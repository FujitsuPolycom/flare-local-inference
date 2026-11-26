# SPDX-License-Identifier: Apache-2.0
"""Identify dependency-closed local tuning work that all tensor-parallel ranks can share."""

from hashlib import sha256
from io import BytesIO
import pickle

from b12x.preparation import CandidateResourceIneligibleError

from b12x.preparation.session import _coalesce_requests, _declaration_key, _expand_requests, _plans_of
from b12x.preparation.types import _CompositePlan, _plan_scope


def portable_declaration(plan):
    if isinstance(plan, _CompositePlan):
        return (plan.component_id, plan.composite_semantic_version, plan.capacity_metadata, plan.shared,
                tuple((rows, portable_declaration(child)) for rows, child in plan.variants.items()))
    # Placement does not specify a different kernel on equivalent devices.
    return (*_declaration_key(plan)[:-1], getattr(plan, "shared", False))


def local_components(requests):
    """Return whole local components, retaining declaration aliases and request dependencies."""
    links = {request.name: request.name for request in requests}
    def root(name):
        while links[name] != name:
            links[name] = links[links[name]]
            name = links[name]
        return name

    aliases = {}
    for request in requests:
        name = request.name
        for dependency in request.dependencies:
            if dependency not in links:
                raise ValueError(f"unknown preparation dependency {dependency!r} for {name!r}")
            links[root(name)] = root(dependency)
        for plan in _plans_of(request):
            other = aliases.setdefault(_declaration_key(plan), name)
            links[root(name)] = root(other)

    groups = {}
    for request in requests:
        groups.setdefault(root(request.name), []).append(request)
    return [tuple(sorted(group, key=lambda request: request.name))
            for group in groups.values() if all(request.collective is None for request in group)]


def component_key(group):
    """Identify the complete request DAG and the declaration bound to every source name."""
    return tuple((r.name, portable_declaration(r.plan), tuple(r.dependencies)) for r in group)


def materialization_key(session, group):
    """Check the candidate factories and per-candidate memory bounds a sharded race will use.

    Some bindings carry rank-local compressed planes. Equal operator queries are not proof that
    their factories, scratch geometry, or simultaneously resident storage also agree.
    """
    digest = sha256()
    owners = {}
    expanded, _ = _expand_requests(group)
    for equivalent in _coalesce_requests(expanded):
        plan = equivalent[0].plan
        choice = plan.contract.configure(plan.query, device=session.device.identity, override=plan.override)
        configs = (choice.pinned,) if choice.pinned is not None else (
            choice.default, *(c for _, c in plan.contract.iterate(choice)))
        seen = set()
        for config in configs:
            payload = plan.contract.config_payload(config)
            if payload in seen:
                continue
            seen.add(payload)
            # Use the declared memory without imposing this rank's scratch admission limit.
            seen_plans = set()
            footprint = []
            for request in equivalent:
                bound = request.plan
                if bound in seen_plans:
                    continue
                seen_plans.add(bound)
                try:
                    with _plan_scope(bound):
                        memory = bound._memory_requirements(config, session.device)
                except CandidateResourceIneligibleError:
                    footprint.append(None)
                    continue
                footprint.append((
                    tuple((s.name, tuple(s.shape), str(s.dtype), s.device.type,
                           None if s.device.index == session.device.ordinal else s.device.index)
                          for s in memory.scratch),
                    tuple((owners.setdefault(m.key, len(owners)), m.required_nbytes, m.resident_nbytes)
                          for m in memory.persistent),
                ))
            jobs = () if None in footprint else tuple(plan._compile_jobs(config, session.device))
            declaration = (payload, jobs, tuple(footprint))
            # Compiler metadata is acyclic. Equal arguments can nonetheless
            # have different object sharing across ranks; identity memoization
            # would distinguish such equivalent compiler specifications.
            stream = BytesIO()
            encoding = pickle.Pickler(stream, protocol=5)
            encoding.fast = True
            encoding.dump(declaration)
            digest.update(stream.getbuffer())
    return digest.digest()
