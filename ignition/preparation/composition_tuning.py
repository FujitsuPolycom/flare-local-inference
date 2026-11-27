# SPDX-License-Identifier: Apache-2.0

"""Race prepared component compositions before compiling model forwards."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from b12x.preparation import (
    CollectiveRequirement, FrozenMapping, Knob, MemoryRequirements,
    PersistentMemory, Plan, PreparationSession, PreparedCall, TuningContract,
)


def plan_identity(plan):
    """Describe selected programs and their invocation without live resource handles."""
    from b12x.preparation.types import _CompositePlan

    if isinstance(plan, _CompositePlan):
        return dict(component=plan.component_id, version=plan.composite_semantic_version,
                    capacity=plan.capacity_metadata,
                    variants=tuple((count, plan_identity(child)) for count, child in plan.variants.items()))
    if plan.selection is None or plan.prepared is None:
        raise ValueError("composition identity requires prepared child plans")
    contract, selection = plan.contract, plan.selection
    return dict(component=plan.component_id,
        versions=(contract.query_schema_version, contract.config_schema_version,
                  contract.semantic_version, contract.candidate_contract_version),
        query=selection.query.to_dict(), invocation=plan.invocation.to_dict(),
        config=contract.config_payload(selection.config).to_dict(),
        programs=tuple(sorted((program.dialect, program.key) for program in plan.prepared.programs)))


@dataclass(frozen=True)
class PreparedComposition:
    """A warmed compiled forward, its child plans and its private trial inputs.

    ``call.run`` must be the compiled forward itself, so its captured computation
    is precisely the code identified by the catalog. Producers and validators
    belong to the declared workload. They must not write model state or restore
    communication counters. The caller retains these borrowed owners until the
    complete race has drained.
    """

    forward: object
    plans: tuple
    call: PreparedCall
    inference_mode: bool = field(default_factory=torch.is_inference_mode_enabled)
    grad_enabled: bool = field(default_factory=torch.is_grad_enabled)

    def identity(self):
        identity = getattr(self.forward, "execution_identity", None)
        if not identity or self.call.run is not self.forward:
            raise ValueError("composition races require the identified compiled forward as their run callable")
        if not self.call.benchmark_producers or self.call.validate is None:
            raise ValueError("composition races require representative producers and output validation")
        if self.call.close is not None or self.call.restore is not None:
            raise ValueError("borrowed composition calls must not transfer close or restore ownership")
        return dict(forward=identity, execution_context=(self.inference_mode, self.grad_enabled),
                    children=tuple((role, plan_identity(plan)) for role, plan in self.plans))

    def invoke(self):
        # b12x benchmarks under inference_mode. Restore the warmed forward's
        # dispatch context while recording it; captured replay runs no Python.
        with torch.inference_mode(self.inference_mode), torch.set_grad_enabled(self.grad_enabled):
            return self.forward()

    def prepared_call(self):
        return replace(self.call, run=self.invoke)


def _validate_query(query, device):
    if (not isinstance(query, FrozenMapping) or not query["operation"] or not query["catalog"]
            or query["default"] not in query["catalog"]):
        raise ValueError("composition queries require named implementations of one operation")


def _validate_config(query, config, device):
    if not isinstance(config, FrozenMapping) or config["implementation"] not in query["catalog"]:
        raise ValueError("composition choice is absent from the prepared catalog")


_CONTRACT = TuningContract(
    component_id="flare.prepared_composition", query_schema_version=1, config_schema_version=1,
    query_fields=frozenset(("operation", "geometry", "numerics", "workload", "default", "catalog")),
    config_fields=frozenset(("implementation",)), encode_query=lambda query: query,
    encode_config=lambda config: config, decode_config=FrozenMapping,
    validate_query=_validate_query, validate_config=_validate_config,
    default_config=lambda query, device: FrozenMapping({"implementation": query["default"]}),
    knobs=(Knob(name="implementation", values=None),),
    parameters=lambda query, device: {"implementation": tuple(query["catalog"])},
    # Rank-specific native programs form each rank's cache key. The collective
    # cache combines those keys and GPU identities after agreeing this contract.
    collective_signature=lambda query: {key: tuple(value) if key == "catalog" else value
                                        for key, value in query.items()},
)


def trial_session(session, *, device, coordinator, **overrides):
    """Open the private preparation session in which a composition trial tunes its own child plans.

    The trial inherits the model session's checkpoint identity, namespace, effort and memory
    limits so that its selections share the model's cache; `overrides` adjust other arguments.
    """
    from .tuning import RecordedPreparationSession

    namespace = session.namespace.to_dict()
    checkpoint, schema, numerics = (namespace.pop(key) for key in (
        "checkpoint", "workload_schema", "numerical_policy"))
    return RecordedPreparationSession(device=device, checkpoint=checkpoint, workload_schema=schema,
        numerical_policy=numerics, namespace=namespace, coordinator=coordinator,
        compile_workers=session.requested_compile_workers, autotune_effort=session.autotune_effort,
        headroom_bytes=session.headroom_bytes, lock_gpu_clocks=session.lock_gpu_clocks, **overrides)


@contextmanager
def composition_race_session(session, *, coordinator, cache_policy):
    """Share one b12x preparation session across a batch of composition races.

    Each standalone ``race_compositions`` call opens and closes its own
    ``PreparationSession``, reloading the selection cache, reopening the
    progress store, and running a full cyclic garbage collection pass on close.
    Holding one session across a batch of composition races reuses the loaded
    cache and defers close-time reclamation to the batch boundary while keeping
    the same cache identity and isolation.
    """
    with PreparationSession(device=session.device, autotune=session.autotune, compile_workers=0,
            prime=False, cache_dir=session.cache_dir, namespace=session.namespace,
            cache_policy=cache_policy, collective_barrier=coordinator.barrier) as race:
        yield race


def _validate_race_session(race_session, *, session, coordinator, cache_policy):
    expected_cache_dir = None if session.cache_dir is None else Path(session.cache_dir)
    actual_cache_dir = None if getattr(race_session, "cache_dir", None) is None else Path(race_session.cache_dir)
    expected_namespace = FrozenMapping(session.namespace or {})
    actual_namespace = FrozenMapping(getattr(race_session, "namespace", None) or {})
    if (not isinstance(race_session, PreparationSession)
            or race_session.device != session.device
            or race_session.autotune != session.autotune
            or race_session.compile_workers != 0
            or race_session.prime
            or actual_cache_dir != expected_cache_dir
            or actual_namespace != expected_namespace
            or race_session.cache_policy != cache_policy
            or race_session.collective_barrier != coordinator.barrier):
        raise ValueError("composition race session must match the caller's session, policy, and coordinator")


def race_compositions(candidates, *, operation, geometry, numerics, workload, default,
                      session, coordinator, cache_policy, resident_bytes=0, race_session=None):
    """Select complete compiled work using b12x's coordinated race and cache.

    This runs after child preparation and compiled warmup. Materialization only
    looks up existing callables: b12x forbids compilation and lowering inside
    this phase. No trial graph or compiled owner is retained by the returned
    record, and the caller binds its selected composition before model capture.
    Pass ``race_session`` from ``composition_race_session`` to reuse one
    ``PreparationSession`` across a batch of races.
    """
    if not candidates or type(resident_bytes) is not int or resident_bytes < 0:
        raise ValueError("composition races require candidates and nonnegative resident storage")
    if race_session is not None:
        _validate_race_session(race_session, session=session, coordinator=coordinator, cache_policy=cache_policy)
    query = FrozenMapping(dict(operation=operation, geometry=geometry, numerics=numerics,
        workload=workload, default=default,
        catalog={name: candidate.identity() for name, candidate in candidates.items()}))
    borrowed = MemoryRequirements(persistent=(PersistentMemory(
        ("prepared-compositions", id(candidates)), resident_bytes, resident_bytes),))
    plan = Plan(contract=_CONTRACT, query=query, _device=f"cuda:{session.device.ordinal}",
        _compile_jobs=lambda config, device: (), _memory_requirements=lambda config, device: borrowed,
        _materialize=lambda selection, device: candidates[selection.config["implementation"]],
        _benchmark_call=lambda candidate: candidate.prepared_call())
    request = plan.request(name=operation, prepare_call=lambda candidate: candidate.prepared_call(),
        collective=CollectiveRequirement(operation, coordinator.ranks))
    scope = (composition_race_session(session, coordinator=coordinator, cache_policy=cache_policy)
             if race_session is None else nullcontext(race_session))
    with scope as race:
        with race.prepare((request,), coordinator=coordinator) as result:
            selected = result.selections[operation]
            return dict(implementation=selected.config["implementation"], source=selected.source,
                cache_hits=result.cache_hits, benchmarked_candidates=result.benchmarked_candidates,
                rejected_candidates=result.rejected_candidates, coverage=dict(result.coverage[operation]))
