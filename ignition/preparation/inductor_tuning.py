# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Agree and retain compiled Inductor kernel selections before graph capture."""

import copy
import inspect
import hashlib
import pickle
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

import torch

from .device_binding import _device_source, _freeze_tuning_winners, get_source_template


def _selected(tuner):
    from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult

    if len(tuner.launchers) != 1:
        raise RuntimeError(f"Inductor kernel {tuner.inductor_meta['kernel_name']} has no final launcher")
    tuner._prune_compile_results_to_launcher(tuner.launchers[0])
    result, = tuner.compile_results
    if not isinstance(result, StaticTritonCompileResult):
        raise TypeError("prepared Inductor selections require a portable static kernel")
    result = copy.copy(result)
    result.kernel = copy.copy(result.kernel)
    result._flare_config_state = copy.deepcopy(vars(result.config))
    if result.kernel.cubin_raw is None:
        result.reload_cubin_path()
        result.kernel.cubin_raw = Path(result.kernel.cubin_path).read_bytes()
    return result


def _selected_digest(tuner):
    from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult

    if len(tuner.launchers) != 1:
        raise RuntimeError(f"Inductor kernel {tuner.inductor_meta['kernel_name']} has no final launcher")
    tuner._prune_compile_results_to_launcher(tuner.launchers[0])
    result, = tuner.compile_results
    if not isinstance(result, StaticTritonCompileResult):
        raise TypeError("prepared Inductor selections require a portable static kernel")
    raw = result.kernel.cubin_raw
    if raw is None:
        result.reload_cubin_path()
        raw = Path(result.kernel.cubin_path).read_bytes()
        result.kernel.cubin_raw = raw
    return dict(vars(result.config)), hashlib.sha256(raw).digest()


# Tuner methods that benchmark or refine kernel configurations.
_SHARDED_SELECTION = ('autotune_to_one_config', '_combo_sequential_autotune',
                      'coordinate_descent_tuning')


def _materialize_selected_cubin(result):
    from torch._inductor.codecache import write_atomic

    raw = result.kernel.cubin_raw
    if raw is None:
        raise ValueError('selected Inductor kernel lacks its compiled cubin bytes')
    result.reload_cubin_path()
    path = result.kernel.cubin_path
    if Path(path).read_bytes() != raw:
        write_atomic(path, raw)


def _install(tuner, encoded):
    result = pickle.loads(encoded)
    result.config.__dict__.clear()
    result.config.__dict__.update(result._flare_config_state)
    result.compile_meta = dict(result.compile_meta, device=tuner.triton_meta['device'])
    _materialize_selected_cubin(result)
    tuner.compile_results = [result]
    tuner.launchers = []
    tuner._cached_launcher = None
    tuner.inductor_meta = {**tuner.inductor_meta, 'dynamic_scale_rblock': False}
    tuner.__dict__.pop('_could_rblock_scale', None)
    tuner._make_launchers()


def _snapshot(entry, tuner):
    from collections import defaultdict

    if len(tuner.compile_results) > 1 and len(tuner.launchers) == 1:
        tuner._prune_compile_results_to_launcher(tuner.launchers[0])
    saved = type(tuner).__new__(type(tuner))
    saved.__dict__.update(tuner.__dict__)
    if getattr(tuner, "fn", None) is not None:
        saved.fn = copy.copy(tuner.fn)
        if hasattr(saved.fn, "create_binder"):
            saved.fn.device_caches = defaultdict(saved.fn.create_binder)
        if hasattr(saved.fn, "kernel"):
            saved.fn.kernel = None
    saved.prepare_for_pickle()
    saved.launchers = []
    if "_cached_launcher" in saved.__dict__:
        saved._cached_launcher = None
    saved._reload_kernel = None
    results = []
    for live in tuner.compile_results:
        result = copy.copy(live)
        result.kernel = copy.copy(live.kernel)
        state = copy.deepcopy(vars(live.config))
        result._flare_config_state = state
        result.config = copy.copy(live.config)
        result.config.__dict__.clear()
        result.config.__dict__.update(copy.deepcopy(state))
        if result.kernel.cubin_raw is None:
            result.reload_cubin_path()
            result.kernel.cubin_raw = Path(result.kernel.cubin_path).read_bytes()
        results.append(result)
    saved.compile_results = results
    return replace(entry, kernel=saved)


def _kernel_globals(graph):
    pending, visited, matches = [graph.current_callable], set(), {}
    while pending:
        function = pending.pop()
        if inspect.ismethod(function):
            function = function.__func__
        if id(function) in visited or not inspect.isfunction(function):
            continue
        visited.add(id(function))
        if Path(function.__code__.co_filename).stem == graph.cache_key:
            matches[id(function.__globals__)] = function.__globals__
        for cell in function.__closure__ or ():
            pending.append(cell.cell_contents)
    if len(matches) != 1:
        raise RuntimeError("compiled forward does not retain one generated Python module")
    return next(iter(matches.values()))


def _execution_identity(source_identity, source, device, kernels, bindings=(), *, prehashed=False,
                        normalized_source=None):
    """Identify executable code without device ordinals or process-local bindings."""
    normalized = _device_source(source, device, 0) if normalized_source is None else normalized_source
    programs = tuple(sorted((name, pickle.dumps(config, protocol=5),
                             cubin if prehashed else hashlib.sha256(cubin).digest())
                            for name, config, cubin in kernels))
    return hashlib.sha256(pickle.dumps(("prepared-forward-v2", source_identity, normalized, programs, bindings),
                                      protocol=5)).hexdigest()


def _unchanged_frozen(selections, bound):
    """Whether a restored forward already holds its frozen static kernel winners."""
    forwards = getattr(selections, "forwards", None)
    if not forwards or id(bound) not in forwards:
        return False
    backend, graph, bindings = forwards[id(bound)]
    if not getattr(backend, "_tuning_frozen", False):
        return False
    bundle = getattr(graph, "_triton_bundle", None)
    if bundle is None:
        return not bindings
    if len(bundle.static_autotuners) != len(bindings):
        return False
    return all(
        len(entry.kernel.compile_results) == 1
        and hasattr(entry.kernel.compile_results[0], "_flare_config_state")
        and InductorSelections._matches(tuner, entry.kernel.compile_results[0])
        for _, entry, tuner in bindings.values()
    )


class InductorSelections:
    """One canonical winner per exact generated kernel across a preparation catalog.

    A kernel held by multiple tensor-parallel ranks is benchmarked on one owning
    rank; every holding rank then runs the winner that owner measured.
    """

    def __init__(self, device, *, coordinator=None, lock_gpu_clocks=False, source_identity=None):
        self.device = torch.device(device)
        self.coordinator = coordinator
        self.lock_gpu_clocks = lock_gpu_clocks
        self.source_identity = source_identity
        self.winners = {}
        self.instances = {}
        self.forwards = {}
        self._active_owners = None
        self._owner_loads = None

    def _gather(self, value):
        if self.coordinator is None:
            return [value]
        import torch.distributed as dist

        values = [None] * len(self.coordinator.ranks)
        dist.all_gather_object(values, value, group=self.coordinator.group)
        return values

    def _check(self, failure):
        failures = self._gather(failure)
        if any(failures):
            raise RuntimeError(f"Inductor preparation failed across ranks: {failures}")

    def _assign_owners(self, local_identities):
        """Choose the one rank whose GPU trials decide each unresolved kernel.

        Kernels held by several ranks are sharded round-robin over their holders, so
        a shared catalog benchmarks each kernel on one rank instead of every rank.
        """
        gathered = self._gather((sorted(local_identities), sorted(self.winners)))
        inventories = [inventory for inventory, _ in gathered]
        resolved = set.intersection(*(set(known) for _, known in gathered)) if gathered else set()
        if self._owner_loads is None or len(self._owner_loads) != len(inventories):
            assigned = [0] * len(inventories)
        else:
            assigned = list(self._owner_loads)
        owners = {}
        for identity in sorted({item for inventory in inventories for item in inventory}):
            if identity in resolved:
                continue
            holders = [rank for rank, inventory in enumerate(inventories) if identity in inventory]
            if not holders:
                continue
            owner = min(holders, key=lambda rank: (assigned[rank], rank))
            owners[identity] = owner
            assigned[owner] += 1
        return owners

    def register(self, bound):
        from torch._inductor.async_compile import CompiledTritonKernels
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        backend = bound.backend
        graph = backend.artifact._flare_graph
        original, = graph.device_idxs
        entries = ({} if graph._triton_bundle is None else
                   {entry.cache_key: entry for entry in graph._triton_bundle.static_autotuners})
        module = _kernel_globals(graph) if entries else {}
        template = get_source_template(graph, original)
        if template.invalid_kernel_assignment:
            raise ValueError("unsupported generated Inductor kernel assignment")
        bindings = {}
        for name, kernel_template, identity in template.assigned_kernels:
            cache_key = CompiledTritonKernels.key(kernel_template.render(original))
            tuner = module.get(name)
            if not isinstance(tuner, CachingAutotuner) or cache_key not in entries:
                kind = f'{type(tuner).__module__}.{type(tuner).__qualname__}'
                raise TypeError(
                    f'generated Inductor kernel {name!r} in graph {graph.cache_key} lacks its static '
                    f'autotuner binding: live_type={kind}, source_key={cache_key}, '
                    f'bundled={cache_key in entries}, static_entries={len(entries)}')
            bindings[cache_key] = identity, entries[cache_key], tuner
            instances = self.instances.setdefault(identity, {})
            if id(tuner) not in instances:
                instances[id(tuner)] = tuner
                if identity in self.winners:
                    _install(tuner, self.winners[identity])
        # Epilogue-fusion benchmarking bundles template candidates that final code never names.
        if any(entries[key].kernel_name in graph.source_code for key in set(entries) - set(bindings)):
            raise ValueError("compiled forward has unbound static Inductor kernels")
        self.forwards[id(bound)] = backend, graph, bindings

    @contextmanager
    def _tuning(self, bindings, *, frozen):
        stack = None
        saved = []
        tuners = {id(tuner): tuner for _, _, tuner in bindings.values()}
        peer_owned = set()
        if not frozen and self._active_owners is not None and self.coordinator is not None:
            import torch.distributed as dist

            me = self.coordinator.ranks.index(dist.get_rank())
            peer_owned = {id(tuner) for identity, _, tuner in bindings.values()
                          if identity in self._active_owners
                          and self._active_owners[identity] != me}

        def selection(method):
            def run(*args, **kwargs):
                nonlocal stack
                if stack is not None:
                    return method(*args, **kwargs)
                with ExitStack() as active:
                    stack = active
                    try:
                        return method(*args, **kwargs)
                    finally:
                        stack = None
            return run

        def benchmark(method):
            def run(*args, **kwargs):
                if frozen:
                    raise RuntimeError("frozen Inductor kernel attempted to benchmark")
                if self.lock_gpu_clocks:
                    if stack is None:
                        raise RuntimeError("Inductor benchmark ran outside kernel selection")
                    if not getattr(stack, '_flare_clocks_locked', False):
                        from .clocks import locked_gpu_clocks
                        uuid = str(torch.cuda.get_device_properties(self.device).uuid)
                        stack.enter_context(locked_gpu_clocks(uuid))
                        stack._flare_clocks_locked = True
                return method(*args, **kwargs)
            return run

        def compile_result(tuner, method):
            def run(*args, **kwargs):
                result = method(*args, **kwargs)
                # Refinement searches otherwise retain only their launcher, which
                # cannot carry the selected cubin into another process.
                if result not in tuner.compile_results:
                    tuner.compile_results.append(result)
                return result
            return run

        def peer_selection(tuner):
            def run(*args, **kwargs):
                # A peer-owned kernel launches one real candidate so the forward
                # produces its activations. The owning rank's trials decide the
                # winner, so this rank runs no trials, refines nothing, and locks
                # no clocks; reconciliation installs the owner's choice.
                if not tuner.launchers:
                    tuner._precompile_worker()
                if not tuner.launchers:
                    tuner._make_launchers()
                if not tuner.launchers:
                    raise RuntimeError(
                        f"Inductor kernel {tuner.inductor_meta['kernel_name']} produced no launcher")
                first = tuner.launchers[0]
                if hasattr(tuner, '_release_static_launchers_except'):
                    tuner._release_static_launchers_except(first)
                tuner.launchers = [first]
                tuner._prune_compile_results_to_launcher(first)
                return first
            return run

        try:
            for tuner in tuners.values():
                for name in ('autotune_to_one_config', '_combo_sequential_autotune',
                             'coordinate_descent_tuning', 'bench', '_precompile_config'):
                    previous = tuner.__dict__.get(name)
                    saved.append((tuner, name, previous, name in tuner.__dict__))
                    method = getattr(tuner, name)
                    wrapper = (peer_selection(tuner) if id(tuner) in peer_owned
                               and name in _SHARDED_SELECTION else
                               compile_result(tuner, method) if name == '_precompile_config' else
                               benchmark(method) if name == 'bench' else selection(method))
                    setattr(tuner, name, wrapper)
            yield
        finally:
            for tuner, name, previous, present in reversed(saved):
                if present:
                    setattr(tuner, name, previous)
                else:
                    delattr(tuner, name)

    def warm(self, bound, warmup, *, before_run=None):
        backend, _, bindings = self.forwards[id(bound)]
        inventory = sorted({identity for identity, _, _ in bindings.values()})
        variants = {}
        for variant, identities in self._gather((getattr(bound.owner, 'compilation_variant', 'shared'), inventory)):
            if variant in variants and variants[variant] != identities:
                raise RuntimeError(f"equivalent compiled forward variant {variant!r} has different Inductor kernels")
            variants[variant] = identities
        adopted, failure = {}, None
        if backend._tuning_frozen:
            # A restored artifact already carries each kernel's frozen winner, so
            # adopting and sharing it lets later forwards reuse those kernels without tuning.
            try:
                for identity, _, tuner in bindings.values():
                    if identity not in self.winners:
                        adopted[identity] = pickle.dumps(_selected(tuner), protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
        self._check(failure)
        adopting = [index for index, present in enumerate(self._gather(bool(adopted))) if present]
        if adopting:
            import torch.distributed as dist

            me = 0 if self.coordinator is None else self.coordinator.ranks.index(dist.get_rank())
            newly_adopted = set()
            for owner in adopting:
                message = [adopted if owner == me else None]
                if self.coordinator is not None:
                    dist.broadcast_object_list(message, src=self.coordinator.ranks[owner],
                                               group=self.coordinator.group)
                for identity, encoded in message[0].items():
                    if identity not in self.winners:
                        self.winners[identity] = encoded
                        newly_adopted.add(identity)
            failure = None
            try:
                candidates = ({identity for identity, _, _ in bindings.values()} & self.winners.keys()) | (
                    newly_adopted & self.instances.keys()
                )
                for identity in sorted(candidates):
                    winner = self.winners[identity]
                    expected = pickle.loads(winner)
                    for tuner in self.instances.get(identity, {}).values():
                        if not self._matches(tuner, expected):
                            _install(tuner, winner)
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
            self._check(failure)
        # A kernel whose winner is unknown is benchmarked in the initial tuning pass.
        # A forward whose bound kernels all hold their winner skips that pass, because
        # register or a restored artifact already selected them.
        needs_tuning = (not backend._tuning_frozen
                        and (not bindings
                             or any(identity not in self.winners for identity, _, _ in bindings.values())))
        if any(self._gather(needs_tuning)):
            self._active_owners = self._assign_owners({identity for identity, _, _ in bindings.values()})
            rank_count = 1 if self.coordinator is None else len(self.coordinator.ranks)
            if self._owner_loads is None or len(self._owner_loads) != rank_count:
                self._owner_loads = [0] * rank_count
            for owner in self._active_owners.values():
                self._owner_loads[owner] += 1
            try:
                failure = None
                try:
                    with warmup(), self._tuning(bindings, frozen=False):
                        if before_run is not None:
                            before_run()
                        bound()
                    torch.cuda.current_stream(self.device).synchronize()
                except Exception as error:
                    failure = f"{type(error).__name__}: {error}"
                self._check(failure)
                self._reconcile(bindings)
            finally:
                self._active_owners = None
        failure = None
        try:
            with warmup(), self._tuning(bindings, frozen=True):
                if before_run is not None:
                    before_run()
                bound()
            torch.cuda.current_stream(self.device).synchronize()
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        self._check(failure)
        actual, failure = {}, None
        try:
            for identity, _, tuner in bindings.values():
                actual[identity] = _selected_digest(tuner)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        self._check(failure)
        gathered = self._gather(actual)
        agreed = {}
        for rank in gathered:
            for identity, selection in rank.items():
                if identity in agreed and agreed[identity] != selection:
                    raise RuntimeError("equivalent Inductor kernels disagree after frozen warmup")
                agreed[identity] = selection

    def _reconcile(self, bindings):
        import torch.distributed as dist

        local = {identity: tuner for identity, _, tuner in bindings.values()}
        owners = (dict(self._active_owners) if self._active_owners is not None
                  else self._assign_owners(local.keys()))
        me = 0 if self.coordinator is None else self.coordinator.ranks.index(dist.get_rank())
        payload, failure = {}, None
        try:
            for identity, owner in owners.items():
                if owner == me:
                    payload[identity] = pickle.dumps(_selected(local[identity]), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        self._check(failure)
        for owner in sorted(set(owners.values())):
            message = [payload if owner == me else None]
            if self.coordinator is not None:
                dist.broadcast_object_list(message, src=self.coordinator.ranks[owner],
                                           group=self.coordinator.group)
            self.winners.update(message[0])
        # Newly resolved kernels replace every instance's selection. Kernels that
        # already held their winner keep their launchers unless the live selection
        # diverged, so a catalog does not rebuild launchers for every earlier
        # forward on every warmup.
        failure = None
        try:
            for identity in sorted(set(local) | set(owners)):
                winner = self.winners[identity]
                expected = None if identity in owners else pickle.loads(winner)
                for tuner in self.instances.get(identity, {}).values():
                    if expected is None or not self._matches(tuner, expected):
                        _install(tuner, winner)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        self._check(failure)

    @staticmethod
    def _matches(tuner, expected):
        """Whether a live tuner already runs the encoded winner."""
        from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult

        if len(tuner.launchers) != 1 or len(tuner.compile_results) != 1:
            return False
        result, = tuner.compile_results
        return (isinstance(result, StaticTritonCompileResult)
                and vars(result.config) == expected._flare_config_state
                and result.kernel.cubin_raw == expected.kernel.cubin_raw)

    def freeze(self, bound):
        import zlib

        backend, graph, bindings = self.forwards[id(bound)]
        serialized = getattr(backend, "_serialized_artifact", None)
        if serialized is not None and not _unchanged_frozen(self, bound):
            from .aot import _RawPortableArtifact

            snapshots = [_snapshot(entry, tuner) for _, entry, tuner in bindings.values()]
            bundle = graph._triton_bundle
            if bundle is not None:
                bundle = replace(bundle, static_autotuners=snapshots)
                _freeze_tuning_winners(bundle)
            raw = (bytes(serialized) if isinstance(serialized, _RawPortableArtifact)
                   else zlib.decompress(serialized))
            entry, kernels, constants = pickle.loads(raw)
            entry.compiled_fw.result._triton_bundle = bundle
            dumped = pickle.dumps((entry, kernels, constants), protocol=pickle.HIGHEST_PROTOCOL)
            backend._serialized_artifact = (_RawPortableArtifact(dumped)
                                            if isinstance(serialized, _RawPortableArtifact)
                                            else zlib.compress(dumped, level=1))
        backend._tuning_frozen = True
        if self.source_identity is not None:
            from torch.utils._pytree import tree_map
            from .aot import _serialized_invocations

            device, = graph.device_idxs
            template = get_source_template(graph, device)
            selected = [(identity, *_selected_digest(tuner)) for identity, _, tuner in bindings.values()]
            description = (backend.signature,
                tuple((item.index, item.view, item.region) for item in backend.inputs),
                _serialized_invocations(backend.invocations))
            description = tree_map(lambda value: torch.device(value.type)
                                   if isinstance(value, torch.device) else value, description)
            bound.execution_identity = _execution_identity(self.source_identity, graph.source_code, device,
                selected, description, prehashed=True, normalized_source=template.render(0))

    def release(self, bound):
        self.forwards.pop(id(bound))
        if not self.forwards:
            self.instances.clear()


_CANONICAL_INDUCTOR_SELECTIONS = InductorSelections
_CANONICAL_FREEZE = InductorSelections.freeze


def _uses_default_portable_freeze(selections):
    """Whether `selections.freeze` runs `InductorSelections.freeze` without subclass or instance hooks."""
    return (type(selections) is _CANONICAL_INDUCTOR_SELECTIONS
            and type(selections).freeze is _CANONICAL_FREEZE
            and "freeze" not in vars(selections))
