# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Ranks compile each forward variant once, and ranks on separate hosts receive the others by path."""

from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path

import pytest


def _rank(rank, world, store, home, owned, outcome):
    import torch.distributed as dist

    from ignition.preparation.aot import _share_artifacts

    # Each rank's working directory stands in for its host's disk; artifact paths are identical.
    os.chdir(home)
    try:
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
        _share_artifacts(owned, dist.group.WORLD, tuple(range(world)))
        outcome.put((rank, {path: Path(path).read_bytes() for path in ("forwards/a.bin", "forwards/b.bin")}))
        dist.destroy_process_group()
    except Exception as error:  # noqa: BLE001 - The parent reports every rank's failure.
        outcome.put((rank, repr(error)))


def test_each_variant_compiles_on_the_least_loaded_rank_that_binds_it():
    from ignition.preparation.aot import _variant_compilers

    shared = [{name: "shared" for name in "abcdef"}] * 4
    assert list(_variant_compilers(shared).values()) == [0, 1, 2, 3, 0, 1]
    # Rank zero samples; its peers consume, so neither variant compiles on a rank that cannot bind it.
    sampled = [{"decode": "sample", "resume": "sample"}] + [{"decode": "consume", "resume": "consume"}] * 3
    assert _variant_compilers(sampled) == {("decode", "sample"): 0, ("decode", "consume"): 1,
                                           ("resume", "sample"): 0, ("resume", "consume"): 2}


def test_every_rank_receives_what_it_lacks_from_the_rank_that_compiled_it(tmp_path):
    context = multiprocessing.get_context("fork")
    homes = [tmp_path / f"host{rank}" for rank in range(3)]
    for home in homes:
        (home / "forwards").mkdir(parents=True)
    (homes[0] / "forwards/a.bin").write_bytes(b"compiled a")
    (homes[1] / "forwards/b.bin").write_bytes(b"compiled b")
    # Rank two compiled nothing, already holds a, and holds a stale b.
    (homes[2] / "forwards/a.bin").write_bytes(b"compiled a")
    (homes[2] / "forwards/b.bin").write_bytes(b"stale b")
    owned = [["forwards/a.bin"], ["forwards/b.bin"], []]
    outcome = context.Queue()
    processes = [context.Process(target=_rank, args=(rank, 3, tmp_path / "store", homes[rank], owned, outcome))
                 for rank in range(3)]
    for process in processes:
        process.start()
    results = dict(outcome.get(timeout=60) for _ in processes)
    for process in processes:
        process.join(timeout=60)
    expected = {"forwards/a.bin": b"compiled a", "forwards/b.bin": b"compiled b"}
    assert results == {rank: expected for rank in range(3)}


def test_ranks_lend_missing_compilations_until_lending_finishes_no_sooner():
    from ignition.preparation.aot import _balance_compilations

    # Rank zero's sampling variants leave it most of a TP16 catalog; lending levels every rank at three waves.
    assigned = _balance_compilations([13] + [2] * 15, [1] * 16)
    assert max(map(len, assigned)) == 3 and len(assigned[0]) == 3
    assert sorted(job for jobs in assigned for job in jobs) == sorted(
        [(0, index) for index in range(13)] + [(rank, index) for rank in range(1, 16) for index in range(2)])
    # Ranks keep their own forwards, and a rank with two workers finishes two per wave.
    assigned = _balance_compilations([9, 0, 0], [2, 1, 1])
    assert [len(jobs) for jobs in assigned] == [5, 2, 2]
    assert all(owner == 0 for jobs in assigned for owner, _ in jobs)
    assert _balance_compilations([1, 1], [1, 1]) == [[(0, 0)], [(1, 0)]]
    # A rank whose GPU has no room for a compiler lends every forward it owns.
    assigned = _balance_compilations([13, 2, 2, 2], [1, 1, 0, 1])
    assert assigned[2] == [] and sorted(job for jobs in assigned for job in jobs) == sorted(
        [(0, index) for index in range(13)] + [(rank, index) for rank in (1, 2, 3) for index in range(2)])
    assert max(map(len, assigned)) == 7


class _Forward:
    def __init__(self, variant):
        self.compilation_variant = variant


class _CompilerMetadata:
    pass


class _Artifact:
    def __init__(self, execute):
        from inspect import signature
        from types import SimpleNamespace

        from ignition.preparation.aot import StorageBackend
        from torch._dynamo.guards import GuardManagerWrapper

        backend = StorageBackend.__new__(StorageBackend)
        backend.artifact = SimpleNamespace(inner_fn=SimpleNamespace(compiled_fn=execute))
        backend._serialized_artifact = b"portable backend"
        backend._cuda_allocation_sizes = ()
        backend._tuning_frozen = False
        guards = SimpleNamespace(check=lambda arguments: isinstance(arguments["forward"], _Forward),
                                 check_verbose=lambda arguments: "forward type differs")
        self._artifacts = SimpleNamespace(signature=signature(lambda forward: None),
            runtime_env=SimpleNamespace(closure=()), guard_manager=GuardManagerWrapper(root=guards),
            compiled_fn=backend, guards_state=b"portable guards", compiler_graph=_CompilerMetadata())
        self.fn = execute

    def __call__(self, owner):
        self.fn(owner)

    def save_compiled_function(self, path):
        assert Path(path).exists()


def _replace_gpu_preparation():
    from types import SimpleNamespace
    import torch
    from ignition.preparation import compile_pool, inductor_tuning

    class Selections(inductor_tuning.InductorSelections):
        def register(self, bound):
            state = bound.artifact._artifacts
            self.forwards[id(bound)] = state.compiled_fn, state.compiler_graph, {}

        def freeze(self, bound):
            self.forwards[id(bound)][0]._tuning_frozen = True

    inductor_tuning.InductorSelections = Selections
    compile_pool.compiler_resources = lambda *args, **kwargs: compile_pool.CompilerResources(
        1, 128 << 30, 16, 32 << 30, 8 << 30)
    torch.cuda.current_stream = lambda device: SimpleNamespace(synchronize=lambda: None)


class _Pool:
    """Compiles by writing the artifact path with the compiling rank, as a worker would."""

    workers = 1

    def __init__(self, device, **kwargs):
        self.workers = kwargs.get("workers", 1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, forward, path):
        from ignition.preparation.forward_metadata import dumps

        return self.submit_payload(dumps(forward), path)

    def submit_payload(self, payload, path):
        from concurrent.futures import Future
        import torch.distributed as dist

        from ignition.preparation.forward_metadata import loads

        forward = loads(payload)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{forward.compilation_variant} compiled by rank {dist.get_rank()}")
        future = Future()
        future.set_result(None)
        return future


def _lending_rank(rank, world, store, root, outcome, lifecycle=None, fail_shutdown=False, scenario=None):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch.distributed as dist

    from ignition.preparation import aot, compile_pool

    messages = []
    try:
        class Pool(_Pool):
            def __exit__(self, *args):
                if lifecycle is not None:
                    started, closed, finished = lifecycle
                    if rank == 0:
                        assert started.wait(20), "borrowed compilation did not start"
                        assert not finished.is_set(), "borrowed compilation finished before owner shutdown"
                        closed.set()
                if fail_shutdown and rank == 0:
                    raise RuntimeError("compiler shutdown failed on rank 0")

            def submit_payload(self, payload, path):
                from ignition.preparation.forward_metadata import loads

                if scenario == "two_hosts":
                    borrowed_forward = loads(payload)
                    if borrowed_forward.compilation_variant == "consume":
                        assert hosts[borrowed_forward.rank] == hosts[rank], "same-host consume forward crossed hosts"
                if scenario == "worker_failure" and rank == 1:
                    from concurrent.futures import Future

                    future = Future()
                    future.set_exception(RuntimeError("forward worker failed on rank 1"))
                    return future
                if lifecycle is not None and rank == 1 and loads(payload).compilation_variant == "sample":
                    started, closed, finished = lifecycle
                    started.set()
                    assert closed.wait(20), "idle owner retained its compiler while borrowed work was running"
                    result = super().submit_payload(payload, path)
                    finished.set()
                    return result
                return super().submit_payload(payload, path)

        compile_pool.CompilePool = Pool
        _replace_gpu_preparation()
        hosts = ("a", "b", "a", "b") if scenario == "two_hosts" else ("host",) * world
        if scenario == "two_hosts":
            os.chdir(root / hosts[rank])
            root = Path("forwards")
        if scenario in ("two_hosts", "zero_owners", "no_capacity", "worker_failure"):
            capacity = (rank in (2, 3) if scenario == "two_hosts" else rank == 1)
            if scenario == "no_capacity":
                capacity = False
            compile_pool.compiler_resources = lambda *args, **kwargs: compile_pool.CompilerResources(
                int(capacity), 40 << 30, 16, 32 << 30, 8 << 30)
        aot.ForwardCache._load = lambda self, forward, path, required=False, before_run=None: aot.BoundForward(
            _Artifact(lambda owner: None), forward, True)
        aot.ForwardCache._allocation_sizes = lambda self, path: ()
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
        cache = aot.ForwardCache.__new__(aot.ForwardCache)
        cache.root, cache.device, cache.headroom_bytes, cache.options = root, "cpu", 0, aot.MappingProxyType({})
        cache.compile_workers = None
        cache.lock_gpu_clocks = False
        variant = "sample" if rank == 0 else "consume"
        functions = {name: _Forward(variant) for name in "abcd"}
        for forward in functions.values():
            forward.rank = rank
        coordinator = SimpleNamespace(group=dist.group.WORLD, ranks=tuple(range(world)), hosts=hosts,
            spans_hosts=len(set(hosts)) > 1,
            local_ranks=world, barrier=lambda key, ranks: dist.monitored_barrier(group=dist.group.WORLD))
        selected = cache._prepare_ranks(functions, {name: name for name in functions}, coordinator,
                                        progress=messages.append,
                                        materialize=lambda: None, warmup=nullcontext, before_run=None)
        if scenario == "two_hosts":
            outcome.put((rank, (sorted(selected), sorted(path.read_text() for path in root.glob("*.bin")))))
        else:
            outcome.put((rank, sorted(selected)))
    except Exception:  # noqa: BLE001 - The parent reports every rank's failure.
        import traceback

        outcome.put((rank, (traceback.format_exc(), messages)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_one_host_lends_a_ranks_missing_compilations_to_idle_ranks(tmp_path):
    context = multiprocessing.get_context("fork")
    root = tmp_path / "forwards"
    root.mkdir()
    outcome = context.Queue()
    processes = [context.Process(target=_lending_rank, args=(rank, 3, tmp_path / "store", root, outcome))
                 for rank in range(3)]
    for process in processes:
        process.start()
    results = dict(outcome.get(timeout=120) for _ in processes)
    for process in processes:
        process.join(timeout=60)
    assert results == {rank: list("abcd") for rank in range(3)}
    # Rank zero owns four sampling variants and its peers two consuming variants each: rank zero lends one.
    compiled = sorted(path.read_text() for path in root.glob("*.bin"))
    assert compiled.count("sample compiled by rank 0") == 3
    assert compiled.count("sample compiled by rank 1") == 1
    assert len(compiled) == 8 and not list((root / "lent").iterdir())


def test_idle_compiler_closes_before_a_peer_finishes_borrowed_work(tmp_path):
    context = multiprocessing.get_context("fork")
    root = tmp_path / "forwards"
    root.mkdir()
    outcome = context.Queue()
    lifecycle = tuple(context.Event() for _ in range(3))
    processes = [context.Process(target=_lending_rank,
        args=(rank, 3, tmp_path / "store", root, outcome, lifecycle)) for rank in range(3)]
    try:
        for process in processes:
            process.start()
        results = dict(outcome.get(timeout=90) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        assert results == {rank: list("abcd") for rank in range(3)}
        assert all(event.is_set() for event in lifecycle)
        assert len(list(root.glob("*.bin"))) == 8
        assert not list((root / "lent").iterdir())
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)


def test_compiler_shutdown_failure_is_reported_to_every_rank(tmp_path):
    context = multiprocessing.get_context("fork")
    root = tmp_path / "forwards"
    root.mkdir()
    outcome = context.Queue()
    processes = [context.Process(target=_lending_rank,
        args=(rank, 3, tmp_path / "store", root, outcome, None, True)) for rank in range(3)]
    try:
        for process in processes:
            process.start()
        results = dict(outcome.get(timeout=90) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        assert set(results) == set(range(3))
        assert all("compiler shutdown failed on rank 0" in result[0] for result in results.values())
        assert len(list(root.glob("*.bin"))) == 8
        assert not list((root / "lent").iterdir())
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)


@pytest.mark.parametrize("scenario", ["zero_owners", "two_hosts", "no_capacity", "worker_failure"])
def test_host_admission_lends_to_local_workers_and_reports_failures_collectively(tmp_path, scenario):
    context = multiprocessing.get_context("fork")
    world = 4 if scenario == "two_hosts" else 3
    root = tmp_path / "forwards"
    root.mkdir()
    if scenario == "two_hosts":
        for host in ("a", "b"):
            (root / host / "forwards").mkdir(parents=True)
    outcome = context.Queue()
    processes = [context.Process(target=_lending_rank,
        args=(rank, world, tmp_path / "store", root, outcome, None, False, scenario)) for rank in range(world)]
    try:
        for process in processes:
            process.start()
        results = dict(outcome.get(timeout=120) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        if scenario == "two_hosts":
            for selected, compiled in results.values():
                assert selected == list("abcd")
                assert len(compiled) == 8
                assert compiled.count("sample compiled by rank 2") == 3
                assert compiled.count("sample compiled by rank 3") == 1
                assert compiled.count("consume compiled by rank 2") == 1
                assert compiled.count("consume compiled by rank 3") == 3
            assert results[0] == results[1] == results[2] == results[3]
        elif scenario == "zero_owners":
            assert results == {rank: list("abcd") for rank in range(world)}
            assert all("rank 1" in path.read_text() for path in root.glob("*.bin"))
        else:
            reason = "no capacity for a forward compiler" if scenario == "no_capacity" else "forward worker failed on rank 1"
            for rank, result in results.items():
                assert isinstance(result, tuple), f"rank {rank} did not fail: {result}"
                failure, messages = result
                assert reason in failure
                # The phase must end as failed, not complete, before the error is raised.
                assert any(message.startswith("Compile model forwards failed in") for message in messages)
                assert not any(message.startswith("Compile model forwards complete in") for message in messages)
        assert not list(root.rglob("*.forward"))
        assert not list(root.rglob("*.tmp"))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)


def _cached_rank(rank, world, store, root, deferred, outcome):
    import gc
    import weakref
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch.distributed as dist

    from ignition.preparation import aot, compile_pool

    try:
        compile_pool.CompilePool = _Pool
        _replace_gpu_preparation()
        restored, portable, warmed = [], [], []

        def load(self, forward, path, required=False, before_run=None):
            if required:
                gc.collect()
                assert all(reference() is None for reference in portable), "previous portable artifact is retained"
            artifact = _Artifact(warmed.append)
            bound = aot.BoundForward(artifact, forward, True)
            if not required:
                restored.append(weakref.ref(bound))
            else:
                portable.extend((weakref.ref(artifact), weakref.ref(artifact._artifacts.compiler_graph)))
            return bound

        def materialize():
            gc.collect()
            assert restored and all(reference() is None for reference in restored)

        aot.ForwardCache._load = load
        aot.ForwardCache._allocation_sizes = lambda self, path: ()
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
        cache = aot.ForwardCache.__new__(aot.ForwardCache)
        cache.root, cache.device, cache.headroom_bytes, cache.options = root, "cpu", 0, aot.MappingProxyType({})
        cache.compile_workers = None
        cache.lock_gpu_clocks = False
        functions = {name: _Forward("shared") for name in "abcd"}
        signatures = {name: name for name in functions}
        for name, forward in functions.items():
            cache._path(forward, name=f"{name}.shared", signature=name).write_text("cached")
        dist.barrier()
        coordinator = SimpleNamespace(group=dist.group.WORLD, ranks=tuple(range(world)), spans_hosts=False,
            hosts=("host",) * world,
            local_ranks=world, barrier=lambda key, ranks: dist.monitored_barrier(group=dist.group.WORLD))
        selected = cache._prepare_ranks(functions, signatures, coordinator, progress=None,
            materialize=materialize if deferred else None, warmup=nullcontext, before_run=None)
        assert set(warmed) == set(functions.values())
        assert all(warmed.count(forward) == 2 for forward in functions.values())
        assert all(bound.cache_hit for bound in selected.values())
        gc.collect()
        assert all(reference() is None for reference in portable)
        for bound in selected.values():
            assert bound.prepare_capture() == 0
            bound()
        assert all(warmed.count(forward) == 3 for forward in functions.values())
        bound = selected["a"]
        owner, bound.owner = bound.owner, object()
        with pytest.raises(aot.BindingMismatch, match="compiled guards"):
            bound()
        bound.owner = owner
        assert all(warmed.count(forward) == 3 for forward in functions.values())
        outcome.put((rank, sorted(selected)))
        dist.destroy_process_group()
    except Exception:
        import traceback

        outcome.put((rank, traceback.format_exc()))


@pytest.mark.parametrize("deferred", [False, True])
def test_cached_catalog_releases_restoration_owners_before_rebinding(tmp_path, deferred):
    context = multiprocessing.get_context("fork")
    root = tmp_path / "forwards"
    root.mkdir()
    outcome = context.Queue()
    processes = [context.Process(target=_cached_rank,
        args=(rank, 3, tmp_path / "store", root, deferred, outcome)) for rank in range(3)]
    for process in processes:
        process.start()
    results = dict(outcome.get(timeout=120) for _ in processes)
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0
    assert results == {rank: list("abcd") for rank in range(3)}


@dataclass
class _StaticEntry:
    cache_key: str
    kernel_name: str
    kernel: object


@dataclass
class _StaticBundle:
    static_autotuners: list


def _static_result(block_size, cubin):
    from types import SimpleNamespace
    from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult

    result = object.__new__(StaticTritonCompileResult)
    result.config = SimpleNamespace(block_size=block_size)
    result._flare_config_state = {"block_size": block_size}
    result.compile_meta = {"device": 0}
    result.kernel = SimpleNamespace(cubin_raw=cubin, cubin_path="")
    return result


class _StaticTuner:
    def __init__(self, result):
        self.launchers = [object()]
        self.compile_results = [result]
        self.inductor_meta = {"kernel_name": "triton_poi_0"}
        self.triton_meta = {"device": 0}
        self._should_coordesc_tune = False

    def is_statically_launchable(self):
        return True

    def _prune_compile_results_to_launcher(self, launcher):
        pass

    def prepare_for_pickle(self):
        return None

    def restore_after_unpickle(self, state):
        pass

    def _make_launchers(self):
        self.launchers = [object()]

    def autotune_to_one_config(self, *args, **kwargs):
        return self.launchers[0]

    def _combo_sequential_autotune(self, *args, **kwargs):
        return self.launchers[0]

    def coordinate_descent_tuning(self, *args, **kwargs):
        return self.launchers[0]

    def bench(self, *args, **kwargs):
        return 0.0

    def _precompile_config(self, *args, **kwargs):
        return self.compile_results[0]


def test_frozen_cached_forward_skips_recompression_and_resave_while_unfrozen_saves(tmp_path, monkeypatch):
    import pickle
    import zlib
    from contextlib import nullcontext
    from types import MappingProxyType, SimpleNamespace
    import torch
    from ignition.preparation import aot, inductor_tuning

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(inductor_tuning, "_materialize_selected_cubin", lambda result: None)

    def make_bound(*, frozen, cache_hit, bundled_block=64, live_block=64, serialized=None):
        bundled_tuner = _StaticTuner(_static_result(bundled_block, f"cubin-{bundled_block}".encode()))
        live_tuner = _StaticTuner(_static_result(live_block, f"cubin-{live_block}".encode()))
        entry = _StaticEntry(cache_key="k0", kernel_name="triton_poi_0", kernel=bundled_tuner)
        bundle = _StaticBundle(static_autotuners=[entry])
        graph = SimpleNamespace(
            _triton_bundle=bundle,
            device_idxs=(0,),
            source_code="def call(args):\n    return None\n",
        )
        cache_entry = SimpleNamespace(compiled_fw=SimpleNamespace(result=SimpleNamespace(_triton_bundle=None)))
        backend = aot.StorageBackend.__new__(aot.StorageBackend)
        backend.artifact = SimpleNamespace(_flare_graph=graph, inner_fn=SimpleNamespace(compiled_fn=SimpleNamespace()))
        backend._serialized_artifact = (
            zlib.compress(pickle.dumps((cache_entry, {}, {})), level=1) if serialized is None else serialized
        )
        backend._tuning_frozen = frozen
        backend.signature = ()
        backend.inputs = []
        backend.invocations = []
        saves = []
        artifact = SimpleNamespace(
            _artifacts=SimpleNamespace(compiled_fn=backend),
            fn=lambda owner: None,
            save_compiled_function=lambda target: saves.append(target),
        )
        artifact_call = type("ArtifactCall", (), {
            "_artifacts": artifact._artifacts,
            "fn": artifact.fn,
            "save_compiled_function": staticmethod(artifact.save_compiled_function),
            "__call__": lambda self, owner: None,
        })()
        bound = aot.BoundForward(artifact_call, _Forward("shared"), cache_hit=cache_hit)
        return bound, entry, live_tuner, saves

    class KernelSelections(inductor_tuning.InductorSelections):
        def __init__(self, entry, live_tuner):
            super().__init__("cpu", source_identity="source-v1")
            self._entry = entry
            self._live_tuner = live_tuner

        def register(self, bound):
            backend = bound.backend
            graph = backend.artifact._flare_graph
            bindings = {"k0": ("id-k0", self._entry, self._live_tuner)}
            self.instances.setdefault("id-k0", {})[id(self._live_tuner)] = self._live_tuner
            self.forwards[id(bound)] = backend, graph, bindings

    cache = aot.ForwardCache.__new__(aot.ForwardCache)
    cache.root, cache.device, cache.headroom_bytes = tmp_path, "cpu", 0
    cache.options, cache.compile_workers, cache.lock_gpu_clocks = MappingProxyType({}), 1, False
    path = tmp_path / "decode.bin"

    # Default single-forward _warm retains _serialized_artifact without decompressing or re-saving,
    # while release_unchanged=True (used by _prepare_many before prepare_capture) drops it early.
    retained_frozen, retained_entry, retained_tuner, retained_saves = make_bound(
        frozen=True, cache_hit=True, serialized=b"invalid-compressed-bytes")
    selections = KernelSelections(retained_entry, retained_tuner)
    cache._warm(retained_frozen, path, name="decode", selections=selections, warmup=nullcontext)
    assert retained_saves == []
    assert retained_frozen.backend._serialized_artifact == b"invalid-compressed-bytes"

    frozen_bound, frozen_entry, frozen_tuner, frozen_saves = make_bound(
        frozen=True, cache_hit=True, serialized=b"invalid-compressed-bytes")
    selections = KernelSelections(frozen_entry, frozen_tuner)
    warmed = cache._warm(frozen_bound, path, name="decode", selections=selections, warmup=nullcontext,
                         release_unchanged=True)
    assert warmed is frozen_bound
    assert frozen_saves == []
    assert frozen_bound.backend._serialized_artifact is None
    assert frozen_bound.backend._tuning_frozen is True
    assert isinstance(frozen_bound.execution_identity, str) and len(frozen_bound.execution_identity) == 64
    with pytest.raises(RuntimeError, match="portable serialized artifact was already released"):
        aot.StorageBackend.serialize_compile_artifacts(frozen_bound.backend)

    # If reconciliation changed a live tuner's winner away from the bundled snapshot, _warm updates
    # the portable bundle and writes the reconciled forward to disk.
    reconciled_bound, reconciled_entry, reconciled_tuner, reconciled_saves = make_bound(
        frozen=True, cache_hit=True, bundled_block=64, live_block=128)
    selections = KernelSelections(reconciled_entry, reconciled_tuner)
    cache._warm(reconciled_bound, path, name="decode", selections=selections, warmup=nullcontext)
    assert reconciled_saves == [str(path)]
    updated_entry, _, _ = pickle.loads(zlib.decompress(reconciled_bound.backend._serialized_artifact))
    saved_result, = updated_entry.compiled_fw.result._triton_bundle.static_autotuners[0].kernel.compile_results
    assert saved_result._flare_config_state == {"block_size": 128}

    # An unfrozen forward updates _serialized_artifact and writes its frozen artifact to disk.
    unfrozen_bound, unfrozen_entry, unfrozen_tuner, unfrozen_saves = make_bound(
        frozen=False, cache_hit=True)
    selections = KernelSelections(unfrozen_entry, unfrozen_tuner)
    cache._warm(unfrozen_bound, path, name="decode", selections=selections, warmup=nullcontext)
    assert unfrozen_saves == [str(path)]
    assert unfrozen_bound.backend._tuning_frozen is True
    updated_entry, _, _ = pickle.loads(zlib.decompress(unfrozen_bound.backend._serialized_artifact))
    saved_result, = updated_entry.compiled_fw.result._triton_bundle.static_autotuners[0].kernel.compile_results
    assert saved_result._flare_config_state == {"block_size": 64}

    import torch._dynamo  # noqa: F401
    from torch._functorch._aot_autograd import aot_autograd_result
    from torch._functorch._aot_autograd.runtime_wrappers import SerializableCompiledFunction

    raw_entry = SimpleNamespace(compiled_fw=SimpleNamespace(result=SimpleNamespace(
        inductor_post_grad_graph_str="graph", cache_linemap=[], _triton_bundle=None)))
    raw_data = pickle.dumps((raw_entry, {}, {}))
    serializer = SerializableCompiledFunction(lambda *args: None, lambda: raw_entry)
    wrapper = SimpleNamespace(
        serialize=serializer.serialize,
        __wrapped__=serializer,
        serialize_fn=serializer.serialize_fn,
    )
    compress_calls = 0
    real_compress = aot.zlib.compress

    def count_compress(payload, level=-1):
        nonlocal compress_calls
        compress_calls += 1
        return real_compress(payload, level=level)

    with monkeypatch.context() as patch:
        patch.setattr(aot.runtime_patches, "apply", lambda: None)
        patch.setattr(aot_autograd_result, "deserialize_bundled_cache_entry", lambda item: wrapper)
        patch.setattr(aot.zlib, "compress", count_compress)
        _, retained_bytes = aot._load_inductor(raw_data, torch.device("cpu"))
        assert retained_bytes is not None and compress_calls == 1
        token = aot._RETAIN_SERIALIZED_ARTIFACT.set(False)
        try:
            _, skipped_bytes = aot._load_inductor(raw_data, torch.device("cpu"))
        finally:
            aot._RETAIN_SERIALIZED_ARTIFACT.reset(token)
        assert skipped_bytes is None and compress_calls == 1


def _serialization_rank(rank, store, root, outcome):
    from concurrent.futures import Future
    from contextlib import nullcontext
    from datetime import timedelta
    from types import MappingProxyType, SimpleNamespace

    import torch
    import torch.distributed as dist

    from ignition.preparation import aot, compile_pool, inductor_tuning

    load_retain_flags = []
    freeze_artifacts = {}
    saved_paths = []
    try:
        class Pool:
            workers = 1

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, forward, path):
                Path(path).write_text("compiled-unfrozen")
                future = Future()
                future.set_result(None)
                return future

        class ObservedBackend(aot.StorageBackend):
            @property
            def _tuning_frozen(self):
                return self._frozen_state

            @_tuning_frozen.setter
            def _tuning_frozen(self, value):
                self._frozen_state = value
                if value and getattr(self, "_observe_name", None) is not None:
                    freeze_artifacts[self._observe_name] = self._serialized_artifact

        def register_bound(self, bound):
            state = bound.artifact._artifacts
            self.forwards[id(bound)] = state.compiled_fn, state.compiler_graph, {}

        compile_pool.CompilePool = Pool
        compile_pool.compiler_resources = lambda *args, **kwargs: compile_pool.CompilerResources(
            1, 128 << 30, 16, 32 << 30, 8 << 30)
        inductor_tuning.InductorSelections.register = register_bound
        torch.cuda.current_stream = lambda device: SimpleNamespace(synchronize=lambda: None)
        aot._guard_code_cache = lambda path: nullcontext()
        aot.ForwardCache._allocation_sizes = lambda self, path: ()

        def load(self, forward, path, required=False, before_run=None):
            import pickle
            import zlib

            retain = aot._RETAIN_SERIALIZED_ARTIFACT.get()
            load_retain_flags.append((forward.name, required, retain))
            artifact = _Artifact(lambda owner: None)
            frozen = Path(path).read_text() == "cached-frozen"
            backend = ObservedBackend.__new__(ObservedBackend)
            backend.artifact = artifact._artifacts.compiled_fn.artifact
            backend._cuda_allocation_sizes = ()
            backend._observe_name = None
            backend._frozen_state = frozen
            entry = SimpleNamespace(compiled_fw=SimpleNamespace(result=SimpleNamespace(_triton_bundle=None)))
            raw = pickle.dumps((entry, {}, {}), protocol=pickle.HIGHEST_PROTOCOL)
            if not retain:
                backend._serialized_artifact = None
            elif aot._DEFER_PORTABLE_COMPRESSION.get() and not frozen:
                backend._serialized_artifact = aot._RawPortableArtifact(raw)
            else:
                backend._serialized_artifact = zlib.compress(raw, level=1)
            backend.signature = ()
            backend.inputs = []
            backend.invocations = []
            backend._observe_name = forward.name
            artifact._artifacts.compiled_fn = backend
            artifact._artifacts.compiler_graph = SimpleNamespace(
                _triton_bundle=None, device_idxs=(0,), source_code="def call(args):\n    return None\n")
            artifact.save_compiled_function = lambda target: saved_paths.append(Path(target).name)
            return aot.BoundForward(artifact, forward, True)

        aot.ForwardCache._load = load
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2,
                                timeout=timedelta(seconds=20))
        cache = aot.ForwardCache.__new__(aot.ForwardCache)
        cache.root, cache.device, cache.headroom_bytes = Path(root), "cpu", 0
        cache.options, cache.compile_workers, cache.lock_gpu_clocks = MappingProxyType({}), 1, False

        functions = {}
        for name in ("frozen_hit", "unfrozen_0", "unfrozen_1"):
            forward = _Forward("shared")
            forward.name = name
            functions[name] = forward
        signatures = {name: name for name in functions}
        frozen_path = cache._path(functions["frozen_hit"], name="frozen_hit.shared", signature="frozen_hit")
        unfrozen_0_path = cache._path(functions["unfrozen_0"], name="unfrozen_0.shared", signature="unfrozen_0")
        unfrozen_1_path = cache._path(functions["unfrozen_1"], name="unfrozen_1.shared", signature="unfrozen_1")
        if rank == 0:
            frozen_path.write_text("cached-frozen")
        dist.barrier()

        coordinator = SimpleNamespace(
            group=dist.group.WORLD, ranks=(0, 1), spans_hosts=False, hosts=("host", "host"),
            local_ranks=2, barrier=lambda key, ranks: dist.monitored_barrier(group=dist.group.WORLD))
        selected = cache._prepare_ranks(
            functions, signatures, coordinator, progress=None,
            materialize=lambda: None, warmup=nullcontext, before_run=None)
        outcome.put((rank, (
            {name: bound.cache_hit for name, bound in selected.items()},
            load_retain_flags,
            {name: value is not None for name, value in freeze_artifacts.items()},
            saved_paths,
            (unfrozen_0_path.name, unfrozen_1_path.name),
        )))
    except Exception:
        import traceback

        outcome.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_prepare_ranks_skips_non_owner_serialization_and_unchanged_frozen_resaves(tmp_path):
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "forwards"
    root.mkdir()
    outcome = context.Queue()
    processes = [context.Process(target=_serialization_rank, args=(rank, tmp_path / "store", root, outcome))
                 for rank in range(2)]
    for process in processes:
        process.start()
    try:
        results = dict(outcome.get(timeout=60) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        for rank, value in results.items():
            assert not isinstance(value, str), f"rank {rank} failed:\n{value}"
        hits_0, flags_0, freeze_0, saves_0, (unfrozen_0_name, unfrozen_1_name) = results[0]
        hits_1, flags_1, freeze_1, saves_1, _ = results[1]
        expected_hits = {"frozen_hit": True, "unfrozen_0": False, "unfrozen_1": False}
        assert hits_0 == expected_hits
        assert hits_1 == expected_hits
        # Variant ownership assigns frozen_hit -> rank 0, unfrozen_0 -> rank 1, unfrozen_1 -> rank 0.
        # Deferred restore skips portable serialization; required load retains it only on the owning rank;
        # _prepare_ranks drops it on rank 0 for frozen_hit before freeze because its static winners are unchanged.
        assert flags_0 == [
            ("frozen_hit", False, False),
            ("frozen_hit", True, True),
            ("unfrozen_0", True, False),
            ("unfrozen_1", True, True),
        ]
        assert flags_1 == [
            ("frozen_hit", True, False),
            ("unfrozen_0", True, True),
            ("unfrozen_1", True, False),
        ]
        assert freeze_0 == {"frozen_hit": False, "unfrozen_0": False, "unfrozen_1": True}
        assert freeze_1 == {"frozen_hit": False, "unfrozen_0": True, "unfrozen_1": False}
        assert saves_0 == [unfrozen_1_name]
        assert saves_1 == [unfrozen_0_name]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


def _probe_unlock_rank(rank, store, root, sync_dir, outcome, probe_rss=3200 << 20, available=88 << 30):
    import time
    from concurrent.futures import Future
    from contextlib import nullcontext
    from datetime import timedelta
    from types import MappingProxyType, SimpleNamespace

    import torch
    import torch.distributed as dist

    from ignition.preparation import aot, compile_pool, inductor_tuning

    messages = []
    pool_events = []
    synced = False
    try:
        class Pool:
            def __init__(self, device, **kwargs):
                self.workers = kwargs["workers"]
                self.max_workers = kwargs["max_workers"]
                self.min_workers = kwargs.get("min_workers")
                self.host_pools = kwargs["host_pools"]
                self.peak_worker_rss = 0
                pool_events.append(("open", self.workers, self.host_pools, self.min_workers))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pool_events.append(("close", self.workers, self.host_pools, self.min_workers, self.peak_worker_rss))

            def reconfigure(self, *, workers, max_workers, host_pools, min_workers=None):
                assert max_workers >= workers >= 1
                self.workers = workers
                self.max_workers = max_workers
                self.min_workers = min_workers
                self.host_pools = host_pools
                pool_events.append(("reconfigure", self.workers, self.host_pools, self.min_workers))

            def observe_worker_rss(self, rss_bytes, *, remaining=None):
                return self.workers

            def submit(self, forward, path):
                import pickle

                return self.submit_payload(pickle.dumps(forward), path)

            def submit_payload(self, payload, path):
                nonlocal synced
                import pickle

                if sync_dir is not None and self.host_pools == 4 and not synced:
                    synced = True
                    (Path(sync_dir) / f"rank_{rank}").write_bytes(b"1")
                    deadline = time.monotonic() + 20.0
                    while not all((Path(sync_dir) / f"rank_{peer}").exists() for peer in range(4)):
                        assert time.monotonic() < deadline, f"rank {rank} timed out waiting for 4-rank concurrency"
                        time.sleep(0.01)
                forward = pickle.loads(payload)
                Path(path).write_text(f"{forward.name} compiled by rank {rank}")
                future = Future()
                future.set_result(probe_rss)
                return future

        class Selections(inductor_tuning.InductorSelections):
            def register(self, bound):
                state = bound.artifact._artifacts
                self.forwards[id(bound)] = state.compiled_fn, state.compiler_graph, {}

            def freeze(self, bound):
                self.forwards[id(bound)][0]._tuning_frozen = True

        compile_pool.CompilePool = Pool
        compile_pool.compiler_resources = lambda *args, **kwargs: compile_pool.CompilerResources(
            8, available, 16, 32 << 30, 8 << 30)
        inductor_tuning.InductorSelections = Selections
        torch.cuda.current_stream = lambda device: SimpleNamespace(synchronize=lambda: None)
        aot._guard_code_cache = lambda path: nullcontext()
        aot.ForwardCache._allocation_sizes = lambda self, path: ()
        aot.ForwardCache._load = lambda self, forward, path, required=False, before_run=None: aot.BoundForward(
            _Artifact(lambda owner: None), forward, True)

        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=4,
                                timeout=timedelta(seconds=30))
        cache = aot.ForwardCache.__new__(aot.ForwardCache)
        cache.root, cache.device, cache.headroom_bytes = Path(root), "cpu", 0
        cache.options, cache.compile_workers, cache.lock_gpu_clocks = MappingProxyType({}), None, False

        functions = {}
        for index in range(21):
            name = "cached_hit" if index == 0 else f"f{index:02d}"
            forward = _Forward("shared")
            forward.name = name
            functions[name] = forward
        signatures = {name: name for name in functions}
        if rank == 0:
            cache._path(functions["cached_hit"], name="cached_hit.shared", signature="cached_hit").write_text(
                "cached-before-probe")
        dist.barrier()

        coordinator = SimpleNamespace(
            group=dist.group.WORLD, ranks=(0, 1, 2, 3), spans_hosts=False, hosts=("host",) * 4,
            local_ranks=4, barrier=lambda key, ranks: dist.monitored_barrier(group=dist.group.WORLD))
        selected = cache._prepare_ranks(
            functions, signatures, coordinator, progress=messages.append,
            materialize=None, warmup=nullcontext, before_run=None)
        outcome.put((rank, (
            sorted(selected),
            selected["cached_hit"].cache_hit,
            sum(not bound.cache_hit for bound in selected.values()),
            pool_events,
            messages,
        )))
    except Exception:
        import traceback

        outcome.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_rss_probe_wave_unlocks_zero_worker_ranks_for_concurrent_compilation(tmp_path):
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "forwards"
    sync_dir = tmp_path / "sync"
    root.mkdir()
    sync_dir.mkdir()
    outcome = context.Queue()
    processes = [
        context.Process(target=_probe_unlock_rank, args=(rank, tmp_path / "store", root, sync_dir, outcome))
        for rank in range(4)
    ]
    for process in processes:
        process.start()
    try:
        results = dict(outcome.get(timeout=90) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        for rank, value in results.items():
            assert not isinstance(value, str), f"rank {rank} failed:\n{value}"
        expected_names = sorted(["cached_hit"] + [f"f{index:02d}" for index in range(1, 21)])
        for rank in range(4):
            names, cached_hit, cold_count, pool_events, messages = results[rank]
            assert names == expected_names
            assert cached_hit is True
            assert cold_count == 20
            assert any("2 compiler workers; per rank [1, 1, 0, 0]" in msg for msg in messages)
            assert any("16 compiler workers; per rank [4, 4, 4, 4]" in msg for msg in messages)
            if rank in (0, 1):
                assert pool_events == [
                    ("open", 1, 2, None),
                    ("reconfigure", 4, 4, 1),
                    ("close", 4, 4, 1, 3200 << 20),
                ]
            else:
                assert pool_events == [("open", 4, 4, 1), ("close", 4, 4, 1, 3200 << 20)]
        compiled_texts = [path.read_text() for path in root.glob("*.bin") if path.read_text() != "cached-before-probe"]
        assert len(compiled_texts) == 20
        for rank in range(4):
            assert any(f"compiled by rank {rank}" in text for text in compiled_texts)
        assert not list(root.rglob("*.forward"))
        assert not list(root.rglob("*.tmp"))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


def test_heavy_rss_probe_shrinks_host_concurrency_and_closes_excess_pool(tmp_path):
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "forwards"
    root.mkdir()
    outcome = context.Queue()
    # 112 GiB available minus 8 GiB headroom = 104 GiB: initial 32 GiB estimate admits [1, 1, 1, 0],
    # and a 40 GiB probe peak (50 GiB with margin) shrinks the host budget to [1, 1, 0, 0] and closes rank 2's pool.
    processes = [
        context.Process(
            target=_probe_unlock_rank,
            args=(rank, tmp_path / "store", root, None, outcome, 40 << 30, 112 << 30),
        )
        for rank in range(4)
    ]
    for process in processes:
        process.start()
    try:
        results = dict(outcome.get(timeout=90) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        for rank, value in results.items():
            assert not isinstance(value, str), f"rank {rank} failed:\n{value}"
        for rank in range(4):
            _, cached_hit, cold_count, pool_events, messages = results[rank]
            assert cached_hit is True and cold_count == 20
            assert any("3 compiler workers; per rank [1, 1, 1, 0]" in msg for msg in messages)
            assert any("2 compiler workers; per rank [1, 1, 0, 0]" in msg for msg in messages)
            if rank in (0, 1):
                assert pool_events == [
                    ("open", 1, 3, None),
                    ("reconfigure", 1, 2, 1),
                    ("close", 1, 2, 1, 40 << 30),
                ]
            elif rank == 2:
                assert pool_events == [("open", 1, 3, None), ("close", 1, 3, None, 0)]
            else:
                assert pool_events == []
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


class _FakeJIT:
    def __init__(self, label):
        from collections import defaultdict
        import threading

        self.fn = lambda x: x
        self.__globals__ = {"__name__": __name__}
        self.used_global_vals = {"scale": 2}
        self.repr = lambda obj: f"jit<{label}>"
        self._hash_lock = threading.Lock()
        self.device_caches = defaultdict(self.create_binder)
        self.kernel = object()

    def create_binder(self):
        return {}


def test_shallow_snapshot_and_deferred_portable_compression_preserve_live_tuners_and_skip_zlib(tmp_path, monkeypatch):
    import copy
    import pickle
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    import torch
    import torch._dynamo  # noqa: F401
    from torch._functorch._aot_autograd import aot_autograd_result
    from torch._functorch._aot_autograd.runtime_wrappers import SerializableCompiledFunction
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner, StaticTritonCompileResult
    from ignition.preparation import aot, inductor_tuning

    jit_fn = _FakeJIT("triton_poi_fused_0")
    orig_fn_callable = jit_fn.fn
    orig_globals = jit_fn.__globals__
    orig_used = jit_fn.used_global_vals
    orig_repr = jit_fn.repr
    orig_hash_lock = jit_fn._hash_lock

    tuner = CachingAutotuner.__new__(CachingAutotuner)
    tuner.lock = threading.Lock()
    tuner._plugins = []
    tuner.benchmark_failure_reasons = {"k": "reason"}
    tuner.inductor_meta = {"kernel_name": "triton_poi_fused_0"}
    tuner.triton_meta = {"device": 0}
    tuner.fn = jit_fn
    live_config = SimpleNamespace(XBLOCK=128, num_warps=4)
    extra_config = SimpleNamespace(XBLOCK=64, num_warps=2)
    live_launcher = SimpleNamespace(config=live_config)
    tuner.launchers = [live_launcher]
    tuner._cached_launcher = live_launcher
    live_result = StaticTritonCompileResult(
        SimpleNamespace(n_regs=32, cubin_raw=b"cubin-payload-128", cubin_path=""),
        live_config,
        {},
        tuner.inductor_meta,
    )
    extra_result = StaticTritonCompileResult(
        SimpleNamespace(n_regs=24, cubin_raw=b"cubin-payload-64", cubin_path=""),
        extra_config,
        {},
        tuner.inductor_meta,
    )
    tuner.compile_results = [extra_result, live_result]

    # Direct copy.copy triggers CachingAutotuner.__getstate__ and fails while launchers are live.
    with pytest.raises(AssertionError, match="make_launchers"):
        copy.copy(tuner)

    entry = _StaticEntry(cache_key="k0", kernel_name="triton_poi_fused_0", kernel=tuner)
    snap = inductor_tuning._snapshot(entry, tuner)
    assert snap.kernel is not tuner
    assert snap.kernel.fn is not jit_fn
    assert tuner.launchers == [live_launcher]
    assert tuner._cached_launcher is live_launcher
    assert jit_fn.fn is orig_fn_callable
    assert jit_fn.__globals__ is orig_globals
    assert jit_fn.used_global_vals is orig_used
    assert jit_fn.repr is orig_repr
    assert jit_fn._hash_lock is orig_hash_lock
    assert tuner.compile_results[0].config is live_config

    unpickled_snap = pickle.loads(pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL))
    assert unpickled_snap.kernel.launchers == []
    assert unpickled_snap.kernel.compile_results[0]._flare_config_state == {"XBLOCK": 128, "num_warps": 4}
    assert unpickled_snap.kernel.compile_results[0].kernel.cubin_raw == b"cubin-payload-128"

    if not hasattr(tuner, "_prune_compile_results_to_launcher"):
        tuner._prune_compile_results_to_launcher = lambda launcher: None
    cfg_dict, cubin_digest = inductor_tuning._selected_digest(tuner)
    assert cfg_dict == {"XBLOCK": 128, "num_warps": 4}
    assert len(cubin_digest) == 32
    expected_id = inductor_tuning._execution_identity(
        "src-v1", "def call(args):\n    return None\n", 0,
        [("id-0", vars(live_config), b"cubin-payload-128")],
    )
    digest_id = inductor_tuning._execution_identity(
        "src-v1", "def call(args):\n    return None\n", 0,
        [("id-0", cfg_dict, cubin_digest)], prehashed=True,
    )
    assert digest_id == expected_id

    raw_entry = SimpleNamespace(compiled_fw=SimpleNamespace(result=SimpleNamespace(
        inductor_post_grad_graph_str="graph", cache_linemap=[], _triton_bundle=None,
        _flare_pool_sizes=(("cuda", 2048),))))
    raw_data = pickle.dumps((raw_entry, {}, {}), protocol=pickle.HIGHEST_PROTOCOL)
    serializer = SerializableCompiledFunction(lambda *args: None, lambda: raw_entry)
    wrapper = SimpleNamespace(
        serialize=serializer.serialize,
        __wrapped__=serializer,
        serialize_fn=serializer.serialize_fn,
    )

    def forbid_zlib(*args, **kwargs):
        raise AssertionError("deferred cold portable serialization must not invoke zlib")

    with monkeypatch.context() as patch:
        patch.setattr(aot.runtime_patches, "apply", lambda: None)
        patch.setattr(aot_autograd_result, "deserialize_bundled_cache_entry", lambda item: wrapper)
        patch.setattr(aot.zlib, "compress", forbid_zlib)
        patch.setattr(aot.zlib, "decompress", forbid_zlib)

        defer_token = aot._DEFER_PORTABLE_COMPRESSION.set(True)
        try:
            _, raw_portable = aot._load_inductor(raw_data, torch.device("cpu"), frozen=False)
        finally:
            aot._DEFER_PORTABLE_COMPRESSION.reset(defer_token)
        assert isinstance(raw_portable, aot._RawPortableArtifact)
        assert isinstance(raw_portable, bytes)

        bundle = _StaticBundle(static_autotuners=[entry])
        graph = SimpleNamespace(
            _triton_bundle=bundle,
            _flare_pool_sizes=(("cuda", 2048),),
            device_idxs=(0,),
            source_code="def call(args):\n    return None\n",
        )
        backend = aot.StorageBackend.__new__(aot.StorageBackend)
        backend.artifact = SimpleNamespace(_flare_graph=graph, inner_fn=SimpleNamespace(compiled_fn=SimpleNamespace()))
        backend._serialized_artifact = raw_portable
        backend._tuning_frozen = False
        backend._cuda_allocation_sizes = None
        backend.signature = ()
        backend.inputs = []
        backend.invocations = []
        bound = aot.BoundForward(SimpleNamespace(_artifacts=SimpleNamespace(compiled_fn=backend)), _Forward("shared"), False)

        selections = inductor_tuning.InductorSelections("cpu", source_identity="src-v1")
        selections.forwards[id(bound)] = backend, graph, {"k0": ("id-0", entry, _StaticTuner(_static_result(64, b"cubin-64")))}
        selections.freeze(bound)
        assert isinstance(backend._serialized_artifact, aot._RawPortableArtifact)
        assert backend._tuning_frozen is True

        wire = aot.StorageBackend.serialize_compile_artifacts(backend)
        decoded_artifact, _, _, _, decoded_frozen, decoded_sizes = pickle.loads(wire)
        assert type(decoded_artifact) is bytes
        assert decoded_frozen is True
        assert decoded_sizes == (2048,)
        frozen_entry, _, _ = pickle.loads(decoded_artifact)
        frozen_res, = frozen_entry.compiled_fw.result._triton_bundle.static_autotuners[0].kernel.compile_results
        assert frozen_res._flare_config_state == {"block_size": 64}

        TorchLikeArtifact = type(
            "TorchLikeArtifact",
            (),
            {
                "__module__": "torch._dynamo.aot_compile",
                "serialize": classmethod(lambda cls, instance: SimpleNamespace(
                    serialized_data=aot.StorageBackend.serialize_compile_artifacts(backend))),
            },
        )
        out_path = tmp_path / "async_saved.bin"
        futures = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            aot._save_compiled_async(TorchLikeArtifact(), out_path, executor, futures)
            backend._serialized_artifact = None
            assert len(futures) == 1
            for future in futures:
                future.result()
        assert out_path.read_bytes() == wire


def _post_warm_failure_rank(rank, store, root, fail_target, fail_step, outcome):
    from concurrent.futures import Future
    from contextlib import nullcontext
    from datetime import timedelta
    from types import MappingProxyType, SimpleNamespace

    import torch
    import torch.distributed as dist

    from ignition.preparation import aot, compile_pool, inductor_tuning

    try:
        class Pool:
            workers = 1

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, forward, path):
                Path(path).write_text("compiled")
                future = Future()
                future.set_result(None)
                return future

        class Selections(inductor_tuning.InductorSelections):
            def register(self, bound):
                state = bound.artifact._artifacts
                self.forwards[id(bound)] = state.compiled_fn, state.compiler_graph, {}

            def freeze(self, bound):
                if rank == 0 and fail_step == "freeze" and bound.owner.name == fail_target:
                    raise RuntimeError(f"injected freeze failure on {fail_target}")
                self.forwards[id(bound)][0]._tuning_frozen = True

        compile_pool.CompilePool = Pool
        compile_pool.compiler_resources = lambda *args, **kwargs: compile_pool.CompilerResources(
            1, 128 << 30, 16, 32 << 30, 8 << 30)
        inductor_tuning.InductorSelections = Selections
        torch.cuda.current_stream = lambda device: SimpleNamespace(synchronize=lambda: None)
        aot._guard_code_cache = lambda path: nullcontext()
        aot.ForwardCache._allocation_sizes = lambda self, path: ()

        def load(self, forward, path, required=False, before_run=None):
            artifact = _Artifact(lambda owner: None)

            def save(target):
                if rank == 0 and fail_step == "save" and forward.name == fail_target:
                    raise OSError(f"injected save failure on {fail_target}")

            artifact.save_compiled_function = save
            return aot.BoundForward(artifact, forward, True)

        aot.ForwardCache._load = load
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2,
                                timeout=timedelta(seconds=20))
        cache = aot.ForwardCache.__new__(aot.ForwardCache)
        cache.root, cache.device, cache.headroom_bytes = Path(root), "cpu", 0
        cache.options, cache.compile_workers, cache.lock_gpu_clocks = MappingProxyType({}), 1, False

        # Assign rank-local variant on fail_target for rank 0 so rank 0 owns its save_compiled_function.
        functions = {}
        for name in ("f0", "f1", "f2"):
            variant = f"rank{rank}" if name == fail_target else "shared"
            forward = _Forward(variant)
            forward.name = name
            functions[name] = forward
        signatures = {name: name for name in functions}
        coordinator = SimpleNamespace(
            group=dist.group.WORLD, ranks=(0, 1), spans_hosts=False, hosts=("host", "host"),
            local_ranks=2, barrier=lambda key, ranks: dist.monitored_barrier(group=dist.group.WORLD))
        try:
            cache._prepare_ranks(
                functions, signatures, coordinator, progress=None,
                materialize=lambda: None, warmup=nullcontext, before_run=None)
            outcome.put((rank, ("ok", None)))
        except RuntimeError as error:
            outcome.put((rank, ("error", str(error))))
    except Exception:
        import traceback

        outcome.put((rank, ("unexpected", traceback.format_exc())))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("fail_target,fail_step", [
    ("f0", "freeze"),
    ("f0", "save"),
    ("f2", "freeze"),
    ("f2", "save"),
])
def test_pipelined_post_warm_failures_propagate_across_all_ranks(tmp_path, fail_target, fail_step):
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "forwards"
    root.mkdir()
    outcome = context.Queue()
    processes = [
        context.Process(
            target=_post_warm_failure_rank,
            args=(rank, tmp_path / "store", root, fail_target, fail_step, outcome),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        results = dict(outcome.get(timeout=60) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        expected_detail = f"injected {fail_step} failure on {fail_target}"
        for rank in (0, 1):
            kind, detail = results[rank]
            assert kind == "error", f"rank {rank} did not raise RuntimeError: {detail}"
            assert "Inductor preparation failed across ranks" in detail
            assert expected_detail in detail
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


def _single_rank_host_worker(rank, world, store, base_dir, outcome):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch.distributed as dist

    from ignition.preparation import aot, compile_pool

    try:
        host_dir = base_dir / f"spark{rank}"
        os.chdir(host_dir)
        root = Path("forwards")
        compile_pool.CompilePool = _Pool
        _replace_gpu_preparation()
        compile_pool.compiler_resources = lambda *args, **kwargs: compile_pool.CompilerResources(
            1, 64 << 30, 16, 32 << 30, 8 << 30)
        aot.ForwardCache._load = lambda self, forward, path, required=False, before_run=None: aot.BoundForward(
            _Artifact(lambda owner: None), forward, True)
        aot.ForwardCache._allocation_sizes = lambda self, path: ()
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
        cache = aot.ForwardCache.__new__(aot.ForwardCache)
        cache.root, cache.device, cache.headroom_bytes, cache.options = root, "cpu", 0, aot.MappingProxyType({})
        cache.compile_workers = None
        cache.lock_gpu_clocks = False

        variant = "sample" if rank == 0 else "consume"
        functions = {name: _Forward(variant) for name in "abcdef"}
        for forward in functions.values():
            forward.rank = rank
        hosts = tuple(f"spark{index}" for index in range(world))
        coordinator = SimpleNamespace(
            group=dist.group.WORLD,
            ranks=tuple(range(world)),
            hosts=hosts,
            spans_hosts=True,
            local_ranks=1,
            barrier=lambda key, ranks: dist.monitored_barrier(group=dist.group.WORLD),
        )
        selected = cache._prepare_ranks(
            functions,
            {name: name for name in functions},
            coordinator,
            progress=None,
            materialize=lambda: None,
            warmup=nullcontext,
            before_run=None,
        )
        lent_exists = (root / "lent").exists()
        contents = sorted(path.read_text() for path in root.glob("*.bin"))
        outcome.put((rank, ("ok", sorted(selected), contents, lent_exists)))
    except Exception:
        import traceback

        outcome.put((rank, ("error", traceback.format_exc())))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_single_rank_hosts_lend_excess_compilations_across_hosts_and_return_artifacts(tmp_path):
    context = multiprocessing.get_context("fork")
    world = 3
    for rank in range(world):
        (tmp_path / f"spark{rank}" / "forwards").mkdir(parents=True)
    outcome = context.Queue()
    processes = [
        context.Process(target=_single_rank_host_worker, args=(rank, world, tmp_path / "store", tmp_path, outcome))
        for rank in range(world)
    ]
    for process in processes:
        process.start()
    try:
        results = dict(outcome.get(timeout=90) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        for rank, value in results.items():
            assert value[0] == "ok", f"rank {rank} failed:\n{value[1]}"
        # Rank 0 owns 6 sampling variants while ranks 1 and 2 own 3 consuming variants each ([6, 3, 3]).
        # Cross-host balancing levels all three single-rank hosts at 4 compilations each without lent files.
        _, names_0, compiled_0, lent_0 = results[0]
        assert names_0 == list("abcdef")
        assert len(compiled_0) == 12
        assert compiled_0.count("sample compiled by rank 0") == 4
        assert compiled_0.count("sample compiled by rank 1") == 1
        assert compiled_0.count("sample compiled by rank 2") == 1
        assert compiled_0.count("consume compiled by rank 1") == 3
        assert compiled_0.count("consume compiled by rank 2") == 3
        for rank in range(world):
            _, names, compiled, lent_exists = results[rank]
            assert names == list("abcdef")
            assert compiled == compiled_0
            assert lent_exists is False
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


def _batched_share_rank(rank, world, store, home, owned, known, outcome):
    import torch.distributed as dist

    from ignition.preparation import aot

    os.chdir(home)
    open_counts = {}
    real_open = Path.open

    def counting_open(self, mode="r", *args, **kwargs):
        if "r" in mode:
            key = self.as_posix()
            open_counts[key] = open_counts.get(key, 0) + 1
        return real_open(self, mode, *args, **kwargs)

    try:
        aot._SHARE_BATCH_BYTES = 1024
        Path.open = counting_open
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
        aot._share_artifacts(owned, dist.group.WORLD, tuple(range(world)), known_digests=known)
        Path.open = real_open
        files = {
            name: Path(name).read_bytes()
            for name in (
                "forwards/current.bin",
                "forwards/size_diff.bin",
                "forwards/b1.bin",
                "forwards/b2.bin",
                "forwards/b3.bin",
                "forwards/oversized.bin",
                "forwards/empty.bin",
            )
        }
        outcome.put((rank, ("ok", files, dict(open_counts))))
    except Exception:
        import traceback

        outcome.put((rank, ("error", traceback.format_exc())))
    finally:
        Path.open = real_open
        if dist.is_initialized():
            dist.destroy_process_group()


def test_batched_share_artifacts_skips_current_detects_size_mismatch_and_uses_known_digests(tmp_path):
    import hashlib

    context = multiprocessing.get_context("fork")
    world = 3
    homes = [tmp_path / f"host{rank}" for rank in range(world)]
    for home in homes:
        (home / "forwards").mkdir(parents=True)

    payloads = {
        "forwards/current.bin": b"already-current-across-all-hosts" * 8,
        "forwards/size_diff.bin": b"updated-size-payload" * 15,
        "forwards/b1.bin": b"batch-item-one-" * 40,
        "forwards/b2.bin": b"batch-item-two-" * 40,
        "forwards/b3.bin": b"batch-item-three-" * 35,
        "forwards/oversized.bin": b"X" * 2600,
        "forwards/empty.bin": b"",
    }
    sender_0 = ["forwards/current.bin", "forwards/size_diff.bin", "forwards/b1.bin", "forwards/b2.bin", "forwards/empty.bin"]
    sender_1 = ["forwards/b3.bin", "forwards/oversized.bin"]
    owned = [sender_0, sender_1, []]

    for path in sender_0:
        (homes[0] / path).write_bytes(payloads[path])
    for path in sender_1:
        (homes[1] / path).write_bytes(payloads[path])

    # Every host already holds current.bin with matching size and digest, so no rank requests or reads it
    # when rank 0 supplies known_digests.
    for rank in (1, 2):
        (homes[rank] / "forwards/current.bin").write_bytes(payloads["forwards/current.bin"])
    # Rank 2 holds a wrong-length size_diff.bin: size mismatch rejects it without reading its old bytes!
    (homes[2] / "forwards/size_diff.bin").write_bytes(b"short")

    known_0 = {path: hashlib.sha256(payloads[path]).hexdigest() for path in sender_0}
    outcome = context.Queue()
    processes = [
        context.Process(
            target=_batched_share_rank,
            args=(rank, world, tmp_path / "store", homes[rank], owned, known_0 if rank == 0 else None, outcome),
        )
        for rank in range(world)
    ]
    for process in processes:
        process.start()
    try:
        results = dict(outcome.get(timeout=90) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        for rank, value in results.items():
            assert value[0] == "ok", f"rank {rank} failed:\n{value[1]}"
            _, files, _ = value
            assert files == payloads
        _, _, opens_0 = results[0]
        _, _, opens_2 = results[2]
        # Rank 0 supplied known_digests: current.bin is wanted by nobody so rank 0 opens it 0 times,
        # and wanted files on rank 0 are opened only once (during slab readinto, never for hashing).
        assert opens_0.get("forwards/current.bin", 0) == 0
        assert opens_0.get("forwards/size_diff.bin", 0) == 1
        assert opens_0.get("forwards/b1.bin", 0) == 1
        # Rank 2 rejected size_diff.bin via stat().st_size != expected_size without opening the stale file.
        assert opens_2.get("forwards/size_diff.bin", 0) == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


def _share_io_failure_rank(rank, world, store, home, mode, outcome):
    from datetime import timedelta
    import torch.distributed as dist

    from ignition.preparation import aot

    os.chdir(home)
    real_open = Path.open
    try:
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world,
                                timeout=timedelta(seconds=20))
        if mode == "manifest_missing":
            owned = [["forwards/missing.bin"], [], []]
            known = None
        else:
            owned = [["forwards/truncated.bin"], [], []]
            known = {"forwards/truncated.bin": "0" * 64} if rank == 0 else None
            if rank == 0:
                def fail_sender_open(self, open_mode="r", *args, **kwargs):
                    if "r" in open_mode and self.as_posix().endswith("forwards/truncated.bin"):
                        raise OSError("sender disk read error")
                    return real_open(self, open_mode, *args, **kwargs)

                Path.open = fail_sender_open
        try:
            aot._share_artifacts(owned, dist.group.WORLD, tuple(range(world)), known_digests=known)
            outcome.put((rank, ("ok", None)))
        except RuntimeError as error:
            exists = (Path("forwards/truncated.bin")).exists()
            outcome.put((rank, ("runtime_error", str(error), exists)))
    except Exception:
        import traceback

        outcome.put((rank, ("unexpected", traceback.format_exc(), False)))
    finally:
        Path.open = real_open
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("mode,expected_msg", [
    ("manifest_missing", "compiled forward artifact manifest collection failed"),
    ("sender_read_error", "compiled forward artifact sharing failed"),
])
def test_share_artifacts_propagates_sender_io_errors_across_all_ranks_without_hanging(tmp_path, mode, expected_msg):
    context = multiprocessing.get_context("fork")
    world = 3
    homes = [tmp_path / f"host{rank}" for rank in range(world)]
    for home in homes:
        (home / "forwards").mkdir(parents=True)
    if mode == "sender_read_error":
        (homes[0] / "forwards/truncated.bin").write_bytes(b"payload-bytes")
    outcome = context.Queue()
    processes = [
        context.Process(target=_share_io_failure_rank, args=(rank, world, tmp_path / "store", homes[rank], mode, outcome))
        for rank in range(world)
    ]
    for process in processes:
        process.start()
    try:
        results = dict(outcome.get(timeout=45) for _ in processes)
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
        for rank in range(world):
            kind, detail, receiver_wrote = results[rank]
            assert kind == "runtime_error", f"rank {rank} did not raise RuntimeError: {detail}"
            assert expected_msg in detail
            if mode == "sender_read_error" and rank != 0:
                assert receiver_wrote is False
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


def test_byte_prefetcher_lazy_executor_and_prefetched_contextvar_scoping(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ignition.preparation import aot

    with aot._BytePrefetcher([]) as empty:
        assert empty._executor is None
        assert empty.pop("missing") is None

    first = tmp_path / "first.bin"
    nested = tmp_path / "nested.bin"
    first.write_bytes(b"outer-bytes")
    nested.write_bytes(b"nested-bytes")
    seen_streams = []

    class FakeArtifact:
        def guard_check(self, bound):
            return bound is owner

        def __call__(self, bound):
            raise aot._BindingsChecked

    def fake_load_compiled(file):
        payload = file.read()
        seen_streams.append(payload)
        if payload == b"outer-bytes":
            assert aot._PREFETCHED_ARTIFACT_BYTES.get() is None
            cache._load(owner, nested)
        return FakeArtifact()

    owner = SimpleNamespace()
    monkeypatch.setattr("torch.compiler.load_compiled_function", fake_load_compiled)
    monkeypatch.setattr("ignition.preparation.aot.release_unused", lambda **_: None)
    cache = aot.ForwardCache.__new__(aot.ForwardCache)
    cache.device = "cpu"
    cache.options = {}

    with aot._BytePrefetcher([("first", first), ("gone", tmp_path / "gone.bin")]) as prefetcher:
        assert prefetcher._executor is not None
        token = aot._PREFETCHED_ARTIFACT_BYTES.set(prefetcher.pop("first"))
        try:
            cache._load(owner, first)
            assert aot._PREFETCHED_ARTIFACT_BYTES.get() is None
        finally:
            aot._PREFETCHED_ARTIFACT_BYTES.reset(token)
        assert aot._PREFETCHED_ARTIFACT_BYTES.get() is None
        assert prefetcher.pop("gone") is None
    assert seen_streams == [b"outer-bytes", b"nested-bytes"]

    token = aot._PREFETCHED_ARTIFACT_BYTES.set(b"unconsumed")
    try:
        assert aot._PREFETCHED_ARTIFACT_BYTES.get() == b"unconsumed"
    finally:
        aot._PREFETCHED_ARTIFACT_BYTES.reset(token)
    assert aot._PREFETCHED_ARTIFACT_BYTES.get() is None


def _shared_host_write_rank(rank, world, store, home, hosts, owned, outcome):
    from datetime import timedelta
    import torch.distributed as dist
    from ignition.preparation import aot

    os.chdir(home)
    writes = []
    real_write = aot._write_shared_file

    def spy_write(path, slab, offset, size):
        writes.append(path)
        return real_write(path, slab, offset, size)

    try:
        aot._write_shared_file = spy_write
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world,
                                timeout=timedelta(seconds=20))
        try:
            aot._share_artifacts(owned, dist.group.WORLD, tuple(range(world)), hosts=hosts)
            files = {path.name: path.read_bytes() for path in sorted(Path("forwards").glob("*.bin"))}
            outcome.put((rank, ("ok", writes, files)))
        except RuntimeError as error:
            outcome.put((rank, ("runtime_error", str(error), writes)))
    except Exception:
        import traceback

        outcome.put((rank, ("unexpected", traceback.format_exc(), writes)))
    finally:
        aot._write_shared_file = real_write
        if dist.is_initialized():
            dist.destroy_process_group()


def test_share_artifacts_writes_once_per_shared_host_and_rejects_conflicting_digests(tmp_path):
    context = multiprocessing.get_context("fork")
    world = 4
    hosts = ("node0", "node0", "node1", "node1")
    node_dirs = {"node0": tmp_path / "node0", "node1": tmp_path / "node1"}
    for directory in node_dirs.values():
        (directory / "forwards").mkdir(parents=True)
    (node_dirs["node0"] / "forwards/from_0.bin").write_bytes(b"rank-0-payload")
    (node_dirs["node1"] / "forwards/from_3.bin").write_bytes(b"rank-3-payload")

    owned = [["forwards/from_0.bin"], [], [], ["forwards/from_3.bin"]]
    outcome = context.Queue()
    processes = [
        context.Process(
            target=_shared_host_write_rank,
            args=(rank, world, tmp_path / "store_ok", node_dirs[hosts[rank]], hosts, owned, outcome),
        )
        for rank in range(world)
    ]
    for process in processes:
        process.start()
    try:
        results = dict(outcome.get(timeout=45) for _ in processes)
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
        # Only rank 0 (leader of node0) writes from_3.bin, and only rank 2 (leader of node1) writes from_0.bin.
        assert results[0] == ("ok", ["forwards/from_3.bin"], {"from_0.bin": b"rank-0-payload", "from_3.bin": b"rank-3-payload"})
        assert results[1] == ("ok", [], {"from_0.bin": b"rank-0-payload", "from_3.bin": b"rank-3-payload"})
        assert results[2] == ("ok", ["forwards/from_0.bin"], {"from_0.bin": b"rank-0-payload", "from_3.bin": b"rank-3-payload"})
        assert results[3] == ("ok", [], {"from_0.bin": b"rank-0-payload", "from_3.bin": b"rank-3-payload"})
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    (node_dirs["node0"] / "forwards/conflict.bin").write_bytes(b"alpha")
    (node_dirs["node1"] / "forwards/conflict.bin").write_bytes(b"bravo-different")
    conflict_owned = [["forwards/conflict.bin"], [], ["forwards/conflict.bin"], []]
    outcome2 = context.Queue()
    processes2 = [
        context.Process(
            target=_shared_host_write_rank,
            args=(rank, world, tmp_path / "store_conflict", node_dirs[hosts[rank]], hosts, conflict_owned, outcome2),
        )
        for rank in range(world)
    ]
    for process in processes2:
        process.start()
    try:
        results2 = dict(outcome2.get(timeout=45) for _ in processes2)
        for process in processes2:
            process.join(timeout=20)
            assert process.exitcode == 0
        for rank in range(world):
            kind, detail, _ = results2[rank]
            assert kind == "runtime_error"
            assert "conflicting compiled forward artifacts" in detail
    finally:
        for process in processes2:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
