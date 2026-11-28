# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kernel ownership shards Inductor tuning and skips resolved warmup passes."""

import pickle
import threading
from contextlib import nullcontext
from threading import BrokenBarrierError
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation import clocks, inductor_tuning


def _result(name, index, cubin_dir):
    """Static compile result for one candidate, with its cubin bytes on disk."""
    from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult

    cubin = f'{name}-candidate-{index}'.encode()
    path = cubin_dir / f'{name}-{index}.cubin'
    path.write_bytes(cubin)
    result = StaticTritonCompileResult(SimpleNamespace(n_regs=0),
                                       SimpleNamespace(XBLOCK=8 * (index + 1)), {},
                                       {'kernel_name': name})
    result.kernel = SimpleNamespace(cubin_raw=cubin, cubin_path=str(path),
                                    hash=f'{name}:{index}', name=name)
    return result


class StubTuner:
    """Stand-in for CachingAutotuner: compiles, loads, and times fixed candidates."""

    def __init__(self, name, cubin_dir, *, candidates=3, preference='first', coordinate_descent=False):
        self.inductor_meta = {'kernel_name': name, 'coordinate_descent_tuning': coordinate_descent}
        self.triton_meta = {'device': 0}
        self.configs = list(range(candidates))
        self.results = {index: _result(name, index, cubin_dir) for index in self.configs}
        self.preference = preference
        self.bench_calls = 0
        self.compile_results = []
        self.launchers = []
        self._cached_launcher = None

    def _precompile_worker(self):
        if self.compile_results:
            return
        self.compile_results = [self._precompile_config(index) for index in self.configs]
        self.configs = None

    def _precompile_config(self, index):
        return self.results[index]

    def _make_launchers(self):
        if len(self.launchers) == len(self.compile_results):
            return
        self.launchers = [SimpleNamespace(config=result.config) for result in self.compile_results]

    def _prune_compile_results_to_launcher(self, launcher):
        self.compile_results = [result for result in self.compile_results
                                if result.config is launcher.config]

    def bench(self, launcher):
        self.bench_calls += 1
        index = vars(launcher.config)['XBLOCK'] // 8 - 1
        return float(index if self.preference == 'first' else -index)

    def autotune_to_one_config(self, *args, **kwargs):
        timings = [(self.bench(launcher), launcher) for launcher in self.launchers]
        self.launchers = [min(timings)[1]]
        return self.launchers[0]

    def _combo_sequential_autotune(self, *args, **kwargs):
        return self.autotune_to_one_config(*args, **kwargs)

    def coordinate_descent_tuning(self, launcher, *args, **kwargs):
        self.bench(launcher)
        return launcher


class StubForward:
    """Runs its kernels the way CachingAutotuner.run selects a launcher."""

    def __init__(self, bindings):
        self.bindings = bindings
        self.tuners = [tuner for _, _, tuner in bindings.values()]
        self.owner = SimpleNamespace()
        self.calls = 0

    def __call__(self):
        self.calls += 1
        for tuner in self.tuners:
            if len(tuner.launchers) != 1:
                if not tuner.launchers:
                    tuner._precompile_worker()
                    tuner._make_launchers()
                if len(tuner.launchers) > 1:
                    tuner.autotune_to_one_config()
            launcher = tuner.launchers[0]
            if (not getattr(launcher.config, 'found_by_coordesc', False)
                    and tuner.inductor_meta.get('coordinate_descent_tuning', False)):
                tuner.launchers = [tuner.coordinate_descent_tuning(launcher)]


def _bindings(cubin_dir, names, *, candidates=3, preference='first', coordinate_descent=False):
    return {f'key-{name}': (name, object(), StubTuner(name, cubin_dir, candidates=candidates,
                                                      preference=preference,
                                                      coordinate_descent=coordinate_descent))
            for name in names}


def _frozen_tuner(name, cubin_dir):
    """Tuner state restored from a frozen artifact: one launcher, winner in place."""
    tuner = StubTuner(name, cubin_dir, candidates=3)
    tuner._precompile_worker()
    tuner._make_launchers()
    tuner.launchers = tuner.launchers[:1]
    tuner._prune_compile_results_to_launcher(tuner.launchers[0])
    return tuner


def _register(selections, forward, *, frozen=False):
    """Bind a forward the way InductorSelections.register does without a graph."""
    selections.forwards[id(forward)] = SimpleNamespace(_tuning_frozen=frozen), None, forward.bindings
    for identity, _, tuner in forward.bindings.values():
        selections.instances.setdefault(identity, {})[id(tuner)] = tuner
        if identity in selections.winners:
            inductor_tuning._install(tuner, selections.winners[identity])


def _selections(*inventories):
    """One selections object per simulated rank, each gathering the same inventories."""
    selections = inductor_tuning.InductorSelections('cpu', coordinator=SimpleNamespace(
        ranks=list(range(len(inventories))), group=None))
    selections._gather = lambda value: [(sorted(item), sorted(selections.winners)) for item in inventories]
    return selections


class ReconcileProbe(inductor_tuning.InductorSelections):
    """Counts reconciliations so a warmup that skips tuning is observable."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconciles = 0

    def _reconcile(self, bindings):
        self.reconciles += 1
        super()._reconcile(bindings)


class Cluster:
    """Runs rank callables concurrently and implements their collectives."""

    def __init__(self, count):
        self.ranks = list(range(count))
        self.coordinator = SimpleNamespace(ranks=self.ranks, group=None)
        self.barrier = threading.Barrier(count, timeout=60)
        self.values = [None] * count
        self.payloads = {}
        self.lock = threading.Lock()
        self.local = threading.local()

    def gather(self, index):
        def gather(value):
            with self.lock:
                self.values[index] = value
            self.barrier.wait()
            snapshot = list(self.values)
            # Separate rounds so a fast rank cannot overwrite the next round's slot.
            self.barrier.wait()
            return snapshot
        return gather

    def broadcast(self, message, src, group=None):
        index = self.local.rank
        with self.lock:
            if self.ranks[index] == src:
                self.payloads[src] = message[0]
        self.barrier.wait()
        message[0] = self.payloads[src]
        self.barrier.wait()

    def install(self, monkeypatch):
        import torch.distributed as dist

        monkeypatch.setattr(dist, 'get_rank', lambda group=None: self.local.rank)
        monkeypatch.setattr(dist, 'broadcast_object_list', self.broadcast)

    def run(self, targets):
        errors = [None] * len(targets)

        def body(index):
            self.local.rank = index
            try:
                targets[index]()
            except BaseException as error:
                errors[index] = error

        threads = [threading.Thread(target=body, args=(index,)) for index in range(len(targets))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert not any(thread.is_alive() for thread in threads), 'rank simulation deadlocked'
        failed = [error for error in errors if error is not None
                  and not isinstance(error, BrokenBarrierError)]
        if failed:
            raise failed[0]
        assert not any(errors), f'rank collectives broke: {errors}'


@pytest.fixture(autouse=True)
def stubbed_kernel_io(monkeypatch):
    """Stub kernels record their cubin file, and CPU runs have no device stream."""
    from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult

    monkeypatch.setattr(StaticTritonCompileResult, 'reload_cubin_path', lambda self: None)
    monkeypatch.setattr(torch.cuda, 'current_stream',
                        lambda device: SimpleNamespace(synchronize=lambda: None))


def test_single_rank_owns_every_unresolved_kernel():
    assert _selections({'k0', 'k1'})._assign_owners({'k0', 'k1'}) == {'k0': 0, 'k1': 0}


def test_shared_kernels_shard_round_robin_across_two_ranks():
    shared = {'k0', 'k1', 'k2', 'k3'}
    owners = _selections(shared, shared)._assign_owners(shared)
    assert owners == {'k0': 0, 'k1': 1, 'k2': 0, 'k3': 1}


def test_shared_kernels_shard_round_robin_across_four_ranks():
    shared = {f'k{index}' for index in range(6)}
    owners = _selections(shared, shared, shared, shared)._assign_owners(shared)
    assert owners == {'k0': 0, 'k1': 1, 'k2': 2, 'k3': 3, 'k4': 0, 'k5': 1}


def test_rank_private_kernels_go_to_their_only_holder():
    owners = _selections({'k0', 'k1'}, {'k1', 'k2'})._assign_owners({'k0', 'k1', 'k2'})
    assert owners == {'k0': 0, 'k1': 1, 'k2': 1}


def test_resolved_kernels_consume_no_shard():
    selections = _selections({'k0', 'k1'}, {'k0', 'k1'})
    selections.winners['k0'] = b'resolved'
    assert selections._assign_owners({'k0', 'k1'}) == {'k1': 0}


def test_resolved_forward_reuses_winners_without_tuning(tmp_path):
    selections = ReconcileProbe('cpu')
    tuning = StubForward(_bindings(tmp_path, ['k0', 'k1']))
    _register(selections, tuning)
    selections.warm(tuning, nullcontext)
    assert tuning.calls == 2
    assert set(selections.winners) == {'k0', 'k1'}
    assert selections.reconciles == 1
    benched = sum(tuner.bench_calls for tuner in tuning.tuners)
    assert benched == 6

    resolved = StubForward(_bindings(tmp_path, ['k0', 'k1']))
    _register(selections, resolved)
    assert all(len(tuner.launchers) == 1 for tuner in resolved.tuners)
    selections.warm(resolved, nullcontext)
    assert resolved.calls == 1
    assert selections.reconciles == 1
    assert sum(tuner.bench_calls for tuner in resolved.tuners) == 0


def test_frozen_forward_warms_once_and_records_winners(tmp_path):
    selections = ReconcileProbe('cpu')
    forward = StubForward({f'key-{name}': (name, object(), _frozen_tuner(name, tmp_path))
                           for name in ('k0', 'k1')})
    _register(selections, forward, frozen=True)
    selections.warm(forward, nullcontext)
    assert forward.calls == 1
    assert selections.reconciles == 0
    assert set(selections.winners) == {'k0', 'k1'}
    for identity, _, tuner in forward.bindings.values():
        winner = pickle.loads(selections.winners[identity])
        assert winner._flare_config_state == vars(tuner.launchers[0].config)
        assert winner.kernel.cubin_raw == tuner.compile_results[0].kernel.cubin_raw
    assert sum(tuner.bench_calls for tuner in forward.tuners) == 0


def test_shared_kernels_benchmark_on_one_rank_each(tmp_path, monkeypatch):
    names = ['k0', 'k1', 'k2', 'k3']
    cluster = Cluster(2)
    cluster.install(monkeypatch)
    selections, forwards = [], []
    for index in range(2):
        choice = inductor_tuning.InductorSelections('cpu', coordinator=cluster.coordinator)
        choice._gather = cluster.gather(index)
        forward = StubForward(_bindings(tmp_path, names,
                                        preference='first' if index == 0 else 'last'))
        _register(choice, forward)
        selections.append(choice)
        forwards.append(forward)
    cluster.run([lambda index=index: selections[index].warm(forwards[index], nullcontext)
                 for index in range(2)])

    trials = [{identity: tuner.bench_calls for identity, _, tuner in forward.bindings.values()}
              for forward in forwards]
    assert trials[0] == {'k0': 3, 'k1': 0, 'k2': 3, 'k3': 0}
    assert trials[1] == {'k0': 0, 'k1': 3, 'k2': 0, 'k3': 3}
    chosen = [{identity: vars(tuner.compile_results[0].config)
               for identity, _, tuner in forward.bindings.values()} for forward in forwards]
    assert chosen[0] == chosen[1]
    assert chosen[0] == {'k0': {'XBLOCK': 8}, 'k1': {'XBLOCK': 24},
                         'k2': {'XBLOCK': 8}, 'k3': {'XBLOCK': 24}}


def test_peer_owned_kernel_selects_without_benchmarking_or_clock_lock(tmp_path, monkeypatch):
    def forbid_lock(uuid):
        raise AssertionError('a rank without trials locked clocks')

    monkeypatch.setattr(clocks, 'locked_gpu_clocks', forbid_lock)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda group=None: 1)
    selections = inductor_tuning.InductorSelections(
        'cpu', coordinator=SimpleNamespace(ranks=[0, 1], group=None), lock_gpu_clocks=True)
    forward = StubForward(_bindings(tmp_path, ['k0'], coordinate_descent=True))
    _register(selections, forward)
    selections._active_owners = {'k0': 0}
    with selections._tuning(forward.bindings, frozen=False):
        forward()
    assert forward.calls == 1
    (tuner,) = forward.tuners
    assert tuner.bench_calls == 0
    assert len(tuner.launchers) == 1
    assert vars(inductor_tuning._selected(tuner).config) == {'XBLOCK': 8}


def test_mixed_frozen_and_unfrozen_ranks_share_adopted_winners_without_deadlock(tmp_path, monkeypatch):
    cluster = Cluster(2)
    cluster.install(monkeypatch)
    selections = []
    for index in range(2):
        choice = inductor_tuning.InductorSelections('cpu', coordinator=cluster.coordinator)
        choice._gather = cluster.gather(index)
        selections.append(choice)
    # Rank 0 restores a frozen variant holding k_shared; rank 1 warms an unfrozen variant holding k_shared and k1.
    rank0_forward = StubForward({'key-k_shared': ('k_shared', object(), _frozen_tuner('k_shared', tmp_path))})
    rank0_forward.owner.compilation_variant = 'sample'
    _register(selections[0], rank0_forward, frozen=True)
    rank1_forward = StubForward(_bindings(tmp_path, ['k_shared', 'k1'], preference='last'))
    rank1_forward.owner.compilation_variant = 'consume'
    _register(selections[1], rank1_forward, frozen=False)

    cluster.run([lambda index=index, forward=(rank0_forward if index == 0 else rank1_forward):
                 selections[index].warm(forward, nullcontext) for index in range(2)])
    assert selections[0].winners.keys() == selections[1].winners.keys() == {'k_shared', 'k1'}
    # k_shared came from rank 0's frozen artifact (XBLOCK=8) without benchmarking on rank 1; rank 1 tuned only k1 (XBLOCK=24).
    assert rank1_forward.bindings['key-k_shared'][2].bench_calls == 0
    assert rank1_forward.bindings['key-k1'][2].bench_calls == 3
    assert vars(inductor_tuning._selected(rank1_forward.bindings['key-k_shared'][2]).config) == {'XBLOCK': 8}
    assert vars(inductor_tuning._selected(rank1_forward.bindings['key-k1'][2]).config) == {'XBLOCK': 24}


def test_consecutive_single_kernel_forwards_alternate_owning_ranks(tmp_path, monkeypatch):
    cluster = Cluster(2)
    cluster.install(monkeypatch)
    selections = [inductor_tuning.InductorSelections('cpu', coordinator=cluster.coordinator) for _ in range(2)]
    for index in range(2):
        selections[index]._gather = cluster.gather(index)
    first = [StubForward(_bindings(tmp_path, ['k_a'])) for _ in range(2)]
    second = [StubForward(_bindings(tmp_path, ['k_b'])) for _ in range(2)]
    for index in range(2):
        _register(selections[index], first[index])
        _register(selections[index], second[index])
    cluster.run([lambda index=index: (selections[index].warm(first[index], nullcontext),
                                      selections[index].warm(second[index], nullcontext))
                 for index in range(2)])
    assert [first[index].tuners[0].bench_calls for index in range(2)] == [3, 0]
    assert [second[index].tuners[0].bench_calls for index in range(2)] == [0, 3]
