# SPDX-License-Identifier: Apache-2.0

"""Independent context groups retain their own collective selection checks."""

from types import SimpleNamespace

import pytest
import torch.distributed as dist

from ignition.preparation.collectives import CollectivePreparation


@pytest.mark.parametrize("rank,mismatch", [(0, 1), (8, 9)])
@pytest.mark.parametrize("event", ["pending", "collective_tuning", "tuning", "callback_pending", "callback_tuning"])
def test_selection_checks_follow_local_group_after_readiness_exchange(monkeypatch, rank, mismatch, event):
    owner = object.__new__(CollectivePreparation)
    owner.group = None
    owner.ranks = tuple(range(16))
    owner.operation_ranks = {}
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    name = "layer.attention.combine"
    states = []
    for peer in owner.ranks:
        ranks = tuple(range(peer // 8 * 8, peer // 8 * 8 + 8))
        pending = ((name, ranks),)
        contribution = SimpleNamespace(key=name, ranks=ranks)
        states.append({
            "pending": (False, pending, None, None, (), None),
            "collective_tuning": (False, (), None, None, (), contribution),
            "tuning": (False, (), None, None, (contribution,), None),
            "callback_pending": (pending, None),
            "callback_tuning": ((), contribution),
        }[event])
    owner._remember_operation_ranks(states)

    def gather(output, local, *, group):
        for peer in owner.ranks:
            output[peer] = {name: ("comm", '"different"' if peer == mismatch else '"same"')}

    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="kernel selections differ"):
        owner.agree_selections([{"name": name, "component": "comm", "config": "same"}])


@pytest.mark.parametrize("rank", [0, 8])
def test_selection_checks_allow_different_choices_between_context_groups(monkeypatch, rank):
    owner = object.__new__(CollectivePreparation)
    owner.group = None
    owner.ranks = tuple(range(16))
    owner.operation_ranks = {}
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    name = "layer.attention.combine"
    states = [
        (False, ((name, tuple(range(peer // 8 * 8, peer // 8 * 8 + 8))),), None, None, (), None)
        for peer in owner.ranks
    ]
    owner._remember_operation_ranks(states)

    def gather(output, local, *, group):
        for peer in owner.ranks:
            output[peer] = {name: ("comm", '"group-a"' if peer < 8 else '"group-b"')}

    monkeypatch.setattr(dist, "all_gather_object", gather)
    owner.agree_selections([{"name": name, "component": "comm",
                             "config": "group-a" if rank < 8 else "group-b"}])


def test_prepare_raises_on_deadlocked_collective_keys(monkeypatch):
    owner = object.__new__(CollectivePreparation)
    owner.group = None
    owner.ranks = (0, 1)
    owner.operation_ranks = {}
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    req0 = SimpleNamespace(key="op.a", step="catalog", ranks=(0, 1))
    req1 = SimpleNamespace(key="op.b", step="catalog", ranks=(0, 1))

    class FakeJob:
        def __init__(self):
            self.closed = False

        def advance(self, **kwargs):
            return SimpleNamespace(
                done=False, ready_collectives=(), ready_cache=None,
                ready_tuning=(), ready_collective_tuning=req0,
            )

        def close(self):
            self.closed = True

    job = FakeJob()
    session = SimpleNamespace(begin=lambda requests, autotune=None: job)
    owner._exchange_progress = lambda local: [
        local,
        (False, (), None, None, (), req1),
    ]
    with pytest.raises(RuntimeError, match="collective preparation deadlocked across ranks:.*op\\.a.*op\\.b"):
        owner.prepare(session, ())
    assert job.closed


def test_prepare_allows_one_subgroup_to_advance_while_other_waits(monkeypatch):
    owner = object.__new__(CollectivePreparation)
    owner.group = None
    owner.ranks = (0, 1, 2, 3)
    owner.operation_ranks = {}
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    req_all = SimpleNamespace(key="tp.all", step="catalog", ranks=(0, 1, 2, 3))
    req_sub = SimpleNamespace(key="dcp.sub", step="catalog", ranks=(2, 3))

    class FakeJob:
        def __init__(self):
            self.calls = 0

        def advance(self, **kwargs):
            self.calls += 1
            if kwargs.get("collective_tuning") is not None:
                return SimpleNamespace(
                    done=True, ready_collectives=(), ready_cache=None,
                    ready_tuning=(), ready_collective_tuning=None,
                )
            return SimpleNamespace(
                done=False, ready_collectives=(), ready_cache=None,
                ready_tuning=(), ready_collective_tuning=req_all,
            )

        def result(self):
            return "ok"

        def close(self):
            pass

    job = FakeJob()
    session = SimpleNamespace(begin=lambda requests, autotune=None: job)
    round_num = {"value": 0}

    def exchange(local):
        round_num["value"] += 1
        if round_num["value"] == 1:
            # Ranks 0,1 wait on TP collective while ranks 2,3 advance their DCP subgroup.
            return [
                local,
                (False, (), None, None, (), req_all),
                (False, (), None, None, (), req_sub),
                (False, (), None, None, (), req_sub),
            ]
        if round_num["value"] == 2:
            # All 4 ranks now match on req_all.
            return [local] * 4
        return [(True, (), None, None, (), None)] * 4

    owner._exchange_progress = exchange
    assert owner.prepare(session, ()) == "ok"


def test_collective_tuning_polling_strips_heavy_payload_and_exchanges_in_subgroup(monkeypatch):
    import pickle
    from ignition.preparation.collectives import _CollectiveTuningHeader

    owner = object.__new__(CollectivePreparation)
    owner.group = "tp4"
    owner.ranks = (0, 1, 2, 3)
    owner.control_groups = {(0, 1, 2, 3): "tp4", (2, 3): "dcp2"}
    owner.operation_ranks = {}
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    heavy_payload = {"candidates": [f"cand-{i:04d}-" + ("x" * 200) for i in range(512)]}
    req_all = SimpleNamespace(key="tp.all", step="catalog", ranks=(0, 1, 2, 3), payload=heavy_payload)
    header_all = _CollectiveTuningHeader("tp.all", (0, 1, 2, 3), "catalog")
    header_sub = _CollectiveTuningHeader("dcp.sub", (2, 3), "queued/1")
    received_authorizations = []

    class FakeJob:
        def advance(self, **kwargs):
            received_authorizations.append(kwargs)
            if kwargs.get("collective_tuning") is not None:
                return SimpleNamespace(
                    done=True, ready_collectives=(), ready_cache=None,
                    ready_tuning=(), ready_collective_tuning=None,
                )
            return SimpleNamespace(
                done=False, ready_collectives=(), ready_cache=None,
                ready_tuning=(), ready_collective_tuning=req_all,
            )

        def result(self):
            return "ok"

        def close(self):
            pass

    polled_sizes = []
    subgroup_calls = []

    def exchange_progress(local):
        polled_sizes.append(len(pickle.dumps(local)))
        assert isinstance(local[5], _CollectiveTuningHeader) or local[0]
        if len(polled_sizes) <= 3:
            # Ranks 0,1 wait on giant TP4 catalog while ranks 2,3 advance their DCP2 subgroup.
            return [
                local,
                (False, (), None, None, (), header_all),
                (False, (), None, None, (), header_sub),
                (False, (), None, None, (), header_sub),
            ]
        if len(polled_sizes) == 4:
            return [local] * 4
        return [(True, (), None, None, (), None)] * 4

    def exchange_subgroup(value, ranks):
        subgroup_calls.append((ranks, value))
        return tuple(value for _ in ranks)

    owner._exchange_progress = exchange_progress
    owner._exchange_subgroup_payload = exchange_subgroup
    job = FakeJob()
    session = SimpleNamespace(begin=lambda requests, autotune=None: job)

    assert owner.prepare(session, ()) == "ok"
    # Every progress-polling packet stayed well under the 2040-byte fast-path limit (no 100 KiB catalog broadcast).
    assert max(polled_sizes) < 512
    assert len(pickle.dumps(req_all)) > 100_000
    # Heavy catalog payload was exchanged exactly once, only after all 4 ranks matched.
    assert subgroup_calls == [((0, 1, 2, 3), req_all)]
    assert received_authorizations[4]["collective_tuning"] == (req_all,) * 4


def test_collective_tuning_requires_matching_step_before_authorization(monkeypatch):
    owner = object.__new__(CollectivePreparation)
    owner.group = None
    owner.ranks = (0, 1)
    owner.operation_ranks = {}
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    req_step0 = SimpleNamespace(key="op.a", step="queued/0", ranks=(0, 1))
    req_step1 = SimpleNamespace(key="op.a", step="queued/1", ranks=(0, 1))

    class FakeJob:
        def __init__(self):
            self.closed = False

        def advance(self, **kwargs):
            return SimpleNamespace(
                done=False, ready_collectives=(), ready_cache=None,
                ready_tuning=(), ready_collective_tuning=req_step0,
            )

        def close(self):
            self.closed = True

    job = FakeJob()
    session = SimpleNamespace(begin=lambda requests, autotune=None: job)
    owner._exchange_progress = lambda local: [
        local,
        (False, (), None, None, (), req_step1),
    ]
    with pytest.raises(RuntimeError, match="collective preparation deadlocked across ranks"):
        owner.prepare(session, ())
    assert job.closed


def _gloo_subgroup_worker(rank, rendezvous, result_queue):
    import time
    from datetime import timedelta
    from b12x.preparation.types import CollectiveTuningRequirement, FrozenMapping

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=30))
    try:
        coordinator = CollectivePreparation(dist.group.WORLD, timeout=timedelta(seconds=30))
        subgroups = [dist.new_group(ranks=[0, 1], backend="gloo"), dist.new_group(ranks=[2, 3], backend="gloo")]
        coordinator.register_group((0, 1), subgroups[0])
        coordinator.register_group((2, 3), subgroups[1])

        heavy = tuple(f"cand-{i:04d}-" + ("x" * 200) for i in range(600))
        req_tp4 = CollectiveTuningRequirement(
            "tp4.reduce", (0, 1, 2, 3), "catalog", FrozenMapping({"rank": rank, "candidates": heavy}),
        )

        class FakeJob:
            def __init__(self):
                self.sub_step = 0
                self.sub_steps_total = 0 if rank < 2 else 150
                self.tp4_done = False

            def advance(self, **kwargs):
                ct = kwargs.get("collective_tuning")
                if self.sub_step < self.sub_steps_total:
                    if ct is not None:
                        assert len(ct) == 2
                        assert [item.payload["rank"] for item in ct] == [2, 3]
                        assert all(item.step == f"queued/{self.sub_step}" for item in ct)
                        self.sub_step += 1
                        ct = None
                    if self.sub_step < self.sub_steps_total:
                        req = CollectiveTuningRequirement(
                            "dcp2.merge", (2, 3), f"queued/{self.sub_step}",
                            FrozenMapping({"rank": rank, "step": self.sub_step}),
                        )
                        return SimpleNamespace(
                            done=False, ready_collectives=(), ready_cache=None,
                            ready_tuning=(), ready_collective_tuning=req,
                        )
                if not self.tp4_done:
                    if ct is not None:
                        assert len(ct) == 4
                        assert [item.payload["rank"] for item in ct] == [0, 1, 2, 3]
                        assert len(ct[0].payload["candidates"]) == 600
                        self.tp4_done = True
                        return SimpleNamespace(
                            done=True, ready_collectives=(), ready_cache=None,
                            ready_tuning=(), ready_collective_tuning=None,
                        )
                    return SimpleNamespace(
                        done=False, ready_collectives=(), ready_cache=None,
                        ready_tuning=(), ready_collective_tuning=req_tp4,
                    )
                return SimpleNamespace(
                    done=True, ready_collectives=(), ready_cache=None,
                    ready_tuning=(), ready_collective_tuning=None,
                )

            def result(self):
                return f"ok-{rank}"

            def close(self):
                pass

        t0 = time.perf_counter()
        out = coordinator.prepare(SimpleNamespace(begin=lambda *a, **k: FakeJob()), ())
        elapsed = time.perf_counter() - t0
        result_queue.put((rank, out, elapsed))
    finally:
        dist.destroy_process_group()


def test_collective_tuning_subgroup_exchange_over_gloo(tmp_path):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    rendezvous = (tmp_path / "gloo-subgroup-exchange").as_uri()
    mp.spawn(_gloo_subgroup_worker, args=(rendezvous, queue), nprocs=4, join=True)
    results = sorted(queue.get(timeout=5) for _ in range(4))
    assert [r[:2] for r in results] == [(i, f"ok-{i}") for i in range(4)]
    assert max(r[2] for r in results) < 5.0
