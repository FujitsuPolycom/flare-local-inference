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
