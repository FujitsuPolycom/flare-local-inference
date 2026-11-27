# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Guard retirement and composition races defer full heap scans to phase boundaries."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from b12x.preparation import FrozenMapping, PreparedCall

from ignition.preparation import aot, capture, composition_tuning


def test_retire_after_capture_collects_young_generation_and_is_idempotent(monkeypatch):
    generations = []
    monkeypatch.setattr(aot, "release_unused", lambda *, generation=2: generations.append(generation))

    backend = SimpleNamespace(name="backend")
    guarded = aot._GuardedForwardCall(function=lambda owner: None, guards=object(), backend=backend)
    bound = aot.BoundForward(artifact=guarded, owner=object(), cache_hit=True)

    bound.retire_after_capture()
    assert isinstance(bound.artifact, aot._CapturedForwardResources)
    assert bound.backend is backend
    assert generations == [0]

    bound.retire_after_capture()
    assert generations == [0]


def test_capture_forwards_defers_full_reclamation_until_after_capture_loop(monkeypatch):
    events = []
    monkeypatch.setattr(aot, "release_unused", lambda *, generation=2: events.append(("retire_gc", generation)))
    monkeypatch.setattr(capture, "release_unused", lambda *, generation=2: events.append(("phase_gc", generation)))

    stream = SimpleNamespace(wait_stream=lambda value: None, synchronize=lambda: None)
    monkeypatch.setattr(capture, "_warmup_stream", lambda device: stream)
    monkeypatch.setattr(capture, "_seed_warmup_workspaces", lambda stream, device: None)
    monkeypatch.setattr(capture.torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(capture.torch.cuda, "stream", lambda value: nullcontext())
    monkeypatch.setattr(capture.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(capture.torch.cuda, "CUDAGraph", lambda: "graph")
    monkeypatch.setattr(capture.torch.cuda, "graph", lambda graph, **kwargs: nullcontext())
    monkeypatch.setattr(capture.torch.distributed, "is_initialized", lambda: False)

    def make_bound(name):
        backend = SimpleNamespace(_cuda_allocation_sizes=())
        guarded = aot._GuardedForwardCall(
            function=lambda owner, name=name: events.append(("capture", name)),
            guards=SimpleNamespace(check=lambda args: True),
            backend=backend,
        )
        return aot.BoundForward(artifact=guarded, owner=object(), cache_hit=True)

    class Cache:
        def __init__(self, **kwargs):
            pass

        def prepare_many(self, functions, signatures, **kwargs):
            return {name: make_bound(name) for name in functions}

    monkeypatch.setattr(capture, "ForwardCache", Cache)
    session = SimpleNamespace(namespace=SimpleNamespace(to_dict=lambda: {}), capture=nullcontext)
    functions = {name: (lambda: None) for name in ("a", "b", "c")}
    signatures = {name: SimpleNamespace(kind="decode") for name in functions}

    graphs, compiled = capture.capture_forwards(
        functions, signatures, session=session, device="cuda:0", stream=stream, pool=None,
    )
    assert set(graphs) == {"a", "b", "c"}
    assert all(isinstance(item.artifact, aot._CapturedForwardResources) for item in compiled.values())
    assert events == [
        ("phase_gc", 2),
        ("capture", "a"),
        ("retire_gc", 0),
        ("capture", "b"),
        ("retire_gc", 0),
        ("capture", "c"),
        ("retire_gc", 0),
        ("phase_gc", 2),
    ]

    events.clear()
    _, retained = capture.capture_forwards(
        functions, signatures, session=session, device="cuda:0", stream=stream, pool=None,
        retain_forwards=True,
    )
    assert all(isinstance(item.artifact, aot._GuardedForwardCall) for item in retained.values())
    assert events == [
        ("phase_gc", 2),
        ("capture", "a"),
        ("capture", "b"),
        ("capture", "c"),
    ]


def _fake_composition_candidate(name):
    forward = SimpleNamespace(execution_identity=f"exec-{name}")
    call = PreparedCall(run=forward, benchmark_producers=(lambda: None,), validate=lambda: None)
    return composition_tuning.PreparedComposition(forward, (), call)


def test_composition_race_session_reuses_one_preparation_session_across_races(monkeypatch):
    created = []
    prepared_ops = []
    closed = []

    class FakeSession:
        def __init__(self, *, device, autotune, compile_workers, prime, cache_dir, namespace,
                     cache_policy, collective_barrier):
            self.device = device
            self.autotune = autotune
            self.compile_workers = compile_workers
            self.prime = prime
            self.cache_dir = None if cache_dir is None else Path(cache_dir)
            self.namespace = FrozenMapping(namespace or {})
            self.cache_policy = cache_policy
            self.collective_barrier = collective_barrier
            created.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(self)
            return False

        def prepare(self, requests, *, coordinator):
            (request,) = requests
            prepared_ops.append((self, request.name))
            selection = SimpleNamespace(config={"implementation": "fused"}, source="cached")
            result = SimpleNamespace(
                selections={request.name: selection},
                cache_hits=1,
                benchmarked_candidates=0,
                rejected_candidates=0,
                coverage={request.name: {"measured_count": 0}},
            )
            return nullcontext(result)

    monkeypatch.setattr(composition_tuning, "PreparationSession", FakeSession)
    session = SimpleNamespace(
        device=SimpleNamespace(ordinal=0),
        autotune=True,
        cache_dir=Path("/tmp/cache"),
        namespace=FrozenMapping({"checkpoint": "rev"}),
    )
    coordinator = SimpleNamespace(ranks=(0, 1), barrier=lambda key, ranks: None)
    candidates = {
        "separate": _fake_composition_candidate("separate"),
        "fused": _fake_composition_candidate("fused"),
    }

    with composition_tuning.composition_race_session(
        session, coordinator=coordinator, cache_policy="warm"
    ) as race_session:
        for index in range(3):
            result = composition_tuning.race_compositions(
                candidates,
                operation=f"op-{index}",
                geometry={"rows": 4 * (index + 1)},
                numerics="exact",
                workload={"version": 1},
                default="separate",
                session=session,
                coordinator=coordinator,
                cache_policy="warm",
                race_session=race_session,
            )
            assert result["implementation"] == "fused"
            assert result["source"] == "cached"
        assert closed == []

    assert len(created) == 1
    assert closed == created
    assert prepared_ops == [(created[0], "op-0"), (created[0], "op-1"), (created[0], "op-2")]

    standalone = composition_tuning.race_compositions(
        candidates,
        operation="op-standalone",
        geometry={"rows": 4},
        numerics="exact",
        workload={"version": 1},
        default="separate",
        session=session,
        coordinator=coordinator,
        cache_policy="warm",
    )
    assert standalone["implementation"] == "fused"
    assert len(created) == 2
    assert closed == created


@pytest.mark.parametrize("field,bad_value", [
    ("cache_policy", "cold"),
    ("autotune", False),
    ("compile_workers", 1),
    ("prime", True),
    ("cache_dir", Path("/tmp/other")),
    ("namespace", FrozenMapping({"checkpoint": "other"})),
    ("collective_barrier", lambda key, ranks: None),
    ("device", SimpleNamespace(ordinal=1)),
])
def test_race_compositions_rejects_mismatched_race_session(monkeypatch, field, bad_value):
    prepared = []

    class FakeSession:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def prepare(self, *args, **kwargs):
            prepared.append(args)
            raise AssertionError("mismatched race session must not prepare")

    monkeypatch.setattr(composition_tuning, "PreparationSession", FakeSession)
    barrier = lambda key, ranks: None  # noqa: E731
    session = SimpleNamespace(
        device=SimpleNamespace(ordinal=0),
        autotune=True,
        cache_dir=Path("/tmp/cache"),
        namespace=FrozenMapping({"checkpoint": "rev"}),
    )
    coordinator = SimpleNamespace(ranks=(0, 1), barrier=barrier)
    base = dict(
        device=session.device,
        autotune=True,
        compile_workers=0,
        prime=False,
        cache_dir=session.cache_dir,
        namespace=session.namespace,
        cache_policy="warm",
        collective_barrier=barrier,
    )
    base[field] = bad_value
    mismatched = FakeSession(**base)
    candidates = {"separate": _fake_composition_candidate("separate")}

    with pytest.raises(ValueError, match="composition race session"):
        composition_tuning.race_compositions(
            candidates,
            operation="op-mismatch",
            geometry={"rows": 4},
            numerics="exact",
            workload={"version": 1},
            default="separate",
            session=session,
            coordinator=coordinator,
            cache_policy="warm",
            race_session=mismatched,
        )
    assert prepared == []
