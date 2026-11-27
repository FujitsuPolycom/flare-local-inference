# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU contract, parity, validation, and FX-tracing tests for compact local-head Engram staging."""

from types import SimpleNamespace

import pytest
import torch
from b12x.sequence import engram as b12x_engram
from b12x.sequence.engram.reference import hash_reference
from torch.fx.experimental.proxy_tensor import make_fx

from ignition.components import engram as core_engram
from ignition.components.engram import EngramInputs, PreparedEngramLookup, PreparedEngramRequests


def _make_inputs(tokens: int, heads: int) -> EngramInputs:
    timings = SimpleNamespace(
        timestamps=torch.zeros((2, 3), dtype=torch.int64),
        frame=torch.zeros((1,), dtype=torch.int32),
    )
    return EngramInputs(
        weights=torch.zeros((tokens * heads, 256), dtype=torch.uint8),
        scales=torch.zeros((tokens * heads, 8), dtype=torch.uint8),
        hashes=torch.full((tokens, heads), -1, dtype=torch.int64),
        ready=SimpleNamespace(cuda_event=12345),
        timings=timings,
        interval=0,
    )


def _partition_mapping(offsets, rows: int, rank: int, world: int):
    heads = tuple(range(rank, len(offsets), world))
    bounds = tuple(zip(offsets, (*offsets[1:], rows), strict=True))
    mapping = [(first, last, -1) for first, last in bounds]
    cursor = 0
    for head in heads:
        first, last = bounds[head]
        mapping[head] = (first, last, cursor)
        cursor += last - first
    return heads, torch.tensor(mapping, dtype=torch.int64), cursor


def test_engram_inputs_allocate_validation_and_shapes(monkeypatch):
    class FakeEvent:
        def __init__(self, *, external=False):
            self.external = external
            self.cuda_event = 99

        def record(self, stream):
            self.stream = stream

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "stream")
    timings = SimpleNamespace(timestamps=torch.zeros((2, 3), dtype=torch.int64), frame=torch.zeros(1, dtype=torch.int32))

    default_inputs = EngramInputs.allocate(16, "cpu", timings=timings)
    assert default_inputs.weights.shape == (16 * 24, 256)
    assert default_inputs.scales.shape == (16 * 24, 8)
    assert default_inputs.hashes.shape == (16, 24)
    assert torch.all(default_inputs.hashes == -1)

    for heads in (3, 4, 6, 8, 12, 24):
        compact = EngramInputs.allocate(128, "cpu", timings=timings, heads=heads)
        assert compact.weights.shape == (128 * heads, 256)
        assert compact.scales.shape == (128 * heads, 8)
        assert compact.hashes.shape == (128, heads)

    for bad in (0, -1, 5, 25, 6.0, True):
        with pytest.raises(ValueError, match="must divide 24"):
            EngramInputs.allocate(8, "cpu", timings=timings, heads=bad)

    for heads, world in ((12, 4), (6, 1), (8, 4)):
        with pytest.raises(ValueError, match="invalid for world size"):
            EngramInputs.allocate(8, "cpu", timings=timings, heads=heads, world=world)


@pytest.mark.parametrize("world", [1, 2, 3, 4, 6, 8])
@pytest.mark.parametrize("tokens", [1, 8, 128])
def test_prepared_engram_requests_compact_matches_legacy_bit_exact(monkeypatch, world, tokens):
    geometry = b12x_engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    token_map = tuple(range(32))
    rows = geometry.num_embeddings[0]
    local_heads = 24 // world

    monkeypatch.setattr(core_engram.engram, "Caps", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(core_engram.engram, "plan", lambda caps, **kw: SimpleNamespace(
        handle=77, request=lambda **rkw: SimpleNamespace(**rkw)))
    monkeypatch.setattr(core_engram, "declare", lambda session, reqs: None)

    def fake_hash_op(handle, token_ids, mask, starts, slots, history, num_seqs, num_tokens,
                     out_hashes, compressed, request_ids, prepared_tokens):
        live_tokens = int(num_tokens.item())
        live_seqs = int(num_seqs.item())
        out_hashes.fill_(-1)
        if live_tokens > 0 and live_seqs > 0:
            active_starts = starts[:live_seqs + 1].tolist()
            active_starts[-1] = min(active_starts[-1], live_tokens)
            ref = hash_reference(
                token_ids[:live_tokens].tolist(),
                mask[:live_tokens].tolist(),
                active_starts,
                slots[:live_seqs].tolist(),
                history.tolist(),
                token_map,
                geometry,
                layer_id=1,
            )
            out_hashes[:ref.shape[0]].copy_(ref)

    published_sizes = []

    def fake_publish(ids, mailbox, sequence):
        n = ids.numel()
        published_sizes.append(n)
        mailbox[:n].copy_(ids.reshape(-1))
        mailbox[-1] = sequence[0]

    monkeypatch.setattr(core_engram, "hash_op", fake_hash_op)
    monkeypatch.setattr(core_engram, "publish_requests", fake_publish)

    seqs = min(4, tokens)
    live_tokens = max(1, tokens - (tokens // 3))
    token_ids = (torch.arange(tokens, dtype=torch.int64) * 7 + 3) % 32
    boundaries = [0] + [live_tokens * (i + 1) // seqs for i in range(seqs)]
    starts = torch.tensor(boundaries, dtype=torch.int32)
    history = (torch.arange(seqs * 3, dtype=torch.int64).reshape(seqs, 3) * 5 + 1) % 32
    history[0, 0] = -1
    counts = torch.tensor([live_tokens, seqs], dtype=torch.int32)
    sequence = torch.tensor([42], dtype=torch.int64)

    for rank in range(world):
        head_indices, full_mapping, _ = _partition_mapping(geometry.offsets[0], rows, rank, world)

        legacy_inputs = _make_inputs(tokens, 24)
        legacy_mailbox = torch.full((tokens * 24 + 1,), -9, dtype=torch.int64)
        legacy_req = PreparedEngramRequests(
            legacy_inputs, token_ids, starts, history, counts, legacy_mailbox, sequence,
            layer=1, token_map=token_map, geometry=geometry, session=None, name="legacy",
            row_mapping=full_mapping,
        ).functional()
        legacy_req(token_ids, starts, history, counts)

        compact_inputs = _make_inputs(tokens, local_heads)
        compact_mailbox = torch.full((tokens * local_heads + 1,), -9, dtype=torch.int64)
        compact_req = PreparedEngramRequests(
            compact_inputs, token_ids, starts, history, counts, compact_mailbox, sequence,
            layer=1, token_map=token_map, geometry=geometry, session=None, name="compact",
            row_mapping=full_mapping, head_indices=head_indices if local_heads < 24 else None,
        ).functional()
        published_sizes.clear()
        compact_req(token_ids, starts, history, counts)

        expected = legacy_inputs.hashes[:, list(head_indices)]
        assert torch.equal(compact_inputs.hashes, expected)
        assert published_sizes == [tokens * local_heads]
        assert torch.equal(compact_mailbox[:tokens * local_heads], expected.reshape(-1))
        assert int(compact_mailbox[-1].item()) == 42
        if live_tokens < tokens:
            assert torch.all(compact_inputs.hashes[live_tokens:] == -1)


def test_prepared_engram_requests_rejects_mismatched_world_head_width(monkeypatch):
    geometry = b12x_engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    token_map = tuple(range(32))
    rows = geometry.num_embeddings[0]
    monkeypatch.setattr(core_engram.engram, "Caps", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(core_engram.engram, "plan", lambda caps, **kw: SimpleNamespace(
        handle=77, request=lambda **rkw: SimpleNamespace(**rkw)))
    monkeypatch.setattr(core_engram, "declare", lambda session, reqs: None)

    tokens = 8
    token_ids = torch.zeros(tokens, dtype=torch.int64)
    starts = torch.tensor([0, tokens], dtype=torch.int32)
    history = torch.zeros((1, 3), dtype=torch.int64)
    counts = torch.tensor([tokens, 1], dtype=torch.int32)
    sequence = torch.tensor([1], dtype=torch.int64)
    _, tp4_mapping, _ = _partition_mapping(geometry.offsets[0], rows, 0, 4)

    # TP4 owns 6 heads; staging 12 heads must be rejected at preparation (matching Helios decode).
    with pytest.raises(ValueError, match="invalid for world size 4"):
        PreparedEngramRequests(
            _make_inputs(tokens, 12), token_ids, starts, history, counts,
            torch.zeros(tokens * 12 + 1, dtype=torch.int64), sequence,
            layer=1, token_map=token_map, geometry=geometry, session=None, name="tp4_wrong_heads",
            row_mapping=tp4_mapping, head_indices=tuple(range(12)),
        )


def test_prepared_engram_lookup_dispatch_validation_and_fx_tracing(monkeypatch):
    geometry = b12x_engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    token_map = tuple(range(32))
    rows = geometry.num_embeddings[0]
    tokens = 16
    heads = 6

    class FakeWait:
        def __init__(self, timings, interval, **kw):
            self.timings = timings
            self.interval = interval
            self.handle = 0

        def functional(self, *, session):
            return self

        def operands(self, source):
            b = torch.zeros((1, 8), dtype=torch.bfloat16, device=source.device)
            return b, torch.empty_like(b)

    monkeypatch.setattr(core_engram, "PreparedReadinessWait", FakeWait)
    monkeypatch.setattr(core_engram.engram, "Caps", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(core_engram.engram, "plan", lambda caps, **kw: SimpleNamespace(
        handle=88, request=lambda **rkw: SimpleNamespace(**rkw)))
    monkeypatch.setattr(core_engram, "declare", lambda session, reqs: None)

    compact_inputs = _make_inputs(tokens, heads)
    compact_out = torch.empty((tokens, heads * 256), dtype=torch.bfloat16)
    count = torch.tensor([10], dtype=torch.int32)
    compact_lookup = PreparedEngramLookup(
        compact_inputs, compact_out, count,
        layer=1, token_map=token_map, geometry=geometry, session=None, name="compact_lookup",
    ).functional(session=None)
    assert compact_lookup.plan is None
    assert compact_lookup.table_rows == rows

    # Wrong output width must raise immediately.
    with pytest.raises(ValueError, match="heads \\* 256"):
        PreparedEngramLookup(
            compact_inputs, torch.empty((tokens, 6144), dtype=torch.bfloat16), count,
            layer=1, token_map=token_map, geometry=geometry, session=None, name="bad_out",
        )

    # Mismatched world size on runtime must raise immediately.
    with pytest.raises(ValueError, match="invalid for world size 4"):
        PreparedEngramLookup(
            _make_inputs(tokens, 12), torch.empty((tokens, 12 * 256), dtype=torch.bfloat16), count,
            layer=1, token_map=token_map, geometry=geometry, session=None, name="bad_world",
            runtime=SimpleNamespace(world_size=4),
        )

    # Legacy 24-head path still uses b12x plan and flare::staged_engram.
    legacy_inputs = _make_inputs(tokens, 24)
    legacy_out = torch.empty((tokens, 6144), dtype=torch.bfloat16)
    legacy_lookup = PreparedEngramLookup(
        legacy_inputs, legacy_out, count,
        layer=1, token_map=token_map, geometry=geometry, session=None, name="legacy_lookup",
    ).functional(session=None)
    assert legacy_lookup.plan is not None and legacy_lookup.plan.handle == 88

    # make_fx(tracing_mode="fake") exercises @register_fake and emits flare::staged_engram_compact vs flare::staged_engram.
    src = torch.zeros((tokens, 5120), dtype=torch.bfloat16)
    compact_gm = make_fx(lambda s, c: compact_lookup(s, c), tracing_mode="fake", _allow_non_fake_inputs=True)(src, count)
    compact_ops = [str(node.target) for node in compact_gm.graph.nodes if node.op == "call_function"]
    assert any("staged_engram_compact" in op for op in compact_ops)
    assert not any(op.endswith("staged_engram.default") for op in compact_ops)

    legacy_gm = make_fx(lambda s, c: legacy_lookup(s, c), tracing_mode="fake", _allow_non_fake_inputs=True)(src, count)
    legacy_ops = [str(node.target) for node in legacy_gm.graph.nodes if node.op == "call_function"]
    assert any(op.endswith("staged_engram.default") for op in legacy_ops)
