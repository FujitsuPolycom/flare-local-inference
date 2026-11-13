# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A row-sharded prefill keeps one capture layout and falls back to full rows when it cannot."""

from types import SimpleNamespace

import pytest
import torch


def _surfaces(monkeypatch, *, abi=11, pipe_bytes=1 << 26):
    import b12x.comm.roce._proxy as proxy
    from ignition.components import collectives

    monkeypatch.setattr(proxy, "ABI_VERSION", abi)
    monkeypatch.setattr(collectives, "_roce", lambda runtime: True)
    return SimpleNamespace(world_size=2, pipe_bytes=pipe_bytes, _group=object(),
        should_reduce_scatter=lambda inp: True, should_gather_shards=lambda inp: True)


def test_row_shard_takes_contiguous_rank_rows_and_gates_on_world_and_call_rows():
    from ignition.recipes.deepseek41.decoder import MHC_CALL_ROWS, RowShard

    shard = RowShard.allocate(1024, 2, 1, "cpu")
    assert (shard.world, shard.rows, MHC_CALL_ROWS) == (2, 512, 512)
    assert shard.rows_index.tolist() == list(range(512, 1024))
    assert shard.residual.shape == (512, 4, 5120) and shard.residual.dtype is torch.bfloat16
    assert shard.pre_mix.shape == (512, 4) and shard.pre_mix.dtype is torch.float32
    assert shard.attention_hc.residual.shape == shard.ffn_hc.residual.shape == (512, 4, 5120)
    assert shard.attention_hc.normalized.shape == (512, 5120)
    assert RowShard.allocate(1024, 1, 0, "cpu") is None
    assert RowShard.allocate(1025, 2, 0, "cpu") is None
    # Shard eligibility accepts whole multiples of 512 rows and the listed remainder blocks.
    assert RowShard.allocate(1024, 4, 0, "cpu") is None


@pytest.mark.parametrize("tokens,rows", [(3072, 768), (5120, 1280), (7168, 1792)])
def test_tail_buckets_accept_the_declared_per_rank_row_counts(tokens, rows):
    """The three TP4 capacities produce the declared local rows and prepared tensor shapes."""
    from ignition.components.mhc import row_call_bounds
    from ignition.recipes.deepseek41.decoder import RowShard

    shard = RowShard.allocate(tokens, 4, 2, "cpu", attention=True, candidate_tokens=16)
    assert (shard.world, shard.rank, shard.rows, shard.start) == (4, 2, rows, 2 * rows)
    assert shard.rows_index.tolist() == list(range(2 * rows, 3 * rows))
    assert shard.residual.shape == (rows, 4, 5120)
    assert shard.attention_hc.normalized.shape == shard.ffn_hc.normalized.shape == (rows, 5120)
    # With an explicit 512-row bound, these counts decompose into whole calls plus 256 rows.
    assert [end - start for start, end in row_call_bounds(rows, 512)] == [512] * (rows // 512) + [256]
    assert shard.attention.query.shape == (rows, 64, 512)
    assert shard.router_logits.shape == (rows, 384)
    # Eligibility accepts these explicit remainder counts; neighbouring capacities keep full rows.
    for other in (tokens - 512, tokens + 512):
        assert RowShard.allocate(other, 4, 0, "cpu") is None


def test_a_whole_call_graph_reaches_every_tail_bucket():
    from ignition.recipes.deepseek41.decoder import RowShard

    captured = RowShard.allocate(8192, 4, 1, "cpu", attention=True, candidate_tokens=16)
    for tokens, rows in ((7168, 1792), (5120, 1280), (3072, 768)):
        shard = captured.graph(tokens)
        assert (shard.rows, shard.start) == (rows, rows)
        assert shard.rows_index.tolist() == list(range(rows, 2 * rows))
        assert shard.attention.query.shape == (rows, 64, 512)
        assert shard.window_values.shape == (tokens, 512)
        assert shard.router_logits.shape == (rows, 384)
    # 2,560 and 3,584 rows do not hold whole calls plus a 256-row tail.
    assert captured.graph(2560) is None and captured.graph(3584) is None


def test_row_shard_graph_shrinks_capacity_and_rejects_rows_that_do_not_divide():
    from ignition.recipes.deepseek41.decoder import RowShard

    shard = RowShard.allocate(2048, 2, 0, "cpu")
    smaller = shard.graph(1024)
    assert smaller.rows == 512 and smaller.rows_index.tolist() == list(range(512))
    assert smaller.residual.shape == (512, 4, 5120) and smaller.attention_hc.normalized.shape == (512, 5120)
    assert shard.graph(2048).rows == 1024
    # Every rank's block of a narrower graph starts at rank * rows of that graph, not of the
    # allocated layout, and stays inside the graph's rows.
    for rank in range(4):
        wide = RowShard.allocate(8192, 4, rank, "cpu")
        for tokens in (2048, 4096, 6144, 8192):
            narrow = wide.graph(tokens)
            rows = tokens // 4
            assert narrow.rows_index.tolist() == list(range(rank * rows, (rank + 1) * rows))
        assert wide.rows_index.tolist() == list(range(rank * 2048, (rank + 1) * 2048))
    assert shard.graph(1023) is None
    # 640 rows per rank holds no whole call and no 256-row tail, so it keeps full rows.
    assert shard.graph(1280) is None
    assert shard.graph(1024 + 2) is None
    # A 1,536-row graph over two ranks uses the accepted 768-row block on each rank.
    tail = shard.graph(1536)
    assert (tail.rows, tail.start) == (768, 0)


def test_decoder_scratch_carries_a_row_shard_only_when_the_bootstrap_asks():
    from ignition.recipes.deepseek41.decoder import Scratch

    plain = Scratch.allocate(1024, 1, 2, "cpu", candidate_tokens=16)
    assert plain.row_shard is None
    asked = Scratch.allocate(1024, 1, 2, "cpu", candidate_tokens=16,
                             row_shard=SimpleNamespace(rank=1))
    assert asked.row_shard.rows == 512 and asked.row_shard.rows_index[0].item() == 512
    assert asked.rows(1024, 1).row_shard.rows == 512
    # A 256-row shard is outside the accepted counts, so this narrower graph keeps full rows.
    assert asked.rows(512, 1).row_shard is None


class _Recorder:
    """A stand-in for a prepared component that returns the surface the forward reads."""

    def __init__(self, value=None):
        self.value, self.calls, self.kwargs = value, [], {}

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        self.kwargs = kwargs
        return self.value

    def _run(self, *args, **kwargs):
        self.calls.append(args)
        self.kwargs = kwargs
        return self.value


def _metadata(tokens=1024):
    return SimpleNamespace(token_ids=torch.zeros(tokens, dtype=torch.int64),
                           positions=torch.arange(tokens))


def _recording_layer(scratch, *, reseed=False):
    """A `PreparedLayer` whose `_run` calls recorders, so a test can read the operands it passes."""
    from ignition.recipes.deepseek41 import decoder

    shard = scratch.row_shard
    layer = object.__new__(decoder.PreparedLayer)
    layer.scratch, layer.row_shard, layer.reseed = scratch, shard, reseed
    layer.attention_shard, layer.selection_move = False, None
    layer.engram = layer.concurrent_shared = None
    layer.image_mask, layer.has_image_mask = None, False
    attended = shard.attention_output if shard is not None else scratch.attention_output
    attention_branch = scratch.attention_hc if shard is None else shard.attention_hc
    ffn_branch = scratch.ffn_hc if shard is None else shard.ffn_hc
    layer.attention = _Recorder((attended, {}, None, {}))
    layer.attention_hc = _Recorder((attention_branch.residual, attention_branch.post,
                                    attention_branch.comb, attention_branch.normalized))
    layer.ffn_hc = _Recorder((ffn_branch.residual, ffn_branch.post, ffn_branch.comb, ffn_branch.normalized))
    layer.post = None if shard is None else _Recorder(shard.residual)
    layer.shared = _Recorder(scratch.shared.output)
    layer.attention_reduce = _Recorder(attended)
    layer.mlp_reduce = _Recorder()
    layer.router_projection = _Recorder(scratch.router_logits)
    layer.router_source = (scratch.ffn_hc.normalized if shard is None else shard.ffn_hc.normalized)
    layer.route_gather, layer.valid_tokens = None, None
    layer.router = _Recorder(SimpleNamespace(ids=torch.zeros(1, dtype=torch.int32), weights=torch.zeros(1),
                                             selected_logits=None))
    layer.experts = _Recorder(scratch.routed)
    if shard is None:
        layer.attention_gather = layer.ffn_gather = None
        layer.residual_gather = layer.mix_gather = None
    else:
        layer.attention_gather, layer.ffn_gather = _Recorder(), _Recorder()
        layer.residual_gather, layer.mix_gather = _Recorder(), _Recorder()
    return layer


@pytest.mark.parametrize("row_shard", [False, True])
def test_the_forward_normalizes_the_ffn_from_the_attention_branch_state(row_shard):
    """A prepared mHC rewrites the residual, post and comb surfaces it reads.

    The feed-forward mHC must therefore take the attention branch's surfaces — the shard's copy
    when the layer shards, the full scratch when it does not — which the attention mHC wrote on
    this call. The feed-forward scratch still holds the previous layer's post until this call
    writes it, so passing its surfaces normalizes a stale mix.
    """
    from ignition.recipes.deepseek41 import decoder

    scratch = decoder.Scratch.allocate(1024, 8, 2, "cpu",
        row_shard=SimpleNamespace(rank=1) if row_shard else None)
    branch = scratch.attention_hc if scratch.row_shard is None else scratch.row_shard.attention_hc
    output = scratch.attention_output if scratch.row_shard is None else scratch.row_shard.attention_output
    residual_source = scratch.residual if scratch.row_shard is None else scratch.row_shard.residual
    mix_source = scratch.pre_mix if scratch.row_shard is None else scratch.row_shard.pre_mix
    layer = _recording_layer(scratch)
    layer._run((residual_source,), mix_source, _metadata(), None, None)
    attention_output, residual, post, comb = layer.ffn_hc.calls[0]
    assert attention_output.data_ptr() == output.data_ptr()
    for received, expected in zip((residual, post, comb),
            (branch.residual, branch.post, branch.comb), strict=True):
        assert received.data_ptr() == expected.data_ptr()


def test_a_sharded_forward_gathers_the_lagged_mix_through_its_bf16_view():
    """The mix gather was prepared over BF16 packs of the FP32 lagged mix, so it takes that view."""
    from ignition.recipes.deepseek41 import decoder

    scratch = decoder.Scratch.allocate(1024, 8, 2, "cpu", row_shard=SimpleNamespace(rank=1))
    shard = scratch.row_shard
    layer = _recording_layer(scratch)
    _, full_mix, _, _, _ = layer._run((shard.residual,), shard.pre_mix, _metadata(), None, None)
    mix, gathered = layer.mix_gather.calls[0]
    assert mix.dtype is torch.bfloat16 and mix.shape == (shard.rows, 8)
    assert mix.is_contiguous() and mix.stride() == (8, 1)
    assert gathered.dtype is torch.bfloat16 and gathered.shape == (1024, 8)
    assert full_mix.dtype is torch.float32 and full_mix.shape == (1024, 4)
    (residual,) = layer.residual_gather.calls[0]
    assert residual.data_ptr() == shard.residual.data_ptr()


@pytest.mark.parametrize("tokens,world,rank", [
    (1024, 2, 1), (3072, 4, 3), (5120, 4, 3), (7168, 4, 3)])
def test_a_resuming_shard_refills_from_the_rows_its_rank_owns(tokens, world, rank):
    """The full-row layers own the rows in between, so a resuming shard takes its own rows out.

    Taking rows from zero instead would hand a rank another rank's state. Each 128-row block of
    the full residual and lagged mix carries its own value, so the two differ. The 768-, 1,280-
    and 1,792-row blocks hold a 256-row tail, which the refill copies with the rest.
    """
    from ignition.recipes.deepseek41 import decoder

    scratch = decoder.Scratch.allocate(tokens, 8, world, "cpu", row_shard=SimpleNamespace(rank=rank))
    shard = scratch.row_shard
    blocks = (torch.arange(tokens) // 128).to(torch.float32)
    scratch.residual.copy_(blocks[:, None, None].expand(tokens, 4, 5120))
    scratch.pre_mix.copy_(blocks[:, None].expand(tokens, 4))
    layer = _recording_layer(scratch, reseed=True)
    layer._run((scratch.residual,), scratch.pre_mix, _metadata(tokens), None, None)
    (refilled,) = layer.attention_hc.calls[0]
    assert torch.equal(refilled, scratch.residual.index_select(0, shard.rows_index))
    assert torch.equal(layer.attention_hc.kwargs["pre_mix"],
                       scratch.pre_mix.index_select(0, shard.rows_index))
    assert not torch.equal(refilled, scratch.residual[:shard.rows])


@pytest.mark.parametrize("abi,pipe_bytes,ready", [
    (11, 1 << 26, True), (10, 1 << 26, False), (11, 1 << 22, False)])
def test_row_shard_ready_requires_the_abi_and_capacity_for_every_surface(monkeypatch, abi, pipe_bytes, ready):
    from ignition.recipes.deepseek41.decoder import RowShard, row_shard_ready

    runtime = _surfaces(monkeypatch, abi=abi, pipe_bytes=pipe_bytes)
    rows = 512
    scratch = SimpleNamespace(
        attention=SimpleNamespace(output=torch.zeros(1024, 5120, dtype=torch.bfloat16)),
        attention_hc=SimpleNamespace(normalized=torch.zeros(1024, 5120, dtype=torch.bfloat16)),
        ffn_hc=SimpleNamespace(normalized=torch.zeros(1024, 5120, dtype=torch.bfloat16)),
        mlp_partial=torch.zeros(1024, 5120, dtype=torch.bfloat16),
        pre_mix=torch.zeros(1024, 4, dtype=torch.float32),
        residual=torch.zeros(1024, 4, 5120, dtype=torch.bfloat16))
    shard = SimpleNamespace(
        attention_output=torch.zeros(rows, 5120, dtype=torch.bfloat16),
        mlp_output=torch.zeros(rows, 5120, dtype=torch.bfloat16),
        attention_hc=SimpleNamespace(normalized=torch.zeros(rows, 5120, dtype=torch.bfloat16)),
        ffn_hc=SimpleNamespace(normalized=torch.zeros(rows, 5120, dtype=torch.bfloat16)),
        pre_mix=torch.zeros(rows, 4, dtype=torch.float32),
        residual=torch.zeros(rows, 4, 5120, dtype=torch.bfloat16))
    assert shard.pre_mix.view(torch.bfloat16).shape == (rows, 8)
    assert row_shard_ready(scratch, shard, runtime) is ready


def test_every_row_count_a_graph_can_use_is_one_segment_of_the_table():
    """One allocation holds every segment, so no narrower graph needs storage of its own.

    The table is int64, and each graph's row numbers are a view at that row count's segment
    offset: 512 rows at 0, then 768 at 512, 1,024 at 1,280, 1,280 at 2,304, 1,536 at 3,584,
    1,792 at 5,120 and 2,048 at 6,912.
    """
    from ignition.recipes.deepseek41.decoder import RowShard

    captured = RowShard.allocate(8192, 4, 1, "cpu")
    assert captured.rows_table.dtype is torch.int64
    assert captured.rows_index.data_ptr() == captured.rows_table.data_ptr() + 6912 * 8
    assert captured.rows_index.tolist() == list(range(2048, 4096))
    offsets = {8192: 6912, 7168: 5120, 6144: 3584, 5120: 2304, 4096: 1280, 3072: 512, 2048: 0}
    for tokens, first in offsets.items():
        graph = captured.graph(tokens)
        assert graph.rows_table is captured.rows_table
        assert graph.rows_index.storage_offset() == first
        rows = tokens // 4
        assert graph.rows_index.tolist() == list(range(rows, 2 * rows))
        assert graph.rows_index.data_ptr() == captured.rows_table.data_ptr() + first * 8


def test_the_row_table_is_written_once_into_persistent_storage(monkeypatch):
    """Prefill scratch exports as zeros, so the table takes the persistent arena instead."""
    from ignition.recipes.deepseek41 import decoder

    calls = []

    def persistent(shape, dtype, device, **kwargs):
        calls.append((tuple(shape), dtype, device, kwargs))
        return torch.zeros(shape, dtype=dtype)

    monkeypatch.setattr(decoder, "persistent_zeros", persistent)
    shard = decoder.RowShard.allocate(4096, 4, 1, "cpu")
    assert calls == [((2304,), torch.int64, "cpu", {})]
    assert shard.rows_table.tolist() == [*range(512, 1024), *range(768, 1536), *range(1024, 2048)]
    assert shard.residual.dtype is torch.bfloat16


def test_the_block_offset_is_snapshot_storage_beside_the_row_numbers(monkeypatch):
    """A captured node reads this rank's block offset, so it needs the row numbers' storage.

    The offset table holds one int32 per 16 bytes, so every graph's single-element view keeps the
    alignment of the first. Prefill scratch loads as zeros; a block offset stored there would be
    zero on every rank.
    """
    from ignition.recipes.deepseek41 import decoder

    calls = []

    def persistent(shape, dtype, device, **kwargs):
        calls.append((tuple(shape), dtype, device, kwargs))
        return torch.zeros(shape, dtype=dtype)

    monkeypatch.setattr(decoder, "persistent_zeros", persistent)
    shards = [decoder.RowShard.allocate(8192, 4, rank, "cpu", candidate_tokens=16, attention=True)
              for rank in range(4)]
    # Attention shards allocate both tables; prefill without attention sharding omits the offset table.
    assert calls == [((8960,), torch.int64, "cpu", {}), ((28,), torch.int32, "cpu", {})] * 4
    assert decoder.RowShard.allocate(8192, 4, 0, "cpu", candidate_tokens=16).live_offset is None
    for rank, shard in enumerate(shards):
        assert shard.live_offset.tolist() == [rank * 2048]
        assert shard.live_offset.stride() == (1,) and shard.live_offset.dtype is torch.int32
        # Every segment starts 16 bytes after the previous one, so one offset per row count.
        assert shard.offset_table.numel() == 28 and shard.offset_table.is_contiguous()
        tokens = (8192, 7168, 6144, 5120, 4096, 3072, 2048)
        offsets = [shard.graph(tokens).live_offset for tokens in tokens]
        assert [offset.storage_offset() for offset in offsets] == [24, 20, 16, 12, 8, 4, 0]
        assert offsets[-1].storage_offset() == 0
        assert [offset.tolist() for offset in reversed(offsets)] == [
            [rank * count] for count in (512, 768, 1024, 1280, 1536, 1792, 2048)]


def test_every_rank_binds_row_shard_tables_at_one_storage_offset():
    """One compiled forward binds every rank, and binding compares storage offsets.

    Each graph's row numbers and block offset therefore sit at the same offset on every rank; only
    their contents differ.
    """
    from ignition.recipes.deepseek41.decoder import RowShard

    shards = [RowShard.allocate(8192, 4, rank, "cpu", candidate_tokens=16, attention=True)
              for rank in range(4)]
    for tokens in (8192, 7168, 6144, 5120, 4096, 3072, 2048):
        graphs = [shard.graph(tokens) for shard in shards]
        rows = tokens // 4
        assert len({graph.rows_index.storage_offset() for graph in graphs}) == 1
        assert len({graph.live_offset.storage_offset() for graph in graphs}) == 1
        for rank, graph in enumerate(graphs):
            assert graph.rows_index.tolist() == list(range(rank * rows, (rank + 1) * rows))
            assert graph.live_offset.tolist() == [rank * rows]
