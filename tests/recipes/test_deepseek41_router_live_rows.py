# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""A row-sharded prefill layer routes the live tokens inside its own block.

A prefill chunk pads to a fixed row count, so a rank's block can hold rows past the batch's last
live token. The shard's router declares the count it derived for the block: rows at or past it
receive expert ID -1 and zero weight, which the route pack skips. A full-row layer routes every
row, so a prefill graph with both row-shard flags off carries no count at all. These cases use the
row-local reference router from the route-record cases, because the native top-k needs a GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import moe
from ignition.components.moe import RouterScratch
from tests.components.test_moe_route_record import _prepared, _route_topk

EXPERTS, TOP_K = 8, 6


@pytest.fixture(autouse=True)
def reference_routes(monkeypatch):
    monkeypatch.setattr(moe.fused_moe, "route_topk", _route_topk)
    monkeypatch.setattr(moe, "declare", lambda *args, **kwargs: None)


@pytest.mark.parametrize("tokens", [3072, 5120, 7168])
def test_a_shard_counts_the_live_tokens_inside_its_block(tokens):
    from ignition.recipes.deepseek41.decoder import RowShard, shard_live_rows

    rows, world = tokens // 4, 4
    for rank in range(world):
        shard = RowShard.allocate(tokens, world, rank, "cpu", attention=True)
        assert shard.live_tokens.dtype is torch.int32 and shard.live_tokens.shape == (1,)
        for live in (0, 1, rows - 1, rows, 2 * rows + 5, tokens - 1, tokens):
            metadata = SimpleNamespace(live_counts=torch.tensor([live, 1], dtype=torch.int32))
            shard_live_rows(metadata, shard)
            inside = min(max(live - shard.start, 0), rows)
            assert shard.live_tokens.tolist() == [inside]
            assert shard.live_tokens.dtype is torch.int32


@pytest.mark.parametrize("tokens", [3072, 7168])
@pytest.mark.parametrize("rank,live_kind", [
    (0, "empty"), (3, "before"), (3, "first"), (3, "partial"), (3, "full")])
def test_a_shard_router_leaves_no_route_past_the_live_count(tokens, rank, live_kind):
    """At both tail-capacity extremes, padding has no routes and live rows retain their routes."""
    from ignition.recipes.deepseek41.decoder import RowShard, shard_live_rows

    world = 4
    rows = tokens // world
    start = rank * rows
    live = {"empty": 0, "before": start, "first": start + 1,
            "partial": tokens - 1, "full": tokens}[live_kind]
    shard = RowShard.allocate(tokens, world, rank, "cpu", attention=True)
    shard_live_rows(SimpleNamespace(live_counts=torch.tensor([live, 1], dtype=torch.int32)), shard)
    inside = min(max(live - shard.start, 0), rows)
    logits = torch.randn(rows, EXPERTS)
    scratch = shard.routes.router_scratch(torch.zeros(rows, TOP_K))
    _prepared(logits, scratch, num_valid_tokens=shard.live_tokens)(logits)
    assert shard.routes.words[inside:, :TOP_K].eq(-1).all()
    assert shard.routes.words[inside:, TOP_K:].eq(0).all()
    # A live row keeps the IDs and the weight bits it routes without the count.
    plain = RouterScratch.allocate(rows, "cpu", experts=EXPERTS, top_k=TOP_K)
    _prepared(logits, plain)(logits)
    assert torch.equal(shard.routes.words[:inside, :TOP_K], plain.ids[:inside])
    assert torch.equal(shard.routes.words[:inside, TOP_K:], plain.weights[:inside].view(torch.int32))


def test_only_a_sharded_prefill_layer_masks_padding_rows():
    from ignition.recipes.deepseek41.decoder import router_valid_tokens

    shard = SimpleNamespace(live_tokens=torch.tensor([3], dtype=torch.int32))
    # A shard router reads the count the shard wrote; a full-row router masks no row, in any mode.
    assert router_valid_tokens("prefill", shard) is shard.live_tokens
    for mode in ("prefill", "decode", "verify", "draft"):
        assert router_valid_tokens(mode) is None


def _component(*args, **kwargs):
    return SimpleNamespace(plan=object(), functional=lambda **options: None)


def _patched_layer(monkeypatch, scratch, metadata, *, mode, row_shard=None, attention_shard=False,
                   real_router=False):
    """`PreparedLayer` over stand-in components, so a test can read the operands it was built on."""
    from ignition.recipes.deepseek41 import decoder

    prepared = {}

    def router(logits, route_scratch, **options):
        prepared.update(options, logits=logits, route_scratch=route_scratch)
        return SimpleNamespace(plan=object(), functional=lambda **options: None)

    for name in ("PreparedMHC", "PreparedAttention", "PreparedAllReduce", "PreparedReduceScatter",
                 "PreparedGatherShards", "PreparedChunkedGatherShards", "UnquantizedProjection",
                 "PreparedRoutedExperts", "PreparedMLP", "PreparedMHCPost"):
        monkeypatch.setattr(decoder, name, _component)
    if not real_router:
        monkeypatch.setattr(decoder, "PreparedRouter", router)
    monkeypatch.setattr(decoder, "_mhc_weights", lambda *args: None)
    text = dict(rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20, scoring_func="sigmoid",
                norm_topk_prob=True, routed_scaling_factor=1., swiglu_limit=10.)
    weights = SimpleNamespace(engram=None, attention=None, experts=SimpleNamespace(
        vision_correction=None, router=None, correction=None, experts=None, shared=None))
    layer = decoder.PreparedLayer(3, weights, None, scratch, metadata,
        SimpleNamespace(config=SimpleNamespace(text=text)), None,
        shard=SimpleNamespace(world_size=4), mode=mode, runtime=None, coordinator=None,
        session=object(), name="layer", workspace=SimpleNamespace(overlap=False), carry=False,
        defer_post=True, row_shard=row_shard, attention_shard=attention_shard)
    return layer, prepared


def test_a_sharded_prefill_layer_routes_the_count_the_shard_derived(monkeypatch):
    from ignition.recipes.deepseek41 import decoder

    tokens = 4096
    scratch = decoder.Scratch.allocate(tokens, 8, 4, "cpu", row_shard=SimpleNamespace(rank=1),
                                       attention_shard=True)
    shard = scratch.row_shard
    metadata = SimpleNamespace(token_ids=torch.empty(tokens), live_counts=torch.tensor([900, 1], dtype=torch.int32))
    _, prepared = _patched_layer(monkeypatch, scratch, metadata, mode="prefill",
                                 row_shard=shard, attention_shard=True)
    assert shard.rows == 1024 and shard.start == 1024
    assert prepared["num_valid_tokens"] is shard.live_tokens
    # The shard router writes the route record this rank gathers, so the count and the routes
    # describe the same block.
    assert prepared["logits"].data_ptr() == shard.router_logits.data_ptr()
    assert prepared["route_scratch"].ids.data_ptr() == shard.routes.words.data_ptr()


def test_a_full_row_prefill_layer_prepares_no_valid_token_count(monkeypatch):
    """With both row-shard flags off the router is prepared as the unsharded decoder prepares it.

    `PreparedDecoder` leaves `scratch.row_shard` unset and `row_shard_attention` false when neither
    flag is set, so every layer routes full rows. Its route plan then declares no valid-token
    operand, so the flags-off prefill layer's route plan is the unsharded decoder's.
    """
    from ignition.recipes.deepseek41 import decoder

    tokens = 1024
    scratch = decoder.Scratch.allocate(tokens, 1, 4, "cpu")
    metadata = SimpleNamespace(token_ids=torch.empty(tokens), live_counts=torch.tensor([900, 1], dtype=torch.int32))
    layer, _ = _patched_layer(monkeypatch, scratch, metadata, mode="prefill", real_router=True)
    assert scratch.row_shard is None and layer.attention_shard is False
    assert "num_valid_tokens" not in layer.router.options
    # The route plan is the one a router prepared without a valid-token operand declares.
    unsharded = moe.PreparedRouter(scratch.router_logits,
        RouterScratch.allocate(tokens, "cpu", experts=384, top_k=6), session=object(), name="layer.topk",
        score_func="sigmoid", renormalize=True, routed_scaling_factor=1.)
    assert layer.router.plan.query == unsharded.plan.query
    assert layer.router.plan.query.has_valid_tokens is False


def test_a_compiled_shard_counts_the_live_rows_without_a_rank_offset():
    """The live-count helper reuses one Dynamo graph across four rank-specific tensor values.

    A Python-integer block offset would create a rank-specific equality guard. The helper reads
    the offset as prepared tensor data. This CPU eager-backend check exercises Dynamo graph
    reuse, not exported-artifact binding, compiled CUDA execution, or native replay.
    """
    from ignition.recipes.deepseek41.decoder import RowShard, shard_live_rows

    tokens, world = 3072, 4
    rows = tokens // world
    live = tokens - 7
    shards = [RowShard.allocate(tokens, world, rank, "cpu", attention=True) for rank in range(world)]
    metadata = SimpleNamespace(live_counts=torch.tensor([live, 1], dtype=torch.int32))

    torch._dynamo.reset()
    compiled = torch.compile(shard_live_rows, backend="eager", fullgraph=True)
    before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    compiled(metadata, shards[0])
    rank_zero = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    for shard in shards[1:]:
        compiled(metadata, shard)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == rank_zero, (
        "the compiled artifact specialized on a rank-local value")
    assert rank_zero > before

    for rank, shard in enumerate(shards):
        inside = min(max(live - rank * rows, 0), rows)
        assert shard.live_tokens.tolist() == [inside]
