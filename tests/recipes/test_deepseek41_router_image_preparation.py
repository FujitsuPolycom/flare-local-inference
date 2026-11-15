# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The real router preparation callback validates the image mask for the rows it routes.

Unrelated layer operators are declaration doubles. PreparedRouter, its b12x plan/query, and
b12x's score-option validator remain real. The callback runs validation without launching a GPU
kernel; these cases establish preparation wiring, not numerical or captured GPU correctness.
"""

from types import SimpleNamespace

import pytest
import torch

from b12x.moe._shared.routing import _validate_score_options
from b12x.moe.fused_moe._tuning import ROUTE_TUNING

from ignition.components.moe import RouteRecord, RouterScratch
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek41 import decoder


TOKENS, WORLD, EXPERTS, TOP_K = 4096, 4, 384, 6


class _Declarations(RecordedPreparationSession):
    def __init__(self):
        self.requests = []
        self.finalizers = []

    def register_finalizer(self, callback):
        self.finalizers.append(callback)

    def declare(self, requests, *, coordinator=None):
        self.requests.extend(requests)


def _operator(*args, **kwargs):
    return SimpleNamespace(plan=object(), functional=lambda **options: None)


def _surfaces(rows, device="cpu"):
    value = torch.zeros((rows, 1), device=device)
    mix = torch.zeros((rows, 4), device=device)
    hc = SimpleNamespace(normalized=value, residual=value, post=value, comb=value)
    return dict(attention_hc=hc, ffn_hc=hc, pre_mix=mix, attn_mix=mix, residual=value,
                attention_output=value, mlp_output=value, attention=SimpleNamespace(output=value))


def _layer(monkeypatch, *, sharded, rank, vision, device="cpu", correction=None, image_correction=None):
    # Only the router is under preparation here; the other operators need no kernels or weights.
    for name in ("PreparedMHC", "PreparedAttention", "PreparedAllReduce", "PreparedReduceScatter",
                 "PreparedChunkedGatherShards", "PreparedGatherShards", "UnquantizedProjection",
                 "PreparedRoutedExperts", "PreparedMLP", "PreparedMHCPost"):
        monkeypatch.setattr(decoder, name, _operator)
    monkeypatch.setattr(decoder, "_mhc_weights", lambda *args: None)
    scratch = SimpleNamespace(**_surfaces(TOKENS, device),
        router_logits=torch.zeros((TOKENS, EXPERTS), device=device),
        router=RouterScratch.allocate(TOKENS, device, experts=EXPERTS, top_k=TOP_K),
        shared=SimpleNamespace(gate_up=None, intermediate=None, output=None),
        routed=None, mlp_partial=None)
    row_shard = None
    if sharded:
        rows, start = TOKENS // WORLD, rank * (TOKENS // WORLD)
        row_shard = SimpleNamespace(**_surfaces(rows, device), rows=rows, start=start,
            rows_index=torch.arange(start, start + rows, device=device), window_values=None,
            router_logits=torch.zeros((rows, EXPERTS), device=device),
            routes=RouteRecord.allocate(rows, TOP_K, device),
            gathered_routes=RouteRecord.allocate(TOKENS, TOP_K, device))
    text = dict(rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20,
                scoring_func="sqrtsoftplus", norm_topk_prob=True,
                routed_scaling_factor=1., swiglu_limit=10.)
    if vision and image_correction is None:
        image_correction = torch.zeros(EXPERTS, device=device)
    weights = SimpleNamespace(engram=None, attention=None, experts=SimpleNamespace(
        router=None, correction=correction, vision_correction=image_correction if vision else None,
        experts=None, shared=None))
    session = _Declarations()
    layer = decoder.PreparedLayer(0, weights, None, scratch,
        SimpleNamespace(token_ids=torch.zeros(TOKENS, dtype=torch.int64, device=device)),
        SimpleNamespace(config=SimpleNamespace(text=text)), None,
        shard=SimpleNamespace(world_size=WORLD), mode="prefill", runtime=None, coordinator=None,
        session=session, name="layer", workspace=SimpleNamespace(overlap=False), carry=False,
        defer_post=False, row_shard=row_shard, attention_shard=sharded)
    assert len(session.requests) == 1
    return layer, session.requests[0]


@pytest.mark.parametrize("sharded,rank,vision", [
    (False, 0, True), (True, 0, True), (True, 1, True), (True, 3, True), (True, 1, False),
])
def test_router_preparation_validates_its_own_image_rows(monkeypatch, sharded, rank, vision):
    layer, request = _layer(monkeypatch, sharded=sharded, rank=rank, vision=vision)
    query = request.plan.query
    rows = TOKENS // WORLD if sharded else TOKENS
    assert query.num_tokens == rows
    assert query.has_image_mask is vision and query.has_image_correction_bias is vision
    full_mask = torch.arange(TOKENS) % 7 == 3
    if vision:
        layer.image_mask.copy_(full_mask)
    captured = []

    def validate(logits, selected, ids, weights, **options):
        # This is the library validator called by native route_topk before kernel dispatch.
        ROUTE_TUNING.validate_query(query, None)
        _validate_score_options(logits, score_func=query.score_func,
            correction_bias=options.get("correction_bias"),
            image_correction_bias=options.get("image_correction_bias"),
            image_mask=options.get("image_mask"))
        captured.append(options.get("image_mask"))

    request.prepare_call(SimpleNamespace(run=validate)).run()
    assert len(captured) == 1
    if vision:
        start = rank * rows if sharded else 0
        assert layer.image_mask.shape == (TOKENS,)
        assert torch.equal(layer.image_mask, full_mask)
        assert torch.equal(captured[0], full_mask[start:start + rows])
    else:
        assert layer.image_mask is None and captured[0] is None
