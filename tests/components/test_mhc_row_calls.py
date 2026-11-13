# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""A row block that is not whole mHC calls still computes every row through the same arithmetic.

The native mHC kernels need a GPU, so these cases replace the bound kernel with b12x's own PyTorch
reference. They qualify the call decomposition: a rank's 768, 1,280 or 1,792 rows must produce the
values a padded whole-call block produces for the same rows.
"""

from types import SimpleNamespace

import pytest
import torch
from b12x.testing.mhc import make_inputs, pre_reference

from ignition.components import mhc as mhc_module
from ignition.components.mhc import MHCScratch, MHCWeights, PreparedMHC, row_call_bounds

EPS = 1e-6


def _normalized_pre(residual, weights, norm_weight, norm_eps):
    """b12x's pre reference followed by the weighted RMSNorm the native pre kernel applies."""
    y, post, comb = pre_reference(residual, weights.fn, weights.scale, weights.base,
                                  rms_eps=EPS, hc_eps=EPS, sinkhorn_iters=20)
    normed = y.float() * torch.rsqrt(y.float().square().mean(-1, keepdim=True) + norm_eps)
    return (normed * norm_weight.float()).to(residual.dtype), post, comb


def _cpu_native(monkeypatch):
    """Replace planning and the bound pre kernel, recording the row count of every call."""
    calls = []

    def bound(tensors, buffers, norm_weight, pre_mix, fn_bf16, operation, plan_handle,
              expected_m, rms_eps, hc_eps, sinkhorn_iters, norm_eps):
        residual, fn, scale, bias = tensors
        assert operation == "pre"
        calls.append(residual.shape[0])
        weights = MHCWeights(fn, scale, bias, norm_weight)
        y, post, comb = _normalized_pre(residual, weights, norm_weight, norm_eps)
        buffers[1].copy_(y)
        buffers[2].copy_(post)
        buffers[3].copy_(comb)

    # The planner answers with the interface `mhc.plan` itself returns: scratch specs, a request
    # that records the call, and a handle.
    plan = SimpleNamespace(scratch_specs=lambda: (), request=lambda **kwargs: SimpleNamespace(**kwargs),
                           handle=None)
    def planner(caps, *, invocation=None):
        return plan
    monkeypatch.setattr(mhc_module.mhc, "plan", planner)
    monkeypatch.setattr(mhc_module._ops, "_bound", bound)
    monkeypatch.setattr(mhc_module, "require_prepared",
                        lambda plan, name: SimpleNamespace(partial_shape=(0,)))
    monkeypatch.setattr(mhc_module, "planned_scratch", lambda specs, workspace=None: None)
    monkeypatch.setattr(mhc_module, "declare", lambda *args, **kwargs: None)
    return calls


def _prepare(block, weights, max_rows_per_call=512):
    scratch = MHCScratch.allocate(block.shape[0], block.shape[-1], block.device)
    return PreparedMHC((block,), weights, scratch, session=None, name="test", workspace=None,
        rms_eps=EPS, hc_eps=EPS, sinkhorn_iters=20, norm_eps=EPS,
        max_rows_per_call=max_rows_per_call)


def _weights(fn, scale, bias):
    return MHCWeights(fn, scale, bias, torch.linspace(.5, 1.5, fn.shape[1] // 4).to(torch.bfloat16))


def _assert_matches(reference, prepared, rows):
    _, post, comb, normalized = prepared
    for want, surface in zip(reference, (normalized, post, comb), strict=True):
        assert torch.equal(surface, want[:rows])


def test_row_call_bounds_end_with_the_rows_that_remain():
    assert row_call_bounds(1536, 512) == ((0, 512), (512, 1024), (1024, 1536))
    # A rank's 768, 1,280 and 1,792 rows are whole calls plus one 256-row tail.
    assert row_call_bounds(768, 512) == ((0, 512), (512, 768))
    assert [end - start for start, end in row_call_bounds(1792, 512)] == [512, 512, 512, 256]
    # A block shorter than one call is one call over its own rows.
    assert row_call_bounds(256, 512) == ((0, 256),)


@pytest.mark.parametrize("rows", [768, 1280, 1792])
def test_a_tail_block_matches_the_padded_whole_call_reference(rows, monkeypatch):
    """The 256-row tail reproduces the rows a padded 512-row call would compute for them.

    mHC mixing, Sinkhorn normalization and the weighted residual sum are row-local, so the call
    boundaries cannot reach a row's values. The block here begins at a nonzero offset of a larger
    row set, the way a rank other than rank zero owns its rows.
    """
    calls = _cpu_native(monkeypatch)
    residual, _, fn, scale, bias = make_inputs(tokens=rows + 1024, hidden_size=4096, seed=3121,
                                               device=torch.device("cpu"))
    weights = _weights(fn, scale, bias)
    block = residual[1024:1024 + rows].contiguous()
    prepared = _prepare(block, weights)
    result = prepared(block)
    # Every whole call keeps the 512-row shape; only the tail runs at 256 rows.
    assert calls == [512] * (rows // 512) + [256]
    assert [(start, end) for start, end, _ in prepared.row_calls] == list(row_call_bounds(rows, 512))

    padded_rows = -(-rows // 512) * 512
    padded = torch.zeros((padded_rows, 4, block.shape[-1]), dtype=torch.bfloat16)
    padded[:rows].copy_(block)
    padded[rows:].copy_(residual[:padded_rows - rows])
    _assert_matches(_normalized_pre(padded, weights, weights.norm, EPS), result, rows)


@pytest.mark.parametrize("rows", [768, 1792])
@pytest.mark.parametrize("max_rows_per_call", [None, 512])
def test_a_tail_block_keeps_every_row_a_full_block_would_compute(
        rows, max_rows_per_call, monkeypatch):
    """Both the decoder's unbounded call and explicit bounded calls retain every row."""
    calls = _cpu_native(monkeypatch)
    residual, _, fn, scale, bias = make_inputs(tokens=rows, hidden_size=4096, seed=4049,
                                               device=torch.device("cpu"))
    weights = _weights(fn, scale, bias)
    block = residual.contiguous()
    prepared = _prepare(block, weights, max_rows_per_call=max_rows_per_call)
    result = prepared(block)
    expected = [rows] if max_rows_per_call is None else [512] * (rows // 512) + [256]
    assert calls == expected
    _assert_matches(_normalized_pre(block, weights, weights.norm, EPS), result, rows)
