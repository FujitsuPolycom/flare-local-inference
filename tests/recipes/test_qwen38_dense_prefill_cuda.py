# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured Qwen GDN prefill with independent token and sequence counts."""

import os
from pathlib import Path

import pytest
import torch

from ignition.recipes import resolve

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA GPU")


@pytest.mark.parametrize("checkpoint,online_mxfp8", [("qwen3.8-27b", False), ("qwen3.8-27b-qad", False), ("qwen3.8-27b-qad", True), ("hybrid", True)])
@torch.no_grad()
def test_checkpoint_captured_chunked_prefill_preserves_counts_and_state(checkpoint, online_mxfp8):
    from b12x.preparation import PreparationSession
    from ignition.components.dflash2 import PreparedContext
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38_dense.decoder import PreparedDecoder, Scratch
    from ignition.recipes.qwen38_dense.pool import ContinuationPool
    from ignition.recipes.qwen38_dense.weights import ModelWeights

    device = torch.device("cuda:0")
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    target = (Path(os.environ.get("FLARE_QWEN38_DENSE_CHECKPOINT",
        "/home/luke/models/Qwen3.8-27B-NVIDIA-NVFP4-QAD-MXFP8")) if checkpoint == "hybrid"
        else resolve(checkpoint).snapshot(hub))
    draft = hub / "models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5"
    weights = ModelWeights.load(inspect_indexed_checkpoint(target), device, mtp=True, draft_snapshot=draft, online_mxfp8=online_mxfp8)
    pool = ContinuationPool.for_model(weights.config, 3, 2, device, mtp=True)
    metadata = Metadata.allocate(128, 1, 1, device)
    starts = torch.zeros(1, dtype=torch.int64, device=device)
    lengths = torch.zeros(1, dtype=torch.int32, device=device)
    rows = torch.ones(1, dtype=torch.int64, device=device)
    pages = torch.zeros((1, 1), dtype=torch.int32, device=device)
    tokens = torch.arange(11, 139, device=device).reshape(1, 128)
    live_rows = torch.ones(1, dtype=torch.int32, device=device)

    def inputs(start, count):
        starts.fill_(start)
        lengths.fill_(count)
        metadata.set(starts, lengths, rows, pages, tokens + start, live_rows)

    with PreparationSession(device=device, autotune=False) as session:
        arena = ScratchArena(device)
        scratch = Scratch.allocate(128, 1, weights.config, device)
        decoder = PreparedDecoder(weights, scratch, metadata, pool, mode="prefill", vocabulary_runtime=None,
            coordinator=None, session=session, name="qwen.actual.prefill128", workspace=arena)
        context = PreparedContext(weights, decoder.auxiliary, metadata, pool, session=session,
            name="qwen.actual.prefill.context", workspace=arena, runtime=None, coordinator=None).functional()
        decoder.release_outputs()
        observed_hidden = torch.empty((1, weights.config.hidden), dtype=torch.bfloat16, device=device)
        observed_features = torch.empty((128, 25600), dtype=torch.bfloat16, device=device)
        observed_counts = torch.empty((2,), dtype=torch.int32, device=device)
        arena.finalize()

        def forward():
            hidden, _, auxiliary, _ = decoder(metadata)
            delta = decoder._delta_metadata(metadata)
            observed_counts.copy_(torch.cat((delta.num_seqs, delta.num_tokens)))
            observed_hidden.copy_(hidden)
            observed_features.copy_(auxiliary)
            context(auxiliary, metadata.lengths, metadata=metadata)

        inputs(0, 17)
        compiled = compile_forward(forward)
        pool.clear_for_preparation()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        pool.clear_for_preparation()
        errors = []
        for start, count in ((0, 17), (17, 13)):
            inputs(start, count)
            before = pool.backing.clone()
            before_pages = {name: value.clone() for name, value in pool.pages.items()}
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert observed_counts[0].item() == 1
            assert observed_counts[1].item() == count
            actual = pool.backing.clone()
            actual_pages = {name: value.clone() for name, value in pool.pages.items()}
            actual_hidden = observed_hidden.clone()
            actual_features = observed_features[:count].clone()
            assert torch.equal(actual[0], before[0])
            assert torch.equal(actual[2], before[2])
            pool.backing.copy_(before)
            for name, value in before_pages.items():
                pool.pages[name].copy_(value)
            forward()
            torch.cuda.synchronize()
            relative = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))
            errors.append(dict(start=start, tokens=count, hidden_relative_l2=relative(actual_hidden, observed_hidden),
                feature_relative_l2=relative(actual_features, observed_features[:count])))
            pool.backing.copy_(actual)
            for name, value in actual_pages.items():
                pool.pages[name].copy_(value)
        print(dict(chunked_prefill=errors, sequence_count=1, capacity=128), flush=True)
        graph.reset()


@torch.no_grad()
def test_compact_prefill_context_rotates_at_the_temporal_axis():
    """Compact context selection and draft rotation use only the temporal column.

    The target decoder's own state depends on all three axes, so the context is
    replayed alone with a fixed auxiliary tensor. Its compact selection indexes
    the prompt by token position and rotates each draft key at the temporal
    column, so changing the height and width columns must leave every draft KV
    ring it writes unchanged.
    """
    from b12x.preparation import PreparationSession
    from ignition.components.dflash2 import PreparedContext
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38_dense.pool import ContinuationPool
    from ignition.recipes.qwen38_dense.weights import ModelWeights

    device = torch.device("cuda:0")
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    target = resolve("qwen3.8-27b").snapshot(hub)
    draft = hub / "models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5"
    weights = ModelWeights.load(inspect_indexed_checkpoint(target), device, mtp=True, draft_snapshot=draft)
    pool = ContinuationPool.for_model(weights.config, 3, 2, device, mtp=True, rotary_offset=True)
    window = weights.config.draft.attention.window
    prefill = window * 2
    metadata = Metadata.allocate(prefill, 1, 1, device, position_dimensions=3)
    starts = torch.zeros(1, dtype=torch.int64, device=device)
    lengths = torch.full((1,), prefill, dtype=torch.int32, device=device)
    rows = torch.ones(1, dtype=torch.int64, device=device)
    pages = torch.zeros((1, 1), dtype=torch.int32, device=device)
    tokens = torch.arange(11, 11 + prefill, device=device).reshape(1, prefill)
    live_rows = torch.ones(1, dtype=torch.int32, device=device)
    positions = torch.arange(prefill, device=device, dtype=torch.int64)
    metadata.set(starts, lengths, rows, pages, tokens, live_rows)
    feature_width = weights.config.hidden * len(weights.config.draft.target_layers)
    auxiliary = torch.arange(window * feature_width, device=device, dtype=torch.float32)
    auxiliary = auxiliary.remainder(97).bfloat16().reshape(window, feature_width)

    def set_rope(height, width):
        metadata.rope_positions.copy_(torch.stack(
            (positions, positions + height, positions + width), dim=1))

    rings = [region.name for region in pool.layout.regions
             if region.name.startswith("draft") and region.name.endswith((".key", ".value"))]
    assert rings
    set_rope(0, 0)
    with PreparationSession(device=device, autotune=False) as session:
        arena = ScratchArena(device)
        context = PreparedContext(weights, auxiliary, metadata, pool, session=session,
            name="qwen.actual.compact", workspace=arena, runtime=None, coordinator=None,
            compact_prefill=True).functional()
        assert context.compact

        def forward():
            context(auxiliary, metadata.lengths, metadata=metadata)

        compiled = compile_forward(forward)
        pool.clear_for_preparation()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()

        def run(height, width):
            set_rope(height, width)
            for name in rings:
                pool.reserved_region(name).zero_()
            graph.replay()
            torch.cuda.synchronize()
            return {name: pool.reserved_region(name).clone() for name in rings}

        first = run(0, 0)
        second = run(100, 200)
        assert any(tensor.count_nonzero() for tensor in first.values())
        assert all(torch.equal(first[name], second[name]) for name in rings)
        graph.reset()


def compact_metadata(chunk, device, three_axis):
    """One-request prompt metadata for a chunk, with token or per-axis positions."""
    from ignition.components.paged_batch import Metadata

    metadata = Metadata.allocate(chunk, 1, 1, device,
        position_dimensions=3 if three_axis else 0)
    starts = torch.zeros(1, dtype=torch.int64, device=device)
    lengths = torch.full((1,), chunk, dtype=torch.int32, device=device)
    rows = torch.ones(1, dtype=torch.int64, device=device)
    pages = torch.zeros((1, 1), dtype=torch.int32, device=device)
    tokens = torch.arange(11, 11 + chunk, device=device).reshape(1, chunk)
    metadata.set(starts, lengths, rows, pages, tokens, torch.ones(1, dtype=torch.int32, device=device))
    if three_axis:
        positions = metadata.positions.clone()
        metadata.rope_positions.copy_(torch.stack((positions, positions, positions), dim=1))
    return metadata


@torch.no_grad()
def test_text_only_compact_prefill_context_matches_the_temporal_axis():
    """A text-only compact prefill writes the draft rings an image preparation writes for text.

    Chunks 2,560 and 4,096 exceed the 2,048-token draft window, so the context
    keeps a suffix. A text-only pool declares no rotary offset region and its
    context rotates the retained rows at their token positions; an image pool
    declares the region and its context rotates at the temporal column of
    per-axis positions, which for text equals the token position. Both contexts
    run eagerly at each size and must write the same draft KV rings.
    """
    from b12x.preparation import PreparationSession
    from ignition.components.dflash2 import PreparedContext
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38_dense.pool import ContinuationPool
    from ignition.recipes.qwen38_dense.weights import ModelWeights

    device = torch.device("cuda:0")
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    target = resolve("qwen3.8-27b").snapshot(hub)
    draft = hub / "models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5"
    weights = ModelWeights.load(inspect_indexed_checkpoint(target), device, mtp=True, draft_snapshot=draft)
    window = weights.config.draft.attention.window
    feature_width = weights.config.hidden * len(weights.config.draft.target_layers)
    auxiliary = torch.arange(window * feature_width, device=device, dtype=torch.float32)
    auxiliary = auxiliary.remainder(97).bfloat16().reshape(window, feature_width)
    with PreparationSession(device=device, autotune=False) as session:
        arena = ScratchArena(device)

        def prepared(three_axis):
            pool = ContinuationPool.for_model(weights.config, 3, 2, device, mtp=True,
                rotary_offset=three_axis)
            rings = [region.name for region in pool.layout.regions
                     if region.name.startswith("draft") and region.name.endswith((".key", ".value"))]
            assert rings
            context = PreparedContext(weights, auxiliary, compact_metadata(window * 2, device, three_axis),
                pool, session=session, name=f"qwen.actual.compact.{int(three_axis)}", workspace=arena,
                runtime=None, coordinator=None, compact_prefill=True).functional()
            assert context.compact

            def run(chunk):
                metadata = compact_metadata(chunk, device, three_axis)
                for name in rings:
                    pool.reserved_region(name).zero_()
                context(auxiliary, metadata.lengths, metadata=metadata)
                torch.cuda.synchronize()
                return {name: pool.reserved_region(name).clone() for name in rings}

            return run

        text_only, image = prepared(False), prepared(True)
        for chunk in (2560, 4096):
            text_rings, image_rings = text_only(chunk), image(chunk)
            assert any(tensor.count_nonzero() for tensor in text_rings.values())
            assert text_rings.keys() == image_rings.keys()
            assert all(torch.equal(text_rings[name], image_rings[name]) for name in text_rings)
