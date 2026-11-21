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
    from dataclasses import replace

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
    target = Path(os.environ.get("FLARE_QWEN38_DENSE_SNAPSHOT") or resolve("qwen3.8-27b").snapshot(hub))
    draft = hub / "models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5"
    weights = ModelWeights.load(inspect_indexed_checkpoint(target), device, mtp=True, draft_snapshot=draft)
    pool = ContinuationPool.for_model(weights.config, 3, 2, device, mtp=True, rotary_offset=True)
    window = weights.config.draft.attention.window
    prefill = window * 2
    base = Metadata.allocate(prefill, 1, 1, device)
    starts = torch.zeros(1, dtype=torch.int64, device=device)
    lengths = torch.full((1,), prefill, dtype=torch.int32, device=device)
    rows = torch.ones(1, dtype=torch.int64, device=device)
    pages = torch.zeros((1, 1), dtype=torch.int32, device=device)
    tokens = torch.arange(11, 11 + prefill, device=device).reshape(1, prefill)
    live_rows = torch.ones(1, dtype=torch.int32, device=device)
    positions = torch.arange(prefill, device=device, dtype=torch.int64)
    base.set(starts, lengths, rows, pages, tokens, live_rows)
    rotary_positions = torch.zeros((prefill, 3), dtype=torch.int64, device=device)
    metadata = replace(base, rotary_positions=rotary_positions)
    feature_width = weights.config.hidden * len(weights.config.draft.target_layers)
    auxiliary = torch.arange(window * feature_width, device=device, dtype=torch.float32)
    auxiliary = auxiliary.remainder(97).bfloat16().reshape(window, feature_width)

    def set_rope(height, width):
        rotary_positions.copy_(torch.stack((positions, positions + height, positions + width), dim=1))

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
    from dataclasses import replace

    from ignition.components.paged_batch import Metadata

    metadata = Metadata.allocate(chunk, 1, 1, device)
    starts = torch.zeros(1, dtype=torch.int64, device=device)
    lengths = torch.full((1,), chunk, dtype=torch.int32, device=device)
    rows = torch.ones(1, dtype=torch.int64, device=device)
    pages = torch.zeros((1, 1), dtype=torch.int32, device=device)
    tokens = torch.arange(11, 11 + chunk, device=device).reshape(1, chunk)
    metadata.set(starts, lengths, rows, pages, tokens, torch.ones(1, dtype=torch.int32, device=device))
    if three_axis:
        positions = metadata.positions.clone()
        metadata = replace(metadata, rotary_positions=torch.stack((positions, positions, positions), dim=1))
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
    target = Path(os.environ.get("FLARE_QWEN38_DENSE_SNAPSHOT") or resolve("qwen3.8-27b").snapshot(hub))
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


def _decode_dense_reference(manifest, device):
    """Build an independent BF16 Transformers Qwen3_5ForConditionalGeneration from checkpoint shards."""
    import json
    from safetensors import safe_open
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    from ignition.preparation.quantization import linear_format
    from ignition.reference.qwen38_quant import decode_nvfp4

    raw = json.loads((manifest.snapshot / "config.json").read_text())
    quant = raw.get("quantization_config", {}).get("quantized_layers", {})
    config = Qwen3_5Config.from_dict(raw)
    with torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(config)
    model.eval().requires_grad_(False)

    shards = {}
    try:
        def read(name):
            entry = manifest.tensors[name]
            handle = shards.get(entry.shard)
            if handle is None:
                handle = safe_open(manifest.snapshot / entry.shard, framework="pt", device="cpu")
                shards[entry.shard] = handle
            return handle.get_tensor(name)

        for name, param in list(model.named_parameters()):
            parent, _, attr = name.rpartition(".")
            prefix = name.removesuffix(".weight")
            fmt = linear_format(quant, prefix) if attr == "weight" and prefix in quant else None
            if fmt == "nvfp4":
                w = decode_nvfp4(
                    read(name),
                    read(prefix + ".weight_scale"),
                    read(prefix + ".weight_scale_2"),
                ).to(dtype=torch.bfloat16, device=device)
            elif fmt == "tensor_fp8":
                w = (read(name).float() * read(prefix + ".weight_scale").float()).to(
                    dtype=torch.bfloat16, device=device
                )
            else:
                w = read(name).to(device=device)
            assert tuple(w.shape) == tuple(param.shape), f"{name}: {w.shape} != {param.shape}"
            setattr(model.get_submodule(parent), attr, torch.nn.Parameter(w, requires_grad=False))
    finally:
        shards.clear()

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextRotaryEmbedding,
        Qwen3_5VisionRotaryEmbedding,
    )

    model.model.language_model.rotary_emb = Qwen3_5TextRotaryEmbedding(config.text_config).to(device)
    model.model.visual.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(config.vision_config).to(device)
    return model


@torch.no_grad()
def test_image_prompt_features_and_logits_match_transformers_reference():
    """End-to-end text-image-text prompt features, MRoPE offset, prefill logits, and decode logits."""
    import gc
    import json
    from PIL import Image
    from b12x.preparation import PreparationSession
    from transformers import AutoProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from transformers import Qwen3_5VisionConfig, Qwen3_5VisionModel
    from transformers.vision_utils import (
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
    from ignition.components.paged_batch import (
        Metadata,
        ROTARY_OFFSET,
        RotaryBatchInputs,
        prepare_batch,
    )
    from ignition.components.sampling import DeviceRequestState, RUNNING
    from ignition.components.vision_state import PrefillVisionInputs, VisionState
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38.vision import PreparedVision, VisionMetadata, VisionScratch
    from ignition.recipes.qwen38_dense.decoder import PreparedDecoder, Scratch
    from ignition.recipes.qwen38_dense.pool import ContinuationPool
    from ignition.recipes.qwen38_dense.vision_state import (
        FEATURES,
        IMAGE_TOKEN_ID,
        PreparedVisionGeneration,
        VisionInputs,
        vision_attention_math_mode,
    )
    from ignition.recipes.qwen38_dense.weights import ModelWeights

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    target = Path(os.environ.get("FLARE_QWEN38_DENSE_SNAPSHOT") or resolve("qwen3.8-27b").snapshot(hub))
    manifest = inspect_indexed_checkpoint(target)

    # Synthetic 256x256 red/blue split image inside a text-image-text prompt.
    width, height = 256, 256
    image = Image.new("RGB", (width, height))
    px = image.load()
    for y in range(height):
        for x in range(width):
            px[x, y] = (230, 20, 20) if x < width // 2 else (20, 40, 230)
    processor = AutoProcessor.from_pretrained(target, local_files_only=True)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Inspect this split image:"},
            {"type": "image"},
            {"type": "text", "text": "What color is the left half? Answer in one word:"},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    encoded = processor(text=[prompt], images=[image], return_tensors="pt")
    input_ids = encoded["input_ids"]
    pixel_values = encoded["pixel_values"]
    image_grid_thw = encoded["image_grid_thw"]
    mm_token_type_ids = encoded["mm_token_type_ids"]
    seq_len = int(input_ids.shape[1])
    num_patches = int(pixel_values.shape[0])
    num_image_tokens = int((input_ids[0] == IMAGE_TOKEN_ID).sum().item())
    assert num_patches == 256 and num_image_tokens == 64

    # Compute canonical 3-axis positions and expected rotary delta from meta model.
    raw_cfg = json.loads((target / "config.json").read_text())
    with torch.device("meta"):
        meta_ref = Qwen3_5ForConditionalGeneration(Qwen3_5Config.from_dict(raw_cfg))
    ref_pos_3d, ref_delta = meta_ref.model.get_rope_index(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=encoded["attention_mask"],
    )
    ref_positions = ref_pos_3d[:, 0, :].T.contiguous()  # (seq_len, 3)
    expected_delta = int(ref_delta.item())
    del meta_ref

    # Also prepare a text-only baseline prompt so the image path's share of the
    # NVFP4/FP8-vs-BF16 logit delta can be compared directly in the same session.
    text_messages = [{
        "role": "user",
        "content": "What is the capital of France? Answer in one word:",
    }]
    text_prompt = processor.apply_chat_template(
        text_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    text_encoded = processor.tokenizer(text_prompt, return_tensors="pt")
    text_input_ids = text_encoded["input_ids"]
    text_seq_len = int(text_input_ids.shape[1])
    text_positions = (
        torch.arange(text_seq_len, dtype=torch.int32).unsqueeze(1).expand(text_seq_len, 3).contiguous()
    )

    prefill_cap = ((max(seq_len, text_seq_len) + 127) // 128) * 128
    blocks = max(4, prefill_cap // 128 + 2)

    # 1. Prepared Ignition path: vision graph -> VisionState -> RotaryBatch -> PreparedDecoder (prefill + decode).
    weights = ModelWeights.load(manifest, device, mtp=False, vision=True)
    pool = ContinuationPool.for_model(weights.config, 4, blocks, device, mtp=False, rotary_offset=True)
    requests = DeviceRequestState.allocate(device, vocab_size=weights.config.vocabulary, capacity=2)
    for slot in (0, 1):
        requests.generation[slot] = 1
        requests.status[slot] = RUNNING
        requests.consumed_tokens[slot] = 0
        requests.pending_token[slot] = -1
    inputs = RotaryBatchInputs.allocate(prefill_cap, blocks, device, capacity=2)
    vision_prefill = PrefillVisionInputs.allocate(prefill_cap, device)
    vision_state = VisionState(
        inputs, vision_prefill, requests,
        image_token=IMAGE_TOKEN_ID, features=FEATURES,
        hidden=weights.config.hidden, image_slots=1,
    )
    vision_inputs = VisionInputs.allocate(device)
    completion = CompletionOutputs.allocate((2, 2, VECTOR_TOKENS_FIELD + 1), device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.ones(1, dtype=torch.int64, device=device)

    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        arena = ScratchArena(device)
        vscratch = VisionScratch.allocate(num_patches, device, weights.config.vision["out_hidden_size"])
        with torch.device("meta"):
            vmodule = Qwen3_5VisionModel(Qwen3_5VisionConfig.from_dict(weights.config.vision))
        vmodule.eval().requires_grad_(False)
        encoder = PreparedVision(
            vmodule, vscratch, VisionMetadata.allocate(num_patches, 1, device),
            weights.vision, session=session, name="qwen.ref.vision",
            projection_workspace=arena, attention_workspace=arena,
            attention_math_mode=vision_attention_math_mode(device),
        )
        prefill_meta = Metadata.allocate(prefill_cap, 1, blocks, device)
        prefill_scratch = Scratch.allocate(prefill_cap, 1, weights.config, device)
        prefill_decoder = PreparedDecoder(
            weights, prefill_scratch, prefill_meta, pool, mode="prefill",
            vocabulary_runtime=None, coordinator=None, session=session,
            name="qwen.ref.prefill", workspace=arena, vision_state=vision_state,
        )
        prefill_batch = prepare_batch(
            inputs, requests, prefill_meta, mode="prefill", shard=weights.shard,
            pool=pool, validate_prefill=vision_state.validate_prefill,
        ).functional()
        decode_meta = Metadata.allocate(1, 1, blocks, device)
        decode_scratch = Scratch.allocate(1, 1, weights.config, device)
        decode_decoder = PreparedDecoder(
            weights, decode_scratch, decode_meta, pool, mode="decode",
            vocabulary_runtime=None, coordinator=None, session=session,
            name="qwen.ref.decode", workspace=arena,
        )
        decode_batch = prepare_batch(
            inputs, requests, decode_meta, mode="decode", shard=weights.shard, pool=pool,
        ).functional()
        arena.finalize()
        encoder.functional(session=session)
        vision_gen = PreparedVisionGeneration(encoder, vision_inputs, vision_state, completion, frame, sequence)
        session.freeze()

        # Run vision publication.
        indices, interp_weights = get_vision_interpolation_indices_and_weights(
            image_grid_thw, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2
        )
        vision_inputs.vision_pixel_values[:num_patches].copy_(pixel_values)
        vision_inputs.vision_interpolation_indices[:num_patches].copy_(indices)
        vision_inputs.vision_interpolation_weights[:num_patches].copy_(interp_weights)
        vision_inputs.vision_position_ids[:num_patches].copy_(get_vision_position_ids(image_grid_thw, 2))
        vision_inputs.vision_cu_seqlens[0] = 0
        vision_inputs.vision_cu_seqlens[1] = num_patches
        vision_inputs.vision_num_patches.fill_(num_patches)
        vision_inputs.vision_slot.zero_()
        vision_inputs.vision_generation.fill_(1)
        vision_inputs.vision_feature_offset.zero_()
        vision_prefill.vision_feature_slot.zero_()

        _, vision_errors = vision_gen.forward()
        torch.cuda.synchronize()
        assert int(vision_errors.item()) == 0
        assert int(vision_state.count[0].item()) == num_image_tokens
        assert int(vision_state.generation[0].item()) == 1
        actual_features = vision_state.arena[0, :num_image_tokens].float().cpu()

        # Run text-image-text prefill.
        inputs.token_ids[:seq_len].copy_(input_ids[0])
        inputs.prefill_positions[:seq_len].copy_(ref_positions)
        inputs.row_slot_ids[0] = 0
        inputs.row_generations[0] = 1
        inputs.request_state_rows[0, 0] = 1
        inputs.live_rows[0] = 1
        inputs.query_lengths[0] = seq_len
        inputs.page_tables[0, :blocks].copy_(torch.arange(blocks, dtype=torch.int32, device=device))
        token_indices = torch.full((seq_len,), -1, dtype=torch.int32)
        image_mask = input_ids[0] == IMAGE_TOKEN_ID
        token_indices[image_mask] = torch.arange(num_image_tokens, dtype=torch.int32)
        vision_prefill.vision_token_indices[:seq_len].copy_(token_indices)

        validation, present = prefill_batch.validate()
        meta = prefill_batch.begin(validation, present=present)
        assert int(meta.lengths[0].item()) == seq_len
        _, prefill_logits, _, _ = prefill_decoder(meta)
        prefill_batch.finish(meta)
        torch.cuda.synchronize()
        assert int(pool.region(ROTARY_OFFSET)[1, 0].item()) == expected_delta
        actual_prefill_logits = prefill_logits[0].float().cpu()
        next_token = int(actual_prefill_logits.argmax().item())

        # Run 1 decode step to verify persisted rotary_offset continuation.
        requests.consumed_tokens[0] = seq_len
        requests.pending_token[0] = next_token
        inputs.query_lengths[0] = 1
        validation, present = decode_batch.validate()
        dmeta = decode_batch.begin(validation, present=present)
        assert int(dmeta.lengths[0].item()) == 1
        assert int(dmeta.rotary_positions[0, 0].item()) == seq_len + expected_delta
        _, decode_logits, _, _ = decode_decoder(dmeta)
        decode_batch.finish(dmeta)
        torch.cuda.synchronize()
        actual_decode_logits = decode_logits[0].float().cpu()

        # Run text-only baseline prefill + decode on slot 1 / state row 2.
        inputs.token_ids[:text_seq_len].copy_(text_input_ids[0])
        inputs.prefill_positions[:text_seq_len].copy_(text_positions)
        inputs.row_slot_ids[0] = 1
        inputs.row_generations[0] = 1
        inputs.request_state_rows[1, 0] = 2
        inputs.live_rows[0] = 1
        inputs.query_lengths[0] = text_seq_len
        inputs.page_tables[1, :blocks].copy_(torch.arange(blocks, dtype=torch.int32, device=device))
        vision_prefill.vision_feature_slot.fill_(-1)
        vision_prefill.vision_token_indices[:text_seq_len].fill_(-1)

        validation, present = prefill_batch.validate()
        tmeta = prefill_batch.begin(validation, present=present)
        assert int(tmeta.lengths[0].item()) == text_seq_len
        _, text_prefill_logits, _, _ = prefill_decoder(tmeta)
        prefill_batch.finish(tmeta)
        torch.cuda.synchronize()
        assert int(pool.region(ROTARY_OFFSET)[2, 0].item()) == 0
        actual_text_prefill_logits = text_prefill_logits[0].float().cpu()
        text_next_token = int(actual_text_prefill_logits.argmax().item())

        requests.consumed_tokens[1] = text_seq_len
        requests.pending_token[1] = text_next_token
        inputs.query_lengths[0] = 1
        validation, present = decode_batch.validate()
        tdmeta = decode_batch.begin(validation, present=present)
        assert int(tdmeta.lengths[0].item()) == 1
        assert int(tdmeta.rotary_positions[0, 0].item()) == text_seq_len
        _, text_decode_logits, _, _ = decode_decoder(tdmeta)
        decode_batch.finish(tdmeta)
        torch.cuda.synchronize()
        actual_text_decode_logits = text_decode_logits[0].float().cpu()

    del (weights, pool, requests, inputs, vision_prefill, vision_state, vision_inputs,
         completion, frame, sequence, arena, vscratch, vmodule, encoder, vision_gen,
         prefill_meta, prefill_scratch, prefill_decoder, prefill_batch,
         decode_meta, decode_scratch, decode_decoder, decode_batch)
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Independent BF16-decoded Transformers Qwen3_5ForConditionalGeneration reference.
    reference = _decode_dense_reference(manifest, device)
    try:
        ref_img_out = reference.model.get_image_features(
            pixel_values.to(device=device, dtype=torch.bfloat16),
            image_grid_thw.to(device=device),
        )
        if hasattr(ref_img_out, "pooler_output"):
            ref_features = ref_img_out.pooler_output
        else:
            ref_features = ref_img_out
        if isinstance(ref_features, (list, tuple)):
            ref_features = torch.cat(list(ref_features), dim=0)
        ref_features = ref_features.reshape(num_image_tokens, -1).float().cpu()

        ref_prefill = reference(
            input_ids=input_ids.to(device),
            attention_mask=encoded["attention_mask"].to(device),
            pixel_values=pixel_values.to(device=device, dtype=torch.bfloat16),
            image_grid_thw=image_grid_thw.to(device),
            mm_token_type_ids=mm_token_type_ids.to(device),
            use_cache=True,
        )
        ref_prefill_logits = ref_prefill.logits[0, -1].float().cpu()

        assert int(reference.model.rope_deltas.item()) == expected_delta
        decode_position_ids = (
            torch.full((3, 1, 1), seq_len, dtype=torch.long, device=device)
            + reference.model.rope_deltas.to(device)
        )
        ref_decode = reference(
            input_ids=torch.tensor([[next_token]], dtype=torch.long, device=device),
            attention_mask=torch.ones((1, seq_len + 1), dtype=torch.long, device=device),
            position_ids=decode_position_ids,
            past_key_values=ref_prefill.past_key_values,
            cache_position=torch.tensor([seq_len], dtype=torch.long, device=device),
            use_cache=True,
        )
        ref_decode_logits = ref_decode.logits[0, -1].float().cpu()

        # Text-only baseline through the same BF16 reference.
        reference.model.rope_deltas = None
        ref_text_prefill = reference(
            input_ids=text_input_ids.to(device),
            attention_mask=text_encoded["attention_mask"].to(device),
            use_cache=True,
        )
        ref_text_prefill_logits = ref_text_prefill.logits[0, -1].float().cpu()
        ref_text_decode = reference(
            input_ids=torch.tensor([[text_next_token]], dtype=torch.long, device=device),
            attention_mask=torch.ones((1, text_seq_len + 1), dtype=torch.long, device=device),
            past_key_values=ref_text_prefill.past_key_values,
            cache_position=torch.tensor([text_seq_len], dtype=torch.long, device=device),
            use_cache=True,
        )
        ref_text_decode_logits = ref_text_decode.logits[0, -1].float().cpu()
    finally:
        del reference
        gc.collect()
        torch.cuda.empty_cache()

    def metrics(actual, expected):
        cos = float(torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0))
        rel_l2 = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
        return dict(
            cosine=round(cos, 6),
            relative_l2=round(rel_l2, 6),
            actual_top5=actual.topk(5).indices.tolist() if actual.ndim == 1 else None,
            ref_top5=expected.topk(5).indices.tolist() if expected.ndim == 1 else None,
        )

    feat_m = metrics(actual_features, ref_features)
    prefill_m = metrics(actual_prefill_logits, ref_prefill_logits)
    decode_m = metrics(actual_decode_logits, ref_decode_logits)
    text_prefill_m = metrics(actual_text_prefill_logits, ref_text_prefill_logits)
    text_decode_m = metrics(actual_text_decode_logits, ref_text_decode_logits)
    print(
        dict(
            seq_len=seq_len,
            rotary_delta=expected_delta,
            next_token_text=processor.tokenizer.decode([next_token]),
            features=feat_m,
            prefill_logits=prefill_m,
            decode_logits=decode_m,
            text_seq_len=text_seq_len,
            text_next_token_text=processor.tokenizer.decode([text_next_token]),
            text_prefill_logits=text_prefill_m,
            text_decode_logits=text_decode_m,
        ),
        flush=True,
    )
    # Vision features share the operator-rounding bound established in
    # `tests/recipes/test_qwen38_vision.py` (_DENSE_RELATIVE_L2 = 0.054).
    # Prefill and decode logits compare 64 layers of NVFP4 MLP / FP8 attention /
    # FP8 KV execution against BF16-dequantized Transformers alongside a
    # text-only baseline in the same session.
    assert feat_m["relative_l2"] < 0.054 and feat_m["cosine"] > 0.999, feat_m
    assert prefill_m["cosine"] > 0.95, prefill_m
    assert prefill_m["actual_top5"][0] == prefill_m["ref_top5"][0], prefill_m
    assert len(set(prefill_m["actual_top5"]) & set(prefill_m["ref_top5"])) >= 4, prefill_m
    assert decode_m["cosine"] > 0.95, decode_m
    assert decode_m["actual_top5"][0] == decode_m["ref_top5"][0], decode_m
    assert len(set(decode_m["actual_top5"]) & set(decode_m["ref_top5"])) >= 3, decode_m
    assert text_prefill_m["cosine"] > 0.95, text_prefill_m
    assert text_prefill_m["actual_top5"][0] == text_prefill_m["ref_top5"][0], text_prefill_m
    assert text_decode_m["cosine"] > 0.95, text_decode_m
    assert text_decode_m["actual_top5"][0] == text_decode_m["ref_top5"][0], text_decode_m
