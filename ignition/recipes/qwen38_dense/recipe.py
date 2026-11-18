# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU Qwen3.8-27B NVFP4 text and image preparation without host-mapped storage."""

from types import MappingProxyType
from pathlib import Path

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange, Speculation


# Image feature slot: 16,384 BF16 vectors at the language-model width 5,120.
VISION_SLOT_BYTES = 16_384 * 5_120 * 2  # 167,772,160
VISION_FEATURE_BYTES = 4 * VISION_SLOT_BYTES  # 671,088,640
# Captured encoder scratch for patch capacities 256, 1,024, 4,096, and 16,384.
# `VisionScratch` holds 18,640 BF16 elements per patch across its eleven tensors:
# patches 1,536 + hidden 1,152 + qkv 3,456 + q/k/v 3 * 16 * 72 + attention
# 1,152 + projected 1,152 + intermediate 4,304 + merger 1,152 + output 1,280.
VISION_SCRATCH_BYTES = (256 + 1_024 + 4_096 + 16_384) * 18_640 * 2  # 811,212,800
# Float32 patch upload and its interpolation, position, and counter metadata.
VISION_INPUT_BYTES = 101_711_900
# Per captured handoff: `VisionMetadata` (64 bytes per patch plus 12) and the
# 4-byte error scalar. The sanitized patch staging is the scratch `patches`
# tensor above, not a second copy. Summed over the four capacities:
# 1,392,688 + 16.
VISION_HANDOFF_BYTES = 1_392_704
# BF16 vision tower weights under `model.visual.*` in the NVFP4 checkpoint.
VISION_TOWER_BYTES = 921_460_192
# Graph-private temporaries of the 16,384-patch forward. Its largest live set is
# the position-embedding path: the [16384, 4, 1152] BF16 gather (150,994,944),
# the [16384, 4, 1152] FP32 weighted product (301,989,888), and the BF16
# [16384, 1152] rounded positions (37,748,736). Inductor may reuse part of that
# set; this reservation keeps the unfused bound.
VISION_GRAPH_BYTES = 150_994_944 + 301_989_888 + 37_748_736  # 490,733,568
VISION_BYTES = (VISION_FEATURE_BYTES + VISION_SCRATCH_BYTES + VISION_INPUT_BYTES
                + VISION_HANDOFF_BYTES + VISION_TOWER_BYTES + VISION_GRAPH_BYTES)


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    from .bootstrap import PreparationConfig, prepare as prepare_model

    if plan.world_size != 1 or ranks is not None:
        raise ValueError("Qwen dense preparation supports one local GPU")
    speculation = plan.speculation
    drafting = {}
    if speculation is not None:
        from huggingface_hub import snapshot_download
        from .draft_config import REPOSITORY, REVISION

        drafting = dict(draft_snapshot=Path(snapshot_download(REPOSITORY, revision=REVISION,
            cache_dir=plan.hub_cache, allow_patterns=("config.json", "model.safetensors"))),
            draft_head_quantization=speculation.draft_head_quantization,
            speculative_graph_budget=speculation.graph_budget or 64)
    selection = plan.benchmark_selection
    config = PreparationConfig(mtp_only=plan.speculation is not None and plan.speculation.mtp_only,
        benchmark_selection=selection, collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder, prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        online_mxfp8=plan.online_mxfp8,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        max_active=plan.concurrency if selection is None else selection.max_batch, context=plan.context,
        image_concurrency=plan.image_concurrency,
        prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache, mtp=speculation is not None,
        speculative_widths=(1,) if speculation is None else speculation.widths,
        speculative_fixed_width=None if speculation is None else speculation.fixed_width,
        speculative_token_capacity=None if speculation is None else plan.speculative_token_capacity, **drafting)
    return prepare_model(snapshot, config, progress=progress, core_storage=core_storage)


def usage(prepared):
    return (RankUsage(prepared.memory_report["pages"], prepared.memory_report["free_device_bytes"]),)


def video_unsupported(snapshot):
    """Why the snapshot's video processor cannot be prepared, or None.

    Preparation with image slots exports the processor's settings when the native
    frontend can consume them, and an image core that rejects video otherwise.
    """
    from ignition.preparation.frontend import video_capability

    return video_capability(snapshot)[1]


RECIPE = Recipe(
    name="qwen38_dense", served_model_name="Qwen3.8-27B", prepare=prepare, usage=usage,
    # Thinking defaults; bounds span thinking/instruct presets: https://huggingface.co/Qwen/Qwen3.8-27B#best-practices.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=20, min_p=0.,
        presence_penalty=0., repetition_penalty=1., frequency_penalty=0.,
        ranges={"temperature": SamplingRange(.7, 1.), "top_p": SamplingRange(.8, .95),
                "presence_penalty": SamplingRange(0., 2.)}),
    checkpoints=(Checkpoint(alias="qwen3.8-27b", repository="nvidia/Qwen3.8-27B-NVFP4",
        revision="482ca0f3832238542f8f5295dde86b5f22711d80", architecture="Qwen3_5ForConditionalGeneration",
        stored_bytes=21_921_428_072, device_bytes=24_104_235_656, draft_bytes=3_953_666_560,
        expansion_bytes=307_871_744),
        Checkpoint(alias="qwen3.8-27b-qad", repository="local-inference-lab/Qwen3.8-27B-NVFP4-QAD",
        revision="b9389787cbb5a09fbcf3517bdf673b5a56b16b02", architecture="Qwen3_5ForConditionalGeneration",
        stored_bytes=25_578_292_192, device_bytes=27_761_099_776, draft_bytes=3_953_666_560,
        expansion_bytes=1_619_001_344)),
    context=262144, page_tokens=128, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7, token_capacity=128, packed_verification=True, graph_budget=64,
                            method="dflash2"), max_draft_tokens=15, verification_rows=0,
    online_mxfp8=True, draft_head_quantizations=("mxfp8", "nvfp4"),
    max_images=4, image_bytes=32 << 20, language_model_only=True, default_image_slots=0,
    # Reserve continuation memory for the supported sixteen-token verification capacity.
    # A plan with image slots stores the three-axis rotary offset in a row 256 bytes larger.
    ranks=MappingProxyType({1: RankCosts(193_865_984, 4_194_304, 4 << 30,
        vision_bytes=VISION_BYTES, vision_slot_bytes=VISION_SLOT_BYTES,
        image_state_row_bytes=193_866_240)}),
    plain_ranks=MappingProxyType({1: RankCosts(158_148_864, 4_194_304, 4 << 30,
        vision_bytes=VISION_BYTES, vision_slot_bytes=VISION_SLOT_BYTES,
        image_state_row_bytes=158_149_120)}),
    vision_prefixes=("model.visual.",), video_unsupported=video_unsupported,
)
