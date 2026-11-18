# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense Qwen text and image inputs, recurrent continuations, and paged attention storage."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p

from .config import PAGE_TOKENS


FEATURE_TOKEN = 248056
VIDEO_FEATURE_TOKEN = 248057
PATCH_CAPACITIES = (256, 1024, 4096, 16384)
VISION = dict(features_per_request=16384, max_images=4, patches_per_feature=4,
              patch_capacities=list(PATCH_CAPACITIES))


def frontend(vision=True, video=False):
    """Qwen chat markers; with ``vision``, Qwen image preparation and feature placement.

    ``video`` declares the video placeholder token, which requires video settings prepared
    from the checkpoint.
    """
    images = {}
    if vision:
        image_tokens = dict(feature_token=FEATURE_TOKEN, span=None)
        if video:
            image_tokens["video_feature_token"] = VIDEO_FEATURE_TOKEN
        images = dict(image_tokens=image_tokens, image_preparation=p.QWEN_IMAGE_PREPARATION,
                      image_layout=p.QWEN_IMAGE_LAYOUT)
    return c.frontend(tools=p.QWEN_CODER, wrapping_newlines=1, terminal_markers=["<tool_call>"],
        assistant_marker="<|im_start|>assistant\n", message_end_markers=["<|im_end|>"],
        prefix_boundary=c.prefix_boundary(["system", "developer"], append_role="user", minimum_messages=0,
            required_start="<|im_start|>system\n", required_marker="<|im_end|>"),
        position_dimensions=3 if vision else 0, **images)


def prepare(signatures, buffers, pool, *, vocabulary, image_slots=0, video=False):
    """Declare the text contract, or with ``image_slots`` the image front end and vision graphs."""
    vision = image_slots > 0
    kinds = {s.kind for s in signatures.values()}
    if "vision" in kinds and not vision:
        raise ValueError("vision graphs require the image frontend and feature storage")
    inputs = {kind: c.text_inputs(kind) for kind in kinds if kind != "vision"}
    if vision:
        inputs["vision"] = c.vision_inputs(
            {field for s in signatures.values() if s.kind == "vision" for field in s.inputs})
    if "decode" in inputs:
        inputs["decode"].pop("token_ids", None)
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    mtp = any(s.speculation_enabled for s in signatures.values())
    consumed = pool.layout.region("consumed_tokens") if mtp else None
    cursor = dict(state=c.state_view("state_pool", pool.layout.row_bytes,
        offset=consumed.offset, nbytes=consumed.nbytes), lag_tokens=0) if mtp else None
    pending = c.state_view("request_pending_token", 8)
    regions = [c.history_region(buffers, f"cache.{name}", tensor, p.PAGED_KEY_VALUE)
               for name, tensor in pool.pages.items()]
    graphs = c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending, cursor=cursor)
    if vision:
        missing = [graph["graph"] for graph in graphs
                   if graph["kind"] in (p.PREFILL, p.SCORED_PREFILL)
                   and not any(value["source"] == p.PROMPT_POSITION for value in graph["inputs"])]
        if missing:
            raise ValueError("three-axis prefill graphs must bind prompt positions: " + ", ".join(missing))
    return c.model_contract(vocabulary=vocabulary, page_tokens=PAGE_TOKENS,
        page_tables=[dict(buffer="page_tables", page_offset=0)],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0),
                 *regions],
        graphs=graphs,
        vision={**VISION, "feature_slots": image_slots} if vision else None,
        frontend=frontend(vision=vision, video=video))
