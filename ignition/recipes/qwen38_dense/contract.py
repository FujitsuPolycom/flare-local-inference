# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense Qwen text and image inputs, recurrent continuations, and paged attention storage."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p
from ignition.recipes.qwen38.contract import FEATURE_TOKEN, PATCH_CAPACITIES, VISION, frontend

from .config import PAGE_TOKENS


def prepare(signatures, buffers, pool, *, vocabulary, image_slots=0):
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
        frontend=frontend(vision=vision))
