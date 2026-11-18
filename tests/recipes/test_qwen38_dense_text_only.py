# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Text-only Qwen3.8 27B preparation declares the text-only contract of commit 5fd4070d of this repository.

A preparation without image graphs assigns every token its index as its rotary
position, so it needs neither three-axis prompt positions nor a continuation
offset. These checks pin that declaration for one synthetic configuration:
continuation regions and row bytes with and without DFlash2, the contract
document and its SHA-256, graph input fields, and the tuning identity. Every
`RECORDED_*` value is what the `qwen38_dense` recipe at commit 5fd4070d of this repository
computes for the configuration built here; the tests rebuild the same values from
this tree's recipe.
"""

import hashlib
import json
from dataclasses import replace

import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.paged_batch import BatchInputs
from ignition.components.sampling import AdmissionBuffers
from ignition.components.state_transfer import StateCommands
from ignition.preparation.execution import GraphSignature
from ignition.recipes.qwen38_dense import bootstrap, contract
from ignition.recipes.qwen38_dense.config import ModelConfig
from ignition.recipes.qwen38_dense.pool import ContinuationPool, continuation_layout

# Values of commit 5fd4070d of this repository for this synthetic configuration.
RECORDED_ROW_BYTES = {False: 3_211_776, True: 3_356_160}
RECORDED_REGIONS = {
    False: [("layer0.convolution", (10240, 3)), ("layer0.recurrent", (48, 128, 128)),
            ("layer1.key", (2048,)), ("layer1.value", (2048,)),
            ("last_hidden", (64,)), ("consumed_tokens", (1,))],
    True: [("layer0.convolution", (10240, 10)), ("layer0.recurrent", (48, 128, 128)),
           ("layer1.key", (2048,)), ("layer1.value", (2048,)),
           ("draft0.key", (64, 1, 8)), ("draft0.value", (64, 1, 8)),
           ("last_hidden", (64,)), ("consumed_tokens", (1,))],
}
RECORDED_CONTRACT_SHA256 = "85d8539d1ea4d304c053211c4142f5b4d5f0187580720dd7937971a7259a7d9f"
RECORDED_SIGNATURE_INPUTS = {
    "decode": ("request_state_rows", "row_slot_ids", "row_generations", "live_rows",
               "query_lengths", "page_tables", "sample_enabled"),
    "prefill": ("request_state_rows", "row_slot_ids", "row_generations", "live_rows",
                "query_lengths", "token_ids", "page_tables", "sample_enabled"),
    "speculative": ("request_state_rows", "row_slot_ids", "row_generations", "live_rows",
                    "query_lengths", "page_tables", "sample_enabled"),
}
# Tuning identity of commit 5fd4070d of this repository for RECORDED_IDENTITY_INPUTS.
RECORDED_IDENTITY = "dfacc12862405d9d364192a4cad7cef25085d801f1de4082f195aaeef40333c0"
RECORDED_IDENTITY_INPUTS = dict(
    config={"hidden_size": 64, "vocab_size": 128},
    headers=["header-a", "header-b"],
    draft={"hidden_size": 64, "num_hidden_layers": 1},
    draft_headers=["draft-header"],
    draft_block_size=8,
    study_widths=(2, 4),
    online_mxfp8=True,
    draft_head_quantization=None,
)
CAPACITY, PREFILL, VOCABULARY = 4, 8, 128


def small_config(mtp=False):
    text = dict(hidden_size=64, vocab_size=128, num_hidden_layers=2, num_key_value_heads=2,
        head_dim=8, linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"])
    model = ModelConfig(text, {})
    if not mtp:
        return model
    from ignition.components.dflash2_geometry import DFlash2Config

    raw = dict(model_type="qwen3", architectures=["DFlash2DraftModel"], hidden_act="silu",
        dtype="bfloat16", attention_bias=False, attention_dropout=0., tie_word_embeddings=False,
        use_sliding_window=True, is_causal=False, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=128, num_target_layers=2, rms_norm_eps=1e-6, sliding_window=64,
        layer_types=["sliding_attention"], rope_parameters=dict(rope_type="default", rope_theta=10000),
        dflash_config=dict(block_size=8, conv_kernel_size=2, conv_group_size=16, selector_rank=16,
            selector_top_k=4, mask_token_id=1, target_layer_ids=[0, 1]))
    return replace(model, draft=DFlash2Config.from_dict(raw, target_hidden=64,
        target_vocabulary=128, target_layers=2, expected=raw))


def declaration(config):
    """The text-only contract and signatures the bootstrap declares for ``config``."""
    from ignition.components.prompt_logprobs import PromptOutputs

    pool = ContinuationPool.for_model(config, CAPACITY + 1, 2, "cpu")
    inputs = BatchInputs.allocate(PREFILL, 2, "cpu", capacity=CAPACITY)
    buffers = inputs.buffers()
    admission = AdmissionBuffers.allocate("cpu", vocab_size=VOCABULARY, capacity=CAPACITY)
    buffers.update({name: getattr(admission, name) for name in admission.__dataclass_fields__})
    buffers.update(StateCommands.allocate("cpu", capacity=CAPACITY).buffers())
    buffers.update(CompletionOutputs.allocate((2, CAPACITY, 16 + 8), device="cpu").buffers())
    buffers.update(PromptOutputs(PREFILL, "cpu").buffers())
    buffers.update(state_pool=pool.backing, memory_capacity=pool.memory_capacity,
        request_pending_token=torch.zeros(CAPACITY, dtype=torch.int64),
        sampling_grammar_mask=torch.zeros((CAPACITY, (VOCABULARY + 31) // 32), dtype=torch.int32))
    signatures = {
        "decode": GraphSignature("decode", CAPACITY, 1, tuple(inputs.buffers())),
        "prefill": GraphSignature("prefill", 1, PREFILL, tuple(inputs.buffers())),
        "speculative": GraphSignature("speculative", 2, 8, tuple(inputs.buffers()),
            speculation_enabled=True, packed_token_capacity=8),
    }
    document = contract.prepare(signatures, buffers, pool, vocabulary=VOCABULARY)
    return document, signatures


def test_text_only_continuation_declaration_matches_the_recorded_layout():
    for drafting in (False, True):
        config = small_config(mtp=drafting)
        layout = continuation_layout(config, mtp=drafting, verification_width=8)
        assert [(region.name, region.shape) for region in layout.regions] == RECORDED_REGIONS[drafting]
        assert layout.row_bytes == RECORDED_ROW_BYTES[drafting]
        assert "rope_delta" not in {region.name for region in layout.regions}
        # Three-axis preparation adds the offset region and nothing else.
        three_axis = continuation_layout(config, mtp=drafting, verification_width=8, rotary_offset=True)
        assert three_axis.row_bytes == layout.row_bytes + 256
        assert three_axis.regions[-1].name == "rope_delta"


def test_text_only_contract_and_signatures_match_the_recorded_contract():
    document, signatures = declaration(small_config(mtp=True))
    encoded = json.dumps(document, sort_keys=True)
    assert hashlib.sha256(encoded.encode()).hexdigest() == RECORDED_CONTRACT_SHA256
    assert document["frontend"]["position_dimensions"] == 0
    assert document["frontend"]["image_tokens"] is None
    for name, fields in RECORDED_SIGNATURE_INPUTS.items():
        assert signatures[name].inputs == fields
        inputs = next(graph for graph in document["graphs"] if graph["graph"] == name)["inputs"]
        assert "prefill_positions" not in {binding["buffer"] for binding in inputs}
    assert document["vision"]["max_images"] == 0


def test_text_only_identity_matches_the_recorded_identity():
    identity = bootstrap.session_identity(**RECORDED_IDENTITY_INPUTS)
    assert identity == RECORDED_IDENTITY
    # An image preparation declares the version-2 graph inputs and a distinct identity.
    image = dict(RECORDED_IDENTITY_INPUTS,
        vision={"image_slots": 4, "patch_capacities": [256]})
    assert bootstrap.session_identity(**image) != identity
