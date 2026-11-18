# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 27B image feature transfer at the 5120-wide language-model width."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

CUDA = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned GPU"
)
WIDTH = 5120


def _snapshot():
    value = os.environ.get("FLARE_QWEN38_DENSE_SNAPSHOT")
    if not value:
        pytest.skip("requires FLARE_QWEN38_DENSE_SNAPSHOT naming a Qwen3.8-27B checkpoint")
    path = Path(value)
    if not path.is_dir():
        pytest.skip("FLARE_QWEN38_DENSE_SNAPSHOT does not name a checkpoint directory")
    return path


def _device():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    return device


def test_dense_contract_declares_the_qwen_image_frontend():
    from ignition.preparation import runtime_core_pb2 as p
    from ignition.recipes.qwen38_dense import contract

    assert contract.PATCH_CAPACITIES == (256, 1024, 4096, 16384)
    assert contract.VISION == dict(features_per_request=16384, max_images=4,
        patches_per_feature=4, patch_capacities=[256, 1024, 4096, 16384])
    frontend = contract.frontend()
    assert frontend["image_tokens"] == dict(feature_token=248056, span=None)
    assert frontend["image_preparation"] == p.QWEN_IMAGE_PREPARATION
    assert frontend["image_layout"] == p.QWEN_IMAGE_LAYOUT
    assert frontend["position_dimensions"] == 3
    assert contract.frontend(vision=False)["image_tokens"] is None
    assert contract.frontend(vision=False)["position_dimensions"] == 0


def test_dense_recipe_declares_the_flash_image_limits():
    from ignition.recipes.qwen38.recipe import RECIPE as FLASH
    from ignition.recipes.qwen38_dense.recipe import (
        RECIPE, VISION_BYTES, VISION_INPUT_BYTES, VISION_SLOT_BYTES, VISION_TOWER_BYTES,
    )

    assert (RECIPE.max_images, RECIPE.image_bytes) == (FLASH.max_images, FLASH.image_bytes) == (4, 32 << 20)
    assert RECIPE.language_model_only
    assert VISION_SLOT_BYTES == 167_772_160
    assert VISION_INPUT_BYTES == 101_711_900
    assert VISION_TOWER_BYTES == 921_460_192
    assert VISION_BYTES == VISION_TOWER_BYTES + VISION_INPUT_BYTES
    for costs in (RECIPE.ranks[1], RECIPE.plain_ranks[1]):
        assert costs.graph_bytes == 4 << 30
        assert costs.vision_bytes == VISION_BYTES
        assert costs.vision_slot_bytes == VISION_SLOT_BYTES


def test_benchmark_selection_takes_no_image_slots():
    """A benchmark prepares text graphs only, so its configuration rejects image slots."""
    from ignition.preparation.selection import BenchmarkSelection
    from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig

    selection = BenchmarkSelection(batches=(1,), contexts=(0,), prefill_chunk=4096)
    config = PreparationConfig(draft_snapshot=Path("draft"), max_active=1,
        prefill_chunk_size=4096, benchmark_selection=selection)
    assert config.image_concurrency == 0
    with pytest.raises(ValueError, match="takes no image slots"):
        PreparationConfig(draft_snapshot=Path("draft"), max_active=1, image_concurrency=1,
            prefill_chunk_size=4096, benchmark_selection=selection)
    assert PreparationConfig(mtp=False).image_concurrency == 0
    assert PreparationConfig(mtp=False, max_active=8, image_concurrency=2).image_concurrency == 2
    with pytest.raises(ValueError, match="max_active"):
        PreparationConfig(mtp=False, max_active=2, image_concurrency=4)
    with pytest.raises(ValueError, match="non-negative"):
        PreparationConfig(mtp=False, image_concurrency=-1)


def test_plan_reserves_vision_storage_only_for_image_slots():
    from ignition import planning
    from ignition.recipes import resolve
    from ignition.recipes.qwen38_dense.recipe import RECIPE, VISION_BYTES, VISION_SLOT_BYTES

    model = resolve("qwen3.8-27b")
    gpus = (planning.Gpu(0, "SM120", "GPU-0", 96 << 30, 95 << 30, (12, 0)),)
    options = dict(concurrency=8, reserve_bytes=0, speculation=RECIPE.speculation)
    default = planning.fit(RECIPE, model.checkpoint, gpus, **options)
    language_only = planning.fit(RECIPE, model.checkpoint, gpus, **options, language_model_only=True)
    one = planning.fit(RECIPE, model.checkpoint, gpus, **options, image_concurrency=1)
    wide = planning.fit(RECIPE, model.checkpoint, gpus, **options, image_concurrency=8)
    assert (language_only.image_concurrency, language_only.max_images) == (0, 0)
    assert language_only.graph_bytes == 4 << 30
    assert (default.image_concurrency, default.max_images) == (4, 4)
    assert default.graph_bytes - language_only.graph_bytes == VISION_BYTES + 4 * VISION_SLOT_BYTES
    assert one.graph_bytes - language_only.graph_bytes == VISION_BYTES + VISION_SLOT_BYTES
    assert wide.graph_bytes - default.graph_bytes == 4 * VISION_SLOT_BYTES
    # A plan that prepares no vision uses the text-only continuation row.
    plain = planning.fit(RECIPE, model.checkpoint, gpus, concurrency=8, reserve_bytes=0,
        speculation=None, language_model_only=True)
    assert language_only.costs == RECIPE.costs(1, RECIPE.speculation, model.checkpoint)
    assert language_only.costs.state_row_bytes == 193_865_984
    assert plain.costs.state_row_bytes == 158_148_864
    assert default.costs.state_row_bytes == 193_866_240
    assert default.costs.graph_bytes == default.graph_bytes
    assert (one.max_images, wide.max_images) == (4, 4)


def test_plan_shortfall_on_32gib_sm120_suggests_language_model_only_when_applicable():
    from ignition import planning
    from ignition.errors import CapacityError, UsageError
    from ignition.recipes import resolve
    from ignition.recipes.qwen38_dense.recipe import RECIPE

    model = resolve("qwen3.8-27b")
    gpus = (planning.Gpu(0, "NVIDIA GeForce RTX 5090", "GPU-0", 32 << 30, 31 << 30, (12, 0)),)
    # Four slots with default four image slots exceed 30 GiB usable (31 GiB free - 1 GiB headroom) and
    # suggest --language-model-only (28.6 GiB) alongside --max-concurrency and --no-mtp.
    with pytest.raises(CapacityError) as default_error:
        planning.plan(RECIPE, model.checkpoint, gpus, selected=(0,), concurrency=4,
            reserve_bytes=0, prefill_chunk=None, speculation=RECIPE.speculation)
    assert "--language-model-only" in str(default_error.value)
    # Explicit --image-concurrency cannot be combined with --language-model-only, so shortfall suggests
    # reducing --image-concurrency instead and omits --language-model-only.
    with pytest.raises(CapacityError) as explicit_error:
        planning.plan(RECIPE, model.checkpoint, gpus, selected=(0,), concurrency=4,
            reserve_bytes=0, prefill_chunk=None, speculation=RECIPE.speculation,
            image_concurrency=4)
    assert "--language-model-only" not in str(explicit_error.value)
    assert "--image-concurrency" in str(explicit_error.value)
    # Direct callers cannot combine language_model_only=True with an explicit image_concurrency.
    with pytest.raises(UsageError, match="--language-model-only"):
        planning.fit(RECIPE, model.checkpoint, gpus, concurrency=4, reserve_bytes=0,
            speculation=RECIPE.speculation, language_model_only=True, image_concurrency=4)
    # Selecting --language-model-only fits four slots on the 32 GiB SM120 card even with a 1 GiB reserve.
    text_only = planning.plan(RECIPE, model.checkpoint, gpus, selected=(0,), concurrency=4,
        reserve_bytes=1 << 30, prefill_chunk=None, speculation=RECIPE.speculation,
        language_model_only=True)
    assert text_only.image_concurrency == 0 and text_only.max_images == 0


def test_dense_config_validates_vision_geometry_and_vision_weights_declares_tensors(tmp_path):
    from types import SimpleNamespace

    from ignition.preparation.rebuild_manifest import CONTAINER_FIELDS
    from ignition.recipes.qwen38.vision import VisionWeights
    from ignition.recipes.qwen38_dense.bootstrap import _declared_weight_roots
    from ignition.recipes.qwen38_dense.config import ModelConfig

    base = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text", "hidden_act": "silu", "attention_bias": False,
            "hidden_size": 5120, "intermediate_size": 17408, "vocab_size": 248320,
            "num_hidden_layers": 2, "num_attention_heads": 24, "num_key_value_heads": 4,
            "head_dim": 256, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "linear_value_head_dim": 128, "tie_word_embeddings": False,
            "max_position_embeddings": 262144, "rms_norm_eps": 1e-6,
            "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {"rope_type": "default", "rope_theta": 1e7,
                                "partial_rotary_factor": 0.25,
                                "mrope_section": [11, 11, 10], "mrope_interleaved": True},
        },
        "vision_config": {
            "model_type": "qwen3_5_vision",
            "hidden_size": 1152, "num_heads": 16, "intermediate_size": 4304, "depth": 27,
            "out_hidden_size": 5120, "num_position_embeddings": 2304, "patch_size": 16,
            "temporal_patch_size": 2, "spatial_merge_size": 2, "in_channels": 3,
        },
        "quantization_config": {
            "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None, "quantized_layers": {},
        },
        "image_token_id": 248056,
    }
    (tmp_path / "config.json").write_text(json.dumps(base))
    loaded = ModelConfig.load(tmp_path, vision=True)
    assert loaded.vision is not None and loaded.vision["out_hidden_size"] == 5120
    # Mutating any prepared vision geometry field fails closed before loading weights.
    for field, bad_value in (("depth", 24), ("num_position_embeddings", 1024),
                             ("patch_size", 14), ("spatial_merge_size", 4)):
        broken = json.loads(json.dumps(base))
        broken["vision_config"][field] = bad_value
        (tmp_path / "config.json").write_text(json.dumps(broken))
        with pytest.raises(ValueError, match="vision"):
            ModelConfig.load(tmp_path, vision=True)
    # VisionWeights registers its fields for rebuild manifests and _declared_weight_roots includes it.
    assert CONTAINER_FIELDS["ignition.recipes.qwen38.vision.VisionWeights"] == ("parameters", "projections")
    vision = VisionWeights(parameters={"p": torch.zeros(1)}, projections={})
    fake_weights = SimpleNamespace(
        tree=SimpleNamespace(embed_tokens=None, norm=None, layers=()),
        head=None, draft=None, draft_head=None, projections={}, vision=vision,
    )
    roots, components = _declared_weight_roots(fake_weights)
    assert roots["vision"] is vision and components["vision"] == "vision"
    text_only_roots, _ = _declared_weight_roots(SimpleNamespace(
        tree=SimpleNamespace(embed_tokens=None, norm=None, layers=()),
        head=None, draft=None, draft_head=None, projections={}, vision=None,
    ))
    assert "vision" not in text_only_roots


def _image_catalog():
    """Contract and manifest of a preparation with four image slots, from CPU buffers."""
    from types import SimpleNamespace

    from ignition.components.capacity import batch_capacities, prefill_capacities
    from ignition.components.completion import CompletionOutputs
    from ignition.components.paged_batch import ROTARY_OFFSET, RotaryBatchInputs
    from ignition.components.prompt_logprobs import PromptOutputs, scored_prefill_capacities
    from ignition.components.sampling import AdmissionBuffers
    from ignition.components.state import StateLayout, StateStorage
    from ignition.components.state_transfer import StateCommands
    from ignition.components.vision_state import PrefillVisionInputs
    from ignition.preparation import contract as c
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.recipes.qwen38_dense import contract
    from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig
    from ignition.recipes.qwen38_dense.vision_state import VisionInputs

    config = PreparationConfig(draft_snapshot=Path("metadata-draft"), max_active=32, image_concurrency=4)
    capacity, vocabulary = config.max_active, 248320
    inputs = RotaryBatchInputs.allocate(config.prefill_chunk_size, config.context // 128, "cpu",
        capacity=capacity)
    admission = AdmissionBuffers.allocate("cpu", vocab_size=vocabulary, capacity=capacity)
    layout = StateLayout.from_specs((("consumed_tokens", (1,), torch.int64),
        (ROTARY_OFFSET, (1,), torch.int64)), rotary_region=ROTARY_OFFSET)
    storage = StateStorage.allocate(layout, capacity * 8 + 1, "cpu")
    pool = SimpleNamespace(layout=layout, pages={}, backing=storage.backing)
    vision_inputs = VisionInputs.allocate("cpu").buffers()
    buffers = dict(inputs.buffers(),
        **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
        **StateCommands.allocate("cpu", capacity=capacity).buffers(),
        state_pool=storage.backing, memory_capacity=storage.memory_capacity,
        request_pending_token=torch.zeros(capacity, dtype=torch.int64),
        sampling_grammar_mask=torch.zeros((capacity, (vocabulary + 31) // 32), dtype=torch.int32),
        frame_index=torch.zeros(1, dtype=torch.int32),
        submission_sequence=torch.zeros(1, dtype=torch.int64),
        **CompletionOutputs.allocate((2, capacity, 16 + config.completion_token_capacity), device="cpu").buffers(),
        **vision_inputs, **PrefillVisionInputs.allocate(config.prefill_chunk_size, "cpu").buffers(),
        **PromptOutputs(config.prefill_chunk_size, "cpu").buffers())
    signatures = {}

    def add(kind, batch, width, *, packed=0):
        # The bootstrap declares the uploads its captured forwards bind.
        if kind == "vision":
            fields = (*vision_inputs, "vision_feature_slot")
        else:
            fields = tuple(name for name in c.text_inputs(kind) if name in buffers
                           and not (kind in ("decode", "speculative") and name == "prefill_positions"))
        signatures[f"{kind}-{batch}-{width}-{packed}"] = GraphSignature(kind, batch, width, fields,
            speculation_enabled=True if kind == "speculative" else None, packed_token_capacity=packed)

    for batch in batch_capacities(capacity):
        for kind in ("init", "restore", "snapshot", "resume", "decode"):
            add(kind, batch, 1)
    for tokens in prefill_capacities(config.prefill_chunk_size):
        add("prefill", 1, tokens)
    for tokens in scored_prefill_capacities(config.prefill_chunk_size):
        add("scored_prefill", 1, tokens)
    for batch, width, rows in config.speculative_round_shapes:
        add("speculative", batch, width, packed=rows)
    for tokens in contract.PATCH_CAPACITIES:
        add("vision", 1, tokens)
    document = contract.prepare(signatures, buffers, pool, vocabulary=vocabulary,
        image_slots=config.image_concurrency)
    bundle = PreparedExecution(buffers=buffers, signatures=signatures,
        graphs=dict.fromkeys(signatures), max_active=capacity, device=0,
        estimated_durations_ns=dict.fromkeys(signatures, 1))
    bundle._validate_resources = lambda: (1, list(range(1, len(signatures) + 1)),
        [(tensor.data_ptr(), tensor.nbytes) for tensor in buffers.values()])
    return document, PreparedExecution.manifest(bundle, all_variants=True)


def test_dense_image_catalog_declares_qwen_layout_and_passes_native_contract_validation():
    import flare_heliopause
    from ignition.preparation import runtime_core_pb2 as p

    contract, manifest = _image_catalog()
    assert contract["vision"] == dict(feature_slots=4, features_per_request=16384,
        max_images=4, patches_per_feature=4, patch_capacities=[256, 1024, 4096, 16384])
    frontend = contract["frontend"]
    assert frontend["image_tokens"] == dict(feature_token=248056, span=None)
    assert frontend["image_preparation"] == p.QWEN_IMAGE_PREPARATION
    assert frontend["image_layout"] == p.QWEN_IMAGE_LAYOUT
    assert frontend["position_dimensions"] == 3
    vision = [graph for graph in manifest["graphs"] if graph["kind"] == "vision"]
    assert sorted(graph["token_capacity"] for graph in vision) == [256, 1024, 4096, 16384]
    prefill = next(graph for graph in contract["graphs"] if graph["graph"] == "prefill-1-4096-0")
    fields = {binding["buffer"] for binding in prefill["inputs"]}
    assert {"vision_token_indices", "vision_feature_slot", "prefill_positions"} <= fields
    assert any(binding["source"] == p.PROMPT_POSITION for binding in prefill["inputs"])
    decode = next(graph for graph in contract["graphs"] if graph["graph"] == "decode-1-1-0")
    assert "prefill_positions" not in {binding["buffer"] for binding in decode["inputs"]}
    spec = next(graph for graph in contract["graphs"] if graph["graph"].startswith("speculative-"))
    assert "prefill_positions" not in {binding["buffer"] for binding in spec["inputs"]}
    flare_heliopause.validate_prepared_contract(manifest, json.dumps(contract))


def test_dense_image_contract_rejects_prefill_without_prompt_positions():
    from types import SimpleNamespace

    from ignition.components.paged_batch import ROTARY_OFFSET
    from ignition.components.state import StateLayout
    from ignition.preparation.execution import GraphSignature
    from ignition.recipes.qwen38_dense import contract

    layout = StateLayout.from_specs((("consumed_tokens", (1,), torch.int64),
                                     (ROTARY_OFFSET, (1,), torch.int64)))
    pool = SimpleNamespace(layout=layout, pages={}, backing=torch.zeros(3, dtype=torch.uint8))
    signatures = {"prefill": GraphSignature("prefill", 1, 4096, ("token_ids", "page_tables"))}
    with pytest.raises(ValueError, match="prompt positions"):
        contract.prepare(signatures, {}, pool, vocabulary=248320, image_slots=4)
    assert contract.prepare(signatures, {}, pool, vocabulary=248320)["frontend"]["position_dimensions"] == 0


def test_dense_vision_state_arena_geometry_and_injection():
    from types import SimpleNamespace

    from ignition.components.paged_batch import RotaryBatchInputs
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.vision_state import PrefillVisionInputs, VisionState
    from ignition.recipes.qwen38_dense.vision_state import FEATURES, IMAGE_TOKEN_ID

    requests = DeviceRequestState.allocate("cpu", vocab_size=248320, capacity=4)
    batch = RotaryBatchInputs.allocate(4, 1, "cpu", capacity=4)
    prefill = PrefillVisionInputs.allocate(4, "cpu")
    state = VisionState(batch, prefill, requests, image_token=IMAGE_TOKEN_ID,
        features=FEATURES, hidden=WIDTH, image_slots=4)
    assert state.arena.shape == (4, FEATURES, WIDTH)
    assert state.arena.numel() * state.arena.element_size() == 4 * 167_772_160
    requests.generation[1], requests.status[1] = 9, 1
    state.feature_slot.fill_(2)
    state.generation[1], state.count[1] = 9, 2
    state.arena[2, 0].fill_(3.0)
    state.arena[2, 1].fill_(5.0)
    batch.row_slot_ids[0], batch.row_generations[0] = 1, 9
    tokens = torch.tensor([11, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12], dtype=torch.int64)
    prefill.vision_token_indices.copy_(torch.tensor([-1, 1, 0, -1], dtype=torch.int32))
    lengths = torch.tensor([4], dtype=torch.int32)
    assert state.validate_prefill(tokens, lengths).item()
    hidden = torch.full((4, WIDTH), 7.0, dtype=torch.bfloat16)
    state.inject(SimpleNamespace(token_ids=tokens), hidden)
    assert torch.all(hidden[[0, 3]] == 7.0)
    assert torch.all(hidden[1] == 5.0) and torch.all(hidden[2] == 3.0)
    # Out-of-range index or mismatched generation is rejected before injection.
    prefill.vision_token_indices[1] = 2
    assert not state.validate_prefill(tokens, lengths).item()
    prefill.vision_token_indices[1] = 1
    state.generation[1] = 8
    assert not state.validate_prefill(tokens, lengths).item()


def test_dense_vision_generation_validates_sanitizes_and_reports_errors():
    """PreparedVisionGeneration sanitizes invalid uploads and flags non-finite encoder outputs."""
    from ignition.components.completion import CompletionOutputs
    from ignition.components.paged_batch import RotaryBatchInputs
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.vision_state import PrefillVisionInputs, VisionState
    from ignition.recipes.qwen38_dense.vision_state import (
        IMAGE_TOKEN_ID, PreparedVisionGeneration, VisionInputs,
    )

    capacity, hidden = 16, 8
    requests = DeviceRequestState.allocate("cpu", vocab_size=64, capacity=2)
    batch = RotaryBatchInputs.allocate(4, 1, "cpu", capacity=2)
    prefill = PrefillVisionInputs.allocate(4, "cpu")
    state = VisionState(batch, prefill, requests, image_token=IMAGE_TOKEN_ID,
        features=16, hidden=hidden, image_slots=2)
    inputs = VisionInputs.allocate("cpu")
    completion = CompletionOutputs.allocate((2, 2, 32), device="cpu")
    frame = torch.zeros(1, dtype=torch.int32)
    sequence = torch.ones(1, dtype=torch.int64)
    seen = {}

    class StubEncoder:
        def __init__(self):
            self.capacity = capacity
            self.emit_nan = False

        def __call__(self, pixels, metadata):
            seen["pixels"] = pixels.clone()
            seen["metadata"] = metadata
            out = pixels.view(capacity // 4, 4, -1).mean(dim=1)[:, :hidden].to(torch.bfloat16)
            if self.emit_nan:
                out = out.clone()
                out[0, 0] = float("nan")
            return out

    encoder = StubEncoder()
    operation = PreparedVisionGeneration(encoder, inputs, state, completion, frame, sequence)
    published = []
    operation.publish = lambda output, count, errors: published.append(
        (output.clone(), count.item(), errors.item()))

    def load_valid(*, count=8, offset=0):
        requests.generation[1], requests.status[1] = 5, 1
        state.generation[1], state.count[1] = 5, offset
        state.feature_slot.fill_(0)
        inputs.vision_slot.fill_(1)
        inputs.vision_generation.fill_(5)
        inputs.vision_num_patches.fill_(count)
        inputs.vision_feature_offset.fill_(offset)
        inputs.vision_cu_seqlens.copy_(torch.tensor([0, count], dtype=torch.int32))
        inputs.vision_pixel_values.fill_(float("nan"))
        inputs.vision_pixel_values[:count].fill_(2.0)
        inputs.vision_interpolation_indices.fill_(9999)
        inputs.vision_interpolation_indices[:count].fill_(7)
        inputs.vision_interpolation_weights.fill_(float("nan"))
        inputs.vision_interpolation_weights[:count].fill_(0.25)
        inputs.vision_position_ids.fill_(-1)
        inputs.vision_position_ids[:count].fill_(3)

    # Valid input ignores NaN/out-of-range garbage past live patches and publishes live features.
    load_valid(count=8, offset=0)
    _, errors = operation.forward()
    assert errors.item() == 0
    assert published[-1][1:] == (2, 0)
    assert torch.isfinite(seen["pixels"]).all()
    assert (seen["pixels"][8:] == 0).all()
    assert seen["metadata"].num_patches.item() == 8

    # Invalid uploads zero the encoder inputs and report non-zero error without publishing live features.
    def check_invalid(mutate):
        load_valid(count=8, offset=0)
        mutate()
        _, err = operation.forward()
        assert err.item() == 1
        assert published[-1][1:] == (0, 1)
        assert (seen["pixels"] == 0).all()
        assert seen["metadata"].num_patches.item() == 0

    check_invalid(lambda: inputs.vision_slot.fill_(2))
    check_invalid(lambda: inputs.vision_generation.fill_(4))
    check_invalid(lambda: inputs.vision_num_patches.fill_(6))
    check_invalid(lambda: inputs.vision_feature_offset.fill_(2))
    check_invalid(lambda: inputs.vision_cu_seqlens.copy_(torch.tensor([0, 4], dtype=torch.int32)))
    check_invalid(lambda: state.feature_slot.fill_(2))
    check_invalid(lambda: inputs.vision_pixel_values[0].fill_(float("nan")))
    check_invalid(lambda: inputs.vision_interpolation_indices[0].fill_(2304))
    check_invalid(lambda: inputs.vision_interpolation_weights[0].fill_(1.5))
    check_invalid(lambda: inputs.vision_position_ids[0].fill_(-1))

    # Non-finite encoder output sets error bit 64.
    load_valid(count=8, offset=0)
    encoder.emit_nan = True
    _, err = operation.forward()
    assert err.item() == 64
    assert published[-1][1:] == (2, 64)


def test_dense_frontend_declares_the_qwen_image_profile():
    from ignition.preparation.frontend import prepare_frontend

    frontend = prepare_frontend(_snapshot(), eos_token_ids=[248046, 248044], image_profile="qwen",
        qwen_model_type="qwen3_5")
    assert frontend["image"]["qwen"] == dict(min_pixels=65536, max_pixels=16777216,
        position_side=48, image_token_id=248056, video_token_id=248057)


def _dense_weights(path, device):
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.components.weights import DirectWeightLoader
    from ignition.recipes.qwen38.vision import VisionWeights
    from transformers import Qwen3_5VisionConfig, Qwen3_5VisionModel

    manifest = inspect_indexed_checkpoint(path)
    config = Qwen3_5VisionConfig.from_dict(
        json.loads((path / "config.json").read_text())["vision_config"])
    with torch.device("meta"):
        module = Qwen3_5VisionModel(config)
    module.eval().requires_grad_(False)
    with DirectWeightLoader(manifest, device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(module, loader)
    return module, weights, config


@CUDA
@torch.no_grad()
def test_dense_vision_generation_publishes_and_injects_wide_features():
    """Capture the 27B encoder with PreparedVisionGeneration, publish two images, and inject them."""
    from types import SimpleNamespace

    from b12x.preparation import PreparationSession
    from ignition.components.completion import CompletionOutputs
    from ignition.components.paged_batch import RotaryBatchInputs
    from ignition.components.sampling import DeviceRequestState, RequestAdmission
    from ignition.components.vision_state import PrefillVisionInputs, VisionState
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38.vision import PreparedVision, VisionMetadata, VisionScratch
    from ignition.recipes.qwen38_dense.vision_state import (
        IMAGE_TOKEN_ID, PreparedVisionGeneration, VisionInputs,
    )
    from transformers.vision_utils import (
        get_vision_interpolation_indices_and_weights, get_vision_position_ids,
    )

    device = _device()
    path = _snapshot()
    module, weights, config = _dense_weights(path, device)
    assert config.out_hidden_size == WIDTH
    arena = ScratchArena(device)
    scratch = VisionScratch.allocate(256, device, WIDTH)
    inputs = VisionInputs.allocate(device)
    requests = DeviceRequestState.allocate(device, vocab_size=248320, capacity=2)
    requests.admit(1, RequestAdmission(7, 3), [1])
    batch = RotaryBatchInputs.allocate(4, 1, device, capacity=2)
    prefill = PrefillVisionInputs.allocate(4, device)
    state = VisionState(batch, prefill, requests, image_token=IMAGE_TOKEN_ID,
        features=256, hidden=WIDTH, image_slots=1)
    state.feature_slot.zero_()
    completion = CompletionOutputs.allocate((2, 2, 32), device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.ones(1, dtype=torch.int64, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        encoder = PreparedVision(module, scratch, VisionMetadata.allocate(256, 1, device),
            weights, session=session, projection_workspace=arena, attention_workspace=arena)
        encoder.functional(session=session)
        session.freeze()
        operation = PreparedVisionGeneration(encoder, inputs, state, completion, frame, sequence)
        published = torch.empty((64, WIDTH), dtype=torch.bfloat16, device=device)
        errors = torch.empty(1, dtype=torch.int32, device=device)

        def forward():
            output, status = operation.forward()
            published.copy_(output)
            errors.copy_(status)

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        worker = PreparedGraphBundle(device=0, graphs={"vision": graph}, buffers={},
            resources=(operation, compiled, weights, session, arena)).start_worker()

        def replay():
            torch.cuda.synchronize(device)
            allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            ticket = worker.submit(0)
            result = worker.poll(timeout_ms=30000)
            assert result is not None and result.ticket == ticket and result.error is None
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations

        try:
            grid = torch.tensor([[1, 16, 16]], device=device)
            indices, factors = get_vision_interpolation_indices_and_weights(
                grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2)
            positions = get_vision_position_ids(grid, 2)
            inputs.vision_slot.fill_(1)
            inputs.vision_generation.fill_(7)
            offset = 0
            for seed in (13, 29):
                inputs.vision_pixel_values.fill_(float("nan"))
                inputs.vision_pixel_values[:256].normal_(generator=torch.Generator(device=device).manual_seed(seed))
                inputs.vision_interpolation_indices[:256].copy_(indices)
                inputs.vision_interpolation_weights[:256].copy_(factors)
                inputs.vision_position_ids[:256].copy_(positions)
                inputs.vision_cu_seqlens.copy_(torch.tensor([0, 256], dtype=torch.int32, device=device))
                inputs.vision_num_patches.fill_(256)
                inputs.vision_feature_offset.fill_(offset)
                requests.status[1] = 1
                requests.error_code[1] = 0
                state.arena[:, offset:].fill_(float("nan"))
                replay()
                assert errors.item() == 0
                assert state.count[1].item() == offset + 64
                assert state.generation[1].item() == 7
                assert torch.equal(state.arena[0, offset:offset + 64], published)
                assert torch.isfinite(published.float()).all()
                assert completion.records[0, 1, 5].item() == 1
                assert completion.records[0, 1, 6].item() == 0
                offset += 64

            # Inject features from both published images into a 5120-wide prefill embedding.
            batch.row_slot_ids[0], batch.row_generations[0] = 1, 7
            batch.token_ids.copy_(torch.tensor([11, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12], device=device))
            prefill.vision_token_indices.copy_(torch.tensor([-1, 0, offset - 1, -1], device=device))
            assert state.validate_prefill(batch.token_ids, torch.tensor([4], device=device)).item()
            embedding = torch.full((4, WIDTH), 7.0, dtype=torch.bfloat16, device=device)
            state.inject(SimpleNamespace(token_ids=batch.token_ids), embedding)
            assert torch.all(embedding[[0, 3]] == 7.0)
            assert torch.equal(embedding[1], state.arena[0, 0])
            assert torch.equal(embedding[2], state.arena[0, offset - 1])

            # Stale generation leaves the arena and running request untouched.
            before = state.arena.clone()
            requests.generation[1] = 8
            replay()
            assert errors.item() != 0
            assert requests.status[1].item() == 1
            assert state.count[1].item() == offset
            assert torch.equal(state.arena[:1, :offset], before[:1, :offset])
        finally:
            worker.close()
            graph.reset()
