# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 27B image feature transfer at the 5120-wide language-model width."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from tests.components.completion_support import completion_outputs

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
    frontend = contract.frontend(video=True)
    assert frontend["image_tokens"] == dict(feature_token=248056,
        video_feature_token=248057, span=None)
    assert frontend["image_preparation"] == p.QWEN_IMAGE_PREPARATION
    assert frontend["image_layout"] == p.QWEN_IMAGE_LAYOUT
    assert frontend["position_dimensions"] == 3
    assert contract.frontend()["image_tokens"] == dict(feature_token=248056, span=None)
    assert contract.frontend(vision=False)["image_tokens"] is None
    assert contract.frontend(vision=False)["position_dimensions"] == 0


def test_dense_recipe_declares_the_flash_image_limits():
    from ignition.recipes.qwen38.recipe import RECIPE as FLASH
    from ignition.recipes.qwen38_dense.recipe import (
        RECIPE, VISION_BYTES, VISION_HANDOFF_BYTES, VISION_FEATURE_BYTES, VISION_GRAPH_BYTES,
        VISION_INPUT_BYTES, VISION_SCRATCH_BYTES, VISION_SLOT_BYTES, VISION_TOWER_BYTES,
    )

    assert (RECIPE.max_images, RECIPE.image_bytes) == (FLASH.max_images, FLASH.image_bytes) == (4, 32 << 20)
    assert RECIPE.language_model_only
    assert VISION_SLOT_BYTES == 167_772_160
    assert VISION_FEATURE_BYTES == 4 * VISION_SLOT_BYTES == 671_088_640
    assert VISION_SCRATCH_BYTES == 811_212_800
    assert VISION_INPUT_BYTES == 101_711_900
    assert VISION_HANDOFF_BYTES == 1_392_704
    assert VISION_TOWER_BYTES == 921_460_192
    assert VISION_GRAPH_BYTES == 490_733_568
    assert VISION_BYTES == 2_997_599_804
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
    image = planning.fit(RECIPE, model.checkpoint, gpus, **options, image_concurrency=4)
    wide = planning.fit(RECIPE, model.checkpoint, gpus, **options, image_concurrency=8)
    zero = planning.fit(RECIPE, model.checkpoint, gpus, **options, image_concurrency=0)
    text_plans = (default, language_only, zero)
    assert {plan.graph_bytes for plan in text_plans} == {4 << 30}
    assert all(plan.image_concurrency == 0 and plan.max_images == 0 for plan in text_plans)
    assert image.graph_bytes - default.graph_bytes == VISION_BYTES
    assert wide.graph_bytes - image.graph_bytes == 4 * VISION_SLOT_BYTES
    # A plan that prepares no vision uses the text-only continuation row.
    plain = planning.fit(RECIPE, model.checkpoint, gpus, concurrency=8, reserve_bytes=0,
        speculation=None)
    assert default.costs == RECIPE.costs(1, RECIPE.speculation, model.checkpoint)
    assert default.costs.state_row_bytes == 193_865_984
    assert plain.costs.state_row_bytes == 158_148_864
    assert image.costs.state_row_bytes == 193_866_240
    assert image.costs.graph_bytes == image.graph_bytes
    assert (image.max_images, wide.max_images) == (4, 4)


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
    from ignition.recipes.qwen38.vision_state import VisionInputs
    from ignition.recipes.qwen38_dense import contract
    from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig

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
        fields = ((*vision_inputs, "vision_feature_slot") if kind == "vision"
                  else tuple(name for name in c.text_inputs(kind) if name in buffers))
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
        image_slots=config.image_concurrency, video=True)
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
    assert frontend["image_tokens"] == dict(feature_token=248056,
        video_feature_token=248057, span=None)
    assert frontend["image_preparation"] == p.QWEN_IMAGE_PREPARATION
    assert frontend["image_layout"] == p.QWEN_IMAGE_LAYOUT
    assert frontend["position_dimensions"] == 3
    vision = [graph for graph in manifest["graphs"] if graph["kind"] == "vision"]
    assert sorted(graph["token_capacity"] for graph in vision) == [256, 1024, 4096, 16384]
    prefill = next(graph for graph in contract["graphs"] if graph["graph"] == "prefill-1-4096-0")
    fields = {binding["buffer"] for binding in prefill["inputs"]}
    assert {"vision_token_indices", "vision_feature_slot", "prefill_positions"} <= fields
    assert any(binding["source"] == p.PROMPT_POSITION for binding in prefill["inputs"])
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


def test_dense_feature_slot_geometry():
    from ignition.recipes.qwen38.vision_state import VisionFeatureState

    features = VisionFeatureState.allocate("cpu", width=WIDTH)
    assert features.arena.shape == (4, 16384, WIDTH)
    assert features.arena.numel() * features.arena.element_size() == 4 * 167_772_160
    with pytest.raises(ValueError, match="positive integer"):
        VisionFeatureState.allocate("cpu", width=0)


def test_dense_frontend_declares_the_qwen_image_profile():
    from ignition.preparation.frontend import prepare_frontend

    from ignition.preparation.frontend import video_settings

    frontend = prepare_frontend(_snapshot(), eos_token_ids=[248046, 248044], image_profile="qwen",
        qwen_model_type="qwen3_5", video=video_settings(_snapshot()))
    assert frontend["image"]["qwen"] == dict(min_pixels=65536, max_pixels=16777216,
        position_side=48, image_token_id=248056, video_token_id=248057,
        video=dict(max_frames=768, min_pixels=4096, max_pixels=25165824, temporal_patch_size=2))


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
def test_dense_vision_generation_publishes_wide_features():
    """Capture the 27B encoder with the shared handoff and publish one image."""
    from b12x.preparation import PreparationSession
    from ignition.components.sampling import DeviceRequestState, RequestAdmission
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38.vision import PreparedVision, VisionScratch
    from ignition.recipes.qwen38.vision_state import (
        PreparedVisionHandoff, VisionFeatureState, VisionInputs,
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
    feature_slot = torch.full((1,), 0, dtype=torch.int32, device=device)
    features = VisionFeatureState.allocate(device, capacity=1, image_slots=1,
        feature_slot=feature_slot, width=WIDTH)
    requests = DeviceRequestState.allocate(device, vocab_size=248320, capacity=1)
    requests.admit(0, RequestAdmission(7, 3), [1])
    records = torch.zeros((2, 1, 32), dtype=torch.int64, device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.ones(1, dtype=torch.int64, device=device)
    handoff = PreparedVisionHandoff(inputs, features, requests, scratch.output,
        completion_outputs(records), frame, sequence, pixels=scratch.patches)
    handoff.prepare()
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        encoder = PreparedVision(module, scratch, handoff.metadata, weights, session=session,
            projection_workspace=arena, attention_workspace=arena)
        handoff.functional()
        encoder.functional(session=session)
        session.freeze()
        grid = torch.tensor([[1, 16, 16]], device=device)
        indices, weights_ = get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2)
        inputs.vision_interpolation_indices[:256].copy_(indices)
        inputs.vision_interpolation_weights[:256].copy_(weights_)
        inputs.vision_position_ids[:256].copy_(get_vision_position_ids(grid, 2))
        inputs.vision_cu_seqlens[1] = 256
        inputs.vision_num_patches.fill_(256)
        inputs.vision_slot.zero_()
        inputs.vision_generation.fill_(7)
        inputs.vision_feature_offset.zero_()
        inputs.vision_pixel_values[:256].normal_()
        published = torch.empty((64, WIDTH), dtype=torch.bfloat16, device=device)
        errors = torch.empty(1, dtype=torch.int32, device=device)

        class Observed:
            def __call__(self):
                values, metadata = handoff.begin()
                output = encoder(values["pixels"], metadata)
                published.copy_(output)
                errors.copy_(values["errors"])
                handoff.finish(values, output)

        forward = Observed()

        def reset():
            features.generation.zero_()
            features.count.zero_()
            requests.status.fill_(1)
            requests.error_code.zero_()

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            forward()
        stream.synchronize()
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
        reset()
        graph.replay()
        torch.cuda.synchronize(device)
        assert published.shape == (64, WIDTH)
        assert errors.item() == 0
        assert features.count[0].item() == 64
        assert features.generation[0].item() == 7
        assert torch.equal(features.arena[0, :64], published)
        assert torch.isfinite(published.float()).all()
        assert records[0, 0, 5].item() == 1
        assert records[0, 0, 6].item() == 0
        graph.reset()


@CUDA
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_dense_prefill_injection_reads_wide_features(compiled):
    """Validate and inject 5120-wide features through the shared prefill kernels."""
    from ignition.components.sampling import DeviceRequestState, RequestAdmission
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.recipes.qwen38.vision_state import PreparedEmbeddingInjection, VisionFeatureState
    from ignition.recipes.qwen38_dense.decoder import PrefillInjection

    device = _device()
    tokens = 256
    features = VisionFeatureState.allocate(device, capacity=1, image_slots=1, width=WIDTH)
    features.feature_slot.fill_(0)
    features.generation[0] = 9
    features.count[0] = 2
    features.arena[0, 0].fill_(1)
    features.arena[0, 1].fill_(2)
    requests = DeviceRequestState.allocate(device, vocab_size=248320, capacity=1)
    requests.admit(0, RequestAdmission(9, 5), [1])
    indices = torch.full((tokens,), -1, dtype=torch.int32, device=device)
    slots = torch.zeros(1, dtype=torch.int32, device=device)
    epochs = torch.full((1,), 9, dtype=torch.int64, device=device)
    lengths = torch.full((1,), tokens, dtype=torch.int32, device=device)
    live = torch.ones(1, dtype=torch.int32, device=device)
    embedding = torch.full((tokens, WIDTH), 7, dtype=torch.bfloat16, device=device)
    operation = PreparedEmbeddingInjection(
        features, requests, indices, slots, epochs, lengths, live, embedding)
    operation.prepare()
    operation.functional()
    injection = PrefillInjection(operation)
    valid = torch.zeros(1, dtype=torch.bool, device=device)
    checked = torch.empty(1, dtype=torch.int32, device=device)

    def forward():
        valid.copy_(injection.validate_prefill(indices, lengths))
        checked.copy_(injection.errors)
        injection.inject(embedding)

    if compiled:
        from ignition.preparation.compilation import compile_forward

        forward = compile_forward(forward)
        forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    worker = PreparedGraphBundle(device=0, graphs={"inject": graph},
        buffers={"embedding": embedding}, resources=(operation, injection, forward)).start_worker()

    def replay():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        ticket = worker.submit(0)
        result = worker.poll(timeout_ms=30000)
        assert result.ticket == ticket and result.error is None
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated

    try:
        indices[:4].copy_(torch.tensor([-1, 1, 0, -1], dtype=torch.int32, device=device))
        replay()
        assert valid.item()
        assert checked.item() == 0
        assert torch.equal(embedding[1], features.arena[0, 1])
        assert torch.equal(embedding[2], features.arena[0, 0])
        assert torch.all(embedding[[0, 3]] == 7)
        assert torch.all(embedding[4:] == 7)
        for bad in (2, 16384):
            requests.status[0] = 1
            embedding.fill_(7)
            indices[-1] = bad
            replay()
            assert not valid.item()
            assert checked.item() == 2
            assert torch.all(embedding == 7)
        requests.status[0] = 1
        indices[-1] = 0
        features.generation[0] = 8
        replay()
        assert not valid.item()
        assert checked.item() == 2
        assert torch.all(embedding == 7)
        requests.status[0] = 1
        indices.fill_(-1)
        features.generation[0] = 9
        replay()
        assert valid.item()
        assert torch.all(embedding == 7)
    finally:
        worker.close()
        graph.reset()


def test_embedding_injection_passes_its_validation_result_explicitly(monkeypatch):
    """The injection keeps no error tensor between preparation and replay."""
    from types import SimpleNamespace

    from ignition.recipes.qwen38.vision_state import PreparedEmbeddingInjection

    operation = object.__new__(PreparedEmbeddingInjection)
    operation.features = SimpleNamespace(arena=SimpleNamespace(device=torch.device("cpu")))
    operation._prepared, operation.dynamic_inputs = True, False
    operation._validate_args = (None,) * 16
    operation._inject_args = (None,) * 18
    # `prepare()` binds compiled kernels; only their launch arguments matter here.
    operation._validate = SimpleNamespace(kernel="validate", compiled=None, grid=(1, 1, 1), options={})
    operation._inject = SimpleNamespace(kernel="inject", compiled=None, grid=(1, 1, 1), options={})
    operation.errors = torch.zeros(1, dtype=torch.int32)
    operation.embedding = torch.zeros((4, 8), dtype=torch.bfloat16)
    operation.functional()
    assert operation.errors is None and operation.embedding is None
    launched = []
    monkeypatch.setattr("ignition.preparation.triton.launch_triton",
        lambda kernel, compiled, args, grid, **options: launched.append(args))
    first = operation.validate()
    second = operation.validate()
    assert first is not second and operation.errors is None
    assert launched[-1][11] is second
    operation.inject(operation.embedding, second)
    assert launched[-1][12] is second
