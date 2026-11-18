# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact-checkpoint vision encoder and native CUDA graph qualification."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from tests.components.completion_support import completion_outputs

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.qwen38.vision import PreparedVision, VisionMetadata, VisionScratch, VisionWeights
from ignition.recipes.qwen38.weights import QwenWeightLoader
from ignition.reference.qwen38_quant import (
    CheckpointTensors, decode_mxfp8, decode_nvfp4, quantize_mxfp8_activation,
    read_nvfp4_projection,
)

CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                         reason="requires explicit GPU assignment and FLARE_RUN_CUDA_TESTS=1")


class ObservedVisionGeneration:
    def __init__(self, handoff, encoder, errors):
        self.handoff, self.encoder, self.errors = handoff, encoder, errors

    def __call__(self):
        values, metadata = self.handoff.begin()
        output = self.encoder(values["pixels"], metadata)
        self.handoff.finish(values, output)
        self.errors.copy_(values["errors"])


def test_vision_buffer_geometry():
    metadata = VisionMetadata.allocate(256, 2, torch.device("cpu"))
    scratch = VisionScratch.allocate(256, torch.device("cpu"))
    assert scratch.patches.shape == (256, 1536)
    assert scratch.query.shape == (256, 16, 72)
    assert scratch.output.shape == (64, 2560)
    assert VisionScratch.allocate(256, torch.device("cpu"), 5120).output.shape == (64, 5120)
    assert metadata.position_ids.shape == (256, 2)
    assert metadata.cu_seqlens.shape == (3,)
    assert metadata.num_patches.item() == 0
    with pytest.raises(ValueError):
        VisionScratch.allocate(255, torch.device("cpu"))


def test_vision_rotation_selects_by_model_class():
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    from transformers.models.qwen4_exp import modeling_qwen4_exp
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

    from ignition.recipes.qwen38.vision import _vision_rotation

    geometry = dict(depth=1, hidden_size=32, num_heads=2, intermediate_size=64,
                    out_hidden_size=32, in_channels=3, patch_size=16)
    flash_config = Qwen4ExpVisionConfig(**geometry, model_type="qwen4_exp")
    with torch.device("meta"):
        flash = Qwen4ExpVisionModel(flash_config)
        dense = Qwen3_5VisionModel(Qwen3_5VisionConfig(**geometry))
    assert flash_config.model_type == "qwen4_exp"
    assert _vision_rotation(flash) == (
        modeling_qwen4_exp.Qwen4ExpVisionRotaryEmbedding,
        modeling_qwen4_exp.apply_rotary_pos_emb_vision)
    assert _vision_rotation(dense) == (
        modeling_qwen3_5.Qwen3_5VisionRotaryEmbedding,
        modeling_qwen3_5.apply_rotary_pos_emb_vision)
    with pytest.raises(ValueError, match=r"unsupported vision model Identity"):
        _vision_rotation(torch.nn.Identity())


def test_vision_rejects_mismatched_output_width():
    from types import SimpleNamespace
    config = SimpleNamespace(hidden_size=1152, num_heads=16, intermediate_size=4304, depth=27,
                             out_hidden_size=2560)
    scratch = VisionScratch.allocate(256, torch.device("cpu"), 5120)
    with pytest.raises(ValueError, match=r"width 5120 differs from the module out_hidden_size 2560"):
        PreparedVision(SimpleNamespace(config=config), scratch, None, None, session=None)


@pytest.fixture(scope="module")
def manifest():
    path = Path("/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    path = Path(os.environ.get("FLARE_QWEN38_SNAPSHOT", path))
    if not path.exists():
        pytest.skip("exact checkpoint is unavailable")
    return inspect_qwen38_checkpoint(path)


def _module(manifest):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel
    config = Qwen4ExpVisionConfig.from_dict(json.loads((manifest.snapshot / "config.json").read_text())["vision_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        module = Qwen4ExpVisionModel(config)
    return module.eval().requires_grad_(False)


class _ReferenceProjection(torch.nn.Module):
    def __init__(self, weight, bias, *, quantized, scale=1.):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.quantized, self.scale = quantized, scale

    def forward(self, source):
        source = quantize_mxfp8_activation(source) if self.quantized else source.float()
        return ((source @ self.weight.T) * self.scale).bfloat16() + self.bias


def _reference(manifest, weights, device):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionRotaryEmbedding
    module = _module(manifest)
    for path, value in weights.parameters.items():
        parent, _, attribute = path.rpartition(".")
        setattr(module.get_submodule(parent), attribute, torch.nn.Parameter(value, requires_grad=False))
    with CheckpointTensors(manifest, byte_budget=512 << 20) as reader:
        for path in weights.projections:
            prefix = f"model.visual.{path}"
            if ".attn." in path:
                decoded = decode_mxfp8(reader.read(prefix + ".weight"), reader.read(prefix + ".weight_scale"))
                scale, quantized = 1., True
            else:
                packed = read_nvfp4_projection(reader, prefix)
                decoded = decode_nvfp4(packed.packed, packed.scales, torch.ones_like(packed.weight_scale_2)).bfloat16().float()
                scale, quantized = packed.weight_scale_2.item(), False
            parent, _, attribute = path.rpartition(".")
            setattr(module.get_submodule(parent), attribute,
                    _ReferenceProjection(decoded.to(device), weights.parameters[path + ".bias"],
                                         quantized=quantized, scale=scale))
    module.rotary_pos_emb = Qwen4ExpVisionRotaryEmbedding(module.config).to(device)
    return module


def _input(manifest, variant):
    from PIL import Image
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(manifest.snapshot, local_files_only=True)
    width, height = 256, 256
    pixels = bytes(component for y in range(height) for x in range(width)
                   for component in ((x * 7 + variant * 31) % 256, (y * 11) % 256, (x + y * 3) % 256))
    image = Image.frombytes("RGB", (width, height), pixels)
    return processor.image_processor(images=[image], return_tensors="pt")


@pytest.fixture(scope="module")
def dense_manifest():
    value = os.environ.get("FLARE_QWEN38_DENSE_SNAPSHOT")
    if not value:
        pytest.skip("requires FLARE_QWEN38_DENSE_SNAPSHOT naming a Qwen3.8-27B checkpoint")
    path = Path(value)
    if not path.is_dir():
        pytest.skip("FLARE_QWEN38_DENSE_SNAPSHOT does not name a checkpoint directory")
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    return inspect_indexed_checkpoint(path)


def _dense_module(manifest):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    config = Qwen3_5VisionConfig.from_dict(json.loads((manifest.snapshot / "config.json").read_text())["vision_config"])
    # The checkpoint states no attention implementation, so Transformers selects its
    # default SDPA kernel. The prepared encoder runs a varlen flash kernel; forcing an
    # eager reference would measure kernel selection instead of this graph.
    with torch.device("meta"):
        module = Qwen3_5VisionModel(config)
    return module.eval().requires_grad_(False)


def _dense_reference(manifest, weights, device):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding
    module = _dense_module(manifest)
    for path, value in weights.parameters.items():
        parent, _, attribute = path.rpartition(".")
        setattr(module.get_submodule(parent), attribute, torch.nn.Parameter(value, requires_grad=False))
    for path, weight in weights.projections.items():
        assert isinstance(weight, torch.Tensor) and weight.dtype == torch.bfloat16, path
        module.get_submodule(path).weight = torch.nn.Parameter(weight, requires_grad=False)
    module.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(module.config).to(device)
    return module


@CUDA
@torch.no_grad()
def test_actual_vision_encoder_native_replay(manifest, record_property):
    from b12x.preparation import PreparationSession
    from transformers.vision_utils import (
        get_vision_position_ids, get_vision_interpolation_indices_and_weights,
    )
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    assert torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    module = _module(manifest)
    with QwenWeightLoader(manifest, device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(module, loader)
    reference = _reference(manifest, weights, device)
    scratch = VisionScratch.allocate(256, device)
    metadata = VisionMetadata.allocate(256, 2, device)
    pixels = torch.zeros((256, 1536), dtype=torch.float32, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        prepared = PreparedVision(module, scratch, metadata, weights, session=session)
        prepared.functional(session=session)
        session.freeze()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        def forward():
            scratch.output.copy_(prepared(pixels, metadata))
        with torch.cuda.stream(stream):
            forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
        worker = PreparedGraphBundle(device=0, graphs={"vision": graph},
            buffers={"pixels": pixels, "embeddings": scratch.output},
            resources=(prepared, weights, session, stream)).start_worker()
        errors = []
        try:
            for variant in (0, 1):
                inputs = _input(manifest, variant)
                assert inputs.pixel_values.shape == pixels.shape
                grid = inputs.image_grid_thw
                indices, weights_ref = get_vision_interpolation_indices_and_weights(
                    grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2)
                metadata.interpolation_indices.copy_(indices)
                metadata.interpolation_weights.copy_(weights_ref)
                metadata.position_ids.copy_(get_vision_position_ids(grid, 2))
                metadata.cu_seqlens.copy_(torch.tensor([0, 256, 256], dtype=torch.int32))
                metadata.num_patches.fill_(256)
                pixels.copy_(inputs.pixel_values)
                expected = reference(inputs.pixel_values.to(device), grid.to(device)).pooler_output.float()
                torch.cuda.synchronize(device)
                scratch.output.fill_(float("nan"))
                torch.cuda.synchronize(device)
                allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                ticket = worker.submit(0)
                completion = worker.poll(timeout_ms=30000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                actual = scratch.output.float()
                relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
                errors.append({"variant": variant, "relative_l2": relative,
                               "max_absolute": float((actual - expected).abs().max())})
                assert relative < .03, errors
                torch.testing.assert_close(actual, expected, rtol=.05, atol=float(expected.abs().max()) * .03)
        finally:
            worker.close()
        record_property("numerical_errors", json.dumps(errors))


# Limits of the prepared 27B vision encoder against the Transformers BF16 forward of the
# same weights. Both evaluate in BF16; the prepared path packs patches with a GEMM instead
# of a convolution and runs varlen flash attention where Transformers selects SDPA, so the
# two outputs differ by operator rounding. The prepared output repeats bit for bit.
#
# Measured for the synthetic image of this test at 256, 1,024, 4,096 and 16,384 patches:
#   relative L2 distance of the whole output to the BF16 forward
#     RTX 5090 (SM120): 0.0343, 0.0329, 0.0228, 0.0260
#     DGX Spark GB10 (SM121): 0.0431, 0.0189, 0.0227, 0.0261
#   fraction of output tokens whose cosine to the reference token is below 0.99
#     DGX Spark GB10: 0, 0, 0, 0.00073 (3 of 4,096 tokens; the lowest cosine is 0.976)
#     RTX 5090: not measured
# The distance limit is the largest measured value, 0.0431, with a margin of one quarter.
# The token limit admits 0.2% of the tokens below 0.99, 2.7 times the largest measured
# fraction; no measured token is below 0.9.
_DENSE_RELATIVE_L2 = 0.054
_DENSE_TOKEN_COSINE = 0.99
_DENSE_TOKEN_FRACTION = 0.002
# Faults in the prepared encoder's inputs, measured on DGX Spark GB10 at the same four
# capacities as (relative L2, fraction of tokens below 0.99):
#   patch rows rotated by one merge block: (0.620, 0.953), (0.755, 0.992), (0.748, 0.990), (0.783, 0.985)
#   row coordinate of the second half of the patches shifted by one:
#     (0.263, 0.266), (0.246, 0.211), (0.199, 0.133), (0.141, 0.070)
#   one merge block exchanged with its neighbour: (0.152, 0.094), (0.072, 0.012), (0.031, 0.0010), (0.030, 0.0010)
# The first two break both limits at every capacity. The exchange changes two output
# tokens; it breaks the limits up to 1,024 patches and stays within them at 4,096 and
# 16,384 patches, where two of 1,024 or 4,096 tokens move the whole output by less than
# operator rounding does. The test requires its detection only up to 1,024 patches.
_DENSE_EXCHANGE_DETECTED_PATCHES = 1024


@CUDA
@pytest.mark.parametrize("capacity", [256, 1024, 4096, 16384])
@torch.no_grad()
def test_dense_vision_encoder_native_replay(dense_manifest, capacity, record_property):
    """Replay the 27B BF16 vision tower and compare it with the Transformers BF16 forward.

    The prepared encoder is deterministic, so a repeat is bit-identical. Its output is
    held to two limits against the BF16 forward: the relative L2 distance of the whole
    output, and the fraction of output tokens whose cosine to the reference token is
    below a floor. Two faults in the prepared encoder's inputs must break a limit at every
    capacity: the patch rows rotated by one merge block, and the row coordinate of the
    second half of the patches shifted by one. A third, one merge block exchanged with
    its neighbour, must break one up to 1,024 patches; the limits cannot detect it beyond.
    """
    from math import isqrt

    from b12x.preparation import PreparationSession
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.vision_utils import (
        get_vision_position_ids, get_vision_interpolation_indices_and_weights,
    )
    from ignition.components.weights import DirectWeightLoader

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    assert torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    module = _dense_module(dense_manifest)
    with DirectWeightLoader(dense_manifest, device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(module, loader)
    reference = _dense_reference(dense_manifest, weights, device)
    processor = AutoProcessor.from_pretrained(dense_manifest.snapshot, local_files_only=True)
    side = 16 * isqrt(capacity)
    axis = torch.arange(side, dtype=torch.int64)
    x, y = torch.meshgrid(axis, axis, indexing="xy")
    pixels = torch.stack([(x * 7) % 256, (y * 11) % 256, (x + y * 3) % 256],
                         dim=-1).to(torch.uint8).numpy()
    processed = processor.image_processor(images=[Image.fromarray(pixels)], return_tensors="pt")
    values = processed["pixel_values"].to(device)
    grid = processed["image_grid_thw"].to(device)
    assert int(grid[0][1] * grid[0][2]) == capacity

    scratch = VisionScratch.allocate(capacity, device, module.config.out_hidden_size)
    metadata = VisionMetadata.allocate(capacity, 1, device)
    buffer = torch.zeros((capacity, 1536), dtype=torch.float32, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        prepared = PreparedVision(module, scratch, metadata, weights, session=session)
        prepared.functional(session=session)
        session.freeze()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))

        def forward():
            scratch.output.copy_(prepared(buffer, metadata))

        with torch.cuda.stream(stream):
            forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
        worker = PreparedGraphBundle(device=0, graphs={"vision": graph},
            buffers={"pixels": buffer, "embeddings": scratch.output},
            resources=(prepared, weights, session, stream)).start_worker()
        try:
            indices, interpolation = get_vision_interpolation_indices_and_weights(
                grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2)
            metadata.interpolation_indices.copy_(indices)
            metadata.interpolation_weights.copy_(interpolation)
            metadata.position_ids.copy_(get_vision_position_ids(grid, 2))
            metadata.cu_seqlens.copy_(torch.tensor([0, capacity], dtype=torch.int32))
            metadata.num_patches.fill_(capacity)
            positions = metadata.position_ids.clone()

            def replay(patches=values, position_ids=positions):
                buffer.copy_(patches)
                metadata.position_ids.copy_(position_ids)
                # Input writes run on the default stream and the graph replays on its own.
                torch.cuda.synchronize(device)
                ticket = worker.submit(0)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket
                assert completion.error is None
                return scratch.output.float().clone()

            first = replay()
            second = replay()
            # The reference keeps the original inputs. A rotation by one merge block encodes
            # every patch at its neighbour's position. Shifting the row coordinate of the
            # second half of the patches changes the rotary offsets between the halves;
            # shifting every row would change no offset. Exchanging one merge block with
            # its neighbour misplaces eight of the patches.
            shifted_positions = positions.clone()
            shifted_positions[capacity // 2:, 0] += 1
            exchanged = values.clone()
            exchanged[:4], exchanged[4:8] = values[4:8], values[:4]
            faults = {
                "rotated": replay(patches=values.roll(4, 0)),
                "shifted": replay(position_ids=shifted_positions),
                "exchanged": replay(patches=exchanged),
            }
            restored = replay()
            expected = reference(values, grid).pooler_output.float()
        finally:
            worker.close()

    def measure(actual):
        cosine = (actual * expected).sum(1) / (actual.norm(dim=1) * expected.norm(dim=1)).clamp_min(1e-12)
        return dict(
            relative_l2=float((actual - expected).norm() / expected.norm()),
            max_absolute=float((actual - expected).abs().max()),
            cosine_min=float(cosine.min()),
            below={floor: float((cosine < floor).float().mean()) for floor in (0.999, 0.99, 0.9)})

    def violates(measured):
        return (measured["relative_l2"] > _DENSE_RELATIVE_L2
                or measured["below"][_DENSE_TOKEN_COSINE] > _DENSE_TOKEN_FRACTION)

    record = dict(capacity=capacity, tokens=first.shape[0], prepared=measure(first),
                  **{name: measure(output) for name, output in faults.items()})
    record_property("numerical_errors", json.dumps(record))
    report = "\n".join(f"{name} at {capacity} patches: {json.dumps(value)}"
                       for name, value in record.items() if isinstance(value, dict))
    assert torch.equal(first, second) and torch.equal(restored, first)
    assert not violates(record["prepared"]), report
    required = [name for name in faults
                if name != "exchanged" or capacity <= _DENSE_EXCHANGE_DETECTED_PATCHES]
    undetected = [name for name in required if not violates(record[name])]
    assert not undetected, report


@CUDA
@torch.no_grad()
def test_vision_generation_compiles_in_worker_and_replays_after_poisoning(manifest, tmp_path):
    from b12x.preparation import PreparationSession
    from ignition.preparation.compile_pool import CompilePool
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38.sampling import DeviceRequestState
    from ignition.recipes.qwen38.vision_state import (
        PreparedVisionHandoff, VisionFeatureState, VisionInputs,
    )
    from transformers.vision_utils import get_vision_position_ids, get_vision_interpolation_indices_and_weights

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    module = _module(manifest)
    with QwenWeightLoader(manifest, device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(module, loader)
    arena = ScratchArena(device)
    scratch = VisionScratch.allocate(256, device)
    inputs, features = VisionInputs.allocate(device), VisionFeatureState.allocate(device, capacity=1)
    requests = DeviceRequestState.allocate(device, capacity=1)
    requests.generation.fill_(7)
    requests.status.fill_(1)
    records = torch.zeros((2, 1, 32), dtype=torch.int64, device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.ones(1, dtype=torch.int64, device=device)
    handoff = PreparedVisionHandoff(inputs, features, requests, scratch.output, completion_outputs(records), frame, sequence,
                                   pixels=scratch.patches)
    handoff.prepare()
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        encoder = PreparedVision(module, scratch, handoff.metadata, weights, session=session,
                                 projection_workspace=arena, attention_workspace=arena)
        arena.finalize()
        metadata_example, errors = handoff.metadata, torch.empty_like(handoff.errors)
        handoff.functional()
        encoder.functional(session=session)
        session.freeze()
        forward = ObservedVisionGeneration(handoff, encoder, errors)
        path = tmp_path / "vision.pt2"
        with CompilePool(device, compile_workers=1) as compilers:
            compilers.submit(forward, path).result()
        with path.open("rb") as file:
            compiled = torch.compiler.load_compiled_function(file)
        grid = torch.tensor([[1, 16, 16]], device=device)
        indices, factors = get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2)
        inputs.vision_interpolation_indices[:256].copy_(indices)
        inputs.vision_interpolation_weights[:256].copy_(factors)
        inputs.vision_position_ids[:256].copy_(get_vision_position_ids(grid, 2))
        inputs.vision_cu_seqlens[1] = 256
        inputs.vision_num_patches.fill_(256)
        inputs.vision_slot.zero_()
        inputs.vision_generation.fill_(7)
        features.feature_slot.zero_()
        inputs.vision_pixel_values[:256].normal_()
        def reset():
            features.count.zero_()
            features.generation.zero_()
            requests.status.fill_(1)

        reset()
        forward()
        expected = features.arena[0, :64].clone()
        reset()
        compiled(forward)
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled(forward)
        worker = PreparedGraphBundle(device=0, graphs={"vision": graph}, buffers={},
            resources=(forward, compiled, arena, session)).start_worker()
        try:
            for _ in range(2):
                for tensor in vars(scratch).values():
                    tensor.fill_(float("nan") if tensor.is_floating_point() else -37)
                errors.fill_(-37)
                for tensor in vars(metadata_example).values():
                    tensor.fill_(float("nan") if tensor.is_floating_point() else -37)
                reset()
                features.arena.fill_(float("nan"))
                torch.cuda.synchronize()
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                ticket = worker.submit(0)
                result = worker.poll(timeout_ms=30000)
                assert result is not None and result.ticket == ticket and result.error is None
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                assert errors.item() == 0
                assert features.count.item() == 64
                torch.testing.assert_close(features.arena[0, :64], expected, rtol=0, atol=0)
        finally:
            worker.close()


@CUDA
@pytest.mark.parametrize("capacity", [256, 16384])
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_vision_native_replay_matches_reference_after_poisoning(manifest, capacity, compiled):
    from b12x.preparation import PreparationSession
    from transformers.vision_utils import get_vision_position_ids, get_vision_interpolation_indices_and_weights

    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    modules = [_module(manifest), _module(manifest)]
    with QwenWeightLoader(manifest, device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(modules[0], loader)
    scratch = [VisionScratch.allocate(capacity, device), VisionScratch.allocate(capacity, device)]
    metadata = [VisionMetadata.allocate(capacity, 1, device),
                VisionMetadata.allocate(capacity, 1, device)]
    pixels = torch.zeros((capacity, 1536), dtype=torch.float32, device=device)
    side = int(capacity ** .5)
    grid = torch.tensor([[1, side, side]], device=device)
    indices, factors = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2)
    positions = get_vision_position_ids(grid, 2)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        prepared = [PreparedVision(module, s, m, weights, session=session, name=f"vision{i}")
                    for i, (module, s, m) in enumerate(zip(modules, scratch, metadata, strict=True))]
        for encoder in prepared:
            encoder.functional(session=session)
        session.freeze()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graphs = {}
        owners = []
        for index, encoder in enumerate(prepared):
            forward = lambda encoder=encoder, index=index: scratch[index].output.copy_(encoder(pixels, metadata[index]))
            if compiled and index == 1:
                import io
                from ignition.preparation.aot import aot_forward

                artifact = aot_forward(forward)
                artifact(forward)
                data = type(artifact).serialize(artifact).serialized_data
                restored = torch.compiler.load_compiled_function(io.BytesIO(data))
                owners.extend((forward, artifact, restored))
                forward = lambda owner=forward, restored=restored: restored(owner)
            with torch.cuda.stream(stream):
                forward()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                forward()
            graphs[str(index)] = graph
        worker = PreparedGraphBundle(device=0, graphs=graphs, buffers={"pixels": pixels},
            resources=(*prepared, *owners, weights, session, stream)).start_worker()
        try:
            for seed in (31, 72):
                pixels.normal_(generator=torch.Generator(device=device).manual_seed(seed))
                for example in scratch:
                    for tensor in vars(example).values():
                        tensor.fill_(float("nan") if tensor.is_floating_point() else -37)
                for m in metadata:
                    m.interpolation_indices.copy_(indices)
                    m.interpolation_weights.copy_(factors)
                    m.position_ids.copy_(positions)
                    m.cu_seqlens.copy_(torch.tensor([0, capacity], dtype=torch.int32, device=device))
                    m.num_patches.fill_(capacity)
                torch.cuda.synchronize(device)
                allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                for index in (0, 1):
                    ticket = worker.submit(index)
                    result = worker.poll(timeout_ms=30000)
                    assert result is not None and result.ticket == ticket and result.error is None
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations
                torch.testing.assert_close(scratch[1].output, scratch[0].output, rtol=0, atol=0)
        finally:
            worker.close()
