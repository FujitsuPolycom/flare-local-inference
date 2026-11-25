# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Indexed checkpoint total_size bounds and ModelOpt quantization declarations."""

import json
import struct

import pytest

from ignition.preparation.checkpoint import CheckpointError, inspect_indexed_checkpoint
from ignition.preparation.quantization import projection_group_format, quantized_layers


def _write_indexed(directory, total_size):
    payload = bytes(range(16))
    header = json.dumps({
        "weight": {"dtype": "U8", "shape": [4, 4], "data_offsets": [0, len(payload)]},
    }).encode()
    (directory / "weights.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + payload
    )
    index = {"weight_map": {"weight": "weights.safetensors"}}
    if total_size is not ...:
        index["metadata"] = {"total_size": total_size}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    return len(payload)


@pytest.mark.parametrize("delta", [0, 23_372_872])
def test_indexed_checkpoint_accepts_total_size_at_or_above_payload_bytes(tmp_path, delta):
    payload_bytes = _write_indexed(tmp_path, 16 + delta)
    manifest = inspect_indexed_checkpoint(tmp_path)
    assert manifest.tensors["weight"].nbytes == payload_bytes


@pytest.mark.parametrize("declared", [..., None, True, False, "16", 16.0, 15, 0, -1])
def test_indexed_checkpoint_rejects_invalid_or_undersized_total_size(tmp_path, declared):
    _write_indexed(tmp_path, declared)
    with pytest.raises(CheckpointError, match="index total_size is invalid"):
        inspect_indexed_checkpoint(tmp_path)


@pytest.mark.parametrize("method", ["modelopt", "modelopt_mixed"])
def test_quantized_layers_accepts_modelopt_and_modelopt_mixed(method):
    layers = {"proj": {"quant_algo": "FP8"}}
    config = {"quantization_config": {
        "quant_method": method,
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": layers,
    }}
    assert quantized_layers(config) is layers


@pytest.mark.parametrize("override", [
    {"quant_method": "fp8"},
    {"quant_algo": "FP8"},
    {"kv_cache_quant_algo": "FP8"},
])
def test_quantized_layers_rejects_unsupported_config(override):
    quant = {
        "quant_method": "modelopt_mixed",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {},
    } | override
    with pytest.raises(ValueError, match="expected modelopt MIXED_PRECISION"):
        quantized_layers({"quantization_config": quant})


def test_projection_group_format_allows_uniform_tensor_fp8_and_rejects_mixed():
    layers = {
        "q": {"quant_algo": "FP8"},
        "k": {"quant_algo": "FP8"},
        "v": {"quant_algo": "FP8"},
        "nv": {"quant_algo": "NVFP4", "group_size": 16},
    }
    assert projection_group_format(layers, ("q", "k", "v")) == "tensor_fp8"
    with pytest.raises(ValueError, match="same quantization format"):
        projection_group_format(layers, ("q", "nv"))
    with pytest.raises(ValueError, match="nonempty"):
        projection_group_format(layers, ())
