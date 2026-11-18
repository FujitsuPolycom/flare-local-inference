# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native Helios Qwen image preprocessing and mRoPE positions against Qwen3.5 27B."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ASSETS = (
    "preprocessor_config.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def _snapshot() -> Path | None:
    env = os.environ.get("FLARE_QWEN38_DENSE_SNAPSHOT")
    if env:
        path = Path(env)
        if path.is_dir():
            return path
    try:
        from ignition.recipes import resolve

        hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
        resolved = resolve("qwen3.8-27b").snapshot(hub)
        if resolved is not None and resolved.is_dir():
            return resolved
    except Exception:
        pass
    return None


def _pattern(width: int, height: int, seed: int) -> np.ndarray:
    state = seed
    data = bytearray(width * height * 3)
    for index in range(len(data)):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        data[index] = state & 255
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def _sha256(array) -> str:
    if isinstance(array, torch.Tensor):
        array = array.cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def test_helios_qwen38_dense_prompt_positions_match_qwen35_reference(tmp_path):
    """Compare Helios-produced tokens, mRoPE positions, rope_delta, and patches against Qwen3_5Model."""
    snapshot = _snapshot()
    if snapshot is None or any(not (snapshot / name).is_file() for name in REQUIRED_ASSETS):
        pytest.skip("Qwen3.8-27B tokenizer/processor snapshot is unavailable")

    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from transformers.vision_utils import (
        get_vision_cu_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    subprocess.run(
        [
            "cargo",
            "build",
            "--quiet",
            "--manifest-path",
            str(ROOT / "helios/Cargo.toml"),
            "--features",
            "images,test-tools",
            "--bin",
            "test_images",
        ],
        check=True,
    )
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "helios/target"))
    tool = target_dir / "debug" / ("test_images.exe" if os.name == "nt" else "test_images")

    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    with torch.device("meta"):
        meta_model = Qwen3_5Model(config)

    case_specs = [
        (
            "text_image_text",
            [{"width": 256, "height": 256, "seed": 11}],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect this split image: "},
                        {"type": "image", "image": "fixture://split"},
                        {"type": "text", "text": " What color is the left half?"},
                    ],
                }
            ],
        ),
        (
            "multi_image_interleaved",
            [
                {"width": 193, "height": 417, "seed": 23},
                {"width": 53, "height": 71, "seed": 37},
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First image: "},
                        {"type": "image", "image": "fixture://first"},
                        {"type": "text", "text": " Second image: "},
                        {"type": "image", "image": "fixture://second"},
                        {"type": "text", "text": " Compare both."},
                    ],
                }
            ],
        ),
        (
            "image_then_text",
            [{"width": 273, "height": 289, "seed": 53}],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "fixture://only"},
                        {"type": "text", "text": "Describe what you see."},
                    ],
                }
            ],
        ),
    ]

    cases = []
    for name, descriptors, messages in case_specs:
        images = [Image.fromarray(_pattern(d["width"], d["height"], d["seed"])) for d in descriptors]
        rendered = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        result = processor(
            text=rendered, images=images, return_tensors="pt", return_mm_token_type_ids=True
        )
        grids = result["image_grid_thw"]
        indices, weights = get_vision_interpolation_indices_and_weights(
            grids, 48, mode="bilinear", align_corners=True, spatial_merge_size=2
        )
        positions = get_vision_position_ids(grids, 2)
        mrope, delta = meta_model.get_rope_index(
            input_ids=result["input_ids"],
            mm_token_type_ids=result["mm_token_type_ids"],
            image_grid_thw=grids,
            attention_mask=result["attention_mask"],
        )
        input_rgb_hashes = []
        resized_rgb_hashes = []
        for pil, grid in zip(images, grids):
            rgb = np.asarray(pil.convert("RGB"))
            resized = processor.image_processor.resize(
                torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0),
                size=processor.image_processor.size,
                factor=32,
                resample=3,
            )
            assert tuple(resized.shape[-2:]) == (int(grid[1]) * 16, int(grid[2]) * 16)
            input_rgb_hashes.append(_sha256(rgb))
            resized_rgb_hashes.append(_sha256(resized[0].permute(1, 2, 0)))
        cases.append(
            {
                "name": name,
                "inputs": descriptors,
                "messages": messages,
                "rendered": rendered,
                "token_ids": result["input_ids"][0].tolist(),
                "mm_token_type_ids": result["mm_token_type_ids"][0].tolist(),
                "grid_thw": grids.tolist(),
                "cu_seqlens": get_vision_cu_seqlens(grids).tolist(),
                "rope_delta": int(delta[0, 0].item()),
                "input_rgb_sha256": input_rgb_hashes,
                "resized_rgb_sha256": resized_rgb_hashes,
                "hashes": {
                    "pixel_values": _sha256(result["pixel_values"]),
                    "position_ids": _sha256(positions),
                    "interpolation_indices": _sha256(indices),
                    "interpolation_weights": _sha256(weights),
                    "mrope": _sha256(mrope[:, 0, :]),
                },
            }
        )

    fixture = {
        "assets": {
            name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
            for name in REQUIRED_ASSETS
        },
        "cases": cases,
        "geometry_cases": [],
    }
    fixture_path = tmp_path / "qwen38_dense_images.json"
    fixture_path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")

    proc = subprocess.run(
        [str(tool), str(snapshot.resolve()), str(fixture_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"test_images failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
