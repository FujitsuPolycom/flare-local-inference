# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Qwen video preparation against the checkpoint's video processor.

Each case renders a chat request with pre-extracted frames, prepares it through the native
request path, and compares every value the prompt and the vision graphs consume with the
reference processor's output: prompt tokens, prompt positions, per-window grids and token
spans, pixel patches, vision positions, and interpolation taps. Comparisons are exact; no
tolerance is applied.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "helios/tests/support/contracts/qwen38.json"
IMAGE_TOKEN = 248056
VIDEO_TOKEN = 248057
PLACEHOLDER = re.compile(r"<\|video_pad\|>|<\|image_pad\|>")


def snapshot() -> Path:
    value = os.environ.get("FLARE_QWEN38_VIDEO_SNAPSHOT")
    if not value:
        pytest.skip("requires FLARE_QWEN38_VIDEO_SNAPSHOT naming a Qwen3.8 checkpoint")
    path = Path(value)
    required = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                "preprocessor_config.json", "config.json")
    if not all((path / name).is_file() for name in required):
        pytest.skip("FLARE_QWEN38_VIDEO_SNAPSHOT does not name a complete checkpoint")
    return path


def pattern(width: int, height: int, seed: int) -> np.ndarray:
    state = seed
    data = bytearray(width * height * 3)
    for index in range(len(data)):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        data[index] = state & 255
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def gradient(width: int, height: int, seed: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    return np.stack([(x * 3 + seed) % 256, (y * 5 + seed) % 256,
                     (x + y + seed) % 256], axis=-1).astype(np.uint8)


def clip(frames: int, width: int, height: int, seed: int, smooth: bool = False) -> list:
    make = gradient if smooth else pattern
    return [make(width, height, seed * 101 + index) for index in range(frames)]


def data_url(frame: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def media_order(rendered: str) -> list:
    order, videos, images = [], 0, 0
    for match in PLACEHOLDER.finditer(rendered):
        if match.group() == "<|video_pad|>":
            order.append(("video", videos))
            videos += 1
        else:
            order.append(("image", images))
            images += 1
    return order


def window_grids(rendered: str, video_grids, image_grids) -> list:
    """Prompt-ordered windows: one per video temporal patch and one per still image."""
    order = []
    for kind, index in media_order(rendered):
        grids = video_grids if kind == "video" else image_grids
        grid = grids[index]
        for _ in range(grid[0] if kind == "video" else 1):
            order.append((kind, [1, int(grid[1]), int(grid[2])]))
    return order


def expected(processor, clips, fps, image=None) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from transformers.video_utils import VideoMetadata
    from transformers.vision_utils import (
        get_vision_interpolation_indices_and_weights, get_vision_position_ids,
    )

    content = [{"type": "text", "text": "Describe the clip."}]
    content += [{"type": "video", "video": "clip"} for _ in clips]
    if image is not None:
        content.append({"type": "image", "image": "image"})
    content.append({"type": "text", "text": "Answer in one word."})
    messages = [{"role": "user", "content": content}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    videos = [torch.from_numpy(np.stack(entry)).permute(0, 3, 1, 2) for entry in clips]
    metadata = [VideoMetadata(total_num_frames=len(entry), fps=fps,
                              frames_indices=list(range(len(entry)))) for entry in clips]
    # The client supplies the sampled frames, so the reference must not resample them at
    # the processor's own fps; do_sample_frames would otherwise replace the clip.
    kwargs = dict(text=[rendered], return_tensors="pt", return_metadata=True,
                  return_mm_token_type_ids=True, do_sample_frames=False)
    if videos:
        kwargs.update(videos=videos, video_metadata=metadata)
    if image is not None:
        kwargs.update(images=[Image.fromarray(image)])
    out = processor(**kwargs)
    video_grids = out["video_grid_thw"].tolist() if "video_grid_thw" in out else []
    image_grids = out["image_grid_thw"].tolist() if "image_grid_thw" in out else []

    def vision_rows(grids):
        if not grids:
            empty = torch.zeros((0, 2), dtype=torch.long)
            return empty, empty, torch.zeros((0, 4), dtype=torch.float32)
        tensor = torch.tensor(grids)
        indices, weights = get_vision_interpolation_indices_and_weights(
            tensor, 48, mode="bilinear", align_corners=True, spatial_merge_size=2)
        return get_vision_position_ids(tensor, 2), indices, weights

    rows = {"video": vision_rows(video_grids), "image": vision_rows(image_grids)}
    pixels = {"video": out["pixel_values_videos"] if video_grids else None,
              "image": out["pixel_values"] if image_grids else None}
    cursors = {"video": 0, "image": 0}
    windows = []
    for kind, grid in window_grids(rendered, video_grids, image_grids):
        patches = grid[1] * grid[2]
        start = cursors[kind]
        positions, indices, weights = rows[kind]
        windows.append(dict(kind=kind, grid=grid, patches=patches,
                            pixels=pixels[kind][start:start + patches],
                            positions=positions[start:start + patches],
                            indices=indices[start:start + patches],
                            weights=weights[start:start + patches]))
        cursors[kind] = start + patches
    positional = SimpleNamespace(config=AutoConfig.from_pretrained(snapshot(), local_files_only=True))
    positional.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids,
                                                    positional)
    rope, _ = Qwen3_5Model.get_rope_index(
        positional, out["input_ids"], out["mm_token_type_ids"],
        image_grid_thw=torch.tensor(image_grids) if image_grids else None,
        video_grid_thw=torch.tensor(video_grids) if video_grids else None)
    return {"rendered": rendered, "out": out, "windows": windows,
            "rope": rope[:, 0, :].numpy()}


def native(binary: Path, directory: Path, clips, fps, image=None):
    """Prepares the request through the native request path; returns (result, output directory)."""
    import sys

    sys.path.insert(0, str(ROOT))
    from ignition.preparation.frontend import prepare_frontend, video_settings

    directory.mkdir(parents=True, exist_ok=True)
    # The contract of a core with video settings: the Qwen3.8 contract fixture with the
    # video placeholder declared as a feature token.
    contract = json.loads(CONTRACT.read_text())
    contract["frontend"]["image_tokens"]["video_feature_token"] = VIDEO_TOKEN
    (directory / "contract.json").write_text(json.dumps(contract))
    content = [{"type": "text", "text": "Describe the clip."}]
    for entry in clips:
        content.append({"type": "video", "video": [data_url(frame) for frame in entry],
                        "fps": fps})
    if image is not None:
        content.append({"type": "image_url", "image_url": {"url": data_url(image)}})
    content.append({"type": "text", "text": "Answer in one word."})
    case = {
        "image_settings": prepare_frontend(snapshot(), eos_token_ids=[248046, 248044],
                                           image_profile="qwen",
                                           qwen_model_type="qwen3_5",
                                           video=video_settings(snapshot()))["image"],
        "contract": str(directory / "contract.json"),
        "model_id": "video-fixture",
        "body": {"model": "video-fixture", "max_tokens": 8,
                 "messages": [{"role": "user", "content": content}],
                 "chat_template_kwargs": {"add_generation_prompt": True}},
    }
    (directory / "case.json").write_text(json.dumps(case))
    output = directory / "native"
    subprocess.run([str(binary), str(snapshot()), str(directory / "case.json"), str(output)],
                   check=True, cwd=ROOT)
    return json.loads((output / "result.json").read_text()), output


def feature_spans(modalities: list) -> list:
    """Token spans of the reference's image and video placeholder runs, in prompt order."""
    spans, start = [], None
    for position, modality in enumerate(modalities + [0]):
        if start is not None and modality != modalities[start]:
            spans.append((start, position))
            start = None
        if start is None and modality != 0:
            start = position
    return spans


def array(directory: Path, name: str, dtype) -> np.ndarray:
    return np.fromfile(directory / name, dtype=dtype)


@pytest.fixture(scope="module")
def harness():
    from transformers import AutoProcessor

    path = snapshot()
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    subprocess.run(["cargo", "build", "--quiet", "--manifest-path", str(ROOT / "helios/Cargo.toml"),
                    "--features", "server,test-tools", "--bin", "test_qwen_video"], check=True)
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(ROOT / "helios/target")))
    return processor, target / "debug" / "test_qwen_video"


CASES = {
    "two_frames": dict(clips=[(2, 64, 64, 8)], fps=2.0),
    "three_frames": dict(clips=[(3, 64, 64, 9)], fps=2.0),
    "even_frames": dict(clips=[(8, 64, 64, 1)], fps=2.0),
    "odd_frames": dict(clips=[(5, 96, 64, 2)], fps=3.0),
    "resized_frames": dict(clips=[(4, 320, 240, 3, True)], fps=30.0),
    "two_clips": dict(clips=[(4, 64, 64, 4), (6, 96, 96, 5)], fps=2.0),
    "image_and_clip": dict(clips=[(4, 96, 96, 6)], fps=2.0, image=(192, 64, 7)),
    # The pixel budget shrinks both axes of a 720p clip.
    "long_720p": dict(clips=[(32, 1280, 720, 10, True)], fps=2.0),
    # 13 frames round their temporal target down to 12 before the budget is tested.
    "fhd_frames": dict(clips=[(13, 1920, 1080, 11, True)], fps=2.0),
    # The pixel budget binds after the temporal target rounds 15 frames to 16. A
    # five-frame clip cannot bind this budget inside the prepared 16,384-patch
    # window: half-even rounding to four temporal rows leaves each frame at least
    # 6.3M pixels, or 24,576 patches, above one vision window.
    "budget_binds": dict(clips=[(15, 2560, 1600, 12, True)], fps=2.0),
    # Frames below 32 pixels are resized up to one patch merge.
    "small_upscale": dict(clips=[(16, 16, 16, 13, True)], fps=2.0),
    # The resized clip falls below the minimum pixel budget.
    "min_growth": dict(clips=[(2, 16, 16, 14, True)], fps=2.0),
    # Half-even temporal rounding and the repeated final frame at one size.
    "round_three": dict(clips=[(3, 64, 64, 15, True)], fps=2.0),
    "round_five": dict(clips=[(5, 64, 64, 16, True)], fps=2.0),
    "round_six": dict(clips=[(6, 64, 64, 17, True)], fps=2.0),
    "round_seven": dict(clips=[(7, 64, 64, 18, True)], fps=2.0),
    # 7/2 rounds half-even upward to four temporal rows, and the pixel budget binds:
    # rounding down to three rows would not shrink the 2048x2048 frames.
    "tie_up_budget": dict(clips=[(7, 2048, 2048, 20, True)], fps=2.0),
    # 9 frames round their temporal target down to eight rows and still take the shrink branch.
    "round_budget": dict(clips=[(9, 2048, 2048, 21, True)], fps=2.0),
    # 32-pixel frames need no upscale and the clip still grows to the minimum budget.
    "min_growth_native": dict(clips=[(2, 32, 32, 22, True)], fps=2.0),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_native_video_preparation_matches_the_reference(harness, tmp_path, name):
    processor, binary = harness
    case = CASES[name]
    clips = [clip(*spec) for spec in case["clips"]]
    image = None if "image" not in case else pattern(*case["image"])
    reference = expected(processor, clips, case["fps"], image)
    result, output = native(binary, tmp_path / name, clips, case["fps"], image)
    out = reference["out"]

    # The checkpoint chat template ignores a video part's frame payload, so the native
    # prompt with compact placeholders must render the reference text.
    assert result["rendered"] == reference["rendered"]
    assert result["token_ids"] == out["input_ids"][0].tolist()
    assert np.array_equal(np.array(result["positions"]), reference["rope"].T)
    assert len(result["windows"]) == len(reference["windows"])
    # The reference marks each image and video placeholder token; each run of them is one
    # vision window, and features are numbered through the runs in prompt order.
    modalities = out["mm_token_type_ids"][0].tolist()
    spans = feature_spans(modalities)
    features = np.cumsum([0] + [end - start for start, end in spans]).tolist()
    indices = [-1] * len(modalities)
    for (start, end), first in zip(spans, features):
        indices[start:end] = range(first, first + end - start)
    assert result["vision_token_indices"] == indices
    assert [(window["token_start"], window["token_end"]) for window in result["windows"]] == spans
    assert [(window["feature_start"], window["feature_end"])
            for window in result["windows"]] == list(zip(features, features[1:]))
    tokens = result["token_ids"]
    for window, want in zip(result["windows"], reference["windows"]):
        assert window["grid_thw"] == want["grid"]
        assert window["patches"] == want["patches"]
        assert window["tokens"] == want["patches"] // 4
        assert window["cu_seqlens"] == [0, want["patches"]]
        placeholder = VIDEO_TOKEN if want["kind"] == "video" else IMAGE_TOKEN
        assert tokens[window["token_start"]:window["token_end"]] == [placeholder] * window["tokens"]
    starts = [window["token_start"] for window in result["windows"]]
    assert [event["position"] for event in result["conditioning"]] == starts
    assert len({event["identity"] for event in result["conditioning"]}) == len(starts)
    assert np.array_equal(
        array(output, "pixels.f32", np.float32),
        np.concatenate([want["pixels"].flatten().numpy() for want in reference["windows"]]))
    assert np.array_equal(
        array(output, "position_ids.i64", np.int64),
        np.concatenate([want["positions"].flatten().numpy() for want in reference["windows"]]))
    assert np.array_equal(
        array(output, "interpolation_indices.i64", np.int64),
        np.concatenate([want["indices"].flatten().numpy() for want in reference["windows"]]))
    assert np.array_equal(
        array(output, "interpolation_weights.f32", np.float32),
        np.concatenate([want["weights"].flatten().numpy() for want in reference["windows"]]))


def test_window_identities_follow_the_frames_of_each_temporal_patch(harness, tmp_path):
    _, binary = harness
    frames = clip(5, 64, 64, 30)

    def identities(directory, clips, fps):
        result, _ = native(binary, tmp_path / directory, clips, fps)
        windows = [window["identity"] for window in result["windows"]]
        assert [event["identity"] for event in result["conditioning"]] == windows
        return windows

    first = identities("first", [frames], 2.0)
    assert len(set(first)) == 3
    # A second preparation of the same clip, and one at another sampling rate, which
    # changes the timestamp markers only, give the same identities.
    assert identities("second", [frames], 2.0) == first
    assert identities("rate", [frames], 4.0) == first
    # Replacing the last frame changes the temporal patch that holds it.
    changed = identities("changed", [frames[:4] + [pattern(64, 64, 999)]], 2.0)
    assert changed[:2] == first[:2]
    assert changed[2] != first[2]
