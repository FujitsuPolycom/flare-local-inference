# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural checkpoint configuration for the Qwen3.5 dense architecture."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from ignition.preparation.quantization import quantized_layers

MODEL_ID = "nvidia/Qwen3.8-27B-NVFP4"
REVISION = "482ca0f3832238542f8f5295dde86b5f22711d80"
TEXT_PREFIX = "model.language_model"
PAGE_TOKENS = 128


DEFAULT_SECTIONS = (11, 11, 10)


def rotary_sections(rotary):
    """Declared position sections, defaulting to the reference's split."""
    sections = rotary.get("mrope_section", DEFAULT_SECTIONS)
    if not isinstance(sections, (list, tuple)):
        raise ValueError("Qwen dense rotary geometry requires three position sections")
    sections = tuple(sections)
    if len(sections) != 3 or any(type(size) is not int or size <= 0 for size in sections):
        raise ValueError("Qwen dense rotary geometry requires three position sections")
    return sections


def rotary_axes(rotary, pairs):
    """Axis index for each rotary frequency pair of a three-axis checkpoint.

    The reference assigns every pair to the temporal axis, then overwrites the
    height and width sections in three-pair strides. Section sizes count pairs,
    and the interleaved layout places each section's pairs at the same offsets.
    """
    sections = rotary_sections(rotary)
    if sum(sections) != pairs:
        raise ValueError("Qwen dense rotary sections must cover every frequency pair")
    axes = torch.zeros(pairs, dtype=torch.int64)
    for axis, size in enumerate(sections):
        axes[axis:size * 3:3] = axis
    return axes


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    quantization: dict
    draft: object = None
    vision: dict | None = None

    @classmethod
    def load(cls, snapshot, *, vision=False):
        """Load structural configuration; ``vision`` also validates the image declarations.

        A text-only preparation rotates every token at its index and loads no
        vision tensors, so it reads neither the position sections nor
        ``vision_config``. A preparation with image slots requires interleaved
        sections that cover every frequency pair and a ``qwen3_5_vision`` encoder
        of the language-model width.
        """
        raw = json.loads((Path(snapshot) / "config.json").read_text())
        if raw.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
            raise ValueError("Qwen dense preparation requires Qwen3_5ForConditionalGeneration")
        text = raw.get("text_config", {})
        required = {"model_type": "qwen3_5_text", "hidden_act": "silu", "attention_bias": False,
                    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
                    "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                    "linear_value_head_dim": 128, "head_dim": 256, "tie_word_embeddings": False}
        for name, expected in required.items():
            if text.get(name) != expected:
                raise ValueError(f"unsupported Qwen dense {name}: {text.get(name)!r}")
        for name in ("hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "linear_num_key_heads",
                     "linear_num_value_heads", "max_position_embeddings"):
            if type(text.get(name)) is not int or text[name] <= 0:
                raise ValueError(f"Qwen dense requires a positive {name}")
        layers = text.get("layer_types", ())
        if (len(layers) != text["num_hidden_layers"]
                or any(kind not in ("linear_attention", "full_attention") for kind in layers)):
            raise ValueError("Qwen dense layer types must describe every attention layer")
        if (text["num_attention_heads"] % text["num_key_value_heads"]
                or text["linear_num_value_heads"] % text["linear_num_key_heads"]):
            raise ValueError("Qwen query and value head counts must divide their key head counts")
        rope = text.get("rope_parameters", {})
        if rope.get("rope_type") != "default" or rope.get("partial_rotary_factor") != .25:
            raise ValueError("Qwen dense preparation requires default partial rotary embeddings")
        if not 0 < float(rope.get("rope_theta", 0)) or not 0 < float(text.get("rms_norm_eps", 0)):
            raise ValueError("Qwen rotary frequency base and normalization epsilon must be positive")
        vision_config = raw.get("vision_config")
        if vision:
            if rope.get("mrope_interleaved", True) is not True:
                raise ValueError("Qwen dense preparation implements interleaved position sections only")
            rotary_axes(rope, int(text["head_dim"] * rope["partial_rotary_factor"]) // 2)
            if not isinstance(vision_config, dict) or vision_config.get("model_type") != "qwen3_5_vision":
                raise ValueError("Qwen dense vision configuration requires the qwen3_5_vision family")
            for name in ("hidden_size", "num_heads", "intermediate_size", "depth",
                         "out_hidden_size", "num_position_embeddings"):
                if type(vision_config.get(name)) is not int or vision_config[name] <= 0:
                    raise ValueError(f"Qwen dense vision requires a positive {name}")
            if (vision_config["hidden_size"], vision_config["num_heads"],
                    vision_config["intermediate_size"], vision_config["depth"]) != (1152, 16, 4304, 27):
                raise ValueError("Qwen dense vision geometry differs from the checkpoint contract")
            if vision_config["num_position_embeddings"] != 2304:
                raise ValueError("Qwen dense vision requires 2,304 learned position embeddings")
            for name, expected in (("patch_size", 16), ("temporal_patch_size", 2),
                                   ("spatial_merge_size", 2), ("in_channels", 3)):
                if vision_config.get(name, expected) != expected:
                    raise ValueError(f"Qwen dense vision requires {name}={expected}")
            if vision_config["out_hidden_size"] != text["hidden_size"]:
                raise ValueError("Qwen dense vision features must match the language-model width")
        return cls(text, dict(quantized_layers(raw)), vision=vision_config if vision else None)

    @property
    def layers(self):
        return tuple(self.text["layer_types"])

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    @property
    def pack_gdn_inputs(self):
        from ignition.preparation.quantization import linear_format

        names = tuple(f"{TEXT_PREFIX}.layers.{index}.linear_attn.in_proj_{part}"
                      for index, kind in enumerate(self.layers) if kind == "linear_attention"
                      for part in ("qkv", "z", "b", "a"))
        return bool(names) and all(name in self.quantization
                                   and linear_format(self.quantization, name) == "mxfp8"
                                   for name in names)
