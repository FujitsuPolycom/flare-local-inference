# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 dense patch uploads and generation-scoped image feature publication."""

from dataclasses import dataclass

import torch

from ignition.components.vision_delivery import PreparedFeaturePublication
from ignition.recipes.qwen38.vision import VisionMetadata

FEATURES = 16384
PATCHES = 16384
IMAGE_TOKEN_ID = 248056


@dataclass(frozen=True)
class VisionInputs:
    vision_pixel_values: torch.Tensor
    vision_interpolation_indices: torch.Tensor
    vision_interpolation_weights: torch.Tensor
    vision_position_ids: torch.Tensor
    vision_cu_seqlens: torch.Tensor
    vision_num_patches: torch.Tensor
    vision_slot: torch.Tensor
    vision_generation: torch.Tensor
    vision_feature_offset: torch.Tensor

    @classmethod
    def allocate(cls, device):
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros((PATCHES, 1536), torch.float32),
            zeros((PATCHES, 4), torch.int64),
            zeros((PATCHES, 4), torch.float32),
            zeros((PATCHES, 2), torch.int64),
            zeros(2, torch.int32),
            zeros(1, torch.int32),
            torch.full((1,), -1, dtype=torch.int32, device=device),
            zeros(1, torch.int64),
            zeros(1, torch.int32),
        )

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class PreparedVisionGeneration:
    """Validate/sanitize uploaded patches, run `PreparedVision`, and publish features."""

    def __init__(self, encoder, inputs, state, completion, frame, sequence):
        self.encoder, self.inputs, self.state = encoder, inputs, state
        self.publish = PreparedFeaturePublication(state, inputs, completion, frame, sequence)

    def __call__(self):
        self.forward()

    def forward(self):
        e, i, s = self.encoder, self.inputs, self.state
        r, capacity = s.requests, e.capacity
        slot = i.vision_slot.long().clamp(0, r.capacity - 1)
        count, offset = i.vision_num_patches, i.vision_feature_offset
        features = count // 4
        current = (i.vision_slot >= 0) & (i.vision_slot < r.capacity)
        current &= (i.vision_generation > 0) & (i.vision_generation == r.generation[slot]) & (r.status[slot] == 1)
        valid = current & (count > 0) & (count <= capacity) & (count % 4 == 0)
        same_epoch = s.generation[slot] == i.vision_generation
        valid &= (offset >= 0) & (offset.long() + features.long() <= s.arena.shape[1])
        valid &= offset == torch.where(same_epoch, s.count[slot], 0)
        valid &= (i.vision_cu_seqlens[0] == 0) & (i.vision_cu_seqlens[1] == count)
        valid &= (s.feature_slot >= 0) & (s.feature_slot < s.arena.shape[0])
        live = torch.arange(capacity, device=count.device) < count
        pixels = i.vision_pixel_values[:capacity]
        indices = i.vision_interpolation_indices[:capacity]
        weights = i.vision_interpolation_weights[:capacity]
        positions = i.vision_position_ids[:capacity]
        valid &= (~live[:, None] | torch.isfinite(pixels)).all()
        valid &= (~live[:, None] | ((indices >= 0) & (indices < 2304))).all()
        valid &= (~live[:, None] | ((weights >= 0.0) & (weights <= 1.0))).all()
        valid &= (~live[:, None] | ((positions >= 0) & (positions < PATCHES))).all()
        errors = (~valid).to(torch.int32)
        m = VisionMetadata(
            torch.where(valid & live[:, None], indices, 0),
            torch.where(valid & live[:, None], weights, 0.0),
            torch.where(valid & live[:, None], positions, 0),
            torch.where(valid, i.vision_cu_seqlens, 0),
            torch.where(valid, count, 0),
        )
        output = e(torch.where(valid & live[:, None], pixels, 0.0).to(torch.bfloat16), m)
        live_features = m.num_patches // 4
        feature_live = torch.arange(capacity // 4, device=count.device) < live_features
        finite = (~feature_live[:, None] | torch.isfinite(output)).all()
        errors = errors | (~finite).to(torch.int32) * 64
        self.publish(output, live_features, errors)
        return output, errors
