# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM context-parallel candidate gathers keep every 512-row slice in place when Inductor compiles them.

Writing each slice's gather through a dtype-viewed row slice of a graph-local FP32 buffer lets Inductor
size the reinplaced buffer to one slice, so later slices land out of bounds. `gather_candidates` writes plain
BF16 row slices; this test runs it compiled (CPU; CUDA with FLARE_RUN_CUDA_TESTS=1) with a mutating custom
op standing in for the b12x all-gather.
"""

import os

import pytest
import torch

from ignition.recipes.glm53.dcp import SLICE_ROWS
from ignition.recipes.glm53.indexer import gather_candidates

TOPK = 512


@torch.library.custom_op("flare_test::glm_candidate_gather", mutates_args=("output",))
def _gather(source: torch.Tensor, output: torch.Tensor, world: int) -> None:
    # Rank r contributes this rank's candidates plus r, in rank order across the row.
    columns = source.shape[1]
    for rank in range(world):
        output[:, rank * columns:(rank + 1) * columns].copy_(source + rank)


@_gather.register_fake
def _(source, output, world):
    pass


@torch.library.custom_op("flare_test::glm_candidate_pack", mutates_args=("packed",))
def _pack(source: torch.Tensor, packed: torch.Tensor) -> None:
    packed.copy_(source)


@_pack.register_fake
def _(source, packed):
    pass


class Gather:
    def __init__(self, world):
        self.world = world

    def __call__(self, source, output=None):
        _gather(source, output, self.world)
        return output


def candidate_merge_input(source, world):
    rows = source.shape[0]
    candidates = torch.zeros((rows, TOPK, 2), dtype=torch.float32, device=source.device)
    _pack(source, candidates)
    slices = [(start, min(start + SLICE_ROWS, rows)) for start in range(0, rows, SLICE_ROWS)]
    gathered = gather_candidates([Gather(world)] * len(slices), slices, candidates, (rows, world * TOPK, 2))
    return gathered.clone()


def reference(source, world):
    rows = source.shape[0]
    packed = source.view(torch.bfloat16).view(rows, -1)
    return torch.cat([(packed + rank) for rank in range(world)], 1).view(torch.float32).view(rows, world * TOPK, 2)


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="CUDA Inductor case runs with FLARE_RUN_CUDA_TESTS=1"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("world", "rows"), [(3, 1), (3, 512), (3, 1152), (3, 1536), (3, 4096), (2, 1536),
                                             (4, 3584)])
def test_candidate_slices_land_in_their_rows_when_compiled(world, rows, device):
    torch.manual_seed(rows + world)
    # Packed (score, pool) pairs are FP32 bit patterns; whole BF16 halves keep the reference exact.
    source = torch.randn(rows, TOPK, 2).to(torch.bfloat16).to(torch.float32).to(device)
    torch._dynamo.reset()
    compiled = torch.compile(lambda s: candidate_merge_input(s, world), backend="inductor", fullgraph=True)
    expected = reference(source, world)
    eager = candidate_merge_input(source, world)
    observed = compiled(source)
    assert torch.equal(eager.view(torch.int32), expected.view(torch.int32))
    wrong = (observed.view(torch.int32) != expected.view(torch.int32)).flatten(1).any(1).nonzero().flatten()
    assert wrong.numel() == 0, f"rows {wrong[:8].tolist()} of {rows} hold another slice's or no candidates"
