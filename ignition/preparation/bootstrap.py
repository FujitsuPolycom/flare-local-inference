# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared preparation configuration fields and plan-forwarding helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from ignition.parameters import RecipeParameters, SpeculationParameters
from ignition.preparation.tuning_replicas import TuningReplica


def validate_host_topology(roce_ring: bool, roce_chain: bool, hosts: tuple[str, ...] | None) -> None:
    """Raise ValueError when requested RoCE ring/chain flags disagree with the host layout."""
    from ignition.planning import host_shape_error

    error = host_shape_error(roce_ring, roce_chain, None if hosts is None else len(hosts))
    if error is not None:
        raise ValueError(error)


@dataclass(frozen=True)
class BasePreparationConfig:
    """Common preparation settings accepted by every model recipe bootstrap.

    Recipe-specific and core ABI preparation options live on `parameters`.
    """

    parameters: RecipeParameters = field(default_factory=RecipeParameters)
    context: int = 262144
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    tuning_replica: TuningReplica | None = None

    @property
    def max_active(self) -> int:
        return self.parameters.max_concurrency

    @property
    def prefill_chunk_size(self) -> int:
        return self.parameters.prefill_chunk

    @property
    def compile_workers(self) -> int | None:
        return self.parameters.compile_workers

    @property
    def autotune_effort(self) -> str:
        return self.parameters.autotune_effort

    @property
    def lock_gpu_clocks(self) -> bool:
        return self.parameters.lock_gpu_clocks and self.parameters.autotune_effort != "off"

    @property
    def compile_decoder(self) -> bool:
        return self.parameters.compiled_forwards

    @property
    def roce_ring(self) -> bool:
        return self.parameters.roce_ring

    @property
    def roce_chain(self) -> bool:
        return self.parameters.roce_chain

    @property
    def speculation(self) -> SpeculationParameters | None:
        return self.parameters.speculation

    @property
    def mtp_only(self) -> bool:
        return bool(self.parameters.speculation and self.parameters.speculation.mtp_only)

    @property
    def speculative_widths(self) -> tuple[int, ...]:
        if self.parameters.speculation is None:
            return (1,)
        return self.parameters.speculation.widths

    @property
    def speculative_fixed_width(self) -> int | None:
        if self.parameters.speculation is None:
            return None
        return self.parameters.speculation.fixed_width

    @property
    def speculative_token_capacity(self) -> int | None:
        if self.parameters.speculation is None:
            return None
        return self.parameters.speculation.effective_token_capacity(self.parameters.max_concurrency)

    @property
    def speculative_graph_budget(self) -> int | None:
        if self.parameters.speculation is None:
            return None
        return self.parameters.speculation.graph_budget

    @property
    def prompt_logprobs(self) -> bool:
        return bool(self.parameters.prompt_logprobs and self.parameters.prompt_logprobs.enabled)

    @property
    def image_concurrency(self) -> int:
        if self.parameters.vision is None:
            return 0
        return self.parameters.vision.effective_image_concurrency(self.parameters.max_concurrency)

    @property
    def language_model_only(self) -> bool:
        return bool(self.parameters.vision and self.parameters.vision.language_model_only)

    @property
    def mixed_prefill(self) -> bool:
        return bool(self.parameters.mixed_prefill and self.parameters.mixed_prefill.enabled)


def common_preparation_kwargs(plan, *, compile_forwards: bool | None = None, kernel_cache: Path | None,
                              retain_graphs: bool) -> dict:
    """Extract the shared PreparationConfig keyword arguments from a sizing Plan."""
    parameters = plan.parameters
    if compile_forwards is not None and parameters.compiled_forwards != compile_forwards:
        parameters = replace(parameters, compiled_forwards=bool(compile_forwards))
    return {
        "parameters": parameters,
        "tuning_replica": plan.tuning_replica,
        "context": plan.declared_context,
        "snapshot_budget_bytes": plan.snapshots * plan.costs.state_row_bytes,
        "headroom_bytes": plan.headroom_bytes + plan.reserve_bytes,
        "retain_graphs": retain_graphs,
        "kernel_cache": kernel_cache,
    }
