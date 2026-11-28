# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Recipe declarations: qualified checkpoints, serving ranges, and memory costs.

Declarations import no model code, so planning and model listing run without
PyTorch.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields, replace
import math
from pathlib import Path
from types import MappingProxyType

from ignition.components.capacity import request_capacity
from ignition.parameters import (
    ParameterContext,
    RecipeParameters,
    SpeculationParameters,
    valid_speculative_method_name,
)


@dataclass(frozen=True)
class SamplingRange:
    """Inclusive bounds from the model author's sampling recommendations."""

    min: float | int
    max: float | int


@dataclass(frozen=True)
class SamplingDefaults:
    """Complete serving parameters manually verified against the model card."""

    temperature: float
    top_p: float
    top_k: int
    seed: int = 0
    min_p: float = 0.
    repetition_penalty: float = 1.
    presence_penalty: float = 0.
    frequency_penalty: float = 0.
    ranges: Mapping[str, SamplingRange] = field(default_factory=dict)

    def __post_init__(self):
        values = (self.temperature, self.top_p, self.min_p, self.repetition_penalty,
                  self.presence_penalty, self.frequency_penalty)
        if (any(type(v) not in (int, float) or not math.isfinite(v)
                or abs(v) > float.fromhex("0x1.fffffep+127") for v in values)
                or any(v < float.fromhex("0x1p-126") for v in
                       (self.temperature, self.top_p, self.repetition_penalty))
                or not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1
                or type(self.top_k) is not int or not -1 <= self.top_k < 2**31
                or type(self.seed) is not int or not 0 <= self.seed < 2**64):
            raise ValueError("invalid recipe sampling defaults")
        names = {item.name for item in fields(self)} - {"seed", "ranges"}
        for name, bounds in self.ranges.items():
            if (name not in names or not isinstance(bounds, SamplingRange)
                    or any(type(v) not in (int, float) or not math.isfinite(v)
                           for v in (bounds.min, bounds.max))
                    or not bounds.min <= getattr(self, name) <= bounds.max):
                raise ValueError(f"invalid sampling range for {name}")
            for value in (bounds.min, bounds.max):
                replace(self, **{name: value}, ranges={})
        object.__setattr__(self, "ranges", MappingProxyType(dict(self.ranges)))

    def as_dict(self):
        result = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "ranges"}
        if self.ranges:
            result["ranges"] = {name: asdict(bounds) for name, bounds in self.ranges.items()}
        return result


def _plain_mappings(value: object) -> object:
    """Replace read-only mapping proxies with plain dicts so checkpoints pickle cleanly."""
    if isinstance(value, Mapping):
        return {key: _plain_mappings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain_mappings(item) for item in value)
    return value


def _restore_checkpoint(cls, values: dict[str, object]) -> "Checkpoint":
    return cls(**values)


@dataclass(frozen=True)
class Checkpoint:
    """One qualified checkpoint revision and where its stored tensors reside."""

    alias: str
    repository: str
    revision: str
    architecture: str
    stored_bytes: int
    device_bytes: int
    mapped_host_bytes: int = 0
    # The share of `device_bytes` that holds the draft model, loaded only for learned drafts.
    draft_bytes: int = 0
    # Device bytes that loading adds beyond the stored tensors, such as block scales
    # expanded from a compressed encoding.
    expansion_bytes: int = 0
    # Estimated resident device weights after optional lossless expert-scale compression.
    scale_compressed_device_bytes: Mapping[str, int] = field(default_factory=dict)
    # Per-rank costs replacing the recipe's `ranks` and `plain_ranks` when this checkpoint's draft or state
    # layout differs, such as a draft model that only this checkpoint carries. The declared sizes may be
    # a subset of the recipe's when the checkpoint's stored shards exclude a tensor-parallel size.
    ranks: Mapping[int, "RankCosts"] | None = None
    # Bytes of weights one decode step reads for one request across every rank: dense weights
    # plus the experts one token activates, at their stored precision. Zero leaves the serving
    # contract without the value, and dashboards then show no weight bandwidth.
    decode_weight_bytes: int = 0
    plain_ranks: Mapping[int, "RankCosts"] | None = None
    # API model name advertised for this checkpoint; empty uses the recipe's `served_model_name`.
    served_model_name: str = ""
    # Longest sequence this checkpoint's configuration serves when it differs from the recipe's
    # `context`; None uses the recipe's `context`. Recipes may declare it statically or derive
    # it from checkpoint metadata in `Recipe.inspect_replacement`.
    context: int | None = None

    def __post_init__(self):
        if self.context is not None and (
            type(self.context) is not int or not 64 <= self.context < 2**31 or self.context % 64
        ):
            raise ValueError("checkpoint context must be a 64-aligned positive int32")

    def __reduce__(self):
        return (
            _restore_checkpoint,
            (type(self), {item.name: _plain_mappings(getattr(self, item.name)) for item in fields(self)}),
        )


@dataclass(frozen=True)
class RankCosts:
    """Device bytes one tensor-parallel rank spends per unit of capacity."""

    state_row_bytes: int
    page_bytes: int
    graph_bytes: int
    # Global positions per page when ranks share each page's context; the recipe's page size otherwise.
    page_tokens: int | None = None
    # Additional costs of a plan with image slots, for a recipe whose language-only
    # plans prepare no vision encoder. `vision_bytes` holds the fixed image tower
    # and upload storage, `vision_slot_bytes` each feature slot, and
    # `image_state_row_bytes` the continuation row that replaces `state_row_bytes`;
    # None keeps the same row.
    vision_bytes: int = 0
    vision_slot_bytes: int = 0
    image_state_row_bytes: int | None = None

    def with_image_slots(self, slots: int) -> "RankCosts":
        """Costs of a plan with `slots` image slots: the image row, and the vision storage as graph bytes."""
        if not slots or (self.image_state_row_bytes is None and not self.vision_bytes and not self.vision_slot_bytes):
            return self
        return replace(self, state_row_bytes=self.image_state_row_bytes or self.state_row_bytes,
            graph_bytes=self.graph_bytes + self.vision_bytes + slots * self.vision_slot_bytes)


@dataclass(frozen=True)
class RankUsage:
    """What one rank allocated and left free after preparation."""

    pages: int
    free_bytes: int


@dataclass(frozen=True)
class SpeculativeMethod:
    """Physical preparation costs for a recipe's learned proposal source."""

    ranks: Mapping[int, RankCosts]
    draft_bytes: int | None = None
    label: str | None = None
    mixed_prefill: bool = True


@dataclass(frozen=True)
class Recipe:
    """One model family's qualified checkpoints, serving range, parameters, costs, and bootstrap.

    `prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress)` loads,
    tunes, and captures the plan's serving catalog and returns the prepared model.
    `usage(prepared)` reports each rank's device memory in rank order.
    `served_model_name` is the canonical model name advertised by the API,
    independent of checkpoint selection aliases and storage formats.

    `parameters` holds the recipe's default `RecipeParameters` instance, which
    defines its supported CLI options, defaults, bounds, validation, planning
    hooks, and `PreparationManifest` serialization.

    `ranks` holds each tensor-parallel size's costs as the recipe serves by
    default. A recipe whose `parameters.speculation` is set drafts with those
    settings unless a plan overrides them, and `plain_ranks` holds its costs
    with speculation disabled. Each active request has one execution state row
    per selected verification position unless `verification_rows` gives another
    count of extra rows.

    `export_repository` names the Hugging Face destination and the default local
    snapshot namespace. Export writes locally; it does not upload or select a
    signing identity.
    `b12x_environment` holds the b12x settings the recipe prepares with. b12x
    reads them only from its environment, so Ignition sets them for the
    preparing process.
    `inspect_replacement`, when provided, is called as
    `inspect_replacement(metadata_dir, replacement)` during resolution of an
    unregistered checkpoint directory or Hugging Face repository so the recipe
    can inspect alternate weight container metadata and return an adjusted
    `Checkpoint` (or `None` to keep the default scaled estimate).
    """

    name: str
    served_model_name: str
    checkpoints: tuple[Checkpoint, ...]
    context: int
    page_tokens: int
    ranks: Mapping[int, RankCosts]
    prepare: Callable
    usage: Callable
    sampling_defaults: SamplingDefaults
    parameters: RecipeParameters = field(default_factory=RecipeParameters)
    mapped_host_name: str = ""
    max_images: int = 0
    image_bytes: int = 0
    speculative_methods: Mapping[str, SpeculativeMethod] = field(default_factory=lambda: MappingProxyType({}))
    plain_ranks: Mapping[int, RankCosts] = field(default_factory=lambda: MappingProxyType({}))
    verification_rows: int | None = None
    b12x_environment: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    export_repository: str = ""
    # Tensor-parallel world sizes that prepare standalone benchmark graph subsets.
    benchmark_world_sizes: tuple[int, ...] = ()
    # Whether standalone benchmark preparation can run the multi-width speculation study.
    speculation_study: bool = False
    # Whether standalone benchmark preparation prepares text-only graphs without a vision encoder.
    benchmark_language_model_only: bool = False
    inspect_replacement: Callable[[Path, Checkpoint], Checkpoint | None] | None = None

    def __post_init__(self):
        if not isinstance(self.parameters, RecipeParameters):
            raise TypeError("recipe parameters must be a RecipeParameters instance")
        if self.inspect_replacement is not None and not callable(self.inspect_replacement):
            raise TypeError("recipe inspect_replacement must be callable")
        if self.export_repository:
            import re
            if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*', self.export_repository)
                    or '..' in self.export_repository):
                raise ValueError('export_repository must be a Hugging Face namespace/repository')
        if any(not name.startswith("B12X_") or not isinstance(value, str)
               for name, value in self.b12x_environment.items()):
            raise ValueError("b12x settings name B12X_ variables with string values")
        if self.parameters.speculation is not None and set(self.plain_ranks) != set(self.ranks):
            raise ValueError("a recipe that speculates declares its costs without speculation for every rank count")
        for name, method in self.speculative_methods.items():
            if not valid_speculative_method_name(name) or set(method.ranks) != set(self.ranks):
                raise ValueError(
                    "speculative methods require lowercase identifier names and every declared tensor-parallel size"
                )
            if method.draft_bytes is not None and method.draft_bytes <= 0:
                raise ValueError("speculative draft bytes must be positive")
            if method.label is not None and (not isinstance(method.label, str) or not method.label.strip()):
                raise ValueError("speculative method label must be a nonempty string")
            if type(method.mixed_prefill) is not bool:
                raise ValueError("speculative method mixed_prefill must be a bool")
        for checkpoint in self.checkpoints:
            if any(costs is not None and (not costs or not set(costs) <= set(self.ranks))
                   for costs in (checkpoint.ranks, checkpoint.plain_ranks)):
                raise ValueError(f"{checkpoint.alias} declares costs for tensor-parallel sizes its recipe does not serve")
            if (checkpoint.ranks is None) != (checkpoint.plain_ranks is None) or (
                    checkpoint.ranks is not None and set(checkpoint.ranks) != set(checkpoint.plain_ranks)):
                raise ValueError(f"{checkpoint.alias} declares costs with and without speculation for different sizes")
            if checkpoint.context is not None and checkpoint.context % self.page_tokens:
                raise ValueError(f"{checkpoint.alias} context must be a multiple of the recipe page size ({self.page_tokens})")

    @property
    def concurrency(self) -> int:
        return self.parameters.max_concurrency

    @property
    def max_concurrency(self) -> int:
        return self.parameters.max_concurrency_limit

    @property
    def prefill_chunk(self) -> int:
        return self.parameters.prefill_chunk

    @property
    def speculation(self) -> SpeculationParameters | None:
        return self.parameters.speculation

    @property
    def max_draft_tokens(self) -> int:
        if self.parameters.speculation is None:
            return 0
        limit = self.parameters.speculation.max_draft_tokens
        return self.parameters.speculation.draft_tokens if limit is None else limit

    @property
    def tensor_parallel(self) -> tuple[int, ...]:
        return tuple(sorted(self.ranks))

    def checkpoint_tensor_parallel(self, checkpoint: Checkpoint) -> tuple[int, ...]:
        """Tensor-parallel sizes `checkpoint` prepares: its own declared sizes, else the recipe's."""
        return self.tensor_parallel if checkpoint.ranks is None else tuple(sorted(checkpoint.ranks))

    def served_name(self, checkpoint: Checkpoint) -> str:
        return checkpoint.served_model_name or self.served_model_name

    def base_context(self, checkpoint: Checkpoint | None = None) -> int:
        """Serving context ceiling declared by `checkpoint`, else the recipe's native context."""
        return self.context if checkpoint is None or checkpoint.context is None else checkpoint.context

    def declared_context(self, checkpoint: Checkpoint | None = None,
                         parameters: RecipeParameters | None = None) -> int:
        """Serving context ceiling for `checkpoint` after applying parameter-specific adjustments."""
        params = self.parameters if parameters is None else parameters
        base = self.base_context(checkpoint)
        context = ParameterContext(
            recipe=self, checkpoint=checkpoint, root=params, defaults=self.parameters,
        )
        return params.adjust_context(base, context)

    def costs(self, world_size: int, speculation: SpeculationParameters | None,
              checkpoint: Checkpoint | None = None) -> RankCosts:
        """Return one rank's costs under the given speculation settings, for `checkpoint` when it declares its own."""
        ranks = self.ranks if checkpoint is None or checkpoint.ranks is None else checkpoint.ranks
        plain = self.plain_ranks if checkpoint is None or checkpoint.plain_ranks is None else checkpoint.plain_ranks
        if speculation is None:
            return (plain or ranks)[world_size]
        method = self.speculative_methods.get(speculation.method)
        return ranks[world_size] if method is None else method.ranks[world_size]

    def base_weights_bytes(self, checkpoint: Checkpoint, world_size: int,
                           speculation: SpeculationParameters | None, *,
                           device_bytes: int | None = None) -> int:
        """Estimate one rank's base device weights from stored target and draft bytes."""
        learned = speculation is not None
        total_device = checkpoint.device_bytes if device_bytes is None else device_bytes
        draft_bytes = checkpoint.draft_bytes
        method = None if speculation is None else self.speculative_methods.get(speculation.method)
        if method is not None and method.draft_bytes is not None:
            draft_bytes = method.draft_bytes
        return -(-(total_device + checkpoint.expansion_bytes - checkpoint.draft_bytes
                   + (draft_bytes if learned else 0)) // world_size)

    def weights_bytes(self, checkpoint: Checkpoint, world_size: int,
                      parameters: RecipeParameters | None = None) -> int:
        """Estimate one rank's device weights after applying parameter-specific adjustments."""
        params = self.parameters if parameters is None else parameters
        base = self.base_weights_bytes(checkpoint, world_size, params.speculation)
        context = ParameterContext(
            recipe=self, checkpoint=checkpoint, root=params, defaults=self.parameters, world_size=world_size,
        )
        return params.adjust_weights_bytes(base, context)

    def live_rows(self, concurrency: int, speculation: SpeculationParameters | None) -> int:
        """Permanently backed null, live, and speculative execution rows."""
        slots = request_capacity(concurrency)
        speculative = (0 if speculation is None else speculation.draft_tokens if self.verification_rows is None
                       else self.verification_rows)
        return 1 + slots * (1 + speculative)
