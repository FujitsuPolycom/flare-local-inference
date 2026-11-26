# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Derive a serving plan from a recipe's cost model and the host's free GPU memory.

Weights, graphs, scratch and execution rows have dedicated storage. KV pages
take the remaining supply after headroom and the configured reserve. Immutable
prefix snapshots borrow KV backing while retained; an idle snapshot slot costs
no physical memory. Active requests need at least 8,192 tokens each.
"""

from dataclasses import dataclass, replace
import os
from pathlib import Path

from ignition.declaration import Checkpoint, Recipe
from ignition.errors import CapacityError, UsageError
from ignition.parameters import ParameterContext, RecipeParameters, SpeculationParameters
from ignition.preparation.selection import BenchmarkSelection
from ignition.preparation.tuning_replicas import TuningReplica

# Free memory kept after the final KV allocation during preparation. Headroom
# covers the transient working memory of compiled warm-up and graph capture,
# measured at 0.83 GiB for one Qwen rank with 4096-token prefill chunks. Part of
# it holds chunk activations, so chunk tokens past 4096 add HEADROOM_CHUNK_TOKEN_BYTES
# each (8192-token chunks need about 1.3 GiB on GB300). Helios sizes its caches
# after graph upload with a separate driver margin. Graphs, scratch, and the CUDA
# context are planned separately.
HEADROOM_BYTES = 1 << 30
HEADROOM_CHUNK_TOKENS = 4096
HEADROOM_CHUNK_TOKEN_BYTES = 128 << 10
MINIMUM_REQUEST_TOKENS = 8192
# Each exchange can retain prompt and terminal checkpoints; snapshot slots have
# virtual capacity for about four recent exchanges per concurrent request.
CACHE_STATES_PER_REQUEST = 8
SUPPORTED_CAPABILITIES = ((10, 3), (12, 0), (12, 1))
SUPPORTED_ARCHITECTURES = "SM103, SM120, or SM121"
_GIB = 1 << 30


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    uuid: str
    total_bytes: int
    free_bytes: int
    capability: tuple[int, int]

    @property
    def supported(self) -> bool:
        return self.capability in SUPPORTED_CAPABILITIES


def discover_gpus() -> tuple[Gpu, ...]:
    """Enumerate NVML devices and include the shared CUDA budget on Windows iGPUs."""
    try:
        from cuda.bindings import nvml

        nvml.init_v2()
    except Exception as error:  # noqa: BLE001 - Import and driver failures share one report.
        raise CapacityError(f"cannot enumerate GPUs through NVML: {error}") from error
    try:
        gpus = []
        for index in range(nvml.device_get_count_v2()):
            handle = nvml.device_get_handle_by_index_v2(index)
            try:
                capability = tuple(nvml.device_get_cuda_compute_capability(handle))
            except nvml.NotSupportedError:
                # NVML also enumerates non-CUDA devices, including Windows NPUs.
                continue
            try:
                memory = nvml.device_get_memory_info_v2(handle)
                total, free = memory.total, memory.free
                if os.name == "nt":
                    shared = _integrated_memory(nvml.device_get_pci_info_v3(handle).bus_id, required=False)
                    if shared is not None:
                        total, free = shared
            except nvml.NotSupportedError:
                # GB10 has no framebuffer for NVML to report; it allocates from system memory.
                total, free = _integrated_memory(nvml.device_get_pci_info_v3(handle).bus_id)
            gpus.append(Gpu(index, nvml.device_get_name(handle), nvml.device_get_uuid(handle), total,
                            free, capability))
        return tuple(gpus)
    finally:
        nvml.shutdown()


def _integrated_memory(bus_id: str, *, required=True) -> tuple[int, int] | None:
    """Query shared DRAM on Linux or the dedicated-plus-shared CUDA budget on Windows."""
    from cuda.bindings import driver

    def checked(call, *arguments):
        status, *values = call(*arguments)
        if status != driver.CUresult.CUDA_SUCCESS:
            raise CapacityError(f"cannot query integrated GPU memory: {call.__name__}: {status}")
        return values[0] if len(values) == 1 else tuple(values) if values else None

    checked(driver.cuInit, 0)
    device = checked(driver.cuDeviceGetByPCIBusId, bus_id.encode())
    if not checked(driver.cuDeviceGetAttribute, driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_INTEGRATED, device):
        if not required:
            return None
        raise CapacityError(f"GPU {bus_id} does not report device memory and is not an integrated GPU")
    if os.name == "nt":
        context = checked(driver.cuDevicePrimaryCtxRetain, device)
        try:
            checked(driver.cuCtxPushCurrent, context)
            try:
                free, total = checked(driver.cuMemGetInfo)
                return total, free
            finally:
                checked(driver.cuCtxPopCurrent)
        finally:
            checked(driver.cuDevicePrimaryCtxRelease, device)
    total = checked(driver.cuDeviceTotalMem, device)
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return total, min(total, int(line.split()[1]) * 1024)
    raise CapacityError("Linux MemAvailable is required to size integrated GPU memory")


def restrict_gpus(gpus: tuple[Gpu, ...], visible: str) -> tuple[Gpu, ...]:
    """Renumber GPUs as a caller's CUDA_VISIBLE_DEVICES value enumerates them.

    Integer entries are NVML indices and other entries are GPU UUID prefixes.
    """
    restricted = []
    for entry in filter(None, (value.strip() for value in visible.split(","))):
        if entry.isdecimal():
            matches = [gpu for gpu in gpus if gpu.index == int(entry)]
        else:
            matches = [gpu for gpu in gpus if gpu.uuid.startswith(entry)]
        if len(matches) != 1 or matches[0] in restricted:
            raise UsageError(f"CUDA_VISIBLE_DEVICES entry {entry!r} does not name one distinct GPU")
        restricted.append(matches[0])
    return tuple(replace(gpu, index=index) for index, gpu in enumerate(restricted))


def parse_gpus(text: str) -> tuple[int, ...]:
    """Parse `0`, `0,1`, or `0-3` into distinct GPU indices."""
    indices = []
    for part in text.split(","):
        first, dash, last = part.strip().partition("-")
        if not first.isdecimal() or (dash and not last.isdecimal()):
            raise UsageError(f"--gpus expects indices such as 0, 0,1, or 0-3; got {text!r}")
        indices.extend(range(int(first), int(last if dash else first) + 1))
    if not indices or len(set(indices)) != len(indices):
        raise UsageError(f"--gpus expects distinct indices in increasing ranges; got {text!r}")
    return tuple(indices)


def format_gpus(indices) -> str:
    indices = tuple(indices)
    if len(indices) > 1 and indices == tuple(range(indices[0], indices[-1] + 1)):
        return f"{indices[0]}-{indices[-1]}"
    return ",".join(map(str, indices))


@dataclass(frozen=True)
class Plan:
    """Per-rank memory demand and supply for one checkpoint on selected GPUs."""

    recipe: Recipe
    checkpoint: Checkpoint
    gpus: tuple[Gpu, ...]
    parameters: RecipeParameters
    cache_target: int
    cache_states: int
    pages: int
    weights_bytes: int
    graph_bytes: int
    hub_cache: Path | None = None
    tuning_gpus: tuple[Gpu, ...] = ()
    tuning_replica: TuningReplica | None = None
    benchmark_selection: BenchmarkSelection | None = None

    def __post_init__(self):
        if not isinstance(self.parameters, RecipeParameters):
            raise TypeError("Plan.parameters must be a RecipeParameters instance")
        if self.parameters.tensor_parallel != self.world_size:
            object.__setattr__(self, "parameters", replace(self.parameters, tensor_parallel=self.world_size))
        context = self.parameter_context
        self.parameters.validate(context)
        if self.benchmark_selection is not None:
            selection = self.benchmark_selection
            if not isinstance(selection, BenchmarkSelection):
                raise UsageError("benchmark selection requires explicit complete-round geometry")
            if self.world_size not in self.recipe.benchmark_world_sizes:
                raise UsageError(
                    f"benchmark graph preparation is not supported for {self.checkpoint.alias} on {self.world_size} GPUs"
                )
            if selection.speculation_study and not self.recipe.speculation_study:
                raise UsageError(
                    f"speculation studies require a recipe that declares speculation_study ({self.checkpoint.alias} does not)"
                )
            required_widths = (8,) if selection.speculation_study else selection.selected_verification_widths
            if not selection.prefill_chunk and (
                self.speculation is None or not set(required_widths) <= set(self.speculation.widths)
            ):
                raise UsageError("benchmark rounds require the selected speculative width")
            if selection.max_batch > self.concurrency:
                raise UsageError("benchmark batches exceed --max-concurrency")
            if not selection.prefill_chunk and selection.max_batch * selection.width > self.speculative_token_capacity:
                raise UsageError("benchmark rounds exceed the speculative token capacity")
            if selection.required_context > self.context:
                raise UsageError("benchmark workload and output allowance exceed the prepared context")
            if selection.prefill_chunk and selection.prefill_chunk != self.prefill_chunk:
                raise UsageError("benchmark chunk must match --prefill-chunk-size")
            if self.prompt_logprobs or self.mixed_prefill:
                raise UsageError("benchmark graph preparation does not implement scored or mixed prefill")

    @property
    def world_size(self) -> int:
        return len(self.gpus)

    @property
    def parameter_context(self) -> ParameterContext:
        return ParameterContext(
            recipe=self.recipe,
            checkpoint=self.checkpoint,
            root=self.parameters,
            defaults=self.recipe.parameters,
            gpus=self.gpus,
            world_size=self.world_size,
        )

    @property
    def concurrency(self) -> int:
        return self.parameters.max_concurrency

    @property
    def prefill_chunk(self) -> int:
        return self.parameters.prefill_chunk

    @property
    def speculation(self) -> SpeculationParameters | None:
        return self.parameters.speculation

    @property
    def reserve_bytes(self) -> int:
        return self.parameters.reserve_bytes

    @property
    def compile_workers(self) -> int | None:
        return self.parameters.compile_workers

    @property
    def autotune_effort(self) -> str:
        return self.parameters.autotune_effort

    @property
    def lock_gpu_clocks(self) -> bool:
        return self.parameters.lock_gpu_clocks

    @property
    def roce_ring(self) -> bool:
        return self.parameters.roce_ring

    @property
    def roce_chain(self) -> bool:
        return self.parameters.roce_chain

    @property
    def prompt_logprobs(self) -> bool:
        return bool(self.parameters.prompt_logprobs and self.parameters.prompt_logprobs.enabled)

    @property
    def mixed_prefill(self) -> bool:
        return bool(self.parameters.mixed_prefill and self.parameters.mixed_prefill.enabled)

    @property
    def mixed_shapes(self):
        if self.parameters.mixed_prefill is None:
            return ()
        return self.parameters.mixed_prefill.shapes(self.concurrency, self.prefill_chunk, self.speculation)

    @property
    def language_model_only(self) -> bool:
        return bool(self.parameters.vision and self.parameters.vision.language_model_only)

    @property
    def image_concurrency(self) -> int:
        if self.parameters.vision is None:
            return 0
        return self.parameters.vision.effective_image_concurrency(self.concurrency)

    @property
    def max_images(self) -> int:
        return 0 if self.language_model_only else self.recipe.max_images

    @property
    def speculative_token_capacity(self) -> int:
        return 0 if self.speculation is None else self.speculation.effective_token_capacity(self.concurrency)

    @property
    def costs(self):
        base = self.recipe.costs(self.world_size, self.speculation, self.checkpoint)
        return self.parameters.adjust_rank_costs(base, self.parameter_context)

    @property
    def live_rows(self) -> int:
        return self.recipe.live_rows(self.concurrency, self.speculation)

    @property
    def live_bytes(self) -> int:
        return self.live_rows * self.costs.state_row_bytes

    @property
    def fixed_bytes(self) -> int:
        return self.weights_bytes + self.graph_bytes + self.live_bytes

    @property
    def demand_pages(self) -> int:
        return self.concurrency * (self.recipe.context // self.page_tokens)

    @property
    def page_tokens(self) -> int:
        return self.parameters.adjust_page_tokens(self.recipe.page_tokens, self.parameter_context)

    @property
    def minimum_pages(self) -> int:
        return self.concurrency * (MINIMUM_REQUEST_TOKENS // self.page_tokens)

    @property
    def free_bytes(self) -> int:
        return min(gpu.free_bytes for gpu in self.gpus)

    @property
    def headroom_bytes(self) -> int:
        base = HEADROOM_BYTES + max(0, self.prefill_chunk - HEADROOM_CHUNK_TOKENS) * HEADROOM_CHUNK_TOKEN_BYTES
        return self.parameters.adjust_headroom_bytes(base, self.parameter_context)

    @property
    def supply_bytes(self) -> int:
        return self.free_bytes - self.headroom_bytes - self.reserve_bytes

    @property
    def page_bytes(self) -> int:
        return self.pages * self.costs.page_bytes

    @property
    def tokens(self) -> int:
        return self.pages * self.page_tokens

    @property
    def context(self) -> int:
        """The longest sequence one request may reach."""
        return min(self.recipe.context, self.tokens)

    @property
    def guaranteed_tokens(self) -> int:
        """Tokens every request can hold while all admitted requests are active."""
        return min(self.recipe.context, self.pages // self.concurrency * self.page_tokens)

    @property
    def snapshots(self) -> int:
        return self.cache_states

    @property
    def fits(self) -> bool:
        return (self.pages >= self.minimum_pages
                and self.fixed_bytes + self.page_bytes <= self.supply_bytes)

    def to_dict(self) -> dict:
        param_dict = self.parameters.to_dict(self.parameter_context)
        return {
            "alias": self.checkpoint.alias,
            "repository": self.checkpoint.repository,
            "revision": self.checkpoint.revision,
            "architecture": self.checkpoint.architecture,
            "gpus": [
                {"index": gpu.index, "name": gpu.name, "uuid": gpu.uuid,
                 "free_bytes": gpu.free_bytes, "total_bytes": gpu.total_bytes}
                for gpu in self.gpus
            ],
            "tensor_parallel": self.world_size,
            **param_dict,
            "tuning_gpus": [gpu.index for gpu in self.tuning_gpus],
            "benchmark_selection": None if self.benchmark_selection is None else self.benchmark_selection.to_dict(),
            "context": self.context,
            "checkpoint_context": self.recipe.context,
            "weights_bytes": self.weights_bytes,
            "mapped_host_bytes": self.checkpoint.mapped_host_bytes,
            "graph_bytes": self.graph_bytes,
            "mixed_shapes": self.mixed_shapes,
            "live_bytes": self.live_bytes,
            "prefix_cache_states": self.cache_states,
            "prefix_cache_target": self.cache_target,
            "snapshot_virtual_bytes": self.snapshots * self.costs.state_row_bytes,
            "snapshots": self.snapshots,
            "pages": self.pages,
            "demand_pages": self.demand_pages,
            "page_tokens": self.page_tokens,
            "page_bytes": self.page_bytes,
            "tokens": self.tokens,
            "guaranteed_tokens": self.guaranteed_tokens,
            "free_bytes": self.free_bytes,
            "headroom_bytes": self.headroom_bytes,
            "reserve_bytes": self.reserve_bytes,
            "supply_bytes": self.supply_bytes,
        }


def _normalize_parameters(recipe: Recipe, parameters: RecipeParameters | None,
                          reserve_bytes: int | None) -> RecipeParameters:
    params = recipe.parameters if parameters is None else parameters
    if reserve_bytes is not None:
        params = replace(params, reserve_bytes=reserve_bytes)
    if (params.speculation is not None and params.speculation.packed_verification
            and params.speculation.draft_tokens > 0
            and params.speculation.graph_budget is None
            and recipe.parameters.speculation is not None):
        params = replace(
            params,
            speculation=replace(params.speculation, graph_budget=recipe.parameters.speculation.graph_budget),
        )
    return params


def fit(recipe: Recipe, checkpoint: Checkpoint, gpus: tuple[Gpu, ...], *,
        parameters: RecipeParameters | None = None,
        reserve_bytes: int | None = None,
        cache_states: int | None = None,
        explicit_flags: frozenset[str] = frozenset()) -> Plan:
    """Size pages and the prefix cache to the supply; the result may not fit.

    Weights come from the checkpoint's stored tensors and graphs and scratch from
    the recipe's estimate, adjusted through `parameters` hooks.
    """
    params = _normalize_parameters(recipe, parameters, reserve_bytes)
    params = replace(params, tensor_parallel=len(gpus))
    context = ParameterContext(
        recipe=recipe,
        checkpoint=checkpoint,
        root=params,
        defaults=recipe.parameters,
        gpus=gpus,
        world_size=len(gpus),
        explicit_flags=explicit_flags,
    )
    params.validate(context)
    base_costs = recipe.costs(len(gpus), params.speculation, checkpoint)
    costs = params.adjust_rank_costs(base_costs, context)
    weights = recipe.weights_bytes(checkpoint, len(gpus), params)
    plan = Plan(
        recipe=recipe,
        checkpoint=checkpoint,
        gpus=gpus,
        parameters=params,
        cache_target=0,
        cache_states=0,
        pages=0,
        weights_bytes=weights,
        graph_bytes=costs.graph_bytes,
    )
    available = plan.supply_bytes - plan.fixed_bytes
    target = min(CACHE_STATES_PER_REQUEST * plan.concurrency, max(1, available // costs.state_row_bytes))
    if cache_states is not None:
        target = cache_states
    return replace(
        plan,
        cache_target=target,
        cache_states=target,
        pages=max(available // costs.page_bytes, plan.minimum_pages),
    )


def _gib(value: int) -> str:
    return f"{value / _GIB:.1f}"


def _per_gpu(plan: Plan) -> str:
    return " per GPU" if plan.world_size > 1 else ""


def _shortfall(plan: Plan, options: list[str], scope: str) -> CapacityError:
    need = plan.fixed_bytes + plan.page_bytes
    lines = [
        f"needs {_gib(plan.weights_bytes)} GiB weights + {_gib(plan.graph_bytes)} GiB graphs"
        f" + {_gib(plan.live_bytes)} GiB live state",
        f"+ {_gib(plan.page_bytes)} GiB minimum pages ({plan.concurrency} × {MINIMUM_REQUEST_TOKENS:,} tokens)"
        f" = {_gib(need)} GiB{_per_gpu(plan)};",
        f"has {_gib(plan.free_bytes)} GiB free − {_gib(plan.headroom_bytes)} GiB headroom"
        f" − {_gib(plan.reserve_bytes)} GiB reserve = {_gib(plan.supply_bytes)} GiB.",
        f"Options: {', or '.join(options)}." if options
        else f"No setting makes {plan.checkpoint.alias} fit on {scope}.",
    ]
    return CapacityError("\n".join(lines))


def host_shape_error(roce_ring: bool, roce_chain: bool, hosts: int | None) -> str | None:
    """Why a ring or chain cannot be formed over ``hosts`` ranks on separate hosts, or None when it can.

    A ring or a chain needs 2 to 16 ranks in rank order on separate hosts; the two shapes exclude each
    other. ``hosts`` is the rank count, or None when ranks share one host."""
    if roce_ring and roce_chain:
        return "--roce-ring and --roce-chain exclude each other"
    if not (roce_ring or roce_chain):
        return None
    flag = "--roce-ring" if roce_ring else "--roce-chain"
    if hosts is None or not 2 <= hosts <= 16:
        return f"{flag} requires at least two hosts in rank order, at most 16, each rank on its own host"
    return None


def plan(recipe: Recipe, checkpoint: Checkpoint, gpus: tuple[Gpu, ...], *,
         selected: tuple[int, ...] | None = None,
         parameters: RecipeParameters | None = None,
         reserve_bytes: int | None = None,
         cache_states: int | None = None,
         explicit_flags: frozenset[str] = frozenset()) -> Plan:
    """Select GPUs and size the plan, or raise a capacity error that lists working settings.

    Without `selected`, the plan uses the smallest supported tensor-parallel size
    for which enough GPUs fit, taking the first such GPUs in index order.
    """
    params = _normalize_parameters(recipe, parameters, reserve_bytes)
    pre_context = ParameterContext(
        recipe=recipe,
        checkpoint=checkpoint,
        root=params,
        defaults=recipe.parameters,
        explicit_flags=explicit_flags,
    )
    params.validate(pre_context)

    world_sizes = recipe.checkpoint_tensor_parallel(checkpoint)
    plan_sizes = params.filter_world_sizes(world_sizes, pre_context)
    if not plan_sizes:
        raise UsageError(f"no supported tensor-parallel size remains for {checkpoint.alias} with the selected parameters")
    sizes = " or ".join(map(str, plan_sizes))
    supported = [gpu for gpu in gpus if gpu.supported]
    chosen = None
    if selected is not None:
        if any(index >= len(gpus) for index in selected):
            raise UsageError(f"--gpus {format_gpus(selected)} names a GPU this host does not have;"
                             f" it has {len(gpus)}")
        chosen = tuple(gpus[index] for index in selected)
        if len(chosen) not in plan_sizes:
            # Let parameter validation surface explicit UsageErrors (such as TP restrictions)
            # before falling back to a generic capacity/GPU-count error.
            params.validate(replace(pre_context, gpus=chosen, world_size=len(chosen)))
            size = min(plan_sizes, key=lambda size: (size < len(chosen), size))
            first = max(0, min(selected[0], len(gpus) - size))
            example = format_gpus(range(first, first + size))
            raise CapacityError(f"{checkpoint.alias} needs {sizes} GPUs; {len(chosen)} selected; use --gpus {example}")
        for gpu in chosen:
            if not gpu.supported:
                raise CapacityError(f"GPU {gpu.index} ({gpu.name}) is not an {SUPPORTED_ARCHITECTURES} device")
    elif len(supported) < plan_sizes[0]:
        raise CapacityError(f"{checkpoint.alias} needs {sizes} {SUPPORTED_ARCHITECTURES} GPUs; this host has {len(supported)}")

    def fitted(group: tuple[Gpu, ...], candidate_params: RecipeParameters = params, *,
               flags: frozenset[str] = explicit_flags) -> Plan:
        return fit(
            recipe, checkpoint, group, parameters=candidate_params, cache_states=cache_states, explicit_flags=flags,
        )

    def automatic(world_size: int, candidate_params: RecipeParameters = params, *,
                  flags: frozenset[str] = explicit_flags) -> Plan:
        """Plan the first GPUs that each fit, else the GPUs with the most free memory."""
        fitting = [gpu for gpu in supported if fitted((gpu,) * world_size, candidate_params, flags=flags).fits]
        if len(fitting) < world_size:
            fitting = sorted(sorted(supported, key=lambda gpu: -gpu.free_bytes)[:world_size],
                             key=lambda gpu: gpu.index)
        return fitted(tuple(fitting[:world_size]), candidate_params, flags=flags)

    def candidates(candidate_params: RecipeParameters = params, *, flags: frozenset[str] = explicit_flags) -> list[Plan]:
        if chosen is not None:
            return [fitted(chosen, candidate_params, flags=flags)]
        candidate_sizes = candidate_params.filter_world_sizes(world_sizes, replace(pre_context, root=candidate_params))
        return [automatic(size, candidate_params, flags=flags) for size in candidate_sizes if size <= len(supported)]

    def attempt(candidate_params: RecipeParameters = params, *, flags: frozenset[str] = frozenset()) -> Plan | None:
        return next((candidate for candidate in candidates(candidate_params, flags=flags) if candidate.fits), None)

    result = attempt(params, flags=explicit_flags)
    if result is not None:
        if params.prefill_chunk > result.context:
            raise UsageError(f"--prefill-chunk-size {params.prefill_chunk:,} exceeds the {result.context:,}-token context"
                             " this plan allows")
        return result

    closest = min(candidates(params, flags=explicit_flags), key=lambda candidate: (
        candidate.fixed_bytes + candidate.page_bytes - candidate.supply_bytes))
    options: list[str] = []
    if params.reserve_bytes > 0:
        for reserve in range((params.reserve_bytes - 1) // _GIB, -1, -1):
            if attempt(replace(params, reserve_bytes=reserve * _GIB)) is not None:
                options.append(f'[memory] reserve = "{reserve}G"' if reserve else '[memory] reserve = "0"')
                break
    relief_context = ParameterContext(
        recipe=recipe,
        checkpoint=checkpoint,
        root=params,
        defaults=recipe.parameters,
        gpus=closest.gpus,
        world_size=closest.world_size,
        explicit_flags=explicit_flags,
    )
    options.extend(params.capacity_relief_options(attempt, relief_context))
    if chosen is not None:
        for size in plan_sizes:
            if len(chosen) < size <= len(supported) and (larger := automatic(size)).fits:
                options.append(f"--gpus {format_gpus(gpu.index for gpu in larger.gpus)}")
                break
    scope = "this host" if chosen is None else f"GPU {format_gpus(gpu.index for gpu in chosen)}"
    raise _shortfall(closest, options, scope)
