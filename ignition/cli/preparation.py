# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Model resolution, parameter resolution, memory planning, and downloads for preparation commands."""

import os
from dataclasses import dataclass, replace

from ignition import planning
from ignition.errors import FetchError, UsageError
from ignition.parameters import ParameterContext, RecipeParameters, explicit_flags_from_args
from ignition.registry import ResolvedModel, resolve

from .config import Configuration


@dataclass(frozen=True)
class Host:
    """GPUs as the command line indexes them, and the caller's restriction on them."""

    gpus: tuple[planning.Gpu, ...]
    restriction: str | None

    @classmethod
    def discover(cls) -> "Host":
        gpus = planning.discover_gpus()
        # The one environment variable Flare inspects: a caller's restriction
        # changes which GPU each index names, and the plan reports that.
        restriction = os.environ.get("CUDA_VISIBLE_DEVICES")
        if restriction is not None:
            gpus = planning.restrict_gpus(gpus, restriction)
        return cls(gpus, restriction)


def resolve_model(configuration: Configuration, name: str | None, *, recipes=None) -> ResolvedModel:
    name = configuration.model if name is None else name
    if name is None:
        raise UsageError("no MODEL given and defaults.model is not set in the configuration file")
    model = resolve(name, configuration.directory("hub_cache"), recipes=recipes)
    model.verify(configuration.directory("hub_cache"))
    return model


def resolve_parameters(model: ResolvedModel, args, configuration: Configuration | None = None, *,
                       multi_host: bool = False) -> RecipeParameters:
    """Resolve `model.recipe.parameters` from CLI arguments and static host configuration."""
    context = ParameterContext(
        recipe=model.recipe,
        checkpoint=model.checkpoint,
        root=model.recipe.parameters,
        defaults=model.recipe.parameters,
        explicit_flags=explicit_flags_from_args(args),
    )
    resolved = model.recipe.parameters.resolve(args, configuration, model.checkpoint.alias, context)
    if multi_host != resolved.multi_host:
        resolved = replace(resolved, multi_host=multi_host)
    resolved.validate(replace(context, root=resolved))
    return resolved


def derive(configuration: Configuration, model: ResolvedModel, host: Host, *,
           gpus: str | None = None,
           parameters: RecipeParameters | None = None,
           tuning_gpus: str | None = None,
           cache_states: int | None = None,
           explicit_flags: frozenset[str] = frozenset()) -> planning.Plan:
    """Select GPUs and estimate memory for one checkpoint from configuration and resolved parameters."""
    params = model.recipe.parameters if parameters is None else parameters
    result = planning.plan(
        model.recipe,
        model.checkpoint,
        host.gpus,
        selected=None if gpus is None else planning.parse_gpus(gpus),
        parameters=params,
        reserve_bytes=configuration.reserve_bytes,
        cache_states=cache_states,
        explicit_flags=explicit_flags,
    )
    result = replace(result, hub_cache=configuration.directory("hub_cache"))
    return tuning_devices(result, host, tuning_gpus)


def tuning_devices(plan: planning.Plan, host: Host, flag: str | None) -> planning.Plan:
    if plan.compile_workers is not None and plan.compile_workers < plan.world_size:
        raise UsageError("--compile-workers must provide at least one process per local serving rank")
    if flag is None:
        return plan
    indices = planning.parse_gpus(flag)
    available = {gpu.index: gpu for gpu in host.gpus}
    if any(index not in available for index in indices):
        raise UsageError("--tuning-gpus names a GPU outside the visible device list")
    count = plan.world_size
    if indices[:count] != tuple(gpu.index for gpu in plan.gpus) or len(indices) % count:
        raise UsageError("--tuning-gpus must start with the serving GPUs and contain complete TP replicas")
    if plan.compile_workers is not None and plan.compile_workers < len(indices):
        raise UsageError("--compile-workers must provide at least one process per local tuning rank")
    selected = tuple(available[index] for index in indices)
    for position, gpu in enumerate(selected[count:], start=count):
        owner = plan.gpus[position % count]
        if (gpu.name, gpu.capability) != (owner.name, owner.capability):
            raise UsageError("each tuning replica must match its corresponding serving GPU model and architecture")
        minimum = (plan.fixed_bytes + plan.headroom_bytes + plan.reserve_bytes
                   + plan.minimum_pages * plan.costs.page_bytes)
        if gpu.free_bytes < minimum:
            raise UsageError(f"tuning GPU {gpu.index} needs at least {minimum >> 20} MiB free for a model replica")
    return replace(plan, tuning_gpus=selected if len(selected) > count else ())


def fetch(model: ResolvedModel, hub_cache, *, progress_bars: bool = True) -> str:
    """Download the pinned revision's missing files and return the snapshot directory."""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        if not progress_bars:
            disable_progress_bars()
        return snapshot_download(model.checkpoint.repository, revision=model.checkpoint.revision,
                                 cache_dir=hub_cache)
    except Exception as error:  # noqa: BLE001 - Network, authorization, and disk failures share one exit code.
        raise FetchError(f"cannot download {model.checkpoint.repository} at revision"
                         f" {model.checkpoint.revision[:8]}: {error}") from error
