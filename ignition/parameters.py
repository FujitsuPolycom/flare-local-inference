# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""First-class, compositional recipe parameters, dynamic CLI help, and manifest serialization.

Core Ignition owns only universal preparation parameters and the four Helios
GraphOperation ABI opt-in groups (`SpeculationParameters`, `VisionParameters`,
`PromptLogprobsParameters`, `MixedPrefillParameters`). Model recipes compose
these groups with recipe-defined `ParameterGroup` classes and subclasses of
`RecipeParameters`.

Every `ParameterGroup` instance is an immutable, picklable dataclass holding
only plain values and composed sub-groups. Parameter descriptors (`ParamSpec`)
live in dataclass field metadata so instances serialize cleanly across local
worker processes and multi-host rank handoffs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from enum import Enum
import re
from typing import Any, ClassVar

from ignition.errors import UsageError

AUTOTUNE_EFFORTS = ("off", "default", "audit")
EXPLICIT_FLAGS_ATTR = "_explicit_flags"
_PARAM_META = "ignition_param_spec"
_SPECULATIVE_METHOD_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")


def valid_speculative_method_name(value: object) -> bool:
    """True when `value` is a non-empty lowercase CLI identifier for a recipe speculative method."""
    return isinstance(value, str) and bool(_SPECULATIVE_METHOD_NAME.fullmatch(value))


class _UnsetType:
    def __repr__(self) -> str:
        return "<UNSET>"


_UNSET = _UnsetType()


class ParameterScope(Enum):
    """Compatibility scope recorded in `PreparationManifest` entries."""

    NONE = None
    DEPLOYMENT = "PREPARATION_PARAMETER_SCOPE_DEPLOYMENT"
    COMBINE = "PREPARATION_PARAMETER_SCOPE_COMBINE"
    TARGET_WEIGHT = "PREPARATION_PARAMETER_SCOPE_TARGET_WEIGHT"
    DRAFT_WEIGHT = "PREPARATION_PARAMETER_SCOPE_DRAFT_WEIGHT"


@dataclass(frozen=True)
class OptionArity:
    """How many following tokens a CLI option may consume during Stage 1 pre-scanning."""

    min_args: int = 1
    max_args: int = 1
    optional_choices: frozenset[str] | None = None


@dataclass(frozen=True)
class ParamSpec:
    """Declarative metadata attached to a `ParameterGroup` dataclass field."""

    flag: str | None = None
    cli: bool = True
    dest: str | None = None
    group: str | None = None
    help: str = ""
    metavar: str | None = None
    choices: tuple[Any, ...] | None = None
    choices_attr: str | None = None
    choices_fn: Callable[[Any, ParameterContext], tuple[Any, ...]] | None = None
    enabled_attr: str | None = None
    enabled_fn: Callable[[Any, ParameterContext], bool] | None = None
    min_value: int | float | None = None
    min_attr: str | None = None
    max_value: int | float | None = None
    max_attr: str | None = None
    max_fn: Callable[[Any, ParameterContext], int | float | None] | None = None
    multiple_of: int | None = None
    negated_flag: str | None = None
    boolean_optional: bool = False
    const: Any = _UNSET
    native_value: str | None = None
    value_type: type | None = None
    config_key: str | None = None
    model_config_key: str | None = None
    scope: ParameterScope = ParameterScope.DEPLOYMENT
    manifest_key: str | None = None
    omit_none: bool = True
    repro_always: bool = False
    repro_flag: str | None = None
    requires_speculation: bool = False
    default_label: str | None = None
    bare_default_label: str | None = None
    help_extra: Callable[[Any, ParameterContext], str | None] | None = None
    status_label: Callable[[Any], str | None] | None = None
    multi_host_only: bool = False
    in_dict: bool = True
    dict_key: str | None = None


def param(default: Any = _UNSET, *, default_factory: Any = _UNSET, **spec_kwargs: Any) -> Any:
    """Declare a dataclass field backed by a `ParamSpec`."""
    spec = ParamSpec(**spec_kwargs)
    kwargs: dict[str, Any] = {"metadata": {_PARAM_META: spec}}
    if default is not _UNSET:
        kwargs["default"] = default
    elif default_factory is not _UNSET:
        kwargs["default_factory"] = default_factory
    return field(**kwargs)


@dataclass(frozen=True)
class ParameterContext:
    """Context supplied to parameter registration, resolution, planning, and serialization."""

    recipe: Any = None
    checkpoint: Any = None
    root: RecipeParameters | None = None
    defaults: RecipeParameters | None = None
    gpus: tuple[Any, ...] = ()
    world_size: int | None = None
    explicit_flags: frozenset[str] = frozenset()
    command: str | None = None

    @property
    def alias(self) -> str:
        if self.checkpoint is not None:
            return self.checkpoint.alias
        if self.recipe is not None:
            return self.recipe.name
        return "model"


def _record_explicit(namespace: argparse.Namespace, option_string: str | None, dest: str) -> None:
    recorded = getattr(namespace, EXPLICIT_FLAGS_ATTR, None)
    if recorded is None:
        recorded = {}
        setattr(namespace, EXPLICIT_FLAGS_ATTR, recorded)
    recorded[dest] = option_string or dest


class _RecordingStoreAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        _record_explicit(namespace, option_string, self.dest)


class _RecordingStoreTrueAction(argparse.Action):
    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=True,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        _record_explicit(namespace, option_string, self.dest)


class _RecordingBooleanOptionalAction(argparse.BooleanOptionalAction):
    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, values, option_string=option_string)
        _record_explicit(namespace, option_string, self.dest)


def explicit_flags_from_args(args: Any) -> frozenset[str]:
    """Return the set of CLI flag strings explicitly passed on `args`."""
    recorded = getattr(args, EXPLICIT_FLAGS_ATTR, None)
    if isinstance(recorded, dict):
        return frozenset(recorded.values())
    return frozenset()


def _param_spec(dc_field) -> ParamSpec | None:
    return dc_field.metadata.get(_PARAM_META)


def _flag_name(dc_field, spec: ParamSpec) -> str | None:
    if not spec.cli:
        return None
    if spec.negated_flag is not None:
        return spec.negated_flag
    if spec.flag is not None:
        return spec.flag
    return "--" + dc_field.name.replace("_", "-")


def _dest_name(dc_field, spec: ParamSpec) -> str:
    if spec.dest is not None:
        return spec.dest
    flag = _flag_name(dc_field, spec)
    if flag is not None:
        return flag.lstrip("-").replace("-", "_")
    return dc_field.name


def _value_type(spec: ParamSpec, default_value: Any) -> type:
    if spec.value_type is not None:
        return spec.value_type
    if spec.negated_flag is not None or spec.boolean_optional:
        return bool
    if isinstance(default_value, bool):
        return bool
    if isinstance(default_value, int):
        return int
    if isinstance(default_value, float):
        return float
    return str


def _is_enabled(group: ParameterGroup, spec: ParamSpec, context: ParameterContext) -> bool:
    if spec.enabled_attr is not None and not getattr(group, spec.enabled_attr):
        return False
    if spec.enabled_fn is not None and not spec.enabled_fn(group, context):
        return False
    return True


def _resolved_choices(group: ParameterGroup, spec: ParamSpec, context: ParameterContext) -> tuple[Any, ...] | None:
    if spec.choices_fn is not None:
        return tuple(spec.choices_fn(group, context))
    if spec.choices_attr is not None:
        return tuple(getattr(group, spec.choices_attr))
    return spec.choices


def _resolved_min(group: ParameterGroup, spec: ParamSpec) -> int | float | None:
    if spec.min_attr is not None:
        return getattr(group, spec.min_attr)
    return spec.min_value


def _resolved_max(group: ParameterGroup, spec: ParamSpec, context: ParameterContext) -> int | float | None:
    if spec.max_fn is not None:
        return spec.max_fn(group, context)
    if spec.max_attr is not None:
        return getattr(group, spec.max_attr)
    return spec.max_value


def _get_or_create_group(parser: argparse.ArgumentParser, title: str | None,
                         description: str | None = None):
    if title is None:
        return parser
    for existing in parser._action_groups:
        if existing.title == title:
            return existing
    return parser.add_argument_group(title, description)


def _format_number(value: int | float) -> str:
    if isinstance(value, int) and value >= 10000:
        return f"{value:,}"
    return str(value)


def _format_help(group: ParameterGroup, dc_field, spec: ParamSpec, context: ParameterContext) -> str:
    default_value = getattr(group, dc_field.name)
    vtype = _value_type(spec, default_value)
    details: list[str] = []

    if context.recipe is None and spec.bare_default_label is not None:
        if spec.multiple_of is not None:
            details.append(f"multiple of {spec.multiple_of}")
        details.append(f"default: {spec.bare_default_label}")
    else:
        min_val = _resolved_min(group, spec)
        max_val = _resolved_max(group, spec, context)
        if min_val is not None and max_val is not None:
            details.append(f"{_format_number(min_val)}..{_format_number(max_val)}")
        elif min_val is not None:
            details.append(f">= {_format_number(min_val)}")
        elif max_val is not None:
            details.append(f"<= {_format_number(max_val)}")
        if spec.multiple_of is not None:
            details.append(f"multiple of {spec.multiple_of}")

        choices = _resolved_choices(group, spec, context)
        if choices and (spec.const is not _UNSET or spec.metavar is not None):
            details.append("choices: " + ", ".join(map(str, choices)))

        if spec.default_label is not None:
            details.append(f"default: {spec.default_label}")
        elif spec.negated_flag is not None:
            details.append("default: enabled" if default_value else "default: disabled")
        elif vtype is bool:
            details.append("default: on" if default_value else "default: off")
        elif default_value is None:
            details.append(f"default: {spec.native_value}" if spec.native_value else "default: none")
        else:
            details.append(f"default: {default_value}")

    if spec.help_extra is not None:
        extra = spec.help_extra(group, context)
        if extra:
            details.append(extra)

    base = spec.help.rstrip(".")
    if not details:
        rendered = base
    else:
        rendered = f"{base} ({'; '.join(details)})" if base else "; ".join(details)
    return rendered.replace("%", "%%")


def _is_arg_explicit(args: Any, dest: str, *, is_store_true: bool) -> bool:
    recorded = getattr(args, EXPLICIT_FLAGS_ATTR, None)
    if isinstance(recorded, dict):
        return dest in recorded
    if not hasattr(args, dest):
        return False
    value = getattr(args, dest)
    return bool(value) if is_store_true else value is not None


@dataclass(frozen=True)
class ParameterGroup:
    """Composable base class for cohesive parameter families.

    Fields carrying `ParamSpec` metadata act as declared parameters. Fields
    holding nested `ParameterGroup` instances are composed recursively. Plain
    fields carry recipe capability constraints (such as bounds or supported
    option tuples) and are never emitted as CLI flags or manifest entries.
    """

    manifest_prefix: ClassVar[str | None] = None
    dict_section: ClassVar[str | None] = None
    default_group: ClassVar[str | None] = None
    group_description: ClassVar[str | None] = None

    def _param_fields(self):
        for dc_field in fields(self):
            spec = _param_spec(dc_field)
            if spec is not None:
                yield dc_field, spec

    def _subgroups(self):
        for dc_field in fields(self):
            if _param_spec(dc_field) is not None:
                continue
            value = getattr(self, dc_field.name)
            if isinstance(value, ParameterGroup):
                yield dc_field, value

    def add_arguments(self, parser: argparse.ArgumentParser, context: ParameterContext, *,
                      include_multi_host: bool = True) -> None:
        """Register CLI flags for this group and all active composed sub-groups."""
        for dc_field, spec in self._param_fields():
            if not spec.cli:
                continue
            if spec.multi_host_only and not include_multi_host:
                continue
            if not _is_enabled(self, spec, context):
                continue
            flag = _flag_name(dc_field, spec)
            assert flag is not None
            dest = _dest_name(dc_field, spec)
            group_title = spec.group if spec.group is not None else self.default_group
            target = _get_or_create_group(parser, group_title, self.group_description)
            default_value = getattr(self, dc_field.name)
            vtype = _value_type(spec, default_value)
            help_text = _format_help(self, dc_field, spec, context)

            if spec.boolean_optional:
                target.add_argument(
                    flag, dest=dest, action=_RecordingBooleanOptionalAction, default=None, help=help_text,
                )
            elif spec.negated_flag is not None or vtype is bool:
                target.add_argument(
                    flag, dest=dest, action=_RecordingStoreTrueAction, default=False, help=help_text,
                )
            else:
                kwargs: dict[str, Any] = {
                    "dest": dest,
                    "action": _RecordingStoreAction,
                    "default": None,
                    "help": help_text,
                }
                if vtype is not str:
                    kwargs["type"] = vtype
                if spec.metavar is not None:
                    kwargs["metavar"] = spec.metavar
                choices = _resolved_choices(self, spec, context)
                if choices is not None:
                    kwargs["choices"] = choices
                if spec.const is not _UNSET:
                    kwargs["nargs"] = "?"
                    kwargs["const"] = spec.const
                target.add_argument(flag, **kwargs)

        for _, subgroup in self._subgroups():
            subgroup.add_arguments(parser, context, include_multi_host=include_multi_host)

    def option_arities(self, context: ParameterContext, *, include_multi_host: bool = True) -> dict[str, OptionArity]:
        """Return CLI flag arities for Stage 1 token pre-scanning."""
        arities: dict[str, OptionArity] = {}
        for dc_field, spec in self._param_fields():
            if not spec.cli:
                continue
            if spec.multi_host_only and not include_multi_host:
                continue
            if not _is_enabled(self, spec, context):
                continue
            flag = _flag_name(dc_field, spec)
            assert flag is not None
            default_value = getattr(self, dc_field.name)
            vtype = _value_type(spec, default_value)
            if spec.boolean_optional:
                arities[flag] = OptionArity(0, 0)
                arities["--no-" + flag.removeprefix("--")] = OptionArity(0, 0)
            elif spec.negated_flag is not None or vtype is bool:
                arities[flag] = OptionArity(0, 0)
            elif spec.const is not _UNSET:
                choices = _resolved_choices(self, spec, context) or ()
                arities[flag] = OptionArity(0, 1, frozenset(map(str, choices)))
            else:
                arities[flag] = OptionArity(1, 1)

        for _, subgroup in self._subgroups():
            arities.update(subgroup.option_arities(context, include_multi_host=include_multi_host))
        return arities

    def explicit_speculation_flags(self, args: Any, context: ParameterContext) -> list[str]:
        """Return CLI flags explicitly passed on `args` whose `ParamSpec` requires speculation."""
        given: list[str] = []
        for dc_field, spec in self._param_fields():
            if not spec.cli or not spec.requires_speculation:
                continue
            if not _is_enabled(self, spec, context):
                continue
            flag = _flag_name(dc_field, spec)
            dest = _dest_name(dc_field, spec)
            default_value = getattr(self, dc_field.name)
            vtype = _value_type(spec, default_value)
            is_store_true = (spec.negated_flag is not None or vtype is bool) and not spec.boolean_optional
            if flag is not None and _is_arg_explicit(args, dest, is_store_true=is_store_true):
                given.append(flag)
        for _, subgroup in self._subgroups():
            given.extend(subgroup.explicit_speculation_flags(args, context))
        return given

    def _resolve_fields(self, args: Any, configuration: Any, alias: str,
                        context: ParameterContext) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        for dc_field, spec in self._param_fields():
            if not spec.cli:
                continue
            if not _is_enabled(self, spec, context):
                continue
            dest = _dest_name(dc_field, spec)
            default_value = getattr(self, dc_field.name)
            vtype = _value_type(spec, default_value)
            is_store_true = (spec.negated_flag is not None or vtype is bool) and not spec.boolean_optional
            explicit = _is_arg_explicit(args, dest, is_store_true=is_store_true)

            if explicit:
                raw = getattr(args, dest)
                if spec.negated_flag is not None:
                    value = not bool(raw)
                elif spec.boolean_optional:
                    value = bool(raw)
                elif vtype is bool:
                    value = bool(raw)
                elif spec.native_value is not None and raw == spec.native_value:
                    value = None
                else:
                    value = raw
                updates[dc_field.name] = value
            elif configuration is not None and spec.model_config_key is not None:
                setting = configuration.max_concurrency(alias, default_value)
                updates[dc_field.name] = setting.value
            elif configuration is not None and spec.config_key is not None:
                setting = configuration.get(spec.config_key)
                if setting.source != "default":
                    updates[dc_field.name] = setting.value
        return updates

    def resolve(self, args: Any, configuration: Any, alias: str,
                context: ParameterContext) -> ParameterGroup:
        """Resolve CLI flags and static host configuration over recipe defaults."""
        updates = self._resolve_fields(args, configuration, alias, context)
        for dc_field, subgroup in self._subgroups():
            updates[dc_field.name] = subgroup.resolve(args, configuration, alias, context)
        return replace(self, **updates) if updates else self

    def reset_speculation_fields(self) -> ParameterGroup:
        """Return a copy with any `requires_speculation` fields reset to their field defaults."""
        updates: dict[str, Any] = {}
        for dc_field, spec in self._param_fields():
            if spec.requires_speculation:
                if dc_field.default is not _UNSET and dc_field.default is not argparse.SUPPRESS:
                    updates[dc_field.name] = dc_field.default
        for dc_field, subgroup in self._subgroups():
            updates[dc_field.name] = subgroup.reset_speculation_fields()
        return replace(self, **updates) if updates else self

    def validate_fields(self, context: ParameterContext) -> None:
        """Validate generic `ParamSpec` constraints (choices, bounds, multiples) for this group."""
        alias = context.alias
        for dc_field, spec in self._param_fields():
            if not _is_enabled(self, spec, context):
                continue
            value = getattr(self, dc_field.name)
            if value is None:
                continue
            flag = _flag_name(dc_field, spec) or dc_field.name
            choices = _resolved_choices(self, spec, context)
            if choices is not None:
                allowed = tuple(c for c in choices if c != spec.native_value)
                if value not in allowed:
                    raise UsageError(
                        f"{flag} {value}: supported values for {alias} are {', '.join(map(str, allowed))}"
                    )
            min_val = _resolved_min(self, spec)
            max_val = _resolved_max(self, spec, context)
            if spec.multiple_of is not None:
                if type(value) is not int or value % spec.multiple_of or (
                        min_val is not None and value < min_val) or (
                        max_val is not None and value > max_val):
                    lo = _format_number(min_val) if min_val is not None else str(spec.multiple_of)
                    hi = f" through {_format_number(max_val)}" if max_val is not None else ""
                    raise UsageError(
                        f"{flag} for {alias} must be a multiple of {spec.multiple_of} from {lo}{hi}"
                    )
            elif min_val is not None or max_val is not None:
                if type(value) not in (int, float) or isinstance(value, bool):
                    raise UsageError(f"{flag} must be a number")
                if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                    if min_val is not None and max_val is not None:
                        raise UsageError(
                            f"{flag} for {alias} must be {_format_number(min_val)} through {_format_number(max_val)}"
                        )
                    if min_val == 1:
                        raise UsageError(f"{flag} must be a positive integer")
                    if min_val is not None:
                        raise UsageError(f"{flag} must be at least {_format_number(min_val)}")
                    raise UsageError(f"{flag} must be at most {_format_number(max_val)}")

    def validate(self, context: ParameterContext) -> None:
        """Validate this group and all composed sub-groups."""
        self.validate_fields(context)
        for _, subgroup in self._subgroups():
            subgroup.validate(context)

    def filter_world_sizes(self, world_sizes: tuple[int, ...], context: ParameterContext) -> tuple[int, ...]:
        """Restrict candidate tensor-parallel world sizes during planning."""
        for _, subgroup in self._subgroups():
            world_sizes = subgroup.filter_world_sizes(world_sizes, context)
        return world_sizes

    def adjust_weights_bytes(self, weights_bytes: int, context: ParameterContext) -> int:
        """Adjust estimated per-rank resident device weight bytes."""
        for _, subgroup in self._subgroups():
            weights_bytes = subgroup.adjust_weights_bytes(weights_bytes, context)
        return weights_bytes

    def adjust_rank_costs(self, costs: Any, context: ParameterContext) -> Any:
        """Adjust per-rank `RankCosts` before sizing."""
        for _, subgroup in self._subgroups():
            costs = subgroup.adjust_rank_costs(costs, context)
        return costs

    def adjust_page_tokens(self, page_tokens: int, context: ParameterContext) -> int:
        """Adjust global tokens per KV page."""
        for _, subgroup in self._subgroups():
            page_tokens = subgroup.adjust_page_tokens(page_tokens, context)
        return page_tokens

    def adjust_context(self, context_tokens: int, context: ParameterContext) -> int:
        """Adjust the declared serving context ceiling in tokens before page limits."""
        for _, subgroup in self._subgroups():
            context_tokens = subgroup.adjust_context(context_tokens, context)
        return context_tokens

    def adjust_headroom_bytes(self, headroom_bytes: int, context: ParameterContext) -> int:
        """Adjust transient warm-up/capture headroom bytes."""
        for _, subgroup in self._subgroups():
            headroom_bytes = subgroup.adjust_headroom_bytes(headroom_bytes, context)
        return headroom_bytes

    def capacity_relief_options(self, attempt: Callable[[RecipeParameters], Any],
                                context: ParameterContext) -> list[str]:
        """Suggest CLI options that make an over-capacity plan fit."""
        options: list[str] = []
        for _, subgroup in self._subgroups():
            options.extend(subgroup.capacity_relief_options(attempt, context))
        return options

    def validate_for_benchmark(self, settings: Any, context: ParameterContext) -> None:
        """Validate compatibility with `ignition bench`."""
        for _, subgroup in self._subgroups():
            subgroup.validate_for_benchmark(settings, context)

    def manifest_parameters(self, context: ParameterContext) -> list[tuple[str, ParameterScope, Any]]:
        """Serialize declared parameters as `(key, scope, value)` tuples for `PreparationManifest`."""
        entries: list[tuple[str, ParameterScope, Any]] = []
        for dc_field, spec in self._param_fields():
            if spec.scope is ParameterScope.NONE:
                continue
            if not _is_enabled(self, spec, context):
                continue
            value = getattr(self, dc_field.name)
            if value is None and spec.omit_none:
                continue
            if spec.manifest_key is not None:
                key = spec.manifest_key
            elif self.manifest_prefix:
                key = f"{self.manifest_prefix}.{dc_field.name}"
            else:
                key = dc_field.name
            entries.append((key, spec.scope, value))
        for _, subgroup in self._subgroups():
            entries.extend(subgroup.manifest_parameters(context))
        return entries

    def reproduction_args(self, context: ParameterContext) -> list[str]:
        """Emit canonical CLI flags reproducing this group's resolved values."""
        args: list[str] = []
        defaults_group = self._matching_defaults(context)
        for dc_field, spec in self._param_fields():
            if not spec.cli:
                continue
            if not _is_enabled(self, spec, context):
                continue
            if spec.multi_host_only and (context.root is None or not context.root.multi_host):
                continue
            value = getattr(self, dc_field.name)
            default_value = (
                getattr(defaults_group, dc_field.name)
                if defaults_group is not None and hasattr(defaults_group, dc_field.name)
                else None
            )
            flag = spec.repro_flag or _flag_name(dc_field, spec)
            assert flag is not None
            vtype = _value_type(spec, value if value is not None else default_value)
            if spec.negated_flag is not None:
                if not value:
                    args.append(spec.negated_flag)
            elif spec.boolean_optional:
                if value != default_value or spec.repro_always:
                    args.append(flag if value else "--no-" + flag.removeprefix("--"))
            elif vtype is bool:
                if value:
                    args.append(flag)
            else:
                if value is None:
                    if spec.native_value is not None and default_value is not None:
                        args.extend([flag, spec.native_value])
                elif spec.repro_always or value != default_value:
                    args.extend([flag, str(value)])
        for _, subgroup in self._subgroups():
            args.extend(subgroup.reproduction_args(context))
        return args

    def _matching_defaults(self, context: ParameterContext) -> ParameterGroup | None:
        if context.defaults is None:
            return None
        if type(context.defaults) is type(self):
            return context.defaults
        for _, subgroup in context.defaults._subgroups():
            if type(subgroup) is type(self):
                return subgroup
        return None

    def to_dict(self, context: ParameterContext | None = None) -> dict[str, Any]:
        """Serialize resolved parameter values into a JSON-compatible dictionary."""
        result: dict[str, Any] = {}
        target = result
        if self.dict_section is not None:
            section: dict[str, Any] = {}
            result[self.dict_section] = section
            target = section
        for dc_field, spec in self._param_fields():
            if not spec.in_dict:
                continue
            if context is not None and not _is_enabled(self, spec, context):
                continue
            value = getattr(self, dc_field.name)
            if value is None and spec.omit_none and spec.dict_key is None and self.dict_section is not None:
                continue
            key = spec.dict_key or dc_field.name
            target[key] = value
        for _, subgroup in self._subgroups():
            sub_dict = subgroup.to_dict(context)
            for key, value in sub_dict.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = {**result[key], **value}
                else:
                    result[key] = value
        return result

    def status_parts(self, context: ParameterContext) -> list[str]:
        """Return concise startup progress fragments describing active non-trivial settings."""
        parts: list[str] = []
        for dc_field, spec in self._param_fields():
            if spec.status_label is None:
                continue
            if not _is_enabled(self, spec, context):
                continue
            label = spec.status_label(getattr(self, dc_field.name))
            if label:
                parts.append(label)
        for _, subgroup in self._subgroups():
            parts.extend(subgroup.status_parts(context))
        return parts


# ---------------------------------------------------------------------------
# Four Helios GraphOperation ABI opt-in ParameterGroups
# ---------------------------------------------------------------------------


def _speculative_methods(group: SpeculationParameters, context: ParameterContext) -> tuple[str, ...]:
    recipe_methods = () if context.recipe is None else tuple(context.recipe.speculative_methods)
    return tuple(dict.fromkeys(name for name in (group.method, *group.supported_methods, *recipe_methods) if name))


def _has_speculative_methods(group: SpeculationParameters, context: ParameterContext) -> bool:
    return bool(_speculative_methods(group, context))


def _token_capacity_help_extra(group: SpeculationParameters, _context: ParameterContext) -> str:
    if group.packed_verification:
        return "at least drafts + 1; packed rows shared across retained request prefixes"
    return "at least drafts + 1; wider rectangular graphs require batch * width <= N"


@dataclass(frozen=True)
class SpeculationParameters(ParameterGroup):
    """Helios `GraphOperation.SPECULATIVE` ABI parameters (learned multi-token drafting)."""

    manifest_prefix: ClassVar[str | None] = "speculation"
    dict_section: ClassVar[str | None] = "speculation"
    default_group: ClassVar[str | None] = "speculative decoding"

    draft_tokens: int = param(
        flag="--mtp-draft-tokens",
        metavar="N",
        value_type=int,
        min_value=0,
        max_attr="max_draft_tokens",
        repro_always=True,
        requires_speculation=True,
        help="most draft tokens verified per round; 0 retains draft state with width-one rounds",
    )
    adaptive: bool = param(
        True,
        negated_flag="--no-mtp-adaptive-verification",
        requires_speculation=True,
        default_label="adaptive",
        help="default to the maximum verification width where captured, otherwise width one",
    )
    token_capacity: int | None = param(
        None,
        flag="--speculative-token-capacity",
        metavar="N",
        value_type=int,
        requires_speculation=True,
        default_label="larger of padded concurrency and drafts + 1",
        help_extra=_token_capacity_help_extra,
        help="maximum target-token rows in wider speculative graphs",
        in_dict=False,
    )
    packed_verification: bool = param(False, cli=False)
    draft_prefix_selection: str = param("cpu", cli=False)
    graph_budget: int | None = param(
        None,
        flag="--speculative-graph-budget",
        metavar="N",
        value_type=int,
        min_value=1,
        enabled_attr="packed_verification",
        requires_speculation=True,
        help="maximum captured packed round graphs, excluding mandatory width-one and ordinary graphs",
    )
    method: str | None = param(
        None,
        flag="--speculative-method",
        choices_fn=_speculative_methods,
        enabled_fn=_has_speculative_methods,
        repro_always=True,
        requires_speculation=True,
        help="learned proposal method",
    )
    mtp_only: bool = param(
        False,
        flag="--mtp-only",
        requires_speculation=True,
        help="omit MTP-disabled execution graphs; exported core requires speculation at runtime",
    )
    # Domain constraints declared by the recipe (never CLI flags or manifest entries).
    max_draft_tokens: int | None = None
    supported_methods: tuple[str, ...] = ()

    def __post_init__(self):
        if self.max_draft_tokens is None:
            object.__setattr__(self, "max_draft_tokens", self.draft_tokens)
        if self.mtp_only and self.draft_tokens < 1:
            raise ValueError("--mtp-only requires positive learned draft depth")
        if self.method is not None and not valid_speculative_method_name(self.method):
            raise ValueError("speculative method must be omitted or a lowercase identifier")
        if any(not valid_speculative_method_name(name) for name in self.supported_methods):
            raise ValueError("supported speculative methods must be lowercase identifiers")
        if self.graph_budget is not None:
            if type(self.graph_budget) is not int or self.graph_budget < 1:
                raise ValueError("speculative graph budget must be a positive integer")
            if self.draft_tokens < 1:
                raise ValueError("speculative graph budget requires positive learned draft depth")
        if self.draft_prefix_selection not in ("cpu", "gpu"):
            raise ValueError("draft prefix selection must be cpu or gpu")
        if self.draft_prefix_selection == "gpu" and (
            not self.packed_verification or not self.adaptive or self.draft_tokens < 1
        ):
            raise ValueError("GPU draft prefix selection requires packed adaptive learned speculation")

    @property
    def widths(self) -> tuple[int, ...]:
        """Verification widths: each draft count from zero through `draft_tokens`, plus the pending input."""
        return tuple(range(1, self.draft_tokens + 2))

    @property
    def fixed_width(self) -> int | None:
        return None if self.adaptive else self.draft_tokens + 1

    def effective_token_capacity(self, max_concurrency: int) -> int:
        from ignition.components.capacity import speculative_token_capacity

        return speculative_token_capacity(max_concurrency, self.draft_tokens + 1, self.token_capacity)

    def add_arguments(self, parser: argparse.ArgumentParser, context: ParameterContext, *,
                      include_multi_host: bool = True) -> None:
        target = _get_or_create_group(parser, self.default_group, self.group_description)
        target.add_argument(
            "--no-mtp",
            dest="no_mtp",
            action=_RecordingStoreTrueAction,
            default=False,
            help="disable multi-token speculation: load no draft model and decode one token per step;"
                 " cannot accompany other speculation flags",
        )
        for dc_field, spec in self._param_fields():
            if not spec.cli or not _is_enabled(self, spec, context):
                continue
            if dc_field.name == "token_capacity" and self.token_capacity is not None:
                spec = replace(spec, default_label=str(self.token_capacity))
            if dc_field.name == "adaptive" and not self.adaptive:
                spec = replace(spec, default_label="fixed width")
            if dc_field.name == "method" and self.method is None:
                spec = replace(spec, default_label="recipe draft")
            flag = _flag_name(dc_field, spec)
            assert flag is not None
            dest = _dest_name(dc_field, spec)
            default_value = getattr(self, dc_field.name)
            vtype = _value_type(spec, default_value)
            help_text = _format_help(self, dc_field, spec, context)
            if spec.negated_flag is not None or vtype is bool:
                target.add_argument(
                    flag, dest=dest, action=_RecordingStoreTrueAction, default=False, help=help_text,
                )
            else:
                kwargs: dict[str, Any] = {
                    "dest": dest,
                    "action": _RecordingStoreAction,
                    "default": None,
                    "help": help_text,
                }
                if vtype is not str:
                    kwargs["type"] = vtype
                if spec.metavar is not None:
                    kwargs["metavar"] = spec.metavar
                choices = _resolved_choices(self, spec, context)
                if choices is not None:
                    kwargs["choices"] = choices
                target.add_argument(flag, **kwargs)

    def option_arities(self, context: ParameterContext, *, include_multi_host: bool = True) -> dict[str, OptionArity]:
        arities = {"--no-mtp": OptionArity(0, 0)}
        arities.update(super().option_arities(context, include_multi_host=include_multi_host))
        return arities

    def resolve(self, args: Any, configuration: Any, alias: str,
                context: ParameterContext) -> SpeculationParameters:
        explicit = explicit_flags_from_args(args)
        method_explicit = _is_arg_explicit(args, "speculative_method", is_store_true=False)
        method = getattr(args, "speculative_method", None) if method_explicit else self.method
        allowed_methods = _speculative_methods(self, context)
        if method is not None and method not in allowed_methods:
            raise UsageError(f"--speculative-method {method}: {alias} does not prepare this proposal method")

        drafts_explicit = _is_arg_explicit(args, "mtp_draft_tokens", is_store_true=False)
        drafts = getattr(args, "mtp_draft_tokens", None) if drafts_explicit else self.draft_tokens
        limit = self.max_draft_tokens if self.max_draft_tokens is not None else self.draft_tokens
        if type(drafts) is not int or not 0 <= drafts <= limit:
            raise UsageError(f"--mtp-draft-tokens must be 0 through {limit}; --no-mtp disables speculation")

        mtp_only = self.mtp_only or bool(getattr(args, "mtp_only", False))
        if mtp_only and drafts == 0:
            raise UsageError("--mtp-only requires positive learned draft depth; --mtp-draft-tokens 0 disables MTP at loading")

        adaptive = self.adaptive and not bool(getattr(args, "no_mtp_adaptive_verification", False))

        cap_explicit = _is_arg_explicit(args, "speculative_token_capacity", is_store_true=False)
        capacity = getattr(args, "speculative_token_capacity", None) if cap_explicit else self.token_capacity
        if capacity is not None and (type(capacity) is not int or capacity < drafts + 1):
            raise UsageError("--speculative-token-capacity must cover the maximum verification width")

        packed = self.packed_verification and drafts > 0
        prefix_selection = self.draft_prefix_selection if drafts else "cpu"
        if prefix_selection == "gpu" and not adaptive:
            raise UsageError(f"{alias} requires adaptive verification; --no-mtp-adaptive-verification does not apply")

        budget_explicit = (
            "--speculative-graph-budget" in explicit
            or _is_arg_explicit(args, "speculative_graph_budget", is_store_true=False)
        )
        raw_budget = getattr(args, "speculative_graph_budget", None) if budget_explicit else self.graph_budget
        if budget_explicit and not (packed and drafts > 0):
            raise UsageError("--speculative-graph-budget requires a packed target catalog with positive draft depth")
        budget = raw_budget if packed and drafts > 0 else None
        if budget is not None and (type(budget) is not int or budget < 1):
            raise UsageError("--speculative-graph-budget must be a positive integer")

        return replace(
            self,
            draft_tokens=drafts,
            adaptive=adaptive,
            token_capacity=capacity,
            packed_verification=packed,
            draft_prefix_selection=prefix_selection,
            graph_budget=budget,
            method=method,
            mtp_only=mtp_only,
        )

    def validate(self, context: ParameterContext) -> None:
        alias = context.alias
        limit = self.max_draft_tokens
        if context.defaults is not None and context.defaults.speculation is not None:
            if context.defaults.speculation.max_draft_tokens is not None:
                limit = context.defaults.speculation.max_draft_tokens
        if limit is None:
            limit = self.draft_tokens
        if not 0 <= self.draft_tokens <= limit:
            raise UsageError(f"--mtp-draft-tokens must be 0 through {limit}; --no-mtp disables speculation")
        if self.mtp_only and self.draft_tokens == 0:
            raise UsageError("--mtp-only requires positive learned draft depth; --mtp-draft-tokens 0 disables MTP at loading")
        if self.method is not None and context.recipe is not None:
            default_spec = (
                context.defaults.speculation if context.defaults is not None
                else context.recipe.parameters.speculation
            )
            allowed = _speculative_methods(
                default_spec if default_spec is not None else replace(self, method=None),
                context,
            )
            if self.method not in allowed:
                raise UsageError(f"--speculative-method {self.method}: {alias} does not prepare this proposal method")
        if self.token_capacity is not None and (type(self.token_capacity) is not int or self.token_capacity < self.draft_tokens + 1):
            raise UsageError("--speculative-token-capacity must cover the maximum verification width")
        if "--speculative-graph-budget" in context.explicit_flags and not (self.packed_verification and self.draft_tokens > 0):
            raise UsageError("--speculative-graph-budget requires a packed target catalog with positive draft depth")
        if self.graph_budget is not None and context.root is not None:
            from ignition.components.capacity import packed_speculative_shapes

            try:
                packed_speculative_shapes(
                    context.root.max_concurrency, self.draft_tokens + 1, self.token_capacity, self.graph_budget,
                )
            except ValueError as error:
                raise UsageError(f"--speculative-graph-budget: {error}") from error

    def manifest_parameters(self, context: ParameterContext) -> list[tuple[str, ParameterScope, Any]]:
        if self.packed_verification and self.draft_tokens > 0 and self.graph_budget is None:
            raise ValueError("packed speculative preparation metadata requires an explicit graph budget")
        assert context.root is not None
        entries: list[tuple[str, ParameterScope, Any]] = [
            ("speculation.has_draft", ParameterScope.DRAFT_WEIGHT, True),
            ("speculation.adaptive", ParameterScope.DEPLOYMENT, bool(self.adaptive)),
            ("speculation.draft_tokens", ParameterScope.DEPLOYMENT, int(self.draft_tokens)),
            ("speculation.mtp_only", ParameterScope.DEPLOYMENT, bool(self.mtp_only)),
            ("speculation.packed_verification", ParameterScope.DEPLOYMENT, bool(self.packed_verification)),
            (
                "speculation.token_capacity",
                ParameterScope.DEPLOYMENT,
                int(self.effective_token_capacity(context.root.max_concurrency)),
            ),
        ]
        if self.draft_prefix_selection:
            entries.append(("speculation.draft_prefix_selection", ParameterScope.DEPLOYMENT, self.draft_prefix_selection))
        if self.graph_budget is not None:
            entries.append(("speculation.graph_budget", ParameterScope.DEPLOYMENT, int(self.graph_budget)))
        if self.method is not None:
            entries.append(("speculation.method", ParameterScope.DEPLOYMENT, self.method))
        return entries

    def reproduction_args(self, context: ParameterContext) -> list[str]:
        assert context.root is not None
        args: list[str] = []
        if self.method is not None:
            args.extend(["--speculative-method", self.method])
        args.extend(["--mtp-draft-tokens", str(self.draft_tokens)])
        if self.mtp_only:
            args.append("--mtp-only")
        if not self.adaptive:
            args.append("--no-mtp-adaptive-verification")
        if self.draft_tokens > 0:
            cap = self.effective_token_capacity(context.root.max_concurrency)
            if cap:
                args.extend(["--speculative-token-capacity", str(cap)])
            if self.packed_verification and self.graph_budget is not None:
                args.extend(["--speculative-graph-budget", str(self.graph_budget)])
        return args

    def to_dict(self, context: ParameterContext | None = None) -> dict[str, Any]:
        spec_dict: dict[str, Any] = {
            "draft_tokens": self.draft_tokens,
            "adaptive": self.adaptive,
            "mtp_only": self.mtp_only,
            "packed_verification": self.packed_verification,
            "draft_prefix_selection": self.draft_prefix_selection,
        }
        if self.method is not None:
            spec_dict["method"] = self.method
        if self.graph_budget is not None:
            spec_dict["graph_budget"] = self.graph_budget
        result: dict[str, Any] = {"speculation": spec_dict}
        if context is not None and context.root is not None:
            result["speculative_token_capacity"] = self.effective_token_capacity(context.root.max_concurrency)
        return result

    def status_parts(self, context: ParameterContext) -> list[str]:
        from ignition.cli import display

        assert context.root is not None
        parts = [display.speculation(self, brief=True)]
        parts.append(f"speculative token capacity {self.effective_token_capacity(context.root.max_concurrency)}")
        if self.graph_budget is not None:
            parts.append(f"packed graph budget {self.graph_budget}")
        method_spec = (
            None if self.method is None or context.recipe is None
            else context.recipe.speculative_methods.get(self.method)
        )
        if method_spec is not None and method_spec.label:
            parts.append(f"{method_spec.label} proposals")
        return parts

    def capacity_relief_options(self, attempt: Callable[[RecipeParameters], Any],
                                context: ParameterContext) -> list[str]:
        assert context.root is not None
        if attempt(context.root.without_speculation()) is not None:
            return ["--no-mtp"]
        return []


@dataclass(frozen=True)
class VisionParameters(ParameterGroup):
    """Helios `GraphOperation.VISION` ABI parameters (image request slots and separable vision encoder)."""

    image_concurrency: int | None = param(
        None,
        flag="--image-concurrency",
        metavar="N",
        value_type=int,
        min_value=1,
        scope=ParameterScope.DEPLOYMENT,
        help="maximum active image requests; excess requests queue independently of text",
    )
    language_model_only: bool = param(
        False,
        flag="--language-model-only",
        group="advanced options",
        enabled_attr="separable_encoder",
        scope=ParameterScope.COMBINE,
        help="prepare no vision encoder and reject images; its weights and scratch become KV pages",
    )
    # Capability flag declared by recipes whose vision encoder can be omitted at preparation time.
    separable_encoder: bool = False

    def effective_image_concurrency(self, max_concurrency: int) -> int:
        if self.language_model_only:
            return 0
        if self.image_concurrency is None:
            return min(4, max_concurrency)
        return self.image_concurrency

    def add_arguments(self, parser: argparse.ArgumentParser, context: ParameterContext, *,
                      include_multi_host: bool = True) -> None:
        default_images = min(4, context.root.max_concurrency) if context.root is not None else 4
        for dc_field, spec in self._param_fields():
            if not _is_enabled(self, spec, context):
                continue
            if dc_field.name == "image_concurrency":
                spec = replace(spec, default_label=f"min(4, max concurrency) = {default_images}")
            flag = _flag_name(dc_field, spec)
            assert flag is not None
            dest = _dest_name(dc_field, spec)
            target = _get_or_create_group(parser, spec.group, self.group_description)
            default_value = getattr(self, dc_field.name)
            vtype = _value_type(spec, default_value)
            help_text = _format_help(self, dc_field, spec, context)
            if vtype is bool:
                target.add_argument(
                    flag, dest=dest, action=_RecordingStoreTrueAction, default=False, help=help_text,
                )
            else:
                target.add_argument(
                    flag, dest=dest, action=_RecordingStoreAction, default=None, type=vtype,
                    metavar=spec.metavar, help=help_text,
                )

    def validate(self, context: ParameterContext) -> None:
        alias = context.alias
        assert context.root is not None
        if self.language_model_only:
            if not self.separable_encoder:
                raise UsageError(f"--language-model-only: {alias} prepares no separable vision encoder")
            if self.image_concurrency is not None or "--image-concurrency" in context.explicit_flags:
                raise UsageError("--image-concurrency requires the vision encoder that --language-model-only leaves unloaded")
            return
        if self.image_concurrency is not None:
            if type(self.image_concurrency) is not int or not 1 <= self.image_concurrency <= context.root.max_concurrency:
                raise UsageError("--image-concurrency requires a vision model and must be 1 through max concurrency")

    def adjust_rank_costs(self, costs: Any, context: ParameterContext) -> Any:
        assert context.root is not None
        return costs.with_image_slots(self.effective_image_concurrency(context.root.max_concurrency))

    def capacity_relief_options(self, attempt: Callable[[RecipeParameters], Any],
                                context: ParameterContext) -> list[str]:
        assert context.root is not None
        options: list[str] = []
        if not self.language_model_only:
            if self.image_concurrency is not None:
                for slots in range(min(self.image_concurrency, context.root.max_concurrency) - 1, 0, -1):
                    candidate = replace(context.root, vision=replace(self, image_concurrency=slots))
                    if attempt(candidate) is not None:
                        options.append(f"--image-concurrency {slots}")
                        break
            if self.separable_encoder and self.image_concurrency is None:
                candidate = replace(context.root, vision=replace(self, language_model_only=True, image_concurrency=None))
                if attempt(candidate) is not None:
                    options.append("--language-model-only")
        return options

    def validate_for_benchmark(self, settings: Any, context: ParameterContext) -> None:
        if context.recipe is not None and context.recipe.benchmark_language_model_only:
            if self.image_concurrency is not None or "--image-concurrency" in context.explicit_flags:
                raise UsageError(
                    f"{context.recipe.name} benchmark graphs prepare no vision encoder; --image-concurrency does not apply"
                )

    def manifest_parameters(self, context: ParameterContext) -> list[tuple[str, ParameterScope, Any]]:
        assert context.root is not None
        entries: list[tuple[str, ParameterScope, Any]] = [
            ("image_concurrency", ParameterScope.DEPLOYMENT, int(self.effective_image_concurrency(context.root.max_concurrency))),
        ]
        if self.separable_encoder:
            entries.append(("language_model_only", ParameterScope.COMBINE, bool(self.language_model_only)))
        return entries

    def reproduction_args(self, context: ParameterContext) -> list[str]:
        assert context.root is not None
        if self.language_model_only:
            return ["--language-model-only"]
        return ["--image-concurrency", str(self.effective_image_concurrency(context.root.max_concurrency))]

    def to_dict(self, context: ParameterContext | None = None) -> dict[str, Any]:
        concurrency = context.root.max_concurrency if context is not None and context.root is not None else 4
        result: dict[str, Any] = {"image_concurrency": self.effective_image_concurrency(concurrency)}
        if self.separable_encoder:
            result["language_model_only"] = self.language_model_only
        return result

    def status_parts(self, context: ParameterContext) -> list[str]:
        assert context.root is not None
        parts = [f"images {self.effective_image_concurrency(context.root.max_concurrency)}"]
        if self.language_model_only:
            parts.append("language model only")
        return parts


@dataclass(frozen=True)
class PromptLogprobsParameters(ParameterGroup):
    """Helios `GraphOperation.SCORED_PREFILL` ABI parameter."""

    default_group: ClassVar[str | None] = "capacity and compression"

    enabled: bool = param(
        False,
        flag="--prompt-logprobs",
        manifest_key="prompt_logprobs",
        dict_key="prompt_logprobs",
        scope=ParameterScope.DEPLOYMENT,
        help="capture dedicated prefill graphs for raw prompt token logprobs",
    )

    def validate_for_benchmark(self, settings: Any, context: ParameterContext) -> None:
        if self.enabled:
            raise UsageError(
                "ignition bench measures speculative rounds or unscored prefill;"
                " --mixed-prefill and --prompt-logprobs do not apply"
            )

    def status_parts(self, context: ParameterContext) -> list[str]:
        return [f"prompt logprobs {'on' if self.enabled else 'off'}"]


@dataclass(frozen=True)
class MixedPrefillParameters(ParameterGroup):
    """Helios `GraphOperation.MIXED` ABI parameter."""

    default_group: ClassVar[str | None] = "capacity and compression"

    enabled: bool = param(
        False,
        flag="--mixed-prefill",
        manifest_key="mixed_prefill",
        dict_key="mixed_prefill",
        scope=ParameterScope.DEPLOYMENT,
        help="capture graphs combining prompt prefill with decode or verification",
    )

    def shapes(self, concurrency: int, chunk: int, speculation: SpeculationParameters | None) -> tuple[tuple[bool, int, int, int], ...]:
        if not self.enabled:
            return ()
        from ignition.components.capacity import mixed_shapes, serving_modes

        modes = serving_modes(speculation is not None, speculation is not None and speculation.mtp_only)
        return tuple(
            (enabled, b, w, p)
            for enabled in modes
            for b, w, p in mixed_shapes(
                concurrency,
                chunk,
                speculation.widths if enabled else (1,),
                speculation.token_capacity if enabled else None,
            )
        )

    def validate(self, context: ParameterContext) -> None:
        if self.enabled and context.root is not None and context.root.speculation is not None:
            method_name = context.root.speculation.method
            method_spec = (
                None if method_name is None or context.recipe is None
                else context.recipe.speculative_methods.get(method_name)
            )
            if method_spec is not None and not method_spec.mixed_prefill:
                raise UsageError(
                    f"--speculative-method {method_name}"
                    " requires separate prefill and verification graphs"
                )

    def adjust_headroom_bytes(self, headroom_bytes: int, context: ParameterContext) -> int:
        if not self.enabled or context.root is None:
            return headroom_bytes
        from ignition import planning

        mixed = self.shapes(context.root.max_concurrency, context.root.prefill_chunk, context.root.speculation)
        tokens = max(
            context.root.prefill_chunk,
            max((p + (b - 1) * w for _, b, w, p in mixed), default=0),
        )
        return max(
            headroom_bytes,
            planning.HEADROOM_BYTES + max(0, tokens - planning.HEADROOM_CHUNK_TOKENS) * planning.HEADROOM_CHUNK_TOKEN_BYTES,
        )

    def validate_for_benchmark(self, settings: Any, context: ParameterContext) -> None:
        if self.enabled:
            raise UsageError(
                "ignition bench measures speculative rounds or unscored prefill;"
                " --mixed-prefill and --prompt-logprobs do not apply"
            )

    def status_parts(self, context: ParameterContext) -> list[str]:
        return [f"mixed prefill {'on' if self.enabled else 'off'}"]


# ---------------------------------------------------------------------------
# Universal RecipeParameters base class
# ---------------------------------------------------------------------------


def _declared_context_limit(group: RecipeParameters, context: ParameterContext) -> int | None:
    if context.recipe is None:
        return None
    root = context.root if context.root is not None else group
    ctx = replace(context, root=root)
    return root.adjust_context(context.recipe.base_context(context.checkpoint), ctx)


@dataclass(frozen=True)
class RecipeParameters(ParameterGroup):
    """Base parameter schema for every Ignition model recipe.

    Carries universal preparation parameters and optional slots for the four
    Helios GraphOperation ABI groups. Model recipes compose additional shared
    `ParameterGroup`s or subclass `RecipeParameters` for model-specific options.
    """

    max_concurrency: int = param(
        4,
        flag="--max-concurrency",
        metavar="N",
        value_type=int,
        min_value=1,
        max_attr="max_concurrency_limit",
        model_config_key="max_concurrency",
        scope=ParameterScope.DEPLOYMENT,
        repro_always=True,
        bare_default_label="defaults.max_concurrency, else see MODEL --help",
        help="maximum simultaneously active requests",
    )
    prefill_chunk: int = param(
        4096,
        flag="--prefill-chunk-size",
        group="capacity and compression",
        metavar="N",
        value_type=int,
        min_value=64,
        max_fn=_declared_context_limit,
        multiple_of=64,
        manifest_key="prefill_chunk",
        scope=ParameterScope.DEPLOYMENT,
        repro_always=True,
        bare_default_label="see MODEL --help",
        help="most prompt tokens per prefill step",
    )
    autotune_effort: str = param(
        "default",
        flag="--autotune-effort",
        group="advanced options",
        choices=AUTOTUNE_EFFORTS,
        config_key="preparation.autotune_effort",
        scope=ParameterScope.DEPLOYMENT,
        repro_always=True,
        help="kernel search effort: off uses heuristics without races; default races the pruned candidate set;"
             " audit lifts performance exclusions",
    )
    compile_workers: int | None = param(
        None,
        flag="--compile-workers",
        group="advanced options",
        metavar="N",
        value_type=int,
        min_value=1,
        config_key="preparation.compile_workers",
        scope=ParameterScope.NONE,
        default_label="automatic",
        help="CPU compiler process budget per host, shared by local preparation ranks",
    )
    lock_gpu_clocks: bool = param(
        False,
        flag="--lock-gpu-clocks",
        group="advanced options",
        scope=ParameterScope.DEPLOYMENT,
        help="lock each tuning GPU to its maximum supported graphics clock during autotuning",
    )
    compiled_forwards: bool = param(
        True,
        negated_flag="--no-compile",
        group="advanced options",
        scope=ParameterScope.DEPLOYMENT,
        default_label="compiled",
        help="development: capture graphs without compiling forwards",
        in_dict=False,
    )
    roce_ring: bool = param(
        False,
        flag="--roce-ring",
        group="multi-host tensor parallelism",
        multi_host_only=True,
        scope=ParameterScope.DEPLOYMENT,
        help="route collectives and fabric TCP through neighbors in rank order",
        status_label=lambda v: "experimental RoCE ring (native neighbor relays)" if v else None,
    )
    roce_chain: bool = param(
        False,
        flag="--roce-chain",
        group="multi-host tensor parallelism",
        multi_host_only=True,
        scope=ParameterScope.DEPLOYMENT,
        help="like --roce-ring for hosts cabled as a chain, with no link between the first and last rank",
        status_label=lambda v: "experimental RoCE chain (native neighbor relays, open ends)" if v else None,
    )
    # Non-CLI execution parameters populated during planning so PreparationManifest
    # is built exclusively from RecipeParameters.
    tensor_parallel: int | None = param(None, cli=False, value_type=int, scope=ParameterScope.DEPLOYMENT, in_dict=False)
    multi_host: bool = param(False, cli=False, value_type=bool, scope=ParameterScope.NONE, in_dict=False)
    reserve_bytes: int = param(0, cli=False, value_type=int, scope=ParameterScope.DEPLOYMENT, in_dict=False)

    # Recipe domain constraint (never a CLI flag or manifest entry).
    max_concurrency_limit: int = 64

    # Optional Helios GraphOperation ABI groups (None = capability absent on this recipe).
    speculation: SpeculationParameters | None = None
    vision: VisionParameters | None = None
    prompt_logprobs: PromptLogprobsParameters | None = None
    mixed_prefill: MixedPrefillParameters | None = None

    def resolve(self, args: Any, configuration: Any, alias: str,
                context: ParameterContext) -> RecipeParameters:
        explicit = explicit_flags_from_args(args)
        ctx = replace(context, root=self, defaults=self, explicit_flags=explicit)

        no_mtp = bool(getattr(args, "no_mtp", False))
        if no_mtp:
            if self.speculation is None:
                raise UsageError(f"{alias} has no multi-token speculation; --no-mtp does not apply")
            given = self.explicit_speculation_flags(args, ctx)
            if given:
                raise UsageError(f"--no-mtp disables speculation; {', '.join(given)} cannot accompany it")

        updates = self._resolve_fields(args, configuration, alias, ctx)
        if configuration is not None:
            updates["reserve_bytes"] = configuration.reserve_bytes
        if "compile_workers" in updates and updates["compile_workers"] == 0:
            if _is_arg_explicit(args, "compile_workers", is_store_true=False):
                raise UsageError("--compile-workers must be a positive integer")
            updates["compile_workers"] = None

        preliminary = replace(self, **updates) if updates else self
        sub_ctx = replace(ctx, root=preliminary)

        for dc_field, subgroup in self._subgroups():
            if dc_field.name == "speculation" and no_mtp:
                updates["speculation"] = None
            else:
                updates[dc_field.name] = subgroup.resolve(args, configuration, alias, sub_ctx)

        resolved = replace(self, **updates) if updates else self
        if no_mtp:
            resolved = resolved.without_speculation()
        return resolved

    def without_speculation(self) -> RecipeParameters:
        """Return a copy with speculation disabled and speculation-only fields cleared."""
        updates: dict[str, Any] = {"speculation": None}
        for dc_field, spec in self._param_fields():
            if spec.requires_speculation and dc_field.default is not _UNSET:
                updates[dc_field.name] = dc_field.default
        for dc_field, subgroup in self._subgroups():
            if dc_field.name != "speculation":
                updates[dc_field.name] = subgroup.reset_speculation_fields()
        return replace(self, **updates)

    def validate(self, context: ParameterContext) -> None:
        ctx = replace(context, root=self)
        if self.roce_ring and self.roce_chain:
            raise UsageError("--roce-ring and --roce-chain exclude each other")
        if self.autotune_effort not in AUTOTUNE_EFFORTS:
            raise UsageError(f"--autotune-effort must be one of {', '.join(AUTOTUNE_EFFORTS)}")
        if self.compile_workers is not None and (type(self.compile_workers) is not int or self.compile_workers < 1):
            raise UsageError("--compile-workers must be a positive integer")
        concurrency_limit = (
            context.defaults.max_concurrency_limit
            if context.defaults is not None
            else self.max_concurrency_limit
        )
        if not 1 <= self.max_concurrency <= concurrency_limit:
            raise UsageError(
                f"--max-concurrency for {ctx.alias} must be 1 through {concurrency_limit}"
            )
        # Validate composed subgroups before calling `adjust_context`, which may depend on them.
        for _, subgroup in self._subgroups():
            subgroup.validate(ctx)
        if ctx.recipe is not None:
            limit = _declared_context_limit(self, ctx)
            if type(limit) is not int or not 64 <= limit < 2**31 or limit % 64:
                raise UsageError(
                    f"{ctx.alias} declared context ({limit!r}) must be a 64-aligned positive int32"
                )
            if limit % ctx.recipe.page_tokens:
                raise UsageError(
                    f"{ctx.alias} context ({limit:,}) must be a multiple of"
                    f" the recipe page size ({ctx.recipe.page_tokens})"
                )
            if self.prefill_chunk % 64 or not 64 <= self.prefill_chunk <= limit:
                raise UsageError(
                    f"--prefill-chunk-size for {ctx.alias} must be a multiple of 64"
                    f" from 64 through {limit:,}"
                )
        self.validate_fields(ctx)

    def manifest_parameters(self, context: ParameterContext) -> list[tuple[str, ParameterScope, Any]]:
        ctx = replace(context, root=self)
        return super().manifest_parameters(ctx)

    def reproduction_args(self, context: ParameterContext) -> list[str]:
        ctx = replace(context, root=self)
        args: list[str] = []
        if self.tensor_parallel is not None:
            if self.multi_host:
                args.extend(["--ranks", str(self.tensor_parallel)])
                if self.roce_ring:
                    args.append("--roce-ring")
                if self.roce_chain:
                    args.append("--roce-chain")
            else:
                gpus = "0" if self.tensor_parallel == 1 else f"0-{self.tensor_parallel - 1}"
                args.extend(["--gpus", gpus])

        args.extend(["--max-concurrency", str(self.max_concurrency)])
        args.extend(["--prefill-chunk-size", str(self.prefill_chunk)])

        if self.speculation is None and ctx.defaults is not None and ctx.defaults.speculation is not None:
            args.append("--no-mtp")
        for _, subgroup in self._subgroups():
            args.extend(subgroup.reproduction_args(ctx))

        base_field_names = {f.name for f in fields(RecipeParameters)}
        defaults_group = self._matching_defaults(ctx)
        for dc_field, spec in self._param_fields():
            if dc_field.name in base_field_names or not spec.cli or not _is_enabled(self, spec, ctx):
                continue
            if spec.multi_host_only and not self.multi_host:
                continue
            value = getattr(self, dc_field.name)
            default_value = (
                getattr(defaults_group, dc_field.name)
                if defaults_group is not None and hasattr(defaults_group, dc_field.name)
                else None
            )
            flag = spec.repro_flag or _flag_name(dc_field, spec)
            assert flag is not None
            vtype = _value_type(spec, value if value is not None else default_value)
            if spec.negated_flag is not None:
                if not value:
                    args.append(spec.negated_flag)
            elif spec.boolean_optional:
                if value != default_value or spec.repro_always:
                    args.append(flag if value else "--no-" + flag.removeprefix("--"))
            elif vtype is bool:
                if value:
                    args.append(flag)
            elif value is None:
                if spec.native_value is not None and default_value is not None:
                    args.extend([flag, spec.native_value])
            elif spec.repro_always or value != default_value:
                args.extend([flag, str(value)])

        if self.autotune_effort:
            args.extend(["--autotune-effort", self.autotune_effort])
        if self.lock_gpu_clocks:
            args.append("--lock-gpu-clocks")
        if not self.compiled_forwards:
            args.append("--no-compile")
        return args

    def to_dict(self, context: ParameterContext | None = None) -> dict[str, Any]:
        ctx = ParameterContext(root=self) if context is None else replace(context, root=self)
        result = super().to_dict(ctx)
        if self.speculation is None:
            result.setdefault("speculation", None)
            result.setdefault("speculative_token_capacity", 0)
        if self.vision is None:
            result.setdefault("image_concurrency", 0)
        return result

    def status_parts(self, context: ParameterContext) -> list[str]:
        ctx = replace(context, root=self)
        parts: list[str] = [f"concurrency {self.max_concurrency}"]
        if self.vision is not None:
            parts.extend(self.vision.status_parts(ctx))
        else:
            parts.append("images 0")
        if self.speculation is not None:
            parts.extend(self.speculation.status_parts(ctx))
        else:
            parts.append("speculation off")
        for dc_field, subgroup in self._subgroups():
            if dc_field.name not in ("vision", "speculation"):
                parts.extend(subgroup.status_parts(ctx))
        for dc_field, spec in self._param_fields():
            if spec.status_label is None or not _is_enabled(self, spec, ctx):
                continue
            label = spec.status_label(getattr(self, dc_field.name))
            if label:
                parts.append(label)
        parts.append(f"compile workers {self.compile_workers or 'auto'} per host")
        parts.append(f"autotune effort {self.autotune_effort}")
        return parts

    def capacity_relief_options(self, attempt: Callable[[RecipeParameters], Any],
                                context: ParameterContext) -> list[str]:
        ctx = replace(context, root=self)
        options: list[str] = []
        for limit in range(self.max_concurrency - 1, 0, -1):
            candidate = replace(self, max_concurrency=limit)
            if candidate.vision is not None and candidate.vision.image_concurrency is not None:
                candidate = replace(
                    candidate,
                    vision=replace(candidate.vision, image_concurrency=min(candidate.vision.image_concurrency, limit)),
                )
            if attempt(candidate) is not None:
                options.append(f"--max-concurrency {limit}")
                break
        if self.vision is not None:
            options.extend(self.vision.capacity_relief_options(attempt, ctx))
        if self.speculation is not None:
            options.extend(self.speculation.capacity_relief_options(attempt, ctx))
        for dc_field, subgroup in self._subgroups():
            if dc_field.name not in ("vision", "speculation"):
                options.extend(subgroup.capacity_relief_options(attempt, ctx))
        return options


def known_core_abi_flags() -> frozenset[str]:
    """Return every CLI flag belonging to the four core Helios ABI groups."""
    prototype = RecipeParameters(
        speculation=SpeculationParameters(
            draft_tokens=1, packed_verification=True, graph_budget=1, method="mtp", supported_methods=("mtp",),
        ),
        vision=VisionParameters(separable_encoder=True),
        prompt_logprobs=PromptLogprobsParameters(),
        mixed_prefill=MixedPrefillParameters(),
    )
    ctx = ParameterContext(root=prototype)
    return frozenset(prototype.option_arities(ctx))
