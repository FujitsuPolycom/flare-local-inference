# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Describe preparation inputs and named views of stored weights without copying payloads.

The catalog describes prepared layouts, not checkpoint tensors. Reading it does
not instantiate Python objects or establish that a recipe can rebuild from it.

Recipes declare semantic weight groups explicitly with `declare_weights`. The
declaration states which container fields hold layers, which groups are tuning
selected, and which views are aliases. Object traversal only enumerates the
resulting views; it never decides immutability, grouping, or alias identity from
object paths or pointer values. Tensors under types without a declared
classification are reported as unavailable instead of being treated as weights.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
import math
import re

from . import runtime_core_pb2 as proto

STABLE = 'stable'
TUNED = 'tuned'

# Field allowlists for external weight container types, keyed by exact type.
# Entries name the fields that carry stored weight bytes; a listed type is never
# traversed through any other field, so runtime buffers of that type stay
# unclassified and end up in the initial-state payload.
CONTAINER_FIELDS: dict[str, tuple[str, ...]] = {}
CONTAINER_TUNED: dict[str, frozenset[str]] = {}
# Per-module-type fields that hold stored bytes. Module parameters are always
# included; buffers are only included for types declared here, so rotary tables,
# masks, and other derived constants never become weight owners.
MODULE_BUFFER_FIELDS: dict[str, frozenset[str]] = {}


def declare_container_fields(type_key, field_names, *, tuned=()):
    """Declare the weight-bearing fields of an external container type."""
    if not isinstance(type_key, str) or not type_key:
        raise ValueError('declared container types need a module-qualified name')
    CONTAINER_FIELDS[type_key] = tuple(field_names)
    CONTAINER_TUNED[type_key] = frozenset(tuned)


def weight_container(field_names, *, tuned=()):
    """Register the weight-bearing fields of a container class at definition time."""
    names = tuple(field_names)

    def decorator(cls):
        if is_dataclass(cls):
            defined = {item.name for item in fields(cls)}
            absent = [name for name in names if name not in defined]
            if absent:
                raise ValueError(
                    f'{cls.__module__}.{cls.__qualname__}: unknown declared weight fields {absent}'
                )
        declare_container_fields(f'{cls.__module__}.{cls.__qualname__}', names, tuned=tuned)
        return cls

    return decorator


def declare_module_buffers(type_key, fields):
    """Declare which module buffers of one type hold stored bytes."""
    if not isinstance(type_key, str) or not type_key:
        raise ValueError('declared module types need a module-qualified name')
    MODULE_BUFFER_FIELDS[type_key] = frozenset(fields)


@dataclass(frozen=True)
class WeightDeclaration:
    """Explicit semantic weight declaration supplied by a recipe."""

    roots: Mapping[str, object]
    # `(path, group)` pairs; a path may name a root or a dotted field/attribute path.
    components: tuple[tuple[str, str], ...]
    layer_fields: tuple[str, ...] = ('layers',)
    tuned_groups: tuple[str, ...] = ()
    aliases: tuple[tuple[str, str], ...] = ()


def declare_weights(roots, *, components, layers=('layers',), tuned=(), aliases=()):
    """Declare the semantic grouping of a recipe's prepared weight objects.

    `components` names the root fields that hold stored weight bytes; other
    fields of the same objects stay unclassified. A name equal to a root keeps
    that whole root in one component group.
    """
    if roots is None:
        raise ValueError('weight declarations require a weight object')
    if not isinstance(roots, Mapping):
        roots = {'weights': roots}
    if not roots:
        raise ValueError('weight declarations require at least one root')
    if not components:
        raise ValueError('weight declarations require declared components')
    if isinstance(components, Mapping):
        entries = tuple((str(path), str(group)) for path, group in components.items())
    else:
        entries = tuple((str(name), str(name)) for name in components)
    return WeightDeclaration(roots=MappingProxyType(dict(roots)), components=entries,
        layer_fields=tuple(layers), tuned_groups=tuple(tuned), aliases=tuple(aliases))


def _type_key(value):
    kind = type(value)
    return f'{kind.__module__}.{kind.__qualname__}'


def _storage(group, declaration):
    return TUNED if group in declaration.tuned_groups else STABLE


def declared_views(declaration):
    """Enumerate declared weight views into layer or misc groups, retaining storage class."""
    import torch

    disabled = (-1,)

    def walk(value, name, owner, group, storage, ancestors):
        if isinstance(value, torch.Tensor):
            payload_group = group if re.fullmatch(r"layer-[0-9]+", group) else "misc"
            yield name, owner, value, payload_group, storage
            return
        if value is None or id(value) in ancestors:
            return
        ancestry = ancestors | {id(value)}
        key = _type_key(value)
        child_storage = {}
        if key in CONTAINER_FIELDS:
            owner = key
            tuned = CONTAINER_TUNED.get(key, frozenset())
            children = [(f'.{field}', getattr(value, field)) for field in CONTAINER_FIELDS[key]]
            child_storage = {f'.{field}': TUNED if field in tuned else storage
                             for field in CONTAINER_FIELDS[key]}
        elif isinstance(value, Mapping):
            children = [(f'[{k!r}]', child) for k, child in value.items()
                        if isinstance(k, (str, int))]
        elif isinstance(value, (tuple, list)):
            children = [(f'[{index}]', child) for index, child in enumerate(value)]
        elif isinstance(value, torch.nn.Module):
            owner = key
            buffers = MODULE_BUFFER_FIELDS.get(key, frozenset())
            children = [(f'.{k}', child) for k, child in value._parameters.items()]
            children += [(f'.{k}', value._buffers[k]) for k in sorted(buffers) if k in value._buffers]
            children += [(f'.{k}', child) for k, child in value._modules.items()]
        elif is_dataclass(value) and not isinstance(value, type):
            owner = key
            children = [(f'.{field.name}', getattr(value, field.name)) for field in fields(value)]
        else:
            return
        for suffix, child in children:
            yield from walk(child, name + suffix, owner, group,
                            child_storage.get(suffix, storage), ancestry)

    def resolve(value, path):
        for segment in path.split('.'):
            if isinstance(value, Mapping):
                if segment not in value:
                    raise KeyError(path)
                value = value[segment]
            elif isinstance(value, (torch.nn.Module,)) or is_dataclass(value):
                value = getattr(value, segment)
            else:
                raise AttributeError(path)
        return value

    components = dict(declaration.components)
    for root, value in sorted(declaration.roots.items()):
        if root in components:
            yield from walk(value, root, '', components[root],
                            _storage(components[root], declaration), frozenset())
            continue
        members = _members(value)
        prefix = root + '.'
        for suffix, child in members:
            path = root + suffix
            relative = path[len(prefix):] if path.startswith(prefix) else path
            field = suffix.rsplit('.', 1)[-1].partition('[')[0]
            group = components.get(path)
            if group is None:
                group = components.get(relative, components.get(field))
            if group is not None:
                yield from walk(child, path, '', group,
                                _storage(group, declaration), frozenset())
                continue
            for declared, declared_group in components.items():
                if not declared.startswith(relative + '.'):
                    continue
                try:
                    selected = resolve(child, declared[len(relative) + 1:])
                except (KeyError, AttributeError) as error:
                    raise ValueError(
                        f'declared weight component {declared!r} does not resolve') from error
                yield from walk(selected, f'{root}.{declared}', '', declared_group,
                                _storage(declared_group, declaration), frozenset())

    for root, value in sorted(declaration.roots.items()):
        if root in components:
            continue
        for declared in declaration.layer_fields:
            try:
                layers = resolve(value, declared)
            except (KeyError, AttributeError):
                continue
            if not isinstance(layers, (tuple, list, torch.nn.ModuleList)):
                continue
            for index, layer in enumerate(layers):
                group = f'layer-{index:03d}'
                yield from walk(layer, f'{root}.{declared}[{index}]', '', group,
                                _storage(group, declaration), frozenset())


def _members(value):
    import torch

    if isinstance(value, Mapping):
        return [(f'[{key!r}]', child) for key, child in value.items() if isinstance(key, (str, int))]
    if isinstance(value, (tuple, list)):
        return [(f'[{index}]', child) for index, child in enumerate(value)]
    if isinstance(value, torch.nn.Module):
        return [(f'.{key}', child) for group in (value._parameters, value._buffers, value._modules)
                for key, child in group.items()]
    return [(f'.{field.name}', getattr(value, field.name)) for field in fields(value)]


def reproduction_arguments(model, plan, *, compiled_forwards, multi_host=False):
    """Build canonical `ignition export` arguments reproducing `plan` without host-local paths."""
    from pathlib import Path

    if model.directory is not None:
        target = Path(model.directory).name
    elif plan.checkpoint.revision:
        target = f'{plan.checkpoint.repository}@{plan.checkpoint.revision}'
    else:
        target = plan.checkpoint.repository
    args = ['ignition', 'export', target]
    if multi_host:
        args.extend(['--ranks', str(plan.world_size)])
        if plan.roce_ring:
            args.append('--roce-ring')
        if plan.roce_chain:
            args.append('--roce-chain')
    else:
        gpus = '0' if plan.world_size == 1 else f'0-{plan.world_size - 1}'
        args.extend(['--gpus', gpus])
    supports_cp = any(costs.page_tokens for costs in plan.recipe.ranks.values())
    if not plan.context_parallel:
        if supports_cp:
            args.append('--no-context-parallel')
    elif plan.context_parallel_size is not None and plan.context_parallel_size != plan.world_size:
        args.extend(['--dcp-size', str(plan.context_parallel_size)])
    args.extend(['--max-concurrency', str(plan.concurrency)])
    if plan.recipe.max_images:
        if plan.language_model_only:
            args.append('--language-model-only')
        else:
            args.extend(['--image-concurrency', str(plan.image_concurrency)])
    args.extend(['--prefill-chunk-size', str(plan.prefill_chunk)])
    if plan.expert_scale_compression is not None:
        args.extend(['--expert-scale-compression', plan.expert_scale_compression])
    if getattr(plan, 'prompt_logprobs', False):
        args.append('--prompt-logprobs')
    if plan.mixed_prefill:
        args.append('--mixed-prefill')
    if plan.recipe.speculation is not None:
        if (s := plan.speculation) is None:
            args.append('--no-mtp')
        else:
            if s.method is not None:
                args.extend(['--speculative-method', s.method])
            args.extend(['--mtp-draft-tokens', str(s.draft_tokens)])
            if s.mtp_only:
                args.append('--mtp-only')
            if not s.adaptive:
                args.append('--no-mtp-adaptive-verification')
            if s.draft_tokens > 0:
                if plan.speculative_token_capacity:
                    args.extend(['--speculative-token-capacity', str(plan.speculative_token_capacity)])
                if s.packed_verification and s.graph_budget is not None:
                    args.extend(['--speculative-graph-budget', str(s.graph_budget)])
                if s.draft_body_quantization is not None:
                    args.extend(['--draft-body-quantization', s.draft_body_quantization])
                if s.draft_head_quantization is not None:
                    args.extend(['--draft-head-quantization', s.draft_head_quantization])
                if s.markov_head_quantization is not None:
                    args.extend(['--markov-head-quantization', s.markov_head_quantization])
                if s.vocabulary_parallel_sampling:
                    args.append('--vocabulary-parallel-sampling')
    if plan.online_mxfp8:
        args.append('--online-mxfp8')
        if plan.online_mxfp8_scope:
            args.extend(['--online-mxfp8-scope', plan.online_mxfp8_scope])
    if plan.mla_compute_mode and plan.mla_compute_mode != 'auto':
        args.extend(['--mla-compute-mode', plan.mla_compute_mode])
    if plan.force_moe_decode_a16:
        args.append('--force-moe-decode-a16')
    if getattr(plan, 'native_dense_activations', False):
        args.append('--native-dense-activations')
    if plan.shard_hyperconnections:
        args.append('--shard-hyperconnections')
    if plan.row_shard_prefill:
        args.append('--row-shard-prefill')
    if plan.joint_tuning:
        args.append('--joint-tuning')
    if not plan.stream_overlap and plan.recipe.stream_overlap:
        args.append('--no-stream-overlap')
    if effort := getattr(plan, 'autotune_effort', ''):
        args.extend(['--autotune-effort', effort])
    if getattr(plan, 'lock_gpu_clocks', False):
        args.append('--lock-gpu-clocks')
    if not compiled_forwards:
        args.append('--no-compile')
    return args


SCOPE_DEPLOYMENT = 'PREPARATION_PARAMETER_SCOPE_DEPLOYMENT'
SCOPE_COMBINE = 'PREPARATION_PARAMETER_SCOPE_COMBINE'
SCOPE_TARGET_WEIGHT = 'PREPARATION_PARAMETER_SCOPE_TARGET_WEIGHT'
SCOPE_DRAFT_WEIGHT = 'PREPARATION_PARAMETER_SCOPE_DRAFT_WEIGHT'


def _preparation_value(value):
    if isinstance(value, bool):
        return {'bool_value': value}
    if isinstance(value, int):
        return {'uint_value': value} if value >= 0 else {'int_value': value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('preparation parameter float value must be finite')
        return {'float_value': value}
    if isinstance(value, str):
        return {'string_value': value}
    if isinstance(value, (list, tuple)):
        return {'list_value': {'values': [_preparation_value(item) for item in value]}}
    raise TypeError(f'unsupported preparation parameter value type: {type(value).__name__}')


def from_plan(model, plan, *, compiled_forwards, multi_host=False):
    from .source_identity import source_identity

    entries: dict[str, tuple[str, object]] = {}

    def add(key, scope, value):
        if value is None:
            return
        if key in entries:
            raise ValueError(f'duplicate preparation parameter key {key!r}')
        entries[key] = (scope, value)

    # Target-model weight preparation semantics (must agree across combine and --reuse-weights).
    online_mxfp8_scope = plan.online_mxfp8_scope or 'all'
    add('online_mxfp8', SCOPE_TARGET_WEIGHT, bool(plan.online_mxfp8))
    add('online_mxfp8_scope', SCOPE_TARGET_WEIGHT, online_mxfp8_scope)
    if plan.expert_scale_compression is not None:
        add('expert_scale_compression', SCOPE_TARGET_WEIGHT, plan.expert_scale_compression)

    # Shared core-combination semantics (must agree across combined deployments).
    add('language_model_only', SCOPE_COMBINE, bool(plan.language_model_only))
    add('mla_compute_mode', SCOPE_COMBINE, plan.mla_compute_mode)

    # Deployment-local Ignition options (preserved for inspection and reproduction;
    # ignored by core combination and cached-weight reuse).
    add('autotune_effort', SCOPE_DEPLOYMENT, getattr(plan, 'autotune_effort', ''))
    add('compiled_forwards', SCOPE_DEPLOYMENT, bool(compiled_forwards))
    add('context_parallel', SCOPE_DEPLOYMENT, bool(plan.context_parallel))
    add(
        'context_parallel_size',
        SCOPE_DEPLOYMENT,
        (plan.context_parallel_size or plan.world_size) if plan.context_parallel else 0,
    )
    add('force_moe_decode_a16', SCOPE_DEPLOYMENT, bool(plan.force_moe_decode_a16))
    add('image_concurrency', SCOPE_DEPLOYMENT, int(plan.image_concurrency))
    add('joint_tuning', SCOPE_DEPLOYMENT, bool(plan.joint_tuning))
    add('lock_gpu_clocks', SCOPE_DEPLOYMENT, bool(getattr(plan, 'lock_gpu_clocks', False)))
    add('max_concurrency', SCOPE_DEPLOYMENT, int(plan.concurrency))
    add('mixed_prefill', SCOPE_DEPLOYMENT, bool(plan.mixed_prefill))
    add(
        'native_dense_activations',
        SCOPE_DEPLOYMENT,
        bool(getattr(plan, 'native_dense_activations', False)),
    )
    add('prefill_chunk', SCOPE_DEPLOYMENT, int(plan.prefill_chunk))
    add('prompt_logprobs', SCOPE_DEPLOYMENT, bool(getattr(plan, 'prompt_logprobs', False)))
    add('reserve_bytes', SCOPE_DEPLOYMENT, int(plan.reserve_bytes))
    add('roce_chain', SCOPE_DEPLOYMENT, bool(plan.roce_chain))
    add('roce_ring', SCOPE_DEPLOYMENT, bool(plan.roce_ring))
    add('row_shard_prefill', SCOPE_DEPLOYMENT, bool(plan.row_shard_prefill))
    add('shard_hyperconnections', SCOPE_DEPLOYMENT, bool(plan.shard_hyperconnections))
    add('stream_overlap', SCOPE_DEPLOYMENT, bool(plan.stream_overlap))
    add('tensor_parallel', SCOPE_DEPLOYMENT, int(plan.world_size))

    if (s := plan.speculation) is not None:
        if s.packed_verification and s.draft_tokens > 0 and s.graph_budget is None:
            raise ValueError('packed speculative preparation metadata requires an explicit graph budget')

        # Draft-model weight semantics. Ignition resolves effective DFlash2 inheritance
        # before export so Rust compatibility checks remain generic.
        add('speculation.has_draft', SCOPE_DRAFT_WEIGHT, True)
        effective_draft_body = s.draft_body_quantization
        if (
            effective_draft_body is None
            and s.method == 'dflash2'
            and plan.online_mxfp8
            and online_mxfp8_scope == 'all'
        ):
            effective_draft_body = 'mxfp8'
        if effective_draft_body is not None:
            add('speculation.draft_body_quantization', SCOPE_DRAFT_WEIGHT, effective_draft_body)
        if s.draft_head_quantization is not None:
            add('speculation.draft_head_quantization', SCOPE_DRAFT_WEIGHT, s.draft_head_quantization)
        if s.markov_head_quantization is not None:
            add('speculation.markov_head_quantization', SCOPE_DRAFT_WEIGHT, s.markov_head_quantization)

        # Shared speculative combination semantics.
        add(
            'speculation.vocabulary_parallel_sampling',
            SCOPE_COMBINE,
            bool(s.vocabulary_parallel_sampling),
        )

        # Deployment-local speculative options.
        add('speculation.adaptive', SCOPE_DEPLOYMENT, bool(s.adaptive))
        if s.draft_prefix_selection:
            add('speculation.draft_prefix_selection', SCOPE_DEPLOYMENT, s.draft_prefix_selection)
        add('speculation.draft_tokens', SCOPE_DEPLOYMENT, int(s.draft_tokens))
        if s.graph_budget is not None:
            add('speculation.graph_budget', SCOPE_DEPLOYMENT, int(s.graph_budget))
        if s.method is not None:
            add('speculation.method', SCOPE_DEPLOYMENT, s.method)
        add('speculation.mtp_only', SCOPE_DEPLOYMENT, bool(s.mtp_only))
        add('speculation.packed_verification', SCOPE_DEPLOYMENT, bool(s.packed_verification))
        add('speculation.token_capacity', SCOPE_DEPLOYMENT, int(plan.speculative_token_capacity))

    parameters = [
        {'key': key, 'scope': scope, 'value': _preparation_value(value)}
        for key, (scope, value) in sorted(entries.items())
    ]
    return {
        'version': 1,
        'recipe': plan.recipe.name,
        'source_identity': source_identity(),
        'checkpoint_repository': plan.checkpoint.repository,
        'checkpoint_revision': plan.checkpoint.revision if model.directory is None else '',
        'local_checkpoint': model.directory is not None,
        'reproduction_args': reproduction_arguments(
            model, plan, compiled_forwards=compiled_forwards, multi_host=multi_host
        ),
        'parameters': parameters,
    }


def validate_preparation_manifest(manifest):
    """Validate `manifest` against Python and native `PreparationManifest` schemas."""
    import json
    from google.protobuf import json_format

    if not isinstance(manifest, Mapping):
        raise ValueError('preparation manifest must be a mapping')
    reproduction = manifest.get('reproduction_args')
    if not isinstance(reproduction, (list, tuple)) or not reproduction or any(
            not isinstance(arg, str) or not arg for arg in reproduction):
        raise ValueError('reproduction_args entries must be non-empty')
    try:
        parsed = json_format.ParseDict(dict(manifest), proto.PreparationManifest())
    except json_format.ParseError as error:
        raise ValueError(f'invalid preparation manifest: {error}') from error
    if parsed.version != 1:
        raise ValueError(f'unsupported preparation manifest version {parsed.version}')
    keys = [parameter.key for parameter in parsed.parameters]
    if keys != sorted(set(keys)):
        raise ValueError('preparation parameters must have unique keys in strictly ascending order')
    if 'speculation.history_drafts' in keys:
        raise ValueError('history_drafts is a runtime option and cannot be exported')
    try:
        import flare_heliopause
    except ImportError:
        return
    validator = getattr(flare_heliopause, 'export_validate_preparation_manifest', None)
    if validator is None:
        raise ValueError(
            'installed flare_heliopause lacks export_validate_preparation_manifest; rebuild ./heliopause'
        )
    validator(json.dumps(dict(manifest)))


def weight_views(roots):
    """Walk named weight owners without evaluating properties, closures, or module code.

    Paths identify fields within the recorded source identity. Aliased tensors keep
    each field name; cycles in container ownership do not recurse indefinitely.
    """
    import torch

    def walk(value, name, owner, ancestors):
        if isinstance(value, torch.Tensor):
            yield name, owner, value
            return
        if value is None or id(value) in ancestors:
            return
        ancestry = ancestors | {id(value)}
        if isinstance(value, Mapping):
            children = ((f'[{key!r}]', child) for key, child in value.items()
                        if isinstance(key, (str, int)))
        elif isinstance(value, (tuple, list)):
            children = ((f'[{index}]', child) for index, child in enumerate(value))
        elif isinstance(value, torch.nn.Module):
            owner = f'{type(value).__module__}.{type(value).__qualname__}'
            children = ((f'.{key}', child) for group in (value._parameters, value._buffers, value._modules)
                        for key, child in group.items())
        elif is_dataclass(value) and not isinstance(value, type):
            owner = f'{type(value).__module__}.{type(value).__qualname__}'
            children = ((f'.{field.name}', getattr(value, field.name)) for field in fields(value))
        elif type(value).__module__.startswith(('b12x.', 'ignition.')) and hasattr(value, '__dict__'):
            owner = f'{type(value).__module__}.{type(value).__qualname__}'
            children = ((f'.{key}', child) for key, child in vars(value).items())
        else:
            return
        for suffix, child in children:
            yield from walk(child, name + suffix, owner, ancestry)

    for name, value in sorted(roots.items()):
        yield from walk(value, name, '', set())


def usable_weight_tensor(tensor):
    """True for a view that has materialized, non-empty, non-negative strides."""
    import torch

    return (tensor.device.type != 'meta' and tensor.layout == torch.strided
            and tensor.numel() and not any(stride < 0 for stride in tensor.stride()))


def weight_payload_declaration(declaration):
    """Return the thin declaration Heliopause resolves against retained storage."""
    views = []
    for name, _owner, tensor, group, storage in declared_views(declaration):
        if not usable_weight_tensor(tensor):
            continue
        entry = dict(name=name, group=group, storage=storage,
                     element_bytes=tensor.element_size(), shape=list(tensor.shape),
                     stride=list(tensor.stride()), address=tensor.data_ptr())
        from . import provenance

        recorded = provenance.lookup(tensor)
        if recorded is not None:
            entry['sources'] = recorded['sources']
            entry['packing'] = recorded['packing']
            if recorded.get('packing_parameters') is not None:
                entry['packing_parameters'] = recorded['packing_parameters']
            entry['statistics_domain'] = recorded['statistics_domain']
        views.append(entry)
    by_name = {entry['name']: entry for entry in views}
    for alias, canonical in declaration.aliases:
        if alias not in by_name:
            raise ValueError(f'declared weight alias {alias!r} is not a declared view')
        if canonical not in by_name:
            raise ValueError(f'declared weight alias target {canonical!r} is not a declared view')
        source, target = by_name[alias], by_name[canonical]
        if (source['address'], source['element_bytes'], source['shape'], source['stride']) != (
                target['address'], target['element_bytes'], target['shape'], target['stride']):
            raise ValueError(f'declared weight alias {alias!r} does not share {canonical!r} storage')
        source['alias_of'] = canonical
    return dict(version=1, views=views)


def weight_payload_request(roots):
    """Return the thin declaration Heliopause resolves, or None for legacy roots."""
    if not isinstance(roots, WeightDeclaration):
        return None
    return weight_payload_declaration(roots)


def _declared_metadata(declaration):
    import torch

    tensors, unavailable = [], []
    for name, owner, tensor, _group, _storage in declared_views(declaration):
        if not usable_weight_tensor(tensor):
            unavailable.append(name)
            continue
        extent = tensor.element_size() * (1 + sum((size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)))
        tensors.append(dict(name=name, dtype=str(tensor.dtype).removeprefix('torch.'),
            element_bytes=tensor.element_size(), shape=list(tensor.shape), stride=list(tensor.stride()),
            layout_type=owner, address=tensor.data_ptr(), extent=extent))
    return dict(version=1, tensors=tensors, unavailable=unavailable)


def rebuild_weights_metadata(roots):
    """Describe tensor views for native resolution against retained storage."""
    import torch

    if isinstance(roots, WeightDeclaration):
        return _declared_metadata(roots)

    tensors, unavailable = [], []
    for name, owner, tensor in weight_views(roots):
        if (tensor.device.type == 'meta' or tensor.layout != torch.strided or not tensor.numel()
                or any(stride < 0 for stride in tensor.stride())):
            unavailable.append(name)
            continue
        extent = tensor.element_size() * (1 + sum((size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)))
        tensors.append(dict(name=name, dtype=str(tensor.dtype).removeprefix('torch.'),
            element_bytes=tensor.element_size(), shape=list(tensor.shape), stride=list(tensor.stride()),
            layout_type=owner, address=tensor.data_ptr(), extent=extent))
    return dict(version=1, tensors=tensors, unavailable=unavailable)


def validate_weights(bundle):
    """Check catalog references before a preparation tool uses them; serving ignores this metadata."""
    if not bundle.HasField('rebuild_weights'):
        return
    catalog = bundle.rebuild_weights
    if catalog.version != 1:
        raise ValueError(f'unsupported prepared-weight catalog version {catalog.version}')
    allocations = {a.name: a for a in bundle.allocations}
    names = set()
    widths = {'bool': 1, 'uint8': 1, 'int8': 1, 'int16': 2, 'uint16': 2, 'float16': 2,
              'bfloat16': 2, 'int32': 4, 'uint32': 4, 'float32': 4, 'int64': 8, 'uint64': 8,
              'float64': 8, 'complex32': 4, 'complex64': 8, 'complex128': 16,
              'float8_e4m3fn': 1, 'float8_e5m2': 1, 'float8_e4m3fnuz': 1,
              'float8_e5m2fnuz': 1, 'float8_e8m0fnu': 1}
    for tensor in catalog.tensors:
        if (not tensor.name or tensor.name in names or not tensor.HasField('storage')
                or tensor.element_bytes != widths.get(tensor.dtype)
                or len(tensor.shape) != len(tensor.stride) or not math.prod(tensor.shape)):
            raise ValueError('invalid prepared-weight tensor description')
        names.add(tensor.name)
        extent = tensor.element_bytes * (1 + sum((size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride, strict=True)))
        allocation = allocations.get(tensor.storage.allocation)
        if (allocation is None or allocation.HasField('sizing') or allocation.memory == proto.PEER
                or tensor.storage.offset + extent > allocation.nbytes):
            raise ValueError(f'prepared-weight tensor {tensor.name!r} exceeds fixed stored allocation')
    for name in catalog.unavailable:
        if not name or name in names:
            raise ValueError('duplicate or empty unavailable prepared-weight name')
        names.add(name)
