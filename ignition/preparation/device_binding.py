# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bind single-device compiled forwards to a tensor-parallel rank's CUDA ordinal."""

import ast
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import hashlib
import re

import torch

_DEVICE = ContextVar("flare_compiled_device", default=None)
_FREEZE_TUNING = ContextVar("flare_compiled_tuning_winners", default=False)
_LINE_BREAK_RE = re.compile(rb"\r\n|\r|\n")


def _line_byte_offsets(data: bytes) -> list[int]:
    offsets = [0]
    for match in _LINE_BREAK_RE.finditer(data):
        offsets.append(match.end())
    return offsets


def _call_target(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        value = func.value
        if isinstance(value, ast.Name):
            return f"{value.id}.{func.attr}"
        if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            return f"{value.value.id}.{value.attr}.{func.attr}"
    return None


@dataclass(frozen=True)
class _KernelTemplate:
    segments: tuple[str, ...]

    def render(self, target: int) -> str:
        return str(target).join(self.segments)


@dataclass(frozen=True)
class _SourceTemplate:
    segments: tuple[str, ...]
    slots: tuple[_KernelTemplate | None, ...]
    assigned_kernels: tuple[tuple[str, _KernelTemplate, str], ...]
    invalid_kernel_assignment: bool
    current_device: int
    source_sha256: bytes

    def render(self, target: int, *, current: int | None = None, kernel_keys: dict | None = None) -> str:
        if not self.slots:
            return self.segments[0]
        from torch._inductor.async_compile import CompiledTritonKernels

        target_text = str(target)
        pieces = [self.segments[0]]
        for slot, tail in zip(self.slots, self.segments[1:]):
            if slot is None:
                pieces.append(target_text)
            else:
                rebound = slot.render(target)
                if kernel_keys is not None and current is not None:
                    kernel_keys[CompiledTritonKernels.key(slot.render(current))] = CompiledTritonKernels.key(rebound)
                pieces.append(repr(rebound))
            pieces.append(tail)
        return "".join(pieces)


def _build_kernel_template(source: str, original: int) -> _KernelTemplate:
    tree = ast.parse(source)
    data = source.encode()
    offsets = _line_byte_offsets(data)
    spans = []

    def mark(node):
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        spans.append((start, end))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_target(node.func)
        if name in ("torch.cuda.set_device", "torch.cuda._DeviceGuard", "get_raw_stream"):
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or node.args[0].value != original:
                raise ValueError(f"unsupported compiled CUDA device binding: {ast.unparse(node)}")
            mark(node.args[0])
        elif name == "DeviceProperties":
            for keyword in node.keywords:
                if keyword.arg == "index":
                    if not isinstance(keyword.value, ast.Constant) or keyword.value.value != original:
                        raise ValueError("compiled kernel spans multiple CUDA devices")
                    mark(keyword.value)
        elif name == "async_compile.triton":
            raise ValueError("nested async_compile.triton calls are unsupported inside kernel source")

    if not spans:
        return _KernelTemplate((source,))
    segments = []
    cursor = 0
    for start, end in sorted(spans):
        if start < cursor:
            raise ValueError("overlapping CUDA device bindings in generated kernel source")
        segments.append(data[cursor:start].decode())
        cursor = end
    segments.append(data[cursor:].decode())
    return _KernelTemplate(tuple(segments))


def build_source_template(source: str, original: int) -> _SourceTemplate:
    """Parse a generated Inductor wrapper once into a reusable device-relocation template."""
    from torch._inductor.async_compile import CompiledTritonKernels

    tree = ast.parse(source)
    data = source.encode()
    offsets = _line_byte_offsets(data)
    replacements = []
    kernel_templates_by_call = {}

    def add_slot(node, slot):
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        replacements.append((start, end, slot))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_target(node.func)
        if name in ("torch.cuda.set_device", "torch.cuda._DeviceGuard", "get_raw_stream"):
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or node.args[0].value != original:
                raise ValueError(f"unsupported compiled CUDA device binding: {ast.unparse(node)}")
            add_slot(node.args[0], None)
        elif name == "DeviceProperties":
            for keyword in node.keywords:
                if keyword.arg == "index":
                    if not isinstance(keyword.value, ast.Constant) or keyword.value.value != original:
                        raise ValueError("compiled kernel spans multiple CUDA devices")
                    add_slot(keyword.value, None)
        elif name == "async_compile.triton":
            if len(node.args) < 2:
                raise ValueError(f"unsupported async_compile.triton call: {ast.unparse(node)}")
            kernel = node.args[1]
            kernel_source = (kernel.value if isinstance(kernel, ast.Constant) and isinstance(kernel.value, str)
                             else ast.literal_eval(kernel))
            if not isinstance(kernel_source, str):
                raise ValueError("async_compile.triton kernel source must evaluate to a string")
            k_template = _build_kernel_template(kernel_source, original)
            kernel_templates_by_call[id(node)] = k_template
            add_slot(kernel, k_template)

    assigned_with_pos = []
    invalid_kernel_assignment = False
    bound_call_ids = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and id(node.value) in kernel_templates_by_call):
            continue
        bound_call_ids.add(id(node.value))
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            invalid_kernel_assignment = True
            continue
        k_template = kernel_templates_by_call[id(node.value)]
        identity = CompiledTritonKernels.key(k_template.render(0))
        assigned_with_pos.append(((node.lineno, node.col_offset), (node.targets[0].id, k_template, identity)))
    if len(bound_call_ids) != len(kernel_templates_by_call):
        invalid_kernel_assignment = True
    assigned_kernels = [item for _, item in sorted(assigned_with_pos, key=lambda entry: entry[0])]

    if not replacements:
        segments = (source,)
        slots = ()
    else:
        ordered = sorted(replacements, key=lambda item: (item[0], item[1]))
        segments_list = []
        slots_list = []
        cursor = 0
        for start, end, slot in ordered:
            if start < cursor:
                raise ValueError("overlapping CUDA device bindings in generated wrapper source")
            segments_list.append(data[cursor:start].decode())
            slots_list.append(slot)
            cursor = end
        segments_list.append(data[cursor:].decode())
        segments = tuple(segments_list)
        slots = tuple(slots_list)

    return _SourceTemplate(
        segments=segments,
        slots=slots,
        assigned_kernels=tuple(assigned_kernels),
        invalid_kernel_assignment=invalid_kernel_assignment,
        current_device=original,
        source_sha256=hashlib.sha256(data).digest(),
    )


def get_source_template(graph, device_index: int) -> _SourceTemplate:
    template = getattr(graph, "_flare_source_template", None)
    if (isinstance(template, _SourceTemplate)
            and template.current_device == device_index
            and hashlib.sha256(graph.source_code.encode()).digest() == template.source_sha256):
        return template
    template = build_source_template(graph.source_code, device_index)
    try:
        graph._flare_source_template = template
    except AttributeError:
        pass
    return template


@contextmanager
def bind_device(device, *, freeze_tuning=False):
    if type(freeze_tuning) is not bool:
        raise TypeError("freeze_tuning must be a boolean")
    token = _DEVICE.set(torch.device(device))
    tuning_token = _FREEZE_TUNING.set(freeze_tuning)
    try:
        yield
    finally:
        _FREEZE_TUNING.reset(tuning_token)
        _DEVICE.reset(token)


def restore_device(kind, index):
    target = _DEVICE.get()
    return target if kind == "cuda" and target is not None else torch.device(kind, index)


def restore_prepared_device(kind, state):
    value = object.__new__(kind)
    target = _DEVICE.get()
    if target is not None:
        state["ordinal"] = target.index
        if isinstance(state.get("uuid"), str):
            state["uuid"] = str(torch.cuda.get_device_properties(target).uuid).strip()
    value.__dict__.update(state)
    return value


def _device_source(source, original, target, kernel_keys=None):
    template = build_source_template(source, original)
    return template.render(target, current=original, kernel_keys=kernel_keys)


def _freeze_tuning_winners(bundle):
    """Preserve completed compile-time selections while rebuilding launchers.

    Torch rechecks a serialized static winner as a one-config cache entry and
    can reopen occupancy search when it rebuilds launchers. Prepared forwards
    retain the selected config; all search and optional refinement must finish
    before serialization.
    """
    # Torch loads duplicate cache keys in list order. Only the final snapshot
    # for each key is effective; preceding snapshots can predate autotuning.
    entries = {entry.cache_key: entry for entry in bundle.static_autotuners}
    for entry in entries.values():
        kernel = entry.kernel
        if len(kernel.compile_results) != 1 or not kernel.is_statically_launchable():
            raise ValueError(f"compiled kernel {entry.kernel_name} has unresolved tuning candidates")
        result, = kernel.compile_results
        config = result.config
        if hasattr(result, '_flare_config_state'):
            config.__dict__.clear()
            config.__dict__.update(result._flare_config_state)
        metadata = kernel.inductor_meta
        if (metadata.get("coordinate_descent_tuning", False)
                and kernel._should_coordesc_tune
                and not getattr(config, "found_by_coordesc", False)):
            raise ValueError(f"compiled kernel {entry.kernel_name} has pending coordinate-descent tuning")
        if (metadata.get("combo_tuning_groups")
                and not getattr(config, "found_by_combo_autotune", False)):
            raise ValueError(f"compiled kernel {entry.kernel_name} has pending combo tuning")
        kernel.inductor_meta = {**metadata, "dynamic_scale_rblock": False,
                                "coordinate_descent_tuning": False}
        kernel.__dict__.pop("_could_rblock_scale", None)
    bundle.static_autotuners[:] = entries.values()


def relocate_inductor(entry, original, *, freeze_tuning=False):
    target = _DEVICE.get()
    relocate = target is not None and target != original
    freeze_tuning = freeze_tuning or _FREEZE_TUNING.get()
    if not relocate and not freeze_tuning:
        return
    from torch._inductor.codecache import code_hash
    from torch._inductor.output_code import CompiledFxGraph
    from torch.utils._ordered_set import OrderedSet

    graph = entry.compiled_fw.result
    if entry.compiled_bw is not None or not isinstance(graph, CompiledFxGraph):
        raise ValueError("device binding requires a compiled inference forward")
    if graph.device_idxs != OrderedSet((original.index,)):
        raise ValueError("compiled forward must use one CUDA device")
    if freeze_tuning and graph._triton_bundle is not None:
        _freeze_tuning_winners(graph._triton_bundle)
    if not relocate:
        return
    template = get_source_template(graph, original.index)
    kernel_keys = {}
    graph.source_code = template.render(target.index, current=original.index, kernel_keys=kernel_keys)
    graph.cache_key = code_hash(graph.source_code)
    graph.device_idxs = OrderedSet((target.index,))
    graph._flare_source_template = replace(
        template,
        current_device=target.index,
        source_sha256=hashlib.sha256(graph.source_code.encode()).digest(),
    )
    if bundle := graph._triton_bundle:
        autotuners = []
        for entry_kernel in bundle.static_autotuners:
            rebound_key = kernel_keys.get(entry_kernel.cache_key)
            if rebound_key is None:
                if entry_kernel.kernel_name in graph.source_code:
                    raise ValueError("compiled forward has unbound static Inductor kernels")
                continue
            kernel = entry_kernel.kernel
            kernel.device_props = kernel.device_props._replace(index=target.index)
            kernel.triton_meta["device"] = target.index
            for result in kernel.compile_results:
                result.compile_meta["device"] = target.index
            autotuners.append(replace(entry_kernel, cache_key=rebound_key))
        graph._triton_bundle = replace(bundle,
            kernel_artifacts=[replace(item, device=target.index) for item in bundle.kernel_artifacts],
            static_autotuners=autotuners)
