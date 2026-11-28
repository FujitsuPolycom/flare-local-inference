# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Serializable compiled forwards with caller-owned storage and runtime handles."""

import hashlib
import importlib
import inspect
import io
import operator
import os
import pickle
import sys
import tempfile
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from threading import RLock
from types import MappingProxyType

import torch
from filelock import FileLock
from torch._dynamo.aot_compile_types import SerializableCallable
from torch._dynamo.guards import GuardsStatePickler, _Missing
from torch.fx.experimental.proxy_tensor import make_fx
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
from torch.utils._pytree import tree_flatten, tree_unflatten, treespec_dumps, treespec_loads

from ignition import runtime_patches

from .compilation import (DEFAULT_CACHE_ROOT, _INVOCATIONS, _OPTIONS, _Invocation, _artifact_kernel_table,
                          _byte_storage, _compile_options, _storage_region, _lower_storage, _compiler_recursion)
from .device_binding import (bind_device, get_source_template, relocate_inductor, restore_device,
                             restore_prepared_device)
from .host_memory import release_unused

_GUARD_LOCK = RLock()
_GUARD_TENSOR_MODE = ContextVar("flare_guard_tensor_mode", default=None)
_DEFERRED_BINDINGS = ContextVar("flare_deferred_bindings", default=False)
_CHECK_BINDINGS = ContextVar("flare_check_bindings", default=False)
_RETAIN_SERIALIZED_ARTIFACT = ContextVar("flare_retain_serialized_artifact", default=True)
_DEFER_PORTABLE_COMPRESSION = ContextVar("flare_defer_portable_compression", default=False)
_PREFETCHED_ARTIFACT_BYTES = ContextVar("flare_prefetched_artifact_bytes", default=None)


class _RawPortableArtifact(bytes):
    """Uncompressed pickled Inductor bundle retained only until warmup freezes and saves it."""


def _write_compiled_with_digest(target, data):
    from torch._inductor.codecache import write_atomic

    write_atomic(target, data, True)
    return target, hashlib.sha256(data).hexdigest()


def _save_compiled_async(artifact, path, executor=None, futures=None):
    serialize = getattr(type(artifact), "serialize", None)
    if (executor is not None and futures is not None
            and callable(serialize)
            and type(artifact).__module__.startswith("torch.")
            and "save_compiled_function" not in getattr(artifact, "__dict__", {})):
        data = serialize(artifact).serialized_data
        futures.append(executor.submit(_write_compiled_with_digest, str(path), data))
        return
    artifact.save_compiled_function(str(path))


class _BytePrefetcher:
    """Read upcoming compiled forward artifacts in the background with bounded lookahead."""

    def __init__(self, items, *, window=2):
        self._items = iter(items)
        self._window = max(1, window)
        self._executor = None
        self._inflight = {}

    def __enter__(self):
        self._fill()
        return self

    def _fill(self):
        while len(self._inflight) < self._window:
            try:
                key, path = next(self._items)
            except StopIteration:
                break
            if key not in self._inflight:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=min(2, self._window))
                self._inflight[key] = self._executor.submit(Path(path).read_bytes)

    def pop(self, key):
        future = self._inflight.pop(key, None)
        self._fill()
        if future is None:
            return None
        try:
            return future.result()
        except OSError:
            return None

    def __exit__(self, *exc):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)


def _deserialize_bundled_entry(entry):
    from torch._functorch._aot_autograd import aot_autograd_result

    impl = aot_autograd_result.deserialize_bundled_cache_entry
    if (getattr(impl, "__module__", None) != "torch._functorch._aot_autograd.aot_autograd_result"
            or getattr(impl, "__name__", None) != "deserialize_bundled_cache_entry"):
        return impl(entry)
    import copy
    from torch._functorch._aot_autograd.runtime_wrappers import SerializableCompiledFunction, simple_wraps
    from torch._inductor.cudagraph_utils import BoxedDeviceIndex
    from torch._inductor.utils import BoxedBool
    from torch._subclasses import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    entry.guards_expr = None
    cudagraphs = BoxedBool(torch._inductor.config.triton.cudagraphs)
    boxed_forward_device_index = BoxedDeviceIndex(None)
    serializable_copy = copy.copy(entry)
    if getattr(entry, "compiled_fw", None) is not None:
        copied_fw = copy.copy(entry.compiled_fw)
        if getattr(copied_fw, "result", None) is not None:
            copied_graph = copy.copy(copied_fw.result)
            if getattr(copied_graph, "_triton_bundle", None) is not None:
                copied_graph._triton_bundle = replace(copied_graph._triton_bundle, kernel_artifacts=[])
            copied_fw.result = copied_graph
        serializable_copy.compiled_fw = copied_fw
    context = torch._guards.TracingContext.try_get()
    if context is None:
        context = torch._guards.TracingContext(FakeTensorMode(shape_env=ShapeEnv()))
    with torch._guards.tracing(context):
        compiled_fn = entry.wrap_post_compile(
            [],
            entry.sanitized_aot_config,
            {
                "cudagraphs": cudagraphs,
                "boxed_forward_device_index": boxed_forward_device_index,
            },
        )
    compiled_fn = SerializableCompiledFunction(compiled_fn, lambda: serializable_copy)

    @simple_wraps(compiled_fn)
    def forward(*runtime_args):
        return compiled_fn(list(runtime_args))

    if not hasattr(compiled_fn, "serialize"):
        raise AssertionError("compiled_fn must have serialize attribute")
    forward.serialize = compiled_fn.serialize
    return forward


def _load_inductor(data, original, *, frozen=False):
    runtime_patches.apply()
    from torch._functorch._aot_autograd.runtime_wrappers import SerializableCompiledFunction
    from torch._inductor.standalone_compile import AOTCompiledArtifact

    entry, kernels, constants = pickle.loads(data)
    if kernels or constants:
        raise ValueError("compiled native launches must use runtime kernel bindings")
    # Torch copies graph attributes into runtime wrappers during deserialization.
    entry.compiled_fw.result.inductor_post_grad_graph_str = None
    entry.compiled_fw.result.cache_linemap = None
    relocate_inductor(entry, original, freeze_tuning=frozen)
    if frozen and entry.compiled_fw.result._triton_bundle is not None:
        from .inductor_tuning import _materialize_selected_cubin

        for tuner in entry.compiled_fw.result._triton_bundle.static_autotuners:
            for result in tuner.kernel.compile_results:
                _materialize_selected_cubin(result)
    if not _RETAIN_SERIALIZED_ARTIFACT.get():
        serialized = None
    elif _DEFER_PORTABLE_COMPRESSION.get() and not frozen:
        serialized = _RawPortableArtifact(
            pickle.dumps((entry, kernels, constants), protocol=pickle.HIGHEST_PROTOCOL))
    else:
        serialized = zlib.compress(
            pickle.dumps((entry, kernels, constants), protocol=pickle.HIGHEST_PROTOCOL), level=1)
    compiled = _deserialize_bundled_entry(entry)
    serializer = getattr(compiled.serialize, "__self__", None)
    if (not isinstance(serializer, SerializableCompiledFunction)
            or getattr(compiled, "__wrapped__", None) is not serializer
            or getattr(compiled, "serialize_fn", None) is not serializer.serialize_fn):
        raise RuntimeError("unsupported bundled Inductor serialization wrappers")
    if frozen and entry.compiled_fw.result._triton_bundle is not None:
        from .device_binding import _freeze_tuning_winners

        # Torch adds config defaults while rebuilding launchers from a static bundle.
        _freeze_tuning_winners(entry.compiled_fw.result._triton_bundle)
    # Loading materializes bundled files in Triton's cache. Keep the portable copy compressed;
    # live kernel owners retain their binaries, launchers, and argument metadata independently.
    for result in (entry, compiled.serialize()):
        graph = result.compiled_fw.result
        if graph._triton_bundle is not None:
            graph._triton_bundle.kernel_artifacts.clear()
    artifact = AOTCompiledArtifact(compiled)
    artifact._flare_graph = entry.compiled_fw.result
    # simple_wraps copies the serializer callback into the execution wrapper's dictionary.
    artifact._flare_serialization_owners = (compiled, serializer)
    return artifact, serialized


class _BindingsChecked(Exception):
    pass


class BindingMismatch(ValueError):
    """Prepared tensor storage differs from the compiled input contract."""


def _guard_object(cls):
    return object.__new__(cls)


class _GuardPickler(GuardsStatePickler):
    """Exclude unguarded kernel resources from prepared Python objects."""

    @classmethod
    def _unpickle_tensor(cls, meta_tensor, device, pytype, dispatch_keys_raw, grad, deferred=False):
        if deferred and _DEFERRED_BINDINGS.get():
            pytype = torch._subclasses.FakeTensor
        mode = _GUARD_TENSOR_MODE.get()
        if mode is None:
            return super()._unpickle_tensor(meta_tensor, device, pytype, dispatch_keys_raw, grad)
        value = mode.fake_tensor_converter.from_meta_and_device(
            mode, meta_tensor, device, pytype,
            torch._C.DispatchKeySet.from_raw_repr(dispatch_keys_raw))
        value.grad = grad
        return value

    def reducer_override(self, obj):
        from triton.runtime import JITFunction
        from torch._dynamo.aot_compile_types import _deserialize_triton_kernel, _serialize_triton_kernel

        if id(obj) in self.empty_values or id(obj) in self.missing_values:
            return super().reducer_override(obj)
        if isinstance(obj, torch.device):
            return restore_device, (obj.type, obj.index)
        if isinstance(obj, torch.Tensor) and obj.device.type != "meta":
            if id(obj) not in self.guard_tree_values:
                return _Missing, ('tensor guard tree',)
            if (type(obj) not in (torch.Tensor, torch.nn.Parameter, torch._subclasses.FakeTensor)
                    or obj.layout != torch.strided or obj.is_nested or obj.is_quantized):
                return super().reducer_override(obj)
            with torch._C._DisableTorchDispatch():
                meta = torch.empty_strided(obj.shape, obj.stride(), dtype=obj.dtype,
                                          device="meta", requires_grad=obj.requires_grad)
            from .forward_metadata import _TENSOR_KINDS, _set_state, _tensor_user_attributes
            if isinstance(obj, torch._subclasses.FakeTensor):
                kinds = _TENSOR_KINDS.get()
                kind = torch._subclasses.FakeTensor if kinds is None else kinds.get(obj, torch._subclasses.FakeTensor)
                deferred = kind is torch._subclasses.FakeTensor
                pytype = torch.Tensor if deferred else kind
            else:
                deferred = False
                pytype = type(obj)
            args = (meta, obj.device, pytype,
                    torch._C._dispatch_keys(obj).raw_repr(), obj.grad, deferred)
            user_attrs = _tensor_user_attributes(obj)
            if user_attrs:
                state = {key: value if id(value) in self.guard_tree_values
                         else _Missing('unguarded attribute')
                         for key, value in user_attrs.items()}
                return type(self)._unpickle_tensor, args, state, None, None, _set_state
            return type(self)._unpickle_tensor, args
        if isinstance(obj, JITFunction):
            return _deserialize_triton_kernel, (_serialize_triton_kernel(obj),)
        from .forward_metadata import _uses_custom_object_state

        if _uses_custom_object_state(obj):
            state = {key: value if id(value) in self.guard_tree_values
                     else _Missing('unguarded attribute')
                     for key, value in vars(obj).items()}
            from b12x.preparation.device import DetectedDevice

            if isinstance(obj, DetectedDevice):
                return restore_prepared_device, (type(obj), state)
            return _guard_object, (type(obj),), state
        return super().reducer_override(obj)


@contextmanager
def _guard_serialization():
    from torch._dynamo import guards

    with _GUARD_LOCK:
        original = guards.GuardsStatePickler
        guards.GuardsStatePickler = _GuardPickler
        try:
            yield
        finally:
            guards.GuardsStatePickler = original


@contextmanager
def _guard_code_cache(root):
    """Reuse guard code without retaining construction stacks in executable guards."""
    from torch._dynamo import guards, package

    with _GUARD_LOCK:
        original = guards.build_guard_function
        original_manager = package.load_guard_manager

        def load_manager(state, *args, **kwargs):
            for guard in state.output_graph._guards:
                guard.stack = None
                guard.user_stack = None
            original_parts = guards.get_verbose_code_parts

            def shared_parts(*args, **kwargs):
                return [sys.intern(part) for part in original_parts(*args, **kwargs)]

            guards.get_verbose_code_parts = shared_parts
            try:
                return original_manager(state, *args, **kwargs)
            finally:
                guards.get_verbose_code_parts = original_parts

        def build(code_parts, closure_args):
            key = hashlib.sha256(pickle.dumps((tuple(code_parts), closure_args))).hexdigest()
            path = root / (key + '.bin')
            if path.is_file():
                return pickle.loads(path.read_bytes())
            result = original(code_parts, closure_args)
            root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=root, suffix='.tmp', delete=False) as file:
                temporary = Path(file.name)
                try:
                    pickle.dump(result, file)
                    file.close()
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            return result

        token = _GUARD_TENSOR_MODE.set(torch._subclasses.FakeTensorMode())
        guards.build_guard_function = build
        package.load_guard_manager = load_manager
        try:
            yield
        finally:
            package.load_guard_manager = original_manager
            guards.build_guard_function = original
            _GUARD_TENSOR_MODE.reset(token)


def _signature(inputs):
    aliases, result = {}, []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            alias = aliases.setdefault(storage._cdata, len(aliases))
            result.append((tuple(map(int, value.shape)), tuple(map(int, value.stride())),
                           int(value.storage_offset()), value.dtype, value.device, alias))
        elif isinstance(value, (int, torch.SymInt)):
            result.append(int)
        else:
            raise TypeError(f'unsupported compiled input: {type(value)}')
    return tuple(result)


@dataclass
class _Input:
    index: int
    view: tuple | None = None
    region: tuple[int, int] | None = None

    def bind(self, args):
        value = args[self.index]
        if self.view is not None:
            dtype, shape, stride, offset = self.view
            storage = _byte_storage(value) if self.region is None else _storage_region(value, *self.region)
            value = storage.view(dtype).as_strided(shape, stride, offset)
        return value


def _share_storage_descriptors(inputs, signature, calls):
    """Share immutable storage descriptors within one deserialized forward."""
    integers, tuples = {}, {}

    def share(value):
        kind = type(value)
        if kind is int:
            return int, integers.setdefault(value, value)
        if kind in (bool, type(None), torch.dtype, torch.device):
            return kind, value
        if kind is not tuple:
            return None, value
        parts = [share(part) for part in value]
        if any(kind is None for kind, _ in parts):
            return None, value
        kinds = tuple(kind for kind, _ in parts)
        result = tuple(part for _, part in parts)
        pool = tuples.setdefault(kinds, {})
        return (tuple, kinds), pool.setdefault(result, result)

    for item in inputs:
        item.index = share(item.index)[1]
        item.view = share(item.view)[1]
        item.region = share(item.region)[1]
    signature = tuple(share(item)[1] for item in signature)
    for _, _, _, operands, positions in calls:
        for index, operand in enumerate(operands):
            operands[index] = share(operand)[1]
        for index, position in enumerate(positions):
            positions[index] = share(position)[1]
    return signature


def _serialized_triton_launch(kernel, constants, arguments):
    """Preserve one traced launch config without serializing an autotuner."""
    from triton.runtime import JITFunction
    from triton.runtime.autotuner import Autotuner, Config

    if isinstance(kernel, JITFunction):
        return kernel, dict(constants)
    if type(kernel) is not Autotuner or len(kernel.configs) != 1 or not isinstance(kernel.fn, JITFunction):
        raise ValueError("opaque Triton serialization requires a JIT function or one fixed launch config")
    config = kernel.configs[0]
    if (type(config) is not Config or config.pre_hook is not None
            or kernel.user_defined_pre_hook or kernel.user_defined_post_hook
            or kernel.reset_to_zero or kernel.restore_value
            or kernel.early_config_prune is not None or kernel.perf_model is not None
            or kernel._do_bench is not None or kernel.num_warmups is not None
            or kernel.num_reps is not None or kernel.use_cuda_graph):
        raise ValueError("opaque Triton serialization cannot discard autotuner hooks or tuning behavior")
    options = config.all_kwargs()
    conflicts = options.keys() & (constants.keys() | arguments.keys())
    if conflicts:
        raise ValueError("opaque Triton launch config conflicts with arguments: " + ", ".join(sorted(conflicts)))
    return kernel.fn, {**constants, **options}


def _deserialized_triton_launch(kernel_info):
    """Resolve the raw JIT whose fixed options were serialized separately."""
    from torch._dynamo.aot_compile_types import _deserialize_triton_kernel, _serialize_triton_kernel
    from triton.runtime import JITFunction
    from triton.runtime.autotuner import Autotuner, Heuristics

    kernel = _deserialize_triton_kernel(kernel_info)
    # Import restores the module global, including decorators removed from the
    # traced launch. Reapplying them can tune again or duplicate fixed options.
    seen = set()
    while type(kernel) in (Autotuner, Heuristics):
        if id(kernel) in seen:
            raise ValueError("serialized fixed Triton launch has cyclic decorators")
        seen.add(id(kernel))
        kernel = kernel.fn
    if not isinstance(kernel, JITFunction):
        raise ValueError("serialized fixed Triton launch does not resolve to a JIT function")
    if _serialize_triton_kernel(kernel) != kernel_info:
        raise ValueError("serialized fixed Triton launch resolves to a different function")
    return kernel


def _serialized_invocations(invocations):
    """Record opaque launches without their process-local Triton table indices."""
    from torch._library.custom_ops import _maybe_get_opdef
    from torch._dynamo.aot_compile_types import _serialize_triton_kernel

    calls = []
    for call in invocations:
        target = call.target
        leaves = call.leaves
        if target is torch.ops.higher_order.triton_kernel_wrapper_mutation:
            args, kwargs = tree_unflatten(leaves, call.tree)
            kwargs = dict(kwargs)
            kernel = kernel_side_table.get_kernel(kwargs['kernel_idx'])
            constants = kernel_side_table.get_constant_args(kwargs['constant_args_idx'])
            kernel, constants = _serialized_triton_launch(kernel, constants, kwargs['kwargs'])
            name = ('triton', _serialize_triton_kernel(kernel), constants)
            kwargs['kernel_idx'] = kwargs['constant_args_idx'] = None
            leaves, _ = tree_flatten((args, kwargs))
        elif isinstance(target, torch._ops.OpOverload):
            definition = _maybe_get_opdef(target)
            registration = None if definition is None else definition._init_fn.__module__
            name = ('op', target.namespace, target._opname, target._overloadname, registration)
        else:
            raise TypeError(f'cannot serialize native operator: {target}')
        calls.append((name, leaves, treespec_dumps(call.tree), call.operands, call.integers))
    return calls


class StorageBackend(SerializableCallable):
    def __init__(self, module, examples, *, options=None):
        runtime_patches.apply()
        from torch._guards import detect_fake_mode
        from torch._inductor import standalone_compile

        module = make_fx(module, tracing_mode='fake', _allow_non_fake_inputs=True,
                         _disable_torch_fn_metadata_mode=True)(*examples)
        graph = module.graph
        arguments = [node for node in graph.nodes if node.op == 'placeholder']
        examples = [node.meta['val'] for node in arguments]
        self.signature = _signature(examples)
        storages, scalar_inputs = {}, {}
        for index, node in enumerate(arguments):
            value = node.meta['val']
            if not isinstance(value, torch.Tensor):
                scalar_inputs[node] = _Input(index)
                continue
            storages.setdefault(value.untyped_storage()._cdata, index)
            name = f'_input_{index}'
            module.register_buffer(name, value)
            with graph.inserting_before(node):
                replacement = graph.get_attr(name)
            replacement.meta = node.meta.copy()
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)

        self.invocations = _lower_storage(module)
        bindings = dict(scalar_inputs)
        for node in list(graph.nodes):
            if node.op != 'get_attr':
                continue
            value = operator.attrgetter(node.target)(module)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'compiled constant must be a tensor: {node.target}')
            origin = node.meta.get('storage_origin')
            index = storages[value.untyped_storage()._cdata if origin is None else origin[0]]
            with graph.inserting_before(next(iter(graph.nodes))):
                argument = graph.placeholder(f'buffer_{len(bindings)}')
            argument.meta['val'] = value
            bindings[argument] = _Input(index, (value.dtype, tuple(map(int, value.shape)),
                                              tuple(map(int, value.stride())), int(value.storage_offset())),
                                        None if origin is None else origin[1:])
            node.replace_all_uses_with(argument)
            graph.erase_node(node)
        graph.eliminate_dead_code()
        graph.lint()
        arguments = [node for node in graph.nodes if node.op == 'placeholder']
        self.inputs = [bindings[node] for node in arguments]
        examples = [node.meta['val'] for node in arguments]
        module = torch.fx.GraphModule({}, graph)
        with _compiler_recursion(module):
            self.artifact = standalone_compile(module, examples, dynamic_shapes='from_example_inputs',
                fake_mode=detect_fake_mode(examples), options={'config_patches': _compile_options(options)},
                aot=True, donate_graph_module=True)
        self._serialized_artifact = None
        self._cuda_allocation_sizes = None
        self._tuning_frozen = False
        self._bound = None

    def __call__(self, *args):
        identity = tuple((id(value), value.untyped_storage()._cdata, value.storage_offset())
                         if isinstance(value, torch.Tensor) else value for value in args)
        if self._bound is None or identity != self._bound[0]:
            signature = _signature(args)
            if signature != self.signature:
                differences = [(index, expected, actual) for index, (expected, actual)
                               in enumerate(zip(self.signature, signature)) if expected != actual]
                raise BindingMismatch('compiled forward input shapes, layouts, or storage aliases changed: '
                    f'{len(self.signature)} expected inputs, {len(signature)} supplied; '
                    f'first differences (index, expected, actual): {differences[:4]}')
            if _CHECK_BINDINGS.get():
                raise _BindingsChecked
            self._bound = identity, [item.bind(args) for item in self.inputs]
        if _CHECK_BINDINGS.get():
            raise _BindingsChecked
        token = _INVOCATIONS.set(self.invocations)
        try:
            return self.artifact(*self._bound[1])
        finally:
            _INVOCATIONS.reset(token)

    @classmethod
    def serialize_compile_artifacts(cls, fn):
        if fn._serialized_artifact is None and hasattr(fn.artifact, "_flare_graph"):
            raise RuntimeError("portable serialized artifact was already released")
        with _artifact_kernel_table():
            if fn._serialized_artifact is None:
                artifact = fn.artifact.serialize()
            elif isinstance(fn._serialized_artifact, _RawPortableArtifact):
                artifact = bytes(fn._serialized_artifact)
            else:
                artifact = zlib.decompress(fn._serialized_artifact)
        calls = _serialized_invocations(fn.invocations)
        sizes = getattr(fn, "_cuda_allocation_sizes", None)
        if sizes is None:
            graph = getattr(fn.artifact, "_flare_graph", None)
            if graph is None:
                entry, kernels, constants = pickle.loads(artifact)
                graph = entry.compiled_fw.result
                if fn._serialized_artifact is None:
                    from torch._inductor.output_code import CompiledFxGraph
                    from torch.utils._ordered_set import OrderedSet

                    if isinstance(graph, CompiledFxGraph):
                        graph.inductor_post_grad_graph_str = None
                        graph.cache_linemap = None
                        cuda_devices = {item[4] for item in fn.signature
                                        if isinstance(item, tuple) and item[4].type == "cuda"}
                        if len(cuda_devices) == 1:
                            original_index = next(iter(cuda_devices)).index
                            if graph.device_idxs == OrderedSet((original_index,)):
                                get_source_template(graph, original_index)
                        artifact = pickle.dumps((entry, kernels, constants), protocol=pickle.HIGHEST_PROTOCOL)
            sizes = tuple(int(size) for device, size in getattr(graph, "_flare_pool_sizes", ())
                          if device == "cuda")
            fn._cuda_allocation_sizes = sizes
        return pickle.dumps((artifact, fn.inputs, calls, fn.signature, fn._tuning_frozen, sizes),
                            protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def deserialize_compile_artifacts(cls, data):
        fn = cls.__new__(cls)
        artifact, fn.inputs, calls, fn.signature, fn._tuning_frozen, fn._cuda_allocation_sizes = pickle.loads(data)
        devices = {item[4] for item in fn.signature if isinstance(item, tuple) and item[4].type == "cuda"}
        if len(devices) != 1:
            raise ValueError("compiled forwards require exactly one CUDA device")
        original_device = devices.pop()
        fn.signature = tuple((*item[:4], restore_device(item[4].type, item[4].index), item[5])
                             if isinstance(item, tuple) else item for item in fn.signature)
        fn.signature = _share_storage_descriptors(fn.inputs, fn.signature, calls)
        fn.invocations = []
        for name, leaves, tree, operands, integers in calls:
            tree = treespec_loads(tree)
            if name[0] == 'triton':
                _, kernel, constants = name
                target = torch.ops.higher_order.triton_kernel_wrapper_mutation
                args, kwargs = tree_unflatten(leaves, tree)
                kwargs = dict(kwargs)
                kwargs['kernel_idx'] = kernel_side_table.add_kernel(_deserialized_triton_launch(kernel))
                kwargs['constant_args_idx'] = kernel_side_table.add_constant_args(constants)
                leaves, _ = tree_flatten((args, kwargs))
            else:
                _, namespace, opname, overload, registration = name
                if registration is not None:
                    importlib.import_module(registration)
                target = getattr(getattr(getattr(torch.ops, namespace), opname), overload)
            fn.invocations.append(_Invocation(target, leaves, tree, operands, integers))
        with _artifact_kernel_table():
            fn.artifact, fn._serialized_artifact = _load_inductor(artifact, original_device,
                                                               frozen=fn._tuning_frozen)
        fn._bound = None
        return fn


def _execute(forward):
    return forward()


def aot_forward(forward, *, options=None):
    runtime_patches.apply()
    options = _compile_options(options)
    sources = r'.*\.(?:_?handle|stream_handle|launch_order|plans\[\d+\])$'
    with _guard_serialization(), torch.compiler.config.patch(unbacked_sources=sources):
        return torch.compile(_execute, backend=partial(StorageBackend, options=options), fullgraph=True,
                             dynamic=False).aot_compile(((forward,), {}))


@dataclass
class _GuardedForwardCall:
    function: object
    guards: object
    backend: StorageBackend

    def __call__(self, owner):
        arguments = {"forward": owner}
        if not self.guards.check(arguments):
            raise BindingMismatch(f"prepared forward no longer matches compiled guards: "
                                  f"{self.guards.check_verbose(arguments)}")
        return self.function(owner)


@dataclass
class _CapturedForwardResources:
    backend: StorageBackend

    def __call__(self, owner):
        raise RuntimeError("captured forward no longer supports Python execution")


@dataclass
class BoundForward:
    artifact: object
    owner: object
    cache_hit: bool
    pending: object = None
    execution_identity: str | None = None

    def __call__(self):
        return self.artifact(self.owner)

    @property
    def backend(self):
        return (self.artifact.backend if isinstance(self.artifact, (_GuardedForwardCall, _CapturedForwardResources))
                else self.artifact._artifacts.compiled_fn)

    @property
    def capture_allocation_sizes(self):
        """CUDA allocation requests from the compiled forward's memory plan."""
        sizes = getattr(self.backend, "_cuda_allocation_sizes", None)
        if sizes is not None:
            return sizes
        return tuple(int(size) for device, size in self.backend.artifact._flare_graph._flare_pool_sizes
                     if device == "cuda")

    def prepare_capture(self):
        """Retain execution guards and kernel owners after portable cache writes finish."""
        if isinstance(self.artifact, _CapturedForwardResources):
            raise RuntimeError("captured forward no longer supports Python execution")
        if isinstance(self.artifact, _GuardedForwardCall):
            return 0
        state = self.artifact._artifacts
        if (tuple(state.signature.parameters) != ("forward",) or state.runtime_env.closure
                or state.guard_manager is None or not isinstance(state.compiled_fn, StorageBackend)):
            raise RuntimeError("capture requires a storage-bound, closed compiled forward")
        guards = state.guard_manager
        if guards.cache_entry is not None or guards.extra_state is not None:
            raise RuntimeError("capture requires standalone compiled guards")
        backend = state.compiled_fn
        serialization_owners = getattr(backend.artifact, "_flare_serialization_owners", ())
        if serialization_owners and any(owner.serialize_fn is not serialization_owners[0].serialize_fn
                                        for owner in serialization_owners):
            raise RuntimeError("bundled Inductor serializer changed before capture")
        released = len(state.guards_state) + len(backend._serialized_artifact or b"")
        backend._serialized_artifact = None
        for owner in serialization_owners:
            owner.serialize_fn = None
        if serialization_owners:
            backend.artifact._flare_serialization_owners = ()
        # Torch copies loading and serialization metadata into execution wrappers.
        pending = [backend.artifact.inner_fn.compiled_fn, getattr(backend.artifact, "_flare_graph", None)]
        visited = set()
        while pending:
            owner = pending.pop()
            if owner is None or id(owner) in visited:
                continue
            visited.add(id(owner))
            for field, value in (("source_code", ""), ("_flare_source_template", None),
                                 ("_triton_bundle", None), ("runnable_graph_str", ""),
                                 ("inductor_provenance_mapping_str", None),
                                 ("inductor_provenance_stack_traces_str", None)):
                if hasattr(owner, field):
                    setattr(owner, field, value)
            pending.append(getattr(owner, "__wrapped__", None))
        # Standalone execution checks the full root, without Dynamo's diff tree or debug lists.
        guards.diff_guard_root = None
        guards.diff_guard_sources.clear()
        guards.code_parts.clear()
        guards.verbose_code_parts = None
        self.artifact = _GuardedForwardCall(self.artifact.fn, guards, backend)
        return released

    def retire_after_capture(self):
        """Release Python guards after capture while retaining graph execution resources."""
        if isinstance(self.artifact, _GuardedForwardCall):
            self.artifact = _CapturedForwardResources(self.artifact.backend)
            # Guard retirement runs per captured or trial forward while the full model
            # catalog stays live. Collect young temporaries here (escalating to
            # generation 2 only under low host RAM) and scan the full heap once at the
            # phase boundary.
            release_unused(generation=0)


def _completed(pending):
    while True:
        for index, item in enumerate(pending):
            future = item[-2].pending
            if future is None or future.done():
                return pending.pop(index)
        wait([item[-2].pending for item in pending], return_when=FIRST_COMPLETED)


def allocation_order(names, sizes, coordinator=None):
    """Order compiled forwards by largest allocation request, agreed across ranks."""
    names = tuple(names)
    ranks = [(names, sizes)]
    if coordinator is not None:
        import torch.distributed as dist

        ranks = [None] * len(coordinator.ranks)
        dist.all_gather_object(ranks, (names, sizes), group=coordinator.group)
        if any(selected != names or requests.keys() != sizes.keys() for selected, requests in ranks):
            raise ValueError("tensor-parallel capture requires matching forward catalogs")

    def priority(name):
        requests = [rank_sizes[name] for _, rank_sizes in ranks if name in rank_sizes]
        return (bool(requests), max((max(allocations, default=0) for allocations in requests), default=0),
                max((sum(allocations) for allocations in requests), default=0))

    return tuple(sorted(names, key=priority, reverse=True))


# Command-line, planning, display, and diagnostic direct-execution sources do not change compiled forwards.
_COMMAND_SOURCES = ('cli', 'direct', 'errors.py', 'planning.py', 'progress.py')
# Nor does writing and signing the core file after preparation.
_PUBLICATION_SOURCES = ('preparation/model_core.py', 'preparation/core_file.py', 'preparation/protection.py')


def _variant_compilers(selections):
    """Assign each selected forward variant to the least loaded rank that selects it.

    `selections[i]` maps each forward name to the variant the group's rank `i` binds. Returns
    {(name, variant): i}; ties go to the lowest rank, so every rank derives the same assignment.
    """
    assigned, compilers = [0] * len(selections), {}
    for name in selections[0]:
        for variant in dict.fromkeys(chosen[name] for chosen in selections):
            compiler = min((index for index, chosen in enumerate(selections) if chosen[name] == variant),
                           key=assigned.__getitem__)
            assigned[compiler] += 1
            compilers[name, variant] = compiler
    return compilers


def _balance_compilations(counts, workers):
    """Spread forwards missing from one host's cache over its ranks' compilers.

    Rank `i` must compile `counts[i]` forwards with `workers[i]` workers; a rank without admitted
    workers lends all of them. Returns, per rank, the (owner, index)
    forwards it compiles: each rank keeps its own until lending one finishes sooner elsewhere.
    Every rank derives the same assignment.
    """
    if not any(workers):
        raise MemoryError("no rank on this host has room for a forward compiler")
    assigned = [[(owner, index) for index in range(count)] for owner, count in enumerate(counts)]
    able = [rank for rank, count in enumerate(workers) if count]

    def finish(rank, extra=0):
        held = len(assigned[rank]) + extra
        return -(-held // workers[rank]) if workers[rank] else float("inf") if held else 0

    while True:
        busiest = max(range(len(counts)), key=lambda rank: (finish(rank), -rank))
        idlest = min(able, key=lambda rank: (finish(rank, 1), rank))
        if finish(idlest, 1) >= finish(busiest):
            return assigned
        assigned[idlest].append(assigned[busiest].pop())


_HASH_CHUNK_BYTES = 4 << 20
_SHARE_BATCH_BYTES = 64 << 20


def _file_meta(path, known_digests=None):
    target = Path(path)
    if known_digests is not None and path in known_digests:
        return target.stat().st_size, known_digests[path]
    digest = hashlib.sha256()
    size = 0
    with target.open("rb") as file:
        while chunk := file.read(_HASH_CHUNK_BYTES):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _write_shared_file(path, slab, offset, size):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        try:
            if size:
                file.write(memoryview(slab.numpy())[offset:offset + size])
            file.close()
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def _share_artifacts(owned, group, ranks, *, known_digests=None, hosts=None):
    """Copy every rank's compiled forwards to ranks on other hosts, which load them by the same path.

    `owned[i]` lists the artifacts rank `ranks[i]` compiled. Each host keeps its own cache
    directory; when `hosts` names each rank's host, only the first rank on each host checks and
    writes shared files on that host's filesystem.
    """
    import torch.distributed as dist

    me = ranks.index(dist.get_rank())
    local_paths = list(dict.fromkeys(owned[me]))
    hash_workers = min(8, os.cpu_count() or 1)
    local_manifest = {}
    meta_error = None
    try:
        if len(local_paths) <= 1 or hash_workers <= 1:
            local_manifest = {path: _file_meta(path, known_digests) for path in local_paths}
        else:
            with ThreadPoolExecutor(max_workers=min(hash_workers, len(local_paths))) as pool:
                local_manifest = dict(
                    zip(local_paths, pool.map(lambda path: _file_meta(path, known_digests), local_paths), strict=True))
    except Exception as failure:
        meta_error = f"{type(failure).__name__}: {failure}"
    gathered_manifests = [None] * len(ranks)
    dist.all_gather_object(gathered_manifests, (local_manifest, meta_error), group=group)
    meta_failures = [(ranks[index], failure) for index, (_, failure) in enumerate(gathered_manifests)
                     if failure is not None]
    if meta_failures:
        raise RuntimeError(f"compiled forward artifact manifest collection failed: {meta_failures}")
    manifests = [compiled for compiled, _ in gathered_manifests]

    owners_by_path = {}
    for index, compiled in enumerate(manifests):
        for path, meta in compiled.items():
            previous = owners_by_path.get(path)
            if previous is None:
                owners_by_path[path] = (index, meta)
            elif previous[1] != meta:
                raise RuntimeError(f"conflicting compiled forward artifacts for {path}")

    if hosts is not None and len(hosts) != len(ranks):
        raise ValueError("hosts must match ranks when sharing compiled forward artifacts")
    host_leader = hosts is None or hosts.index(hosts[me]) == me
    same_host_manifest = dict(local_manifest)
    if hosts is not None:
        for index, compiled in enumerate(manifests):
            if hosts[index] == hosts[me]:
                same_host_manifest.update(compiled)

    def current(item):
        path, (expected_size, expected_digest) = item
        if same_host_manifest.get(path) == (expected_size, expected_digest):
            return True
        try:
            target = Path(path)
            if target.stat().st_size != expected_size:
                return False
            digest = hashlib.sha256()
            with target.open("rb") as file:
                while chunk := file.read(_HASH_CHUNK_BYTES):
                    digest.update(chunk)
            return digest.hexdigest() == expected_digest
        except OSError:
            return False

    if host_leader:
        checks = [
            (path, meta)
            for path, (owner_index, meta) in owners_by_path.items()
            if owner_index != me and (hosts is None or hosts[owner_index] != hosts[me])
        ]
        if len(checks) <= 1 or hash_workers <= 1:
            results = [current(item) for item in checks]
        else:
            with ThreadPoolExecutor(max_workers=min(hash_workers, len(checks))) as pool:
                results = list(pool.map(current, checks))
        missing = {path for (path, _), ok in zip(checks, results, strict=True) if not ok}
    else:
        missing = set()
    wanted = [None] * len(ranks)
    dist.all_gather_object(wanted, sorted(missing), group=group)
    wanted = set().union(*wanted)
    if not wanted:
        return

    transfer_error = None
    write_error = None
    in_flight_batches = []

    def drain_batch(batch_futures):
        nonlocal write_error
        for future in batch_futures:
            try:
                future.result()
            except Exception as failure:
                if write_error is None:
                    write_error = f"{type(failure).__name__}: {failure}"

    with ThreadPoolExecutor(max_workers=4) as write_pool:
        try:
            for index, compiled in enumerate(manifests):
                paths = sorted(
                    path for path in wanted.intersection(compiled)
                    if owners_by_path[path][0] == index
                )
                if not paths:
                    continue
                batches = []
                batch = []
                batch_bytes = 0
                for path in paths:
                    size, _ = compiled[path]
                    if batch and batch_bytes + size > _SHARE_BATCH_BYTES:
                        batches.append((batch, batch_bytes))
                        batch = []
                        batch_bytes = 0
                    batch.append((path, size))
                    batch_bytes += size
                if batch:
                    batches.append((batch, batch_bytes))

                for items, total_bytes in batches:
                    limit = 0 if total_bytes > _SHARE_BATCH_BYTES else 1
                    while len(in_flight_batches) > limit:
                        drain_batch(in_flight_batches.pop(0))
                    slab = torch.zeros(1 + total_bytes, dtype=torch.uint8)
                    if index == me:
                        try:
                            view = memoryview(slab.numpy())[1:]
                            offset = 0
                            for path, size in items:
                                with Path(path).open("rb") as file:
                                    if size:
                                        read = 0
                                        target_view = view[offset:offset + size]
                                        while read < size:
                                            count = file.readinto(target_view[read:])
                                            if not count:
                                                raise OSError(
                                                    f"truncated read while sharing {path}: {read} of {size} bytes")
                                            read += count
                                        offset += size
                        except Exception as failure:
                            slab[0] = 1
                            if transfer_error is None:
                                transfer_error = f"{type(failure).__name__}: {failure}"
                    dist.broadcast(slab, src=ranks[index], group=group)

                    batch_futures = []
                    if int(slab[0].item()) == 0 and write_error is None:
                        offset = 0
                        for path, size in items:
                            if path in missing:
                                batch_futures.append(write_pool.submit(_write_shared_file, path, slab, 1 + offset, size))
                                missing.discard(path)
                            offset += size
                    if batch_futures:
                        in_flight_batches.append(batch_futures)
                        if total_bytes > _SHARE_BATCH_BYTES:
                            drain_batch(in_flight_batches.pop(0))
        finally:
            while in_flight_batches:
                drain_batch(in_flight_batches.pop(0))

    errors = [None] * len(ranks)
    dist.all_gather_object(errors, transfer_error or write_error, group=group)
    failures = [(ranks[index], failure) for index, failure in enumerate(errors) if failure is not None]
    if failures:
        raise RuntimeError(f"compiled forward artifact sharing failed: {failures}")


class ForwardCache:
    """Load complete compiled forwards before tracing the serving catalog."""

    def __init__(self, *, namespace, device, root=None, headroom_bytes=0, options=None, compile_workers=None,
                 lock_gpu_clocks=False):
        runtime_patches.apply()
        import flashinfer
        from torch._inductor.codecache import torch_key

        self.device = torch.device(device)
        self.headroom_bytes = headroom_bytes
        self.compile_workers = compile_workers
        self.lock_gpu_clocks = lock_gpu_clocks
        self.options = MappingProxyType(_compile_options(options))
        properties = torch.cuda.get_device_properties(device)
        options = {name: value for name, value in self.options.items() if name != 'compile_threads'}
        key = hashlib.sha256(pickle.dumps((namespace, options, torch_key(), runtime_patches.identity(),
            properties.name, properties.major, properties.minor, properties.multi_processor_count)))
        for source in (Path(__file__).resolve().parents[1], Path(flashinfer.__file__).resolve().parent):
            for path in sorted(source.rglob('*.py')):
                if source.name == 'ignition' and (path.name == '__main__.py'
                                               or path.relative_to(source).parts[0] in _COMMAND_SOURCES
                                               or path.relative_to(source).as_posix() in _PUBLICATION_SOURCES):
                    continue
                key.update(str(path.relative_to(source)).encode())
                key.update(path.read_bytes())
        root = DEFAULT_CACHE_ROOT if root is None else Path(root)
        self.root = root / 'forwards' / key.hexdigest()

    def prepare(self, forward, *, name, signature, progress=None):
        """Bind and warm one callable on the caller's preparation stream."""
        with self._publication_lock(), _guard_code_cache(self.root / 'guards'):
            bound, path = self._prepare(forward, name=name, signature=signature, progress=progress)
            result = self._warm(bound, path, name=name, progress=progress)
            release_unused()
            return result

    @contextmanager
    def _publication_lock(self, coordinator=None):
        """Hold this host's cache against other launches from compilation through binding.

        Cached variants share filenames, so another launch must not replace a catalog still in
        use. Ranks on one host share its cache directory: the host's first rank holds the lock for
        all of them. No rank compiles before every host's lock is held, and no host releases it
        before every rank has bound its forwards.
        """
        host, hosts, holder = None, [None], True
        if coordinator is not None:
            import torch.distributed as dist

            index = coordinator.ranks.index(dist.get_rank())
            host = coordinator.hosts[index]
            holder = coordinator.hosts.index(host) == index
            hosts = sorted(set(coordinator.hosts))
        lock = None
        try:
            # Hosts lock in one order, so launches sharing hosts cannot each hold a lock the other awaits.
            for turn in hosts:
                if holder and turn == host:
                    self.root.mkdir(parents=True, exist_ok=True)
                    lock = FileLock(self.root / 'publication.lock')
                    lock.acquire()
                if coordinator is not None:
                    dist.barrier(group=coordinator.group)
            yield
            # A failure skips this barrier: the failing rank's peers learn of it collectively.
            if coordinator is not None:
                dist.barrier(group=coordinator.group)
        finally:
            if lock is not None:
                lock.release()

    def prepare_many(self, functions, signatures, *, progress=None, coordinator=None, materialize=None,
                     warmup=nullcontext, before_run=None):
        """Trace and compile in workers, then bind and warm on the owning rank.

        ``before_run`` restores mutable input state before every parent-rank
        binding check and warmup; worker compilation uses tensor metadata.
        """
        # Compiler workers own parallel compilation; artifact binding needs no second worker pool.
        with self._publication_lock(coordinator), torch._inductor.config.patch(compile_threads=1):
            return self._prepare_many(functions, signatures, progress=progress,
                                      coordinator=coordinator, materialize=materialize, warmup=warmup,
                                      before_run=before_run)

    def _prepare_many(self, functions, signatures, *, progress, coordinator, materialize, warmup,
                      before_run):
        from ignition.progress import phase
        from .compile_pool import CompilePool
        from .inductor_tuning import InductorSelections, _uses_default_portable_freeze

        if coordinator is not None:
            return self._prepare_ranks(functions, signatures, coordinator, progress=progress,
                                       materialize=materialize, warmup=warmup, before_run=before_run)
        selections = InductorSelections(self.device, lock_gpu_clocks=self.lock_gpu_clocks,
                                        source_identity=self.root.name)
        default_freeze = _uses_default_portable_freeze(selections)
        compiled, loaded, paths, pending = {}, {}, {}, []
        release_unused()
        with phase(progress, "Compile model forwards", total=len(functions)) as stage, CompilePool(self.device,
                headroom_bytes=self.headroom_bytes, options=self.options, compile_workers=self.compile_workers) as compiler:
            unsubmitted = len(functions)

            def finish():
                name, bound, path = _completed(pending)
                try:
                    with _guard_code_cache(self.root / 'guards'):
                        if bound.pending is not None:
                            rss = bound.pending.result()
                            if rss is not None:
                                compiler.observe_worker_rss(rss, remaining=unsubmitted + len(pending))
                        elif materialize is None and bound.cache_hit:
                            loaded[name] = bound
                        paths[name] = path
                except Exception as error:
                    raise RuntimeError(f"failed to prepare compiled forward {name}") from error
                stage.update(advance=1)

            for name, forward in functions.items():
                unsubmitted -= 1
                with _guard_code_cache(self.root / 'guards'):
                    bound, path = self._prepare(forward, name=name, signature=signatures[name], compiler=compiler,
                                                deferred=materialize is not None,
                                                before_run=None if before_run is None else lambda name=name: before_run(name))
                pending.append((name, bound, path))
                while len(pending) >= compiler.workers:
                    finish()
            while pending:
                finish()
        release_unused()
        if materialize is not None:
            materialize()
        if len(functions) > 1:
            sizes = {
                name: (loaded[name].capture_allocation_sizes if name in loaded
                       else self._allocation_sizes(paths[name]))
                for name in functions
            }
            ordered_names = allocation_order(functions, sizes)
        else:
            ordered_names = tuple(functions)
        prefetch_items = [(name, paths[name]) for name in ordered_names if name not in loaded]
        with (phase(progress, "Warm compiled forwards", total=len(functions)) as stage,
              ThreadPoolExecutor(max_workers=2) as save_executor,
              _BytePrefetcher(prefetch_items) as prefetcher):
            save_futures = []
            for name in ordered_names:
                forward = functions[name]
                try:
                    with _guard_code_cache(self.root / 'guards'):
                        if name in loaded:
                            bound = loaded.pop(name)
                        else:
                            defer_token = _DEFER_PORTABLE_COMPRESSION.set(default_freeze)
                            prefetch_token = _PREFETCHED_ARTIFACT_BYTES.set(prefetcher.pop(name))
                            try:
                                bound = self._load(forward, paths[name], required=True,
                                                   before_run=None if before_run is None else lambda name=name: before_run(name))
                            except BindingMismatch:
                                if materialize is None:
                                    raise
                                bound = BoundForward(aot_forward(forward, options=self.options), forward, False)
                            else:
                                if materialize is None:
                                    bound.cache_hit = False
                            finally:
                                _PREFETCHED_ARTIFACT_BYTES.reset(prefetch_token)
                                _DEFER_PORTABLE_COMPRESSION.reset(defer_token)
                        compiled[name] = self._warm(bound, paths[name], name=name, progress=None,
                                                    selections=selections, warmup=warmup,
                                                    before_run=None if before_run is None else lambda name=name: before_run(name),
                                                    release_unchanged=True, save_executor=save_executor,
                                                    save_futures=save_futures)
                        compiled[name].prepare_capture()
                        selections.release(compiled[name])
                        release_unused(generation=0)
                except Exception as error:
                    if materialize is None:
                        raise RuntimeError(f"failed to prepare compiled forward {name}") from error
                    raise
                stage.update(advance=1)
            for future in save_futures:
                try:
                    future.result()
                except Exception as error:
                    if materialize is None:
                        raise RuntimeError("failed to save compiled forward") from error
                    raise
        release_unused()
        return compiled

    def _prepare_ranks(self, functions, signatures, coordinator, *, progress, materialize, warmup,
                       before_run):
        """Compile the catalog for a rank group, then bind and warm it on every rank.

        Compiled forwards serve every rank, so the catalog compiles once: each forward variant a
        rank selects through `compilation_variant` compiles on a rank that selects it, from that
        rank's own forward, unless an idle rank on the same or another host borrows it and
        compiles it from the owner's traced forward metadata. Compiler pools are budgeted for each
        host's ranks. Ranks on one host share its cache directory; ranks on other hosts receive
        the artifacts they lack from the ranks whose disks hold them. Every rank then binds the
        same artifacts.
        """
        import torch.distributed as dist
        from ignition.progress import phase
        from . import host_memory
        from .compile_pool import CompilePool, allocate_workers, compiler_resources, cpu_workers
        from .inductor_tuning import InductorSelections, _unchanged_frozen, _uses_default_portable_freeze

        group, ranks = coordinator.group, coordinator.ranks
        share = ranks.index(dist.get_rank())
        selections = [None] * len(ranks)
        dist.all_gather_object(selections, {name: getattr(forward, "compilation_variant", "shared")
                                            for name, forward in functions.items()}, group=group)
        if any(chosen.keys() != selections[0].keys() for chosen in selections):
            raise RuntimeError("ranks prepared different forward catalogs")
        compilers = _variant_compilers(selections)
        work = [(name, variant, functions[name]) for (name, variant), compiler in compilers.items() if compiler == share]
        selected, outcome, error, compiler, paths, lent, sent = {}, (None, None, None), None, None, {}, set(), []
        sent_files = []
        local_compiled = []
        loans = []
        with ExitStack() as scope:
            stage = scope.enter_context(phase(progress, "Compile model forwards", total=len(work)))
            compiler_scope = scope.enter_context(ExitStack())
            # Variants that only one rank selects, such as rank zero's sampling, can leave that rank most of
            # the catalog. Ranks on one host share its cache, so a rank lends forwards it cannot restore, whether
            # missing or bound differently by another launch, to ranks whose compilers would otherwise finish
            # first; the borrower compiles the owner's metadata.
            restored, missing = {}, []
            try:
                for index, (name, variant, forward) in enumerate(work):
                    with _guard_code_cache(self.root / 'guards'):
                        bound, path = self._restore(forward, name=f"{name}.{variant}", signature=signatures[name],
                                                    deferred=materialize is not None,
                                                    before_run=None if before_run is None else lambda name=name: before_run(name))
                    if bound is None:
                        missing.append(index)
                    else:
                        restored[index] = bound, path
            except Exception as failure:
                error = error or f"{type(failure).__name__}: {failure}"
            resources = None
            try:
                release_unused()
            except Exception as failure:
                error = error or f"{type(failure).__name__}: {failure}"
            # Include every rank's retained cache-restoration owners in host memory admission.
            dist.barrier(group=group)
            try:
                resources = compiler_resources(self.device, headroom_bytes=self.headroom_bytes,
                    options=self.options, compile_workers=self.compile_workers)
            except Exception as failure:
                error = error or f"{type(failure).__name__}: {failure}"
            demand = [None] * len(ranks)
            dist.all_gather_object(demand, (len(missing), resources, error), group=group)

            def pool_bounds(active_workers):
                host = coordinator.hosts[share]
                members = [index for index, name in enumerate(coordinator.hosts) if name == host]
                host_pools = sum(1 for index in members if active_workers[index] > 0)
                host_available = min(demand[index][1].available for index in members)
                local_resources = demand[share][1]
                device_cap = (local_resources.capacity if local_resources.max_capacity is None
                              else local_resources.max_capacity)
                if local_resources.worker_bytes > host_memory.FORWARD_WORKER_BYTES:
                    device_cap //= host_pools
                if self.compile_workers is not None and self.compile_workers < host_pools:
                    cpus = 1
                elif host_pools == 1:
                    cpus = cpu_workers(self.compile_workers)
                else:
                    cpus = cpu_workers(self.compile_workers, pools=host_pools)
                max_workers = max(active_workers[share], min(cpus, device_cap))
                return host_pools, host_available, max_workers

            counts = []
            workers = []
            probe_hosts = set()
            probe_host_rss = {}
            probe_assigned = [[] for _ in ranks]
            probe_consumed = [0] * len(ranks)
            probe_done = set()
            try:
                failures = [(rank, failure) for rank, (_, _, failure) in zip(ranks, demand) if failure is not None]
                if failures:
                    raise RuntimeError(f"forward compiler resource collection failed: {failures}")
                counts = [count for count, _, _ in demand]
                workers = allocate_workers([resources for _, resources, _ in demand], counts, coordinator.hosts)
                stage.update(f"{sum(workers)} compiler workers; per rank {workers}")
                for host in dict.fromkeys(coordinator.hosts):
                    members = [index for index, name in enumerate(coordinator.hosts) if name == host]
                    host_jobs = sum(counts[index] for index in members)
                    host_workers = sum(workers[index] for index in members)
                    host_limit = min(demand[index][1].limit for index in members)
                    if (host_workers > 0
                            and host_jobs > host_workers
                            and host_workers < host_limit
                            and any(workers[index] == 0 and demand[index][1].capacity > 0 for index in members)):
                        probe_hosts.add(host)
                        for member in members:
                            if not workers[member]:
                                continue
                            own_take = min(workers[member], counts[member] - probe_consumed[member])
                            for offset in range(own_take):
                                probe_assigned[member].append((member, probe_consumed[member] + offset))
                            probe_consumed[member] += own_take
                        for member in members:
                            need = workers[member] - len(probe_assigned[member])
                            if need <= 0:
                                continue
                            for donor in members:
                                if need <= 0:
                                    break
                                available_jobs = counts[donor] - probe_consumed[donor]
                                if available_jobs <= 0:
                                    continue
                                take = min(need, available_jobs)
                                for offset in range(take):
                                    probe_assigned[member].append((donor, probe_consumed[donor] + offset))
                                probe_consumed[donor] += take
                                need -= take
            except Exception as failure:
                error = error or f"{type(failure).__name__}: {failure}"
            if probe_hosts:
                # When the conservative startup estimate leaves capable GPUs without workers on a host,
                # run one wave on the admitted ranks first to measure actual worker RSS and admit idle GPUs.
                probe_error = None
                probe_sent = []
                probe_lent = set()
                try:
                    if probe_assigned[share]:
                        host_pools, host_available, max_workers = pool_bounds(workers)
                        compiler = compiler_scope.enter_context(CompilePool(
                            self.device, options=self.options, workers=workers[share],
                            headroom_bytes=self.headroom_bytes, max_workers=max_workers,
                            host_pools=host_pools, host_available=host_available))
                    probe_loans = [
                        (borrower, missing[local_idx])
                        for borrower, forwards in enumerate(probe_assigned)
                        for owner, local_idx in forwards
                        if owner == share and borrower != share
                    ]
                    probe_lent = {work[index][:2] for _, index in probe_loans}
                    if probe_loans:
                        from .forward_metadata import dumps

                        jobs = self.root / 'lent'
                        jobs.mkdir(parents=True, exist_ok=True)
                        for borrower, index in probe_loans:
                            name, variant, forward = work[index]
                            path = self._path(forward, name=f"{name}.{variant}", signature=signatures[name])
                            job = jobs / f"{path.stem}.probe.forward"
                            with tempfile.NamedTemporaryFile(dir=jobs, suffix='.tmp', delete=False) as file:
                                temporary = Path(file.name)
                            try:
                                with temporary.open('wb') as file:
                                    pickle.dump((dumps(forward), str(path)), file)
                                os.replace(temporary, job)
                            finally:
                                temporary.unlink(missing_ok=True)
                            probe_sent.append((borrower, str(job)))
                            paths[name, variant] = str(path), True
                except Exception as failure:
                    probe_error = f"{type(failure).__name__}: {failure}"
                probe_mailed = [None] * len(ranks)
                dist.all_gather_object(probe_mailed, probe_sent, group=group)
                probe_borrowed = [job for jobs in probe_mailed for borrower, job in jobs if borrower == share]
                probe_peak_rss = 0
                try:
                    if probe_error is not None:
                        raise RuntimeError(probe_error)
                    probe_pending = []
                    for owner, local_idx in probe_assigned[share]:
                        if owner != share:
                            continue
                        index = missing[local_idx]
                        name, variant, forward = work[index]
                        path = self._path(forward, name=f"{name}.{variant}", signature=signatures[name])
                        bound = BoundForward(None, forward, False, compiler.submit(forward, path))
                        probe_pending.append((name, variant, bound, path))
                    for job in probe_borrowed:
                        with open(job, 'rb') as file:
                            payload, path = pickle.load(file)
                        os.unlink(job)
                        future = compiler.submit_payload(payload, Path(path))
                        probe_pending.append((None, None, BoundForward(None, None, False, future), path))
                    while probe_pending:
                        name, variant, bound, path = _completed(probe_pending)
                        rss = bound.pending.result()
                        if rss is not None and rss > probe_peak_rss:
                            probe_peak_rss = int(rss)
                        local_compiled.append(str(path))
                        if name is not None:
                            paths[name, variant] = str(path), True
                            stage.update(advance=1)
                except Exception as failure:
                    probe_error = probe_error or f"{type(failure).__name__}: {failure}"
                probe_results = [None] * len(ranks)
                dist.all_gather_object(probe_results, (probe_peak_rss, probe_error), group=group)
                for _, job in probe_sent:
                    Path(job).unlink(missing_ok=True)
                probe_failures = [(rank, failure) for rank, (_, failure) in zip(ranks, probe_results)
                                  if failure is not None]
                if probe_failures:
                    try:
                        compiler_scope.close()
                    except Exception as failure:
                        probe_failures.append((ranks[share], f"{type(failure).__name__}: {failure}"))
                    raise RuntimeError(f"forward compilation failed: {probe_failures}")
                stage.update(advance=len(probe_lent))
                probe_done = set(missing[:probe_consumed[share]])
                missing = missing[probe_consumed[share]:]
                counts = [count - consumed for count, consumed in zip(counts, probe_consumed)]
                try:
                    sized_resources = [item for _, item, _ in demand]
                    for host in probe_hosts:
                        members = [index for index, name in enumerate(coordinator.hosts) if name == host]
                        max_host_rss = max(probe_results[index][0] for index in members)
                        if max_host_rss > 0:
                            probe_host_rss[host] = int(max_host_rss)
                            measured_unit = max(host_memory._WORKER_BYTES, (max_host_rss * 5 + 3) // 4)
                            for index in members:
                                orig_res = demand[index][1]
                                sized_resources[index] = replace(
                                    orig_res,
                                    worker_bytes=measured_unit + (orig_res.worker_bytes - host_memory.FORWARD_WORKER_BYTES),
                                )
                    workers = allocate_workers(sized_resources, counts, coordinator.hosts)
                    stage.update(f"{sum(workers)} compiler workers; per rank {workers}")
                except Exception as failure:
                    error = error or f"{type(failure).__name__}: {failure}"
            try:
                if error is None:
                    assigned = [[] for _ in ranks]
                    for host in dict.fromkeys(coordinator.hosts):
                        members = [index for index, name in enumerate(coordinator.hosts) if name == host]
                        if not any(counts[index] for index in members):
                            continue
                        local = _balance_compilations([counts[index] for index in members],
                                                      [workers[index] for index in members])
                        for borrower, forwards in zip(members, local):
                            assigned[borrower] = [(members[owner], index) for owner, index in forwards]
                    if coordinator.spans_hosts and any(workers) and any(counts):
                        able = [rank for rank, count in enumerate(workers) if count]

                        def finish_wave(rank, extra=0):
                            held = len(assigned[rank]) + extra
                            return -(-held // workers[rank]) if workers[rank] else float("inf") if held else 0

                        while True:
                            busiest = max(range(len(ranks)), key=lambda rank: (finish_wave(rank), -rank))
                            idlest = min(able, key=lambda rank: (finish_wave(rank, 1), rank))
                            if finish_wave(idlest, 1) >= finish_wave(busiest):
                                break
                            assigned[idlest].append(assigned[busiest].pop())
                    if workers[share] and assigned[share]:
                        host_pools, host_available, max_workers = pool_bounds(workers)
                        local_host = coordinator.hosts[share]
                        min_workers = 1 if local_host in probe_hosts else None
                        if compiler is None:
                            compiler = compiler_scope.enter_context(CompilePool(
                                self.device, options=self.options, workers=workers[share],
                                headroom_bytes=self.headroom_bytes, max_workers=max_workers,
                                min_workers=min_workers, host_pools=host_pools, host_available=host_available))
                        else:
                            compiler.reconfigure(
                                workers=workers[share], max_workers=max_workers,
                                host_pools=host_pools, min_workers=min_workers)
                        if local_host in probe_host_rss and hasattr(compiler, "peak_worker_rss"):
                            compiler.peak_worker_rss = max(compiler.peak_worker_rss, probe_host_rss[local_host])
                    elif compiler is not None:
                        compiler_scope.close()
                        compiler = None
                    loans = [(borrower, missing[index]) for borrower, forwards in enumerate(assigned)
                             for owner, index in forwards if owner == share and borrower != share]
                    lent = {work[index][:2] for _, index in loans}
                    from .forward_metadata import dumps

                    if loans:
                        jobs = None
                        for borrower, index in loans:
                            name, variant, forward = work[index]
                            path = self._path(forward, name=f"{name}.{variant}", signature=signatures[name])
                            payload = dumps(forward)
                            if coordinator.hosts[borrower] == coordinator.hosts[share]:
                                if jobs is None:
                                    jobs = self.root / 'lent'
                                    jobs.mkdir(parents=True, exist_ok=True)
                                job = jobs / f"{path.stem}.forward"
                                with tempfile.NamedTemporaryFile(dir=jobs, suffix='.tmp', delete=False) as file:
                                    temporary = Path(file.name)
                                try:
                                    with temporary.open('wb') as file:
                                        pickle.dump((payload, str(path)), file)
                                    os.replace(temporary, job)
                                finally:
                                    temporary.unlink(missing_ok=True)
                                sent.append((borrower, ("file", str(job))))
                                sent_files.append(str(job))
                            else:
                                sent.append((borrower, ("payload", payload, str(path))))
                            # The borrower compiles it before every rank gathers the outcome below.
                            paths[name, variant] = str(path), True
            except Exception as failure:
                error = error or f"{type(failure).__name__}: {failure}"
            mailed = [None] * len(ranks)
            dist.all_gather_object(mailed, sent, group=group)
            borrowed = [job for jobs in mailed for borrower, job in jobs if borrower == share]
            try:
                if error is not None:
                    raise RuntimeError(error)
                pending = []
                unsubmitted_compiles = sum(
                    1 for index, (name, variant, _) in enumerate(work)
                    if index not in probe_done and (name, variant) not in lent and index not in restored
                ) + len(borrowed)

                def finish():
                    name, variant, bound, path = _completed(pending)
                    if bound.pending is not None:
                        rss = bound.pending.result()
                        if rss is not None:
                            in_flight = sum(1 for item in pending if item[2].pending is not None)
                            compiler.observe_worker_rss(rss, remaining=unsubmitted_compiles + in_flight)
                    elif materialize is None:
                        # Restored and bound to this rank's own forward already.
                        selected[name] = bound
                    local_compiled.append(str(path))
                    if name is not None:
                        # Each forward's path, and whether this launch compiled it rather than restoring it.
                        paths[name, variant] = str(path), bound.pending is not None
                        stage.update(advance=1)

                for index, (name, variant, forward) in enumerate(work):
                    if index in probe_done or (name, variant) in lent:
                        continue
                    if index in restored:
                        bound, path = restored[index]
                    else:
                        unsubmitted_compiles -= 1
                        path = self._path(forward, name=f"{name}.{variant}", signature=signatures[name])
                        bound = BoundForward(None, forward, False, compiler.submit(forward, path))
                    pending.append((name, variant, bound, path))
                    while compiler is not None and len(pending) >= compiler.workers:
                        finish()
                for item in borrowed:
                    unsubmitted_compiles -= 1
                    if item[0] == "file":
                        _, job = item
                        with open(job, 'rb') as file:
                            payload, path = pickle.load(file)
                        os.unlink(job)
                    else:
                        _, payload, path = item
                    future = compiler.submit_payload(payload, Path(path))
                    pending.append((None, None, BoundForward(None, None, False, future), path))
                    while len(pending) >= compiler.workers:
                        finish()
                while pending:
                    finish()
                outcome = paths, sorted(set(local_compiled)), None
            except Exception as failure:
                outcome = None, None, f"{type(failure).__name__}: {failure}"
            try:
                compiler_scope.close()
            except Exception as failure:
                outcome = None, None, outcome[2] or f"{type(failure).__name__}: {failure}"
            outcomes = [None] * len(ranks)
            dist.all_gather_object(outcomes, outcome, group=group)
            for job in sent_files:
                Path(job).unlink(missing_ok=True)
            # Raise inside the phase so it ends as failed rather than complete. Every rank holds the same
            # outcomes, so every rank raises here and none skips a collective.
            failures = [(index, failure) for index, (_, _, failure) in zip(ranks, outcomes) if failure is not None]
            if failures:
                raise RuntimeError(f"forward compilation failed: {failures}")
            # Forwards lent to other ranks are complete once every rank reports.
            stage.update(advance=len(lent))
        restored.clear()
        bound = None
        paths = {key: value for compiled, _, _ in outcomes for key, value in compiled.items()}
        # Ranks on one host share its cache directory; ranks on other hosts receive every artifact from the rank holding it.
        if coordinator.spans_hosts:
            _share_artifacts([local for _, local, _ in outcomes], group, ranks, hosts=coordinator.hosts)
        error, sizes = None, None
        try:
            release_unused()
            if materialize is not None:
                materialize()
            sizes = {
                name: (selected[name].capture_allocation_sizes if name in selected
                       else self._allocation_sizes(paths[name, getattr(forward, "compilation_variant", "shared")][0]))
                for name, forward in functions.items()
            }
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
        errors = [None] * len(ranks)
        dist.all_gather_object(errors, error, group=group)
        if any(errors):
            raise RuntimeError(f"compiled forward materialization failed: {errors}")
        ordered_names = allocation_order(functions, sizes, coordinator)
        selections = InductorSelections(self.device, coordinator=coordinator,
                                        lock_gpu_clocks=self.lock_gpu_clocks, source_identity=self.root.name)
        default_freeze = _uses_default_portable_freeze(selections)
        published, released = [], 0
        saved_digests = {}
        prefetch_items = [
            (name, paths[name, getattr(functions[name], "compilation_variant", "shared")][0])
            for name in ordered_names if name not in selected
        ]
        with (phase(progress, "Load and warm compiled forwards", total=len(functions)) as stage,
              ThreadPoolExecutor(max_workers=2) as save_executor,
              _BytePrefetcher(prefetch_items) as prefetcher):
            save_futures = []
            post_failure = None
            for name in ordered_names:
                forward = functions[name]
                variant = getattr(forward, "compilation_variant", "shared")
                owns_variant = compilers[name, variant] == share
                load_failure = None
                bound = None
                if post_failure is None:
                    try:
                        if name not in selected:
                            path, compiled = paths[name, variant]
                            token = _RETAIN_SERIALIZED_ARTIFACT.set(owns_variant or not default_freeze)
                            defer_token = _DEFER_PORTABLE_COMPRESSION.set(owns_variant and default_freeze)
                            prefetch_token = _PREFETCHED_ARTIFACT_BYTES.set(prefetcher.pop(name))
                            try:
                                with _guard_code_cache(self.root / 'guards'):
                                    selected[name] = self._load(
                                        forward, Path(path), required=True,
                                        before_run=None if before_run is None else lambda name=name: before_run(name))
                            finally:
                                _PREFETCHED_ARTIFACT_BYTES.reset(prefetch_token)
                                _DEFER_PORTABLE_COMPRESSION.reset(defer_token)
                                _RETAIN_SERIALIZED_ARTIFACT.reset(token)
                            selected[name].cache_hit = not compiled
                        bound = selected[name]
                        selections.register(bound)
                    except Exception as error:
                        import traceback

                        traceback.print_exc(limit=-40)
                        load_failure = f"{name}: {type(error).__name__}: {error}"
                selections._check(post_failure or load_failure)
                coordinator.barrier(f"compiled warmup {name}", ranks)
                try:
                    selections.warm(bound, warmup,
                                    before_run=None if before_run is None else lambda name=name: before_run(name))
                except Exception as error:
                    error.add_note(f"Warming compiled forward {name!r}")
                    raise
                try:
                    unchanged = bound.cache_hit and _unchanged_frozen(selections, bound)
                    if (not owns_variant or unchanged) and default_freeze:
                        serialized = getattr(bound.backend, "_serialized_artifact", None)
                        if serialized is not None:
                            released += len(serialized)
                            bound.backend._serialized_artifact = None
                    selections.freeze(bound)
                    if not owns_variant or unchanged:
                        serialized = getattr(bound.backend, "_serialized_artifact", None)
                        if serialized is not None:
                            released += len(serialized)
                            bound.backend._serialized_artifact = None
                    if owns_variant and not unchanged:
                        path, _ = paths[name, variant]
                        _save_compiled_async(bound.artifact, path, save_executor, save_futures)
                        published.append(path)
                    released += bound.prepare_capture()
                    selections.release(bound)
                    release_unused(generation=0)
                except Exception as error:
                    post_failure = f"{name}: {type(error).__name__}: {error}"
                stage.update(advance=1)
            for future in save_futures:
                try:
                    saved = future.result()
                    if saved is not None:
                        saved_path, saved_digest = saved
                        saved_digests[saved_path] = saved_digest
                except Exception as error:
                    if post_failure is None:
                        post_failure = f"{type(error).__name__}: {error}"
            selections._check(post_failure)
            stage.update(f"{released >> 20} MiB portable state released; guards and kernel owners retained")
        release_unused()
        if coordinator.spans_hosts:
            _share_artifacts(selections._gather(published), group, ranks,
                             known_digests=saved_digests, hosts=coordinator.hosts)
        return selected

    def _allocation_sizes(self, path):
        from torch._dynamo.aot_compile import AOTCompileUnpickler

        with Path(path).open('rb') as file:
            compiled = AOTCompileUnpickler({}, file).load()
        deserializer, payload = compiled['compiled_fn']
        if deserializer != StorageBackend.deserialize_compile_artifacts:
            raise TypeError(f"{path} does not contain a Flare storage backend")
        _, _, _, _, _, sizes = pickle.loads(payload)
        return sizes

    def _load(self, forward, path, *, required=False, before_run=None):
        prefetched = _PREFETCHED_ARTIFACT_BYTES.get()
        if prefetched is not None:
            _PREFETCHED_ARTIFACT_BYTES.set(None)
            stream = io.BytesIO(prefetched)
        else:
            stream = Path(path).open('rb')
        with bind_device(self.device, freeze_tuning=self.options.get(
                'triton.autotune_at_compile_time', False)), stream as file:
            artifact = torch.compiler.load_compiled_function(file)
        # Bound forwards remain live for the complete catalog. Collect young
        # deserialization temporaries here; scan the retained heap once at each
        # preparation phase boundary instead of once per restored forward.
        release_unused(generation=0)
        if not artifact.guard_check(forward):
            if required:
                reason = artifact._artifacts.guard_manager.check_verbose(artifact.prepare_f_locals(forward))
                raise BindingMismatch(f"rank-local forward does not match compiled guards: {reason}")
            return None
        # Dynamo guards omit some offsets and aliases checked by the storage backend.
        token = _CHECK_BINDINGS.set(True)
        try:
            if before_run is not None:
                before_run()
            artifact(forward)
        except _BindingsChecked:
            return BoundForward(artifact, forward, True)
        except BindingMismatch:
            if required:
                raise
            return None
        finally:
            _CHECK_BINDINGS.reset(token)
        raise RuntimeError("compiled forward did not validate its storage bindings")

    def _path(self, forward, *, name, signature):
        identity = (type(forward).__module__, type(forward).__qualname__, name, signature)
        return self.root / (hashlib.sha256(pickle.dumps(identity)).hexdigest() + '.bin')

    def _restore(self, forward, *, name, signature, deferred=False, report=None,
                 before_run=None):
        """The cached forward bound to `forward`, or None when it is missing or its guards reject it, and its path."""
        path = self._path(forward, name=name, signature=signature)
        if not path.is_file():
            return None, path
        if report is not None:
            report("restoring compiled forward")
        token = _DEFERRED_BINDINGS.set(deferred)
        serialized_token = _RETAIN_SERIALIZED_ARTIFACT.set(
            _RETAIN_SERIALIZED_ARTIFACT.get() and not deferred)
        try:
            bound = self._load(forward, path, before_run=before_run)
        finally:
            _RETAIN_SERIALIZED_ARTIFACT.reset(serialized_token)
            _DEFERRED_BINDINGS.reset(token)
        if report is not None:
            report("checking compiled bindings")
        return bound, path

    def _prepare(self, forward, *, name, signature, compiler=None, progress=None, deferred=False,
                 before_run=None):
        def report(operation):
            if progress is not None:
                progress.update(f"{name} · {operation}")
        bound, path = self._restore(forward, name=name, signature=signature, deferred=deferred,
                                    report=report, before_run=before_run)
        if bound is not None:
            return bound, path
        report("tracing and compiling forward")
        if compiler is not None:
            return BoundForward(None, forward, False, compiler.submit(forward, path)), path
        return BoundForward(aot_forward(forward, options=self.options), forward, False), path

    def _warm(self, bound, path, *, name, progress=None, selections=None, warmup=nullcontext,
              before_run=None, release_unchanged=False, save_executor=None, save_futures=None):
        def report(operation):
            if progress is not None:
                progress.update(f"{name} · {operation}")

        from .inductor_tuning import InductorSelections, _unchanged_frozen

        if selections is None:
            selections = InductorSelections(self.device, lock_gpu_clocks=self.lock_gpu_clocks,
                                        source_identity=self.root.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        rewritten = not hasattr(bound.artifact._artifacts.compiled_fn.artifact, '_flare_graph')
        if rewritten:
            bound.artifact.save_compiled_function(str(path))
            cached = bound.cache_hit
            bound = self._load(bound.owner, path, required=True, before_run=before_run)
            bound.cache_hit = cached
        report("warming restored forward" if bound.cache_hit else "warming compiled forward")
        selections.register(bound)
        selections.warm(bound, warmup, before_run=before_run)
        unchanged = bound.cache_hit and not rewritten and _unchanged_frozen(selections, bound)
        selections.freeze(bound)
        if release_unchanged and unchanged and getattr(bound.backend, "_serialized_artifact", None) is not None:
            bound.backend._serialized_artifact = None
        if not unchanged:
            report("saving compiled forward")
            _save_compiled_async(bound.artifact, path, save_executor, save_futures)
        return bound
