# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiler workers receive tensor layouts and aliases without tensor contents."""

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses import FakeTensor

from ignition.preparation.forward_metadata import dumps, loads


@pytest.mark.parametrize("integrated,expected", [(False, 2), (True, 1)])
def test_compiler_concurrency_accounts_for_shared_device_ram(monkeypatch, integrated, expected):
    from ignition.preparation import compile_pool, host_memory

    free = (24 if host_memory.os.name == "nt" else 2) if integrated else 72
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free << 30, 121 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=integrated))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 72 << 30)
    monkeypatch.setitem(compile_pool._OPTIONS, "compile_threads", 16)
    pool = compile_pool.CompilePool("cuda:0")
    assert pool.workers == expected
    assert pool.options["compile_threads"] == 1


def test_forward_compiler_uses_explicit_host_budget_without_eight_worker_cap(monkeypatch):
    from ignition.preparation import compile_pool, host_memory

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (96 << 30, 96 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 1032 << 30)
    monkeypatch.setattr(host_memory, "cpu_workers", lambda requested=None: requested or 64)
    monkeypatch.setattr(compile_pool, "cpu_workers", host_memory.cpu_workers)
    assert compile_pool.CompilePool("cuda:0", compile_workers=32).workers == 32
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (12 << 30, 96 << 30))
    # A discrete GPU's worker holds 1 GiB.
    assert compile_pool.CompilePool("cuda:0", compile_workers=32, headroom_bytes=2 << 30).workers == 10


def test_metadata_preserves_tensor_aliases_and_offsets():
    storage = torch.empty(1024, dtype=torch.uint8)
    original = SimpleNamespace(storage=storage, part=storage.view(torch.float32)[16:48],
        alias=storage.view(torch.bfloat16), independent=torch.empty(1024, dtype=torch.uint8))
    original.again = original.part
    restored = loads(dumps(original))
    assert restored.again is restored.part
    assert isinstance(restored.storage, FakeTensor)
    for name in ('storage', 'part', 'alias', 'independent'):
        source, target = getattr(original, name), getattr(restored, name)
        assert (target.shape, target.stride(), target.storage_offset(), target.dtype, target.device) == (
            source.shape, source.stride(), source.storage_offset(), source.dtype, source.device)
    assert restored.storage.untyped_storage()._cdata == restored.part.untyped_storage()._cdata
    assert restored.storage.untyped_storage()._cdata == restored.alias.untyped_storage()._cdata
    assert restored.storage.untyped_storage()._cdata != restored.independent.untyped_storage()._cdata


def test_metadata_size_is_independent_of_storage_size():
    small = torch.empty(16, device='meta')
    large = torch.empty(1 << 35, device='meta')
    assert len(dumps(large)) < len(dumps(small)) + 32
    restored = loads(dumps(large))
    assert restored.numel() == 1 << 35
    assert restored.untyped_storage().device.type == 'meta'


def test_checkpoint_model_class_loads_in_fresh_worker(tmp_path):
    import subprocess
    import sys

    cache = tmp_path / 'modules'
    source = tmp_path / 'checkpoint'
    source.mkdir()
    (source / 'modeling_test.py').write_text(
        'import torch\n'
        'class CheckpointLayer(torch.nn.Linear):\n'
        '    pass\n'
    )
    payload = tmp_path / 'forward.pkl'
    producer = '''
import sys
from pathlib import Path
from transformers import dynamic_module_utils
from ignition.preparation.forward_metadata import dumps
dynamic_module_utils.HF_MODULES_CACHE = sys.argv[1]
layer = dynamic_module_utils.get_class_from_dynamic_module(
    'modeling_test.CheckpointLayer', sys.argv[2], local_files_only=True)
Path(sys.argv[3]).write_bytes(dumps(layer(16, 32, device='meta')))
'''
    subprocess.run([sys.executable, '-c', producer, str(cache), str(source), str(payload)],
                   check=True, capture_output=True, text=True)
    code = '''
import sys
from pathlib import Path
from transformers import dynamic_module_utils
from torch._subclasses import FakeTensor
from ignition.preparation.forward_metadata import loads
dynamic_module_utils.HF_MODULES_CACHE = sys.argv[1]
assert sys.argv[1] not in sys.path
assert not any(name.startswith('transformers_modules.') for name in sys.modules)
layer = loads(Path(sys.argv[2]).read_bytes())
assert type(layer).__name__ == 'CheckpointLayer'
assert isinstance(layer.weight, FakeTensor)
assert tuple(layer.weight.shape) == (32, 16)
'''
    subprocess.run([sys.executable, '-c', code, str(cache), str(payload)],
                   check=True, capture_output=True, text=True)


def test_preparation_closures_cannot_execute_in_compiler():
    owner = SimpleNamespace(callback=lambda: pytest.fail('preparation must not run'))
    restored = loads(dumps(owner))
    with pytest.raises(RuntimeError, match='process-local resource'):
        restored.callback.anything


@pytest.mark.parametrize("kind", ["storage", "session", "recorded_session"])
def test_preparation_resources_do_not_serialize_active_core_storage(kind):
    from b12x.preparation import PreparationSession
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.tensor_metadata import NumericStorage, _NUMERIC_STORAGE
    from ignition.preparation.tuning import RecordedPreparationSession

    with NumericStorage() as registry:
        storage = CoreStorage.__new__(CoreStorage)
        storage.numeric_storage = registry
        resource = storage
        if kind != "storage":
            cls = PreparationSession if kind == "session" else RecordedPreparationSession
            resource = cls.__new__(cls)
            resource.benchmark_storage = storage
        restored = loads(dumps(SimpleNamespace(resource=resource, alias=resource)))
        assert restored.resource is restored.alias
        with pytest.raises(RuntimeError, match="process-local resource"):
            restored.resource.numeric_storage
        assert storage.numeric_storage is registry and _NUMERIC_STORAGE.get() is registry


def test_native_lock_methods_are_not_rebound_in_worker():
    from threading import Lock

    lock = Lock()
    restored = loads(dumps(SimpleNamespace(lock=lock, acquire=lock.acquire)))
    assert not lock.locked()
    with pytest.raises(RuntimeError, match='process-local resource'):
        restored.lock.acquire


def _probe_worker_runtime():
    import os
    import torch
    from ignition import runtime_patches

    torch_root = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
    return (
        runtime_patches._APPLIED,
        os.environ.get("PYTHONPATH", "").split(os.pathsep),
        torch_root,
    )


@pytest.mark.parametrize('operator', [torch.ops.aten, torch.ops.aten.add, torch.ops.aten.add.Tensor])
def test_resolvable_torch_operator_bindings_survive_worker_roundtrip(operator):
    restored = loads(dumps(SimpleNamespace(operator=operator)))
    assert restored.operator is operator
    if operator is torch.ops.aten:
        result = restored.operator.add.Tensor(torch.ones(1), torch.ones(1))
    else:
        result = restored.operator(torch.ones(1), torch.ones(1))
    assert torch.equal(result, torch.full((1,), 2.))


def test_custom_resolvable_and_rank_local_torch_ops_roundtrip_accurately():
    from ignition.preparation import compilation

    assert compilation._snapshot is not None
    restored_global = loads(dumps(SimpleNamespace(op=torch.ops.flare.snapshot.default)))
    assert restored_global.op is torch.ops.flare.snapshot.default

    @torch.library.custom_op("flare_test_local_ops::rank_only", mutates_args=())
    def _rank_only(x: torch.Tensor) -> torch.Tensor:
        return x

    restored_local = loads(dumps(SimpleNamespace(
        overload=torch.ops.flare_test_local_ops.rank_only.default,
        packet=torch.ops.flare_test_local_ops.rank_only,
        missing_ns=torch.ops.flare_nonexistent_ns_474,
    )))
    with pytest.raises(RuntimeError, match='process-local resource Torch operator binding'):
        restored_local.overload(torch.ones(1))
    with pytest.raises(RuntimeError, match='process-local resource Torch operator binding'):
        restored_local.packet(torch.ones(1))
    with pytest.raises(RuntimeError, match='process-local resource Torch operator binding'):
        restored_local.missing_ns.anything


def test_anonymous_builtin_method_is_process_local():
    import types

    anonymous = torch.ops.aten.add.Tensor._op
    assert isinstance(anonymous, types.BuiltinMethodType) and not anonymous.__name__
    restored = loads(dumps(SimpleNamespace(method=anonymous)))
    with pytest.raises(RuntimeError, match=r'process-local resource .*\.<anonymous>'):
        restored.method()


def test_importable_native_extensions_remain_available_to_workers(monkeypatch):
    import sys
    import types
    from importlib.machinery import ExtensionFileLoader, ModuleSpec

    import torch._C

    assert isinstance(torch._C.__spec__.loader, ExtensionFileLoader)
    restored = loads(dumps(SimpleNamespace(module=torch._C)))
    assert restored.module is torch._C

    editable_name = "flare_editable_ext_test"
    editable = types.ModuleType(editable_name)
    editable_spec = ModuleSpec(editable_name, ExtensionFileLoader(editable_name, "/virtual/editable_ext.so"))
    editable.__spec__ = editable_spec

    class EditableFinder:
        @staticmethod
        def find_spec(fullname, path=None, target=None):
            return editable_spec if fullname == editable_name else None

    monkeypatch.setitem(sys.modules, editable_name, editable)
    monkeypatch.setattr(sys, "meta_path", [EditableFinder, *sys.meta_path])
    restored_editable = loads(dumps(SimpleNamespace(module=editable)))
    assert restored_editable.module is editable


def test_unimportable_native_extension_module_is_process_local():
    import types
    from importlib.machinery import ExtensionFileLoader, ModuleSpec

    extension = types.ModuleType("flare_nonexistent_jit_ext")
    extension.__spec__ = ModuleSpec("flare_nonexistent_jit_ext",
                                    ExtensionFileLoader("flare_nonexistent_jit_ext", "/nonexistent/ext.so"))
    restored = loads(dumps(SimpleNamespace(module=extension)))
    with pytest.raises(RuntimeError, match="process-local resource native extension flare_nonexistent_jit_ext"):
        restored.module.launch


def test_metadata_preserves_custom_tensor_attributes_and_ignores_lazy_fake_attrs():
    from torch._subclasses import FakeTensorMode
    from ignition.preparation.forward_metadata import _tensor_user_attributes

    plain = torch.empty(16, dtype=torch.float32)
    annotated = torch.empty((16, 32), dtype=torch.float8_e4m3fn)
    annotated.sf = torch.empty((16, 1), dtype=torch.int32)
    annotated.scale_mode = "block"
    restored = loads(dumps(SimpleNamespace(plain=plain, annotated=annotated)))
    assert isinstance(restored.plain, FakeTensor)
    assert isinstance(restored.annotated, FakeTensor)
    assert not hasattr(restored.plain, "sf")
    assert isinstance(restored.annotated.sf, FakeTensor)
    assert tuple(restored.annotated.sf.shape) == (16, 1)
    assert restored.annotated.sf.dtype == torch.int32
    assert restored.annotated.scale_mode == "block"

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    fake = mode.from_tensor(torch.ones(4))
    fake.constant = torch.ones(4)
    fake.real_tensor = torch.ones(4)
    fake.item_memo = 1.0
    fake.pytype = torch.Tensor
    fake._debug_trace = "trace"
    fake.user_tag = "keep"
    assert _tensor_user_attributes(fake) == {"user_tag": "keep"}


import enum


class _DynamicIgnitionEnum(enum.Enum):
    __module__ = "ignition.test_enum_identity"
    ALPHA = "alpha"

    def __getattr__(self, name):
        raise AttributeError(name)


def test_metadata_preserves_ignition_enum_and_builtin_subclass_identity(monkeypatch):
    import sys
    import types
    from ignition.preparation.frontend import PromptEncoder

    member = PromptEncoder.PROMPT_JINJA
    assert loads(dumps(member)) is member

    mod = types.ModuleType("ignition.test_enum_identity")
    mod._DynamicIgnitionEnum = _DynamicIgnitionEnum
    monkeypatch.setitem(sys.modules, "ignition.test_enum_identity", mod)
    assert loads(dumps(_DynamicIgnitionEnum.ALPHA)) is _DynamicIgnitionEnum.ALPHA


def test_compile_pool_starts_real_worker_with_runtime_patches_and_sys_path_precedence(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    if os.name == "nt":
        pytest.skip("SubprocPool child runtime-patch test targets POSIX SubprocPool")

    sentinel = str(tmp_path / "overlay-site-packages")
    code = '''
import os
import sys
from pathlib import Path

sentinel = sys.argv[1]
output = Path(sys.argv[2])
sys.path.insert(0, sentinel)

from ignition import runtime_patches
from ignition.preparation import compile_pool
from tests.preparation.test_compile_pool import _probe_worker_runtime

assert runtime_patches._APPLIED is False
with compile_pool.CompilePool("cuda:0", workers=1) as pool:
    future = pool.submit_payload(b"invalid", output)
    try:
        future.result(timeout=30)
    except Exception:
        pass
    assert runtime_patches._APPLIED is True
    worker_applied, worker_pythonpath, torch_root = pool.pool.submit(_probe_worker_runtime).result(timeout=30)
assert worker_applied is True
assert worker_pythonpath[0] == sentinel
assert worker_pythonpath[-1] == torch_root
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    subprocess.run([sys.executable, "-c", code, sentinel, str(tmp_path / "dummy.bin")],
                   check=True, env=env, capture_output=True, text=True)


def test_selected_plan_metadata_survives_without_registering_a_worker_plan():
    from b12x.preparation.types import Plan, plan_from_handle, require_prepared

    plan = Plan(contract=SimpleNamespace(component_id='test.layout'), query=None,
                _compile_jobs=lambda *args: (), _memory_requirements=lambda *args: None,
                _materialize=lambda *args: None)
    selected = SimpleNamespace(closed=False, state=SimpleNamespace(tiles=17, scratch=torch.empty(32)),
                               selection=SimpleNamespace(config='selected'))
    object.__setattr__(plan, '_prepared', selected)
    restored = loads(dumps(plan))
    state = require_prepared(restored, 'test.layout')
    assert state.tiles == 17 and isinstance(state.scratch, FakeTensor)
    assert restored.selection.config == 'selected'
    assert restored.handle == plan.handle and plan_from_handle(plan.handle) is plan


def test_attention_layout_restores_before_forwarding_attribute_lookups():
    from b12x.attention._shared.contiguous.api import AttentionPlan, VarlenAttentionPlan

    for kind in (AttentionPlan, VarlenAttentionPlan):
        plan = kind(key=SimpleNamespace(tokens=128), compiled=object(), cutlass_dtype=float)
        restored = loads(dumps(plan))
        assert restored.tokens == 128



# A discrete GPU's worker holds 1 GiB, so 3 GiB free compile three forwards at once.
@pytest.mark.parametrize(("requested", "free_gib", "workers", "threads"),
                         [(None, 95, 8, 1), (None, 3, 3, 1), (32, 95, 32, 1), (32, 3, 3, 1)])
def test_compile_pool_sizes_workers_from_free_device_memory(monkeypatch, requested, free_gib, workers, threads):
    from ignition.preparation import compile_pool

    monkeypatch.setitem(compile_pool._OPTIONS, "compile_threads", 16)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_gib << 30, 96 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr("ignition.preparation.host_memory.available_bytes", lambda: 1032 << 30)
    monkeypatch.setattr(compile_pool, "cpu_workers", lambda requested=None: requested or 64)
    pool = compile_pool.CompilePool("cuda:0", compile_workers=requested)
    assert (pool.workers, pool.options["compile_threads"]) == (workers, threads)


@pytest.mark.parametrize("integrated,workers,threads", [(False, 2, 1), (True, 1, 1)])
def test_compiler_reuses_its_budgeted_worker_pool(monkeypatch, tmp_path, integrated, workers, threads):
    from concurrent.futures import Future
    from torch._inductor.compile_worker import subproc_pool
    from ignition.preparation import compile_pool, host_memory

    monkeypatch.setitem(compile_pool._OPTIONS, "compile_threads", 16)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (72 << 30, 121 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=integrated))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 72 << 30)
    executors = []

    class Executor:
        def __init__(self, count, *, kind=None, mp_context=None):
            self.calls, self.closed = [], False
            assert count == workers
            if compile_pool.os.name == "nt":
                assert mp_context.get_start_method() == "spawn"
            else:
                assert kind == subproc_pool.SubprocKind.SPAWN
            executors.append(self)

        def submit(self, function, *arguments):
            self.calls.append(arguments)
            future = Future()
            future.set_result(None)
            return future

        def shutdown(self):
            self.closed = True

    monkeypatch.setattr(subproc_pool, "SubprocPool", Executor)
    monkeypatch.setattr(compile_pool, "ProcessPoolExecutor", Executor)
    with compile_pool.CompilePool("cuda:0") as pool:
        pool.submit(SimpleNamespace(), tmp_path / "first.bin")
        pool.submit(SimpleNamespace(), tmp_path / "second.bin")
    (executor,) = executors
    assert [call[3] for call in executor.calls] == [str(tmp_path / "first.bin"), str(tmp_path / "second.bin")]
    assert all(call[2]["compile_threads"] == threads for call in executor.calls)
    assert executor.closed and compile_pool._OPTIONS["compile_threads"] == 16


@pytest.mark.parametrize("integrated,workers", [(False, 2), (True, 1)])
def test_compilation_preserves_configured_headroom(monkeypatch, integrated, workers):
    from ignition.preparation import compile_pool, host_memory

    monkeypatch.setitem(compile_pool._OPTIONS, "compile_threads", 16)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (72 << 30, 121 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=integrated))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 72 << 30)
    assert compile_pool.CompilePool("cuda:0", headroom_bytes=16 << 30).workers == workers
    with pytest.raises(MemoryError, match="headroom"):
        compile_pool.CompilePool("cuda:0", headroom_bytes=(72 << 30) - (512 << 20))


@pytest.mark.parametrize("pools,workers", [(1, [2]), (2, [1, 1]), (4, [1, 1, 0, 0])])
def test_ranks_sharing_a_host_split_its_compilers(monkeypatch, pools, workers):
    from ignition.preparation import compile_pool, host_memory

    monkeypatch.setitem(compile_pool._OPTIONS, "compile_threads", 16)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (72 << 30, 96 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 72 << 30)
    resources = [compile_pool.compiler_resources("cuda:0") for _ in range(pools)]
    assert compile_pool.allocate_workers(resources, [16] * pools, ["host"] * pools) == workers


def test_windows_compiler_workers_fit_host_ram_with_larger_device_budget(monkeypatch):
    from ignition.preparation import compile_pool, host_memory

    monkeypatch.setattr(compile_pool, "os", SimpleNamespace(name="nt"))
    monkeypatch.setitem(compile_pool._OPTIONS, "compile_threads", 16)
    monkeypatch.setattr(compile_pool, "device_memory_info", lambda device: (105 << 30, 110 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=True))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 48 << 30)
    pool = compile_pool.CompilePool("cuda:0", headroom_bytes=8 << 30)
    assert (pool.workers, pool.options["compile_threads"]) == (1, 1)


def test_host_admission_leaves_zero_worker_ranks_when_ram_cannot_cover_every_owner():
    from ignition.preparation.compile_pool import CompilerResources, allocate_workers

    resources = [CompilerResources(32, 104 << 30, 16, 32 << 30, 8 << 30)] * 16
    workers = allocate_workers(resources, [8] * 16, ["host"] * 16)
    assert workers == [1] * 3 + [0] * 13
    assert sum(workers) * (32 << 30) + (8 << 30) == 104 << 30


@pytest.mark.parametrize("limit,capacity,jobs,expected", [(3, 8, 100, 3), (16, 1, 100, 2), (16, 8, 1, 1)])
def test_host_admission_respects_cpu_device_and_missing_work_limits(limit, capacity, jobs, expected):
    from ignition.preparation.compile_pool import CompilerResources, allocate_workers

    resources = [CompilerResources(capacity, 1024 << 30, limit, 32 << 30, 8 << 30)] * 2
    workers = allocate_workers(resources, [jobs, 0], ["host"] * 2)
    assert sum(workers) == expected
    assert max(workers) <= capacity


def test_shared_device_admission_counts_host_and_device_storage_once():
    from ignition.preparation.compile_pool import CompilerResources, allocate_workers

    resources = [CompilerResources(16, 76 << 30, 16, 34 << 30, 8 << 30)] * 4
    assert allocate_workers(resources, [10] * 4, ["host"] * 4) == [1, 1, 0, 0]
    resources = [CompilerResources(16, 84 << 30, 16, 34 << 30, 16 << 30)] * 4
    assert allocate_workers(resources, [10] * 4, ["host"] * 4) == [1, 1, 0, 0]


def test_host_admission_uses_conservative_peer_samples_and_independent_host_budgets():
    from ignition.preparation.compile_pool import CompilerResources, allocate_workers

    resources = [CompilerResources(8, free << 30, limit, 32 << 30, 8 << 30)
                 for free, limit in [(136, 16), (136, 4), (72, 16), (136, 16)]]
    assert allocate_workers(resources, [10] * 4, ["a", "b", "a", "b"]) == [1, 2, 1, 2]


def test_cached_host_needs_no_compiler_capacity_but_missing_work_requires_one():
    from ignition.preparation.compile_pool import CompilerResources, allocate_workers

    resources = [CompilerResources(0, 0, 16, 32 << 30, 8 << 30)]
    assert allocate_workers(resources, [0], ["cached"]) == [0]
    with pytest.raises(MemoryError, match="host 'missing'.*1 missing forwards.*no capacity"):
        allocate_workers(resources, [1], ["missing"])


@pytest.mark.parametrize("shared_bytes,available", [(0, 10), (2, 12), (2, 24)])
def test_minimum_worker_allowance_preserves_one_worker_per_host(shared_bytes, available):
    from ignition.preparation.compile_pool import CompilerResources, allocate_workers

    resources = [CompilerResources(8, available << 30, 16, (32 + shared_bytes) << 30, 8 << 30)] * 4
    assert allocate_workers(resources, [10] * 4, ["a", "b", "a", "b"]) == [1, 1, 0, 0]
    resources = [CompilerResources(8, 11 << 30, 16, 34 << 30, 8 << 30)] * 2
    with pytest.raises(MemoryError, match="no capacity"):
        allocate_workers(resources, [10] * 2, ["a"] * 2)


def test_explicit_forward_workers_are_bounded_by_cpu_affinity(monkeypatch):
    from ignition.preparation import compile_pool, host_memory

    monkeypatch.setattr(host_memory.os, "sched_getaffinity", lambda _: set(range(3)))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 1024 << 30)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (96 << 30, 96 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    assert compile_pool.CompilePool("cuda:0", compile_workers=32).workers == 3


def test_admitted_pool_does_not_resample_resources(monkeypatch):
    from ignition.preparation import compile_pool

    monkeypatch.setattr(compile_pool, "compiler_resources", lambda *args, **kwargs: pytest.fail("resampled"))
    pool = compile_pool.CompilePool("cuda:0", workers=2)
    assert pool.workers == 2 and pool.options["compile_threads"] == 1


@pytest.mark.parametrize("recorded", [False, True])
def test_compiler_does_not_serialize_preparation_session_resources(recorded):
    from contextvars import ContextVar
    from b12x.preparation import PreparationSession
    from ignition.preparation.tuning import RecordedPreparationSession

    session = object.__new__(RecordedPreparationSession if recorded else PreparationSession)
    context = ContextVar("test_preparation_resource")
    token = context.set("owning rank")
    session.resource = token
    try:
        original = SimpleNamespace(session=session, selected=SimpleNamespace(tile=64),
                                   source=torch.empty(16))
        restored = loads(dumps(original))
        assert restored.selected.tile == 64
        assert isinstance(restored.source, FakeTensor)
        with pytest.raises(RuntimeError, match="process-local resource b12x preparation session"):
            restored.session.prepare()
        assert session.resource is token
    finally:
        context.reset(token)


def test_describe_workers_reports_host_ram_gpu_and_cpu_clamping():
    from ignition.preparation.compile_pool import CompilerResources, allocate_workers, describe_workers

    ram_limited = [CompilerResources(16, 72 << 30, 16, 32 << 30, 8 << 30)] * 4
    workers = allocate_workers(ram_limited, [10] * 4, ["host"] * 4)
    assert describe_workers(ram_limited, [10] * 4, workers, ["host"] * 4, requested=12) == (
        "2 compiler workers (requested 12; host RAM limit, 72 GiB avail); per rank [1, 1, 0, 0]"
    )

    gpu_limited = [CompilerResources(2, 512 << 30, 16, 32 << 30, 8 << 30)]
    workers = allocate_workers(gpu_limited, [8], ["host"])
    assert describe_workers(gpu_limited, [8], workers, ["host"], requested=8) == (
        "2 compiler workers (requested 8; GPU memory limit)"
    )

    cpu_limited = [CompilerResources(16, 512 << 30, 3, 32 << 30, 8 << 30)]
    workers = allocate_workers(cpu_limited, [8], ["host"])
    assert describe_workers(cpu_limited, [8], workers, ["host"], requested=8) == (
        "3 compiler workers (requested 8; CPU limit 3)"
    )

    unclamped = [CompilerResources(16, 512 << 30, 16, 32 << 30, 8 << 30)]
    workers = allocate_workers(unclamped, [2], ["host"])
    assert describe_workers(unclamped, [2], workers, ["host"], requested=12) == "2 compiler workers"
