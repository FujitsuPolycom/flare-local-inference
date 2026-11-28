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
    import torch._inductor.async_compile  # noqa: F401
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


def test_peak_rss_bytes_reports_process_memory():
    from ignition.preparation.compile_pool import _peak_rss_bytes

    assert _peak_rss_bytes() > (1 << 20)


def _simulate_host(monkeypatch, available, *, free=30 << 30, cpus=16):
    from ignition.preparation import compile_pool, host_memory

    state = {"available": available, "free": free}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (state["free"], 32 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: state["available"])
    monkeypatch.setattr(compile_pool, "cpu_workers",
                        lambda requested=None, *, pools=1: max(1, (requested or cpus) // pools))
    return state


def test_worker_pool_grows_by_completed_waves_within_host_budget(monkeypatch):
    from ignition.preparation import compile_pool

    state = _simulate_host(monkeypatch, 31 << 30)
    # 31 GiB minus 8 GiB headroom cannot hold a 32 GiB forward; the minimum allowance keeps one worker.
    pool = compile_pool.CompilePool("cuda:0")
    assert (pool.workers, pool.max_workers) == (1, 16)
    state["available"] = 29 << 30
    # A 2 GiB peak costs 2.5 GiB with margin, so 23 GiB of pre-spawn budget caps growth at nine workers.
    assert pool.observe_worker_rss(2 << 30) == 2
    assert pool.observe_worker_rss(2 << 30) == 4
    # A two-worker increment requires two completions before doubling again.
    assert pool.observe_worker_rss(2 << 30) == 4
    assert pool.observe_worker_rss(2 << 30) == 8
    for _ in range(3):
        assert pool.observe_worker_rss(2 << 30) == 8
    assert pool.observe_worker_rss(2 << 30) == 9
    assert pool.observe_worker_rss(2 << 30) == 9
    # A later heavy forward shrinks the admitted concurrency back to the host's safe ceiling.
    assert pool.observe_worker_rss(16 << 30) == 1


def test_multi_rank_host_pools_partition_host_memory_without_oversubscribing(monkeypatch):
    from ignition.preparation import compile_pool

    state = _simulate_host(monkeypatch, 56 << 30, cpus=16)
    pools = [compile_pool.CompilePool("cuda:0", workers=1, max_workers=4, host_pools=4,
                                      host_available=56 << 30)
             for _ in range(4)]
    state["available"] = 28 << 30
    # 20 GiB live remainder above 8 GiB headroom fits four 5 GiB workers total: one extra per rank.
    assert [pool.observe_worker_rss(4 << 30) for pool in pools] == [2, 2, 2, 2]
    # 48 GiB total pre-spawn budget above headroom caps four pools at two 5 GiB workers each (40 GiB).
    for pool in pools:
        assert pool.observe_worker_rss(4 << 30) == 2
        assert pool.observe_worker_rss(4 << 30) == 2


def test_huge_reported_peak_keeps_the_pool_at_one_worker(monkeypatch):
    from ignition.preparation import compile_pool

    state = _simulate_host(monkeypatch, 31 << 30)
    pool = compile_pool.CompilePool("cuda:0")
    state["available"] = 26 << 30
    assert pool.observe_worker_rss(28 << 30) == 1
    assert pool.peak_worker_rss == 28 << 30


def test_explicit_compile_workers_bounds_peak_rss_growth(monkeypatch):
    from ignition.preparation import compile_pool

    state = _simulate_host(monkeypatch, 72 << 30)
    pool = compile_pool.CompilePool("cuda:0", compile_workers=2)
    assert (pool.workers, pool.max_workers) == (2, 2)
    state["available"] = 200 << 30
    assert pool.observe_worker_rss(1 << 30) == 2

    # When 32 GiB startup admission clamps below an explicit compile_workers=8, measured RSS grows up to 8.
    state["available"] = 72 << 30
    capped = compile_pool.CompilePool("cuda:0", compile_workers=8)
    assert (capped.workers, capped.max_workers) == (2, 8)
    state["available"] = 64 << 30
    assert capped.observe_worker_rss(4 << 30) == 4
    assert capped.observe_worker_rss(4 << 30) == 4
    assert capped.observe_worker_rss(4 << 30) == 8
    for _ in range(5):
        assert capped.observe_worker_rss(4 << 30) == 8
    # Remaining work caps growth when fewer forwards remain than the next wave.
    state["available"] = 72 << 30
    tail = compile_pool.CompilePool("cuda:0", compile_workers=8)
    assert tail.observe_worker_rss(4 << 30, remaining=3) == 3


def test_worker_growth_adds_pool_shards_and_shutdown_closes_each(monkeypatch, tmp_path):
    from concurrent.futures import Future
    import torch._inductor.async_compile  # noqa: F401
    from torch._inductor.compile_worker import subproc_pool
    from ignition.preparation import compile_pool

    state = _simulate_host(monkeypatch, 31 << 30)
    executors = []

    class Executor:
        def __init__(self, count, *, kind=None, mp_context=None):
            self.count, self.calls, self.closed, self.fail_next = count, [], False, False
            executors.append(self)

        def submit(self, function, *arguments):
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("executor rejected submission")
            future = Future()
            self.calls.append((arguments, future))
            return future

        def shutdown(self):
            self.closed = True

    monkeypatch.setattr(subproc_pool, "SubprocPool", Executor)
    monkeypatch.setattr(compile_pool, "ProcessPoolExecutor", Executor)
    with compile_pool.CompilePool("cuda:0") as pool:
        first = pool.submit_payload("first", tmp_path / "first.bin")
        assert [executor.count for executor in executors] == [1]
        assert pool.pool is executors[0]
        state["available"] = 26 << 30
        assert pool.observe_worker_rss(4 << 30) == 2
        second = pool.submit_payload("second", tmp_path / "second.bin")
        # The first shard is fully busy, so the expanded capacity starts a second shard.
        assert [executor.count for executor in executors] == [1, 1]
        first.set_result(1 << 20)
        second.set_result(1 << 20)
        executors[0].fail_next = True
        with pytest.raises(RuntimeError, match="executor rejected submission"):
            pool.submit_payload("failed", tmp_path / "failed.bin")
        assert pool._shard_pending == [0, 0]
        third = pool.submit_payload("third", tmp_path / "third.bin")
        fourth = pool.submit_payload("fourth", tmp_path / "fourth.bin")
        # Released slots take the next forwards, one per shard.
        assert [executor.calls[0][0][0] for executor in executors] == ["first", "second"]
        assert [executor.calls[1][0][0] for executor in executors] == ["third", "fourth"]
        third.set_result(1 << 20)
        fourth.set_result(1 << 20)
    assert all(executor.closed for executor in executors)


def test_shutdown_closes_remaining_shards_when_one_raises(monkeypatch, tmp_path):
    from concurrent.futures import Future
    import torch._inductor.async_compile  # noqa: F401
    from torch._inductor.compile_worker import subproc_pool
    from ignition.preparation import compile_pool

    state = _simulate_host(monkeypatch, 31 << 30)
    executors = []

    class Executor:
        def __init__(self, count, *, kind=None, mp_context=None):
            self.calls, self.closed = [], False
            executors.append(self)

        def submit(self, function, *arguments):
            future = Future()
            self.calls.append((arguments, future))
            return future

        def shutdown(self):
            self.closed = True
            if self is executors[1]:
                raise RuntimeError("shutdown failed")

    monkeypatch.setattr(subproc_pool, "SubprocPool", Executor)
    monkeypatch.setattr(compile_pool, "ProcessPoolExecutor", Executor)
    pool = compile_pool.CompilePool("cuda:0")
    pool.submit_payload("first", tmp_path / "first.bin")
    state["available"] = 26 << 30
    pool.observe_worker_rss(4 << 30)
    pool.submit_payload("second", tmp_path / "second.bin")
    with pytest.raises(RuntimeError, match="shutdown failed"):
        pool.__exit__(None, None, None)
    assert all(executor.closed for executor in executors)


def test_compiler_resources_reclaims_cached_allocator_segments_when_cuda_initialized(monkeypatch):
    from ignition.preparation import compile_pool, host_memory

    gib = 1 << 30
    state = {"free": round(2.65 * gib), "available": 64 * gib}
    events = []

    def empty_cache():
        events.append("empty_cache")
        state["free"] = round(11.4 * gib)
        state["available"] = 72 * gib

    def available_bytes():
        events.append("available_bytes")
        return state["available"]

    def mem_get_info(device):
        events.append("mem_get_info")
        return state["free"], 96 * gib

    monkeypatch.setitem(compile_pool._OPTIONS, "compile_threads", 16)
    monkeypatch.setattr(compile_pool, "cpu_workers", lambda requested=None: requested or 16)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(host_memory, "available_bytes", available_bytes)
    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    uninitialized = compile_pool.compiler_resources("cuda:0")
    assert events == ["available_bytes", "mem_get_info"]
    assert (uninitialized.capacity, uninitialized.max_capacity, uninitialized.available) == (2, 2, 64 * gib)

    events.clear()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    initialized = compile_pool.compiler_resources("cuda:0")
    assert events == ["empty_cache", "available_bytes", "mem_get_info"]
    assert (initialized.capacity, initialized.max_capacity, initialized.available) == (8, 11, 72 * gib)

    state["free"] = round(2.65 * gib)
    explicit = compile_pool.compiler_resources("cuda:0", compile_workers=16)
    assert (explicit.capacity, explicit.max_capacity) == (11, 11)


def test_observe_worker_rss_counts_spawned_shards_once_and_respects_external_vram_drop(monkeypatch, tmp_path):
    from concurrent.futures import Future
    import torch._inductor.async_compile  # noqa: F401
    from torch._inductor.compile_worker import subproc_pool
    from ignition.preparation import compile_pool

    gib = 1 << 30
    worker_vram = round(0.8 * gib)
    state = _simulate_host(monkeypatch, 31 * gib, free=round(4.5 * gib), cpus=16)
    executors = []

    class Executor:
        def __init__(self, count, *, kind=None, mp_context=None):
            self.count, self.calls, self.closed = count, [], False
            state["free"] -= count * worker_vram
            executors.append(self)

        def submit(self, function, *arguments):
            future = Future()
            self.calls.append((arguments, future))
            return future

        def shutdown(self):
            self.closed = True

    monkeypatch.setattr(subproc_pool, "SubprocPool", Executor)
    monkeypatch.setattr(compile_pool, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    with compile_pool.CompilePool("cuda:0") as pool:
        assert (pool.workers, pool.max_workers) == (1, 4)
        state["available"] = 29 * gib

        # Wave 1 spawns 1 worker: live free VRAM drops 4.5 -> 3.7 GiB.
        first = pool.submit_payload("w1", tmp_path / "w1.bin")
        assert state["free"] == round(3.7 * gib)
        first.set_result(2 * gib)
        assert pool.observe_worker_rss(2 * gib) == 2

        # Wave 2 spawns 1 additional worker: live free VRAM drops 3.7 -> 2.9 GiB.
        second = pool.submit_payload("w2", tmp_path / "w2.bin")
        third = pool.submit_payload("w3", tmp_path / "w3.bin")
        assert state["free"] == round(2.9 * gib)
        second.set_result(2 * gib)
        assert pool.observe_worker_rss(2 * gib) == 4
        third.set_result(2 * gib)

        # Wave 3 spawns 2 additional workers: live free VRAM drops 2.9 -> 1.3 GiB.
        wave3 = [pool.submit_payload(f"w{index}", tmp_path / f"w{index}.bin") for index in range(4, 8)]
        assert [executor.count for executor in executors] == [1, 1, 2]
        assert abs(state["free"] - round(1.3 * gib)) <= 1
        for future in wave3:
            future.set_result(2 * gib)
            assert pool.observe_worker_rss(2 * gib) == 4

    executors.clear()
    state["available"] = 31 * gib
    state["free"] = round(4.5 * gib)
    with compile_pool.CompilePool("cuda:0") as constrained:
        state["available"] = 29 * gib
        first = constrained.submit_payload("c1", tmp_path / "c1.bin")
        first.set_result(2 * gib)
        assert constrained.observe_worker_rss(2 * gib) == 2

        second = constrained.submit_payload("c2", tmp_path / "c2.bin")
        third = constrained.submit_payload("c3", tmp_path / "c3.bin")
        assert [executor.count for executor in executors] == [1, 1]
        # External GPU occupancy drops live free VRAM below 1 GiB after two shards have started.
        state["free"] = round(0.7 * gib)
        second.set_result(2 * gib)
        assert constrained.observe_worker_rss(2 * gib) == 2
        third.set_result(2 * gib)
        assert constrained.observe_worker_rss(2 * gib) == 2

        # When free VRAM recovers for growth and then drops externally before additional shards spawn,
        # concurrency shrinks back to the already-spawned shard capacity.
        state["free"] = round(2.9 * gib)
        assert constrained.observe_worker_rss(2 * gib) == 4
        state["free"] = round(0.7 * gib)
        assert constrained.observe_worker_rss(2 * gib) == 2


def test_compile_pool_reconfigure_expands_in_place_and_validates_arguments(monkeypatch, tmp_path):
    from concurrent.futures import Future
    import torch._inductor.async_compile  # noqa: F401
    from torch._inductor.compile_worker import subproc_pool
    from ignition.preparation import compile_pool

    gib = 1 << 30
    state = _simulate_host(monkeypatch, 56 * gib, free=20 * gib, cpus=32)
    executors = []

    class Executor:
        def __init__(self, count, *, kind=None, mp_context=None):
            self.count, self.calls, self.closed = count, [], False
            executors.append(self)

        def submit(self, function, *arguments):
            future = Future()
            self.calls.append((arguments, future))
            return future

        def shutdown(self):
            self.closed = True

    monkeypatch.setattr(subproc_pool, "SubprocPool", Executor)
    monkeypatch.setattr(compile_pool, "ProcessPoolExecutor", Executor)

    with compile_pool.CompilePool(
        "cuda:0", workers=1, max_workers=1, host_pools=2, host_available=56 * gib
    ) as pool:
        probe = pool.submit_payload("probe", tmp_path / "probe.bin")
        assert [executor.count for executor in executors] == [1]
        probe.set_result(2 * gib)
        assert pool.observe_worker_rss(2 * gib) == 1

        for bad_pools in (0, -1, True, 1.5, None):
            with pytest.raises(ValueError, match="host_pools must be a positive integer"):
                pool.reconfigure(workers=2, max_workers=4, host_pools=bad_pools)
        for bad_workers in (0, -1, True, 2.0, None):
            with pytest.raises(ValueError, match="positive admitted worker count"):
                pool.reconfigure(workers=bad_workers, max_workers=4, host_pools=4)
        for bad_max in (2, 0, True, 3.5, None):
            with pytest.raises(ValueError, match="maximum must cover its admitted worker count"):
                pool.reconfigure(workers=3, max_workers=bad_max, host_pools=4)
        assert (pool.workers, pool.max_workers, pool.host_pools) == (1, 1, 2)

        pool.reconfigure(workers=3, max_workers=6, host_pools=4)
        assert (pool.workers, pool.max_workers, pool.host_pools) == (3, 6, 4)
        assert pool.pool is executors[0] and not executors[0].closed
        assert len(executors) == 1

        in_flight = [pool.submit_payload(f"job{index}", tmp_path / f"job{index}.bin") for index in range(4)]
        assert [executor.count for executor in executors] == [1, 2]
        assert not executors[0].closed and not executors[1].closed

        state["available"] = 48 * gib
        in_flight[0].set_result(2 * gib)
        # Two slots were added beyond the existing 1-worker shard, so the wave needs two completions.
        assert pool.observe_worker_rss(2 * gib) == 3
        in_flight[1].set_result(2 * gib)
        # Reconfigured host_pools=4 partitions the 48 GiB host budget above 8 GiB headroom into 4 workers per pool.
        assert pool.observe_worker_rss(2 * gib) == 4
        for future in in_flight[2:]:
            future.set_result(2 * gib)
    assert all(executor.closed for executor in executors)

    shrunk = compile_pool.CompilePool("cuda:0", workers=4, max_workers=4, host_pools=1, host_available=56 * gib)
    shrunk.reconfigure(workers=2, max_workers=6, host_pools=4)
    assert shrunk.observe_worker_rss(20 * gib) == 2

    # Default reconfigure updates the floor to `workers`, while an explicit `min_workers=1`
    # lets post-probe pools shrink back down to 1 if a subsequent forward spikes host RSS.
    pinned = compile_pool.CompilePool("cuda:0", workers=1, max_workers=1, host_pools=2, host_available=56 * gib)
    pinned.reconfigure(workers=4, max_workers=4, host_pools=4)
    assert pinned.observe_worker_rss(20 * gib) == 4

    shrinkable = compile_pool.CompilePool("cuda:0", workers=4, max_workers=4, min_workers=1,
                                          host_pools=4, host_available=56 * gib)
    assert shrinkable.observe_worker_rss(20 * gib) == 1
    reconfigured_shrinkable = compile_pool.CompilePool("cuda:0", workers=1, max_workers=1,
                                                       host_pools=2, host_available=56 * gib)
    reconfigured_shrinkable.reconfigure(workers=4, max_workers=4, host_pools=4, min_workers=1)
    assert reconfigured_shrinkable.observe_worker_rss(20 * gib) == 1
    for bad_min in (0, -1, 5, True, 1.5):
        with pytest.raises(ValueError, match="minimum must be between 1"):
            reconfigured_shrinkable.reconfigure(workers=4, max_workers=4, host_pools=4, min_workers=bad_min)


def test_post_probe_sized_resources_preserve_minimum_worker_floor(monkeypatch):
    from dataclasses import replace
    from ignition.preparation import compile_pool, host_memory

    gib = 1 << 30
    monkeypatch.setitem(compile_pool._OPTIONS, "compile_threads", 16)
    monkeypatch.setattr(compile_pool, "cpu_workers", lambda requested=None: requested or 16)

    # Heavy probe RSS (50 GiB with margin) on a discrete-GPU host with 24 GiB available (16 GiB above headroom):
    # initial admission granted 1 worker under the 2 GiB minimum allowance, and post-probe sizing must keep 1 worker.
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (40 * gib, 80 * gib))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 24 * gib)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    base_discrete = [compile_pool.compiler_resources("cuda:0") for _ in range(4)]
    assert compile_pool.allocate_workers(base_discrete, [5] * 4, ["host"] * 4) == [1, 0, 0, 0]
    heavy_sized = [replace(res, worker_bytes=50 * gib) for res in base_discrete]
    assert compile_pool.allocate_workers(heavy_sized, [4, 5, 5, 5], ["host"] * 4) == [1, 0, 0, 0]

    # Light probe RSS (4 GiB + 2 GiB shared device = 6 GiB) on an integrated GPU whose remaining memory (3 GiB)
    # is below the 4 GiB minimum (2 GiB host + 2 GiB shared device) must still reject instead of underflowing.
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=True))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (11 * gib, 128 * gib))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 11 * gib)
    base_integrated = [compile_pool.compiler_resources("cuda:0") for _ in range(2)]
    light_sized = [replace(res, worker_bytes=6 * gib) for res in base_integrated]
    with pytest.raises(MemoryError, match="no capacity"):
        compile_pool.allocate_workers(light_sized, [4, 4], ["spark"] * 2)


def test_integrated_gpu_small_forward_rss_does_not_oversubscribe_unified_memory(monkeypatch):
    from ignition.preparation import compile_pool, host_memory

    gib = 1 << 30
    state = {"available": round(16.4 * gib), "free": round(12.4 * gib)}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (state["free"], 122 * gib))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=True))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(host_memory, "available_bytes", lambda: state["available"])
    monkeypatch.setattr(compile_pool, "cpu_workers", lambda requested=None, *, pools=1: max(1, (requested or 20) // pools))

    # After graph measurement on GB10 (~108 GiB outside PyTorch, ~16.4 GiB MemAvailable above 0),
    # a small 1.3 GiB vision forward must not grow concurrency past 1 worker into unified-memory OOM.
    constrained = compile_pool.CompilePool("cuda:0", headroom_bytes=1 * gib)
    assert constrained.workers == 1
    state["available"] = round(12.1 * gib)
    assert constrained.observe_worker_rss(round(1.3 * gib)) == 1

    # Before KV/graph capture on GB10 (~44 GiB MemAvailable), unified memory comfortably scales to 4 workers.
    state["available"] = 44 * gib
    state["free"] = 44 * gib
    spacious = compile_pool.CompilePool("cuda:0", headroom_bytes=1 * gib)
    assert spacious.workers == 1
    state["available"] = 40 * gib
    assert spacious.observe_worker_rss(round(1.3 * gib)) == 2
    assert spacious.observe_worker_rss(round(1.3 * gib)) == 4
    assert spacious.observe_worker_rss(round(1.3 * gib)) == 4
