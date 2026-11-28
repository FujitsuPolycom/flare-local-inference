# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Restored static reduction winners retain their compiled candidate set."""

import threading
import pickle
import copy
from collections import namedtuple
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import triton

from torch._inductor.codecache import StaticAutotunerFuture

from ignition.preparation.device_binding import _freeze_tuning_winners, bind_device, relocate_inductor
from torch._inductor.runtime.hints import HeuristicType
from torch._inductor.runtime.triton_heuristics import (
    CachingAutotuner,
    StaticTritonCompileResult,
)


def restored_reduction(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU restore regression initialized CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    tuner = CachingAutotuner.__new__(CachingAutotuner)
    tuner.lock = threading.Lock()
    tuner._plugins = []
    tuner.inductor_meta = {"kernel_name": "restored_reduction"}
    tuner.filename = "/nonexistent/restored_reduction.py"
    tuner.triton_meta = {"device": 0}
    tuner.autotune_cache_info = {"autotune_cache_state": "hit"}
    tuner.device_props = SimpleNamespace(type="cuda", major=10,
        regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048,
        multi_processor_count=152, warp_size=32, warp_size_or_default=32)
    tuner.size_hints = {"x": 16384, "r0_": 4096}
    tuner.heuristic_type = HeuristicType.REDUCTION
    tuner.deterministic_mode = False
    tuner.launchers = []
    tuner.configs = None
    tuner.fn = SimpleNamespace(fn=object())
    tuner._cached_launcher = None

    def compile_config(config):
        return StaticTritonCompileResult(
            SimpleNamespace(n_regs=64), config, {}, tuner.inductor_meta)

    selected = triton.Config({"XBLOCK": 8, "R0_BLOCK": 256}, num_warps=4,
                             num_stages=1)
    tuner.compile_results = [compile_config(selected)]
    # Driver-bound compilation/loading is replaced at its boundary. Cache
    # recheck, precompile ordering and occupancy candidate generation are real.
    tuner._precompile_worker = lambda: None
    tuner._ensure_kernel_loaded = lambda: None
    tuner._precompile_config = compile_config
    tuner._make_launchers = lambda: setattr(tuner, "launchers", [
        SimpleNamespace(config=result.config) for result in tuner.compile_results])
    return tuner, selected


def test_static_restore_reintroduces_candidates_without_preserved_winner(monkeypatch):
    tuner, selected = restored_reduction(monkeypatch)
    future = StaticAutotunerFuture(tuner)
    future.reload_kernel_from_src = lambda: tuner
    assert future.result() is tuner
    assert tuner.autotune_cache_info["autotune_cache_state"] == "miss"
    assert [result.config.kwargs["R0_BLOCK"] for result in tuner.compile_results] == [256, 128, 64]
    assert tuner.compile_results[0].config is selected
    assert len(tuner.launchers) == 3


def test_prepared_winner_does_not_reopen_occupancy_search(monkeypatch):
    tuner, selected = restored_reduction(monkeypatch)
    # Cached eligibility can already exist when the artifact is copied.
    assert tuner._could_rblock_scale
    metadata = tuner.inductor_meta
    _freeze_tuning_winners(SimpleNamespace(static_autotuners=[
        SimpleNamespace(cache_key="reduction-key", kernel=tuner, kernel_name="restored_reduction")]))
    assert metadata.get("dynamic_scale_rblock") is None
    future = StaticAutotunerFuture(tuner)
    future.reload_kernel_from_src = lambda: tuner
    assert future.result() is tuner
    assert len(tuner.launchers) == len(tuner.compile_results) == 1
    assert tuner.compile_results[0].config is selected
    assert tuner.launchers[0].config.kwargs == {"XBLOCK": 8, "R0_BLOCK": 256}


def test_frozen_config_restores_exact_state_after_torch_cache_recheck(monkeypatch):
    tuner, selected = restored_reduction(monkeypatch)
    state = copy.deepcopy(vars(selected))
    tuner.compile_results[0]._flare_config_state = state
    bundle = SimpleNamespace(static_autotuners=[
        SimpleNamespace(cache_key='reduction-key', kernel=tuner, kernel_name='restored_reduction')])
    _freeze_tuning_winners(bundle)
    future = StaticAutotunerFuture(tuner)
    future.reload_kernel_from_src = lambda: tuner
    assert future.result() is tuner
    assert 'found_by_coordesc' not in state
    assert vars(selected)['found_by_coordesc'] is False
    _freeze_tuning_winners(bundle)
    assert vars(selected) == state


@pytest.mark.parametrize("pending", ["multiple", "empty", "coordesc", "combo"])
def test_prepared_winner_rejects_unresolved_selection(monkeypatch, pending):
    tuner, _ = restored_reduction(monkeypatch)
    if pending == "multiple":
        tuner.compile_results *= 2
    elif pending == "empty":
        tuner.compile_results.clear()
    elif pending == "coordesc":
        tuner.inductor_meta["coordinate_descent_tuning"] = True
    else:
        tuner.inductor_meta["combo_tuning_groups"] = [1]
    with pytest.raises(ValueError, match="unresolved|pending"):
        _freeze_tuning_winners(SimpleNamespace(static_autotuners=[
            SimpleNamespace(cache_key="reduction-key", kernel=tuner, kernel_name="restored_reduction")]))


@pytest.mark.parametrize("selection,marker", [
    ("coordinate_descent_tuning", "found_by_coordesc"),
    ("combo_tuning_groups", "found_by_combo_autotune"),
])
def test_prepared_winner_accepts_completed_refinement(monkeypatch, selection, marker):
    tuner, selected = restored_reduction(monkeypatch)
    tuner.inductor_meta[selection] = True
    setattr(selected, marker, True)
    _freeze_tuning_winners(SimpleNamespace(static_autotuners=[
        SimpleNamespace(cache_key="reduction-key", kernel=tuner, kernel_name="restored_reduction")]))
    assert tuner.compile_results[0].config is selected
    assert getattr(selected, marker)


@pytest.mark.parametrize("kind,deterministic", [
    (HeuristicType.TEMPLATE, False), (HeuristicType.USER_AUTOTUNE, False),
    (HeuristicType.FIXED, False), (HeuristicType.REDUCTION, True),
])
def test_ineligible_coordinate_descent_needs_no_completion_marker(monkeypatch, kind, deterministic):
    tuner, selected = restored_reduction(monkeypatch)
    tuner.heuristic_type = kind
    tuner.deterministic_mode = deterministic
    tuner.inductor_meta["coordinate_descent_tuning"] = True
    _freeze_tuning_winners(SimpleNamespace(static_autotuners=[
        SimpleNamespace(cache_key="reduction-key", kernel=tuner, kernel_name="restored_reduction")]))
    assert tuner.compile_results[0].config is selected


def test_frozen_tuning_contract_is_scoped_and_default_binding_is_unchanged():
    from ignition.preparation.device_binding import _FREEZE_TUNING

    entry = object()
    assert relocate_inductor(entry, torch.device("cuda", 0)) is None
    with bind_device("cuda:0"):
        assert relocate_inductor(entry, torch.device("cuda", 0)) is None
        with pytest.raises(RuntimeError, match="binding failed"):
            with bind_device("cuda:0", freeze_tuning=True):
                assert _FREEZE_TUNING.get()
                raise RuntimeError("binding failed")
        assert not _FREEZE_TUNING.get()
    assert not _FREEZE_TUNING.get()
    with pytest.raises(TypeError, match="boolean"):
        with bind_device("cuda:0", freeze_tuning=1):
            pass


def test_forward_cache_load_requests_its_compile_time_contract(tmp_path, monkeypatch):
    from ignition.preparation.aot import ForwardCache, _BindingsChecked, _CHECK_BINDINGS
    from ignition.preparation.device_binding import _FREEZE_TUNING

    path = tmp_path / "artifact.bin"
    path.write_bytes(b"compiled-forward")
    owner = object()
    observed = []
    class Artifact:
        def guard_check(self, bound):
            return bound is owner
        def __call__(self, bound):
            assert _CHECK_BINDINGS.get()
            raise _BindingsChecked
    def load(file):
        observed.append(_FREEZE_TUNING.get())
        return Artifact()
    monkeypatch.setattr(torch.compiler, "load_compiled_function", load)
    monkeypatch.setattr("ignition.preparation.aot.release_unused", lambda **_: None)
    cache = ForwardCache.__new__(ForwardCache)
    cache.device = torch.device("cuda", 0)
    for frozen in (False, True):
        cache.options = {"triton.autotune_at_compile_time": frozen}
        assert cache._load(owner, path).cache_hit
    assert observed == [False, True]
    assert not _FREEZE_TUNING.get()


DeviceProperties = namedtuple("DeviceProperties", "index")


@dataclass
class StoredEntry:
    cache_key: str
    kernel: object
    kernel_name: str


@dataclass
class StoredArtifact:
    device: int


@dataclass
class StoredBundle:
    static_autotuners: list
    kernel_artifacts: list


class StoredTuner:
    is_statically_launchable = CachingAutotuner.is_statically_launchable


@dataclass
class StoredKernel:
    cubin_raw: bytes
    n_regs: int = 64


class PicklableJIT:
    fn = None
    __globals__ = None
    used_global_vals = None

    @staticmethod
    def repr(_self):
        return "restored_reduction"


@pytest.mark.parametrize("target", [0, 1])
def test_artifact_binding_freezes_an_unpickled_entry_before_deserialization(monkeypatch, target):
    import importlib
    import zlib
    from torch._inductor.output_code import CompiledFxGraph
    from torch.utils._ordered_set import OrderedSet
    from torch._inductor.async_compile import CompiledTritonKernels
    from torch._inductor.codecache import code_hash
    from torch._functorch._aot_autograd import aot_autograd_result
    from torch._functorch._aot_autograd.runtime_wrappers import SerializableCompiledFunction
    from torch._functorch._aot_autograd.utils import simple_wraps
    from ignition.preparation import aot
    standalone_compile = importlib.import_module('torch._inductor.standalone_compile')

    tuner, selected = restored_reduction(monkeypatch)
    stored = StoredTuner()
    stored.compile_results = tuner.compile_results
    stored.compile_results[0].compile_meta["device"] = 0
    stored.inductor_meta = tuner.inductor_meta
    stored.triton_meta = {"device": 0}
    stored.device_props = DeviceProperties(0)
    stored._could_rblock_scale = True
    kernel_source = "device = DeviceProperties(index=0)\n"
    graph = CompiledFxGraph.__new__(CompiledFxGraph)
    graph.compiled_fn_runner = None
    graph.device_idxs = OrderedSet((0,))
    graph.source_code = ("torch.cuda.set_device(0)\n"
                         "stream = get_raw_stream(0)\n"
                         "kernel = async_compile.triton('reduction', " + repr(kernel_source) + ")\n")
    graph.cache_key = code_hash(graph.source_code)
    graph._triton_bundle = StoredBundle(static_autotuners=[
        StoredEntry(CompiledTritonKernels.key(kernel_source), stored, "restored_reduction")],
        kernel_artifacts=[StoredArtifact(0)])
    entry = SimpleNamespace(compiled_fw=SimpleNamespace(result=graph), compiled_bw=None)
    data = pickle.dumps((entry, {}, {}))
    observed = []
    def deserialize(copied):
        observed.append(copied)
        frozen = copied.compiled_fw.result._triton_bundle.static_autotuners[0].kernel
        assert frozen.inductor_meta["dynamic_scale_rblock"] is False
        assert "_could_rblock_scale" not in vars(frozen)
        owner = SerializableCompiledFunction(lambda args: None, lambda: copied)
        @simple_wraps(owner)
        def compiled(*args):
            return owner(list(args))
        compiled.serialize = owner.serialize
        return compiled
    monkeypatch.setattr(aot_autograd_result, "deserialize_bundled_cache_entry", deserialize)
    monkeypatch.setattr(standalone_compile, "AOTCompiledArtifact", lambda compiled: compiled)
    with bind_device(f"cuda:{target}", freeze_tuning=True):
        compiled, compressed = aot._load_inductor(data, torch.device("cuda", 0))
    copied = observed[0]
    copied_graph = copied.compiled_fw.result
    portable = pickle.loads(zlib.decompress(compressed))[0].compiled_fw.result
    restored = portable._triton_bundle.static_autotuners[0].kernel
    assert copied is not entry
    assert restored.compile_results[0].config.kwargs == selected.kwargs
    assert restored.device_props.index == target
    assert restored.triton_meta["device"] == target
    assert restored.compile_results[0].compile_meta["device"] == target
    assert portable.device_idxs == OrderedSet((target,))
    assert portable._triton_bundle.kernel_artifacts[0].device == target
    assert portable.cache_key == code_hash(portable.source_code)
    if target:
        assert f"torch.cuda.set_device({target})" in portable.source_code
        assert portable._triton_bundle.static_autotuners[0].cache_key != graph._triton_bundle.static_autotuners[0].cache_key
    else:
        assert portable.source_code == graph.source_code
    assert copied_graph._triton_bundle.kernel_artifacts == []
    assert compiled.serialize() is copied
    assert stored._could_rblock_scale is True
    assert "dynamic_scale_rblock" not in stored.inductor_meta
    assert graph._triton_bundle.kernel_artifacts == [StoredArtifact(0)]


def test_effective_last_snapshot_preserves_the_tuned_winner(monkeypatch):
    initial, _ = restored_reduction(monkeypatch)
    initial.compile_results *= 2
    winner, selected = restored_reduction(monkeypatch)
    snapshots = [SimpleNamespace(cache_key="same-key", kernel_name="reduction", kernel=kernel)
                 for kernel in (initial, winner)]
    bundle = SimpleNamespace(static_autotuners=snapshots.copy())
    _freeze_tuning_winners(bundle)
    assert bundle.static_autotuners == snapshots[1:]
    assert bundle.static_autotuners[0].kernel.compile_results[0].config is selected
    assert len(initial.compile_results) == 2
    with pytest.raises(ValueError, match="unresolved"):
        _freeze_tuning_winners(SimpleNamespace(static_autotuners=snapshots[::-1]))


def test_source_template_relocates_and_registers_without_reparsing(monkeypatch):
    from torch._inductor.async_compile import CompiledTritonKernels
    from torch._inductor.codecache import code_hash
    from torch._inductor.output_code import CompiledFxGraph
    from torch.utils._ordered_set import OrderedSet
    from ignition.preparation import device_binding
    from ignition.preparation.device_binding import bind_device, build_source_template, relocate_inductor
    from ignition.preparation.inductor_tuning import InductorSelections

    DeviceProps = namedtuple("DeviceProps", ["index"])
    tuner, _ = restored_reduction(monkeypatch)
    tuner.compile_results[0].compile_meta["device"] = 0
    tuner.compile_results[0].kernel.cubin_raw = b"cubin-bytes"
    tuner._make_launchers()

    stored = StoredTuner()
    stored.compile_results = tuner.compile_results
    stored.inductor_meta = tuner.inductor_meta
    stored.triton_meta = {"device": 0}
    stored.device_props = DeviceProps(0)

    kernel_source = "device = DeviceProperties(index=0)\nstream0 = get_raw_stream(0)\n"
    wrapper_source = (
        "import torch\n"
        "torch.cuda.set_device(0)\n"
        "with torch.cuda._DeviceGuard(0):\n"
        "    stream0 = get_raw_stream(0)\n"
        "kernel = async_compile.triton('reduction', " + repr(kernel_source) + ")\n"
    )
    template = build_source_template(wrapper_source, 0)

    graph = CompiledFxGraph.__new__(CompiledFxGraph)
    graph.compiled_fn_runner = None
    graph.source_code = wrapper_source
    graph.cache_key = code_hash(wrapper_source)
    graph.device_idxs = OrderedSet((0,))
    graph._flare_source_template = template
    graph._triton_bundle = StoredBundle(
        static_autotuners=[StoredEntry(CompiledTritonKernels.key(kernel_source), stored, "restored_reduction")],
        kernel_artifacts=[StoredArtifact(0)],
    )
    entry = SimpleNamespace(compiled_fw=SimpleNamespace(result=graph), compiled_bw=None)

    def forbid_reparse(*args, **kwargs):
        raise AssertionError("cached _SourceTemplate should not rebuild or reparse generated AST")

    monkeypatch.setattr(device_binding, "build_source_template", forbid_reparse)

    with bind_device("cuda:3", freeze_tuning=True):
        relocate_inductor(entry, torch.device("cuda", 0))

    assert "torch.cuda.set_device(3)" in graph.source_code
    assert "torch.cuda._DeviceGuard(3)" in graph.source_code
    assert "index=3" in graph.source_code
    assert graph.device_idxs == OrderedSet((3,))
    assert graph._flare_source_template.current_device == 3

    fake_code = compile("def call(args):\n    return args\n", f"/tmp/{graph.cache_key}.py", "exec")
    namespace = {"kernel": tuner}
    exec(fake_code, namespace)
    graph.current_callable = namespace["call"]

    backend = SimpleNamespace(
        artifact=SimpleNamespace(_flare_graph=graph),
        _serialized_artifact=None,
        _tuning_frozen=True,
        signature=(),
        inputs=[],
        invocations=[],
    )
    bound = SimpleNamespace(backend=backend, owner=SimpleNamespace(), execution_identity=None)
    selections = InductorSelections(torch.device("cuda", 3), source_identity="unit")
    selections.register(bound)
    selections.freeze(bound)
    assert bound.execution_identity is not None
    assert len(selections.forwards[id(bound)][2]) == 1


def test_source_template_handles_unicode_separators_utf8_same_line_assignments_and_same_length_mutation():
    from torch._inductor.output_code import CompiledFxGraph
    from ignition.preparation.device_binding import (
        _device_source,
        build_source_template,
        get_source_template,
    )

    kernel_a = "# formfeed \x0c and unicode sep \u2028 αβγ\r\ndevice = DeviceProperties(index=0); s = get_raw_stream(0)\n"
    kernel_b = "x = 'Ω'; stream = get_raw_stream(0)\n"
    wrapper = (
        "# comment with \x0b \x0c \x85 \u2028 \u2029 and UTF-8 🚀\n"
        "label = 'αβγ'; torch.cuda.set_device(0)\r\n"
        "k_first = async_compile.triton('k_first', " + repr(kernel_a) + "); "
        "k_second = async_compile.triton('k_second', " + repr(kernel_b) + ")\n"
    )
    template = build_source_template(wrapper, 0)
    assert [name for name, _, _ in template.assigned_kernels] == ["k_first", "k_second"]
    assert template.invalid_kernel_assignment is False
    rendered = template.render(2)
    assert "label = 'αβγ'; torch.cuda.set_device(2)" in rendered
    assert "DeviceProperties(index=2); s = get_raw_stream(2)" in rendered
    assert "x = 'Ω'; stream = get_raw_stream(2)" in rendered
    assert _device_source(wrapper, 0, 2) == rendered

    unassigned = "async_compile.triton('orphan', 'get_raw_stream(0)')\n"
    assert build_source_template(unassigned, 0).invalid_kernel_assignment is True

    graph = CompiledFxGraph.__new__(CompiledFxGraph)
    graph.compiled_fn_runner = None
    graph.source_code = "tag = 'aaaa'\ntorch.cuda.set_device(0)\n"
    first = get_source_template(graph, 0)
    assert get_source_template(graph, 0) is first
    # Mutating source_code to a distinct string of identical character length must rebuild the template.
    graph.source_code = "tag = 'bbbb'\ntorch.cuda.set_device(0)\n"
    assert len(graph.source_code) == len("tag = 'aaaa'\ntorch.cuda.set_device(0)\n")
    second = get_source_template(graph, 0)
    assert second is not first
    assert "tag = 'bbbb'" in second.render(1)


def test_relocate_inductor_drops_unreferenced_candidates_and_rejects_named_unbound_kernels(monkeypatch):
    from torch._inductor.async_compile import CompiledTritonKernels
    from torch._inductor.codecache import code_hash
    from torch._inductor.output_code import CompiledFxGraph
    from torch.utils._ordered_set import OrderedSet

    tuner, _ = restored_reduction(monkeypatch)
    tuner.compile_results[0].compile_meta["device"] = 0
    stored = StoredTuner()
    stored.compile_results = tuner.compile_results
    stored.inductor_meta = tuner.inductor_meta
    stored.triton_meta = {"device": 0}
    stored.device_props = DeviceProperties(0)

    orphan = StoredTuner()
    orphan.compile_results = tuner.compile_results
    orphan.inductor_meta = {"kernel_name": "UNREFERENCED_TEMPLATE"}
    orphan.triton_meta = {"device": 0}
    orphan.device_props = DeviceProperties(0)

    kernel_source = "device = DeviceProperties(index=0)\n"
    wrapper_source = (
        "torch.cuda.set_device(0)\n"
        "kernel = async_compile.triton('restored_reduction', " + repr(kernel_source) + ")\n"
    )

    graph = CompiledFxGraph.__new__(CompiledFxGraph)
    graph.compiled_fn_runner = None
    graph.source_code = wrapper_source
    graph.cache_key = code_hash(wrapper_source)
    graph.device_idxs = OrderedSet((0,))
    graph._triton_bundle = StoredBundle(
        static_autotuners=[
            StoredEntry(CompiledTritonKernels.key(kernel_source), stored, "restored_reduction"),
            StoredEntry("epilogue-candidate-key", orphan, "UNREFERENCED_TEMPLATE"),
        ],
        kernel_artifacts=[StoredArtifact(0)],
    )
    entry = SimpleNamespace(compiled_fw=SimpleNamespace(result=graph), compiled_bw=None)
    with bind_device("cuda:1"):
        relocate_inductor(entry, torch.device("cuda", 0))
    assert len(graph._triton_bundle.static_autotuners) == 1
    assert graph._triton_bundle.static_autotuners[0].kernel_name == "restored_reduction"

    bad_graph = CompiledFxGraph.__new__(CompiledFxGraph)
    bad_graph.compiled_fn_runner = None
    bad_graph.source_code = wrapper_source
    bad_graph.cache_key = code_hash(wrapper_source)
    bad_graph.device_idxs = OrderedSet((0,))
    bad_graph._triton_bundle = StoredBundle(
        static_autotuners=[
            StoredEntry("mismatched-key", stored, "restored_reduction"),
        ],
        kernel_artifacts=[StoredArtifact(0)],
    )
    bad_entry = SimpleNamespace(compiled_fw=SimpleNamespace(result=bad_graph), compiled_bw=None)
    with bind_device("cuda:1"):
        with pytest.raises(ValueError, match="unbound static Inductor kernels"):
            relocate_inductor(bad_entry, torch.device("cuda", 0))


def test_custom_inductor_selections_subclass_reads_zlib_portable_artifact_in_freeze(tmp_path, monkeypatch):
    import zlib
    from concurrent.futures import Future
    from torch._inductor.async_compile import CompiledTritonKernels
    from torch._inductor.codecache import code_hash
    from torch._inductor.output_code import CompiledFxGraph
    from torch.utils._ordered_set import OrderedSet
    from ignition.preparation import aot, compile_pool, inductor_tuning
    from ignition.preparation.aot import BoundForward, ForwardCache, StorageBackend, _BindingsChecked, _CHECK_BINDINGS
    from ignition.preparation.device_binding import build_source_template
    from ignition.preparation.inductor_tuning import _uses_default_portable_freeze

    tuner, _ = restored_reduction(monkeypatch)
    if not hasattr(CachingAutotuner, "_prune_compile_results_to_launcher"):
        monkeypatch.setattr(CachingAutotuner, "_prune_compile_results_to_launcher",
                            lambda self, launcher: None, raising=False)
    tuner.fn = PicklableJIT()
    tuner.benchmark_failure_reasons = {}
    tuner.device_props = DeviceProperties(0)
    tuner.compile_results[0].compile_meta["device"] = 0
    tuner.compile_results[0].kernel = StoredKernel(b"cubin-bytes")
    tuner._make_launchers()
    for local_hook in ("_precompile_worker", "_ensure_kernel_loaded", "_precompile_config", "_make_launchers"):
        tuner.__dict__.pop(local_hook, None)

    kernel_source = "device = DeviceProperties(index=0)\nstream0 = get_raw_stream(0)\n"
    wrapper_source = (
        "import torch\n"
        "torch.cuda.set_device(0)\n"
        "kernel = async_compile.triton('restored_reduction', " + repr(kernel_source) + ")\n"
    )
    kernel_key = CompiledTritonKernels.key(kernel_source)

    saved_targets = []

    def make_bound(*, frozen, cache_hit):
        stored = StoredTuner()
        stored_result = copy.copy(tuner.compile_results[0])
        if frozen:
            stored_result._flare_config_state = copy.deepcopy(vars(stored_result.config))
        stored.compile_results = [stored_result]
        stored.inductor_meta = dict(tuner.inductor_meta, dynamic_scale_rblock=False)
        stored.triton_meta = {"device": 0}
        stored.device_props = DeviceProperties(0)

        graph = CompiledFxGraph.__new__(CompiledFxGraph)
        graph.compiled_fn_runner = None
        graph.source_code = wrapper_source
        graph.cache_key = code_hash(wrapper_source)
        graph.device_idxs = OrderedSet((0,))
        graph._flare_source_template = build_source_template(wrapper_source, 0)
        graph._triton_bundle = StoredBundle(
            static_autotuners=[StoredEntry(kernel_key, stored, "restored_reduction")],
            kernel_artifacts=[],
        )
        fake_code = compile("def call(args):\n    return args\n", f"/tmp/{graph.cache_key}.py", "exec")
        namespace = {"kernel": tuner}
        exec(fake_code, namespace)
        graph.current_callable = namespace["call"]

        portable_graph = SimpleNamespace(
            _triton_bundle=StoredBundle(
                static_autotuners=[StoredEntry(kernel_key, stored, "restored_reduction")],
                kernel_artifacts=[],
            )
        )
        entry = SimpleNamespace(compiled_fw=SimpleNamespace(result=portable_graph), compiled_bw=None)
        raw = pickle.dumps((entry, {}, {}), protocol=pickle.HIGHEST_PROTOCOL)
        serialized = (
            aot._RawPortableArtifact(raw)
            if aot._DEFER_PORTABLE_COMPRESSION.get() and not frozen
            else zlib.compress(raw, level=1)
        )
        backend = StorageBackend.__new__(StorageBackend)
        backend.artifact = SimpleNamespace(_flare_graph=graph)
        backend._serialized_artifact = serialized
        backend._tuning_frozen = frozen
        backend._cuda_allocation_sizes = (1024,)
        backend.signature = ()
        backend.inputs = []
        backend.invocations = []

        class FakeArtifact:
            def __init__(self):
                self._artifacts = SimpleNamespace(compiled_fn=backend)

            def __call__(self, owner):
                if _CHECK_BINDINGS.get():
                    raise _BindingsChecked
                return None

            def save_compiled_function(self, target):
                saved_targets.append(str(target))
                Path(target).write_bytes(b"saved-compiled-forward")

        from pathlib import Path
        return BoundForward(FakeArtifact(), SimpleNamespace(), cache_hit)

    default_selections = inductor_tuning.InductorSelections(torch.device("cuda", 0), source_identity="unit")
    assert _uses_default_portable_freeze(default_selections) is True

    freeze_checks = []

    class Probe(inductor_tuning.InductorSelections):
        def freeze(self, bound):
            super().freeze(bound)
            payload = bound.backend._serialized_artifact
            assert type(payload) is bytes
            entry, _, _ = pickle.loads(zlib.decompress(payload))
            assert len(entry.compiled_fw.result._triton_bundle.static_autotuners) == 1
            freeze_checks.append(bound.cache_hit)

    monkeypatch.setattr(inductor_tuning, "InductorSelections", Probe)
    assert _uses_default_portable_freeze(inductor_tuning.InductorSelections(torch.device("cuda", 0))) is False

    class DummyPool:
        workers = 1

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def observe_worker_rss(self, *args, **kwargs):
            return 1

    monkeypatch.setattr(compile_pool, "CompilePool", DummyPool)
    monkeypatch.setattr(aot, "release_unused", lambda **_: None)
    monkeypatch.setattr(aot, "_guard_code_cache", lambda _root: aot.nullcontext())
    monkeypatch.setattr(BoundForward, "prepare_capture", lambda self: 0)

    cache = ForwardCache.__new__(ForwardCache)
    cache.device = torch.device("cuda", 0)
    cache.headroom_bytes = 0
    cache.compile_workers = 1
    cache.lock_gpu_clocks = False
    cache.options = {"triton.autotune_at_compile_time": True}
    cache.root = tmp_path

    artifact_path = tmp_path / "reduce.bin"
    owner = SimpleNamespace()

    def fake_prepare_cold(forward, *, name, signature, compiler=None, progress=None, deferred=False, before_run=None):
        done = Future()
        done.set_result(1 << 20)
        artifact_path.write_bytes(b"cold-compiled")
        return BoundForward(None, forward, False, done), artifact_path

    monkeypatch.setattr(cache, "_prepare", fake_prepare_cold)
    monkeypatch.setattr(cache, "_load", lambda forward, path, **_: make_bound(frozen=False, cache_hit=False))
    monkeypatch.setattr(Probe, "warm", lambda self, bound, warmup, before_run=None: None)

    cold = cache._prepare_many(
        {"reduce": owner}, {"reduce": "sig"},
        progress=None, coordinator=None, materialize=None, warmup=aot.nullcontext, before_run=None,
    )["reduce"]
    assert not cold.cache_hit
    assert freeze_checks == [False]

    warm_bound = make_bound(frozen=True, cache_hit=True)
    monkeypatch.setattr(
        cache, "_prepare",
        lambda forward, *, name, signature, **_: (warm_bound, artifact_path),
    )
    restored = cache._prepare_many(
        {"reduce": owner}, {"reduce": "sig"},
        progress=None, coordinator=None, materialize=None, warmup=aot.nullcontext, before_run=None,
    )["reduce"]
    assert restored.cache_hit
    assert freeze_checks == [False, True]
    assert saved_targets == [str(artifact_path)]
    assert warm_bound.backend._serialized_artifact is None
