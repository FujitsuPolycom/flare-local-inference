# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Restored AOT forward guards preserve diagnostics and release portable compiler state."""

import gc
import pickle
import weakref

import pytest
import torch

from ignition.preparation.aot import (
    BindingMismatch, BoundForward, StorageBackend, _guard_code_cache, _load_inductor,
)


class GuardedForward:
    def __init__(self):
        self.value = torch.ones(2)
        self.offset = 1
        self.ready = True

    def __call__(self):
        return self.value + self.offset


def _execute_guarded(forward):
    return forward()


def _compiled_forward():
    forward = GuardedForward()
    artifact = torch.compile(_execute_guarded,
        backend=lambda graph, inputs: StorageBackend(graph, inputs, options={"compile_threads": 1}),
        fullgraph=True, dynamic=False).aot_compile(((forward,), {}))
    return artifact, forward


def _compiled_guard_state(tmp_path):
    from torch._dynamo import package

    artifact, _ = _compiled_forward()
    path = tmp_path / "forward.bin"
    artifact.save_compiled_function(str(path))
    with path.open("rb") as file:
        return package.load_guards_state(pickle.load(file)["guards_state"])


def _native_guard_diagnostics(wrapper):
    result = []
    pending = [wrapper.root]
    while pending:
        manager = pending.pop()
        result.extend(leaf.verbose_code_parts() for leaf in manager.get_leaf_guards())
        pending.extend(reversed(manager.get_child_managers()))
    result.extend(leaf.verbose_code_parts() for leaf in wrapper.root.get_epilogue_lambda_guards())
    return result


def test_restored_guards_share_exact_native_diagnostic_strings(tmp_path, monkeypatch):
    from torch._dynamo import guards, package

    state = _compiled_guard_state(tmp_path)
    for guard in state.output_graph._guards:
        guard.stack = None
        guard.user_stack = None
    arguments = (state, _execute_guarded.__code__, _execute_guarded.__globals__)
    original_parts = guards.get_verbose_code_parts
    formatted = set()

    def record_parts(*args, **kwargs):
        parts = original_parts(*args, **kwargs)
        formatted.update(parts)
        return parts

    with monkeypatch.context() as patch:
        patch.setattr(guards, "get_verbose_code_parts", record_parts)
        baseline = package.load_guard_manager(*arguments)
    with _guard_code_cache(tmp_path / "guards"):
        first = package.load_guard_manager(*arguments)
        assert guards.get_verbose_code_parts is original_parts
        second = package.load_guard_manager(*arguments)
        assert guards.get_verbose_code_parts is original_parts
    assert first.root is not second.root
    diagnostics = [_native_guard_diagnostics(manager) for manager in (baseline, first, second)]
    assert diagnostics[0] == diagnostics[1] == diagnostics[2]
    assert all(isinstance(parts, list) for lists in diagnostics for parts in lists)
    strings = [[text for parts in lists for text in parts if text in formatted]
               for lists in diagnostics]
    assert strings[0]
    assert all(left is right for left, right in zip(strings[1], strings[2]))

    for manager in (baseline, first, second):
        forward = GuardedForward()
        assert manager.check({"forward": forward})
        for attribute, value in (("offset", 7), ("value", torch.ones(3)),
                                 ("value", torch.ones(2, dtype=torch.float64))):
            changed = GuardedForward()
            setattr(changed, attribute, value)
            assert not manager.check({"forward": changed})
            actual = manager.check_verbose({"forward": changed})
            expected = baseline.check_verbose({"forward": changed})
            assert not actual.result
            assert actual.verbose_code_parts == expected.verbose_code_parts


def test_guard_diagnostic_formatter_restores_after_load_failure(tmp_path, monkeypatch):
    from torch._dynamo import guards, package

    state = _compiled_guard_state(tmp_path)
    original_parts = guards.get_verbose_code_parts
    formatted = []

    def fail_formatting(*args, **kwargs):
        formatted.append(original_parts(*args, **kwargs))
        raise RuntimeError("guard diagnostic formatter failed")

    monkeypatch.setattr(guards, "get_verbose_code_parts", fail_formatting)
    with _guard_code_cache(tmp_path / "guards"):
        with pytest.raises(RuntimeError, match="guard diagnostic formatter failed"):
            package.load_guard_manager(
                state, _execute_guarded.__code__, _execute_guarded.__globals__)
        assert formatted
        assert guards.get_verbose_code_parts is fail_formatting
    assert guards.get_verbose_code_parts is fail_formatting


def _metadata_owners(backend):
    owners = []
    wrapped = backend.artifact.inner_fn.compiled_fn
    while wrapped is not None and all(wrapped is not owner for owner in owners):
        owners.append(wrapped)
        wrapped = getattr(wrapped, "__wrapped__", None)
    graph = getattr(backend.artifact, "_flare_graph", None)
    if graph is not None and all(graph is not owner for owner in owners):
        owners.append(graph)
    return owners


@pytest.mark.parametrize("restored", [False, True])
def test_finalization_releases_portable_owners_and_retains_live_checks(tmp_path, restored):
    class Callback:
        def check(self, arguments):
            return arguments["forward"].ready

    artifact, forward = _compiled_forward()
    backend = artifact._artifacts.compiled_fn
    portable_refs = ()
    execution_refs = ()
    if restored:
        backend.artifact, backend._serialized_artifact = _load_inductor(
            backend.artifact.serialize(), torch.device("cpu"))
        compiled = backend.artifact.inner_fn.compiled_fn
        portable = compiled.serialize()
        portable_refs = (weakref.ref(portable), weakref.ref(portable.compiled_fw.result),
                         weakref.ref(compiled.serialize_fn))
        execution_refs = (weakref.ref(compiled.serialize.__self__.compiled_fn),
                          weakref.ref(backend.artifact._flare_graph))
        torch.testing.assert_close(artifact(forward), torch.full((2,), 2.0))
        artifact.save_compiled_function(str(tmp_path / "forward.bin"))
        del compiled, portable
    metadata = _metadata_owners(backend)
    assert len(metadata) > 1
    assert all(value.source_code for value in metadata)
    bundle_refs = [weakref.ref(value._triton_bundle) for value in metadata
                   if value._triton_bundle is not None]
    assert bundle_refs
    structure = [tuple(id(getattr(value, field, None))
                       for field in ("__wrapped__", "compiled_fn", "current_callable", "__closure__"))
                 for value in metadata]
    del backend
    wrapper = artifact._artifacts.guard_manager
    callback = Callback()
    wrapper.root.add_lambda_guard(callback.check, ["forward.ready must be True"], None)
    wrapper.diff_guard_root = wrapper.root.clone_manager(lambda manager: True)
    wrapper.populate_code_parts_for_debugging()
    clone_ref = weakref.ref(wrapper.diff_guard_root)
    wrapper_ref, callback_ref = weakref.ref(wrapper), weakref.ref(callback)
    manager_ref = weakref.ref(wrapper._check_function_manager)
    backend_ref = weakref.ref(artifact._artifacts.compiled_fn)
    bound = BoundForward(artifact, forward, cache_hit=True)
    del artifact, wrapper, callback
    assert bound.prepare_capture() > 0
    gc.collect()
    assert clone_ref() is None
    assert wrapper_ref() is not None
    assert manager_ref() is not None
    assert not wrapper_ref().code_parts
    assert callback_ref() is not None
    assert all(reference() is None for reference in portable_refs)
    assert all(reference() is not None for reference in execution_refs)
    assert all(reference() is None for reference in bundle_refs)
    assert all(value.source_code == "" and value._triton_bundle is None
               and value.runnable_graph_str == ""
               and value.inductor_provenance_mapping_str is None
               and value.inductor_provenance_stack_traces_str is None for value in metadata)
    assert structure == [tuple(id(getattr(value, field, None))
                               for field in ("__wrapped__", "compiled_fn", "current_callable", "__closure__"))
                         for value in metadata]
    del metadata
    assert bound.prepare_capture() == 0
    torch.testing.assert_close(bound(), torch.full((2,), 2.0))
    forward.ready = False
    with pytest.raises(BindingMismatch, match="forward.ready must be True"):
        bound()
    del bound
    gc.collect()
    assert wrapper_ref() is None
    assert manager_ref() is None
    assert backend_ref() is None
    assert callback_ref() is None
    assert all(reference() is None for reference in execution_refs)
    assert type(forward) is GuardedForward
