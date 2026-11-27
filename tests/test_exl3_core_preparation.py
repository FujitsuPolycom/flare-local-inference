# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for chunked concurrent reductions and recipe replacement inspection hooks."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from ignition import registry
from ignition.components import collectives
from ignition.declaration import (
    Checkpoint,
    RankCosts,
    Recipe,
    SamplingDefaults,
)
from ignition.errors import UsageError
from ignition.parameters import RecipeParameters

GIB = 1 << 30
MIB = 1 << 20


def _make_recipe(*, inspect_replacement=None):
    checkpoint = Checkpoint(
        alias="toy",
        repository="example/toy",
        revision="a" * 40,
        architecture="ToyForCausalLM",
        stored_bytes=100 * GIB,
        device_bytes=100 * GIB,
        scale_compressed_device_bytes={"csf": 96 * GIB},
    )
    costs = RankCosts(state_row_bytes=64 * MIB, page_bytes=2 * MIB, graph_bytes=4 * GIB, page_tokens=1024)
    return Recipe(
        name="toy",
        served_model_name="Toy",
        checkpoints=(checkpoint,),
        context=131072,
        page_tokens=128,
        ranks={8: costs},
        prepare=lambda *args, **kwargs: None,
        usage=lambda *args, **kwargs: None,
        sampling_defaults=SamplingDefaults(temperature=1.0, top_p=0.95, top_k=0),
        parameters=RecipeParameters(),
        inspect_replacement=inspect_replacement,
    )


def test_prepared_chunked_all_reduce_ranges_and_dependencies():
    calls = []

    class FakePart:
        def __init__(self, idx):
            self.idx = idx
            self.output_layout = ((4, 8), (8, 1), torch.bfloat16)

        def __call__(self, source, output=None, *, dependencies=()):
            calls.append((self.idx, tuple(source.shape), len(dependencies)))
            output.copy_(source + self.idx + 1)
            return output

    chunked = object.__new__(collectives.PreparedChunkedAllReduce)
    chunked.bounds = [(0, 4), (4, 8), (8, 12)]
    chunked.parts = [FakePart(0), FakePart(1), FakePart(2)]
    chunked.output = None
    chunked.output_layout = ((12, 8), (8, 1), torch.bfloat16)

    dep = torch.zeros(1)
    inp = torch.zeros((8, 8), dtype=torch.bfloat16)
    out = chunked(inp, parts=(1, 3), dependencies=(dep,))
    assert out.shape == (8, 8)
    assert torch.equal(out[:4], torch.full((4, 8), 2, dtype=torch.bfloat16))
    assert torch.equal(out[4:], torch.full((4, 8), 3, dtype=torch.bfloat16))
    # First chunk carries caller dep; second chunk chains prior chunk output as dependency.
    assert calls == [(1, (4, 8), 1), (2, (4, 8), 2)]

    with pytest.raises(ValueError, match="out of bounds"):
        chunked(inp, parts=(2, 4))
    with pytest.raises(ValueError, match="expected 8"):
        chunked(torch.zeros((4, 8), dtype=torch.bfloat16), parts=(1, 3))


def test_concurrent_chunked_all_reduce_chains_starts_and_joins(monkeypatch):
    started = []
    joined = []

    class FakeReduce:
        def __init__(self, handle, rows):
            self.handle = handle
            self.output = None
            self.output_layout = ((rows, 8), (8, 1), torch.bfloat16)
            self.concurrent_input = None

        def functional(self, *, session):
            return self

    class FakeConcurrent:
        def __init__(self, part, *, workspace, session, parallel_plans, send_buffer=None):
            self.reduce = part
            self.stream_handle = workspace.auxiliary_stream.cuda_stream
            self.send_buffer = send_buffer
            self.validated = False

        def validate_scratch(self):
            self.validated = True

        def functional(self, *, session):
            self.reduce.functional(session=session)
            return self

    monkeypatch.setattr(collectives, "ConcurrentAllReduce", FakeConcurrent)
    monkeypatch.setattr(
        collectives,
        "_start_all_reduce",
        lambda send, out, deps, handle, stream: started.append((handle, tuple(send.shape), len(deps), stream)),
    )
    monkeypatch.setattr(
        collectives,
        "_join_all_reduce",
        lambda send, out, deps, dep, stream: joined.append((tuple(out.shape), len(deps), stream)),
    )

    workspace = SimpleNamespace(auxiliary_stream=SimpleNamespace(cuda_stream=77))
    parts = [FakeReduce(10, 4), FakeReduce(11, 4)]
    cc = collectives.ConcurrentChunkedAllReduce(
        parts, [(4, 8), (8, 12)], workspace=workspace, session=None, parallel_plans=()
    ).functional(session=None)
    cc.validate_scratch()
    assert all(p.validated for p in cc.parts)

    src = torch.randn((8, 8), dtype=torch.bfloat16)
    dep = torch.zeros(1)
    pending = cc.start(src, dependencies=(dep,))
    assert started == [(10, (4, 8), 1, 77), (11, (4, 8), 2, 77)]
    res = cc.finish(pending, torch.zeros(1))
    assert res.shape == (8, 8)
    assert joined == [((8, 8), 2, 77)]

    with pytest.raises(ValueError, match="contiguous bounds"):
        collectives.ConcurrentChunkedAllReduce(
            parts, [(0, 4), (5, 8)], workspace=workspace, session=None, parallel_plans=()
        )


def test_registry_resolve_invokes_recipe_inspect_replacement_and_wraps_errors(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "alt-checkpoint"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text(json.dumps({"architectures": ["ToyForCausalLM"], "custom_fmt": True}))
    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 50 * GIB}, "weight_map": {}})
    )

    seen = []

    def inspect_alt(metadata, replacement):
        seen.append((metadata, replacement.stored_bytes))
        return replace(replacement, device_bytes=52 * GIB, scale_compressed_device_bytes={})

    recipe = _make_recipe(inspect_replacement=inspect_alt)
    resolved = registry.resolve(str(ckpt_dir), recipes=(recipe,))
    assert seen == [(ckpt_dir.resolve(), 50 * GIB)]
    assert resolved.checkpoint.device_bytes == 52 * GIB
    assert resolved.checkpoint.scale_compressed_device_bytes == {}

    # Returning None preserves default scaled replacement.
    none_recipe = _make_recipe(inspect_replacement=lambda metadata, replacement: None)
    resolved_default = registry.resolve(str(ckpt_dir), recipes=(none_recipe,))
    assert resolved_default.checkpoint.device_bytes == 50 * GIB
    assert resolved_default.checkpoint.scale_compressed_device_bytes == {"csf": 48 * GIB}

    # Malformed metadata errors raised by the recipe hook surface as UsageError.
    bad_recipe = _make_recipe(
        inspect_replacement=lambda metadata, replacement: (_ for _ in ()).throw(ValueError("bad manifest"))
    )
    with pytest.raises(UsageError, match="unreadable checkpoint metadata: bad manifest"):
        registry.resolve(str(ckpt_dir), recipes=(bad_recipe,))

    wrong_type = _make_recipe(inspect_replacement=lambda metadata, replacement: {"not": "a checkpoint"})
    with pytest.raises(UsageError, match="returned dict"):
        registry.resolve(str(ckpt_dir), recipes=(wrong_type,))

    # Non-callable inspect_replacement is rejected on Recipe construction.
    with pytest.raises(TypeError, match="callable"):
        _make_recipe(inspect_replacement="not-callable")

    # Hugging Face repository resolution invokes the same hook on the fetched snapshot metadata directory.
    hub_snap = tmp_path / "hub" / "snapshots" / ("b" * 40)
    hub_snap.mkdir(parents=True)
    cfg = hub_snap / "config.json"
    idx = hub_snap / "model.safetensors.index.json"
    cfg.write_text(json.dumps({"architectures": ["ToyForCausalLM"]}))
    idx.write_text(json.dumps({"metadata": {"total_size": 40 * GIB}, "weight_map": {}}))
    monkeypatch.setattr(registry, "_hub_metadata", lambda repo, rev, cache, recs: (cfg, idx))

    seen.clear()
    hub_resolved = registry.resolve("example/toy-alt", recipes=(recipe,))
    assert seen == [(hub_snap, 40 * GIB)]
    assert hub_resolved.checkpoint.revision == "b" * 40
    assert hub_resolved.checkpoint.device_bytes == 52 * GIB
