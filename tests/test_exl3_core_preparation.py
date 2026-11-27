# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for recipe replacement checkpoint inspection hooks."""

from dataclasses import replace
import json

import pytest

from ignition import registry
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
