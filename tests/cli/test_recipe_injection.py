# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Verify recipe-free core CLI behavior and explicit recipe/source-root injection."""

from types import MappingProxyType

import pytest

from ignition.cli import main
from ignition.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults
from ignition.parameters import RecipeParameters
from ignition.preparation import provenance
from ignition.preparation.source_identity import source_identity


def _sample_recipe():
    return Recipe(
        name="custom_toy",
        served_model_name="Custom-Toy-1B",
        checkpoints=(
            Checkpoint(
                alias="custom-toy-1b",
                repository="example/Custom-Toy-1B",
                revision="0123456789abcdef0123456789abcdef01234567",
                architecture="CustomToyForCausalLM",
                stored_bytes=1 << 30,
                device_bytes=1 << 30,
            ),
        ),
        context=8192,
        page_tokens=128,
        parameters=RecipeParameters(
            max_concurrency=4,
            max_concurrency_limit=8,
            prefill_chunk=1024,
        ),
        ranks=MappingProxyType({1: RankCosts(1 << 20, 1 << 20, 1 << 28)}),
        prepare=lambda *args, **kwargs: None,
        usage=lambda prepared: (),
        sampling_defaults=SamplingDefaults(temperature=1.0, top_p=0.95, top_k=20),
    )


def test_bare_cli_help_and_missing_recipe_error_work_without_recipes(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "No model recipes are registered in this CLI." in out
    assert "ignition.cli.main(recipes=...)" in out

    assert main(["export", "unknown-model"]) == 2
    err = capsys.readouterr().err
    assert "no model recipes are registered in this CLI" in err


def test_explicit_recipe_and_prog_appear_in_help_without_global_registry(capsys):
    recipe = _sample_recipe()
    with pytest.raises(SystemExit) as exc:
        main(["--help"], recipes=(recipe,), prog="custom-ignition")
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "custom-toy-1b" in out
    assert "custom-ignition export MODEL --help" in out

    with pytest.raises(SystemExit) as exc:
        main(["export", "custom-toy-1b", "--help"], recipes=(recipe,), prog="custom-ignition")
    assert exc.value.code == 0
    out = capsys.readouterr().out
    normalized = " ".join(out.split())
    assert "Model recipe: custom-toy-1b" in out
    assert "example/Custom-Toy-1B" in out
    assert "1..8; default: 4" in normalized
    assert "64..8192; multiple of 64; default: 1024" in normalized


def test_main_binds_source_roots_for_source_identity(tmp_path, monkeypatch):
    pkg = tmp_path / "external_recipes"
    pkg.mkdir()
    (pkg / "recipe.py").write_text("VALUE = 1\n")

    recorded = {}

    def capture_handler(args):
        recorded["first"] = source_identity()
        (pkg / "recipe.py").write_text("VALUE = 2\n")
        recorded["second"] = source_identity()
        return 0

    from ignition.cli import publishing

    monkeypatch.setattr(publishing, "convert", capture_handler)
    baseline = source_identity()
    assert main(["convert", "dummy"], source_roots=(("external_recipes", pkg),)) == 0
    assert recorded["first"] != baseline
    assert recorded["second"] != recorded["first"]
    assert source_identity() == baseline


def test_provenance_finalizer_flushes_live_owners_once():
    class Owner:
        pass

    owner = Owner()
    seen = []
    assert provenance.defer_finalizer(owner, seen.append)
    provenance.flush_finalizers()
    provenance.flush_finalizers()
    assert seen == [owner]
