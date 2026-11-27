# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit and integration tests for first-class RecipeParameters, dynamic CLI help, and manifests."""

from dataclasses import dataclass, replace
import pickle
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from google.protobuf import json_format

from ignition import planning
from ignition.cli import main, parse_args, preparation
from ignition.cli.config import Configuration
from ignition.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults
from ignition.errors import CapacityError, UsageError
from ignition.parameters import (
    MixedPrefillParameters,
    ParameterContext,
    ParameterGroup,
    ParameterScope,
    PromptLogprobsParameters,
    RecipeParameters,
    SpeculationParameters,
    VisionParameters,
    explicit_flags_from_args,
    param,
)
from ignition.preparation import rebuild_manifest, runtime_core_pb2 as proto
from ignition.preparation.bootstrap import BasePreparationConfig, common_preparation_kwargs
from ignition.registry import ResolvedModel

GIB = 1 << 30
MIB = 1 << 20


@dataclass(frozen=True)
class OptionalCompressionGroup(ParameterGroup):
    """Test ParameterGroup with nargs='?' optional choice argument."""

    default_group = "capacity and compression"
    encoding: str | None = param(
        None,
        flag="--toy-compression",
        metavar="MODE",
        choices=("csf", "native"),
        const="csf",
        native_value="native",
        scope=ParameterScope.TARGET_WEIGHT,
        manifest_key="toy_compression",
        dict_key="toy_compression",
        help="optional toy compression mode",
    )


@dataclass(frozen=True)
class AlphaParameters(RecipeParameters):
    compute_mode: str = param(
        "fast",
        flag="--alpha-mode",
        group="advanced options",
        choices=("auto", "fast"),
        scope=ParameterScope.COMBINE,
        repro_always=True,
        help="alpha arithmetic mode",
    )
    compression: OptionalCompressionGroup = OptionalCompressionGroup(encoding="csf")


@dataclass(frozen=True)
class BetaParameters(RecipeParameters):
    decode_override: bool = param(
        False,
        flag="--beta-override",
        group="advanced options",
        boolean_optional=True,
        scope=ParameterScope.DEPLOYMENT,
        help="beta boolean optional override",
    )


def _make_recipe(name: str, alias: str, parameters: RecipeParameters, *, max_images: int = 0) -> Recipe:
    checkpoint = Checkpoint(
        alias=alias,
        repository=f"example/{alias}",
        revision="a" * 40,
        architecture=f"{name.title()}ForCausalLM",
        stored_bytes=40 * GIB,
        device_bytes=40 * GIB,
        draft_bytes=2 * GIB if parameters.speculation is not None else 0,
    )
    ranks = MappingProxyType({
        1: RankCosts(100 * MIB, 2 * MIB, 4 * GIB, vision_bytes=2 * GIB if max_images else 0, vision_slot_bytes=512 * MIB if max_images else 0),
        2: RankCosts(50 * MIB, 1 * MIB, 4 * GIB, vision_bytes=2 * GIB if max_images else 0, vision_slot_bytes=512 * MIB if max_images else 0),
    })
    plain = MappingProxyType({
        1: RankCosts(80 * MIB, 3 * MIB // 2, 3 * GIB, vision_bytes=2 * GIB if max_images else 0, vision_slot_bytes=512 * MIB if max_images else 0),
        2: RankCosts(40 * MIB, 3 * MIB // 4, 3 * GIB, vision_bytes=2 * GIB if max_images else 0, vision_slot_bytes=512 * MIB if max_images else 0),
    }) if parameters.speculation is not None else MappingProxyType({})
    return Recipe(
        name=name,
        served_model_name=alias,
        checkpoints=(checkpoint,),
        context=131072,
        page_tokens=64,
        parameters=parameters,
        ranks=ranks,
        plain_ranks=plain,
        max_images=max_images,
        prepare=lambda *args, **kwargs: None,
        usage=lambda prepared: (),
        sampling_defaults=SamplingDefaults(temperature=1.0, top_p=1.0, top_k=-1),
    )


def _recipes():
    alpha = _make_recipe(
        "alpha", "alpha-1b",
        AlphaParameters(
            max_concurrency=16,
            max_concurrency_limit=32,
            prefill_chunk=4096,
            speculation=SpeculationParameters(
                draft_tokens=5, max_draft_tokens=7, token_capacity=64,
                packed_verification=True, draft_prefix_selection="gpu", graph_budget=32,
            ),
            vision=VisionParameters(separable_encoder=True),
            prompt_logprobs=PromptLogprobsParameters(),
        ),
        max_images=4,
    )
    beta = _make_recipe(
        "beta", "beta-1b",
        BetaParameters(
            max_concurrency=8,
            max_concurrency_limit=16,
            prefill_chunk=2048,
            speculation=SpeculationParameters(
                draft_tokens=3, max_draft_tokens=5, token_capacity=32, method="mtp", supported_methods=("mtp", "dflash2"),
            ),
            mixed_prefill=MixedPrefillParameters(),
        ),
    )
    plain = _make_recipe(
        "plain", "plain-1b",
        RecipeParameters(max_concurrency=4, max_concurrency_limit=8, prefill_chunk=1024),
    )
    return (alpha, beta, plain)


def _gpu(index=0, free_gib=95, capability=(12, 0)):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}",
                        96 * GIB, int(free_gib * GIB), capability)


def test_dynamic_parsing_handles_flag_ordering_config_model_and_cross_recipe_errors(tmp_path):
    recipes = _recipes()
    alpha, beta, plain = recipes

    # Flag after MODEL and flag before MODEL both resolve cleanly.
    after = parse_args(["export", "alpha-1b", "--alpha-mode", "auto", "--toy-compression", "native"], recipes=recipes)
    before = parse_args(["export", "--alpha-mode", "auto", "--toy-compression", "native", "alpha-1b"], recipes=recipes)
    assert after.alpha_mode == before.alpha_mode == "auto"
    assert after.toy_compression == before.toy_compression == "native"

    # Default model from config.toml resolves the recipe for Stage 2 parsing.
    cfg = tmp_path / "config.toml"
    cfg.write_text('[defaults]\nmodel = "beta-1b"\n')
    from_cfg = parse_args(["export", "--config", str(cfg), "--beta-override", "--mixed-prefill"], recipes=recipes)
    assert from_cfg.beta_override is True and from_cfg.mixed_prefill is True

    # Flag belonging to another registered recipe yields a UsageError naming the model.
    with pytest.raises(UsageError, match="--alpha-mode is not supported by beta-1b"):
        parse_args(["export", "beta-1b", "--alpha-mode", "fast"], recipes=recipes)
    with pytest.raises(UsageError, match="--mixed-prefill is not supported by alpha-1b"):
        parse_args(["export", "alpha-1b", "--mixed-prefill"], recipes=recipes)
    with pytest.raises(UsageError, match="--no-mtp is not supported by plain-1b"):
        parse_args(["export", "plain-1b", "--no-mtp"], recipes=recipes)

    # Truly unknown flags remain argparse exit-2 errors.
    with pytest.raises(SystemExit) as exc:
        parse_args(["export", "alpha-1b", "--nonexistent-option"], recipes=recipes)
    assert exc.value.code == 2


def test_bare_and_model_help_render_dynamically_and_stay_offline(capsys, tmp_path, monkeypatch):
    recipes = _recipes()
    # Point HOME to an unreadable config path to guarantee --help never reads config.toml.
    bad_cfg = tmp_path / ".config/flare/config.toml"
    bad_cfg.parent.mkdir(parents=True)
    bad_cfg.write_text("[broken\n")
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(SystemExit) as exc:
        parse_args(["export", "--help"], recipes=recipes)
    assert exc.value.code == 0
    bare = capsys.readouterr().out
    assert "Available model recipes:" in bare
    assert "alpha-1b" in bare and "beta-1b" in bare and "plain-1b" in bare
    assert "--alpha-mode" not in bare
    assert "--beta-override" not in bare
    assert "--mixed-prefill" not in bare
    assert "--mtp-draft-tokens" not in bare

    # MODEL --help (in either order) renders only that recipe's options and exact defaults.
    for argv in (["export", "alpha-1b", "--help"], ["export", "--help", "alpha-1b"]):
        with pytest.raises(SystemExit) as exc:
            parse_args(argv, recipes=recipes)
        assert exc.value.code == 0
        out = " ".join(capsys.readouterr().out.split())
        assert "--alpha-mode" in out and "default: fast" in out
        assert "--toy-compression" in out and "default: csf" in out
        assert "--language-model-only" in out
        assert "--speculative-graph-budget" in out and "default: 32" in out
        assert "1..32; default: 16" in out
        assert "--beta-override" not in out
        assert "--mixed-prefill" not in out

    with pytest.raises(SystemExit) as exc:
        parse_args(["export", "plain-1b", "--help"], recipes=recipes)
    assert exc.value.code == 0
    plain_out = capsys.readouterr().out
    assert "Multi-token speculation: unavailable." in plain_out
    assert "Images: unavailable." in plain_out
    assert "--mtp-draft-tokens" not in plain_out
    assert "--image-concurrency" not in plain_out


def test_help_imports_no_torch_or_network_dependencies():
    root = Path(__file__).resolve().parents[2]
    script = """
import contextlib, io, sys
from ignition.cli import main
with contextlib.redirect_stdout(io.StringIO()):
    for argv in (['--help'], ['export', '--help']):
        try:
            main(argv)
        except SystemExit as error:
            assert error.code == 0
assert not {'torch', 'transformers', 'flare_heliopause', 'huggingface_hub', 'pynvml'} & set(sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_parameters_and_base_preparation_config_round_trip_through_pickle():
    alpha, _, _ = _recipes()
    plan = planning.plan(alpha, alpha.checkpoints[0], (_gpu(0),), selected=(0,))
    config = BasePreparationConfig(**common_preparation_kwargs(
        plan, compile_forwards=False, kernel_cache=None, retain_graphs=True,
    ))
    restored = pickle.loads(pickle.dumps(config))
    assert restored == config
    assert restored.parameters == replace(plan.parameters, compiled_forwards=False)
    assert restored.max_active == 16
    assert restored.compile_decoder is False
    assert restored.image_concurrency == 4
    assert restored.speculative_widths == (1, 2, 3, 4, 5, 6)
    assert restored.speculative_graph_budget == 32


def test_core_abi_validations_and_capacity_relief():
    alpha, beta, _ = _recipes()
    alpha_model = ResolvedModel(alpha, alpha.checkpoints[0])
    beta_model = ResolvedModel(beta, beta.checkpoints[0])

    # --no-mtp rejects accompanying speculation flags.
    args = parse_args(["export", "alpha-1b", "--no-mtp", "--mtp-draft-tokens", "3"], recipes=(alpha,))
    with pytest.raises(UsageError, match="--no-mtp disables speculation; --mtp-draft-tokens cannot accompany it"):
        preparation.resolve_parameters(alpha_model, args)

    # --language-model-only rejects explicit --image-concurrency.
    args = parse_args(["export", "alpha-1b", "--language-model-only", "--image-concurrency", "2"], recipes=(alpha,))
    with pytest.raises(UsageError, match="--image-concurrency requires the vision encoder"):
        preparation.resolve_parameters(alpha_model, args)

    # --mixed-prefill rejects --speculative-method dflash2.
    args = parse_args(["export", "beta-1b", "--mixed-prefill", "--speculative-method", "dflash2"], recipes=(beta,))
    with pytest.raises(UsageError, match="separate prefill and verification graphs"):
        preparation.resolve_parameters(beta_model, args)

    # Capacity relief suggests --max-concurrency, --language-model-only, and --no-mtp generically.
    tight_gpu = (_gpu(0, free_gib=50),)
    with pytest.raises(CapacityError) as exc:
        planning.plan(alpha, alpha.checkpoints[0], tight_gpu, selected=(0,))
    msg = str(exc.value)
    assert "--max-concurrency" in msg or "--no-mtp" in msg or "--language-model-only" in msg


def test_pure_manifest_emission_and_reproduction_args_round_trip(monkeypatch):
    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    recipes = _recipes()
    alpha, beta, plain = recipes

    for recipe, cli_flags in (
        (alpha, ["--max-concurrency", "8", "--alpha-mode", "auto", "--toy-compression", "native",
                 "--language-model-only", "--prompt-logprobs", "--no-compile"]),
        (beta, ["--max-concurrency", "4", "--beta-override", "--mixed-prefill", "--no-mtp"]),
        (plain, ["--max-concurrency", "2", "--prefill-chunk-size", "2048", "--autotune-effort", "off"]),
    ):
        model = ResolvedModel(recipe, recipe.checkpoints[0])
        args = parse_args(["export", recipe.checkpoints[0].alias, *cli_flags], recipes=recipes)
        resolved = preparation.resolve_parameters(model, args)
        plan = planning.plan(recipe, recipe.checkpoints[0], (_gpu(0),), selected=(0,), parameters=resolved)

        manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), plan)
        encoded = json_format.ParseDict(manifest, proto.PreparationManifest())
        decoded = proto.PreparationManifest.FromString(encoded.SerializeToString(deterministic=True))
        by_key = {p.key: p for p in decoded.parameters}

        # Verify strictly pure manifest keys: plain emits zero ABI or subclass keys.
        if recipe is plain:
            assert not any(k.startswith("speculation.") for k in by_key)
            assert "image_concurrency" not in by_key
            assert "language_model_only" not in by_key
            assert "prompt_logprobs" not in by_key
            assert "mixed_prefill" not in by_key
            assert "compute_mode" not in by_key
            assert "decode_override" not in by_key

        if recipe is alpha:
            assert by_key["compute_mode"].scope == proto.PREPARATION_PARAMETER_SCOPE_COMBINE
            assert by_key["compute_mode"].value.string_value == "auto"
            assert "toy_compression" not in by_key
            assert by_key["language_model_only"].scope == proto.PREPARATION_PARAMETER_SCOPE_COMBINE
            assert by_key["language_model_only"].value.bool_value is True
            assert by_key["prompt_logprobs"].value.bool_value is True
            assert by_key["compiled_forwards"].value.bool_value is False
            assert "mixed_prefill" not in by_key
            assert "decode_override" not in by_key

        if recipe is beta:
            assert by_key["decode_override"].value.bool_value is True
            assert by_key["mixed_prefill"].value.bool_value is True
            assert not any(k.startswith("speculation.") for k in by_key)
            assert "compute_mode" not in by_key

        # Golden invariant: reproduction_args parses cleanly back into an equivalent RecipeParameters!
        repro_argv = manifest["reproduction_args"][1:]
        assert repro_argv[0] == "export"
        reparsed_args = parse_args(repro_argv, recipes=recipes)
        reparsed_params = preparation.resolve_parameters(model, reparsed_args)
        reparsed_plan = planning.plan(recipe, recipe.checkpoints[0], (_gpu(0),), selected=(0,), parameters=reparsed_params)
        repro_manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), reparsed_plan)
        assert repro_manifest["parameters"] == manifest["parameters"]


def test_dfly2_speculative_method_declaration_cli_display_and_planning():
    from ignition.cli import display
    from ignition.declaration import SamplingDefaults, SpeculativeMethod

    spec = SpeculationParameters(
        draft_tokens=5,
        max_draft_tokens=7,
        token_capacity=64,
        method="dspark",
        supported_methods=("dspark", "dfly2"),
    )
    params = BetaParameters(
        max_concurrency=8,
        speculation=spec,
        mixed_prefill=MixedPrefillParameters(),
    )
    recipe = Recipe(
        name="dfly2_test",
        served_model_name="DFly2-Test",
        prepare=lambda *a, **k: None,
        usage=lambda *a, **k: (),
        sampling_defaults=SamplingDefaults(temperature=1.0, top_p=1.0, top_k=-1),
        checkpoints=(
            Checkpoint(
                alias="dfly2-1b",
                repository="example/dfly2-1b",
                revision="deadbeef",
                architecture="DFly2ForCausalLM",
                stored_bytes=1 << 30,
                device_bytes=2 << 30,
                draft_bytes=512 << 20,
            ),
        ),
        context=4096,
        page_tokens=16,
        parameters=params,
        ranks=MappingProxyType({1: RankCosts(1 << 20, 64 << 10, 1 << 28)}),
        plain_ranks=MappingProxyType({1: RankCosts(1 << 20, 64 << 10, 1 << 28)}),
        speculative_methods=MappingProxyType({
            "dfly2": SpeculativeMethod(
                MappingProxyType({1: RankCosts(512 << 10, 64 << 10, 1 << 28)}),
                draft_bytes=256 << 20,
            ),
        }),
    )
    model = SimpleNamespace(recipe=recipe, checkpoint=recipe.checkpoints[0], alias="dfly2-1b")

    # --speculative-method dfly2 parses, resolves, formats, and plans cleanly.
    args = parse_args(["export", "dfly2-1b", "--speculative-method", "dfly2"], recipes=(recipe,))
    resolved = preparation.resolve_parameters(model, args)
    assert resolved.speculation is not None
    assert resolved.speculation.method == "dfly2"
    assert "DFly2" in display.speculation(resolved.speculation)
    ctx = ParameterContext(recipe=recipe, checkpoint=recipe.checkpoints[0], root=resolved)
    assert "DFly2 proposals" in resolved.speculation.status_parts(ctx)
    plan = planning.plan(recipe, recipe.checkpoints[0], (_gpu(0),), selected=(0,), parameters=resolved)
    assert plan.speculation is not None and plan.speculation.method == "dfly2"

    # --mixed-prefill rejects --speculative-method dfly2.
    bad = parse_args(["export", "dfly2-1b", "--mixed-prefill", "--speculative-method", "dfly2"], recipes=(recipe,))
    with pytest.raises(UsageError, match="separate prefill and verification graphs"):
        preparation.resolve_parameters(model, bad)
