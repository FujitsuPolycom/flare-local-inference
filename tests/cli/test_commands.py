# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Recipe-free dynamic CLI parsing and RecipeParameters forwarding."""

from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType

import pytest

from ignition import planning
from ignition.cli import bench, parse_args, preparation, serve
from ignition.cli.config import Configuration
from ignition.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults
from ignition.errors import UsageError
from ignition.parameters import (
    ParameterContext,
    ParameterScope,
    PromptLogprobsParameters,
    RecipeParameters,
    SpeculationParameters,
    param,
)
from ignition.preparation.bootstrap import BasePreparationConfig, common_preparation_kwargs
from ignition.registry import ResolvedModel

GIB = 1 << 30
MIB = 1 << 20


@dataclass(frozen=True)
class FixtureShardingParameters(RecipeParameters):
    """Recipe-defined subclass exercising custom flags, world-size filtering, weights, and status."""

    sharded_prefill: bool = param(
        False,
        flag="--sharded-prefill",
        group="advanced options",
        scope=ParameterScope.DEPLOYMENT,
        help="shard prefill reductions across ranks",
        status_label=lambda v: "sharded prefill" if v else None,
    )
    sharded_attention: bool = param(
        False,
        flag="--sharded-attention",
        group="advanced options",
        scope=ParameterScope.DEPLOYMENT,
        help="replicate attention projections alongside --sharded-prefill",
        status_label=lambda v: "sharded attention" if v else None,
    )
    sharded_world_sizes: tuple[int, ...] = (2, 4)
    replicated_attention_bytes: int = 2 * GIB

    def filter_world_sizes(self, world_sizes: tuple[int, ...], context: ParameterContext) -> tuple[int, ...]:
        sizes = super().filter_world_sizes(world_sizes, context)
        if self.sharded_prefill:
            return tuple(s for s in sizes if s in self.sharded_world_sizes)
        return sizes

    def validate(self, context: ParameterContext) -> None:
        super().validate(context)
        if self.sharded_attention and not self.sharded_prefill:
            raise UsageError("--sharded-attention requires --sharded-prefill")
        if self.sharded_prefill and context.world_size is not None:
            if context.world_size not in self.sharded_world_sizes:
                raise UsageError(f"--sharded-prefill is not supported at TP{context.world_size}")

    def adjust_weights_bytes(self, weights_bytes: int, context: ParameterContext) -> int:
        weights_bytes = super().adjust_weights_bytes(weights_bytes, context)
        if self.sharded_attention:
            return weights_bytes + self.replicated_attention_bytes
        return weights_bytes


def _fixture_recipe(*, ranks=(1, 2, 4)):
    checkpoint = Checkpoint(
        alias="fixture-tp-model",
        repository="example/Fixture-TP-Model",
        revision="0123456789abcdef0123456789abcdef01234567",
        architecture="FixtureForCausalLM",
        stored_bytes=32 * GIB,
        device_bytes=32 * GIB,
        draft_bytes=2 * GIB,
    )
    params = FixtureShardingParameters(
        max_concurrency=4,
        max_concurrency_limit=32,
        prefill_chunk=4096,
        speculation=SpeculationParameters(draft_tokens=3, max_draft_tokens=7),
        prompt_logprobs=PromptLogprobsParameters(),
    )
    return Recipe(
        name="fixture_tp",
        served_model_name="Fixture-TP-Model",
        checkpoints=(checkpoint,),
        context=65536,
        page_tokens=64,
        parameters=params,
        ranks=MappingProxyType({size: RankCosts(64 * MIB // size, 2 * MIB // size, 2 * GIB) for size in ranks}),
        plain_ranks=MappingProxyType({size: RankCosts(48 * MIB // size, MIB // size, 2 * GIB) for size in ranks}),
        prepare=lambda *args, **kwargs: None,
        usage=lambda prepared: (),
        sampling_defaults=SamplingDefaults(temperature=1.0, top_p=1.0, top_k=-1),
        benchmark_world_sizes=(2, 4),
    )


def _gpu(index, free_gib=80, capability=(12, 0)):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}",
                        96 * GIB, free_gib * GIB, capability)


@pytest.mark.parametrize("command", ["export", "debug-serve", "bench"])
def test_bare_help_stays_recipe_free_and_model_help_is_dynamic(command, capsys):
    recipe = _fixture_recipe()
    with pytest.raises(SystemExit) as exc:
        parse_args([command, "--help"], recipes=(recipe,))
    assert exc.value.code == 0
    bare_out = capsys.readouterr().out
    assert "--sharded-prefill" not in bare_out
    assert "--sharded-attention" not in bare_out
    assert "--mtp-draft-tokens" not in bare_out
    assert "fixture-tp-model" in bare_out

    with pytest.raises(SystemExit) as exc:
        parse_args([command, "fixture-tp-model", "--help"], recipes=(recipe,))
    assert exc.value.code == 0
    model_out = " ".join(capsys.readouterr().out.split())
    assert "--sharded-prefill" in model_out
    assert "--sharded-attention" in model_out
    assert "--mtp-draft-tokens" in model_out
    assert "0..7; default: 3" in model_out
    assert "1..32; default: 4" in model_out


def test_derive_and_base_preparation_config_embed_parameters(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("")
    configuration = Configuration.load(cfg_path)
    recipe = _fixture_recipe()
    model = ResolvedModel(recipe, recipe.checkpoints[0])
    host = preparation.Host(tuple(_gpu(i) for i in range(4)), None)

    args = parse_args(["export", "fixture-tp-model", "--gpus", "0,1", "--sharded-prefill"], recipes=(recipe,))
    params = preparation.resolve_parameters(model, args, configuration)
    plan = preparation.derive(configuration, model, host, gpus=args.gpus, parameters=params)
    assert plan.parameters.sharded_prefill is True
    assert plan.to_dict()["sharded_prefill"] is True
    kwargs = common_preparation_kwargs(plan, compile_forwards=True, kernel_cache=None, retain_graphs=False)
    config = BasePreparationConfig(**kwargs)
    assert config.parameters is plan.parameters
    assert config.prefill_chunk_size == 4096
    assert config.max_active == 4
    assert config.speculative_widths == (1, 2, 3, 4)


def test_custom_recipe_parameters_forward_through_serve_and_bench(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("")
    configuration = Configuration.load(cfg_path)
    recipe = _fixture_recipe()
    model = ResolvedModel(recipe, recipe.checkpoints[0])
    host = preparation.Host(tuple(_gpu(i) for i in range(4)), None)

    args = parse_args(
        ["export", "fixture-tp-model", "--config", str(cfg_path), "--gpus", "0,1",
         "--sharded-prefill", "--sharded-attention"],
        recipes=(recipe,),
    )
    resolved_params = preparation.resolve_parameters(model, args, configuration)
    derived = preparation.derive(configuration, model, host, gpus="0,1", parameters=resolved_params)
    assert derived.parameters.sharded_attention is True
    assert derived.to_dict()["sharded_attention"] is True

    class StopAfterSelect(RuntimeError):
        pass

    seen = []
    stage_updates = []

    def spy_derive(*a, **kwargs):
        seen.append((kwargs["parameters"].sharded_prefill, kwargs["parameters"].sharded_attention))
        return derived

    @contextmanager
    def fake_phase(_progress, label, **_kwargs):
        class Stage:
            def update(self, detail):
                stage_updates.append((label, detail))
        yield Stage()
        if label == "Select GPUs":
            raise StopAfterSelect()

    monkeypatch.setattr(serve.Configuration, "load", lambda _path: configuration)
    monkeypatch.setattr(serve, "_publisher", lambda *_a, **_kw: None)
    monkeypatch.setattr(serve, "resolve_model", lambda _cfg, _arg: model)
    monkeypatch.setattr(serve.Host, "discover", lambda: host)
    monkeypatch.setattr(serve, "derive", spy_derive)
    monkeypatch.setattr("ignition.progress.phase", fake_phase)
    with pytest.raises(StopAfterSelect):
        serve._prepare(args)
    assert seen == [(True, True)]
    assert any(
        label == "Select GPUs" and "sharded prefill · sharded attention" in detail
        for label, detail in stage_updates
    )

    bench_seen = []

    def spy_bench_derive(*a, **kwargs):
        bench_seen.append((kwargs["parameters"].sharded_prefill, kwargs["parameters"].sharded_attention))
        raise StopAfterSelect()

    monkeypatch.setattr(bench.Configuration, "load", lambda _path: configuration)
    monkeypatch.setattr(bench, "resolve_model", lambda _cfg, _arg: model)
    monkeypatch.setattr(bench.Host, "discover", lambda: host)
    monkeypatch.setattr(bench, "derive", spy_bench_derive)
    bench_args = parse_args(
        ["bench", "fixture-tp-model", "--config", str(cfg_path), "--gpus", "0,1",
         "--sharded-prefill", "--sharded-attention"],
        recipes=(recipe,),
    )
    with pytest.raises(StopAfterSelect):
        bench.run(bench_args)
    assert bench_seen == [(True, True)]
