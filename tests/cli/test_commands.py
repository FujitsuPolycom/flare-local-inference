# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Recipe-free command-line parsing and preparation-flag forwarding."""

from types import MappingProxyType

import pytest

from ignition import planning
from ignition.cli import _parser, preparation
from ignition.cli.config import Configuration
from ignition.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults, Speculation
from ignition.errors import UsageError
from ignition.preparation.bootstrap import BasePreparationConfig, common_preparation_kwargs
from ignition.registry import ResolvedModel

GIB = 1 << 30
MIB = 1 << 20


def _fixture_recipe(*, ranks=(1, 2, 4), row_shard_world_sizes=(2, 4),
                    replicated_attention_bytes=None, joint_tuning=True):
    speculation = Speculation(draft_tokens=3)
    checkpoint = Checkpoint(
        alias="fixture-tp-model",
        repository="example/Fixture-TP-Model",
        revision="0123456789abcdef0123456789abcdef01234567",
        architecture="FixtureForCausalLM",
        stored_bytes=32 * GIB,
        device_bytes=32 * GIB,
        draft_bytes=2 * GIB,
    )
    return Recipe(
        name="fixture_tp",
        served_model_name="Fixture-TP-Model",
        checkpoints=(checkpoint,),
        context=65536,
        page_tokens=64,
        concurrency=4,
        max_concurrency=32,
        max_draft_tokens=7,
        prefill_chunk=4096,
        ranks=MappingProxyType({size: RankCosts(64 * MIB // size, 2 * MIB // size, 2 * GIB) for size in ranks}),
        plain_ranks=MappingProxyType({size: RankCosts(48 * MIB // size, MIB // size, 2 * GIB) for size in ranks}),
        prepare=lambda *args, **kwargs: None,
        usage=lambda prepared: (),
        sampling_defaults=SamplingDefaults(temperature=1.0, top_p=1.0, top_k=-1),
        speculation=speculation,
        joint_tuning=joint_tuning,
        benchmark_world_sizes=(2, 4),
        row_shard_prefill_world_sizes=row_shard_world_sizes,
        replicated_attention_bytes=replicated_attention_bytes,
    )


def _gpu(index, free_gib=80, capability=(12, 0)):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}",
                        96 * GIB, free_gib * GIB, capability)


@pytest.mark.parametrize("command", ["export", "debug-serve", "bench"])
@pytest.mark.parametrize(
    ("flag", "attr"),
    [("--row-shard-prefill", "row_shard_prefill"), ("--row-shard-attention", "row_shard_attention")],
)
def test_row_shard_flags_are_model_agnostic(command, flag, attr, capsys):
    parser = _parser()
    assert not getattr(parser.parse_args([command, "fixture-tp-model"]), attr)
    assert getattr(parser.parse_args([command, "fixture-tp-model", flag]), attr)

    with pytest.raises(SystemExit) as exc:
        _parser().parse_args([command, "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    section = help_text.split(flag, 1)[1].split("--lock-gpu-clocks", 1)[0]
    assert "DeepSeek" not in section
    assert "supported TP recipes" in section


def test_derive_forwards_row_shard_prefill_and_rejects_joint_tuning(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("")
    configuration = Configuration.load(cfg_path)
    recipe = _fixture_recipe()
    model = ResolvedModel(recipe, recipe.checkpoints[0], "alias")
    host = preparation.Host(tuple(_gpu(i) for i in range(4)), None)
    concurrency = configuration.max_concurrency(model.checkpoint.alias, recipe.concurrency, None)

    plan = preparation.derive(
        configuration, model, host, gpus="0,1", concurrency=concurrency,
        speculation=recipe.speculation, row_shard_prefill=True,
    )
    assert plan.row_shard_prefill
    assert plan.to_dict()["row_shard_prefill"] is True
    kwargs = common_preparation_kwargs(plan, compile_forwards=True, kernel_cache=None, retain_graphs=False)
    config = BasePreparationConfig(**kwargs)
    assert config.prefill_chunk_size == 4096
    assert config.max_active == 4

    with pytest.raises(UsageError, match="--row-shard-prefill cannot be combined with --joint-tuning"):
        preparation.derive(
            configuration, model, host, gpus="0,1", concurrency=concurrency,
            speculation=recipe.speculation, row_shard_prefill=True, joint_tuning=True,
        )


def test_row_shard_attention_forwarding_and_serve_progress(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from ignition.cli import bench, serve

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("")
    configuration = Configuration.load(cfg_path)
    recipe = _fixture_recipe(
        replicated_attention_bytes=lambda w, *, prompt_logprobs=False, mtp_only=False: 2 * GIB,
    )
    model = ResolvedModel(recipe, recipe.checkpoints[0], "alias")
    host = preparation.Host(tuple(_gpu(i) for i in range(4)), None)
    concurrency = configuration.max_concurrency(model.checkpoint.alias, recipe.concurrency, None)

    derived = preparation.derive(
        configuration, model, host, gpus="0,1", concurrency=concurrency,
        speculation=recipe.speculation, row_shard_prefill=True, row_shard_attention=True,
    )
    assert derived.row_shard_attention and derived.to_dict()["row_shard_attention"] is True

    class StopAfterSelect(RuntimeError):
        pass

    seen = []
    stage_updates = []

    def spy_derive(*args, **kwargs):
        seen.append((kwargs["row_shard_prefill"], kwargs["row_shard_attention"]))
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
    args = _parser().parse_args(
        ["export", "fixture-tp-model", "--config", str(cfg_path), "--gpus", "0,1",
         "--row-shard-prefill", "--row-shard-attention"]
    )
    with pytest.raises(StopAfterSelect):
        serve._prepare(args)
    assert seen == [(True, True)]
    assert any(
        label == "Select GPUs" and " · row-sharded prefill · row-sharded attention" in detail
        for label, detail in stage_updates
    )

    bench_seen = []

    def spy_bench_derive(*args, **kwargs):
        bench_seen.append((kwargs["row_shard_prefill"], kwargs["row_shard_attention"]))
        raise StopAfterSelect()

    monkeypatch.setattr(bench.Configuration, "load", lambda _path: configuration)
    monkeypatch.setattr(bench, "resolve_model", lambda _cfg, _arg: model)
    monkeypatch.setattr(bench.Host, "discover", lambda: host)
    monkeypatch.setattr(bench, "derive", spy_bench_derive)
    bench_args = _parser().parse_args(
        ["bench", "fixture-tp-model", "--config", str(cfg_path), "--gpus", "0,1",
         "--row-shard-prefill", "--row-shard-attention"]
    )
    with pytest.raises(StopAfterSelect):
        bench.run(bench_args)
    assert bench_seen == [(True, True)]
