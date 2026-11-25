# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Export preflight validation before GPU preparation begins."""

import json
from types import MappingProxyType, SimpleNamespace

import pytest

from ignition import planning
from ignition.cli import preparation, serve
from ignition.cli.config import Configuration
from ignition.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults, Speculation
from ignition.errors import PreparationError, UsageError
from ignition.preparation import rebuild_manifest
from ignition.registry import ResolvedModel, snapshot_directory

GIB = 1 << 30
MIB = 1 << 20


def _fixture_model():
    checkpoint = Checkpoint(
        alias="fixture-tp-model",
        repository="example/Fixture-TP-Model",
        revision="0123456789abcdef0123456789abcdef01234567",
        architecture="FixtureForCausalLM",
        stored_bytes=32 * GIB,
        device_bytes=32 * GIB,
        draft_bytes=2 * GIB,
    )
    recipe = Recipe(
        name="fixture_tp",
        served_model_name="Fixture-TP-Model",
        checkpoints=(checkpoint,),
        context=65536,
        page_tokens=64,
        concurrency=4,
        max_concurrency=32,
        max_draft_tokens=7,
        prefill_chunk=4096,
        ranks=MappingProxyType({1: RankCosts(64 * MIB, 2 * MIB, 2 * GIB),
                                4: RankCosts(16 * MIB, 512 << 10, 2 * GIB)}),
        plain_ranks=MappingProxyType({1: RankCosts(48 * MIB, MIB, 2 * GIB),
                                      4: RankCosts(12 * MIB, 256 << 10, 2 * GIB)}),
        prepare=lambda *args, **kwargs: pytest.fail("GPU preparation must not start when preflight fails"),
        usage=lambda prepared: (),
        sampling_defaults=SamplingDefaults(temperature=1.0, top_p=1.0, top_k=-1),
        speculation=Speculation(draft_tokens=3),
        row_shard_prefill_world_sizes=(4,),
    )
    return ResolvedModel(recipe, checkpoint)


def _write_indexed_snapshot(snapshot):
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "README.md").write_text("---\nlicense: apache-2.0\n---\n# Fixture\n")
    shard = snapshot / "model-00001-of-00001.safetensors"
    header = json.dumps({"weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
    shard.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00" * 8)
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 8},
        "weight_map": {"weight": shard.name},
    }))


def _fixture_plan(tmp_path, model):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        f'[paths]\n'
        f'hub_cache = "{(tmp_path / "hub").as_posix()}"\n'
        f'core_cache = "{(tmp_path / "cores").as_posix()}"\n'
        f'export_staging = "{(tmp_path / "staging").as_posix()}"\n'
        f'kernel_cache = "{(tmp_path / "kernels").as_posix()}"\n'
    )
    configuration = Configuration.load(cfg_path)
    gpus = tuple(
        planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}",
                     96 * GIB, 80 * GIB, (12, 0))
        for index in range(4)
    )
    host = preparation.Host(gpus, None)
    concurrency = configuration.max_concurrency(model.checkpoint.alias, model.recipe.concurrency, None)
    plan = preparation.derive(
        configuration, model, host, gpus="0-3", concurrency=concurrency,
        speculation=model.recipe.speculation, row_shard_prefill=True,
    )
    return configuration, plan


def test_validate_preparation_manifest_rejects_unknown_fields_and_outdated_extension(tmp_path, monkeypatch):
    import flare_heliopause

    model = _fixture_model()
    _, plan = _fixture_plan(tmp_path, model)
    valid = rebuild_manifest.from_plan(model, plan, compiled_forwards=True)
    rebuild_manifest.validate_preparation_manifest(valid)

    bad = dict(valid, row_shard_attention=True)
    with pytest.raises(ValueError, match="row_shard_attention"):
        rebuild_manifest.validate_preparation_manifest(bad)

    unsorted = dict(valid, parameters=list(reversed(valid["parameters"])))
    with pytest.raises(ValueError, match="strictly ascending order"):
        rebuild_manifest.validate_preparation_manifest(unsorted)

    monkeypatch.setattr(flare_heliopause, "export_validate_preparation_manifest", None, raising=False)
    with pytest.raises(ValueError, match="rebuild ./heliopause"):
        rebuild_manifest.validate_preparation_manifest(valid)

    drift_check = lambda encoded: (_ for _ in ()).throw(
        ValueError("unrecognized field name 'row_shard_attention'"))
    monkeypatch.setattr(flare_heliopause, "export_validate_preparation_manifest", drift_check, raising=False)
    with pytest.raises(ValueError, match="row_shard_attention"):
        rebuild_manifest.validate_preparation_manifest(valid)


def test_preflight_export_fails_before_prepare_on_manifest_drift_or_unwritable_staging(tmp_path, monkeypatch):
    model = _fixture_model()
    configuration, plan = _fixture_plan(tmp_path, model)
    hub_cache = configuration.directory("hub_cache")
    _write_indexed_snapshot(snapshot_directory(hub_cache, model.checkpoint))
    args = SimpleNamespace(config=configuration.path, official_source=None, no_compile=False)
    signer = serve._publisher(configuration, args)

    publisher, options = serve._preflight_export(
        configuration, args, model, plan, hub_cache=hub_cache, signer=signer)
    assert publisher.license is not None and publisher.license.name == "apache-2.0"
    manifest = options["preparation_manifest"]
    params = {entry["key"]: entry["value"] for entry in manifest["parameters"]}
    assert params["row_shard_prefill"] == {"bool_value": True}
    assert manifest["checkpoint_revision"] == model.checkpoint.revision

    monkeypatch.setattr(
        rebuild_manifest, "from_plan",
        lambda *a, **kw: {**manifest, "row_shard_attention": True},
    )
    with pytest.raises(PreparationError, match="row_shard_attention"):
        serve._preflight_export(configuration, args, model, plan, hub_cache=hub_cache, signer=signer)


def test_preflight_export_rejects_file_blocking_staging_directory(tmp_path):
    model = _fixture_model()
    configuration, plan = _fixture_plan(tmp_path, model)
    hub_cache = configuration.directory("hub_cache")
    _write_indexed_snapshot(snapshot_directory(hub_cache, model.checkpoint))
    staging_blocker = configuration.directory("export_staging")
    staging_blocker.write_text("not a directory")
    args = SimpleNamespace(config=configuration.path, official_source=None, no_compile=False)
    signer = serve._publisher(configuration, args)
    with pytest.raises(UsageError, match="export staging or core cache is not writable"):
        serve._preflight_export(configuration, args, model, plan, hub_cache=hub_cache, signer=signer)
