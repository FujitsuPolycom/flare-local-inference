# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Upfront export disk-space preflight across single-host and multi-host topologies."""

from collections import namedtuple
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from ignition import planning
from ignition.cli import parse_args, preparation, serve
from ignition.cli.config import Configuration
from ignition.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults
from ignition.errors import UsageError
from ignition.parameters import RecipeParameters, SpeculationParameters
from ignition.registry import ResolvedModel

GIB = 1 << 30
MIB = 1 << 20
DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])


def _deepseek_like_recipe(*, stored_bytes=300 * GIB):
    """TP4 recipe with ~75 GiB per-rank weights (300 GiB device total) and optional non-device stored bytes."""
    speculation = SpeculationParameters(draft_tokens=5, max_draft_tokens=7)
    checkpoint = Checkpoint(
        alias="fixture-300b",
        repository="deepseek-ai/Fixture-V4.1-Flash",
        revision="1bae0e1088993b9b82de08f59de5be5a34e42170",
        architecture="FixtureV41ForCausalLM",
        stored_bytes=stored_bytes,
        device_bytes=300 * GIB,
        draft_bytes=8 * GIB,
    )
    return Recipe(
        name="fixture41",
        served_model_name="Fixture-V4.1-Flash",
        checkpoints=(checkpoint,),
        context=262144,
        page_tokens=128,
        parameters=RecipeParameters(
            max_concurrency=8,
            max_concurrency_limit=64,
            prefill_chunk=4096,
            speculation=speculation,
        ),
        ranks=MappingProxyType({size: RankCosts(3 * MIB, 128 * 1024, 8 * GIB) for size in (1, 2, 4)}),
        plain_ranks=MappingProxyType({size: RankCosts(3 * MIB, 128 * 1024, 8 * GIB) for size in (1, 2, 4)}),
        prepare=lambda *args, **kwargs: pytest.fail("GPU preparation must not start when disk preflight fails"),
        usage=lambda prepared: (),
        sampling_defaults=SamplingDefaults(temperature=1.0, top_p=1.0, top_k=-1),
    )


def _gpu(index, free_gib=115):
    return planning.Gpu(
        index,
        "NVIDIA GB10",
        f"GPU-{index:04d}",
        128 * GIB,
        free_gib * GIB,
        (12, 1),
    )


def _mock_filesystems(monkeypatch, mapping):
    """Map resolved path prefixes to synthetic `(st_dev, free_bytes)` tuples."""
    import os

    entries = sorted(
        ((os.path.realpath(os.fspath(prefix)), dev, free) for prefix, (dev, free) in mapping.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    real_stat = os.stat

    def lookup(target):
        if isinstance(target, int):
            return None
        normalized = os.path.abspath(os.fspath(target))
        for prefix, dev, free in entries:
            if normalized == prefix or normalized.startswith(prefix + os.sep):
                return dev, free
        return None

    def fake_stat(target, *args, **kwargs):
        base = real_stat(target, *args, **kwargs)
        matched = lookup(target)
        if matched is not None:
            dev, _ = matched
            attrs = {name: getattr(base, name) for name in dir(base) if name.startswith("st_")}
            attrs["st_dev"] = dev
            return SimpleNamespace(**attrs)
        return base

    def fake_disk_usage(target):
        matched = lookup(target)
        if matched is None:
            raise AssertionError(f"unmocked disk_usage target: {target}")
        _, free = matched
        return DiskUsage(total=max(free, 1000 * GIB), used=0, free=free)

    monkeypatch.setattr(os, "stat", fake_stat)
    monkeypatch.setattr(planning.shutil, "disk_usage", fake_disk_usage)


def test_single_host_enospc_fails_immediately_and_weight_reuse_flags_bypass(tmp_path, monkeypatch):
    recipe = _deepseek_like_recipe()
    gpus = tuple(_gpu(i) for i in range(4))
    tp4_plan = planning.plan(recipe, recipe.checkpoints[0], gpus, selected=(0, 1, 2, 3))
    assert tp4_plan.weights_bytes == 75 * GIB

    cache_dir = tmp_path / "cores"
    # 119 GiB free cannot hold 4 x 75 GiB weights + 8 GiB executables/staging (308 GiB).
    _mock_filesystems(monkeypatch, {tmp_path: (101, 119 * GIB)})
    with pytest.raises(UsageError) as caught:
        planning.check_export_disk_space(
            tp4_plan,
            recipe.checkpoints[0].repository,
            core_cache=cache_dir,
            host_name="spark-0",
        )
    message = str(caught.value)
    assert "rank 0 (spark-0)" in message
    assert str(cache_dir) in message
    assert "119.0 GiB available" in message
    assert "308.0 GiB" in message
    assert "--reuse-weights" in message and "--no-weights" in message

    # Both --no-weights and --reuse-weights require only executables + staging (8.0 GiB across 4 ranks) and pass.
    planning.check_export_disk_space(
        tp4_plan,
        recipe.checkpoints[0].repository,
        core_cache=cache_dir,
        no_weights=True,
        host_name="spark-0",
    )
    planning.check_export_disk_space(
        tp4_plan,
        recipe.checkpoints[0].repository,
        core_cache=cache_dir,
        reuse_weights=True,
        host_name="spark-0",
    )

    # When free space cannot even hold executables + staging (e.g. 3 GiB < 8 GiB), --no-weights still fails without suggesting --reuse-weights.
    _mock_filesystems(monkeypatch, {tmp_path: (101, 3 * GIB)})
    with pytest.raises(UsageError) as exec_shortfall:
        planning.check_export_disk_space(
            tp4_plan,
            recipe.checkpoints[0].repository,
            core_cache=cache_dir,
            no_weights=True,
            host_name="spark-0",
        )
    assert "3.0 GiB available" in str(exec_shortfall.value)
    assert "8.0 GiB" in str(exec_shortfall.value)
    assert "--reuse-weights" not in str(exec_shortfall.value)


def test_same_filesystem_vs_split_filesystem_staging_and_cache_accounting(tmp_path, monkeypatch):
    recipe = _deepseek_like_recipe()
    gpus = tuple(_gpu(i) for i in range(4))
    tp4_plan = planning.plan(recipe, recipe.checkpoints[0], gpus, selected=(0, 1, 2, 3))

    cache_dir = tmp_path / "nvme_cache" / "cores"
    stage_dir = tmp_path / "fast_scratch" / "staging"
    (tmp_path / "nvme_cache").mkdir()
    (tmp_path / "fast_scratch").mkdir()

    # Split filesystems (st_dev 201 vs 202): core_cache needs 304 GiB (300 weights + 4 exec) and export_staging needs 4 GiB.
    _mock_filesystems(monkeypatch, {
        tmp_path / "nvme_cache": (201, 306 * GIB),
        tmp_path / "fast_scratch": (202, 6 * GIB),
    })
    planning.check_export_disk_space(
        tp4_plan,
        recipe.checkpoints[0].repository,
        core_cache=cache_dir,
        export_staging=stage_dir,
        host_name="spark-0",
    )

    # Same two paths on a single shared filesystem (both st_dev 201) with 306 GiB free fail because 304 + 4 = 308 GiB > 306 GiB.
    _mock_filesystems(monkeypatch, {
        tmp_path / "nvme_cache": (201, 306 * GIB),
        tmp_path / "fast_scratch": (201, 306 * GIB),
    })
    with pytest.raises(UsageError) as shared_err:
        planning.check_export_disk_space(
            tp4_plan,
            recipe.checkpoints[0].repository,
            core_cache=cache_dir,
            export_staging=stage_dir,
            host_name="spark-0",
        )
    assert "shared filesystem" in str(shared_err.value)
    assert str(stage_dir) in str(shared_err.value)
    assert "308.0 GiB" in str(shared_err.value)

    # Split filesystems where core_cache has plenty of space (500 GiB) but export_staging has 0.5 GiB (< 4 GiB) fail naming export_staging.
    _mock_filesystems(monkeypatch, {
        tmp_path / "nvme_cache": (201, 500 * GIB),
        tmp_path / "fast_scratch": (202, 512 * MIB),
    })
    with pytest.raises(UsageError) as stage_err:
        planning.check_export_disk_space(
            tp4_plan,
            recipe.checkpoints[0].repository,
            core_cache=cache_dir,
            export_staging=stage_dir,
            host_name="spark-0",
        )
    assert str(stage_dir) in str(stage_err.value)
    assert "0.5 GiB available" in str(stage_err.value)
    assert "4.0 GiB" in str(stage_err.value)
    assert "--reuse-weights" not in str(stage_err.value)


def test_existing_same_model_same_tp_cached_blobs_prevent_false_positive(tmp_path, monkeypatch):
    recipe = _deepseek_like_recipe()
    model_id = recipe.checkpoints[0].repository
    gpus = tuple(_gpu(i) for i in range(4))
    tp4_plan = planning.plan(recipe, recipe.checkpoints[0], gpus, selected=(0, 1, 2, 3))

    cache_dir = tmp_path / "cores"
    model_key = planning._model_cache_key(model_id)
    assert model_key == "deepseek-ai_2FFixture-V4.1-Flash"

    # Cached blobs for a DIFFERENT TP width (tp2) or DIFFERENT model do not waive TP4's 300 GiB requirement.
    tp2_weights = cache_dir / "models" / model_key / "entry-tp2" / "weights" / "tp2" / "rank-0"
    tp2_weights.mkdir(parents=True)
    other_weights = cache_dir / "models" / planning._model_cache_key("other/Model") / "entry-tp4" / "weights" / "tp4" / "rank-0"
    other_weights.mkdir(parents=True)
    tp2_blob = tp2_weights / "layer-000.safetensors"
    other_blob = other_weights / "layer-000.safetensors"
    tp2_blob.write_bytes(b"x")
    other_blob.write_bytes(b"y")

    real_stat = Path.stat

    def sized_stat(self, *args, **kwargs):
        base = real_stat(self, *args, **kwargs)
        if self.suffix == ".safetensors":
            attrs = {k: getattr(base, k) for k in dir(base) if k.startswith("st_")}
            attrs["st_size"] = 300 * GIB
            return SimpleNamespace(**attrs)
        return base

    monkeypatch.setattr(Path, "stat", sized_stat)
    _mock_filesystems(monkeypatch, {tmp_path: (301, 25 * GIB)})

    with pytest.raises(UsageError, match="308.0 GiB"):
        planning.check_export_disk_space(tp4_plan, model_id, core_cache=cache_dir, host_name="spark-0")

    # Once a TP4 entry for this exact model is present in core_cache, re-exporting TP4 with 25 GiB free succeeds.
    tp4_weights = cache_dir / "models" / model_key / "entry-tp4" / "weights" / "tp4" / "rank-0"
    tp4_weights.mkdir(parents=True)
    (tp4_weights / "layer-000.safetensors").write_bytes(b"z")
    planning.check_export_disk_space(tp4_plan, model_id, core_cache=cache_dir, host_name="spark-0")


def test_host_resident_engram_bytes_checked_on_rank_zero_and_deployment_shards_credited(tmp_path, monkeypatch):
    recipe = _deepseek_like_recipe(stored_bytes=510 * GIB)
    model_id = recipe.checkpoints[0].repository
    gpus = tuple(_gpu(i) for i in range(4))
    tp4_plan = planning.plan(recipe, recipe.checkpoints[0], gpus, selected=(0, 1, 2, 3))

    cache_dir = tmp_path / "cores"
    # 373 GiB free on rank 0 exceeds 308 GiB GPU weights + executables, yet falls short of 518 GiB once
    # 210 GiB of non-device stored Engram tensors (510 GiB stored - 300 GiB device) are included on rank 0.
    topologies = (
        {
            "host": "spark-0",
            "boot_id": "boot-0",
            "storage": {
                "core_cache": {"path": str(cache_dir), "device": 10, "free_bytes": 373 * GIB},
                "export_staging": {"path": str(cache_dir), "device": 10, "free_bytes": 373 * GIB},
            },
        },
        *(
            {
                "host": f"spark-{r}",
                "boot_id": f"boot-{r}",
                "storage": {
                    "core_cache": {"path": f"/var/spark-{r}/cores", "device": 10, "free_bytes": 95 * GIB},
                    "export_staging": {"path": f"/var/spark-{r}/cores", "device": 10, "free_bytes": 95 * GIB},
                },
            }
            for r in (1, 2, 3)
        ),
    )
    with pytest.raises(UsageError) as engram_err:
        planning.check_export_disk_space(tp4_plan, model_id, core_cache=cache_dir, topologies=topologies)
    assert "rank 0 (spark-0)" in str(engram_err.value)
    assert "373.0 GiB available" in str(engram_err.value)
    assert "518.0 GiB" in str(engram_err.value)

    # Peer ranks still require only their per-rank GPU weights (75 GiB + 2 GiB = 77 GiB), so 95 GiB on peers passes
    # once rank 0 either has >= 518 GiB free OR already holds both TP4 rank weights and partitioned Engram shards.
    model_key = planning._model_cache_key(model_id)
    tp4_weights = cache_dir / "models" / model_key / "entry-tp4" / "weights" / "tp4" / "rank-0"
    tp4_deploy = cache_dir / "models" / model_key / "entry-tp4" / "deployments" / "gb10-tp4-202611271655"
    tp4_weights.mkdir(parents=True)
    tp4_deploy.mkdir(parents=True)
    (tp4_weights / "layer-000.safetensors").write_bytes(b"w")
    (tp4_deploy / "engram-0-shard-0-weights.safetensors").write_bytes(b"e")

    real_stat = Path.stat

    def sized_stat(self, *args, **kwargs):
        base = real_stat(self, *args, **kwargs)
        if self.name == "layer-000.safetensors":
            attrs = {k: getattr(base, k) for k in dir(base) if k.startswith("st_")}
            attrs["st_size"] = 300 * GIB
            return SimpleNamespace(**attrs)
        if self.name.startswith("engram-"):
            attrs = {k: getattr(base, k) for k in dir(base) if k.startswith("st_")}
            attrs["st_size"] = 210 * GIB
            return SimpleNamespace(**attrs)
        return base

    monkeypatch.setattr(Path, "stat", sized_stat)
    planning.check_export_disk_space(tp4_plan, model_id, core_cache=cache_dir, topologies=topologies)


def test_multihost_peer_enospc_fails_with_peer_rank_and_host_identity(tmp_path):
    recipe = _deepseek_like_recipe()
    gpus = tuple(_gpu(i) for i in range(4))
    tp4_plan = planning.plan(recipe, recipe.checkpoints[0], gpus, selected=(0, 1, 2, 3))

    def host_topology(name, cache_free_gib, *, stage_free_gib=None, split_stage=False):
        stage_dev = 20 if split_stage else 10
        stage_free = cache_free_gib if stage_free_gib is None else stage_free_gib
        return {
            "host": name,
            "boot_id": f"boot-{name}",
            "storage": {
                "core_cache": {"path": f"/var/{name}/cores", "device": 10, "free_bytes": cache_free_gib * GIB},
                "export_staging": {
                    "path": f"/scratch/{name}/stage" if split_stage else f"/var/{name}/cores",
                    "device": stage_dev,
                    "free_bytes": int(stage_free * GIB),
                },
            },
        }

    # Peer rank 2 lacks space for its 75 GiB rank weights + 2 GiB executables/staging (has 42 GiB free).
    topologies = (
        host_topology("spark-0", 500),
        host_topology("spark-1", 200),
        host_topology("spark-2", 42),
        host_topology("spark-3", 200),
    )
    with pytest.raises(UsageError) as peer_err:
        planning.check_export_disk_space(
            tp4_plan,
            recipe.checkpoints[0].repository,
            core_cache=tmp_path / "cores",
            topologies=topologies,
        )
    text = str(peer_err.value)
    assert "rank 2 (spark-2)" in text
    assert "/var/spark-2/cores" in text
    assert "42.0 GiB available" in text
    assert "77.0 GiB" in text
    assert "--reuse-weights" in text

    # With --reuse-weights, peer rank 2 only needs 2.0 GiB and passes.
    planning.check_export_disk_space(
        tp4_plan,
        recipe.checkpoints[0].repository,
        core_cache=tmp_path / "cores",
        reuse_weights=True,
        topologies=topologies,
    )

    # Peer rank 3 split export_staging with 0.3 GiB (< 1.0 GiB) fails even under --reuse-weights.
    split_topologies = (
        host_topology("spark-0", 500),
        host_topology("spark-1", 200),
        host_topology("spark-2", 200),
        host_topology("spark-3", 200, stage_free_gib=0.3, split_stage=True),
    )
    with pytest.raises(UsageError) as split_peer_err:
        planning.check_export_disk_space(
            tp4_plan,
            recipe.checkpoints[0].repository,
            core_cache=tmp_path / "cores",
            reuse_weights=True,
            topologies=split_topologies,
        )
    assert "rank 3 (spark-3)" in str(split_peer_err.value)
    assert "/scratch/spark-3/stage" in str(split_peer_err.value)
    assert "0.3 GiB available" in str(split_peer_err.value)
    assert "1.0 GiB" in str(split_peer_err.value)


def test_cli_prepare_and_export_hosts_abort_before_gpu_preparation(tmp_path, monkeypatch):
    recipe = _deepseek_like_recipe()
    model = ResolvedModel(recipe, recipe.checkpoints[0])
    cfg_path = tmp_path / "config.toml"
    cache_dir = tmp_path / "cores"
    cfg_path.write_text(f'[paths]\ncore_cache = "{cache_dir.as_posix()}"\nhub_cache = "{(tmp_path / "hub").as_posix()}"\n')
    configuration = Configuration.load(cfg_path)
    host = preparation.Host(tuple(_gpu(i) for i in range(4)), None)

    monkeypatch.setattr(serve.Configuration, "load", lambda _path: configuration)
    monkeypatch.setattr(serve, "_publisher", lambda *_a, **_kw: object())
    monkeypatch.setattr(serve, "resolve_model", lambda _cfg, _arg: model)
    monkeypatch.setattr(serve.Host, "discover", lambda: host)
    monkeypatch.setattr(ResolvedModel, "cached", lambda self, _hub: True)

    # 1. Single-host _prepare rejects 119 GiB free before fetch or prepare_recipe.
    _mock_filesystems(monkeypatch, {tmp_path: (401, 119 * GIB)})
    local_args = parse_args(
        ["export", "fixture-300b", "--config", str(cfg_path), "--gpus", "0,1,2,3"],
        recipes=(recipe,),
    )
    with pytest.raises(UsageError, match="119.0 GiB available"):
        serve._prepare(local_args)

    # 2. Multi-host _export_hosts rejects Rank 0 ENOSPC before opening HostRendezvous.
    import sys
    import types

    try:
        import ignition.preparation.host_ranks as host_ranks_mod
    except ImportError:
        host_ranks_mod = types.ModuleType("ignition.preparation.host_ranks")
        host_ranks_mod.HostRendezvous = None
        host_ranks_mod.local_topology = None
        host_ranks_mod.source_identity = None
        host_ranks_mod.topology_gpu = lambda topology, index: planning.Gpu(
            index,
            topology["gpu"]["name"],
            topology["gpu"]["uuid"],
            topology["gpu"]["total_bytes"],
            topology["gpu"]["free_bytes"],
            tuple(topology["gpu"]["capability"]),
        )
        monkeypatch.setitem(sys.modules, "ignition.preparation.host_ranks", host_ranks_mod)

    rendezvous_opened = []
    monkeypatch.setattr(
        host_ranks_mod,
        "HostRendezvous",
        lambda *a, **kw: rendezvous_opened.append(True),
    )
    mh_args = parse_args(
        ["export", "fixture-300b", "--config", str(cfg_path), "--ranks", "4", "--rendezvous", "10.0.0.1:29630"],
        recipes=(recipe,),
    )
    with pytest.raises(UsageError, match="119.0 GiB available"):
        serve._export_hosts(mh_args)
    assert rendezvous_opened == []

    # 3. Multi-host _export_hosts with ample Rank 0 space (500 GiB) accepts peers, catches Peer Rank 1 ENOSPC,
    # and calls rendezvous.abort() before recipe.prepare runs.
    _mock_filesystems(monkeypatch, {tmp_path: (401, 500 * GIB)})
    aborted = []

    class FakeRendezvous:
        def __init__(self, address, world, *, identity, topology, **kwargs):
            self.world = world
            self.topology = topology

        def accept(self, *, report=None):
            def peer(rank, free_gib):
                g = _gpu(0)
                return {
                    "host": f"spark-{rank}",
                    "boot_id": f"boot-{rank}",
                    "gpu": {
                        "name": g.name,
                        "uuid": f"GPU-PEER-{rank}",
                        "total_bytes": g.total_bytes,
                        "free_bytes": g.free_bytes,
                        "capability": list(g.capability),
                    },
                    "rdma": [],
                    "storage": {
                        "core_cache": {"path": f"/var/spark-{rank}/cores", "device": 10, "free_bytes": free_gib * GIB},
                        "export_staging": {"path": f"/var/spark-{rank}/cores", "device": 10, "free_bytes": free_gib * GIB},
                    },
                }

            return (self.topology, peer(1, 30), peer(2, 200), peer(3, 200))

        def abort(self):
            aborted.append(True)

    monkeypatch.setattr(host_ranks_mod, "HostRendezvous", FakeRendezvous)
    monkeypatch.setattr(host_ranks_mod, "source_identity", lambda: "src-id")
    monkeypatch.setattr(
        host_ranks_mod,
        "local_topology",
        lambda gpu, *, storage=None: {
            "host": "spark-0",
            "boot_id": "boot-0",
            "gpu": {
                "name": gpu.name,
                "uuid": gpu.uuid,
                "total_bytes": gpu.total_bytes,
                "free_bytes": gpu.free_bytes,
                "capability": list(gpu.capability),
            },
            "rdma": [],
            "storage": storage,
        },
    )
    with pytest.raises(UsageError, match=r"rank 1 \(spark-1\).*30\.0 GiB available"):
        serve._export_hosts(mh_args)
    assert aborted == [True]
