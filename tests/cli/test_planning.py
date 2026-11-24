# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Recipe-free planning, world-size validation, and RebuildManifest serialization."""

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest
from google.protobuf import json_format

from ignition import planning
from ignition.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults, Speculation
from ignition.errors import CapacityError, UsageError
from ignition.preparation import rebuild_manifest, runtime_core_pb2 as proto

GIB = 1 << 30
MIB = 1 << 20
DRAFTING = Speculation(draft_tokens=3)


def _recipe(name="fixture", *, device_gib=60, ranks=(1, 2, 4), speculation=DRAFTING,
            row_shard_prefill_world_sizes=(), joint_tuning=False):
    checkpoint = Checkpoint(
        alias=f"{name}-alias",
        repository=f"example/{name}",
        revision="ab12cd34" + "0" * 32,
        architecture="FixtureForCausalLM",
        stored_bytes=device_gib * GIB,
        device_bytes=device_gib * GIB,
        draft_bytes=2 * GIB if speculation is not None else 0,
    )
    return Recipe(
        name=name,
        served_model_name=name,
        checkpoints=(checkpoint,),
        context=262144,
        page_tokens=64,
        concurrency=4,
        max_concurrency=64,
        max_draft_tokens=7 if speculation is not None else 0,
        prefill_chunk=4096,
        ranks=MappingProxyType({size: RankCosts(100 * MIB // size, 2 * MIB // size, 4 * GIB) for size in ranks}),
        plain_ranks=MappingProxyType(
            {size: RankCosts(80 * MIB // size, 3 * MIB // (2 * size), 3 * GIB) for size in ranks}
            if speculation is not None else {}
        ),
        prepare=lambda *args, **kwargs: None,
        usage=lambda prepared: (),
        sampling_defaults=SamplingDefaults(temperature=1.0, top_p=1.0, top_k=-1),
        speculation=speculation,
        joint_tuning=joint_tuning,
        row_shard_prefill_world_sizes=row_shard_prefill_world_sizes,
    )


def _gpu(index, free_gib=95, capability=(12, 0)):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}",
                        96 * GIB, int(free_gib * GIB), capability)


def test_recipe_validates_row_shard_prefill_world_sizes():
    valid = _recipe(ranks=(1, 2, 4), row_shard_prefill_world_sizes=(2, 4))
    assert valid.row_shard_prefill_world_sizes == (2, 4)

    for bad_sizes in ((1,), (3,), (0,), (-2,), (True,)):
        with pytest.raises(ValueError, match="row_shard_prefill_world_sizes"):
            _recipe(ranks=(1, 2, 4), row_shard_prefill_world_sizes=bad_sizes)


def test_row_shard_prefill_validation_automatic_sizing_and_rebuild_manifest(monkeypatch):
    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    unsupported = _recipe("plain", ranks=(1, 2, 4), row_shard_prefill_world_sizes=())
    supported = _recipe("sharded", ranks=(1, 2, 4), row_shard_prefill_world_sizes=(2, 4), joint_tuning=True)
    tp34 = _recipe("tp34", ranks=(3, 4), row_shard_prefill_world_sizes=(4,))

    pair_gpus = (_gpu(0), _gpu(1))
    settings = dict(selected=(0, 1), concurrency=4, reserve_bytes=0, speculation=DRAFTING,
                    online_mxfp8_scope="all")
    ordinary = planning.plan(supported, supported.checkpoints[0], pair_gpus, **settings)
    sharded = planning.plan(supported, supported.checkpoints[0], pair_gpus, row_shard_prefill=True, **settings)
    assert not ordinary.row_shard_prefill and ordinary.online_mxfp8_scope == "all"
    assert sharded.row_shard_prefill and sharded.online_mxfp8_scope == "all"
    assert sharded.to_dict()["row_shard_prefill"] is True
    assert replace(sharded, row_shard_prefill=False) == ordinary

    fitted = planning.fit(supported, supported.checkpoints[0], pair_gpus, concurrency=4,
                          reserve_bytes=0, speculation=DRAFTING, online_mxfp8_scope="all",
                          row_shard_prefill=True)
    assert fitted.row_shard_prefill

    for candidate in (ordinary, sharded, fitted):
        manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), candidate, compiled_forwards=True)
        assert ("--row-shard-prefill" in manifest["reproduction_args"]) == candidate.row_shard_prefill
        encoded = json_format.ParseDict(manifest, proto.RebuildManifest())
        decoded = proto.RebuildManifest.FromString(encoded.SerializeToString())
        assert decoded.row_shard_prefill == candidate.row_shard_prefill

    # Unsupported recipe is rejected explicitly by alias.
    with pytest.raises(UsageError, match="--row-shard-prefill is not supported by plain-alias"):
        planning.plan(unsupported, unsupported.checkpoints[0], pair_gpus, row_shard_prefill=True, **settings)

    # TP1 is rejected on both TP1-capable and multi-GPU-only recipes.
    with pytest.raises(UsageError, match="--row-shard-prefill requires tensor parallelism"):
        planning.plan(supported, supported.checkpoints[0], (_gpu(0),), selected=(0,),
                      concurrency=4, reserve_bytes=0, speculation=DRAFTING, row_shard_prefill=True)
    with pytest.raises(UsageError, match="--row-shard-prefill requires tensor parallelism"):
        planning.plan(tp34, tp34.checkpoints[0], (_gpu(0), _gpu(1), _gpu(2), _gpu(3)), selected=(0,),
                      concurrency=4, reserve_bytes=0, speculation=DRAFTING, row_shard_prefill=True)

    # Unsupported TP3 is rejected explicitly while automatic planning skips TP3 and selects TP4.
    four_gpus = tuple(_gpu(i) for i in range(4))
    with pytest.raises(UsageError, match="--row-shard-prefill is not supported at TP3 for tp34-alias; supported tensor-parallel sizes: 4"):
        planning.plan(tp34, tp34.checkpoints[0], four_gpus[:3], selected=(0, 1, 2),
                      concurrency=4, reserve_bytes=0, speculation=DRAFTING, row_shard_prefill=True)
    auto_tp4 = planning.plan(tp34, tp34.checkpoints[0], four_gpus, selected=None,
                             concurrency=4, reserve_bytes=0, speculation=DRAFTING, row_shard_prefill=True)
    assert auto_tp4.world_size == 4 and auto_tp4.row_shard_prefill
    with pytest.raises(CapacityError, match="tp34-alias needs 4"):
        planning.plan(tp34, tp34.checkpoints[0], four_gpus[:3], selected=None,
                      concurrency=4, reserve_bytes=0, speculation=DRAFTING, row_shard_prefill=True)

    # SM103 and --joint-tuning are rejected.
    with pytest.raises(UsageError, match="SM103"):
        planning.plan(supported, supported.checkpoints[0], (_gpu(0, 95, (10, 3)), _gpu(1, 95, (10, 3))),
                      selected=(0, 1), concurrency=4, reserve_bytes=0, speculation=DRAFTING,
                      row_shard_prefill=True)
    with pytest.raises(UsageError, match="--row-shard-prefill cannot be combined with --joint-tuning"):
        replace(sharded, joint_tuning=True)
