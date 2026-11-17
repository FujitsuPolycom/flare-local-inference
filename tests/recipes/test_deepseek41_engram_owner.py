# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU eligibility and constructor guards for the owner-row Engram handoff.

Meta tensors keep the full declaration geometry without allocating model buffers.
These tests do not execute lookup, projection, readiness, or CUDA collectives.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def declaration(monkeypatch):
    from b12x.comm.roce import _proxy
    from ignition.components import collectives
    from ignition.components.paged_batch import Metadata
    from ignition.recipes.deepseek41.decoder import RowShard
    from ignition.recipes.deepseek41.engram import EngramScratch

    # The transport predicate is tested separately by the shared collective suite.
    # This fixture exercises the recipe's actual shape/stride/precision guards.
    monkeypatch.setattr(_proxy, "ABI_VERSION", 14)
    monkeypatch.setattr(collectives, "_roce", lambda runtime: True)
    runtime = SimpleNamespace(world_size=4, pipe_bytes=80 << 20,
        should_reduce_scatter=lambda value: True, should_gather_shards=lambda value: True)
    metadata = Metadata.allocate(8192, 1, 96, "meta")
    scratch = SimpleNamespace(
        engram_residual=torch.empty((8192, 4, 5120), dtype=torch.bfloat16, device="meta"),
        engram=EngramScratch.allocate(8192, "meta", world=4))
    return SimpleNamespace(owner=RowShard.allocate(8192, 4, 0, "meta"),
        metadata=metadata, scratch=scratch,
        tables=SimpleNamespace(shard=SimpleNamespace(world_size=4)), runtime=runtime)


def fits(value):
    from ignition.recipes.deepseek41.engram import owner_input_fits
    return owner_input_fits(value.owner, value.metadata, value.scratch, value.tables, value.runtime)


@pytest.mark.parametrize("rank", range(4))
def test_owner_geometry_accepts_every_tp4_rank(declaration, rank):
    value = declaration
    value.owner = replace(value.owner, rank=rank, start=rank * 2048)
    assert fits(value)


@pytest.mark.parametrize("change", (
    "missing_owner", "missing_runtime", "wrong_world", "wrong_table_world", "wrong_pipe",
    "wrong_rank", "wrong_start", "wrong_rows", "short_index", "index_dtype",
    "source_dtype", "source_stride", "projection_shape", "projection_dtype", "projection_stride",
    "full_output_shape", "multiple_requests", "smaller_capacity"))
def test_owner_geometry_rejects_incompatible_declarations(declaration, change):
    from ignition.components.paged_batch import Metadata
    value = declaration
    if change == "missing_owner": value.owner = None
    elif change == "missing_runtime": value.runtime = None
    elif change == "wrong_world": value.runtime.world_size = 2
    elif change == "wrong_table_world": value.tables.shard.world_size = 2
    elif change == "wrong_pipe": value.runtime.pipe_bytes = 64 << 20
    elif change == "wrong_rank": value.owner = replace(value.owner, rank=4)
    elif change == "wrong_start": value.owner = replace(value.owner, start=1)
    elif change == "wrong_rows": value.owner = replace(value.owner, rows=1024)
    elif change == "short_index": value.owner = replace(value.owner, rows_index=value.owner.rows_index[:2047])
    elif change == "index_dtype": value.owner = replace(value.owner, rows_index=value.owner.rows_index.int())
    elif change == "source_dtype": value.owner = replace(value.owner, residual=value.owner.residual.float())
    elif change == "source_stride":
        source = torch.empty((2048, 4, 5121), dtype=torch.bfloat16, device="meta")[:, :, :5120]
        value.owner = replace(value.owner, residual=source)
    elif change == "projection_shape":
        value.scratch.engram = replace(value.scratch.engram, partial=value.scratch.engram.partial[:, :25592])
    elif change == "projection_dtype":
        value.scratch.engram = replace(value.scratch.engram, partial=value.scratch.engram.partial.float())
    elif change == "projection_stride":
        partial = torch.empty((8192, 25608), dtype=torch.bfloat16, device="meta")[:, :25600]
        value.scratch.engram = replace(value.scratch.engram, partial=partial)
    elif change == "full_output_shape": value.scratch.engram_residual = value.scratch.engram_residual[:4096]
    elif change == "multiple_requests": value.metadata = Metadata.allocate(8192, 2, 96, "meta")
    elif change == "smaller_capacity": value.metadata = Metadata.allocate(4096, 1, 96, "meta")
    assert not fits(value)


@pytest.mark.parametrize("owner_rows,output_gather", ((object(), None), (None, object())))
def test_constructor_requires_owner_rows_and_output_gather_together(owner_rows, output_gather):
    from ignition.recipes.deepseek41.engram import PreparedEngram
    with pytest.raises(ValueError, match="owner-row Engram requires its full-output gather"):
        PreparedEngram(1, None, None, None, None, None, None, scratch=None,
            session=None, name="invalid-owner-pair", workspace=None,
            owner_rows=owner_rows, output_gather=output_gather)


@pytest.mark.parametrize("scored", (False, True))
def test_decoder_moves_only_the_eligible_predecessor_residual_gather(monkeypatch, declaration, scored):
    from dataclasses import dataclass
    from ignition.recipes.deepseek41 import decoder

    class Leaf:
        def __init__(self, *args, **kwargs):
            pass
        def functional(self, **kwargs):
            return self

    class Layer:
        def __init__(self, index, weights, tables, scratch, metadata, pool, rotary, **options):
            self.options = options
            self.row_shard = options["row_shard"]
            self.post = None if options["defer_post"] else object()
            gather = object() if self.row_shard is not None and self.post is not None else None
            self.residual_gather = self.original_gather = gather
            self.mix_gather = gather

    @dataclass
    class LayerWeights:
        engram: object
        attention: dict

    for name in ("PreparedTableRequests", "PreparedEmbedding", "PreparedDecoderBoundary", "AttentionIndices"):
        monkeypatch.setattr(decoder, name, Leaf)
    monkeypatch.setattr(decoder, "PreparedLayer", Layer)
    monkeypatch.setattr(decoder, "Rotary", lambda *args: object())
    monkeypatch.setattr(decoder, "register_finalizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder.PreparedDecoder, "_prepare_head", lambda *args, **kwargs: None)
    scratch = decoder.Scratch.allocate(8192, 1, 4, "meta", row_shard=SimpleNamespace(rank=0), attention_shard=True)
    config = SimpleNamespace(text={"engram_layer_ids": [1, 14], "candidate_source_layer_id": 14,
        "rms_norm_eps": 1e-6, "compress_ratios": [2] * 20 + [1] * 20}, layers=40,
        decoder_start=20, sources=(), draft=SimpleNamespace(target_layers=(33, 34, 35)))
    config.prefill_stop = lambda scoring: 39 if scoring else 19
    config.shardable_attention_layers = lambda stop, aux: tuple(
        i for i in range(stop) if i not in (1, 14) and i not in aux)
    config.index_source = lambda index: None
    weights = SimpleNamespace(config=config,
        layers=[LayerWeights({} if i in (1, 14) else None, {}) for i in range(40)],
        draft=object(), embedding=torch.empty((129280, 5120), dtype=torch.bfloat16, device="meta"),
        head=None, shard=SimpleNamespace(world_size=4))
    scoring = SimpleNamespace(prepare=lambda *args, **kwargs: object()) if scored else None
    model = decoder.PreparedDecoder(weights, declaration.tables, scratch, declaration.metadata,
        SimpleNamespace(config=config), mode="prefill", runtime=declaration.runtime,
        vocabulary_runtime=declaration.runtime, coordinator=None, session=None, name="decoder",
        workspace=SimpleNamespace(overlap=False), row_shard_attention=True, scoring=scoring)
    for index in (1, 14):
        previous = model.layers[index - 1]
        handoff = model.layers[index].options.get("engram_input")
        if scored:
            assert handoff is None
        else:
            assert handoff == (previous.row_shard, previous.original_gather)
            assert previous.residual_gather is None
            assert previous.mix_gather is previous.original_gather
        assert model.layers[index].options["row_shard"] is None
