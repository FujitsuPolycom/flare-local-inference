# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DFlash2 proposal scratch for attention widths distinct from model hidden widths."""

from dataclasses import replace
from types import SimpleNamespace
import copy

import pytest
import torch

from ignition.components.dflash2 import DraftScratch, _draft_templates
from ignition.components.dflash2_geometry import DraftAttentionGeometry
from ignition.components.parallel import TensorParallelShard


@pytest.mark.parametrize('world', [1, 2, 4, 8])
def test_non_square_attention_scratch_and_proposal_views(world):
    config = SimpleNamespace(block_size=8,
        text=dict(hidden_size=5120, intermediate_size=12288, vocab_size=128),
        attention=DraftAttentionGeometry(32, 8, 128, 128, 128, 10000000., 2048))
    shard = TensorParallelShard(0, world)
    scratch = DraftScratch.allocate(config, shard, 32, 'meta')
    templates = _draft_templates(config, shard, scratch, 'meta')
    assert scratch.logit_shape == (32, 128)
    assert templates.embedding.shape == (32, 5120)
    assert templates.local_embedding.shape == (32, 5120 // world)
    assert templates.query.shape == templates.attended.shape == (32, 32 // world, 128)
    assert templates.key.shape == templates.value.shape == (32, 8 // world, 128)
    assert templates.qkv.shape == (32, 6144 // world)
    assert templates.partial.shape == templates.output.shape == (32, 5120)
    assert templates.window_key.shape == (68, 128, 8 // world, 128)
    assert templates.window_value.shape == (68, 128, 8 // world, 128)
    assert templates.local_logits.shape == (32, 128 if world == 1 else 64)
    assert templates.local_logits.dtype == torch.bfloat16
    view = scratch.rows(8, 8)
    view_templates = _draft_templates(config, shard, view, 'meta')
    assert view.logit_shape == (8, 128)
    assert view_templates.query.shape == (8, 32 // world, 128)
    assert view_templates.embedding.shape == (8, 5120)
    assert view_templates.window_key.shape == (17, 128, 8 // world, 128)
    assert view_templates.convolution["before"].shape == (8, 5120)
    assert scratch.scratch_tensors() == view.scratch_tensors() == ()
    with pytest.raises(ValueError, match='complete proposal blocks'):
        scratch.rows(7, 7)


@pytest.mark.parametrize("width", [8, 13, 16])
def test_effective_block_preserves_checkpoint_and_scratch_fields(width):
    config = SimpleNamespace(block_size=8,
        text=dict(hidden_size=5120, intermediate_size=17408, vocab_size=128),
        attention=DraftAttentionGeometry(32, 8, 128, 128, 128, 10000000., 2048))
    shard = TensorParallelShard(0, 1)
    scratch = DraftScratch.allocate(config, shard, width, "meta", block_size=width)
    view = scratch.rows(width, width)
    templates = _draft_templates(config, shard, view, "meta")
    assert config.block_size == 8
    assert view.block_size == width
    assert templates.query.shape == (width, 32, 128)
    assert templates.window_key.shape[0] == 17
    assert templates.mlp.intermediate.shape == (width, 17408)
    assert templates.local_logits.shape == (width, 128)
    assert templates.gathered_logits.dtype == torch.bfloat16
    assert view.scratch_tensors() == ()


@pytest.mark.parametrize("model", ["dense", "flash"])
@pytest.mark.parametrize("world", [1, 2, 4])
def test_checkpoint_geometry_drives_shared_context_storage(model, world):
    from ignition.components.dflash2_geometry import DFlash2Config, draft_context_specs
    from ignition.recipes.qwen38.draft_config import EXPECTED
    from ignition.recipes.qwen38_dense.draft_config import PROFILE

    profile = EXPECTED if model == "flash" else PROFILE
    text = dict(model_type="qwen3", architectures=["DFlash2DraftModel"], hidden_act="silu",
        dtype="bfloat16", attention_bias=False, attention_dropout=0., tie_word_embeddings=False,
        use_sliding_window=True, is_causal=False, rms_norm_eps=1e-6, sliding_window=2048,
        layer_types=["sliding_attention"] * 5) | copy.deepcopy(profile)
    hidden, layers = (2560, 48) if model == "flash" else (5120, 64)
    config = DFlash2Config.from_dict(text, target_hidden=hidden, target_vocabulary=248320,
        target_layers=layers, expected=profile)
    shard = TensorParallelShard(0, world)
    config.validate_shard(shard)
    regions = dict((name, (shape, dtype)) for name, shape, dtype in draft_context_specs(config, shard))
    assert set(regions) == {f"draft{i}.{part}" for i in range(5) for part in ("key", "value")}
    assert set(regions.values()) == {((2048, 8 // world, 128), torch.float8_e4m3fn)}
    scratch = DraftScratch.allocate(config, shard, 8, "meta")
    templates = _draft_templates(config, shard, scratch, "meta")
    assert scratch.hidden_size == hidden
    assert templates.embedding.shape == (8, hidden)
    assert templates.query.shape == (8, 32 // world, 128)
    assert templates.local_embedding.shape == (8, hidden // world)
    assert templates.qkv.shape == (8, 6144 // world)
    with pytest.raises(ValueError, match="does not match the target"):
        DFlash2Config.from_dict(text, target_hidden=hidden, target_vocabulary=248320,
            target_layers=layers + 1, expected=profile)


KEY, VALUE, WINDOW = 8, 4, 2048


def _context(compact):
    """A draft context whose projections are identities and whose writer records its inputs."""
    from ignition.components.dflash2 import PreparedContext

    written = []
    context = object.__new__(PreparedContext)
    context.compact, context.window = compact, WINDOW
    context.weights = SimpleNamespace(shard=None, config=SimpleNamespace(draft=SimpleNamespace(
        attention=SimpleNamespace(local=lambda shard: (1, 1), key_dim=KEY, value_dim=VALUE))))
    context.project = context.gather = context.norm = context.project_kv = lambda value: value
    context.inverse = 1.0 / (10000 ** (torch.arange(0, KEY, 2, dtype=torch.float32) / KEY))
    context.layers = [(lambda value: value, lambda key, value, slots: written.append((key, value, slots)))]
    return context, written


def _chunk_metadata(chunk, three_axis):
    from ignition.components.paged_batch import Metadata

    metadata = Metadata.allocate(chunk, 1, 1, "cpu")
    metadata.set(torch.zeros(1, dtype=torch.int64), torch.full((1,), chunk, dtype=torch.int32),
        torch.ones(1, dtype=torch.int64), torch.zeros((1, 1), dtype=torch.int32),
        torch.arange(chunk, dtype=torch.int64).reshape(1, chunk), torch.ones(1, dtype=torch.int32))
    if three_axis:
        positions = metadata.positions
        return replace(metadata, rotary_positions=torch.stack(
            (positions + 7, positions + 100, positions + 200), dim=1))
    return metadata


@pytest.mark.parametrize("chunk", [1024, 2560, 4096])
@pytest.mark.parametrize("metadata_type", ["paged", "plain", "three_axis"])
def test_context_rotates_and_places_retained_tokens(chunk, metadata_type):
    """Non-compact chunks write every token; compact chunks keep the trailing window.

    Each written row rotates at its own position: the token index for plain or
    paged metadata (even when a recipe-specific ``rope_positions`` attribute is
    present) and the temporal column when ``rotary_positions`` is supplied.
    """
    from ignition.components.dflash2 import _rotary

    compact = chunk > WINDOW
    rows = WINDOW if compact else chunk
    three_axis = metadata_type == "three_axis"
    metadata = _chunk_metadata(chunk, three_axis)
    if metadata_type == "plain":
        metadata = SimpleNamespace(positions=metadata.positions, starts=metadata.starts,
            requests=metadata.requests, state_rows=metadata.state_rows,
            query_start_loc=metadata.query_start_loc, rope_positions="recipe-specific")
    counts = torch.tensor([chunk], dtype=torch.int32)
    auxiliary = torch.randn((rows, KEY + VALUE), generator=torch.Generator().manual_seed(2)).bfloat16()
    context, written = _context(compact)
    context(auxiliary, counts, metadata)
    (key, value, slots), = written
    retained = torch.arange(chunk - rows, chunk)
    assert slots.tolist() == (WINDOW + retained.remainder(WINDOW)).tolist()
    source = auxiliary[:, :KEY].view(rows, 1, KEY)
    assert torch.equal(key, _rotary(source, retained + 7 if three_axis else retained, context.inverse))
    if compact:
        assert not torch.equal(key, _rotary(source, torch.arange(rows), context.inverse))
    assert torch.equal(value, auxiliary[:, KEY:].view(rows, 1, VALUE))
