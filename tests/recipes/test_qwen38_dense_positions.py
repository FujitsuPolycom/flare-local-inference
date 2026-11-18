# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen dense three-axis rotary positions, their offset, and its transfers.

The reference is ``Qwen3_5TextRotaryEmbedding`` from the checkpoint's
Transformers implementation. Prompt positions follow the served frontend: text
tokens advance every axis together, image tokens occupy a rectangular grid whose
temporal position is the run's start, and text after an image continues from the
largest axis position plus one.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import (BatchInputs, Metadata, RotaryBatch, RotaryBatchInputs,
    gather_rotary_offset, prepare_batch, rotary_positions)
from ignition.recipes.qwen38_dense.config import ModelConfig, rotary_axes
from ignition.recipes.qwen38_dense.decoder import rotary_embeddings

HEAD_DIM = 256
PAIRS = 32
ROTARY = dict(rope_type="default", rope_theta=1e7, partial_rotary_factor=.25,
              mrope_section=[11, 11, 10], mrope_interleaved=True)


def text_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(hidden_size=5120, num_attention_heads=40, num_key_value_heads=4,
        head_dim=HEAD_DIM, max_position_embeddings=262144, rope_parameters=dict(ROTARY))


def inverse_frequency():
    return 1.0 / (ROTARY["rope_theta"] ** (
        torch.arange(0, 64, 2, dtype=torch.float32) / 64))


def prompt_positions(segments):
    """Three-axis positions for text runs and merged image grids.

    ``segments`` holds ``("text", tokens)`` and ``("image", height, width)`` with
    the merged grid shape the served image layout reports.
    """
    positions, current = [], 0
    for segment in segments:
        if segment[0] == "text":
            for index in range(segment[1]):
                positions.append([current + index] * 3)
            current += segment[1]
        else:
            height, width = segment[1:]
            for y in range(height):
                for x in range(width):
                    positions.append([current, current + y, current + x])
            current += max(height, width)
    return torch.tensor(positions, dtype=torch.int64)


def prompt_offset(segments):
    """The offset the frontend implies: next axis position minus token count."""
    positions = prompt_positions(segments)
    return int(positions.max()) + 1 - positions.shape[0]


def reference_angles(positions):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    module = Qwen3_5TextRotaryEmbedding(text_config())
    hidden = torch.zeros(1, positions.shape[0], dtype=torch.bfloat16)
    cos, sin = module(hidden, positions.T[:, None])
    return cos[0], sin[0]


@pytest.mark.parametrize("segments", [
    [("text", 5)],
    [("text", 3), ("image", 6, 4), ("text", 4)],
    [("text", 2), ("image", 2, 2), ("text", 1), ("image", 8, 10), ("text", 3)],
    [("text", 4), ("image", 30, 4)],
    [("text", 4), ("image", 4, 30)],
    [("image", 6, 6), ("text", 5), ("image", 2, 2)],
])
def test_three_axis_angles_match_the_reference_model(segments):
    positions = prompt_positions(segments)
    expected_cos, expected_sin = reference_angles(positions)
    axes = rotary_axes(ROTARY, PAIRS)
    cos, sin = rotary_embeddings(positions, inverse_frequency(), axes, 3)
    assert torch.equal(cos, expected_cos) and torch.equal(sin, expected_sin)


def test_reference_rotation_of_queries_and_keys_matches():
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    positions = prompt_positions([("text", 3), ("image", 6, 4), ("text", 2)])
    reference = reference_angles(positions)
    actual = rotary_embeddings(positions, inverse_frequency(), rotary_axes(ROTARY, PAIRS), 3)
    generator = torch.Generator().manual_seed(7)
    query = torch.randn((positions.shape[0], 40, HEAD_DIM), generator=generator).bfloat16()
    key = torch.randn((positions.shape[0], 4, HEAD_DIM), generator=generator).bfloat16()
    expected_query, expected_key = apply_rotary_pos_emb(query, key, *reference)
    query_out, key_out = apply_rotary_pos_emb(query, key, *actual)
    assert torch.equal(query_out, expected_query) and torch.equal(key_out, expected_key)


@pytest.mark.parametrize("count", [1, 7, 128, 4096])
def test_text_only_angles_match_reference_and_one_axis(count):
    positions = prompt_positions([("text", count)])
    inverse, axes = inverse_frequency(), rotary_axes(ROTARY, PAIRS)
    expected_cos, expected_sin = reference_angles(positions)
    for candidate in (positions[:, 0], positions, positions[:, :1]):
        cos, sin = rotary_embeddings(candidate, inverse, axes, 3)
        assert torch.equal(cos, expected_cos) and torch.equal(sin, expected_sin)


def prefill_metadata(positions, start=0):
    """Prefill metadata as the batch builds it: prompt positions from the upload."""
    count = positions.shape[0]
    metadata = Metadata.allocate(count, 1, 1, "cpu")
    metadata.set(torch.tensor([start], dtype=torch.int64), torch.tensor([count], dtype=torch.int32),
        torch.ones(1, dtype=torch.int64), torch.zeros((1, 1), dtype=torch.int32),
        (positions[:, 0] - start).reshape(1, count), torch.ones(1, dtype=torch.int32))
    return replace(metadata, rotary_positions=positions.clone())


def test_zero_rotary_offset_reproduces_token_positions():
    positions = prompt_positions([("text", 16)])
    inverse, axes = inverse_frequency(), rotary_axes(ROTARY, PAIRS)
    metadata = Metadata.pack(torch.zeros(1, dtype=torch.int64), torch.tensor([16], dtype=torch.int32),
        torch.ones(1, dtype=torch.int64), torch.zeros((1, 1), dtype=torch.int32),
        positions[:, 0].reshape(1, 16), torch.ones(1, dtype=torch.int32), token_capacity=16,
        rotary_offset=torch.zeros(1, dtype=torch.int64))
    assert torch.equal(rotary_positions(metadata), metadata.positions[:, None])
    assert torch.equal(rotary_positions(metadata)[:, 0], metadata.positions)
    cos, sin = rotary_embeddings(rotary_positions(metadata), inverse, axes, 3)
    expected_cos, expected_sin = reference_angles(positions)
    assert torch.equal(cos, expected_cos) and torch.equal(sin, expected_sin)


def small_config(mtp=False):
    text = dict(hidden_size=64, vocab_size=128, num_hidden_layers=2, num_key_value_heads=2,
        head_dim=8, linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"])
    model = ModelConfig(text, {})
    if not mtp:
        return model
    from ignition.components.dflash2_geometry import DFlash2Config

    raw = dict(model_type="qwen3", architectures=["DFlash2DraftModel"], hidden_act="silu",
        dtype="bfloat16", attention_bias=False, attention_dropout=0., tie_word_embeddings=False,
        use_sliding_window=True, is_causal=False, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=128, num_target_layers=2, rms_norm_eps=1e-6, sliding_window=64,
        layer_types=["sliding_attention"], rope_parameters=dict(rope_type="default", rope_theta=10000),
        dflash_config=dict(block_size=8, conv_kernel_size=2, conv_group_size=16, selector_rank=16,
            selector_top_k=4, mask_token_id=1, target_layer_ids=[0, 1]))
    return replace(model, draft=DFlash2Config.from_dict(raw, target_hidden=64,
        target_vocabulary=128, target_layers=2, expected=raw))


def synthetic_pool():
    from ignition.recipes.qwen38_dense.pool import ContinuationPool

    return ContinuationPool.for_model(small_config(), 6, 3, "cpu", rotary_offset=True)


@pytest.mark.parametrize("segments", [
    [("text", 9)],
    [("text", 3), ("image", 6, 4), ("text", 4)],
    [("text", 5), ("image", 8, 10)],
    [("text", 2), ("image", 2, 30), ("text", 4)],
])
def test_prompt_offset_continues_from_the_largest_axis_position(monkeypatch, segments):
    positions = prompt_positions(segments)
    batch, _, pool, _, _ = request_batch(monkeypatch, "prefill", positions.shape[0], [1])
    metadata = prefill_metadata(positions)
    batch.commit_rotary(metadata)
    rows = metadata.state_rows
    expected = prompt_offset(segments)
    assert gather_rotary_offset(batch.rotary_offset, rows).tolist() == [expected]
    consumed = int(metadata.starts[0] + metadata.lengths[0])
    decode = Metadata.pack(torch.tensor([consumed], dtype=torch.int64),
        torch.ones(1, dtype=torch.int32), rows, torch.zeros((1, 1), dtype=torch.int32),
        torch.zeros((1, 1), dtype=torch.int64), torch.ones(1, dtype=torch.int32),
        token_capacity=1, rotary_offset=gather_rotary_offset(batch.rotary_offset, rows))
    assert int(rotary_positions(decode)[0, 0]) == expected + consumed


def test_second_prefill_chunk_keeps_the_prompt_offset(monkeypatch):
    segments = [("text", 3), ("image", 6, 4), ("text", 12)]
    positions = prompt_positions(segments)
    batch, _, pool, _, _ = request_batch(monkeypatch, "prefill", positions.shape[0], [1])
    chunks = (14, positions.shape[0])
    for begin, end in ((0, chunks[0]), (chunks[0], chunks[1])):
        metadata = prefill_metadata(positions[begin:end], start=begin)
        assert int(metadata.starts[0]) == begin
        batch.commit_rotary(metadata)
    rows = metadata.state_rows
    offset = gather_rotary_offset(batch.rotary_offset, rows)
    assert offset.item() == prompt_offset(segments)
    consumed = chunks[1]
    following = Metadata.pack(torch.tensor([consumed], dtype=torch.int64),
        torch.ones(1, dtype=torch.int32), rows, torch.zeros((1, 1), dtype=torch.int32),
        torch.zeros((1, 1), dtype=torch.int64), torch.ones(1, dtype=torch.int32),
        token_capacity=1, rotary_offset=offset)
    assert int(rotary_positions(following)[0, 0]) == consumed + prompt_offset(segments)


def test_pool_declares_one_int64_rotary_offset_per_continuation_row():
    from ignition.components.state import StateLayout
    from ignition.recipes.qwen38_dense.pool import ContinuationPool

    pool = synthetic_pool()
    region = pool.layout.region("rotary_offset")
    assert region.dtype == torch.int64 and region.shape == (1,)
    assert pool.region("rotary_offset").shape == (pool.backing.shape[0], 1)
    assert pool.layout.rotary_region == "rotary_offset"
    assert StateLayout.from_specs((("consumed_tokens", (1,), torch.int64),)).rotary_region is None
    assert all(name != "rotary_offset" for name, _ in pool.partial_pages())
    # A text-only preparation declares one position per token and no offset.
    text_only = ContinuationPool.for_model(small_config(), 6, 3, "cpu")
    assert "rotary_offset" not in {region.name for region in text_only.layout.regions}
    assert text_only.layout.row_bytes == pool.layout.row_bytes - 256


def test_three_axis_inputs_upload_prompt_positions_after_the_token_ids():
    inputs = BatchInputs.allocate(16, 2, "cpu", capacity=4)
    assert "prefill_positions" not in inputs.buffers()
    three = RotaryBatchInputs.allocate(16, 2, "cpu", capacity=4)
    assert three.prefill_positions.shape == (16, 3) and three.prefill_positions.dtype == torch.int64
    names = list(inputs.buffers())
    names.insert(names.index("token_ids") + 1, "prefill_positions")
    assert list(three.buffers()) == names
    assert all(torch.equal(three.buffers()[name], value) for name, value in inputs.buffers().items())


def test_token_position_metadata_has_no_axis_columns():
    starts = torch.tensor([0], dtype=torch.int64)
    lengths = torch.tensor([4], dtype=torch.int32)
    states = torch.ones(1, dtype=torch.int64)
    pages = torch.zeros((1, 1), dtype=torch.int32)
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    live = torch.ones(1, dtype=torch.int32)
    plain = Metadata.allocate(4, 1, 1, "cpu")
    plain.set(starts, lengths, states, pages, tokens, live)
    assert plain.rotary_positions is None
    assert torch.equal(rotary_positions(plain), plain.positions)
    packed = Metadata.pack(starts, lengths, states, pages, tokens, live, token_capacity=4)
    assert packed.rotary_positions is None
    assert torch.equal(rotary_positions(packed), packed.positions)


def test_pack_applies_per_row_rotary_offsets():
    starts = torch.tensor([3, 40], dtype=torch.int64)
    lengths = torch.tensor([2, 1], dtype=torch.int32)
    states = torch.tensor([1, 2], dtype=torch.int64)
    pages = torch.zeros((2, 1), dtype=torch.int32)
    tokens = torch.tensor([[1, 2], [3, 0]], dtype=torch.int64)
    live = torch.tensor([2], dtype=torch.int32)
    delta = torch.tensor([10, -5], dtype=torch.int64)
    packed = Metadata.pack(starts, lengths, states, pages, tokens, live, token_capacity=4,
        rotary_offset=delta)
    assert packed.positions.tolist() == [3, 4, 40, -1]
    assert packed.rotary_positions.tolist() == [[13], [14], [35], [-1]]


def test_draft_rotary_uses_the_temporal_axis():
    from ignition.components.dflash2 import _rotary

    generator = torch.Generator().manual_seed(3)
    positions = prompt_positions([("text", 2), ("image", 2, 3)])
    values = torch.randn((positions.shape[0], 2, 8), generator=generator).bfloat16()
    inverse = inverse_frequency()[:4]
    assert torch.equal(_rotary(values, positions, inverse), _rotary(values, positions[:, 0], inverse))
    assert torch.equal(_rotary(values, positions[:, :1], inverse), _rotary(values, positions[:, 0], inverse))
    assert not torch.equal(_rotary(values, positions, inverse), _rotary(values, positions[:, 2], inverse))


def test_rotary_axes_follow_the_three_sections():
    axes = rotary_axes(ROTARY, PAIRS)
    assert axes.tolist()[:6] == [0, 1, 2, 0, 1, 2]
    assert [int((axes == axis).sum()) for axis in range(3)] == [11, 11, 10]
    with pytest.raises(ValueError):
        rotary_axes(dict(ROTARY, mrope_section=[11, 11, 9]), PAIRS)
    with pytest.raises(ValueError):
        rotary_axes(dict(ROTARY, mrope_section=[11, 11]), PAIRS)


class FixedTokenState:
    """Replace the prepared token gather so batch wiring can run without kernels."""

    positions_seed: tuple = (10, 20, 30, 40)
    pending_seed: tuple = (7, 7, 7, 7)

    def __init__(self, requests, tokens, query_start_loc, slots, epochs, live, *, buffers=None):
        self.rows, self.tokens = slots.numel(), tokens
        self.pending_tokens = torch.tensor(self.pending_seed[:self.rows], dtype=torch.int64)
        self.positions = torch.tensor(self.positions_seed[:self.rows], dtype=torch.int64)
        self.active = torch.ones(self.rows, dtype=torch.int32)
        self.gathered = self.consumed = None

    def prepare(self):
        return self

    def functional(self):
        return self

    def gather(self):
        self.gathered = (self.pending_tokens, self.positions, None, self.active)
        return self.gathered

    def consume(self, tokens, query_start_loc):
        self.consumed = (tokens, query_start_loc)


def request_batch(monkeypatch, mode, tokens, state_rows):
    from ignition.components.sampling import RUNNING, DeviceRequestState

    import ignition.components.paged_batch as batch_module

    capacity = 4
    rows = 1 if mode == "prefill" else capacity

    class State(FixedTokenState):
        positions_seed = (0,) if mode == "prefill" else FixedTokenState.positions_seed

    monkeypatch.setattr(batch_module, "PreparedTokenState", State)
    requests = DeviceRequestState.allocate("cpu", vocab_size=64, capacity=capacity)
    requests.generation.fill_(3)
    requests.status.fill_(RUNNING)
    inputs = RotaryBatchInputs.allocate(tokens, 2, "cpu", capacity=capacity)
    inputs.row_slot_ids[:rows].copy_(torch.arange(rows))
    inputs.row_generations[:rows].fill_(3)
    inputs.live_rows.fill_(rows)
    inputs.request_state_rows[:rows, 0].copy_(torch.tensor(state_rows, dtype=torch.int64))
    inputs.page_tables.fill_(0)
    inputs.token_ids[:tokens].copy_(torch.arange(tokens) + 1)
    if mode == "prefill":
        inputs.query_lengths[0] = tokens
    pool = synthetic_pool()
    metadata = Metadata.allocate(tokens, rows, 2, "cpu")
    batch = prepare_batch(inputs, requests, metadata, mode=mode, shard=SimpleNamespace(world_size=1),
        pool=pool)
    assert type(batch) is RotaryBatch
    totals = torch.zeros((1, 2 * capacity), dtype=torch.bfloat16)
    present = torch.zeros(capacity, dtype=torch.bool)
    present[:rows] = True
    return batch, inputs, pool, totals, present


def test_decode_positions_add_each_request_offset(monkeypatch):
    batch, inputs, pool, totals, present = request_batch(monkeypatch, "decode", 4, [1, 2, 3, 4])
    deltas = torch.tensor([0, 5, -3, 11], dtype=torch.int64)
    pool.region("rotary_offset")[1:5, 0] = deltas
    metadata = batch.begin(totals, present=present)
    assert metadata.positions.tolist() == [10, 20, 30, 40]
    assert rotary_positions(metadata)[:, 0].tolist() == [10, 25, 27, 51]
    assert rotary_positions(metadata).shape == (4, 1)
    batch.finish(metadata)
    assert pool.region("rotary_offset")[1:5, 0].tolist() == deltas.tolist()


def test_prefill_positions_come_from_the_upload_and_commit_the_offset(monkeypatch):
    segments = [("text", 3), ("image", 4, 6), ("text", 2)]
    positions = prompt_positions(segments)
    count = positions.shape[0]
    batch, inputs, pool, totals, present = request_batch(monkeypatch, "prefill", count, [2])
    inputs.prefill_positions[:count].copy_(positions)
    metadata = batch.begin(totals, present=present)
    assert torch.equal(rotary_positions(metadata), positions)
    assert metadata.positions.tolist() == list(range(count))
    batch.finish(metadata)
    assert gather_rotary_offset(pool.region("rotary_offset"), torch.tensor([2])).item() == prompt_offset(segments)


def test_vision_state_validates_and_injects_image_features(monkeypatch):
    """Batch validation rejects invalid image bindings and VisionState injects valid features."""
    from ignition.components.vision_state import PrefillVisionInputs, VisionState

    batch, inputs, pool, totals, present = request_batch(monkeypatch, "prefill", 4, [2])
    prefill = PrefillVisionInputs.allocate(4, "cpu")
    vision_state = VisionState(inputs, prefill, batch.requests, image_token=7,
        features=8, hidden=4, image_slots=2)
    batch.validate_prefill = vision_state.validate_prefill
    inputs.token_ids[:4].copy_(torch.tensor([1, 7, 7, 2]))
    prefill.vision_token_indices.copy_(torch.tensor([-1, 0, 1, -1], dtype=torch.int32))
    prefill.vision_feature_slot.fill_(1)
    # Without published features for the request epoch, validation rejects slot 0.
    validation, slots = batch.validate()
    assert validation[0, :batch.requests.capacity].tolist() == [1, 0, 0, 0]
    assert slots.tolist() == [True, False, False, False]
    # Publish two features in epoch 3 for slot 0 and confirm validation and injection succeed.
    vision_state.generation[0] = 3
    vision_state.count[0] = 2
    vision_state.arena[1, 0].fill_(2.5)
    vision_state.arena[1, 1].fill_(4.0)
    validation, _ = batch.validate()
    assert validation[0, :batch.requests.capacity].tolist() == [0, 0, 0, 0]
    metadata = batch.begin(totals, present=present)
    embedding = torch.ones((4, 4), dtype=torch.bfloat16)
    vision_state.inject(metadata, embedding)
    assert embedding[0].tolist() == [1.0] * 4
    assert embedding[1].tolist() == [2.5] * 4
    assert embedding[2].tolist() == [4.0] * 4
    assert embedding[3].tolist() == [1.0] * 4


def test_draft_round_rotates_draft_and_target_rows_at_the_stored_offsets(monkeypatch):
    """A round takes each row's offset from its batch's decode metadata.

    The batch is built by its constructor from a pool that declares the region.
    The draft rows of a proposal and the target rows of its verification rotate at
    the token position plus the offset stored in the request's continuation row.
    """
    from ignition.components.parallel_draft import PreparedParallelDraftRound

    batch, inputs, pool, _, _ = request_batch(monkeypatch, "decode", 4, [1, 2, 3, 4])
    requests = batch.requests
    deltas = torch.tensor([0, 5, -3, 11], dtype=torch.int64)
    pool.region("rotary_offset")[1:5, 0] = deltas
    requests.consumed_tokens.copy_(torch.tensor(FixedTokenState.positions_seed))
    requests.pending_token.fill_(7)
    inputs.query_lengths.fill_(2)
    owner = SimpleNamespace(requests=requests, inputs=inputs, block_size=3, logit_offset=1,
        weights=SimpleNamespace(config=SimpleNamespace(draft=SimpleNamespace(noise_token=9))),
        generation=torch.zeros(4, dtype=torch.int64), offset=torch.zeros(4, dtype=torch.int64))

    class Sampler:
        has_source_logits = True
        layout = {"probabilities": ((4, 1, 64), torch.float32), "valid": ((4, 1), torch.bool)}

        def run(self, scratch, logits, *, source_logits, shortlist_ids, candidates, seed, offset, sampled):
            sampled.fill_(5)
            scratch["valid"].fill_(True)
            scratch["probabilities"].zero_()
            scratch["probabilities"][..., 5] = 1.
            return scratch["probabilities"]

    observed = []

    def draft(metadata):
        observed.append(metadata)
        return torch.zeros(12, 64), None

    class Decoder:
        def __call__(self, metadata):
            observed.append(metadata)
            raise StopIteration

    operation = object.__new__(PreparedParallelDraftRound)
    operation.owner, operation.batch, operation.width = owner, 4, 2
    operation.target_token_capacity = 8
    operation.local_metadata, operation.generate_proposals = True, True
    operation.vocabulary_parallel = False
    operation.operations = None, (Sampler(),), None, None
    operation.validator, operation.validation = batch, lambda values: values
    operation.draft, operation.decoder = draft, Decoder()
    operation.shortlisted_selector = operation.selector = operation.markov = None
    operation.full_proposal = operation.proposal_block = None
    proposal = operation.propose(probabilities=torch.empty(4, 1, 64))
    assert proposal.rotary_offsets.tolist() == deltas.tolist()
    assert proposal.before.tolist() == [10, 20, 30, 40]
    assert observed[0].rotary_positions[:, 0].tolist() == [10, 11, 12, 25, 26, 27, 27, 28, 29, 51, 52, 53]
    with pytest.raises(StopIteration):
        operation.verify(probabilities=torch.empty(4, 1, 64), proposal=proposal)
    assert observed[1].positions.tolist() == [10, 11, 20, 21, 30, 31, 40, 41]
    assert observed[1].rotary_positions[:, 0].tolist() == [10, 11, 25, 26, 27, 28, 51, 52]


def test_rotary_offset_uses_reserved_region_when_state_rows_grow(monkeypatch):
    """Dynamic STATE_ROWS growth places continuation rows past the initial backing extent."""
    from ignition.components.paged_batch import rotary_offset_region

    class GrownPool:
        def __init__(self, base, reserved_rows):
            self.layout = base.layout
            self._reserved = torch.zeros((reserved_rows, base.layout.row_bytes), dtype=torch.uint8)
            self.backing = self._reserved[:base.backing.shape[0]]

        def region(self, name):
            region = self.layout.region(name)
            return self.backing[:, region.offset:region.offset + region.nbytes].view(
                region.dtype).view(self.backing.shape[0], *region.shape)

        def reserved_region(self, name):
            region = self.layout.region(name)
            return self._reserved[:, region.offset:region.offset + region.nbytes].view(
                region.dtype).view(self._reserved.shape[0], *region.shape)

    base = synthetic_pool()
    grown = GrownPool(base, reserved_rows=base.backing.shape[0] + 8)
    assert rotary_offset_region(grown).shape[0] == base.backing.shape[0] + 8

    # Commit and read a rotary offset on a state row past the initial backing height.
    high_row = base.backing.shape[0] + 3
    segments = [("text", 3), ("image", 4, 6), ("text", 2)]
    positions = prompt_positions(segments)
    count = positions.shape[0]
    batch, inputs, _, totals, present = request_batch(monkeypatch, "prefill", count, [high_row])
    batch = prepare_batch(inputs, batch.requests, batch.metadata, mode="prefill",
        shard=SimpleNamespace(world_size=1), pool=grown)
    inputs.prefill_positions[:count].copy_(positions)
    metadata = batch.begin(totals, present=present)
    batch.finish(metadata)
    expected = prompt_offset(segments)
    assert grown.reserved_region("rotary_offset")[high_row, 0].item() == expected

    decode_batch, decode_inputs, _, decode_totals, decode_present = request_batch(
        monkeypatch, "decode", 4, [1, high_row, 2, 3])
    decode_batch = prepare_batch(decode_inputs, decode_batch.requests, decode_batch.metadata,
        mode="decode", shard=SimpleNamespace(world_size=1), pool=grown)
    decode_meta = decode_batch.begin(decode_totals, present=decode_present)
    assert rotary_positions(decode_meta)[1, 0].item() == 20 + expected
