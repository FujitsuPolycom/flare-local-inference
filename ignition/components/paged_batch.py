# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed request metadata for windowed and compressed-history attention."""

from dataclasses import dataclass, replace

import torch

from ignition.components.sampling import ERROR, RUNNING
from ignition.components.state import gather_rows, scatter_rows
from ignition.components.token_state import PreparedTokenState

PAGE_TOKENS = 128
ROTARY_OFFSET = "rope_delta"


def rotary_offset_region(pool):
    """Continuation region holding each row's rotary offset, or None for token indices.

    A pool declares the region by naming it in its state layout's ``rotary_region``.
    A layout that names none, and a storage object without a state layout, hold no
    offsets: their batches and draft rounds rotate at token indices.
    """
    declared = getattr(getattr(pool, "layout", None), "rotary_region", None)
    return None if declared is None else pool.region(declared)


def gather_rotary_offset(region, state_rows):
    """Read each request's rotary offset from its continuation row."""
    offset = region.new_empty((state_rows.numel(), 1))
    return gather_rows(region, state_rows, offset).view(-1)


def rotary_positions(metadata):
    """Per-token rotary positions in one column per declared axis.

    ``Metadata`` for a model with three-axis positions carries them explicitly;
    every other metadata object rotates with its token positions. Other request
    metadata can hold an attribute of the same name with another meaning, so
    only ``Metadata`` declares per-axis rotary columns here.
    """
    rope = metadata.rope_positions if isinstance(metadata, Metadata) else None
    return metadata.positions if rope is None else rope


def row_rotary_offsets(metadata):
    """Each row's rotary offset in the decode metadata of a `RotaryBatch`, or None.

    Decode metadata holds one token per active row at the row's token position plus
    its offset. Metadata without per-axis positions, and any other metadata object,
    carries no offsets.
    """
    if not isinstance(metadata, Metadata) or metadata.rope_positions is None:
        return None
    first = metadata.query_start_loc[:-1].long().clamp(max=metadata.positions.numel() - 1)
    return torch.where(metadata.lengths > 0, metadata.rope_positions[first, 0] - metadata.starts, 0)


@dataclass(frozen=True)
class BatchInputs:
    request_state_rows: torch.Tensor
    row_slot_ids: torch.Tensor
    row_generations: torch.Tensor
    live_rows: torch.Tensor
    query_lengths: torch.Tensor
    token_ids: torch.Tensor
    page_tables: torch.Tensor
    sample_enabled: torch.Tensor

    @classmethod
    def allocate(cls, tokens, blocks, device, *, capacity):
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros((capacity, 8), torch.int64),
            zeros(capacity) - 1,
            zeros(capacity, torch.int64),
            zeros(1),
            zeros(capacity),
            zeros(tokens, torch.int64),
            zeros((capacity, blocks)) - 1,
            zeros(capacity),
        )

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class RotaryBatchInputs(BatchInputs):
    """Uploads of a three-axis frontend: per-axis prompt positions for each prefill token."""
    prefill_positions: torch.Tensor

    @classmethod
    def allocate(cls, tokens, blocks, device, *, capacity):
        inputs = BatchInputs.allocate(tokens, blocks, device, capacity=capacity)
        return cls(**inputs.buffers(),
            prefill_positions=torch.zeros((tokens, 3), dtype=torch.int64, device=device))

    def buffers(self):
        """Upload buffers in graph input order: prompt positions follow the token ids."""
        buffers = {}
        for name, value in super().buffers().items():
            if name != "prefill_positions":
                buffers[name] = value
            if name == "token_ids":
                buffers["prefill_positions"] = self.prefill_positions
        return buffers


@dataclass(frozen=True)
class Metadata:
    token_ids: torch.Tensor
    sample_indices: torch.Tensor
    query_start_loc: torch.Tensor
    state_rows: torch.Tensor
    starts: torch.Tensor
    lengths: torch.Tensor
    requests: torch.Tensor
    positions: torch.Tensor
    page_table: torch.Tensor
    token_page_table: torch.Tensor
    live_counts: torch.Tensor
    max_query_tokens: int
    rope_positions: torch.Tensor | None = None

    @classmethod
    def allocate(cls, tokens, rows, blocks, device, *, samples=None, buffers=None, max_query_tokens=None,
                 position_dimensions=0):
        query_capacity = (tokens + rows - 1) // rows if max_query_tokens is None else max_query_tokens
        if type(query_capacity) is not int or query_capacity < 1:
            raise ValueError("per-request query capacity must be a positive integer")
        if position_dimensions not in (0, 3):
            raise ValueError("rotary position dimensions must be zero or three")
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device) if buffers is None else buffers.zeros(shape, dtype)

        pages = zeros((rows, blocks)).fill_(-1)
        per_token = pages.expand(tokens, -1) if rows == 1 else zeros((tokens, blocks)).fill_(-1)
        rope = zeros((tokens, 3), torch.int64).fill_(-1) if position_dimensions else None
        return cls(
            zeros(tokens, torch.int64),
            zeros(rows if samples is None else samples, torch.int64),
            zeros(rows + 1),
            zeros(rows, torch.int64),
            zeros(rows, torch.int64),
            zeros(rows),
            zeros(tokens, torch.int64),
            zeros(tokens, torch.int64).fill_(-1),
            pages,
            per_token,
            zeros(2),
            query_capacity,
            rope,
        )

    def set(self, starts, lengths, state_rows, pages, tokens, live_rows, *, rope_delta=None):
        rows, width = tokens.shape
        offsets = torch.arange(self.token_ids.numel(), device=tokens.device)
        ends = lengths.cumsum(0)
        begins = ends - lengths
        request = (offsets[:, None] >= ends[None, :]).sum(1).clamp(max=rows - 1)
        column = offsets - begins[request]
        live = offsets < ends[-1]
        self.query_start_loc[0].zero_()
        self.query_start_loc[1:].copy_(ends)
        self.lengths.copy_(lengths)
        self.starts.copy_(starts)
        self.state_rows.copy_(torch.where(lengths > 0, state_rows, 0))
        self.page_table.copy_(pages)
        if rows > 1:
            self.token_page_table.copy_(pages[request])
        self.live_counts[0:1].copy_(ends[-1:])
        self.live_counts[1:].copy_(live_rows.clamp(0, rows))
        self.requests.copy_(request)
        self.positions.copy_(torch.where(live, starts[request] + column, -1))
        indices = (request * width + column).clamp(0, tokens.numel() - 1)
        self.token_ids.copy_(torch.where(live, tokens.reshape(-1)[indices], 0))
        columns = torch.arange(self.sample_indices.numel() // rows, device=tokens.device)
        self.sample_indices.copy_(
            (begins[:, None] + columns).clamp(0, self.token_ids.numel() - 1).reshape(-1)
        )
        if self.rope_positions is None:
            if rope_delta is not None:
                raise ValueError("rotary offsets require metadata allocated with per-axis positions")
            return
        # Text occupies every axis at its token position plus the row's rotary offset. Without
        # `rope_delta` the offset is zero; a caller filling rows that hold a stored offset
        # passes it.
        rope = self.positions if rope_delta is None else self.positions + rope_delta[request]
        self.rope_positions.copy_(torch.where(live, rope, -1)[:, None].expand_as(self.rope_positions))

    @classmethod
    def pack(cls, starts, lengths, state_rows, pages, tokens, live_rows, *, token_capacity,
             sample_count=None, max_query_tokens=None, rope_delta=None):
        """Return packed metadata with forward-local destinations.

        ``rope_delta`` supplies each row's rotary offset from its token index,
        which three-axis models keep in continuation state.
        """
        rows, width = tokens.shape
        if rows < 1 or width < 1 or type(token_capacity) is not int or token_capacity < 1:
            raise ValueError("packed metadata requires positive row, token, and input capacities")
        if rope_delta is not None and tuple(rope_delta.shape) != (rows,):
            raise ValueError("packed rotary offsets require one value per row")
        sample_count = rows if sample_count is None else sample_count
        if type(sample_count) is not int or sample_count < 0 or sample_count % rows:
            raise ValueError("packed sample capacity must contain an equal number of columns per row")
        query_capacity = ((token_capacity + rows - 1) // rows
                          if max_query_tokens is None else max_query_tokens)
        if type(query_capacity) is not int or query_capacity < 1:
            raise ValueError("per-request query capacity must be a positive integer")
        offsets = torch.arange(token_capacity, device=tokens.device)
        ends = lengths.cumsum(0)
        begins = ends - lengths
        request = (offsets[:, None] >= ends[None, :]).sum(1).clamp(max=rows - 1)
        column = offsets - begins[request]
        live = offsets < ends[-1]
        indices = (request * width + column).clamp(0, tokens.numel() - 1)
        columns = torch.arange(sample_count // rows, device=tokens.device)
        packed_pages = pages.to(dtype=torch.int32, copy=True)
        packed = cls(
            torch.where(live, tokens.reshape(-1)[indices], 0).to(torch.int64),
            (begins[:, None] + columns).clamp(0, token_capacity - 1).reshape(-1).to(torch.int64),
            torch.cat((torch.zeros(1, dtype=torch.int32, device=tokens.device), ends.to(torch.int32))),
            torch.where(lengths > 0, state_rows, 0).to(torch.int64),
            starts.to(dtype=torch.int64, copy=True),
            lengths.to(dtype=torch.int32, copy=True),
            request,
            torch.where(live, starts[request] + column, -1).to(torch.int64),
            packed_pages,
            packed_pages.expand(token_capacity, -1) if rows == 1 else packed_pages[request],
            torch.cat((ends[-1:].to(torch.int32), live_rows.clamp(0, rows).reshape(1).to(torch.int32))),
            query_capacity,
        )
        if rope_delta is None:
            return packed
        positions = packed.positions
        return replace(packed, rope_positions=torch.where(
            positions >= 0, positions + rope_delta[request], -1)[:, None])


class PreparedBatch:
    def __init__(
        self,
        inputs,
        requests,
        metadata,
        *,
        mode,
        shard,
        pool,
        verification_width=1,
        target_token_capacity=None,
        input_vocabulary=None,
        validate_prefill=None,
        page_tokens=PAGE_TOKENS,
        buffers=None,
    ):
        self.inputs, self.requests, self.metadata, self.shard = inputs, requests, metadata, shard
        self.rows, self.tokens, self.mode = metadata.starts.numel(), metadata.token_ids.numel(), mode
        self.sample_count = metadata.sample_indices.numel()
        self.max_query_tokens = metadata.max_query_tokens
        if mode not in ("prefill", "decode") or mode == "prefill" and self.rows != 1:
            raise ValueError("Paged prefill requires one packed sequence")
        self.pool = pool
        self.page_tokens = page_tokens
        self.verification_width = verification_width
        self.target_token_capacity = self.rows * verification_width if target_token_capacity is None else target_token_capacity
        if not self.rows <= self.target_token_capacity <= self.rows * verification_width:
            raise ValueError("target token capacity must fit the request rows and verification width")
        self.input_vocabulary = requests.vocab_size if input_vocabulary is None else input_vocabulary
        self.validate_prefill = validate_prefill
        device, capacity = metadata.token_ids.device, requests.capacity
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device) if buffers is None else buffers.zeros(shape, dtype)
        self.validation_input = zeros((1, max(8, 2 * capacity)), torch.bfloat16)
        self.validation_sum = zeros(self.validation_input.shape, torch.bfloat16)
        self.token_state = PreparedTokenState(
            requests,
            metadata.token_ids,
            metadata.query_start_loc,
            inputs.row_slot_ids[: self.rows],
            inputs.row_generations[: self.rows],
            inputs.live_rows,
            buffers=buffers,
        )
        self.token_state.prepare()
        self.token_state.functional()

    def functional(self):
        """Return validation and packed metadata as local forward values."""
        if self.metadata is not None:
            self.metadata = self.validation_input = self.validation_sum = None
        return self

    def validate(self):
        b, r = self.inputs, self.requests
        device = r.generation.device
        row_ids = torch.arange(self.rows, device=device)
        slot_ids = torch.arange(r.capacity, device=device)
        block_ids = torch.arange(b.page_tables.shape[1], device=device)
        slots = b.row_slot_ids[: self.rows].long()
        safe = slots.clamp(0, r.capacity - 1)
        resident = (row_ids < b.live_rows) & (slots >= 0) & (slots < r.capacity)
        resident &= b.row_generations[: self.rows] == r.generation[safe]
        duplicate = ((slots[:, None] == slots[None, :]) & resident[:, None] & resident[None, :]).sum(1) > 1
        lengths = (
            b.query_lengths[: self.rows]
            if self.mode == "prefill" or self.verification_width > 1
            else torch.ones_like(safe)
        )
        position = r.consumed_tokens[safe]
        end = position + lengths
        capacity = self.tokens if self.mode == "prefill" else self.verification_width
        valid = ~duplicate & (b.live_rows <= self.rows) & (lengths > 0) & (lengths <= capacity)
        if self.mode == "decode":
            offered = torch.where(resident & (r.status[safe] == RUNNING), lengths, 0)
            valid &= offered.sum() <= self.target_token_capacity
        valid &= (position >= 0) & (end <= b.page_tables.shape[1] * self.page_tokens)
        pages = b.page_tables[safe]
        valid &= (
            ((pages >= 0) & (pages < self.pool.memory_capacity[1]))
            | (block_ids >= (end[:, None] + self.page_tokens - 1) // self.page_tokens)
        ).all(1)
        state = b.request_state_rows[safe, 0]
        valid &= (state > 0) & (state < self.pool.memory_capacity[0])
        pending, tokens = r.pending_token[safe], b.token_ids[: self.tokens]
        if self.mode == "prefill":
            offsets = torch.arange(self.tokens, device=device)
            invalid = ((tokens < 0) | (tokens >= self.input_vocabulary)) & (offsets < lengths[0])
            valid &= (pending < 0) & ~invalid.any()
            if self.validate_prefill is not None:
                valid &= self.validate_prefill(tokens, lengths)
        else:
            valid &= (pending >= 0) & (pending < r.vocab_size)
        present = resident & (r.status[safe] == RUNNING)
        bad = present & ~valid
        bad_slots = ((slot_ids[:, None] == slots[None, :]) & bad[None, :]).any(1)
        present_slots = ((slot_ids[:, None] == slots[None, :]) & present[None, :]).any(1)
        validation = torch.zeros((1, max(8, 2 * r.capacity)), dtype=torch.bfloat16, device=device)
        validation[0, :r.capacity].copy_(bad_slots)
        validation[0, r.capacity:2 * r.capacity].copy_(present_slots)
        return validation, present_slots

    def begin(self, totals, *, present):
        b, r, t = self.inputs, self.requests, self.token_state
        participants = totals[0, r.capacity : 2 * r.capacity]
        bad = present & (
            (totals[0, : r.capacity] > 0) | ((participants > 0) & (participants < self.shard.world_size))
        )
        r.status.copy_(torch.where(bad, ERROR, r.status))
        r.error_code.copy_(torch.where(bad, 32, r.error_code))
        pending_tokens, positions, _, active = t.gather()
        active = active.bool()
        slots = b.row_slot_ids[: self.rows].long().clamp(0, r.capacity - 1)
        lengths = (b.query_lengths[: self.rows] if self.mode == "prefill"
                   else torch.ones((self.rows,), dtype=torch.int32, device=r.generation.device))
        lengths = torch.where(active, lengths, 0)
        ends = lengths.cumsum(0, dtype=torch.int32)
        starts = ends - lengths
        offsets = torch.arange(self.tokens, device=r.generation.device)
        request = (offsets[:, None] >= ends[None, :]).sum(1).clamp(max=self.rows - 1)
        live = offsets < ends[-1]
        pages = torch.where(active[:, None], b.page_tables[slots], -1)
        tokens = b.token_ids[:self.tokens] if self.mode == "prefill" else pending_tokens[request]
        return Metadata(
            torch.where(live, tokens, 0),
            (ends - 1).clamp(min=0).to(torch.int64).expand(self.sample_count),
            torch.cat((torch.zeros(1, dtype=torch.int32, device=ends.device), ends)),
            torch.where(active, b.request_state_rows[slots, 0], 0),
            positions,
            lengths,
            request,
            torch.where(live, positions[request] + offsets - starts[request], -1),
            pages,
            pages.expand(self.tokens, -1) if self.rows == 1 else pages[request],
            torch.cat((ends[-1:], b.live_rows.clamp(0, self.rows))),
            self.max_query_tokens,
        )

    def finish(self, metadata):
        self.token_state.consume(metadata.token_ids, metadata.query_start_loc)


class RotaryBatch(PreparedBatch):
    """Packed metadata for a pool whose layout declares a rotary offset region.

    Prefill takes per-axis rotary positions from the uploaded prompt positions and
    commits each request's offset to its continuation row. Decode rotates at the
    token index plus that offset.
    """

    def __init__(self, inputs, requests, metadata, *, pool, **options):
        super().__init__(inputs, requests, metadata, pool=pool, **options)
        self.rotary_offset = rotary_offset_region(pool)
        if (self.rotary_offset is None or self.rotary_offset.shape[1:] != (1,)
                or self.rotary_offset.dtype != torch.int64):
            raise ValueError("three-axis batches require one int64 rotary offset per continuation row")
        if self.mode == "prefill" and not isinstance(inputs, RotaryBatchInputs):
            raise ValueError("three-axis prefill requires uploaded prompt positions")

    def begin(self, totals, *, present):
        metadata = super().begin(totals, present=present)
        if self.mode == "prefill":
            live = torch.arange(self.tokens, device=metadata.positions.device) < metadata.live_counts[0]
            rope = torch.where(live[:, None], self.inputs.prefill_positions[:self.tokens], -1)
        else:
            # `state_rows` is zero for rows that are inactive or failed validation.
            offset = gather_rotary_offset(self.rotary_offset, metadata.state_rows)
            logical = metadata.positions
            rope = torch.where(logical >= 0, logical + offset[metadata.requests], -1)[:, None]
        return replace(metadata, rope_positions=rope)

    def finish(self, metadata):
        if self.mode == "prefill":
            self.commit_rotary(metadata)
        super().finish(metadata)

    def commit_rotary(self, metadata):
        """Continue each request's rotary positions past the prompt's largest axis.

        Text continues at the token index, so its offset stays zero. Image
        positions occupy a rectangular grid; the next token continues from the
        largest axis position plus one.
        """
        if metadata.rope_positions is None:
            raise ValueError("committing rotary offsets requires per-axis prompt positions")
        offsets = torch.arange(metadata.token_ids.numel(), device=metadata.starts.device)
        live = offsets < metadata.live_counts[0]
        maximum = torch.where(live[:, None], metadata.rope_positions, -1).max()
        previous = gather_rotary_offset(self.rotary_offset, metadata.state_rows)
        endpoint = metadata.starts + metadata.lengths
        delta = torch.maximum(metadata.starts + previous, maximum + 1) - endpoint
        scatter_rows(delta[:, None], metadata.state_rows, self.rotary_offset, metadata.lengths > 0)


def prepare_batch(inputs, requests, metadata, *, pool, **options):
    """The batch for ``pool``: a `RotaryBatch` when its layout declares a rotary offset region."""
    batch = RotaryBatch if rotary_offset_region(pool) is not None else PreparedBatch
    return batch(inputs, requests, metadata, pool=pool, **options)
