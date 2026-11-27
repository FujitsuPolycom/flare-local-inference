# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compact b12x Engram lookup behind a captured external readiness event."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import triton
import triton.language as tl
from b12x.preparation import PreparedCall
from b12x.sequence import engram
from b12x.sequence.engram._kernels import hash_op, lookup_op

from ignition.components.ngram_delivery import publish_requests
from ignition.components.readiness import PreparedReadinessWait, ReadinessTimings, wait_for_readiness
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tuning import declare


@triton.jit(do_not_specialize=["prepared_tokens"])
def _compact_lookup(
    weight,
    scales,
    hashes,
    num_tokens,
    out,
    prepared_tokens,
    T: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
):
    t, head = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, 256)
    nt = tl.load(num_tokens)
    live = (t < nt) & (t < prepared_tokens) & (nt >= 0) & (nt <= T)
    local_row = t.to(tl.int64) * HEADS + head
    row = tl.load(hashes + local_row, live, -1).to(tl.int64)
    local = live & (row >= 0) & (row < ROWS)
    quant = tl.load(weight + local_row * 256 + col.to(tl.int64), local, 0.0).to(tl.float32)
    exponent = tl.load(scales + local_row * 8 + (col // 32).to(tl.int64), local, 127).to(tl.uint32)
    # E8M0 byte zero is 2^-127, not floating point zero; 255 is NaN.
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exponent == 0, 2.0**-127, scale)
    scale = tl.where(exponent == 255, float("nan"), scale)
    result = (quant * scale).to(tl.bfloat16)
    tl.store(out + local_row * 256 + col.to(tl.int64), tl.where(local, result, 0))


@torch.library.custom_op("flare::staged_engram", mutates_args=("out", "timings", "barrier_output"))
def _lookup(source: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
            hashes: torch.Tensor, count: torch.Tensor, out: torch.Tensor,
            timings: torch.Tensor, frame: torch.Tensor, barrier_input: torch.Tensor,
            barrier_output: torch.Tensor, interval: int, barrier: int, plan: int, event: int) -> None:
    wait_for_readiness(source, event, timings, frame, interval, barrier_input, barrier_output, barrier)
    lookup_op(plan, weight.view(torch.float8_e4m3fn), scales, hashes, count, out, hashes.shape[0], True)


@_lookup.register_fake
def _lookup_fake(source, weight, scales, hashes, count, out, timings, frame,
                 barrier_input, barrier_output, interval, barrier, plan, event):
    pass


@torch.library.custom_op("flare::staged_engram_compact", mutates_args=("out", "timings", "barrier_output"))
def _lookup_compact(source: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
                    hashes: torch.Tensor, count: torch.Tensor, out: torch.Tensor,
                    timings: torch.Tensor, frame: torch.Tensor, barrier_input: torch.Tensor,
                    barrier_output: torch.Tensor, interval: int, barrier: int,
                    rows: int, event: int) -> None:
    wait_for_readiness(source, event, timings, frame, interval, barrier_input, barrier_output, barrier)
    tokens, heads = hashes.shape
    if tokens:
        _compact_lookup[(tokens, heads, 1)](
            weight.view(torch.float8_e4m3fn), scales, hashes, count, out, tokens,
            T=tokens, HEADS=heads, ROWS=rows, num_warps=4)


@_lookup_compact.register_fake
def _lookup_compact_fake(source, weight, scales, hashes, count, out, timings, frame,
                         barrier_input, barrier_output, interval, barrier, rows, event):
    pass


_CANONICAL_HEADS = 24


def _validate_staged_heads(heads: int, *, world: int | None = None) -> None:
    if type(heads) is not int or not 1 <= heads <= _CANONICAL_HEADS or _CANONICAL_HEADS % heads != 0:
        raise ValueError("Engram staged heads must divide 24")
    if world is not None:
        if type(world) is not int or world < 1:
            raise ValueError("Engram world size must be a positive integer")
        legal = (
            (_CANONICAL_HEADS,)
            if (world == 1 or _CANONICAL_HEADS % world != 0)
            else (_CANONICAL_HEADS, _CANONICAL_HEADS // world)
        )
        if heads not in legal:
            raise ValueError(
                f"Engram staged heads {heads} is invalid for world size {world}; expected one of {legal}"
            )


@dataclass(frozen=True)
class EngramInputs:
    weights: torch.Tensor
    scales: torch.Tensor
    hashes: torch.Tensor
    ready: object
    timings: ReadinessTimings
    interval: int

    @classmethod
    def allocate(cls, tokens, device, *, timings, interval=0, heads=24, world=None):
        _validate_staged_heads(heads, world=world)
        ready = torch.cuda.Event(external=True)
        ready.record(torch.cuda.current_stream(device))
        return cls(torch.zeros((tokens * heads, 256), dtype=torch.uint8, device=device),
                   torch.zeros((tokens * heads, 8), dtype=torch.uint8, device=device),
                   torch.full((tokens, heads), -1, dtype=torch.int64, device=device), ready, timings, interval)

    def buffers(self, layer):
        return {f"engram_{layer}_{name}": getattr(self, name) for name in ("weights", "scales", "hashes")}


class PreparedEngramLookup:
    def __init__(self, inputs, output, count, *, layer, token_map, geometry, session, name,
                 runtime=None, coordinator=None):
        tokens = output.shape[0]
        heads = inputs.hashes.shape[1]
        _validate_staged_heads(heads, world=getattr(runtime, "world_size", None))
        if (output.ndim != 2 or output.shape[1] != heads * 256
                or output.dtype != torch.bfloat16 or not output.is_contiguous()):
            raise ValueError("Engram lookup output must match contiguous [tokens, heads * 256] bfloat16")
        self.weights = inputs.weights[:tokens * heads]
        self.scales = inputs.scales[:tokens * heads]
        self.hashes = inputs.hashes[:tokens]
        self.count, self.output = count, output
        self.handle = inputs.ready.cuda_event
        self.wait = PreparedReadinessWait(inputs.timings, inputs.interval, runtime=runtime, session=session,
                                         coordinator=coordinator, name=name + ".readiness")
        self.table_rows = geometry.num_embeddings[geometry.layer_ids.index(layer)]
        if heads == _CANONICAL_HEADS:
            self.plan = engram.plan(engram.Caps(device=output.device, max_tokens=tokens,
                max_seqs=tokens, max_requests=tokens, vocab_size=len(token_map), layer_id=layer,
                tp_size=1, tp_rank=0), token_map=token_map, geometry=geometry,
                invocation={"operation": "lookup", "compact_rows": True})
            binding = SimpleNamespace(weight=self.weights.view(torch.float8_e4m3fn),
                scale_bytes=self.scales, hash_ids=self.hashes, num_tokens=count, out=output)
            declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
                run=lambda: state.run_lookup(binding, tokens, clear_tail=True), owners=(inputs, binding))),))
        else:
            self.plan = None

    def functional(self, *, session):
        self.output_layout = tuple(self.output.shape), self.output.dtype
        self.count = self.output = None
        self.wait.functional(session=session)
        return self

    def __call__(self, source, count=None):
        output = self.output
        if output is None:
            shape, dtype = self.output_layout
            output = torch.empty(shape, dtype=dtype, device=source.device)
        barrier_input, barrier_output = self.wait.operands(source)
        live = self.count if count is None else count
        if self.plan is None:
            _lookup_compact(source, self.weights, self.scales, self.hashes, live,
                            output, self.wait.timings.timestamps, self.wait.timings.frame,
                            barrier_input, barrier_output, self.wait.interval, self.wait.handle,
                            self.table_rows, self.handle)
        else:
            _lookup(source, self.weights, self.scales, self.hashes, live,
                    output, self.wait.timings.timestamps, self.wait.timings.frame,
                    barrier_input, barrier_output, self.wait.interval, self.wait.handle,
                    self.plan.handle, self.handle)
        return output


class PreparedEngramRequests:
    """Hash packed token rows on the GPU and release their mapped delivery mailbox."""

    def __init__(self, inputs, token_ids, starts, history, counts, mailbox, sequence, *,
                 layer, token_map, geometry, session, name, row_mapping=None, head_indices=None):
        tokens, rows = token_ids.numel(), history.shape[0]
        device = token_ids.device
        self.mask = torch.ones(tokens, dtype=torch.bool, device=device)
        self.slots = torch.arange(rows, dtype=torch.int32, device=device)
        self.compressed = torch.empty(tokens, dtype=torch.int64, device=device)
        self.requests = torch.empty(tokens, dtype=torch.int32, device=device)
        self.hashes = inputs.hashes[:tokens]
        local_heads = self.hashes.shape[1]
        world = None
        if row_mapping is not None:
            if row_mapping.ndim != 2 or row_mapping.shape[1] != 3 or row_mapping.dtype != torch.int64 or row_mapping.device != device:
                raise ValueError("embedding row mapping must have int64 source bounds and a destination per head")
            if row_mapping.shape[0] == _CANONICAL_HEADS:
                owned = int((row_mapping[:, 2] >= 0).sum().item())
                world = _CANONICAL_HEADS // owned if owned > 0 and _CANONICAL_HEADS % owned == 0 else 1
        _validate_staged_heads(local_heads, world=world)
        self.mailbox, self.sequence = mailbox, sequence
        if head_indices is not None:
            indices_list = head_indices.tolist() if isinstance(head_indices, torch.Tensor) else list(head_indices)
            if (len(indices_list) != local_heads
                    or any(type(head) is not int or not 0 <= head < _CANONICAL_HEADS for head in indices_list)
                    or len(set(indices_list)) != local_heads):
                raise ValueError("head_indices must name distinct canonical heads matching staged width")
            self.head_indices = persistent_copy(torch.tensor(indices_list, dtype=torch.int64, device=device))
        elif local_heads < _CANONICAL_HEADS:
            raise ValueError("compact Engram request staging requires explicit head_indices")
        else:
            self.head_indices = None
        if row_mapping is not None:
            if row_mapping.shape[0] == _CANONICAL_HEADS and self.head_indices is not None:
                row_mapping = persistent_copy(torch.index_select(row_mapping, 0, self.head_indices))
            elif row_mapping.shape[0] != local_heads:
                raise ValueError("embedding row mapping must have int64 source bounds and a destination per head")
        self.row_mapping = row_mapping
        self.full_hashes = (
            torch.empty((tokens, 24), dtype=torch.int64, device=device)
            if self.head_indices is not None else self.hashes
        )
        self.args = (token_ids, self.mask, starts, self.slots, history, counts[1:], counts[:1],
                     self.full_hashes, self.compressed, self.requests, tokens)
        self.plan = engram.plan(engram.Caps(device=device, max_tokens=tokens, max_seqs=rows,
            max_requests=rows, vocab_size=len(token_map), layer_id=layer, tp_size=1, tp_rank=0),
            token_map=token_map, geometry=geometry, invocation={"operation": "hash"})
        binding = SimpleNamespace(**dict(zip(("token_ids", "token_mask", "query_start_loc", "request_slots",
            "committed_history", "num_seqs", "num_tokens", "hash_ids", "compressed", "request_ids"), self.args[:-1])))
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(binding, tokens), owners=(binding,))),))

    def functional(self):
        self.mask = self.slots = self.compressed = self.requests = self.full_hashes = self.args = None
        return self

    def __call__(self, token_ids=None, starts=None, history=None, counts=None):
        args = self.args
        if args is None:
            tokens, rows = token_ids.numel(), history.shape[0]
            mask = torch.ones(tokens, dtype=torch.bool, device=token_ids.device)
            slots = torch.arange(rows, dtype=torch.int32, device=token_ids.device)
            compressed = torch.empty(tokens, dtype=torch.int64, device=token_ids.device)
            requests = torch.empty(tokens, dtype=torch.int32, device=token_ids.device)
            full_hashes = (
                torch.empty((tokens, 24), dtype=torch.int64, device=token_ids.device)
                if self.head_indices is not None else self.hashes
            )
            args = (token_ids, mask, starts, slots, history, counts[1:], counts[:1],
                    full_hashes, compressed, requests, tokens)
        else:
            full_hashes = self.full_hashes
        hash_op(self.plan.handle, *args)
        selected = (
            torch.index_select(full_hashes, 1, self.head_indices)
            if self.head_indices is not None else self.hashes
        )
        if self.row_mapping is not None:
            # Preserve canonical hash geometry; remap only the storage rows of owned heads.
            start, end, destination = self.row_mapping.unbind(1)
            valid = (destination >= 0) & (selected >= start) & (selected < end)
            self.hashes.copy_(torch.where(valid, selected - start + destination, -1))
        elif self.head_indices is not None:
            self.hashes.copy_(selected)
        publish_requests(self.hashes, self.mailbox, self.sequence)
