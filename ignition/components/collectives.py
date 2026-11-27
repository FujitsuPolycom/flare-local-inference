# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared b12x reductions and gathers embedded in compiled CUDA graphs.

Ranks on one host communicate through b12x PCIe runtimes; ranks on separate
DGX Sparks through RoCEnante (b12x.comm.roce), whose one-shot all-reduce and
all-gather are captured in the same graphs. A component given no runtime serves
one rank: the reduction or gather is the rank's own values, copied into the
output its consumers read.
"""

import itertools
from math import prod
import weakref
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from b12x.preparation import CollectiveRequirement, PreparedCall

from ignition.preparation.collectives import VARLEN_BANK_BYTES
from ignition.preparation.tuning import declare, register_finalizer

_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


def packed_broadcast_layout(fields):
    """Return byte offsets and extent for eight-byte-aligned typed fields."""
    offsets, size = [], 0
    for shape, dtype in fields:
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError("broadcast field dimensions must be nonnegative integers")
        size = (size + 7) // 8 * 8
        offsets.append(size)
        size += prod(shape) * dtype.itemsize
    return tuple(offsets), size


def _roce(runtime):
    """True for a RoCEnante runtime joining ranks on separate hosts."""
    from b12x.comm import roce

    return isinstance(runtime, roce.AllReduce)


def _roce_plan(runtime, coordinator, surface):
    """One fixed RoCEnante plan; every operation's geometry is validated by the runtime at launch."""
    from b12x.comm import roce

    query = roce.query_from_runtime(runtime, surface=surface, call={"dtypes": ("bfloat16",)},
                                    topology="roce_rdma", peer_hosts=coordinator.hosts)
    return roce.plan(query, runtime=runtime)


@torch.library.custom_op("flare::pcie_all_reduce", mutates_args=("output",))
def _all_reduce(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.all_reduce(source, out=output, plan=owner.plan)


@_all_reduce.register_fake
def _fake_all_reduce(source, output, handle):
    pass


@torch.library.custom_op("flare::pcie_all_reduce_addend", mutates_args=("output",))
def _all_reduce_addend(source: torch.Tensor, addend: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.all_reduce(source, out=output, addend=addend, plan=owner.plan)


@_all_reduce_addend.register_fake
def _fake_all_reduce_addend(source, addend, output, handle):
    pass


@torch.library.custom_op("flare::pcie_all_reduce_dep", mutates_args=("output",))
def _all_reduce_dep(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor], handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.all_reduce(source, out=output, plan=owner.plan)


@_all_reduce_dep.register_fake
def _fake_all_reduce_dep(source, output, dependencies, handle):
    pass


@torch.library.custom_op("flare::pcie_all_reduce_addend_dep", mutates_args=("output",))
def _all_reduce_addend_dep(source: torch.Tensor, addend: torch.Tensor, output: torch.Tensor,
                           dependencies: list[torch.Tensor], handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.all_reduce(source, out=output, addend=addend, plan=owner.plan)


@_all_reduce_addend_dep.register_fake
def _fake_all_reduce_addend_dep(source, addend, output, dependencies, handle):
    pass


@torch.library.custom_op("flare::start_all_reduce", mutates_args=("output",))
def _start_all_reduce(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor],
                      handle: int, stream: int) -> None:
    owner = _OWNERS[handle]
    if owner.concurrent_input is not None:
        send = owner.concurrent_input
        if send.data_ptr() != source.data_ptr():
            send.copy_(source)
    else:
        send = source
    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        owner.runtime.all_reduce(send, out=output, plan=owner.plan)


@_start_all_reduce.register_fake
def _fake_start_all_reduce(source, output, dependencies, handle, stream):
    pass


@torch.library.custom_op("flare::join_all_reduce", mutates_args=("output",))
def _join_all_reduce(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor],
                     dependency: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(source.device).wait_stream(torch.cuda.ExternalStream(stream, device=source.device))


@_join_all_reduce.register_fake
def _fake_join_all_reduce(source, output, dependencies, dependency, stream):
    pass


@dataclass(frozen=True)
class PendingAllReduce:
    source: torch.Tensor
    output: torch.Tensor
    dependencies: list[torch.Tensor]


class ConcurrentAllReduce:
    """Overlap a prepared PCIe all-reduce with local compute on an auxiliary stream."""

    def __init__(self, reduce, *, workspace, session, parallel_plans, send_buffer=None):
        from ignition.preparation.scratch import ScratchArena

        plans = tuple(parallel_plans)
        if reduce.single or _roce(reduce.runtime):
            raise ValueError("concurrent all-reduce requires a prepared PCIe channel")
        if not isinstance(workspace, ScratchArena) or not workspace.overlap or not plans:
            raise ValueError("concurrent all-reduce requires an overlap arena and local plans")
        self.reduce = reduce
        self.reduce.concurrent_input = send_buffer
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        for plan in plans:
            workspace.register_parallel(reduce.plan, plan)
        register_finalizer(session, self.validate_scratch)

    def validate_scratch(self):
        if self.reduce.plan.memory_requirements().scratch_nbytes:
            raise ValueError("concurrent all-reduce requires a plan without caller scratch")

    def functional(self, *, session):
        self.reduce.functional(session=session)
        return self

    def start(self, source, output=None, *, dependencies=()):
        if output is None:
            output = self.reduce.output
        if output is None:
            shape, stride, dtype = self.reduce.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        send = source.clone() if self.reduce.concurrent_input is None else source
        deps = list(dependencies)
        _start_all_reduce(send, output, deps, self.reduce.handle, self.stream_handle)
        return PendingAllReduce(send, output, deps)

    def finish(self, pending, dependency):
        _join_all_reduce(pending.source, pending.output, pending.dependencies, dependency, self.stream_handle)
        return pending.output


class PreparedAllReduce:
    """One fixed-shape reduction through an established b12x PCIe runtime.

    The bootstrap owns a single-channel runtime, exchange group, and coordinator.
    Every participating rank must declare the same operation name and enter
    preparation together. Graph ownership must retain this component and its
    runtime until every rank has drained native replay.
    """

    def __init__(self, source, output, *, runtime, session, coordinator, name, has_addend=False):
        import torch.distributed as dist
        from b12x.comm import pcie

        self.runtime, self.output = runtime, output
        self.fused_addend = False
        self.concurrent_input = None
        # Traced forwards test this flag, never the runtime: compile workers see a runtime that
        # owns RDMA or IPC resources only as a placeholder.
        self.single = runtime is None
        if runtime is None:
            if source.shape != output.shape or source.dtype != output.dtype:
                raise ValueError("a single-rank reduction copies its source into a matching output")
            return
        if _roce(runtime):
            if (source.shape != output.shape or source.dtype != torch.bfloat16 or output.dtype != source.dtype
                    or not source.is_contiguous() or not output.is_contiguous()
                    or not 0 < source.numel() * 2 <= max(runtime.max_size, runtime.pipe_bytes)
                    or source.numel() * 2 % 16):
                raise ValueError("RoCE reductions require contiguous BF16 16-byte packs within the runtime's capacity")
            self.plan = _roce_plan(runtime, coordinator, "AllReduce.all_reduce")
            group = runtime._group

            def prepare(state):
                return PreparedCall(run=lambda: state.all_reduce(source, out=output), output=output,
                                    owners=(runtime,))
        elif isinstance(runtime, pcie.DmaAllReduce):
            # Copy-engine ring for prefill-sized activations.
            from b12x.comm.pcie._dma_preparation import query_from_metadata

            self.plan = pcie.plan(query_from_metadata(runtime, shape=tuple(source.shape), dtype=source.dtype),
                                  runtime=runtime)
            group = runtime.group

            def prepare(state):
                return PreparedCall(run=lambda: state.run(source, out=output), output=output,
                                    owners=(runtime,))
        elif runtime.algorithm == "hierarchical":
            from b12x.comm.pcie._owner_preparation import prepared_call

            self.plan = runtime.plan(source, out=output)
            group = runtime.for_stream().group

            def prepare(state):
                return prepared_call(state, inp=source, out=output)
        else:
            if (runtime.world_size < 2 or runtime.algorithm != "oneshot"
                    or not runtime.single_channel):
                raise ValueError("prepared PCIe reductions require single-channel b12x direct all-reduce")
            channel = runtime.for_stream()
            plan_options = {}
            if (has_addend and source.dtype == torch.bfloat16
                    and getattr(channel, "_eager_ptrs", None) is not None):
                plan_options["has_addend"] = True
                self.fused_addend = True
            self.plan = runtime.plan(source, out=output, **plan_options)
            group = runtime.exchange_group
            zero_addend = torch.zeros_like(source) if self.fused_addend else None

            def prepare(state):
                if state.query.surface == "PCIeTwoShotBF16.all_reduce":
                    from b12x.comm.pcie._twoshot_preparation import prepared_call

                    if zero_addend is not None:
                        call = prepared_call(state, payload=source, out=output, addend=zero_addend)
                        return PreparedCall(
                            run=call.run,
                            output=call.output,
                            capture_safe=call.capture_safe,
                            owners=(*call.owners, runtime, zero_addend),
                        )
                    return prepared_call(state, payload=source, out=output)
                state.bind_input(source)
                return PreparedCall(
                    run=((lambda: state.run_plain(source, output, addend=zero_addend))
                         if zero_addend is not None else (lambda: state.run_plain(source, output))),
                    output=output,
                    owners=(runtime,) if zero_addend is None else (runtime, zero_addend),
                )

        ranks = tuple(dist.get_process_group_ranks(group))
        request = self.plan.request(name=name, prepare_call=prepare,
            collective=CollectiveRequirement(key=name, ranks=ranks))
        declare(session, (request,), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def functional(self, *, session):
        """Allocate the reduction destination inside the traced forward."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
        return self

    def __call__(self, source, output=None, *, dependencies=(), addend=None):
        if output is None:
            output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        if self.single:
            if addend is not None:
                torch.add(source, addend, out=output)
            else:
                output.copy_(source)
        elif addend is not None and self.fused_addend:
            if dependencies:
                _all_reduce_addend_dep(source, addend, output, list(dependencies), self.handle)
            else:
                _all_reduce_addend(source, addend, output, self.handle)
        else:
            if addend is not None:
                source = source + addend
            if dependencies:
                _all_reduce_dep(source, output, list(dependencies), self.handle)
            else:
                _all_reduce(source, output, self.handle)
        return output


def _dma(runtime):
    """True for a PCIe Copy-Engine ring runtime joining ranks on one host."""
    from b12x.comm import pcie

    return isinstance(runtime, pcie.DmaAllReduce)


def _island_row_shard(runtime):
    """True for a TP16 PCIe island runtime exposing row-shard surfaces."""
    from b12x.comm import pcie

    return (
        isinstance(runtime, pcie.AllReduce)
        and getattr(runtime, "_island_rs", None) is not None
        and getattr(runtime, "_algorithm_override", None) != "hierarchical"
        and hasattr(runtime, "should_reduce_scatter")
        and hasattr(runtime, "should_all_gather")
    )


def _row_shard_capacity_bytes(runtime):
    """Maximum contiguous message payload supported by one row-shard collective call."""
    if _dma(runtime):
        return runtime.max_bytes
    if _island_row_shard(runtime):
        return runtime._island_rs.max_elements * 2
    return runtime.pipe_bytes


def row_shard_surfaces(runtime):
    """True for a RoCE (ABI >= 11), PCIe DMA, or TP16 PCIe island runtime exposing row-shard surfaces."""
    if _dma(runtime):
        return hasattr(runtime, "should_reduce_scatter") and hasattr(runtime, "should_all_gather")
    if _island_row_shard(runtime):
        return True
    if not _roce(runtime):
        return False
    from b12x.comm.roce._proxy import ABI_VERSION

    return (ABI_VERSION >= 11
            and hasattr(runtime, "should_reduce_scatter") and hasattr(runtime, "should_gather_shards"))


@torch.library.custom_op("flare::roce_reduce_scatter", mutates_args=("output",))
def _reduce_scatter(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.reduce_scatter(source, out=output, plan=owner.plan)


@_reduce_scatter.register_fake
def _fake_reduce_scatter(source, output, handle):
    pass


def _dma_reduce_scatter_exec(owner, source, output):
    owner.runtime.reduce_scatter(source, out=output, plan=owner.plan)


@torch.library.custom_op("flare::pcie_reduce_scatter", mutates_args=("output",))
def _pcie_reduce_scatter(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    _dma_reduce_scatter_exec(_OWNERS[handle], source, output)


@_pcie_reduce_scatter.register_fake
def _fake_pcie_reduce_scatter(source, output, handle):
    pass


# `dependencies` is an unread graph-ordering operand that orders preceding writes before the collective.
@torch.library.custom_op("flare::pcie_reduce_scatter_dep", mutates_args=("output",))
def _pcie_reduce_scatter_dep(source: torch.Tensor, output: torch.Tensor,
                             dependencies: list[torch.Tensor], handle: int) -> None:
    _dma_reduce_scatter_exec(_OWNERS[handle], source, output)


@_pcie_reduce_scatter_dep.register_fake
def _fake_pcie_reduce_scatter_dep(source, output, dependencies, handle):
    pass


@torch.library.custom_op("flare::start_reduce_scatter", mutates_args=("output",))
def _start_reduce_scatter(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor],
                          handle: int, stream: int) -> None:
    owner = _OWNERS[handle]
    if owner.concurrent_input is not None:
        send = owner.concurrent_input
        if send.data_ptr() != source.data_ptr():
            send.copy_(source)
    else:
        send = source
    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        _dma_reduce_scatter_exec(owner, send, output)


@_start_reduce_scatter.register_fake
def _fake_start_reduce_scatter(source, output, dependencies, handle, stream):
    pass


@torch.library.custom_op("flare::join_reduce_scatter", mutates_args=("output",))
def _join_reduce_scatter(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor],
                         dependency: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(source.device).wait_stream(torch.cuda.ExternalStream(stream, device=source.device))


@_join_reduce_scatter.register_fake
def _fake_join_reduce_scatter(source, output, dependencies, dependency, stream):
    pass


@dataclass(frozen=True)
class PendingReduceScatter:
    source: torch.Tensor
    output: torch.Tensor
    dependencies: list[torch.Tensor]


class ConcurrentReduceScatter:
    """Overlap a prepared PCIe DMA reduce-scatter with local compute on an auxiliary stream."""

    def __init__(self, reduce_scatter, *, workspace, session, parallel_plans, send_buffer=None):
        from ignition.preparation.scratch import ScratchArena

        plans = tuple(parallel_plans)
        if reduce_scatter.single or not reduce_scatter.pcie:
            raise ValueError("concurrent reduce-scatter requires a prepared PCIe DMA channel")
        if not isinstance(workspace, ScratchArena) or not workspace.overlap or not plans:
            raise ValueError("concurrent reduce-scatter requires an overlap arena and local plans")
        self.reduce = reduce_scatter
        self.reduce.concurrent_input = send_buffer
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        for plan in plans:
            workspace.register_parallel(reduce_scatter.plan, plan)
        register_finalizer(session, self.validate_scratch)

    def validate_scratch(self):
        if self.reduce.plan.memory_requirements().scratch_nbytes:
            raise ValueError("concurrent reduce-scatter requires a plan without caller scratch")

    def functional(self, *, session):
        self.reduce.functional(session=session)
        return self

    def start(self, source, output=None, *, dependencies=(), clone=True):
        if output is None:
            output = self.reduce.output
        if output is None:
            shape, stride, dtype = self.reduce.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        send = source.clone() if (clone and self.reduce.concurrent_input is None) else source
        deps = list(dependencies)
        _start_reduce_scatter(send, output, deps, self.reduce.handle, self.stream_handle)
        return PendingReduceScatter(send, output, deps)

    def finish(self, pending, dependency):
        _join_reduce_scatter(pending.source, pending.output, pending.dependencies, dependency, self.stream_handle)
        return pending.output


class PreparedReduceScatter:
    """One fixed-shape reduction into this rank's contiguous row shard through RoCE or PCIe DMA.

    The reduced rows split into `world_size` equal shards and each rank keeps its own. The
    bootstrap owns the runtime, exchange group and coordinator; every participating rank declares
    the same operation name and enters preparation together.
    """

    def __init__(self, source, output, *, runtime, session, coordinator, name, concurrent=False):
        import torch.distributed as dist

        self.runtime, self.output = runtime, output
        self.dma = False
        self.pcie = False
        self.concurrent_input = None
        # Traced forwards test this flag, never the runtime: compile workers see a runtime that
        # owns RDMA or IPC resources only as a placeholder.
        self.single = runtime is None
        if runtime is None:
            if source.shape != output.shape or source.dtype != output.dtype:
                raise ValueError("a single-rank reduction copies its source into a matching output")
            return
        if not self.fits(runtime, source, output):
            raise ValueError("row reductions require contiguous BF16 16-byte packs dividing into"
                             " this rank's shard within the runtime's capacity")
        self.dma = _dma(runtime)
        island = _island_row_shard(runtime)
        self.pcie = self.dma or island
        if self.dma:
            from b12x.comm import pcie
            from b12x.comm.pcie._dma_preparation import query_from_metadata

            self.plan = pcie.plan(query_from_metadata(runtime, shape=tuple(source.shape), dtype=source.dtype),
                                  runtime=runtime)
            ranks = tuple(dist.get_process_group_ranks(runtime.group))

            def prepare(state):
                return PreparedCall(run=lambda: state.run_reduce_scatter(source, out=output), output=output,
                                    owners=(runtime,))
        elif island:
            from b12x.comm.pcie._owner_preparation import prepared_call

            plan_kwargs = {"transport": "ce_local"} if concurrent else {}
            self.plan = runtime.plan(source, operation="reduce_scatter", out=output, **plan_kwargs)
            ranks = tuple(dist.get_process_group_ranks(runtime._island_rs.group))

            def prepare(state):
                return prepared_call(state, inp=source, out=output)
        else:
            self.plan = _roce_plan(runtime, coordinator, "AllReduce.reduce_scatter")
            ranks = tuple(dist.get_process_group_ranks(runtime._group))

            def prepare(state):
                return PreparedCall(run=lambda: state.reduce_scatter(source, out=output), output=output,
                                    owners=(runtime, source, output))
        declare(session, (self.plan.request(name=name, prepare_call=prepare,
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    @staticmethod
    def fits(runtime, source, output):
        """True when this runtime can split `source`'s rows into the shard `output` holds."""
        if runtime is None or not row_shard_surfaces(runtime) or source.ndim < 1:
            return False
        rows, world = source.shape[0], runtime.world_size
        row_bytes = source.numel() // max(rows, 1) * source.element_size()
        return (rows % world == 0
                and tuple(output.shape) == (rows // world, *source.shape[1:])
                and source.dtype is torch.bfloat16 and output.dtype is source.dtype
                and source.is_contiguous() and output.is_contiguous()
                and row_bytes % 16 == 0
                and 0 < source.numel() * source.element_size() <= _row_shard_capacity_bytes(runtime)
                and runtime.should_reduce_scatter(source))

    def functional(self, *, session):
        """Allocate this rank's shard destination inside the traced forward."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
        return self

    def __call__(self, source, output=None, *, dependencies=()):
        if output is None:
            output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        if self.single:
            output.copy_(source)
        elif self.pcie:
            if dependencies:
                _pcie_reduce_scatter_dep(source, output, list(dependencies), self.handle)
            else:
                _pcie_reduce_scatter(source, output, self.handle)
        else:
            if dependencies:
                raise ValueError("RoCE reduce-scatter does not accept graph-ordering dependencies")
            _reduce_scatter(source, output, self.handle)
        return output


@torch.library.custom_op("flare::roce_gather_shards", mutates_args=("output",))
def _gather_shards(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.gather_shards(source, out=output, plan=owner.plan)


@_gather_shards.register_fake
def _fake_gather_shards(source, output, handle):
    pass


def _dma_gather_shards_exec(owner, source, output):
    owner.runtime.all_gather(
        source, out=output.view(owner.runtime.world_size, *source.shape), plan=owner.plan
    )


@torch.library.custom_op("flare::pcie_gather_shards", mutates_args=("output",))
def _pcie_gather_shards(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    _dma_gather_shards_exec(_OWNERS[handle], source, output)


@_pcie_gather_shards.register_fake
def _fake_pcie_gather_shards(source, output, handle):
    pass


# `dependencies` is an unread graph-ordering operand that orders preceding writes before the collective.
@torch.library.custom_op("flare::pcie_gather_shards_dep", mutates_args=("output",))
def _pcie_gather_shards_dep(source: torch.Tensor, output: torch.Tensor,
                            dependencies: list[torch.Tensor], handle: int) -> None:
    _dma_gather_shards_exec(_OWNERS[handle], source, output)


@_pcie_gather_shards_dep.register_fake
def _fake_pcie_gather_shards_dep(source, output, dependencies, handle):
    pass


@torch.library.custom_op("flare::start_gather_shards", mutates_args=("output",))
def _start_gather_shards(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor],
                         handle: int, stream: int) -> None:
    owner = _OWNERS[handle]
    if owner.concurrent_input is not None:
        send = owner.concurrent_input
        if send.data_ptr() != source.data_ptr():
            send.copy_(source)
    else:
        send = source
    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        _dma_gather_shards_exec(owner, send, output)


@_start_gather_shards.register_fake
def _fake_start_gather_shards(source, output, dependencies, handle, stream):
    pass


@torch.library.custom_op("flare::join_gather_shards", mutates_args=("output",))
def _join_gather_shards(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor],
                        dependency: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(source.device).wait_stream(torch.cuda.ExternalStream(stream, device=source.device))


@_join_gather_shards.register_fake
def _fake_join_gather_shards(source, output, dependencies, dependency, stream):
    pass


@dataclass(frozen=True)
class PendingGatherShards:
    source: torch.Tensor
    output: torch.Tensor
    dependencies: list[torch.Tensor]


class ConcurrentGatherShards:
    """Overlap a prepared PCIe DMA row-shard gather with local compute on an auxiliary stream."""

    def __init__(self, gather, *, workspace, session, parallel_plans, send_buffer=None):
        from ignition.preparation.scratch import ScratchArena

        plans = tuple(parallel_plans)
        if gather.single or not gather.pcie:
            raise ValueError("concurrent shard gather requires a prepared PCIe DMA channel")
        if not isinstance(workspace, ScratchArena) or not workspace.overlap or not plans:
            raise ValueError("concurrent shard gather requires an overlap arena and local plans")
        self.gather = gather
        self.gather.concurrent_input = send_buffer
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        for plan in plans:
            workspace.register_parallel(gather.plan, plan)
        register_finalizer(session, self.validate_scratch)

    def validate_scratch(self):
        if self.gather.plan.memory_requirements().scratch_nbytes:
            raise ValueError("concurrent shard gather requires a plan without caller scratch")

    def functional(self, *, session):
        self.gather.functional(session=session)
        return self

    def start(self, source, output=None, *, dependencies=(), clone=True):
        if output is None:
            output = self.gather.output
        if output is None:
            shape, stride, dtype = self.gather.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        send = source.clone() if (clone and self.gather.concurrent_input is None) else source
        deps = list(dependencies)
        _start_gather_shards(send, output, deps, self.gather.handle, self.stream_handle)
        return PendingGatherShards(send, output, deps)

    def finish(self, pending, dependency):
        _join_gather_shards(pending.source, pending.output, pending.dependencies, dependency, self.stream_handle)
        return pending.output


class PreparedGatherShards:
    """One fixed-shape gather of every rank's row shard into all rows through RoCE or PCIe DMA.

    Rank zero's shard comes first in the output rows, then rank one's, and so on, copied without
    arithmetic. The bootstrap owns the runtime, exchange group and coordinator; every
    participating rank declares the same operation name and enters preparation together.
    """

    def __init__(self, source, output, *, runtime, session, coordinator, name, concurrent=False):
        import torch.distributed as dist

        self.runtime, self.output = runtime, output
        self.dma = False
        self.pcie = False
        self.concurrent_input = None
        # Traced forwards test this flag, never the runtime: compile workers see a runtime that
        # owns RDMA or IPC resources only as a placeholder.
        self.single = runtime is None
        if runtime is None:
            if source.shape != output.shape or source.dtype != output.dtype:
                raise ValueError("a single-rank gather copies its shard into a matching output")
            return
        if not self.fits(runtime, source, output):
            raise ValueError("shard gathers require contiguous BF16 16-byte packs gathering into"
                             " the runtime's capacity")
        self.dma = _dma(runtime)
        island = _island_row_shard(runtime)
        self.pcie = self.dma or island
        if self.dma:
            from b12x.comm import pcie
            from b12x.comm.pcie._dma_preparation import query_from_metadata

            self.plan = pcie.plan(
                query_from_metadata(runtime, shape=(runtime.world_size, *source.shape), dtype=source.dtype),
                runtime=runtime,
            )
            ranks = tuple(dist.get_process_group_ranks(runtime.group))

            def prepare(state):
                return PreparedCall(
                    run=lambda: state.run_all_gather(source, out=output.view(runtime.world_size, *source.shape)),
                    output=output, owners=(runtime,),
                )
        elif island:
            from b12x.comm.pcie._owner_preparation import prepared_call

            out_view = output.view(runtime.world_size, *source.shape)
            plan_kwargs = {"transport": "ce_local"} if concurrent else {}
            self.plan = runtime.plan(source, operation="all_gather", out=out_view, **plan_kwargs)
            ranks = tuple(dist.get_process_group_ranks(runtime._island_rs.group))

            def prepare(state):
                call = prepared_call(state, inp=source, out=out_view)
                return PreparedCall(
                    run=call.run,
                    output=output,
                    capture_safe=call.capture_safe,
                    owners=call.owners,
                )
        else:
            self.plan = _roce_plan(runtime, coordinator, "AllReduce.gather_shards")
            ranks = tuple(dist.get_process_group_ranks(runtime._group))

            def prepare(state):
                return PreparedCall(run=lambda: state.gather_shards(source, out=output), output=output,
                                    owners=(runtime, source, output))
        declare(session, (self.plan.request(name=name, prepare_call=prepare,
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    @staticmethod
    def fits(runtime, source, output):
        """True when this runtime gathers `source`'s shard rows into every rank's `output`."""
        if runtime is None or not row_shard_surfaces(runtime) or source.ndim < 1:
            return False
        rows, world = source.shape[0], runtime.world_size
        row_bytes = source.numel() // max(rows, 1) * source.element_size()
        accepts = (runtime.should_all_gather(source) if (_dma(runtime) or _island_row_shard(runtime))
                   else runtime.should_gather_shards(source))
        return (tuple(output.shape) == (rows * world, *source.shape[1:])
                and source.dtype is torch.bfloat16 and output.dtype is source.dtype
                and source.is_contiguous() and output.is_contiguous()
                and row_bytes % 16 == 0
                and 0 < output.numel() * output.element_size() <= _row_shard_capacity_bytes(runtime)
                and accepts)

    def functional(self, *, session):
        """Allocate every rank's gathered rows inside the traced forward."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
        return self

    def __call__(self, source, output=None, *, dependencies=()):
        if output is None:
            output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        if self.single:
            output.copy_(source)
        elif self.pcie:
            if dependencies:
                _pcie_gather_shards_dep(source, output, list(dependencies), self.handle)
            else:
                _pcie_gather_shards(source, output, self.handle)
        else:
            if dependencies:
                raise ValueError("RoCE shard gather does not accept graph-ordering dependencies")
            _gather_shards(source, output, self.handle)
        return output


class PreparedChunkedGatherShards:
    """Shard gathers larger than the runtime's row-shard capacity, gathered in shard-row chunks.

    Each chunk gathers ``world_size`` slices of `step` shard rows into a contiguous message, which is
    copied into the matching rows of every rank's full-size output. When all shard rows fit in one
    collective message, the gather writes directly into `output` without intermediate staging.
    """

    def __init__(self, source, output, *, runtime, session, coordinator, name, examples=None):
        from ignition.preparation.examples import declaration_tensor

        rows, world = source.shape[0], 1 if runtime is None else runtime.world_size
        step = rows
        if runtime is not None:
            # Each rank contributes one shard of the gathered message.
            row_bytes = source.numel() // max(rows, 1) * source.element_size()
            step = max(1, min(rows, _row_shard_capacity_bytes(runtime) // (row_bytes * world)))
        self.bounds = [(start, min(start + step, rows)) for start in range(0, rows, step)]
        if len(self.bounds) == 1:
            self.parts = [PreparedGatherShards(source, output, runtime=runtime,
                                               session=session, coordinator=coordinator, name=name)]
            self.staging_layout = None
        else:
            staging = declaration_tensor(examples, "chunked_gather_shards.staging",
                                         (world * step, *source.shape[1:]), source.dtype, source.device)
            self.parts = [PreparedGatherShards(source[begin:end], staging[:world * (end - begin)], runtime=runtime,
                                               session=session, coordinator=coordinator,
                                               name=f"{name}.{index}")
                          for index, (begin, end) in enumerate(self.bounds)]
            self.staging_layout = (tuple(staging.shape), staging.stride())
        self.output, self.world = output, world

    @staticmethod
    def fits(runtime, source, output):
        """True when this runtime gathers `source`'s shards into `output` in whole shard-row chunks."""
        if runtime is None or not row_shard_surfaces(runtime) or source.ndim < 1 or not source.shape[0]:
            return False
        world = runtime.world_size
        # One shard row per rank is the smallest chunk, so its eligibility covers every larger chunk.
        return (tuple(output.shape) == (source.shape[0] * world, *source.shape[1:])
                and PreparedGatherShards.fits(runtime, source[:1], output[:world]))

    def functional(self, *, session):
        """Allocate the gathered rows and any multi-chunk staging inside the traced forward."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
            for part in self.parts:
                part.functional(session=session)
        return self

    def __call__(self, source, output=None, *, dependencies=()):
        if output is None:
            output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        if self.staging_layout is None:
            return self.parts[0](source, output, dependencies=dependencies)
        shape, stride = self.staging_layout
        staging = torch.empty_strided(shape, stride, dtype=source.dtype, device=source.device)
        window = output.view(self.world, source.shape[0], *output.shape[1:])
        deps = dependencies
        for (begin, end), part in zip(self.bounds, self.parts, strict=True):
            chunk = staging[:self.world * (end - begin)]
            part(source[begin:end], chunk, dependencies=deps)
            window[:, begin:end].copy_(chunk.view(self.world, end - begin, *window.shape[2:]))
            if part.pcie:
                deps = (output,)
        return output


def _dma_all_to_all_exec(owner, source, output):
    owner.runtime.all_to_all(source, out=output, plan=owner.plan)


@torch.library.custom_op("flare::pcie_all_to_all", mutates_args=("output",))
def _pcie_all_to_all(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    _dma_all_to_all_exec(_OWNERS[handle], source, output)


@_pcie_all_to_all.register_fake
def _fake_pcie_all_to_all(source, output, handle):
    pass


# `dependencies` is an unread graph-ordering operand that orders preceding writes before the collective.
@torch.library.custom_op("flare::pcie_all_to_all_dep", mutates_args=("output",))
def _pcie_all_to_all_dep(source: torch.Tensor, output: torch.Tensor,
                         dependencies: list[torch.Tensor], handle: int) -> None:
    _dma_all_to_all_exec(_OWNERS[handle], source, output)


@_pcie_all_to_all_dep.register_fake
def _fake_pcie_all_to_all_dep(source, output, dependencies, handle):
    pass


@torch.library.custom_op("flare::start_all_to_all", mutates_args=("output",))
def _start_all_to_all(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor],
                      handle: int, stream: int) -> None:
    owner = _OWNERS[handle]
    if owner.concurrent_input is not None:
        send = owner.concurrent_input
        if send.data_ptr() != source.data_ptr():
            send.copy_(source)
    else:
        send = source
    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        _dma_all_to_all_exec(owner, send, output)


@_start_all_to_all.register_fake
def _fake_start_all_to_all(source, output, dependencies, handle, stream):
    pass


@torch.library.custom_op("flare::join_all_to_all", mutates_args=("output",))
def _join_all_to_all(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor],
                     dependency: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(source.device).wait_stream(torch.cuda.ExternalStream(stream, device=source.device))


@_join_all_to_all.register_fake
def _fake_join_all_to_all(source, output, dependencies, dependency, stream):
    pass


@dataclass(frozen=True)
class PendingAllToAll:
    source: torch.Tensor
    output: torch.Tensor
    dependencies: list[torch.Tensor]


class ConcurrentAllToAll:
    """Overlap a prepared PCIe DMA all-to-all with local compute on an auxiliary stream."""

    def __init__(self, all_to_all, *, workspace, session, parallel_plans, send_buffer=None):
        from ignition.preparation.scratch import ScratchArena

        plans = tuple(parallel_plans)
        if all_to_all.single or not all_to_all.dma:
            raise ValueError("concurrent all-to-all requires a prepared PCIe DMA channel")
        if not isinstance(workspace, ScratchArena) or not workspace.overlap or not plans:
            raise ValueError("concurrent all-to-all requires an overlap arena and local plans")
        self.all_to_all = all_to_all
        self.all_to_all.concurrent_input = send_buffer
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        for plan in plans:
            workspace.register_parallel(all_to_all.plan, plan)
        register_finalizer(session, self.validate_scratch)

    def validate_scratch(self):
        if self.all_to_all.plan.memory_requirements().scratch_nbytes:
            raise ValueError("concurrent all-to-all requires a plan without caller scratch")

    def functional(self, *, session):
        self.all_to_all.functional(session=session)
        return self

    def start(self, source, output=None, *, dependencies=(), clone=True):
        if output is None:
            output = self.all_to_all.output
        if output is None:
            shape, stride, dtype = self.all_to_all.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        send = source.clone() if (clone and self.all_to_all.concurrent_input is None) else source
        deps = list(dependencies)
        _start_all_to_all(send, output, deps, self.all_to_all.handle, self.stream_handle)
        return PendingAllToAll(send, output, deps)

    def finish(self, pending, dependency):
        _join_all_to_all(pending.source, pending.output, pending.dependencies, dependency, self.stream_handle)
        return pending.output


class PreparedAllToAll:
    """One fixed-shape peer-targeted exchange `[world_size, *slice] -> [world_size, *slice]` via PCIe DMA.

    Each rank sends `source[peer]` to `peer` and receives `peer`'s slice for this rank in
    `output[peer]` without arithmetic. The bootstrap owns the runtime, exchange group, and
    coordinator; every participating rank declares the same operation name and enters
    preparation together.
    """

    def __init__(self, source, output, *, runtime, session, coordinator, name):
        import torch.distributed as dist
        from b12x.comm import pcie
        from b12x.comm.pcie._dma_preparation import query_from_metadata

        self.runtime, self.output = runtime, output
        self.dma = False
        self.concurrent_input = None
        # Traced forwards test this flag, never the runtime: compile workers see a runtime that
        # owns RDMA or IPC resources only as a placeholder.
        self.single = runtime is None
        if runtime is None:
            if source.shape != output.shape or source.dtype != output.dtype:
                raise ValueError("a single-rank all-to-all copies its source into a matching output")
            return
        if not self.fits(runtime, source, output):
            raise ValueError("PCIe DMA all-to-all requires contiguous 16-byte-aligned "
                             "[world_size, *slice] tensors within runtime capacity")
        self.dma = True
        if (source.dtype in (torch.bfloat16, torch.float16, torch.float32)
                and source.numel() % (runtime.world_size * 8) == 0):
            meta_shape, meta_dtype = tuple(source.shape), source.dtype
        else:
            # All-to-all transfers raw 16-byte-aligned bytes; express the byte footprint in BF16
            # for the DMA metadata query without changing the executed tensor dtype.
            slice_bytes = source.numel() * source.element_size() // runtime.world_size
            meta_shape, meta_dtype = (runtime.world_size, slice_bytes // 2), torch.bfloat16
        self.plan = pcie.plan(query_from_metadata(runtime, shape=meta_shape, dtype=meta_dtype),
                              runtime=runtime)
        ranks = tuple(dist.get_process_group_ranks(runtime.group))

        def prepare(state):
            return PreparedCall(run=lambda: state.run_all_to_all(source, out=output), output=output,
                                owners=(runtime,))

        declare(session, (self.plan.request(name=name, prepare_call=prepare,
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    @staticmethod
    def fits(runtime, source, output):
        """True when this PCIe DMA runtime can exchange `source[peer]` into `output[src]`."""
        if runtime is None or not _dma(runtime) or not hasattr(runtime, "should_all_to_all"):
            return False
        if source.ndim < 2 or source.shape[0] != runtime.world_size:
            return False
        slice_bytes = source.numel() // runtime.world_size * source.element_size()
        return (tuple(output.shape) == tuple(source.shape)
                and output.dtype is source.dtype
                and output.device == source.device
                and source.is_contiguous() and output.is_contiguous()
                and slice_bytes > 0 and slice_bytes % 16 == 0
                and source.numel() * source.element_size() <= runtime.max_bytes
                and runtime.should_all_to_all(source))

    def functional(self, *, session):
        """Allocate this rank's all-to-all destination inside the traced forward."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
        return self

    def __call__(self, source, output=None, *, dependencies=()):
        if output is None:
            output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        if self.single:
            if output is not source:
                output.copy_(source)
        elif dependencies:
            _pcie_all_to_all_dep(source, output, list(dependencies), self.handle)
        else:
            _pcie_all_to_all(source, output, self.handle)
        return output


def _pcie_row_to_col_all_to_all_exec(owner, source, output):
    owner.runtime.row_to_col_all_to_all(source, out=output, plan=owner.plan)


@torch.library.custom_op("flare::pcie_row_to_col_all_to_all", mutates_args=("output",))
def _pcie_row_to_col_all_to_all(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    _pcie_row_to_col_all_to_all_exec(_OWNERS[handle], source, output)


@_pcie_row_to_col_all_to_all.register_fake
def _fake_pcie_row_to_col_all_to_all(source, output, handle):
    pass


# `dependencies` is an unread graph-ordering operand that orders preceding writes before the collective.
@torch.library.custom_op("flare::pcie_row_to_col_all_to_all_dep", mutates_args=("output",))
def _pcie_row_to_col_all_to_all_dep(source: torch.Tensor, output: torch.Tensor,
                                    dependencies: list[torch.Tensor], handle: int) -> None:
    _pcie_row_to_col_all_to_all_exec(_OWNERS[handle], source, output)


@_pcie_row_to_col_all_to_all_dep.register_fake
def _fake_pcie_row_to_col_all_to_all_dep(source, output, dependencies, handle):
    pass


@torch.library.custom_op("flare::start_row_to_col_all_to_all", mutates_args=("output",))
def _start_row_to_col_all_to_all(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor],
                                 handle: int, stream: int) -> None:
    owner = _OWNERS[handle]
    if owner.concurrent_input is not None:
        send = owner.concurrent_input
        if send.data_ptr() != source.data_ptr():
            send.copy_(source)
    else:
        send = source
    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        _pcie_row_to_col_all_to_all_exec(owner, send, output)


@_start_row_to_col_all_to_all.register_fake
def _fake_start_row_to_col_all_to_all(source, output, dependencies, handle, stream):
    pass


@torch.library.custom_op("flare::join_row_to_col_all_to_all", mutates_args=("output",))
def _join_row_to_col_all_to_all(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor],
                                dependency: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(source.device).wait_stream(torch.cuda.ExternalStream(stream, device=source.device))


@_join_row_to_col_all_to_all.register_fake
def _fake_join_row_to_col_all_to_all(source, output, dependencies, dependency, stream):
    pass


@dataclass(frozen=True)
class PendingRowToColAllToAll:
    source: torch.Tensor
    output: torch.Tensor
    dependencies: list[torch.Tensor]


class ConcurrentRowToColAllToAll:
    """Overlap a prepared PCIe row-to-column all-to-all with local compute on an auxiliary stream."""

    def __init__(self, all_to_all, *, workspace, session, parallel_plans, send_buffer=None):
        from ignition.preparation.scratch import ScratchArena

        plans = tuple(parallel_plans)
        if all_to_all.single or not all_to_all.pcie:
            raise ValueError("concurrent row-to-column all-to-all requires a prepared PCIe DMA channel")
        if not isinstance(workspace, ScratchArena) or not workspace.overlap or not plans:
            raise ValueError("concurrent row-to-column all-to-all requires an overlap arena and local plans")
        self.all_to_all = all_to_all
        self.all_to_all.concurrent_input = send_buffer
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        for plan in plans:
            workspace.register_parallel(all_to_all.plan, plan)
        register_finalizer(session, self.validate_scratch)

    def validate_scratch(self):
        if self.all_to_all.plan.memory_requirements().scratch_nbytes:
            raise ValueError("concurrent row-to-column all-to-all requires a plan without caller scratch")

    def functional(self, *, session):
        self.all_to_all.functional(session=session)
        return self

    def start(self, source, output=None, *, dependencies=(), clone=True):
        if output is None:
            output = self.all_to_all.output
        if output is None:
            shape, stride, dtype = self.all_to_all.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        send = source.clone() if (clone and self.all_to_all.concurrent_input is None) else source
        deps = list(dependencies)
        _start_row_to_col_all_to_all(send, output, deps, self.all_to_all.handle, self.stream_handle)
        return PendingRowToColAllToAll(send, output, deps)

    def finish(self, pending, dependency):
        _join_row_to_col_all_to_all(pending.source, pending.output, pending.dependencies, dependency, self.stream_handle)
        return pending.output


class PreparedRowToColAllToAll:
    """Repartition BF16 row shards [st, world * c] into column shards [world * st, c]."""

    def __init__(self, source, output, *, runtime, session, coordinator, name):
        import torch.distributed as dist

        self.runtime, self.output = runtime, output
        self.pcie = False
        self.concurrent_input = None
        self.single = runtime is None
        if runtime is None:
            if source.shape != output.shape or source.dtype != output.dtype:
                raise ValueError("a single-rank all-to-all copies its source into a matching output")
            return
        if not self.fits(runtime, source, output):
            raise ValueError("row-to-column all-to-all requires contiguous BF16 matrices within island capacity")
        from b12x.comm.pcie._owner_preparation import prepared_call

        self.pcie = True
        self.plan = runtime.plan(source, operation="row_to_col_all_to_all", out=output)
        ranks = tuple(dist.get_process_group_ranks(runtime._island_rs.group))

        def prepare(state):
            return prepared_call(state, inp=source, out=output)

        declare(session, (self.plan.request(name=name, prepare_call=prepare,
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    @staticmethod
    def fits(runtime, source, output):
        """True when this runtime repartitions `source` [st, world * c] into `output` [world * st, c]."""
        if (runtime is None or not _island_row_shard(runtime)
                or not hasattr(runtime, "should_row_to_col_all_to_all") or source.ndim != 2):
            return False
        st, full_c = source.shape
        world = runtime.world_size
        return (st > 0 and full_c > 0 and full_c % world == 0
                and tuple(output.shape) == (st * world, full_c // world)
                and source.dtype is torch.bfloat16 and output.dtype is source.dtype
                and source.is_contiguous() and output.is_contiguous()
                and runtime.should_row_to_col_all_to_all(source))

    def functional(self, *, session):
        """Allocate this rank's column-shard destination inside the traced forward."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
        return self

    def __call__(self, source, output=None, *, dependencies=()):
        if output is None:
            output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        if self.single:
            output.copy_(source)
        elif dependencies:
            _pcie_row_to_col_all_to_all_dep(source, output, list(dependencies), self.handle)
        else:
            _pcie_row_to_col_all_to_all(source, output, self.handle)
        return output


class PreparedChunkedAllGather:
    """Column gathers of more rows than the channel's batch capacity, gathered in row chunks it accepts.

    Each chunk is a PreparedAllGather over contiguous row views of the source and output, so a channel sized
    for decode can also gather prefill activations; the chunks run in order on one stream.
    """

    def __init__(self, source, output, *, runtime, session, coordinator, name):
        rows = source.shape[0]
        step = rows if runtime is None or _roce(runtime) else max(1, min(rows, runtime.max_batch_size))
        self.bounds = [(start, min(start + step, rows)) for start in range(0, rows, step)]
        self.parts = [PreparedAllGather(source[begin:end], output[begin:end], runtime=runtime, session=session,
                                        coordinator=coordinator, name=name if len(self.bounds) == 1 else f"{name}.{index}")
                      for index, (begin, end) in enumerate(self.bounds)]
        self.output = output

    def functional(self, *, session):
        """Allocate one destination shared by the ordered row chunks."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
            for part in self.parts:
                part.functional(session=session)
        return self

    def __call__(self, source):
        output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        for (begin, end), part in zip(self.bounds, self.parts, strict=True):
            part(source[begin:end], output[begin:end])
        return output

    def run_eager(self, source):
        """Gather on the current stream from inside another eager operation."""
        output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        for (begin, end), part in zip(self.bounds, self.parts, strict=True):
            part.run_eager(source[begin:end], output[begin:end])
        return output


class PreparedChunkedAllReduce:
    """A row-major reduction larger than its PCIe runtime's capacity, reduced in row chunks the runtime accepts.

    Each chunk is a PreparedAllReduce over contiguous row views of the source and output, so a channel sized
    for decode can also reduce prefill activations; the chunks run in order on one stream.
    """

    def __init__(self, source, output, *, runtime, session, coordinator, name):
        rows = source.shape[0]
        step = rows
        if runtime is not None and _roce(runtime):
            row_bytes = source[0].numel() * source.element_size()
            step = min(rows, max(runtime.max_size, runtime.pipe_bytes) // row_bytes)
            if step < 1:
                raise ValueError("one reduction row exceeds the RoCE channel capacity")
        elif runtime is not None:
            from b12x.comm import pcie

            if isinstance(runtime, pcie.DmaAllReduce):
                row_bytes = source[0].numel() * source.element_size()
                step = min(rows, runtime.max_bytes // row_bytes)
                while step > 1 and not runtime.should_allreduce(source[:step]):
                    step -= 1
            else:
                # One-shot pools answer through their stream channel; hierarchical runtimes directly.
                accepts = (runtime.for_stream().should_allreduce
                           if runtime.algorithm == "oneshot"
                           else runtime.should_allreduce)
                while step > 1 and not accepts(source[:step]):
                    step = (step + 1) // 2
        self.bounds = [(start, min(start + step, rows)) for start in range(0, rows, step)]
        self.parts = [PreparedAllReduce(source[begin:end], output[begin:end], runtime=runtime, session=session,
                                        coordinator=coordinator, name=name if len(self.bounds) == 1 else f"{name}.{index}")
                      for index, (begin, end) in enumerate(self.bounds)]
        self.output = output

    def functional(self, *, session):
        """Allocate one destination shared by the ordered row chunks."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
            for part in self.parts:
                part.functional(session=session)
        return self

    def __call__(self, source):
        output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        for (begin, end), part in zip(self.bounds, self.parts, strict=True):
            part(source[begin:end], output[begin:end])
        return output


@dataclass(frozen=True)
class WeightedGroupSource:
    """Ordered weighted groups with physical [rows, groups, hidden] storage.

    Weights are FP32 [rows, groups]; IDs are int32/int64 with the same shape.
    IDs outside [0, id_limit) supply zero values before multiplication.
    """

    values: torch.Tensor
    weights: torch.Tensor
    ids: torch.Tensor
    id_limit: int

    def arguments(self):
        return dict(group_weights=self.weights, group_ids=self.ids, group_id_limit=self.id_limit)


@torch.library.custom_op("flare::pcie_all_reduce_rmsnorm", mutates_args=("residual", "output"))
def _all_reduce_rmsnorm(source: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                       output: torch.Tensor, handle: int, input_addend: torch.Tensor | None = None,
                       group_weights: torch.Tensor | None = None, group_ids: torch.Tensor | None = None) -> None:
    owner = _OWNERS[handle]
    owner.runtime.all_reduce_fused_add_rms_norm(source, residual, weight, owner.eps, plan=owner.plan,
        out=output, residual_out=residual, zero_centered=owner.zero_centered, reduction_dtype=owner.reduction_dtype,
        input_addend=input_addend, group_weights=group_weights, group_ids=group_ids,
        group_id_limit=owner.group_id_limit)


@_all_reduce_rmsnorm.register_fake
def _fake_all_reduce_rmsnorm(source, residual, weight, output, handle, input_addend=None,
                             group_weights=None, group_ids=None):
    pass


class PreparedAllReduceRMSNorm:
    """Reduce an optional rounded source sum, update the residual, and normalize.

    Weighted groups reduce in group order with separate FP32 multiplies and
    adds, then round to source precision. An input addend shares the logical
    [rows, hidden] geometry and dtype; its FP32 addition rounds to source
    precision before tensor-parallel reduction.
    """

    @staticmethod
    def fits(runtime, source):
        from b12x.comm import pcie

        if isinstance(source, WeightedGroupSource):
            from b12x.comm.pcie._oneshot_preparation import _grouped_inputs, _state

            if not isinstance(runtime, pcie.AllReduce) or runtime.algorithm != "oneshot" or not runtime.single_channel:
                return False
            channel = runtime._runtime._prepared_channel_for_stream(None, None)
            if _state(channel).eager_tables is None:
                return False
            source, *_ = _grouped_inputs(source.values, **source.arguments())
        return (isinstance(runtime, pcie.AllReduce) and runtime.algorithm == "oneshot"
                and runtime.single_channel and source.ndim == 2 and source.shape[0] > 0
                and source.is_contiguous() and source.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and source.shape[1] * source.element_size() % 16 == 0
                and 0 < source.numel() * source.element_size() <= runtime.max_size)

    def __init__(self, source, residual, weight, output, *, eps, runtime, session, coordinator, name,
                 zero_centered=False, reduction_dtype=None, input_addend=None):
        import torch.distributed as dist
        from b12x.comm.pcie._oneshot_preparation import _prepare_fused_call

        if not self.fits(runtime, source):
            raise ValueError("fused all-reduce RMSNorm requires a single-channel one-shot runtime within its capacity")
        self.runtime, self.residual, self.weight, self.output = runtime, residual, weight, output
        self.input_addend = input_addend
        grouped = source if isinstance(source, WeightedGroupSource) else None
        group_arguments = {} if grouped is None else grouped.arguments()
        self.group_weights = None if grouped is None else grouped.weights
        self.group_ids = None if grouped is None else grouped.ids
        self.group_id_limit = 0 if grouped is None else grouped.id_limit
        source = source if grouped is None else grouped.values
        self.eps, self.zero_centered, self.reduction_dtype = eps, zero_centered, reduction_dtype
        self.plan = runtime.plan(source, operation="all_reduce_fused_add_rms_norm", residual=residual,
            weight=weight, out=output, residual_out=residual, zero_centered=zero_centered, reduction_dtype=reduction_dtype,
            input_addend=input_addend, **group_arguments)
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: _prepare_fused_call(state, inp=source, residual=residual, weight=weight,
                out=output, residual_out=residual, epsilon=eps, input_addend=input_addend, **group_arguments,
                owners=(runtime, source, residual, weight, output, input_addend, self.group_weights, self.group_ids)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def functional(self, *, session):
        self.output_layout = tuple(self.output.shape), tuple(self.output.stride()), self.output.dtype
        self.output = self.residual = self.input_addend = self.group_weights = self.group_ids = None
        return self

    def __call__(self, source, residual, *, input_addend=None, group_weights=None, group_ids=None):
        output = self.output
        if output is None:
            output = torch.empty_strided(self.output_layout[0], self.output_layout[1], dtype=self.output_layout[2], device=source.device)
        else:
            input_addend, group_weights, group_ids = self.input_addend, self.group_weights, self.group_ids
        _all_reduce_rmsnorm(source, residual, self.weight, output, self.handle, input_addend,
                           group_weights, group_ids)
        return output


@torch.library.custom_op("flare::pcie_all_reduce_add", mutates_args=("residual",))
def _all_reduce_add(source: torch.Tensor, residual: torch.Tensor, handle: int,
                    input_addend: torch.Tensor | None = None,
                    group_weights: torch.Tensor | None = None,
                    group_ids: torch.Tensor | None = None) -> None:
    owner = _OWNERS[handle]
    owner.runtime.all_reduce_fused_add(source, residual, plan=owner.plan,
        input_addend=input_addend, group_weights=group_weights,
        group_ids=group_ids, group_id_limit=owner.group_id_limit)


@_all_reduce_add.register_fake
def _fake_all_reduce_add(source, residual, handle, input_addend=None, group_weights=None, group_ids=None):
    pass


class PreparedAllReduceAdd:
    """Reduce weighted contributions and update every residual row in place."""

    fits = staticmethod(PreparedAllReduceRMSNorm.fits)

    def __init__(self, source, residual, *, runtime, session, coordinator, name,
                 reduction_dtype=None, input_addend=None):
        import torch.distributed as dist
        from b12x.comm.pcie._oneshot_preparation import _prepare_fused_call

        if not self.fits(runtime, source):
            raise ValueError("residual reduction requires a single-channel one-shot runtime within capacity")
        self.runtime, self.input_addend = runtime, input_addend
        grouped = source if isinstance(source, WeightedGroupSource) else None
        arguments = {} if grouped is None else grouped.arguments()
        self.group_weights = None if grouped is None else grouped.weights
        self.group_ids = None if grouped is None else grouped.ids
        self.group_id_limit = 0 if grouped is None else grouped.id_limit
        source = source if grouped is None else grouped.values
        self.plan = runtime.plan(source, operation="all_reduce_fused_add", residual=residual,
            residual_out=residual, reduction_dtype=reduction_dtype, input_addend=input_addend, **arguments)
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: _prepare_fused_call(state, inp=source, residual=residual,
                weight=None, out=None, residual_out=residual, epsilon=0.0,
                input_addend=input_addend, **arguments,
                owners=(runtime, source, residual, input_addend, self.group_weights, self.group_ids)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def functional(self, *, session):
        self.input_addend = self.group_weights = self.group_ids = None
        return self

    def __call__(self, source, residual, *, input_addend=None, group_weights=None, group_ids=None):
        _all_reduce_add(source, residual, self.handle,
                        self.input_addend if input_addend is None else input_addend,
                        self.group_weights if group_weights is None else group_weights,
                        self.group_ids if group_ids is None else group_ids)
        return residual


def _gather_transport_view(tensor):
    # Two BF16 transport lanes preserve every FP32 bit without conversion.
    return tensor.view(torch.bfloat16) if tensor.dtype == torch.float32 else tensor


def _gather_columns(source, output, handle, gate=None):
    owner = _OWNERS[handle]
    runtime = owner.runtime
    if owner.dma:
        runtime.all_gather(source, out=output, plan=owner.plan)
        return
    source, output = _gather_transport_view(source), _gather_transport_view(output)
    if owner.roce:
        runtime.all_gather(source, dim=-1, out=output, plan=owner.plan)
        return
    kwargs = {} if gate is None else {"gate": gate}
    runtime.all_gather_heads(
        source.view(source.shape[0], runtime.heads_per_rank, runtime.query_head_dim),
        out=output.view(output.shape[0], runtime.total_heads, runtime.query_head_dim),
        plan=owner.plan, **kwargs)


@torch.library.custom_op("flare::pcie_all_gather", mutates_args=("output",))
def _all_gather(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor], handle: int) -> None:
    _gather_columns(source, output, handle)


@_all_gather.register_fake
def _fake_all_gather(source, output, dependencies, handle):
    pass


@torch.library.custom_op("flare::pcie_all_gather_gated", mutates_args=("output",))
def _all_gather_gated(source: torch.Tensor, gate: torch.Tensor, output: torch.Tensor,
                      dependencies: list[torch.Tensor], handle: int) -> None:
    _gather_columns(source, output, handle, gate=gate)


@_all_gather_gated.register_fake
def _fake_all_gather_gated(source, gate, output, dependencies, handle):
    pass


@torch.library.custom_op("flare::start_all_gather", mutates_args=("output",))
def _start_all_gather(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor],
                      handle: int, stream: int) -> None:
    # A compiled forward may reuse the input's storage on its main stream as
    # soon as this call returns. Stage it before recording the auxiliary wait
    # unless the caller already supplies a graph-owned unaliased send tensor.
    owner = _OWNERS[handle]
    if owner.concurrent_input is not None:
        send = owner.concurrent_input
        if send.data_ptr() != source.data_ptr():
            send.copy_(source)
    else:
        send = source
    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        _gather_columns(send, output, handle)


@_start_all_gather.register_fake
def _fake_start_all_gather(source, output, dependencies, handle, stream):
    pass


@torch.library.custom_op("flare::join_all_gather", mutates_args=("output",))
def _join_all_gather(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor],
                     dependency: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(source.device).wait_stream(torch.cuda.ExternalStream(stream, device=source.device))


@_join_all_gather.register_fake
def _fake_join_all_gather(source, output, dependencies, dependency, stream):
    pass


@dataclass(frozen=True)
class PendingAllGather:
    source: torch.Tensor
    output: torch.Tensor
    dependencies: list[torch.Tensor]


class ConcurrentAllGather:
    """Overlap a prepared PCIe gather with local work, joining before channel reuse.

    The join retains the source through its asynchronous read and follows a local
    output dependency. A subsequent collective must read the joined output as a
    dependency, keeping communication serialized through compilation.
    """

    def __init__(self, gather, *, workspace, session, parallel_plans, send_buffer=None):
        from ignition.preparation.scratch import ScratchArena

        plans = tuple(parallel_plans)
        if gather.single or gather.roce:
            raise ValueError("concurrent column gather requires a prepared PCIe channel")
        if not isinstance(workspace, ScratchArena) or not workspace.overlap or not plans:
            raise ValueError("concurrent column gather requires an overlap arena and local plans")
        self.gather = gather
        # The channel retains the send buffer for all captured replays.
        # Operations sharing it join before the next write.
        self.gather.concurrent_input = send_buffer
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        for plan in plans:
            workspace.register_parallel(gather.plan, plan)
        register_finalizer(session, self.validate_scratch)

    def validate_scratch(self):
        # PCIe gather scratch belongs to its persistent channel, outside the shared arena.
        if self.gather.plan.memory_requirements().scratch_nbytes:
            raise ValueError("concurrent column gather requires a plan without caller scratch")

    def functional(self, *, session):
        self.gather.functional(session=session)
        return self

    def start(self, source, *, dependencies=()):
        output = self.gather.output
        if output is None:
            shape, stride, dtype = self.gather.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        dependencies = [*self.gather.dependencies, *dependencies]
        _start_all_gather(source, output, dependencies, self.gather.handle, self.stream_handle)
        return PendingAllGather(source, output, dependencies)

    def finish(self, pending, dependency):
        _join_all_gather(pending.source, pending.output, pending.dependencies, dependency, self.stream_handle)
        return pending.output


@triton.jit
def _broadcast_pack(sources, send, OFFSETS: tl.constexpr, SIZES: tl.constexpr,
                    BOOLS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    for binding in tl.static_range(len(SIZES)):
        value = tl.load(sources[binding] + index, index < SIZES[binding], other=0)
        if BOOLS[binding]:
            value = (value != 0).to(tl.uint8)
        tl.store(send + OFFSETS[binding] + index, value, index < SIZES[binding])


@triton.jit
def _broadcast_unpack(received, destinations, OFFSETS: tl.constexpr, SIZES: tl.constexpr,
                      BOOLS: tl.constexpr, ROW_BYTES: tl.constexpr, WORLD: tl.constexpr,
                      BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    for binding in tl.static_range(len(SIZES)):
        offset = OFFSETS[binding] + index
        # Every received row places rank zero before the other rank partitions.
        source = offset // ROW_BYTES * (ROW_BYTES * WORLD) + offset % ROW_BYTES
        value = tl.load(received + source, index < SIZES[binding], other=0)
        if BOOLS[binding]:
            value = (value != 0).to(tl.uint8)
        tl.store(destinations[binding] + index, value, index < SIZES[binding])


# Typed operands preserve bool consumers and storage aliases during compilation.
@torch.library.custom_op("flare::broadcast_pack", mutates_args=("send",))
def _broadcast_pack_call(sources: list[torch.Tensor], send: torch.Tensor, handle: int) -> None:
    operation = _OWNERS[handle]._pack
    operation.compiled[operation.grid](tuple(source.view(torch.uint8).view(-1) for source in sources), send)


@_broadcast_pack_call.register_fake
def _fake_broadcast_pack_call(sources, send, handle):
    pass


@torch.library.custom_op("flare::broadcast_unpack", mutates_args=("destinations",))
def _broadcast_unpack_call(received: torch.Tensor, destinations: list[torch.Tensor], handle: int,
                           group: int) -> None:
    operation = _OWNERS[handle]._unpacks[group]
    operation.compiled[operation.grid](received,
        tuple(destination.view(torch.uint8).view(-1) for destination in destinations))


@_broadcast_unpack_call.register_fake
def _fake_broadcast_unpack_call(received, destinations, handle, group):
    pass


class PreparedBroadcast:
    """Copy rank zero's tensor bytes through a serial all-gather channel.

    Integer decisions and floating-point payloads retain their exact bit patterns.
    The BF16 channel views transport bytes without floating-point arithmetic.
    Overlapping destination views are restored in tensor order.
    """

    def __init__(self, tensors, *, runtime, session, coordinator, name, buffers=None):
        self.single = runtime is None
        if self.single:
            return
        tensors = tuple(tensors)
        if not tensors or any(not tensor.is_contiguous() or tensor.device != tensors[0].device
                              for tensor in tensors):
            raise ValueError("broadcast requires contiguous tensors on one device")
        offsets, size = packed_broadcast_layout((tuple(tensor.shape), tensor.dtype) for tensor in tensors)
        columns = ((size + 15) // 16 * 8 if _roce(runtime)
                   else runtime.heads_per_rank * runtime.query_head_dim)
        rows = (size + 2 * columns - 1) // (2 * columns)

        def allocate(shape):
            return (torch.zeros(shape, dtype=torch.uint8, device=tensors[0].device) if buffers is None
                    else buffers.zeros(shape, torch.uint8))

        self.send = allocate((rows, columns * 2))
        self.received = allocate((rows, columns * 2 * runtime.world_size))
        self.gather = PreparedAllGather(self.send.view(torch.bfloat16), self.received.view(torch.bfloat16),
            runtime=runtime, session=session, coordinator=coordinator, name=name)

        self._tensors = tensors
        self._byte_views = tuple(tensor.view(torch.uint8).view(-1) for tensor in tensors)
        self._offsets = tuple(offsets)
        self._sizes = tuple(tensor.numel() * tensor.element_size() for tensor in tensors)
        self._bools = tuple(tensor.dtype == torch.bool for tensor in tensors)
        groups, group, ranges = [], [], []
        for index, tensor in enumerate(tensors):
            begin, size = tensor.data_ptr(), self._sizes[index]
            end = begin + size
            if size and any(begin < other_end and other_begin < end for other_begin, other_end in ranges):
                groups.append(tuple(group))
                group, ranges = [], []
            group.append(index)
            if size:
                ranges.append((begin, end))
        groups.append(tuple(group))
        self._unpack_groups = tuple(groups)
        self._row_bytes = columns * 2
        self._world = runtime.world_size
        register_finalizer(session, self.prepare)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def prepare(self):
        from ignition.preparation.triton import prepare_triton

        if self._byte_views is None:
            return
        options = dict(OFFSETS=self._offsets, SIZES=self._sizes, BOOLS=self._bools, BLOCK=256, num_warps=4)
        grid = (triton.cdiv(max(self._sizes), 256), 1, 1)
        self._pack = prepare_triton(_broadcast_pack, (self._byte_views, self.send), grid, **options)
        self._unpacks = tuple(prepare_triton(_broadcast_unpack,
            (self.received, tuple(self._byte_views[index] for index in group)),
            (triton.cdiv(max(self._sizes[index] for index in group), 256), 1, 1),
            OFFSETS=tuple(self._offsets[index] for index in group),
            SIZES=tuple(self._sizes[index] for index in group),
            BOOLS=tuple(self._bools[index] for index in group),
            ROW_BYTES=self._row_bytes, WORLD=self._world, BLOCK=256, num_warps=4)
            for group in self._unpack_groups)

    def functional(self, *, session):
        """Accept runtime payloads and allocate transport scratch in the forward."""
        if not self.single and not hasattr(self, "send_shape"):
            self.send_shape = tuple(self.send.shape)
            self.gather.functional(session=session)
            register_finalizer(session, self._release_buffers)
        return self

    def _release_buffers(self):
        self.send = self.received = self._tensors = self._byte_views = None
        self._pack.args = ()
        for operation in self._unpacks:
            operation.args = ()

    def __call__(self, tensors=None):
        if self.single:
            return
        tensors = self._tensors if tensors is None else tuple(tensors)
        send = self.send
        if send is None:
            send = torch.zeros(self.send_shape, dtype=torch.uint8, device=tensors[0].device)
        _broadcast_pack_call(list(tensors), send, self.handle)
        received = self.gather(send.view(torch.bfloat16)).view(torch.uint8)
        # Aliased destinations keep their field order across separate stream-ordered launches.
        # Disjoint fields share one launch, including the ordinary sampling payload.
        for index, group in enumerate(self._unpack_groups):
            _broadcast_unpack_call(received, [tensors[field] for field in group], self.handle, index)


class PreparedAllGather:
    """Gather contiguous column partitions through a caller-owned b12x channel.

    Each output row contains rank zero's columns, then rank one's, and so on.
    Operations sharing the channel must execute serially on every rank.
    Dependencies order storage writes before the collective publishes them to peers.
    """

    def __init__(self, source, output, *, runtime, session, coordinator, name, dependencies=(), has_gate=False):
        import torch.distributed as dist
        from b12x.comm import pcie

        self.runtime, self.output = runtime, output
        self.dependencies = list(dependencies)
        self.gated = False
        self.roce = False
        self.dma = False
        self.concurrent_input = None
        # Traced forwards test this flag, never the runtime: compile workers see a runtime that
        # owns RDMA or IPC resources only as a placeholder.
        self.single = runtime is None
        if runtime is None:
            if source.shape != output.shape or source.dtype != output.dtype:
                raise ValueError("a single-rank gather copies its source into a matching output")
            return
        self.dma = isinstance(runtime, pcie.DmaAllReduce)
        if self.dma:
            if (not runtime.should_all_gather(source)
                    or tuple(output.shape) != (runtime.world_size, *source.shape)
                    or output.dtype != source.dtype or output.device != source.device
                    or not output.is_contiguous()):
                raise ValueError("DMA all-gather requires contiguous aligned input within shard capacity "
                                 "and matching [world, *source] output")
            from b12x.comm.pcie._dma_preparation import query_from_metadata

            self.plan = pcie.plan(query_from_metadata(runtime, shape=tuple(output.shape), dtype=output.dtype),
                                  runtime=runtime)
            ranks = tuple(dist.get_process_group_ranks(runtime.group))
            declare(session, (self.plan.request(name=name,
                prepare_call=lambda state: PreparedCall(
                    run=lambda: state.run_all_gather(source, out=output),
                    output=output, owners=(runtime,)),
                collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
            self.handle = next(_HANDLES)
            _OWNERS[self.handle] = self
            return
        if (source.ndim != 2 or source.dtype not in (torch.bfloat16, torch.float32, torch.float8_e4m3fn)
                or output.dtype != source.dtype or not source.is_contiguous() or not output.is_contiguous()):
            raise ValueError("column gathers require contiguous matching BF16, FP32 or E4M3 matrices")
        transport_source = _gather_transport_view(source)
        transport_output = _gather_transport_view(output)
        self.roce = _roce(runtime)
        if self.roce:
            if (transport_source.dtype != torch.bfloat16
                    or transport_output.shape != (source.shape[0], transport_source.shape[1] * runtime.world_size)
                    or transport_source.shape[1] * 2 % 16
                    or not 0 < transport_source.numel() * 2 <= runtime.max_gather_bytes):
                raise ValueError("RoCE gathers require BF16 or FP32 column partitions within the byte capacity")
            self.plan = _roce_plan(runtime, coordinator, "AllReduce.all_gather")
            ranks = tuple(dist.get_process_group_ranks(runtime._group))
            # Priming runs before the plan is published, so it launches the materialized gather directly.
            declare(session, (self.plan.request(name=name,
                prepare_call=lambda state: PreparedCall(
                    run=lambda: runtime._run_prepared_all_gather(transport_source, prepared=state, dim=-1, out=transport_output),
                    output=output, owners=(runtime,)),
                collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
            self.handle = next(_HANDLES)
            _OWNERS[self.handle] = self
            return
        if (runtime.world_size < 2 or source.ndim != 2
                or source.dtype not in (torch.bfloat16, torch.float32, torch.float8_e4m3fn) or output.dtype != source.dtype
                or source.device != runtime.device or output.device != source.device
                or not source.is_contiguous() or not output.is_contiguous()
                or not 0 < source.shape[0] <= runtime.max_batch_size
                or transport_source.shape[1] != runtime.heads_per_rank * runtime.query_head_dim
                or output.shape != (source.shape[0], source.shape[1] * runtime.world_size)):
            raise ValueError("all-gather requires contiguous BF16, FP32 or E4M3 column partitions matching its collective "
                             "channel")
        self.runtime, self.output = runtime, output
        call = {"local_input": transport_source.view(source.shape[0], runtime.heads_per_rank, runtime.query_head_dim),
                "out": transport_output.view(output.shape[0], runtime.total_heads, runtime.query_head_dim)}
        if has_gate:
            call["has_gate"] = True
            self.gated = True
        query = pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_heads", call=call)
        self.plan = pcie.plan(query, runtime=runtime)
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        one_gate = torch.ones((1,), dtype=torch.int32, device=source.device) if self.gated else None
        run_call = {k: v for k, v in call.items() if k != "has_gate"}
        prepare_call_kwargs = run_call if one_gate is None else {**run_call, "gate": one_gate}
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.run(**prepare_call_kwargs), output=output,
                                                    owners=(runtime,) if one_gate is None else (runtime, one_gate)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def functional(self, *, session):
        """Allocate the gathered destination inside the traced forward."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
        self.dependencies = []
        return self

    def __call__(self, source, output=None, *, dependencies=(), gate=None):
        if output is None:
            output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        if self.single:
            output.copy_(source)
        elif self.gated and gate is not None:
            _all_gather_gated(source, gate, output, [*self.dependencies, *dependencies], self.handle)
        else:
            _all_gather(source, output, [*self.dependencies, *dependencies], self.handle)
        return output

    def run_eager(self, source, output=None):
        """Gather on the current stream from inside another eager operation."""
        if output is None:
            output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        if self.single:
            output.copy_(source)
        else:
            _gather_columns(source, output, self.handle)
        return output


@torch.library.custom_op("flare::all_gather_varlen", mutates_args=("output",))
def _all_gather_varlen(source: torch.Tensor, counts: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    # Rank-local slicing stays inside the opaque call so every rank shares the compiled input layout.
    owner.runtime.all_gather_varlen(source[owner.local_rank], counts, out=output, plan=owner.plan)


@_all_gather_varlen.register_fake
def _fake_all_gather_varlen(source, counts, output, handle):
    pass


class PreparedVarlenGather:
    """Gather byte prefixes of every rank's row of `output` [world, capacity] to every rank.

    Each rank writes its contribution into its own row in place; `counts` is a [world] int64 device
    tensor of each rank's bytes, a multiple of 16, holding the same values on every rank and free to
    change between replays. Row r then holds rank r's first counts[r] bytes on every rank; bytes
    past them keep their values. Operations sharing the runtime must execute serially on every rank.

    `runtime` is a RoCEnante runtime across hosts. On one host it is a PCIe all-gather runtime created by
    create_varlen_gather_runtime, which admits two or four ranks.
    """

    def __init__(self, output, counts, *, runtime, session, coordinator, name):
        import torch.distributed as dist

        roce = _roce(runtime)
        ranks = tuple(dist.get_process_group_ranks(runtime._group if roce else runtime.exchange_group))
        if (output.shape[0] != len(ranks) or not output.is_contiguous() or counts.shape != (len(ranks),)
                or counts.dtype != torch.int64 or output[0].numel() * output.element_size() % 16):
            raise ValueError("variable-length gathers require one contiguous 16-byte-multiple row per rank")
        self.runtime, self.output, self.counts = runtime, output, counts
        # A rank's own row is its source, so the gather leaves it in place.
        self.local_rank = ranks.index(dist.get_rank())
        self.source = output[self.local_rank]
        if roce:
            self.plan = _roce_plan(runtime, coordinator, "AllReduce.all_gather_varlen")

            def prepare_call(state):
                return PreparedCall(
                    run=lambda: runtime._run_prepared_all_gather_varlen(self.source, counts, prepared=state, out=output),
                    output=output, owners=(runtime, self.source, counts, output))
        else:
            from b12x.comm import pcie

            query = pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_varlen",
                call={"out": output, "counts": counts, "bank_bytes": VARLEN_BANK_BYTES})
            self.plan = pcie.plan(query, runtime=runtime)

            def prepare_call(state):
                return PreparedCall(run=lambda: state.run(out=output, counts=counts),
                    output=output, owners=(runtime, counts, output))
        declare(session, (self.plan.request(name=name, prepare_call=prepare_call,
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def functional(self, *, session):
        register_finalizer(session, self._release_examples)
        return self

    def _release_examples(self):
        self.source = self.counts = self.output = None

    def __call__(self, output=None, counts=None):
        output = self.output if output is None else output
        counts = self.counts if counts is None else counts
        _all_gather_varlen(output, counts, output, self.handle)
        return output
