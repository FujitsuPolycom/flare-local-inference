"""Forward objects and minimal direct b12x gather wrappers of helios/tests/gather_core_fixture.py."""

import itertools
import weakref

import torch

_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()
VARLEN_BANK_BYTES = 4 << 20


def _transport_view(tensor):
    return tensor.view(torch.bfloat16) if tensor.dtype == torch.float32 else tensor


@torch.library.custom_op("flare_fixture::all_gather", mutates_args=("output",))
def _all_gather(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    rt = owner.runtime
    rt.all_gather_heads(
        _transport_view(source).view(source.shape[0], rt.heads_per_rank, rt.query_head_dim),
        out=_transport_view(output).view(output.shape[0], rt.total_heads, rt.query_head_dim),
        plan=owner.plan,
    )


@_all_gather.register_fake
def _fake_all_gather(source, output, handle):
    pass


@torch.library.custom_op("flare_fixture::all_gather_varlen", mutates_args=("output",))
def _all_gather_varlen(source: torch.Tensor, counts: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.all_gather_varlen(source[owner.local_rank], counts, out=output, plan=owner.plan)


@_all_gather_varlen.register_fake
def _fake_all_gather_varlen(source, counts, output, handle):
    pass


class FixtureAllGather:
    def __init__(self, source, output, *, runtime, session, coordinator, name):
        import torch.distributed as dist
        from b12x.comm import pcie
        from b12x.preparation import CollectiveRequirement, PreparedCall
        from ignition.preparation.tuning import declare

        self.runtime, self.output = runtime, output
        ts, to = _transport_view(source), _transport_view(output)
        call = {
            "local_input": ts.view(source.shape[0], runtime.heads_per_rank, runtime.query_head_dim),
            "out": to.view(output.shape[0], runtime.total_heads, runtime.query_head_dim),
        }
        query = pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_heads", call=call)
        self.plan = pcie.plan(query, runtime=runtime)
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        declare(session, (self.plan.request(
            name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.run(**call), output=output, owners=(runtime,)),
            collective=CollectiveRequirement(key=name, ranks=ranks),
        ),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self, source):
        _all_gather(source, self.output, self.handle)
        return self.output


class FixtureVarlenGather:
    def __init__(self, output, counts, *, runtime, session, coordinator, name):
        import torch.distributed as dist
        from b12x.comm import pcie
        from b12x.preparation import CollectiveRequirement, PreparedCall
        from ignition.preparation.tuning import declare

        self.runtime, self.output, self.counts = runtime, output, counts
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        self.local_rank = ranks.index(dist.get_rank())
        query = pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_varlen",
            call={"out": output, "counts": counts, "bank_bytes": VARLEN_BANK_BYTES})
        self.plan = pcie.plan(query, runtime=runtime)
        declare(session, (self.plan.request(
            name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.run(out=output, counts=counts),
                output=output, owners=(runtime, counts, output)),
            collective=CollectiveRequirement(key=name, ranks=ranks),
        ),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self, output=None, counts=None):
        output = self.output if output is None else output
        counts = self.counts if counts is None else counts
        _all_gather_varlen(output, counts, output, self.handle)
        return output


class GatherForward:
    def __init__(self, operation, source):
        self.operation, self.source = operation, source

    def __call__(self):
        return self.operation(self.source)


class RecordForward:
    """Pack, gather and scatter one window sequence of latent records, as a prefill layer's ring does."""

    def __init__(self, windows, rank_index, world, cache, table, output, window):
        self.windows, self.rank_index, self.world = windows, rank_index, world
        self.cache, self.table, self.output, self.window = cache, table, output, window

    def __call__(self):
        slab = torch.zeros_like(self.output)
        packed = torch.full((self.world, self.window * 528), 0xA5, dtype=torch.uint8, device=self.cache.device)
        cache_records = self.cache.view(-1, 528)
        end = 128
        span = 4 * self.world
        local_offsets = torch.arange(self.window, dtype=torch.int64, device=self.cache.device)
        owners = torch.arange(self.world, dtype=torch.int64, device=self.cache.device)
        for first, gather in self.windows:
            local = first + local_offsets
            pages = self.table[(local // 256).clamp(max=self.table.shape[0] - 1)].long()
            physical = pages * 256 + (local % 256)
            rank_records = cache_records.index_select(0, physical)
            count = (end // span) * 4 + (end % span - self.rank_index * 4).clamp(0, 4)
            live = (local < count).unsqueeze(-1)
            packed_view = packed.view(self.world, self.window, 528)
            existing = packed_view.index_select(0, self.rank_index).squeeze(0)
            packed_view.index_copy_(0, self.rank_index, torch.where(live, rank_records, existing).unsqueeze(0))
            # Counts deliberately have no data dependency on the packed cache records.
            counts = torch.full((self.world,), min(max(32 - first, 0), self.window) * 528,
                dtype=torch.int64, device=self.cache.device)
            gather(output=packed, counts=counts)
            gathered = packed.view(self.world, self.window, 528)
            tokens = (local[None, :] // 4) * span + owners[:, None] * 4 + (local[None, :] % 4)
            flat_tokens = tokens.reshape(-1)
            valid = flat_tokens < end
            safe_tokens = flat_tokens.clamp(0, slab.shape[0] - 1)
            slab_target = slab.index_select(0, safe_tokens)
            slab.index_copy_(0, safe_tokens, torch.where(valid.unsqueeze(-1), gathered.reshape(-1, 528), slab_target))
        self.output.copy_(slab)
