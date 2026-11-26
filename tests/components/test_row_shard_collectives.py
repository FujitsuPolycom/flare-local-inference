# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Row-sharded reductions and shard gathers keep the all-reduce path's layout and eligibility."""

from types import SimpleNamespace

import pytest
import torch


def _surfaces(monkeypatch, *, abi=11, roce=True):
    """Enable the ABI 11 row-shard surfaces on the module the wrappers test at run time."""
    import b12x.comm.roce._proxy as proxy
    from ignition.components import collectives

    monkeypatch.setattr(proxy, "ABI_VERSION", abi)
    monkeypatch.setattr(collectives, "_roce", lambda runtime: roce)


def _runtime(*, world=4, pipe_bytes=1 << 26, gather=None):
    calls = []

    def reduce_scatter(inp, *, out, plan):
        calls.append(("reduce_scatter", tuple(inp.shape), tuple(out.shape)))

    def gather_shards(inp, *, out, plan):
        calls.append(("gather_shards", tuple(inp.shape), tuple(out.shape)))

    return SimpleNamespace(world_size=world, pipe_bytes=pipe_bytes, _group=object(),
        should_reduce_scatter=lambda inp: True, should_gather_shards=lambda inp: True,
        reduce_scatter=reduce_scatter, gather_shards=gather or gather_shards, calls=calls)


@pytest.fixture
def prepared(monkeypatch):
    """Prepare row-shard operations without a device or a preparation session."""
    from ignition.components import collectives

    _surfaces(monkeypatch)
    surfaces, plan = [], SimpleNamespace(request=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(collectives, "_roce_plan",
                        lambda runtime, coordinator, surface: surfaces.append(surface) or plan)
    monkeypatch.setattr(collectives, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1, 2, 3])
    return surfaces, plan


@pytest.mark.parametrize("abi,roce,drop,expected", [
    (11, True, None, True), (10, True, None, False), (11, False, None, False),
    (11, True, "should_reduce_scatter", False), (11, True, "should_gather_shards", False)])
def test_row_shard_surfaces_require_abi_eleven_and_both_runtime_surfaces(monkeypatch, abi, roce, drop, expected):
    from ignition.components import collectives

    _surfaces(monkeypatch, abi=abi, roce=roce)
    runtime = _runtime()
    if drop is not None:
        delattr(runtime, drop)
    assert collectives.row_shard_surfaces(runtime) is expected


def test_reduce_scatter_fits_splits_whole_row_packs_into_the_shard(monkeypatch):
    from ignition.components.collectives import PreparedReduceScatter

    _surfaces(monkeypatch)
    runtime = _runtime()
    shard = torch.zeros(2, 5120, dtype=torch.bfloat16)
    assert PreparedReduceScatter.fits(runtime, torch.zeros(8, 5120, dtype=torch.bfloat16), shard)
    # A four-stream residual shards along its rows, not its trailing dims.
    assert PreparedReduceScatter.fits(runtime, torch.zeros(8, 4, 5120, dtype=torch.bfloat16),
                                      torch.zeros(2, 4, 5120, dtype=torch.bfloat16))
    assert not PreparedReduceScatter.fits(runtime, torch.zeros(8, 5120, dtype=torch.bfloat16), torch.zeros(4, 5120, dtype=torch.bfloat16))
    assert not PreparedReduceScatter.fits(runtime, torch.zeros(6, 5120, dtype=torch.bfloat16), shard)
    assert not PreparedReduceScatter.fits(runtime, torch.zeros(8, 5120, dtype=torch.bfloat16)[:, :1], torch.zeros(2, 1, dtype=torch.bfloat16))
    assert not PreparedReduceScatter.fits(runtime, torch.zeros(8, 7, dtype=torch.bfloat16), torch.zeros(2, 7, dtype=torch.bfloat16))
    assert not PreparedReduceScatter.fits(runtime, torch.zeros(8, 5120), torch.zeros(2, 5120))
    assert not PreparedReduceScatter.fits(_runtime(pipe_bytes=1 << 15), torch.zeros(8, 5120, dtype=torch.bfloat16), shard)


def test_gather_shards_fits_gathers_every_shard_row_and_chunked_gathers_accept_one_row_per_rank(monkeypatch):
    from ignition.components.collectives import PreparedChunkedGatherShards, PreparedGatherShards

    _surfaces(monkeypatch)
    runtime = _runtime()
    shard = torch.zeros(2, 4, 5120, dtype=torch.bfloat16)
    assert PreparedGatherShards.fits(runtime, shard, torch.zeros(8, 4, 5120, dtype=torch.bfloat16))
    assert not PreparedGatherShards.fits(runtime, shard, torch.zeros(4, 4, 5120, dtype=torch.bfloat16))
    assert not PreparedGatherShards.fits(_runtime(pipe_bytes=1 << 18), shard,
                                         torch.zeros(8, 4, 5120, dtype=torch.bfloat16))
    assert PreparedChunkedGatherShards.fits(runtime, shard, torch.zeros(8, 4, 5120, dtype=torch.bfloat16))
    # One shard row per rank is the smallest chunk, so nothing below its message fits.
    assert not PreparedChunkedGatherShards.fits(_runtime(pipe_bytes=4 * 40960 - 1), shard,
                                                torch.zeros(8, 4, 5120, dtype=torch.bfloat16))


@pytest.mark.parametrize("wrapper", ["PreparedReduceScatter", "PreparedGatherShards"])
def test_single_rank_row_shard_operations_copy_the_source(monkeypatch, wrapper):
    from ignition.components import collectives

    operation = getattr(collectives, wrapper)
    source = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    output = torch.empty_like(source)
    prepared = operation(source, output, runtime=None, session=None, coordinator=None, name="row_shard")
    assert prepared.single
    assert prepared(source) is output
    assert torch.equal(source, output)
    with pytest.raises(ValueError, match="single-rank"):
        operation(source, torch.empty(4, 4, dtype=torch.bfloat16), runtime=None, session=None,
                  coordinator=None, name="row_shard")


def test_prepared_row_shard_operations_request_their_surface_and_launch_the_runtime(prepared):
    from ignition.components.collectives import PreparedGatherShards, PreparedReduceScatter

    surfaces, plan = prepared
    runtime = _runtime()
    source = torch.zeros(8, 5120, dtype=torch.bfloat16)
    shard = torch.zeros(2, 5120, dtype=torch.bfloat16)
    reduce = PreparedReduceScatter(source, shard, runtime=runtime, session=object(),
                                   coordinator=SimpleNamespace(hosts=()), name="layer.attention_reduce")
    gather_output = torch.zeros(8, 5120, dtype=torch.bfloat16)
    gather = PreparedGatherShards(shard, gather_output, runtime=runtime, session=object(),
                                  coordinator=SimpleNamespace(hosts=()), name="layer.attention_gather")
    assert surfaces == ["AllReduce.reduce_scatter", "AllReduce.gather_shards"]
    assert reduce.plan is plan and gather.plan is plan
    assert reduce(source) is shard and gather(shard) is gather_output
    assert runtime.calls == [("reduce_scatter", (8, 5120), (2, 5120)),
                             ("gather_shards", (2, 5120), (8, 5120))]


def test_single_message_chunked_gather_writes_directly_without_staging(prepared):
    from ignition.components.collectives import PreparedChunkedGatherShards

    rows, world, columns = 2, 2, 8
    row_bytes = columns * 2
    destinations = []

    def gather(inp, *, out, plan):
        destinations.append(out.data_ptr())
        out.view(world, rows, columns)[:] = inp

    runtime = _runtime(world=world, pipe_bytes=row_bytes * rows * world, gather=gather)
    shard = torch.arange(rows * columns, dtype=torch.bfloat16).reshape(rows, columns)
    output = torch.zeros(rows * world, columns, dtype=torch.bfloat16)
    operation = PreparedChunkedGatherShards(shard, output, runtime=runtime, session=object(),
                                            coordinator=SimpleNamespace(hosts=()), name="layer.residual_gather")
    assert operation.bounds == [(0, 2)]
    assert operation.staging_layout is None
    assert operation(shard) is output
    assert destinations == [output.data_ptr()]
    for rank in range(world):
        assert torch.equal(output.view(world, rows, columns)[rank], shard)

    assert operation.functional(session=object()) is operation
    assert operation.output is None
    first = operation(shard)
    second = operation(shard)
    assert first.shape == second.shape == (rows * world, columns) and first.dtype is torch.bfloat16
    assert first.data_ptr() != second.data_ptr()
    assert destinations[-2:] == [first.data_ptr(), second.data_ptr()]


def test_chunked_gather_scatters_each_chunk_into_its_rank_major_window(prepared):
    from ignition.components.collectives import PreparedChunkedGatherShards

    rows, world, columns = 5, 2, 8
    row_bytes = columns * 2
    # Two shard rows per gathered message, so the five shard rows arrive in three chunks.
    runtime = _runtime(world=world, pipe_bytes=row_bytes * 2 * world)

    def gather(inp, *, out, plan):
        # A gather publishes rank-major rows, so a single-rank test writes the shard to each rank.
        out.view(world, -1, columns)[:] = inp
    runtime.gather_shards = gather
    shard = torch.arange(rows * columns, dtype=torch.bfloat16).reshape(rows, columns)
    output = torch.zeros(rows * world, columns, dtype=torch.bfloat16)
    operation = PreparedChunkedGatherShards(shard, output, runtime=runtime, session=object(),
                                            coordinator=SimpleNamespace(hosts=()), name="layer.residual_gather")
    assert operation.bounds == [(0, 2), (2, 4), (4, 5)]
    assert operation.staging_layout is not None
    assert operation(shard) is output
    for rank in range(world):
        assert torch.equal(output.view(world, rows, columns)[rank], shard)


def test_a_functional_chunked_gather_allocates_its_staging_and_output_in_the_forward(prepared):
    """A captured graph owns every operand, so the wrapper declares shapes and allocates on call."""
    from ignition.components.collectives import PreparedChunkedGatherShards

    rows, world, columns = 5, 2, 8
    row_bytes = columns * 2

    def gather(inp, *, out, plan):
        out.view(world, -1, columns)[:] = inp

    runtime = _runtime(world=world, pipe_bytes=row_bytes * 2 * world, gather=gather)
    shard = torch.arange(rows * columns, dtype=torch.bfloat16).reshape(rows, columns)
    operation = PreparedChunkedGatherShards(shard, torch.zeros(rows * world, columns, dtype=torch.bfloat16),
                                            runtime=runtime, session=object(),
                                            coordinator=SimpleNamespace(hosts=()), name="layer.residual_gather")
    assert operation.functional(session=object()) is operation
    assert operation.output is None
    first = operation(shard)
    second = operation(shard)
    assert first.shape == second.shape == (rows * world, columns) and first.dtype is torch.bfloat16
    assert first.data_ptr() != second.data_ptr()
    for rank in range(world):
        assert torch.equal(second.view(world, rows, columns)[rank], shard)


class _FakeDmaAllReduce:
    """Stand-in for `b12x.comm.pcie.DmaAllReduce` in CPU unit tests."""

    def __init__(self, *, world=4, max_bytes=1 << 26, accept_rs=True, accept_ag=True):
        self.world_size = world
        self.max_bytes = max_bytes
        self.group = object()
        self.calls = []
        self._accept_rs = accept_rs
        self._accept_ag = accept_ag

    def should_reduce_scatter(self, inp):
        return self._accept_rs

    def should_all_gather(self, inp):
        return self._accept_ag

    def reduce_scatter(self, inp, *, out, plan):
        self.calls.append(("reduce_scatter", tuple(inp.shape), tuple(out.shape), out.data_ptr(), plan))
        out.copy_(inp.view(self.world_size, *out.shape).sum(dim=0))

    def all_gather(self, inp, *, out, plan):
        self.calls.append(("all_gather", tuple(inp.shape), tuple(out.shape), out.data_ptr(), plan))
        for rank in range(self.world_size):
            out[rank].copy_(inp + rank)


@pytest.fixture
def prepared_dma(monkeypatch):
    """Patch PCIe DMA planning and runtime isinstance checks for CPU execution."""
    import b12x.comm.pcie as pcie
    import b12x.comm.pcie._dma_preparation as dma_prep
    from ignition.components import collectives

    monkeypatch.setattr(pcie, "DmaAllReduce", _FakeDmaAllReduce)
    queries = []
    plans = []
    declarations = []

    def query_from_metadata(runtime, *, shape, dtype):
        query = SimpleNamespace(runtime=runtime, shape=tuple(shape), dtype=dtype)
        queries.append(query)
        return query

    def make_plan(query, *, runtime):
        plan = SimpleNamespace(
            query=query,
            scratch_nbytes=0,
            memory_requirements=lambda: SimpleNamespace(scratch_nbytes=plan.scratch_nbytes),
            request=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        plans.append(plan)
        return plan

    monkeypatch.setattr(dma_prep, "query_from_metadata", query_from_metadata)
    monkeypatch.setattr(pcie, "plan", make_plan)
    monkeypatch.setattr(collectives, "declare", lambda session, requests, coordinator=None: declarations.extend(requests))
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1, 2, 3])
    return queries, plans, declarations


@pytest.mark.parametrize("drop,expected", [
    (None, True),
    ("should_reduce_scatter", False),
    ("should_all_gather", False),
])
def test_dma_row_shard_surfaces_require_reduce_scatter_and_all_gather(monkeypatch, prepared_dma, drop, expected):
    import b12x.comm.roce._proxy as proxy
    from ignition.components import collectives

    # PCIe DMA eligibility is independent of the multi-host RoCE proxy ABI version.
    monkeypatch.setattr(proxy, "ABI_VERSION", 10)
    runtime = _FakeDmaAllReduce()
    if drop is not None:
        monkeypatch.delattr(_FakeDmaAllReduce, drop)
    assert collectives.row_shard_surfaces(runtime) is expected


def test_dma_row_shard_fits_enforces_max_bytes_and_runtime_predicates(prepared_dma):
    from ignition.components.collectives import (
        PreparedChunkedGatherShards, PreparedGatherShards, PreparedReduceScatter,
    )

    source = torch.zeros(8, 5120, dtype=torch.bfloat16)
    shard = torch.zeros(2, 5120, dtype=torch.bfloat16)
    full_bytes = source.numel() * source.element_size()
    one_row_gather_bytes = 4 * 5120 * 2

    assert PreparedReduceScatter.fits(_FakeDmaAllReduce(max_bytes=full_bytes), source, shard)
    assert not PreparedReduceScatter.fits(_FakeDmaAllReduce(max_bytes=full_bytes - 1), source, shard)
    assert not PreparedReduceScatter.fits(_FakeDmaAllReduce(max_bytes=full_bytes, accept_rs=False), source, shard)

    assert PreparedGatherShards.fits(_FakeDmaAllReduce(max_bytes=full_bytes), shard, source)
    assert not PreparedGatherShards.fits(_FakeDmaAllReduce(max_bytes=full_bytes - 1), shard, source)
    assert not PreparedGatherShards.fits(_FakeDmaAllReduce(max_bytes=full_bytes, accept_ag=False), shard, source)

    assert PreparedChunkedGatherShards.fits(_FakeDmaAllReduce(max_bytes=one_row_gather_bytes), shard, source)
    assert not PreparedChunkedGatherShards.fits(_FakeDmaAllReduce(max_bytes=one_row_gather_bytes - 1), shard, source)
    assert not PreparedChunkedGatherShards.fits(
        _FakeDmaAllReduce(max_bytes=one_row_gather_bytes, accept_ag=False), shard, source
    )


def test_dma_dispatch_and_dependency_custom_op_selection(prepared_dma):
    from torch.fx.experimental.proxy_tensor import make_fx
    from ignition.components.collectives import (
        PreparedChunkedGatherShards, PreparedGatherShards, PreparedReduceScatter,
    )

    queries, plans, declarations = prepared_dma
    runtime = _FakeDmaAllReduce(world=4)
    # Use 8-column BF16 rows so each row is 16 bytes.
    source = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8)
    shard = torch.zeros(2, 8, dtype=torch.bfloat16)
    gathered = torch.zeros(8, 8, dtype=torch.bfloat16)

    reduce = PreparedReduceScatter(source, shard, runtime=runtime, session=object(),
                                   coordinator=SimpleNamespace(hosts=()), name="dma.rs")
    gather = PreparedGatherShards(shard, gathered, runtime=runtime, session=object(),
                                  coordinator=SimpleNamespace(hosts=()), name="dma.gs")
    assert reduce.dma and gather.dma
    assert [(q.shape, q.dtype) for q in queries] == [((8, 8), torch.bfloat16), ((4, 2, 8), torch.bfloat16)]

    # Verify prepare_call binds run_reduce_scatter and run_all_gather over the rank-major view.
    state_calls = []
    fake_state = SimpleNamespace(
        run_reduce_scatter=lambda inp, *, out: state_calls.append(("run_rs", tuple(inp.shape), tuple(out.shape))),
        run_all_gather=lambda inp, *, out: state_calls.append(("run_ag", tuple(inp.shape), tuple(out.shape), out.data_ptr())),
    )
    declarations[0].prepare_call(fake_state).run()
    declarations[1].prepare_call(fake_state).run()
    assert state_calls == [
        ("run_rs", (8, 8), (2, 8)),
        ("run_ag", (2, 8), (4, 2, 8), gathered.data_ptr()),
    ]

    # Verify eager call-level dispatch without and with dependencies.
    dep = torch.ones(1)
    assert reduce(source) is shard
    assert torch.equal(shard, source.view(4, 2, 8).sum(dim=0))
    assert gather(shard) is gathered
    for rank in range(4):
        assert torch.equal(gathered.view(4, 2, 8)[rank], shard + rank)
    assert runtime.calls == [
        ("reduce_scatter", (8, 8), (2, 8), shard.data_ptr(), plans[0]),
        ("all_gather", (2, 8), (4, 2, 8), gathered.data_ptr(), plans[1]),
    ]

    shard_dep = torch.zeros_like(shard)
    gathered_dep = torch.zeros_like(gathered)
    assert reduce(source, shard_dep, dependencies=(dep,)) is shard_dep
    assert gather(shard_dep, gathered_dep, dependencies=(dep,)) is gathered_dep
    assert torch.equal(shard_dep, shard) and torch.equal(gathered_dep, gathered)

    # Verify FX custom-op selection for empty vs non-empty dependencies, plus multi-chunk DMA chaining.
    reduce.functional(session=object())
    gather.functional(session=object())
    examples = {}
    chunk_runtime = _FakeDmaAllReduce(world=2, max_bytes=2 * 2 * 16)
    chunk_shard = torch.ones(5, 8, dtype=torch.bfloat16)
    chunked = PreparedChunkedGatherShards(
        chunk_shard, torch.zeros(10, 8, dtype=torch.bfloat16),
        runtime=chunk_runtime, session=object(), coordinator=SimpleNamespace(hosts=()),
        name="dma.chunked", examples=examples,
    ).functional(session=object())
    assert ("chunked_gather_shards.staging", (4, 8), torch.bfloat16, torch.device("cpu")) in examples
    assert chunked.bounds == [(0, 2), (2, 4), (4, 5)]

    def forward(x, c_in, ordering):
        s0 = reduce(x)
        s1 = reduce(x, dependencies=(ordering,))
        g0 = gather(s0)
        g1 = gather(s1, dependencies=(g0,))
        cg = chunked(c_in)
        return g1, cg

    graph = make_fx(forward, tracing_mode="fake")(source, chunk_shard, dep).graph
    rs_plain = [n for n in graph.nodes if n.target == torch.ops.flare.pcie_reduce_scatter.default]
    rs_dep = [n for n in graph.nodes if n.target == torch.ops.flare.pcie_reduce_scatter_dep.default]
    gs_plain = [n for n in graph.nodes if n.target == torch.ops.flare.pcie_gather_shards.default]
    gs_dep = [n for n in graph.nodes if n.target == torch.ops.flare.pcie_gather_shards_dep.default]
    assert len(rs_plain) == 1 and len(rs_dep) == 1
    assert rs_dep[0].args[2][0].op == "placeholder"
    # Single-shot gather uses plain once, plus chunk 0 of chunked gather; chunks 1 and 2 chain via _dep.
    assert len(gs_plain) == 2 and len(gs_dep) == 3
    for chained in gs_dep[1:]:
        assert len(chained.args[2]) == 1 and chained.args[2][0].target == torch.ops.aten.empty_strided.default


def test_roce_row_shard_calls_reject_non_empty_dependencies(prepared):
    from ignition.components.collectives import (
        PreparedChunkedGatherShards, PreparedGatherShards, PreparedReduceScatter,
    )

    runtime = _runtime()
    source = torch.zeros(8, 8, dtype=torch.bfloat16)
    shard = torch.zeros(2, 8, dtype=torch.bfloat16)
    gathered = torch.zeros(8, 8, dtype=torch.bfloat16)
    dep = torch.ones(1)

    reduce = PreparedReduceScatter(source, shard, runtime=runtime, session=object(),
                                   coordinator=SimpleNamespace(hosts=()), name="roce.rs")
    gather = PreparedGatherShards(shard, gathered, runtime=runtime, session=object(),
                                  coordinator=SimpleNamespace(hosts=()), name="roce.gs")
    chunked = PreparedChunkedGatherShards(
        torch.zeros(5, 8, dtype=torch.bfloat16), torch.zeros(10, 8, dtype=torch.bfloat16),
        runtime=_runtime(world=2, pipe_bytes=2 * 2 * 16), session=object(),
        coordinator=SimpleNamespace(hosts=()), name="roce.chunked",
    )

    with pytest.raises(ValueError, match="does not accept graph-ordering dependencies"):
        reduce(source, dependencies=(dep,))
    with pytest.raises(ValueError, match="does not accept graph-ordering dependencies"):
        gather(shard, dependencies=(dep,))
    with pytest.raises(ValueError, match="does not accept graph-ordering dependencies"):
        chunked(torch.zeros(5, 8, dtype=torch.bfloat16), dependencies=(dep,))


@pytest.mark.parametrize(
    "wrapper_cls,op_cls,is_gather",
    [
        ("ConcurrentReduceScatter", "PreparedReduceScatter", False),
        ("ConcurrentGatherShards", "PreparedGatherShards", True),
    ],
)
def test_concurrent_row_shard_validation_and_scratch_checks(
    monkeypatch, prepared, prepared_dma, wrapper_cls, op_cls, is_gather
):
    from ignition.components import collectives
    from ignition.preparation.scratch import ScratchArena

    wrapper = getattr(collectives, wrapper_cls)
    op = getattr(collectives, op_cls)
    finalizers = []
    monkeypatch.setattr(collectives, "register_finalizer", lambda session, fn: finalizers.append(fn))

    full = torch.zeros(8, 8, dtype=torch.bfloat16)
    shard = torch.zeros(2, 8, dtype=torch.bfloat16)
    src, dst = (shard, full) if is_gather else (full, shard)
    local_plan = object()

    arena = ScratchArena("cpu", overlap=True)
    arena._auxiliary_stream = SimpleNamespace(cuda_stream=17)

    # Single-rank and RoCE operations are rejected.
    single_op = op(src, src.clone(), runtime=None, session=None, coordinator=None, name="single")
    with pytest.raises(ValueError, match="requires a prepared PCIe DMA channel"):
        wrapper(single_op, workspace=arena, session=object(), parallel_plans=(local_plan,))

    roce_op = op(src, dst.clone(), runtime=_runtime(), session=object(),
                 coordinator=SimpleNamespace(hosts=()), name="roce")
    with pytest.raises(ValueError, match="requires a prepared PCIe DMA channel"):
        wrapper(roce_op, workspace=arena, session=object(), parallel_plans=(local_plan,))

    # Non-arena workspace, overlap=False, and empty parallel_plans are rejected.
    dma_op = op(src, dst.clone(), runtime=_FakeDmaAllReduce(), session=object(),
                coordinator=SimpleNamespace(hosts=()), name="dma")
    for bad_ws, bad_plans in (
        (object(), (local_plan,)),
        (ScratchArena("cpu", overlap=False), (local_plan,)),
        (arena, ()),
    ):
        with pytest.raises(ValueError, match="requires an overlap arena and local plans"):
            wrapper(dma_op, workspace=bad_ws, session=object(), parallel_plans=bad_plans)

    # Valid DMA overlap registers parallel plans, binds send_buffer, and enforces zero caller scratch.
    send_buf = src.clone()
    concurrent = wrapper(dma_op, workspace=arena, session=object(),
                         parallel_plans=(local_plan,), send_buffer=send_buf)
    assert concurrent.stream_handle == 17
    assert dma_op.concurrent_input is send_buf
    assert arena._parallel_plans == [(dma_op.plan, local_plan)]
    assert finalizers == [concurrent.validate_scratch]

    concurrent.validate_scratch()
    dma_op.plan.scratch_nbytes = 256
    with pytest.raises(ValueError, match="requires a plan without caller scratch"):
        concurrent.validate_scratch()


class _FakeIslandAllReduce:
    """Stand-in for TP16 `b12x.comm.pcie.AllReduce` with an active island runtime."""

    def __init__(self, *, world=16, max_elements=1024 * 7168, override="island_rs", has_island=True):
        self.world_size = world
        self._algorithm_override = override
        self._island_rs = (
            SimpleNamespace(max_elements=max_elements, group=object()) if has_island else None
        )
        self.calls = []
        self.plans = []

    def should_reduce_scatter(self, inp):
        return self._island_rs is not None and inp.numel() <= self._island_rs.max_elements

    def should_all_gather(self, inp):
        return self._island_rs is not None and inp.numel() * self.world_size <= self._island_rs.max_elements

    def should_row_to_col_all_to_all(self, inp):
        return (
            self._island_rs is not None
            and inp.ndim == 2
            and inp.shape[1] % (self.world_size * 2) == 0
            and inp.numel() <= self._island_rs.max_elements
        )

    def plan(self, inp, *, operation, out=None, transport=None):
        p = SimpleNamespace(
            operation=operation,
            shape=tuple(inp.shape),
            out_shape=None if out is None else tuple(out.shape),
            transport=transport,
            scratch_nbytes=0,
            memory_requirements=lambda: SimpleNamespace(scratch_nbytes=p.scratch_nbytes),
            request=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        self.plans.append(p)
        return p

    def reduce_scatter(self, inp, *, out, plan):
        self.calls.append(("reduce_scatter", tuple(inp.shape), tuple(out.shape), plan.operation))
        out.copy_(inp.view(self.world_size, *out.shape).sum(dim=0))

    def all_gather(self, inp, *, out, plan):
        self.calls.append(("all_gather", tuple(inp.shape), tuple(out.shape), plan.operation))
        for r in range(self.world_size):
            out[r].copy_(inp + r)

    def row_to_col_all_to_all(self, inp, *, out, plan):
        self.calls.append(("row_to_col_all_to_all", tuple(inp.shape), tuple(out.shape), plan.operation))
        st, full_c = inp.shape
        c = full_c // self.world_size
        out.view(self.world_size, st, c)[:] = inp[:, :c]


def test_tp16_island_row_shard_and_all_to_all_dispatch(monkeypatch):
    import b12x.comm.pcie as pcie
    import b12x.comm.pcie._owner_preparation as owner_prep
    from ignition.components import collectives
    from ignition.components.collectives import (
        ConcurrentGatherShards,
        ConcurrentReduceScatter,
        ConcurrentRowToColAllToAll,
        PreparedGatherShards,
        PreparedReduceScatter,
        PreparedRowToColAllToAll,
    )
    from ignition.preparation.scratch import ScratchArena

    monkeypatch.setattr(pcie, "AllReduce", _FakeIslandAllReduce)
    declarations = []
    monkeypatch.setattr(collectives, "declare", lambda session, requests, coordinator=None: declarations.extend(requests))
    monkeypatch.setattr(collectives, "register_finalizer", lambda session, fn: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: list(range(16)))
    monkeypatch.setattr(
        owner_prep,
        "prepared_call",
        lambda state, *, inp, out: SimpleNamespace(run=lambda: (inp.shape, out.shape), output=out, capture_safe=True, owners=()),
    )

    rt = _FakeIslandAllReduce()
    assert collectives.row_shard_surfaces(rt)
    assert not collectives.row_shard_surfaces(_FakeIslandAllReduce(override="hierarchical"))
    assert not collectives.row_shard_surfaces(_FakeIslandAllReduce(has_island=False))

    rs_src = torch.ones(32, 16, dtype=torch.bfloat16)
    rs_dst = torch.zeros(2, 16, dtype=torch.bfloat16)
    ag_dst = torch.zeros(32, 16, dtype=torch.bfloat16)
    a2a_src = torch.ones(2, 32, dtype=torch.bfloat16)
    a2a_dst = torch.zeros(32, 2, dtype=torch.bfloat16)

    assert PreparedReduceScatter.fits(rt, rs_src, rs_dst)
    assert PreparedGatherShards.fits(rt, rs_dst, ag_dst)
    assert PreparedRowToColAllToAll.fits(rt, a2a_src, a2a_dst)

    rs = PreparedReduceScatter(rs_src, rs_dst, runtime=rt, session=object(), coordinator=SimpleNamespace(hosts=()), name="isl.rs")
    ag = PreparedGatherShards(rs_dst, ag_dst, runtime=rt, session=object(), coordinator=SimpleNamespace(hosts=()), name="isl.ag")
    a2a = PreparedRowToColAllToAll(a2a_src, a2a_dst, runtime=rt, session=object(), coordinator=SimpleNamespace(hosts=()), name="isl.a2a")
    rs_conc = PreparedReduceScatter(rs_src, rs_dst, runtime=rt, session=object(), coordinator=SimpleNamespace(hosts=()), name="isl.rs.conc", concurrent=True)
    ag_conc = PreparedGatherShards(rs_dst, ag_dst, runtime=rt, session=object(), coordinator=SimpleNamespace(hosts=()), name="isl.ag.conc", concurrent=True)
    assert rs.pcie and not rs.dma
    assert ag.pcie and not ag.dma
    assert a2a.pcie
    assert rs_conc.pcie and ag_conc.pcie
    assert [(p.operation, p.transport) for p in rt.plans] == [
        ("reduce_scatter", None),
        ("all_gather", None),
        ("row_to_col_all_to_all", None),
        ("reduce_scatter", "ce_local"),
        ("all_gather", "ce_local"),
    ]

    assert rs(rs_src) is rs_dst
    assert ag(rs_dst) is ag_dst
    assert a2a(a2a_src) is a2a_dst
    assert rt.calls == [
        ("reduce_scatter", (32, 16), (2, 16), "reduce_scatter"),
        ("all_gather", (2, 16), (16, 2, 16), "all_gather"),
        ("row_to_col_all_to_all", (2, 32), (32, 2), "row_to_col_all_to_all"),
    ]

    # Concurrent wrappers accept TP16 island ops and register parallel plans.
    arena = ScratchArena("cpu", overlap=True)
    arena._auxiliary_stream = SimpleNamespace(cuda_stream=23)
    local_plan = object()
    crs = ConcurrentReduceScatter(rs, workspace=arena, session=object(), parallel_plans=(local_plan,))
    cag = ConcurrentGatherShards(ag, workspace=arena, session=object(), parallel_plans=(local_plan,))
    ca2a = ConcurrentRowToColAllToAll(a2a, workspace=arena, session=object(), parallel_plans=(local_plan,))
    assert arena._parallel_plans == [(rs.plan, local_plan), (ag.plan, local_plan), (a2a.plan, local_plan)]
    crs.validate_scratch()
    cag.validate_scratch()
    ca2a.validate_scratch()

    started = []
    monkeypatch.setattr(collectives, "_start_reduce_scatter", lambda send, *args: started.append(("rs", send)))
    monkeypatch.setattr(collectives, "_start_gather_shards", lambda send, *args: started.append(("ag", send)))
    monkeypatch.setattr(collectives, "_start_row_to_col_all_to_all", lambda send, *args: started.append(("a2a", send)))
    p_rs_clone = crs.start(rs_src)
    p_rs_direct = crs.start(rs_src, clone=False)
    p_ag_direct = cag.start(rs_dst, clone=False)
    p_a2a_direct = ca2a.start(a2a_src, clone=False)
    assert p_rs_clone.source is not rs_src
    assert p_rs_direct.source is rs_src
    assert p_ag_direct.source is rs_dst
    assert p_a2a_direct.source is a2a_src
    assert started[-3:] == [("rs", rs_src), ("ag", rs_dst), ("a2a", a2a_src)]
