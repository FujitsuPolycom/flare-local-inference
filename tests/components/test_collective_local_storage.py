# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Collective invocation storage and asynchronous operands are explicit tensors."""

import gc
import weakref

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from ignition.components.collectives import (
    ConcurrentAllGather, ConcurrentAllReduce, ConcurrentAllToAll,
    ConcurrentGatherShards, ConcurrentReduceScatter,
    PreparedAllGather, PreparedAllReduce, PreparedAllToAll,
    PreparedChunkedAllGather, PreparedGatherShards, PreparedReduceScatter,
)


def test_functional_gather_releases_declaration_storage_and_copies_each_source():
    source = torch.arange(12, dtype=torch.float32).view(3, 4)
    output = torch.empty_like(source)
    dependency = torch.ones(1)
    output_ref, dependency_ref = weakref.ref(output), weakref.ref(dependency)
    gather = PreparedAllGather(source, output, runtime=None, session=None,
        coordinator=None, name="single", dependencies=(dependency,)).functional(session=None)
    del output, dependency
    gc.collect()
    assert output_ref() is None and dependency_ref() is None
    first, second = gather(source), gather(source + 1)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, source)
    torch.testing.assert_close(second, source + 1)


def test_functional_chunked_gather_owns_one_invocation_destination():
    source = torch.arange(12, dtype=torch.float32).view(3, 4)
    output = torch.empty_like(source)
    output_ref = weakref.ref(output)
    gather = PreparedChunkedAllGather(source, output, runtime=None, session=None,
        coordinator=None, name="chunks").functional(session=None)
    del output
    gc.collect()
    assert output_ref() is None
    torch.testing.assert_close(gather(source), source)
    torch.testing.assert_close(gather.run_eager(source + 2), source + 2)


def test_concurrent_gather_trace_retains_pending_operands_until_join():
    source = torch.ones((2, 4), dtype=torch.float32)
    gather = PreparedAllGather(source, torch.empty_like(source), runtime=None,
        session=None, coordinator=None, name="trace")
    gather.handle = 0
    concurrent = object.__new__(ConcurrentAllGather)
    concurrent.gather, concurrent.stream_handle = gather, 1
    concurrent.functional(session=None)

    def forward(value, history):
        pending = concurrent.start(value * 2, dependencies=(history,))
        local = value + 3
        return concurrent.finish(pending, local) + local

    graph = make_fx(forward, tracing_mode="fake")(source, torch.ones(1)).graph
    start, = [node for node in graph.nodes if node.target == torch.ops.flare.start_all_gather.default]
    join, = [node for node in graph.nodes if node.target == torch.ops.flare.join_all_gather.default]
    assert join.args[:3] == start.args[:3]
    assert join.args[3].target == torch.ops.aten.add.Tensor
    assert start.args[0].target == torch.ops.aten.mul.Tensor
    assert start.args[1].target == torch.ops.aten.empty_strided.default
    assert start.args[2][0].op == "placeholder"


def test_functional_gather_trace_uses_invocation_dependencies():
    source = torch.ones((2, 4), dtype=torch.float32)
    gather = PreparedAllGather(source, torch.empty_like(source), runtime=None,
        session=None, coordinator=None, name="trace").functional(session=None)
    gather.single, gather.handle = False, 0

    def forward(value, history):
        return gather(value, dependencies=(history,))

    graph = make_fx(forward, tracing_mode="fake")(source, torch.ones(1)).graph
    call, = [node for node in graph.nodes if node.target == torch.ops.flare.pcie_all_gather.default]
    assert call.args[2][0].op == "placeholder"
    assert call.args[1].target == torch.ops.aten.empty_strided.default


def test_concurrent_reduce_scatter_and_gather_shards_trace():
    full = torch.ones((8, 8), dtype=torch.bfloat16)
    shard = torch.ones((2, 8), dtype=torch.bfloat16)

    rs = PreparedReduceScatter(full, full.clone(), runtime=None, session=None, coordinator=None, name="rs")
    rs.single, rs.dma, rs.handle, rs.concurrent_input = False, True, 0, None
    rs.output = shard.clone()
    crs = object.__new__(ConcurrentReduceScatter)
    crs.reduce, crs.stream_handle = rs, 1
    crs.functional(session=None)

    gs = PreparedGatherShards(shard, shard.clone(), runtime=None, session=None, coordinator=None, name="gs")
    gs.single, gs.dma, gs.handle, gs.concurrent_input = False, True, 1, None
    gs.output = full.clone()
    cgs = object.__new__(ConcurrentGatherShards)
    cgs.gather, cgs.stream_handle = gs, 1
    cgs.functional(session=None)

    def forward(x, dep):
        p_rs = crs.start(x, dependencies=(dep,))
        local = x + 1
        out_sh = crs.finish(p_rs, local)
        p_gs = cgs.start(out_sh, dependencies=(out_sh,))
        local2 = out_sh + 2
        return cgs.finish(p_gs, local2)

    graph = make_fx(forward, tracing_mode="fake")(full, torch.ones(1)).graph
    s_rs, = [n for n in graph.nodes if n.target == torch.ops.flare.start_reduce_scatter.default]
    j_rs, = [n for n in graph.nodes if n.target == torch.ops.flare.join_reduce_scatter.default]
    s_gs, = [n for n in graph.nodes if n.target == torch.ops.flare.start_gather_shards.default]
    j_gs, = [n for n in graph.nodes if n.target == torch.ops.flare.join_gather_shards.default]
    assert j_rs.args[:3] == s_rs.args[:3]
    assert j_gs.args[:3] == s_gs.args[:3]


def test_functional_all_reduce_fused_addend_with_dependencies():
    source = torch.ones((2, 4), dtype=torch.bfloat16)
    addend = torch.full((2, 4), 2.0, dtype=torch.bfloat16)
    reduce = PreparedAllReduce(source, torch.empty_like(source), runtime=None,
        session=None, coordinator=None, name="reduce").functional(session=None)
    reduce.single, reduce.fused_addend, reduce.handle = False, True, 0

    def forward(value, extra, dep):
        return reduce(value, addend=extra, dependencies=(dep,))

    graph = make_fx(forward, tracing_mode="fake")(source, addend, torch.ones(1)).graph
    call, = [node for node in graph.nodes if node.target == torch.ops.flare.pcie_all_reduce_addend_dep.default]
    assert call.args[0].op == "placeholder"
    assert call.args[1].op == "placeholder"
    assert call.args[2].target == torch.ops.aten.empty_strided.default
    assert call.args[3][0].op == "placeholder"
    assert not any(node.target == torch.ops.aten.add.Tensor for node in graph.nodes)


def test_concurrent_copy_input_controls_fx_clone_and_retains_live_ssa_source():
    full = torch.ones((8, 8), dtype=torch.bfloat16)
    shard = torch.ones((2, 8), dtype=torch.bfloat16)
    a2a = torch.ones((4, 2, 8), dtype=torch.bfloat16)
    dep = torch.ones(1)

    def _make_cases(copy_input):
        ar = PreparedAllReduce(full, full.clone(), runtime=None, session=None, coordinator=None, name="ar")
        ar.single, ar.handle, ar.concurrent_input = False, 0, None
        car = object.__new__(ConcurrentAllReduce)
        car.reduce, car.stream_handle, car.copy_input = ar, 1, copy_input
        car.functional(session=None)

        rs = PreparedReduceScatter(full, full.clone(), runtime=None, session=None, coordinator=None, name="rs")
        rs.single, rs.dma, rs.handle, rs.concurrent_input = False, True, 1, None
        rs.output = shard.clone()
        crs = object.__new__(ConcurrentReduceScatter)
        crs.reduce, crs.stream_handle, crs.copy_input = rs, 1, copy_input
        crs.functional(session=None)

        gs = PreparedGatherShards(shard, shard.clone(), runtime=None, session=None, coordinator=None, name="gs")
        gs.single, gs.dma, gs.handle, gs.concurrent_input = False, True, 2, None
        gs.output = full.clone()
        cgs = object.__new__(ConcurrentGatherShards)
        cgs.gather, cgs.stream_handle, cgs.copy_input = gs, 1, copy_input
        cgs.functional(session=None)

        aa = PreparedAllToAll(a2a, a2a.clone(), runtime=None, session=None, coordinator=None, name="a2a")
        aa.single, aa.dma, aa.handle, aa.concurrent_input = False, True, 3, None
        caa = object.__new__(ConcurrentAllToAll)
        caa.all_to_all, caa.stream_handle, caa.copy_input = aa, 1, copy_input
        caa.functional(session=None)

        return (
            (car, full, torch.ops.flare.start_all_reduce.default, torch.ops.flare.join_all_reduce.default),
            (crs, full, torch.ops.flare.start_reduce_scatter.default, torch.ops.flare.join_reduce_scatter.default),
            (cgs, shard, torch.ops.flare.start_gather_shards.default, torch.ops.flare.join_gather_shards.default),
            (caa, a2a, torch.ops.flare.start_all_to_all.default, torch.ops.flare.join_all_to_all.default),
        )

    # Default copy_input=True emits aten.clone and retains the cloned send tensor until join.
    for concurrent, sample, start_op, join_op in _make_cases(copy_input=True):
        def forward_copy(x, d):
            pending = concurrent.start(x * 2, dependencies=(d,))
            local = x + 3
            return concurrent.finish(pending, local)

        graph = make_fx(forward_copy, tracing_mode="fake")(sample, dep).graph
        start, = [n for n in graph.nodes if n.target == start_op]
        join, = [n for n in graph.nodes if n.target == join_op]
        clones = [n for n in graph.nodes if n.target == torch.ops.aten.clone.default]
        assert len(clones) == 1
        assert start.args[0] is clones[0]
        assert join.args[:3] == start.args[:3]

    # Opt-in copy_input=False emits zero aten.clone nodes and retains the functional SSA source in join.
    for concurrent, sample, start_op, join_op in _make_cases(copy_input=False):
        def forward_zerocopy(x, d):
            pending = concurrent.start(x * 2, dependencies=(d,))
            local = x + 3
            return concurrent.finish(pending, local)

        graph = make_fx(forward_zerocopy, tracing_mode="fake")(sample, dep).graph
        start, = [n for n in graph.nodes if n.target == start_op]
        join, = [n for n in graph.nodes if n.target == join_op]
        clones = [n for n in graph.nodes if n.target == torch.ops.aten.clone.default]
        assert not clones
        assert start.args[0].target == torch.ops.aten.mul.Tensor
        assert join.args[0] is start.args[0]
        assert join.args[:3] == start.args[:3]

    # ConcurrentAllGather defaults to zero-copy (copy_input=False) and clones when copy_input=True.
    gather = PreparedAllGather(shard, shard.clone(), runtime=None, session=None, coordinator=None, name="ag")
    gather.handle = 4
    cag = object.__new__(ConcurrentAllGather)
    cag.gather, cag.stream_handle, cag.copy_input = gather, 1, True
    cag.functional(session=None)

    def forward_ag(x, d):
        pending = cag.start(x * 2, dependencies=(d,))
        return cag.finish(pending, x + 3)

    graph = make_fx(forward_ag, tracing_mode="fake")(shard, dep).graph
    start, = [n for n in graph.nodes if n.target == torch.ops.flare.start_all_gather.default]
    join, = [n for n in graph.nodes if n.target == torch.ops.flare.join_all_gather.default]
    clones = [n for n in graph.nodes if n.target == torch.ops.aten.clone.default]
    assert len(clones) == 1 and start.args[0] is clones[0] and join.args[0] is clones[0]
