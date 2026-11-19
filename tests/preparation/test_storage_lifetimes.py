# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Storage snapshots preserve reads across aliased writes and last-use boundaries."""

import operator
import random

import pytest
import torch

from ignition.preparation import compilation


def reference_reads_across_mutations(nodes, keys, mutations):
    """Compute snapshot requirements by scanning live values."""
    positions = {node: index for index, node in enumerate(nodes)}
    last_use = {node: max((positions[user] for user in node.users), default=-1) for node in nodes}
    for node in reversed(nodes):
        if (isinstance(node.target, torch._ops.OpOverload)
                and node.target.namespace == 'aten' and not node.target._schema.is_mutable):
            for operand in node.all_input_nodes:
                last_use[operand] = max(last_use[operand], last_use[node])
    mutations = {node: {keys[operand] for operand in operands}
                 for node, operands in mutations.items() if operands}
    mutable = set().union(*mutations.values()) if mutations else set()
    live, fresh, required = {}, set(), {}
    for index, node in enumerate(nodes):
        operands = node.all_input_nodes
        reads = set().union(*(live.get(arg, set()) for arg in operands)) if operands else set()
        reads.update(keys[arg] for arg in operands if keys.get(arg) in mutable)
        if node in mutations:
            for previous, pending in live.items():
                if previous in fresh and last_use[previous] > index and pending & mutations[node]:
                    required.setdefault(previous, []).append(node)
                    pending.clear()
        live = {value: pending for value, pending in live.items() if pending and last_use[value] > index}
        if (node in keys and node.op == 'call_function'
                and isinstance(node.target, torch._ops.OpOverload) and node.target.namespace == 'aten'):
            if keys[node] not in {keys[arg] for arg in operands if arg in keys}:
                fresh.add(node)
            live[node] = reads
        elif node in keys and keys[node] in mutable:
            live[node] = {keys[node]}
    return {node: required[node] for node in nodes if node in required}


def assert_matches_reference(graph, keys, mutations):
    graph.lint()
    nodes = list(graph.nodes)
    expected = reference_reads_across_mutations(nodes, keys, mutations)
    actual = compilation._reads_across_mutations(nodes, keys, mutations)
    assert list(actual.items()) == list(expected.items())
    return actual


def random_storage_graph(seed, size=160):
    rng = random.Random(seed)
    graph = torch.fx.Graph()
    values = [graph.placeholder(f'input_{index}') for index in range(8)]
    keys = {value: index // 2 for index, value in enumerate(values)}
    mutations = {}
    next_key = 4
    for _ in range(size):
        left, right, destination = rng.choices(values, k=3)
        kind = rng.randrange(8)
        if kind == 0:
            node = graph.call_function(torch.ops.aten.sin.default, (left,))
        elif kind == 1:
            node = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
        elif kind == 2:
            node = graph.call_function(torch.ops.aten.view.default, (left, [8]))
            keys[node] = keys[left]
        elif kind == 3:
            node = graph.call_function(torch.ops.aten.copy_.default, (destination, left))
            keys[node] = keys[destination]
            mutations[node] = [destination]
        elif kind == 4:
            node = graph.call_function(torch.ops.aten.add.out, (left, right), {'out': destination})
            keys[node] = keys[destination]
            mutations[node] = [destination]
        elif kind == 5:
            node = graph.call_function(torch.ops.aten._foreach_add_.List, ([left, right], [right, left]))
            mutations[node] = [left, right]
            node = graph.call_function(operator.getitem, (node, 0))
            keys[node] = keys[left]
        elif kind == 6:
            node = graph.call_function(torch.ops.aten.mul.Tensor, (left, right))
        else:
            node = graph.call_function(torch.ops.aten.neg.default, (left,))
        if node not in keys:
            keys[node] = next_key
            next_key += 1
        values.append(node)
    graph.output(tuple(rng.sample(values, 16)))
    return graph, keys, mutations


@pytest.mark.parametrize('seed', range(200))
def test_random_aliases_mutations_and_out_arguments(seed):
    graph, keys, mutations = random_storage_graph(seed)
    assert_matches_reference(graph, keys, mutations)


def test_reads_precede_mutation_and_last_use_expiry():
    graph = torch.fx.Graph()
    source = graph.placeholder('source')
    saved = graph.call_function(torch.ops.aten.sin.default, (source,))
    write = graph.call_function(torch.ops.aten.copy_.default, (source, saved))
    graph.output(source)
    keys = {source: 0, saved: 1, write: 0}
    assert assert_matches_reference(graph, keys, {write: [source]}) == {}


def test_snapshot_clears_all_pending_storage_dependencies():
    graph = torch.fx.Graph()
    left, right = graph.placeholder('left'), graph.placeholder('right')
    saved = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
    first = graph.call_function(torch.ops.aten.zero_.default, (left,))
    second = graph.call_function(torch.ops.aten.zero_.default, (right,))
    consumed = graph.call_function(torch.ops.aten.sin.default, (saved,))
    graph.output(consumed)
    keys = {left: 0, right: 1, saved: 2, first: 0, second: 1, consumed: 3}
    actual = assert_matches_reference(graph, keys, {first: [left], second: [right]})
    assert actual == {saved: [first]}


def test_mutation_reads_propagate_before_snapshot_clears():
    graph = torch.fx.Graph()
    source = graph.placeholder('source')
    saved = graph.call_function(torch.ops.aten.sin.default, (source,))
    first = graph.call_function(torch.ops.aten.add_.Tensor, (source, saved))
    derived = graph.call_function(torch.ops.aten.cos.default, (first,))
    second = graph.call_function(torch.ops.aten.zero_.default, (source,))
    graph.output((saved, derived))
    keys = {source: 0, saved: 1, first: 0, derived: 2, second: 0}
    actual = assert_matches_reference(graph, keys, {first: [source], second: [source]})
    assert actual == {saved: [first], derived: [second]}


def test_immutable_dependency_chain_extends_last_use():
    graph = torch.fx.Graph()
    source = graph.placeholder('source')
    first = graph.call_function(torch.ops.aten.sin.default, (source,))
    second = graph.call_function(torch.ops.aten.cos.default, (first,))
    write = graph.call_function(torch.ops.aten.zero_.default, (source,))
    graph.output(second)
    keys = {source: 0, first: 1, second: 2, write: 0}
    actual = assert_matches_reference(graph, keys, {write: [source]})
    assert list(actual) == [first, second]
    assert all(consumers == [write] for consumers in actual.values())


def test_empty_pending_dead_nodes_and_no_mutations():
    graph = torch.fx.Graph()
    source = graph.placeholder('source')
    dead = graph.call_function(torch.ops.aten.sin.default, (source,))
    empty = graph.call_function(torch.ops.aten.ones.default, ([8],))
    graph.output(source)
    assert assert_matches_reference(graph, {source: 0, dead: 1, empty: 2}, {}) == {}


@pytest.mark.parametrize('alias_target', [torch.ops.aten.view.default, torch.ops.aten.slice.Tensor])
def test_snapshot_tracks_alias_storage_and_out_writes(alias_target):
    graph = torch.fx.Graph()
    source = graph.placeholder('source')
    unrelated = graph.placeholder('unrelated')
    args = (source, [8]) if alias_target is torch.ops.aten.view.default else (source, 0, 0, 8)
    alias = graph.call_function(alias_target, args)
    saved = graph.call_function(torch.ops.aten.sin.default, (alias,))
    dead = graph.call_function(torch.ops.aten.cos.default, (alias,))
    other = graph.call_function(torch.ops.aten.neg.default, (unrelated,))
    write = graph.call_function(torch.ops.aten.add.out, (unrelated, unrelated), {'out': source})
    graph.output((saved, other, alias))
    keys = {source: 0, unrelated: 1, alias: 0, saved: 2, dead: 3, other: 4, write: 0}
    assert assert_matches_reference(graph, keys, {write: [source]}) == {saved: [write]}


def test_snapshot_deduplicates_multi_storage_mutations_in_graph_order():
    graph = torch.fx.Graph()
    left, right = graph.placeholder('left'), graph.placeholder('right')
    first = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
    second = graph.call_function(torch.ops.aten.mul.Tensor, (left, right))
    write = graph.call_function(torch.ops.aten._foreach_add_.List, ([left, right], [right, left]))
    later = graph.call_function(torch.ops.aten.zero_.default, (left,))
    graph.output((first, second))
    keys = {left: 0, right: 1, first: 2, second: 3, later: 0}
    actual = assert_matches_reference(graph, keys, {write: [left, right], later: [left]})
    assert list(actual.items()) == [(first, [write]), (second, [write])]


@pytest.mark.parametrize('separate_views', [False, True])
def test_lowered_cpu_snapshots_preserve_reused_scratch(separate_views):
    from ignition.preparation.compilation import _INVOCATIONS, _lower_storage
    from torch.fx.experimental.proxy_tensor import make_fx

    scratch = torch.zeros(8)
    alias = scratch.view(2, 4)[0] if separate_views else scratch

    def forward():
        saved = alias.sin().cos()
        scratch.fill_(2.)
        return saved, alias

    module = make_fx(forward)()
    calls = _lower_storage(module)
    module.recompile()
    token = _INVOCATIONS.set(calls)
    try:
        for initial in (0., 3., -7.):
            scratch.fill_(initial)
            saved, result = module()
            torch.testing.assert_close(saved, torch.full_like(alias, initial).sin().cos())
            torch.testing.assert_close(result, torch.full_like(alias, 2.))
            assert result.data_ptr() == alias.data_ptr()
    finally:
        _INVOCATIONS.reset(token)


_FILL_ADDRESSES = []


@torch.library.custom_op('flare_test_lifetimes::fill', mutates_args=('destination',))
def _fill(destination: torch.Tensor, value: float) -> None:
    _FILL_ADDRESSES.append(destination.data_ptr())
    destination.fill_(value)


@_fill.register_fake
def _fake_fill(destination, value):
    pass


@pytest.mark.parametrize('offset', [0, 16])
def test_lowered_opaque_writes_keep_caller_storage_and_intermediate_values(offset):
    storage = torch.zeros(64)
    source = storage[:8]
    destination = storage[offset:offset + 8]
    output = torch.empty(3, 8)

    def forward():
        first = source + 3
        _fill(destination, 7.)
        second = source + 5
        _fill(destination, 11.)
        output[0].copy_(first)
        output[1].copy_(second)
        output[2].copy_(source)
        return output, destination

    module = torch.fx.experimental.proxy_tensor.make_fx(forward)()
    calls = compilation._lower_storage(module)
    module.recompile()
    token = compilation._INVOCATIONS.set(calls)
    try:
        for initial in (2., -17., 101.):
            _FILL_ADDRESSES.clear()
            storage.fill_(initial)
            output.fill_(float('nan'))
            result, written = module()
            expected = torch.tensor([initial + 3, (7 if offset == 0 else initial) + 5,
                                     11 if offset == 0 else initial]).view(3, 1).expand_as(output)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            assert _FILL_ADDRESSES == [destination.data_ptr(), destination.data_ptr()]
            assert result.data_ptr() == output.data_ptr()
            assert written.data_ptr() == destination.data_ptr()
            assert torch.all(destination == 11)
            assert torch.all(storage[offset + 8:] == initial)
            if offset:
                assert torch.all(storage[:offset] == initial)
    finally:
        compilation._INVOCATIONS.reset(token)


@torch.library.custom_op('flare_test_lifetimes::copy_from', mutates_args=('destination',))
def _copy_from(destination: torch.Tensor, source: torch.Tensor) -> None:
    destination.copy_(source)


@_copy_from.register_fake
def _fake_copy_from(destination, source):
    pass


def test_lowered_dtype_views_of_mutated_temporaries_pass_through_their_owner(monkeypatch):
    seed = torch.arange(32.)

    def forward():
        storage = seed + 1
        other = seed * 2
        plain = seed - 1
        _fill(storage[8:16].view(torch.bfloat16).view(4, 4), 5.)
        _copy_from(storage[:8].view(torch.bfloat16), other[8:16].view(torch.bfloat16))
        _fill(plain[4:8], 7.)
        return storage, plain

    expected = forward()
    module = torch.fx.experimental.proxy_tensor.make_fx(forward)()
    calls = compilation._lower_storage(module)
    assert len(calls) == 3
    module.recompile()
    owners = []
    original = compilation._Invocation.run

    def observe(invocation, reads, writes, integers=()):
        owners.append((len(reads), len(writes), [array.numel() for array in (*reads, *writes)]))
        return original(invocation, reads, writes, integers)

    monkeypatch.setattr(compilation._Invocation, 'run', observe)
    token = compilation._INVOCATIONS.set(calls)
    try:
        torch.testing.assert_close(module(), expected, rtol=0, atol=0)
    finally:
        compilation._INVOCATIONS.reset(token)
    # A mutated allocation that is also viewed as bfloat16 is passed whole. The read of a bfloat16 slice
    # and the mutation of a slice that has only one dtype are passed as the slices themselves.
    assert owners == [(0, 1, [32]), (1, 1, [16, 32]), (0, 1, [4])]


def test_lowering_rejects_temporary_owner_that_cannot_be_viewed_as_operand_dtype():
    def forward():
        scratch = torch.zeros(7, dtype=torch.bfloat16)
        _fill(scratch[2:6].view(torch.float32), 1.)

    module = torch.fx.experimental.proxy_tensor.make_fx(forward)()
    with pytest.raises(ValueError, match='cannot view temporary storage-call owner'):
        compilation._lower_storage(module)


@torch.library.custom_op('flare_test_lifetimes::gather_ranks', mutates_args=('output',))
def _gather_ranks(source: torch.Tensor, output: torch.Tensor, world: int) -> None:
    # Rank r contributes the candidates plus r, in rank order across each row.
    columns = source.shape[1]
    for rank in range(world):
        output[:, rank * columns:(rank + 1) * columns].copy_(source + rank)


@_gather_ranks.register_fake
def _fake_gather_ranks(source, output, world):
    pass


@pytest.mark.parametrize(('world', 'rows'), [(3, 1), (3, 512), (3, 1152), (3, 1536), (3, 4096), (2, 1536), (4, 3584)])
def test_compiled_dtype_view_gathers_write_every_slice_in_place(world, rows):
    # The context-parallel indexer gathers each 512-row slice of packed float32 candidates into a
    # float32 buffer through bfloat16 views of both, and merges the whole buffer afterwards.
    from torch._inductor import standalone_compile
    from torch.fx.experimental.proxy_tensor import make_fx

    from ignition import runtime_patches

    runtime_patches.apply()
    seed = torch.randn(rows, 512, 2).to(torch.bfloat16).to(torch.float32)

    def forward(seed):
        candidates = seed.clone()
        gathered = torch.zeros(rows, world * 512, 2)
        for start in range(0, rows, 512):
            end = min(start + 512, rows)
            _gather_ranks(candidates[start:end].view(torch.bfloat16).view(end - start, -1),
                          gathered[start:end].view(torch.bfloat16).view(end - start, -1), world)
        return (gathered,)

    packed = seed.view(torch.bfloat16).view(rows, -1)
    expected = torch.cat([packed + rank for rank in range(world)], 1).view(torch.float32).view(rows, world * 512, 2)
    module = make_fx(forward)(seed)
    calls = compilation._lower_storage(module)
    module.recompile()
    compiled = standalone_compile(module, [seed], dynamic_shapes='from_example_inputs', aot=True,
        options={'config_patches': {'memory_planning': True, 'memory_pool': 'combined', 'compile_threads': 1}})
    token = compilation._INVOCATIONS.set(calls)
    try:
        observed, = compiled(seed)
    finally:
        compilation._INVOCATIONS.reset(token)
    torch.testing.assert_close(observed.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)
