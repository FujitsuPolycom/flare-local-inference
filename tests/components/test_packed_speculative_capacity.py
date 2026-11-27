# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed catalogs refine physical row budgets independently of proposal depth."""

from fractions import Fraction

import pytest

from ignition.components.capacity import batch_capacities, packed_speculative_shapes


@pytest.mark.parametrize("concurrency", [1, 3, 8, 32, 64])
@pytest.mark.parametrize("width", [2, 6, 8])
@pytest.mark.parametrize("capacity", [8, 31, 128, 512])
def test_catalog_covers_capacity_endpoints_without_exceeding_row_budget(concurrency, width, capacity):
    shapes = packed_speculative_shapes(concurrency, width, capacity)
    assert len(shapes) == len(set(shapes)) <= 64
    for batch in batch_capacities(concurrency):
        physical = [tokens for b, tokens in shapes if b == batch]
        upper = min(capacity, batch * width)
        if batch < upper:
            assert physical[0] == batch + 1
            assert physical[-1] == upper
        else:
            assert not physical
        assert all(batch < tokens <= upper for tokens in physical)
    assert (1, width) in shapes


@pytest.mark.parametrize("width,counts,saturation", [
    (6, (5, 7, 7, 8, 7, 9, 9, 7, 5), 445),
    (8, (5, 6, 8, 8, 9, 9, 9, 5, 5), 543),
])
def test_default_density_covers_intermediate_batches_within_graph_budget(width, counts, saturation):
    assert batch_capacities(32) == (1, 2, 4, 6, 8, 12, 16, 24, 32)
    shapes = packed_speculative_shapes(32, width, 128)
    assert len(shapes) == 64
    assert max(batch for batch, _ in shapes) == 32
    assert max(tokens for _, tokens in shapes) == 128
    assert tuple(sum(b == batch for b, _ in shapes) for batch in batch_capacities(32)) == counts
    assert (1, 2) in shapes and (1, width) in shapes
    complete = packed_speculative_shapes(32, width, 128, 1000)
    assert len(complete) == saturation
    assert set(complete) == {(batch, tokens) for batch in batch_capacities(32)
                             for tokens in range(batch + 1, min(128, batch * width) + 1)}


def test_each_budget_adds_one_graph_by_refining_the_largest_relative_gap():
    previous = packed_speculative_shapes(32, 6, 128, 18)
    for budget in range(19, 90):
        shapes = packed_speculative_shapes(32, 6, 128, budget)
        assert set(previous) < set(shapes)
        assert len(shapes) == budget
        batch, middle = (set(shapes) - set(previous)).pop()
        intervals = [(Fraction(upper, lower), b, lower, upper)
            for b in batch_capacities(32)
            for lower, upper in zip([p for old_b, p in previous if old_b == b],
                                    [p for old_b, p in previous if old_b == b][1:])
            if upper - lower > 1]
        _, selected_batch, lower, upper = min(intervals, key=lambda gap: (-gap[0], *gap[1:]))
        assert batch == selected_batch
        assert middle == min(range(lower + 1, upper), key=lambda value:
            (max(Fraction(value, lower), Fraction(upper, value)), value))
        previous = shapes


def test_small_domains_saturate_without_duplicate_endpoints():
    assert packed_speculative_shapes(1, 2, graph_budget=1) == ((1, 2),)
    assert packed_speculative_shapes(1, 2, graph_budget=64) == ((1, 2),)
    assert packed_speculative_shapes(32, 6, 6) == (
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
        (2, 3), (2, 4), (2, 5), (2, 6), (4, 5), (4, 6))


@pytest.mark.parametrize("budget", [0, -1, 12, 17, True, 64.0, None])
def test_budget_must_cover_mandatory_endpoints(budget):
    with pytest.raises(ValueError, match="at least 18.*9 eligible request buckets"):
        packed_speculative_shapes(32, 6, 128, budget)


@pytest.mark.parametrize("width", [0, 1, 1 << 31, True, 6.0])
def test_logical_width_is_a_supported_positive_draft_block(width):
    with pytest.raises(ValueError, match="width"):
        packed_speculative_shapes(32, width)
