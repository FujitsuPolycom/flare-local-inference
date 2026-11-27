# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the decoupled readiness barrier protocol and sampler broadcast attachment."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import readiness
from ignition.components.readiness import BARRIER_DTYPE, BARRIER_SHAPE, PreparedReadinessWait


def _fake_timings(intervals=2):
    return SimpleNamespace(
        frame=torch.zeros(1, dtype=torch.int32),
        timestamps=torch.zeros((2, 3 * intervals), dtype=torch.int64),
    )


def test_readiness_wait_single_rank_and_no_barrier_register_no_handle():
    timings = _fake_timings(2)
    wait_none = PreparedReadinessWait(timings, 0)
    assert wait_none.handle == 0
    assert wait_none.input.shape == BARRIER_SHAPE and wait_none.input.dtype == BARRIER_DTYPE

    single_barrier = SimpleNamespace(single=True)
    wait_single = PreparedReadinessWait(timings, 1, barrier=single_barrier)
    assert wait_single.handle == 0

    with pytest.raises(ValueError, match="outside its timestamp buffer"):
        PreparedReadinessWait(timings, 2)


def test_readiness_wait_registers_barrier_callable_and_forwards_functional():
    calls = []
    functional_sessions = []

    class FakeBarrier:
        single = False

        def functional(self, *, session):
            functional_sessions.append(session)
            return self

        def __call__(self, inp, out):
            calls.append((inp, out))
            out.copy_(inp + 1)
            return out

    timings = _fake_timings(1)
    barrier = FakeBarrier()
    wait = PreparedReadinessWait(timings, 0, barrier=barrier)
    assert wait.handle > 0
    assert readiness._BARRIERS[wait.handle - 1] is barrier

    inp, out = wait.operands(torch.zeros(1))
    readiness._BARRIERS[wait.handle - 1](inp, out)
    assert len(calls) == 1
    assert torch.all(out == 1)

    sentinel = object()
    wait.functional(session=sentinel)
    assert functional_sessions == [sentinel]
    assert wait.input is None and wait.output is None
    fresh_in, fresh_out = wait.operands(torch.zeros(1))
    assert fresh_in.shape == BARRIER_SHAPE and fresh_out.shape == BARRIER_SHAPE
