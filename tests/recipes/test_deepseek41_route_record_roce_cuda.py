# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preserve native router IDs and FP32 weight bits through captured four-host shard gathers.

Run one process on each of four assigned RoCE hosts with FLARE_RUN_CUDA_TESTS=1,
FLARE_RUN_TP_CUDA_TESTS=1, FLARE_RUN_ROCE_TESTS=1, FLARE_TEST_RANK=0..3, and the same unused
rank-zero fabric tcp://HOST:PORT in FLARE_TEST_RENDEZVOUS. Use matching preparation sources
and dependencies. Initialize the configured dependency caches before importing pytest or Torch;
otherwise Inductor can select a different Triton cache from the one used during preparation.
From the checkout root, with those test variables set:

    python - <<'PY'
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply
    from ignition.cli.preparation import Host
    gpus = {gpu.index: gpu for gpu in Host.discover().gpus}
    apply(Configuration.load(), (gpus[0],))
    import pytest
    raise SystemExit(pytest.main([
        "-p", "no:cacheprovider", "-q", "-s",
        "tests/recipes/test_deepseek41_route_record_roce_cuda.py",
    ]))
    PY

Pass an explicit preparation configuration path to Configuration.load when its cache settings
are not in the normal user configuration. No checkpoint is
required. The test exercises the component sequence used by row-sharded prefill, not a complete
model or exported package. Padding follows the decoder's unmasked router contract: padded rows
are compared with the real full-row router rather than assigned a new inactive-row sentinel.
"""

from datetime import timedelta
import os
import socket
import sys
from urllib.parse import urlsplit

import pytest
import torch
import torch.distributed as dist


pytestmark = pytest.mark.skipif(
    any(os.environ.get(name) != "1" for name in (
        "FLARE_RUN_CUDA_TESTS", "FLARE_RUN_TP_CUDA_TESTS", "FLARE_RUN_ROCE_TESTS")),
    reason="requires four explicitly assigned CUDA/RoCE hosts and their GPU test opt-ins",
)

WORLD, EXPERTS, TOP_K = 4, 384, 6
CAPACITIES = (2048, 4096, 6144, 8192)


def _logits(rows, live, seed, *, tied=False):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn((rows, EXPERTS), generator=generator, dtype=torch.float32)
    if tied:
        values.zero_()
    values[live:].zero_()
    return values


def _assert_routes(ids, weights, expected, *, capacity, live, compiled, tied):
    context = dict(capacity=capacity, live=live, compiled=compiled, tied=tied)
    assert torch.equal(ids, expected.ids), f"expert IDs differ: {context}"
    assert torch.equal(weights.view(torch.int32), expected.weights.view(torch.int32)), (
        f"FP32 route-weight bits differ: {context}")
    if tied:
        # b12x _route_topk_values selects tl.max(expert_id) among equal maximum scores.
        descending = torch.arange(EXPERTS - 1, EXPERTS - TOP_K - 1, -1,
                                  device=ids.device, dtype=torch.int32)
        assert torch.equal(ids, descending.expand_as(ids)), (
            f"native descending-ID tie ordering differs: {context}")
    elif live:
        # A BF16 conversion would discard these low FP32 bits even if selected IDs survived.
        assert bool((expected.weights[:live].view(torch.int32) & 0xffff).any()), context


def _close_resources(device, graphs, session, runtime):
    original = sys.exception()
    errors = []
    try:
        torch.cuda.synchronize(device)
    except BaseException as error:
        errors.append(error)
    else:
        # Kernel and transport owners remain alive until all captured device work drains.
        closers = [graph.reset for _, _, graph, _ in graphs]
        closers += ([] if session is None else [session.close])
        closers += ([] if runtime is None else [runtime.close])
        for close in closers:
            try:
                close()
            except BaseException as error:
                errors.append(error)
    try:
        dist.destroy_process_group()
    except BaseException as error:
        errors.append(error)
    if errors:
        failure = original if original is not None else errors[0]
        for error in errors:
            failure.add_note(f"route-record resource cleanup failed: {error!r}")
        if original is None:
            raise failure


@torch.no_grad()
def test_native_route_records_survive_roce_graph_transitions():
    from ignition.components.collectives import PreparedGatherShards
    from ignition.components.moe import PreparedRouter, RouteRecord, RouterScratch
    from ignition.preparation.collectives import CollectivePreparation, create_roce_runtime
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.host_ranks import _use_interface
    from ignition.preparation.tuning import RecordedPreparationSession

    assert torch.cuda.is_available(), "the opted-in RoCE test requires an assigned CUDA device"
    rank = int(os.environ["FLARE_TEST_RANK"])
    assert 0 <= rank < WORLD
    rendezvous = os.environ["FLARE_TEST_RENDEZVOUS"]
    address = urlsplit(rendezvous)
    assert address.scheme == "tcp" and address.hostname and address.port
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((address.hostname, address.port))
        _use_interface(route.getsockname()[0])
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=5))
    coordinator = CollectivePreparation(dist.group.WORLD)
    runtime = session = None
    graphs = []
    try:
        assert coordinator.spans_hosts, "this case requires four separate RoCE hosts"
        # Six ID/weight pairs occupy 48 bytes, or 24 BF16 transport words per row.
        packed_columns = 4 * TOP_K
        runtime = create_roce_runtime(dist.group.WORLD, device, max_tokens=1,
            pipe_tokens=max(CAPACITIES), hidden_size=packed_columns,
            gather_bytes=max(CAPACITIES) * packed_columns * 2, ring=True)
        cases = []
        session = RecordedPreparationSession(device=device, checkpoint="synthetic-route-records",
                workload_schema="deepseek41-native-topk-roce", numerical_policy="exact-route-bits",
                coordinator=coordinator, compile_workers=2)
        for capacity in CAPACITIES:
            local_rows = capacity // WORLD
            # The existing decoder shards whole 512-row blocks per rank.
            assert capacity % WORLD == 0 and local_rows % 512 == 0, capacity
            full_logits = torch.zeros((capacity, EXPERTS), dtype=torch.float32, device=device)
            local_logits = torch.zeros((local_rows, EXPERTS), dtype=torch.float32, device=device)
            declaration = RouteRecord.allocate(local_rows, TOP_K, device)
            gathered_declaration = RouteRecord.allocate(capacity, TOP_K, device)
            selected = torch.zeros((local_rows, TOP_K), dtype=torch.float32, device=device)
            reference_scratch = RouterScratch.allocate(capacity, device, experts=EXPERTS, top_k=TOP_K)
            options = dict(score_func="sqrtsoftplus", renormalize=True, routed_scaling_factor=1.)
            router = PreparedRouter(local_logits, declaration.router_scratch(selected),
                session=session, name=f"route-{capacity}", **options)
            reference = PreparedRouter(full_logits, reference_scratch,
                session=session, name=f"full-row-reference-{capacity}", **options)
            gather = PreparedGatherShards(declaration.packed, gathered_declaration.packed,
                runtime=runtime, coordinator=coordinator, session=session,
                name=f"route-record-gather-{capacity}")
            ids = torch.empty((capacity, TOP_K), dtype=torch.int32, device=device)
            weights = torch.empty((capacity, TOP_K), dtype=torch.float32, device=device)
            router.functional(session=session)
            reference.functional(session=session)
            gather.functional(session=session)

            def forward(router=router, gather=gather, local_logits=local_logits,
                        ids=ids, weights=weights):
                routed = router(local_logits)
                packed = RouteRecord.allocate(routed.ids.shape[0], TOP_K, routed.ids.device)
                packed.write(routed.ids, routed.weights)
                gathered = gather(packed.packed)
                RouteRecord(gathered.view(torch.int32)).read(ids, weights)

            cases.append((capacity, local_logits, full_logits, reference, ids, weights, forward))
        session.tune(show_progress=False)
        pool = torch.cuda.graph_pool_handle()
        functions = [(case, compiled, compile_forward(case[-1]) if compiled else case[-1])
                     for case in cases for compiled in (False, True)]
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        # Warm the complete catalog before retaining any graph in the shared pool.
        for _ in range(2):
            for case, _, function in functions:
                case[3](case[2])
                dist.barrier()
                with torch.cuda.stream(stream):
                    function()
                torch.cuda.synchronize(device)
        for case, compiled, function in functions:
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                    graph, pool=pool, stream=stream):
                function()
            graphs.append((case, compiled, graph, function))
        session.freeze()

        # Replay small and large capacities in both orders with changing rank-local data.
        for iteration in range(2):
            ordered = graphs if iteration == 0 else list(reversed(graphs))
            for case, compiled, graph, _ in ordered:
                capacity, local_logits, full_logits, reference, ids, weights, _ = case
                local_rows = capacity // WORLD
                live_counts = sorted({0, 1, capacity - 1, capacity,
                    *(boundary * local_rows + offset
                      for boundary in range(1, WORLD) for offset in (-1, 0, 1))})
                for live in live_counts:
                    values = _logits(capacity, live, 284000 + iteration * 100 + live)
                    full_logits.copy_(values)
                    local_logits.copy_(values[rank * local_rows:(rank + 1) * local_rows])
                    expected = reference(full_logits)
                    ids.fill_(-12345)
                    weights.fill_(float("nan"))
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    before = torch.cuda.memory_stats(device)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    after = torch.cuda.memory_stats(device)
                    runtime.check_health()
                    for counter in ("allocation.all.allocated", "allocation.all.freed",
                                    "segment.all.allocated", "segment.all.freed"):
                        assert before[counter] == after[counter], (capacity, compiled, counter)
                    _assert_routes(ids, weights, expected, capacity=capacity, live=live,
                                   compiled=compiled, tied=False)
                values = _logits(capacity, capacity, 0, tied=True)
                full_logits.copy_(values)
                local_logits.copy_(values[rank * local_rows:(rank + 1) * local_rows])
                expected = reference(full_logits)
                ids.fill_(-12345)
                weights.fill_(float("nan"))
                torch.cuda.synchronize(device)
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize(device)
                runtime.check_health()
                _assert_routes(ids, weights, expected, capacity=capacity, live=capacity,
                               compiled=compiled, tied=True)
    finally:
        _close_resources(device, graphs, session, runtime)
