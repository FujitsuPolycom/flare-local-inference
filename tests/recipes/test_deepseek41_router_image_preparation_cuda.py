# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare image-enabled shard routing and preserve its native routes through four-host replay.

Use the configured dependency bootstrap and four-host test opt-ins documented in
`test_deepseek41_route_record_roce_cuda.py`. This case uses actual PreparedLayer router
declarations, native router preparation/execution, and RoCE record transport. Other layer
operators are declaration-only doubles; the case does not execute or qualify a complete model.
"""

from datetime import timedelta
import os
import socket
from urllib.parse import urlsplit

import pytest
import torch
import torch.distributed as dist

from .test_deepseek41_route_record_roce_cuda import _close_resources
from .test_deepseek41_router_image_preparation import EXPERTS, TOKENS, TOP_K, WORLD, _layer


pytestmark = pytest.mark.skipif(
    any(os.environ.get(name) != "1" for name in (
        "FLARE_RUN_CUDA_TESTS", "FLARE_RUN_TP_CUDA_TESTS", "FLARE_RUN_ROCE_TESTS")),
    reason="requires four assigned CUDA/RoCE hosts and their GPU test opt-ins",
)


@torch.no_grad()
def test_image_router_preparation_and_gather_match_native_full_rows(monkeypatch):
    from ignition.components.collectives import PreparedGatherShards
    from ignition.components.moe import PreparedRouter, RouteRecord, RouterScratch
    from ignition.preparation.collectives import CollectivePreparation, create_roce_runtime
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.host_ranks import _use_interface
    from ignition.preparation.tuning import RecordedPreparationSession

    assert torch.cuda.is_available()
    rank = int(os.environ["FLARE_TEST_RANK"])
    assert 0 <= rank < WORLD
    address = urlsplit(os.environ["FLARE_TEST_RENDEZVOUS"])
    assert address.scheme == "tcp" and address.hostname and address.port
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((address.hostname, address.port))
        _use_interface(route.getsockname()[0])
    dist.init_process_group("gloo", init_method=address.geturl(), rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=5))
    runtime = session = None
    graphs = []
    try:
        coordinator = CollectivePreparation(dist.group.WORLD)
        assert coordinator.spans_hosts
        rows, start = TOKENS // WORLD, rank * (TOKENS // WORLD)
        runtime = create_roce_runtime(dist.group.WORLD, device, max_tokens=1,
            pipe_tokens=TOKENS, hidden_size=4 * TOP_K, gather_bytes=TOKENS * TOP_K * 8,
            ring=True)
        session = RecordedPreparationSession(device=device, checkpoint="synthetic-image-router",
            workload_schema="deepseek41-image-mask-roce", numerical_policy="exact-route-bits",
            coordinator=coordinator, compile_workers=2)
        correction = torch.linspace(-.5, .5, EXPERTS).to(device)
        image_correction = -correction
        layer, request = _layer(monkeypatch, sharded=True, rank=rank, vision=True,
            device=device, correction=correction, image_correction=image_correction)
        session.declare((request,))
        full_logits = ((torch.arange(TOKENS * EXPERTS).reshape(TOKENS, EXPERTS) % 31 - 15)
                       .float() / 64).to(device)
        layer.row_shard.router_logits.copy_(full_logits[start:start + rows])
        full_mask = layer.image_mask
        assert full_mask.shape == (TOKENS,)
        local_mask = full_mask[start:start + rows]
        full_mask.copy_(torch.arange(TOKENS) % 7 == 3)
        reference = PreparedRouter(full_logits,
            RouterScratch.allocate(TOKENS, device, experts=EXPERTS, top_k=TOP_K),
            session=session, name="full-image-reference", score_func="sqrtsoftplus",
            renormalize=True, routed_scaling_factor=1., correction_bias=correction,
            image_correction_bias=image_correction, image_mask=full_mask)
        gathered_records = RouteRecord.allocate(TOKENS, TOP_K, device)
        gather = PreparedGatherShards(layer.row_shard.routes.packed, gathered_records.packed,
            runtime=runtime, coordinator=coordinator, session=session, name="image-route-gather")
        ids = torch.empty((TOKENS, TOP_K), dtype=torch.int32, device=device)
        weights = torch.empty((TOKENS, TOP_K), dtype=torch.float32, device=device)
        session.tune(show_progress=False)
        # Run the actual declaration callback with the actual prepared native state. A full-row
        # mask passed to this 1024-row router must fail here, before runtime functional routing.
        request.prepare_call(layer.router.plan.prepared.state).run()
        torch.cuda.synchronize(device)
        router = layer.router
        local_logits = layer.row_shard.router_logits
        router.functional(session=session)
        reference.functional(session=session)
        gather.functional(session=session)
        reference(full_logits, image_mask=full_mask)
        torch.cuda.synchronize(device)

        def forward():
            result = router(local_logits, image_mask=local_mask)
            record = RouteRecord.allocate(rows, TOP_K, device)
            record.write(result.ids, result.weights)
            RouteRecord(gather(record.packed).view(torch.int32)).read(ids, weights)

        functions = [(False, forward), (True, compile_forward(forward))]
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        for _, function in functions:
            dist.barrier()
            with torch.cuda.stream(stream):
                function()
            torch.cuda.synchronize(device)
        pool = torch.cuda.graph_pool_handle()
        for compiled, function in functions:
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                    graph, pool=pool, stream=stream):
                function()
            graphs.append((None, compiled, graph, function))
        session.freeze()
        text_ids = None
        for label in ("text", "mixed", "image"):
            mask = (torch.zeros(TOKENS, dtype=torch.bool) if label == "text" else
                    torch.ones(TOKENS, dtype=torch.bool) if label == "image" else
                    torch.arange(TOKENS) % 7 == 3)
            full_mask.copy_(mask)
            expected = reference(full_logits, image_mask=full_mask)
            if label == "text":
                text_ids = expected.ids.clone()
            else:
                assert bool((expected.ids[full_mask] != text_ids[full_mask]).any())
            for _, compiled, graph, _ in graphs:
                ids.fill_(-12345)
                weights.fill_(float("nan"))
                torch.cuda.synchronize(device)
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize(device)
                runtime.check_health()
                assert torch.equal(ids, expected.ids), (rank, label, compiled)
                assert torch.equal(weights.view(torch.int32), expected.weights.view(torch.int32)), (
                    rank, label, compiled)
    finally:
        _close_resources(device, graphs, session, runtime)
