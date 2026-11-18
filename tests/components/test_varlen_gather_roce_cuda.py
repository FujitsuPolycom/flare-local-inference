# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compile one variable-length gather artifact for two RoCE hosts.

Run on two assigned hosts with FLARE_RUN_ROCE_TESTS=1, FLARE_TEST_RANK=0 or 1,
and the same FLARE_TEST_RENDEZVOUS=tcp://ADDRESS:PORT.
"""

from datetime import timedelta
import os
from pathlib import Path
import socket
from urllib.parse import urlsplit

import pytest
import torch
import torch.distributed as dist

from ignition.components.collectives import PreparedVarlenGather
from ignition.preparation.aot import ForwardCache
from ignition.preparation.collectives import CollectivePreparation, create_roce_runtime
from ignition.preparation.execution import GraphSignature
from ignition.preparation.host_ranks import _use_interface
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_ROCE_TESTS") != "1",
                               reason="requires two assigned RoCE hosts")


class GatherForward:
    def __init__(self, gather, device):
        self.gather = gather
        self.rows = torch.zeros((2, 256), dtype=torch.uint8, device=device)
        self.counts = torch.full((2,), 256, dtype=torch.int64, device=device)

    def __call__(self):
        self.gather(output=self.rows, counts=self.counts)


@torch.no_grad()
def test_shared_compiled_varlen_gather_preserves_rank_rows_and_tails(tmp_path):
    rank = int(os.environ["FLARE_TEST_RANK"])
    rendezvous = os.environ["FLARE_TEST_RENDEZVOUS"]
    address = urlsplit(rendezvous)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((address.hostname, address.port))
        _use_interface(route.getsockname()[0])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(minutes=5))
    runtime = None
    graphs = []
    try:
        coordinator = CollectivePreparation(dist.group.WORLD)
        assert coordinator.spans_hosts
        runtime = create_roce_runtime(dist.group.WORLD, device, max_tokens=1, pipe_tokens=1,
            hidden_size=8, gather_bytes=256, varlen_bytes=256)
        arena = ScratchArena(device)
        with RecordedPreparationSession(device=device, checkpoint="varlen-gather",
                workload_schema="rank-local-source-v1", numerical_policy="exact-bytes",
                coordinator=coordinator, compile_workers=1) as session:
            session.register_scratch(arena)
            rows = torch.zeros((2, 256), dtype=torch.uint8, device=device)
            counts = torch.full((2,), 256, dtype=torch.int64, device=device)
            gather = PreparedVarlenGather(rows, counts, runtime=runtime, session=session,
                coordinator=coordinator, name="gather").functional(session=session)
            session.tune(show_progress=False)
            arena.release_preparation()
            roots = [str(tmp_path) if rank == 0 else None]
            dist.broadcast_object_list(roots, src=0)
            cache = ForwardCache(namespace={"test": "shared-varlen-gather"}, device=device,
                                 root=Path(roots[0]), compile_workers=1)
            signatures = {"gather": GraphSignature("prefill", 1, 1)}
            for restored in (False, True):
                forward = GatherForward(gather, device)
                bound = cache.prepare_many({"gather": forward}, signatures,
                    coordinator=coordinator, materialize=lambda: None)["gather"]
                assert bound.cache_hit is restored
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                    bound()
                graphs.append((graph, bound, forward))

                for iteration, lengths in enumerate(((256, 256), (16, 48), (0, 80), (112, 0), (0, 0), (256, 256))):
                    local = torch.full((2, 256), 239, dtype=torch.uint8)
                    local[rank] = ((torch.arange(256) + rank * 71 + iteration * 13) % 239).to(torch.uint8)
                    expected = local.clone()
                    for peer, length in enumerate(lengths):
                        expected[peer, :length] = (
                            (torch.arange(length) + peer * 71 + iteration * 13) % 239).to(torch.uint8)
                    forward.rows.copy_(local)
                    forward.counts.copy_(torch.tensor(lengths, dtype=torch.int64))
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    before = torch.cuda.memory_stats(device)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    after = torch.cuda.memory_stats(device)
                    runtime.check_health()
                    torch.testing.assert_close(forward.rows.cpu(), expected, rtol=0, atol=0)
                    for counter in ("allocation.all.allocated", "allocation.all.freed",
                                    "segment.all.allocated", "segment.all.freed"):
                        assert before[counter] == after[counter]
    finally:
        for graph, _, _ in graphs:
            graph.reset()
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()
