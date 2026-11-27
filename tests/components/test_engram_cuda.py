# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disk row staging overlaps captured work until the Engram consumer waits."""

import os
import threading
import time

import pytest
import torch
from b12x.preparation import PreparationSession
from b12x.preparation.types import require_prepared
from b12x.sequence import engram
from b12x.sequence._shared.disk_table import MappedHostAllocation
from cuda.bindings import driver

from ignition.components.engram import EngramInputs, PreparedEngramLookup
from ignition.components.readiness import ReadinessTimings

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


class LookupForward:
    def __init__(self, source, lookup):
        self.source, self.lookup = source, lookup

    def __call__(self):
        self.source.add_(1)
        return self.lookup(self.source)


@pytest.mark.parametrize("tokens", [1, 4, 4096])
@torch.no_grad()
def test_compiled_disk_lookup_waits_at_consumer(tmp_path, tokens):
    from ignition.preparation.compile_pool import CompilePool

    device = torch.device("cuda:0")
    geometry = engram.build_geometry(base_table_size=101, compressed_vocab_size=32)
    table_rows = geometry.num_embeddings[0]
    weight = (torch.arange(table_rows * 256).remainder(15).float() - 7).reshape(table_rows, 256).to(torch.float8_e4m3fn)
    scales = torch.arange(table_rows * 8).remainder(5).add(125).byte().reshape(table_rows, 8)
    paths = [tmp_path / "weights.bin", tmp_path / "scales.bin"]
    for path, tensor in zip(paths, (weight, scales), strict=True):
        path.write_bytes(bytes(4093) + tensor.view(torch.uint8).numpy().tobytes())
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    inputs = EngramInputs.allocate(tokens, device, timings=ReadinessTimings.allocate(frame, 1))
    output = torch.empty((tokens, 6144), dtype=torch.bfloat16, device=device)
    count = torch.tensor([tokens], dtype=torch.int32, device=device)
    ready = MappedHostAllocation((1,), torch.int32, device)
    progress = MappedHostAllocation((1,), torch.int32, device)
    ready.host_view.zero_()
    progress.host_view.zero_()
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        lookup = PreparedEngramLookup(inputs, output, count, layer=1, token_map=tuple(range(32)),
            geometry=geometry, session=session, name="disk_lookup")
        forward = LookupForward(progress.device_view, lookup)
        path = tmp_path / "compiled.bin"
        with CompilePool(device) as pool:
            pool.submit(forward, path).result()
        with path.open("rb") as file:
            compiled = torch.compiler.load_compiled_function(file)
        compiled(forward)
        table = engram.DiskTable(require_prepared(lookup.plan, "sequence.engram", device), queue_depth=8)
        for scale, path in zip((False, True), paths, strict=True):
            table.add_shard(0, str(path), 4093, scale=scale)
        table.freeze()
        copy_stream, compute = torch.cuda.Stream(), torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph, stream=compute):
            compiled(forward)
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        try:
            for generation, live in enumerate((tokens, max(1, tokens // 2), 0), 1):
                frame.fill_(generation % 2)
                ids = (torch.arange(tokens * 24).reshape(tokens, 24) * 19 + generation) % table_rows
                inputs.hashes.copy_(ids)
                count.fill_(live)
                output.fill_(float("nan"))
                progress.host_view.zero_()
                torch.cuda.synchronize()
                errors = []

                def read():
                    try:
                        deadline = time.monotonic() + 10
                        while progress.host_view.item() != 1:
                            if time.monotonic() > deadline:
                                raise RuntimeError("captured work did not run before the disk read")
                            time.sleep(.0001)
                        time.sleep(.01)
                        cache = table._cache
                        cache.ids_host.copy_(ids.flatten())
                        cache._native.ple_reader_run(cache._reader, cache._ids_buffer,
                            cache._weight_buffer, cache._scale_buffer, tokens * 24)
                    except BaseException as error:
                        errors.append(error)
                    finally:
                        ready.host_view.fill_(generation)

                with torch.cuda.stream(copy_stream):
                    status, = driver.cuStreamWaitValue32(driver.CUstream(copy_stream.cuda_stream),
                        ready.device_view.data_ptr(), generation, driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ)
                    assert int(status) == 0
                    inputs.weights.copy_(table._cache.weight_host, non_blocking=True)
                    inputs.scales.copy_(table._cache.scale_host, non_blocking=True)
                    inputs.ready.record(copy_stream)
                worker = threading.Thread(target=read)
                worker.start()
                with torch.cuda.stream(compute):
                    graph.replay()
                compute.synchronize()
                worker.join()
                if errors:
                    raise errors[0]
                assert torch.cuda.memory_allocated() == allocated
                before, ready_at, synchronized = inputs.timings.timestamps.cpu()[generation % 2].tolist()
                assert before < ready_at <= synchronized
                assert 8_000_000 <= ready_at - before < 1_000_000_000
                expected = (weight.float()[ids] * scales.view(torch.float8_e8m0fnu).float()[ids]
                            .repeat_interleave(32, dim=-1)).bfloat16().flatten(1)
                expected[live:].zero_()
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        finally:
            graph.reset()
            table.close()
    ready.close()
    progress.close()


@pytest.mark.parametrize("tokens", [1, 8, 128, 4096])
@torch.no_grad()
def test_compact_local_heads_matches_full_lookup_cuda_graph(tokens):
    device = torch.device("cuda:0")
    geometry = engram.build_geometry(base_table_size=101, compressed_vocab_size=32)
    table_rows = geometry.num_embeddings[0]
    token_map = tuple(range(32))
    local_heads = (0, 4, 8, 12, 16, 20)
    heads = len(local_heads)
    columns = torch.tensor(
        [head * 256 + col for head in local_heads for col in range(256)],
        dtype=torch.int64,
        device=device,
    )

    frame = torch.zeros(1, dtype=torch.int32, device=device)
    full_inputs = EngramInputs.allocate(tokens, device, timings=ReadinessTimings.allocate(frame, 1))
    compact_inputs = EngramInputs.allocate(tokens, device, timings=ReadinessTimings.allocate(frame, 1), heads=heads)
    full_out = torch.empty((tokens, 24 * 256), dtype=torch.bfloat16, device=device)
    compact_out = torch.empty((tokens, heads * 256), dtype=torch.bfloat16, device=device)
    count = torch.tensor([tokens], dtype=torch.int32, device=device)
    source = torch.zeros((1,), dtype=torch.int32, device=device)

    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        full_lookup = PreparedEngramLookup(
            full_inputs, full_out, count, layer=1, token_map=token_map,
            geometry=geometry, session=session, name="full_lookup",
        )
        compact_lookup = PreparedEngramLookup(
            compact_inputs, compact_out, count, layer=1, token_map=token_map,
            geometry=geometry, session=session, name="compact_lookup",
        )
        # Warm eager compilation outside CUDA graph capture.
        full_inputs.ready.record(torch.cuda.current_stream(device))
        compact_inputs.ready.record(torch.cuda.current_stream(device))
        full_lookup(source)
        compact_lookup(source)
        torch.cuda.synchronize()

        compute = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph, stream=compute):
            compact_lookup(source)
        torch.cuda.synchronize()
        try:
            weights_24 = (
                (torch.arange(tokens * 24 * 256, device=device).remainder(15).float() - 7)
                .reshape(tokens, 24, 256)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
            scales_24 = (
                (torch.arange(tokens * 24 * 8, device=device).remainder(7) + 124)
                .to(torch.uint8)
                .reshape(tokens, 24, 8)
            )
            # Exercise E8M0 0 (2^-127 subnormal, not zero) and 255 (NaN) on owned head 0.
            weights_24[0, 0].copy_(torch.full((256,), 1.0, dtype=torch.float8_e4m3fn, device=device).view(torch.uint8))
            scales_24[0, 0, 0] = 0
            scales_24[0, 0, 1] = 128
            scales_24[0, 0, 2] = 255

            hashes_24 = (torch.arange(tokens * 24, device=device).reshape(tokens, 24) * 19 + 3) % table_rows
            # Exercise -1 sentinel and out-of-range row on owned heads 4 and 8 (local indices 1 and 2)
            # with non-zero staged weights/scales behind them.
            hashes_24[0, 4] = -1
            hashes_24[0, 8] = table_rows
            scales_24[0, 4, :] = 255
            scales_24[0, 8, :] = 255

            head_idx = torch.tensor(local_heads, dtype=torch.int64, device=device)
            full_inputs.weights.copy_(weights_24.reshape(tokens * 24, 256))
            full_inputs.scales.copy_(scales_24.reshape(tokens * 24, 8))
            full_inputs.hashes.copy_(hashes_24)
            compact_inputs.weights.copy_(weights_24[:, head_idx].reshape(tokens * heads, 256))
            compact_inputs.scales.copy_(scales_24[:, head_idx].reshape(tokens * heads, 8))
            compact_inputs.hashes.copy_(hashes_24[:, head_idx])
            del weights_24, scales_24, hashes_24, head_idx
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()

            live_cases = (tokens, 0) if tokens == 1 else (tokens, max(1, tokens // 2), 0)
            for generation, live in enumerate(live_cases, 1):
                frame.fill_(generation % 2)
                count.fill_(live)
                full_out.fill_(float("nan"))
                compact_out.fill_(float("nan"))
                stream = torch.cuda.current_stream(device)
                full_inputs.ready.record(stream)
                compact_inputs.ready.record(stream)
                full_lookup(source)
                torch.cuda.synchronize()

                before_replay = torch.cuda.memory_allocated()
                with torch.cuda.stream(compute):
                    graph.replay()
                compute.synchronize()

                assert torch.cuda.memory_allocated() == before_replay == allocated
                expected = torch.index_select(full_out, 1, columns)
                assert torch.equal(compact_out.view(torch.uint16), expected.view(torch.uint16))

                if live > 0:
                    # Explicit edge-case assertions prevent vacuous parity.
                    assert torch.equal(
                        compact_out[0, 0:32],
                        torch.full((32,), 2.0**-127, dtype=torch.bfloat16, device=device),
                    )
                    assert torch.all(compact_out[0, 32:64] == 2.0)
                    assert torch.all(torch.isnan(compact_out[0, 64:96]))
                    assert torch.all(compact_out[0, 256:768] == 0.0)
                if live < tokens:
                    assert torch.all(compact_out[live:] == 0.0)
                del expected
        finally:
            graph.reset()
