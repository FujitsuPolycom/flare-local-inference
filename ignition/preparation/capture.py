# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compile prepared model forwards and capture CUDA graphs for their shapes."""

from contextlib import nullcontext
from time import monotonic

import torch

from ignition.progress import graph_catalog, phase

from .aot import ForwardCache, allocation_order
from .collectives import warmup_wait
from .compilation import _defer_full_collection
from .runtime_core import capture_graph, instantiate_graph, record_export_graph


_warmup_streams = {}
_warmup_pools = {}


def _warmup_stream(device):
    """Return the per-device stream that eager warmups run on.

    PyTorch caches one cuBLAS/cuBLASLt workspace per (handle, stream).
    Warmups allocate inside transient storage, so their workspaces must
    stay off the capture stream: the first capture then allocates the
    stream's workspace inside the recording pool, and the cache retains
    it for every graph the stream captures afterward.
    """
    # The stream must be created outside PyTorch's round-robin pool: a
    # caller's later torch.cuda.Stream can alias a pool stream, and
    # workspaces are cached per CUDA stream, so an aliased capture stream
    # would reuse a warmup workspace's transient address. The stream and
    # its CUDA handle live for the life of the process.
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
        stream = _warmup_streams.get(index)
        if stream is None:
            from cuda.bindings import driver as cuda

            status, handle = cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING)
            if status != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cannot create warmup stream: {status}")
            stream = _warmup_streams[index] = torch.cuda.ExternalStream(int(handle), device=device)
    return stream


def _seed_warmup_workspaces(stream, device):
    """Seed the calling thread's warmup-stream workspace in a dedicated pool.

    Workspaces are cached per thread's cuBLAS handle, so seeding runs on
    each capture call under the ambient BLAS backend, the backend the
    warmups use. A workspace created inside transient storage keeps its
    address in the cache after the transient backing is released, and a
    later call could reuse it; a dedicated pool keeps the seed's backing
    mapped for the life of the cache entry, outside transient storage,
    any caller-owned pool, and the exported graph pool.
    """
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
        pool = _warmup_pools.get(index)
        if pool is None:
            pool = _warmup_pools[index] = torch.cuda.MemPool()
    with torch.cuda.device(index), torch.cuda.stream(stream), torch.cuda.use_mem_pool(pool):
        operand = torch.empty((512, 512), dtype=torch.bfloat16, device=device)
        torch.mm(operand, operand, out=torch.empty_like(operand))
    stream.synchronize()


def _capture_order(names, compiled, coordinator):
    sizes = {name: forward.capture_allocation_sizes for name, forward in compiled.items()}
    return allocation_order(names, sizes, coordinator)


@_defer_full_collection()
def capture_forwards(functions, signatures, *, session, device, stream, pool,
                     collective=None, coordinator=None, compile_decoder=True, materialize=None,
                     progress=None, label="Prepare serving graphs", cache_root=None, retain_graphs=False,
                     warmup_storage=None, capture_names=None, before_run=None,
                     report_catalog=True, retain_forwards=False, compile_options=None,
                     compile_kinds=("prefill", "terminal_prefill", "scored_prefill", "decode", "resume", "speculative", "mixed", "vision")):
    """Retain compiled resources and capture with an optional b12x IPC runtime.

    Persistent state and graph outputs must live in caller-owned storage.
    Only graph-local intermediates may use the shared graph memory pool;
    serving replays variants serially and may choose them in any order.
    All eager warmups precede capture so their transient workspaces do not
    coexist with retained graph intermediates.
    Graphs captured on one stream share that stream's cached workspace;
    resetting any of them frees the workspace the others still reference.
    Callers must reset all of a capture stream's graphs together or keep
    separately reset groups on separate streams.
    ``retain_graphs`` keeps templates and executables for export when
    ``warmup_storage`` supplies external CUDA pool owners.
    ``before_run`` restores caller-owned trial inputs outside compiled warmup
    and graph capture; it is unused for the ordinary serving catalog.
    ``retain_forwards`` preserves guarded Python execution for eager diagnostics.
    ``compile_options`` overrides Inductor configuration passed to ``ForwardCache``
    without mutating the module defaults.
    """
    if (report_catalog and progress is not None and not getattr(session, "_graph_catalog_reported", False)
            and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)):
        graph_catalog(progress, (signatures[name] for name in functions))
        session._graph_catalog_reported = True
    names = tuple(functions) if capture_names is None else tuple(capture_names)
    if len(set(names)) != len(names) or not set(names) <= functions.keys():
        raise ValueError("capture selection must name distinct prepared forwards")
    graphs, compiled = {}, {}
    warmup = nullcontext if warmup_storage is None else warmup_storage.transient
    warm = _warmup_stream(device)
    _seed_warmup_workspaces(warm, device)
    if compile_decoder:
        cache = ForwardCache(namespace=session.namespace.to_dict(), device=device, root=cache_root,
                             headroom_bytes=getattr(session, "headroom_bytes", 0),
                             options=compile_options,
                             compile_workers=getattr(session, "requested_compile_workers", None),
                             lock_gpu_clocks=getattr(session, "lock_gpu_clocks", False) and session.autotune)
        forwards = {name: function for name, function in functions.items()
                    if name in names and signatures[name].kind in compile_kinds}
        warm.wait_stream(torch.cuda.current_stream(device))
        warm.wait_stream(stream)
        retention = nullcontext() if warmup_storage is None else warmup_storage.retain_transient_backing()
        with retention, warmup_wait(collective, coordinator):
            with torch.cuda.stream(warm):
                compiled = cache.prepare_many(forwards, signatures, progress=progress,
                                              coordinator=coordinator, materialize=materialize, warmup=warmup,
                                              before_run=before_run)
            warm.synchronize()
        with phase(progress, "Release serialized compiler state", total=len(compiled)) as stage:
            released, counted = 0, set()
            for forward in compiled.values():
                artifact = id(getattr(forward, "artifact", forward))
                size = forward.prepare_capture()
                if artifact not in counted:
                    released += size
                    counted.add(artifact)
                stage.update(advance=1)
            stage.update(f"{released >> 20} MiB; guards and kernel owners retained")
        from .host_memory import release_unused

        release_unused()
    elif materialize is not None:
        materialize()
    with phase(progress, "Warm serving graphs", total=len(names)) as stage:
        for name in names:
            function = functions[name]
            if coordinator is not None:
                coordinator.barrier(f"capture warmup {name}", coordinator.ranks)
            warm.wait_stream(torch.cuda.current_stream(device))
            warm.wait_stream(stream)
            with warmup(), torch.cuda.stream(warm):
                if before_run is not None:
                    before_run(name)
                if name not in compiled:
                    function()
            warm.synchronize()
            stage.update(advance=1)
    stream.wait_stream(warm)
    ordered_names = _capture_order(names, compiled, coordinator)
    with phase(progress, label, total=len(names)) as stage:
        for name in ordered_names:
            function = compiled.get(name, functions[name])
            if coordinator is not None:
                coordinator.barrier(f"capture {name}", coordinator.ranks)
            started = monotonic()
            stream.wait_stream(torch.cuda.current_stream(device))
            exporting = retain_graphs and warmup_storage is not None
            graph = torch.cuda.CUDAGraph(keep_graph=True) if exporting else capture_graph(retain_graphs=retain_graphs)
            communication = nullcontext() if collective is None else collective.capture(stream=stream)
            def memory_detail():
                free, total = torch.cuda.mem_get_info(device)
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                return (f"{free / (1 << 20):.0f} MiB free, "
                        f"{allocated / (1 << 20):.0f} MiB allocated, "
                        f"{reserved / (1 << 20):.0f} MiB reserved, "
                        f"{(total - free - reserved) / (1 << 20):.0f} MiB outside PyTorch")

            try:
                if progress is not None:
                    progress(f"capturing {name}: {memory_detail()}")
                if before_run is not None:
                    warm.wait_stream(stream)
                    with torch.cuda.stream(warm):
                        before_run(name)
                    stream.wait_stream(warm)
                with communication, session.capture(), torch.cuda.graph(graph, pool=pool, stream=stream):
                    function()
                if exporting:
                    graph = record_export_graph(graph, warmup_storage, owners=(function,))
                elif retain_graphs:
                    if progress is not None:
                        progress(f"instantiating {name}: {memory_detail()}")
                    instantiate_graph(graph)
            except Exception as error:
                error.add_note(f"Graph {name!r} on {device}: {memory_detail()}")
                raise
            if name in compiled and not retain_forwards:
                function.retire_after_capture()
            graphs[name] = graph
            stage.update(advance=1)
            if progress is not None:
                cached = " (compiled cache hit)" if name in compiled and compiled[name].cache_hit else ""
                progress(f"captured {name} in {monotonic() - started:.2f}s{cached}")
    torch.cuda.synchronize(device)
    return graphs, compiled
