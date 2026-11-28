# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Trace and compile prepared forwards in workers without model storage."""

import multiprocessing
import os
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from ignition import runtime_patches

from . import host_memory
from .compilation import _OPTIONS, _compile_options, _defer_full_collection
from .host_memory import cpu_workers, device_memory_info

# Device memory one compiler worker can hold: its CUDA context plus Inductor's constant folding.
_WORKER_DEVICE_BYTES = 2 << 30
# Minimum unified-memory unit when growing an integrated GPU's pool past its initial admission:
# a small vision or draft forward can report <2 GiB Python RSS before a full-model forward
# spikes Inductor, child C++/ptxas compilers, and unified CUDA context memory together.
_SHARED_GROWTH_WORKER_BYTES = 8 << 30
# A discrete GPU's worker holds only device memory, about 0.7 GiB for Kimi K3's 93-layer forwards.
_DISCRETE_WORKER_DEVICE_BYTES = 1 << 30


@dataclass(frozen=True)
class CompilerResources:
    capacity: int
    available: int
    limit: int
    worker_bytes: int
    headroom: int
    max_capacity: int | None = None
    min_worker_bytes: int | None = None


def compiler_resources(device, *, headroom_bytes=0, options=None, compile_workers=None):
    """Sample a rank's device capacity and its host's compiler memory budget."""
    if type(headroom_bytes) is not int or headroom_bytes < 0:
        raise ValueError("headroom_bytes must be a nonnegative byte count")
    options = _compile_options(options)
    limit = min(cpu_workers(compile_workers), cpu_workers())
    if compile_workers is None:
        limit = min(limit, options['compile_threads'])
    if torch.cuda.is_initialized():
        # Preparation scratch and declaration examples release back to the CUDA caching allocator
        # before forward compilation; return cached segments to the driver before sampling free memory.
        torch.cuda.empty_cache()
    available = host_memory.available_bytes()
    free = device_memory_info(device)[0]
    shared = torch.cuda.get_device_properties(device).is_integrated
    device_bytes = _WORKER_DEVICE_BYTES if shared else _DISCRETE_WORKER_DEVICE_BYTES
    max_capacity = max(0, (free - headroom_bytes) // device_bytes)
    capacity = min(8, max_capacity) if compile_workers is None else max_capacity
    if shared:
        available = min(available, free)
    shared_device_bytes = device_bytes if shared else 0
    return CompilerResources(capacity, available, limit,
        host_memory.FORWARD_WORKER_BYTES + shared_device_bytes,
        max(host_memory._HEADROOM_BYTES, headroom_bytes if shared else 0),
        max_capacity=max_capacity,
        min_worker_bytes=host_memory._WORKER_BYTES + shared_device_bytes)


def _minimum_worker_bytes(resource: CompilerResources) -> int:
    if resource.min_worker_bytes is not None:
        return resource.min_worker_bytes
    return max(0, resource.worker_bytes - host_memory.FORWARD_WORKER_BYTES) + host_memory._WORKER_BYTES


def allocate_workers(resources, counts, hosts):
    """Admit forward compiler workers once per host, including zero-worker owners."""
    workers = [0] * len(resources)
    for host in dict.fromkeys(hosts):
        members = [rank for rank, name in enumerate(hosts) if name == host]
        jobs = sum(counts[rank] for rank in members)
        if not jobs:
            continue
        available = min(resources[rank].available for rank in members)
        headroom = max(resources[rank].headroom for rank in members)
        limit = min(jobs, min(resources[rank].limit for rank in members))
        remaining = available - headroom
        for _ in range(limit):
            able = [rank for rank in members if workers[rank] < resources[rank].capacity
                    and resources[rank].worker_bytes <= remaining]
            if not able:
                break
            rank = min(able, key=lambda rank: (workers[rank], rank))
            workers[rank] += 1
            remaining -= resources[rank].worker_bytes
        if not any(workers[rank] for rank in members):
            # Small forwards can fit below the concurrency estimate. Preserve one worker per host
            # under the minimum allowance without multiplying that allowance by the rank count.
            able = [rank for rank in members if resources[rank].capacity
                    and _minimum_worker_bytes(resources[rank]) <= remaining]
            if able:
                workers[able[0]] = 1
            else:
                raise MemoryError(f"host {host!r} has {jobs} missing forwards but no capacity for a forward compiler "
                                  f"({available >> 20} MiB available host memory, {headroom >> 20} MiB headroom)")
    return workers


def _device_worker_bytes(device):
    """Report the device bytes one forward compiler worker holds and whether device memory is shared."""
    shared = torch.cuda.get_device_properties(device).is_integrated
    return (_WORKER_DEVICE_BYTES if shared else _DISCRETE_WORKER_DEVICE_BYTES), shared


def device_worker_capacity(device, *, headroom_bytes=0):
    """Report how many forward compiler workers a device's free memory can hold."""
    device_bytes, _ = _device_worker_bytes(device)
    free = device_memory_info(device)[0]
    return max(1, (free - headroom_bytes) // device_bytes)


def _peak_rss_bytes() -> int:
    """Report this process's peak resident set size in bytes."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        try:
            query = kernel.K32GetProcessMemoryInfo
        except AttributeError:
            query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    import resource
    import sys

    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_rss = max(0, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    rss = self_rss + child_rss
    return rss if sys.platform == "darwin" else rss * 1024


@_defer_full_collection()
def _compile_forward(payload, device, options, path, num_threads, grad_enabled):
    runtime_patches.apply()
    from .aot import aot_forward
    from .device_binding import bind_device
    from .forward_metadata import loads

    torch.set_num_threads(num_threads)
    torch.set_grad_enabled(grad_enabled)
    torch.cuda.set_device(device)
    # A lent forward's metadata names its owner's GPU; it loads bound to this worker's.
    with bind_device(device):
        forward = loads(payload)
    artifact = aot_forward(forward, options=options)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as file:
        temporary = Path(file.name)
    try:
        artifact.save_compiled_function(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    # The parent admits further workers from this forward's measured host peak.
    return _peak_rss_bytes()


class CompilePool:
    def __init__(self, device, *, headroom_bytes=0, options=None, compile_workers=None, workers=None,
                 max_workers=None, min_workers=None, host_pools=1, host_available=None):
        """Compile forwards in admitted workers, growing as completions report their peak host RSS.

        A rank sized here admits its own worker count from its host and device resources. Ranks
        sharing a host pass counts admitted once per host by `allocate_workers` as `workers`, with
        `host_pools` naming how many active pools share that host's memory budget; `workers` bounds
        growth unless the caller supplies `max_workers`, and `min_workers` sets the floor when a
        heavier forward shrinks concurrency. An explicit `compile_workers` caps growth at that
        process budget rather than the cold 32 GiB per-worker startup estimate.
        """
        if type(host_pools) is not int or host_pools < 1:
            raise ValueError("host_pools must be a positive integer")
        self.device = torch.device(device)
        admitted = workers is not None
        options = _compile_options(options)
        resources = None
        if workers is None:
            resources = compiler_resources(self.device, headroom_bytes=headroom_bytes, options=options,
                                           compile_workers=compile_workers)
            workers = allocate_workers([resources], [resources.limit], ["local"])[0]
        if type(workers) is not int or workers < 1:
            raise ValueError("forward compiler pool requires a positive admitted worker count")
        if min_workers is not None and (type(min_workers) is not int or min_workers < 1 or min_workers > workers):
            raise ValueError("forward compiler pool minimum must be between 1 and its admitted worker count")
        self.workers = workers
        self._initial_workers = workers if min_workers is None else min_workers
        self.compile_workers = compile_workers
        self.headroom_bytes = headroom_bytes
        self.host_pools = host_pools
        self._host_available = (resources.available if host_available is None and resources is not None
                                else host_available)
        self.peak_worker_rss = 0
        self._wave_remaining = 0
        if max_workers is None:
            if admitted:
                max_workers = workers
            else:
                device_cap = resources.capacity if resources.max_capacity is None else resources.max_capacity
                cpus = (cpu_workers(compile_workers) if host_pools == 1
                        else cpu_workers(compile_workers, pools=host_pools))
                max_workers = max(workers, min(cpus, device_cap))
        if type(max_workers) is not int or max_workers < workers:
            raise ValueError("forward compiler pool maximum must cover its admitted worker count")
        self.max_workers = max_workers
        self._pools = []
        self._shard_capacity = []
        self._shard_pending = []
        self._pool_lock = threading.Lock()
        # Nested Inductor pools would escape the host's admitted process budget.
        self.options = dict(options, compile_threads=1)

    @property
    def pool(self):
        """Report the first compiler pool, or `None` before the first submission."""
        return self._pools[0] if self._pools else None

    def submit(self, forward, path):
        from .forward_metadata import dumps

        return self.submit_payload(dumps(forward), path)

    def submit_payload(self, payload, path):
        """Compile a forward serialized by `forward_metadata.dumps`, possibly on another rank, into `path`."""
        pool, index = self._select_shard()
        try:
            future = pool.submit(_compile_forward, payload, self.device, self.options, str(path),
                                 torch.get_num_threads(), torch.is_grad_enabled())
        except BaseException:
            self._release_slot(index)
            raise
        if future is not None and hasattr(future, "add_done_callback"):
            future.add_done_callback(lambda _: self._release_slot(index))
        else:
            self._release_slot(index)
        return future

    def _select_shard(self):
        """Reserve a worker slot in an existing shard, or create an additional shard of the pool."""
        with self._pool_lock:
            for index, capacity in enumerate(self._shard_capacity):
                if self._shard_pending[index] < capacity:
                    self._shard_pending[index] += 1
                    return self._pools[index], index
            capacity = self.workers - sum(self._shard_capacity)
            if capacity < 1:
                # Every admitted worker is busy; queue the forward on the least loaded shard.
                index = min(range(len(self._pools)), key=self._shard_pending.__getitem__)
                self._shard_pending[index] += 1
                return self._pools[index], index
            from torch._inductor.compile_worker.subproc_pool import SubprocKind, SubprocPool

            if not self._pools:
                runtime_patches.apply()
            if os.name == "nt":
                pool = ProcessPoolExecutor(capacity, mp_context=multiprocessing.get_context("spawn"))
            else:
                pool = SubprocPool(capacity, kind=SubprocKind.SPAWN)
            self._pools.append(pool)
            self._shard_capacity.append(capacity)
            self._shard_pending.append(1)
            return pool, len(self._pools) - 1

    def _release_slot(self, index):
        with self._pool_lock:
            self._shard_pending[index] -= 1

    def reconfigure(self, *, workers: int, max_workers: int, host_pools: int,
                    min_workers: int | None = None) -> None:
        """Update admitted worker bounds and active host-pool sharing in place.

        Existing worker shards remain open; subsequent submissions spawn additional shards only
        when `workers` exceeds the capacity already started. `min_workers` sets the floor retained
        when a subsequent forward reports a peak RSS above the reconfigured budget.
        """
        if type(host_pools) is not int or host_pools < 1:
            raise ValueError("host_pools must be a positive integer")
        if type(workers) is not int or workers < 1:
            raise ValueError("forward compiler pool requires a positive admitted worker count")
        if type(max_workers) is not int or max_workers < workers:
            raise ValueError("forward compiler pool maximum must cover its admitted worker count")
        if min_workers is not None and (type(min_workers) is not int or min_workers < 1 or min_workers > workers):
            raise ValueError("forward compiler pool minimum must be between 1 and its admitted worker count")
        with self._pool_lock:
            self.host_pools = host_pools
            self.max_workers = max_workers
            self._initial_workers = workers if min_workers is None else min_workers
            self.workers = workers
            self._wave_remaining = max(0, workers - sum(self._shard_capacity))

    def observe_worker_rss(self, rss_bytes: int | None, *, remaining: int | None = None) -> int:
        """Adjust admitted worker count from the peak host RSS a completed worker reported.

        Returns the worker count after observation. The largest observed peak sizes subsequent
        worker admissions within this pool's share of the host memory budget, and each grown wave
        completes its added slots before doubling again. When `remaining` names the forwards still
        queued or compiling in this pool, growth stops once active capacity covers them.
        """
        if rss_bytes is None or rss_bytes <= 0:
            return self.workers
        with self._pool_lock:
            self.peak_worker_rss = max(self.peak_worker_rss, int(rss_bytes))
            if self.max_workers <= self._initial_workers and self.workers <= self._initial_workers:
                return self.workers
            device_bytes, shared = _device_worker_bytes(self.device)
            # Keep the minimum per-worker floor (`host_memory._WORKER_BYTES`) and add a 25% margin to the measured peak.
            unit_bytes = max(host_memory._WORKER_BYTES, (self.peak_worker_rss * 5 + 3) // 4)
            if shared:
                unit_bytes = max(_SHARED_GROWTH_WORKER_BYTES, unit_bytes + device_bytes)
            live_available = host_memory.available_bytes()
            free = device_memory_info(self.device)[0]
            if shared:
                live_available = min(live_available, free)
            if self._host_available is None:
                self._host_available = live_available
            headroom = max(host_memory._HEADROOM_BYTES, self.headroom_bytes if shared else 0)
            budget_cap = max(self._initial_workers,
                             (self._host_available - headroom) // (self.host_pools * unit_bytes))
            spawned = sum(self._shard_capacity)
            # Started worker shards already hold their CUDA contexts in `free`; count free device
            # memory toward additional workers beyond `spawned` while `self.max_workers` enforces
            # the pre-spawn device ceiling.
            device_headroom = headroom if shared else self.headroom_bytes
            live_device_additional = max(
                0, (free - device_headroom) // (self.host_pools * device_bytes if shared else device_bytes)
            )
            device_cap = max(self._initial_workers, spawned + live_device_additional)
            ceiling = min(self.max_workers, budget_cap, device_cap)
            if ceiling < self.workers:
                self.workers = ceiling
                self._wave_remaining = 0
                return self.workers
            if self._wave_remaining > 0:
                self._wave_remaining -= 1
            if self._wave_remaining > 0 or self.workers >= ceiling:
                return self.workers
            live_additional = max(0, (live_available - headroom) // (self.host_pools * unit_bytes))
            target = min(ceiling, self.workers + live_additional, self.workers * 2)
            if remaining is not None:
                target = min(target, max(self.workers, remaining))
            if target > self.workers:
                self._wave_remaining = target - self.workers
                self.workers = target
            return self.workers

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        failure = None
        for pool in self._pools:
            try:
                pool.shutdown()
            except Exception as error:
                failure = failure or error
        if failure is not None and kind is None:
            raise failure
