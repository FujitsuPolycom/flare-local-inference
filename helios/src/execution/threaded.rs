/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Coordinated CUDA ranks sharing one process address space.

use super::{ExecutionManifest, FrameCompletion, GraphExecutor, Upload};
use crate::{Error, ExecutionError, Result, Shutdown, cache::Fence};
use std::{
    cell::UnsafeCell,
    collections::VecDeque,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const TIMEOUT: Duration = Duration::from_secs(120);
const HANDOFF_POLL: Duration = Duration::from_micros(500);
const STACK_UPLOAD_CAPACITY: usize = 64;
const MAX_RANKS: usize = 16;

pub(super) mod protocol;
#[cfg(test)]
mod protocol_tests;

// Startup and drain replies are brief. Poll that interval before parking a rank,
// while retaining blocking waits for idle or slow tuning graphs.
fn receive<T>(
    receiver: &mpsc::Receiver<T>,
    timeout: Option<Duration>,
) -> std::result::Result<T, mpsc::RecvTimeoutError> {
    let started = Instant::now();
    let poll = timeout.map_or(HANDOFF_POLL, |limit| limit.min(HANDOFF_POLL));
    loop {
        match receiver.try_recv() {
            Ok(value) => return Ok(value),
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(mpsc::RecvTimeoutError::Disconnected);
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
        if started.elapsed() >= poll {
            break;
        }
        std::hint::spin_loop();
    }
    match timeout {
        Some(limit) => receiver.recv_timeout(limit.saturating_sub(started.elapsed())),
        None => receiver
            .recv()
            .map_err(|_| mpsc::RecvTimeoutError::Disconnected),
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::execution(ExecutionError::Worker, message)
}

fn spin_wait_until<T>(timeout: Duration, mut poll: impl FnMut() -> Option<T>) -> Option<T> {
    // Fast spin before reading the monotonic clock so back-to-back handoffs incur zero vDSO reads.
    for _ in 0..32 {
        if let Some(value) = poll() {
            return Some(value);
        }
        std::hint::spin_loop();
    }
    let started = Instant::now();
    loop {
        if let Some(value) = poll() {
            return Some(value);
        }
        let elapsed = started.elapsed();
        if elapsed >= timeout {
            return None;
        }
        if elapsed < HANDOFF_POLL {
            std::hint::spin_loop();
        } else {
            thread::yield_now();
        }
    }
}

#[repr(align(64))]
struct CacheAligned<T>(T);

#[derive(Clone, Copy)]
struct SlotUpload {
    name_start: usize,
    name_end: usize,
    offset: usize,
    data_start: usize,
    data_end: usize,
}

/// Reusable ping-pong upload slot shared between rank zero and peer rank threads.
struct StepSlot {
    fence: Fence,
    graph: usize,
    grammar: bool,
    names: String,
    payload: Vec<u8>,
    uploads: Vec<SlotUpload>,
}

impl StepSlot {
    fn with_capacity(payload_bytes: usize, upload_count: usize) -> Self {
        Self {
            fence: Fence(0),
            graph: 0,
            grammar: false,
            names: String::with_capacity(upload_count.saturating_mul(24)),
            payload: Vec::with_capacity(payload_bytes),
            uploads: Vec::with_capacity(upload_count),
        }
    }

    fn write(&mut self, fence: Fence, graph: usize, grammar: bool, uploads: &[Upload<'_>]) {
        self.fence = fence;
        self.graph = graph;
        self.grammar = grammar;
        self.names.clear();
        self.payload.clear();
        self.uploads.clear();
        for upload in uploads {
            let name_start = self.names.len();
            self.names.push_str(upload.buffer);
            let name_end = self.names.len();
            let data_start = self.payload.len();
            self.payload.extend_from_slice(upload.bytes);
            let data_end = self.payload.len();
            self.uploads.push(SlotUpload {
                name_start,
                name_end,
                offset: upload.offset,
                data_start,
                data_end,
            });
        }
    }

    fn upload(&self, src: &SlotUpload) -> Upload<'_> {
        Upload {
            buffer: &self.names[src.name_start..src.name_end],
            offset: src.offset,
            bytes: &self.payload[src.data_start..src.data_end],
        }
    }

    fn with_uploads<R>(&self, f: impl FnOnce(&[Upload<'_>]) -> R) -> R {
        let len = self.uploads.len();
        if len <= STACK_UPLOAD_CAPACITY {
            let mut stack: [std::mem::MaybeUninit<Upload<'_>>; STACK_UPLOAD_CAPACITY] =
                [const { std::mem::MaybeUninit::uninit() }; STACK_UPLOAD_CAPACITY];
            for (dst, src) in stack.iter_mut().zip(&self.uploads) {
                dst.write(self.upload(src));
            }
            // SAFETY: The first `len` elements were initialized in the loop above.
            let slice =
                unsafe { std::slice::from_raw_parts(stack.as_ptr().cast::<Upload<'_>>(), len) };
            f(slice)
        } else {
            let uploads: Vec<Upload<'_>> = self.uploads.iter().map(|src| self.upload(src)).collect();
            f(&uploads)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DispatchEvent {
    Step,
    Drain,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecisionEvent {
    Commit,
    Cancel,
    Drain,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WaitPeersError {
    PeerExited,
    Timeout,
}

/// Single-producer, multi-consumer lock-free ping-pong dispatch ring for local ranks.
///
/// Each coordinated step runs an inline Prepare -> Ready -> Commit/Cancel barrier over
/// cache-aligned atomics without per-step heap allocations or control-channel traffic:
/// 1. Rank zero writes `slots[epoch & 1]` and publishes `prepare_epoch = epoch`.
/// 2. Every peer thread validates its local upload layout, frame capacity, and rank-local
///    Engram staging state without launching CUDA work, drops its `&StepSlot` borrow, and
///    stores `epoch << 1` in its `peer_ack` slot.
/// 3. Only after every peer acknowledges readiness and rank zero re-validates locally does rank
///    zero publish `decision = (epoch << 1) | 1` (`Commit`) BEFORE calling `local.submit()`, so
///    every rank launches multi-GPU collectives concurrently without rank zero blocking inside
///    CUDA graph launch while peers wait for `Commit`. If any peer fails validation or exits, rank
///    zero requests drain without launching locally, preventing stranded collectives. If rank
///    zero encounters `Ok(None)` backpressure during pre-commit validation, it publishes
///    `decision = epoch << 1` (`Cancel`), waits for peers to acknowledge `(epoch << 1) | 1`, and
///    returns `Ok(None)` without poisoning.
/// 4. Parity slot reuse (`slots[epoch & 1]` for `epoch >= 3`) is synchronized by
///    `wait_peer_acks(peer_count, (epoch - 1) << 1)`: every peer drops its `&StepSlot` borrow
///    for `epoch - 2` inside `executor.submit()` (or after observing `Cancel(epoch - 2)`) before
///    advancing to `epoch - 1` and releasing `peer_ack >= (epoch - 1) << 1`. This holds even if
///    `epoch - 1` is cancelled while `epoch - 2`'s GPU frame is still in flight.
struct DispatchRing {
    slots: [CacheAligned<UnsafeCell<StepSlot>>; 2],
    prepare_epoch: CacheAligned<AtomicU64>,
    peer_ack: [CacheAligned<AtomicU64>; MAX_RANKS],
    decision: CacheAligned<AtomicU64>,
    drain: CacheAligned<AtomicBool>,
    peer_exited: CacheAligned<AtomicBool>,
    parked: CacheAligned<AtomicUsize>,
    idle_lock: Mutex<()>,
    idle_wake: Condvar,
}

// SAFETY: Rank zero is the sole writer to `slots[epoch & 1]`. For `epoch >= 3`, rank zero writes
// `slots[epoch & 1]` only after `wait_peer_acks(peer_count, (epoch - 1) << 1)` acquires each
// peer's release store `mark_peer_ready(peer, epoch - 1)`, which happens strictly after the peer
// drops its `&StepSlot` borrow for `epoch - 2` inside `executor.submit()` (or cancel handling).
unsafe impl Sync for DispatchRing {}

impl DispatchRing {
    fn new(payload_bytes: usize, upload_count: usize) -> Self {
        Self {
            slots: [
                CacheAligned(UnsafeCell::new(StepSlot::with_capacity(
                    payload_bytes,
                    upload_count,
                ))),
                CacheAligned(UnsafeCell::new(StepSlot::with_capacity(
                    payload_bytes,
                    upload_count,
                ))),
            ],
            prepare_epoch: CacheAligned(AtomicU64::new(0)),
            peer_ack: std::array::from_fn(|_| CacheAligned(AtomicU64::new(0))),
            decision: CacheAligned(AtomicU64::new(0)),
            drain: CacheAligned(AtomicBool::new(false)),
            peer_exited: CacheAligned(AtomicBool::new(false)),
            parked: CacheAligned(AtomicUsize::new(0)),
            idle_lock: Mutex::new(()),
            idle_wake: Condvar::new(),
        }
    }

    /// Stage one step into its parity slot and publish `prepare_epoch` for peer validation.
    ///
    /// # Safety
    /// Caller must be the sole producer and guarantee no peer still holds `&StepSlot` for
    /// `slots[epoch & 1]` (satisfied for `epoch <= 2` or after `wait_peer_acks((epoch - 1) << 1)`).
    unsafe fn prepare(
        &self,
        epoch: u64,
        fence: Fence,
        graph: usize,
        grammar: bool,
        uploads: &[Upload<'_>],
    ) {
        let slot = (epoch as usize) & 1;
        // SAFETY: Exclusive write access to this parity slot is guaranteed by the caller.
        unsafe { (*self.slots[slot].0.get()).write(fence, graph, grammar, uploads) };
        self.prepare_epoch.0.store(epoch, Ordering::SeqCst);
        if self.parked.0.load(Ordering::SeqCst) != 0 {
            let _guard = self.idle_lock.lock().unwrap_or_else(|e| e.into_inner());
            self.idle_wake.notify_all();
        }
    }

    fn mark_peer_ready(&self, peer_index: usize, epoch: u64) {
        self.peer_ack[peer_index]
            .0
            .store(epoch << 1, Ordering::Release);
    }

    fn mark_peer_cancelled(&self, peer_index: usize, epoch: u64) {
        self.peer_ack[peer_index]
            .0
            .store((epoch << 1) | 1, Ordering::Release);
    }

    fn commit(&self, epoch: u64) {
        self.decision.0.store((epoch << 1) | 1, Ordering::SeqCst);
    }

    fn cancel(&self, epoch: u64) {
        self.decision.0.store(epoch << 1, Ordering::SeqCst);
    }

    fn poll_peer_acks(&self, peer_count: usize, target: u64) -> Option<std::result::Result<(), WaitPeersError>> {
        if (0..peer_count).all(|index| self.peer_ack[index].0.load(Ordering::Acquire) >= target) {
            return Some(Ok(()));
        }
        if self.peer_exited() {
            return Some(Err(WaitPeersError::PeerExited));
        }
        None
    }

    fn wait_peer_acks(
        &self,
        peer_count: usize,
        target: u64,
        timeout: Duration,
    ) -> std::result::Result<(), WaitPeersError> {
        spin_wait_until(timeout, || self.poll_peer_acks(peer_count, target))
            .unwrap_or(Err(WaitPeersError::Timeout))
    }

    fn poll_decision(&self, epoch: u64) -> Option<DecisionEvent> {
        // Check `decision` before `drain`: under 2-deep pipelining a later step can fail
        // and request drain after `epoch` was already committed and launched on other ranks.
        // Every surviving peer must still observe `Commit(epoch)` so no collective is stranded.
        let value = self.decision.0.load(Ordering::Acquire);
        if (value >> 1) >= epoch {
            return Some(if (value & 1) != 0 {
                DecisionEvent::Commit
            } else {
                DecisionEvent::Cancel
            });
        }
        if self.drain.0.load(Ordering::SeqCst) {
            // Re-check `decision` after observing `drain`: `commit`/`cancel` (SeqCst)
            // precedes any subsequent `request_drain` (SeqCst) in the total order.
            let value = self.decision.0.load(Ordering::SeqCst);
            if (value >> 1) >= epoch {
                return Some(if (value & 1) != 0 {
                    DecisionEvent::Commit
                } else {
                    DecisionEvent::Cancel
                });
            }
            return Some(DecisionEvent::Drain);
        }
        None
    }

    fn wait_decision(&self, epoch: u64, timeout: Duration) -> Result<DecisionEvent> {
        spin_wait_until(timeout, || self.poll_decision(epoch))
            .ok_or_else(|| invalid("rank step decision deadline expired"))
    }

    fn request_drain(&self) {
        self.drain.0.store(true, Ordering::SeqCst);
        let _guard = self.idle_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.idle_wake.notify_all();
    }

    fn mark_peer_exited(&self) {
        self.peer_exited.0.store(true, Ordering::SeqCst);
    }

    fn peer_exited(&self) -> bool {
        self.peer_exited.0.load(Ordering::Acquire)
    }

    fn has_step(&self, next_epoch: u64) -> bool {
        self.prepare_epoch.0.load(Ordering::Acquire) >= next_epoch
    }

    fn poll_next(&self, next_epoch: u64) -> Option<DispatchEvent> {
        // Safe to check `drain` first here: rank zero cannot commit `next_epoch` until
        // every peer has already observed `prepare_epoch >= next_epoch` and signalled ready.
        if self.drain.0.load(Ordering::Acquire) {
            return Some(DispatchEvent::Drain);
        }
        if self.prepare_epoch.0.load(Ordering::Acquire) >= next_epoch {
            return Some(DispatchEvent::Step);
        }
        None
    }

    fn wait_next(&self, next_epoch: u64) -> DispatchEvent {
        if let Some(event) = spin_wait_until(HANDOFF_POLL, || self.poll_next(next_epoch)) {
            return event;
        }
        let mut guard = self.idle_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.parked.0.fetch_add(1, Ordering::SeqCst);
        loop {
            if self.drain.0.load(Ordering::SeqCst) {
                self.parked.0.fetch_sub(1, Ordering::Release);
                return DispatchEvent::Drain;
            }
            if self.prepare_epoch.0.load(Ordering::SeqCst) >= next_epoch {
                self.parked.0.fetch_sub(1, Ordering::Release);
                return DispatchEvent::Step;
            }
            guard = self
                .idle_wake
                .wait(guard)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Borrow the immutable slot for `step_epoch` after observing `prepare_epoch >= step_epoch`.
    ///
    /// # Safety
    /// Caller must have observed `prepare_epoch >= step_epoch` and must drop the returned
    /// reference before calling `mark_peer_ready(peer, step_epoch + 1)` (i.e. before
    /// advancing past `step_epoch`).
    unsafe fn slot(&self, step_epoch: u64) -> &StepSlot {
        let index = (step_epoch as usize) & 1;
        // SAFETY: Acquire/SeqCst load of `prepare_epoch >= step_epoch` synchronizes with `prepare`.
        unsafe { &*self.slots[index].0.get() }
    }
}

/// Coordinate peer pre-launch validation, commit publication, and rank-zero submission over `ring`.
///
/// # Safety
/// Caller must be the sole producer and guarantee no peer still reads `ring.slots[*next_epoch & 1]`.
#[allow(clippy::too_many_arguments)]
unsafe fn coordinate_step<L, V, S>(
    ring: &DispatchRing,
    peer_count: usize,
    next_epoch: &mut u64,
    fence: Fence,
    graph: usize,
    grammar: bool,
    uploads: &[Upload<'_>],
    timeout: Duration,
    on_peer_exit: impl FnOnce() -> Error,
    local: &mut L,
    local_validate: V,
    local_submit: S,
) -> Result<Option<Fence>>
where
    V: FnOnce(&mut L) -> Result<Option<Fence>>,
    S: FnOnce(&mut L) -> Result<Option<Fence>>,
{
    let epoch = *next_epoch;
    // SAFETY: Caller guarantees exclusive write access to `slots[epoch & 1]`.
    unsafe {
        ring.prepare(epoch, fence, graph, grammar, uploads);
    }
    match ring.wait_peer_acks(peer_count, epoch << 1, timeout) {
        Ok(()) => {}
        Err(WaitPeersError::PeerExited) => {
            ring.request_drain();
            return Err(on_peer_exit());
        }
        Err(WaitPeersError::Timeout) => {
            ring.request_drain();
            return Err(invalid("rank thread readiness deadline expired"));
        }
    }
    match local_validate(local) {
        Ok(Some(local_fence)) if local_fence == fence => {
            // Publish Commit before local CUDA launch so peer rank threads launch
            // multi-GPU collectives concurrently with rank zero; calling local_submit()
            // first deadlocks if cuGraphLaunch blocks on a collective while peers wait.
            ring.commit(epoch);
            *next_epoch += 1;
            match local_submit(local) {
                Ok(Some(submitted)) if submitted == fence => Ok(Some(fence)),
                Ok(_) => {
                    ring.request_drain();
                    Err(invalid("rank-zero fence changed after validation"))
                }
                Err(error) => {
                    ring.request_drain();
                    Err(error)
                }
            }
        }
        Ok(None) => {
            ring.cancel(epoch);
            match ring.wait_peer_acks(peer_count, (epoch << 1) | 1, timeout) {
                Ok(()) => {
                    *next_epoch += 1;
                    Ok(None)
                }
                Err(WaitPeersError::PeerExited) => {
                    ring.request_drain();
                    Err(on_peer_exit())
                }
                Err(WaitPeersError::Timeout) => {
                    ring.request_drain();
                    Err(invalid("rank cancel acknowledgement deadline expired"))
                }
            }
        }
        Ok(Some(_)) => {
            ring.request_drain();
            Err(invalid("rank-zero fence changed after validation"))
        }
        Err(error) => {
            ring.request_drain();
            Err(error)
        }
    }
}

/// Drain every already-committed in-flight frame on a rank thread before closing its executor.
fn drain_outstanding_frames<S, T>(
    state: &mut S,
    mut outstanding: impl FnMut(&S) -> usize,
    mut poll: impl FnMut(&mut S) -> Result<Option<T>>,
    started: &mut VecDeque<Instant>,
    timeout: Duration,
    mut on_complete: impl FnMut(T),
) -> Result<()> {
    while outstanding(state) != 0 {
        if let Some(completion) = poll(state)? {
            started.pop_front();
            on_complete(completion);
        } else if started.front().is_some_and(|time| time.elapsed() > timeout) {
            return Err(invalid("rank graph completion deadline expired during drain"));
        } else {
            std::hint::spin_loop();
            thread::yield_now();
        }
    }
    Ok(())
}

struct PeerExitGuard(Arc<DispatchRing>);
impl Drop for PeerExitGuard {
    fn drop(&mut self) {
        self.0.mark_peer_exited();
    }
}

enum Command {
    Tune(usize),
    FinishStartup,
    Drain,
}
enum Reply {
    Initialized,
    Tuned(usize, Result<()>),
    Started,
    Complete(FrameCompletion),
    Drained,
}
struct Peer {
    sender: mpsc::Sender<Command>,
    receiver: mpsc::Receiver<Result<Reply>>,
    thread: Option<JoinHandle<Shutdown>>,
}
impl Peer {
    fn send(&self, command: Command) -> Result<()> {
        self.sender
            .send(command)
            .map_err(|_| invalid("rank thread disconnected"))
    }
    fn receive(&self) -> Result<Reply> {
        receive(&self.receiver, Some(TIMEOUT))
            .map_err(|_| invalid("rank thread response deadline expired"))?
    }
}

struct InFlightFrame {
    fence: Fence,
    graph: usize,
    grammar: bool,
    dispatched: Instant,
    queued: bool,
    world_size: usize,
    completions: [Option<FrameCompletion>; MAX_RANKS],
}

impl protocol::PendingFrame for InFlightFrame {
    fn fence(&self) -> Fence {
        self.fence
    }
    fn graph(&self) -> usize {
        self.graph
    }
    fn grammar(&self) -> bool {
        self.grammar
    }
    fn committed(&self) -> bool {
        true
    }
    fn ready(&self) -> &[bool] {
        &[]
    }
    fn ready_mut(&mut self) -> &mut [bool] {
        &mut []
    }
    fn completions(&self) -> &[Option<FrameCompletion>] {
        &self.completions[..self.world_size]
    }
    fn completions_mut(&mut self) -> &mut [Option<FrameCompletion>] {
        &mut self.completions[..self.world_size]
    }
}

pub(crate) struct ThreadedExecutor {
    local: GraphExecutor,
    peers: Vec<Peer>,
    ring: Arc<DispatchRing>,
    next_epoch: u64,
    pending: VecDeque<InFlightFrame>,
    poison: Option<Error>,
    shutdown: Option<Shutdown>,
}
impl ThreadedExecutor {
    pub unsafe fn new(manifest: ExecutionManifest, ranks: Vec<ExecutionManifest>) -> Result<Self> {
        if ranks.len() >= MAX_RANKS {
            return Err(invalid("tensor parallelism supports at most sixteen ranks"));
        }
        if ranks.is_empty() {
            return Err(invalid("tensor parallelism requires at least two ranks"));
        }
        if !manifest.persistent_workers.is_empty()
            || ranks.iter().any(|rank| !rank.persistent_workers.is_empty())
        {
            return Err(invalid(
                "persistent workers are only supported on single-rank cores",
            ));
        }
        for rank in &ranks {
            rank.validate()?;
            if !same_layout(&manifest, rank) {
                return Err(invalid("rank graph I/O schemas differ"));
            }
        }
        let max_uploads = manifest
            .graphs
            .iter()
            .map(|graph| graph.input_bindings.len())
            .max()
            .unwrap_or(0)
            .max(16);
        let ring = Arc::new(DispatchRing::new(manifest.input_frame_bytes, max_uploads));
        let local = unsafe { GraphExecutor::new_unstarted(manifest)? };
        let mut result = Self {
            local,
            peers: Vec::new(),
            ring,
            next_epoch: 1,
            pending: VecDeque::with_capacity(2),
            poison: None,
            shutdown: None,
        };
        for (index, manifest) in ranks.into_iter().enumerate() {
            let (sender, commands) = mpsc::channel();
            let (replies, receiver) = mpsc::channel();
            let peer_ring = result.ring.clone();
            let thread = thread::Builder::new()
                .name(format!("flare-rank-{}", index + 1))
                .spawn(move || {
                    let _exit_guard = PeerExitGuard(peer_ring.clone());
                    let mut executor = match unsafe { GraphExecutor::new_unstarted(manifest) } {
                        Ok(executor) => executor,
                        Err(error) => {
                            let _ = replies.send(Err(error.clone()));
                            return Shutdown {
                                drained: true,
                                error: Some(error),
                            };
                        }
                    };
                    executor.copy_logprobs = false;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        rank_loop(index, &mut executor, commands, &replies, &peer_ring)
                    }))
                    .unwrap_or_else(|_| Err(invalid("rank execution thread panicked")));
                    match result {
                        Ok(()) => executor.close(),
                        Err(error) => {
                            let _ = replies.send(Err(error.clone()));
                            executor.abandon(error)
                        }
                    }
                })
                .map_err(|e| invalid(format!("cannot start rank thread: {e}")))?;
            let mut peer = Peer {
                sender,
                receiver,
                thread: Some(thread),
            };
            let ready = match peer.receiver.recv_timeout(TIMEOUT) {
                Ok(reply) => reply,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    Err(invalid("rank initialization thread exited"))
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // A thread inside CUDA may still use resources supplied by the caller.
                    eprintln!(
                        "rank initialization deadline expired; terminating the CUDA process"
                    );
                    std::process::abort();
                }
            };
            if !matches!(ready, Ok(Reply::Initialized)) {
                drop(peer.sender);
                match peer.thread.take().unwrap().join() {
                    Ok(shutdown) if shutdown.drained => {}
                    _ => {
                        eprintln!(
                            "rank initialization thread did not confirm drain; terminating the CUDA process"
                        );
                        std::process::abort();
                    }
                }
                return Err(ready
                    .err()
                    .unwrap_or_else(|| invalid("rank startup did not report readiness")));
            }
            result.peers.push(peer);
        }
        let startup = (|| -> Result<()> {
            for index in result.local.manifest().referenced_tuning_graphs() {
                for (sent, peer) in result.peers.iter().enumerate() {
                    if let Err(error) = peer.send(Command::Tune(index)) {
                        if sent != 0 {
                            eprintln!(
                                "startup rank dispatch failed with unconfirmed CUDA work: {error}; terminating the CUDA process"
                            );
                            std::process::abort();
                        }
                        return Err(error);
                    }
                }
                let mut failure = result.local.run_tuning_graph(index).err();
                for peer in &result.peers {
                    match peer.receive() {
                        Ok(Reply::Tuned(value, status)) if value == index => {
                            if let Err(error) = status {
                                failure.get_or_insert(error);
                            }
                        }
                        reply => {
                            let message = reply.err().map_or_else(
                                || "rank tuning completion identity differs".to_string(),
                                |error| error.to_string(),
                            );
                            eprintln!(
                                "startup rank completion is unconfirmed: {message}; terminating the CUDA process"
                            );
                            std::process::abort();
                        }
                    }
                }
                if let Some(error) = failure {
                    return Err(error);
                }
            }
            for peer in &result.peers {
                peer.send(Command::FinishStartup)?;
            }
            result.local.finish_startup()?;
            for peer in &result.peers {
                if !matches!(peer.receive()?, Reply::Started) {
                    return Err(invalid("rank startup did not complete"));
                }
            }
            Ok(())
        })();
        if let Err(error) = startup {
            // Every issued tuning command has a completed acknowledgement here.
            result.ring.request_drain();
            let sent: Vec<_> = result
                .peers
                .iter()
                .map(|peer| peer.send(Command::Drain).is_ok())
                .collect();
            for (peer, sent) in result.peers.iter_mut().zip(sent) {
                if sent {
                    loop {
                        match peer.receive() {
                            Ok(Reply::Drained) => break,
                            Ok(Reply::Started) => continue,
                            reply => {
                                eprintln!(
                                    "startup rank cleanup was not confirmed: {:?}; terminating the CUDA process",
                                    reply.err()
                                );
                                std::process::abort();
                            }
                        }
                    }
                }
                if let Some(thread) = peer.thread.take() {
                    match thread.join() {
                        Ok(shutdown) if shutdown.drained => {}
                        _ => {
                            eprintln!(
                                "startup rank thread did not confirm drain; terminating the CUDA process"
                            );
                            std::process::abort();
                        }
                    }
                }
            }
            let shutdown = result.local.close();
            if !shutdown.drained {
                eprintln!(
                    "startup coordinator cleanup did not drain; terminating the CUDA process"
                );
                std::process::abort();
            }
            result.shutdown = Some(shutdown);
            return Err(error);
        }
        Ok(result)
    }
    #[cfg(feature = "server")]
    pub fn manifest(&self) -> &ExecutionManifest {
        self.local.manifest()
    }
    pub fn outstanding(&self) -> usize {
        self.pending.len()
    }
    pub fn can_submit(&self, grammar: bool) -> bool {
        self.poison.is_none()
            && self.shutdown.is_none()
            && protocol::can_submit_pending(&self.pending, grammar, self.local.manifest())
            && self.local.can_submit(grammar)
    }
    pub fn next_fence(&self) -> Result<Fence> {
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        if self.shutdown.is_some() {
            return Err(invalid("rank executor is closed"));
        }
        self.local.next_fence()
    }
    fn surface_peer_error(peers: &[Peer]) -> Error {
        for peer in peers {
            while let Ok(reply) = peer.receiver.try_recv() {
                if let Err(error) = reply {
                    return error;
                }
            }
        }
        invalid("rank thread disconnected")
    }
    pub fn submit(
        &mut self,
        graph: usize,
        uploads: &[Upload<'_>],
        grammar: bool,
    ) -> Result<Option<Fence>> {
        self.next_fence()?;
        if self.ring.peer_exited() {
            let error = Self::surface_peer_error(&self.peers);
            self.poison = Some(error.clone());
            return Err(error);
        }
        let barrier = grammar
            || !matches!(
                self.local.manifest().graphs.get(graph).map(|g| g.kind),
                Some(super::GraphKind::Decode | super::GraphKind::Speculative)
            );
        if !self.can_submit(barrier) {
            return Ok(None);
        }
        let Some(fence) = self.local.validate_submission(graph, uploads, grammar)? else {
            return Ok(None);
        };
        let queued = !self.pending.is_empty();
        let dispatched = Instant::now();
        let peers = &self.peers;
        let local = &mut self.local;
        // SAFETY: Rank zero is the sole producer. For `next_epoch <= 2`, parity slot
        // `next_epoch & 1` has never been read by any peer. For `next_epoch >= 3`, advancing
        // past `next_epoch - 1` required `wait_peer_acks(peers.len(), (next_epoch - 1) << 1)`,
        // acquiring every peer's release store made strictly after it dropped its `&StepSlot`
        // borrow for `next_epoch - 2` inside `executor.submit()` (or cancel handling).
        let coordinated = unsafe {
            coordinate_step(
                &self.ring,
                peers.len(),
                &mut self.next_epoch,
                fence,
                graph,
                grammar,
                uploads,
                TIMEOUT,
                || Self::surface_peer_error(peers),
                local,
                |exec| exec.validate_submission(graph, uploads, grammar),
                |exec| exec.submit(graph, uploads, grammar),
            )
        };
        match coordinated {
            Ok(Some(submitted_fence)) => {
                self.pending.push_back(InFlightFrame {
                    fence: submitted_fence,
                    graph,
                    grammar,
                    dispatched,
                    queued,
                    world_size: self.peers.len() + 1,
                    completions: std::array::from_fn(|_| None),
                });
                Ok(Some(submitted_fence))
            }
            Ok(None) => Ok(None),
            Err(error) => {
                self.poison = Some(error.clone());
                Err(error)
            }
        }
    }
    pub fn poll(&mut self) -> Result<Option<FrameCompletion>> {
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        match self.poll_ranks() {
            Ok(None) => Ok(None),
            Ok(Some(completion)) => Ok(Some(completion)),
            Err(error) => {
                self.poison = Some(error.clone());
                Err(error)
            }
        }
    }
    fn poll_ranks(&mut self) -> Result<Option<FrameCompletion>> {
        if self
            .pending
            .iter()
            .any(|frame| frame.dispatched.elapsed() > TIMEOUT)
        {
            return Err(invalid("tensor-parallel graph completion deadline expired"));
        }
        if let Some(completion) = self.local.poll()? {
            protocol::record_local_completion(&mut self.pending, completion)?;
        }
        for (index, peer) in self.peers.iter_mut().enumerate() {
            loop {
                let reply = match peer.receiver.try_recv() {
                    Ok(reply) => reply?,
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(_) => return Err(invalid("rank thread disconnected")),
                };
                match reply {
                    Reply::Complete(completion) => {
                        protocol::record_peer_completion(&mut self.pending, index, completion)?;
                    }
                    _ => return Err(invalid("unexpected rank thread reply")),
                }
            }
        }
        let Some(mut frame) = protocol::pop_completed_front(&mut self.pending) else {
            return Ok(None);
        };
        let mut root = frame.completions[0].take().unwrap();
        for index in 0..self.peers.len() {
            let completion = frame.completions[index + 1].take().unwrap();
            let agreement = if self.local.manifest().graphs[root.graph_index].kind
                == super::GraphKind::Speculative
            {
                super::parallel::agree_speculative
            } else {
                super::parallel::agree
            };
            agreement(&root, &completion)
                .map_err(|error| invalid(format!("rank {}: {error}", index + 1)))?;
            root.timing.merge_rank(index + 1, completion.timing)?;
        }
        root.timing.queued = frame.queued;
        root.timing.wall_ns = frame.dispatched.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        Ok(Some(root))
    }
    pub fn close(&mut self) -> Shutdown {
        if let Some(shutdown) = &self.shutdown {
            return shutdown.clone();
        }
        let result = (|| {
            while !self.pending.is_empty() {
                self.poll()?;
                thread::sleep(Duration::from_micros(50));
            }
            if let Some(error) = &self.poison {
                return Err(error.clone());
            }
            self.ring.request_drain();
            for peer in &mut self.peers {
                if !matches!(peer.receive()?, Reply::Drained) {
                    return Err(invalid("rank thread did not drain"));
                }
                let shutdown = peer
                    .thread
                    .take()
                    .unwrap()
                    .join()
                    .map_err(|_| invalid("rank thread panicked"))?;
                if !shutdown.drained {
                    return Err(invalid("rank thread did not confirm drain"));
                }
                if let Some(error) = shutdown.error {
                    return Err(error);
                }
            }
            Ok(())
        })();
        if result.is_err() {
            // Best-effort drain and join of any peers that can still exit cleanly so live
            // peer threads are not detached on poisoned teardown.
            self.ring.request_drain();
            for peer in &mut self.peers {
                if peer.thread.is_none() {
                    continue;
                }
                let deadline = Instant::now() + TIMEOUT;
                let mut exited = false;
                while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
                    match receive(&peer.receiver, Some(remaining)) {
                        Ok(Ok(Reply::Drained))
                        | Ok(Err(_))
                        | Err(mpsc::RecvTimeoutError::Disconnected) => {
                            exited = true;
                            break;
                        }
                        Ok(Ok(_)) => continue,
                        Err(mpsc::RecvTimeoutError::Timeout) => break,
                    }
                }
                if exited {
                    if let Some(thread) = peer.thread.take() {
                        let _ = thread.join();
                    }
                }
            }
        }
        let shutdown = match result {
            Ok(()) => self.local.close(),
            Err(error) => self.local.abandon(error),
        };
        self.shutdown = Some(shutdown.clone());
        shutdown
    }
}
impl Drop for ThreadedExecutor {
    fn drop(&mut self) {
        self.close();
    }
}

fn drain_rank(
    executor: &mut GraphExecutor,
    started: &mut VecDeque<Instant>,
    send: &impl Fn(Reply) -> Result<()>,
) -> Result<()> {
    drain_outstanding_frames(
        executor,
        |exec| exec.outstanding(),
        |exec| exec.poll(),
        started,
        TIMEOUT,
        |completion| {
            let _ = send(Reply::Complete(completion));
        },
    )?;
    let shutdown = executor.close();
    if !shutdown.drained {
        return Err(invalid("rank streams did not drain"));
    }
    if let Some(error) = shutdown.error {
        return Err(error);
    }
    let _ = send(Reply::Drained);
    Ok(())
}

fn rank_loop(
    peer_index: usize,
    executor: &mut GraphExecutor,
    commands: mpsc::Receiver<Command>,
    replies: &mpsc::Sender<Result<Reply>>,
    ring: &DispatchRing,
) -> Result<()> {
    let send = |reply| {
        replies
            .send(Ok(reply))
            .map_err(|_| invalid("rank coordinator disconnected"))
    };
    send(Reply::Initialized)?;
    let mut started = VecDeque::with_capacity(2);
    // Startup phase: run startup tuning graphs and FinishStartup over the control channel.
    loop {
        match receive(&commands, None) {
            Ok(Command::Tune(index)) => {
                let status = executor.run_tuning_graph(index);
                send(Reply::Tuned(index, status))?;
            }
            Ok(Command::FinishStartup) => {
                if let Err(error) = executor.finish_startup() {
                    let _ = replies.send(Err(error));
                    return Ok(());
                }
                send(Reply::Started)?;
                break;
            }
            Ok(Command::Drain) => {
                return drain_rank(executor, &mut started, &send);
            }
            Err(_) => return Ok(()),
        }
    }
    // Serving phase: lock-free Prepare -> Ready -> Commit/Cancel step dispatch over DispatchRing.
    let mut next_epoch = 1u64;
    loop {
        let event = if executor.outstanding() == 0 {
            Some(ring.wait_next(next_epoch))
        } else {
            ring.poll_next(next_epoch)
        };
        match event {
            Some(DispatchEvent::Step) => {
                // SAFETY: `ring` published `prepare_epoch >= next_epoch`; this borrow ends
                // before signalling readiness or advancing past `next_epoch`.
                let (fence, graph, grammar, validated) = {
                    let slot = unsafe { ring.slot(next_epoch) };
                    let fence = slot.fence;
                    let graph = slot.graph;
                    let grammar = slot.grammar;
                    let validated = slot.with_uploads(|uploads| {
                        executor.validate_submission(graph, uploads, grammar)
                    });
                    (fence, graph, grammar, validated)
                };
                let validation_error = match validated {
                    Ok(Some(v)) if v == fence => None,
                    Ok(_) => Some(invalid("rank submission fence differs")),
                    Err(error) => Some(error),
                };
                if let Some(error) = validation_error {
                    // Signal failure immediately so rank zero aborts this step without
                    // launching a stranded collective, then drain any prior in-flight frame.
                    let _ = replies.send(Err(error.clone()));
                    ring.mark_peer_exited();
                    let _ = drain_outstanding_frames(
                        executor,
                        |exec| exec.outstanding(),
                        |exec| exec.poll(),
                        &mut started,
                        TIMEOUT,
                        |completion| {
                            let _ = send(Reply::Complete(completion));
                        },
                    );
                    return Err(error);
                }
                ring.mark_peer_ready(peer_index, next_epoch);
                match ring.wait_decision(next_epoch, TIMEOUT)? {
                    DecisionEvent::Commit => {
                        // SAFETY: Rank zero committed `next_epoch` and will not overwrite
                        // `slots[next_epoch & 1]` until `wait_peer_acks((next_epoch + 1) << 1)`,
                        // which happens strictly after this block drops `&StepSlot`.
                        let submitted = {
                            let slot = unsafe { ring.slot(next_epoch) };
                            slot.with_uploads(|uploads| executor.submit(graph, uploads, grammar))?
                        };
                        if submitted != Some(fence) {
                            return Err(invalid("rank submission fence differs"));
                        }
                        next_epoch += 1;
                        started.push_back(Instant::now());
                    }
                    DecisionEvent::Cancel => {
                        ring.mark_peer_cancelled(peer_index, next_epoch);
                        next_epoch += 1;
                    }
                    DecisionEvent::Drain => {
                        return drain_rank(executor, &mut started, &send);
                    }
                }
            }
            Some(DispatchEvent::Drain) => {
                return drain_rank(executor, &mut started, &send);
            }
            None => {}
        }
        if let Some(completion) = executor.poll()? {
            started.pop_front();
            send(Reply::Complete(completion))?;
        }
        if started.front().is_some_and(|time| time.elapsed() > TIMEOUT) {
            return Err(invalid("rank graph completion deadline expired"));
        }
        if executor.outstanding() != 0 && !ring.has_step(next_epoch) {
            thread::yield_now();
        }
    }
}

fn same_layout(a: &ExecutionManifest, b: &ExecutionManifest) -> bool {
    match (&a.engram, &b.engram) {
        // Partitioned tables have independent delivery groups, each numbered from zero.
        (Some(a), Some(b)) if Arc::ptr_eq(&a.shared, &b.shared) && a.rank == b.rank => {
            return false;
        }
        (Some(_), None) | (None, Some(_)) => return false,
        _ => {}
    }
    match (&a.offload, &b.offload) {
        (Some(a), Some(b)) if !a.shares(b) || a.rank() == b.rank() => return false,
        (Some(_), None) | (None, Some(_)) => return false,
        _ => {}
    }
    match (&a.cache, &b.cache) {
        (Some(a), Some(b)) if !a.same_layout(b) => return false,
        (Some(_), None) | (None, Some(_)) => return false,
        _ => {}
    }
    if a.persistent_workers.len() != b.persistent_workers.len()
        || a.persistent_workers
            .iter()
            .zip(&b.persistent_workers)
            .any(|(a, b)| {
                a.worker_sm_count != b.worker_sm_count
                    || a.graph != b.graph
                    || a.metrics_count != b.metrics_count
            })
    {
        return false;
    }
    a.completion_token_capacity == b.completion_token_capacity
        && a.max_active == b.max_active
        && a.input_frame_bytes == b.input_frame_bytes
        && a.buffers.len() == b.buffers.len()
        && a.tuning_graphs.len() == b.tuning_graphs.len()
        && a.tuning_graphs.iter().zip(&b.tuning_graphs).all(|(a, b)| {
            a.name == b.name && a.outputs == b.outputs && a.status_buffer == b.status_buffer
        })
        && a.graphs.len() == b.graphs.len()
        && a.buffers.iter().zip(&b.buffers).all(|(a, b)| {
            a.name == b.name
                && a.dtype == b.dtype
                && a.nbytes == b.nbytes
                && a.shape == b.shape
                && a.strides == b.strides
        })
        && a.graphs.iter().zip(&b.graphs).all(|(a, b)| {
            a.name == b.name
                && a.kind == b.kind
                && a.sampling_mode == b.sampling_mode
                && a.batch_capacity == b.batch_capacity
                && a.token_capacity == b.token_capacity
                && a.draft_tokens == b.draft_tokens
                && a.confidence_width == b.confidence_width
                && a.packed_token_capacity == b.packed_token_capacity
                && a.prefill_capacity == b.prefill_capacity
                && a.tuned_bindings == b.tuned_bindings
                && a.input_bindings.len() == b.input_bindings.len()
                && a.input_bindings
                    .iter()
                    .zip(&b.input_bindings)
                    .all(|(a, b)| {
                        a.buffer_name == b.buffer_name
                            && a.host_offset == b.host_offset
                            && a.nbytes == b.nbytes
                    })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};

    #[test]
    fn rank_layout_uses_completion_geometry_instead_of_abi_labels() {
        let a = super::super::tests::manifest();
        let mut b = a.clone();
        b.abi_version = 5;
        assert!(same_layout(&a, &b));
        b.completion_token_capacity += 1;
        assert!(!same_layout(&a, &b));
    }

    #[test]
    fn handoff_receive_preserves_queued_messages_and_disconnects() {
        let (sender, receiver) = mpsc::channel();
        sender.send(7).unwrap();
        sender.send(9).unwrap();
        drop(sender);
        assert_eq!(receive(&receiver, Some(Duration::ZERO)), Ok(7));
        assert_eq!(receive(&receiver, None), Ok(9));
        assert_eq!(
            receive(&receiver, None),
            Err(mpsc::RecvTimeoutError::Disconnected)
        );
    }

    #[test]
    fn handoff_receive_parks_for_late_messages_and_respects_deadlines() {
        let (sender, receiver) = mpsc::channel();
        assert_eq!(
            receive(&receiver, Some(Duration::ZERO)),
            Err(mpsc::RecvTimeoutError::Timeout)
        );
        let worker = thread::spawn(move || {
            thread::sleep(HANDOFF_POLL * 4);
            sender.send(17).unwrap();
        });
        assert_eq!(receive(&receiver, Some(Duration::from_secs(1))), Ok(17));
        worker.join().unwrap();
    }

    #[test]
    fn startup_tuning_layout_compares_producers_and_consumers() {
        let root = super::super::tests::tuning_manifest();
        let mut peer = root.clone();
        peer.tuning_graphs[0].handle += 32;
        assert!(same_layout(&root, &peer));
        for mutation in 0..4 {
            let mut peer = root.clone();
            match mutation {
                0 => peer.tuning_graphs[0].name.push('x'),
                1 => peer.tuning_graphs[0].outputs[0].push('x'),
                2 => peer.tuning_graphs[0].status_buffer.push('x'),
                _ => peer.graphs[0].tuned_bindings[0].buffer.push('x'),
            }
            assert!(!same_layout(&root, &peer), "mutation {mutation}");
        }
    }

    #[test]
    fn rank_layout_compares_bindings_but_not_cuda_addresses() {
        let root = super::super::tests::manifest();
        let mut peer = root.clone();
        peer.context += 1;
        peer.device += 1;
        for buffer in &mut peer.buffers {
            buffer.address += 0x10000;
        }
        for graph in &mut peer.graphs {
            graph.handle += 0x10000;
        }
        assert!(same_layout(&root, &peer));
        peer.graphs[0].input_bindings[0].host_offset += 8;
        assert!(!same_layout(&root, &peer));
        peer = root.clone();
        peer.buffers.pop();
        assert!(!same_layout(&root, &peer));
        peer = root.clone();
        peer.graphs[0].token_capacity += 1;
        assert!(!same_layout(&root, &peer));
        peer = root.clone();
        peer.graphs[0].prefill_capacity = 64;
        assert!(!same_layout(&root, &peer));
        let geometry = flare_selene::testing::geometry(64, &[("k", 8192)], 4096);
        peer = root.clone();
        peer.cache = Some(geometry.clone());
        assert!(!same_layout(&root, &peer));
        let mut root = root;
        root.cache = Some(geometry.clone());
        for plane in &mut peer.cache.as_mut().unwrap().planes {
            plane.address += 0x1000;
        }
        assert!(same_layout(&root, &peer));
        peer.cache.as_mut().unwrap().planes[0].unit_bytes += 1;
        assert!(!same_layout(&root, &peer));
    }

    #[test]
    fn rank_layout_accepts_shared_and_partitioned_engram_delivery() {
        use crate::engram::{Binding, Index, Plane, Shared, Table};

        for partitioned in [false, true] {
            let groups: Vec<_> = (0..if partitioned { 4 } else { 1 })
                .map(|rank| {
                    Shared::layout_fixture(
                        Index {
                            tables: vec![Table {
                                planes: [256, 8].map(|row_bytes| Plane {
                                    file: rank,
                                    offset: rank as u64 * 4096,
                                    nbytes: (100 + rank) * row_bytes,
                                    row_bytes,
                                }),
                                heads: 24,
                                weights: "weights".into(),
                                scales: "scales".into(),
                                hashes: "hashes".into(),
                                event: "ready".into(),
                                request: Some("requests".into()),
                                prefetch: None,
                            }],
                        },
                        if partitioned { 1 } else { 4 },
                    )
                })
                .collect();
            let ranks: Vec<_> = (0..4)
                .map(|rank| {
                    let mut manifest = super::super::tests::manifest();
                    manifest.context += rank;
                    manifest.device = rank as i32;
                    for buffer in &mut manifest.buffers {
                        buffer.address += rank * 0x100000;
                    }
                    for graph in &mut manifest.graphs {
                        graph.handle += rank * 0x100000;
                    }
                    manifest.engram = Some(Binding {
                        shared: groups[if partitioned { rank } else { 0 }].clone(),
                        rank: if partitioned { 0 } else { rank },
                        events: vec![rank + 1],
                    });
                    manifest.validate().unwrap();
                    manifest
                })
                .collect();
            for peer in &ranks[1..] {
                assert!(same_layout(&ranks[0], peer));
            }
            // A delivery group cannot count the same member twice.
            let mut duplicate = ranks[1].clone();
            duplicate.engram = ranks[0].engram.clone();
            assert!(!same_layout(&ranks[0], &duplicate));
            let mut missing = ranks[1].clone();
            missing.engram = None;
            assert!(!same_layout(&ranks[0], &missing));
            let mut mismatch = ranks[1].clone();
            mismatch.graphs[0].input_bindings[0].host_offset += 8;
            assert!(!same_layout(&ranks[0], &mismatch));
        }
    }

    #[test]
    fn tp4_catalog_accepts_shared_and_partitioned_embedding_delivery() {
        use super::super::{BufferDescriptor, Dtype, GraphKind, InputBinding};
        use crate::engram::{Binding, Index, Plane, Shared, Table};

        for logprobs in [false, true] {
            let mut root = if logprobs {
                super::super::tests::with_logprobs(20)
            } else {
                super::super::tests::manifest()
            };
            root.abi_version = super::super::ABI_VERSION;
            let records = root
                .buffers
                .iter_mut()
                .find(|b| b.name == "completion_records")
                .unwrap();
            records.nbytes = 2 * 4 * 32 * 8;
            records.shape = vec![2, 4, 32];
            records.strides = vec![4 * 32, 32, 1];
            root.buffers.push(BufferDescriptor {
                name: "proposal_confidence".into(),
                address: 0x30000,
                nbytes: 2 * 4 * 15 * 4,
                dtype: Dtype::Float32,
                shape: vec![2, 4, 15],
                strides: vec![60, 15, 1],
            });
            root.buffers.push(BufferDescriptor {
                name: "prefill_row".into(),
                address: 0x40000,
                nbytes: 4,
                dtype: Dtype::Int32,
                shape: vec![1],
                strides: vec![1],
            });
            let template = root.graphs[0].clone();
            root.graphs = [
                ("prefill", GraphKind::Prefill, 1, 4096, 0, 0),
                ("decode", GraphKind::Decode, 4, 1, 0, 0),
                ("resume", GraphKind::Resume, 4, 1, 0, 0),
                ("speculative", GraphKind::Speculative, 4, 6, 0, 12),
                ("mixed", GraphKind::Mixed, 4, 6, 4096, 0),
            ]
            .into_iter()
            .enumerate()
            .map(|(index, (name, kind, batch, width, prefill, packed))| {
                let mut graph = template.clone();
                graph.name = name.into();
                graph.kind = kind;
                graph.handle += index;
                graph.batch_capacity = batch;
                graph.token_capacity = width;
                graph.prefill_capacity = prefill;
                graph.packed_token_capacity = packed;
                if kind == GraphKind::Speculative {
                    graph.draft_tokens = 5;
                    graph.confidence_width = 5;
                }
                if kind == GraphKind::Mixed {
                    graph.input_bindings.push(InputBinding {
                        buffer_name: "prefill_row".into(),
                        host_offset: root.input_frame_bytes,
                        nbytes: 4,
                    });
                }
                graph
            })
            .collect();
            root.input_frame_bytes += 8;
            root.validate().unwrap();

            for partitioned in [false, true] {
                let groups: Vec<_> = (0..if partitioned { 4 } else { 1 })
                    .map(|rank| {
                        Shared::layout_fixture(
                            Index {
                                tables: vec![Table {
                                    planes: [256, 8].map(|row_bytes| Plane {
                                        file: rank,
                                        offset: rank as u64 * 4096,
                                        nbytes: (100 + rank) * row_bytes,
                                        row_bytes,
                                    }),
                                    heads: 24,
                                    weights: "weights".into(),
                                    scales: "scales".into(),
                                    hashes: "hashes".into(),
                                    event: "ready".into(),
                                    request: Some("requests".into()),
                                    prefetch: None,
                                }],
                            },
                            if partitioned { 1 } else { 4 },
                        )
                    })
                    .collect();
                let ranks: Vec<_> = (0..4)
                    .map(|rank| {
                        let mut manifest = root.clone();
                        manifest.context += rank;
                        manifest.device = rank as i32;
                        for buffer in &mut manifest.buffers {
                            buffer.address += rank * 0x100000;
                        }
                        for graph in &mut manifest.graphs {
                            graph.handle += rank * 0x100000;
                        }
                        manifest.engram = Some(Binding {
                            shared: groups[if partitioned { rank } else { 0 }].clone(),
                            rank: if partitioned { 0 } else { rank },
                            events: vec![rank + 1],
                        });
                        manifest.validate().unwrap();
                        manifest
                    })
                    .collect();
                for peer in &ranks[1..] {
                    assert!(same_layout(&ranks[0], peer));
                }
                // A repeated member of one delivery group would corrupt its completion accounting.
                assert!(!same_layout(&ranks[0], &ranks[0]));
                let mut bad = ranks[1].clone();
                bad.graphs[3].input_bindings[0].host_offset += 8;
                assert!(!same_layout(&ranks[0], &bad));
                let mut bad = ranks[1].clone();
                bad.buffers[2].shape[1] += 1;
                assert!(!same_layout(&ranks[0], &bad));
            }
        }
    }

    #[test]
    fn dispatch_ring_ten_thousand_steps_across_four_ranks_without_races_or_drops() {
        const STEPS: u64 = 10_000;
        const PEERS: usize = 3; // 4 ranks total (coordinator + 3 worker threads)

        let ring = Arc::new(DispatchRing::new(4096, 16));
        let mut receivers = Vec::with_capacity(PEERS);
        let mut workers = Vec::with_capacity(PEERS);
        for peer_id in 0..PEERS {
            let (tx, rx) = mpsc::channel::<(Fence, usize, u64)>();
            let peer_ring = ring.clone();
            let handle = thread::spawn(move || {
                let mut next_epoch = 1u64;
                loop {
                    match peer_ring.wait_next(next_epoch) {
                        DispatchEvent::Step => {
                            let (fence, graph) = {
                                let slot = unsafe { peer_ring.slot(next_epoch) };
                                (slot.fence, slot.graph)
                            };
                            peer_ring.mark_peer_ready(peer_id, next_epoch);
                            match peer_ring
                                .wait_decision(next_epoch, Duration::from_secs(10))
                                .unwrap()
                            {
                                DecisionEvent::Commit => {
                                    let slot = unsafe { peer_ring.slot(next_epoch) };
                                    let checksum = slot.with_uploads(|uploads| {
                                        assert_eq!(uploads.len(), 3);
                                        assert_eq!(uploads[0].buffer, "slot_ids");
                                        assert_eq!(uploads[1].buffer, "positions");
                                        assert_eq!(uploads[2].buffer, "tokens");
                                        let mut acc = 0u64;
                                        for u in uploads {
                                            for chunk in u.bytes.chunks_exact(8) {
                                                acc = acc.wrapping_add(u64::from_le_bytes(
                                                    chunk.try_into().unwrap(),
                                                ));
                                            }
                                        }
                                        acc ^ (peer_id as u64)
                                    });
                                    tx.send((fence, graph, checksum)).unwrap();
                                    next_epoch += 1;
                                }
                                DecisionEvent::Cancel => {
                                    peer_ring.mark_peer_cancelled(peer_id, next_epoch);
                                    next_epoch += 1;
                                }
                                DecisionEvent::Drain => break,
                            }
                        }
                        DispatchEvent::Drain => break,
                    }
                }
                next_epoch - 1
            });
            receivers.push(rx);
            workers.push(handle);
        }

        // Exercise rapid back-to-back 2-deep pipelining across all 10,000 steps.
        let mut in_flight: VecDeque<(u64, usize, u64)> = VecDeque::with_capacity(2);
        let mut next_epoch = 1u64;
        let mut buf0 = [0u8; 64];
        let mut buf1 = [0u8; 64];
        let mut buf2 = [0u8; 64];
        for step in 1..=STEPS {
            if in_flight.len() == 2 {
                let (expected_fence, expected_graph, expected_sum) = in_flight.pop_front().unwrap();
                for (peer_id, rx) in receivers.iter().enumerate() {
                    let (fence, graph, sum) = rx.recv_timeout(Duration::from_secs(10)).unwrap();
                    assert_eq!(fence, Fence(expected_fence));
                    assert_eq!(graph, expected_graph);
                    assert_eq!(sum, expected_sum ^ (peer_id as u64));
                }
            }
            let v0 = step.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let v1 = step.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let v2 = step.wrapping_mul(0x94D0_49BB_1331_11EB);
            for chunk in buf0.chunks_exact_mut(8) {
                chunk.copy_from_slice(&v0.to_le_bytes());
            }
            for chunk in buf1.chunks_exact_mut(8) {
                chunk.copy_from_slice(&v1.to_le_bytes());
            }
            for chunk in buf2.chunks_exact_mut(8) {
                chunk.copy_from_slice(&v2.to_le_bytes());
            }
            let expected_sum = v0
                .wrapping_mul(8)
                .wrapping_add(v1.wrapping_mul(8))
                .wrapping_add(v2.wrapping_mul(8));
            let uploads = [
                Upload {
                    buffer: "slot_ids",
                    offset: 0,
                    bytes: &buf0,
                },
                Upload {
                    buffer: "positions",
                    offset: 64,
                    bytes: &buf1,
                },
                Upload {
                    buffer: "tokens",
                    offset: 128,
                    bytes: &buf2,
                },
            ];
            let graph = (step as usize) % 7;
            let submitted = unsafe {
                coordinate_step(
                    &ring,
                    PEERS,
                    &mut next_epoch,
                    Fence(step),
                    graph,
                    false,
                    &uploads,
                    Duration::from_secs(10),
                    || invalid("unexpected peer exit"),
                    &mut (),
                    |_| Ok(Some(Fence(step))),
                    |_| Ok(Some(Fence(step))),
                )
                .unwrap()
            };
            assert_eq!(submitted, Some(Fence(step)));
            in_flight.push_back((step, graph, expected_sum));
        }
        while let Some((expected_fence, expected_graph, expected_sum)) = in_flight.pop_front() {
            for (peer_id, rx) in receivers.iter().enumerate() {
                let (fence, graph, sum) = rx.recv_timeout(Duration::from_secs(10)).unwrap();
                assert_eq!(fence, Fence(expected_fence));
                assert_eq!(graph, expected_graph);
                assert_eq!(sum, expected_sum ^ (peer_id as u64));
            }
        }
        ring.request_drain();
        for worker in workers {
            assert_eq!(worker.join().unwrap(), STEPS);
        }
        // Verify ping-pong slot allocations stayed bounded without per-step growth.
        for index in 0..2 {
            let slot = unsafe { &*ring.slots[index].0.get() };
            assert!(slot.payload.capacity() <= 4096);
            assert!(slot.uploads.capacity() <= 16);
        }
    }

    #[test]
    fn dispatch_ring_wakes_parked_peers_and_tracks_peer_exit() {
        let ring = Arc::new(DispatchRing::new(512, 8));
        let (tx, rx) = mpsc::channel();
        let peer_ring = ring.clone();
        let worker = thread::spawn(move || {
            let _guard = PeerExitGuard(peer_ring.clone());
            assert_eq!(peer_ring.wait_next(1), DispatchEvent::Step);
            peer_ring.mark_peer_ready(0, 1);
            assert_eq!(
                peer_ring.wait_decision(1, Duration::from_secs(2)).unwrap(),
                DecisionEvent::Commit
            );
            let slot = unsafe { peer_ring.slot(1) };
            tx.send((slot.fence, slot.graph)).unwrap();
            assert_eq!(peer_ring.wait_next(2), DispatchEvent::Drain);
        });
        // Wait past HANDOFF_POLL so the peer parks on the Condvar.
        let deadline = Instant::now() + Duration::from_secs(2);
        while ring.parked.0.load(Ordering::Acquire) == 0 {
            assert!(Instant::now() < deadline, "peer never entered idle park");
            thread::sleep(Duration::from_micros(200));
        }
        assert!(!ring.peer_exited());
        let data = 42u64.to_le_bytes();
        let uploads = [Upload {
            buffer: "tokens",
            offset: 0,
            bytes: &data,
        }];
        let mut next_epoch = 1u64;
        let fence = unsafe {
            coordinate_step(
                &ring,
                1,
                &mut next_epoch,
                Fence(1),
                3,
                false,
                &uploads,
                Duration::from_secs(2),
                || invalid("unexpected peer exit"),
                &mut (),
                |_| Ok(Some(Fence(1))),
                |_| Ok(Some(Fence(1))),
            )
            .unwrap()
        };
        assert_eq!(fence, Some(Fence(1)));
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            (Fence(1), 3)
        );
        ring.request_drain();
        worker.join().unwrap();
        assert!(ring.peer_exited());
    }

    #[test]
    fn coordinate_step_aborts_before_local_submit_when_peer_validation_fails() {
        const PEERS: usize = 3;
        let ring = Arc::new(DispatchRing::new(512, 8));
        let peer_launches = Arc::new(AtomicUsize::new(0));
        let (err_tx, err_rx) = mpsc::channel::<Error>();
        let mut workers = Vec::with_capacity(PEERS);
        for peer_id in 0..PEERS {
            let peer_ring = ring.clone();
            let launches = peer_launches.clone();
            let peer_err_tx = err_tx.clone();
            workers.push(thread::spawn(move || {
                let _guard = PeerExitGuard(peer_ring.clone());
                let mut next_epoch = 1u64;
                loop {
                    match peer_ring.wait_next(next_epoch) {
                        DispatchEvent::Step => {
                            let fence = unsafe { peer_ring.slot(next_epoch).fence };
                            // Simulate rank-local validation failure (e.g., partitioned Engram staging error)
                            // on peer 1 during step 2 before signalling readiness.
                            if peer_id == 1 && fence == Fence(2) {
                                let _ = peer_err_tx.send(invalid("rank-1 engram staging error"));
                                return;
                            }
                            peer_ring.mark_peer_ready(peer_id, next_epoch);
                            match peer_ring
                                .wait_decision(next_epoch, Duration::from_secs(5))
                                .unwrap()
                            {
                                DecisionEvent::Commit => {
                                    launches.fetch_add(1, Ordering::SeqCst);
                                    next_epoch += 1;
                                }
                                DecisionEvent::Cancel => {
                                    peer_ring.mark_peer_cancelled(peer_id, next_epoch);
                                    next_epoch += 1;
                                }
                                DecisionEvent::Drain => return,
                            }
                        }
                        DispatchEvent::Drain => return,
                    }
                }
            }));
        }
        let data = 9u64.to_le_bytes();
        let uploads = [Upload {
            buffer: "tokens",
            offset: 0,
            bytes: &data,
        }];
        let mut next_epoch = 1u64;
        let mut local_launches = 0usize;

        // Step 1 succeeds across all ranks.
        let step1 = unsafe {
            coordinate_step(
                &ring,
                PEERS,
                &mut next_epoch,
                Fence(1),
                0,
                false,
                &uploads,
                Duration::from_secs(5),
                || invalid("unexpected peer exit"),
                &mut (),
                |_| Ok(Some(Fence(1))),
                |_| {
                    local_launches += 1;
                    Ok(Some(Fence(1)))
                },
            )
            .unwrap()
        };
        assert_eq!(step1, Some(Fence(1)));
        assert_eq!(local_launches, 1);

        // Step 2 fails validation on peer 1; rank zero must NOT call local_submit and
        // healthy peers must drain without launching step 2.
        let step2 = unsafe {
            coordinate_step(
                &ring,
                PEERS,
                &mut next_epoch,
                Fence(2),
                0,
                false,
                &uploads,
                Duration::from_secs(5),
                || {
                    err_rx
                        .try_recv()
                        .unwrap_or_else(|_| invalid("rank thread disconnected"))
                },
                &mut (),
                |_| Ok(Some(Fence(2))),
                |_| {
                    local_launches += 1;
                    Ok(Some(Fence(2)))
                },
            )
        };
        assert_eq!(step2.unwrap_err().to_string(), "rank-1 engram staging error");
        assert_eq!(
            local_launches, 1,
            "rank zero must not launch when peer validation fails"
        );
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(
            peer_launches.load(Ordering::SeqCst),
            PEERS,
            "healthy peers must not launch the aborted step"
        );
    }

    #[test]
    fn coordinate_step_cancels_cleanly_on_local_backpressure_without_poisoning() {
        const PEERS: usize = 3;
        let ring = Arc::new(DispatchRing::new(512, 8));
        let mut receivers = Vec::with_capacity(PEERS);
        let mut workers = Vec::with_capacity(PEERS);
        for peer_id in 0..PEERS {
            let (tx, rx) = mpsc::channel::<(Fence, usize)>();
            let peer_ring = ring.clone();
            workers.push(thread::spawn(move || {
                let mut next_epoch = 1u64;
                let mut cancels = 0usize;
                loop {
                    match peer_ring.wait_next(next_epoch) {
                        DispatchEvent::Step => {
                            let (fence, graph) = {
                                let slot = unsafe { peer_ring.slot(next_epoch) };
                                (slot.fence, slot.graph)
                            };
                            peer_ring.mark_peer_ready(peer_id, next_epoch);
                            match peer_ring
                                .wait_decision(next_epoch, Duration::from_secs(5))
                                .unwrap()
                            {
                                DecisionEvent::Commit => {
                                    tx.send((fence, graph)).unwrap();
                                    next_epoch += 1;
                                }
                                DecisionEvent::Cancel => {
                                    cancels += 1;
                                    peer_ring.mark_peer_cancelled(peer_id, next_epoch);
                                    next_epoch += 1;
                                }
                                DecisionEvent::Drain => break,
                            }
                        }
                        DispatchEvent::Drain => break,
                    }
                }
                cancels
            }));
            receivers.push(rx);
        }
        let data = 11u64.to_le_bytes();
        let uploads = [Upload {
            buffer: "tokens",
            offset: 0,
            bytes: &data,
        }];
        let mut next_epoch = 1u64;

        // Step 1 commits normally.
        assert_eq!(
            unsafe {
                coordinate_step(
                    &ring,
                    PEERS,
                    &mut next_epoch,
                    Fence(1),
                    2,
                    false,
                    &uploads,
                    Duration::from_secs(5),
                    || invalid("unexpected peer exit"),
                    &mut (),
                    |_| Ok(Some(Fence(1))),
                    |_| Ok(Some(Fence(1))),
                )
                .unwrap()
            },
            Some(Fence(1))
        );
        for rx in &receivers {
            assert_eq!(
                rx.recv_timeout(Duration::from_secs(5)).unwrap(),
                (Fence(1), 2)
            );
        }

        // Next attempt hits documented Ok(None) backpressure from pre-commit validation: must
        // cancel peers cleanly without poisoning and without launching Fence(2).
        assert_eq!(
            unsafe {
                coordinate_step(
                    &ring,
                    PEERS,
                    &mut next_epoch,
                    Fence(2),
                    4,
                    false,
                    &uploads,
                    Duration::from_secs(5),
                    || invalid("unexpected peer exit"),
                    &mut (),
                    |_| Ok(None),
                    |_| unreachable!("backpressure must cancel before local_submit"),
                )
                .unwrap()
            },
            None
        );
        for rx in &receivers {
            assert!(matches!(rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
        }

        // Retrying Fence(2) after backpressure clears succeeds across every rank.
        assert_eq!(
            unsafe {
                coordinate_step(
                    &ring,
                    PEERS,
                    &mut next_epoch,
                    Fence(2),
                    5,
                    false,
                    &uploads,
                    Duration::from_secs(5),
                    || invalid("unexpected peer exit"),
                    &mut (),
                    |_| Ok(Some(Fence(2))),
                    |_| Ok(Some(Fence(2))),
                )
                .unwrap()
            },
            Some(Fence(2))
        );
        for rx in &receivers {
            assert_eq!(
                rx.recv_timeout(Duration::from_secs(5)).unwrap(),
                (Fence(2), 5)
            );
        }

        ring.request_drain();
        for worker in workers {
            assert_eq!(worker.join().unwrap(), 1);
        }
    }

    #[test]
    fn poll_decision_observes_commit_before_drain_under_two_deep_pipelining() {
        // Reproduces the 3-rank (rank 0 + 2 peers) 2-deep race:
        // Step 1 commits on rank 0; peer 0 launches step 1 and fails validation on step 2,
        // causing rank 0 to request drain while slow peer 1 is still in wait_decision(1).
        // Peer 1 MUST still observe Commit(1) before draining at epoch 2 so GPU collectives
        // already launched on rank 0 and peer 0 cannot deadlock.
        let ring = Arc::new(DispatchRing::new(256, 4));
        let data = 7u64.to_le_bytes();
        let uploads = [Upload {
            buffer: "tokens",
            offset: 0,
            bytes: &data,
        }];
        let mut next_epoch = 1u64;

        // Step 1: prepare, both peers validate, rank zero commits.
        unsafe { ring.prepare(1, Fence(1), 0, false, &uploads) };
        assert_eq!(ring.poll_next(1), Some(DispatchEvent::Step));
        ring.mark_peer_ready(0, 1);
        ring.mark_peer_ready(1, 1);
        ring.commit(1);
        next_epoch += 1;

        // Peer 0 observes Commit(1) and advances to epoch 2. Peer 1 has NOT polled decision(1) yet.
        assert_eq!(ring.poll_decision(1), Some(DecisionEvent::Commit));

        // Step 2: rank zero stages epoch 2 into parity slot 0 while peer 1 still needs slot 1.
        let data2 = 13u64.to_le_bytes();
        let uploads2 = [Upload {
            buffer: "tokens",
            offset: 0,
            bytes: &data2,
        }];
        unsafe { ring.prepare(next_epoch, Fence(2), 1, false, &uploads2) };
        assert_eq!(ring.poll_next(2), Some(DispatchEvent::Step));

        // Peer 0 fails validation on step 2 -> peer_exited + coordinator requests drain.
        ring.mark_peer_exited();
        ring.request_drain();

        // Slow peer 1 now polls decision(1): MUST observe Commit(1), NOT Drain!
        assert_eq!(
            ring.poll_decision(1),
            Some(DecisionEvent::Commit),
            "committed step must be observed before drain"
        );
        // Slot 1 still intact for peer 1's submission of step 1!
        let slot1 = unsafe { ring.slot(1) };
        assert_eq!(slot1.fence, Fence(1));
        assert_eq!(slot1.graph, 0);

        // Only after launching step 1 and advancing to epoch 2 does peer 1 observe Drain.
        assert_eq!(ring.poll_next(2), Some(DispatchEvent::Drain));
    }

    #[test]
    fn drain_outstanding_frames_completes_in_flight_work_before_close() {
        let mut remaining = VecDeque::from([Fence(1), Fence(2)]);
        let mut started = VecDeque::from([Instant::now(), Instant::now()]);
        let mut poll_calls = 0usize;
        let mut delivered = Vec::new();
        drain_outstanding_frames(
            &mut remaining,
            |q| q.len(),
            |q| {
                poll_calls += 1;
                if poll_calls % 2 == 0 {
                    Ok(q.pop_front())
                } else {
                    Ok(None)
                }
            },
            &mut started,
            Duration::from_secs(2),
            |fence| delivered.push(fence),
        )
        .expect("surviving peer must drain in-flight work before closing");
        assert!(remaining.is_empty());
        assert!(started.is_empty());
        assert_eq!(delivered, vec![Fence(1), Fence(2)]);

        // An expired in-flight frame during drain must still fail cleanly instead of hanging.
        let mut stuck_started = VecDeque::from([Instant::now() - Duration::from_secs(5)]);
        let mut stuck = 1usize;
        let err = drain_outstanding_frames::<usize, Fence>(
            &mut stuck,
            |count| *count,
            |_| Ok(None),
            &mut stuck_started,
            Duration::from_millis(10),
            |_| unreachable!(),
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "rank graph completion deadline expired during drain"
        );
    }

    #[test]
    fn parity_slot_reuse_after_cancel_while_prior_frame_in_flight() {
        // Exercises the exact slot-reuse safety boundary:
        // Step 1 commits (slot 1) and remains IN FLIGHT (not completed yet).
        // Step 2 prepares (slot 0) and is cancelled by local backpressure.
        // Step 3 reuses slot 1 while step 1's GPU frame is still in flight; this is safe
        // because every peer already dropped its `&StepSlot` borrow for slot 1 before
        // acknowledging readiness for step 2.
        const PEERS: usize = 2;
        let ring = Arc::new(DispatchRing::new(256, 4));
        let (launched_tx, launched_rx) = mpsc::channel::<(usize, Fence, u64)>();
        let mut workers = Vec::with_capacity(PEERS);
        for peer_id in 0..PEERS {
            let peer_ring = ring.clone();
            let tx = launched_tx.clone();
            workers.push(thread::spawn(move || {
                let mut next_epoch = 1u64;
                let mut in_flight: Vec<Fence> = Vec::new();
                loop {
                    match peer_ring.wait_next(next_epoch) {
                        DispatchEvent::Step => {
                            let fence = unsafe { peer_ring.slot(next_epoch).fence };
                            peer_ring.mark_peer_ready(peer_id, next_epoch);
                            match peer_ring
                                .wait_decision(next_epoch, Duration::from_secs(5))
                                .unwrap()
                            {
                                DecisionEvent::Commit => {
                                    let payload = {
                                        let slot = unsafe { peer_ring.slot(next_epoch) };
                                        slot.with_uploads(|uploads| {
                                            u64::from_le_bytes(
                                                uploads[0].bytes.try_into().unwrap(),
                                            )
                                        })
                                    };
                                    in_flight.push(fence);
                                    tx.send((peer_id, fence, payload)).unwrap();
                                    next_epoch += 1;
                                }
                                DecisionEvent::Cancel => {
                                    peer_ring.mark_peer_cancelled(peer_id, next_epoch);
                                    next_epoch += 1;
                                }
                                DecisionEvent::Drain => break,
                            }
                        }
                        DispatchEvent::Drain => break,
                    }
                }
                in_flight
            }));
        }

        let mut next_epoch = 1u64;
        let bytes1 = 0x1111_2222_3333_4444u64.to_le_bytes();
        let bytes2 = 0x5555_6666_7777_8888u64.to_le_bytes();
        let bytes3 = 0x9999_AAAA_BBBB_CCCCu64.to_le_bytes();

        // Step 1 (slot 1): commit and keep in flight.
        let u1 = [Upload { buffer: "tokens", offset: 0, bytes: &bytes1 }];
        assert_eq!(
            unsafe {
                coordinate_step(
                    &ring, PEERS, &mut next_epoch, Fence(1), 0, false, &u1,
                    Duration::from_secs(5), || invalid("exit"), &mut (), |_| Ok(Some(Fence(1))), |_| Ok(Some(Fence(1))),
                )
                .unwrap()
            },
            Some(Fence(1))
        );

        // Step 2 (slot 0): cancel on local backpressure while step 1 is still in flight.
        let u2 = [Upload { buffer: "tokens", offset: 0, bytes: &bytes2 }];
        assert_eq!(
            unsafe {
                coordinate_step(
                    &ring, PEERS, &mut next_epoch, Fence(2), 1, false, &u2,
                    Duration::from_secs(5), || invalid("exit"), &mut (), |_| Ok(None),
                    |_| unreachable!("backpressure must cancel before local_submit"),
                )
                .unwrap()
            },
            None
        );

        // Step 3 (reuses slot 1!): commit while step 1 is STILL in flight on all peers.
        let u3 = [Upload { buffer: "tokens", offset: 0, bytes: &bytes3 }];
        assert_eq!(
            unsafe {
                coordinate_step(
                    &ring, PEERS, &mut next_epoch, Fence(2), 2, false, &u3,
                    Duration::from_secs(5), || invalid("exit"), &mut (), |_| Ok(Some(Fence(2))), |_| Ok(Some(Fence(2))),
                )
                .unwrap()
            },
            Some(Fence(2))
        );

        let mut observed = Vec::new();
        for _ in 0..(PEERS * 2) {
            observed.push(launched_rx.recv_timeout(Duration::from_secs(5)).unwrap());
        }
        ring.request_drain();
        for worker in workers {
            assert_eq!(worker.join().unwrap(), vec![Fence(1), Fence(2)]);
        }
        for peer_id in 0..PEERS {
            assert!(observed.contains(&(peer_id, Fence(1), 0x1111_2222_3333_4444)));
            assert!(observed.contains(&(peer_id, Fence(2), 0x9999_AAAA_BBBB_CCCC)));
        }
    }

    #[test]
    fn coordinate_step_commits_peers_before_blocking_local_collective_launch() {
        // Multi-GPU collective graphs (e.g., 16-GPU prefill) can block rank zero inside
        // cuGraphLaunch when GPU 0 enters a collective and waits for peer GPUs while its
        // pushbuffer drains. Peer Commit MUST already be published before local_submit runs
        // so every rank enters the collective concurrently rather than deadlocking.
        const PEERS: usize = 3;
        let ring = Arc::new(DispatchRing::new(256, 4));
        let collective = Arc::new(std::sync::Barrier::new(PEERS + 1));
        let mut workers = Vec::with_capacity(PEERS);
        for peer_id in 0..PEERS {
            let peer_ring = ring.clone();
            let barrier = collective.clone();
            workers.push(thread::spawn(move || {
                assert_eq!(peer_ring.wait_next(1), DispatchEvent::Step);
                peer_ring.mark_peer_ready(peer_id, 1);
                assert_eq!(
                    peer_ring.wait_decision(1, Duration::from_secs(5)).unwrap(),
                    DecisionEvent::Commit
                );
                // Simulate peer GPU entering the collective launched in executor.submit().
                barrier.wait();
                assert_eq!(peer_ring.wait_next(2), DispatchEvent::Drain);
            }));
        }
        let data = 1u64.to_le_bytes();
        let uploads = [Upload {
            buffer: "tokens",
            offset: 0,
            bytes: &data,
        }];
        let mut next_epoch = 1u64;
        let result = unsafe {
            coordinate_step(
                &ring,
                PEERS,
                &mut next_epoch,
                Fence(1),
                0,
                false,
                &uploads,
                Duration::from_secs(5),
                || invalid("unexpected peer exit"),
                &mut (),
                |_| Ok(Some(Fence(1))),
                |_| {
                    assert_eq!(
                        ring.poll_decision(1),
                        Some(DecisionEvent::Commit),
                        "peer commit must be published before local CUDA submission"
                    );
                    // Blocks until every peer observed Commit and joined the collective.
                    collective.wait();
                    Ok(Some(Fence(1)))
                },
            )
            .unwrap()
        };
        assert_eq!(result, Some(Fence(1)));
        ring.request_drain();
        for worker in workers {
            worker.join().unwrap();
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
    enum ModelCoordPhase {
        Idle = 0,
        WaitingReady = 1,
        WaitingCancel = 2,
        Draining = 3,
        LaunchingLocal = 4,
    }

    #[derive(Clone, Copy, Eq, PartialEq, Hash)]
    enum ModelPeerPhase {
        WaitingStep = 0,
        ReadingPrepare = 1,
        WaitingDecision = 2,
        ReadingCommit = 3,
        Draining = 4,
        Closed = 5,
        Abandoned = 6,
    }

    #[derive(Clone)]
    struct RingModelState {
        prepare_epoch: u64,
        decision_word: u64,
        drain: bool,
        peer_exited: bool,
        coord_phase: ModelCoordPhase,
        coord_epoch: u64,
        pending: u8,
        committed: u8,
        peer_phase: [ModelPeerPhase; 2],
        peer_epoch: [u64; 2],
        peer_ack: [u64; 2],
        peer_in_flight: [u8; 2],
        slot_write_conflict: bool,
        stranded_on_drain: bool,
        abandoned_in_flight: bool,
        local_launch_before_commit: bool,
    }

    impl RingModelState {
        fn initial() -> Self {
            Self {
                prepare_epoch: 0,
                decision_word: 0,
                drain: false,
                peer_exited: false,
                coord_phase: ModelCoordPhase::Idle,
                coord_epoch: 1,
                pending: 0,
                committed: 0,
                peer_phase: [ModelPeerPhase::WaitingStep; 2],
                peer_epoch: [1; 2],
                peer_ack: [0; 2],
                peer_in_flight: [0; 2],
                slot_write_conflict: false,
                stranded_on_drain: false,
                abandoned_in_flight: false,
                local_launch_before_commit: false,
            }
        }

        fn hydrate_ring(&self) -> DispatchRing {
            let ring = DispatchRing::new(64, 4);
            ring.prepare_epoch.0.store(self.prepare_epoch, Ordering::SeqCst);
            ring.decision.0.store(self.decision_word, Ordering::SeqCst);
            ring.drain.0.store(self.drain, Ordering::SeqCst);
            ring.peer_exited.0.store(self.peer_exited, Ordering::SeqCst);
            for p in 0..2 {
                ring.peer_ack[p].0.store(self.peer_ack[p], Ordering::SeqCst);
            }
            ring
        }

        fn sync_from_ring(&mut self, ring: &DispatchRing) {
            self.prepare_epoch = ring.prepare_epoch.0.load(Ordering::SeqCst);
            self.decision_word = ring.decision.0.load(Ordering::SeqCst);
            self.drain = ring.drain.0.load(Ordering::SeqCst);
            self.peer_exited = ring.peer_exited.0.load(Ordering::SeqCst);
            for p in 0..2 {
                self.peer_ack[p] = ring.peer_ack[p].0.load(Ordering::SeqCst);
            }
        }

        fn projection(&self) -> [usize; 18] {
            let flags = usize::from(self.slot_write_conflict)
                | (usize::from(self.stranded_on_drain) << 1)
                | (usize::from(self.abandoned_in_flight) << 2)
                | (usize::from(self.local_launch_before_commit) << 3);
            [
                self.prepare_epoch as usize,
                (self.decision_word >> 1) as usize,
                (self.decision_word & 1) as usize,
                usize::from(self.drain),
                usize::from(self.peer_exited),
                self.coord_phase as usize,
                self.coord_epoch as usize,
                self.pending as usize,
                self.committed as usize,
                self.peer_phase[0] as usize,
                self.peer_epoch[0] as usize,
                self.peer_ack[0] as usize,
                self.peer_in_flight[0] as usize,
                self.peer_phase[1] as usize,
                self.peer_epoch[1] as usize,
                self.peer_ack[1] as usize,
                self.peer_in_flight[1] as usize,
                flags,
            ]
        }

        fn peer_reads_slot(&self, peer: usize, slot: u64) -> bool {
            matches!(
                self.peer_phase[peer],
                ModelPeerPhase::ReadingPrepare | ModelPeerPhase::ReadingCommit
            ) && (self.peer_epoch[peer] & 1) == slot
        }
    }

    struct LiveRingTrace {
        state: RingModelState,
        edges: Vec<([usize; 18], [usize; 3], [usize; 18])>,
        actions: HashSet<usize>,
    }

    impl LiveRingTrace {
        fn new() -> Self {
            Self {
                state: RingModelState::initial(),
                edges: Vec::new(),
                actions: HashSet::new(),
            }
        }

        fn record(&mut self, action: [usize; 3], update: impl FnOnce(&mut RingModelState)) {
            let before = self.state.projection();
            update(&mut self.state);
            let after = self.state.projection();
            self.actions.insert(action[0]);
            self.edges.push((before, action, after));
        }

        fn observe_prepare(&mut self, ring: &DispatchRing) {
            let prepared = ring.prepare_epoch.0.load(Ordering::Acquire);
            if self.state.coord_phase == ModelCoordPhase::Idle
                && prepared >= self.state.coord_epoch
            {
                let epoch = self.state.coord_epoch;
                assert_eq!(prepared, epoch, "prepare_epoch must advance sequentially");
                let slot = epoch & 1;
                let conflict = (0..2).any(|p| self.state.peer_reads_slot(p, slot));
                self.record([1, epoch as usize, 0], |st| {
                    st.prepare_epoch = prepared;
                    st.coord_phase = ModelCoordPhase::WaitingReady;
                    if conflict {
                        st.slot_write_conflict = true;
                    }
                });
            }
        }

        fn observe_decision(&mut self, ring: &DispatchRing) {
            let decision = ring.decision.0.load(Ordering::Acquire);
            if self.state.coord_phase == ModelCoordPhase::WaitingReady
                && (decision >> 1) >= self.state.coord_epoch
            {
                let epoch = self.state.coord_epoch;
                if (decision & 1) != 0 {
                    self.record([5, epoch as usize, 0], |st| {
                        st.decision_word = decision;
                        st.coord_phase = ModelCoordPhase::LaunchingLocal;
                        st.committed |= 1 << (epoch - 1);
                    });
                } else {
                    self.record([6, epoch as usize, 0], |st| {
                        st.decision_word = decision;
                        st.coord_phase = ModelCoordPhase::WaitingCancel;
                    });
                }
            }
        }

        fn observe_drain(&mut self, ring: &DispatchRing) {
            if ring.drain.0.load(Ordering::SeqCst) && !self.state.drain {
                if self.state.coord_phase == ModelCoordPhase::LaunchingLocal {
                    self.record([18, 0, 0], |st| {
                        st.drain = true;
                        st.coord_phase = ModelCoordPhase::Draining;
                        st.coord_epoch += 1;
                    });
                } else {
                    self.record([8, 0, 0], |st| {
                        st.drain = true;
                        st.coord_phase = ModelCoordPhase::Draining;
                    });
                }
            }
        }
    }

    fn record_production_coordinate_step_edges(
        edges: &mut BTreeSet<([usize; 18], [usize; 3], [usize; 18])>,
    ) {
        let mut observed_actions = HashSet::new();
        let data = 1u64.to_le_bytes();
        let uploads = [Upload {
            buffer: "tokens",
            offset: 0,
            bytes: &data,
        }];

        // Scenario A: production coordinate_step across epoch 1 Commit (with blocking
        // 3-rank collective Barrier), epoch 2 Cancel (backpressure), and epoch 3 Commit
        // followed by post-commit local_submit error -> Drain with in-flight frame drain.
        {
            let ring = Arc::new(DispatchRing::new(64, 4));
            let trace = Arc::new(Mutex::new(LiveRingTrace::new()));
            let collective = Arc::new(std::sync::Barrier::new(3));
            let mut workers = Vec::with_capacity(2);
            for peer in 0..2 {
                let peer_ring = ring.clone();
                let peer_trace = trace.clone();
                let peer_barrier = collective.clone();
                workers.push(thread::spawn(move || {
                    let mut in_flight = VecDeque::new();
                    let mut started = VecDeque::new();

                    // Epoch 1: Step -> Ready -> Commit -> blocking collective -> SubmitCommitted.
                    assert_eq!(peer_ring.wait_next(1), DispatchEvent::Step);
                    {
                        let mut tr = peer_trace.lock().unwrap();
                        tr.observe_prepare(&peer_ring);
                        tr.record([2, peer, 1], |st| {
                            st.peer_phase[peer] = ModelPeerPhase::ReadingPrepare;
                        });
                        peer_ring.mark_peer_ready(peer, 1);
                        let ack = peer_ring.peer_ack[peer].0.load(Ordering::Acquire);
                        tr.record([3, peer, 1], |st| {
                            st.peer_ack[peer] = ack;
                            st.peer_phase[peer] = ModelPeerPhase::WaitingDecision;
                        });
                    }
                    assert_eq!(
                        peer_ring.wait_decision(1, Duration::from_secs(5)).unwrap(),
                        DecisionEvent::Commit
                    );
                    {
                        let mut tr = peer_trace.lock().unwrap();
                        tr.observe_decision(&peer_ring);
                        tr.record([9, peer, 0], |st| {
                            st.peer_phase[peer] = ModelPeerPhase::ReadingCommit;
                        });
                    }
                    peer_barrier.wait();
                    {
                        let mut tr = peer_trace.lock().unwrap();
                        tr.record([12, peer, 1], |st| {
                            st.peer_in_flight[peer] |= 1;
                            st.peer_epoch[peer] += 1;
                            st.peer_phase[peer] = ModelPeerPhase::WaitingStep;
                        });
                    }
                    in_flight.push_back(1u64);
                    started.push_back(Instant::now());

                    // Epoch 2: Step -> Ready -> Cancel -> ack cancel.
                    assert_eq!(peer_ring.wait_next(2), DispatchEvent::Step);
                    {
                        let mut tr = peer_trace.lock().unwrap();
                        tr.observe_prepare(&peer_ring);
                        tr.record([2, peer, 2], |st| {
                            st.peer_phase[peer] = ModelPeerPhase::ReadingPrepare;
                        });
                        peer_ring.mark_peer_ready(peer, 2);
                        let ack = peer_ring.peer_ack[peer].0.load(Ordering::Acquire);
                        tr.record([3, peer, 2], |st| {
                            st.peer_ack[peer] = ack;
                            st.peer_phase[peer] = ModelPeerPhase::WaitingDecision;
                        });
                    }
                    assert_eq!(
                        peer_ring.wait_decision(2, Duration::from_secs(5)).unwrap(),
                        DecisionEvent::Cancel
                    );
                    {
                        let mut tr = peer_trace.lock().unwrap();
                        tr.observe_decision(&peer_ring);
                        peer_ring.mark_peer_cancelled(peer, 2);
                        let ack = peer_ring.peer_ack[peer].0.load(Ordering::Acquire);
                        tr.record([10, peer, 0], |st| {
                            st.peer_ack[peer] = ack;
                            st.peer_epoch[peer] += 1;
                            st.peer_phase[peer] = ModelPeerPhase::WaitingStep;
                        });
                    }

                    // Epoch 3 (reuses parity slot 1 while epoch 1 is in flight): Commit then Drain.
                    assert_eq!(peer_ring.wait_next(3), DispatchEvent::Step);
                    {
                        let mut tr = peer_trace.lock().unwrap();
                        tr.observe_prepare(&peer_ring);
                        tr.record([2, peer, 3], |st| {
                            st.peer_phase[peer] = ModelPeerPhase::ReadingPrepare;
                        });
                        peer_ring.mark_peer_ready(peer, 3);
                        let ack = peer_ring.peer_ack[peer].0.load(Ordering::Acquire);
                        tr.record([3, peer, 3], |st| {
                            st.peer_ack[peer] = ack;
                            st.peer_phase[peer] = ModelPeerPhase::WaitingDecision;
                        });
                    }
                    assert_eq!(
                        peer_ring.wait_decision(3, Duration::from_secs(5)).unwrap(),
                        DecisionEvent::Commit
                    );
                    {
                        let mut tr = peer_trace.lock().unwrap();
                        tr.observe_decision(&peer_ring);
                        tr.record([9, peer, 0], |st| {
                            st.peer_phase[peer] = ModelPeerPhase::ReadingCommit;
                        });
                        tr.record([12, peer, 3], |st| {
                            st.peer_in_flight[peer] |= 1 << 2;
                            st.peer_epoch[peer] += 1;
                            st.peer_phase[peer] = ModelPeerPhase::WaitingStep;
                        });
                    }
                    in_flight.push_back(3u64);
                    started.push_back(Instant::now());

                    // Epoch 4: Drain observed after post-commit local_submit error; drain in-flight frames.
                    assert_eq!(peer_ring.wait_next(4), DispatchEvent::Drain);
                    {
                        let mut tr = peer_trace.lock().unwrap();
                        tr.observe_drain(&peer_ring);
                        tr.record([13, peer, 4], |st| {
                            st.peer_phase[peer] = ModelPeerPhase::Draining;
                        });
                    }
                    drain_outstanding_frames(
                        &mut in_flight,
                        |pending| pending.len(),
                        |pending| Ok(pending.pop_front()),
                        &mut started,
                        Duration::from_secs(5),
                        |completed_epoch| {
                            let mut tr = peer_trace.lock().unwrap();
                            tr.record([14, peer, completed_epoch as usize], |st| {
                                st.peer_in_flight[peer] &= !(1 << (completed_epoch - 1));
                            });
                        },
                    )
                    .unwrap();
                    {
                        let mut tr = peer_trace.lock().unwrap();
                        tr.record([15, peer, 0], |st| {
                            st.peer_phase[peer] = ModelPeerPhase::Closed;
                        });
                    }
                }));
            }

            let mut next_epoch = 1u64;
            // Epoch 1: Commit + blocking collective Barrier.
            let res1 = unsafe {
                coordinate_step(
                    &ring,
                    2,
                    &mut next_epoch,
                    Fence(1),
                    0,
                    false,
                    &uploads,
                    Duration::from_secs(5),
                    || invalid("unexpected peer exit"),
                    &mut (),
                    |_| {
                        assert!(
                            (0..2).all(|p| ring.peer_ack[p].0.load(Ordering::Acquire) >= (1 << 1)),
                            "commit must wait for every peer readiness"
                        );
                        Ok(Some(Fence(1)))
                    },
                    |_| {
                        {
                            let mut tr = trace.lock().unwrap();
                            tr.observe_decision(&ring);
                            let decision = ring.decision.0.load(Ordering::Acquire);
                            if (decision >> 1) < 1 || (decision & 1) == 0 {
                                tr.state.local_launch_before_commit = true;
                            }
                            assert!(
                                !tr.state.local_launch_before_commit,
                                "peer commit must be published before local CUDA submission"
                            );
                            assert_eq!(tr.state.coord_phase, ModelCoordPhase::LaunchingLocal);
                        }
                        collective.wait();
                        {
                            let mut tr = trace.lock().unwrap();
                            tr.record([17, 1, 0], |st| {
                                st.coord_phase = ModelCoordPhase::Idle;
                                st.coord_epoch += 1;
                                st.pending |= 1;
                            });
                        }
                        Ok(Some(Fence(1)))
                    },
                )
                .unwrap()
            };
            assert_eq!(res1, Some(Fence(1)));

            // Epoch 2: Backpressure Cancel.
            let res2 = unsafe {
                coordinate_step(
                    &ring,
                    2,
                    &mut next_epoch,
                    Fence(2),
                    1,
                    false,
                    &uploads,
                    Duration::from_secs(5),
                    || invalid("unexpected peer exit"),
                    &mut (),
                    |_| {
                        assert!(
                            (0..2).all(|p| ring.peer_ack[p].0.load(Ordering::Acquire) >= (2 << 1)),
                            "commit must wait for every peer readiness"
                        );
                        Ok(None)
                    },
                    |_| unreachable!("backpressure must cancel before local_submit"),
                )
                .unwrap()
            };
            assert_eq!(res2, None);
            {
                let mut tr = trace.lock().unwrap();
                tr.observe_decision(&ring);
                assert_eq!(ring.poll_peer_acks(2, (2 << 1) | 1), Some(Ok(())));
                tr.record([7, 2, 0], |st| {
                    st.coord_phase = ModelCoordPhase::Idle;
                    st.coord_epoch += 1;
                });
            }

            // Epoch 3: Commit followed by post-commit local_submit error -> Drain.
            let res3 = unsafe {
                coordinate_step(
                    &ring,
                    2,
                    &mut next_epoch,
                    Fence(2),
                    2,
                    false,
                    &uploads,
                    Duration::from_secs(5),
                    || invalid("unexpected peer exit"),
                    &mut (),
                    |_| {
                        assert!(
                            (0..2).all(|p| ring.peer_ack[p].0.load(Ordering::Acquire) >= (3 << 1)),
                            "commit must wait for every peer readiness"
                        );
                        Ok(Some(Fence(2)))
                    },
                    |_| {
                        let mut tr = trace.lock().unwrap();
                        tr.observe_decision(&ring);
                        let decision = ring.decision.0.load(Ordering::Acquire);
                        if (decision >> 1) < 3 || (decision & 1) == 0 {
                            tr.state.local_launch_before_commit = true;
                        }
                        assert!(
                            !tr.state.local_launch_before_commit,
                            "peer commit must be published before local CUDA submission"
                        );
                        Err(invalid("simulated post-commit local submit failure"))
                    },
                )
            };
            assert!(res3.is_err());
            {
                let mut tr = trace.lock().unwrap();
                tr.observe_drain(&ring);
            }
            for worker in workers {
                worker.join().unwrap();
            }
            {
                let mut tr = trace.lock().unwrap();
                tr.record([16, 1, 0], |st| {
                    st.pending &= !1;
                });
                observed_actions.extend(tr.actions.iter().copied());
                edges.extend(tr.edges.iter().copied());
            }
        }

        // Scenario B: production coordinate_step aborts with Drain when peer 0 fails validation.
        {
            let ring = Arc::new(DispatchRing::new(64, 4));
            let trace = Arc::new(Mutex::new(LiveRingTrace::new()));
            let peer1_ready = Arc::new(std::sync::Barrier::new(2));

            let ring0 = ring.clone();
            let trace0 = trace.clone();
            let ready0 = peer1_ready.clone();
            let w0 = thread::spawn(move || {
                assert_eq!(ring0.wait_next(1), DispatchEvent::Step);
                {
                    let mut tr = trace0.lock().unwrap();
                    tr.observe_prepare(&ring0);
                    tr.record([2, 0, 1], |st| {
                        st.peer_phase[0] = ModelPeerPhase::ReadingPrepare;
                    });
                }
                // Let peer 1 enter WaitingDecision cleanly before peer 0 triggers peer exit.
                ready0.wait();
                {
                    let mut tr = trace0.lock().unwrap();
                    ring0.mark_peer_exited();
                    let exited = ring0.peer_exited.0.load(Ordering::SeqCst);
                    tr.record([4, 0, 1], |st| {
                        st.peer_exited = exited;
                        st.peer_phase[0] = ModelPeerPhase::Abandoned;
                    });
                }
            });

            let ring1 = ring.clone();
            let trace1 = trace.clone();
            let ready1 = peer1_ready.clone();
            let w1 = thread::spawn(move || {
                assert_eq!(ring1.wait_next(1), DispatchEvent::Step);
                {
                    let mut tr = trace1.lock().unwrap();
                    tr.observe_prepare(&ring1);
                    tr.record([2, 1, 1], |st| {
                        st.peer_phase[1] = ModelPeerPhase::ReadingPrepare;
                    });
                    ring1.mark_peer_ready(1, 1);
                    let ack = ring1.peer_ack[1].0.load(Ordering::Acquire);
                    tr.record([3, 1, 1], |st| {
                        st.peer_ack[1] = ack;
                        st.peer_phase[1] = ModelPeerPhase::WaitingDecision;
                    });
                }
                ready1.wait();
                assert_eq!(
                    ring1.wait_decision(1, Duration::from_secs(5)).unwrap(),
                    DecisionEvent::Drain
                );
                {
                    let mut tr = trace1.lock().unwrap();
                    tr.observe_drain(&ring1);
                    tr.record([11, 1, 0], |st| {
                        st.peer_phase[1] = ModelPeerPhase::Draining;
                    });
                    tr.record([15, 1, 0], |st| {
                        st.peer_phase[1] = ModelPeerPhase::Closed;
                    });
                }
            });

            let mut next_epoch = 1u64;
            let err = unsafe {
                coordinate_step(
                    &ring,
                    2,
                    &mut next_epoch,
                    Fence(1),
                    0,
                    false,
                    &uploads,
                    Duration::from_secs(5),
                    || invalid("peer validation failed"),
                    &mut (),
                    |_| unreachable!("peer failure must abort before local validation"),
                    |_| unreachable!("peer failure must abort before local submission"),
                )
            };
            assert!(err.is_err());
            {
                let mut tr = trace.lock().unwrap();
                tr.observe_drain(&ring);
            }
            w0.join().unwrap();
            w1.join().unwrap();
            let tr = trace.lock().unwrap();
            observed_actions.extend(tr.actions.iter().copied());
            edges.extend(tr.edges.iter().copied());
        }

        for required in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] {
            assert!(
                observed_actions.contains(&required),
                "production coordinate_step trace must exercise action {required}"
            );
        }
    }

    #[test]
    fn dispatch_ring_matches_model_edges() {
        const MAX_EPOCHS: u64 = 3;
        let mut work = vec![RingModelState::initial()];
        let mut states = HashSet::new();
        let mut edges = BTreeSet::new();
        let mut observed_commit_after_drain = false;
        let mut observed_drain_with_in_flight = false;
        let mut observed_cancel_slot_reuse = false;

        while let Some(state) = work.pop() {
            let before = state.projection();
            if !states.insert(before) {
                continue;
            }
            assert!(
                !state.slot_write_conflict,
                "parity slot overwritten while peer held borrow"
            );
            assert!(
                !state.stranded_on_drain,
                "committed step must be observed before drain"
            );
            assert!(
                !state.abandoned_in_flight,
                "surviving peer must drain in-flight work before closing"
            );
            assert!(
                !state.local_launch_before_commit,
                "peer commit must be published before local CUDA submission"
            );
            for e in 1..=MAX_EPOCHS {
                if (state.committed & (1 << (e - 1))) != 0 {
                    for p in 0..2 {
                        assert!(
                            state.peer_epoch[p] > e || state.peer_ack[p] >= (e << 1),
                            "commit must wait for every peer readiness"
                        );
                    }
                }
            }

            let mut emit = |after: RingModelState, action: [usize; 3]| {
                edges.insert((before, action, after.projection()));
                work.push(after);
            };

            // 1. CoordPrepare
            if state.coord_phase == ModelCoordPhase::Idle
                && !state.drain
                && !state.peer_exited
                && state.coord_epoch <= MAX_EPOCHS
                && state.pending.count_ones() < 2
            {
                let mut after = state.clone();
                let ring = after.hydrate_ring();
                let epoch = after.coord_epoch;
                let slot = epoch & 1;
                if (0..2).any(|p| state.peer_reads_slot(p, slot)) {
                    after.slot_write_conflict = true;
                }
                if epoch == 3 && (state.pending & 1) != 0 && (state.committed & 2) == 0 {
                    observed_cancel_slot_reuse = true;
                }
                unsafe { ring.prepare(epoch, Fence(epoch), 0, false, &[]) };
                after.sync_from_ring(&ring);
                after.coord_phase = ModelCoordPhase::WaitingReady;
                emit(after, [1, epoch as usize, 0]);
            }

            // 5, 6, 7, 8: Coordinator transitions driven by production poll_peer_acks / commit / cancel / request_drain.
            let ring_ro = state.hydrate_ring();
            match state.coord_phase {
                ModelCoordPhase::WaitingReady => {
                    let ack_status = ring_ro.poll_peer_acks(2, state.coord_epoch << 1);
                    if matches!(ack_status, Some(Ok(()))) && !state.peer_exited {
                        // 5. CoordCommit (publishes Commit to peers before local CUDA launch)
                        let mut after = state.clone();
                        let ring = after.hydrate_ring();
                        let epoch = after.coord_epoch;
                        ring.commit(epoch);
                        after.sync_from_ring(&ring);
                        after.coord_phase = ModelCoordPhase::LaunchingLocal;
                        after.committed |= 1 << (epoch - 1);
                        emit(after, [5, epoch as usize, 0]);

                        // 6. CoordCancel
                        let mut after = state.clone();
                        let ring = after.hydrate_ring();
                        let epoch = after.coord_epoch;
                        ring.cancel(epoch);
                        after.sync_from_ring(&ring);
                        after.coord_phase = ModelCoordPhase::WaitingCancel;
                        emit(after, [6, epoch as usize, 0]);
                    }
                    if !state.drain
                        && (matches!(ack_status, Some(Err(WaitPeersError::PeerExited)))
                            || matches!(ack_status, Some(Ok(()))))
                    {
                        // 8. CoordRequestDrain (peer exited or local validate error before commit)
                        let mut after = state.clone();
                        let ring = after.hydrate_ring();
                        ring.request_drain();
                        after.sync_from_ring(&ring);
                        after.coord_phase = ModelCoordPhase::Draining;
                        emit(after, [8, 0, 0]);
                    }
                }
                ModelCoordPhase::LaunchingLocal => {
                    if !state.drain {
                        // 17. CoordLaunchLocal (local submit completes after peer Commit published)
                        let mut after = state.clone();
                        let epoch = after.coord_epoch;
                        if (state.decision_word >> 1) < epoch || (state.decision_word & 1) == 0 {
                            after.local_launch_before_commit = true;
                        }
                        after.coord_phase = ModelCoordPhase::Idle;
                        after.coord_epoch += 1;
                        after.pending |= 1 << (epoch - 1);
                        emit(after, [17, epoch as usize, 0]);

                        // 18. CoordLaunchLocalFail (local submit error after Commit requests drain)
                        let mut after = state.clone();
                        let ring = after.hydrate_ring();
                        ring.request_drain();
                        after.sync_from_ring(&ring);
                        after.coord_phase = ModelCoordPhase::Draining;
                        after.coord_epoch += 1;
                        emit(after, [18, 0, 0]);
                    }
                }
                ModelCoordPhase::WaitingCancel => {
                    let cancel_status = ring_ro.poll_peer_acks(2, (state.coord_epoch << 1) | 1);
                    if matches!(cancel_status, Some(Ok(()))) && !state.peer_exited {
                        // 7. CoordCancelComplete
                        let mut after = state.clone();
                        let epoch = after.coord_epoch;
                        after.coord_phase = ModelCoordPhase::Idle;
                        after.coord_epoch += 1;
                        emit(after, [7, epoch as usize, 0]);
                    }
                    if !state.drain && matches!(cancel_status, Some(Err(WaitPeersError::PeerExited)))
                    {
                        // 8. CoordRequestDrain
                        let mut after = state.clone();
                        let ring = after.hydrate_ring();
                        ring.request_drain();
                        after.sync_from_ring(&ring);
                        after.coord_phase = ModelCoordPhase::Draining;
                        emit(after, [8, 0, 0]);
                    }
                }
                ModelCoordPhase::Idle => {
                    if !state.drain && (state.pending == 0 || state.peer_exited) {
                        // 8. CoordRequestDrain
                        let mut after = state.clone();
                        let ring = after.hydrate_ring();
                        ring.request_drain();
                        after.sync_from_ring(&ring);
                        after.coord_phase = ModelCoordPhase::Draining;
                        emit(after, [8, 0, 0]);
                    }
                }
                ModelCoordPhase::Draining => {}
            }

            // 16. CoordRetireFront
            if state.pending != 0 {
                let oldest = (state.pending.trailing_zeros() + 1) as u64;
                let bit = 1u8 << (oldest - 1);
                if (0..2).all(|p| {
                    state.peer_epoch[p] > oldest && (state.peer_in_flight[p] & bit) == 0
                }) {
                    let mut after = state.clone();
                    after.pending &= !bit;
                    emit(after, [16, oldest as usize, 0]);
                }
            }

            // Peer transitions driven by production poll_next / mark_peer_ready / poll_decision / mark_peer_cancelled / drain_outstanding_frames.
            for p in 0..2 {
                let epoch = state.peer_epoch[p];
                match state.peer_phase[p] {
                    ModelPeerPhase::WaitingStep => {
                        match ring_ro.poll_next(epoch) {
                            Some(DispatchEvent::Step) => {
                                // 2. PeerObserveStep
                                let mut after = state.clone();
                                after.peer_phase[p] = ModelPeerPhase::ReadingPrepare;
                                emit(after, [2, p, epoch as usize]);
                            }
                            Some(DispatchEvent::Drain) => {
                                // 13. PeerObserveStepDrain
                                let mut after = state.clone();
                                if epoch <= MAX_EPOCHS && (state.committed & (1 << (epoch - 1))) != 0 {
                                    after.stranded_on_drain = true;
                                }
                                after.peer_phase[p] = ModelPeerPhase::Draining;
                                emit(after, [13, p, epoch as usize]);
                            }
                            None => {}
                        }
                    }
                    ModelPeerPhase::ReadingPrepare => {
                        // 3. PeerValidateOk
                        let mut after = state.clone();
                        let ring = after.hydrate_ring();
                        ring.mark_peer_ready(p, epoch);
                        after.sync_from_ring(&ring);
                        after.peer_phase[p] = ModelPeerPhase::WaitingDecision;
                        emit(after, [3, p, epoch as usize]);

                        // 4. PeerValidateFail
                        let mut after = state.clone();
                        let ring = after.hydrate_ring();
                        ring.mark_peer_exited();
                        after.sync_from_ring(&ring);
                        after.peer_phase[p] = if state.peer_in_flight[p] == 0 {
                            ModelPeerPhase::Abandoned
                        } else {
                            ModelPeerPhase::Draining
                        };
                        emit(after, [4, p, epoch as usize]);
                    }
                    ModelPeerPhase::WaitingDecision => {
                        match ring_ro.poll_decision(epoch) {
                            Some(DecisionEvent::Commit) => {
                                // 9. PeerObserveCommit
                                if state.drain {
                                    observed_commit_after_drain = true;
                                }
                                let mut after = state.clone();
                                after.peer_phase[p] = ModelPeerPhase::ReadingCommit;
                                emit(after, [9, p, 0]);
                            }
                            Some(DecisionEvent::Cancel) => {
                                // 10. PeerObserveCancel
                                let mut after = state.clone();
                                let ring = after.hydrate_ring();
                                ring.mark_peer_cancelled(p, epoch);
                                after.sync_from_ring(&ring);
                                after.peer_epoch[p] += 1;
                                after.peer_phase[p] = ModelPeerPhase::WaitingStep;
                                emit(after, [10, p, 0]);
                            }
                            Some(DecisionEvent::Drain) => {
                                // 11. PeerObserveDecisionDrain
                                let mut after = state.clone();
                                if epoch <= MAX_EPOCHS && (state.committed & (1 << (epoch - 1))) != 0 {
                                    after.stranded_on_drain = true;
                                }
                                after.peer_phase[p] = ModelPeerPhase::Draining;
                                emit(after, [11, p, 0]);
                            }
                            None => {}
                        }
                    }
                    ModelPeerPhase::ReadingCommit => {
                        // 12. PeerSubmitCommitted
                        let mut after = state.clone();
                        after.peer_in_flight[p] |= 1 << (epoch - 1);
                        after.peer_epoch[p] += 1;
                        after.peer_phase[p] = ModelPeerPhase::WaitingStep;
                        emit(after, [12, p, epoch as usize]);
                    }
                    ModelPeerPhase::Draining => {
                        if state.peer_in_flight[p] != 0 {
                            observed_drain_with_in_flight = true;
                        } else {
                            // 15. PeerFinishDrain (verified via production drain_outstanding_frames)
                            let mut started = VecDeque::new();
                            let mut empty = 0usize;
                            drain_outstanding_frames::<usize, ()>(
                                &mut empty,
                                |count| *count,
                                |_| Ok(None),
                                &mut started,
                                Duration::from_secs(1),
                                |_| {},
                            )
                            .unwrap();
                            let mut after = state.clone();
                            after.peer_phase[p] = ModelPeerPhase::Closed;
                            emit(after, [15, p, 0]);
                        }
                    }
                    ModelPeerPhase::Closed | ModelPeerPhase::Abandoned => {}
                }

                // 14. PeerCompleteFrame (FIFO completion on any non-closed/abandoned peer)
                if !matches!(
                    state.peer_phase[p],
                    ModelPeerPhase::Closed | ModelPeerPhase::Abandoned
                ) && state.peer_in_flight[p] != 0
                {
                    let oldest = (state.peer_in_flight[p].trailing_zeros() + 1) as u64;
                    let mut after = state.clone();
                    after.peer_in_flight[p] &= !(1 << (oldest - 1));
                    emit(after, [14, p, oldest as usize]);
                }
            }
        }

        assert!(
            observed_commit_after_drain,
            "exploration must exercise committed step observation after drain"
        );
        assert!(
            observed_drain_with_in_flight,
            "exploration must exercise draining peers with in-flight GPU work"
        );
        assert!(
            observed_cancel_slot_reuse,
            "exploration must exercise parity slot reuse after cancel while prior frame is in flight"
        );
        record_production_coordinate_step_edges(&mut edges);
        println!(
            "DISPATCH_RING_COUNTS states={} edges={}",
            states.len(),
            edges.len()
        );
        for (before, action, after) in edges {
            println!("DISPATCH_RING_EDGE [{before:?}, {action:?}, {after:?}]");
        }
    }
}

#[cfg(all(test, feature = "runtime-core"))]
#[path = "threaded/cuda_tests.rs"]
mod cuda_tests;
