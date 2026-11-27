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
fn receive<T>(receiver: &mpsc::Receiver<T>, timeout: Option<Duration>) -> std::result::Result<T, mpsc::RecvTimeoutError> {
    let started = Instant::now();
    let poll = timeout.map_or(HANDOFF_POLL, |limit| limit.min(HANDOFF_POLL));
    loop {
        match receiver.try_recv() {
            Ok(value) => return Ok(value),
            Err(mpsc::TryRecvError::Disconnected) => return Err(mpsc::RecvTimeoutError::Disconnected),
            Err(mpsc::TryRecvError::Empty) => {}
        }
        if started.elapsed() >= poll { break; }
        std::hint::spin_loop();
    }
    match timeout {
        Some(limit) => receiver.recv_timeout(limit.saturating_sub(started.elapsed())),
        None => receiver.recv().map_err(|_| mpsc::RecvTimeoutError::Disconnected),
    }
}
fn invalid(message: impl Into<String>) -> Error {
    Error::execution(ExecutionError::Worker, message)
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
            let slice = unsafe { std::slice::from_raw_parts(stack.as_ptr().cast::<Upload<'_>>(), len) };
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
/// 1. Rank zero writes `slots[epoch & 1]` (only while no in-flight frame occupies that parity
///    slot) and publishes `prepare_epoch = epoch`.
/// 2. Every peer thread validates its local upload layout, frame capacity, and rank-local
///    Engram staging state without launching CUDA work, then stores `epoch << 1` in its
///    `peer_ack` slot.
/// 3. Only after every peer acknowledges readiness does rank zero submit locally and publish
///    `decision = (epoch << 1) | 1` (`Commit`). If any peer fails validation or exits, rank
///    zero requests drain without launching locally, preventing stranded collectives. If rank
///    zero encounters `Ok(None)` backpressure, it publishes `decision = epoch << 1` (`Cancel`),
///    waits for peers to acknowledge `(epoch << 1) | 1`, and returns `Ok(None)` without poisoning.
/// 4. Rank zero does not reuse `slots[epoch & 1]` until every peer has returned completion for
///    the prior committed step in that slot (or acknowledged a cancel), establishing a
///    happens-before edge before the next write.
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

// SAFETY: Rank zero is the sole writer to `slots[epoch & 1]` and writes a parity slot only
// after every peer has returned completion (or cancel acknowledgement) for the prior step
// occupying that slot. Peers obtain shared `&StepSlot` references only after observing
// `prepare_epoch >= step_epoch` and drop them before reporting completion or cancel ack.
unsafe impl Sync for DispatchRing {}

impl DispatchRing {
    fn new(payload_bytes: usize, upload_count: usize) -> Self {
        Self {
            slots: [
                CacheAligned(UnsafeCell::new(StepSlot::with_capacity(payload_bytes, upload_count))),
                CacheAligned(UnsafeCell::new(StepSlot::with_capacity(payload_bytes, upload_count))),
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
    /// Caller must be the sole producer and guarantee no peer is still reading `slots[epoch & 1]`.
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
        self.peer_ack[peer_index].0.store(epoch << 1, Ordering::Release);
    }

    fn mark_peer_cancelled(&self, peer_index: usize, epoch: u64) {
        self.peer_ack[peer_index].0.store((epoch << 1) | 1, Ordering::Release);
    }

    fn commit(&self, epoch: u64) {
        self.decision.0.store((epoch << 1) | 1, Ordering::SeqCst);
    }

    fn cancel(&self, epoch: u64) {
        self.decision.0.store(epoch << 1, Ordering::SeqCst);
    }

    fn wait_peer_acks(
        &self,
        peer_count: usize,
        target: u64,
        timeout: Duration,
    ) -> std::result::Result<(), WaitPeersError> {
        let first_pending = || {
            (0..peer_count)
                .find(|&index| self.peer_ack[index].0.load(Ordering::Acquire) < target)
        };
        for _ in 0..32 {
            if first_pending().is_none() {
                return Ok(());
            }
            if self.peer_exited() {
                return Err(WaitPeersError::PeerExited);
            }
            std::hint::spin_loop();
        }
        let started = Instant::now();
        loop {
            let Some(pending_peer) = first_pending() else {
                return Ok(());
            };
            if self.peer_exited() {
                return Err(WaitPeersError::PeerExited);
            }
            if started.elapsed() > timeout {
                return Err(WaitPeersError::Timeout);
            }
            #[cfg(any(all(target_arch = "aarch64", target_os = "linux"), target_arch = "x86_64"))]
            {
                let seen = self.peer_ack[pending_peer].0.load(Ordering::Acquire);
                if seen < target && !self.peer_exited() {
                    let _ = super::transport::host_wait::wait_u64_change(
                        &self.peer_ack[pending_peer].0,
                        seen,
                        25,
                    );
                }
            }
            #[cfg(not(any(all(target_arch = "aarch64", target_os = "linux"), target_arch = "x86_64")))]
            {
                let _ = pending_peer;
                std::hint::spin_loop();
            }
        }
    }

    fn poll_decision(&self, epoch: u64) -> Option<DecisionEvent> {
        if self.drain.0.load(Ordering::Acquire) {
            return Some(DecisionEvent::Drain);
        }
        let value = self.decision.0.load(Ordering::Acquire);
        if (value >> 1) >= epoch {
            return Some(if (value & 1) != 0 {
                DecisionEvent::Commit
            } else {
                DecisionEvent::Cancel
            });
        }
        None
    }

    fn wait_decision(&self, epoch: u64, timeout: Duration) -> Result<DecisionEvent> {
        for _ in 0..32 {
            if let Some(event) = self.poll_decision(epoch) {
                return Ok(event);
            }
            std::hint::spin_loop();
        }
        let started = Instant::now();
        loop {
            if let Some(event) = self.poll_decision(epoch) {
                return Ok(event);
            }
            if started.elapsed() > timeout {
                return Err(invalid("rank step decision deadline expired"));
            }
            #[cfg(any(all(target_arch = "aarch64", target_os = "linux"), target_arch = "x86_64"))]
            {
                let seen = self.decision.0.load(Ordering::Acquire);
                if (seen >> 1) < epoch && !self.drain.0.load(Ordering::Acquire) {
                    let _ = super::transport::host_wait::wait_u64_change(&self.decision.0, seen, 25);
                }
            }
            #[cfg(not(any(all(target_arch = "aarch64", target_os = "linux"), target_arch = "x86_64")))]
            {
                std::hint::spin_loop();
            }
        }
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
        if self.drain.0.load(Ordering::Acquire) {
            return Some(DispatchEvent::Drain);
        }
        if self.prepare_epoch.0.load(Ordering::Acquire) >= next_epoch {
            return Some(DispatchEvent::Step);
        }
        None
    }

    fn wait_next(&self, next_epoch: u64) -> DispatchEvent {
        // Fast spin before touching the monotonic clock so back-to-back handoffs incur zero vDSO reads.
        for _ in 0..32 {
            if let Some(event) = self.poll_next(next_epoch) {
                return event;
            }
            std::hint::spin_loop();
        }
        let started = Instant::now();
        let expected_prev = next_epoch - 1;
        while started.elapsed() < HANDOFF_POLL {
            if let Some(event) = self.poll_next(next_epoch) {
                return event;
            }
            #[cfg(any(all(target_arch = "aarch64", target_os = "linux"), target_arch = "x86_64"))]
            {
                let _ = super::transport::host_wait::wait_u64_change(
                    &self.prepare_epoch.0,
                    expected_prev,
                    25,
                );
            }
            #[cfg(not(any(all(target_arch = "aarch64", target_os = "linux"), target_arch = "x86_64")))]
            {
                let _ = expected_prev;
                std::hint::spin_loop();
            }
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
            guard = self.idle_wake.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Borrow the immutable slot for `step_epoch` after observing `prepare_epoch >= step_epoch`.
    ///
    /// # Safety
    /// Caller must have observed `prepare_epoch >= step_epoch` and must drop the returned
    /// reference before reporting completion or cancel acknowledgement to rank zero.
    unsafe fn slot(&self, step_epoch: u64) -> &StepSlot {
        let index = (step_epoch as usize) & 1;
        // SAFETY: Acquire/SeqCst load of `prepare_epoch >= step_epoch` synchronizes with `prepare`.
        unsafe { &*self.slots[index].0.get() }
    }
}

/// Coordinate peer pre-launch validation, rank-zero submission, and commit/cancel over `ring`.
///
/// # Safety
/// Caller must be the sole producer and guarantee no peer still reads `ring.slots[*next_epoch & 1]`.
#[allow(clippy::too_many_arguments)]
unsafe fn coordinate_step<F>(
    ring: &DispatchRing,
    peer_count: usize,
    next_epoch: &mut u64,
    fence: Fence,
    graph: usize,
    grammar: bool,
    uploads: &[Upload<'_>],
    timeout: Duration,
    on_peer_exit: impl FnOnce() -> Error,
    local_submit: F,
) -> Result<Option<Fence>>
where
    F: FnOnce() -> Result<Option<Fence>>,
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
    match local_submit() {
        Ok(Some(local_fence)) if local_fence == fence => {
            ring.commit(epoch);
            *next_epoch += 1;
            Ok(Some(fence))
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

fn can_submit_window(
    pending: &VecDeque<InFlightFrame>,
    barrier: bool,
    manifest: &ExecutionManifest,
) -> bool {
    pending.len() < 2
        && (!barrier || pending.is_empty())
        && !pending.iter().any(|frame| {
            frame.grammar
                || !matches!(
                    manifest.graphs[frame.graph].kind,
                    super::GraphKind::Decode | super::GraphKind::Speculative
                )
        })
}

fn record_local_completion(
    pending: &mut VecDeque<InFlightFrame>,
    completion: FrameCompletion,
) -> Result<()> {
    let frame = pending
        .iter_mut()
        .find(|frame| frame.fence == completion.fence)
        .ok_or_else(|| invalid("unexpected rank-zero completion fence"))?;
    if frame.graph != completion.graph_index {
        return Err(invalid("rank-zero completion identity or state differs"));
    }
    if frame.completions[0].is_some() {
        return Err(invalid("duplicate rank-zero completion"));
    }
    frame.completions[0] = Some(completion);
    Ok(())
}

fn record_peer_completion(
    pending: &mut VecDeque<InFlightFrame>,
    peer: usize,
    completion: FrameCompletion,
) -> Result<()> {
    let expected = pending
        .iter_mut()
        .find(|frame| frame.completions[peer + 1].is_none())
        .ok_or_else(|| invalid("unexpected rank completion"))?;
    if expected.fence != completion.fence || expected.graph != completion.graph_index {
        return Err(invalid("rank completion order or identity differs"));
    }
    expected.completions[peer + 1] = Some(completion);
    Ok(())
}

fn pop_completed_front(pending: &mut VecDeque<InFlightFrame>) -> Option<InFlightFrame> {
    if !pending
        .front()
        .is_some_and(|frame| frame.completions[..frame.world_size].iter().all(Option::is_some))
    {
        return None;
    }
    pending.pop_front()
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
                    eprintln!("rank initialization deadline expired; terminating the CUDA process");
                    std::process::abort();
                }
            };
            if !matches!(ready, Ok(Reply::Initialized)) {
                drop(peer.sender);
                match peer.thread.take().unwrap().join() {
                    Ok(shutdown) if shutdown.drained => {}
                    _ => {
                        eprintln!("rank initialization thread did not confirm drain; terminating the CUDA process");
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
                            eprintln!("startup rank dispatch failed with unconfirmed CUDA work: {error}; terminating the CUDA process");
                            std::process::abort();
                        }
                        return Err(error);
                    }
                }
                let mut failure = result.local.run_tuning_graph(index).err();
                for peer in &result.peers {
                    match peer.receive() {
                        Ok(Reply::Tuned(value, status)) if value == index => {
                            if let Err(error) = status { failure.get_or_insert(error); }
                        }
                        reply => {
                            let message = reply.err().map_or_else(|| "rank tuning completion identity differs".to_string(), |error| error.to_string());
                            eprintln!("startup rank completion is unconfirmed: {message}; terminating the CUDA process");
                            std::process::abort();
                        }
                    }
                }
                if let Some(error) = failure { return Err(error); }
            }
            for peer in &result.peers { peer.send(Command::FinishStartup)?; }
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
            let sent: Vec<_> = result.peers.iter().map(|peer| peer.send(Command::Drain).is_ok()).collect();
            for (peer, sent) in result.peers.iter_mut().zip(sent) {
                if sent {
                    loop {
                        match peer.receive() {
                            Ok(Reply::Drained) => break,
                            Ok(Reply::Started) => continue,
                            reply => {
                                eprintln!("startup rank cleanup was not confirmed: {:?}; terminating the CUDA process", reply.err());
                                std::process::abort();
                            }
                        }
                    }
                }
                if let Some(thread) = peer.thread.take() {
                    match thread.join() {
                        Ok(shutdown) if shutdown.drained => {}
                        _ => {
                            eprintln!("startup rank thread did not confirm drain; terminating the CUDA process");
                            std::process::abort();
                        }
                    }
                }
            }
            let shutdown = result.local.close();
            if !shutdown.drained {
                eprintln!("startup coordinator cleanup did not drain; terminating the CUDA process");
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
    pub fn park(&self) -> bool {
        self.local.park()
    }
    pub fn can_submit(&self, grammar: bool) -> bool {
        self.poison.is_none()
            && self.shutdown.is_none()
            && can_submit_window(&self.pending, grammar, self.local.manifest())
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
        let barrier = grammar || !matches!(self.local.manifest().graphs.get(graph).map(|g| g.kind),
            Some(super::GraphKind::Decode | super::GraphKind::Speculative));
        if !self.can_submit(barrier) { return Ok(None); }
        let Some(fence) = self.local.validate_submission(graph, uploads, grammar)? else {
            return Ok(None);
        };
        let queued = !self.pending.is_empty();
        let dispatched = Instant::now();
        let peers = &self.peers;
        // SAFETY: `can_submit` limits `self.pending.len() < 2` and every prior frame in parity
        // slot `next_epoch & 1` has already completed (or acknowledged cancel) on every peer.
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
                || self.local.submit(graph, uploads, grammar),
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
        if self.pending.iter().any(|frame| frame.dispatched.elapsed() > TIMEOUT) {
            return Err(invalid("tensor-parallel graph completion deadline expired"));
        }
        if let Some(completion) = self.local.poll()? {
            record_local_completion(&mut self.pending, completion)?;
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
                        record_peer_completion(&mut self.pending, index, completion)?;
                    }
                    _ => return Err(invalid("unexpected rank thread reply")),
                }
            }
        }
        let Some(mut frame) = pop_completed_front(&mut self.pending) else { return Ok(None); };
        let mut root = frame.completions[0].take().unwrap();
        for index in 0..self.peers.len() {
            let completion = frame.completions[index + 1].take().unwrap();
            let agreement = if self.local.manifest().graphs[root.graph_index].kind == super::GraphKind::Speculative {
                super::parallel::agree_speculative
            } else { super::parallel::agree };
            agreement(&root, &completion).map_err(|error| invalid(format!("rank {}: {error}", index + 1)))?;
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
                        Ok(Ok(Reply::Drained)) | Ok(Err(_)) | Err(mpsc::RecvTimeoutError::Disconnected) => {
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

fn drain_rank(executor: &mut GraphExecutor, send: &impl Fn(Reply) -> Result<()>) -> Result<()> {
    if executor.outstanding() != 0 {
        return Err(invalid("rank drain arrived with outstanding frames"));
    }
    let shutdown = executor.close();
    if !shutdown.drained {
        return Err(invalid("rank streams did not drain"));
    }
    if let Some(error) = shutdown.error {
        return Err(error);
    }
    send(Reply::Drained)?;
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
                return drain_rank(executor, &send);
            }
            Err(_) => return Ok(()),
        }
    }
    // Serving phase: lock-free Prepare -> Ready -> Commit/Cancel step dispatch over DispatchRing.
    let mut next_epoch = 1u64;
    let mut started = VecDeque::with_capacity(2);
    loop {
        let event = if executor.outstanding() == 0 {
            Some(ring.wait_next(next_epoch))
        } else {
            ring.poll_next(next_epoch)
        };
        match event {
            Some(DispatchEvent::Step) => {
                // SAFETY: `ring` published `prepare_epoch >= next_epoch`; this borrow ends
                // before signalling readiness or cancel acknowledgement.
                let (fence, graph, grammar, validated) = {
                    let slot = unsafe { ring.slot(next_epoch) };
                    let fence = slot.fence;
                    let graph = slot.graph;
                    let grammar = slot.grammar;
                    let validated =
                        slot.with_uploads(|uploads| executor.validate_submission(graph, uploads, grammar))?;
                    (fence, graph, grammar, validated)
                };
                if validated != Some(fence) {
                    return Err(invalid("rank submission fence differs"));
                }
                ring.mark_peer_ready(peer_index, next_epoch);
                match ring.wait_decision(next_epoch, TIMEOUT)? {
                    DecisionEvent::Commit => {
                        // SAFETY: Rank zero committed `next_epoch` and retains slot ownership
                        // until this peer reports completion.
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
                        return drain_rank(executor, &send);
                    }
                }
            }
            Some(DispatchEvent::Drain) => {
                return drain_rank(executor, &send);
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
            if !executor.park() {
                std::hint::spin_loop();
                thread::yield_now();
            }
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
        && a.tuning_graphs.iter().zip(&b.tuning_graphs).all(|(a, b)|
            a.name == b.name && a.outputs == b.outputs && a.status_buffer == b.status_buffer)
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
        assert_eq!(receive(&receiver, None), Err(mpsc::RecvTimeoutError::Disconnected));
    }

    #[test]
    fn handoff_receive_parks_for_late_messages_and_respects_deadlines() {
        let (sender, receiver) = mpsc::channel();
        assert_eq!(receive(&receiver, Some(Duration::ZERO)), Err(mpsc::RecvTimeoutError::Timeout));
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
            let records = root.buffers.iter_mut().find(|b| b.name == "completion_records").unwrap();
            records.nbytes = 2 * 4 * 32 * 8;
            records.shape = vec![2, 4, 32];
            records.strides = vec![4 * 32, 32, 1];
            root.buffers.push(BufferDescriptor {
                name: "proposal_confidence".into(), address: 0x30000, nbytes: 2 * 4 * 15 * 4,
                dtype: Dtype::Float32, shape: vec![2, 4, 15], strides: vec![60, 15, 1],
            });
            root.buffers.push(BufferDescriptor {
                name: "prefill_row".into(), address: 0x40000, nbytes: 4,
                dtype: Dtype::Int32, shape: vec![1], strides: vec![1],
            });
            let template = root.graphs[0].clone();
            root.graphs = [
                ("prefill", GraphKind::Prefill, 1, 4096, 0, 0),
                ("decode", GraphKind::Decode, 4, 1, 0, 0),
                ("resume", GraphKind::Resume, 4, 1, 0, 0),
                ("speculative", GraphKind::Speculative, 4, 6, 0, 12),
                ("mixed", GraphKind::Mixed, 4, 6, 4096, 0),
            ].into_iter().enumerate().map(|(index, (name, kind, batch, width, prefill, packed))| {
                let mut graph = template.clone();
                graph.name = name.into();
                graph.kind = kind;
                graph.handle += index;
                graph.batch_capacity = batch;
                graph.token_capacity = width;
                graph.prefill_capacity = prefill;
                graph.packed_token_capacity = packed;
                if kind == GraphKind::Speculative { graph.draft_tokens = 5; graph.confidence_width = 5; }
                if kind == GraphKind::Mixed {
                    graph.input_bindings.push(InputBinding {
                        buffer_name: "prefill_row".into(), host_offset: root.input_frame_bytes, nbytes: 4,
                    });
                }
                graph
            }).collect();
            root.input_frame_bytes += 8;
            root.validate().unwrap();

            for partitioned in [false, true] {
                let groups: Vec<_> = (0..if partitioned { 4 } else { 1 }).map(|rank| {
                    Shared::layout_fixture(Index { tables: vec![Table {
                        planes: [256, 8].map(|row_bytes| Plane {
                            file: rank, offset: rank as u64 * 4096,
                            nbytes: (100 + rank) * row_bytes, row_bytes,
                        }),
                        heads: 24, weights: "weights".into(), scales: "scales".into(),
                        hashes: "hashes".into(), event: "ready".into(), request: Some("requests".into()),
                    }] }, if partitioned { 1 } else { 4 })
                }).collect();
                let ranks: Vec<_> = (0..4).map(|rank| {
                    let mut manifest = root.clone();
                    manifest.context += rank;
                    manifest.device = rank as i32;
                    for buffer in &mut manifest.buffers { buffer.address += rank * 0x100000; }
                    for graph in &mut manifest.graphs { graph.handle += rank * 0x100000; }
                    manifest.engram = Some(Binding {
                        shared: groups[if partitioned { rank } else { 0 }].clone(),
                        rank: if partitioned { 0 } else { rank }, events: vec![rank + 1],
                    });
                    manifest.validate().unwrap();
                    manifest
                }).collect();
                for peer in &ranks[1..] { assert!(same_layout(&ranks[0], peer)); }
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
                            match peer_ring.wait_decision(next_epoch, Duration::from_secs(10)).unwrap() {
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
                                                acc = acc.wrapping_add(u64::from_le_bytes(chunk.try_into().unwrap()));
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
            for chunk in buf0.chunks_exact_mut(8) { chunk.copy_from_slice(&v0.to_le_bytes()); }
            for chunk in buf1.chunks_exact_mut(8) { chunk.copy_from_slice(&v1.to_le_bytes()); }
            for chunk in buf2.chunks_exact_mut(8) { chunk.copy_from_slice(&v2.to_le_bytes()); }
            let expected_sum = v0.wrapping_mul(8)
                .wrapping_add(v1.wrapping_mul(8))
                .wrapping_add(v2.wrapping_mul(8));
            let uploads = [
                Upload { buffer: "slot_ids", offset: 0, bytes: &buf0 },
                Upload { buffer: "positions", offset: 64, bytes: &buf1 },
                Upload { buffer: "tokens", offset: 128, bytes: &buf2 },
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
                    || Ok(Some(Fence(step))),
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
        let uploads = [Upload { buffer: "tokens", offset: 0, bytes: &data }];
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
                || Ok(Some(Fence(1))),
            )
            .unwrap()
        };
        assert_eq!(fence, Some(Fence(1)));
        assert_eq!(rx.recv_timeout(Duration::from_secs(2)).unwrap(), (Fence(1), 3));
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
                            match peer_ring.wait_decision(next_epoch, Duration::from_secs(5)).unwrap() {
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
        let uploads = [Upload { buffer: "tokens", offset: 0, bytes: &data }];
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
                || {
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
                || err_rx.try_recv().unwrap_or_else(|_| invalid("rank thread disconnected")),
                || {
                    local_launches += 1;
                    Ok(Some(Fence(2)))
                },
            )
        };
        assert_eq!(step2.unwrap_err().to_string(), "rank-1 engram staging error");
        assert_eq!(local_launches, 1, "rank zero must not launch when peer validation fails");
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
                            match peer_ring.wait_decision(next_epoch, Duration::from_secs(5)).unwrap() {
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
        let uploads = [Upload { buffer: "tokens", offset: 0, bytes: &data }];
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
                    || Ok(Some(Fence(1))),
                )
                .unwrap()
            },
            Some(Fence(1))
        );
        for rx in &receivers {
            assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), (Fence(1), 2));
        }

        // Next attempt hits documented Ok(None) backpressure from local.submit: must cancel
        // peers cleanly without poisoning and without launching Fence(2).
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
                    || Ok(None),
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
                    || Ok(Some(Fence(2))),
                )
                .unwrap()
            },
            Some(Fence(2))
        );
        for rx in &receivers {
            assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), (Fence(2), 5));
        }

        ring.request_drain();
        for worker in workers {
            assert_eq!(worker.join().unwrap(), 1);
        }
    }
}

#[cfg(all(test, feature = "runtime-core"))]
#[path = "threaded/cuda_tests.rs"]
mod cuda_tests;
