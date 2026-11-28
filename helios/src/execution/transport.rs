/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use std::{marker::PhantomData, rc::Rc, time::Instant};

use crate::{
    Error, ExecutionError, Failure, Result, Shutdown,
    cache::Fence,
    cuda::{BATCH_BYTES, BATCH_ENTRIES, Event, PinnedHost, Stream},
};

use super::{ExecutionManifest, GraphKind, DEFAULT_COMPLETION_TOKEN_CAPACITY, REQUEST_LANES, COMPLETION_METADATA_FIELDS, ReadinessIntervals};

/// Frame-completion wake for integrated GPUs whose CPU and GPU share a SoC power budget.
/// Discrete GPUs keep the lowest-latency spin-yield loop without parking. On Linux integrated
/// aarch64 SoCs with FEAT_WFxT (HWCAP2_WFXT, e.g. GB10), the copy stream writes each frame's
/// fence to a mapped host word right after the frame's `transferred` event, and
/// [`GraphExecutor::park`] waits in WFET with the exclusive monitor armed on that word. On Linux
/// integrated SoCs without FEAT_WFxT (e.g. Jetson Orin), [`GraphExecutor::park`] uses per-graph
/// GPU execution duration to sleep in bounded OS `nanosleep` intervals during the body of a frame
/// (so Linux `schedutil` can lower CPU frequency instead of starving the GPU on shared power),
/// switching to a low-latency landing wait right before predicted completion (hardware WFE on
/// aarch64 `EVTSTRM` SoCs, or spin-yield otherwise). The event query in [`GraphExecutor::poll`]
/// stays the only completion signal; the wake word can only end a wait early, best effort.
#[cfg_attr(not(all(target_arch = "aarch64", target_os = "linux")), allow(dead_code))]
struct CompletionWake {
    word: PinnedHost,
    device: usize,
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
mod host_wait {
    use std::sync::OnceLock;

    const HWCAP_EVTSTRM: libc::c_ulong = 1 << 11;
    const HWCAP2_WFXT: libc::c_ulong = 1 << 31;

    pub(super) fn wfxt_available() -> bool {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        // SAFETY: getauxval reads the process auxiliary vector and has no preconditions.
        *AVAILABLE.get_or_init(|| unsafe { libc::getauxval(libc::AT_HWCAP2) } & HWCAP2_WFXT != 0)
    }

    pub(super) fn evtstrm_available() -> bool {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        // SAFETY: getauxval reads the process auxiliary vector and has no preconditions.
        *AVAILABLE.get_or_init(|| unsafe { libc::getauxval(libc::AT_HWCAP) } & HWCAP_EVTSTRM != 0)
    }

    pub(super) fn ticks() -> u64 {
        let ticks: u64;
        // SAFETY: CNTVCT_EL0 is readable at EL0 on Linux; the ISB orders the read after earlier loads.
        unsafe { std::arch::asm!("isb", "mrs {t}, cntvct_el0", t = out(reg) ticks, options(nostack)) };
        ticks
    }

    pub(super) fn frequency() -> u64 {
        let hz: u64;
        // SAFETY: CNTFRQ_EL0 is readable at EL0 on Linux.
        unsafe { std::arch::asm!("mrs {f}, cntfrq_el0", f = out(reg) hz, options(nomem, nostack)) };
        hz
    }

    /// Park until `word` may differ from `old` or CNTVCT reaches `deadline`, whichever is first.
    /// Same sequence as Linux `__cmpwait_case_32`, with WFET bounded by an absolute deadline.
    ///
    /// # Safety
    /// `word` must point to a live, 4-byte aligned u32 in coherent memory, and WFxT must be available.
    pub(super) unsafe fn wait_change_wfet(word: *const u32, old: u32, deadline: u64) {
        unsafe {
            std::arch::asm!(
                "sevl",
                "wfe",
                "ldxr {tmp:w}, [{word}]",
                "eor {tmp:w}, {tmp:w}, {old:w}",
                "cbnz {tmp:w}, 2f",
                ".inst 0xd5031009", // wfet x9
                "2:",
                word = in(reg) word,
                old = in(reg) old,
                tmp = out(reg) _,
                in("x9") deadline,
                options(nostack),
            );
        }
    }

    /// Park until `word` may differ from `old` or the architectural event stream wakes the core.
    ///
    /// # Safety
    /// `word` must point to a live, 4-byte aligned u32 in coherent memory, and EVTSTRM must be
    /// available so WFE wakes periodically even if a store does not clear the local monitor.
    pub(super) unsafe fn wait_change_wfe(word: *const u32, old: u32) {
        unsafe {
            std::arch::asm!(
                "sevl",
                "wfe",
                "ldxr {tmp:w}, [{word}]",
                "eor {tmp:w}, {tmp:w}, {old:w}",
                "cbnz {tmp:w}, 2f",
                "wfe",
                "2:",
                word = in(reg) word,
                old = in(reg) old,
                tmp = out(reg) _,
                options(nostack),
            );
        }
    }
}

/// Longest intentional wait per park in WFET on FEAT_WFxT cores (DGX Spark / GB10).
/// Anything that cannot touch the wake word (a peer's socket message, a new request, a deadline
/// check) waits for this deadline, GB10's ~100-130 us WFE event stream, or a spurious wake.
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
const WFET_PARK_LIMIT_US: u64 = 100;

/// Minimum sleep duration worth issuing a Linux `nanosleep` syscall given 50 us default timer slack.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const PARK_MIN_SLEEP_US: u64 = 100;
/// Wake this many microseconds before predicted GPU completion to enter the landing wait.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const PARK_LANDING_US: u64 = 600;
/// Upper bound per OS sleep so incoming requests, cancellations, and peer messages stay responsive
/// and sudden frame-time drops are observed promptly in [`GraphExecutor::poll`].
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const PARK_MAX_SLEEP_US: u64 = 500;
/// Return to bounded OS sleep if a frame runs this many microseconds past its predicted duration.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const PARK_OVERSHOOT_CAP_US: u64 = 1_000;
/// Bounded OS sleep step for unmeasured graphs and frames that run past the landing window.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const PARK_FALLBACK_SLEEP_US: u64 = 250;

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParkAction {
    /// Put the thread to sleep in `TASK_INTERRUPTIBLE` so Linux `schedutil` can lower CPU clocks.
    Sleep(std::time::Duration),
    /// Enter the low-latency completion landing wait (WFE on aarch64 EVTSTRM, spin-yield otherwise).
    Land,
}

/// Update a graph's conservative GPU duration estimate in nanoseconds.
/// Adapts faster downward (1/2) than upward (1/4) so predictive sleeps wake before GPU completion
/// even when frame time drops on shorter chunks or smaller contexts.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn update_duration_estimate(estimate_ns: u64, observed_ns: u64) -> u64 {
    if observed_ns == 0 {
        return estimate_ns;
    }
    if estimate_ns == 0 {
        return observed_ns;
    }
    if observed_ns < estimate_ns {
        estimate_ns - (estimate_ns - observed_ns) / 2
    } else {
        estimate_ns + (observed_ns - estimate_ns) / 4
    }
}

/// Choose between a bounded OS sleep and the low-latency landing wait on an integrated SoC.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn decide_park(elapsed_us: u64, expected_us: u64) -> ParkAction {
    if expected_us == 0 {
        return ParkAction::Sleep(std::time::Duration::from_micros(PARK_FALLBACK_SLEEP_US));
    }
    if let Some(until_landing) = expected_us
        .saturating_sub(PARK_LANDING_US)
        .checked_sub(elapsed_us)
    {
        if until_landing >= PARK_MIN_SLEEP_US {
            return ParkAction::Sleep(std::time::Duration::from_micros(
                until_landing.min(PARK_MAX_SLEEP_US),
            ));
        }
    }
    if elapsed_us > expected_us.saturating_add(PARK_OVERSHOOT_CAP_US) {
        return ParkAction::Sleep(std::time::Duration::from_micros(PARK_FALLBACK_SLEEP_US));
    }
    ParkAction::Land
}

enum ParkMode {
    /// Discrete GPUs and non-Linux hosts keep zero-overhead spin-yield polling.
    Spin,
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    Wfet(CompletionWake),
    #[cfg(target_os = "linux")]
    Predictive {
        #[cfg(target_arch = "aarch64")]
        wfe_wake: Option<CompletionWake>,
    },
}

impl ParkMode {
    fn wake(&self) -> Option<&CompletionWake> {
        match self {
            Self::Spin => None,
            #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
            Self::Wfet(wake) => Some(wake),
            #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
            Self::Predictive { wfe_wake } => wfe_wake.as_ref(),
            #[cfg(all(not(target_arch = "aarch64"), target_os = "linux"))]
            Self::Predictive {} => None,
        }
    }
}

pub struct Upload<'a> {
    pub buffer: &'a str,
    pub offset: usize,
    pub bytes: &'a [u8],
}

#[cfg(kani)]
mod kani_proofs;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenVector {
    pub emitted_count: i64,
    pub offered_drafts: i64,
    pub accepted_drafts: i64,
    pub copy_offered_mask: i64,
    pub model_reserved: i64,
    pub draft_cache_consumed: i64,
    pub verification_width: i64,
    pub tokens: Vec<i64>,
}
impl Default for TokenVector {
    fn default() -> Self {
        Self {
            emitted_count: 0,
            offered_drafts: 0,
            accepted_drafts: 0,
            copy_offered_mask: 0,
            model_reserved: -1,
            draft_cache_consumed: 0,
            verification_width: 1,
            tokens: vec![-1; DEFAULT_COMPLETION_TOKEN_CAPACITY],
        }
    }
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CompletionRecord {
    pub generation: i64,
    pub submission_sequence: i64,
    pub consumed_tokens: i64,
    pub generated_tokens: i64,
    pub token_id: i64,
    pub status: i64,
    pub error_code: i64,
    pub reserved: i64,
    pub vector: Option<TokenVector>,
}

impl CompletionRecord {
    pub(crate) fn validate_vector(&self) -> Result<()> {
        if self.status != 0 {
            if let Some(v) = &self.vector {
                if !(1..=v.tokens.len() as i64).contains(&v.verification_width)
                    || !(0..=v.verification_width).contains(&v.emitted_count)
                    || !(0..v.verification_width).contains(&v.offered_drafts)
                    || v.accepted_drafts < v.emitted_count.saturating_sub(1).max(0)
                    || v.accepted_drafts > v.emitted_count
                    || v.accepted_drafts > v.offered_drafts
                    || v.copy_offered_mask < 0
                    || (v.offered_drafts < 63 && (v.copy_offered_mask as u64) >= (1u64 << v.offered_drafts))
                    || v.draft_cache_consumed < -1
                    || v.tokens[..v.emitted_count as usize]
                        .iter()
                        .any(|id| u32::try_from(*id).is_err())
                    || v.tokens[v.emitted_count as usize..]
                        .iter()
                        .any(|id| *id != -1)
                    || self.token_id
                        != if v.emitted_count == 0 {
                            -1
                        } else {
                            v.tokens[v.emitted_count as usize - 1]
                        }
                    || (self.status == 3 && v.emitted_count != 0)
                {
                    return Err(Error::execution(
                        ExecutionError::Completion,
                        "invalid bounded token-vector completion",
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn fields(&self) -> Vec<i64> {
        let mut values = vec![
            self.generation,
            self.submission_sequence,
            self.consumed_tokens,
            self.generated_tokens,
            self.token_id,
            self.status,
            self.error_code,
            self.reserved,
        ];
        let v = self.vector.as_ref().expect("completion serialization requires a token vector");
        values.extend([
            v.emitted_count,
            v.offered_drafts,
            v.accepted_drafts,
            v.copy_offered_mask,
            v.model_reserved,
            v.draft_cache_consumed,
            0,
            v.verification_width,
        ]);
        values.extend_from_slice(&v.tokens);
        values
    }
}
/// Completion storage sized to the prepared request and token capacities.
#[derive(Debug, Clone)]
pub struct CompletionRecords {
    records: Vec<CompletionRecord>,
}

impl std::ops::Deref for CompletionRecords {
    type Target = [CompletionRecord];
    fn deref(&self) -> &Self::Target {
        &self.records
    }
}
impl std::ops::DerefMut for CompletionRecords {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.records
    }
}
impl<const N: usize> From<[CompletionRecord; N]> for CompletionRecords {
    fn from(values: [CompletionRecord; N]) -> Self {
        assert!(N <= REQUEST_LANES);
        Self { records: Vec::from(values) }
    }
}

impl From<Vec<CompletionRecord>> for CompletionRecords {
    fn from(records: Vec<CompletionRecord>) -> Self {
        assert!(records.len() <= REQUEST_LANES);
        Self { records }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprobs {
    pub token: u32,
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PromptLogprobs {
    pub value: TokenLogprobs,
    pub rank: u32,
}

#[derive(Debug)]
pub struct FrameCompletion {
    pub fence: Fence,
    pub graph_index: usize,
    pub timing: ExecutionTiming,
    pub records: CompletionRecords,
    pub logprobs: Option<Vec<Vec<Option<TokenLogprobs>>>>,
    pub prompt_logprobs: Option<Vec<Option<PromptLogprobs>>>,
    /// Confidence by proposal batch row; rows and columns outside the graph shape are zero.
    pub proposal_confidence: Option<Vec<Vec<f32>>>,
    /// Verification prefix survival by physical slot; -1 marks an unobserved prefix.
    pub speculative_survival: Option<Vec<Vec<f32>>>,
}

pub(crate) fn validate_speculative_survival(rows: &[Vec<f32>], records: &[CompletionRecord]) -> Result<()> {
    if rows.len() != records.len() || rows.len() > REQUEST_LANES {
        return Err("speculative survival does not cover completion slots".into());
    }
    for (slot, row) in rows.iter().enumerate() {
        if records.get(slot).and_then(|record| record.vector.as_ref())
            .is_some_and(|vector| row.len() + 1 != vector.tokens.len()) {
            return Err("speculative survival width differs from completion capacity".into());
        }
        let offered = records.get(slot).filter(|record| matches!(record.status, 1 | 2))
            .and_then(|record| record.vector.as_ref()).map_or(0, |vector| {
                if vector.emitted_count > vector.accepted_drafts { vector.offered_drafts }
                else { vector.accepted_drafts.min(vector.offered_drafts) }
            }).clamp(0, row.len() as i64) as usize;
        let mut bound = 1.0;
        let mut missing = false;
        for (depth, &value) in row.iter().enumerate() {
            missing |= value == -1.0;
            if value != -1.0 && (missing || depth >= offered || !value.is_finite() || !(0.0..=bound).contains(&value)) {
                return Err(format!("invalid speculative survival at slot {slot}, depth {depth}: value {value}, preceding bound {bound}, observed extent {offered}, row {row:?}").into());
            }
            bound = value;
        }
    }
    Ok(())
}

fn reads_speculative_survival(graph: &super::GraphDescriptor) -> bool {
    graph.token_capacity > 1 && matches!(graph.kind, GraphKind::Speculative | GraphKind::Mixed)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutionTiming {
    /// Local CUDA event intervals for up to sixteen ranks; absent ranks have no interval.
    pub gpu_ns: [Option<u64>; 16],
    /// Per-table minima of the readiness-through-barrier and barrier-only intervals.
    pub readiness: Option<ReadinessIntervals>,
    /// Host interval from dispatch through observation of every rank's completion.
    /// Includes input staging, coordination, device queueing and completion delivery.
    pub wall_ns: u64,
    /// Another frame was outstanding when this frame was dispatched.
    pub queued: bool,
}

impl ExecutionTiming {
    pub(crate) fn merge_rank(&mut self, rank: usize, peer: Self) -> Result<()> {
        self.gpu_ns[rank] = peer.gpu_ns[0];
        match (&mut self.readiness, peer.readiness) {
            (Some(local), Some(remote)) => local.merge(remote),
            (None, None) => Ok(()),
            _ => Err(Error::execution(ExecutionError::Completion, "rank readiness timestamp bindings differ")),
        }
    }
}

struct Pending {
    logprobs: u64,
    logprob_width: usize,
    logprob_token_ids: Vec<Vec<u32>>,
    prompt_rows: usize,
    fence: Fence,
    graph_index: usize,
    grammar: bool,
    dispatched: Instant,
    queued: bool,
}

struct Frame {
    input: PinnedHost,
    output: PinnedHost,
    confidence: Option<Readback>,
    survival: Option<Readback>,
    readiness: Option<Readback>,
    logprobs: Option<Readback>,
    prompt_logprobs: Option<Readback>,
    produced: Event,
    started: Event,
    transferred: Event,
    pending: Option<Pending>,
}

enum Readback {
    MappedHost { address: usize, nbytes: usize },
    DeviceCopy(PinnedHost),
}

impl Readback {
    fn new(stream: &Stream, address: usize, nbytes: usize) -> Result<Self> {
        Ok(match stream.mapped_host_address(address, nbytes)? {
            Some(address) => Self::MappedHost { address, nbytes },
            None => Self::DeviceCopy(stream.host_buffer(nbytes)?),
        })
    }

    unsafe fn enqueue_copy(&self, stream: &Stream, source: usize, nbytes: usize) -> Result<()> {
        match self {
            Self::MappedHost { .. } => Ok(()),
            // SAFETY: The caller retains the device source and this pinned
            // destination until the transfer stream completes.
            Self::DeviceCopy(host) => unsafe { stream.download(host, source, nbytes) },
        }
    }

    unsafe fn enqueue_pitched_copy(
        &self,
        stream: &Stream,
        source: usize,
        source_pitch: usize,
        width: usize,
        rows: usize,
    ) -> Result<()> {
        match self {
            Self::MappedHost { .. } => Ok(()),
            // SAFETY: The caller retains both allocations through the transfer event.
            Self::DeviceCopy(host) => unsafe {
                stream.download_2d(host, source, source_pitch, width, rows)
            },
        }
    }

    /// The caller has observed the frame's transfer event and retains the source
    /// allocation; proposal feedback prevents another graph from overwriting it.
    unsafe fn bytes(&self) -> &[u8] {
        match self {
            Self::MappedHost { address, nbytes } => {
                // SAFETY: The mapped allocation and producing graph obey the
                // executor's ownership and completion requirements above.
                unsafe { std::slice::from_raw_parts(*address as *const u8, *nbytes) }
            }
            Self::DeviceCopy(host) => host.bytes(),
        }
    }
}

struct InputCopyBatches {
    destinations: Vec<u64>,
    sources: Vec<u64>,
    sizes: Vec<usize>,
    bytes: usize,
}

impl InputCopyBatches {
    fn new(capacity: usize) -> Self {
        let capacity = capacity.min(BATCH_ENTRIES);
        Self {
            destinations: Vec::with_capacity(capacity),
            sources: Vec::with_capacity(capacity),
            sizes: Vec::with_capacity(capacity),
            bytes: 0,
        }
    }

    fn clear(&mut self) {
        self.destinations.clear();
        self.sources.clear();
        self.sizes.clear();
        self.bytes = 0;
    }

    fn submit(
        &mut self,
        host: usize,
        copies: impl Iterator<Item = ResolvedUpload>,
        mut emit: impl FnMut(&[u64], &[u64], &[usize]) -> Result<()>,
    ) -> Result<()> {
        self.clear();
        for copy in copies {
            let mut offset = 0;
            while offset < copy.nbytes {
                if self.bytes == BATCH_BYTES || self.sizes.len() == BATCH_ENTRIES {
                    emit(&self.destinations, &self.sources, &self.sizes)?;
                    self.clear();
                }
                let nbytes = (copy.nbytes - offset).min(BATCH_BYTES - self.bytes);
                self.destinations.push((copy.device_address + offset) as u64);
                self.sources.push((host + copy.host_offset + offset) as u64);
                self.sizes.push(nbytes);
                self.bytes += nbytes;
                offset += nbytes;
            }
        }
        if !self.sizes.is_empty() {
            emit(&self.destinations, &self.sources, &self.sizes)?;
        }
        self.clear();
        Ok(())
    }
}

struct Resources {
    model: Stream,
    copy: Stream,
    frames: [Frame; 2],
    engram: Option<crate::engram::Transfers>,
    input_copies: InputCopyBatches,
    park_mode: ParkMode,
}

/// Thread-bound executor over one borrowed prepared model and its CUDA context.
pub struct GraphExecutor {
    manifest: ExecutionManifest,
    pub(super) copy_logprobs: bool,
    resources: Option<Resources>,
    resolved_uploads: Vec<ResolvedUpload>,
    startup_next: usize,
    startup_ready: bool,
    next_sequence: u64,
    completed: Fence,
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    graph_duration_ns: Vec<u64>,
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    last_completion_at: Option<Instant>,
    poison: Option<Error>,
    shutdown: Option<Shutdown>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl GraphExecutor {
    /// # Safety
    /// All manifest handles and allocations must belong to the given context,
    /// remain stable, and obey helios/manifest.md. The caller retains the owner
    /// until close confirms both streams drained. No other executor may mutate
    /// these inputs, graph executables, scratch or completion rows concurrently.
    pub unsafe fn new(manifest: ExecutionManifest) -> Result<Self> {
        let mut executor = unsafe { Self::new_unstarted(manifest)? };
        for index in executor.manifest.referenced_tuning_graphs() {
            executor.run_tuning_graph(index)?;
        }
        executor.finish_startup()?;
        Ok(executor)
    }

    /// Construct rank-local streams without launching any collective graphs.
    pub(super) unsafe fn new_unstarted(manifest: ExecutionManifest) -> Result<Self> {
        manifest.validate()?;
        let upload_capacity = manifest
            .graphs
            .iter()
            .map(|graph| graph.input_bindings.len())
            .max()
            .unwrap_or(0)
            .saturating_mul(manifest.request_capacity());
        let model = Stream::open(manifest.context, manifest.device)?;
        let copy = Stream::open(manifest.context, manifest.device)?;
        let mut frame_index = 0;
        let mut frame = || -> Result<Frame> {
            let index = frame_index;
            frame_index += 1;
            Ok(Frame {
                input: model.host_buffer(manifest.input_frame_bytes)?,
                output: model.host_buffer(manifest.completion_bytes())?,
                logprobs: if manifest.max_top_logprobs()? != 0 {
                    let b = manifest.buffer("completion_logprobs")?;
                    Some(Readback::new(
                        &model, b.address + index * b.nbytes / 2, b.nbytes / 2,
                    )?)
                } else {
                    None
                },
                prompt_logprobs: if manifest.prompt_logprob_capacity()? != 0 {
                    let b = manifest.buffer("prompt_logprobs")?;
                    Some(Readback::new(
                        &model, b.address + index * b.nbytes / 2, b.nbytes / 2,
                    )?)
                } else {
                    None
                },
                confidence: manifest
                    .graphs
                    .iter()
                    .any(|graph| graph.confidence_width > 0)
                    .then(|| {
                        let buffer = manifest.buffer("proposal_confidence")?;
                        Readback::new(&model, buffer.address + index * buffer.nbytes / 2, buffer.nbytes / 2)
                    })
                    .transpose()?,
                survival: manifest.buffers.iter().find(|buffer| buffer.name == "completion_speculative_survival")
                    .map(|buffer| Readback::new(&model, buffer.address + index * buffer.nbytes / 2, buffer.nbytes / 2))
                    .transpose()?,
                readiness: manifest.buffers.iter().find(|buffer| buffer.name == "readiness_timestamps")
                    .map(|buffer| Readback::new(&model, buffer.address + index * buffer.nbytes / 2, buffer.nbytes / 2))
                    .transpose()?,
                produced: model.event_with_timing(true)?,
                started: model.event_with_timing(true)?,
                transferred: copy.event()?,
                pending: None,
            })
        };
        let frames = [frame()?, frame()?];
        let graph_count = manifest.graphs.len();
        #[cfg(target_os = "linux")]
        let park_mode = if model.is_integrated(manifest.device)? {
            #[cfg(target_arch = "aarch64")]
            {
                if host_wait::wfxt_available() {
                    let word = copy.mapped_host_buffer(64)?;
                    let device = copy.mapped_address(&word)?;
                    ParkMode::Wfet(CompletionWake { word, device })
                } else {
                    let wfe_wake = if host_wait::evtstrm_available() {
                        let word = copy.mapped_host_buffer(64)?;
                        let device = copy.mapped_address(&word)?;
                        Some(CompletionWake { word, device })
                    } else {
                        None
                    };
                    ParkMode::Predictive { wfe_wake }
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                ParkMode::Predictive {}
            }
        } else {
            ParkMode::Spin
        };
        #[cfg(not(target_os = "linux"))]
        let park_mode = ParkMode::Spin;
        #[cfg(target_os = "linux")]
        let graph_duration_ns = if matches!(park_mode, ParkMode::Predictive { .. }) {
            vec![0; graph_count]
        } else {
            Vec::new()
        };
        #[cfg(not(target_os = "linux"))]
        let graph_duration_ns = {
            let _ = graph_count;
            Vec::new()
        };
        let engram = manifest
            .engram
            .clone()
            .map(|binding| crate::engram::Transfers::new(binding, &manifest))
            .transpose()?;
        Ok(Self {
            manifest,
            copy_logprobs: true,
            resources: Some(Resources {
                model,
                copy,
                frames,
                engram,
                input_copies: InputCopyBatches::new(upload_capacity.saturating_add(2)),
                park_mode,
            }),
            resolved_uploads: Vec::with_capacity(upload_capacity),
            startup_next: 0,
            startup_ready: false,
            next_sequence: 1,
            completed: Fence(0),
            graph_duration_ns,
            last_completion_at: None,
            poison: None,
            shutdown: None,
            _thread_bound: PhantomData,
        })
    }

    pub(super) fn run_tuning_graph(&mut self, index: usize) -> Result<()> {
        let order = self.manifest.referenced_tuning_graphs();
        if self.startup_ready || order.get(self.startup_next) != Some(&index) {
            return Err("startup tuning graph order differs from the manifest".into());
        }
        let graph = &self.manifest.tuning_graphs[index];
        let status = self.manifest.buffer(&graph.status_buffer)?;
        let resources = self.resources.as_ref().ok_or("executor is closed")?;
        let mut reset = resources.model.host_buffer(4)?;
        reset.bytes_mut().copy_from_slice(&(-1i32).to_ne_bytes());
        let readback = resources.model.host_buffer(4)?;
        let completed = resources.model.event()?;
        let started = Instant::now();
        let result = (|| -> Result<()> {
            // Pinned buffers remain alive until the event confirms all transfers.
            unsafe { resources.model.upload(status.address, &reset, 0, 4)?; }
            resources.model.launch(graph.handle)?;
            unsafe { resources.model.download(&readback, status.address, 4)?; }
            resources.model.record(&completed)?;
            while !completed.completed()? {
                if started.elapsed() > std::time::Duration::from_secs(120) {
                    return Err("startup tuning graph completion deadline expired".into());
                }
                if let Some(health) = &self.manifest.collectives { (health.0)()?; }
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
            Ok(())
        })();
        if let Err(error) = result {
            // Record after any accepted commands so even a failed enqueue can drain.
            let drained = resources.model.record(&completed).and_then(|()| {
                loop {
                    if completed.completed()? { return Ok(()); }
                    if started.elapsed() > std::time::Duration::from_secs(120) {
                        return Err("startup CUDA drain deadline expired".into());
                    }
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
            });
            if drained.is_err() {
                // Constructor failure cannot return an undrained ownership lease.
                eprintln!("startup tuning failed before CUDA drain: {error}; terminating the CUDA process");
                std::process::abort();
            }
            return Err(error);
        }
        let status = i32::from_ne_bytes(readback.bytes().try_into().unwrap());
        if status != 0 {
            return Err(format!("startup tuning graph {:?} failed with status {status}", graph.name).into());
        }
        self.startup_next += 1;
        Ok(())
    }

    pub(super) fn finish_startup(&mut self) -> Result<()> {
        if self.startup_next != self.manifest.referenced_tuning_graphs().len() {
            return Err("startup tuning graphs have not completed".into());
        }
        if let Some(worker) = self.manifest.persistent_workers.first() {
            let control = worker.control.as_ref().ok_or_else(|| {
                Error::execution(
                    ExecutionError::Manifest,
                    "persistent worker requires green-context runtime control (export a core and serve with flare run)",
                )
            })?;
            control.start()?;
        }
        self.startup_ready = true;
        Ok(())
    }

    pub fn manifest(&self) -> &ExecutionManifest {
        &self.manifest
    }

    /// Lock-free telemetry snapshot from the manifest's persistent worker, when present.
    pub fn persistent_worker_snapshot(&self) -> Option<super::PersistentWorkerSnapshot> {
        self.manifest
            .persistent_workers
            .first()
            .and_then(super::PersistentWorkerDescriptor::snapshot)
    }
    pub fn completed_fence(&self) -> Fence {
        self.completed
    }
    /// Fence to record on resources before the next successful submission attempt.
    /// A failed CUDA enqueue poisons the executor; shutdown must drain before reuse.
    pub fn next_fence(&self) -> Result<Fence> {
        if let Some(error) = &self.poison {
            return Err(ExecutionError::Poisoned(Failure::caused_by(
                "executor is closed or failed",
                error.clone(),
            ))
            .into());
        }
        if self.resources.is_none() {
            return Err(Error::execution(
                ExecutionError::Closed,
                "executor is closed or failed",
            ));
        }
        if self.next_sequence > i64::MAX as u64 {
            return Err(Error::execution(
                ExecutionError::Capacity,
                "submission sequence exhausted",
            ));
        }
        Ok(Fence(self.next_sequence))
    }
    pub fn can_submit(&self, grammar_constrained: bool) -> bool {
        let Some(resources) = &self.resources else {
            return false;
        };
        self.startup_ready && self.poison.is_none()
            && resources.frames[self.next_sequence as usize % 2].pending.is_none()
            && !resources
                .frames
                .iter()
                .filter_map(|frame| frame.pending.as_ref())
                .any(|pending| pending.grammar)
            && (!grammar_constrained
                || resources.frames.iter().all(|frame| frame.pending.is_none()))
    }
    /// On integrated Linux SoCs, wait briefly for the oldest pending frame so the CPU does not
    /// burn shared SoC power in a `sched_yield` spin loop. Discrete GPUs and non-Linux hosts
    /// return `false` immediately to preserve minimum-latency spin-yield polling.
    pub fn park(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            let Some(resources) = self.resources.as_ref() else { return false };
            if matches!(resources.park_mode, ParkMode::Spin) {
                return false;
            }
            let Some(pending) = resources
                .frames
                .iter()
                .filter_map(|frame| frame.pending.as_ref())
                .min_by_key(|pending| pending.fence)
            else {
                return false;
            };
            match &resources.park_mode {
                ParkMode::Spin => false,
                #[cfg(target_arch = "aarch64")]
                ParkMode::Wfet(wake) => {
                    let word = wake.word.address() as *const u32;
                    // SAFETY: The mapped word is live, aligned and written only by the copy stream.
                    let seen = unsafe { std::sync::atomic::AtomicU32::from_ptr(word as *mut u32) }
                        .load(std::sync::atomic::Ordering::Acquire);
                    if (seen.wrapping_sub(pending.fence.0 as u32) as i32) >= 0 {
                        return false;
                    }
                    let deadline = host_wait::ticks()
                        + host_wait::frequency() * WFET_PARK_LIMIT_US / 1_000_000;
                    // SAFETY: `Wfet` is constructed only when HWCAP2_WFXT is available.
                    unsafe { host_wait::wait_change_wfet(word, seen, deadline) };
                    true
                }
                ParkMode::Predictive {
                    #[cfg(target_arch = "aarch64")]
                    wfe_wake,
                } => {
                    // For a frame pipelined behind an earlier frame, GPU execution begins once
                    // the earlier frame finishes rather than when this frame was queued.
                    let gpu_start = match self.last_completion_at {
                        Some(completed) if completed > pending.dispatched => completed,
                        _ => pending.dispatched,
                    };
                    let elapsed_us = gpu_start.elapsed().as_micros().min(u64::MAX as u128) as u64;
                    let expected_us = self
                        .graph_duration_ns
                        .get(pending.graph_index)
                        .copied()
                        .unwrap_or(0)
                        / 1_000;
                    match decide_park(elapsed_us, expected_us) {
                        ParkAction::Sleep(duration) => {
                            std::thread::sleep(duration);
                            true
                        }
                        ParkAction::Land => {
                            #[cfg(target_arch = "aarch64")]
                            if let Some(wake) = wfe_wake.as_ref() {
                                let word = wake.word.address() as *const u32;
                                // SAFETY: The mapped word is live, aligned and written only by the copy stream.
                                let seen = unsafe {
                                    std::sync::atomic::AtomicU32::from_ptr(word as *mut u32)
                                }
                                .load(std::sync::atomic::Ordering::Acquire);
                                if (seen.wrapping_sub(pending.fence.0 as u32) as i32) >= 0 {
                                    return false;
                                }
                                // SAFETY: `wfe_wake` is constructed only when HWCAP_EVTSTRM is available.
                                unsafe { host_wait::wait_change_wfe(word, seen) };
                                return true;
                            }
                            false
                        }
                    }
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = &self.resources;
            false
        }
    }

    pub fn outstanding(&self) -> usize {
        self.resources.as_ref().map_or(0, |resources| {
            resources
                .frames
                .iter()
                .filter(|frame| frame.pending.is_some())
                .count()
        })
    }

    /// Validate a coordinated submission without touching CUDA or advancing its fence.
    pub(super) fn validate_submission(
        &mut self,
        graph: usize,
        uploads: &[Upload<'_>],
        grammar: bool,
    ) -> Result<Option<Fence>> {
        let fence = self.next_fence()?;
        validate_uploads(&self.manifest, graph, uploads, &mut self.resolved_uploads)?;
        if !self.can_submit(grammar || !matches!(self.manifest.graphs[graph].kind,
            GraphKind::Decode | GraphKind::Speculative)) {
            return Ok(None);
        }
        if crate::engram::consumes(self.manifest.graphs[graph].kind) {
            if let Some(engram) = self.resources.as_ref().and_then(|resources| resources.engram.as_ref()) {
                engram.validate(fence.0, self.manifest.graphs[graph].packed_tokens(), uploads)?;
            }
        }
        Ok(Some(fence))
    }

    /// Retain allocations after a peer failure that may strand a GPU collective.
    pub(super) fn abandon(&mut self, error: Error) -> Shutdown {
        if let Some(resources) = self.resources.take() {
            std::mem::forget(resources);
        }
        let shutdown = Shutdown {
            drained: false,
            error: Some(error),
        };
        self.shutdown = Some(shutdown.clone());
        shutdown
    }

    /// Returns None on frame or grammar backpressure without changing CUDA state.
    pub fn submit(
        &mut self,
        graph_index: usize,
        uploads: &[Upload<'_>],
        grammar_constrained: bool,
    ) -> Result<Option<Fence>> {
        if let Some(error) = &self.poison {
            return Err(ExecutionError::Poisoned(Failure::caused_by(
                format!("executor failed: {error}"),
                error.clone(),
            ))
            .into());
        }
        if !self.startup_ready {
            return Err(Error::execution(
                ExecutionError::State,
                "startup tuning has not completed",
            ));
        }
        let resources = self
            .resources
            .as_mut()
            .ok_or_else(|| Error::execution(ExecutionError::Closed, "executor is closed"))?;
        let graph =
            self.manifest.graphs.get(graph_index).ok_or_else(|| {
                Error::execution(ExecutionError::State, "graph index out of range")
            })?;
        let grammar_constrained = grammar_constrained
            || !matches!(graph.kind, GraphKind::Decode | GraphKind::Speculative);
        let mut resolved = validate_uploads(
            &self.manifest,
            graph_index,
            uploads,
            &mut self.resolved_uploads,
        )?;
        if (resources.frames.iter().any(|frame| frame.pending.is_some()) && grammar_constrained)
            || resources
                .frames
                .iter()
                .filter_map(|frame| frame.pending.as_ref())
                .any(|pending| pending.grammar)
        {
            return Ok(None);
        }
        let index = self.next_sequence as usize % 2;
        if resources.frames[index].pending.is_some() {
            return Ok(None);
        }
        if self.next_sequence > i64::MAX as u64 {
            return Err(Error::execution(
                ExecutionError::Capacity,
                "submission sequence exhausted",
            ));
        }
        if crate::engram::consumes(graph.kind) {
            if let Some(engram) = &resources.engram {
                engram.validate(self.next_sequence, graph.packed_tokens(), uploads)?;
            }
        }
        let fence = Fence(self.next_sequence);
        let queued = resources.frames.iter().any(|frame| frame.pending.is_some());
        let frame = &mut resources.frames[index];
        let dispatched = Instant::now();
        resolved.stage(frame.input.bytes_mut(), index as i32, fence.0 as i64);
        let logprobs = if self.copy_logprobs { resolved.logprobs } else { 0 };
        let logprob_width = resolved.logprob_width;
        let logprob_token_ids = if self.copy_logprobs {
            std::mem::take(&mut resolved.logprob_token_ids)
        } else {
            Vec::new()
        };
        let prompt_rows = if self.copy_logprobs && graph.kind == GraphKind::ScoredPrefill {
            let bytes = uploads.iter().filter(|u| u.buffer == "prompt_target_tokens")
                .map(|u| u.offset + u.bytes.len()).max().unwrap_or(0);
            if bytes == 0 || bytes % 4 != 0 || bytes / 4 > graph.token_capacity {
                return Err("scored prefill requires one target per live prompt row".into());
            }
            bytes / 4
        } else { 0 };
        frame.pending = Some(Pending {
            logprobs,
            logprob_width,
            logprob_token_ids,
            prompt_rows,
            fence,
            graph_index,
            grammar: grammar_constrained,
            dispatched,
            queued,
        });
        self.next_sequence += 1;
        let mut operation = || -> Result<()> {
            // Validated destinations are disjoint, so copies within a batch do
            // not depend on each other. Omitted input ranges remain untouched.
            resources.input_copies.submit(frame.input.address(), resolved.copies(), |dst, src, sizes| {
                // SAFETY: The manifest and pinned frame retain every validated
                // range. Pending frames remain immutable until completion, and
                // enqueue failures retain their owners until both streams drain.
                // Descriptor slices are consumed synchronously by the driver.
                unsafe { resources.model.copy_batch(dst, src, sizes) }
            })?;
            resources.model.record(&frame.started)?;
            if crate::engram::consumes(graph.kind) {
                if let Some(engram) = &resources.engram {
                    engram.enqueue(fence.0, graph.packed_tokens(), uploads, &frame.started)?;
                }
            }
            resources.model.launch(graph.handle)?;
            resources.model.record(&frame.produced)?;
            resources.copy.wait(&frame.produced)?;
            let address = self.manifest.buffer("completion_records")?.address
                + index * self.manifest.completion_bytes();
            // SAFETY: The manifest bounds both completion rows; index selects one
            // row in that retained allocation. The copy stream waits for the model
            // event, and the pinned output is not read or reused before transferred
            // completes. Shutdown drains both streams before releasing either owner.
            unsafe {
                resources.copy.download(
                    &frame.output,
                    address,
                    self.manifest.completion_bytes(),
                )?;
                if logprobs != 0 {
                    let b = self.manifest.buffer("completion_logprobs")?;
                    frame.logprobs.as_ref().unwrap().enqueue_pitched_copy(
                        &resources.copy, b.address + index * b.nbytes / 2,
                        b.shape[3] * 4, (2 + 2 * logprob_width) * 4,
                        self.manifest.request_capacity() * self.manifest.completion_token_capacity,
                    )?;
                }
                if prompt_rows != 0 {
                    let b = self.manifest.buffer("prompt_logprobs")?;
                    frame.prompt_logprobs.as_ref().unwrap().enqueue_copy(&resources.copy,
                        b.address + index * b.nbytes / 2, prompt_rows * super::PROMPT_FIELDS * 4)?;
                }
                if graph.confidence_width > 0 {
                    let confidence = self.manifest.buffer("proposal_confidence")?;
                    frame.confidence.as_ref().unwrap().enqueue_copy(
                        &resources.copy, confidence.address + index * confidence.nbytes / 2, confidence.nbytes / 2,
                    )?;
                }
                if reads_speculative_survival(graph) {
                    if let Some(readback) = &frame.survival {
                        let buffer = self.manifest.buffer("completion_speculative_survival")?;
                        readback.enqueue_copy(&resources.copy, buffer.address + index * buffer.nbytes / 2,
                                              buffer.nbytes / 2)?;
                    }
                }
                if crate::engram::consumes(graph.kind) {
                    if let Some(readback) = &frame.readiness {
                        let buffer = self.manifest.buffer("readiness_timestamps")?;
                        readback.enqueue_copy(&resources.copy, buffer.address + index * buffer.nbytes / 2,
                                              buffer.nbytes / 2)?;
                    }
                }
            }
            resources.copy.record(&frame.transferred)?;
            if let Some(wake) = resources.park_mode.wake() {
                // SAFETY: Resources retain the mapped word until close drains both streams.
                unsafe { resources.copy.write_value(wake.device, fence.0 as u32) }?;
            }
            Ok(())
        };
        if let Err(error) = operation() {
            self.poison = Some(error.clone());
            return Err(error);
        }
        Ok(Some(fence))
    }

    pub fn poll(&mut self) -> Result<Option<FrameCompletion>> {
        if let Some(error) = &self.poison {
            return Err(ExecutionError::Poisoned(Failure::caused_by(
                format!("executor failed: {error}"),
                error.clone(),
            ))
            .into());
        }
        let resources = self
            .resources
            .as_mut()
            .ok_or_else(|| Error::execution(ExecutionError::Closed, "executor is closed"))?;
        let frame = resources
            .frames
            .iter_mut()
            .filter(|frame| frame.pending.is_some())
            .min_by_key(|frame| frame.pending.as_ref().unwrap().fence);
        let Some(frame) = frame else {
            return Ok(None);
        };
        match frame.transferred.completed() {
            Ok(false) => return Ok(None),
            Err(error) => {
                self.poison = Some(error.clone());
                return Err(error);
            }
            Ok(true) => {}
        }
        if let Some(health) = &self.manifest.collectives {
            if let Err(error) = (health.0)() {
                self.poison = Some(error.clone());
                return Err(error);
            }
        }
        let pending = frame.pending.take().unwrap();
        if crate::engram::consumes(self.manifest.graphs[pending.graph_index].kind) {
            if let Some(engram) = &resources.engram {
                if let Err(error) = engram.finish(pending.fence.0) {
                    self.poison = Some(error.clone());
                    return Err(error);
                }
            }
        }
        self.completed = pending.fence;
        let records = match decode_records(
            frame.output.bytes(),
            pending.fence,
            self.manifest.request_capacity(),
            self.manifest.completion_fields(),
        ) {
            Ok(records) => records,
            Err(error) => {
                self.poison = Some(error.clone());
                return Err(error);
            }
        };
        let logprobs = if pending.logprobs != 0 {
            // SAFETY: This frame's transfer event covers its sidecar; the source
            // and pinned destination remain owned until decoding completes.
            let readback = frame.logprobs.as_ref().unwrap();
            let fields = match readback {
                Readback::MappedHost { .. } => self.manifest.buffer("completion_logprobs")?.shape[3],
                Readback::DeviceCopy(_) => 2 + 2 * pending.logprob_width,
            };
            let bytes = unsafe { readback.bytes() };
            let nbytes = records.len() * self.manifest.completion_token_capacity * fields * size_of::<f32>();
            match decode_logprobs(
                &bytes[..nbytes],
                &records,
                self.manifest.max_top_logprobs()?,
                fields,
                pending.logprobs,
                &pending.logprob_token_ids,
            ) {
                Ok(values) => Some(values),
                Err(error) => {
                    self.poison = Some(error.clone());
                    return Err(error);
                }
            }
        } else {
            None
        };
        let prompt_logprobs = if pending.prompt_rows != 0 && !records.iter().any(|r| r.status == 3) {
            // SAFETY: The transfer event covers this frame; decoding precedes its release.
            let bytes = unsafe { frame.prompt_logprobs.as_ref().unwrap().bytes() };
            match decode_prompt_logprobs(bytes, pending.prompt_rows) {
                Ok(values) => Some(values),
                Err(error) => { self.poison = Some(error.clone()); return Err(error); }
            }
        } else { None };
        let elapsed_ns = match frame.produced.elapsed_ns(&frame.started) {
            Ok(value) => value,
            Err(error) => {
                self.poison = Some(error.clone());
                return Err(error);
            }
        };
        #[cfg(target_os = "linux")]
        if let Some(slot) = self.graph_duration_ns.get_mut(pending.graph_index) {
            self.last_completion_at = Some(Instant::now());
            *slot = update_duration_estimate(*slot, elapsed_ns);
        }
        let readiness = if crate::engram::consumes(self.manifest.graphs[pending.graph_index].kind) {
            frame.readiness.as_ref().map(|readback| {
                // SAFETY: The frame's transfer event completed before decoding its retained readback.
                ReadinessIntervals::decode(unsafe { readback.bytes() })
            }).transpose()
        } else { Ok(None) };
        let readiness = match readiness {
            Ok(value) if value.is_none_or(|sample| sample.wait_ns() <= elapsed_ns) => value,
            value => {
                let error = value.err().unwrap_or_else(|| Error::execution(ExecutionError::Completion,
                    "readiness delay exceeds its enclosing CUDA interval"));
                self.poison = Some(error.clone());
                return Err(error);
            }
        };
        let speculative_survival = if reads_speculative_survival(&self.manifest.graphs[pending.graph_index]) {
            frame.survival.as_ref().map(|readback| {
                let mut rows = vec![vec![-1.0; self.manifest.completion_token_capacity - 1]; self.manifest.request_capacity()];
                // SAFETY: The completion transfer protects this frame's retained feedback bytes.
                for (row, bytes) in rows.iter_mut().zip(unsafe { readback.bytes() }.chunks_exact((self.manifest.completion_token_capacity - 1) * 4)) {
                    for (value, bytes) in row.iter_mut().zip(bytes.chunks_exact(4)) {
                        *value = f32::from_ne_bytes(bytes.try_into().unwrap());
                    }
                }
                rows
            })
        } else { None };
        if let Some(rows) = &speculative_survival {
            if let Err(error) = validate_speculative_survival(rows, &records) {
                self.poison = Some(error.clone());
                return Err(error);
            }
        }
        Ok(Some(FrameCompletion {
            fence: pending.fence,
            graph_index: pending.graph_index,
            timing: ExecutionTiming {
                readiness,
                gpu_ns: [
                    Some(elapsed_ns),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                wall_ns: pending
                    .dispatched
                    .elapsed()
                    .as_nanos()
                    .min(u64::MAX as u128) as u64,
                queued: pending.queued,
            },
            records,
            logprobs,
            prompt_logprobs,
            speculative_survival,
            proposal_confidence: if self.manifest.graphs[pending.graph_index].confidence_width > 0
            {
                let mut confidence = vec![vec![0.0; self.manifest.completion_token_capacity - 1]; self.manifest.request_capacity()];
                let graph = &self.manifest.graphs[pending.graph_index];
                for (row, bytes) in confidence
                    .iter_mut()
                    .take(graph.batch_capacity)
                    // SAFETY: The transfer event follows the producing graph;
                    // this frame remains reserved while its sidecar is decoded.
                    .zip(unsafe { frame.confidence.as_ref().unwrap().bytes() }
                        .chunks_exact(4 * self.manifest.buffer("proposal_confidence")?.shape[2]))
                {
                    for (value, bytes) in row.iter_mut().take(graph.confidence_width).zip(bytes.chunks_exact(4)) {
                        *value = f32::from_ne_bytes(bytes.try_into().unwrap());
                    }
                }
                Some(confidence)
            } else {
                None
            },
        }))
    }

    pub fn close(&mut self) -> Shutdown {
        if let Some(shutdown) = &self.shutdown {
            return shutdown.clone();
        }
        let resources = self.resources.take().unwrap();
        if let Some(engram) = &resources.engram {
            engram.cancel();
        }
        let compute = resources.model.synchronize().map_err(Error::drain);
        let copy = resources.copy.synchronize().map_err(Error::drain);
        let engram = resources
            .engram
            .as_ref()
            .map_or(Ok(()), |e| e.drain())
            .map_err(Error::drain);
        let worker = self
            .manifest
            .persistent_workers
            .first()
            .and_then(|worker| worker.control.as_ref())
            .map_or(Ok(()), |control| control.stop())
            .map_err(Error::drain);
        let drained = compute.is_ok() && copy.is_ok() && engram.is_ok() && worker.is_ok();
        let error = compute
            .err()
            .or(copy.err())
            .or(engram.err())
            .or(worker.err())
            .or(self.poison.take());
        if drained {
            drop(resources);
        } else {
            std::mem::forget(resources);
        }
        let shutdown = Shutdown { drained, error };
        self.shutdown = Some(shutdown.clone());
        shutdown
    }
}

impl Drop for GraphExecutor {
    fn drop(&mut self) {
        self.close();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ResolvedUpload {
    pub(super) host_offset: usize,
    pub(super) device_address: usize,
    pub(super) nbytes: usize,
}

/// Borrowed descriptors are consumed synchronously; CUDA retains only frame bytes.
pub(super) struct ResolvedUploads<'uploads, 'bytes, 'storage> {
    uploads: &'uploads [Upload<'bytes>],
    bindings: &'storage [ResolvedUpload],
    frame_index: ResolvedUpload,
    submission_sequence: ResolvedUpload,
    logprobs: u64,
    logprob_width: usize,
    logprob_token_ids: Vec<Vec<u32>>,
}

impl ResolvedUploads<'_, '_, '_> {
    pub(super) fn stage(&self, frame: &mut [u8], index: i32, sequence: i64) {
        for (upload, binding) in self.uploads.iter().zip(self.bindings) {
            frame[binding.host_offset..binding.host_offset + binding.nbytes]
                .copy_from_slice(upload.bytes);
        }
        for (binding, bytes) in [
            (self.frame_index, index.to_le_bytes().as_slice()),
            (self.submission_sequence, sequence.to_le_bytes().as_slice()),
        ] {
            frame[binding.host_offset..binding.host_offset + binding.nbytes].copy_from_slice(bytes);
        }
    }

    pub(super) fn copies(&self) -> impl Iterator<Item = ResolvedUpload> + '_ {
        self.bindings
            .iter()
            .copied()
            .chain([self.frame_index, self.submission_sequence])
    }
}

/// Resolve one command against an already validated, immutable manifest.
/// Omitted inputs and disjoint partial uploads preserve resident device ranges.
/// Storage grows only to the fragment-count high-water; valid nonempty disjoint
/// fragments are bounded by input_frame_bytes. The executor reserves four uploads
/// per binding for physical request lanes before accepting any submissions.
pub(super) fn validate_uploads<'uploads, 'bytes, 'storage>(
    manifest: &ExecutionManifest,
    graph_index: usize,
    uploads: &'uploads [Upload<'bytes>],
    storage: &'storage mut Vec<ResolvedUpload>,
) -> Result<ResolvedUploads<'uploads, 'bytes, 'storage>> {
    storage.clear();
    let graph = manifest
        .graphs
        .get(graph_index)
        .ok_or_else(|| Error::execution(ExecutionError::Upload, "graph index out of range"))?;
    if graph.kind == GraphKind::ScoredPrefill {
        let bytes = uploads.iter().filter(|upload| upload.buffer == "prompt_target_tokens")
            .map(|upload| upload.offset.saturating_add(upload.bytes.len())).max().unwrap_or(0);
        if bytes == 0 || bytes % 4 != 0 || bytes / 4 > graph.token_capacity {
            return Err("scored prefill requires one target per live prompt row".into());
        }
    }
    if uploads.len() > manifest.input_frame_bytes {
        return Err(Error::execution(
            ExecutionError::Upload,
            "upload count exceeds the nonempty input-fragment bound",
        ));
    }
    let reserved = |name: &str| -> Result<ResolvedUpload> {
        let binding = graph
            .input_bindings
            .iter()
            .find(|binding| binding.buffer_name == name)
            .ok_or_else(|| {
                Error::execution(
                    ExecutionError::Upload,
                    "graph lacks a reserved input binding",
                )
            })?;
        let buffer = manifest.buffer(name)?;
        Ok(ResolvedUpload {
            host_offset: binding.host_offset,
            device_address: buffer.address,
            nbytes: buffer.nbytes,
        })
    };
    let frame_index = reserved("frame_index")?;
    let submission_sequence = reserved("submission_sequence")?;
    for upload in uploads {
        if ["frame_index", "submission_sequence"].contains(&upload.buffer) {
            return Err(Error::execution(
                ExecutionError::Upload,
                "reserved executor inputs cannot be overridden",
            ));
        }
        let binding = graph
            .input_bindings
            .iter()
            .find(|binding| binding.buffer_name == upload.buffer)
            .ok_or_else(|| {
                Error::execution(ExecutionError::Upload, "upload is not bound by this graph")
            })?;
        let buffer = manifest.buffer(upload.buffer)?;
        if upload.bytes.is_empty()
            || upload.offset % buffer.dtype.size() != 0
            || upload.bytes.len() % buffer.dtype.size() != 0
            || upload
                .offset
                .checked_add(upload.bytes.len())
                .is_none_or(|end| end > binding.nbytes)
        {
            return Err(Error::execution(
                ExecutionError::Upload,
                format!("unaligned or out-of-range upload: {}", upload.buffer),
            ));
        }
        let resolved = ResolvedUpload {
            host_offset: binding.host_offset + upload.offset,
            device_address: buffer.address + upload.offset,
            nbytes: upload.bytes.len(),
        };
        // Manifest validation makes different buffers' host spans disjoint.
        if storage.iter().any(|prior| {
            prior.host_offset < resolved.host_offset + resolved.nbytes
                && resolved.host_offset < prior.host_offset + prior.nbytes
        }) {
            return Err(Error::execution(
                ExecutionError::Upload,
                "uploads overlap within one input buffer",
            ));
        }
        storage.push(resolved);
    }
    let mut logprobs = 0u64;
    if graph.kind.samples() {
        for upload in uploads.iter().filter(|u| u.buffer == "logprob_requested") {
            for (i, bytes) in upload.bytes.chunks_exact(size_of::<i32>()).enumerate() {
                if i32::from_ne_bytes(bytes.try_into().unwrap()) != 0 {
                    logprobs |= 1 << (upload.offset / size_of::<i32>() + i);
                }
            }
        }
    }
    let mut logprob_width = if logprobs != 0 {
        (manifest.buffer("completion_logprobs")?.shape[3] / 2 - 1).min(super::MAX_TOP_LOGPROBS)
    } else {
        0
    };
    let mut logprob_token_ids = Vec::new();
    if graph.kind.samples() && manifest.logprob_token_id_capacity() != 0 {
        let capacity = manifest.request_capacity();
        if !uploads.iter().any(|u| {
            u.buffer == "logprob_requested" && u.offset == 0 && u.bytes.len() == capacity * 4
        }) {
            return Err(
                "sampling requires all generation logprob gates on every submission".into(),
            );
        }
        let counts = uploads
            .iter()
            .find(|u| {
                u.buffer == "logprob_token_id_counts"
                    && u.offset == 0
                    && u.bytes.len() == capacity * 4
            })
            .ok_or("sampling requires all logprob token ID counts on every submission")?;
        for (slot, count) in counts.bytes.chunks_exact(4).enumerate() {
            let count = i32::from_ne_bytes(count.try_into().unwrap());
            if !(0..=super::MAX_LOGPROB_TOKEN_IDS as i32).contains(&count)
                || (count != 0 && logprobs & (1 << slot) == 0)
            {
                return Err("invalid logprob token ID count or request gate".into());
            }
            if count == 0 {
                continue;
            }
            if logprob_token_ids.is_empty() {
                logprob_token_ids.resize_with(capacity, Vec::new);
            }
            let ids = uploads
                .iter()
                .find(|u| {
                    u.buffer == "logprob_token_ids"
                        && u.offset == slot * super::MAX_LOGPROB_TOKEN_IDS * 4
                        && u.bytes.len() == count as usize * 4
                })
                .ok_or("missing requested logprob token ID prefix")?;
            for id in ids.bytes.chunks_exact(4) {
                let id = i32::from_ne_bytes(id.try_into().unwrap());
                if id < 0 || logprob_token_ids[slot].contains(&(id as u32)) {
                    return Err("invalid or duplicate requested logprob token ID".into());
                }
                logprob_token_ids[slot].push(id as u32);
            }
            logprob_width = logprob_width.max(count as usize);
        }
    }
    Ok(ResolvedUploads {
        uploads,
        bindings: storage,
        frame_index,
        submission_sequence,
        logprobs,
        logprob_width,
        logprob_token_ids,
    })
}

pub(super) fn decode_records(
    bytes: &[u8],
    fence: Fence,
    capacity: usize,
    fields: usize,
) -> Result<CompletionRecords> {
    if !(1..=REQUEST_LANES).contains(&capacity)
        || fields <= COMPLETION_METADATA_FIELDS
        || capacity.checked_mul(fields).and_then(|n| n.checked_mul(8)) != Some(bytes.len())
    {
        return Err(Error::execution(
            ExecutionError::Completion,
            "completion extent mismatch",
        ));
    }
    let mut records = CompletionRecords {
        records: vec![CompletionRecord::default(); capacity],
    };
    for (record, bytes) in records.iter_mut().zip(bytes.chunks_exact(fields * 8)) {
        let mut fields = vec![0i64; fields];
        for (field, bytes) in fields.iter_mut().zip(bytes.chunks_exact(8)) {
            *field = i64::from_le_bytes(bytes.try_into().unwrap());
        }
        *record = CompletionRecord {
            generation: fields[0],
            submission_sequence: fields[1],
            consumed_tokens: fields[2],
            generated_tokens: fields[3],
            token_id: fields[4],
            status: fields[5],
            error_code: fields[6],
            reserved: fields[7],
            vector: Some(TokenVector {
                emitted_count: fields[8],
                offered_drafts: fields[9],
                accepted_drafts: fields[10],
                copy_offered_mask: fields[11],
                model_reserved: fields[12],
                draft_cache_consumed: fields[13],
                verification_width: fields[15],
                tokens: fields[COMPLETION_METADATA_FIELDS..].to_vec(),
            }),
        };
        if record.status != 0 && fields[14] != 0 {
            return Err(Error::execution(
                ExecutionError::Completion,
                "reserved vector completion field must be zero",
            ));
        }
        record.validate_vector()?;
        if !(0..=3).contains(&record.status)
            || record.reserved != 0
            || (record.status != 0
                && (record.generation <= 0
                    || record.submission_sequence != fence.0 as i64
                    || record.consumed_tokens < 0
                    || record.generated_tokens < 0
                    || record.token_id < -1
                    || (record.status != 3 && record.error_code != 0)))
        {
            return Err(Error::execution(
                ExecutionError::Completion,
                "invalid or stale device completion record",
            ));
        }
    }
    Ok(records)
}

pub(super) fn decode_logprobs(
    bytes: &[u8],
    records: &[CompletionRecord],
    k: usize,
    fields: usize,
    requested: u64,
    selections: &[Vec<u32>],
) -> Result<Vec<Vec<Option<TokenLogprobs>>>> {
    let token_capacity = records.first().and_then(|r| r.vector.as_ref()).map_or(DEFAULT_COMPLETION_TOKEN_CAPACITY, |v| v.tokens.len());
    if !(1..=super::MAX_TOP_LOGPROBS).contains(&k)
        || fields < 2 + 2 * k
        || fields > super::LOGPROB_FIELDS
        || fields % 2 != 0
        || bytes.len() != records.len() * token_capacity * fields * size_of::<i32>()
    {
        return Err(Error::execution(
            ExecutionError::Completion,
            "generation logprob frame extent mismatch",
        ));
    }
    let mut result: Vec<Vec<Option<TokenLogprobs>>> = (0..records.len())
        .map(|_| vec![None; token_capacity])
        .collect();
    for (slot, record) in records
        .iter()
        .enumerate()
        .filter(|(slot, r)| matches!(r.status, 1 | 2) && requested & (1 << slot) != 0)
    {
        let count = record
            .vector.as_ref()
            .map_or(usize::from(record.token_id >= 0), |v| {
                v.emitted_count as usize
            });
        if count > token_capacity {
            return Err("generation logprob emitted count exceeds the frame".into());
        }
        for position in 0..count {
            let start = (slot * token_capacity + position) * fields * size_of::<i32>();
            let int = |at: usize| {
                let offset = start + at * size_of::<i32>();
                i32::from_ne_bytes(bytes[offset..offset + size_of::<i32>()].try_into().unwrap())
            };
            let float = |at| f32::from_bits(int(at) as u32);
            let expected = record
                .vector.as_ref()
                .map_or(record.token_id, |v| v.tokens[position]);
            if int(0) < 0 || i64::from(int(0)) != expected || float(1).is_nan() {
                return Err("generated token is missing its raw logprob record".into());
            }
            let ids = selections.get(slot).map_or(&[][..], Vec::as_slice);
            let count = if ids.is_empty() { k } else { ids.len() };
            if count > super::MAX_LOGPROB_TOKEN_IDS || 2 + 2 * count > fields {
                return Err("generation logprob selection exceeds the copied width".into());
            }
            let mut value = TokenLogprobs {
                token: int(0) as u32,
                logprob: float(1),
                top: Vec::with_capacity(count + usize::from(!ids.is_empty())),
            };
            for i in 0..count {
                let id = int(super::LOGPROB_PAIR_FIELDS * (1 + i));
                if id < 0 && ids.is_empty() {
                    break;
                }
                if id < 0 || (!ids.is_empty() && id as u32 != ids[i]) {
                    return Err(
                        "generation logprob ID does not match the requested selection".into(),
                    );
                }
                let lp = float(super::LOGPROB_PAIR_FIELDS * (1 + i) + 1);
                if lp.is_nan() {
                    return Err(Error::execution(
                        ExecutionError::Completion,
                        "invalid generation logprob alternative",
                    ));
                }
                if ids.is_empty() && i > 0 {
                    let (previous, p) = value.top[i - 1];
                    if lp > p || (lp == p && id as u32 <= previous) {
                        return Err(Error::execution(
                            ExecutionError::Completion,
                            "generation logprob alternatives are not ordered",
                        ));
                    }
                }
                value.top.push((id as u32, lp));
            }
            if !ids.is_empty() && !ids.contains(&value.token) {
                value.top.push((value.token, value.logprob));
            }
            result[slot][position] = Some(value);
        }
    }
    Ok(result)
}

pub(super) fn decode_prompt_logprobs(bytes: &[u8], rows: usize) -> Result<Vec<Option<PromptLogprobs>>> {
    let fields = super::PROMPT_FIELDS;
    if bytes.len() < rows * fields * 4 {
        return Err("prompt logprob frame extent mismatch".into());
    }
    bytes[..rows * fields * 4].chunks_exact(fields * 4).map(|row| {
        let int = |i: usize| i32::from_ne_bytes(row[i * 4..(i + 1) * 4].try_into().unwrap());
        let float = |i: usize| f32::from_bits(int(i) as u32);
        if int(0) == -1 {
            if int(2) != 0 || !float(1).is_nan() {
                return Err("invalid prompt logprob guard".into());
            }
            return Ok(None);
        }
        if int(0) < 0 || int(2) <= 0 || float(1).is_nan() || float(1) > 0. {
            return Err("invalid prompt logprob result".into());
        }
        let mut value = TokenLogprobs {
            token: int(0) as u32, logprob: float(1),
            top: Vec::with_capacity(super::MAX_TOP_LOGPROBS),
        };
        for i in 0..super::MAX_TOP_LOGPROBS {
            let token = int(3 + 2 * i);
            if token == -1 {
                break;
            }
            let lp = float(4 + 2 * i);
            if token < 0 || lp.is_nan() || lp > 0. {
                return Err("invalid prompt alternative".into());
            }
            if i > 0 {
                let (previous, previous_lp) = value.top[i - 1];
                if lp > previous_lp || (lp == previous_lp && token as u32 <= previous) {
                    return Err("prompt alternatives are not ordered".into());
                }
            }
            value.top.push((token as u32, lp));
        }
        Ok(Some(PromptLogprobs { value, rank: int(2) as u32 }))
    }).collect()
}

#[cfg(test)]
mod input_copy_tests {
    use super::*;

    #[test]
    fn input_batches_preserve_every_partial_range_across_both_driver_limits() {
        let host = 0x10000;
        let mut copies = vec![ResolvedUpload {
            host_offset: 128, device_address: 0x20000000, nbytes: BATCH_BYTES + 16,
        }];
        copies.extend((0..BATCH_ENTRIES + 3).map(|i| ResolvedUpload {
            host_offset: BATCH_BYTES + 1024 + i * 8,
            device_address: 0x40000000 + i * 16,
            nbytes: 8,
        }));
        let mut batches = InputCopyBatches::new(4);
        let mut recorded = Vec::new();
        batches.submit(host, copies.iter().copied(), |dst, src, sizes| {
            assert!(sizes.len() <= BATCH_ENTRIES);
            assert!(sizes.iter().sum::<usize>() <= BATCH_BYTES);
            assert!(sizes.iter().all(|&size| size > 0));
            recorded.extend(dst.iter().zip(src).zip(sizes).map(|((&d, &s), &n)| (d, s, n)));
            Ok(())
        }).unwrap();
        let mut actual = recorded.into_iter();
        for copy in copies {
            let mut offset = 0;
            while offset < copy.nbytes {
                let (dst, src, nbytes) = actual.next().unwrap();
                assert_eq!(dst, (copy.device_address + offset) as u64);
                assert_eq!(src, (host + copy.host_offset + offset) as u64);
                assert!(nbytes <= copy.nbytes - offset);
                offset += nbytes;
            }
        }
        assert!(actual.next().is_none());
        assert!(batches.sizes.is_empty());
    }

    #[test]
    fn failed_batch_stops_submission_and_does_not_retain_descriptors_for_reuse() {
        let mut batches = InputCopyBatches::new(4);
        let copy = ResolvedUpload { host_offset: 16, device_address: 4096, nbytes: BATCH_BYTES + 8 };
        let mut calls = 0;
        assert!(batches.submit(8192, std::iter::once(copy), |_, _, _| {
            calls += 1;
            Err("copy failed".into())
        }).is_err());
        assert_eq!(calls, 1);
        batches.submit(8192, std::iter::once(ResolvedUpload { nbytes: 8, ..copy }), |dst, src, sizes| {
            assert_eq!(dst, &[4096]);
            assert_eq!(src, &[8208]);
            assert_eq!(sizes, &[8]);
            Ok(())
        }).unwrap();
        batches.submit(8192, std::iter::empty(), |_, _, _| panic!("empty input must not enqueue")).unwrap();
    }
}

#[cfg(all(test, feature = "runtime-core"))]
mod confidence_tests {
    use super::*;
    use crate::runtime_core::driver::{CopyEndpoint, CopyParams, Driver};

    #[test]
    #[ignore = "requires a CUDA device selected through CUDA_VISIBLE_DEVICES"]
    fn mapped_and_device_confidence_survive_repeated_graph_replay() {
        const BYTES: usize = 160;
        let driver = Driver::load().unwrap();
        let mut context = std::ptr::null_mut();
        let (mut source, mut device_output) = (0, 0);
        unsafe {
            driver.check("cuInit", (driver.init)(0)).unwrap();
            let mut device = 0;
            driver
                .check("cuDeviceGet", (driver.device)(&mut device, 0))
                .unwrap();
            driver
                .check(
                    "cuCtxCreate_v2",
                    (driver.context_create)(&mut context, 0, device),
                )
                .unwrap();
            driver
                .check("cuMemAlloc_v2", (driver.allocate)(&mut source, BYTES))
                .unwrap();
            driver
                .check(
                    "cuMemAlloc_v2",
                    (driver.allocate)(&mut device_output, BYTES),
                )
                .unwrap();
        }
        {
            let model = Stream::open(context as usize, 0).unwrap();
            let copy = Stream::open(context as usize, 0).unwrap();
            let mapped = model.mapped_host_buffer(BYTES).unwrap();
            let mapped_address = model.mapped_address(&mapped).unwrap();
            let mut input = model.host_buffer(BYTES).unwrap();
            for address in [mapped_address, device_output as usize] {
                let confidence = Readback::new(&model, address, BYTES).unwrap();
                assert_eq!(
                    matches!(confidence, Readback::MappedHost { .. }),
                    address == mapped_address
                );
                let mut graph = std::ptr::null_mut();
                let mut executable = std::ptr::null_mut();
                unsafe {
                    driver
                        .check("cuGraphCreate", (driver.graph_create)(&mut graph, 0))
                        .unwrap();
                    let endpoint = |address| CopyEndpoint {
                        memory_type: 2,
                        device: address,
                        pitch: BYTES,
                        height: 1,
                        ..CopyEndpoint::default()
                    };
                    let params = CopyParams {
                        source: endpoint(source),
                        destination: endpoint(address as u64),
                        width: BYTES,
                        height: 1,
                        depth: 1,
                    };
                    let mut node = std::ptr::null_mut();
                    driver
                        .check(
                            "cuGraphAddMemcpyNode",
                            (driver.graph_copy)(
                                &mut node,
                                graph,
                                std::ptr::null(),
                                0,
                                &params,
                                context,
                            ),
                        )
                        .unwrap();
                    driver
                        .check(
                            "cuGraphInstantiateWithFlags",
                            (driver.graph_instantiate)(&mut executable, graph, 0),
                        )
                        .unwrap();
                }
                let produced = model.event().unwrap();
                let transferred = copy.event().unwrap();
                for generation in 1..=16 {
                    let values: Vec<u8> = (0..BYTES / 4)
                        .flat_map(|i| ((generation * 100 + i) as f32).to_ne_bytes())
                        .collect();
                    input.bytes_mut().copy_from_slice(&values);
                    unsafe {
                        model.upload(source as usize, &input, 0, BYTES).unwrap();
                    }
                    model.launch(executable as usize).unwrap();
                    model.record(&produced).unwrap();
                    copy.wait(&produced).unwrap();
                    unsafe {
                        confidence.enqueue_copy(&copy, address, BYTES).unwrap();
                    }
                    copy.record(&transferred).unwrap();
                    while !transferred.completed().unwrap() {
                        std::thread::yield_now();
                    }
                    // The transfer fence covers the mapped graph output as well
                    // as the copied device output before either is read or reused.
                    assert_eq!(unsafe { confidence.bytes() }, values);
                }
                copy.synchronize().unwrap();
                model.synchronize().unwrap();
                unsafe {
                    driver
                        .check(
                            "cuGraphExecDestroy",
                            (driver.executable_destroy)(executable),
                        )
                        .unwrap();
                    driver
                        .check("cuGraphDestroy", (driver.graph_destroy)(graph))
                        .unwrap();
                }
            }
        }
        unsafe {
            driver.check("cuMemFree_v2", (driver.free)(source)).unwrap();
            driver
                .check("cuMemFree_v2", (driver.free)(device_output))
                .unwrap();
            driver
                .check("cuCtxDestroy_v2", (driver.context_destroy)(context))
                .unwrap();
        }
    }

    #[test]
    fn duration_estimate_adapts_faster_downward_than_upward() {
        assert_eq!(update_duration_estimate(0, 0), 0);
        assert_eq!(update_duration_estimate(0, 10_000), 10_000);
        assert_eq!(update_duration_estimate(10_000, 0), 10_000);
        // Shorter observation moves halfway down immediately.
        assert_eq!(update_duration_estimate(10_000, 6_000), 8_000);
        // Longer observation moves one quarter upward.
        assert_eq!(update_duration_estimate(10_000, 14_000), 11_000);
    }

    #[test]
    fn predictive_park_cruises_lands_and_bounds_overshoot_on_integrated_socs() {
        // Unmeasured integrated SoC frame sleeps in bounded steps rather than burning shared power.
        assert_eq!(
            decide_park(0, 0),
            ParkAction::Sleep(std::time::Duration::from_micros(PARK_FALLBACK_SLEEP_US))
        );
        // Sub-landing frame enters the low-latency landing wait immediately.
        assert_eq!(decide_park(0, 500), ParkAction::Land);
        // 120 ms frame sleeps in 500 us chunks until 600 us before predicted completion, then lands.
        assert_eq!(
            decide_park(0, 120_000),
            ParkAction::Sleep(std::time::Duration::from_micros(PARK_MAX_SLEEP_US))
        );
        assert_eq!(
            decide_park(119_250, 120_000),
            ParkAction::Sleep(std::time::Duration::from_micros(150))
        );
        assert_eq!(decide_park(119_350, 120_000), ParkAction::Land);
        assert_eq!(decide_park(120_500, 120_000), ParkAction::Land);
        // Overshoot past expected + PARK_OVERSHOOT_CAP_US falls back to bounded sleep.
        assert_eq!(
            decide_park(121_001, 120_000),
            ParkAction::Sleep(std::time::Duration::from_micros(PARK_FALLBACK_SLEEP_US))
        );
    }
}
