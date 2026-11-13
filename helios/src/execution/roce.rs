/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! RoCEnante's RDMA proxy for cores whose ranks run on separate hosts.
//!
//! A rank's graphs launch b12x's RoCE kernels against a pinned region restored at its recorded
//! address. This side opens the devices of the fabric this host discovered (`roce_fabric`),
//! registers that region, exchanges queue-pair records with every peer, connects each peer's
//! rails, and runs the posting thread (`native/roce_proxy.c`, the proxy the kernels' ABI was
//! compiled against).

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};

use super::roce_fabric::Shape;
use super::roce_fabric::Topology;
use crate::runtime_core::RoceSpec;
use crate::{ExecutionError, Failure, Result};

/// The proxy ABI compiled into this runtime and required in live peer records.
///
/// Operation credits prevent a rank from overwriting receive rows while a peer still uses them.
/// Open lines of four or more ranks broadcast pipelined all-gather chunks. Connection records
/// carry the ring geometry so peers can refuse incompatible allocations before using addresses.
///
/// Core compatibility is checked separately against the recorded GPU-visible region layout.
pub const PROXY_ABI: u32 = crate::runtime_core::roce_layout::PROXY_ABI;
/// Control word 7 after a variable-length all-gather met a byte count it cannot send.
const INVALID_VARLEN_COUNT: u32 = 1;
/// Queue pairs this rank opens per rail and peer, and the value every rank must pass to
/// `roce_create`: ABI 15 refuses a group whose peers report a different lane count at connect.
///
/// The relay itself always uses lane 0, so the count only stripes this runtime's direct unicast
/// and pipelined traffic. Four is the deployed TP4 value; `roce_create` accepts 1 through
/// `ROCE_MAX_LANES`, and the blob carries the count so a group mixing builds is refused before any
/// queue pair is connected rather than mis-striped.
const LANES: c_int = 4;

#[repr(C)]
struct Context {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn roce_abi_version() -> c_int;
    // The pure layout in `runtime_core::roce_layout` answers the runtime's questions, so the
    // native query is retained only as the parity oracle its tests compare against.
    #[allow(dead_code)]
    fn roce_layout(
        world: c_int,
        slot_bytes: u64,
        pipe_bytes: u64,
        varlen_bytes: u64,
        out: *mut u64,
    ) -> c_int;
    fn roce_blob_bytes() -> u64;
    fn roce_create(
        world: c_int,
        rank: c_int,
        hca_names: *const *const c_char,
        n_hca: c_int,
        gid_index: c_int,
        traffic_class: c_int,
        n_lane: c_int,
        region: *mut c_void,
        region_bytes: u64,
        slot_bytes: u64,
        pipe_bytes: u64,
        varlen_bytes: u64,
        err: *mut c_char,
        err_len: u64,
    ) -> *mut Context;
    fn roce_enable_ring(c: *mut Context) -> c_int;
    fn roce_enable_ring_open(c: *mut Context) -> c_int;
    fn roce_ring_bytes(world: c_int, open: c_int) -> u64;
    fn roce_local_blob(c: *mut Context, out: *mut c_void, out_len: u64) -> c_int;
    fn roce_connect(
        c: *mut Context,
        blobs: *const c_void,
        blobs_len: u64,
        routes: *const i32,
        n_rail: c_int,
    ) -> c_int;
    fn roce_start(c: *mut Context) -> c_int;
    fn roce_failed(c: *mut Context) -> c_int;
    fn roce_error(c: *mut Context) -> *const c_char;
    fn roce_destroy(c: *mut Context);
}

fn failure(message: impl Into<String>) -> crate::Error {
    ExecutionError::Poisoned(Failure::new(message.into())).into()
}

/// The ring mailbox area a `world`-rank ring or chain allocates at startup.
///
/// The proxy owns this memory: `roce_enable_ring` or `roce_enable_ring_open` allocates and
/// registers it after `roce_create`, so it is not part of `roce_layout`'s captured region. Reserve
/// it before the transport exists, for the topology the rank will build: an `open` line allocates
/// 192.19 MiB per rank at four ranks, 896.90 MiB at eight and 3.75 GiB at sixteen, because a
/// broadcast uses a 256 KiB packet and a 32-slot window per destination. Lines of two or three
/// ranks retain per-destination delivery. A closed ring allocates 4.02 MiB at four
/// ranks and 16.06 MiB at sixteen: 64 KiB packets behind an eight-slot window shared by the
/// origin's destinations. Zero for a world the proxy does not support.
pub fn ring_bytes(world: usize, open: bool) -> u64 {
    // SAFETY: This query has no arguments beyond the world size and the topology and allocates
    // nothing.
    unsafe { roce_ring_bytes(world as c_int, open as c_int) }
}

/// One transport's lifecycle: created, connected exactly once, then terminal forever.
///
/// In-process quiesce, reset and rejoin are not implemented, so a failed transport is terminal.
/// A failed context has posted serials that may not be replayed and ABI 15's proxy refuses to
/// start it again; replacing it in place would have to publish the new context's record to peers
/// that have already read the old one, which this wrapper does not attempt. `open` shows the
/// supported order — create, publish this context's record, connect — and it only ever runs once
/// per transport. Recovery is therefore a fresh group: a coordinated restart of every rank
/// process, which reloads the core and restores the pinned region and the device counters
/// (epoch and poison) at their recorded addresses and zeroed, so every rank begins again at
/// sequence 1 with empty flags. Each failed setup path latches its diagnostic here, and `health`
/// reports that diagnostic instead of trusting the proxy's error words, which stay zero for setup
/// failures.
#[derive(Default)]
struct LinkState {
    state: AtomicU8,
    /// The first failure that made this transport terminal; every later check repeats it.
    diagnostic: Mutex<Option<String>>,
}

const LINK_CREATED: u8 = 0;
const LINK_CONNECTED: u8 = 1;
const LINK_TERMINAL: u8 = 2;

impl LinkState {
    /// Take the one connection slot. A second connect would replay the dead context's serials.
    fn admit_connect(&self) -> Result<()> {
        match self.state.compare_exchange(
            LINK_CREATED,
            LINK_CONNECTED,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(()),
            Err(LINK_CONNECTED) => Err(failure(
                "RoCE transport is already connected; a context connects once, and a failed group \
                 is recreated rather than restarted",
            )),
            Err(_) => Err(self.terminal_failure()),
        }
    }

    /// Whether the context's one connection completed; `health` reports nothing before that.
    fn connected(&self) -> bool {
        self.state.load(Ordering::Acquire) == LINK_CONNECTED
    }

    fn terminal(&self) -> bool {
        self.state.load(Ordering::Acquire) == LINK_TERMINAL
    }

    /// Make the transport terminal for good, keeping the first diagnostic.
    fn fail(&self, diagnostic: String) {
        // Publish the diagnostic before the state, so a reader that observes TERMINAL can read it.
        let mut slot = self
            .diagnostic
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        slot.get_or_insert(diagnostic);
        self.state.store(LINK_TERMINAL, Ordering::Release);
    }

    fn diagnostic(&self) -> Option<String> {
        self.diagnostic
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn terminal_failure(&self) -> crate::Error {
        failure(
            self.diagnostic()
                .unwrap_or_else(|| "RoCE transport is terminal after a failure".to_owned()),
        )
    }
}

fn peer_blobs(records: &[Vec<u8>], world: usize) -> Result<Vec<u8>> {
    // Every live peer must use the same proxy protocol, independently of the core's captured
    // layout compatibility checked by open().
    let size = unsafe { roce_blob_bytes() } as usize;
    if records.len() != world
        || records
            .iter()
            .any(|record| record.len() != size + 4 || record[..4] != PROXY_ABI.to_le_bytes())
    {
        return Err(failure(
            "RoCE queue-pair records must cover every rank with the same native proxy ABI",
        ));
    }
    Ok(records
        .iter()
        .flat_map(|record| record[4..].iter().copied())
        .collect())
}

/// The calls one proxy context accepts, separated from the FFI so the lifecycle above them can be
/// driven by a deterministic shim in tests. Native transports bind the vendored C proxy.
trait Proxy {
    /// This rank's connection record, without the ABI prefix.
    fn local_blob(&self) -> std::result::Result<Vec<u8>, String>;
    /// Connect every peer's rails on the current context.
    fn connect(
        &self,
        blobs: &[u8],
        routes: &[i32],
        rails: usize,
    ) -> std::result::Result<(), String>;
    /// Start the posting thread on a connected context.
    fn start(&self) -> std::result::Result<(), String>;
    fn failed(&self) -> bool;
    fn error(&self) -> String;
    /// The eight control words inside the core region.
    fn control(&self) -> *const AtomicU32;
    /// Stop the posting thread and release the context's queue pairs, registrations and memory.
    ///
    /// # Safety
    /// Nothing may drive the context afterwards; the owner makes this call once.
    unsafe fn destroy(&mut self);
}

/// The vendored C proxy's context over the core's pinned region.
struct Native {
    context: *mut Context,
    // The control record: words 2, 3 and 6 name a failed wait's sequence, peer and rail; word 7
    // why a kernel stopped without a timeout.
    control: *const AtomicU32,
}

// SAFETY: The proxy context is driven only through its thread-safe C entry points, and the error
// words are read atomically. Setup serializes through `&mut self`, health polls through `&self`,
// and the core region outlives the context.
unsafe impl Send for Native {}
unsafe impl Sync for Native {}

impl Native {
    fn message(&self) -> String {
        let text = unsafe { roce_error(self.context) };
        if text.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .into_owned()
        }
    }
}

impl Proxy for Native {
    fn local_blob(&self) -> std::result::Result<Vec<u8>, String> {
        let mut record = vec![0u8; unsafe { roce_blob_bytes() } as usize];
        let status = unsafe {
            roce_local_blob(
                self.context,
                record.as_mut_ptr().cast(),
                record.len() as u64,
            )
        };
        if status != 0 {
            return Err("the queue-pair record does not fit its buffer".to_owned());
        }
        Ok(record)
    }

    fn connect(
        &self,
        blobs: &[u8],
        routes: &[i32],
        rails: usize,
    ) -> std::result::Result<(), String> {
        let status = unsafe {
            roce_connect(
                self.context,
                blobs.as_ptr().cast(),
                blobs.len() as u64,
                routes.as_ptr(),
                rails as c_int,
            )
        };
        if status != 0 { Err(self.message()) } else { Ok(()) }
    }

    fn start(&self) -> std::result::Result<(), String> {
        if unsafe { roce_start(self.context) } != 0 {
            Err(self.message())
        } else {
            Ok(())
        }
    }

    fn failed(&self) -> bool {
        unsafe { roce_failed(self.context) != 0 }
    }

    fn error(&self) -> String {
        self.message()
    }

    fn control(&self) -> *const AtomicU32 {
        self.control
    }

    unsafe fn destroy(&mut self) {
        if !self.context.is_null() {
            unsafe { roce_destroy(self.context) };
            self.context = std::ptr::null_mut();
        }
    }
}

impl Drop for Native {
    fn drop(&mut self) {
        // Stops the posting thread before releasing queue pairs and memory registrations, and
        // never waits for health: teardown must not skip QP destruction because the context was
        // interrupted.
        unsafe { self.destroy() };
    }
}

/// One context's lifecycle over whatever proxy backs it: the state machine and the health checks
/// every transport runs. Serving drives it with [`Native`]; tests drive it with a shim.
struct Session<P: Proxy> {
    proxy: P,
    world: usize,
    rank: usize,
    rails: usize,
    routes: Vec<i32>,
    link: LinkState,
}

impl<P: Proxy> Session<P> {
    /// This rank's queue-pair record, sent to every peer before [`Session::connect`].
    fn local_record(&self) -> Result<Vec<u8>> {
        // A terminal context's record names queue pairs that are gone, or that are connected to
        // peers which are also gone. Publishing it again would poison the group it is offered to,
        // so a fresh context must publish instead.
        if self.link.terminal() {
            return Err(self.link.terminal_failure());
        }
        let record = self
            .proxy
            .local_blob()
            .map_err(|message| failure(format!("RoCE queue-pair record failed: {message}")))?;
        let mut versioned = Vec::with_capacity(record.len() + 4);
        versioned.extend_from_slice(&PROXY_ABI.to_le_bytes());
        versioned.extend_from_slice(&record);
        Ok(versioned)
    }

    /// Connect every peer's rails from all ranks' records, in rank order, and start posting.
    ///
    /// This is the context's one connection. Every failure from here on — records that do not
    /// cover the group, a native connect refusal, a posting thread that will not start — latches
    /// its diagnostic and makes the transport terminal.
    fn connect(&mut self, records: &[Vec<u8>]) -> Result<()> {
        self.link.admit_connect()?;
        match self.join(records) {
            Ok(()) => Ok(()),
            Err(error) => {
                // Setup failures leave the proxy's `failed` word and the kernel's error words at
                // zero, so the state machine, not those words, is what health reports afterwards.
                self.link
                    .fail(format!("RoCE transport is terminal: {error}"));
                Err(error)
            }
        }
    }

    /// Post every peer's rails and start the posting thread on the current context.
    fn join(&mut self, records: &[Vec<u8>]) -> Result<()> {
        let joined = peer_blobs(records, self.world)?;
        self.proxy
            .connect(&joined, &self.routes, self.rails)
            .map_err(|message| failure(format!("RoCE queue-pair connect failed: {message}")))?;
        self.proxy
            .start()
            .map_err(|message| failure(format!("RoCE proxy thread failed to start: {message}")))?;
        Ok(())
    }

    /// Whether this context can never be used again.
    fn terminal(&self) -> bool {
        self.link.terminal()
    }

    /// Fail when the proxy stopped or a kernel's peer wait timed out; a frame that completed after
    /// either carries no trustworthy collective result.
    ///
    /// The answer comes from the lifecycle state first: a setup failure is terminal even though the
    /// proxy's `failed` word and the kernel's error words are still zero, and a latched diagnostic
    /// outlives any later poll.
    fn health(&self) -> Result<()> {
        if self.link.terminal() {
            return Err(self.link.terminal_failure());
        }
        if !self.link.connected() {
            return Err(failure(
                "RoCE transport has not connected yet; no collective may launch on it",
            ));
        }
        if self.proxy.failed() {
            let diagnostic = format!(
                "RoCE transport is terminal: RoCE proxy failed: {}",
                self.proxy.error()
            );
            self.link.fail(diagnostic.clone());
            return Err(failure(diagnostic));
        }
        // SAFETY: The control record lies inside the region the transport was opened over.
        let words = unsafe { std::slice::from_raw_parts(self.proxy.control(), 8) };
        match check_control(self.rank, words) {
            Ok(()) => Ok(()),
            Err(error) => {
                let diagnostic = format!("RoCE transport is terminal: {error}");
                self.link.fail(diagnostic.clone());
                Err(failure(diagnostic))
            }
        }
    }
}

/// The (local HCA, peer HCA) pair for every peer's rails, in rank order.
fn routes(spec: &RoceSpec, topology: &Topology) -> Vec<i32> {
    let rails = topology.rails;
    let mut routes = Vec::with_capacity(spec.world * rails * 2);
    for (peer, links) in topology.routes.iter().enumerate() {
        for rail in 0..rails {
            let (local, remote) =
                if peer == spec.rank || topology.shape.relays() && links.is_empty() {
                    (0, 0)
                } else {
                    links[rail]
                };
            routes.push(local as i32);
            routes.push(remote as i32);
        }
    }
    routes
}

/// One rank's connected transport. Dropping it stops the proxy and releases the HCAs; the region
/// itself belongs to the core.
pub struct RoceTransport {
    spec: RoceSpec,
    topology: Topology,
    session: Session<Native>,
}

impl RoceTransport {
    /// Open the topology's devices and register the restored region.
    ///
    /// This is the only place a proxy context is created, and it runs before `connect` and before
    /// any graph has touched the region, so zeroing the region here cannot race a live GPU or NIC
    /// user. A context that fails afterwards is never recreated in place: the group is recreated by
    /// a coordinated restart of every rank process, which reloads the core and restores the
    /// pinned region and the device counters (epoch and poison) at their recorded addresses and
    /// zeroed, so every rank begins again at sequence 1.
    ///
    /// # Safety
    /// `spec.region_address` must name `spec.region_bytes` of mapped, locked host memory that the
    /// core restored for this rank and that outlives the transport; no graph may run on it
    /// before `connect` returns.
    pub unsafe fn open(spec: &RoceSpec, topology: &Topology) -> Result<Self> {
        let abi = unsafe { roce_abi_version() } as u32;
        if abi != PROXY_ABI {
            return Err(failure(format!(
                "compiled RoCE proxy reports ABI {abi}, this runtime requires {PROXY_ABI}"
            )));
        }
        // Repeated from the loader's metadata gate: a spec built without the loader is still
        // refused before a device is touched.
        crate::runtime_core::roce_layout::check_core_layout(spec)?;
        if (topology.shape.relays() && !(2..=16).contains(&spec.world))
            || topology.routes.len() != spec.world
            || !(1..=super::roce_fabric::MAX_RAILS).contains(&topology.rails)
        {
            return Err(failure(format!(
                "RoCE topology does not match the core's ranks: {} routes and {} rails for {} ranks",
                topology.routes.len(),
                topology.rails,
                spec.world
            )));
        }
        // The relay's mailbox area is group-negotiated: every rank's proxy must allocate the same
        // `ring_bytes` and agree on its protocol, window and chunk size before any address is used.
        if topology.shape.relays()
            && ring_bytes(spec.world, topology.shape == Shape::Chain) == 0
        {
            return Err(failure(format!(
                "RoCE relay ring geometry is unsupported for {} ranks",
                spec.world
            )));
        }
        for (peer, links) in topology.routes.iter().enumerate() {
            let absent = !topology.shape.linked(spec.rank, peer, spec.world);
            if absent != links.is_empty()
                || (!absent && links.len() != topology.rails)
                || links
                    .iter()
                    .any(|(local, _)| *local as usize >= topology.hcas.len())
            {
                return Err(failure(format!(
                    "RoCE topology does not match the core's ranks: rank {} has {} links to rank {peer},                      expected {} ({} rails, {} HCAs)",
                    spec.rank,
                    links.len(),
                    if absent { 0 } else { topology.rails },
                    topology.rails,
                    topology.hcas.len()
                )));
            }
        }
        let routes = routes(spec, topology);
        let context = unsafe { Self::create(spec, topology)? };
        Ok(Self {
            spec: spec.clone(),
            topology: topology.clone(),
            session: Session {
                proxy: Native {
                    context,
                    control: (spec.region_address + spec.ctrl_offset) as *const AtomicU32,
                },
                world: spec.world,
                rank: spec.rank,
                rails: topology.rails,
                routes,
                link: LinkState::default(),
            },
        })
    }

    /// Create one proxy context over the core's region and enable the topology's relay.
    ///
    /// # Safety
    /// As [`RoceTransport::open`]; the region stays live for the context's whole life.
    unsafe fn create(spec: &RoceSpec, topology: &Topology) -> Result<*mut Context> {
        let names = topology
            .hcas
            .iter()
            .map(|name| CString::new(name.as_str()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| failure("RoCE HCA name contains a NUL byte"))?;
        let pointers: Vec<*const c_char> = names.iter().map(|name| name.as_ptr()).collect();
        let region = spec.region_address as *mut c_void;
        // Flags and the control record must start at zero: every rank begins again at sequence 1.
        unsafe { std::ptr::write_bytes(region.cast::<u8>(), 0, spec.region_bytes as usize) };
        let mut message = [0 as c_char; 512];
        let context = unsafe {
            roce_create(
                spec.world as c_int,
                spec.rank as c_int,
                pointers.as_ptr(),
                pointers.len() as c_int,
                topology.gid_index as c_int,
                topology.traffic_class as c_int,
                LANES,
                region,
                spec.region_bytes,
                spec.slot_bytes,
                spec.pipe_bytes,
                spec.varlen_bytes,
                message.as_mut_ptr(),
                message.len() as u64,
            )
        };
        if context.is_null() {
            let text = unsafe { CStr::from_ptr(message.as_ptr()) }.to_string_lossy();
            return Err(failure(format!("RoCE transport setup failed: {text}")));
        }
        let enabled = match topology.shape {
            Shape::Mesh => 0,
            Shape::Ring => unsafe { roce_enable_ring(context) },
            Shape::Chain => unsafe { roce_enable_ring_open(context) },
        };
        if enabled != 0 {
            let message = unsafe { CStr::from_ptr(roce_error(context)) }
                .to_string_lossy()
                .into_owned();
            unsafe { roce_destroy(context) };
            return Err(failure(format!("RoCE ring setup failed: {message}")));
        }
        Ok(context)
    }

    pub fn spec(&self) -> &RoceSpec {
        &self.spec
    }

    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    /// This rank's queue-pair record, sent to every peer before [`RoceTransport::connect`].
    pub fn local_record(&self) -> Result<Vec<u8>> {
        self.session.local_record()
    }

    /// Connect every peer's rails from all ranks' records, in rank order, and start posting.
    ///
    /// A context connects at most once. ABI 15's proxy refuses `roce_start` on a context that
    /// failed or was stopped inside an operation, and its serials may not be replayed, so a failed
    /// transport is never retried or rebuilt in place: every rank process is restarted so the group
    /// reloads its core and rejoins with fresh contexts and newly exchanged records.
    pub fn connect(&mut self, records: &[Vec<u8>]) -> Result<()> {
        self.session.connect(records)
    }

    /// Whether this transport is terminal: the proxy stopped, a kernel's wait timed out, or its
    /// setup failed. Such a context is never restarted; the group is recreated.
    pub fn terminal(&self) -> bool {
        self.session.terminal()
    }

    /// Fail when the transport is terminal — including a setup failure the proxy's own error words
    /// do not report — or a kernel's peer wait timed out. A frame that completed after either
    /// carries no trustworthy collective result.
    pub fn health(&self) -> Result<()> {
        self.session.health()
    }
}

fn check_control(rank: usize, words: &[AtomicU32]) -> Result<()> {
    let sequence = words[2].load(Ordering::Acquire);
    if sequence != 0 && words[7].load(Ordering::Relaxed) == INVALID_VARLEN_COUNT {
        return Err(failure(format!(
            "RoCE variable-length all-gather at sequence {sequence} received an unusable \
             byte count for rank {}; every rank stopped before sending, and the transport \
             is poisoned",
            words[3].load(Ordering::Relaxed),
        )));
    }
    if sequence != 0 {
        return Err(failure(format!(
            "RoCE collective on rank {} timed out waiting for rank {}, HCA {}, at sequence \
             {sequence}; the transport is poisoned",
            rank,
            words[3].load(Ordering::Relaxed),
            words[6].load(Ordering::Relaxed),
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_core::roce_layout::{
        OLDEST_PACKAGE_ABI, PARITY_PACKAGE_ABI, check_core_abi, layout as pure_layout,
        parity_pipe_excess,
    };

    fn specification() -> (Vec<u8>, RoceSpec, Topology) {
        let mut layout = [0; 17];
        // SAFETY: The C layout function writes exactly seventeen words and opens no devices.
        assert_eq!(
            unsafe { roce_layout(4, 4096, 4096, 4096, layout.as_mut_ptr()) },
            0
        );
        let mut region = vec![0x5a; layout[4] as usize];
        let spec = RoceSpec {
            proxy_abi: PROXY_ABI,
            world: 4,
            rank: 0,
            region_address: region.as_mut_ptr() as u64,
            region_bytes: layout[4],
            counters_address: 0,
            counters_bytes: 0,
            slot_bytes: 4096,
            pipe_bytes: 4096,
            varlen_bytes: 4096,
            ctrl_offset: layout[3],
        };
        let topology = Topology {
            hcas: vec!["rail0".into(), "rail1".into()],
            gid_index: 3,
            traffic_class: 0,
            rails: 2,
            routes: vec![vec![], vec![(0, 0), (1, 1)], vec![], vec![(0, 0), (1, 1)]],
            shape: Shape::Ring,
        };
        (region, spec, topology)
    }

    fn layout_of(spec: &RoceSpec) -> [u64; 17] {
        let mut layout = [0; 17];
        // SAFETY: Layout calculation opens no devices and writes seventeen words.
        assert_eq!(
            unsafe {
                roce_layout(
                    spec.world as c_int,
                    spec.slot_bytes,
                    spec.pipe_bytes,
                    spec.varlen_bytes,
                    layout.as_mut_ptr(),
                )
            },
            0
        );
        layout
    }

    #[test]
    fn pure_layout_matches_the_native_proxy_layout() {
        for (world, slot_bytes, pipe_bytes, varlen_bytes) in [
            (2, 4096, 0, 0),
            (4, 4096, 4096, 4096),
            (8, 8192, 65_536, 0),
            (16, 1 << 20, 1 << 20, 1 << 20),
            // The native ceilings are inclusive, so the pure layout must accept them too.
            (16, 1 << 40, 0, 0),
            (4, 4096, 1 << 32, 0),
            (4, 4096, 0, 1 << 32),
        ] {
            let mut native = [0u64; 17];
            // SAFETY: The layout query opens no device and writes exactly seventeen words.
            assert_eq!(
                unsafe {
                    roce_layout(
                        world,
                        slot_bytes,
                        pipe_bytes,
                        varlen_bytes,
                        native.as_mut_ptr(),
                    )
                },
                0
            );
            assert_eq!(
                pure_layout(world as usize, slot_bytes, pipe_bytes, varlen_bytes).unwrap(),
                native,
                "world {world}, slot {slot_bytes}, pipe {pipe_bytes}, varlen {varlen_bytes}"
            );
        }
        // Both implementations refuse the same invalid geometries.
        for (world, slot_bytes, pipe_bytes, varlen_bytes) in [
            (1, 4096, 0, 0),
            (17, 4096, 0, 0),
            (4, 4097, 0, 0),
            (4, 4096, 1, 0),
            (4, (1 << 40) + 4096, 0, 0),
        ] {
            let mut native = [0u64; 17];
            // SAFETY: The layout query opens no device and writes at most seventeen words.
            assert_ne!(
                unsafe {
                    roce_layout(
                        world,
                        slot_bytes,
                        pipe_bytes,
                        varlen_bytes,
                        native.as_mut_ptr(),
                    )
                },
                0
            );
            assert!(
                pure_layout(world as usize, slot_bytes, pipe_bytes, varlen_bytes).is_err(),
                "world {world}, slot {slot_bytes}, pipe {pipe_bytes}, varlen {varlen_bytes}"
            );
        }
    }

    #[test]
    fn native_proxy_abi_matches_core_contract() {
        // SAFETY: This query has no arguments, device access, or allocation.
        assert_eq!(unsafe { roce_abi_version() } as u32, PROXY_ABI);
        assert_eq!(PROXY_ABI, 15);
    }

    #[test]
    fn ring_mailboxes_are_budgeted_before_the_transport_exists() {
        // The exact native allocations `roce_enable_ring` and `roce_enable_ring_open` make, checked
        // against the slot arithmetic in `native/roce_proxy.c` rather than a literal.
        for open in [false, true] {
            assert_eq!(ring_bytes(1, open), 0);
            assert_eq!(ring_bytes(17, open), 0);
            // Lines of two or three ranks retain per-destination delivery and the closed geometry.
            assert_eq!(ring_bytes(2, open), 2_105_600);
            assert_eq!(ring_bytes(3, open), 3_158_400);
        }
        // A closed ring: 64 KiB packets behind one eight-slot window per origin and rail.
        assert_eq!(ring_bytes(4, false), 4_211_200);
        assert_eq!(ring_bytes(8, false), 8_422_400);
        assert_eq!(ring_bytes(16, false), 16_844_800);
        // An open line of four or more ranks: 256 KiB packets behind a 32-slot window per
        // destination, which is what the broadcast needs.
        assert_eq!(ring_bytes(4, true), 201_529_344);
        assert_eq!(ring_bytes(8, true), 940_470_272);
        assert_eq!(ring_bytes(16, true), 4_030_586_880);
    }

    #[test]
    fn core_abis_validate_receive_layout_before_touching_storage() {
        let (region, spec, topology) = specification();
        for pipe_bytes in [0, 4096] {
            for abi in 8..=16 {
                let mut spec = spec.clone();
                spec.proxy_abi = abi;
                spec.pipe_bytes = pipe_bytes;
                spec.ctrl_offset += 128;
                // SAFETY: ABI or layout validation rejects before touching the live region.
                let Err(error) = (unsafe { RoceTransport::open(&spec, &topology) }) else {
                    panic!("invalid layout must fail");
                };
                let error = error.to_string();
                let expected = if !(OLDEST_PACKAGE_ABI..=PROXY_ABI).contains(&abi) {
                    "proxy ABI"
                } else {
                    "region differs"
                };
                assert!(
                    error.contains(expected),
                    "ABI {abi}, pipe {pipe_bytes}: {error}"
                );
                assert!(region.iter().all(|byte| *byte == 0x5a));
            }
        }
    }

    #[test]
    fn an_older_proxy_refuses_a_newer_core_in_both_directions() {
        let (_, spec, _) = specification();
        let layout = layout_of(&spec);
        // A core newer than the runtime is refused, and an older runtime refuses this one.
        let error = check_core_abi(16, 15, &spec, &layout)
            .expect_err("newer core must be refused");
        assert!(error.to_string().contains("proxy ABI 16"), "{error}");
        assert!(error.to_string().contains("provides 15"), "{error}");
        let error = check_core_abi(15, 14, &spec, &layout)
            .expect_err("older runtime must be refused");
        assert!(error.to_string().contains("provides 14"), "{error}");
        // Every older ABI shares this kernel-visible region, so it is served here unchanged.
        for core_abi in OLDEST_PACKAGE_ABI..=PROXY_ABI {
            assert!(
                check_core_abi(core_abi, 15, &spec, &layout).is_ok(),
                "ABI {core_abi}"
            );
        }
        // Without a pipelined area the region keeps its size across all of them.
        let mut unpipelined = spec.clone();
        unpipelined.pipe_bytes = 0;
        let unpipelined_layout = layout_of(&unpipelined);
        for core_abi in OLDEST_PACKAGE_ABI..=PROXY_ABI {
            assert!(check_core_abi(core_abi, 15, &unpipelined, &unpipelined_layout).is_ok());
        }
    }

    #[test]
    fn sequence_parity_cores_are_refused_by_their_layout_not_their_number() {
        // ABI 12 names two wires: one with a single pipelined receive row per peer and chunk
        // (this layout) and one with two rows selected by `seq & 1`, which doubled the pipelined
        // rows. Only the region tells them apart, so the gate checks the geometry the export
        // recorded.
        let (region, spec, topology) = specification();
        let layout = layout_of(&spec);
        let mut parity = spec.clone();
        parity.proxy_abi = PARITY_PACKAGE_ABI;
        parity.region_bytes = layout[4] + parity_pipe_excess(spec.world as u64, &layout);
        let error = check_core_abi(parity.proxy_abi, 15, &parity, &layout)
            .expect_err("the doubled-row pipelined layout must be refused")
            .to_string();
        assert!(error.contains("sequence-parity"), "{error}");
        assert!(error.contains("single-copy"), "{error}");
        assert!(error.contains("re-export from a proxy ABI 15"), "{error}");
        // The refusal happens in preflight, before the region is zeroed or a device opened.
        // SAFETY: The parity geometry fails validation before touching the live allocation.
        let Err(error) = (unsafe { RoceTransport::open(&parity, &topology) }) else {
            panic!("parity region must fail preflight");
        };
        let error = error.to_string();
        assert!(error.contains("sequence-parity"), "{error}");
        assert!(region.iter().all(|byte| *byte == 0x5a));
        // The same number with upstream's single-copy region is served.
        assert!(check_core_abi(PARITY_PACKAGE_ABI, 15, &spec, &layout).is_ok());
        // Any other surplus is an ordinary geometry mismatch, refused by the layout check.
        let mut other = parity.clone();
        other.region_bytes += 4096;
        assert!(check_core_abi(PARITY_PACKAGE_ABI, 15, &other, &layout).is_ok());
        // SAFETY: A region that does not match the layout is rejected before device access.
        let Err(error) = (unsafe { RoceTransport::open(&other, &topology) }) else {
            panic!("a mismatched region must fail");
        };
        let error = error.to_string();
        assert!(error.contains("region differs"), "{error}");
        assert!(region.iter().all(|byte| *byte == 0x5a));
    }

    /// The proxy calls one deterministic shim answers, so the lifecycle every transport runs can
    /// be tested without a NIC or a GPU: a fresh context publishes fresh queue-pair numbers, setup
    /// can fail while the proxy's own `failed` word and the kernel's error words stay clear, and
    /// the control record is a real array the test can poison.
    #[derive(Default)]
    struct ShimState {
        generation: u32,
        connects: usize,
        starts: usize,
        destroys: usize,
        failed: bool,
        error: String,
        connect_error: Option<String>,
        start_error: Option<String>,
    }

    type Shared = std::sync::Arc<Mutex<ShimState>>;

    struct Shim {
        state: Shared,
        control: std::sync::Arc<[AtomicU32; 8]>,
    }

    impl Proxy for Shim {
        fn local_blob(&self) -> std::result::Result<Vec<u8>, String> {
            let size = unsafe { roce_blob_bytes() } as usize;
            Ok(vec![self.state.lock().unwrap().generation as u8; size])
        }

        fn connect(
            &self,
            _blobs: &[u8],
            _routes: &[i32],
            _rails: usize,
        ) -> std::result::Result<(), String> {
            let mut state = self.state.lock().unwrap();
            state.connects += 1;
            match &state.connect_error {
                Some(error) => Err(error.clone()),
                None => Ok(()),
            }
        }

        fn start(&self) -> std::result::Result<(), String> {
            let mut state = self.state.lock().unwrap();
            state.starts += 1;
            match &state.start_error {
                Some(error) => Err(error.clone()),
                None => Ok(()),
            }
        }

        fn failed(&self) -> bool {
            self.state.lock().unwrap().failed
        }

        fn error(&self) -> String {
            self.state.lock().unwrap().error.clone()
        }

        fn control(&self) -> *const AtomicU32 {
            self.control.as_ptr()
        }

        unsafe fn destroy(&mut self) {
            self.state.lock().unwrap().destroys += 1;
        }
    }

    impl Drop for Shim {
        // A proxy context releases its queue pairs and registrations when its owner drops it.
        fn drop(&mut self) {
            unsafe { self.destroy() };
        }
    }

    fn shim() -> (Shared, std::sync::Arc<[AtomicU32; 8]>, Session<Shim>) {
        let state = Shared::default();
        let control = std::sync::Arc::new(std::array::from_fn(|_| AtomicU32::new(0)));
        let session = Session {
            proxy: Shim {
                state: state.clone(),
                control: control.clone(),
            },
            world: 2,
            rank: 0,
            rails: 1,
            routes: vec![0, 0, 0, 0],
            link: LinkState::default(),
        };
        (state, control, session)
    }

    fn shim_records() -> Vec<Vec<u8>> {
        let size = unsafe { roce_blob_bytes() } as usize;
        (0..2u8)
            .map(|rank| {
                let mut record = PROXY_ABI.to_le_bytes().to_vec();
                record.extend(vec![rank; size]);
                record
            })
            .collect()
    }

    #[test]
    fn a_connected_context_never_connects_twice() {
        let (state, _control, mut session) = shim();
        session.connect(&shim_records()).unwrap();
        session.health().unwrap();
        assert!(!session.terminal());
        // A second start on a live context would replay its serials: refused.
        let error = session
            .connect(&shim_records())
            .expect_err("a second connect must fail")
            .to_string();
        assert!(error.contains("already connected"), "{error}");
        // Refusing the duplicate does not poison the context that is already posting.
        session.health().unwrap();
        assert!(!session.terminal());
        let state = state.lock().unwrap();
        assert_eq!(state.connects, 1, "the proxy saw one connect");
        assert_eq!(state.starts, 1, "the proxy saw one start");
    }

    #[test]
    fn records_that_do_not_cover_the_group_are_terminal() {
        let (state, _control, mut session) = shim();
        let error = session
            .connect(&[])
            .expect_err("a short record set must fail")
            .to_string();
        assert!(error.contains("queue-pair records"), "{error}");
        assert!(session.terminal(), "an invalid group is terminal");
        let first = session.health().unwrap_err().to_string();
        assert!(first.contains("terminal"), "{first}");
        assert!(first.contains("queue-pair records"), "{first}");
        // Every later poll reports the same diagnostic, even though the proxy is silent.
        assert_eq!(session.health().unwrap_err().to_string(), first);
        // The dead context is neither connected again nor republished to a new group.
        let error = session
            .connect(&shim_records())
            .expect_err("a terminal context must not connect")
            .to_string();
        assert!(error.contains("terminal"), "{error}");
        let error = session
            .local_record()
            .expect_err("a terminal context must not publish its queue pairs")
            .to_string();
        assert!(error.contains("terminal"), "{error}");
        assert_eq!(state.lock().unwrap().connects, 0, "nothing reached the proxy");
    }

    #[test]
    fn setup_failures_are_terminal_and_visible_in_health() {
        // The proxy's `failed` word and the kernel error words are zero for a setup failure, so
        // only the state machine can make them visible.
        for (connect_error, start_error, expected) in [
            (
                Some("rank 1 has no route on rail 2".to_owned()),
                None,
                "RoCE queue-pair connect failed: rank 1 has no route on rail 2",
            ),
            (
                None,
                Some("the posting thread could not be created".to_owned()),
                "RoCE proxy thread failed to start: the posting thread could not be created",
            ),
        ] {
            let (state, _control, mut session) = shim();
            {
                let mut state = state.lock().unwrap();
                state.connect_error = connect_error;
                state.start_error = start_error;
                state.failed = false;
            }
            let error = session
                .connect(&shim_records())
                .expect_err("a failed setup must report its failure")
                .to_string();
            assert!(error.contains(expected), "{error}");
            assert!(session.terminal(), "{expected} must be terminal");
            let health = session.health().unwrap_err().to_string();
            assert!(health.contains("terminal"), "{health}");
            assert!(health.contains(expected), "{health}");
            assert_eq!(session.health().unwrap_err().to_string(), health);
        }
    }

    #[test]
    fn a_proxy_or_kernel_failure_poisons_health_for_good() {
        // The posting thread stopped.
        let (state, _control, mut session) = shim();
        session.connect(&shim_records()).unwrap();
        {
            let mut state = state.lock().unwrap();
            state.failed = true;
            state.error = "doorbell skipped 2 ops (last 1, now 3)".into();
        }
        let error = session.health().unwrap_err().to_string();
        assert!(error.contains("RoCE proxy failed: doorbell skipped"), "{error}");
        assert!(session.terminal());
        assert_eq!(session.health().unwrap_err().to_string(), error);

        // A kernel's wait timed out: the control record names the sequence and peer.
        let (_state, control, mut session) = shim();
        session.connect(&shim_records()).unwrap();
        control[2].store(37, Ordering::Release);
        control[3].store(1, Ordering::Relaxed);
        control[6].store(1, Ordering::Relaxed);
        let error = session.health().unwrap_err().to_string();
        assert!(error.contains("sequence 37"), "{error}");
        assert!(error.contains("transport is poisoned"), "{error}");
        assert!(session.terminal());
        // Clearing the kernel's word cannot make a poisoned transport healthy again.
        control[2].store(0, Ordering::Release);
        let later = session.health().unwrap_err().to_string();
        assert!(later.contains("sequence 37"), "{later}");
    }

    #[test]
    fn records_name_the_context_that_created_them() {
        let size = unsafe { roce_blob_bytes() } as usize;
        let (_state, _control, session) = shim();
        let first = session.local_record().unwrap();
        assert_eq!(&first[..4], &PROXY_ABI.to_le_bytes());
        assert_eq!(first.len(), size + 4);
        // A fresh context — a restarted rank's proxy — publishes different queue pairs and rkeys,
        // so a group may only connect records published after its own creation.
        let (state, _control, fresh) = shim();
        state.lock().unwrap().generation = 1;
        let second = fresh.local_record().unwrap();
        assert_ne!(first[4..], second[4..]);
    }

    #[test]
    fn dropping_a_session_destroys_its_context_once() {
        let (state, _control, session) = shim();
        drop(session);
        assert_eq!(state.lock().unwrap().destroys, 1);
    }

    #[test]
    fn pipelined_receive_region_is_single_copy() {
        let mut layout = [0; 17];
        // SAFETY: Layout calculation opens no devices and writes seventeen words.
        assert_eq!(
            unsafe { roce_layout(4, 4096, 4096, 4096, layout.as_mut_ptr()) },
            0
        );
        // ABI 15 keeps one copy of the pipelined rows and flags: phase per source rank, no slot.
        assert_eq!(layout[9] - layout[8], 4 * 2 * layout[10]);
        assert_eq!(layout[13] - layout[9], 2 * 4 * 64 * 2 * 128);
        let (region, mut spec, topology) = specification();
        // Mislabeling a core as ABI 15 must not bypass size validation.
        spec.region_bytes -= 4 * 2 * layout[10] + 2 * 4 * 64 * 2 * 128;
        // SAFETY: The malformed region is rejected before modifying it or opening devices.
        let Err(error) = (unsafe { RoceTransport::open(&spec, &topology) }) else {
            panic!("a short region must fail");
        };
        let error = error.to_string();
        assert!(error.contains("region differs"));
        assert!(region.iter().all(|byte| *byte == 0x5a));
    }

    #[test]
    fn peer_records_require_one_native_ring_protocol() {
        // SAFETY: This query has no arguments, device access, or allocation.
        let size = unsafe { roce_blob_bytes() } as usize;
        // ABI 15's connection record adds the ring geometry: 1184 bytes, not ABI 14's 1168.
        assert_eq!(size, 1184);
        let record = |abi: u32, value: u8| {
            let mut bytes = abi.to_le_bytes().to_vec();
            bytes.extend(vec![value; size]);
            bytes
        };
        let records = vec![record(PROXY_ABI, 0x12), record(PROXY_ABI, 0x34)];
        let expected = [vec![0x12; size], vec![0x34; size]].concat();
        assert_eq!(peer_blobs(&records, 2).unwrap(), expected);
        assert!(peer_blobs(&records, 3).is_err());
        for abi in [9, 10, 11, 12, 13, 14, 16] {
            let mut foreign = records.clone();
            foreign[1] = record(abi, 0x34);
            assert!(peer_blobs(&foreign, 2).is_err(), "ABI {abi}");
        }
        let mut unversioned = records.clone();
        unversioned[1] = vec![0; size];
        assert!(peer_blobs(&unversioned, 2).is_err());
        // A record of the previous size is refused even if its version word matches.
        let mut short = records.clone();
        short[1] = record(PROXY_ABI, 0x34);
        short[1].truncate(size + 3);
        assert!(peer_blobs(&short, 2).is_err());
        let mut truncated = records;
        truncated[1].truncate(3);
        assert!(peer_blobs(&truncated, 2).is_err());
    }

    #[test]
    fn rejects_incompatible_core_and_routes_before_modifying_region() {
        let (region, spec, topology) = specification();
        for case in 0..10 {
            let mut spec = spec.clone();
            let mut topology = topology.clone();
            match case {
                0 => spec.proxy_abi += 1,
                1 => spec.ctrl_offset += 128,
                2 => spec.region_bytes -= 128,
                3 => {
                    topology.routes.pop();
                }
                4 => topology.rails = 0,
                5 => topology.rails = 3,
                6 => topology.routes[0] = vec![(0, 0), (1, 1)],
                7 => topology.routes[1].clear(),
                8 => topology.routes[2] = vec![(0, 0), (1, 1)],
                9 => topology.routes[3][1].0 = 2,
                _ => unreachable!(),
            }
            // SAFETY: Every case fails preflight before accessing or registering the region.
            // The backing allocation nevertheless remains live throughout the call.
            let result = unsafe { RoceTransport::open(&spec, &topology) };
            assert!(
                result.is_err(),
                "invalid transport case {case} passed preflight"
            );
            assert!(region.iter().all(|byte| *byte == 0x5a));
        }
    }

    #[test]
    fn control_errors_stay_poisoned_across_health_checks() {
        let words: [AtomicU32; 8] = std::array::from_fn(|_| AtomicU32::new(0));
        check_control(2, &words).unwrap();
        // Diagnostic words without an error sequence do not poison the transport.
        words[3].store(1, Ordering::Relaxed);
        words[6].store(1, Ordering::Relaxed);
        words[7].store(INVALID_VARLEN_COUNT, Ordering::Relaxed);
        check_control(2, &words).unwrap();
        words[2].store(37, Ordering::Release);
        let error = check_control(2, &words).unwrap_err().to_string();
        assert!(error.contains("variable-length all-gather at sequence 37"));
        assert!(error.contains("rank 1"));
        assert!(error.contains("transport is poisoned"));
        words[7].store(0, Ordering::Relaxed);
        let error = check_control(2, &words).unwrap_err().to_string();
        assert!(error.contains("rank 2 timed out waiting for rank 1, HCA 1"));
        assert!(error.contains("sequence 37"));
        assert!(error.contains("transport is poisoned"));
        // A subsequent health poll cannot turn the failed sequence into success.
        assert!(check_control(2, &words).is_err());
    }
}
