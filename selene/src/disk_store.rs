/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Records retained as files, moved through registered host staging.
//!
//! [`DiskStore`] indexes the record files under one directory, accounts their
//! bytes against a limit, and evicts under the tree policy of the index: the
//! least recently used unpinned leaf is unlinked, and when every leaf is pinned
//! the least recently used single-child ancestor is compacted by truncating its
//! file at the row segment. Every byte movement goes through a [`DiskTransport`].
//! [`Staged`] streams records between GPU ids and files through a registered
//! shared slot pool, the shared transfer threads, and the disk thread, in windows of
//! at most [`WINDOW_PAGES`] pages. Outbound and caller-buffer pipelines stage at
//! most [`WINDOWS_IN_FLIGHT`] windows per record. Each device `fetch` holds at
//! most [`WINDOWS_IN_FLIGHT`] active window credits and runs plans out of order:
//! concurrent fetches reading the same record file share one open handle and
//! attach as followers to [`WINDOW_PAGES`]-aligned windows (staging raw v1–v3
//! records in shared slots or v4 packed records in shared `ReadPacked` buffers)
//! while those windows remain referenced, submitting one GPU upload per follower.
//! It also moves records between files and caller-owned host buffers retained by
//! the composite's memory tier. Submissions never block the caller (`store`,
//! `fetch`, `truncate`, and `unlink` enqueue work, and `poll` advances every
//! pipeline as free slots and queue space allow); the last user of a shared
//! segment holds its [`TransportEvent::Fetched`] event until the file close
//! settles on the disk thread so no disk read outlives the caller's index pin.

#[cfg(test)]
use crate::store::TierStore;

use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant, SystemTime},
};

use crate::levels::Rate;
use crate::{
    CheckpointId, Error, ExportRefusal, History, PageId, PrefixKey, Result, StateSlotId,
    index::NodeEvent,
    pool::{
        Announcer, FilePin, HostPin, IndexEntry, Releases, Tier as PoolTier, TierView,
        pins::{FileSpan, host_chain_kept},
    },
    shared::SliceSums,
};

use super::{
    CacheGeometry, HostHit, HostStore, INITIAL_STAGED_DISK_RATE, InboundTicket, JobId, Layout,
    PublishedCheckpoint, Retirement, Retiring, StoreEvent, StoreOutcome, StoreReason, StoreStats,
    Tier,
    copier::{self, Direction, PageCopier, SharedCopier},
    disk::{self, DiskThread, Operation},
    geometry::invalid,
    index::{Eviction, Index, Match, Node},
    record::{self, Header, Scanned},
    slab::{SlabPool, Slot, WINDOW_PAGES},
};

/// Active staging window credits per outbound record, host read, or device fetch.
pub const WINDOWS_IN_FLIGHT: usize = crate::shared::WINDOW_CREDITS as usize;
/// Records per direction a disk store keeps in flight. Further `store` calls
/// are skipped and further `fetch` calls refused until one completes, so GPU
/// pins and ticket ids held for the store stay bounded.
pub const RECORDS_IN_FLIGHT: usize = 8;
/// Leave room in the sharded transport's backlog for stores and fetches.
const MAINTENANCE_IN_FLIGHT: usize = crate::shard::BACKLOG / 2;

/// A record file as the disk index sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiskRecord {
    pub path: PathBuf,
    pub header: Header,
    /// Reserved or committed file length. Unreserved DRAM records and
    /// discarded writes consume zero disk bytes.
    pub bytes: u64,
    /// A failed record remains indexed while its children refer to it.
    failed: bool,
}

/// Completion the transport reports to the disk store.
#[derive(Debug)]
pub enum TransportEvent {
    Stored {
        id: super::JobId,
        result: Result<u64>,
        header: Option<Box<Header>>,
    },
    /// A failed store changed the destination on some ranks; its prior file is unusable.
    StoreInvalidated { id: super::JobId, error: Error },
    Fetched {
        id: super::JobId,
        result: Result<u64>,
    },
    Truncated {
        id: super::JobId,
        result: Result<()>,
    },
    Unlinked {
        id: super::JobId,
        result: Result<()>,
    },
}

/// Moves whole records between GPU ids and record files.
pub trait DiskTransport {
    /// Stream a published GPU checkpoint into a new record file at `temporary`,
    /// renamed to `path` on completion. The record's pages are the last
    /// `header.tail_pages` entries of the checkpoint's chain.
    fn store(
        &mut self,
        id: super::JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        checkpoint: &PublishedCheckpoint,
        detect: bool,
    ) -> Result<()>;
    /// Persist a zero-page Endpoint link without copying recurrent state.
    fn store_empty(
        &mut self,
        id: super::JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
    ) -> Result<()> {
        self.staged_mut()
            .ok_or_else(|| invalid("disk transport cannot store a header-only record"))?
            .write_from(id, header, temporary, path, Vec::new(), None, None)
    }
    /// Stream the chain into the ticket's ids: for each record, the page
    /// records `first..first + count` and, for the last record, the row.
    fn fetch(
        &mut self,
        id: super::JobId,
        plan: Vec<FetchSegment>,
        ticket: &InboundTicket,
    ) -> Result<()>;
    /// Drop the row segment of a committed record.
    fn truncate(&mut self, id: super::JobId, path: PathBuf, length: u64) -> Result<()>;
    fn unlink(&mut self, id: super::JobId, path: PathBuf) -> Result<()>;
    /// Report completed jobs. `Err` means this transport can no longer report
    /// them: a job it accepted may still run, so the caller must stop rather
    /// than reuse the ids that job holds.
    fn poll(&mut self, events: &mut Vec<TransportEvent>) -> Result<()>;
    /// Drain physical inbound window read samples `(bytes, elapsed)` for rate tracking.
    fn drain_inbound_samples(&mut self, _samples: &mut Vec<(u64, Duration)>) {}
    fn close(&mut self) -> Result<()>;
    fn staged(&self) -> Option<&Staged> {
        None
    }
    /// The staged transport behind this handle, for the composite store's
    /// transfers between files and its memory slots.
    fn staged_mut(&mut self) -> Option<&mut Staged> {
        None
    }
    fn codec_receive_profiles(&self) -> Vec<[u8; 32]> {
        Vec::new()
    }
    fn decode_counters(&self) -> Option<crate::codec_copy::DecodeCounters> {
        None
    }
}

/// One record's contribution to a fetch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchSegment {
    pub path: PathBuf,
    pub header: Header,
    pub first_page: u32,
    pub pages: u32,
    pub row: bool,
}

fn lock<'a>(
    transfers: &'a Mutex<dyn PageCopier + 'static>,
) -> MutexGuard<'a, dyn PageCopier + 'static> {
    transfers
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Host addresses of one window's page and row records, with the staging
/// slots behind them when the transport allocated them.
struct Buffers {
    pages: Vec<usize>,
    row: Option<Vec<usize>>,
    slots: Option<(Vec<Slot>, Option<Vec<Slot>>)>,
}

/// The threads and pool every pipeline draws on, with the counters that keep
/// submissions within the queues' depths so a refusal is a failure, not backpressure.
struct Resources {
    /// Declared before `pool` so Drop joins `DiskThread` before unmapping pinned slabs.
    disk: DiskThread,
    /// Shared slots for device transfers; host-buffer-only users may omit them.
    pool: Option<SlabPool>,
    transfers: SharedCopier,
    /// Disk jobs submitted whose replies have not been polled.
    disk_outstanding: usize,
    next_disk: u64,
    /// Physical `disk::Job::Open` calls submitted by this transport.
    #[cfg(any(test, feature = "test-support"))]
    disk_opens: u64,
    /// Physical disk window reads submitted by this transport.
    #[cfg(any(test, feature = "test-support"))]
    disk_window_reads: u64,
    /// Logical device-fetch windows satisfied by attaching to an in-flight shared window.
    #[cfg(any(test, feature = "test-support"))]
    coalesced_windows: u64,
}

impl Resources {
    fn disk_id(&mut self) -> JobId {
        let id = JobId(self.next_disk);
        self.next_disk += 1;
        id
    }

    fn disk_space(&self) -> bool {
        self.disk_outstanding < disk::QUEUE_DEPTH
    }

    fn submit_disk(&mut self, job: disk::Job) -> Result<()> {
        self.disk.submit(job)?;
        self.disk_outstanding += 1;
        Ok(())
    }

    fn transfer_space(&self) -> bool {
        lock(&self.transfers).in_flight() < copier::QUEUE_DEPTH
    }

    fn submit_transfer(&mut self, job: copier::Job) -> Result<()> {
        lock(&self.transfers).submit(job)
    }

    fn codec_enabled(&self) -> bool {
        lock(&self.transfers).codec_profile().is_some()
    }

    /// Free slots in the shared pool; zero when this transport has no pool.
    fn free_slots(&self) -> usize {
        self.pool.as_ref().map_or(0, SlabPool::free_slots)
    }

    /// Slots for `pages` page records and, when `row`, all row pieces;
    /// `None` leaves the pool unchanged when it cannot supply all of them.
    fn allocate(&mut self, pages: usize, row: bool) -> Option<Buffers> {
        let pool = self.pool.as_mut()?;
        let needed = lock(&self.transfers).layout().record_slots(pages, row);
        let mut slots = pool.allocate(needed)?;
        let row_slots = slots.split_off(pages);
        let row = row.then_some(row_slots);
        Some(Buffers {
            pages: pool.addresses(&slots),
            row: row.as_ref().map(|slots| pool.addresses(slots)),
            slots: Some((slots, row)),
        })
    }

    fn release(&mut self, buffers: &mut Buffers) {
        if let (Some(pool), Some((pages, row))) = (self.pool.as_ref(), buffers.slots.take()) {
            pool.release(pages.into_iter().chain(row.into_iter().flatten()));
        }
    }
}

/// The store job a disk job belongs to.
#[derive(Clone, Copy)]
enum Owner {
    Outbound(JobId),
    HostInbound(JobId, usize),
    SharedSegment(SharedSegmentId),
    Truncate(JobId),
    Unlink(JobId),
}

/// Where an outbound record's bytes come from.
enum Source {
    /// GPU ids and the row's state slot, staged through the pool.
    Device {
        pages: Vec<PageId>,
        slot: StateSlotId,
        detect: bool,
    },
    /// Caller-owned registered buffers already holding every record, and their checksums if known.
    Host {
        pages: Vec<usize>,
        row: Option<Vec<usize>>,
        sums: Option<SliceSums>,
    },
    /// Already-encoded packed units residing in caller-pinned DRAM slabs.
    HostPacked {
        units: Option<Vec<crate::shared::wire::PackedUnit>>,
        spans: Vec<(usize, usize)>,
        row: bool,
    },
}

fn build_packed_write_pinned(
    disk: JobId,
    temporary: PathBuf,
    path: PathBuf,
    header: Header,
    units: Vec<crate::shared::wire::PackedUnit>,
    spans: Vec<(usize, usize)>,
    row: bool,
) -> Result<disk::Job> {
    use crate::shared::wire::{PackedOffer, packed_unit::Representation};
    let expected_bytes: usize = units
        .iter()
        .try_fold(0usize, |acc, u| {
            usize::try_from(u.stored_length)
                .ok()
                .and_then(|n| acc.checked_add(n))
        })
        .ok_or_else(|| invalid("pinned packed payload length overflow"))?;
    let actual_bytes: usize = spans.iter().map(|&(_, len)| len).sum();
    if actual_bytes != expected_bytes {
        return Err(invalid(format!(
            "pinned packed payload spans total {actual_bytes} bytes; units expect {expected_bytes}"
        )));
    }
    let ordered = units
        .windows(2)
        .all(|w| (w[0].kind, w[0].logical_index, w[0].stored_rank) <= (w[1].kind, w[1].logical_index, w[1].stored_rank));
    let any_packet = units
        .iter()
        .any(|u| u.representation == Representation::Packet as i32);
    if any_packet && ordered {
        let first_page = header
            .parent
            .map_or(0, |p| p.consumed_tokens / header.page_tokens);
        let packed_header = header.with_packed(PackedOffer { first_page, units })?;
        let metadata = packed_header.packed_metadata()?;
        return Ok(disk::Job::WritePacked {
            id: disk,
            temporary,
            path,
            header: packed_header,
            metadata,
            payloads: disk::PackedPayload::Pinned(spans),
            sync: false,
        });
    }
    // Rare fallback (all-literal or unsorted units): materialize owned buffers once
    // so `build_packed_write` can sort and emit a canonical raw v3 record.
    let mut flat = Vec::with_capacity(actual_bytes);
    for &(addr, len) in &spans {
        // SAFETY: caller pins the source slabs until the disk write completes.
        flat.extend_from_slice(unsafe { std::slice::from_raw_parts(addr as *const u8, len) });
    }
    let mut cursor = 0usize;
    let mut slots = Vec::with_capacity(units.len());
    for unit in units {
        let len = unit.stored_length as usize;
        slots.push(crate::codec_copy::CaptureSlot {
            unit,
            bytes: flat[cursor..cursor + len].to_vec(),
        });
        cursor += len;
    }
    build_packed_write(disk, temporary, path, header, slots, row)
}

fn build_packed_write(
    disk: JobId,
    temporary: PathBuf,
    path: PathBuf,
    header: Header,
    mut slots: Vec<crate::codec_copy::CaptureSlot>,
    row: bool,
) -> Result<disk::Job> {
    use crate::shared::wire::{
        PackedOffer,
        packed_unit::{Kind, Representation},
    };
    slots.sort_by_key(|s| (s.unit.kind, s.unit.logical_index, s.unit.stored_rank));
    let any_packet = slots
        .iter()
        .any(|s| s.unit.representation == Representation::Packet as i32);
    if any_packet {
        let first_page = header
            .parent
            .map_or(0, |p| p.consumed_tokens / header.page_tokens);
        let units = slots.iter().map(|s| s.unit.clone()).collect();
        let packed_header = header.with_packed(PackedOffer { first_page, units })?;
        let metadata = packed_header.packed_metadata()?;
        let payloads = disk::PackedPayload::Owned(slots.into_iter().map(|s| s.bytes).collect());
        return Ok(disk::Job::WritePacked {
            id: disk,
            temporary,
            path,
            header: packed_header,
            metadata,
            payloads,
            sync: false,
        });
    }
    // All units fell back to Literal: verify each captured literal checksum before
    // emitting fresh whole-page/row raw v3 checksums so in-DRAM corruption cannot pass.
    for slot in &slots {
        if record::checksum(&slot.bytes) != slot.unit.raw_xxh3 {
            return Err(Error::carrying(crate::shared::packed::Checksum::Stored));
        }
    }
    let mut raw_header = header;
    raw_header.set_replication(Default::default())?;
    let ranks = raw_header.ranks.len();
    let tail = raw_header.tail_pages as usize;
    let (page_slots, row_slots): (Vec<_>, Vec<_>) = slots
        .into_iter()
        .partition(|s| s.unit.kind == Kind::Page as i32);
    let mut expanded_pages: Vec<Vec<u8>> = Vec::with_capacity(tail + usize::from(row));
    let mut checksums = vec![0u64; tail];
    for page in 0..tail {
        let mut by_rank: Vec<Option<&[u8]>> = vec![None; ranks];
        for slot in &page_slots {
            if slot.unit.logical_index
                == raw_header
                    .parent
                    .map_or(0, |p| p.consumed_tokens / raw_header.page_tokens)
                    + page as u64
            {
                by_rank[slot.unit.stored_rank as usize] = Some(&slot.bytes);
                for &replica in &slot.unit.replicas {
                    by_rank[replica as usize] = Some(&slot.bytes);
                }
            }
        }
        let mut page_buf = Vec::with_capacity(raw_header.page_record_bytes() as usize);
        for chunk in by_rank {
            let bytes = chunk.ok_or_else(|| invalid("missing literal rank share for page"))?;
            page_buf.extend_from_slice(bytes);
        }
        checksums[page] = record::checksum(&page_buf);
        expanded_pages.push(page_buf);
    }
    if row {
        let mut by_rank: Vec<Option<&[u8]>> = vec![None; ranks];
        for slot in &row_slots {
            by_rank[slot.unit.stored_rank as usize] = Some(&slot.bytes);
            for &replica in &slot.unit.replicas {
                by_rank[replica as usize] = Some(&slot.bytes);
            }
        }
        let mut row_buf = Vec::with_capacity(raw_header.row_record_bytes() as usize);
        for chunk in by_rank {
            let bytes = chunk.ok_or_else(|| invalid("missing literal rank share for row"))?;
            row_buf.extend_from_slice(bytes);
        }
        raw_header.row_checksum = record::checksum(&row_buf);
        expanded_pages.push(row_buf);
    } else {
        raw_header.row_checksum = 0;
    }
    let checksum_bytes = raw_header.checksum_bytes() as usize;
    let mut metadata = vec![0u8; record::HEADER_BYTES + checksum_bytes];
    metadata[..record::HEADER_BYTES].copy_from_slice(&raw_header.encode()?);
    for (slot, sum) in metadata[record::HEADER_BYTES..]
        .chunks_exact_mut(8)
        .zip(&checksums)
    {
        slot.copy_from_slice(&sum.to_le_bytes());
    }
    Ok(disk::Job::WritePacked {
        id: disk,
        temporary,
        path,
        header: raw_header,
        metadata,
        payloads: disk::PackedPayload::Owned(expanded_pages),
        sync: false,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TemporaryFile {
    Unsubmitted,
    Creating,
    Open,
    /// Never created, removed by a failed commit, or removed by an abort.
    Absent,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OutPhase {
    Transferring,
    Transferred,
    Writing,
}

struct OutWindow {
    index: u32,
    first: u32,
    buffers: Buffers,
    phase: OutPhase,
    pages_submitted: bool,
    row_submitted: bool,
    /// Jobs that may still touch the buffers.
    referenced: usize,
}

struct OutboundRecord {
    disk: JobId,
    header: Header,
    temporary: PathBuf,
    path: PathBuf,
    source: Source,
    packed: bool,
    captures: Vec<Arc<crate::codec_copy::CaptureWindow>>,
    windows: u32,
    next_window: u32,
    active: Vec<OutWindow>,
    written: u32,
    file: TemporaryFile,
    commit_submitted: bool,
    abort_submitted: bool,
    /// Jobs submitted for this record whose replies are awaited.
    outstanding: usize,
    committed: Option<u64>,
    error: Option<Error>,
}

impl OutboundRecord {
    fn new(
        disk: JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        source: Source,
        packed: bool,
    ) -> Self {
        let windows = if matches!(source, Source::HostPacked { .. }) {
            0
        } else {
            (header.tail_pages as usize).div_ceil(WINDOW_PAGES).max(1) as u32
        };
        Self {
            disk,
            header,
            temporary,
            path,
            source,
            packed,
            captures: Vec::new(),
            windows,
            next_window: 0,
            active: Vec::with_capacity(WINDOWS_IN_FLIGHT),
            written: 0,
            file: if packed {
                TemporaryFile::Absent
            } else {
                TemporaryFile::Unsubmitted
            },
            commit_submitted: false,
            abort_submitted: false,
            outstanding: 0,
            committed: None,
            error: None,
        }
    }

    fn fail(&mut self, error: Error) {
        self.error.get_or_insert(error);
    }

    fn apply_disk(&mut self, reply: disk::Reply) {
        self.outstanding = self.outstanding.saturating_sub(1);
        match reply.operation {
            Operation::Create => match reply.result {
                Ok(()) => self.file = TemporaryFile::Open,
                Err(error) => {
                    self.file = TemporaryFile::Absent;
                    self.fail(error);
                }
            },
            Operation::WritePages | Operation::WriteRow => {
                let window = self.active.iter_mut().find(|window| {
                    if reply.operation == Operation::WritePages {
                        window.index == reply.window
                    } else {
                        window.buffers.row.is_some()
                    }
                });
                if let Some(window) = window {
                    window.referenced = window.referenced.saturating_sub(1);
                }
                if let Err(error) = reply.result {
                    self.fail(error);
                }
            }
            Operation::Commit | Operation::WritePacked => match reply.result {
                Ok(()) => {
                    if let Some(header) = reply.header {
                        self.header = header;
                    }
                    self.committed = Some(reply.bytes);
                }
                Err(error) => {
                    self.file = TemporaryFile::Absent;
                    self.fail(error);
                }
            },
            Operation::Abort => self.file = TemporaryFile::Absent,
            _ => {}
        }
    }

    fn apply_transfer(&mut self, reply: copier::Reply) {
        self.outstanding = self.outstanding.saturating_sub(1);
        let Some(window) = self
            .active
            .iter_mut()
            .find(|window| window.index == reply.window)
        else {
            return;
        };
        window.referenced = window.referenced.saturating_sub(1);
        match reply.result {
            Ok(()) => window.phase = OutPhase::Transferred,
            Err(error) => self.fail(error),
        }
    }

    /// Advance as far as slots and queue space allow; `Some` when the record
    /// is finished and nothing references its buffers or its temporary file.
    fn pump(&mut self, id: JobId, resources: &mut Resources) -> Option<Result<u64>> {
        if let Some(bytes) = self.committed {
            return Some(Ok(bytes));
        }
        if self.packed {
            if self.error.is_some() {
                return self.unwind(resources);
            }
            if let Source::HostPacked { units, spans, row } = &mut self.source {
                if !self.commit_submitted && resources.disk_space() {
                    let taken_units = units.take().unwrap_or_default();
                    let taken_spans = std::mem::take(spans);
                    let has_row = *row;
                    match build_packed_write_pinned(
                        self.disk,
                        self.temporary.clone(),
                        self.path.clone(),
                        self.header.clone(),
                        taken_units,
                        taken_spans,
                        has_row,
                    )
                    .and_then(|job| resources.submit_disk(job))
                    {
                        Ok(()) => {
                            self.commit_submitted = true;
                            self.outstanding += 1;
                        }
                        Err(error) => {
                            self.fail(error);
                            return self.unwind(resources);
                        }
                    }
                }
                return None;
            }
            let mut index = 0;
            while index < self.active.len() {
                let window = &self.active[index];
                if window.phase == OutPhase::Transferred && window.referenced == 0 {
                    self.active.remove(index);
                    self.written += 1;
                } else {
                    index += 1;
                }
            }
            while self.active.len() < WINDOWS_IN_FLIGHT && self.next_window < self.windows {
                match self.start_window(id, resources) {
                    Ok(true) => {}
                    Ok(false) => break,
                    Err(error) => {
                        self.fail(error);
                        return self.unwind(resources);
                    }
                }
            }
            if self.written == self.windows
                && !self.commit_submitted
                && resources.disk_space()
            {
                let mut slots = Vec::new();
                for cap in &self.captures {
                    slots.extend(cap.take_sorted());
                }
                match build_packed_write(
                    self.disk,
                    self.temporary.clone(),
                    self.path.clone(),
                    self.header.clone(),
                    slots,
                    true,
                )
                .and_then(|job| resources.submit_disk(job))
                {
                    Ok(()) => {
                        self.commit_submitted = true;
                        self.outstanding += 1;
                    }
                    Err(error) => {
                        self.fail(error);
                        return self.unwind(resources);
                    }
                }
            }
            return None;
        }
        if self.file == TemporaryFile::Unsubmitted {
            if !resources.disk_space() {
                return None;
            }
            let job = disk::Job::Create {
                id: self.disk,
                temporary: self.temporary.clone(),
                header: self.header.clone(),
                row: match &self.source {
                    Source::Host { row, .. } => row.is_some(),
                    Source::Device { .. } => true,
                    Source::HostPacked { row, .. } => *row,
                },
                sums: match &self.source {
                    Source::Host { sums, .. } => sums.clone(),
                    Source::Device { .. } | Source::HostPacked { .. } => None,
                },
                replication: match &self.source {
                    Source::Host { .. } | Source::HostPacked { .. } => {
                        Some(self.header.replication())
                    }
                    Source::Device { detect, .. } => (!detect).then_some(Default::default()),
                },
            };
            match resources.submit_disk(job) {
                Ok(()) => {
                    self.file = TemporaryFile::Creating;
                    self.outstanding += 1;
                }
                Err(error) => {
                    self.file = TemporaryFile::Absent;
                    self.fail(error);
                }
            }
        }
        if self.error.is_some() {
            return self.unwind(resources);
        }
        let mut index = 0;
        while index < self.active.len() {
            let window = &self.active[index];
            if window.phase == OutPhase::Writing && window.referenced == 0 {
                let mut window = self.active.remove(index);
                resources.release(&mut window.buffers);
                self.written += 1;
            } else {
                index += 1;
            }
        }
        while self.active.len() < WINDOWS_IN_FLIGHT && self.next_window < self.windows {
            match self.start_window(id, resources) {
                Ok(true) => {}
                Ok(false) => break,
                Err(error) => {
                    self.fail(error);
                    return self.unwind(resources);
                }
            }
        }
        for index in 0..self.active.len() {
            if self.active[index].phase == OutPhase::Transferred
                && let Err(error) = self.submit_writes(index, resources)
            {
                self.fail(error);
                return self.unwind(resources);
            }
        }
        if self.written == self.windows
            && !self.commit_submitted
            && self.file == TemporaryFile::Open
            && resources.disk_space()
        {
            let job = disk::Job::Commit {
                id: self.disk,
                path: self.path.clone(),
                sync: false,
            };
            match resources.submit_disk(job) {
                Ok(()) => {
                    self.commit_submitted = true;
                    self.outstanding += 1;
                }
                Err(error) => {
                    self.fail(error);
                    return self.unwind(resources);
                }
            }
        }
        None
    }

    /// Stage the next window: allocate its slots and copy it out of the GPU,
    /// or take the caller's buffers. `Ok(false)` holds the window until slots
    /// or queue space free up.
    fn start_window(&mut self, id: JobId, resources: &mut Resources) -> Result<bool> {
        let index = self.next_window;
        let first = index as usize * WINDOW_PAGES;
        let count = (self.header.tail_pages as usize - first).min(WINDOW_PAGES);
        let last = index + 1 == self.windows;
        let (buffers, phase, referenced) = match &self.source {
            Source::Device { pages, slot, .. } if self.packed => {
                if !resources.transfer_space() {
                    return Ok(false);
                }
                let first_logical = self
                    .header
                    .parent
                    .map_or(0, |p| p.consumed_tokens / self.header.page_tokens)
                    + first as u64;
                let win = crate::codec_copy::Window::capture(first_logical);
                let capture = win.capture.clone().unwrap();
                let job = copier::Job {
                    codec: Some(win),
                    replication: Default::default(),
                    id,
                    window: index,
                    direction: Direction::Out,
                    pages: pages[first..first + count]
                        .iter()
                        .copied()
                        .map(|p| (p, 0))
                        .collect(),
                    row: last.then_some((*slot, vec![0])),
                };
                resources.submit_transfer(job)?;
                self.captures.push(capture);
                self.outstanding += 1;
                (
                    Buffers {
                        pages: Vec::new(),
                        row: None,
                        slots: None,
                    },
                    OutPhase::Transferring,
                    1,
                )
            }
            Source::Device { pages, slot, .. } => {
                if !resources.transfer_space() {
                    return Ok(false);
                }
                let Some(mut buffers) = resources.allocate(count, last) else {
                    return Ok(false);
                };
                let job = copier::Job {
                    codec: None,
                    replication: Default::default(),
                    id,
                    window: index,
                    direction: Direction::Out,
                    pages: pages[first..first + count]
                        .iter()
                        .copied()
                        .zip(buffers.pages.iter().copied())
                        .collect(),
                    row: buffers.row.clone().map(|address| (*slot, address)),
                };
                if let Err(error) = resources.submit_transfer(job) {
                    resources.release(&mut buffers);
                    return Err(error);
                }
                self.outstanding += 1;
                (buffers, OutPhase::Transferring, 1)
            }
            Source::Host { pages, row, .. } => (
                Buffers {
                    pages: pages[first..first + count].to_vec(),
                    row: row.clone().filter(|_| last),
                    slots: None,
                },
                OutPhase::Transferred,
                0,
            ),
            Source::HostPacked { .. } => unreachable!("HostPacked submits directly in pump"),
        };
        self.active.push(OutWindow {
            index,
            first: first as u32,
            buffers,
            phase,
            pages_submitted: false,
            row_submitted: false,
            referenced,
        });
        self.next_window += 1;
        Ok(true)
    }

    fn submit_writes(&mut self, index: usize, resources: &mut Resources) -> Result<()> {
        let window = &mut self.active[index];
        if !window.buffers.pages.is_empty() && !window.pages_submitted {
            if !resources.disk_space() {
                return Ok(());
            }
            resources.submit_disk(disk::Job::WritePages {
                id: self.disk,
                window: window.index,
                first: window.first,
                buffers: window.buffers.pages.clone(),
            })?;
            window.pages_submitted = true;
            window.referenced += 1;
            self.outstanding += 1;
        }
        if let Some(buffers) = window.buffers.row.clone()
            && !window.row_submitted
        {
            if !resources.disk_space() {
                return Ok(());
            }
            resources.submit_disk(disk::Job::WriteRow {
                id: self.disk,
                buffers,
            })?;
            window.row_submitted = true;
            window.referenced += 1;
            self.outstanding += 1;
        }
        if (window.buffers.pages.is_empty() || window.pages_submitted)
            && (window.buffers.row.is_none() || window.row_submitted)
        {
            window.phase = OutPhase::Writing;
        }
        Ok(())
    }

    /// Release every window no job references; once nothing is outstanding,
    /// abort the temporary file and report the failure after the abort replies.
    fn unwind(&mut self, resources: &mut Resources) -> Option<Result<u64>> {
        self.active.retain_mut(|window| {
            if window.referenced == 0 {
                resources.release(&mut window.buffers);
                false
            } else {
                true
            }
        });
        if self.outstanding > 0 || !self.active.is_empty() {
            return None;
        }
        match self.file {
            TemporaryFile::Open if !self.abort_submitted => {
                if !resources.disk_space() {
                    return None;
                }
                match resources.submit_disk(disk::Job::Abort { id: self.disk }) {
                    Ok(()) => {
                        self.abort_submitted = true;
                        self.outstanding += 1;
                        None
                    }
                    Err(_) => Some(Err(self.error.clone()?)),
                }
            }
            TemporaryFile::Absent => Some(Err(self.error.clone()?)),
            _ => None,
        }
    }
}

/// Where a private host-buffer inbound read's bytes go.
enum HostDestination {
    /// Caller-owned registered buffers.
    Slots {
        pages: Vec<usize>,
        row: Option<Vec<usize>>,
    },
    /// Contiguous packed payload for a single v4 record, handed back to the caller.
    Packed,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FileState {
    Unopened,
    Opened,
    Closing,
    Closed,
}

struct Segment {
    disk: JobId,
    path: PathBuf,
    /// The indexed header the disk thread validates before reading payloads.
    expected: Header,
    state: FileState,
    /// Windows of this segment whose reads have not completed.
    reads_left: usize,
}

impl Segment {
    fn settled(&self) -> bool {
        matches!(self.state, FileState::Unopened | FileState::Closed)
    }
}

/// One window of a fetch: `count` page records from `first` of one segment,
/// landing at `offset` in the destination pages, plus the row on the last.
#[derive(Clone, Copy)]
struct WindowPlan {
    segment: usize,
    index: u32,
    first: u32,
    count: usize,
    offset: usize,
    row: bool,
}

/// Split segments of `(first page, page count, row)` into staging windows.
/// Caller-owned `read_into` keeps `align_to_window = false` so each host read packs
/// segment-relative windows without splitting an unaligned suffix into an extra job.
fn plan_windows(
    segments: impl IntoIterator<Item = (u32, usize, bool)>,
    align_to_window: bool,
) -> Vec<WindowPlan> {
    let mut plans = Vec::new();
    let mut offset = 0;
    for (segment, (first, pages, row)) in segments.into_iter().enumerate() {
        if pages == 0 {
            if row {
                plans.push(WindowPlan {
                    segment,
                    index: plans.len() as u32,
                    first,
                    count: 0,
                    offset,
                    row: true,
                });
            }
            continue;
        }
        let mut cursor = first as usize;
        let end = cursor + pages;
        let mut dst_offset = offset;
        while cursor < end {
            let limit = if align_to_window {
                ((cursor / WINDOW_PAGES) + 1) * WINDOW_PAGES
            } else {
                cursor + WINDOW_PAGES
            };
            let chunk_end = end.min(limit);
            let count = chunk_end - cursor;
            plans.push(WindowPlan {
                segment,
                index: plans.len() as u32,
                first: cursor as u32,
                count,
                offset: dst_offset,
                row: row && chunk_end == end,
            });
            cursor = chunk_end;
            dst_offset += count;
        }
        offset += pages;
    }
    plans
}

fn is_packed_content_error(error: &Error) -> bool {
    error.carried().is_some_and(|cause| {
        cause
            .downcast_ref::<crate::shared::packed::Checksum>()
            .is_some()
            || cause
                .downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::InvalidData)
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HostInPhase {
    Reading,
    Read,
}

struct HostWindow {
    plan: WindowPlan,
    pages: Vec<usize>,
    row: Option<Vec<usize>>,
    packed_buf: Option<Arc<Vec<u8>>>,
    phase: HostInPhase,
    pages_submitted: bool,
    row_submitted: bool,
    #[cfg(any(test, feature = "test-support"))]
    disk_read_counted: bool,
    referenced: usize,
}

/// An isolated host-buffer inbound read (`Staged::read_into` or
/// `Staged::read_into_packed`) targeting caller-owned buffers rather than the
/// shared device staging pool.
struct HostInbound {
    segments: Vec<Segment>,
    destination: HostDestination,
    plans: Vec<WindowPlan>,
    next_window: usize,
    active: Vec<HostWindow>,
    done: usize,
    outstanding: usize,
    bytes: u64,
    completed_packed: Option<(Header, Arc<Vec<u8>>)>,
    error: Option<Error>,
}

impl HostInbound {
    fn new(segments: Vec<Segment>, destination: HostDestination, plans: Vec<WindowPlan>) -> Self {
        Self {
            segments,
            destination,
            plans,
            next_window: 0,
            active: Vec::with_capacity(WINDOWS_IN_FLIGHT),
            done: 0,
            outstanding: 0,
            bytes: 0,
            completed_packed: None,
            error: None,
        }
    }

    fn fail(&mut self, error: Error) {
        self.error.get_or_insert(error);
    }

    fn apply_disk(&mut self, segment: usize, reply: disk::Reply) {
        self.outstanding = self.outstanding.saturating_sub(1);
        let Some(file) = self.segments.get_mut(segment) else {
            return;
        };
        match reply.operation {
            Operation::Open => match reply.result {
                Ok(()) => {}
                Err(error) => {
                    file.state = FileState::Closed;
                    self.fail(error.with_record_read());
                }
            },
            Operation::ReadPacked => {
                let window = self
                    .active
                    .iter_mut()
                    .find(|window| {
                        window.plan.segment == segment && window.plan.index == reply.window
                    });
                let mut finished = false;
                if let Some(window) = window {
                    window.referenced = window.referenced.saturating_sub(1);
                    window.packed_buf = reply.payload;
                    if window.phase == HostInPhase::Reading
                        && window.pages_submitted
                        && window.row_submitted
                        && window.referenced == 0
                    {
                        window.phase = HostInPhase::Read;
                        finished = true;
                    }
                }
                if finished {
                    file.reads_left = file.reads_left.saturating_sub(1);
                }
                match reply.result {
                    Ok(()) => self.bytes += reply.bytes,
                    Err(error) => self.fail(error.with_record_read()),
                }
            }
            Operation::ReadPages | Operation::ReadRow => {
                let window = self.active.iter_mut().find(|window| {
                    window.plan.segment == segment
                        && if reply.operation == Operation::ReadPages {
                            window.plan.index == reply.window
                        } else {
                            window.plan.row
                        }
                });
                let mut finished = false;
                if let Some(window) = window {
                    window.referenced = window.referenced.saturating_sub(1);
                    if window.phase == HostInPhase::Reading
                        && window.pages_submitted
                        && window.row_submitted
                        && window.referenced == 0
                    {
                        window.phase = HostInPhase::Read;
                        finished = true;
                    }
                }
                if finished {
                    file.reads_left = file.reads_left.saturating_sub(1);
                }
                match reply.result {
                    Ok(()) => self.bytes += reply.bytes,
                    Err(error) => self.fail(error.with_record_read()),
                }
            }
            Operation::Close => file.state = FileState::Closed,
            _ => {}
        }
    }

    fn pump(&mut self, resources: &mut Resources) -> Option<Result<u64>> {
        if self.error.is_some() {
            return self.unwind(resources);
        }
        let mut index = 0;
        while index < self.active.len() {
            if self.active[index].phase == HostInPhase::Read {
                let mut window = self.active.remove(index);
                if matches!(self.destination, HostDestination::Packed)
                    && let Some(buf) = window.packed_buf.take()
                {
                    let header = self.segments[window.plan.segment].expected.clone();
                    self.completed_packed = Some((header, buf));
                }
                self.done += 1;
            } else {
                index += 1;
            }
        }
        while self.active.len() < WINDOWS_IN_FLIGHT && self.next_window < self.plans.len() {
            match self.start_window(resources) {
                Ok(true) => {}
                Ok(false) => break,
                Err(error) => {
                    self.fail(error);
                    return self.unwind(resources);
                }
            }
        }
        for index in 0..self.active.len() {
            if self.active[index].phase == HostInPhase::Reading
                && let Err(error) = self.submit_reads(index, resources)
            {
                self.fail(error);
                return self.unwind(resources);
            }
        }
        for segment in &mut self.segments {
            if segment.state == FileState::Opened
                && segment.reads_left == 0
                && resources.disk_space()
            {
                match resources.submit_disk(disk::Job::Close { id: segment.disk }) {
                    Ok(()) => {
                        segment.state = FileState::Closing;
                        self.outstanding += 1;
                    }
                    Err(error) => {
                        self.fail(error);
                        return self.unwind(resources);
                    }
                }
            }
        }
        if self.done == self.plans.len() && self.segments.iter().all(Segment::settled) {
            return Some(Ok(self.bytes));
        }
        None
    }

    fn start_window(&mut self, resources: &mut Resources) -> Result<bool> {
        let plan = self.plans[self.next_window];
        let segment = &mut self.segments[plan.segment];
        if segment.state == FileState::Unopened {
            if !resources.disk_space() {
                return Ok(false);
            }
            resources.submit_disk(disk::Job::Open {
                id: segment.disk,
                path: segment.path.clone(),
                expected: segment.expected.clone(),
            })?;
            #[cfg(any(test, feature = "test-support"))]
            {
                resources.disk_opens += 1;
            }
            segment.state = FileState::Opened;
            self.outstanding += 1;
        }
        let (pages, row) = if segment.expected.packed.is_some() {
            (Vec::new(), None)
        } else {
            match &self.destination {
                HostDestination::Slots { pages, row } => (
                    pages[plan.offset..plan.offset + plan.count].to_vec(),
                    if plan.row { row.clone() } else { None },
                ),
                HostDestination::Packed => (Vec::new(), None),
            }
        };
        self.active.push(HostWindow {
            plan,
            pages,
            row,
            packed_buf: None,
            phase: HostInPhase::Reading,
            pages_submitted: false,
            row_submitted: false,
            #[cfg(any(test, feature = "test-support"))]
            disk_read_counted: false,
            referenced: 0,
        });
        self.next_window += 1;
        Ok(true)
    }

    fn submit_reads(&mut self, index: usize, resources: &mut Resources) -> Result<()> {
        let window = &mut self.active[index];
        let seg = &self.segments[window.plan.segment];
        let disk = seg.disk;
        if let Some(layout) = &seg.expected.packed {
            if !window.pages_submitted {
                if !resources.disk_space() {
                    return Ok(());
                }
                let first = window.plan.first as usize;
                let count = window.plan.count;
                let start_rel = if count == 0 && window.plan.row {
                    layout.page_offset(layout.page_count())
                } else {
                    layout.page_offset(first)
                };
                let end_rel = if window.plan.row {
                    layout.physical_bytes()
                } else {
                    layout.page_offset(first + count)
                };
                let offset = seg.expected.page_segment + start_rel;
                let length = usize::try_from(end_rel - start_rel)
                    .map_err(|_| invalid("packed window length overflow"))?;
                resources.submit_disk(disk::Job::ReadPacked {
                    id: disk,
                    window: window.plan.index,
                    offset,
                    length,
                })?;
                #[cfg(any(test, feature = "test-support"))]
                if !window.disk_read_counted {
                    window.disk_read_counted = true;
                    resources.disk_window_reads += 1;
                }
                window.pages_submitted = true;
                window.row_submitted = true;
                window.referenced += 1;
                self.outstanding += 1;
            }
            return Ok(());
        }
        if !window.pages_submitted {
            if window.pages.is_empty() {
                window.pages_submitted = true;
            } else {
                if !resources.disk_space() {
                    return Ok(());
                }
                resources.submit_disk(disk::Job::ReadPages {
                    id: disk,
                    window: window.plan.index,
                    first: window.plan.first,
                    buffers: window.pages.clone(),
                })?;
                #[cfg(any(test, feature = "test-support"))]
                if !window.disk_read_counted {
                    window.disk_read_counted = true;
                    resources.disk_window_reads += 1;
                }
                window.pages_submitted = true;
                window.referenced += 1;
                self.outstanding += 1;
            }
        }
        if !window.row_submitted {
            match window.row.clone() {
                None => window.row_submitted = true,
                Some(buffers) => {
                    if !resources.disk_space() {
                        return Ok(());
                    }
                    resources.submit_disk(disk::Job::ReadRow { id: disk, buffers })?;
                    #[cfg(any(test, feature = "test-support"))]
                    if !window.disk_read_counted {
                        window.disk_read_counted = true;
                        resources.disk_window_reads += 1;
                    }
                    window.row_submitted = true;
                    window.referenced += 1;
                    self.outstanding += 1;
                }
            }
        }
        Ok(())
    }

    /// Wait for every referenced window reply, close every opened file, and
    /// report the failure once every reply has arrived.
    fn unwind(&mut self, resources: &mut Resources) -> Option<Result<u64>> {
        self.active.retain(|window| window.referenced > 0);
        for segment in &mut self.segments {
            if segment.state == FileState::Opened && resources.disk_space() {
                match resources.submit_disk(disk::Job::Close { id: segment.disk }) {
                    Ok(()) => {
                        segment.state = FileState::Closing;
                        self.outstanding += 1;
                    }
                    Err(_) => segment.state = FileState::Closed,
                }
            }
        }
        if self.outstanding == 0
            && self.active.is_empty()
            && self.segments.iter().all(Segment::settled)
        {
            return Some(Err(self.error.clone()?));
        }
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SharedSegmentId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SharedWindowId(u64);

/// Reference-counted record header wrapper with pointer-equality fast path.
#[derive(Clone, Debug)]
struct HeaderIdentity(Arc<Header>);

impl PartialEq for HeaderIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || *self.0 == *other.0
    }
}

impl Eq for HeaderIdentity {}

impl std::hash::Hash for HeaderIdentity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SegmentKey {
    path: Arc<Path>,
    expected: HeaderIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct WindowKey {
    segment: SegmentKey,
    first: u32,
    count: usize,
    row: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SharedFileState {
    /// No `disk::Job::Open` has been submitted yet.
    Unopened,
    /// `disk::Job::Open` is in flight; pipelined reads may already be submitted.
    Opening,
    /// `disk::Job::Open` succeeded and the record file stays open on the disk thread.
    Opened,
    /// `disk::Job::Close` is in flight; no new followers or reads may attach.
    Closing,
    /// Closed cleanly, or `Open` failed before creating a disk-thread reader entry.
    Closed,
}

/// A record file opened once and shared across concurrent device fetches whose
/// path and expected header match exactly.
struct SharedSegment {
    disk: Option<JobId>,
    key: SegmentKey,
    state: SharedFileState,
    /// Submitted disk jobs (`Open`, `ReadPages`, `ReadRow`, `ReadPacked`, `Close`) awaiting replies.
    disk_outstanding: usize,
    next_disk_window: u32,
    pending_page_reads: HashMap<u32, SharedWindowId>,
    pending_packed_reads: HashMap<u32, SharedWindowId>,
    /// `disk::Job::ReadRow` carries no window id; replies arrive in FIFO order on `disk`.
    pending_row_reads: VecDeque<SharedWindowId>,
    open_error: Option<Error>,
    users: HashSet<JobId>,
}

impl SharedSegment {
    fn reusable(&self) -> bool {
        self.open_error.is_none()
            && matches!(
                self.state,
                SharedFileState::Unopened | SharedFileState::Opening | SharedFileState::Opened
            )
    }

    fn disk_settled(&self) -> bool {
        self.state == SharedFileState::Unopened
            || (self.state == SharedFileState::Closed && self.disk_outstanding == 0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SharedWindowPhase {
    Reading,
    Read,
    Failed,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct WindowFollower {
    record: JobId,
    plan_index: usize,
}

impl WindowFollower {
    fn matches(self, record: JobId, plan_index: usize) -> bool {
        self.record == record && self.plan_index == plan_index
    }
}

/// One staged read window shared across concurrent device fetches.
///
/// Its `SlabPool` slots or v4 `packed_buf` remain retained until every
/// in-flight disk read and every attached follower's GPU upload have completed
/// (`followers.is_empty() && disk_referenced == 0 && upload_referenced == 0`).
struct SharedWindow {
    segment: SharedSegmentId,
    key: WindowKey,
    buffers: Buffers,
    packed_buf: Option<Arc<Vec<u8>>>,
    phase: SharedWindowPhase,
    error: Option<Error>,
    pages_submitted: bool,
    row_submitted: bool,
    #[cfg(any(test, feature = "test-support"))]
    disk_read_counted: bool,
    /// Cumulative disk-thread execution time across this window's read jobs.
    read_elapsed: Duration,
    /// In-flight `ReadPages`, `ReadRow`, or `ReadPacked` jobs populating this window.
    disk_referenced: usize,
    /// In-flight `copier::Job` uploads reading from `buffers` or `packed_buf`.
    upload_referenced: usize,
    /// Bytes read from disk for this window, credited to each follower.
    read_bytes: u64,
    followers: Vec<WindowFollower>,
}

#[derive(Clone, Copy, Debug)]
struct DeviceActiveWindow {
    plan_index: usize,
    shared: SharedWindowId,
    upload_submitted: bool,
    upload_done: bool,
}

struct DeviceSegmentRef {
    segment: SegmentKey,
    /// `None` for zero-window segments that contribute neither pages nor a row.
    shared: Option<SharedSegmentId>,
}

/// One device fetch (`Staged::fetch`) executing aligned windows out of order
/// over shared open segments and shared staging windows.
struct DeviceInbound {
    segments: Vec<DeviceSegmentRef>,
    pages: Vec<PageId>,
    slot: StateSlotId,
    plans: Vec<WindowPlan>,
    started: Vec<bool>,
    active: Vec<DeviceActiveWindow>,
    done: usize,
    bytes: u64,
    error: Option<Error>,
}

impl DeviceInbound {
    fn new(
        segments: Vec<DeviceSegmentRef>,
        pages: Vec<PageId>,
        slot: StateSlotId,
        plans: Vec<WindowPlan>,
    ) -> Self {
        let started = vec![false; plans.len()];
        Self {
            segments,
            pages,
            slot,
            plans,
            started,
            active: Vec::with_capacity(WINDOWS_IN_FLIGHT),
            done: 0,
            bytes: 0,
            error: None,
        }
    }

    fn window_key(&self, plan_index: usize) -> WindowKey {
        let plan = self.plans[plan_index];
        WindowKey {
            segment: self.segments[plan.segment].segment.clone(),
            first: plan.first,
            count: plan.count,
            row: plan.row,
        }
    }

    fn has_unstarted_packed(&self) -> bool {
        self.plans.iter().enumerate().any(|(idx, plan)| {
            !self.started[idx]
                && self.segments[plan.segment]
                    .segment
                    .expected
                    .0
                    .packed
                    .is_some()
        })
    }

    fn fail(&mut self, error: Error) {
        self.error.get_or_insert(error);
    }
}

/// The staged transport: record files move through a shared slot
/// pool and the shared transfer threads. It also moves records between files
/// and caller-owned registered buffers for the composite store.
///
/// Without a pool the transport never submits transfer jobs and never polls
/// the shared transfer threads, whose replies then belong to the memory store.
pub struct Staged {
    resources: Resources,
    outbound: HashMap<JobId, OutboundRecord>,
    host_inbound: HashMap<JobId, HostInbound>,
    device_inbound: HashMap<JobId, DeviceInbound>,
    completed_packed_reads: HashMap<JobId, (Header, Arc<Vec<u8>>)>,
    shared_segments: HashMap<SharedSegmentId, SharedSegment>,
    shared_windows: HashMap<SharedWindowId, SharedWindow>,
    window_index: HashMap<WindowKey, SharedWindowId>,
    next_shared_segment: u64,
    next_shared_window: u64,
    /// Disk job ids to the store job or shared segment each belongs to.
    owners: HashMap<JobId, Owner>,
    /// Truncate and unlink jobs awaiting disk queue space.
    maintenance: VecDeque<(JobId, disk::Job)>,
    events: Vec<TransportEvent>,
    /// Physical shared-window read samples `(bytes, elapsed)` drained out of band.
    inbound_samples: Vec<(u64, Duration)>,
    closed: bool,
}

impl Staged {
    pub fn new(pool: Option<SlabPool>, transfers: SharedCopier, disk: DiskThread) -> Self {
        Self {
            resources: Resources {
                disk,
                pool,
                transfers,
                disk_outstanding: 0,
                next_disk: 1,
                #[cfg(any(test, feature = "test-support"))]
                disk_opens: 0,
                #[cfg(any(test, feature = "test-support"))]
                disk_window_reads: 0,
                #[cfg(any(test, feature = "test-support"))]
                coalesced_windows: 0,
            },
            outbound: HashMap::new(),
            host_inbound: HashMap::new(),
            device_inbound: HashMap::new(),
            completed_packed_reads: HashMap::new(),
            shared_segments: HashMap::new(),
            shared_windows: HashMap::new(),
            window_index: HashMap::new(),
            next_shared_segment: 1,
            next_shared_window: 1,
            owners: HashMap::new(),
            maintenance: VecDeque::new(),
            events: Vec::new(),
            inbound_samples: Vec::new(),
            closed: false,
        }
    }

    pub fn shares_transfers(&self, transfers: &SharedCopier) -> bool {
        Arc::ptr_eq(&self.resources.transfers, transfers)
    }

    pub fn pool(&self) -> Option<&SlabPool> {
        self.resources.pool.as_ref()
    }

    /// Physical `disk::Job::Open` calls submitted by this transport.
    #[cfg(any(test, feature = "test-support"))]
    pub fn disk_opens(&self) -> u64 {
        self.resources.disk_opens
    }

    /// Physical disk window reads submitted by this transport.
    #[cfg(any(test, feature = "test-support"))]
    pub fn disk_window_reads(&self) -> u64 {
        self.resources.disk_window_reads
    }

    /// Logical device-fetch windows satisfied by attaching to an in-flight shared window.
    #[cfg(any(test, feature = "test-support"))]
    pub fn coalesced_windows(&self) -> u64 {
        self.resources.coalesced_windows
    }

    /// Records of each direction whose pipelines have not finished.
    pub fn in_flight(&self) -> (usize, usize) {
        (
            self.outbound.len(),
            self.host_inbound.len() + self.device_inbound.len(),
        )
    }

    /// Write a record whose page and row bytes already sit in caller-owned
    /// registered buffers (host addresses in record order).
    #[allow(clippy::too_many_arguments)]
    pub fn write_from(
        &mut self,
        id: super::JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        pages: Vec<usize>,
        row: Option<Vec<usize>>,
        sums: Option<SliceSums>,
    ) -> Result<()> {
        self.accept(id)?;
        if pages.len() != header.tail_pages as usize {
            return Err(invalid(format!(
                "record {} has {} tail pages; {} page buffers were supplied",
                record::file_name(&header.key),
                header.tail_pages,
                pages.len()
            )));
        }
        let source = Source::Host { pages, row, sums };
        self.start_outbound(id, header, temporary, path, source);
        Ok(())
    }

    /// Write an already-encoded packed record directly from caller-pinned host slab spans.
    ///
    /// # Safety
    /// The caller must keep every `(addr, len)` span in `spans` valid and immutable until the
    /// matching `TransportEvent::Stored` arrives or the transport closes.
    pub fn write_packed(
        &mut self,
        id: super::JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        units: Vec<crate::shared::wire::PackedUnit>,
        spans: Vec<(usize, usize)>,
        row: bool,
    ) -> Result<()> {
        self.accept(id)?;
        let source = Source::HostPacked {
            units: Some(units),
            spans,
            row,
        };
        self.start_outbound(id, header, temporary, path, source);
        Ok(())
    }

    /// Read a v4 packed record's pages (and optional row) into a contiguous host buffer.
    pub fn read_into_packed(
        &mut self,
        id: super::JobId,
        path: PathBuf,
        expected: Header,
        row: bool,
    ) -> Result<()> {
        self.accept(id)?;
        let layout = expected
            .packed
            .as_ref()
            .ok_or_else(|| invalid("read_into_packed requires a v4 packed header"))?;
        let count = layout.page_count();
        if count == 0 && !row {
            return Err(invalid(
                "a packed record read must request pages or the row",
            ));
        }
        let plan = WindowPlan {
            segment: 0,
            index: 0,
            first: 0,
            count,
            offset: 0,
            row,
        };
        let disk = self.resources.disk_id();
        self.owners.insert(disk, Owner::HostInbound(id, 0));
        let segment = Segment {
            disk,
            path,
            expected,
            state: FileState::Unopened,
            reads_left: 1,
        };
        self.host_inbound.insert(
            id,
            HostInbound::new(vec![segment], HostDestination::Packed, vec![plan]),
        );
        self.pump_host_inbound(id);
        Ok(())
    }

    pub fn take_packed_read(&mut self, id: JobId) -> Option<(Header, Arc<Vec<u8>>)> {
        self.completed_packed_reads.remove(&id)
    }

    /// Read a record's page records `first..first + buffers.len()` and row into
    /// caller-owned registered buffers.
    pub fn read_into(
        &mut self,
        id: super::JobId,
        path: PathBuf,
        expected: Header,
        first_page: u32,
        pages: Vec<usize>,
        row: Option<Vec<usize>>,
    ) -> Result<()> {
        self.accept(id)?;
        if pages.is_empty() && row.is_none() {
            return Err(invalid(
                "a record read must name page buffers or the row buffer",
            ));
        }
        let plans = plan_windows([(first_page, pages.len(), row.is_some())], false);
        let disk = self.resources.disk_id();
        self.owners.insert(disk, Owner::HostInbound(id, 0));
        let segment = Segment {
            disk,
            path,
            expected,
            state: FileState::Unopened,
            reads_left: plans.len(),
        };
        self.host_inbound.insert(
            id,
            HostInbound::new(
                vec![segment],
                HostDestination::Slots { pages, row },
                plans,
            ),
        );
        self.pump_host_inbound(id);
        Ok(())
    }

    /// Stream the GPU ids `pages` and the row in `slot` into a new record file,
    /// as `store` does for the record's pages of a published checkpoint.
    pub fn store_pages(
        &mut self,
        id: super::JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        pages: Vec<PageId>,
        slot: StateSlotId,
        detect: bool,
    ) -> Result<()> {
        self.accept(id)?;
        if self.resources.pool.is_none() {
            return Err(invalid(
                "the staged transport has no slot pool; records move only through caller-owned buffers",
            ));
        }
        if pages.len() != header.tail_pages as usize {
            return Err(invalid(format!(
                "record {} holds {} tail pages; {} GPU pages were supplied",
                record::file_name(&header.key),
                header.tail_pages,
                pages.len()
            )));
        }
        let source = Source::Device {
            pages,
            slot,
            detect,
        };
        self.start_outbound(id, header, temporary, path, source);
        Ok(())
    }

    fn accept(&self, id: JobId) -> Result<()> {
        if self.closed {
            return Err(invalid("the staged transport is closed"));
        }
        if self.outbound.contains_key(&id)
            || self.host_inbound.contains_key(&id)
            || self.device_inbound.contains_key(&id)
        {
            return Err(invalid(format!(
                "disk transport job {} is already in flight",
                id.0
            )));
        }
        Ok(())
    }

    fn next_shared_segment_id(&mut self) -> SharedSegmentId {
        let id = SharedSegmentId(self.next_shared_segment);
        self.next_shared_segment += 1;
        id
    }

    fn next_shared_window_id(&mut self) -> SharedWindowId {
        let id = SharedWindowId(self.next_shared_window);
        self.next_shared_window += 1;
        id
    }

    fn acquire_shared_segment(&mut self, user: JobId, key: SegmentKey) -> (SharedSegmentId, SegmentKey) {
        let existing = self
            .shared_segments
            .iter()
            .filter(|(_, seg)| seg.reusable() && seg.key == key)
            .min_by_key(|&(&id, _)| id)
            .map(|(&id, seg)| (id, seg.key.clone()));
        if let Some((seg_id, shared_key)) = existing {
            self.shared_segments
                .get_mut(&seg_id)
                .unwrap()
                .users
                .insert(user);
            return (seg_id, shared_key);
        }
        let seg_id = self.next_shared_segment_id();
        let mut users = HashSet::new();
        users.insert(user);
        self.shared_segments.insert(
            seg_id,
            SharedSegment {
                disk: None,
                key: key.clone(),
                state: SharedFileState::Unopened,
                disk_outstanding: 0,
                next_disk_window: 0,
                pending_page_reads: HashMap::new(),
                pending_packed_reads: HashMap::new(),
                pending_row_reads: VecDeque::new(),
                open_error: None,
                users,
            },
        );
        (seg_id, key)
    }

    fn start_outbound(
        &mut self,
        id: JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        source: Source,
    ) {
        let packed = match &source {
            Source::HostPacked { .. } => true,
            Source::Device { pages, .. } => {
                self.resources.codec_enabled()
                    && crate::shared::packed::fits_units(pages.len(), true, header.ranks.len())
            }
            Source::Host { .. } => false,
        };
        let disk = self.resources.disk_id();
        self.owners.insert(disk, Owner::Outbound(id));
        self.outbound.insert(
            id,
            OutboundRecord::new(disk, header, temporary, path, source, packed),
        );
        self.pump_outbound(id);
    }

    fn pump_outbound(&mut self, id: JobId) {
        let Some(record) = self.outbound.get_mut(&id) else {
            return;
        };
        if let Some(result) = record.pump(id, &mut self.resources) {
            let record = self.outbound.remove(&id).unwrap();
            self.owners.remove(&record.disk);
            self.events.push(TransportEvent::Stored {
                id,
                result,
                header: record.committed.map(|_| Box::new(record.header)),
            });
        }
    }

    fn pump_host_inbound(&mut self, id: JobId) {
        let Some(record) = self.host_inbound.get_mut(&id) else {
            return;
        };
        if let Some(result) = record.pump(&mut self.resources) {
            let mut record = self.host_inbound.remove(&id).unwrap();
            for segment in &record.segments {
                self.owners.remove(&segment.disk);
            }
            if result.is_ok()
                && let Some(packed) = record.completed_packed.take()
            {
                self.completed_packed_reads.insert(id, packed);
            }
            self.events.push(TransportEvent::Fetched { id, result });
        }
    }

    fn unindex_window(
        window_index: &mut HashMap<WindowKey, SharedWindowId>,
        key: &WindowKey,
        id: SharedWindowId,
    ) {
        if window_index.get(key) == Some(&id) {
            window_index.remove(key);
        }
    }

    fn fail_segment_users(&mut self, seg_id: SharedSegmentId, error: &Error) {
        let users: Vec<JobId> = self
            .shared_segments
            .get(&seg_id)
            .map(|seg| seg.users.iter().copied().collect())
            .unwrap_or_default();
        for user in users {
            if let Some(record) = self.device_inbound.get_mut(&user) {
                record.fail(error.clone());
            }
        }
        let mut window_followers = Vec::new();
        for (&win_id, win) in &mut self.shared_windows {
            if win.segment == seg_id && win.phase == SharedWindowPhase::Reading {
                win.phase = SharedWindowPhase::Failed;
                win.error.get_or_insert_with(|| error.clone());
                Self::unindex_window(&mut self.window_index, &win.key, win_id);
                window_followers.extend(win.followers.iter().map(|f| f.record));
            }
        }
        for record_id in window_followers {
            if let Some(record) = self.device_inbound.get_mut(&record_id) {
                record.fail(error.clone());
            }
        }
    }

    fn fail_window(&mut self, win_id: SharedWindowId, error: Error) {
        let Some(win) = self.shared_windows.get_mut(&win_id) else {
            return;
        };
        win.phase = SharedWindowPhase::Failed;
        Self::unindex_window(&mut self.window_index, &win.key, win_id);
        let err = win.error.get_or_insert(error).clone();
        let followers: Vec<JobId> = win.followers.iter().map(|f| f.record).collect();
        for record_id in followers {
            if let Some(record) = self.device_inbound.get_mut(&record_id) {
                record.fail(err.clone());
            }
        }
    }

    fn apply_shared_disk(&mut self, seg_id: SharedSegmentId, reply: disk::Reply) {
        let target_window = {
            let Some(seg) = self.shared_segments.get_mut(&seg_id) else {
                return;
            };
            seg.disk_outstanding = seg.disk_outstanding.saturating_sub(1);
            match reply.operation {
                Operation::Open => {
                    match reply.result {
                        Ok(()) => {
                            seg.state = SharedFileState::Opened;
                            return;
                        }
                        Err(error) => {
                            let err = error.with_record_read();
                            seg.state = SharedFileState::Closed;
                            let err = seg.open_error.get_or_insert(err).clone();
                            self.fail_segment_users(seg_id, &err);
                            return;
                        }
                    }
                }
                Operation::ReadPages => seg.pending_page_reads.remove(&reply.window),
                Operation::ReadPacked => seg.pending_packed_reads.remove(&reply.window),
                Operation::ReadRow => seg.pending_row_reads.pop_front(),
                Operation::Close => {
                    seg.state = SharedFileState::Closed;
                    return;
                }
                _ => return,
            }
        };
        let Some(win_id) = target_window else {
            return;
        };
        let mut sample = None;
        let failed = {
            let Some(win) = self.shared_windows.get_mut(&win_id) else {
                return;
            };
            win.disk_referenced = win.disk_referenced.saturating_sub(1);
            if reply.operation == Operation::ReadPacked {
                win.packed_buf = reply.payload;
            }
            match reply.result {
                Ok(()) => {
                    win.read_bytes += reply.bytes;
                    win.read_elapsed += reply.elapsed;
                    if win.phase == SharedWindowPhase::Reading
                        && win.pages_submitted
                        && win.row_submitted
                        && win.disk_referenced == 0
                    {
                        win.phase = SharedWindowPhase::Read;
                        if win.read_bytes > 0 {
                            sample = Some((
                                win.read_bytes,
                                win.read_elapsed.max(Duration::from_nanos(1)),
                            ));
                        }
                    }
                    None
                }
                Err(error) => Some(error.with_record_read()),
            }
        };
        if let Some((bytes, elapsed)) = sample {
            self.inbound_samples.push((bytes, elapsed));
        }
        if let Some(err) = failed {
            self.fail_window(win_id, err);
        }
    }

    fn apply_device_transfer(&mut self, reply: copier::Reply) {
        let Some(record) = self.device_inbound.get_mut(&reply.id) else {
            return;
        };
        let Some(active) = record
            .active
            .iter_mut()
            .find(|active| active.plan_index as u32 == reply.window)
        else {
            return;
        };
        active.upload_done = true;
        let shared_id = active.shared;
        let mut packed_has_buf = false;
        if let Some(win) = self.shared_windows.get_mut(&shared_id) {
            win.upload_referenced = win.upload_referenced.saturating_sub(1);
            packed_has_buf = win.packed_buf.is_some();
        }
        if let Err(error) = reply.result {
            if packed_has_buf && is_packed_content_error(&error) {
                let err = error.with_record_read();
                record.fail(err.clone());
                self.fail_window(shared_id, err);
            } else {
                record.fail(error);
            }
        }
    }

    fn find_reusable_window(&self, key: &WindowKey) -> Option<(SharedWindowId, SharedWindowPhase)> {
        let &win_id = self.window_index.get(key)?;
        let win = self.shared_windows.get(&win_id)?;
        let reusable = match win.phase {
            SharedWindowPhase::Read => true,
            SharedWindowPhase::Reading => {
                self.shared_segments.get(&win.segment).is_some_and(|seg| {
                    seg.open_error.is_none()
                        && matches!(
                            seg.state,
                            SharedFileState::Opening | SharedFileState::Opened
                        )
                })
            }
            SharedWindowPhase::Failed => false,
        };
        reusable.then_some((win_id, win.phase))
    }

    fn find_best_existing_window_for(&self, id: JobId) -> Option<(usize, SharedWindowId)> {
        let record = self.device_inbound.get(&id)?;
        let mut best: Option<(bool, usize, SharedWindowId)> = None;
        for plan_index in 0..record.plans.len() {
            if record.started[plan_index] {
                continue;
            }
            let key = record.window_key(plan_index);
            if let Some((win_id, phase)) = self.find_reusable_window(&key) {
                let candidate = (phase != SharedWindowPhase::Read, plan_index, win_id);
                if best.is_none_or(|current| candidate < current) {
                    best = Some(candidate);
                }
            }
        }
        best.map(|(_, plan_index, win_id)| (plan_index, win_id))
    }

    fn has_window_credit(&self, id: JobId) -> bool {
        self.device_inbound
            .get(&id)
            .is_some_and(|record| record.error.is_none() && record.active.len() < WINDOWS_IN_FLIGHT)
    }

    fn attach_existing_window(&mut self, id: JobId, plan_index: usize, win_id: SharedWindowId) {
        if let Some(win) = self.shared_windows.get_mut(&win_id) {
            win.followers.push(WindowFollower {
                record: id,
                plan_index,
            });
        }
        if let Some(record) = self.device_inbound.get_mut(&id) {
            record.started[plan_index] = true;
            record.active.push(DeviceActiveWindow {
                plan_index,
                shared: win_id,
                upload_submitted: false,
                upload_done: false,
            });
        }
        #[cfg(any(test, feature = "test-support"))]
        {
            self.resources.coalesced_windows += 1;
        }
    }

    fn attach_pending(
        &mut self,
        id: JobId,
        demand: Option<&mut HashMap<WindowKey, usize>>,
    ) -> bool {
        if !self.has_window_credit(id) {
            return false;
        }
        let Some((plan_index, win_id)) = self.find_best_existing_window_for(id) else {
            return false;
        };
        if let Some(demand_map) = demand {
            let key = self.device_inbound[&id].window_key(plan_index);
            if let Some(count) = demand_map.get_mut(&key) {
                *count = count.saturating_sub(1);
            }
        }
        self.attach_existing_window(id, plan_index, win_id);
        true
    }

    /// Build a per-window demand table across healthy fetches so Pass 4 selects
    /// shared ancestor windows in linear time without per-plan path allocations.
    fn build_pending_demand(&self) -> HashMap<WindowKey, usize> {
        let mut demand = HashMap::new();
        if self.device_inbound.len() <= 1 {
            return demand;
        }
        for record in self.device_inbound.values() {
            if record.error.is_some() {
                continue;
            }
            let mut seen_in_record: HashSet<(usize, u32, usize, bool)> = HashSet::new();
            for (idx, plan) in record.plans.iter().enumerate() {
                if record.started[idx] {
                    continue;
                }
                if !seen_in_record.insert((plan.segment, plan.first, plan.count, plan.row)) {
                    continue;
                }
                *demand.entry(record.window_key(idx)).or_insert(0) += 1;
            }
        }
        demand
    }

    /// Choose the next unstarted window for `id`, preferring windows also
    /// pending in sibling concurrent fetches so they stay synchronized on
    /// shared ancestors before fanning out to private segments.
    fn pick_next_pending_plan(
        &self,
        id: JobId,
        demand: &HashMap<WindowKey, usize>,
    ) -> Option<usize> {
        let record = self.device_inbound.get(&id)?;
        self.resources.pool.as_ref()?;
        let free = self.resources.free_slots();
        let transfers = lock(&self.resources.transfers);
        let layout = transfers.layout();
        let mut best: Option<(usize, usize)> = None;
        for (plan_index, plan) in record.plans.iter().enumerate() {
            if record.started[plan_index] {
                continue;
            }
            let packed = record.segments[plan.segment]
                .segment
                .expected
                .0
                .packed
                .is_some();
            if !packed && layout.record_slots(plan.count, plan.row) > free {
                continue;
            }
            let peer_demand = demand
                .get(&record.window_key(plan_index))
                .map_or(0, |total| total.saturating_sub(1));
            if best.is_none_or(|(best_demand, _)| peer_demand > best_demand) {
                best = Some((peer_demand, plan_index));
            }
        }
        best.map(|(_, plan_index)| plan_index)
    }

    fn detach_unwinding_windows(&mut self, ids: &[JobId]) {
        for &id in ids {
            let Some(record) = self.device_inbound.get_mut(&id) else {
                continue;
            };
            if record.error.is_none() {
                continue;
            }
            let mut index = 0;
            while index < record.active.len() {
                let active = record.active[index];
                let upload_busy = active.upload_submitted && !active.upload_done;
                let can_detach = !upload_busy
                    && self.shared_windows.get(&active.shared).is_none_or(|win| {
                        win.followers.iter().any(|f| f.record != id) || win.disk_referenced == 0
                    });
                if can_detach {
                    let active = record.active.remove(index);
                    if let Some(win) = self.shared_windows.get_mut(&active.shared) {
                        win.followers.retain(|f| !f.matches(id, active.plan_index));
                    }
                } else {
                    index += 1;
                }
            }
        }
    }

    fn sweep_shared_windows(&mut self) {
        let dead: Vec<SharedWindowId> = self
            .shared_windows
            .iter()
            .filter(|(_, win)| {
                win.followers.is_empty()
                    && win.disk_referenced == 0
                    && win.upload_referenced == 0
            })
            .map(|(&id, _)| id)
            .collect();
        for win_id in dead {
            if let Some(mut win) = self.shared_windows.remove(&win_id) {
                Self::unindex_window(&mut self.window_index, &win.key, win_id);
                self.resources.release(&mut win.buffers);
            }
        }
    }

    fn segment_needs_disk_reads(&self, seg_id: SharedSegmentId) -> bool {
        let Some(seg) = self.shared_segments.get(&seg_id) else {
            return false;
        };
        if seg.disk_outstanding > 0 {
            return true;
        }
        if self.shared_windows.values().any(|win| {
            win.segment == seg_id
                && (win.disk_referenced > 0
                    || (win.phase == SharedWindowPhase::Reading
                        && win.followers.iter().any(|f| {
                            self.device_inbound
                                .get(&f.record)
                                .is_some_and(|record| record.error.is_none())
                        })))
        }) {
            return true;
        }
        seg.users.iter().any(|user_id| {
            self.device_inbound.get(user_id).is_some_and(|record| {
                record.error.is_none()
                    && record.plans.iter().enumerate().any(|(plan_index, plan)| {
                        !record.started[plan_index]
                            && record.segments[plan.segment].shared == Some(seg_id)
                    })
            })
        })
    }

    fn remove_shared_segment(&mut self, seg_id: SharedSegmentId) {
        if let Some(seg) = self.shared_segments.remove(&seg_id)
            && let Some(disk) = seg.disk
        {
            self.owners.remove(&disk);
        }
    }

    fn pump_device_inbound(&mut self) {
        let mut ids: Vec<JobId> = self.device_inbound.keys().copied().collect();
        ids.sort_unstable();

        // Pass 1: reap completed uploads on healthy records and detach finished
        // or shared windows from unwinding records.
        for &id in &ids {
            let Some(record) = self.device_inbound.get_mut(&id) else {
                continue;
            };
            if record.error.is_some() {
                continue;
            }
            let mut index = 0;
            while index < record.active.len() {
                if record.active[index].upload_done {
                    let active = record.active.remove(index);
                    if let Some(win) = self.shared_windows.get_mut(&active.shared) {
                        record.bytes += win.read_bytes;
                        win.followers.retain(|f| !f.matches(id, active.plan_index));
                    }
                    record.done += 1;
                } else {
                    index += 1;
                }
            }
        }
        self.detach_unwinding_windows(&ids);

        // Pass 2: attach healthy records with free window credit to already
        // in-flight shared windows before sweeping finished buffers or starting
        // new disk reads.
        for &id in &ids {
            while self.attach_pending(id, None) {}
        }

        // Pass 3: release staging slots for shared windows no follower or job
        // still references.
        self.sweep_shared_windows();

        // Pass 4: start new shared windows (or attach to one started by an
        // earlier record in this pass) as slots and disk queue depth allow.
        let any_packed = self
            .device_inbound
            .values()
            .any(|record| record.error.is_none() && record.has_unstarted_packed());
        let mut demand = if self.resources.free_slots() > 0 || any_packed {
            Some(self.build_pending_demand())
        } else {
            None
        };
        for &id in &ids {
            while self.has_window_credit(id) {
                if self.attach_pending(id, demand.as_mut()) {
                    continue;
                }
                if self.resources.free_slots() == 0
                    && !self.device_inbound[&id].has_unstarted_packed()
                {
                    break;
                }
                let demand_ref = demand.get_or_insert_with(|| self.build_pending_demand());
                let Some(plan_index) = self.pick_next_pending_plan(id, demand_ref) else {
                    break;
                };
                let (plan, seg_id, key) = {
                    let record = &self.device_inbound[&id];
                    let plan = record.plans[plan_index];
                    let seg_ref = &record.segments[plan.segment];
                    (plan, seg_ref.shared.unwrap(), record.window_key(plan_index))
                };
                let seg_state = self.shared_segments[&seg_id].state;
                debug_assert!(matches!(
                    seg_state,
                    SharedFileState::Unopened
                        | SharedFileState::Opening
                        | SharedFileState::Opened
                ));
                if seg_state == SharedFileState::Unopened {
                    if !self.resources.disk_space() {
                        break;
                    }
                    let disk = self.resources.disk_id();
                    let job = disk::Job::Open {
                        id: disk,
                        path: key.segment.path.as_ref().to_path_buf(),
                        expected: (*key.segment.expected.0).clone(),
                    };
                    match self.resources.submit_disk(job) {
                        Ok(()) => {
                            self.owners.insert(disk, Owner::SharedSegment(seg_id));
                            let seg = self.shared_segments.get_mut(&seg_id).unwrap();
                            seg.disk = Some(disk);
                            seg.state = SharedFileState::Opening;
                            seg.disk_outstanding += 1;
                            #[cfg(any(test, feature = "test-support"))]
                            {
                                self.resources.disk_opens += 1;
                            }
                        }
                        Err(error) => {
                            let seg = self.shared_segments.get_mut(&seg_id).unwrap();
                            seg.state = SharedFileState::Closed;
                            let err = seg.open_error.get_or_insert(error).clone();
                            self.fail_segment_users(seg_id, &err);
                            break;
                        }
                    }
                }
                let packed = key.segment.expected.0.packed.is_some();
                let buffers = if packed {
                    Buffers {
                        pages: Vec::new(),
                        row: None,
                        slots: None,
                    }
                } else {
                    let Some(buffers) = self.resources.allocate(plan.count, plan.row) else {
                        break;
                    };
                    buffers
                };
                let win_id = self.next_shared_window_id();
                if let Some(demand_map) = demand.as_mut()
                    && let Some(count) = demand_map.get_mut(&key)
                {
                    *count = count.saturating_sub(1);
                }
                self.window_index.insert(key.clone(), win_id);
                self.shared_windows.insert(
                    win_id,
                    SharedWindow {
                        segment: seg_id,
                        key,
                        buffers,
                        packed_buf: None,
                        phase: SharedWindowPhase::Reading,
                        error: None,
                        pages_submitted: !packed && plan.count == 0,
                        row_submitted: !packed && !plan.row,
                        #[cfg(any(test, feature = "test-support"))]
                        disk_read_counted: false,
                        read_elapsed: Duration::ZERO,
                        disk_referenced: 0,
                        upload_referenced: 0,
                        read_bytes: 0,
                        followers: vec![WindowFollower {
                            record: id,
                            plan_index,
                        }],
                    },
                );
                let record = self.device_inbound.get_mut(&id).unwrap();
                record.started[plan_index] = true;
                record.active.push(DeviceActiveWindow {
                    plan_index,
                    shared: win_id,
                    upload_submitted: false,
                    upload_done: false,
                });
            }
        }

        // Pass 5a: submit pending disk reads for shared windows with at least
        // one healthy follower.
        let mut win_ids: Vec<SharedWindowId> = self.shared_windows.keys().copied().collect();
        win_ids.sort_unstable();
        for win_id in win_ids {
            let has_healthy_follower = self.shared_windows.get(&win_id).is_some_and(|win| {
                win.phase == SharedWindowPhase::Reading
                    && win.followers.iter().any(|f| {
                        self.device_inbound
                            .get(&f.record)
                            .is_some_and(|record| record.error.is_none())
                    })
            });
            if !has_healthy_follower {
                continue;
            }
            let (
                seg_id,
                pages_needed,
                row_needed,
                first,
                count,
                row,
                page_segment,
                packed_layout,
                page_buffers,
                row_buffers,
            ) = {
                let win = &self.shared_windows[&win_id];
                let expected = &win.key.segment.expected.0;
                (
                    win.segment,
                    !win.pages_submitted,
                    !win.row_submitted,
                    win.key.first,
                    win.key.count,
                    win.key.row,
                    expected.page_segment,
                    expected.packed.clone(),
                    if !win.pages_submitted && expected.packed.is_none() {
                        Some(win.buffers.pages.clone())
                    } else {
                        None
                    },
                    if !win.row_submitted && expected.packed.is_none() {
                        win.buffers.row.clone()
                    } else {
                        None
                    },
                )
            };
            let Some(disk) = self
                .shared_segments
                .get(&seg_id)
                .filter(|seg| {
                    seg.open_error.is_none()
                        && matches!(seg.state, SharedFileState::Opening | SharedFileState::Opened)
                })
                .and_then(|seg| seg.disk)
            else {
                continue;
            };
            if let Some(layout) = packed_layout {
                if !pages_needed {
                    continue;
                }
                if !self.resources.disk_space() {
                    continue;
                }
                let start_rel = if count == 0 && row {
                    layout.page_offset(layout.page_count())
                } else {
                    layout.page_offset(first as usize)
                };
                let end_rel = if row {
                    layout.physical_bytes()
                } else {
                    layout.page_offset(first as usize + count)
                };
                let offset = page_segment + start_rel;
                let length = match usize::try_from(end_rel - start_rel) {
                    Ok(len) => len,
                    Err(_) => {
                        self.fail_window(win_id, invalid("packed window length overflow"));
                        continue;
                    }
                };
                let token = {
                    let seg = self.shared_segments.get_mut(&seg_id).unwrap();
                    let token = seg.next_disk_window;
                    seg.next_disk_window = seg.next_disk_window.wrapping_add(1);
                    token
                };
                let job = disk::Job::ReadPacked {
                    id: disk,
                    window: token,
                    offset,
                    length,
                };
                match self.resources.submit_disk(job) {
                    Ok(()) => {
                        let seg = self.shared_segments.get_mut(&seg_id).unwrap();
                        seg.pending_packed_reads.insert(token, win_id);
                        seg.disk_outstanding += 1;
                        let win = self.shared_windows.get_mut(&win_id).unwrap();
                        win.pages_submitted = true;
                        win.row_submitted = true;
                        win.disk_referenced += 1;
                        #[cfg(any(test, feature = "test-support"))]
                        if !win.disk_read_counted {
                            win.disk_read_counted = true;
                            self.resources.disk_window_reads += 1;
                        }
                    }
                    Err(error) => {
                        self.fail_window(win_id, error);
                    }
                }
                continue;
            }
            if pages_needed {
                if !self.resources.disk_space() {
                    continue;
                }
                let token = {
                    let seg = self.shared_segments.get_mut(&seg_id).unwrap();
                    let token = seg.next_disk_window;
                    seg.next_disk_window = seg.next_disk_window.wrapping_add(1);
                    token
                };
                let job = disk::Job::ReadPages {
                    id: disk,
                    window: token,
                    first,
                    buffers: page_buffers.unwrap(),
                };
                match self.resources.submit_disk(job) {
                    Ok(()) => {
                        let seg = self.shared_segments.get_mut(&seg_id).unwrap();
                        seg.pending_page_reads.insert(token, win_id);
                        seg.disk_outstanding += 1;
                        let win = self.shared_windows.get_mut(&win_id).unwrap();
                        win.pages_submitted = true;
                        win.disk_referenced += 1;
                        #[cfg(any(test, feature = "test-support"))]
                        if !win.disk_read_counted {
                            win.disk_read_counted = true;
                            self.resources.disk_window_reads += 1;
                        }
                    }
                    Err(error) => {
                        self.fail_window(win_id, error);
                        continue;
                    }
                }
            }
            if row_needed {
                if !self.resources.disk_space() {
                    continue;
                }
                let job = disk::Job::ReadRow {
                    id: disk,
                    buffers: row_buffers.unwrap(),
                };
                match self.resources.submit_disk(job) {
                    Ok(()) => {
                        let seg = self.shared_segments.get_mut(&seg_id).unwrap();
                        seg.pending_row_reads.push_back(win_id);
                        seg.disk_outstanding += 1;
                        let win = self.shared_windows.get_mut(&win_id).unwrap();
                        win.row_submitted = true;
                        win.disk_referenced += 1;
                        #[cfg(any(test, feature = "test-support"))]
                        if !win.disk_read_counted {
                            win.disk_read_counted = true;
                            self.resources.disk_window_reads += 1;
                        }
                    }
                    Err(error) => {
                        self.fail_window(win_id, error);
                    }
                }
            }
        }

        // Pass 5b: submit per-follower GPU uploads for shared windows whose
        // disk reads have completed.
        for &id in &ids {
            let active_len = match self.device_inbound.get(&id) {
                Some(record) if record.error.is_none() => record.active.len(),
                _ => continue,
            };
            for index in 0..active_len {
                let active = self.device_inbound[&id].active[index];
                if active.upload_submitted {
                    continue;
                }
                let ready = self
                    .shared_windows
                    .get(&active.shared)
                    .is_some_and(|win| win.phase == SharedWindowPhase::Read);
                if !ready {
                    continue;
                }
                if !self.resources.transfer_space() {
                    break;
                }
                let job_result: Result<copier::Job> = (|| {
                    let record = &self.device_inbound[&id];
                    let plan = record.plans[active.plan_index];
                    let win = &self.shared_windows[&active.shared];
                    if let Some(layout) = &win.key.segment.expected.0.packed {
                        let buf = win
                            .packed_buf
                            .as_ref()
                            .ok_or_else(|| invalid("packed window missing disk payload"))?;
                        let units = layout
                            .units(plan.first as usize, plan.count, plan.row)
                            .ok_or_else(|| invalid("packed window out of layout range"))?;
                        let base = buf.as_ptr() as usize;
                        let mut cursor = 0usize;
                        let mut packets = Vec::with_capacity(units.len());
                        for unit in units {
                            let len = usize::try_from(unit.stored_length)
                                .map_err(|_| invalid("packed unit length overflow"))?;
                            if cursor + len > buf.len() {
                                return Err(invalid("packed disk payload truncated"));
                            }
                            packets.push(crate::codec_copy::PinnedPacket {
                                unit: unit.clone(),
                                spans: vec![(base + cursor, len)],
                            });
                            cursor += len;
                        }
                        let first_logical = layout.offer().first_page + plan.first as u64;
                        return Ok(copier::Job {
                            codec: Some(crate::codec_copy::Window::restore_pinned(
                                first_logical,
                                packets,
                            )),
                            replication: Default::default(),
                            id,
                            window: plan.index,
                            direction: Direction::In,
                            pages: record.pages[plan.offset..plan.offset + plan.count]
                                .iter()
                                .copied()
                                .map(|p| (p, 0))
                                .collect(),
                            row: plan.row.then_some((record.slot, vec![0])),
                        });
                    }
                    Ok(copier::Job {
                        codec: None,
                        replication: Default::default(),
                        id,
                        window: plan.index,
                        direction: Direction::In,
                        pages: record.pages[plan.offset..plan.offset + plan.count]
                            .iter()
                            .copied()
                            .zip(win.buffers.pages.iter().copied())
                            .collect(),
                        row: win.buffers.row.clone().map(|addrs| (record.slot, addrs)),
                    })
                })();
                match job_result.and_then(|job| self.resources.submit_transfer(job)) {
                    Ok(()) => {
                        self.device_inbound.get_mut(&id).unwrap().active[index].upload_submitted =
                            true;
                        self.shared_windows
                            .get_mut(&active.shared)
                            .unwrap()
                            .upload_referenced += 1;
                    }
                    Err(error) => {
                        self.device_inbound.get_mut(&id).unwrap().fail(error);
                        break;
                    }
                }
            }
        }

        // Pass 5c: detach and release windows for any record that failed during
        // read or transfer submission in this step.
        self.detach_unwinding_windows(&ids);
        self.sweep_shared_windows();

        // Pass 5d: close open shared segments that no longer need disk reads.
        let mut seg_ids: Vec<SharedSegmentId> = self.shared_segments.keys().copied().collect();
        seg_ids.sort_unstable();
        for seg_id in seg_ids {
            let should_close = self
                .shared_segments
                .get(&seg_id)
                .is_some_and(|seg| seg.state == SharedFileState::Opened)
                && !self.segment_needs_disk_reads(seg_id);
            if should_close && self.resources.disk_space() {
                let disk = self.shared_segments[&seg_id].disk.unwrap();
                match self.resources.submit_disk(disk::Job::Close { id: disk }) {
                    Ok(()) => {
                        let seg = self.shared_segments.get_mut(&seg_id).unwrap();
                        seg.state = SharedFileState::Closing;
                        seg.disk_outstanding += 1;
                    }
                    Err(_) => {
                        let seg = self.shared_segments.get_mut(&seg_id).unwrap();
                        seg.state = SharedFileState::Closed;
                    }
                }
            }
        }

        // Pass 5e: complete finished or fully unwound device fetches. The last
        // user of a shared segment waits for it to settle before completing so
        // no disk I/O outlives the caller's index pin on that file.
        for &id in &ids {
            let outcome = {
                let Some(record) = self.device_inbound.get(&id) else {
                    continue;
                };
                let windows_done = if record.error.is_some() {
                    record.active.is_empty()
                } else {
                    record.done == record.plans.len() && record.active.is_empty()
                };
                if !windows_done {
                    continue;
                }
                let segments_ready = record
                    .segments
                    .iter()
                    .filter_map(|seg_ref| seg_ref.shared)
                    .all(|seg_id| {
                        self.shared_segments.get(&seg_id).is_none_or(|seg| {
                            seg.users.iter().any(|&other| other != id) || seg.disk_settled()
                        })
                    });
                if !segments_ready {
                    continue;
                }
                match &record.error {
                    Some(error) => Err(error.clone()),
                    None => Ok(record.bytes),
                }
            };
            let record = self.device_inbound.remove(&id).unwrap();
            for seg_id in record.segments.iter().filter_map(|seg_ref| seg_ref.shared) {
                let remove = match self.shared_segments.get_mut(&seg_id) {
                    Some(seg) => {
                        seg.users.remove(&id);
                        seg.users.is_empty() && seg.disk_settled()
                    }
                    None => false,
                };
                if remove {
                    self.remove_shared_segment(seg_id);
                }
            }
            self.events.push(TransportEvent::Fetched {
                id,
                result: outcome,
            });
        }

        let empty_segments: Vec<SharedSegmentId> = self
            .shared_segments
            .iter()
            .filter(|(_, seg)| seg.users.is_empty() && seg.disk_settled())
            .map(|(&seg_id, _)| seg_id)
            .collect();
        for seg_id in empty_segments {
            self.remove_shared_segment(seg_id);
        }
    }

    fn pump_all(&mut self) {
        while let Some((disk, _)) = self.maintenance.front()
            && self.resources.disk_space()
        {
            let disk = *disk;
            let (_, job) = self.maintenance.pop_front().unwrap();
            if let Err(error) = self.resources.submit_disk(job) {
                self.finish_maintenance(disk, Err(error));
            }
        }
        let mut ids: Vec<JobId> = self.outbound.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            self.pump_outbound(id);
        }
        let mut ids: Vec<JobId> = self.host_inbound.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            self.pump_host_inbound(id);
        }
        self.pump_device_inbound();
    }

    fn finish_maintenance(&mut self, disk: JobId, result: Result<()>) {
        match self.owners.remove(&disk) {
            Some(Owner::Truncate(id)) => self.events.push(TransportEvent::Truncated { id, result }),
            Some(Owner::Unlink(id)) => self.events.push(TransportEvent::Unlinked { id, result }),
            _ => {}
        }
    }

    /// Absorb every reply and advance every pipeline.
    fn step(&mut self) {
        let mut replies = Vec::new();
        self.resources.disk.poll(&mut replies);
        for reply in replies {
            self.resources.disk_outstanding = self.resources.disk_outstanding.saturating_sub(1);
            match self.owners.get(&reply.id).copied() {
                Some(Owner::Outbound(id)) => {
                    if let Some(record) = self.outbound.get_mut(&id) {
                        record.apply_disk(reply);
                    }
                }
                Some(Owner::HostInbound(id, segment)) => {
                    if let Some(record) = self.host_inbound.get_mut(&id) {
                        record.apply_disk(segment, reply);
                    }
                }
                Some(Owner::SharedSegment(seg_id)) => {
                    self.apply_shared_disk(seg_id, reply);
                }
                Some(Owner::Truncate(_) | Owner::Unlink(_)) => {
                    self.finish_maintenance(reply.id, reply.result);
                }
                None => {}
            }
        }
        if self.resources.pool.is_some() {
            let mut replies = Vec::new();
            lock(&self.resources.transfers).poll(&mut replies);
            for reply in replies {
                match reply.direction {
                    Direction::Out => {
                        if let Some(record) = self.outbound.get_mut(&reply.id) {
                            record.apply_transfer(reply);
                        }
                    }
                    Direction::In => {
                        self.apply_device_transfer(reply);
                    }
                }
            }
        }
        self.pump_all();
    }

    fn busy(&self) -> bool {
        !self.outbound.is_empty()
            || !self.host_inbound.is_empty()
            || !self.device_inbound.is_empty()
            || !self.shared_segments.is_empty()
            || !self.shared_windows.is_empty()
            || !self.owners.is_empty()
    }
}

impl DiskTransport for Staged {
    fn staged(&self) -> Option<&Staged> {
        Some(self)
    }

    fn store(
        &mut self,
        id: super::JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        checkpoint: &PublishedCheckpoint,
        detect: bool,
    ) -> Result<()> {
        let tail = header.tail_pages as usize;
        let chain = checkpoint.chain_pages.len();
        if tail > chain {
            return Err(invalid(format!(
                "record {} holds {tail} tail pages but the published chain has only {chain}",
                record::file_name(&header.key)
            )));
        }
        self.store_pages(
            id,
            header,
            temporary,
            path,
            checkpoint.chain_pages[chain - tail..].to_vec(),
            checkpoint.slot,
            detect,
        )
    }

    fn fetch(
        &mut self,
        id: super::JobId,
        plan: Vec<FetchSegment>,
        ticket: &InboundTicket,
    ) -> Result<()> {
        self.accept(id)?;
        if self.resources.pool.is_none() {
            return Err(invalid(
                "the staged transport has no slot pool; records move only through caller-owned buffers",
            ));
        }
        let pages: usize = plan.iter().map(|segment| segment.pages as usize).sum();
        if pages != ticket.pages.len() {
            return Err(invalid(format!(
                "fetch plan covers {pages} pages; the ticket reserved {}",
                ticket.pages.len()
            )));
        }
        let rows = plan.iter().filter(|segment| segment.row).count();
        if rows != 1 || !plan.last().is_some_and(|segment| segment.row) {
            return Err(invalid(
                "fetch plan must take the row from its last record only",
            ));
        }
        let plans = plan_windows(
            plan.iter()
                .map(|segment| (segment.first_page, segment.pages as usize, segment.row)),
            true,
        );
        let mut segments = Vec::with_capacity(plan.len());
        for (index, segment) in plan.into_iter().enumerate() {
            let key = SegmentKey {
                path: Arc::from(segment.path),
                expected: HeaderIdentity(Arc::new(segment.header)),
            };
            let has_windows = plans.iter().any(|window| window.segment == index);
            let (shared, segment) = if has_windows {
                let (seg_id, shared_key) = self.acquire_shared_segment(id, key);
                (Some(seg_id), shared_key)
            } else {
                (None, key)
            };
            segments.push(DeviceSegmentRef { segment, shared });
        }
        self.device_inbound.insert(
            id,
            DeviceInbound::new(segments, ticket.pages.clone(), ticket.slot, plans),
        );
        self.pump_device_inbound();
        Ok(())
    }

    fn truncate(&mut self, id: super::JobId, path: PathBuf, length: u64) -> Result<()> {
        if self.closed {
            return Err(invalid("the staged transport is closed"));
        }
        let disk = self.resources.disk_id();
        self.owners.insert(disk, Owner::Truncate(id));
        self.maintenance.push_back((
            disk,
            disk::Job::Truncate {
                id: disk,
                path,
                length,
            },
        ));
        Ok(())
    }

    fn unlink(&mut self, id: super::JobId, path: PathBuf) -> Result<()> {
        if self.closed {
            return Err(invalid("the staged transport is closed"));
        }
        let disk = self.resources.disk_id();
        self.owners.insert(disk, Owner::Unlink(id));
        self.maintenance
            .push_back((disk, disk::Job::Unlink { id: disk, path }));
        Ok(())
    }

    fn poll(&mut self, events: &mut Vec<TransportEvent>) -> Result<()> {
        if !self.closed {
            self.step();
        }
        events.append(&mut self.events);
        Ok(())
    }

    fn drain_inbound_samples(&mut self, samples: &mut Vec<(u64, Duration)>) {
        samples.append(&mut self.inbound_samples);
    }

    /// Finish outstanding pipelines within [`copier::CLOSE_TIMEOUT`], then
    /// stop the disk thread and, when this transport owns the slot pool,
    /// the transfer threads. Completions reached before the deadline remain
    /// available through `poll`.
    fn close(&mut self) -> Result<()> {
        let deadline = Instant::now() + copier::CLOSE_TIMEOUT;
        loop {
            self.step();
            if !self.busy() || Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        self.closed = true;
        let maintenance_jobs = self
            .owners
            .values()
            .filter(|owner| matches!(owner, Owner::Truncate(_) | Owner::Unlink(_)))
            .count();
        let store_jobs = self.outbound.len()
            + self.host_inbound.len()
            + self.device_inbound.len()
            + maintenance_jobs;
        let abandoned_jobs = store_jobs
            .max(self.owners.len())
            .max(self.maintenance.len());
        let leaked_shared = self.shared_segments.len() + self.shared_windows.len();
        let disk = self.resources.disk.close();
        let transfers = match self.resources.pool {
            Some(_) => lock(&self.resources.transfers).close(),
            None => Ok(()),
        };
        if (disk.is_err() || transfers.is_err())
            && let Some(pool) = &self.resources.pool
        {
            pool.leak();
        }
        if transfers.is_err() {
            for win in self.shared_windows.values_mut() {
                if win.upload_referenced > 0 {
                    std::mem::forget(win.packed_buf.take());
                }
            }
        }
        disk.and(transfers)?;
        if abandoned_jobs > 0 {
            return Err(invalid(format!(
                "{abandoned_jobs} disk transport jobs did not complete within {:?}",
                copier::CLOSE_TIMEOUT
            )));
        }
        if leaked_shared > 0 {
            return Err(invalid(format!(
                "{leaked_shared} shared staging entries remained after close within {:?}",
                copier::CLOSE_TIMEOUT
            )));
        }
        Ok(())
    }

    fn staged_mut(&mut self) -> Option<&mut Staged> {
        Some(self)
    }

    fn codec_receive_profiles(&self) -> Vec<[u8; 32]> {
        lock(&self.resources.transfers).codec_receive_profiles()
    }

    fn decode_counters(&self) -> Option<crate::codec_copy::DecodeCounters> {
        lock(&self.resources.transfers).decode_counters()
    }
}

struct Outbound {
    checkpoint: CheckpointId,
    key: PrefixKey,
    started: Instant,
}

struct Inbound {
    key: PrefixKey,
    nodes: Vec<PrefixKey>,
    /// Tokens the restore saves over prefill.
    tokens: u64,
}

/// Bytes charged until the unlink or truncate succeeds.
struct Reclaim {
    /// The compacted record, for a truncate; `None` for an unlink.
    key: Option<PrefixKey>,
    bytes: u64,
}

pub struct DiskStore {
    index: Index<DiskRecord>,
    identity: [u8; 32],
    fingerprint: [u8; 16],
    limit: u64,
    budget: Option<Arc<super::input::DiskBudget>>,
    used: u64,
    transport: Box<dyn DiskTransport>,
    stats: StoreStats,
    next_job: u64,
    geometry: Vec<CacheGeometry>,
    layout: Layout,
    outbound: HashMap<JobId, Outbound>,
    inbound: HashMap<JobId, Inbound>,
    /// Unlink and truncate jobs whose replies are awaited.
    maintenance: HashMap<JobId, Reclaim>,
    replacements: HashMap<PrefixKey, DiskRecord>,
    /// Keys whose files were unlinked since the last `take_evicted`.
    evicted: Vec<PrefixKey>,
    outbound_rate: Rate,
    inbound_rate: Rate,
    announce: Option<Announce>,
    /// Eviction hands records to the pool's manager instead of unlinking them.
    spill: bool,
    pub(crate) preserve_committed: bool,
    /// Records chosen for the manager and not yet taken by the pool store.
    retiring: Vec<Retiring>,
    /// Records held in place until their move ends, with the bytes freed when
    /// they go.
    held: HashMap<PrefixKey, u64>,
    /// Records whose moves ended and that leave once nothing reads them.
    finishing: Vec<PrefixKey>,
    largest: u64,
    maintenance_progress: u64,
    /// Committed children made these parents eligible for row truncation.
    thinning: HashSet<PrefixKey>,
    pub directory_lock: record::DirectoryLock,
}

/// What a disk store keeps to announce its records and serve them as exports.
struct Announce {
    view: TierView,
    releases: Releases,
}

impl DiskStore {
    /// Open a store over the locked directory, already scanned by
    /// [`super::record::scan`], whose records are inserted parents first and
    /// take their access order from file age, oldest first.
    pub fn new(
        directory_lock: record::DirectoryLock,
        identity: [u8; 32],
        fingerprint: [u8; 16],
        limit: u64,
        geometry: Vec<CacheGeometry>,
        scanned: Vec<Scanned>,
        transport: Box<dyn DiskTransport>,
    ) -> Result<Self> {
        let layout = Layout::new(&geometry)?;
        let mut index = Index::new(layout.page_tokens);
        let receive_profiles = transport.codec_receive_profiles();
        let can_read = |record: &Scanned| match &record.header.packed {
            None => true,
            Some(layout) => layout
                .required_profiles(record.row)
                .iter()
                .all(|p| receive_profiles.iter().any(|r| r.as_slice() == p.as_slice())),
        };
        let scanned = if scanned.iter().any(|record| !can_read(record)) {
            let mut kept = Vec::with_capacity(scanned.len());
            for record in scanned {
                if can_read(&record) {
                    kept.push(record);
                } else {
                    record::remove(&record.path)?;
                }
            }
            record::remove_orphans(kept)?
        } else {
            scanned
        };
        let mut used = 0u64;
        for record in &scanned {
            index.insert(
                record.header.key,
                record.header.parent,
                record.header.bucket,
                record.header.tail_pages as usize,
                record.row,
                true,
                DiskRecord {
                    path: record.path.clone(),
                    header: record.header.clone(),
                    bytes: record.bytes,
                    failed: false,
                },
            )?;
            used += record.bytes;
        }
        let mut by_age: Vec<&Scanned> = scanned.iter().collect();
        by_age.sort_by_key(|record| (record.modified, record.header.key.consumed_tokens));
        for (tick, record) in by_age.into_iter().enumerate() {
            index.access_at(&record.header.key, tick as u64 + 1);
            let node = index.node_mut(&record.header.key).unwrap();
            node.last_used = record.modified;
        }
        Ok(Self {
            index,
            identity,
            fingerprint,
            limit,
            budget: None,
            used,
            transport,
            stats: StoreStats::default(),
            next_job: 1,
            geometry,
            layout,
            outbound: HashMap::new(),
            inbound: HashMap::new(),
            maintenance: HashMap::new(),
            replacements: HashMap::new(),
            evicted: Vec::new(),
            outbound_rate: Rate::new(INITIAL_STAGED_DISK_RATE),
            inbound_rate: Rate::new(INITIAL_STAGED_DISK_RATE),
            announce: None,
            spill: false,
            preserve_committed: false,
            retiring: Vec::new(),
            held: HashMap::new(),
            finishing: Vec::new(),
            largest: scanned.iter().map(|r| r.bytes).max().unwrap_or(0),
            maintenance_progress: 0,
            thinning: scanned.iter().filter_map(|r| r.header.parent).collect(),
            directory_lock,
        })
    }

    /// Announce this store's records through `announcer` and serve them as
    /// exports. Recovered holdings are announced before this returns.
    pub fn announce_records(&mut self, announcer: Announcer) {
        if self.announce.is_some() {
            return;
        }
        let view = TierView::new(announcer, PoolTier::Nvme, &self.layout);
        self.index.observe(view.sink());
        view.sink()
            .lock()
            .unwrap()
            .extend(self.index.ready_notices());
        self.announce = Some(Announce {
            view,
            releases: Releases::default(),
        });
        self.flush_announcements();
    }

    /// What this tier announced for a complete record, as the next tier below
    /// memory: the entry a record spilled here is announced as.
    pub fn announced_entry(&self, key: &PrefixKey) -> Option<IndexEntry> {
        let node = self.index.node(key).filter(|node| node.ready)?;
        if !self.announce.as_ref()?.view.contains(key) {
            return None;
        }
        let (bytes, packed, required_profiles) = match &node.location.header.packed {
            Some(layout) => (
                if node.row {
                    layout.physical_bytes()
                } else {
                    layout.page_offset(layout.page_count())
                },
                true,
                layout.required_profiles(node.row),
            ),
            None => (
                node.tail_pages as u64 * self.layout.page_record_bytes
                    + if node.row {
                        self.layout.row_record_bytes
                    } else {
                        0
                    },
                false,
                Vec::new(),
            ),
        };
        Some(IndexEntry {
            key: *key,
            parent: node.parent,
            bucket: node.bucket,
            tail_pages: node.tail_pages as u32,
            row: node.row,
            tier: PoolTier::Nvme,
            bytes,
            packed,
            required_profiles,
        })
    }

    /// Release finished exports' pins and announce what changed.
    pub fn flush_announcements(&mut self) {
        let Some(a) = self.announce.as_mut() else {
            return;
        };
        for key in a.releases.drain() {
            let _ = self.index.unpin(&key);
            self.stats.progress += 1;
        }
        for notice in a.view.drain() {
            match notice.event {
                NodeEvent::Ready | NodeEvent::RowDropped => {
                    if let Some(node) = self
                        .index
                        .node(&notice.key)
                        .filter(|n| !n.location.header.replicated())
                    {
                        let packed = node.location.header.packed.as_ref().map(|layout| {
                            let bytes = if notice.row {
                                layout.physical_bytes()
                            } else {
                                layout.page_offset(layout.page_count())
                            };
                            (bytes, layout.required_profiles(notice.row))
                        });
                        a.view.announce_packed(&notice, notice.parent, packed);
                    }
                }
                NodeEvent::Gone => a.view.retire(&notice.key),
            }
        }
        let _ = self.finish_retired();
    }

    /// Pin the chain of complete record files ending at `key` for an export
    /// from `begin_tokens`. The files' own checksums are what is served.
    pub fn pin_export_files(
        &mut self,
        key: PrefixKey,
        begin_tokens: u64,
        want_row: bool,
    ) -> std::result::Result<Box<dyn HostPin>, ExportRefusal> {
        let Some(a) = self.announce.as_ref() else {
            return Err(ExportRefusal::NotFound);
        };
        if !a.view.contains(&key) {
            return Err(ExportRefusal::NotFound);
        }
        let page_tokens = self.layout.page_tokens;
        let start_page = begin_tokens / page_tokens;
        let mut keys = Vec::new();
        let mut nodes = Vec::new();
        let mut spans = Vec::new();
        let mut cursor = key;
        loop {
            let node = self
                .index
                .node(&cursor)
                .filter(|node| node.ready)
                .ok_or(ExportRefusal::ChainGap)?;
            if !a.view.contains(&cursor) {
                return Err(ExportRefusal::ChainGap);
            }
            if want_row && cursor == key && !node.row {
                return Err(ExportRefusal::ChainGap);
            }
            keys.push(cursor);
            nodes.push((cursor, node.tail_pages));
            spans.push(FileSpan::new(
                node.location.path.clone(),
                node.location.header.clone(),
            ));
            let start =
                (cursor.consumed_tokens / page_tokens).saturating_sub(node.tail_pages as u64);
            if start <= start_page {
                break;
            }
            cursor = node.parent.ok_or(ExportRefusal::ChainGap)?;
        }
        keys.reverse();
        nodes.reverse();
        spans.reverse();
        let (chain, kept) = host_chain_kept(page_tokens, begin_tokens, want_row, &nodes);
        if chain.total_pages != (key.consumed_tokens / page_tokens).saturating_sub(start_page) {
            return Err(ExportRefusal::ChainGap);
        }
        // Windows name records by their place in the chain, which leaves out nodes with no page.
        let mut spans: Vec<Option<FileSpan>> = spans.into_iter().map(Some).collect();
        let spans: Vec<FileSpan> = kept.iter().filter_map(|&i| spans[i].take()).collect();
        for (position, key) in keys.iter().enumerate() {
            if self.index.pin(key).is_err() {
                for pinned in &keys[..position] {
                    let _ = self.index.unpin(pinned);
                }
                return Err(ExportRefusal::NotFound);
            }
        }
        Ok(Box::new(FilePin {
            chain,
            spans,
            page_bytes: self.layout.page_record_bytes,
            row_bytes: self.layout.row_record_bytes,
            keys,
            releases: a.releases.clone(),
        }))
    }

    pub fn share_budget(&mut self, budget: Arc<super::input::DiskBudget>) {
        budget.set_checkpoint(self.used);
        self.budget = Some(budget);
    }

    fn reclaim_inputs(&mut self) {
        let pressure = self.budget.as_ref().map_or(0, |b| b.checkpoint_pressure());
        if pressure != 0 {
            let _ = self.make_room(pressure);
        }
    }

    fn sync_budget(&self) {
        if let Some(budget) = &self.budget {
            budget.set_checkpoint(self.used.saturating_add(self.pending_bytes()));
        }
    }

    #[cfg(test)]
    pub(crate) fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    pub fn index(&self) -> &Index<DiskRecord> {
        &self.index
    }

    pub fn index_mut(&mut self) -> &mut Index<DiskRecord> {
        &mut self.index
    }

    pub fn directory(&self) -> &Path {
        self.directory_lock.path()
    }

    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }

    pub fn fingerprint(&self) -> [u8; 16] {
        self.fingerprint
    }

    pub fn limit(&self) -> u64 {
        self.limit
    }

    pub fn used(&self) -> u64 {
        self.used
    }

    pub fn geometry(&self) -> &[CacheGeometry] {
        &self.geometry
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn transport_mut(&mut self) -> &mut dyn DiskTransport {
        self.transport.as_mut()
    }

    pub fn pending_store_key(&self, id: JobId) -> Option<PrefixKey> {
        self.outbound.get(&id).map(|job| job.key)
    }

    pub fn next_job(&mut self) -> super::JobId {
        let id = super::JobId(self.next_job);
        self.next_job += 1;
        id
    }

    pub fn record(&self, key: &PrefixKey) -> Option<&DiskRecord> {
        self.index.node(key).map(|node| &node.location)
    }

    /// Keys whose files were unlinked since the previous call.
    pub fn take_evicted(&mut self) -> Vec<PrefixKey> {
        std::mem::take(&mut self.evicted)
    }

    fn observe_inbound_samples(&mut self) {
        let mut samples = Vec::new();
        self.transport.drain_inbound_samples(&mut samples);
        for (bytes, elapsed) in samples {
            self.inbound_rate.observe(bytes, elapsed);
        }
    }

    /// Drain the transport's completions without applying them, for a
    /// composite store that drives the transport directly.
    pub fn poll_transport(&mut self, events: &mut Vec<TransportEvent>) -> Result<()> {
        self.reclaim_inputs();
        let polled = self.transport.poll(events);
        self.observe_inbound_samples();
        polled
    }

    fn accepts_children(&self, key: &PrefixKey) -> bool {
        self.index
            .node(key)
            .is_some_and(|node| !node.location.failed)
    }

    /// The one parent choice for a published checkpoint, used by the planner and
    /// by the row decision so a record's row and its link agree: the first
    /// offered ancestor this index accepts children under.
    fn select_parent(&self, checkpoint: &PublishedCheckpoint) -> Option<PrefixKey> {
        checkpoint
            .offered()
            .filter(|key| self.accepts_children(key))
            .find(|key| {
                checkpoint
                    .tail_below(key, self.layout.page_tokens)
                    .is_some()
            })
    }

    /// The record shape for a published checkpoint: a child of its selected
    /// ancestor when the parent is indexed, otherwise a root record holding the
    /// whole chain. The remnant of a failed write, which has neither a row nor
    /// readable pages, counts as absent.
    pub fn plan_record(
        &self,
        checkpoint: &PublishedCheckpoint,
    ) -> (Option<PrefixKey>, Vec<PageId>) {
        if let Some(node) = self.index.node(&checkpoint.key) {
            let parent = node.parent;
            let first = parent.map_or(0, |p| p.consumed_tokens / self.layout.page_tokens) as usize;
            return (parent, checkpoint.chain_pages[first..].to_vec());
        }
        let parent = self.select_parent(checkpoint);
        let tail = match parent.and_then(|key| checkpoint.tail_below(&key, self.layout.page_tokens))
        {
            Some(tail) => tail.to_vec(),
            None => checkpoint.chain_pages.clone(),
        };
        (parent, tail)
    }

    pub(crate) fn covered_endpoint(
        &self,
        checkpoint: &PublishedCheckpoint,
        readable: impl Fn(&Node<DiskRecord>) -> bool,
    ) -> bool {
        let held = self.index.node(&checkpoint.key);
        let parent = held.map_or_else(|| self.select_parent(checkpoint), |n| n.parent);
        let role = held.map_or(checkpoint.capture_role, |n| {
            n.location
                .header
                .capture_role()
                .merge(checkpoint.capture_role)
        });
        self.index.covered_endpoint(
            checkpoint.key,
            parent,
            role,
            held.map_or(0, |n| n.children()),
            readable,
        )
    }

    fn needs_row(&self, checkpoint: &PublishedCheckpoint) -> bool {
        self.index
            .node(&checkpoint.key)
            .is_some_and(|n| n.location.failed)
            || !self.covered_endpoint(checkpoint, |n| {
                n.ready || self.outbound.values().any(|write| write.key == n.key)
            })
    }

    pub fn store_status(&self, checkpoint: &PublishedCheckpoint) -> Result<Option<StoreOutcome>> {
        self.store_status_with_row(checkpoint, self.needs_row(checkpoint))
    }

    pub(crate) fn store_status_with_row(
        &self,
        checkpoint: &PublishedCheckpoint,
        row: bool,
    ) -> Result<Option<StoreOutcome>> {
        if checkpoint.tail_pages > checkpoint.chain_pages.len() {
            return Err(invalid("published checkpoint tail exceeds its chain"));
        }
        if let Some(node) = self.index.node(&checkpoint.key) {
            let stored = node.location.header.capture_role();
            let covers = stored.merge(checkpoint.capture_role) == stored;
            if node.ready && covers && ((node.row && self.index.complete(&checkpoint.key)) || !row)
            {
                return Ok(Some(StoreOutcome::Present));
            }
            let writing = self.outbound.values().any(|write| write.key == checkpoint.key);
            if (!node.ready && !node.location.failed)
                || writing
                || (node.row && covers)
                || node.pinned()
                || self
                    .maintenance
                    .values()
                    .any(|job| job.key == Some(checkpoint.key))
            {
                return Ok(Some(StoreOutcome::Retry(StoreReason::Inflight)));
            }
        }
        if self.outbound.len() >= RECORDS_IN_FLIGHT {
            return Ok(Some(StoreOutcome::Retry(StoreReason::Queue)));
        }
        let (parent, tail) = self.plan_record(checkpoint);
        let header = Header::new(
            &self.geometry,
            self.identity,
            self.fingerprint,
            checkpoint.key,
            parent,
            checkpoint.bucket,
            tail.len() as u32,
            SystemTime::now(),
        )?;
        let mut footprint = header.length(row);
        let mut cursor = parent;
        while let Some(key) = cursor {
            let node = &self.index[&key];
            // Admission pins the immediate parent. Older protected rows still count.
            let bytes = if Some(key) == parent && node.location.bytes != 0 {
                node.location.bytes
            } else {
                let protected = Some(key) == parent
                    || node.parent.is_none()
                    || node.location.header.capture_role() != crate::CaptureRole::Endpoint;
                node.location.header.length(node.row && protected)
            };
            footprint = footprint.saturating_add(bytes);
            cursor = node.parent;
        }
        Ok((footprint > self.limit).then_some(StoreOutcome::Oversize))
    }

    /// With `reserve=false`, index only uncharged metadata for a DRAM copy.
    /// Make room for `tail_pages` and, when requested, a row; index it as writing, pin it,
    /// and return its header and file names; `None` when evicting every
    /// unpinned record would still not bring it under the limit. The key must
    /// be absent or eligible for replacement. Its parent may still be writing.
    /// The parent is pinned while room is made, so eviction neither
    /// unlinks nor compacts it, and removing its other children never cascades
    /// into it, before the record names it.
    pub(crate) fn reserve_capture(
        &mut self,
        key: PrefixKey,
        parent: Option<PrefixKey>,
        bucket: [u8; 16],
        tail_pages: usize,
        row: bool,
        reserve: bool,
        capture_role: crate::CaptureRole,
    ) -> Result<Option<(Header, PathBuf, PathBuf)>> {
        let parked = self
            .index
            .node(&key)
            .is_some_and(|n| n.location.bytes == 0 && !n.location.failed);
        let capture_role = self.index.node(&key).map_or(capture_role, |node| {
            node.location.header.capture_role().merge(capture_role)
        });
        let replacement = if let Some(node) = self.index.node(&key).filter(|_| !parked) {
            let stored = node.location.header.capture_role();
            if node.ready && node.row && stored == capture_role {
                return Err(invalid("host record key is already indexed"));
            }
            let writing = self.outbound.values().any(|write| write.key == key);
            if (!node.ready && !node.location.failed)
                || writing
                || node.pinned()
                || self.maintenance.values().any(|job| job.key == Some(key))
            {
                return Ok(None);
            }
            if node.parent != parent || node.tail_pages != tail_pages {
                return Err(invalid(
                    "replacement must preserve the stored parent and page tail",
                ));
            }
            Some(node.location.clone())
        } else {
            None
        };
        if let Some(parent) = &parent
            && !self.accepts_children(parent)
        {
            return Err(invalid(
                "host record parent is not indexed or is the remnant of a failed write",
            ));
        }
        let tail = u32::try_from(tail_pages)
            .map_err(|_| invalid(format!("a record cannot hold {tail_pages} tail pages")))?;
        let header = Header::new(
            &self.geometry,
            self.identity,
            self.fingerprint,
            key,
            parent,
            bucket,
            tail,
            SystemTime::now(),
        )?
        .with_capture_role(capture_role);
        let bytes = header.length(row);
        if let Some(parent) = &parent {
            self.index.pin(parent)?;
        }
        if replacement.is_some() {
            self.index.pin(&key)?;
        }
        let room = if reserve {
            self.make_room(bytes)
        } else {
            Ok(true)
        };
        if replacement.is_some() {
            self.index.unpin(&key)?;
        }
        if let Some(parent) = &parent {
            self.index.unpin(parent)?;
        }
        if !room? {
            return Ok(None);
        }
        if reserve
            && let Some(budget) = &self.budget
            && !budget.reserve_checkpoint(bytes)
        {
            return Ok(None);
        }
        let bytes = if reserve { bytes } else { 0 };
        let path = self.directory().join(record::file_name(&key));
        let temporary = self.directory().join(record::temporary_name(&key));
        let location = DiskRecord {
            path: path.clone(),
            header: header.clone(),
            bytes,
            failed: false,
        };
        if parked {
            self.index.node_mut(&key).unwrap().location = location;
        } else if let Some(old) = replacement {
            self.replacements.insert(key, old);
            self.index.node_mut(&key).unwrap().location = location;
            self.index.set_ready(&key, false)?;
        } else {
            self.index
                .insert(key, parent, bucket, tail_pages, row, false, location)?;
        }
        if !parked {
            self.index.pin(&key)?;
        }
        self.used += bytes;
        self.largest = self.largest.max(bytes);
        self.stats.progress += 1;
        Ok(Some((header, temporary, path)))
    }

    /// Finish a reserved record: a successful write makes it ready; a failed
    /// one returns its bytes to the budget and drops it from the index.
    pub fn commit_record(
        &mut self,
        key: PrefixKey,
        result: Result<u64>,
        header: Option<Header>,
    ) -> Result<()> {
        self.finish_record(key, result, header, true)
    }

    fn finish_record(
        &mut self,
        key: PrefixKey,
        result: Result<u64>,
        header: Option<Header>,
        prior_readable: bool,
    ) -> Result<()> {
        self.stats.progress += 1;
        if let Ok(bytes) = &result {
            let node = self
                .index
                .node_mut(&key)
                .ok_or_else(|| invalid("cannot commit an unindexed host record"))?;
            if let Some(header) = header {
                node.location.header = header;
            }
            if node.location.header.replicated() {
                self.stats.packed_records[1] += 1;
            } else {
                self.stats.per_rank_records[1] += 1;
            }
            self.used -= node.location.bytes;
            self.used += *bytes;
            node.location.bytes = *bytes;
            node.row = node.location.header.row_present(*bytes);
            if let Some(parent) = node.parent {
                self.thinning.insert(parent);
            }
        }
        if let Some(old) = self.replacements.remove(&key) {
            let node = self.index.node_mut(&key).unwrap();
            let ready = if result.is_ok() {
                self.thinning.remove(&key);
                self.used -= old.bytes;
                self.stats.row_refills += 1;
                self.stats.stored += 1;
                self.stats.bytes_out += result.as_ref().copied().unwrap();
                self.evicted.push(key);
                true
            } else {
                self.stats.store_failed += 1;
                if prior_readable && !old.failed {
                    self.used -= node.location.bytes;
                    node.row = old.header.row_present(old.bytes);
                    node.location = old;
                    true
                } else if prior_readable {
                    // Replacement failed before overwriting the corrupt file; restore the
                    // failed record's accounting without making it readable again.
                    self.used -= node.location.bytes;
                    node.location = old;
                    node.row = false;
                    false
                } else {
                    // A failed cleanup can leave a complete replacement on one rank.
                    self.used -= old.bytes;
                    node.location.failed = true;
                    node.row = false;
                    false
                }
            };
            self.index.set_ready(&key, ready)?;
            self.index.unpin(&key)?;
            if !prior_readable {
                self.evicted.push(key);
                if self.index.node(&key).unwrap().children() == 0 {
                    for removed in self.index.remove(&key, |_| false)? {
                        self.unlink(removed)?;
                    }
                }
            }
            self.sync_budget();
            return Ok(());
        }
        match result {
            Ok(bytes) => {
                self.index.set_ready(&key, true)?;
                self.index.unpin(&key)?;
                self.stats.stored += 1;
                self.stats.bytes_out += bytes;
            }
            Err(_) => {
                self.stats.store_failed += 1;
                self.discard(&key)?;
            }
        }
        self.sync_budget();
        Ok(())
    }

    pub(crate) fn thin_rows(&mut self) {
        self.thin_rows_for(self.layout.row_record_bytes.saturating_mul(2));
    }

    fn free_retention_bytes(&self) -> u64 {
        self.budget
            .as_ref()
            .map_or(self.limit, |budget| budget.checkpoint_limit())
            .min(self.limit)
            .saturating_sub(self.used)
    }

    fn thin_rows_for(&mut self, headroom: u64) {
        if self.free_retention_bytes() > headroom {
            return;
        }
        let mut queued = self.index.thinning_order(&mut self.thinning);
        while self.maintenance.len() < MAINTENANCE_IN_FLIGHT {
            let Some(key) = queued.next() else {
                break;
            };
            if self.index[&key].location.header.capture_role() != crate::CaptureRole::Endpoint
                || !self.index.row_redundant(&key, true)
                || !self.index[&key].row
            {
                continue;
            }
            let node = &self.index[&key];
            if !self.index.has_row_below(&key, |n| n.ready && n.row) {
                self.thinning.insert(key);
                continue;
            }
            if !node.ready || node.pinned() {
                self.thinning.insert(key);
            } else if self.compact(&key).is_ok() {
                self.index.drop_row(&key);
                self.stats.rows_thinned[1] += 1;
            }
            if self.free_retention_bytes() > headroom {
                break;
            }
        }
        self.thinning.extend(queued);
    }

    /// Evict until `bytes` more fit under the limit; false when nothing
    /// evictable remains, and false before evicting anything when `bytes`
    /// alone exceed the limit.
    fn make_room(&mut self, bytes: u64) -> Result<bool> {
        if bytes > self.limit {
            return Ok(false);
        }
        if !self.preserve_committed && self.free_retention_bytes() < bytes {
            self.thin_rows_for(bytes);
        }
        while self.free_retention_bytes() < bytes {
            if self.preserve_committed {
                return Ok(false);
            }
            if self.spill && self.retire_for(bytes) {
                return Ok(false);
            }
            // At the truncate cap, whole-record eviction can precede further row reclaim.
            match self.index.evict(
                true,
                |node| {
                    node.children() == 0
                        || (node.parent.is_some()
                            && node.location.header.capture_role() == crate::CaptureRole::Endpoint)
                },
                |_| false,
            ) {
                Some(Eviction::Leaves(nodes)) => {
                    self.stats.dropped[3] += nodes.len() as u64;
                    for node in nodes {
                        self.unlink(node)?;
                    }
                }
                Some(Eviction::Compacted(key)) => {
                    self.stats.dropped[3] += 1;
                    self.compact(&key)?;
                }
                None => {
                    if let Some(budget) = &self.budget {
                        budget.request_input_reclaim(bytes);
                    }
                    return Ok(false);
                }
            }
        }
        Ok(self.budget.is_none()
            || self
                .used
                .saturating_add(self.pending_bytes())
                .saturating_add(bytes)
                <= self
                    .budget
                    .as_ref()
                    .unwrap()
                    .checkpoint_limit()
                    .min(self.limit))
    }

    fn pending_bytes(&self) -> u64 {
        self.maintenance.values().map(|reclaim| reclaim.bytes).sum()
    }

    pub(crate) fn maintain(&mut self) {
        let progress = self.stats().progress;
        if !self.preserve_committed && self.spill && self.maintenance_progress != progress {
            self.maintenance_progress = progress;
            let target = crate::store::headroom(self.limit, self.largest)
                .saturating_add(self.pending_bytes());
            // Leaves already held for handover free their bytes on their own.
            let held: u64 = self.held.values().sum();
            self.thin_rows_for(target.saturating_sub(held));
            self.retire_for(target);
        }
    }

    /// Hold bounded cold leaves for handover until they cover the demand.
    /// A full tier waits while any such records remain held.
    fn retire_for(&mut self, bytes: u64) -> bool {
        let limit = self
            .budget
            .as_ref()
            .map_or(self.limit, |b| b.checkpoint_limit())
            .min(self.limit);
        let mut freed: u64 = self.held.values().sum();
        for _ in 0..crate::pool::cascade::MOVES_MAX {
            if self.used.saturating_sub(freed).saturating_add(bytes) <= limit
                || self.held.len() >= crate::pool::cascade::MOVES_MAX
            {
                break;
            }
            let announce = self.announce.as_ref();
            let Some(key) = self.index.candidate(false, |node| {
                announce.is_some_and(|a| a.view.contains(&node.key))
            }) else {
                break;
            };
            let Some(node) = self.index.node(&key) else {
                break;
            };
            let retiring = Retiring {
                descendant_row: false,
                tier: PoolTier::Nvme,
                capture_role: node.location.header.capture_role(),
                child_paths: node.children(),
                key,
                parent: node.parent,
                ancestor_keys: self.index.ancestor_keys(&key),
                bucket: node.bucket,
                tail_pages: node.tail_pages,
                row: node.row
                    && !self.index.covered_endpoint(
                        key,
                        node.parent,
                        node.location.header.capture_role(),
                        node.children(),
                        |n| n.ready,
                    ),
            };
            let node_bytes = node.location.bytes;
            if self.index.pin(&key).is_err() {
                break;
            }
            freed += node_bytes;
            self.held.insert(key, node_bytes);
            self.retiring.push(retiring);
        }
        !self.held.is_empty()
    }

    /// The move of a held record ended: unlink it, or let it be offered again.
    fn retired(&mut self, key: &PrefixKey, how: Retirement) -> Result<()> {
        if self.held.remove(key).is_none() {
            return Ok(());
        }
        self.index.unpin(key)?;
        self.stats.held[1] += u64::from(how == Retirement::Failed);
        self.stats.progress += 1;
        if how != Retirement::Failed {
            self.finishing.push(*key);
            self.finish_retired()?;
        }
        Ok(())
    }

    /// Unlink the records whose moves ended, once no export pin still reads
    /// them. A record that gained a child meanwhile stays.
    fn finish_retired(&mut self) -> Result<()> {
        let mut finishing = std::mem::take(&mut self.finishing);
        let mut result = Ok(());
        finishing.retain(|key| {
            if self.index.node(key).is_some_and(|node| node.pinned()) {
                return true;
            }
            if let Ok(removed) = self.index.remove(key, |_| false) {
                for node in removed {
                    if let Err(error) = self.unlink(node) {
                        result = Err(error);
                    }
                }
                self.stats.progress += 1;
            }
            false
        });
        self.finishing = finishing;
        result
    }

    /// Return a removed node's bytes to the budget and unlink its file.
    fn unlink(&mut self, node: Node<DiskRecord>) -> Result<()> {
        self.thinning.remove(&node.key);
        self.used = self.used.saturating_sub(node.location.bytes);
        if node.location.bytes == 0 {
            return Ok(());
        }
        let id = self.next_job();
        self.maintenance.insert(
            id,
            Reclaim {
                key: None,
                bytes: node.location.bytes,
            },
        );
        self.sync_budget();
        if let Err(error) = self.transport.unlink(id, node.location.path) {
            self.apply(
                TransportEvent::Unlinked {
                    id,
                    result: Err(error.clone()),
                },
                &mut Vec::new(),
            );
            return Err(error);
        }
        self.evicted.push(node.key);
        Ok(())
    }

    pub(crate) fn expire(&mut self, key: PrefixKey) -> Result<()> {
        for node in self.index.remove(&key, |_| false)? {
            if node.location.bytes == 0 {
                self.evicted.push(node.key);
            }
            // Unready records are parked memory copies in a composite.
            self.stats.expired[usize::from(node.ready)] += 1;
            self.unlink(node)?;
        }
        Ok(())
    }

    /// Drop a compacted node's row segment from its file and the budget.
    fn compact(&mut self, key: &PrefixKey) -> Result<()> {
        let node = self
            .index
            .node_mut(key)
            .ok_or_else(|| invalid("cannot compact an unindexed host record"))?;
        let length = node.location.header.row_segment;
        let freed = node.location.bytes.saturating_sub(length);
        node.location.bytes = length;
        let path = node.location.path.clone();
        self.used = self.used.saturating_sub(freed);
        let id = self.next_job();
        self.maintenance.insert(
            id,
            Reclaim {
                key: Some(*key),
                bytes: freed,
            },
        );
        self.sync_budget();
        if let Err(error) = self.transport.truncate(id, path, length) {
            self.apply(
                TransportEvent::Truncated {
                    id,
                    result: Err(error.clone()),
                },
                &mut Vec::new(),
            );
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn is_failed(&self, key: &PrefixKey) -> bool {
        self.index.node(key).is_some_and(|n| n.location.failed)
    }

    /// Quarantine a corrupt disk record whose pages failed checksum verification.
    /// Drops readiness and lookup eligibility so descendants cannot re-select it,
    /// while marking it `failed` so a subsequent capture can replace it in place
    /// once outstanding pins drain (instead of spinning in `Retry(Inflight)`).
    pub(crate) fn quarantine_corrupt(&mut self, key: &PrefixKey) {
        let _ = self.index.set_ready(key, false);
        self.index.drop_row(key);
        if let Some(node) = self.index.node_mut(key) {
            node.location.failed = true;
        }
    }

    /// Drop a record whose write failed. It leaves the index unless children
    /// already name it as their parent; then it stays as a remnant without a
    /// row or a file until the last child is removed.
    fn discard(&mut self, key: &PrefixKey) -> Result<()> {
        self.index.unpin(key)?;
        self.thinning.remove(key);
        let node = self
            .index
            .node_mut(key)
            .ok_or_else(|| invalid("cannot discard an unindexed host record"))?;
        node.location.failed = true;
        let bytes = std::mem::take(&mut node.location.bytes);
        self.used = self.used.saturating_sub(bytes);
        if node.children() == 0 {
            for removed in self.index.remove(key, |_| false)? {
                self.unlink(removed)?;
            }
        } else {
            self.index.drop_row(key);
        }
        self.sync_budget();
        Ok(())
    }

    /// Plan the segments that restore a looked-up chain: the first record
    /// contributes its tail after the pages the GPU holds, every other record
    /// its whole tail, and the last record the row.
    pub fn plan_fetch(&self, chain: &Match) -> Result<Vec<FetchSegment>> {
        let last = chain
            .nodes
            .len()
            .checked_sub(1)
            .ok_or_else(|| invalid("host record chain is empty"))?;
        let mut plan = Vec::with_capacity(chain.nodes.len());
        for (position, key) in chain.nodes.iter().enumerate() {
            let node = self
                .index
                .node(key)
                .ok_or_else(|| invalid("host record chain names an unindexed record"))?;
            if !node.ready {
                return Err(invalid(
                    "host record chain names a record whose write has not completed",
                ));
            }
            let skip = if position == 0 { chain.skip_pages } else { 0 };
            let pages = node.tail_pages.checked_sub(skip).ok_or_else(|| {
                invalid("host record chain skips more pages than its first record holds")
            })?;
            let row = position == last;
            if row && !node.row {
                return Err(invalid("host record chain ends at a record without a row"));
            }
            plan.push(FetchSegment {
                path: node.location.path.clone(),
                header: node.location.header.clone(),
                first_page: skip as u32,
                pages: pages as u32,
                row,
            });
        }
        Ok(plan)
    }

    /// A hit for `chain` priced at this store's measured inbound rate.
    pub fn hit(&self, chain: Match) -> HostHit {
        let bytes = self.layout.record_bytes(chain.pages as u64);
        HostHit {
            key: chain.key,
            consumed_tokens: chain.consumed_tokens,
            resident_tokens: chain.resident_tokens,
            pages: chain.pages,
            bytes,
            tier: Tier::Nvme,
            estimated_seconds: bytes as f64 / self.inbound_rate.get().max(1.0),
            chain,
        }
    }

    /// Apply one transport completion to the index, counters, and rates.
    pub fn apply(&mut self, event: TransportEvent, events: &mut Vec<StoreEvent>) {
        self.stats.progress += 1;
        match event {
            TransportEvent::Stored { id, result, header } => {
                if self.outbound.contains_key(&id)
                    && result.as_ref().err().is_some_and(|e| {
                        e.carried().is_some_and(|e| e.is::<disk::ReplicaMismatch>())
                    })
                {
                    self.stats.abandoned_stores += 1;
                }
                self.finish_outbound(id, result, header.map(|h| *h), true, events);
            }
            TransportEvent::StoreInvalidated { id, error } => {
                self.finish_outbound(id, Err(error), None, false, events);
            }
            TransportEvent::Fetched { id, result } => {
                let Some(job) = self.inbound.remove(&id) else {
                    return;
                };
                for key in &job.nodes {
                    let _ = self.index.unpin(key);
                }
                match &result {
                    Ok(bytes) => {
                        for key in job.nodes.iter().rev().skip(1).rev() {
                            self.index.access(key);
                        }
                        if let Some(key) = job.nodes.last() {
                            self.reused(*key);
                        }
                        self.stats.fetched += 1;
                        self.stats.restores_by_tier[crate::Tier::Nvme as usize] += 1;
                        self.stats.bytes_in += bytes;
                        self.stats.reused_tokens += job.tokens;
                    }
                    Err(error) => {
                        self.stats.fetch_failed += 1;
                        // Only failures opening or reading the record file
                        // invalidate the lookup row; other failures keep it.
                        if error.is_record_read()
                            && let Some(endpoint) = job.nodes.last()
                        {
                            // Its pages stay in the index for descendants that
                            // hold it as an ancestor; only its lookup row is dropped.
                            self.index.drop_row(endpoint);
                        }
                    }
                }
                events.push(StoreEvent::Fetched {
                    key: job.key,
                    result,
                });
            }
            TransportEvent::Truncated { id, result } | TransportEvent::Unlinked { id, result } => {
                let Some(reclaim) = self.maintenance.remove(&id) else {
                    return;
                };
                let Reclaim { key, bytes } = reclaim;
                if result.is_ok() {
                    self.sync_budget();
                    return;
                }
                // The file keeps the bytes, so they stay charged against the
                // limit until the next startup scan reconciles the directory.
                // Only an accepted compaction makes the row unmatchable. An
                // unlinked record is already out of the index.
                if let Some(node) = key.and_then(|key| self.index.node_mut(&key))
                    && node.location.bytes == node.location.header.row_segment
                {
                    node.location.bytes += bytes;
                }
                self.used = self.used.saturating_add(bytes);
                self.sync_budget();
            }
        }
    }

    fn finish_outbound(
        &mut self,
        id: JobId,
        result: Result<u64>,
        header: Option<Header>,
        prior_readable: bool,
        events: &mut Vec<StoreEvent>,
    ) {
        let Some(job) = self.outbound.remove(&id) else {
            return;
        };
        if let Ok(bytes) = &result {
            self.outbound_rate.observe(*bytes, job.started.elapsed());
        }
        let pressure = result.as_ref().err().is_some_and(Error::no_space);
        let result = match self.finish_record(job.key, result.clone(), header, prior_readable) {
            Ok(()) => result,
            Err(error) => Err(error),
        };
        self.stats.dropped[5] += u64::from(result.is_err() && !pressure);
        self.stats.held[1] += u64::from(pressure);
        events.push(if pressure {
            StoreEvent::Deferred {
                checkpoint: job.checkpoint,
                reason: StoreReason::Budget,
            }
        } else {
            StoreEvent::Stored {
                checkpoint: job.checkpoint,
                key: job.key,
                result,
            }
        });
    }
}

impl HostStore for DiskStore {
    fn reused(&mut self, key: PrefixKey) {
        self.index.touch(&key);
    }

    fn lookup(&mut self, history: &dyn History, resident: u64, before: u64) -> Option<HostHit> {
        self.stats.lookups += 1;
        let chain = self
            .index
            .lookup_below(history, resident, before, |node| node.ready)?;
        self.stats.hits += 1;
        Some(self.hit(chain))
    }

    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> Result<StoreOutcome> {
        let row = self.needs_row(checkpoint);
        if let Some(outcome) = self.store_status_with_row(checkpoint, row)? {
            self.stats.store_skipped += 1;
            if outcome == StoreOutcome::Present {
                self.index.touch_ancestors(&checkpoint.key);
            }
            self.stats.dropped[4] += u64::from(outcome == StoreOutcome::Oversize);
            return Ok(outcome);
        }
        let (parent, tail) = self.plan_record(checkpoint);
        let Some((header, temporary, path)) = self.reserve_capture(
            checkpoint.key,
            parent,
            checkpoint.bucket,
            tail.len(),
            row,
            true,
            checkpoint.capture_role,
        )?
        else {
            self.stats.store_skipped += 1;
            return Ok(StoreOutcome::Retry(StoreReason::Budget));
        };
        let id = self.next_job();
        let detect = self.announce.is_none() && self.geometry.len() > 1;
        self.index.set_published_ancestors(&checkpoint.key, parent, &checkpoint.ancestors);
        let write = if row {
            self.transport
                .store(id, header, temporary, path, checkpoint, detect)
        } else {
            self.transport.store_empty(id, header, temporary, path)
        };
        if let Err(error) = write {
            self.commit_record(checkpoint.key, Err(error.clone()), None)?;
            return Err(error);
        }
        self.outbound.insert(
            id,
            Outbound {
                checkpoint: checkpoint.id,
                key: checkpoint.key,
                started: Instant::now(),
            },
        );
        Ok(StoreOutcome::Queued)
    }

    fn fetch(&mut self, hit: HostHit, ticket: InboundTicket) -> Result<()> {
        if hit.key != ticket.key {
            return Err(invalid(
                "inbound ticket names a different key than the host hit",
            ));
        }
        if self.inbound.len() >= RECORDS_IN_FLIGHT {
            return Err(invalid(format!(
                "disk store already has {RECORDS_IN_FLIGHT} inbound records in flight"
            )));
        }
        let chain = hit.chain;
        let plan = self.plan_fetch(&chain)?;
        let pages: usize = plan.iter().map(|segment| segment.pages as usize).sum();
        if pages != ticket.pages.len() || pages != chain.pages {
            return Err(invalid(format!(
                "host record chain supplies {pages} pages; the hit covers {} and the ticket reserved {}",
                chain.pages,
                ticket.pages.len()
            )));
        }
        for (pinned, key) in chain.nodes.iter().enumerate() {
            if let Err(error) = self.index.pin(key) {
                for key in &chain.nodes[..pinned] {
                    let _ = self.index.unpin(key);
                }
                return Err(error);
            }
        }
        let id = self.next_job();
        if let Err(error) = self.transport.fetch(id, plan, &ticket) {
            for key in &chain.nodes {
                let _ = self.index.unpin(key);
            }
            return Err(error);
        }
        self.inbound_rate.started();
        self.inbound.insert(
            id,
            Inbound {
                key: hit.key,
                nodes: chain.nodes,
                tokens: chain.consumed_tokens.saturating_sub(chain.resident_tokens),
            },
        );
        Ok(())
    }

    fn poll(&mut self, events: &mut Vec<StoreEvent>) -> Result<()> {
        for key in self.index.expired(|node| !node.pinned()) {
            self.expire(key)?;
        }
        self.reclaim_inputs();
        let mut completed = Vec::new();
        let polled = self.transport.poll(&mut completed);
        self.observe_inbound_samples();
        for event in completed {
            self.apply(event, events);
        }
        self.flush_announcements();
        self.thin_rows();
        self.maintain();
        polled
    }

    fn stats(&self) -> StoreStats {
        let pool = self.transport.staged().and_then(Staged::pool);
        let decode = self
            .transport
            .decode_counters()
            .map(|c| *c.lock().unwrap())
            .unwrap_or_default();
        StoreStats {
            decode,
            slots_used: pool.map_or(0, |pool| pool.slot_count() - pool.free_slots()),
            slots_total: pool.map_or(0, SlabPool::slot_count),
            intake_slots: pool.map_or(0, |pool| pool.intake_slots()),
            outbound_in_flight: self.outbound.len(),
            inbound_in_flight: self.inbound.len(),
            disk_records: self.index.len(),
            disk_bytes: self.used,
            progress: self.stats.progress
                + self.budget.as_ref().map_or(0, |budget| budget.progress()),
            outbound_rate: self.outbound_rate.get(),
            inbound_rate: self.inbound_rate.get(),
            ..self.stats
        }
    }

    fn close(&mut self) -> Result<()> {
        let closed = self.transport.close();
        let mut completed = Vec::new();
        let _ = self.transport.poll(&mut completed);
        self.observe_inbound_samples();
        let mut events = Vec::new();
        for event in completed {
            self.apply(event, &mut events);
        }
        closed
    }
}

impl crate::store::TierStore for DiskStore {
    fn announce_to(&mut self, announcer: Announcer) {
        self.announce_records(announcer);
    }
    fn pin_export(
        &mut self,
        tier: PoolTier,
        key: PrefixKey,
        begin_tokens: u64,
        want_row: bool,
    ) -> std::result::Result<Box<dyn HostPin>, ExportRefusal> {
        if tier != PoolTier::Nvme {
            return Err(ExportRefusal::NotFound);
        }
        self.pin_export_files(key, begin_tokens, want_row)
    }
    fn set_spill(&mut self, enabled: bool) {
        let enabled = enabled && self.announce.is_some();
        self.stats.progress += u64::from(self.spill != enabled);
        self.spill = enabled;
    }
    fn take_retiring(&mut self) -> Vec<Retiring> {
        std::mem::take(&mut self.retiring)
    }
    fn retirement_predecessor(&self, key: PrefixKey, candidates: bool) -> Option<Retiring> {
        let node = self.index.node(&key)?;
        if !node.ready {
            return None;
        }
        Some(Retiring {
            descendant_row: false,
            tier: PoolTier::Nvme,
            capture_role: node.location.header.capture_role(),
            child_paths: node.children(),
            key,
            parent: node.parent,
            ancestor_keys: if candidates {
                self.index.ancestor_keys(&key)
            } else {
                Vec::new()
            },
            bucket: node.bucket,
            tail_pages: node.tail_pages,
            row: node.row
                && !self.index.covered_endpoint(
                    key,
                    node.parent,
                    node.location.header.capture_role(),
                    node.children(),
                    |n| n.ready,
                ),
        })
    }
    fn retired(&mut self, key: PrefixKey, tier: PoolTier, how: Retirement) {
        if tier == PoolTier::Nvme {
            let _ = DiskStore::retired(self, &key, how);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::poll_until;
    use crate::testing::{Ids, checkpoint, geometry as rank, pattern, test_directory};
    use crate::testing::{StoppedCopier, TestHistory as PrefixHistory};
    use std::{
        cell::RefCell,
        fs,
        io::{Read as _, Seek, SeekFrom, Write as _},
        path::Path,
        rc::Rc,
        time::UNIX_EPOCH,
    };

    const IDENTITY: [u8; 32] = [0x5a; 32];
    const FINGERPRINT: [u8; 16] = [0xa5; 16];
    const PAGE_TOKENS: u64 = 64;
    /// Page record bytes of the test geometry: two 8 KiB planes.
    const PAGE_RECORD: u64 = 16384;
    /// Row record bytes of the test geometry: one 4 KiB row.
    const ROW_RECORD: u64 = 4096;
    const CREATED: Duration = Duration::from_secs(1_700_000_000);

    fn geometry() -> Vec<CacheGeometry> {
        vec![rank(PAGE_TOKENS, &[("k", 8192), ("v", 8192)], 4096)]
    }

    fn history(tokens: u64) -> PrefixHistory {
        let mut history = PrefixHistory::new(IDENTITY);
        let tokens: Vec<u32> = (0..tokens as u32)
            .map(|t| t.wrapping_mul(2654435761))
            .collect();
        history.append(&tokens, &[]).unwrap();
        history
    }

    fn header(
        geometry: &[CacheGeometry],
        history: &PrefixHistory,
        consumed: u64,
        parent: Option<u64>,
    ) -> Header {
        let tail = consumed / PAGE_TOKENS - parent.map_or(0, |parent| parent / PAGE_TOKENS);
        Header::new(
            geometry,
            IDENTITY,
            FINGERPRINT,
            history.key_at(consumed).unwrap(),
            parent.map(|parent| history.key_at(parent).unwrap()),
            history.bucket_at(consumed).unwrap(),
            tail as u32,
            UNIX_EPOCH + CREATED,
        )
        .unwrap()
    }

    fn endpoint_header(
        geometry: &[CacheGeometry],
        history: &PrefixHistory,
        consumed: u64,
        parent: Option<u64>,
    ) -> Header {
        header(geometry, history, consumed, parent).with_capture_role(crate::CaptureRole::Endpoint)
    }

    fn scanned(directory: &Path, header: Header, age: u64, row: bool) -> Scanned {
        let bytes = if row {
            header.length_with_row()
        } else {
            header.row_segment
        };
        Scanned {
            path: directory.join(record::file_name(&header.key)),
            header,
            row,
            bytes,
            modified: SystemTime::now() - Duration::from_secs(50u64.saturating_sub(age)),
        }
    }

    fn pages(range: std::ops::Range<u32>) -> Vec<PageId> {
        range.map(PageId).collect()
    }

    #[derive(Debug)]
    enum Call {
        Empty {
            id: JobId,
            header: Header,
        },
        Store {
            id: JobId,
            header: Header,
            temporary: PathBuf,
            path: PathBuf,
            pages: Vec<PageId>,
            slot: StateSlotId,
        },
        Fetch {
            id: JobId,
            plan: Vec<FetchSegment>,
            ticket: InboundTicket,
        },
        Truncate {
            id: JobId,
            path: PathBuf,
            length: u64,
        },
        Unlink {
            id: JobId,
            path: PathBuf,
        },
    }

    #[derive(Default)]
    struct Log {
        calls: Vec<Call>,
        events: Vec<TransportEvent>,
        closed: bool,
        fail_truncate: bool,
    }

    /// A transport that records what the store asks of it and replies with
    /// whatever events the test queues.
    struct Recorder(Rc<RefCell<Log>>);

    impl Drop for Recorder {
        fn drop(&mut self) {
            let _ = fs::remove_dir(directory());
        }
    }

    impl DiskTransport for Recorder {
        fn store_empty(&mut self, id: JobId, header: Header, _: PathBuf, _: PathBuf) -> Result<()> {
            self.0.borrow_mut().calls.push(Call::Empty { id, header });
            Ok(())
        }

        fn store(
            &mut self,
            id: JobId,
            header: Header,
            temporary: PathBuf,
            path: PathBuf,
            checkpoint: &PublishedCheckpoint,
            _detect: bool,
        ) -> Result<()> {
            let tail = header.tail_pages as usize;
            let pages = checkpoint.chain_pages[checkpoint.chain_pages.len() - tail..].to_vec();
            self.0.borrow_mut().calls.push(Call::Store {
                id,
                header,
                temporary,
                path,
                pages,
                slot: checkpoint.slot,
            });
            Ok(())
        }

        fn fetch(
            &mut self,
            id: JobId,
            plan: Vec<FetchSegment>,
            ticket: &InboundTicket,
        ) -> Result<()> {
            self.0.borrow_mut().calls.push(Call::Fetch {
                id,
                plan,
                ticket: ticket.clone(),
            });
            Ok(())
        }

        fn truncate(&mut self, id: JobId, path: PathBuf, length: u64) -> Result<()> {
            self.0
                .borrow_mut()
                .calls
                .push(Call::Truncate { id, path, length });
            if self.0.borrow().fail_truncate {
                return Err(invalid("truncate submission failed"));
            }
            Ok(())
        }

        fn unlink(&mut self, id: JobId, path: PathBuf) -> Result<()> {
            self.0.borrow_mut().calls.push(Call::Unlink { id, path });
            Ok(())
        }

        fn poll(&mut self, events: &mut Vec<TransportEvent>) -> Result<()> {
            events.append(&mut self.0.borrow_mut().events);
            Ok(())
        }

        fn close(&mut self) -> Result<()> {
            self.0.borrow_mut().closed = true;
            Ok(())
        }
    }

    fn directory() -> PathBuf {
        std::env::temp_dir().join(format!(
            "flare-disk-store-mock-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn store_with(scanned: Vec<Scanned>, limit: u64) -> (DiskStore, Rc<RefCell<Log>>) {
        let log = Rc::new(RefCell::new(Log::default()));
        let store = DiskStore::new(
            crate::testing::directory_lock(&directory()).unwrap(),
            IDENTITY,
            FINGERPRINT,
            limit,
            geometry(),
            scanned,
            Box::new(Recorder(log.clone())),
        )
        .unwrap();
        (store, log)
    }

    #[test]
    fn pressure_thinning_keeps_recovered_roots_branches_and_both_leaves() {
        let full = history(256);
        let headers = [
            endpoint_header(&geometry(), &full, 64, None),
            endpoint_header(&geometry(), &full, 128, Some(64)),
            endpoint_header(&geometry(), &full, 192, Some(128)),
            endpoint_header(&geometry(), &full, 256, Some(128)),
        ];
        let records = headers
            .iter()
            .enumerate()
            .map(|(i, h)| scanned(&directory(), h.clone(), i as u64, true))
            .collect();
        let (mut store, log) = store_with(records, 1 << 30);
        store.set_limit(store.used());
        store.thin_rows();
        assert!(store.index.nodes().all(|n| n.row));
        assert!(log.borrow().calls.is_empty());
    }

    #[test]
    fn parent_rows_survive_headroom_and_pressure_reclaims_in_access_order() {
        let full = history(5 * PAGE_TOKENS);
        let headers: Vec<_> = (1..=5)
            .map(|turn| {
                endpoint_header(
                    &geometry(),
                    &full,
                    turn * PAGE_TOKENS,
                    (turn > 1).then(|| (turn - 1) * PAGE_TOKENS),
                )
            })
            .collect();
        let records = headers
            .iter()
            .enumerate()
            .map(|(age, h)| scanned(&directory(), h.clone(), age as u64, true))
            .collect();
        let (mut store, log) = store_with(records, 1 << 30);
        let used = store.used();
        store.set_limit(used + 2 * ROW_RECORD + 1);
        store.poll(&mut Vec::new()).unwrap();
        assert!(store.index.nodes().all(|node| node.row));
        assert!(log.borrow().calls.is_empty());

        store.reused(headers[1].key);
        store.index.pin(&headers[2].key).unwrap();
        store.set_limit(used + 2 * ROW_RECORD);
        store.poll(&mut Vec::new()).unwrap();
        assert!(store.index[&headers[1].key].row);
        assert!(store.index[&headers[2].key].row);
        assert!(!store.index[&headers[3].key].row);
        assert_eq!(store.stats().rows_thinned, [0, 1]);
        assert_eq!(store.used(), used - ROW_RECORD);

        store.index.unpin(&headers[2].key).unwrap();
        assert!(store.make_room(3 * ROW_RECORD + 1).unwrap());
        assert!(!store.index[&headers[2].key].row);
        assert!(store.index[&headers[1].key].row);
        assert!(store.index[&headers[0].key].row && store.index[&headers[4].key].row);
        assert_eq!(store.index.len(), headers.len());
        assert_eq!(store.stats().rows_thinned, [0, 2]);
        assert!(matches!(log.borrow().calls.as_slice(),
            [Call::Truncate { path: first, .. }, Call::Truncate { path: second, .. }]
                if *first == store.record(&headers[3].key).unwrap().path
                && *second == store.record(&headers[2].key).unwrap().path));
    }

    #[test]
    fn maintenance_reclaims_parent_rows_before_retiring_leaves() {
        let full = history(66 * PAGE_TOKENS);
        let headers = [
            endpoint_header(&geometry(), &full, 64 * PAGE_TOKENS, None),
            endpoint_header(&geometry(), &full, 65 * PAGE_TOKENS, Some(64 * PAGE_TOKENS)),
            endpoint_header(&geometry(), &full, 66 * PAGE_TOKENS, Some(65 * PAGE_TOKENS)),
        ];
        let records = headers
            .iter()
            .enumerate()
            .map(|(age, h)| scanned(&directory(), h.clone(), age as u64, true))
            .collect();
        let (mut store, log) = store_with(records, 1 << 30);
        store.set_limit(store.used() * 10 / 9 - ROW_RECORD / 2);
        let target = crate::store::headroom(store.limit, store.largest);
        assert!(store.free_retention_bytes() > 2 * ROW_RECORD);
        assert!(store.free_retention_bytes() < target);
        assert!(store.free_retention_bytes() + ROW_RECORD > target);
        let _announcer = announced(&mut store);
        store.set_spill(true);
        store.poll(&mut Vec::new()).unwrap();
        assert!(!store.index[&headers[1].key].row);
        assert!(store.take_retiring().is_empty());
        assert!(store.held.is_empty());
        assert_eq!(store.index.len(), headers.len());
        assert!(store.index[&headers[2].key].row);
        assert!(matches!(log.borrow().calls.as_slice(),
            [Call::Truncate { length, .. }] if *length == headers[1].row_segment));
    }

    #[test]
    fn thinning_queue_keeps_only_resident_records_after_removal_and_eviction() {
        for removal in ["expire", "evict", "discard", "retire"] {
            let (mut store, _) = store_with(Vec::new(), 1 << 30);
            let _announcer = announced(&mut store);
            for salt in 1..=64 {
                let mut full = PrefixHistory::new(IDENTITY);
                full.append(&vec![salt; 192], &[]).unwrap();
                let headers = [
                    header(&geometry(), &full, 64, None),
                    header(&geometry(), &full, 128, Some(64)),
                    header(&geometry(), &full, 192, Some(128)),
                ];
                for header in &headers {
                    assert!(
                        store
                            .reserve_capture(
                                header.key,
                                header.parent,
                                header.bucket,
                                header.tail_pages as usize,
                                true,
                                true,
                                crate::CaptureRole::Endpoint,
                            )
                            .unwrap()
                            .is_some()
                    );
                    store
                        .commit_record(header.key, Ok(header.length_with_row()), None)
                        .unwrap();
                }
                assert!(store.thinning.contains(&headers[1].key));
                for header in headers.iter().rev() {
                    match removal {
                        "expire" => store.expire(header.key).unwrap(),
                        "evict" => {
                            let Some(Eviction::Leaves(nodes)) =
                                store.index.evict(false, |_| true, |_| false)
                            else {
                                panic!("the chain must have an evictable leaf");
                            };
                            for node in nodes {
                                store.unlink(node).unwrap();
                            }
                        }
                        "discard" => {
                            store.index.pin(&header.key).unwrap();
                            store.discard(&header.key).unwrap();
                        }
                        "retire" => {
                            store.flush_announcements();
                            assert!(store.retire_for(store.limit));
                            assert_eq!(store.take_retiring()[0].key, header.key);
                            store.retired(&header.key, Retirement::Moved).unwrap();
                        }
                        _ => unreachable!(),
                    }
                    assert!(
                        store.thinning.iter().all(|key| store.index.contains(key)),
                        "stale thinning keys after chain {salt}, removal={removal}"
                    );
                    assert!(store.thinning.len() <= store.index.len());
                }
                assert!(store.index.is_empty());
                assert!(store.thinning.is_empty());
            }
        }
    }

    #[test]
    fn recovered_message_and_anchor_rows_survive_a_same_page_endpoint() {
        let full = history(256);
        let records: Vec<_> = [
            endpoint_header(&geometry(), &full, 32, None),
            header(&geometry(), &full, 64, Some(32)),
            header(&geometry(), &full, 128, Some(64))
                .with_capture_role(crate::CaptureRole::Message),
            header(&geometry(), &full, 130, Some(128))
                .with_capture_role(crate::CaptureRole::Anchor),
            endpoint_header(&geometry(), &full, 192, Some(130)),
        ]
        .into_iter()
        .map(|h| scanned(&directory(), h, 0, true))
        .collect();
        let (mut store, log) = store_with(records, 1 << 30);
        store.thin_rows();
        let endpoint = checkpoint(
            &mut Ids::new(),
            &full,
            134,
            Some(130),
            pages(0..2),
            0,
            StateSlotId(1),
        );
        assert_eq!(store.store(&endpoint).unwrap(), StoreOutcome::Queued);
        let (id, header) = match log.borrow().calls.as_slice() {
            [Call::Empty { id, header }] => (*id, header.clone()),
            calls => panic!("expected a header-only link, got {calls:?}"),
        };
        complete(
            &log,
            TransportEvent::Stored {
                id,
                result: Ok(header.length(false)),
                header: None,
            },
            &mut store,
        );
        assert_eq!(store.store(&endpoint).unwrap(), StoreOutcome::Present);
        store.set_limit(store.used());
        store.thin_rows();
        for tokens in [32, 64, 128, 130, 192] {
            assert!(store.index[&full.key_at(tokens).unwrap()].row);
        }
        assert!(!store.index[&endpoint.key].row);
        assert_eq!(store.lookup(&full, 0, 135).unwrap().consumed_tokens, 130);
    }

    #[test]
    fn a_covered_endpoint_uses_its_held_ancestors_row() {
        // The Message at 128 holds a row. The prompt-end Endpoint at 130 was never stored.
        // The terminal Endpoint at 134 shares their page, so the Message covers it, and
        // its record must hang under the Message with no row of its own.
        let full = history(256);
        let records: Vec<_> = [
            endpoint_header(&geometry(), &full, 32, None),
            header(&geometry(), &full, 64, Some(32)),
            header(&geometry(), &full, 128, Some(64))
                .with_capture_role(crate::CaptureRole::Message),
        ]
        .into_iter()
        .map(|h| scanned(&directory(), h, 0, true))
        .collect();
        let (mut store, log) = store_with(records, 1 << 30);
        let message = full.key_at(128).unwrap();
        let mut endpoint = checkpoint(
            &mut Ids::new(),
            &full,
            134,
            Some(130),
            pages(0..2),
            0,
            StateSlotId(1),
        );
        endpoint.ancestors = vec![full.key_at(130).unwrap(), message];
        let (parent, tail) = store.plan_record(&endpoint);
        assert_eq!(parent, Some(message));
        assert!(tail.is_empty());
        assert!(store.covered_endpoint(&endpoint, |n| n.ready));
        assert_eq!(store.store(&endpoint).unwrap(), StoreOutcome::Queued);
        let (id, header) = match log.borrow().calls.as_slice() {
            [Call::Empty { id, header }] => (*id, header.clone()),
            calls => panic!("expected a header-only link, got {calls:?}"),
        };
        assert_eq!(header.parent, Some(message));
        complete(
            &log,
            TransportEvent::Stored {
                id,
                result: Ok(header.length(false)),
                header: None,
            },
            &mut store,
        );
        let node = &store.index[&endpoint.key];
        assert_eq!(node.parent, Some(message));
        assert_eq!(node.published_ancestors, [full.key_at(130).unwrap()]);
        assert!(!node.row);
        // Without any ancestor the disk keeps a root and its row.
        let mut lone = endpoint.clone();
        lone.key = full.key_at(135).unwrap();
        lone.ancestors.clear();
        lone.tail_pages = lone.chain_pages.len();
        assert_eq!(store.plan_record(&lone).0, None);
        assert!(!store.covered_endpoint(&lone, |n| n.ready));
    }

    #[test]
    fn recovered_linear_chain_thins_and_reannounces_its_interior() {
        let full = history(192);
        let headers = [
            endpoint_header(&geometry(), &full, 64, None),
            endpoint_header(&geometry(), &full, 128, Some(64)),
            endpoint_header(&geometry(), &full, 192, Some(128)),
        ];
        let records = headers
            .iter()
            .map(|h| scanned(&directory(), h.clone(), 0, true))
            .collect();
        let (mut store, log) = store_with(records, 1 << 30);
        store.set_limit(store.used() + 2 * ROW_RECORD);
        let announcer = Announcer::new(Arc::new(tokio::sync::Notify::new()));
        announcer.begin_session();
        store.announce_records(announcer.clone());
        let initial = announcer.take_batch(32).unwrap();
        assert_eq!(initial.len(), 3);
        let parent = &headers[1];
        store.index.pin(&parent.key).unwrap();
        store.poll(&mut Vec::new()).unwrap();
        assert!(store.index[&parent.key].row);
        assert!(log.borrow().calls.is_empty());
        store.index.unpin(&parent.key).unwrap();
        store.poll(&mut Vec::new()).unwrap();
        assert!(store.index[&headers[0].key].row && store.index[&headers[2].key].row);
        assert!(!store.index[&parent.key].row);
        assert_eq!(store.stats().rows_thinned, [0, 1]);
        assert!(matches!(
            log.borrow().calls.as_slice(),
            [Call::Truncate { length, .. }] if *length == parent.row_segment
        ));
        log.borrow_mut().events.push(TransportEvent::Truncated {
            id: JobId(1),
            result: Ok(()),
        });
        store.poll(&mut Vec::new()).unwrap();
        let notices = announcer.take_batch(32).unwrap();
        assert!(
            matches!(notices.as_slice(), [crate::pool::Notice::Saved(entry)]
            if entry.key == parent.key && !entry.row && entry.tier == PoolTier::Nvme)
        );
        assert!(matches!(
            store.pin_export_files(parent.key, 0, true),
            Err(ExportRefusal::ChainGap)
        ));
        store.poll(&mut Vec::new()).unwrap();
        assert_eq!(store.stats().rows_thinned, [0, 1]);
    }

    #[test]
    fn recovered_row_thinning_waits_for_maintenance_completions() {
        let maintenance_limit = crate::shard::BACKLOG / 2;
        let records = 200;
        let full = history(records * PAGE_TOKENS);
        let headers: Vec<_> = (1..=records)
            .map(|i| {
                endpoint_header(
                    &geometry(),
                    &full,
                    i * PAGE_TOKENS,
                    (i > 1).then(|| (i - 1) * PAGE_TOKENS),
                )
            })
            .collect();
        let recovered = headers
            .iter()
            .map(|header| scanned(&directory(), header.clone(), 0, true))
            .collect();
        let (mut store, log) = store_with(recovered, 1 << 30);
        store.set_limit(store.used() - (records - 2) * ROW_RECORD);
        store.poll(&mut Vec::new()).unwrap();
        assert_eq!(store.maintenance.len(), maintenance_limit);
        assert_eq!(log.borrow().calls.len(), maintenance_limit);
        assert_eq!(store.stats().rows_thinned[1], maintenance_limit as u64);
        store.poll(&mut Vec::new()).unwrap();
        assert_eq!(log.borrow().calls.len(), maintenance_limit);
        assert_eq!(store.stats().rows_thinned[1], maintenance_limit as u64);
        let id = *store.maintenance.keys().next().unwrap();
        log.borrow_mut()
            .events
            .push(TransportEvent::Truncated { id, result: Ok(()) });
        store.poll(&mut Vec::new()).unwrap();
        assert_eq!(store.maintenance.len(), maintenance_limit);
        assert_eq!(log.borrow().calls.len(), maintenance_limit + 1);
        for _ in 0..records {
            log.borrow_mut().events.extend(
                store
                    .maintenance
                    .keys()
                    .map(|&id| TransportEvent::Truncated { id, result: Ok(()) }),
            );
            store.poll(&mut Vec::new()).unwrap();
            assert!(store.maintenance.len() <= maintenance_limit);
            if store.thinning.is_empty() && store.maintenance.is_empty() {
                break;
            }
        }
        assert!(store.thinning.is_empty());
        assert!(store.maintenance.is_empty());
        assert_eq!(log.borrow().calls.len() as u64, records - 2);
        assert_eq!(store.stats().rows_thinned[1], records - 2);
        assert_eq!(
            store.used(),
            headers.iter().map(|header| header.row_segment).sum::<u64>() + 2 * ROW_RECORD
        );
        for (i, header) in headers.iter().enumerate() {
            assert_eq!(
                store.index[&header.key].row,
                i == 0 || i == headers.len() - 1
            );
        }
    }

    #[test]
    fn thinning_submission_failure_rolls_back_bytes_and_preserves_poll_events() {
        let full = history(192);
        let headers = [
            endpoint_header(&geometry(), &full, 64, None),
            endpoint_header(&geometry(), &full, 128, Some(64)),
            endpoint_header(&geometry(), &full, 192, Some(128)),
        ];
        let records = headers
            .iter()
            .map(|h| scanned(&directory(), h.clone(), 0, true))
            .collect();
        let (mut store, log) = store_with(records, 1 << 30);
        store.set_limit(store.used() + 2 * ROW_RECORD);
        let used = store.used();
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        store
            .fetch(
                hit.clone(),
                InboundTicket {
                    key: hit.key,
                    pages: pages(10..13),
                    slot: StateSlotId(1),
                },
            )
            .unwrap();
        log.borrow_mut().fail_truncate = true;
        log.borrow_mut().events.push(TransportEvent::Fetched {
            id: JobId(1),
            result: Ok(hit.bytes),
        });
        let mut events = Vec::new();
        store.poll(&mut events).unwrap();
        assert!(matches!(
            events.as_slice(),
            [StoreEvent::Fetched { result: Ok(_), .. }]
        ));
        assert!(store.index[&headers[1].key].row);
        assert_eq!(
            store
                .lookup(&full, 0, headers[2].key.consumed_tokens)
                .unwrap()
                .key,
            headers[1].key
        );
        assert_eq!(
            store.record(&headers[1].key).unwrap().bytes,
            headers[1].length_with_row()
        );
        assert_eq!(store.used(), used);
        assert!(store.maintenance.is_empty());
        assert_eq!(store.stats().dropped[5], 0);
        assert_eq!(store.stats().rows_thinned, [0, 0]);
    }

    #[test]
    fn disk_pressure_keeps_the_last_row_under_a_covered_tip() {
        let full = history(192);
        let root = endpoint_header(&geometry(), &full, 64, None);
        let parent = endpoint_header(&geometry(), &full, 128, Some(64));
        let tip = endpoint_header(&geometry(), &full, 134, Some(128));
        let records = [(&root, true), (&parent, true), (&tip, false)]
            .into_iter()
            .map(|(h, row)| scanned(&directory(), h.clone(), 0, row))
            .collect();
        let (mut store, _) = store_with(records, 1 << 30);
        store.thin_rows_for(u64::MAX);
        assert!(store.index[&parent.key].row);
        assert_eq!(store.lookup(&full, 0, u64::MAX).unwrap().key, parent.key);
    }

    #[test]
    fn deferred_thinning_does_not_count_an_uncommitted_replacement_child() {
        let full = history(192);
        let root = endpoint_header(&geometry(), &full, 64, None);
        let parent = endpoint_header(&geometry(), &full, 128, Some(64));
        let child = endpoint_header(&geometry(), &full, 192, Some(128));
        let records = [&root, &parent, &child]
            .into_iter()
            .map(|h| scanned(&directory(), h.clone(), 0, true))
            .collect();
        let (mut store, log) = store_with(records, 1 << 30);
        store.set_limit(store.used() + 2 * ROW_RECORD);
        store.index.pin(&parent.key).unwrap();
        store.poll(&mut Vec::new()).unwrap();
        store.expire(child.key).unwrap();
        store
            .reserve_capture(
                child.key,
                Some(parent.key),
                child.bucket,
                1,
                true,
                true,
                crate::CaptureRole::Endpoint,
            )
            .unwrap()
            .unwrap();
        store.index.unpin(&parent.key).unwrap();
        store.poll(&mut Vec::new()).unwrap();
        assert!(store.index[&parent.key].row);
        assert!(
            log.borrow()
                .calls
                .iter()
                .all(|call| !matches!(call, Call::Truncate { .. }))
        );
        store
            .commit_record(child.key, Err(invalid("write failed")), None)
            .unwrap();
        assert!(store.index[&parent.key].row);
        assert_eq!(store.index[&parent.key].children(), 0);
        assert_eq!(store.stats().rows_thinned, [0, 0]);
    }

    #[test]
    fn partial_row_replacement_failure_invalidates_ancestry_until_reclaimed() {
        for has_child in [false, true] {
            let full = history(1000);
            let root = header(&geometry(), &full, 128, None);
            let child = header(&geometry(), &full, 256, Some(128));
            let mut records = vec![scanned(&directory(), root.clone(), 0, false)];
            if has_child {
                records.push(scanned(&directory(), child.clone(), 1, true));
            }
            let (mut store, log) = store_with(records, 1 << 30);
            let original = store.used();
            let mut ids = Ids::new();
            let publication =
                checkpoint(&mut ids, &full, 128, None, pages(0..2), 2, StateSlotId(1));
            assert_eq!(store.store(&publication).unwrap(), StoreOutcome::Queued);
            let events = complete(
                &log,
                TransportEvent::StoreInvalidated {
                    id: JobId(1),
                    error: invalid("rank 1: disk full"),
                },
                &mut store,
            );
            assert!(matches!(
                &events[..],
                [StoreEvent::Stored { result: Err(_), .. }]
            ));
            assert!(store.lookup(&full, 0, u64::MAX).is_none());
            assert_eq!(store.stats().store_failed, 1);
            assert_eq!(store.stats().row_refills, 0);
            assert_eq!(store.stats().outbound_in_flight, 0);
            if has_child {
                let node = store.index.node(&root.key).unwrap();
                assert!(!node.ready && !node.row && !node.pinned());
                assert_eq!(node.children(), 1);
                assert_eq!(store.used(), original + ROW_RECORD);
                assert_eq!(store.index.node(&child.key).unwrap().parent, Some(root.key));
                // A GPU-resident ancestor still supplies a valid prefix for the child's tail.
                assert_eq!(store.lookup(&full, 128, u64::MAX).unwrap().key, child.key);
                assert!(store.make_room(store.limit()).unwrap());
            }
            assert!(store.index.is_empty());
            assert_eq!(store.used(), 0);
            let unlinks: Vec<_> = log
                .borrow()
                .calls
                .iter()
                .filter_map(|call| match call {
                    Call::Unlink { id, path } => Some((*id, path.clone())),
                    _ => None,
                })
                .collect();
            assert_eq!(unlinks.len(), if has_child { 2 } else { 1 });
            for (id, _) in unlinks {
                complete(
                    &log,
                    TransportEvent::Unlinked { id, result: Ok(()) },
                    &mut store,
                );
            }
            assert!(store.maintenance.is_empty());
            assert_eq!(store.used(), 0);
        }
    }

    #[test]
    fn compacted_row_replacement_preserves_links_and_rolls_back_failure() {
        for success in [false, true] {
            let full = history(1000);
            let root = header(&geometry(), &full, 128, None);
            let compacted = header(&geometry(), &full, 512, Some(128));
            let child = header(&geometry(), &full, 768, Some(512));
            let records = vec![
                scanned(&directory(), root.clone(), 0, true),
                scanned(&directory(), compacted.clone(), 1, false),
                scanned(&directory(), child.clone(), 2, true),
            ];
            let (mut store, log) = store_with(records, 1 << 30);
            let original = store.used();
            let mut ids = Ids::new();
            // The GPU has a different parent: the stored parent determines the rewrite tail.
            let publication =
                checkpoint(&mut ids, &full, 512, None, pages(0..8), 8, StateSlotId(3));
            assert_eq!(store.store(&publication).unwrap(), StoreOutcome::Queued);
            let node = store.index.node(&compacted.key).unwrap();
            assert!(!node.ready && !node.row && node.pinned());
            assert_eq!(node.parent, Some(root.key));
            assert_eq!(node.children(), 1);
            assert_eq!(store.used(), original + compacted.length_with_row());
            match &log.borrow().calls[0] {
                Call::Store {
                    header,
                    pages: copied,
                    ..
                } => {
                    assert_eq!(header.parent, Some(root.key));
                    assert_eq!(copied, &pages(2..8));
                }
                other => panic!("unexpected call {other:?}"),
            }
            store.apply(
                TransportEvent::Stored {
                    header: None,
                    id: JobId(1),
                    result: if success {
                        Ok(compacted.length_with_row())
                    } else {
                        Err(invalid("write failed"))
                    },
                },
                &mut Vec::new(),
            );
            let node = store.index.node(&compacted.key).unwrap();
            assert!(node.ready && !node.pinned());
            assert_eq!(node.row, success);
            assert_eq!(node.children(), 1);
            assert_eq!(
                store.index.node(&child.key).unwrap().parent,
                Some(compacted.key)
            );
            assert_eq!(
                store.used(),
                original + if success { ROW_RECORD } else { 0 }
            );
            assert_eq!(store.stats().row_refills, u64::from(success));
            store.poll(&mut Vec::new()).unwrap();
            assert_eq!(store.index[&compacted.key].row, success);
            assert_eq!(store.stats().rows_thinned, [0, 0]);
            assert_eq!(log.borrow().calls.len(), 1);
            assert!(store.lookup(&full, 0, u64::MAX).is_some());
        }
    }

    #[test]
    fn disk_store_keeps_exclusive_directory_ownership_until_drop() {
        let (mut store, _) = store_with(Vec::new(), 1 << 30);
        assert!(crate::testing::directory_lock(store.directory()).is_err());
        store.close().unwrap();
        assert!(crate::testing::directory_lock(store.directory()).is_err());
        let path = store.directory().to_owned();
        drop(store);
        let reopened = crate::testing::directory_lock(&path).unwrap();
        drop(reopened);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn feasible_ancestor_row_compaction_is_not_oversize() {
        let full = history(192);
        let root = endpoint_header(&geometry(), &full, 1, None);
        let interior = endpoint_header(&geometry(), &full, 64, Some(1));
        let parent = endpoint_header(&geometry(), &full, 128, Some(64));
        let leaf = endpoint_header(&geometry(), &full, 192, Some(128));
        let limit = root.length_with_row()
            + interior.row_segment
            + parent.length_with_row()
            + leaf.length_with_row();
        let (mut store, _) = store_with(
            vec![
                scanned(&directory(), root.clone(), 3, true),
                scanned(&directory(), interior.clone(), 0, true),
                scanned(&directory(), parent.clone(), 1, true),
            ],
            limit,
        );
        let record = checkpoint(
            &mut Ids::new(),
            &full,
            192,
            Some(128),
            pages(0..3),
            1,
            StateSlotId(0),
        );
        assert_eq!(store.store(&record).unwrap(), StoreOutcome::Queued);
        assert!(store.index[&root.key].row);
        assert!(!store.index.node(&interior.key).unwrap().row);
        assert!(store.index.node(&parent.key).unwrap().row);

        drop(store);
        let message = interior.with_capture_role(crate::CaptureRole::Message);
        let (mut store, log) = store_with(
            vec![
                scanned(&directory(), root, 3, true),
                scanned(&directory(), message, 0, true),
                scanned(&directory(), parent, 1, true),
            ],
            limit,
        );
        assert_eq!(store.store(&record).unwrap(), StoreOutcome::Oversize);
        assert!(log.borrow().calls.is_empty());
    }

    #[test]
    fn filesystem_full_defers_a_disk_only_copy_after_drain() {
        let full = history(64);
        let record = checkpoint(
            &mut Ids::new(),
            &full,
            64,
            None,
            vec![PageId(0)],
            1,
            StateSlotId(0),
        );
        let (mut store, _) = store_with(Vec::new(), 1 << 30);
        assert_eq!(store.store(&record).unwrap(), StoreOutcome::Queued);
        let id = *store.outbound.keys().next().unwrap();
        let mut events = Vec::new();
        store.apply(
            TransportEvent::Stored {
                header: None,
                id,
                result: Err(Error::caused_by(
                    "disk full",
                    std::io::Error::from(std::io::ErrorKind::StorageFull),
                )),
            },
            &mut events,
        );
        assert!(
            matches!(&events[..], [StoreEvent::Deferred { checkpoint, reason: StoreReason::Budget }] if *checkpoint == record.id)
        );
        assert!(store.outbound.is_empty());
        assert_eq!(store.stats().dropped, [0; 8]);
        assert_eq!(store.stats().held[1], 1);
        assert!(!store.index.contains(&record.key));
    }

    #[test]
    fn a_full_local_tier_keeps_every_record_until_one_is_needed() {
        let full = history(1000);
        let headers: Vec<_> = (1..=3)
            .map(|n| header(&geometry(), &full, n * 64, None))
            .collect();
        let limit: u64 = headers.iter().map(Header::length_with_row).sum();
        let records = headers
            .iter()
            .enumerate()
            .map(|(i, h)| scanned(&directory(), h.clone(), i as u64, true))
            .collect();
        let (mut store, _log) = store_with(records, limit);
        let _announcer = announced(&mut store);
        // Local mode never sets spill: no proactive demotion, however full the tier.
        store.maintain();
        assert!(store.take_retiring().is_empty());
        assert!(store.maintenance.is_empty());
        assert_eq!(store.stats().disk_records, 3);
        // With the pool ready the same tier does demote ahead of need.
        store.set_spill(true);
        store.maintain();
        assert!(!store.take_retiring().is_empty());
    }

    #[test]
    fn pending_unlinks_are_included_in_proactive_headroom() {
        let full = history(1000);
        let headers: Vec<_> = (1..=3)
            .map(|n| header(&geometry(), &full, n * 64, None))
            .collect();
        let limit: u64 = headers.iter().map(Header::length_with_row).sum();
        let records = headers
            .iter()
            .enumerate()
            .map(|(i, h)| scanned(&directory(), h.clone(), i as u64, true))
            .collect();
        let (mut store, log) = store_with(records, limit);
        let budget = super::super::input::DiskBudget::new(limit, limit);
        store.share_budget(budget);
        let _announcer = announced(&mut store);
        store.set_spill(true);
        store.maintain();
        let first = store.take_retiring();
        assert!(!first.is_empty());
        for r in first {
            store.retired(&r.key, Retirement::Moved).unwrap();
        }
        let pending = store.pending_bytes();
        assert!(pending > 0);
        store.maintain();
        let next = store.take_retiring();
        assert!(!next.is_empty(), "pending maintenance advances retirement");
        for r in next {
            store.retired(&r.key, Retirement::Moved).unwrap();
        }
        let id = *store.maintenance.keys().next().unwrap();
        complete(
            &log,
            TransportEvent::Unlinked { id, result: Ok(()) },
            &mut store,
        );
        let candidate = checkpoint(
            &mut Ids::new(),
            &full,
            64,
            None,
            vec![PageId(0)],
            1,
            StateSlotId(0),
        );
        let key = PrefixKey {
            digest: [93; 16],
            ..candidate.key
        };
        assert!(store.pending_bytes() > 0);
        assert!(
            store
                .reserve_capture(
                    key,
                    None,
                    candidate.bucket,
                    1,
                    true,
                    true,
                    crate::CaptureRole::Endpoint
                )
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn shared_disk_budget_keeps_maintenance_bytes_until_success() {
        let history = history(700);
        let header = header(&geometry(), &history, 700, None);
        let bytes = header.length_with_row();
        let (mut store, log) =
            store_with(vec![scanned(&directory(), header.clone(), 0, true)], bytes);
        let budget = super::super::input::DiskBudget::new(bytes, bytes);
        store.share_budget(budget.clone());
        let nodes = store.index.remove(&header.key, |_| false).unwrap();
        for node in nodes {
            store.unlink(node).unwrap();
        }
        assert_eq!(store.used(), 0);
        assert!(!budget.reserve_checkpoint(1));
        let id = *store.maintenance.keys().next().unwrap();
        complete(
            &log,
            TransportEvent::Unlinked {
                id,
                result: Err(invalid("unlink refused")),
            },
            &mut store,
        );
        assert_eq!(store.used(), bytes);
        assert!(!budget.reserve_checkpoint(1));
        drop(store);

        let (mut store, log) =
            store_with(vec![scanned(&directory(), header.clone(), 0, true)], bytes);
        let budget = super::super::input::DiskBudget::new(bytes, bytes);
        store.share_budget(budget.clone());
        let nodes = store.index.remove(&header.key, |_| false).unwrap();
        for node in nodes {
            store.unlink(node).unwrap();
        }
        let id = *store.maintenance.keys().next().unwrap();
        complete(
            &log,
            TransportEvent::Unlinked { id, result: Ok(()) },
            &mut store,
        );
        assert!(budget.reserve_checkpoint(bytes));
    }

    fn complete(
        log: &Rc<RefCell<Log>>,
        event: TransportEvent,
        store: &mut DiskStore,
    ) -> Vec<StoreEvent> {
        log.borrow_mut().events.push(event);
        let mut events = Vec::new();
        store.poll(&mut events).unwrap();
        events
    }

    fn unlinked(log: &Rc<RefCell<Log>>) -> Vec<PathBuf> {
        log.borrow()
            .calls
            .iter()
            .filter_map(|call| match call {
                Call::Unlink { path, .. } => Some(path.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn startup_pressure_keeps_the_last_row_above_a_page_only_leaf() {
        let geometry = geometry();
        let directory = directory();
        let full = history(2000);
        let root = endpoint_header(&geometry, &full, 700, None);
        let child = endpoint_header(&geometry, &full, 1300, Some(700));
        let leaf = endpoint_header(&geometry, &full, 1990, Some(1300));
        let mut branch = history(1300);
        branch.append(&[5; 100], &[]).unwrap();
        let sibling = endpoint_header(&geometry, &branch, 1400, Some(1300));
        assert_eq!(
            (
                root.tail_pages,
                child.tail_pages,
                sibling.tail_pages,
                leaf.tail_pages
            ),
            (10, 10, 1, 11)
        );
        let records = vec![
            scanned(&directory, root.clone(), 30, true),
            scanned(&directory, child.clone(), 10, true),
            scanned(&directory, sibling.clone(), 0, true),
            scanned(&directory, leaf.clone(), 20, false),
        ];
        let total: u64 = records.iter().map(|record| record.bytes).sum();
        let (mut store, log) = store_with(records, total);
        assert_eq!(store.used(), total);
        assert_eq!(store.index().len(), 4);
        assert!(store.index().node(&root.key).unwrap().ready);
        assert!(!store.index().node(&leaf.key).unwrap().row);
        assert_eq!(store.record(&leaf.key).unwrap().bytes, leaf.row_segment);
        assert_eq!(
            store.record(&leaf.key).unwrap().path,
            directory.join(record::file_name(&leaf.key))
        );
        let stats = store.stats();
        assert_eq!((stats.disk_records, stats.disk_bytes), (4, total));
        assert_eq!(stats.inbound_rate, INITIAL_STAGED_DISK_RATE);
        assert_eq!(stats.outbound_rate, INITIAL_STAGED_DISK_RATE);
        assert_eq!(store.limit(), total);

        // After the older sibling leaves, only the child can restore this path.
        let key = full.key_at(300).unwrap();
        let (reserved, temporary, path) = store
            .reserve_capture(
                key,
                None,
                full.bucket_at(300).unwrap(),
                4,
                true,
                true,
                crate::CaptureRole::Endpoint,
            )
            .unwrap()
            .unwrap();
        assert_eq!(reserved.tail_pages, 4);
        assert_eq!(reserved.key, key);
        assert_eq!(path, directory.join(record::file_name(&key)));
        assert_eq!(temporary, directory.join(record::temporary_name(&key)));
        let bytes = reserved.length_with_row();
        assert_eq!(bytes, 2 * 4096 + 4 * PAGE_RECORD + ROW_RECORD);
        let sibling_bytes = sibling.length_with_row();
        assert_eq!(sibling_bytes, 2 * 4096 + PAGE_RECORD + ROW_RECORD);
        assert_eq!(
            unlinked(&log),
            [
                directory.join(record::file_name(&sibling.key)),
                directory.join(record::file_name(&leaf.key))
            ]
        );
        assert_eq!(log.borrow().calls.len(), 2);
        assert_eq!(
            store.used(),
            total - sibling_bytes - leaf.row_segment + bytes
        );
        assert_eq!(store.take_evicted(), [sibling.key, leaf.key]);
        assert!(store.take_evicted().is_empty());
        assert_eq!(store.index().len(), 3);
        let node = store.index().node(&key).unwrap();
        assert!(node.pinned() && !node.ready && node.row);
        assert_eq!(node.location.bytes, bytes);
        assert!(store.index()[&child.key].row);
        assert_eq!(store.index().node(&root.key).unwrap().children(), 1);
    }

    #[test]
    fn a_stored_root_keeps_its_offered_ancestors_for_the_pool_put() {
        let (mut store, _log) = store_with(Vec::new(), 1 << 30);
        let full = history(2000);
        let mut ids = Ids::new();
        // No ancestor is indexed, so the record is a root that still names them all.
        let mut root = checkpoint(
            &mut ids,
            &full,
            1300,
            Some(700),
            pages(0..20),
            10,
            StateSlotId(1),
        );
        root.ancestors.push(full.key_at(300).unwrap());
        assert_eq!(store.store(&root).unwrap(), StoreOutcome::Queued);
        let stats = store.stats();
        assert_eq!(stats.root_records, [[0; crate::RootReason::COUNT]; 2]);
        assert_eq!(stats.root_pages, [[0; crate::RootReason::COUNT]; 2]);
        assert_eq!(stats.deeper_link_records, [0; 2]);
        assert_eq!(stats.deeper_link_pages, [0; 2]);
        assert_eq!(
            store.index().ancestor_keys(&root.key),
            [full.key_at(700).unwrap(), full.key_at(300).unwrap()]
        );
    }

    #[test]
    fn store_dedupes_shapes_records_and_applies_completions() {
        let (mut store, log) = store_with(Vec::new(), 1 << 30);
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        assert_eq!(store.store(&root).unwrap(), StoreOutcome::Queued);
        match &log.borrow().calls[..] {
            [
                Call::Store {
                    id,
                    header,
                    temporary,
                    path,
                    pages,
                    slot,
                },
            ] => {
                assert_eq!(*id, JobId(1));
                assert_eq!(header.parent, None);
                assert_eq!(header.tail_pages, 10);
                assert_eq!(header.key, root.key);
                assert_eq!(header.bucket, root.bucket);
                assert_eq!(pages, &root.chain_pages);
                assert_eq!(*slot, StateSlotId(1));
                assert_eq!(*path, store.directory().join(record::file_name(&root.key)));
                assert_eq!(
                    *temporary,
                    store.directory().join(record::temporary_name(&root.key))
                );
            }
            other => panic!("unexpected calls {other:?}"),
        }
        let node = store.index().node(&root.key).unwrap();
        assert!(node.pinned() && !node.ready && node.row);
        let root_bytes = node.location.bytes;
        assert_eq!(root_bytes, 2 * 4096 + 10 * PAGE_RECORD + ROW_RECORD);
        assert_eq!(store.used(), root_bytes);

        // The same key again is skipped without a second call, and a record
        // being written never matches a lookup.
        assert_ne!(store.store(&root).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.stats().store_skipped, 1);
        assert_eq!(store.stats().outbound_in_flight, 1);
        assert_eq!(log.borrow().calls.len(), 1);
        assert!(store.lookup(&full, 0, u64::MAX).is_none());

        // A child of a record being written is shaped as its child.
        let child = checkpoint(
            &mut ids,
            &full,
            1300,
            Some(700),
            pages(0..20),
            10,
            StateSlotId(2),
        );
        assert_eq!(store.store(&child).unwrap(), StoreOutcome::Queued);
        match &log.borrow().calls[1] {
            Call::Store {
                id, header, pages, ..
            } => {
                assert_eq!(*id, JobId(2));
                assert_eq!(header.parent, Some(root.key));
                assert_eq!(header.tail_pages, 10);
                assert_eq!(pages, &child.chain_pages[10..]);
            }
            other => panic!("unexpected call {other:?}"),
        }
        assert_eq!(store.index().node(&root.key).unwrap().children(), 1);

        // A checkpoint whose parent the store never saw becomes a root record
        // holding its whole chain.
        let orphan = checkpoint(
            &mut ids,
            &full,
            1990,
            Some(1500),
            pages(0..31),
            8,
            StateSlotId(3),
        );
        assert_eq!(
            store.plan_record(&orphan),
            (None, orphan.chain_pages.clone())
        );
        assert_eq!(store.store(&orphan).unwrap(), StoreOutcome::Queued);
        match &log.borrow().calls[2] {
            Call::Store {
                id, header, pages, ..
            } => {
                assert_eq!(*id, JobId(3));
                assert_eq!(header.parent, None);
                assert_eq!(header.tail_pages, 31);
                assert_eq!(pages, &orphan.chain_pages);
            }
            other => panic!("unexpected call {other:?}"),
        }
        assert_eq!(store.stats().outbound_in_flight, 3);
        let orphan_bytes = store.record(&orphan.key).unwrap().bytes;

        // The root's completion makes it ready, unpinned, and matchable.
        let events = complete(
            &log,
            TransportEvent::Stored {
                header: None,
                id: JobId(1),
                result: Ok(root_bytes),
            },
            &mut store,
        );
        match &events[..] {
            [
                StoreEvent::Stored {
                    checkpoint,
                    key,
                    result,
                },
            ] => {
                assert_eq!(*checkpoint, root.id);
                assert_eq!(*key, root.key);
                assert_eq!(*result.as_ref().unwrap(), root_bytes);
            }
            other => panic!("unexpected events {other:?}"),
        }
        let node = store.index().node(&root.key).unwrap();
        assert!(!node.pinned() && node.ready);
        let stats = store.stats();
        assert_eq!(
            (stats.stored, stats.bytes_out, stats.outbound_in_flight),
            (1, root_bytes, 2)
        );
        assert!(stats.outbound_rate > 0.0 && stats.outbound_rate.is_finite());
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(hit.key, root.key);
        assert_eq!(
            (hit.consumed_tokens, hit.resident_tokens, hit.pages),
            (700, 0, 10)
        );
        assert_eq!(hit.bytes, 10 * PAGE_RECORD + ROW_RECORD);
        assert_eq!(hit.tier, Tier::Nvme);
        assert_eq!(
            hit.estimated_seconds,
            hit.bytes as f64 / INITIAL_STAGED_DISK_RATE
        );
        assert_eq!(hit.chain.nodes, [root.key]);
        assert_eq!((store.stats().lookups, store.stats().hits), (2, 1));

        // The deeper root record wins once it completes.
        complete(
            &log,
            TransportEvent::Stored {
                header: None,
                id: JobId(3),
                result: Ok(orphan_bytes),
            },
            &mut store,
        );
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(hit.consumed_tokens, 1990);
        assert_eq!(hit.chain.nodes, [orphan.key]);
        assert_eq!(hit.pages, 31);

        // A failed write leaves no record and returns its bytes.
        let child_bytes = store.record(&child.key).unwrap().bytes;
        let used = store.used();
        let events = complete(
            &log,
            TransportEvent::Stored {
                header: None,
                id: JobId(2),
                result: Err("disk full".into()),
            },
            &mut store,
        );
        match &events[..] {
            [
                StoreEvent::Stored {
                    checkpoint,
                    key,
                    result,
                },
            ] => {
                assert_eq!(*checkpoint, child.id);
                assert_eq!(*key, child.key);
                assert_eq!(result.as_ref().unwrap_err().to_string(), "disk full");
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(store.record(&child.key).is_none());
        assert_eq!(store.used(), used - child_bytes);
        assert_eq!(store.index().node(&root.key).unwrap().children(), 0);
        let stats = store.stats();
        assert_eq!(
            (
                stats.store_failed,
                stats.outbound_in_flight,
                stats.disk_records
            ),
            (1, 0, 2)
        );
        assert_eq!(stats.disk_bytes, root_bytes + orphan_bytes);
        assert!(store.take_evicted().is_empty());
        assert_eq!(log.borrow().calls.len(), 3);

        // Completions for unknown jobs are ignored.
        let events = complete(
            &log,
            TransportEvent::Stored {
                header: None,
                id: JobId(9),
                result: Ok(1),
            },
            &mut store,
        );
        assert!(events.is_empty());

        // The store refuses a checkpoint whose chain cannot hold its tail.
        let mut short = checkpoint(&mut ids, &full, 1000, None, pages(0..3), 15, StateSlotId(1));
        short.tail_pages = 15;
        assert!(store.store(&short).is_err());
        assert!(store.record(&short.key).is_none());

        store.close().unwrap();
        assert!(log.borrow().closed);
    }

    #[test]
    fn a_failed_parent_write_leaves_a_remnant_until_its_children_go() {
        let root_bytes = 2 * 4096 + 10 * PAGE_RECORD + ROW_RECORD;
        let (mut store, log) = store_with(Vec::new(), 2 * root_bytes);
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        let child = checkpoint(
            &mut ids,
            &full,
            1300,
            Some(700),
            pages(0..20),
            10,
            StateSlotId(2),
        );
        assert_eq!(store.store(&root).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.store(&child).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.used(), 2 * root_bytes);

        let events = complete(
            &log,
            TransportEvent::Stored {
                header: None,
                id: JobId(1),
                result: Err("torn".into()),
            },
            &mut store,
        );
        assert_eq!(events.len(), 1);
        let remnant = store.index().node(&root.key).unwrap();
        assert!(!remnant.ready && !remnant.row && !remnant.pinned());
        assert_eq!((remnant.children(), remnant.location.bytes), (1, 0));
        assert_eq!(store.used(), root_bytes);
        assert_eq!(store.stats().store_failed, 1);

        // The remnant is no parent for later records.
        store.index.pin(&root.key).unwrap();
        let grandchild = checkpoint(
            &mut ids,
            &full,
            1990,
            Some(700),
            pages(0..31),
            21,
            StateSlotId(3),
        );
        assert_eq!(
            store.plan_record(&grandchild),
            (None, grandchild.chain_pages.clone())
        );
        assert!(
            store
                .reserve_capture(
                    grandchild.key,
                    Some(root.key),
                    grandchild.bucket,
                    21,
                    true,
                    true,
                    crate::CaptureRole::Endpoint,
                )
                .is_err()
        );
        assert!(store.record(&grandchild.key).is_none());

        store.index.unpin(&root.key).unwrap();

        // The child becomes ready, but no chain passes through the remnant.
        complete(
            &log,
            TransportEvent::Stored {
                header: None,
                id: JobId(2),
                result: Ok(root_bytes),
            },
            &mut store,
        );
        assert!(store.index().node(&child.key).unwrap().ready);
        assert!(store.lookup(&full, 0, u64::MAX).is_none());

        // Evicting the child takes the remnant with it; only the child's file is unlinked.
        let key = full.key_at(704).unwrap();
        let (reserved, _, _) = store
            .reserve_capture(
                key,
                None,
                full.bucket_at(704).unwrap(),
                11,
                true,
                true,
                crate::CaptureRole::Endpoint,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            unlinked(&log),
            [store.directory().join(record::file_name(&child.key))]
        );
        assert_eq!(log.borrow().calls.len(), 3);
        assert_eq!(store.take_evicted(), [child.key]);
        assert!(store.index().node(&root.key).is_none());
        assert!(store.index().node(&child.key).is_none());
        assert_eq!(store.used(), reserved.length_with_row());
        assert_eq!(store.index().len(), 1);
    }

    #[test]
    fn a_transient_restore_failure_keeps_the_record_offered() {
        let geometry = geometry();
        let directory = directory();
        let full = history(2000);
        let root = header(&geometry, &full, 700, None);
        let leaf = header(&geometry, &full, 1990, Some(700));
        let records = vec![
            scanned(&directory, root, 0, true),
            scanned(&directory, leaf.clone(), 1, true),
        ];
        let total: u64 = records.iter().map(|record| record.bytes).sum();
        let (mut store, log) = store_with(records, 2 * total);
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(hit.key, leaf.key);
        let ticket = InboundTicket {
            key: hit.key,
            pages: (100..100 + hit.pages as u32).map(PageId).collect(),
            slot: StateSlotId(2),
        };
        store.fetch(hit, ticket).unwrap();
        let events = complete(
            &log,
            TransportEvent::Fetched {
                id: JobId(1),
                result: Err(invalid("copy cancelled")),
            },
            &mut store,
        );
        assert!(matches!(
            events.as_slice(),
            [StoreEvent::Fetched { key, result: Err(_) }] if *key == leaf.key
        ));
        assert!(store.index().nodes().all(|node| !node.pinned()));
        assert_eq!(store.stats().fetch_failed, 1);
        assert_eq!(store.lookup(&full, 700, u64::MAX).unwrap().key, leaf.key);
    }

    #[test]
    fn a_record_read_failure_is_not_offered_again() {
        let geometry = geometry();
        let directory = directory();
        let full = history(2000);
        let root = header(&geometry, &full, 700, None);
        let leaf = header(&geometry, &full, 1990, Some(700));
        let records = vec![
            scanned(&directory, root.clone(), 0, true),
            scanned(&directory, leaf.clone(), 1, true),
        ];
        let total: u64 = records.iter().map(|record| record.bytes).sum();
        let (mut store, log) = store_with(records, 2 * total);

        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(hit.key, leaf.key);
        let ticket = InboundTicket {
            key: hit.key,
            pages: (100..100 + hit.pages as u32).map(PageId).collect(),
            slot: StateSlotId(2),
        };
        store.fetch(hit, ticket).unwrap();
        log.borrow_mut().events.push(TransportEvent::Fetched {
            id: JobId(1),
            result: Err(invalid("page checksum mismatch").with_record_read()),
        });
        let mut events = Vec::new();
        store.poll(&mut events).unwrap();
        assert!(matches!(
            events.as_slice(),
            [StoreEvent::Fetched { result: Err(_), .. }]
        ));

        // A record whose bytes could not be restored is not offered again, so
        // every later admission does not repeat the same failing transfer. Its
        // ancestor stays available, because only the failed endpoint's row was
        // dropped.
        assert_eq!(store.lookup(&full, 0, u64::MAX).unwrap().key, root.key);
        assert!(store.lookup(&full, 700, u64::MAX).is_none());
    }

    #[test]
    fn fetch_plans_segments_after_skipped_pages_and_pins_the_chain() {
        let geometry = geometry();
        let directory = directory();
        let full = history(2000);
        let root = header(&geometry, &full, 700, None);
        let child = endpoint_header(&geometry, &full, 1300, Some(700));
        let leaf = header(&geometry, &full, 1990, Some(1300));
        let records = vec![
            scanned(&directory, root.clone(), 0, true),
            scanned(&directory, child.clone(), 1, true),
            scanned(&directory, leaf.clone(), 2, true),
        ];
        let total: u64 = records.iter().map(|record| record.bytes).sum();
        let (mut store, log) = store_with(records, 2 * total);

        let hit = store.lookup(&full, 1000, u64::MAX).unwrap();
        assert_eq!(hit.chain.nodes, [child.key, leaf.key]);
        assert_eq!(
            (hit.chain.skip_pages, hit.pages, hit.resident_tokens),
            (5, 16, 1000)
        );
        let bytes = 16 * PAGE_RECORD + ROW_RECORD;
        assert_eq!(hit.bytes, bytes);
        let ticket = |pages: u32| InboundTicket {
            key: hit.key,
            pages: (100..100 + pages).map(PageId).collect(),
            slot: StateSlotId(2),
        };

        // A ticket of the wrong size or key is refused before anything is pinned.
        assert!(store.fetch(hit.clone(), ticket(15)).is_err());
        let mut wrong = ticket(16);
        wrong.key = root.key;
        assert!(store.fetch(hit.clone(), wrong).is_err());
        assert!(store.index().nodes().all(|node| !node.pinned()));
        assert!(log.borrow().calls.is_empty());

        store.fetch(hit.clone(), ticket(16)).unwrap();
        match &log.borrow().calls[..] {
            [Call::Fetch { id, plan, ticket }] => {
                assert_eq!(*id, JobId(1));
                assert_eq!(ticket.pages.len(), 16);
                assert_eq!(
                    *plan,
                    [
                        FetchSegment {
                            path: directory.join(record::file_name(&child.key)),
                            header: child.clone(),
                            first_page: 5,
                            pages: 5,
                            row: false,
                        },
                        FetchSegment {
                            path: directory.join(record::file_name(&leaf.key)),
                            header: leaf.clone(),
                            first_page: 0,
                            pages: 11,
                            row: true,
                        }
                    ]
                );
            }
            other => panic!("unexpected calls {other:?}"),
        }
        let pins = |store: &DiskStore| {
            [&root.key, &child.key, &leaf.key].map(|key| store.index().node(key).unwrap().pins())
        };
        assert_eq!(pins(&store), [0, 1, 1]);
        assert_eq!(store.stats().inbound_in_flight, 1);

        // A second restore of the same chain is an independent job.
        let again = store.lookup(&full, 1000, u64::MAX).unwrap();
        store.fetch(again, ticket(16)).unwrap();
        assert_eq!(pins(&store), [0, 2, 2]);

        // Both reads pin their segments. Pressure must also preserve the unpinned root row.
        let key = full.key_at(300).unwrap();
        assert!(
            store
                .reserve_capture(
                    key,
                    None,
                    full.bucket_at(300).unwrap(),
                    4,
                    true,
                    true,
                    crate::CaptureRole::Endpoint
                )
                .unwrap()
                .is_some()
        );
        assert!(store.index().node(&root.key).unwrap().row);
        store.limit = store.used();
        assert!(
            store
                .reserve_capture(
                    full.key_at(400).unwrap(),
                    None,
                    full.bucket_at(400).unwrap(),
                    6,
                    true,
                    true,
                    crate::CaptureRole::Endpoint,
                )
                .unwrap()
                .is_none()
        );
        assert!(log.borrow().calls[2..].is_empty());
        assert!(store.index()[&root.key].row);
        assert_eq!(
            store.record(&root.key).unwrap().bytes,
            root.length_with_row()
        );
        assert_eq!(pins(&store), [0, 2, 2]);

        let events = complete(
            &log,
            TransportEvent::Fetched {
                id: JobId(1),
                result: Ok(bytes),
            },
            &mut store,
        );
        match &events[..] {
            [StoreEvent::Fetched { key, result }] => {
                assert_eq!(*key, leaf.key);
                assert_eq!(*result.as_ref().unwrap(), bytes);
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert_eq!(pins(&store), [0, 1, 1]);
        let stats = store.stats();
        assert_eq!(
            (stats.fetched, stats.bytes_in, stats.reused_tokens),
            (1, bytes, 990)
        );
        assert_eq!(stats.inbound_in_flight, 1);
        assert!(stats.inbound_rate > 0.0 && stats.inbound_rate.is_finite());

        let events = complete(
            &log,
            TransportEvent::Fetched {
                id: JobId(2),
                result: Err("checksum".into()),
            },
            &mut store,
        );
        match &events[..] {
            [StoreEvent::Fetched { key, result }] => {
                assert_eq!(*key, leaf.key);
                assert!(result.is_err());
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert_eq!(pins(&store), [0, 0, 0]);
        let stats = store.stats();
        assert_eq!(
            (stats.fetched, stats.fetch_failed, stats.inbound_in_flight),
            (1, 1, 0)
        );
        assert_eq!(stats.reused_tokens, 990);

        // Releasing both readers lets the interior row compact. A failed truncate
        // keeps its physical bytes charged, and the root still supplies its row.
        let used = store.used();
        assert!(
            matches!(log.borrow().calls.last(), Some(Call::Truncate { id: JobId(3), path, .. })
            if *path == directory.join(record::file_name(&child.key)))
        );
        assert!(!store.index()[&child.key].row);
        let events = complete(
            &log,
            TransportEvent::Truncated {
                id: JobId(3),
                result: Err("busy".into()),
            },
            &mut store,
        );
        assert!(events.is_empty());
        assert_eq!(
            store.record(&child.key).unwrap().bytes,
            child.length_with_row()
        );
        assert!(store.index()[&root.key].row);
        assert_eq!(store.used(), used + ROW_RECORD);
        assert!(store.maintenance.is_empty());
    }

    #[test]
    fn reserving_a_child_never_evicts_its_parent_and_an_oversized_record_evicts_nothing() {
        let root_bytes = 2 * 4096 + 10 * PAGE_RECORD + ROW_RECORD;
        let (mut store, log) = store_with(Vec::new(), root_bytes + 4096);
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        assert_eq!(store.store(&root).unwrap(), StoreOutcome::Queued);
        complete(
            &log,
            TransportEvent::Stored {
                header: None,
                id: JobId(1),
                result: Ok(root_bytes),
            },
            &mut store,
        );

        // With only the parent to evict, the child is skipped and the parent stays.
        let child = checkpoint(
            &mut ids,
            &full,
            1300,
            Some(700),
            pages(0..20),
            10,
            StateSlotId(2),
        );
        assert_ne!(store.store(&child).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.stats().store_skipped, 1);
        assert!(unlinked(&log).is_empty());
        assert_eq!(log.borrow().calls.len(), 1);
        let node = store.index().node(&root.key).unwrap();
        assert!(node.ready && !node.pinned());
        assert_eq!(node.children(), 0);
        assert_eq!((store.used(), store.index().len()), (root_bytes, 1));
        assert!(store.take_evicted().is_empty());

        // Beside its parent the child fits, and the parent ends unpinned.
        store.limit = 2 * root_bytes;
        assert_eq!(store.store(&child).unwrap(), StoreOutcome::Queued);
        let node = store.index().node(&root.key).unwrap();
        assert_eq!((node.children(), node.pins()), (1, 0));
        complete(
            &log,
            TransportEvent::Stored {
                header: None,
                id: JobId(2),
                result: Ok(root_bytes),
            },
            &mut store,
        );
        let mut branch = history(700);
        branch.append(&[5; 700], &[]).unwrap();
        let sibling = checkpoint(
            &mut ids,
            &branch,
            1400,
            Some(700),
            pages(0..21),
            11,
            StateSlotId(3),
        );
        let sibling_bytes = 2 * 4096 + 11 * PAGE_RECORD + ROW_RECORD;
        store.limit = 2 * root_bytes + sibling_bytes;
        assert_eq!(store.store(&sibling).unwrap(), StoreOutcome::Queued);
        complete(
            &log,
            TransportEvent::Stored {
                header: None,
                id: JobId(3),
                result: Ok(sibling_bytes),
            },
            &mut store,
        );
        assert_eq!(store.index().node(&root.key).unwrap().children(), 2);

        // A grandchild's parent is the least recently used leaf, so the
        // younger sibling is the one evicted.
        let grandchild = checkpoint(
            &mut ids,
            &full,
            1990,
            Some(1300),
            pages(0..31),
            11,
            StateSlotId(4),
        );
        assert_eq!(store.store(&grandchild).unwrap(), StoreOutcome::Queued);
        assert_eq!(
            unlinked(&log),
            [store.directory().join(record::file_name(&sibling.key))]
        );
        assert_eq!(store.take_evicted(), [sibling.key]);
        let parent = store.index().node(&child.key).unwrap();
        assert!(parent.ready && !parent.pinned());
        assert_eq!(parent.children(), 1);
        assert_eq!(store.index().node(&root.key).unwrap().children(), 1);
        assert_eq!(
            store.index().node(&grandchild.key).unwrap().parent,
            Some(child.key)
        );
        assert_eq!(store.used(), 2 * root_bytes + sibling_bytes);
        // The sibling's unlink took job 4; the grandchild's write is job 5.
        complete(
            &log,
            TransportEvent::Stored {
                header: None,
                id: JobId(5),
                result: Ok(sibling_bytes),
            },
            &mut store,
        );
        // Committing the grandchild submits a sixth call to truncate its
        // non-root parent's row; the root and the new leaf retain theirs.
        assert!(matches!(
            &log.borrow().calls[5],
            Call::Truncate { id: JobId(6), path, length }
                if *path == store.directory().join(record::file_name(&child.key))
                    && *length == root_bytes - ROW_RECORD
        ));
        assert!(store.index().node(&root.key).unwrap().row);
        assert!(!store.index().node(&child.key).unwrap().row);
        assert!(store.index().node(&grandchild.key).unwrap().row);
        assert_eq!(store.used(), 2 * root_bytes + sibling_bytes - ROW_RECORD);
        assert_eq!(store.pending_bytes(), sibling_bytes + ROW_RECORD);

        // A record larger than the whole limit is skipped before anything is evicted.
        store.limit = store.used();
        let long = history(4000);
        let huge = checkpoint(
            &mut ids,
            &long,
            3010,
            None,
            pages(0..47),
            47,
            StateSlotId(5),
        );
        assert!(2 * 4096 + 47 * PAGE_RECORD + ROW_RECORD > store.limit());
        assert_ne!(store.store(&huge).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.stats().store_skipped, 2);
        assert_eq!(log.borrow().calls.len(), 6);
        assert_eq!(store.index().len(), 3);
        assert_eq!(store.used(), store.limit());

        // Reservation refuses an indexed key and an unknown parent before
        // evicting anything.
        assert!(
            store
                .reserve_capture(
                    root.key,
                    None,
                    root.bucket,
                    10,
                    true,
                    true,
                    crate::CaptureRole::Endpoint
                )
                .is_err()
        );
        let unknown = full.key_at(1000).unwrap();
        assert!(
            store
                .reserve_capture(
                    full.key_at(1100).unwrap(),
                    Some(unknown),
                    full.bucket_at(1100).unwrap(),
                    1,
                    true,
                    true,
                    crate::CaptureRole::Endpoint,
                )
                .is_err()
        );
        assert_eq!(log.borrow().calls.len(), 6);
        assert_eq!(store.index().len(), 3);
        assert!(store.index().nodes().all(|node| !node.pinned()));
        complete(
            &log,
            TransportEvent::Truncated {
                id: JobId(6),
                result: Ok(()),
            },
            &mut store,
        );
        assert_eq!(store.used(), store.limit());
        // Only the earlier sibling unlink remains charged as pending reclaim.
        assert_eq!(store.pending_bytes(), sibling_bytes);
    }

    #[test]
    fn reserving_under_a_page_only_parent_keeps_it_through_the_cascade() {
        let geometry = geometry();
        let directory = directory();
        let full = history(2000);
        let root = header(&geometry, &full, 700, None);
        let leaf = header(&geometry, &full, 1300, Some(700));
        let records = vec![
            scanned(&directory, root.clone(), 0, false),
            scanned(&directory, leaf.clone(), 1, true),
        ];
        let total: u64 = records.iter().map(|record| record.bytes).sum();
        let (mut store, log) = store_with(records, total);

        // A second child of the page-only root needs the leaf's bytes. Removing
        // the leaf would cascade into the childless root, which is the parent
        // being named, so the root survives and the child is indexed under it.
        let mut branch = history(700);
        branch.append(&[5; 700], &[]).unwrap();
        let key = branch.key_at(1340).unwrap();
        let (reserved, _, _) = store
            .reserve_capture(
                key,
                Some(root.key),
                branch.bucket_at(1340).unwrap(),
                10,
                true,
                true,
                crate::CaptureRole::Endpoint,
            )
            .unwrap()
            .unwrap();
        assert_eq!(reserved.length_with_row(), leaf.length_with_row());
        assert_eq!(
            unlinked(&log),
            [directory.join(record::file_name(&leaf.key))]
        );
        assert_eq!(store.take_evicted(), [leaf.key]);
        let parent = store.index().node(&root.key).unwrap();
        assert!(parent.ready && !parent.row && !parent.pinned());
        assert_eq!(parent.children(), 1);
        assert_eq!(store.index().node(&key).unwrap().parent, Some(root.key));
        assert_eq!(store.used(), root.row_segment + reserved.length_with_row());
        assert_eq!(store.index().len(), 2);

        // Once the child is ready, the chain through the page-only root matches.
        store
            .commit_record(key, Ok(reserved.length_with_row()), None)
            .unwrap();
        let hit = store.lookup(&branch, 0, u64::MAX).unwrap();
        assert_eq!(hit.chain.nodes, [root.key, key]);
        assert_eq!(hit.pages, 20);
    }

    #[test]
    fn failed_unlinks_and_truncates_keep_their_bytes_against_the_limit() {
        let geometry = geometry();
        let directory = directory();
        let full = history(2000);
        let root = endpoint_header(&geometry, &full, 64, None);
        let interior = endpoint_header(&geometry, &full, 700, Some(64));
        let child = endpoint_header(&geometry, &full, 1300, Some(700));
        let leaf = endpoint_header(&geometry, &full, 1990, Some(1300));
        let records = vec![
            scanned(&directory, root.clone(), 40, true),
            scanned(&directory, interior.clone(), 15, true),
            scanned(&directory, child.clone(), 10, true),
            scanned(&directory, leaf.clone(), 20, true),
        ];
        let total: u64 = records.iter().map(|record| record.bytes).sum();
        let (mut store, log) = store_with(records, total);
        store.index_mut().pin(&leaf.key).unwrap();
        let key = full.key_at(300).unwrap();
        let bucket = full.bucket_at(300).unwrap();
        assert!(
            store
                .reserve_capture(
                    key,
                    None,
                    bucket,
                    4,
                    true,
                    true,
                    crate::CaptureRole::Endpoint
                )
                .unwrap()
                .is_none()
        );
        match &log.borrow().calls[..] {
            [
                Call::Truncate { id: first, .. },
                Call::Truncate { id: second, .. },
            ] => assert_eq!((*first, *second), (JobId(1), JobId(2))),
            other => panic!("unexpected calls {other:?}"),
        }
        assert_eq!(store.used(), total - 2 * ROW_RECORD);

        // Redundant child rows are reclaimed before eviction compacts the older interior row.
        // The older interior row's truncate fails: its file keeps the row bytes, so the
        // record regains its length and the budget its bytes; the child's
        // succeeds and changes nothing further.
        let events = complete(
            &log,
            TransportEvent::Truncated {
                id: JobId(2),
                result: Err("busy".into()),
            },
            &mut store,
        );
        assert!(events.is_empty());
        assert_eq!(
            store.record(&interior.key).unwrap().bytes,
            interior.length_with_row()
        );
        assert!(!store.index().node(&interior.key).unwrap().row);
        assert_eq!(store.used(), total - ROW_RECORD);
        complete(
            &log,
            TransportEvent::Truncated {
                id: JobId(1),
                result: Ok(()),
            },
            &mut store,
        );
        assert_eq!(store.record(&child.key).unwrap().bytes, child.row_segment);
        assert_eq!(store.stats().disk_bytes, total - ROW_RECORD);

        // Releasing the leaf lets the whole chain go. The child's unlink fails,
        // so its bytes stay charged although the record left the index.
        store.index_mut().unpin(&leaf.key).unwrap();
        let (reserved, _, _) = store
            .reserve_capture(
                key,
                None,
                bucket,
                4,
                true,
                true,
                crate::CaptureRole::Endpoint,
            )
            .unwrap()
            .unwrap();
        assert_eq!(store.take_evicted(), [leaf.key, child.key, interior.key]);
        assert_eq!(
            store.used(),
            root.length_with_row() + reserved.length_with_row()
        );
        match &log.borrow().calls[2..] {
            [
                Call::Unlink { id: first, .. },
                Call::Unlink { id: second, .. },
                Call::Unlink { id: third, .. },
            ] => assert_eq!((*first, *second, *third), (JobId(3), JobId(4), JobId(5))),
            other => panic!("unexpected calls {other:?}"),
        }
        for (id, result) in [
            (JobId(3), Ok(())),
            (JobId(4), Err("permission denied".into())),
            (JobId(5), Ok(())),
            (JobId(9), Err("unknown".into())),
        ] {
            let events = complete(&log, TransportEvent::Unlinked { id, result }, &mut store);
            assert!(events.is_empty());
        }
        assert_eq!(
            store.used(),
            root.length_with_row() + reserved.length_with_row() + child.row_segment
        );
        let stats = store.stats();
        assert_eq!(
            (stats.disk_records, stats.disk_bytes),
            (
                2,
                root.length_with_row() + reserved.length_with_row() + child.row_segment
            )
        );
        assert!(store.maintenance.is_empty());
    }

    #[test]
    fn pinned_leaves_compact_ancestors_and_release_cascades_to_page_only_parents() {
        let geometry = geometry();
        let directory = directory();
        let full = history(2000);
        let root = endpoint_header(&geometry, &full, 64, None);
        let interior = endpoint_header(&geometry, &full, 700, Some(64));
        let child = endpoint_header(&geometry, &full, 1300, Some(700));
        let leaf = endpoint_header(&geometry, &full, 1990, Some(1300));
        let records = vec![
            scanned(&directory, root.clone(), 40, true),
            scanned(&directory, interior.clone(), 15, true),
            scanned(&directory, child.clone(), 10, true),
            scanned(&directory, leaf.clone(), 20, true),
        ];
        let total: u64 = records.iter().map(|record| record.bytes).sum();
        let (mut store, log) = store_with(records, total);
        store.index_mut().pin(&leaf.key).unwrap();

        // With the leaf pinned, reclaim the redundant child row before
        // eviction compacts the older interior row. The record still does not fit.
        let key = full.key_at(300).unwrap();
        let bucket = full.bucket_at(300).unwrap();
        assert!(
            store
                .reserve_capture(
                    key,
                    None,
                    bucket,
                    4,
                    true,
                    true,
                    crate::CaptureRole::Endpoint
                )
                .unwrap()
                .is_none()
        );
        match &log.borrow().calls[..] {
            [
                Call::Truncate {
                    path: first,
                    length: first_length,
                    ..
                },
                Call::Truncate {
                    path: second,
                    length: second_length,
                    ..
                },
            ] => {
                assert_eq!(*first, directory.join(record::file_name(&child.key)));
                assert_eq!(*first_length, child.row_segment);
                assert_eq!(*second, directory.join(record::file_name(&interior.key)));
                assert_eq!(*second_length, interior.row_segment);
            }
            other => panic!("unexpected calls {other:?}"),
        }
        assert_eq!(store.used(), total - 2 * ROW_RECORD);
        assert!(!store.index().node(&interior.key).unwrap().row);
        assert!(!store.index().node(&child.key).unwrap().row);
        assert_eq!(store.record(&child.key).unwrap().bytes, child.row_segment);
        assert!(store.take_evicted().is_empty());
        assert!(store.index().node(&key).is_none());

        // Page-only ancestors still supply pages to the leaf's chain but no
        // longer match on their own.
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(
            hit.chain.nodes,
            [root.key, interior.key, child.key, leaf.key]
        );
        assert_eq!(hit.pages, 31);
        assert_eq!(
            store.lookup(&history(1300), 0, u64::MAX).unwrap().key,
            root.key
        );

        // Releasing the pin lets the leaf go, and its page-only ancestors follow.
        store.index_mut().unpin(&leaf.key).unwrap();
        let (reserved, _, _) = store
            .reserve_capture(
                key,
                None,
                bucket,
                4,
                true,
                true,
                crate::CaptureRole::Endpoint,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            unlinked(&log),
            [&leaf, &child, &interior].map(|header| directory.join(record::file_name(&header.key)))
        );
        assert_eq!(store.take_evicted(), [leaf.key, child.key, interior.key]);
        assert_eq!(
            store.used(),
            root.length_with_row() + reserved.length_with_row()
        );
        assert_eq!(store.index().len(), 2);
        assert!(store.index()[&root.key].row);
        assert!(store.index().node(&key).unwrap().pinned());

        // Committing the reserved record directly, as the composite does.
        store
            .commit_record(key, Ok(reserved.length_with_row()), None)
            .unwrap();
        let node = store.index().node(&key).unwrap();
        assert!(node.ready && !node.pinned());
        assert_eq!(store.stats().stored, 1);
        assert!(store.commit_record(interior.key, Ok(0), None).is_err());
    }

    #[test]
    fn windows_split_segments_at_the_window_size_and_carry_the_row_last() {
        let plans = plan_windows([(5u32, 3usize, false), (0, 0, false), (2, 130, true)], false);
        let summary: Vec<_> = plans
            .iter()
            .map(|plan| {
                (
                    plan.segment,
                    plan.index,
                    plan.first,
                    plan.count,
                    plan.offset,
                    plan.row,
                )
            })
            .collect();
        assert_eq!(
            summary,
            [
                (0, 0, 5, 3, 0, false),
                (2, 1, 2, 64, 3, false),
                (2, 2, 66, 64, 67, false),
                (2, 3, 130, 2, 131, true),
            ]
        );
        let row_only = plan_windows([(0, 0, true)], false);
        assert_eq!(row_only.len(), 1);
        assert_eq!((row_only[0].count, row_only[0].row), (0, true));
        let exact = plan_windows([(0, 64, true)], false);
        assert_eq!(exact.len(), 1);
        assert_eq!((exact[0].count, exact[0].row), (64, true));
        assert!(plan_windows([], false).is_empty());
        assert!(plan_windows([(0, 0, false)], false).is_empty());
    }

    fn stopped_transfers() -> SharedCopier {
        StoppedCopier::shared(&geometry()).unwrap()
    }

    /// Caller-owned aligned buffers: the slots of an unregistered pool.
    fn host_pool() -> SlabPool {
        let layout = Layout::new(&geometry()).unwrap();
        SlabPool::new(
            layout.page_record_bytes,
            crate::offload::intake_slots(&layout, 1),
            crate::offload::intake_slots(&layout, 1),
        )
        .unwrap()
    }

    fn page_slabs(pool: &mut SlabPool, count: usize, seed: u64) -> Vec<Slot> {
        (0..count as u64)
            .map(|index| {
                let slot = pool.allocate_page().unwrap();
                pattern(pool.page_mut(slot), seed + index);
                slot
            })
            .collect()
    }

    fn row_slab(pool: &mut SlabPool, seed: u64) -> Slot {
        let slot = pool.allocate_page().unwrap();
        pattern(pool.page_mut(slot), seed);
        slot
    }

    fn addresses(pool: &SlabPool, slots: &[Slot]) -> Vec<usize> {
        slots.iter().map(|slot| pool.slot_address(*slot)).collect()
    }

    fn transport_events(staged: &mut Staged, count: usize) -> Vec<TransportEvent> {
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut events = Vec::new();
        while events.len() < count {
            assert!(
                Instant::now() < deadline,
                "{} of {count} transport events within 60 s",
                events.len()
            );
            staged.poll(&mut events).unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(events.len(), count);
        events
    }

    /// Nothing in flight, no disk job owned, no shared segment or window held,
    /// and no disk reply awaited.
    fn settled(staged: &Staged) -> bool {
        staged.in_flight() == (0, 0)
            && staged.owners.is_empty()
            && staged.resources.disk_outstanding == 0
            && staged.maintenance.is_empty()
            && staged.shared_segments.is_empty()
            && staged.shared_windows.is_empty()
            && staged.window_index.is_empty()
    }

    fn corrupt(path: &Path, offset: u64) {
        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&[byte[0] ^ 0x5a]).unwrap();
    }

    #[test]
    fn staged_reads_reject_changed_headers_before_writing_host_or_staging_slabs() {
        let Some(directory) = test_directory("disk-store", "changed-header") else {
            return;
        };
        let geometry = geometry();
        let layout = Layout::new(&geometry).unwrap();
        let expected = header(&geometry, &history(200), 200, None);
        let mut variants = Vec::new();
        let mut changed = expected.clone();
        changed.key.digest[0] ^= 1;
        variants.push(("key", changed));
        let mut changed = expected.clone();
        changed.ranks[0].planes[0] -= 1;
        variants.push(("geometry", changed));
        let mut changed = expected.clone();
        changed.ranks[0].planes[0] += 4096;
        changed.row_segment += u64::from(changed.tail_pages) * 4096;
        variants.push(("page-size", changed));
        let mut changed = expected.clone();
        changed.ranks[0].row_bytes += 4096;
        variants.push(("row-size", changed));
        for (name, changed) in variants {
            let path = directory.join(format!("{name}.ckpt"));
            let mut bytes = vec![0x5a; changed.length_with_row() as usize];
            bytes[..record::HEADER_BYTES].copy_from_slice(&changed.encode().unwrap());
            fs::write(&path, bytes).unwrap();
            for device in [false, true] {
                let mut pool = SlabPool::new(
                    PAGE_RECORD,
                    layout.record_slots(4, true) + layout.row_slots(),
                    0,
                )
                .unwrap();
                for index in 0..4 {
                    pool.page_mut(Slot(index)).fill(0xee);
                }
                for index in 0..2 {
                    pool.page_mut(Slot(index)).fill(0xee);
                }
                let page = pool.slot_address(Slot(0));
                let row = pool.slot_address(Slot(0));
                let (host, staging) = if device {
                    (None, Some(pool))
                } else {
                    (Some(pool), None)
                };
                let mut staged = Staged::new(
                    staging,
                    stopped_transfers(),
                    DiskThread::start("flare-offload-disk").unwrap(),
                );
                if device {
                    staged
                        .fetch(
                            JobId(1),
                            vec![FetchSegment {
                                path: path.clone(),
                                header: expected.clone(),
                                first_page: 0,
                                pages: 1,
                                row: true,
                            }],
                            &InboundTicket {
                                key: expected.key,
                                pages: pages(0..1),
                                slot: StateSlotId(0),
                            },
                        )
                        .unwrap();
                } else {
                    staged
                        .read_into(
                            JobId(1),
                            path.clone(),
                            expected.clone(),
                            0,
                            vec![page],
                            Some(vec![row]),
                        )
                        .unwrap();
                }
                match &transport_events(&mut staged, 1)[..] {
                    [
                        TransportEvent::Fetched {
                            result: Err(error), ..
                        },
                    ] => assert!(
                        error.to_string().contains("different header"),
                        "{name}, device={device}: {error}"
                    ),
                    other => panic!("{name}, device={device}: {other:?}"),
                }
                assert!(settled(&staged));
                let pool = host.as_ref().or_else(|| staged.pool()).unwrap();
                for index in 0..4 {
                    assert!(
                        pool.page(Slot(index)).iter().all(|byte| *byte == 0xee),
                        "{name} changed page {index}, device={device}"
                    );
                }
                for index in 0..2 {
                    assert!(
                        pool.page(Slot(index)).iter().all(|byte| *byte == 0xee),
                        "{name} changed row {index}, device={device}"
                    );
                }
                staged.close().unwrap();
            }
        }
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn compacted_record_is_atomically_rewritten_and_survives_rescan() {
        let Some(directory) = test_directory("disk-store", "row-replacement") else {
            return;
        };
        let full = history(500);
        let root = header(&geometry(), &full, 128, None);
        let child = header(&geometry(), &full, 256, Some(128));
        let mut pool = host_pool();
        let source = page_slabs(&mut pool, 2, 17);
        let row = row_slab(&mut pool, 19);
        let mut staged = Staged::new(
            None,
            stopped_transfers(),
            DiskThread::start("row-refill").unwrap(),
        );
        for (id, header) in [(1, &root), (2, &child)] {
            staged
                .write_from(
                    JobId(id),
                    header.clone(),
                    directory.join(record::temporary_name(&header.key)),
                    directory.join(record::file_name(&header.key)),
                    addresses(&pool, &source),
                    Some(vec![pool.slot_address(row)]),
                    None,
                )
                .unwrap();
            assert!(matches!(
                &transport_events(&mut staged, 1)[0],
                TransportEvent::Stored { result: Ok(_), .. }
            ));
        }
        staged
            .truncate(
                JobId(3),
                directory.join(record::file_name(&root.key)),
                root.row_segment,
            )
            .unwrap();
        assert!(matches!(
            &transport_events(&mut staged, 1)[0],
            TransportEvent::Truncated { result: Ok(_), .. }
        ));
        let guard = crate::testing::directory_lock(&directory).unwrap();
        let scanned = record::scan(&guard, &IDENTITY, &FINGERPRINT, geometry()[0].page_tokens, &record::rank_sizes(&geometry())).unwrap();
        let mut store = DiskStore::new(
            guard,
            IDENTITY,
            FINGERPRINT,
            1 << 30,
            geometry(),
            scanned,
            Box::new(staged),
        )
        .unwrap();
        let (replacement, temporary, path) = store
            .reserve_capture(
                root.key,
                None,
                root.bucket,
                2,
                true,
                true,
                crate::CaptureRole::Endpoint,
            )
            .unwrap()
            .unwrap();
        assert!(!store.index.complete(&child.key));
        for slot in &source {
            pool.page_mut(*slot).fill(0x87);
        }
        pool.page_mut(row).fill(0x91);
        store
            .transport
            .staged_mut()
            .unwrap()
            .write_from(
                JobId(4),
                replacement.clone(),
                temporary,
                path.clone(),
                addresses(&pool, &source),
                Some(vec![pool.slot_address(row)]),
                None,
            )
            .unwrap();
        assert!(matches!(
            &transport_events(store.transport.staged_mut().unwrap(), 1)[0],
            TransportEvent::Stored { result: Ok(_), .. }
        ));
        store
            .commit_record(root.key, Ok(replacement.length_with_row()), None)
            .unwrap();
        assert!(store.index.complete(&child.key));
        let bytes = fs::read(&path).unwrap();
        let layout = Layout::new(&geometry()).unwrap();
        for page in 0..2 {
            for plane in 0..2 {
                let offset = (replacement.page_segment
                    + page * layout.page_record_bytes
                    + layout.ranks[0].plane_offsets[plane]) as usize;
                assert!(
                    bytes[offset..offset + 8192]
                        .iter()
                        .all(|byte| *byte == 0x87)
                );
            }
        }
        assert!(
            bytes[replacement.row_segment as usize..]
                .iter()
                .all(|byte| *byte == 0x91)
        );
        store.close().unwrap();
        drop(store);
        let guard = crate::testing::directory_lock(&directory).unwrap();
        let scanned = record::scan(&guard, &IDENTITY, &FINGERPRINT, geometry()[0].page_tokens, &record::rank_sizes(&geometry())).unwrap();
        assert_eq!(scanned.len(), 2);
        assert!(scanned.iter().all(|record| record.row));
        drop(guard);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn staged_moves_records_between_files_and_caller_buffers_without_a_pool() {
        let Some(directory) = test_directory("disk-store", "host-buffers") else {
            return;
        };
        let geometry = geometry();
        let layout = Layout::new(&geometry).unwrap();
        const PAGES: usize = WINDOW_PAGES + 6;
        let mut pool = SlabPool::new(PAGE_RECORD, 2 * layout.record_slots(PAGES, true), 0).unwrap();
        let mut staged = Staged::new(
            None,
            stopped_transfers(),
            DiskThread::start("flare-offload-disk").unwrap(),
        );
        let full = history(5000);
        let consumed = PAGES as u64 * PAGE_TOKENS + 3;
        let header = header(&geometry, &full, consumed, None);
        assert_eq!(header.tail_pages as usize, PAGES);
        let sources = page_slabs(&mut pool, PAGES, 1);
        let source_row = row_slab(&mut pool, 99);
        let source_addresses = addresses(&pool, &sources);
        let path = directory.join(record::file_name(&header.key));
        let temporary = directory.join(record::temporary_name(&header.key));

        // Without a pool, device pipelines are refused; host pipelines check
        // their arguments before taking the job.
        let mut ids = Ids::new();
        let checkpoint = checkpoint(
            &mut ids,
            &full,
            consumed,
            None,
            pages(0..PAGES as u32),
            PAGES,
            StateSlotId(1),
        );
        assert!(
            staged
                .store(
                    JobId(1),
                    header.clone(),
                    temporary.clone(),
                    path.clone(),
                    &checkpoint,
                    false,
                )
                .is_err()
        );
        let segment = FetchSegment {
            path: path.clone(),
            header: header.clone(),
            first_page: 0,
            pages: PAGES as u32,
            row: true,
        };
        let ticket = InboundTicket {
            key: header.key,
            pages: pages(0..PAGES as u32),
            slot: StateSlotId(1),
        };
        assert!(staged.fetch(JobId(1), vec![segment], &ticket).is_err());
        assert!(
            staged
                .write_from(
                    JobId(1),
                    header.clone(),
                    temporary.clone(),
                    path.clone(),
                    source_addresses[..PAGES - 1].to_vec(),
                    Some(vec![pool.slot_address(source_row)]),
                    None,
                )
                .is_err()
        );
        assert!(
            staged
                .read_into(JobId(1), path.clone(), header.clone(), 0, Vec::new(), None)
                .is_err()
        );
        assert!(settled(&staged));

        // Two windows and the row stream into the file; a second job under
        // the same id is refused while the first is in flight.
        staged
            .write_from(
                JobId(1),
                header.clone(),
                temporary.clone(),
                path.clone(),
                source_addresses.clone(),
                Some(vec![pool.slot_address(source_row)]),
                None,
            )
            .unwrap();
        assert!(
            staged
                .write_from(
                    JobId(1),
                    header.clone(),
                    temporary.clone(),
                    path.clone(),
                    source_addresses.clone(),
                    Some(vec![pool.slot_address(source_row)]),
                    None,
                )
                .is_err()
        );
        assert_eq!(staged.in_flight(), (1, 0));
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Stored { id, result, .. }] => {
                assert_eq!(*id, JobId(1));
                assert_eq!(*result.as_ref().unwrap(), header.length_with_row());
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert_eq!(fs::metadata(&path).unwrap().len(), header.length_with_row());
        assert!(!temporary.exists());
        let scanned = record::scan(
            &crate::testing::directory_lock(&directory).unwrap(),
            &IDENTITY,
            &FINGERPRINT,
            geometry[0].page_tokens,
            &record::rank_sizes(&geometry),
        )
        .unwrap();
        assert_eq!(scanned.len(), 1);
        assert_eq!(scanned[0].header.key, header.key);
        assert!(scanned[0].row);
        assert!(settled(&staged));

        // A suffix of the pages and the row land in caller buffers byte-exact,
        // and pages can be read without the row.
        let targets = page_slabs(&mut pool, PAGES, 1000);
        let target_row = row_slab(&mut pool, 1999);
        const SKIP: usize = 5;
        staged
            .read_into(
                JobId(2),
                path.clone(),
                header.clone(),
                SKIP as u32,
                addresses(&pool, &targets[..PAGES - SKIP]),
                Some(vec![pool.slot_address(target_row)]),
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { id, result }] => {
                assert_eq!(*id, JobId(2));
                assert_eq!(
                    *result.as_ref().unwrap(),
                    layout.record_bytes((PAGES - SKIP) as u64)
                );
            }
            other => panic!("unexpected events {other:?}"),
        }
        for (target, source) in targets.iter().zip(&sources[SKIP..]) {
            assert!(pool.page(*target) == pool.page(*source));
        }
        assert_eq!(
            &pool.page(target_row)[..ROW_RECORD as usize],
            &pool.page(source_row)[..ROW_RECORD as usize]
        );
        staged
            .read_into(
                JobId(3),
                path.clone(),
                header.clone(),
                PAGES as u32 - 2,
                addresses(&pool, &targets[PAGES - 2..]),
                None,
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { id, result }] => {
                assert_eq!(*id, JobId(3));
                assert_eq!(*result.as_ref().unwrap(), 2 * layout.page_record_bytes);
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(pool.page(targets[PAGES - 2]) == pool.page(sources[PAGES - 2]));
        assert!(pool.page(targets[PAGES - 1]) == pool.page(sources[PAGES - 1]));
        assert!(settled(&staged));

        // A missing file fails at open and a corrupted page fails its
        // checksum; both leave nothing in flight.
        staged
            .read_into(
                JobId(4),
                directory.join("missing.ckpt"),
                header.clone(),
                0,
                addresses(&pool, &targets[..1]),
                None,
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { id, result }] => {
                assert_eq!(*id, JobId(4));
                assert!(result.as_ref().unwrap_err().is_record_read());
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(settled(&staged));
        corrupt(
            &path,
            header.page_segment + 10 * layout.page_record_bytes + 100,
        );
        staged
            .read_into(
                JobId(5),
                path.clone(),
                header.clone(),
                0,
                addresses(&pool, &targets),
                Some(vec![pool.slot_address(target_row)]),
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { id, result }] => {
                assert_eq!(*id, JobId(5));
                assert!(result.as_ref().unwrap_err().is_record_read());
                assert!(
                    result
                        .as_ref()
                        .unwrap_err()
                        .to_string()
                        .contains("checksum")
                );
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(settled(&staged));

        // Truncation drops the row, after which reading it fails; unlinking
        // removes the file, and unlinking again fails.
        staged
            .truncate(JobId(6), path.clone(), header.row_segment)
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Truncated { id, result }] => {
                assert_eq!(*id, JobId(6));
                result.as_ref().unwrap();
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert_eq!(fs::metadata(&path).unwrap().len(), header.row_segment);
        staged
            .read_into(
                JobId(7),
                path.clone(),
                header.clone(),
                WINDOW_PAGES as u32,
                addresses(&pool, &targets[..PAGES - WINDOW_PAGES]),
                Some(vec![pool.slot_address(target_row)]),
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { id, result }] => {
                assert_eq!(*id, JobId(7));
                assert!(result.is_err());
            }
            other => panic!("unexpected events {other:?}"),
        }
        staged.unlink(JobId(8), path.clone()).unwrap();
        staged.unlink(JobId(9), path.clone()).unwrap();
        match &transport_events(&mut staged, 2)[..] {
            [
                TransportEvent::Unlinked {
                    id: first,
                    result: Ok(()),
                },
                TransportEvent::Unlinked {
                    id: second,
                    result: Ok(()),
                },
            ] => assert_eq!((*first, *second), (JobId(8), JobId(9))),
            other => panic!("unexpected events {other:?}"),
        }
        assert!(!path.exists());
        assert!(settled(&staged));

        staged.close().unwrap();
        assert!(
            staged
                .write_from(
                    JobId(10),
                    header.clone(),
                    temporary.clone(),
                    path.clone(),
                    source_addresses.clone(),
                    Some(vec![pool.slot_address(source_row)]),
                    None,
                )
                .is_err()
        );
        assert!(staged.truncate(JobId(10), path.clone(), 0).is_err());
        assert!(staged.unlink(JobId(10), path.clone()).is_err());
        let mut events = Vec::new();
        staged.poll(&mut events).unwrap();
        assert!(events.is_empty());
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn staged_failures_release_staging_slabs_and_abort_temporary_files() {
        let Some(directory) = test_directory("disk-store", "failures") else {
            return;
        };
        let geometry = geometry();
        let layout = Layout::new(&geometry).unwrap();
        let mut host = host_pool();
        let mut staged = Staged::new(
            Some(
                SlabPool::new(
                    layout.page_record_bytes,
                    crate::offload::intake_slots(&layout, 1),
                    crate::offload::intake_slots(&layout, 1),
                )
                .unwrap(),
            ),
            stopped_transfers(),
            DiskThread::start("flare-offload-disk").unwrap(),
        );
        let full = history(2000);
        let mut ids = Ids::new();
        let free = |staged: &Staged| {
            let pool = staged.pool().unwrap();
            pool.free_slots()
        };
        let count = crate::offload::intake_slots(&layout, 1);
        let all = count;
        assert_eq!(free(&staged), all);

        // A record written from caller buffers gives the fetches below a file.
        let stored = header(&geometry, &full, 3 * PAGE_TOKENS + 1, None);
        let sources = page_slabs(&mut host, 3, 1);
        let source_row = row_slab(&mut host, 9);
        let source_addresses = addresses(&host, &sources);
        let path = directory.join(record::file_name(&stored.key));
        staged
            .write_from(
                JobId(1),
                stored.clone(),
                directory.join(record::temporary_name(&stored.key)),
                path.clone(),
                source_addresses.clone(),
                Some(vec![host.slot_address(source_row)]),
                None,
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Stored { id, result, .. }] => {
                assert_eq!(*id, JobId(1));
                result.as_ref().unwrap();
            }
            other => panic!("unexpected events {other:?}"),
        }

        // Without transfer threads a store fails at its first window: the
        // slots return to the pool at once, the temporary file is aborted once
        // its create reply arrives, and the failure is reported once.
        let device = header(&geometry, &full, 3 * PAGE_TOKENS + 2, None);
        let checkpoint = checkpoint(
            &mut ids,
            &full,
            3 * PAGE_TOKENS + 2,
            None,
            pages(0..3),
            3,
            StateSlotId(1),
        );
        let device_temporary = directory.join(record::temporary_name(&device.key));
        let device_path = directory.join(record::file_name(&device.key));
        staged
            .store(
                JobId(2),
                device.clone(),
                device_temporary.clone(),
                device_path.clone(),
                &checkpoint,
                false,
            )
            .unwrap();
        assert_eq!(staged.in_flight(), (1, 0));
        assert_eq!(free(&staged), all);
        assert!(staged.outbound[&JobId(2)].file == TemporaryFile::Creating);
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Stored { id, result, .. }] => {
                assert_eq!(*id, JobId(2));
                assert!(
                    result
                        .as_ref()
                        .unwrap_err()
                        .to_string()
                        .contains("no offload threads")
                );
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(!device_temporary.exists() && !device_path.exists());
        assert!(settled(&staged));
        assert_eq!(free(&staged), all);

        // A fetch reads its window from the file into staging and fails at
        // the upload: the slots are released after the read replies and the
        // file is closed.
        let ticket = InboundTicket {
            key: stored.key,
            pages: pages(10..13),
            slot: StateSlotId(2),
        };
        let segment = FetchSegment {
            path: path.clone(),
            header: stored.clone(),
            first_page: 0,
            pages: 3,
            row: true,
        };
        staged
            .fetch(JobId(3), vec![segment.clone()], &ticket)
            .unwrap();
        assert_eq!(free(&staged), count - 4);
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { id, result }] => {
                assert_eq!(*id, JobId(3));
                assert!(!result.as_ref().unwrap_err().is_record_read());
                assert!(
                    result
                        .as_ref()
                        .unwrap_err()
                        .to_string()
                        .contains("no offload threads")
                );
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(settled(&staged));
        assert_eq!(free(&staged), all);

        // A file holding a different record than the index expects fails the
        // fetch at open.
        let mut other = segment.clone();
        other.header = device.clone();
        staged.fetch(JobId(4), vec![other], &ticket).unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { id, result }] => {
                assert_eq!(*id, JobId(4));
                assert!(
                    result
                        .as_ref()
                        .unwrap_err()
                        .to_string()
                        .contains("different header")
                );
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(settled(&staged));
        assert_eq!(free(&staged), all);

        // A plan must take the row from its last record and cover the ticket.
        let mut no_row = segment.clone();
        no_row.row = false;
        assert!(staged.fetch(JobId(5), vec![no_row], &ticket).is_err());
        let short = InboundTicket {
            pages: pages(10..12),
            ..ticket.clone()
        };
        assert!(
            staged
                .fetch(JobId(5), vec![segment.clone()], &short)
                .is_err()
        );
        assert!(settled(&staged));

        // A commit that cannot rename into place removes the temporary file;
        // a create that cannot open its temporary file fails without an abort.
        let lost = header(&geometry, &full, 3 * PAGE_TOKENS + 4, None);
        let lost_temporary = directory.join(record::temporary_name(&lost.key));
        let missing = directory.join("missing");
        staged
            .write_from(
                JobId(6),
                lost.clone(),
                lost_temporary.clone(),
                missing.join(record::file_name(&lost.key)),
                source_addresses.clone(),
                Some(vec![host.slot_address(source_row)]),
                None,
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { .. }] => panic!("unexpected fetch event"),
            [TransportEvent::Stored { id, result, .. }] => {
                assert_eq!(*id, JobId(6));
                assert!(result.as_ref().unwrap_err().to_string().contains("rename"));
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(!lost_temporary.exists());
        staged
            .write_from(
                JobId(7),
                lost.clone(),
                missing.join(record::temporary_name(&lost.key)),
                directory.join(record::file_name(&lost.key)),
                source_addresses.clone(),
                Some(vec![host.slot_address(source_row)]),
                None,
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Stored { id, result, .. }] => {
                assert_eq!(*id, JobId(7));
                assert!(result.is_err());
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(settled(&staged));
        assert_eq!(free(&staged), all);
        assert_eq!(
            record::scan(
                &crate::testing::directory_lock(&directory).unwrap(),
                &IDENTITY,
                &FINGERPRINT,
                geometry[0].page_tokens,
                &record::rank_sizes(&geometry)
            )
            .unwrap()
            .len(),
            1
        );
        staged.close().unwrap();
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn staged_holds_submissions_at_the_disk_queue_depth_and_finishes_every_record() {
        let Some(directory) = test_directory("disk-store", "queue-depth") else {
            return;
        };
        let geometry = geometry();
        let mut host = host_pool();
        let mut staged = Staged::new(
            None,
            stopped_transfers(),
            DiskThread::start("flare-offload-disk").unwrap(),
        );
        // Three windows, two staged at a time: a record first submits its
        // create and two page writes, then the third window with the row,
        // then the commit.
        const PAGES: usize = 2 * WINDOW_PAGES + 1;
        const RECORDS: u64 = 24;
        let sources = page_slabs(&mut host, PAGES, 1);
        let source_row = row_slab(&mut host, 7);
        let source_addresses = addresses(&host, &sources);
        let full = history(PAGES as u64 * PAGE_TOKENS + 100);
        let headers: Vec<Header> = (0..RECORDS)
            .map(|offset| header(&geometry, &full, PAGES as u64 * PAGE_TOKENS + offset, None))
            .collect();
        for (index, header) in headers.iter().enumerate() {
            staged
                .write_from(
                    JobId(index as u64 + 1),
                    header.clone(),
                    directory.join(record::temporary_name(&header.key)),
                    directory.join(record::file_name(&header.key)),
                    source_addresses.clone(),
                    Some(vec![host.slot_address(source_row)]),
                    None,
                )
                .unwrap();
        }
        // Twenty-one records submit their create and both page writes, the
        // next record's create fills the queue and its page writes hold, and
        // the last two hold before their create.
        assert_eq!(staged.resources.disk_outstanding, disk::QUEUE_DEPTH);
        let (held, partial) = staged
            .outbound
            .values()
            .fold((0, 0), |(held, partial), record| {
                if record.file == TemporaryFile::Unsubmitted {
                    (held + 1, partial)
                } else if record.active.iter().any(|window| !window.pages_submitted) {
                    (held, partial + 1)
                } else {
                    (held, partial)
                }
            });
        assert_eq!((held, partial), (2, 1));
        assert_eq!(staged.in_flight(), (RECORDS as usize, 0));

        let deadline = Instant::now() + Duration::from_secs(120);
        let mut events = Vec::new();
        while events.len() < RECORDS as usize {
            assert!(
                Instant::now() < deadline,
                "{} of {RECORDS} records within 120 s",
                events.len()
            );
            staged.poll(&mut events).unwrap();
            assert!(staged.resources.disk_outstanding <= disk::QUEUE_DEPTH);
            std::thread::sleep(Duration::from_millis(1));
        }
        let mut done: Vec<u64> = events
            .iter()
            .map(|event| match event {
                TransportEvent::Stored { id, result, .. } => {
                    result.as_ref().unwrap();
                    id.0
                }
                other => panic!("unexpected event {other:?}"),
            })
            .collect();
        done.sort_unstable();
        assert_eq!(done, (1..=RECORDS).collect::<Vec<_>>());
        for header in &headers {
            assert_eq!(
                fs::metadata(directory.join(record::file_name(&header.key)))
                    .unwrap()
                    .len(),
                header.length_with_row()
            );
        }
        assert!(settled(&staged));
        assert_eq!(
            record::scan(
                &crate::testing::directory_lock(&directory).unwrap(),
                &IDENTITY,
                &FINGERPRINT,
                geometry[0].page_tokens,
                &record::rank_sizes(&geometry)
            )
            .unwrap()
            .len(),
            RECORDS as usize
        );
        staged.close().unwrap();
        fs::remove_dir_all(&directory).unwrap();
    }

    /// Write the record file of `header` with known pages, their checksums and a
    /// row; returns the pages and the row.
    fn write_record(directory: &Path, header: &mut Header, seed: u8) -> (Vec<Vec<u8>>, Vec<u8>) {
        let tail = header.tail_pages as usize;
        let pages: Vec<Vec<u8>> = (0..tail)
            .map(|p| {
                (0..PAGE_RECORD as usize)
                    .map(|i| (i * 3 + p + seed as usize) as u8)
                    .collect()
            })
            .collect();
        let row: Vec<u8> = (0..ROW_RECORD as usize)
            .map(|i| (i + seed as usize) as u8)
            .collect();
        header.row_checksum = record::checksum(&row);
        let mut file = vec![0u8; header.length_with_row() as usize];
        file[..record::HEADER_BYTES].copy_from_slice(&header.encode().unwrap());
        for (index, page) in pages.iter().enumerate() {
            let at = header.page_segment as usize + index * PAGE_RECORD as usize;
            file[at..at + page.len()].copy_from_slice(page);
            let sum = header.checksum_segment() as usize + index * 8;
            file[sum..sum + 8].copy_from_slice(&record::checksum(page).to_le_bytes());
        }
        let at = header.row_segment as usize;
        file[at..at + row.len()].copy_from_slice(&row);
        fs::write(directory.join(record::file_name(&header.key)), file).unwrap();
        (pages, row)
    }

    #[test]
    fn records_on_disk_are_announced_and_exported_with_the_checksums_their_files_persist() {
        use crate::pool::{Announcer, HostWindow, Notice, Tier as PoolTier};
        let geometry = geometry();
        let directory = directory();
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        let full = history(2000);
        let mut root = header(&geometry, &full, 700, None);
        let mut leaf = header(&geometry, &full, 1300, Some(700));
        let (root_pages, _) = write_record(&directory, &mut root, 1);
        let (leaf_pages, leaf_row) = write_record(&directory, &mut leaf, 2);
        let records = vec![
            scanned(&directory, root.clone(), 0, true),
            scanned(&directory, leaf.clone(), 1, true),
        ];
        let (mut store, _log) = store_with(records, 1 << 30);
        let announcer = Announcer::new(Arc::new(tokio::sync::Notify::new()));
        announcer.begin_session();
        store.announce_to(announcer.clone());
        store.flush_announcements();
        let notices = announcer.take_batch(10).unwrap();
        assert_eq!(notices.len(), 2, "both files are announced");
        assert!(
            notices
                .iter()
                .all(|n| matches!(n, Notice::Saved(e) if e.tier == PoolTier::Nvme))
        );
        assert_eq!(
            store.pin_export(PoolTier::Dram, leaf.key, 0, true).err(),
            Some(ExportRefusal::NotFound)
        );
        let mut pin = store.pin_export(PoolTier::Nvme, leaf.key, 0, true).unwrap();
        let chain = pin.chain().clone();
        assert_eq!(chain.records.len(), 2);
        assert_eq!(chain.total_pages, 20);
        assert_eq!(chain.records[0].key, root.key);
        assert!(store.index.nodes().all(|node| node.pinned()));
        // Pages and checksums come from the files.
        let window = HostWindow {
            record: 0,
            first_page: 2,
            page_count: 3,
            row: false,
        };
        let mut dst = vec![0u8; 3 * PAGE_RECORD as usize];
        let sums = pin.read(&window, &mut dst).unwrap();
        assert_eq!(dst[..PAGE_RECORD as usize], root_pages[2][..]);
        assert_eq!(sums.pages[0], record::checksum(&root_pages[2]));
        let window = HostWindow {
            record: 1,
            first_page: 9,
            page_count: 1,
            row: true,
        };
        let mut dst = vec![0u8; (PAGE_RECORD + ROW_RECORD) as usize];
        let sums = pin.read(&window, &mut dst).unwrap();
        assert_eq!(dst[PAGE_RECORD as usize..], leaf_row[..]);
        assert_eq!(sums.row, Some(record::checksum(&leaf_row)));
        // A page that rotted on disk is served as it is, with the checksum it
        // was written with, so the requester's check fails.
        let path = directory.join(record::file_name(&leaf.key));
        let mut file = fs::read(&path).unwrap();
        file[leaf.page_segment as usize + 5] ^= 1;
        fs::write(&path, file).unwrap();
        let window = HostWindow {
            record: 1,
            first_page: 0,
            page_count: 1,
            row: false,
        };
        let mut dst = vec![0u8; PAGE_RECORD as usize];
        let sums = pin.read(&window, &mut dst).unwrap();
        assert_ne!(record::checksum(&dst), sums.pages[0]);
        assert_eq!(sums.pages[0], record::checksum(&leaf_pages[0]));
        // Starting inside the leaf needs only the leaf.
        let suffix = store
            .pin_export(PoolTier::Nvme, leaf.key, 700 + 640, false)
            .unwrap();
        assert_eq!(suffix.chain().records.len(), 1);
        drop(suffix);
        // Dropping the pins releases the records at the next flush.
        drop(pin);
        store.flush_announcements();
        assert!(store.index.nodes().all(|node| !node.pinned()));
        // Nothing further changed, so nothing further is announced.
        store.flush_announcements();
        assert!(announcer.take_batch(10).is_none());
    }

    fn announced(store: &mut DiskStore) -> crate::pool::Announcer {
        let announcer = crate::pool::Announcer::new(Arc::new(tokio::sync::Notify::new()));
        announcer.begin_session();
        store.announce_to(announcer.clone());
        store.flush_announcements();
        announcer
    }

    #[test]
    fn a_spilling_store_holds_the_coldest_leaf_until_its_move_ends() {
        let full = history(1000);
        let cold = header(&geometry(), &full, 128, None);
        let warm = header(&geometry(), &full, 256, None);
        let limit = cold.length_with_row() + warm.length_with_row();
        let records = vec![
            scanned(&directory(), cold.clone(), 0, true),
            scanned(&directory(), warm.clone(), 1, true),
        ];
        let (mut store, log) = store_with(records, limit);
        let _announcer = announced(&mut store);
        store.set_spill(true);

        // Room for one more page: the coldest file is held, nothing is unlinked.
        assert!(!store.make_room(PAGE_RECORD).unwrap());
        let retiring = store.take_retiring();
        assert_eq!(retiring.len(), 1);
        assert_eq!(
            (retiring[0].key, retiring[0].tier, retiring[0].row),
            (cold.key, PoolTier::Nvme, true)
        );
        assert!(unlinked(&log).is_empty());
        assert!(store.index.node(&cold.key).unwrap().pinned());

        // The held file already covers the request: nothing more is chosen.
        assert!(!store.make_room(PAGE_RECORD).unwrap());
        assert!(store.take_retiring().is_empty());

        // A failed move keeps the file, which is offered again.
        crate::store::TierStore::retired(&mut store, cold.key, PoolTier::Nvme, Retirement::Failed);
        assert!(unlinked(&log).is_empty());
        assert!(!store.index.node(&cold.key).unwrap().pinned());
        assert!(!store.make_room(PAGE_RECORD).unwrap());
        assert_eq!(store.take_retiring()[0].key, cold.key);

        // Once moved, the file is unlinked and its bytes are free.
        crate::store::TierStore::retired(&mut store, cold.key, PoolTier::Nvme, Retirement::Moved);
        assert_eq!(unlinked(&log).len(), 1);
        assert!(!store.index.contains(&cold.key));
        assert_eq!(store.take_evicted(), [cold.key]);
        assert!(store.make_room(PAGE_RECORD).unwrap());
    }

    #[test]
    fn a_busy_or_unavailable_manager_leaves_local_lru_as_the_end() {
        let full = history(1000);
        let cold = header(&geometry(), &full, 128, None);
        let warm = header(&geometry(), &full, 256, None);
        let limit = cold.length_with_row() + warm.length_with_row();
        let (mut store, log) = store_with(
            vec![
                scanned(&directory(), cold.clone(), 0, true),
                scanned(&directory(), warm.clone(), 1, true),
            ],
            limit,
        );
        let _announcer = announced(&mut store);
        store.set_spill(true);
        store.maintain();
        let selected = store.take_retiring();
        assert!(!selected.is_empty());
        for r in selected {
            store.retired(&r.key, Retirement::Failed).unwrap();
        }
        store.set_spill(false);
        assert!(store.index.contains(&cold.key));
        assert!(unlinked(&log).is_empty());
        assert!(store.make_room(PAGE_RECORD).unwrap());
        assert!(!store.index.contains(&cold.key));
        assert!(store.index.contains(&warm.key));
        assert_eq!(unlinked(&log).len(), 1);
        assert_eq!(store.stats().dropped[3], 1);
        assert!(store.stats().held[1] > 0);
    }

    #[test]
    fn a_store_that_is_not_spilling_unlinks_the_coldest_leaf_as_before() {
        let full = history(1000);
        let cold = header(&geometry(), &full, 128, None);
        let warm = header(&geometry(), &full, 256, None);
        let limit = cold.length_with_row() + warm.length_with_row();
        let records = vec![
            scanned(&directory(), cold.clone(), 0, true),
            scanned(&directory(), warm.clone(), 1, true),
        ];
        let (mut store, log) = store_with(records, limit);
        let _announcer = announced(&mut store);
        // Spilling was on and the manager went away.
        store.set_spill(true);
        store.set_spill(false);
        assert!(store.make_room(PAGE_RECORD).unwrap());
        assert!(store.take_retiring().is_empty());
        assert_eq!(unlinked(&log).len(), 1);
        assert!(!store.index.contains(&cold.key));
    }

    #[test]
    fn a_moved_record_stays_until_the_pin_that_reads_it_is_released() {
        use crate::pool::Tier as PoolTier;
        let geometry = geometry();
        let directory = directory();
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        let full = history(2000);
        let mut leaf = header(&geometry, &full, 700, None);
        write_record(&directory, &mut leaf, 1);
        let (mut store, log) = store_with(
            vec![scanned(&directory, leaf.clone(), 0, true)],
            leaf.length_with_row(),
        );
        let _announcer = announced(&mut store);
        store.set_spill(true);
        assert!(!store.make_room(PAGE_RECORD).unwrap());
        assert_eq!(store.take_retiring().remove(0).key, leaf.key);
        let pin = store.pin_export(PoolTier::Nvme, leaf.key, 0, true).unwrap();

        crate::store::TierStore::retired(&mut store, leaf.key, PoolTier::Nvme, Retirement::Moved);
        assert!(unlinked(&log).is_empty(), "a pin still reads the file");
        drop(pin);
        store.flush_announcements();
        assert_eq!(unlinked(&log).len(), 1);
        assert!(!store.index.contains(&leaf.key));
    }

    #[test]
    fn a_publication_starts_the_disk_copy() {
        let (mut store, log) = store_with(Vec::new(), 1 << 30);
        let full = history(1000);
        let mut ids = Ids::new();
        let publication = checkpoint(&mut ids, &full, 128, None, pages(0..2), 2, StateSlotId(1));
        assert_eq!(store.store(&publication).unwrap(), StoreOutcome::Queued);
        assert_eq!(log.borrow().calls.len(), 1);
    }
    #[test]
    fn adaptive_records_spill_recover_restore_and_reject_corrupt_shares() {
        use crate::{composite::CompositeStore, memory::MemoryStore, testing::SimulatedGpu};
        const PAGES: usize = WINDOW_PAGES + 3;
        for composite in [false, true] {
            for pooled in [false, true] {
                for mode in 0..3 {
                    for same_row in [false, true] {
                        let Some(directory) = test_directory(
                            "replicated",
                            &format!("{composite}-{pooled}-{mode}-{same_row}"),
                        ) else {
                            return;
                        };
                        let geometry = vec![rank(PAGE_TOKENS, &[("kv", 4096)], 12288); 4];
                        let layout = Layout::new(&geometry).unwrap();
                        let intake = crate::offload::intake_slots(&layout, 1);
                        let pool = SlabPool::new(
                            layout.page_record_bytes,
                            layout.record_slots(PAGES, true) + intake + 16,
                            intake,
                        )
                        .unwrap();
                        let (copies, device) = SimulatedGpu::shared_mapped(&geometry).unwrap();
                        let (memory_copies, memory_device) =
                            SimulatedGpu::shared_mapped(&geometry).unwrap();
                        let page_data: Vec<Vec<u8>> = (0..PAGES)
                            .map(|page| {
                                let mut bytes = vec![0; layout.page_record_bytes as usize];
                                for (rank, part) in layout.ranks.iter().enumerate() {
                                    pattern(
                                        &mut bytes[part.page_offset as usize
                                            ..part.page_offset as usize + 4096],
                                        (page
                                            + if mode == 0 || (mode == 2 && page % 2 == 0) {
                                                0
                                            } else {
                                                rank
                                            }) as u64,
                                    );
                                }
                                bytes
                            })
                            .collect();
                        let mut row_data = vec![0; layout.row_record_bytes as usize];
                        for (rank, part) in layout.ranks.iter().enumerate() {
                            pattern(
                                &mut row_data
                                    [part.row_offset as usize..part.row_offset as usize + 12288],
                                if same_row { 100 } else { 100 + rank as u64 },
                            );
                        }
                        for gpu in [&device, &memory_device] {
                            for (page, bytes) in page_data.iter().enumerate() {
                                gpu.pages.lock().unwrap().insert(page as u32, bytes.clone());
                            }
                            gpu.rows.lock().unwrap().insert(1, row_data.clone());
                        }
                        let disk = DiskStore::new(
                            record::DirectoryLock::acquire(&directory).unwrap(),
                            IDENTITY,
                            FINGERPRINT,
                            1 << 30,
                            geometry.clone(),
                            Vec::new(),
                            Box::new(Staged::new(
                                Some(pool.clone()),
                                copies,
                                DiskThread::start("replicated-write").unwrap(),
                            )),
                        )
                        .unwrap();
                        let mut store: Box<dyn TierStore> = if composite {
                            let memory =
                                MemoryStore::new_flat(IDENTITY, pool.clone(), memory_copies)
                                    .unwrap();
                            let mut store = CompositeStore::new(memory, disk).unwrap();
                            store.drain_on_shutdown = Duration::from_secs(5);
                            Box::new(store)
                        } else {
                            Box::new(disk)
                        };
                        if pooled {
                            store.announce_to(Announcer::new(Arc::new(tokio::sync::Notify::new())));
                        }
                        let full = history(PAGES as u64 * PAGE_TOKENS + 3);
                        let checkpoint = checkpoint(
                            &mut Ids::new(),
                            &full,
                            full.token_count(),
                            None,
                            pages(0..PAGES as u32),
                            PAGES,
                            StateSlotId(1),
                        );
                        assert_eq!(store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
                        let deadline = Instant::now() + Duration::from_secs(10);
                        poll_until(&mut *store, deadline, |e| {
                            matches!(e, StoreEvent::Stored { result: Ok(_), .. })
                        });
                        store.close().unwrap();
                        let packed = !pooled && (mode == 0 || same_row);
                        assert_eq!(
                            store.stats().packed_records,
                            [u64::from(composite && packed), u64::from(packed)]
                        );
                        assert_eq!(
                            store.stats().per_rank_records,
                            [u64::from(composite && !packed), u64::from(!packed)]
                        );
                        assert_eq!(store.stats().abandoned_stores, 0);
                        drop(store);
                        let pool = SlabPool::new(
                            layout.page_record_bytes,
                            layout.record_slots(PAGES, true) + intake + 16,
                            intake,
                        )
                        .unwrap();
                        let guard = record::DirectoryLock::acquire(&directory).unwrap();
                        let mut scanned = record::scan(
                            &guard,
                            &IDENTITY,
                            &FINGERPRINT,
                            PAGE_TOKENS,
                            &record::rank_sizes(&geometry),
                        )
                        .unwrap();
                        assert_eq!(scanned.len(), 1);
                        let saved = scanned.remove(0);
                        let header = saved.header.clone();
                        if pooled || (mode != 0 && !same_row) {
                            assert!(!header.replicated());
                            assert_eq!(saved.bytes, header.length_with_row());
                        } else {
                            assert_eq!(header.replication().pages, mode == 0);
                            assert_eq!(header.replication().row, same_row);
                            assert_eq!(saved.bytes, header.length_with_row());
                            let full_bytes = header.page_segment
                                + PAGES as u64 * layout.page_record_bytes
                                + layout.row_record_bytes;
                            assert!(saved.bytes < full_bytes);
                        }
                        assert!(saved.row);
                        assert_eq!(header.fingerprint, FINGERPRINT);
                        let (copies, restored) = SimulatedGpu::shared_mapped(&geometry).unwrap();
                        let mut recovered = DiskStore::new(
                            guard,
                            IDENTITY,
                            FINGERPRINT,
                            1 << 30,
                            geometry.clone(),
                            vec![saved.clone()],
                            Box::new(Staged::new(
                                Some(pool.clone()),
                                copies,
                                DiskThread::start("replicated-read").unwrap(),
                            )),
                        )
                        .unwrap();
                        assert_eq!(recovered.used(), saved.bytes);
                        // Replicated local files are kept local when a pool is enabled later.
                        recovered
                            .announce_records(Announcer::new(Arc::new(tokio::sync::Notify::new())));
                        assert_eq!(
                            recovered.announced_entry(&checkpoint.key).is_some(),
                            !header.replicated()
                        );
                        let hit = recovered.lookup(&full, 2 * PAGE_TOKENS, u64::MAX).unwrap();
                        recovered
                            .fetch(
                                hit,
                                InboundTicket {
                                    key: checkpoint.key,
                                    pages: pages(1000..1000 + PAGES as u32 - 2),
                                    slot: StateSlotId(7),
                                },
                            )
                            .unwrap();
                        poll_until(&mut recovered, deadline, |e| {
                            matches!(e, StoreEvent::Fetched { result: Ok(_), .. })
                        });
                        for (index, bytes) in page_data.iter().skip(2).enumerate() {
                            assert_eq!(
                                &restored.pages.lock().unwrap()[&(1000 + index as u32)],
                                bytes
                            );
                        }
                        assert_eq!(restored.rows.lock().unwrap()[&7], row_data);
                        if !composite && !pooled && mode < 2 && !same_row {
                            let slots = pool.allocate(layout.record_slots(1, true)).unwrap();
                            for (number, offset) in [header.page_segment, header.row_segment]
                                .into_iter()
                                .enumerate()
                            {
                                let mut file = fs::OpenOptions::new()
                                    .read(true)
                                    .write(true)
                                    .open(&saved.path)
                                    .unwrap();
                                file.seek(SeekFrom::Start(offset)).unwrap();
                                let mut byte = [0];
                                file.read_exact(&mut byte).unwrap();
                                file.seek(SeekFrom::Start(offset)).unwrap();
                                file.write_all(&[byte[0] ^ 1]).unwrap();
                                let staged = recovered.transport.staged_mut().unwrap();
                                staged
                                    .read_into(
                                        JobId(100 + number as u64),
                                        saved.path.clone(),
                                        header.clone(),
                                        0,
                                        vec![pool.slot_address(slots[0])],
                                        Some(pool.addresses(&slots[1..])),
                                    )
                                    .unwrap();
                                let result = transport_events(staged, 1);
                                assert!(
                                    matches!(&result[..], [TransportEvent::Fetched { result: Err(error), .. }] if error.to_string().contains("checksum verification")),
                                    "{result:?}"
                                );
                                file.seek(SeekFrom::Start(offset)).unwrap();
                                file.write_all(&byte).unwrap();
                            }
                            pool.release(slots);
                        }
                        if composite && !pooled {
                            let (copier, gpu) = SimulatedGpu::shared_mapped(&geometry).unwrap();
                            let memory =
                                MemoryStore::new_flat(IDENTITY, pool.clone(), copier).unwrap();
                            let mut promoted = CompositeStore::new(memory, recovered).unwrap();
                            let hit = promoted.lookup(&full, 0, u64::MAX).unwrap();
                            promoted
                                .fetch(
                                    hit,
                                    InboundTicket {
                                        key: checkpoint.key,
                                        pages: pages(3000..3000 + PAGES as u32),
                                        slot: StateSlotId(9),
                                    },
                                )
                                .unwrap();
                            poll_until(&mut promoted, deadline, |e| {
                                matches!(e, StoreEvent::Fetched { result: Ok(_), .. })
                            });
                            for (i, bytes) in page_data.iter().enumerate() {
                                assert_eq!(&gpu.pages.lock().unwrap()[&(3000 + i as u32)], bytes);
                            }
                            assert_eq!(gpu.rows.lock().unwrap()[&9], row_data);
                            assert_eq!(
                                promoted.stats().packed_records[0],
                                u64::from(mode == 0 || same_row)
                            );
                            promoted.close().unwrap();
                        } else {
                            recovered.close().unwrap();
                        }
                        fs::remove_dir_all(directory).unwrap();
                    }
                }
            }
        }
    }
    /// Release the last GPU window first, one reply per poll, so writes arrive out of order.
    struct ReverseWindows {
        copier: SharedCopier,
        layout: Layout,
        replies: Vec<copier::Reply>,
        completed: Arc<Mutex<Vec<u32>>>,
    }
    impl PageCopier for ReverseWindows {
        fn layout(&self) -> &Layout {
            &self.layout
        }
        fn submit(&mut self, job: copier::Job) -> Result<()> {
            lock(&self.copier).submit(job)
        }
        fn poll(&mut self, replies: &mut Vec<copier::Reply>) {
            lock(&self.copier).poll(&mut self.replies);
            if let Some(reply) = self.replies.pop() {
                self.completed.lock().unwrap().push(reply.window);
                replies.push(reply);
            }
        }
        fn in_flight(&self) -> usize {
            lock(&self.copier).in_flight() + self.replies.len()
        }
        fn close(&mut self) -> Result<()> {
            lock(&self.copier).close()
        }
    }

    #[test]
    fn disk_only_windows_complete_out_of_order_and_abandon_a_later_mismatch() {
        for mismatch in [false, true] {
            let Some(directory) = test_directory("replicated", &format!("reverse-{mismatch}"))
            else {
                return;
            };
            let geometry = vec![rank(PAGE_TOKENS, &[("kv", 4096)], 12288); 4];
            let layout = Layout::new(&geometry).unwrap();
            let intake = crate::offload::intake_slots(&layout, 1);
            let pool = SlabPool::new(layout.page_record_bytes, intake, intake).unwrap();
            let (copier, gpu) = crate::testing::SimulatedGpu::shared_mapped(&geometry).unwrap();
            let order = Arc::new(Mutex::new(Vec::new()));
            let copier = Arc::new(Mutex::new(ReverseWindows {
                copier,
                layout: layout.clone(),
                replies: Vec::new(),
                completed: order.clone(),
            }));
            let mut store = DiskStore::new(
                record::DirectoryLock::acquire(&directory).unwrap(),
                IDENTITY,
                FINGERPRINT,
                1 << 30,
                geometry,
                Vec::new(),
                Box::new(Staged::new(
                    Some(pool.clone()),
                    copier,
                    DiskThread::start("reverse-windows").unwrap(),
                )),
            )
            .unwrap();
            for page in 0..WINDOW_PAGES + 3 {
                let mut bytes = vec![7; layout.page_record_bytes as usize];
                // Window 1 decides packed first. A difference in window 0 must abandon this store.
                if mismatch && page == 0 {
                    bytes[4096] = 8;
                }
                gpu.pages.lock().unwrap().insert(page as u32, bytes);
            }
            gpu.rows
                .lock()
                .unwrap()
                .insert(1, vec![9; layout.row_record_bytes as usize]);
            let h = history((WINDOW_PAGES + 3) as u64 * PAGE_TOKENS + 3);
            let checkpoint = checkpoint(
                &mut Ids::new(),
                &h,
                h.token_count(),
                None,
                pages(0..(WINDOW_PAGES + 3) as u32),
                WINDOW_PAGES + 3,
                StateSlotId(1),
            );
            assert_eq!(store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
            let events = poll_until(&mut store, Instant::now() + Duration::from_secs(10), |e| {
                matches!(e, StoreEvent::Stored { .. })
            });
            assert!(matches!(order.lock().unwrap().as_slice(), [1, 0]));
            assert_eq!(store.stats().abandoned_stores, u64::from(mismatch));
            assert_eq!(store.stats().store_failed, u64::from(mismatch));
            assert_eq!(store.stats().packed_records, [0, u64::from(!mismatch)]);
            assert_eq!(store.stats().per_rank_records, [0, 0]);
            if mismatch {
                assert!(
                    matches!(&events[..], [StoreEvent::Stored { result: Err(error), .. }] if error.carried().is_some_and(|e| e.is::<disk::ReplicaMismatch>()))
                );
                assert_eq!(store.used(), 0);
                assert!(store.record(&checkpoint.key).is_none());
                assert!(!directory.join(record::file_name(&checkpoint.key)).exists());
                assert!(
                    !directory
                        .join(record::temporary_name(&checkpoint.key))
                        .exists()
                );
            } else {
                assert!(matches!(
                    &events[..],
                    [StoreEvent::Stored { result: Ok(_), .. }]
                ));
                assert!(
                    store
                        .record(&checkpoint.key)
                        .unwrap()
                        .header
                        .replication()
                        .pages
                );
                let hit = store.lookup(&h, 0, u64::MAX).unwrap();
                store
                    .fetch(
                        hit,
                        InboundTicket {
                            key: checkpoint.key,
                            pages: pages(1000..1000 + (WINDOW_PAGES + 3) as u32),
                            slot: StateSlotId(9),
                        },
                    )
                    .unwrap();
                let deadline = Instant::now() + Duration::from_secs(10);
                poll_until(&mut store, deadline, |e| {
                    matches!(e, StoreEvent::Fetched { result: Ok(_), .. })
                });
                assert_eq!(
                    gpu.pages.lock().unwrap()[&1000],
                    vec![7; layout.page_record_bytes as usize]
                );
                assert_eq!(
                    gpu.rows.lock().unwrap()[&9],
                    vec![9; layout.row_record_bytes as usize]
                );
            }
            assert_eq!(pool.free_slots(), pool.slot_count());
            store.close().unwrap();
            drop(store);
            fs::remove_dir_all(directory).unwrap();
        }
    }

    #[test]
    fn compact_record_row_thins_recovers_and_refills() {
        let Some(directory) = test_directory("replicated", "thin-recover-refill") else {
            return;
        };
        let geometry = vec![rank(PAGE_TOKENS, &[("kv", 4096)], 12288); 4];
        let layout = Layout::new(&geometry).unwrap();
        let intake = crate::offload::intake_slots(&layout, 1);
        let pool = SlabPool::new(layout.page_record_bytes, intake, intake).unwrap();
        let (copier, gpu) = crate::testing::SimulatedGpu::shared_mapped(&geometry).unwrap();
        let mut store = DiskStore::new(
            record::DirectoryLock::acquire(&directory).unwrap(),
            IDENTITY,
            FINGERPRINT,
            1 << 30,
            geometry.clone(),
            Vec::new(),
            Box::new(Staged::new(
                Some(pool.clone()),
                copier,
                DiskThread::start("thin-compact").unwrap(),
            )),
        )
        .unwrap();
        let h = history(3 * PAGE_TOKENS + 3);
        let mut ids = Ids::new();
        let mut checkpoints = Vec::new();
        for i in 0..3 {
            gpu.pages
                .lock()
                .unwrap()
                .insert(i, vec![i as u8 + 7; layout.page_record_bytes as usize]);
            gpu.rows
                .lock()
                .unwrap()
                .insert(i, vec![i as u8 + 19; layout.row_record_bytes as usize]);
            let parent = checkpoints
                .last()
                .map(|c: &PublishedCheckpoint| c.key.consumed_tokens);
            let c = checkpoint(
                &mut ids,
                &h,
                (i as u64 + 1) * PAGE_TOKENS + 3,
                parent,
                pages(0..i + 1),
                1,
                StateSlotId(i),
            );
            assert_eq!(store.store(&c).unwrap(), StoreOutcome::Queued);
            assert!(matches!(
                &poll_until(
                    &mut store,
                    Instant::now() + Duration::from_secs(10),
                    |e| matches!(e, StoreEvent::Stored { .. })
                )[..],
                [StoreEvent::Stored { result: Ok(_), .. }]
            ));
            checkpoints.push(c);
        }
        store.set_limit(store.used());
        let middle = &checkpoints[1];
        let deadline = Instant::now() + Duration::from_secs(10);
        while fs::metadata(&store.record(&middle.key).unwrap().path)
            .unwrap()
            .len()
            != store.record(&middle.key).unwrap().header.row_segment
        {
            assert!(Instant::now() < deadline);
            store.poll(&mut Vec::new()).unwrap();
        }
        assert_eq!(store.stats().rows_thinned, [0, 1]);
        assert_eq!(store.stats().packed_records, [0, 3]);
        store.close().unwrap();
        drop(store);
        let guard = crate::testing::directory_lock(&directory).unwrap();
        let scanned = record::scan(
            &guard,
            &IDENTITY,
            &FINGERPRINT,
            PAGE_TOKENS,
            &record::rank_sizes(&geometry),
        )
        .unwrap();
        assert_eq!(scanned.len(), 3);
        assert!(scanned.iter().all(|r| r.header.replicated()));
        assert!(
            !scanned
                .iter()
                .find(|r| r.header.key == middle.key)
                .unwrap()
                .row
        );
        let (copier, gpu) = crate::testing::SimulatedGpu::shared_mapped(&geometry).unwrap();
        let mut store = DiskStore::new(
            guard,
            IDENTITY,
            FINGERPRINT,
            1 << 30,
            geometry,
            scanned,
            Box::new(Staged::new(
                Some(pool),
                copier,
                DiskThread::start("refill-compact").unwrap(),
            )),
        )
        .unwrap();
        assert_eq!(
            store
                .lookup(&h, 0, middle.key.consumed_tokens + 1)
                .unwrap()
                .key,
            checkpoints[0].key
        );
        gpu.pages
            .lock()
            .unwrap()
            .insert(1, vec![8; layout.page_record_bytes as usize]);
        gpu.rows
            .lock()
            .unwrap()
            .insert(1, vec![20; layout.row_record_bytes as usize]);
        assert_eq!(store.store(middle).unwrap(), StoreOutcome::Queued);
        assert!(matches!(
            &poll_until(
                &mut store,
                Instant::now() + Duration::from_secs(10),
                |e| matches!(e, StoreEvent::Stored { .. })
            )[..],
            [StoreEvent::Stored { result: Ok(_), .. }]
        ));
        let record = store.record(&middle.key).unwrap();
        assert!(
            record
                .header
                .row_present(fs::metadata(&record.path).unwrap().len())
        );
        assert_eq!(store.stats().row_refills, 1);
        assert_eq!(store.stats().packed_records, [0, 1]);
        let hit = store
            .lookup(&h, PAGE_TOKENS, middle.key.consumed_tokens + 1)
            .unwrap();
        assert_eq!(hit.key, middle.key);
        store
            .fetch(
                hit,
                InboundTicket {
                    key: middle.key,
                    pages: pages(100..101),
                    slot: StateSlotId(9),
                },
            )
            .unwrap();
        poll_until(&mut store, deadline, |e| {
            matches!(e, StoreEvent::Fetched { result: Ok(_), .. })
        });
        assert_eq!(
            gpu.pages.lock().unwrap()[&100],
            vec![8; layout.page_record_bytes as usize]
        );
        assert_eq!(
            gpu.rows.lock().unwrap()[&9],
            vec![20; layout.row_record_bytes as usize]
        );
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "codec")]
    #[test]
    fn packed_disk_round_trip_restart_mixed_profiles_and_compaction() {
        use crate::shared::codec::Plan;
        let Some(directory) = test_directory("disk", "packed-roundtrip") else {
            return;
        };

        let geometry = geometry();
        let layout = Layout::new(&geometry).unwrap();
        let gpu = crate::testing::GpuMemory::default();
        let profile_a = [0xAA; 32];
        let profile_b = [0xBB; 32];
        let open_store = |profile: [u8; 32], receive: Vec<[u8; 32]>| -> DiskStore {
            let guard = crate::testing::directory_lock(&directory).unwrap();
            let scanned = record::scan(
                &guard,
                &IDENTITY,
                &FINGERPRINT,
                PAGE_TOKENS,
                &record::rank_sizes(&geometry),
            )
            .unwrap();
            let pool = SlabPool::new(layout.page_record_bytes, 16, 0).unwrap();
            let copier = crate::testing::CodecCopier::new(
                layout.clone(),
                profile,
                Arc::new(Plan::new(layout.page_record_bytes as usize, &[]).unwrap()),
                Arc::new(Plan::new(layout.row_record_bytes as usize, &[]).unwrap()),
                gpu.clone(),
            )
            .with_receive(receive)
            .shared();
            DiskStore::new(
                guard,
                IDENTITY,
                FINGERPRINT,
                1 << 30,
                geometry.clone(),
                scanned,
                Box::new(Staged::new(
                    Some(pool),
                    copier,
                    DiskThread::start("packed-disk-test").unwrap(),
                )),
            )
            .unwrap()
        };

        let mut ids = Ids::new();
        let h = history(10 * PAGE_TOKENS + 5);
        let expected_pages: Vec<Vec<u8>> = (0..10u32)
            .map(|p| {
                (0..layout.page_record_bytes as usize)
                    .map(|j| ((j / 32 + p as usize) % 5) as u8)
                    .collect()
            })
            .collect();
        let expected_rows: Vec<Vec<u8>> = (0..3u32)
            .map(|r| {
                (0..layout.row_record_bytes as usize)
                    .map(|j| ((j / 16 + r as usize) % 4) as u8)
                    .collect()
            })
            .collect();
        for (i, p) in expected_pages.iter().enumerate() {
            gpu.pages.lock().unwrap().insert(i as u32, p.clone());
        }
        for (i, r) in expected_rows.iter().enumerate() {
            gpu.rows.lock().unwrap().insert(i as u32, r.clone());
        }
        let c1 = checkpoint(&mut ids, &h, 6 * PAGE_TOKENS + 1, None, pages(0..6), 6, StateSlotId(0));
        let c2 = checkpoint(
            &mut ids,
            &h,
            8 * PAGE_TOKENS + 2,
            Some(c1.key.consumed_tokens),
            pages(0..8),
            2,
            StateSlotId(1),
        );
        let c3 = checkpoint(
            &mut ids,
            &h,
            10 * PAGE_TOKENS + 3,
            Some(c2.key.consumed_tokens),
            pages(0..10),
            2,
            StateSlotId(2),
        );

        let deadline = || Instant::now() + Duration::from_secs(10);

        // Phase 1: Write c1 + c2 with profile A as v4 packed files.
        let mut store = open_store(profile_a, vec![profile_a, profile_b]);
        for c in [&c1, &c2] {
            assert_eq!(store.store(c).unwrap(), StoreOutcome::Queued);
            assert!(matches!(
                &poll_until(&mut store, deadline(), |e| matches!(e, StoreEvent::Stored { .. }))[..],
                [StoreEvent::Stored { result: Ok(_), .. }]
            ));
        }
        let r1 = store.record(&c1.key).unwrap();
        assert!(r1.header.packed.is_some());
        let raw_len = header(&geometry, &h, c1.key.consumed_tokens, None).length(true);
        assert!(
            r1.bytes < raw_len / 2,
            "expected compressed v4 file well below raw length ({} vs {raw_len})",
            r1.bytes
        );
        assert_eq!(fs::metadata(&r1.path).unwrap().len(), r1.bytes);
        store.close().unwrap();
        drop(store);

        // Step 2: Restart with write profile B + receive [A, B], restore c2, write c3 with B,
        // compact c2 (row truncation), and verify mixed-profile + compacted chain across restart.
        let mut store = open_store(profile_b, vec![profile_a, profile_b]);
        assert_eq!(store.index().len(), 2);
        let hit = store.lookup(&h, 0, u64::MAX).unwrap();
        assert_eq!(hit.key, c2.key);
        store
            .fetch(
                hit,
                InboundTicket {
                    key: c2.key,
                    pages: pages(100..108),
                    slot: StateSlotId(10),
                },
            )
            .unwrap();
        poll_until(&mut store, deadline(), |e| {
            matches!(e, StoreEvent::Fetched { result: Ok(_), .. })
        });
        for i in 0..8usize {
            assert_eq!(gpu.pages.lock().unwrap()[&(100 + i as u32)], expected_pages[i]);
        }
        assert_eq!(gpu.rows.lock().unwrap()[&10], expected_rows[1]);

        assert_eq!(store.store(&c3).unwrap(), StoreOutcome::Queued);
        poll_until(&mut store, deadline(), |e| {
            matches!(e, StoreEvent::Stored { result: Ok(_), .. })
        });
        assert!(store.record(&c3.key).unwrap().header.packed.is_some());
        store.compact(&c2.key).unwrap();
        store.index_mut().drop_row(&c2.key);
        let t_deadline = deadline();
        while fs::metadata(&store.record(&c2.key).unwrap().path)
            .unwrap()
            .len()
            != store.record(&c2.key).unwrap().header.row_segment
        {
            assert!(Instant::now() < t_deadline);
            store.poll(&mut Vec::new()).unwrap();
        }
        store.close().unwrap();
        drop(store);

        // Phase 3: Restart again with profile A + receive [A, B] and restore c3 across
        // profile A parent (c1) + compacted profile A middle (c2) + profile B leaf (c3).
        let mut store = open_store(profile_a, vec![profile_a, profile_b]);
        assert_eq!(store.index().len(), 3);
        assert!(!store.index().node(&c2.key).unwrap().row);
        let hit = store.lookup(&h, 0, u64::MAX).unwrap();
        assert_eq!(hit.key, c3.key);
        store
            .fetch(
                hit,
                InboundTicket {
                    key: c3.key,
                    pages: pages(200..210),
                    slot: StateSlotId(20),
                },
            )
            .unwrap();
        poll_until(&mut store, deadline(), |e| {
            matches!(e, StoreEvent::Fetched { result: Ok(_), .. })
        });
        for i in 0..10usize {
            assert_eq!(gpu.pages.lock().unwrap()[&(200 + i as u32)], expected_pages[i]);
        }
        assert_eq!(gpu.rows.lock().unwrap()[&20], expected_rows[2]);
        store.close().unwrap();
        drop(store);

        // Phase 4: Restart with compression OFF (no receive profiles) prunes unreadable v4 files.
        let guard = crate::testing::directory_lock(&directory).unwrap();
        let scanned = record::scan(
            &guard,
            &IDENTITY,
            &FINGERPRINT,
            PAGE_TOKENS,
            &record::rank_sizes(&geometry),
        )
        .unwrap();
        assert_eq!(scanned.len(), 3);
        let pool = SlabPool::new(layout.page_record_bytes, 16, 0).unwrap();
        let (raw_copier, _) = crate::testing::SimulatedGpu::shared_mapped(&geometry).unwrap();
        let mut store = DiskStore::new(
            guard,
            IDENTITY,
            FINGERPRINT,
            1 << 30,
            geometry,
            scanned,
            Box::new(Staged::new(
                Some(pool),
                raw_copier,
                DiskThread::start("packed-disk-off").unwrap(),
            )),
        )
        .unwrap();
        assert_eq!(store.index().len(), 0);
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn all_literal_disk_conversion_rejects_corrupted_literal_bytes() {
        use crate::shared::{
            packed::Checksum,
            wire::{
                PackedUnit,
                packed_unit::{Kind, Representation},
            },
        };
        let geometry = vec![rank(PAGE_TOKENS, &[("kv", 4096)], 4096)];
        let key = PrefixKey {
            digest: [7u8; 16],
            consumed_tokens: PAGE_TOKENS,
        };
        let header = record::Header::new(
            &geometry,
            IDENTITY,
            FINGERPRINT,
            key,
            None,
            [0u8; 16],
            1,
            SystemTime::now(),
        )
        .unwrap();
        let page_bytes = vec![0x3Cu8; 4096];
        let row_bytes = vec![0xA5u8; 4096];
        let make_slots = |p: Vec<u8>, r: Vec<u8>| {
            vec![
                crate::codec_copy::CaptureSlot {
                    unit: PackedUnit {
                        kind: Kind::Page as i32,
                        logical_index: 0,
                        stored_rank: 0,
                        representation: Representation::Literal as i32,
                        original_length: 4096,
                        stored_length: 4096,
                        raw_xxh3: record::checksum(&page_bytes),
                        ..Default::default()
                    },
                    bytes: p,
                },
                crate::codec_copy::CaptureSlot {
                    unit: PackedUnit {
                        kind: Kind::Row as i32,
                        logical_index: 0,
                        stored_rank: 0,
                        representation: Representation::Literal as i32,
                        original_length: 4096,
                        stored_length: 4096,
                        raw_xxh3: record::checksum(&row_bytes),
                        ..Default::default()
                    },
                    bytes: r,
                },
            ]
        };
        // Intact literal slots convert cleanly to canonical raw v3.
        let ok = build_packed_write(
            JobId(1),
            PathBuf::from("/tmp/ok.tmp"),
            PathBuf::from("/tmp/ok.rec"),
            header.clone(),
            make_slots(page_bytes.clone(), row_bytes.clone()),
            true,
        );
        assert!(ok.is_ok(), "intact all-literal record must succeed: {ok:?}");

        // Flipping a byte in a literal page must fail with Checksum::Stored.
        let mut bad_p = page_bytes.clone();
        bad_p[123] ^= 0x01;
        let err = build_packed_write(
            JobId(2),
            PathBuf::from("/tmp/bad-page.tmp"),
            PathBuf::from("/tmp/bad-page.rec"),
            header.clone(),
            make_slots(bad_p, row_bytes.clone()),
            true,
        )
        .unwrap_err();
        assert_eq!(
            err.carried().and_then(|e| e.downcast_ref::<Checksum>()),
            Some(&Checksum::Stored)
        );

        // Flipping a byte in a literal row must also fail with Checksum::Stored.
        let mut bad_r = row_bytes.clone();
        bad_r[77] ^= 0x80;
        let err = build_packed_write(
            JobId(3),
            PathBuf::from("/tmp/bad-row.tmp"),
            PathBuf::from("/tmp/bad-row.rec"),
            header,
            make_slots(page_bytes.clone(), bad_r),
            true,
        )
        .unwrap_err();
        assert_eq!(
            err.carried().and_then(|e| e.downcast_ref::<Checksum>()),
            Some(&Checksum::Stored)
        );
    }

    #[test]
    fn packed_transfer_error_preserves_lookup_row_for_transient_errors_only() {
        let transient = Error::new("CUDA stream upload error");
        let io_other = Error::carrying(std::io::Error::new(
            std::io::ErrorKind::Other,
            "device busy",
        ));
        let stored_bad = Error::carrying(crate::shared::packed::Checksum::Stored);
        let orig_bad = Error::carrying(crate::shared::packed::Checksum::Original);
        let corrupt_pkt = Error::carrying(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "truncated GKV1 packet",
        ));
        assert!(!is_packed_content_error(&transient));
        assert!(!is_packed_content_error(&io_other));
        assert!(is_packed_content_error(&stored_bad));
        assert!(is_packed_content_error(&orig_bad));
        assert!(is_packed_content_error(&corrupt_pkt));
    }

    /// Wraps a `PageCopier` so tests can hold `Direction::In` replies or fail a
    /// chosen job id without affecting sibling coalesced followers.
    struct GatedCopier {
        inner: SharedCopier,
        layout: Layout,
        hold_in: Arc<Mutex<bool>>,
        held_in: Arc<Mutex<VecDeque<copier::Reply>>>,
        fail_job: Arc<Mutex<Option<JobId>>>,
        reply_error: Arc<Mutex<Option<(JobId, Error)>>>,
    }

    struct GatedCopierHandles {
        copier: SharedCopier,
        hold_in: Arc<Mutex<bool>>,
        held_in: Arc<Mutex<VecDeque<copier::Reply>>>,
        fail_job: Arc<Mutex<Option<JobId>>>,
        reply_error: Arc<Mutex<Option<(JobId, Error)>>>,
    }

    impl GatedCopier {
        fn wrap(inner: SharedCopier, layout: Layout, hold_in: bool) -> GatedCopierHandles {
            let hold = Arc::new(Mutex::new(hold_in));
            let held = Arc::new(Mutex::new(VecDeque::new()));
            let fail = Arc::new(Mutex::new(None));
            let reply_err = Arc::new(Mutex::new(None));
            let copier = Arc::new(Mutex::new(Self {
                inner,
                layout,
                hold_in: hold.clone(),
                held_in: held.clone(),
                fail_job: fail.clone(),
                reply_error: reply_err.clone(),
            }));
            GatedCopierHandles {
                copier,
                hold_in: hold,
                held_in: held,
                fail_job: fail,
                reply_error: reply_err,
            }
        }
    }

    impl PageCopier for GatedCopier {
        fn layout(&self) -> &Layout {
            &self.layout
        }

        fn decode_counters(&self) -> Option<crate::codec_copy::DecodeCounters> {
            lock(&self.inner).decode_counters()
        }

        fn codec_profile(&self) -> Option<[u8; 32]> {
            lock(&self.inner).codec_profile()
        }

        fn codec_receive_profiles(&self) -> Vec<[u8; 32]> {
            lock(&self.inner).codec_receive_profiles()
        }

        #[cfg(feature = "codec")]
        fn async_host_codec(&self) -> Option<crate::codec_copy::HostEncoder> {
            lock(&self.inner).async_host_codec()
        }

        fn submit(&mut self, job: copier::Job) -> Result<()> {
            if job.direction == Direction::In
                && *self.fail_job.lock().unwrap() == Some(job.id)
            {
                return Err(invalid("injected private transfer failure"));
            }
            lock(&self.inner).submit(job)
        }

        fn poll(&mut self, replies: &mut Vec<copier::Reply>) {
            let mut incoming = Vec::new();
            lock(&self.inner).poll(&mut incoming);
            let hold = *self.hold_in.lock().unwrap();
            let mut held = self.held_in.lock().unwrap();
            let mut injected = self.reply_error.lock().unwrap();
            for mut reply in incoming {
                if reply.direction == Direction::In
                    && injected.as_ref().is_some_and(|(id, _)| *id == reply.id)
                {
                    let (_, err) = injected.take().unwrap();
                    reply.result = Err(err);
                }
                if hold && reply.direction == Direction::In {
                    held.push_back(reply);
                } else {
                    replies.push(reply);
                }
            }
            if !hold {
                while let Some(reply) = held.pop_front() {
                    replies.push(reply);
                }
            }
        }

        fn in_flight(&self) -> usize {
            lock(&self.inner).in_flight() + self.held_in.lock().unwrap().len()
        }

        fn close(&mut self) -> Result<()> {
            *self.hold_in.lock().unwrap() = false;
            self.held_in.lock().unwrap().clear();
            lock(&self.inner).close()
        }
    }

    type ChildBranch = (PrefixHistory, Header, Vec<Vec<u8>>, Vec<u8>);

    struct AncestorBranchFixture {
        parent_header: Header,
        parent_pages: Vec<Vec<u8>>,
        children: Vec<ChildBranch>,
    }

    /// Write one multi-window parent record and `branch_count` distinct child
    /// records under it so each child shares the parent's full page chain.
    fn write_shared_ancestor_tree(
        directory: &Path,
        geometry: &[CacheGeometry],
        parent_pages_count: usize,
        child_pages_count: usize,
        branch_count: usize,
    ) -> AncestorBranchFixture {
        let parent_tokens = parent_pages_count as u64 * PAGE_TOKENS;
        let base_history = history(parent_tokens);
        let mut parent_header = header(geometry, &base_history, parent_tokens, None);
        let (parent_pages, _) = write_record(directory, &mut parent_header, 11);
        let mut children = Vec::with_capacity(branch_count);
        for branch in 0..branch_count {
            let mut child_history = base_history.clone();
            let extra_tokens: Vec<u32> = (0..(child_pages_count as u32 * PAGE_TOKENS as u32 + 5))
                .map(|i| i.wrapping_mul(17) ^ (branch as u32 + 1).wrapping_mul(0x9e37_79b9))
                .collect();
            child_history.append(&extra_tokens, &[]).unwrap();
            let child_consumed = child_history.token_count();
            let mut child_header = header(
                geometry,
                &child_history,
                child_consumed,
                Some(parent_tokens),
            );
            let (child_pages, child_row) =
                write_record(directory, &mut child_header, 30 + branch as u8);
            children.push((child_history, child_header, child_pages, child_row));
        }
        AncestorBranchFixture {
            parent_header,
            parent_pages,
            children,
        }
    }

    fn poll_all_fetches(
        store: &mut DiskStore,
        expected: usize,
        deadline: Instant,
    ) -> Vec<StoreEvent> {
        let mut events = Vec::new();
        while events.len() < expected {
            assert!(
                Instant::now() < deadline,
                "received {} of {expected} fetch events: {events:?}",
                events.len()
            );
            store.poll(&mut events).unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(events.len(), expected);
        events
    }

    #[test]
    fn aligned_windows_align_to_window_boundaries_for_device_coalescing() {
        let from_zero = plan_windows([(0u32, 130usize, false)], true);
        let from_ten = plan_windows([(10u32, 120usize, false)], true);
        let summarize = |plans: &[WindowPlan]| -> Vec<_> {
            plans
                .iter()
                .map(|p| (p.segment, p.index, p.first, p.count, p.offset, p.row))
                .collect()
        };
        assert_eq!(
            summarize(&from_zero),
            [
                (0, 0, 0, 64, 0, false),
                (0, 1, 64, 64, 64, false),
                (0, 2, 128, 2, 128, false),
            ]
        );
        assert_eq!(
            summarize(&from_ten),
            [
                (0, 0, 10, 54, 0, false),
                (0, 1, 64, 64, 54, false),
                (0, 2, 128, 2, 118, false),
            ]
        );
        // Suffix windows from boundary 64 onward share exact `(first, count, row)`.
        assert_eq!(
            (from_zero[1].first, from_zero[1].count, from_zero[1].row),
            (from_ten[1].first, from_ten[1].count, from_ten[1].row)
        );
        assert_eq!(
            (from_zero[2].first, from_zero[2].count, from_zero[2].row),
            (from_ten[2].first, from_ten[2].count, from_ten[2].row)
        );
        assert_eq!(
            summarize(&plan_windows(
                [(5, 3, false), (0, 0, false), (10, 120, true)],
                true,
            )),
            [
                (0, 0, 5, 3, 0, false),
                (2, 1, 10, 54, 3, false),
                (2, 2, 64, 64, 57, false),
                (2, 3, 128, 2, 121, true),
            ]
        );
        assert_eq!(
            summarize(&plan_windows([(0, 0, true)], true)),
            [(0, 0, 0, 0, 0, true)]
        );
        assert_eq!(
            summarize(&plan_windows([(64, 64, true)], true)),
            [(0, 0, 64, 64, 0, true)]
        );
        assert!(plan_windows([], true).is_empty());
        assert!(plan_windows([(0, 0, false)], true).is_empty());
    }

    #[test]
    fn concurrent_burst_coalesces_shared_ancestor_opens_and_window_reads() {
        use crate::testing::SimulatedGpu;
        let Some(directory) = test_directory("disk-store", "coalesce-burst") else {
            return;
        };
        const PARENT_PAGES: usize = 3 * WINDOW_PAGES + 5; // 4 aligned windows
        const CHILD_PAGES: usize = 3; // 1 window each
        const BRANCHES: usize = 6;
        let geometry = geometry();
        let layout = Layout::new(&geometry).unwrap();
        let fixture =
            write_shared_ancestor_tree(&directory, &geometry, PARENT_PAGES, CHILD_PAGES, BRANCHES);
        let mut records = vec![scanned(&directory, fixture.parent_header.clone(), 0, true)];
        for (i, (_, child_header, _, _)) in fixture.children.iter().enumerate() {
            records.push(scanned(&directory, child_header.clone(), i as u64 + 1, true));
        }
        // Size the pool to minimal intake (two windows), proving six concurrent
        // restores share staging slots on the common ancestor without inflating
        // staging capacity.
        let intake = crate::offload::intake_slots(&layout, 1);
        let pool = SlabPool::new(layout.page_record_bytes, intake, intake).unwrap();
        let (copier, gpu) = SimulatedGpu::shared_mapped(&geometry).unwrap();
        let mut store = DiskStore::new(
            record::DirectoryLock::acquire(&directory).unwrap(),
            IDENTITY,
            FINGERPRINT,
            1 << 30,
            geometry,
            records,
            Box::new(Staged::new(
                Some(pool.clone()),
                copier,
                DiskThread::start("coalesce-burst").unwrap(),
            )),
        )
        .unwrap();

        let total_pages = (PARENT_PAGES + CHILD_PAGES) as u32;
        let mut tickets = Vec::with_capacity(BRANCHES);
        for (branch, (child_history, child_header, _, _)) in fixture.children.iter().enumerate() {
            let hit = store.lookup(child_history, 0, u64::MAX).unwrap();
            assert_eq!(hit.key, child_header.key);
            assert_eq!(hit.pages, total_pages as usize);
            let base = (branch as u32 + 1) * 1000;
            let ticket = InboundTicket {
                key: child_header.key,
                pages: pages(base..base + total_pages),
                slot: StateSlotId(branch as u32 + 1),
            };
            store.fetch(hit, ticket.clone()).unwrap();
            tickets.push(ticket);
        }

        let events =
            poll_all_fetches(&mut store, BRANCHES, Instant::now() + Duration::from_secs(30));
        let expected_bytes = (PARENT_PAGES + CHILD_PAGES) as u64 * PAGE_RECORD + ROW_RECORD;
        for event in &events {
            match event {
                StoreEvent::Fetched {
                    result: Ok(bytes), ..
                } => {
                    assert_eq!(*bytes, expected_bytes);
                }
                other => panic!("unexpected fetch event: {other:?}"),
            }
        }

        let staged = store.transport_mut().staged().unwrap();
        assert!(settled(staged));
        // Parent is opened once and each of its four windows is read once;
        // each of the six distinct child files is opened and read once.
        assert_eq!(staged.disk_opens(), 1 + BRANCHES as u64);
        assert_eq!(staged.disk_window_reads(), 4 + BRANCHES as u64);
        assert_eq!(staged.coalesced_windows(), 4 * (BRANCHES as u64 - 1));
        assert_eq!(pool.free_slots(), pool.slot_count());
        assert!(store.index().nodes().all(|node| !node.pinned()));

        let gpu_pages = gpu.pages.lock().unwrap();
        let gpu_rows = gpu.rows.lock().unwrap();
        for (branch, (_, _, child_pages, child_row)) in fixture.children.iter().enumerate() {
            let ticket = &tickets[branch];
            for (index, expected) in fixture.parent_pages.iter().enumerate() {
                assert_eq!(&gpu_pages[&ticket.pages[index].0], expected);
            }
            for (index, expected) in child_pages.iter().enumerate() {
                assert_eq!(&gpu_pages[&ticket.pages[PARENT_PAGES + index].0], expected);
            }
            assert_eq!(&gpu_rows[&ticket.slot.0], child_row);
        }

        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn staggered_arrival_attaches_out_of_order_to_in_flight_windows() {
        use crate::testing::SimulatedGpu;
        let Some(directory) = test_directory("disk-store", "coalesce-staggered") else {
            return;
        };
        const PARENT_PAGES: usize = 4 * WINDOW_PAGES; // 4 aligned windows: W0, W1, W2, W3
        const CHILD_PAGES: usize = 4;
        let geometry = geometry();
        let layout = Layout::new(&geometry).unwrap();
        let fixture =
            write_shared_ancestor_tree(&directory, &geometry, PARENT_PAGES, CHILD_PAGES, 2);
        let records = vec![
            scanned(&directory, fixture.parent_header.clone(), 0, true),
            scanned(&directory, fixture.children[0].1.clone(), 1, true),
            scanned(&directory, fixture.children[1].1.clone(), 2, true),
        ];
        let intake = crate::offload::intake_slots(&layout, 1);
        let pool = SlabPool::new(layout.page_record_bytes, intake, intake).unwrap();
        let (inner, gpu) = SimulatedGpu::shared_mapped(&geometry).unwrap();
        let gates = GatedCopier::wrap(inner, layout.clone(), true);
        let mut store = DiskStore::new(
            record::DirectoryLock::acquire(&directory).unwrap(),
            IDENTITY,
            FINGERPRINT,
            1 << 30,
            geometry,
            records,
            Box::new(Staged::new(
                Some(pool.clone()),
                gates.copier,
                DiskThread::start("coalesce-staggered").unwrap(),
            )),
        )
        .unwrap();

        let total_pages = (PARENT_PAGES + CHILD_PAGES) as u32;
        let hit_a = store.lookup(&fixture.children[0].0, 0, u64::MAX).unwrap();
        let ticket_a = InboundTicket {
            key: hit_a.key,
            pages: pages(1000..1000 + total_pages),
            slot: StateSlotId(1),
        };
        store.fetch(hit_a, ticket_a.clone()).unwrap();

        // Advance request 1 until W0 and W1 have both completed disk reads and
        // are held in the copier queue.
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut events = Vec::new();
        while gates.held_in.lock().unwrap().len() < 2 {
            assert!(Instant::now() < deadline);
            store.poll(&mut events).unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
        // Release W0 so W0 completes and is swept while W1 remains active and
        // W2 enters flight.
        {
            let w0 = gates.held_in.lock().unwrap().pop_front().unwrap();
            assert_eq!(w0.window, 0);
            store
                .transport_mut()
                .staged_mut()
                .unwrap()
                .apply_device_transfer(w0);
        }
        // Pump until W2's disk read lands and its upload is held alongside W1.
        while gates.held_in.lock().unwrap().len() < 2 {
            assert!(Instant::now() < deadline);
            store.poll(&mut events).unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(events.is_empty());

        // Start request 2 midway through the parent (W0 is gone; W1 and W2 are
        // live in shared staging). Request 2 must piggyback on W1 and W2
        // immediately out of order without allocating extra staging windows.
        let hit_b = store.lookup(&fixture.children[1].0, 0, u64::MAX).unwrap();
        let ticket_b = InboundTicket {
            key: hit_b.key,
            pages: pages(2000..2000 + total_pages),
            slot: StateSlotId(2),
        };
        store.fetch(hit_b, ticket_b.clone()).unwrap();
        assert_eq!(
            store.transport_mut().staged().unwrap().coalesced_windows(),
            2
        );

        // Release the copier gate and complete both requests.
        *gates.hold_in.lock().unwrap() = false;
        let done = poll_all_fetches(&mut store, 2, deadline);
        assert!(
            done.iter()
                .all(|e| matches!(e, StoreEvent::Fetched { result: Ok(_), .. }))
        );

        let staged = store.transport_mut().staged().unwrap();
        assert!(settled(staged));
        assert_eq!(staged.disk_opens(), 3);
        assert!(staged.coalesced_windows() >= 2);
        assert_eq!(staged.disk_window_reads() + staged.coalesced_windows(), 10);
        assert_eq!(pool.free_slots(), pool.slot_count());

        let gpu_pages = gpu.pages.lock().unwrap();
        let gpu_rows = gpu.rows.lock().unwrap();
        for (ticket, (_, _, child_pages, child_row)) in
            [(&ticket_a, &fixture.children[0]), (&ticket_b, &fixture.children[1])]
        {
            for (index, expected) in fixture.parent_pages.iter().enumerate() {
                assert_eq!(&gpu_pages[&ticket.pages[index].0], expected);
            }
            for (index, expected) in child_pages.iter().enumerate() {
                assert_eq!(&gpu_pages[&ticket.pages[PARENT_PAGES + index].0], expected);
            }
            assert_eq!(&gpu_rows[&ticket.slot.0], child_row);
        }

        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn different_skip_pages_coalesce_aligned_windows_to_exact_gpu_pages() {
        use crate::testing::SimulatedGpu;
        let Some(directory) = test_directory("disk-store", "coalesce-skip-pages") else {
            return;
        };
        const PARENT_PAGES: usize = 3 * WINDOW_PAGES + 7; // 199 pages: [0..64), [64..128), [128..192), [192..199)
        const CHILD_PAGES: usize = 5;
        const SKIP_PAGES: usize = 10;
        let geometry = geometry();
        let layout = Layout::new(&geometry).unwrap();
        let fixture =
            write_shared_ancestor_tree(&directory, &geometry, PARENT_PAGES, CHILD_PAGES, 2);
        let records = vec![
            scanned(&directory, fixture.parent_header.clone(), 0, true),
            scanned(&directory, fixture.children[0].1.clone(), 1, true),
            scanned(&directory, fixture.children[1].1.clone(), 2, true),
        ];
        let intake = crate::offload::intake_slots(&layout, 2);
        let pool = SlabPool::new(layout.page_record_bytes, intake, intake).unwrap();
        let (inner, gpu) = SimulatedGpu::shared_mapped(&geometry).unwrap();
        let gates = GatedCopier::wrap(inner, layout.clone(), true);
        let mut store = DiskStore::new(
            record::DirectoryLock::acquire(&directory).unwrap(),
            IDENTITY,
            FINGERPRINT,
            1 << 30,
            geometry,
            records,
            Box::new(Staged::new(
                Some(pool.clone()),
                gates.copier,
                DiskThread::start("coalesce-skip").unwrap(),
            )),
        )
        .unwrap();

        // Request 1 starts at skip_pages = 0; request 2 starts at skip_pages = 10.
        let hit_a = store.lookup(&fixture.children[0].0, 0, u64::MAX).unwrap();
        let hit_b = store
            .lookup(
                &fixture.children[1].0,
                SKIP_PAGES as u64 * PAGE_TOKENS,
                u64::MAX,
            )
            .unwrap();
        assert_eq!(hit_a.chain.skip_pages, 0);
        assert_eq!(hit_b.chain.skip_pages, SKIP_PAGES);
        assert_eq!(hit_a.pages, PARENT_PAGES + CHILD_PAGES);
        assert_eq!(hit_b.pages, PARENT_PAGES - SKIP_PAGES + CHILD_PAGES);

        let ticket_a = InboundTicket {
            key: hit_a.key,
            pages: pages(1000..1000 + hit_a.pages as u32),
            slot: StateSlotId(1),
        };
        let ticket_b = InboundTicket {
            key: hit_b.key,
            pages: pages(2000..2000 + hit_b.pages as u32),
            slot: StateSlotId(2),
        };
        store.fetch(hit_a, ticket_a.clone()).unwrap();
        store.fetch(hit_b, ticket_b.clone()).unwrap();

        let deadline = Instant::now() + Duration::from_secs(30);
        let mut events = Vec::new();
        while gates.held_in.lock().unwrap().len() < 2 {
            assert!(Instant::now() < deadline);
            store.poll(&mut events).unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
        *gates.hold_in.lock().unwrap() = false;
        let done = poll_all_fetches(&mut store, 2, deadline);
        assert!(
            done.iter()
                .all(|e| matches!(e, StoreEvent::Fetched { result: Ok(_), .. }))
        );

        let staged = store.transport_mut().staged().unwrap();
        assert!(settled(staged));
        assert_eq!(staged.disk_opens(), 3);
        assert_eq!(staged.coalesced_windows(), 3);
        assert_eq!(staged.disk_window_reads(), 7);
        assert_eq!(pool.free_slots(), pool.slot_count());

        let gpu_pages = gpu.pages.lock().unwrap();
        let gpu_rows = gpu.rows.lock().unwrap();
        for (index, expected) in fixture.parent_pages.iter().enumerate() {
            assert_eq!(&gpu_pages[&ticket_a.pages[index].0], expected);
        }
        for (index, expected) in fixture.children[0].2.iter().enumerate() {
            assert_eq!(&gpu_pages[&ticket_a.pages[PARENT_PAGES + index].0], expected);
        }
        assert_eq!(&gpu_rows[&ticket_a.slot.0], &fixture.children[0].3);

        for (index, expected) in fixture.parent_pages[SKIP_PAGES..].iter().enumerate() {
            assert_eq!(&gpu_pages[&ticket_b.pages[index].0], expected);
        }
        for (index, expected) in fixture.children[1].2.iter().enumerate() {
            assert_eq!(
                &gpu_pages[&ticket_b.pages[PARENT_PAGES - SKIP_PAGES + index].0],
                expected
            );
        }
        assert_eq!(&gpu_rows[&ticket_b.slot.0], &fixture.children[1].3);

        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn private_failure_isolation_preserves_sibling_sharing_ancestor() {
        use crate::testing::SimulatedGpu;
        for inject_transfer_failure in [false, true] {
            let Some(directory) = test_directory(
                "disk-store",
                &format!("coalesce-private-fail-{inject_transfer_failure}"),
            ) else {
                return;
            };
            const PARENT_PAGES: usize = 2 * WINDOW_PAGES + 4; // 3 aligned windows
            const CHILD_PAGES: usize = 3;
            let geometry = geometry();
            let layout = Layout::new(&geometry).unwrap();
            let fixture =
                write_shared_ancestor_tree(&directory, &geometry, PARENT_PAGES, CHILD_PAGES, 2);
            if !inject_transfer_failure {
                // Corrupt a byte in child 0's private page segment.
                let bad_child_path = directory.join(record::file_name(&fixture.children[0].1.key));
                corrupt(&bad_child_path, fixture.children[0].1.page_segment + 64);
            }
            let records = vec![
                scanned(&directory, fixture.parent_header.clone(), 0, true),
                scanned(&directory, fixture.children[0].1.clone(), 1, true),
                scanned(&directory, fixture.children[1].1.clone(), 2, true),
            ];
            let intake = crate::offload::intake_slots(&layout, 1);
            let pool = SlabPool::new(layout.page_record_bytes, intake, intake).unwrap();
            let (inner, gpu) = SimulatedGpu::shared_mapped(&geometry).unwrap();
            let gates = GatedCopier::wrap(inner, layout.clone(), false);
            if inject_transfer_failure {
                // First `DiskStore::fetch` allocates `JobId(1)`.
                *gates.fail_job.lock().unwrap() = Some(JobId(1));
            }
            let mut store = DiskStore::new(
                record::DirectoryLock::acquire(&directory).unwrap(),
                IDENTITY,
                FINGERPRINT,
                1 << 30,
                geometry,
                records,
                Box::new(Staged::new(
                    Some(pool.clone()),
                    gates.copier,
                    DiskThread::start("coalesce-private-fail").unwrap(),
                )),
            )
            .unwrap();

            let total_pages = (PARENT_PAGES + CHILD_PAGES) as u32;
            let hit_bad = store.lookup(&fixture.children[0].0, 0, u64::MAX).unwrap();
            let hit_good = store.lookup(&fixture.children[1].0, 0, u64::MAX).unwrap();
            let ticket_bad = InboundTicket {
                key: hit_bad.key,
                pages: pages(1000..1000 + total_pages),
                slot: StateSlotId(1),
            };
            let ticket_good = InboundTicket {
                key: hit_good.key,
                pages: pages(2000..2000 + total_pages),
                slot: StateSlotId(2),
            };
            store.fetch(hit_bad, ticket_bad).unwrap();
            store.fetch(hit_good, ticket_good.clone()).unwrap();

            let events =
                poll_all_fetches(&mut store, 2, Instant::now() + Duration::from_secs(30));
            let mut saw_bad = false;
            let mut saw_good = false;
            for event in events {
                match event {
                    StoreEvent::Fetched { key, result } if key == fixture.children[0].1.key => {
                        let error = result.unwrap_err();
                        assert_eq!(error.is_record_read(), !inject_transfer_failure);
                        saw_bad = true;
                    }
                    StoreEvent::Fetched { key, result } if key == fixture.children[1].1.key => {
                        let expected_bytes =
                            (PARENT_PAGES + CHILD_PAGES) as u64 * PAGE_RECORD + ROW_RECORD;
                        assert_eq!(result.unwrap(), expected_bytes);
                        saw_good = true;
                    }
                    other => panic!("unexpected event: {other:?}"),
                }
            }
            assert!(saw_bad && saw_good);

            let staged = store.transport_mut().staged().unwrap();
            assert!(settled(staged));
            assert!(staged.coalesced_windows() >= 1);
            assert_eq!(pool.free_slots(), pool.slot_count());
            assert!(store.index().nodes().all(|node| !node.pinned()));

            let gpu_pages = gpu.pages.lock().unwrap();
            let gpu_rows = gpu.rows.lock().unwrap();
            for (index, expected) in fixture.parent_pages.iter().enumerate() {
                assert_eq!(&gpu_pages[&ticket_good.pages[index].0], expected);
            }
            for (index, expected) in fixture.children[1].2.iter().enumerate() {
                assert_eq!(
                    &gpu_pages[&ticket_good.pages[PARENT_PAGES + index].0],
                    expected
                );
            }
            assert_eq!(&gpu_rows[&ticket_good.slot.0], &fixture.children[1].3);

            store.close().unwrap();
            drop(store);
            fs::remove_dir_all(directory).unwrap();
        }
    }

    #[test]
    fn shared_ancestor_failure_propagates_to_all_followers_without_leaks() {
        use crate::testing::SimulatedGpu;
        for corrupt_header in [false, true] {
            let Some(directory) = test_directory(
                "disk-store",
                &format!("coalesce-shared-fail-{corrupt_header}"),
            ) else {
                return;
            };
            const PARENT_PAGES: usize = 2 * WINDOW_PAGES + 4;
            const CHILD_PAGES: usize = 3;
            const BRANCHES: usize = 4;
            let geometry = geometry();
            let layout = Layout::new(&geometry).unwrap();
            let fixture = write_shared_ancestor_tree(
                &directory,
                &geometry,
                PARENT_PAGES,
                CHILD_PAGES,
                BRANCHES,
            );
            let parent_path = directory.join(record::file_name(&fixture.parent_header.key));
            if corrupt_header {
                // Flip a byte inside the magic field so `disk::Job::Open` fails.
                corrupt(&parent_path, 0);
            } else {
                // Flip a byte inside page 0 so `disk::Job::ReadPages` fails checksum verification.
                corrupt(&parent_path, fixture.parent_header.page_segment + 128);
            }
            let mut records = vec![scanned(&directory, fixture.parent_header.clone(), 0, true)];
            for (i, (_, child_header, _, _)) in fixture.children.iter().enumerate() {
                records.push(scanned(&directory, child_header.clone(), i as u64 + 1, true));
            }
            let intake = crate::offload::intake_slots(&layout, 1);
            let pool = SlabPool::new(layout.page_record_bytes, intake, intake).unwrap();
            let (copier, _) = SimulatedGpu::shared_mapped(&geometry).unwrap();
            let mut store = DiskStore::new(
                record::DirectoryLock::acquire(&directory).unwrap(),
                IDENTITY,
                FINGERPRINT,
                1 << 30,
                geometry,
                records,
                Box::new(Staged::new(
                    Some(pool.clone()),
                    copier,
                    DiskThread::start("coalesce-shared-fail").unwrap(),
                )),
            )
            .unwrap();

            let total_pages = (PARENT_PAGES + CHILD_PAGES) as u32;
            for (branch, (child_history, child_header, _, _)) in fixture.children.iter().enumerate()
            {
                let hit = store.lookup(child_history, 0, u64::MAX).unwrap();
                let base = (branch as u32 + 1) * 1000;
                store
                    .fetch(
                        hit,
                        InboundTicket {
                            key: child_header.key,
                            pages: pages(base..base + total_pages),
                            slot: StateSlotId(branch as u32 + 1),
                        },
                    )
                    .unwrap();
            }

            let events =
                poll_all_fetches(&mut store, BRANCHES, Instant::now() + Duration::from_secs(30));
            for event in events {
                match event {
                    StoreEvent::Fetched {
                        result: Err(error), ..
                    } => {
                        assert!(
                            error.is_record_read(),
                            "corrupt_header={corrupt_header}: expected record_read error, got {error}"
                        );
                    }
                    other => panic!("corrupt_header={corrupt_header}: unexpected event {other:?}"),
                }
            }

            let staged = store.transport_mut().staged().unwrap();
            assert!(settled(staged));
            assert_eq!(staged.disk_opens(), 1);
            assert_eq!(pool.free_slots(), pool.slot_count());
            assert!(store.index().nodes().all(|node| !node.pinned()));

            store.close().unwrap();
            drop(store);
            fs::remove_dir_all(directory).unwrap();
        }
    }

    #[test]
    fn staged_close_drains_active_coalesced_fetches_cleanly() {
        use crate::testing::SimulatedGpu;
        let Some(directory) = test_directory("disk-store", "coalesce-close-drain") else {
            return;
        };
        const PARENT_PAGES: usize = 3 * WINDOW_PAGES; // 3 aligned windows
        const CHILD_PAGES: usize = 4;
        const BRANCHES: usize = 4;
        let geometry = geometry();
        let layout = Layout::new(&geometry).unwrap();
        let fixture =
            write_shared_ancestor_tree(&directory, &geometry, PARENT_PAGES, CHILD_PAGES, BRANCHES);
        let mut records = vec![scanned(&directory, fixture.parent_header.clone(), 0, true)];
        for (i, (_, child_header, _, _)) in fixture.children.iter().enumerate() {
            records.push(scanned(&directory, child_header.clone(), i as u64 + 1, true));
        }
        let intake = crate::offload::intake_slots(&layout, 1);
        let pool = SlabPool::new(layout.page_record_bytes, intake, intake).unwrap();
        let (copier, gpu) = SimulatedGpu::shared_mapped(&geometry).unwrap();
        let mut store = DiskStore::new(
            record::DirectoryLock::acquire(&directory).unwrap(),
            IDENTITY,
            FINGERPRINT,
            1 << 30,
            geometry,
            records,
            Box::new(Staged::new(
                Some(pool.clone()),
                copier,
                DiskThread::start("coalesce-close-drain").unwrap(),
            )),
        )
        .unwrap();

        let total_pages = (PARENT_PAGES + CHILD_PAGES) as u32;
        let mut tickets = Vec::with_capacity(BRANCHES);
        for (branch, (child_history, child_header, _, _)) in fixture.children.iter().enumerate() {
            let hit = store.lookup(child_history, 0, u64::MAX).unwrap();
            let base = (branch as u32 + 1) * 1000;
            let ticket = InboundTicket {
                key: child_header.key,
                pages: pages(base..base + total_pages),
                slot: StateSlotId(branch as u32 + 1),
            };
            store.fetch(hit, ticket.clone()).unwrap();
            tickets.push(ticket);
        }

        // Close immediately while coalesced fetches are actively in flight.
        assert_eq!(store.transport_mut().staged().unwrap().in_flight().1, BRANCHES);
        store.close().unwrap();

        let staged = store.transport_mut().staged().unwrap();
        assert!(settled(staged));
        assert_eq!(staged.disk_opens(), 1 + BRANCHES as u64);
        assert_eq!(staged.coalesced_windows(), 3 * (BRANCHES as u64 - 1));
        assert_eq!(pool.free_slots(), pool.slot_count());
        assert!(store.index().nodes().all(|node| !node.pinned()));
        assert_eq!(store.stats().fetched, BRANCHES as u64);

        let gpu_pages = gpu.pages.lock().unwrap();
        let gpu_rows = gpu.rows.lock().unwrap();
        for (branch, (_, _, child_pages, child_row)) in fixture.children.iter().enumerate() {
            let ticket = &tickets[branch];
            for (index, expected) in fixture.parent_pages.iter().enumerate() {
                assert_eq!(&gpu_pages[&ticket.pages[index].0], expected);
            }
            for (index, expected) in child_pages.iter().enumerate() {
                assert_eq!(&gpu_pages[&ticket.pages[PARENT_PAGES + index].0], expected);
            }
            assert_eq!(&gpu_rows[&ticket.slot.0], child_row);
        }

        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn misaligned_follower_attaching_to_shared_partial_tail_and_row_matches_private_restore() {
        use crate::testing::SimulatedGpu;
        let Some(directory) = test_directory("disk-store", "coalesce-misaligned-follower-row")
        else {
            return;
        };
        const PARENT_PAGES: usize = WINDOW_PAGES + 7; // 71 pages: [0..64) and partial [64..71)
        const LEAF_PAGES: usize = 19; // partial < 64 trailing chunk carrying row = true
        const SKIP_PAGES: usize = 13; // non-64-multiple resident prefix
        let geometry = geometry();
        let layout = Layout::new(&geometry).unwrap();
        let fixture =
            write_shared_ancestor_tree(&directory, &geometry, PARENT_PAGES, LEAF_PAGES, 1);
        let (ref leaf_history, ref leaf_header, ref leaf_pages, ref leaf_row) = fixture.children[0];
        let records = vec![
            scanned(&directory, fixture.parent_header.clone(), 0, true),
            scanned(&directory, leaf_header.clone(), 1, true),
        ];

        // Baseline: run restore B (`skip_pages = 13`) privately on an isolated store.
        let private_dir = directory.join("private-lock");
        fs::create_dir_all(&private_dir).unwrap();
        let intake = crate::offload::intake_slots(&layout, 2);
        let private_pool = SlabPool::new(layout.page_record_bytes, intake, intake).unwrap();
        let (private_copier, private_gpu) = SimulatedGpu::shared_mapped(&geometry).unwrap();
        let mut private_store = DiskStore::new(
            record::DirectoryLock::acquire(&private_dir).unwrap(),
            IDENTITY,
            FINGERPRINT,
            1 << 30,
            geometry.clone(),
            records.clone(),
            Box::new(Staged::new(
                Some(private_pool),
                private_copier,
                DiskThread::start("private-baseline").unwrap(),
            )),
        )
        .unwrap();
        let hit_private = private_store
            .lookup(leaf_history, SKIP_PAGES as u64 * PAGE_TOKENS, u64::MAX)
            .unwrap();
        assert_eq!(hit_private.chain.skip_pages, SKIP_PAGES);
        assert_eq!(hit_private.pages, PARENT_PAGES - SKIP_PAGES + LEAF_PAGES);
        let ticket_private = InboundTicket {
            key: hit_private.key,
            pages: pages(3000..3000 + hit_private.pages as u32),
            slot: StateSlotId(5),
        };
        private_store
            .fetch(hit_private, ticket_private.clone())
            .unwrap();
        let private_done = poll_all_fetches(
            &mut private_store,
            1,
            Instant::now() + Duration::from_secs(30),
        );
        assert!(matches!(
            private_done.as_slice(),
            [StoreEvent::Fetched { result: Ok(_), .. }]
        ));
        assert_eq!(
            private_store
                .transport_mut()
                .staged()
                .unwrap()
                .coalesced_windows(),
            0
        );

        // Coalesced run: restore A (`skip_pages = 0`) creates the shared windows first,
        // then restore B (`skip_pages = 13`) attaches as a follower to both the partial
        // ancestor tail `[64..71)` and the final partial chunk `[0..19)` carrying `row = true`.
        let pool = SlabPool::new(layout.page_record_bytes, intake, intake).unwrap();
        let (inner, gpu) = SimulatedGpu::shared_mapped(&geometry).unwrap();
        let gates = GatedCopier::wrap(inner, layout, true);
        let mut store = DiskStore::new(
            record::DirectoryLock::acquire(&directory).unwrap(),
            IDENTITY,
            FINGERPRINT,
            1 << 30,
            geometry,
            records,
            Box::new(Staged::new(
                Some(pool.clone()),
                gates.copier,
                DiskThread::start("coalesce-misaligned-follower").unwrap(),
            )),
        )
        .unwrap();

        let hit_a = store.lookup(leaf_history, 0, u64::MAX).unwrap();
        let ticket_a = InboundTicket {
            key: hit_a.key,
            pages: pages(1000..1000 + (PARENT_PAGES + LEAF_PAGES) as u32),
            slot: StateSlotId(1),
        };
        store.fetch(hit_a, ticket_a.clone()).unwrap();

        let deadline = Instant::now() + Duration::from_secs(30);
        let mut events = Vec::new();
        while gates.held_in.lock().unwrap().len() < 2 {
            assert!(Instant::now() < deadline);
            store.poll(&mut events).unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
        {
            let w0 = gates.held_in.lock().unwrap().pop_front().unwrap();
            assert_eq!(w0.window, 0);
            store
                .transport_mut()
                .staged_mut()
                .unwrap()
                .apply_device_transfer(w0);
        }
        while gates.held_in.lock().unwrap().len() < 2 {
            assert!(Instant::now() < deadline);
            store.poll(&mut events).unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(events.is_empty());

        let hit_b = store
            .lookup(leaf_history, SKIP_PAGES as u64 * PAGE_TOKENS, u64::MAX)
            .unwrap();
        assert_eq!(hit_b.chain.skip_pages, SKIP_PAGES);
        let ticket_b = InboundTicket {
            key: hit_b.key,
            pages: pages(2000..2000 + hit_b.pages as u32),
            slot: StateSlotId(2),
        };
        store.fetch(hit_b, ticket_b.clone()).unwrap();
        assert_eq!(
            store.transport_mut().staged().unwrap().coalesced_windows(),
            2
        );

        *gates.hold_in.lock().unwrap() = false;
        let done = poll_all_fetches(&mut store, 2, deadline);
        assert!(
            done.iter()
                .all(|e| matches!(e, StoreEvent::Fetched { result: Ok(_), .. }))
        );

        let staged = store.transport_mut().staged().unwrap();
        assert!(settled(staged));
        assert_eq!(staged.disk_opens(), 3);
        assert_eq!(staged.disk_window_reads(), 4);
        assert_eq!(staged.coalesced_windows(), 2);
        assert_eq!(pool.free_slots(), pool.slot_count());

        let gpu_pages = gpu.pages.lock().unwrap();
        let gpu_rows = gpu.rows.lock().unwrap();
        let baseline_pages = private_gpu.pages.lock().unwrap();
        let baseline_rows = private_gpu.rows.lock().unwrap();

        assert_eq!(ticket_b.pages.len(), ticket_private.pages.len());
        for (coalesced_id, private_id) in ticket_b.pages.iter().zip(&ticket_private.pages) {
            assert_eq!(&gpu_pages[&coalesced_id.0], &baseline_pages[&private_id.0]);
        }
        assert_eq!(
            &gpu_rows[&ticket_b.slot.0],
            &baseline_rows[&ticket_private.slot.0]
        );
        for (index, expected) in fixture.parent_pages[SKIP_PAGES..]
            .iter()
            .chain(leaf_pages.iter())
            .enumerate()
        {
            assert_eq!(&gpu_pages[&ticket_b.pages[index].0], expected);
        }
        assert_eq!(&gpu_rows[&ticket_b.slot.0], leaf_row);

        for (index, expected) in fixture
            .parent_pages
            .iter()
            .chain(leaf_pages.iter())
            .enumerate()
        {
            assert_eq!(&gpu_pages[&ticket_a.pages[index].0], expected);
        }
        assert_eq!(&gpu_rows[&ticket_a.slot.0], leaf_row);

        drop(gpu_pages);
        drop(gpu_rows);
        drop(baseline_pages);
        drop(baseline_rows);
        private_store.close().unwrap();
        store.close().unwrap();
        drop(private_store);
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn coalesced_restores_keep_inbound_rate_physical_and_bytes_in_logical() {
        use crate::testing::SimulatedGpu;
        struct ObservedTransport {
            inner: Staged,
            samples: Arc<Mutex<Vec<(u64, Duration)>>>,
        }
        impl DiskTransport for ObservedTransport {
            fn store(
                &mut self,
                id: JobId,
                header: Header,
                temporary: PathBuf,
                path: PathBuf,
                checkpoint: &PublishedCheckpoint,
                detect: bool,
            ) -> Result<()> {
                self.inner
                    .store(id, header, temporary, path, checkpoint, detect)
            }
            fn fetch(
                &mut self,
                id: JobId,
                plan: Vec<FetchSegment>,
                ticket: &InboundTicket,
            ) -> Result<()> {
                self.inner.fetch(id, plan, ticket)
            }
            fn truncate(&mut self, id: JobId, path: PathBuf, length: u64) -> Result<()> {
                self.inner.truncate(id, path, length)
            }
            fn unlink(&mut self, id: JobId, path: PathBuf) -> Result<()> {
                self.inner.unlink(id, path)
            }
            fn poll(&mut self, events: &mut Vec<TransportEvent>) -> Result<()> {
                self.inner.poll(events)
            }
            fn drain_inbound_samples(&mut self, samples: &mut Vec<(u64, Duration)>) {
                let start = samples.len();
                self.inner.drain_inbound_samples(samples);
                self.samples
                    .lock()
                    .unwrap()
                    .extend_from_slice(&samples[start..]);
            }
            fn close(&mut self) -> Result<()> {
                self.inner.close()
            }
            fn staged(&self) -> Option<&Staged> {
                Some(&self.inner)
            }
            fn staged_mut(&mut self) -> Option<&mut Staged> {
                Some(&mut self.inner)
            }
            fn codec_receive_profiles(&self) -> Vec<[u8; 32]> {
                self.inner.codec_receive_profiles()
            }
            fn decode_counters(&self) -> Option<crate::codec_copy::DecodeCounters> {
                self.inner.decode_counters()
            }
        }

        let Some(directory) = test_directory("disk-store", "coalesce-inbound-rate") else {
            return;
        };
        const PAGES: usize = 2 * WINDOW_PAGES; // 2 aligned windows
        const CONCURRENT: usize = RECORDS_IN_FLIGHT; // 8 restores of the same ancestor
        const POLL_DELAY: Duration = Duration::from_millis(20);
        let geometry = geometry();
        let layout = Layout::new(&geometry).unwrap();
        let full = history(PAGES as u64 * PAGE_TOKENS);
        let mut record_header = header(&geometry, &full, PAGES as u64 * PAGE_TOKENS, None);
        write_record(&directory, &mut record_header, 21);
        let scanned_records = vec![scanned(&directory, record_header.clone(), 0, true)];
        let expected_logical_bytes = PAGES as u64 * PAGE_RECORD + ROW_RECORD;

        let single_samples = Arc::new(Mutex::new(Vec::new()));
        let single_stats = {
            let lock_dir = directory.join("single-lock");
            fs::create_dir_all(&lock_dir).unwrap();
            let intake = crate::offload::intake_slots(&layout, 1);
            let pool = SlabPool::new(layout.page_record_bytes, intake, intake).unwrap();
            let (copier, _) = SimulatedGpu::shared_mapped(&geometry).unwrap();
            let mut store = DiskStore::new(
                record::DirectoryLock::acquire(&lock_dir).unwrap(),
                IDENTITY,
                FINGERPRINT,
                1 << 30,
                geometry.clone(),
                scanned_records.clone(),
                Box::new(ObservedTransport {
                    inner: Staged::new(
                        Some(pool),
                        copier,
                        DiskThread::start("rate-single").unwrap(),
                    ),
                    samples: Arc::clone(&single_samples),
                }),
            )
            .unwrap();
            let hit = store.lookup(&full, 0, u64::MAX).unwrap();
            store
                .fetch(
                    hit,
                    InboundTicket {
                        key: record_header.key,
                        pages: pages(100..100 + PAGES as u32),
                        slot: StateSlotId(1),
                    },
                )
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(30);
            let mut events = Vec::new();
            while events.is_empty() {
                assert!(Instant::now() < deadline);
                std::thread::sleep(POLL_DELAY);
                store.poll(&mut events).unwrap();
            }
            let stats = store.stats();
            store.close().unwrap();
            stats
        };

        assert_eq!(single_stats.fetched, 1);
        assert_eq!(single_stats.progress, 1);
        assert_eq!(single_stats.bytes_in, expected_logical_bytes);
        assert_eq!(single_samples.lock().unwrap().len(), 2);

        for staggered in [false, true] {
            let lock_dir = directory.join(format!("coalesced-lock-{staggered}"));
            fs::create_dir_all(&lock_dir).unwrap();
            let intake = crate::offload::intake_slots(&layout, 1);
            let pool = SlabPool::new(layout.page_record_bytes, intake, intake).unwrap();
            let (inner, _) = SimulatedGpu::shared_mapped(&geometry).unwrap();
            let gates = GatedCopier::wrap(inner, layout.clone(), staggered);
            let recorded_samples = Arc::new(Mutex::new(Vec::new()));
            let mut store = DiskStore::new(
                record::DirectoryLock::acquire(&lock_dir).unwrap(),
                IDENTITY,
                FINGERPRINT,
                1 << 30,
                geometry.clone(),
                scanned_records.clone(),
                Box::new(ObservedTransport {
                    inner: Staged::new(
                        Some(pool),
                        gates.copier,
                        DiskThread::start("rate-coalesced").unwrap(),
                    ),
                    samples: Arc::clone(&recorded_samples),
                }),
            )
            .unwrap();

            let deadline = Instant::now() + Duration::from_secs(30);
            let submit_count = if staggered { 1 } else { CONCURRENT };
            for i in 0..submit_count {
                let hit = store.lookup(&full, 0, u64::MAX).unwrap();
                let base = (i as u32 + 1) * 1000;
                store
                    .fetch(
                        hit,
                        InboundTicket {
                            key: record_header.key,
                            pages: pages(base..base + PAGES as u32),
                            slot: StateSlotId(i as u32 + 1),
                        },
                    )
                    .unwrap();
            }

            let mut events = Vec::new();
            let mut rate_after_physical_reads = None;
            if staggered {
                while gates.held_in.lock().unwrap().len() < 2 {
                    assert!(Instant::now() < deadline);
                    std::thread::sleep(POLL_DELAY);
                    store.poll(&mut events).unwrap();
                }
                rate_after_physical_reads = Some(store.stats().inbound_rate);
                for i in 1..CONCURRENT {
                    let hit = store.lookup(&full, 0, u64::MAX).unwrap();
                    let base = (i as u32 + 1) * 1000;
                    store
                        .fetch(
                            hit,
                            InboundTicket {
                                key: record_header.key,
                                pages: pages(base..base + PAGES as u32),
                                slot: StateSlotId(i as u32 + 1),
                            },
                        )
                        .unwrap();
                }
                *gates.hold_in.lock().unwrap() = false;
            }

            while events.len() < CONCURRENT {
                assert!(Instant::now() < deadline);
                if !staggered {
                    std::thread::sleep(POLL_DELAY);
                } else {
                    std::thread::sleep(Duration::from_millis(1));
                }
                store.poll(&mut events).unwrap();
            }

            let staged = store.transport_mut().staged().unwrap();
            assert!(settled(staged));
            assert_eq!(staged.disk_window_reads(), 2);
            assert_eq!(staged.coalesced_windows(), 2 * (CONCURRENT as u64 - 1));

            let samples = recorded_samples.lock().unwrap().clone();
            assert_eq!(
                samples.len(),
                2,
                "staggered={staggered}: expected 2 physical window rate samples, got {}",
                samples.len()
            );
            assert_eq!(
                samples.iter().map(|(bytes, _)| *bytes).sum::<u64>(),
                expected_logical_bytes
            );
            let mut expected_rate = Rate::new(INITIAL_STAGED_DISK_RATE);
            for &(bytes, elapsed) in &samples {
                expected_rate.observe(bytes, elapsed);
            }
            let expected_bps = expected_rate.get();

            let coalesced_stats = store.stats();
            assert_eq!(coalesced_stats.fetched, CONCURRENT as u64);
            assert_eq!(coalesced_stats.progress, CONCURRENT as u64);
            assert_eq!(
                coalesced_stats.bytes_in,
                CONCURRENT as u64 * single_stats.bytes_in
            );
            if let Some(before_followers) = rate_after_physical_reads {
                assert!(
                    (coalesced_stats.inbound_rate - before_followers).abs()
                        < 0.05 * before_followers,
                    "follower completions changed inbound_rate from {before_followers} to {}",
                    coalesced_stats.inbound_rate
                );
            }
            assert!(
                (coalesced_stats.inbound_rate - expected_bps).abs() <= 0.05 * expected_bps,
                "staggered={staggered}: DiskStore inbound_rate ({}) diverged from physical window median ({expected_bps})",
                coalesced_stats.inbound_rate
            );

            store.close().unwrap();
        }

        fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "codec")]
    #[test]
    fn concurrent_packed_fetches_coalesce_opens_and_read_packed_windows() {
        use crate::shared::{codec::Plan, packed::Checksum};
        use crate::testing::{CodecCopier, GpuMemory};
        let Some(directory) = test_directory("disk-store", "coalesce-packed-v4") else {
            return;
        };
        const PARENT_PAGES: usize = 2 * WINDOW_PAGES + 5; // 133 pages -> 3 aligned ReadPacked windows
        const CHILD_PAGES: usize = 3; // 1 ReadPacked window + row per branch
        const BRANCHES: usize = 4;
        let geometry = geometry();
        let layout = Layout::new(&geometry).unwrap();
        let profile = [0xCC; 32];
        let page_plan = Arc::new(Plan::new(layout.page_record_bytes as usize, &[]).unwrap());
        let row_plan = Arc::new(Plan::new(layout.row_record_bytes as usize, &[]).unwrap());

        let write_gpu = GpuMemory::default();
        let parent_pages_expected: Vec<Vec<u8>> = (0..PARENT_PAGES)
            .map(|p| {
                (0..layout.page_record_bytes as usize)
                    .map(|j| ((j / 32 + p) % 7) as u8)
                    .collect()
            })
            .collect();
        for (i, bytes) in parent_pages_expected.iter().enumerate() {
            write_gpu.pages.lock().unwrap().insert(i as u32, bytes.clone());
        }
        write_gpu
            .rows
            .lock()
            .unwrap()
            .insert(0, vec![0x51; layout.row_record_bytes as usize]);

        let mut ids = Ids::new();
        let parent_tokens = PARENT_PAGES as u64 * PAGE_TOKENS;
        let base_history = history(parent_tokens);
        let parent_cp = checkpoint(
            &mut ids,
            &base_history,
            parent_tokens,
            None,
            pages(0..PARENT_PAGES as u32),
            PARENT_PAGES,
            StateSlotId(0),
        );

        let mut branches = Vec::with_capacity(BRANCHES);
        for branch in 0..BRANCHES {
            let mut branch_history = base_history.clone();
            let extra_tokens: Vec<u32> = (0..(CHILD_PAGES as u32 * PAGE_TOKENS as u32 + 5))
                .map(|i| i.wrapping_mul(23) ^ (branch as u32 + 1).wrapping_mul(0x85eb_ca6b))
                .collect();
            branch_history.append(&extra_tokens, &[]).unwrap();
            let consumed = branch_history.token_count();
            let base_id = 500 + (branch as u32) * 10;
            let child_pages_expected: Vec<Vec<u8>> = (0..CHILD_PAGES)
                .map(|p| {
                    (0..layout.page_record_bytes as usize)
                        .map(|j| ((j / 16 + p + branch * 3) % 6) as u8)
                        .collect()
                })
                .collect();
            for (i, bytes) in child_pages_expected.iter().enumerate() {
                write_gpu
                    .pages
                    .lock()
                    .unwrap()
                    .insert(base_id + i as u32, bytes.clone());
            }
            let child_row_expected: Vec<u8> = (0..layout.row_record_bytes as usize)
                .map(|j| ((j / 16 + branch + 1) % 5) as u8)
                .collect();
            let slot_id = StateSlotId(10 + branch as u32);
            write_gpu
                .rows
                .lock()
                .unwrap()
                .insert(slot_id.0, child_row_expected.clone());
            let mut chain_ids = pages(0..PARENT_PAGES as u32);
            chain_ids.extend(pages(base_id..base_id + CHILD_PAGES as u32));
            let cp = checkpoint(
                &mut ids,
                &branch_history,
                consumed,
                Some(parent_tokens),
                chain_ids,
                CHILD_PAGES,
                slot_id,
            );
            branches.push((branch_history, cp, child_pages_expected, child_row_expected));
        }

        // Write v4 packed parent and 4 v4 packed children to disk.
        {
            let guard = crate::testing::directory_lock(&directory).unwrap();
            let pool = SlabPool::new(layout.page_record_bytes, 4, 0).unwrap();
            let copier = CodecCopier::new(
                layout.clone(),
                profile,
                page_plan.clone(),
                row_plan.clone(),
                write_gpu,
            )
            .with_receive(vec![profile])
            .shared();
            let mut writer = DiskStore::new(
                guard,
                IDENTITY,
                FINGERPRINT,
                1 << 30,
                geometry.clone(),
                Vec::new(),
                Box::new(Staged::new(
                    Some(pool),
                    copier,
                    DiskThread::start("coalesce-packed-write").unwrap(),
                )),
            )
            .unwrap();
            assert_eq!(writer.store(&parent_cp).unwrap(), StoreOutcome::Queued);
            assert!(matches!(
                &poll_until(&mut writer, Instant::now() + Duration::from_secs(30), |e| {
                    matches!(e, StoreEvent::Stored { .. })
                })[..],
                [StoreEvent::Stored { result: Ok(_), .. }]
            ));
            assert!(writer.record(&parent_cp.key).unwrap().header.packed.is_some());
            for (_, cp, _, _) in &branches {
                assert_eq!(writer.store(cp).unwrap(), StoreOutcome::Queued);
                assert!(matches!(
                    &poll_until(&mut writer, Instant::now() + Duration::from_secs(30), |e| {
                        matches!(e, StoreEvent::Stored { .. })
                    })[..],
                    [StoreEvent::Stored { result: Ok(_), .. }]
                ));
                assert!(writer.record(&cp.key).unwrap().header.packed.is_some());
            }
            writer.close().unwrap();
        }

        // Reopen with a 1-slot SlabPool (packed windows need 0 SlabPool slots) and
        // execute 4 concurrent branch fetches sharing the 3-window v4 ancestor.
        {
            let guard = crate::testing::directory_lock(&directory).unwrap();
            let scanned_records = record::scan(
                &guard,
                &IDENTITY,
                &FINGERPRINT,
                PAGE_TOKENS,
                &record::rank_sizes(&geometry),
            )
            .unwrap();
            assert_eq!(scanned_records.len(), 1 + BRANCHES);
            assert!(scanned_records.iter().all(|r| r.header.packed.is_some()));

            let pool = SlabPool::new(layout.page_record_bytes, 1, 1).unwrap();
            let read_gpu = GpuMemory::default();
            let inner = CodecCopier::new(
                layout.clone(),
                profile,
                page_plan.clone(),
                row_plan.clone(),
                read_gpu.clone(),
            )
            .with_receive(vec![profile])
            .shared();
            let mut store = DiskStore::new(
                guard,
                IDENTITY,
                FINGERPRINT,
                1 << 30,
                geometry.clone(),
                scanned_records,
                Box::new(Staged::new(
                    Some(pool.clone()),
                    inner,
                    DiskThread::start("coalesce-packed-read").unwrap(),
                )),
            )
            .unwrap();

            let total_pages = (PARENT_PAGES + CHILD_PAGES) as u32;
            let mut tickets = Vec::with_capacity(BRANCHES);
            for (branch, (branch_history, cp, _, _)) in branches.iter().enumerate() {
                let hit = store.lookup(branch_history, 0, u64::MAX).unwrap();
                assert_eq!(hit.key, cp.key);
                assert_eq!(hit.pages, total_pages as usize);
                let base = (branch as u32 + 1) * 1000;
                let ticket = InboundTicket {
                    key: cp.key,
                    pages: pages(base..base + total_pages),
                    slot: StateSlotId(branch as u32 + 1),
                };
                store.fetch(hit, ticket.clone()).unwrap();
                tickets.push(ticket);
            }

            let events =
                poll_all_fetches(&mut store, BRANCHES, Instant::now() + Duration::from_secs(30));
            for event in &events {
                assert!(
                    matches!(event, StoreEvent::Fetched { result: Ok(bytes), .. } if *bytes > 0),
                    "unexpected fetch event: {event:?}"
                );
            }

            let staged = store.transport_mut().staged().unwrap();
            assert!(settled(staged));
            assert_eq!(staged.disk_opens(), 1 + BRANCHES as u64);
            assert_eq!(staged.disk_window_reads(), 3 + BRANCHES as u64);
            assert_eq!(staged.coalesced_windows(), 3 * (BRANCHES as u64 - 1));
            assert_eq!(pool.free_slots(), pool.slot_count());
            assert!(store.index().nodes().all(|node| !node.pinned()));

            let gpu_pages = read_gpu.pages.lock().unwrap();
            let gpu_rows = read_gpu.rows.lock().unwrap();
            for (branch, (_, _, child_pages_expected, child_row_expected)) in
                branches.iter().enumerate()
            {
                let ticket = &tickets[branch];
                for (index, expected) in parent_pages_expected.iter().enumerate() {
                    assert_eq!(&gpu_pages[&ticket.pages[index].0], expected);
                }
                for (index, expected) in child_pages_expected.iter().enumerate() {
                    assert_eq!(&gpu_pages[&ticket.pages[PARENT_PAGES + index].0], expected);
                }
                assert_eq!(&gpu_rows[&ticket.slot.0], child_row_expected);
            }
            drop(gpu_pages);
            drop(gpu_rows);
            store.close().unwrap();
        }

        // Verify a packed content checksum error on a shared ReadPacked window marks
        // `is_record_read()`, fails all attached followers, and drops their lookup rows.
        {
            let guard = crate::testing::directory_lock(&directory).unwrap();
            let scanned_records = record::scan(
                &guard,
                &IDENTITY,
                &FINGERPRINT,
                PAGE_TOKENS,
                &record::rank_sizes(&geometry),
            )
            .unwrap();
            let pool = SlabPool::new(layout.page_record_bytes, 1, 1).unwrap();
            let read_gpu = GpuMemory::default();
            let inner = CodecCopier::new(
                layout.clone(),
                profile,
                page_plan,
                row_plan,
                read_gpu,
            )
            .with_receive(vec![profile])
            .shared();
            let gates = GatedCopier::wrap(inner, layout, false);
            *gates.reply_error.lock().unwrap() =
                Some((JobId(1), Error::carrying(Checksum::Stored)));
            let mut store = DiskStore::new(
                guard,
                IDENTITY,
                FINGERPRINT,
                1 << 30,
                geometry,
                scanned_records,
                Box::new(Staged::new(
                    Some(pool.clone()),
                    gates.copier,
                    DiskThread::start("coalesce-packed-corrupt").unwrap(),
                )),
            )
            .unwrap();

            let total_pages = (PARENT_PAGES + CHILD_PAGES) as u32;
            for (branch, (branch_history, cp, _, _)) in branches[..2].iter().enumerate() {
                let hit = store.lookup(branch_history, 0, u64::MAX).unwrap();
                let base = (branch as u32 + 1) * 1000;
                store
                    .fetch(
                        hit,
                        InboundTicket {
                            key: cp.key,
                            pages: pages(base..base + total_pages),
                            slot: StateSlotId(branch as u32 + 1),
                        },
                    )
                    .unwrap();
            }

            let events = poll_all_fetches(&mut store, 2, Instant::now() + Duration::from_secs(30));
            for event in events {
                match event {
                    StoreEvent::Fetched {
                        key,
                        result: Err(error),
                    } => {
                        assert!(error.is_record_read());
                        assert!(!store.index().node(&key).unwrap().row);
                    }
                    other => panic!("expected record_read error, got {other:?}"),
                }
            }
            let staged = store.transport_mut().staged().unwrap();
            assert!(settled(staged));
            assert_eq!(pool.free_slots(), pool.slot_count());
            assert!(store.index().nodes().all(|node| !node.pinned()));
            store.close().unwrap();
        }

        fs::remove_dir_all(directory).unwrap();
    }
}
