/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Budget and in-flight refusals retry the same put once. A permanent refusal
//! or an exhausted retry on an ancestor leaves one attempt to send the leaf.
//! Other failures keep the record local for a later move.
//! Manager loss unwinds waiting selections; active moves retain their pins
//! until the drained terminal event. While the manager cannot accept records,
//! the local tier's LRU is the end of the chain.

use std::{
    collections::{HashMap, VecDeque},
    time::{Duration, Instant},
};

use crate::{Error, Layout, Result, Retirement, Retiring, Tier};

use super::{
    exporter::{Checksums, HostPin, HostWindow, PutBytes},
    shared::{PutOutcome, PutRefusal, PutRequest, PutSource},
    store::PoolPort,
};

/// Records waiting for a put; selection stops at this bound.
pub(crate) const PENDING_MAX: usize = 64;
/// Puts started at once.
pub(crate) const MOVES_MAX: usize = 4;

/// Supplies a put's bytes from a pinned host chain.
struct PinBytes {
    pin: Box<dyn HostPin>,
    page_bytes: usize,
    row_bytes: usize,
    packed: bool,
}

impl PutBytes for PinBytes {
    fn min_first_page(&self) -> u64 {
        self.pin.homogeneous_suffix_first_page()
    }

    fn fill(
        &mut self,
        first_page: u64,
        page_count: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<Checksums> {
        let records = self.pin.chain().records.clone();
        let last = records
            .len()
            .checked_sub(1)
            .ok_or_else(|| Error::new("the chain to write is empty"))?;
        let end = first_page + u64::from(page_count);
        let mut sums = Checksums {
            pages: Vec::with_capacity(if self.packed { 0 } else { page_count as usize }),
            row: None,
        };
        let (mut base, mut offset) = (0u64, 0usize);
        for (index, record) in records.iter().enumerate() {
            let from = first_page.max(base);
            let to = end.min(base + u64::from(record.pages));
            let count = to.saturating_sub(from) as u32;
            let with_row = row && index == last;
            if count > 0 || with_row {
                let first = if count > 0 {
                    record.first_page + (from - base) as u32
                } else {
                    record.first_page + record.pages
                };
                let window = HostWindow {
                    record: index as u32,
                    first_page: first,
                    page_count: count,
                    row: with_row,
                };
                if self.packed {
                    let n = self.pin.read_packed(&window, &mut dst[offset..])?;
                    offset += n;
                } else {
                    let len = count as usize * self.page_bytes
                        + if with_row { self.row_bytes } else { 0 };
                    let chunk = dst
                        .get_mut(offset..offset + len)
                        .ok_or_else(|| Error::new("the window is smaller than its records"))?;
                    let read = self.pin.read(&window, chunk)?;
                    sums.pages.extend(read.pages);
                    if with_row {
                        sums.row = read.row;
                    }
                    offset += len;
                }
            }
            base += u64::from(record.pages);
        }
        if offset != dst.len() {
            return Err(Error::new("the window is larger than its records"));
        }
        Ok(sums)
    }
}

struct Move {
    record: Retiring,
    failures: u8,
    leaf_only: bool,
    mixed_prefix_queued: bool,
    path: VecDeque<crate::PrefixKey>,
    depth: u64,
    /// Shapes offered on this pass, checked before releasing a locally promoted row.
    offered: Vec<Retiring>,
    started: Instant,
    /// A predecessor's put has its own candidates; the leaf uses `record.ancestor_keys`.
    predecessor_candidates: Option<Vec<crate::PrefixKey>>,
}

/// Walk local ancestors of `start` into a parent-first retirement path, dropping any
/// leading page-only roots that cannot start a manager chain on their own.
fn ancestor_path<S: crate::store::TierStore>(
    inner: &S,
    mut parent: Option<crate::PrefixKey>,
) -> VecDeque<crate::PrefixKey> {
    let mut path = VecDeque::new();
    while let Some(local) = parent.and_then(|key| inner.retirement_predecessor(key, false)) {
        path.push_front(local.key);
        parent = local.parent;
    }
    if path
        .front()
        .and_then(|key| inner.retirement_predecessor(*key, false))
        .is_some_and(|r| r.parent.is_none() && !r.row)
    {
        while path
            .front()
            .and_then(|key| inner.retirement_predecessor(*key, false))
            .is_some_and(|r| !r.row)
        {
            path.pop_front();
        }
    }
    path
}

impl Move {
    fn candidates(&self) -> &[crate::PrefixKey] {
        self.predecessor_candidates.as_deref().unwrap_or(&self.record.ancestor_keys)
    }
}

pub(crate) struct Cascade {
    /// Records chosen by the local tier, coldest first.
    pending: VecDeque<Move>,
    /// Records whose put is running, by put id.
    moving: HashMap<u64, Move>,
    retry_after: Instant,
    refused_until: Instant,
    #[cfg(test)]
    pub(crate) now: Option<Instant>,
    /// Refusal discards by DRAM/NVMe source tier.
    pub(crate) dropped: [u64; 2],
    pub(crate) refusal_drops: [u64; 5],
    page_bytes: usize,
    row_bytes: usize,
    pub(crate) paths: u64,
    pub(crate) max_depth: u64,
    pub(crate) latency_ns: u64,
    pub(crate) max_latency_ns: u64,
    pub(crate) latency_buckets: [u64; 8],
    /// Root puts the manager stored, by [`crate::RootReason`], with their pages.
    pub(crate) root_records: [u64; crate::RootReason::COUNT],
    pub(crate) root_pages: [u64; crate::RootReason::COUNT],
    pub(crate) deeper_link_records: u64,
    pub(crate) deeper_link_pages: u64,
}

impl Cascade {
    pub(crate) fn new(layout: &Layout) -> Self {
        let now = Instant::now();
        Self {
            paths: 0,
            max_depth: 0,
            latency_ns: 0,
            max_latency_ns: 0,
            latency_buckets: [0; 8],
            root_records: [0; crate::RootReason::COUNT],
            root_pages: [0; crate::RootReason::COUNT],
            deeper_link_records: 0,
            deeper_link_pages: 0,
            pending: VecDeque::new(),
            moving: HashMap::new(),
            retry_after: now,
            refused_until: now,
            #[cfg(test)]
            now: None,
            dropped: [0; 2],
            refusal_drops: [0; 5],
            page_bytes: layout.page_record_bytes as usize,
            row_bytes: layout.row_record_bytes as usize,
        }
    }

    /// Records waiting for a put or being written.
    pub(crate) fn in_progress(&self) -> usize {
        self.pending.len() + self.moving.len()
    }

    fn now(&self) -> Instant {
        #[cfg(test)]
        if let Some(now) = self.now {
            return now;
        }
        Instant::now()
    }

    pub(crate) fn progress(&mut self) {
        self.retry_after = self.now().max(self.refused_until);
    }

    /// Start bounded puts; unavailable destinations leave records local.
    pub(crate) fn pump<S: crate::store::TierStore, P: PoolPort>(
        &mut self,
        inner: &mut S,
        pool: &mut P,
    ) {
        let up = pool.ready();
        let ready = up && self.now() >= self.retry_after;
        inner.set_spill(ready);
        for record in inner.take_retiring() {
            // A failed put's backoff holds the queue; it does not fail what waits in it.
            if up && self.pending.len() + self.moving.len() < PENDING_MAX + MOVES_MAX {
                let path = if record.tier == Tier::Dram && record.parent.is_some() {
                    ancestor_path(inner, record.parent)
                } else {
                    VecDeque::new()
                };
                self.pending.push_back(Move {
                    depth: 1 + path.len() as u64,
                    started: self.now(),
                    offered: Vec::new(),
                    path,
                    record,
                    failures: 0,
                    leaf_only: false,
                    mixed_prefix_queued: false,
                    predecessor_candidates: None,
                });
            } else {
                inner.retired(record.key, record.tier, Retirement::Failed);
            }
        }
        while ready && self.moving.len() < MOVES_MAX && self.now() >= self.retry_after {
            let Some(entry) = self.pending.pop_front() else {
                break;
            };
            self.begin(inner, pool, entry);
        }
        if !up {
            for Move { record, .. } in self.pending.drain(..) {
                inner.retired(record.key, record.tier, Retirement::Failed);
            }
        }
    }

    fn begin<S: crate::store::TierStore, P: PoolPort>(
        &mut self,
        inner: &mut S,
        pool: &mut P,
        mut entry: Move,
    ) {
        let key = entry.path.front().copied().unwrap_or(entry.record.key);
        let is_predecessor = key != entry.record.key;
        let current = (entry.record.tier == Tier::Dram || is_predecessor)
            .then(|| inner.retirement_predecessor(key, is_predecessor))
            .flatten();
        let mut record = match current {
            Some(record) => record,
            None if !is_predecessor => entry.record.clone(),
            None => return self.failed(inner, &entry.record),
        };
        let pin = match inner.pin_export(record.tier, record.key, 0, record.row) {
            Ok(pin) => pin,
            Err(_) => return self.failed(inner, &entry.record),
        };
        // A chain mixing raw (v3) and packed (v4) spans cannot be streamed as one homogeneous
        // root put against a cold manager. Establish row-bearing predecessors first so the
        // manager holds the prefix before accepting this record's homogeneous suffix.
        if pin.has_mixed_representations()
            && entry.path.is_empty()
            && !entry.leaf_only
            && !entry.mixed_prefix_queued
        {
            entry.mixed_prefix_queued = true;
            let prefix = ancestor_path(inner, record.parent);
            if !prefix.is_empty() {
                drop(pin);
                entry.path = prefix;
                return self.begin(inner, pool, entry);
            }
        }
        entry.predecessor_candidates = is_predecessor
            .then(|| std::mem::take(&mut record.ancestor_keys));
        let packed_offer = pin.packed_offer();
        let is_packed = packed_offer.is_some();
        let request = PutRequest {
            key: record.key,
            parent: record.parent,
            ancestor_keys: entry.candidates().to_vec(),
            bucket: record.bucket,
            chain_pages: pin.chain().total_pages,
            tail_pages: record.tail_pages as u64,
            capture_role: record.capture_role,
            child_paths: record.child_paths,
            descendant_row: record.descendant_row,
            next_child: entry
                .path
                .get(1)
                .copied()
                .or_else(|| entry.path.front().map(|_| entry.record.key)),
            row: record.row,
            packed: packed_offer,
            source: PutSource::Host(Box::new(PinBytes {
                pin,
                page_bytes: self.page_bytes,
                row_bytes: self.row_bytes,
                packed: is_packed,
            })),
        };
        if record.tier == Tier::Dram {
            entry.offered.push(record);
        }
        match pool.put(request) {
            Ok(id) => {
                self.moving.insert(id, entry);
            }
            Err(_) => self.failed(inner, &entry.record),
        }
    }

    fn back_off(&mut self) {
        self.retry_after = self.now() + Duration::from_millis(50);
    }

    fn failed<S: crate::store::TierStore>(&mut self, inner: &mut S, record: &Retiring) {
        self.back_off();
        inner.set_spill(false);
        inner.retired(record.key, record.tier, Retirement::Failed);
    }

    /// A put ended. False when it was not one of the cascade's.
    pub(crate) fn ended<S: crate::store::TierStore>(
        &mut self,
        inner: &mut S,
        id: u64,
        outcome: &PutOutcome,
    ) -> bool {
        let Some(mut entry) = self.moving.remove(&id) else {
            return false;
        };
        if let PutOutcome::Stored { pages, linked, .. } = outcome {
            let put = entry.offered.last().unwrap_or(&entry.record);
            let candidates = entry.candidates();
            if let Some(parent) = linked {
                if candidates.iter().skip(1).any(|key| key == parent) {
                    self.deeper_link_records += 1;
                    self.deeper_link_pages += pages;
                }
            } else if put.parent.is_none() {
                let reason = crate::RootReason::from_nearest(candidates.first().copied()) as usize;
                self.root_records[reason] += 1;
                self.root_pages[reason] += pages;
            }
        }
        let record = &entry.record;
        match outcome {
            PutOutcome::Stored { .. } | PutOutcome::Present => {
                if entry.path.pop_front().is_some() {
                    entry.failures = 0;
                    self.pending.push_front(entry);
                } else {
                    if record.tier == Tier::Dram {
                        if entry.offered.iter().any(|old| {
                            inner
                                .retirement_predecessor(old.key, false)
                                .is_some_and(|current| {
                                    current.row
                                        && (!old.row
                                            || current.capture_role > old.capture_role
                                            || (old.descendant_row && !current.descendant_row)
                                            || (old.child_paths < 2 && current.child_paths >= 2))
                                })
                        }) {
                            self.failed(inner, record);
                            return true;
                        }
                        self.paths += 1;
                        self.max_depth = self.max_depth.max(entry.depth);
                        let elapsed = self.now().duration_since(entry.started).as_nanos() as u64;
                        for (count, ms) in self
                            .latency_buckets
                            .iter_mut()
                            .zip(crate::RETIREMENT_LATENCY_MS)
                        {
                            *count += u64::from(elapsed <= ms * 1_000_000);
                        }
                        self.latency_ns += elapsed;
                        self.max_latency_ns = self.max_latency_ns.max(elapsed);
                    }
                    let how = if entry.leaf_only {
                        Retirement::MovedLeaf
                    } else {
                        Retirement::Moved
                    };
                    inner.retired(record.key, record.tier, how);
                }
            }
            PutOutcome::Refused(
                reason @ (PutRefusal::Invalid
                | PutRefusal::Unlinked
                | PutRefusal::Oversize
                | PutRefusal::Budget
                | PutRefusal::Inflight),
            ) => {
                let temporary = matches!(reason, PutRefusal::Budget | PutRefusal::Inflight);
                if temporary {
                    self.back_off();
                    self.refused_until = self.retry_after;
                    if !entry.leaf_only
                        && crate::store::retry_once(&mut entry.failures) == Retirement::Failed
                    {
                        entry.offered.pop();
                        self.pending.push_back(entry);
                        return true;
                    }
                }
                let index = match reason {
                    PutRefusal::Invalid => 0,
                    PutRefusal::Unlinked => 1,
                    PutRefusal::Oversize => 2,
                    PutRefusal::Budget => 3,
                    PutRefusal::Inflight => 4,
                    _ => unreachable!(),
                };
                if temporary || entry.path.is_empty() {
                    self.refusal_drops[index] += 1;
                }
                if entry.path.is_empty() {
                    self.dropped[usize::from(record.tier == Tier::Nvme)] += 1;
                    inner.retired(record.key, record.tier, Retirement::Dropped);
                } else {
                    entry.path.clear();
                    entry.offered.clear();
                    entry.leaf_only = true;
                    entry.depth = 1;
                    self.pending.push_front(entry);
                }
            }
            _ => {
                self.failed(inner, record);
                if record.tier == Tier::Dram {
                    self.refused_until = self.retry_after;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::PrefixKey;
    use crate::pool::{
        pins::{MemoryPin, MemorySpan, host_chain},
        tiers::{RecordSums, Releases},
    };

    fn key(tokens: u64, salt: u8) -> PrefixKey {
        PrefixKey {
            consumed_tokens: tokens,
            digest: [salt; 16],
        }
    }

    #[test]
    fn put_bytes_cross_the_records_of_the_chain_and_carry_the_persisted_checksums() {
        const PAGE: usize = 64;
        const ROW: usize = 32;
        // Pages of 16 tokens: the first record owns pages 0..3, the second 3..5.
        let first: Vec<u8> = (0..3 * PAGE + ROW).map(|i| i as u8).collect();
        let second: Vec<u8> = (0..2 * PAGE + ROW).map(|i| (i * 3 + 1) as u8).collect();
        let (a, b) = (first.as_ptr() as usize, second.as_ptr() as usize);
        let releases = Releases::default();
        let nodes = [(key(48, 1), 3), (key(80, 2), 2)];
        let pin = MemoryPin {
            chain: host_chain(16, 0, true, &nodes),
            spans: vec![
                MemorySpan {
                    pages: vec![a, a + PAGE, a + 2 * PAGE],
                    row: Some(vec![a + 3 * PAGE]),
                    sums: Arc::new(RecordSums {
                        pages: vec![10, 11, 12],
                        row: Some(13),
                    }),
                    packed: None,
                    slab_bytes: PAGE,
                },
                MemorySpan {
                    pages: vec![b, b + PAGE],
                    row: Some(vec![b + 2 * PAGE]),
                    sums: Arc::new(RecordSums {
                        pages: vec![20, 21],
                        row: Some(22),
                    }),
                    packed: None,
                    slab_bytes: PAGE,
                },
            ],
            layout: Arc::new(crate::Layout {
                page_tokens: 16,
                page_record_bytes: PAGE as u64,
                row_record_bytes: ROW as u64,
                ranks: vec![crate::geometry::RankLayout {
                    page_offset: 0,
                    plane_offsets: vec![0],
                    plane_bytes: vec![64],
                    row_bytes: 64,
                    row_offset: 0,
                }],
            }),
            keys: vec![key(48, 1), key(80, 2)],
            releases: releases.clone(),
        };
        let mut bytes = PinBytes {
            pin: Box::new(pin),
            page_bytes: PAGE,
            row_bytes: ROW,
            packed: false,
        };

        // A window across the boundary between the records.
        let mut dst = vec![0u8; 3 * PAGE];
        let sums = bytes.fill(2, 3, false, &mut dst).unwrap();
        assert_eq!(dst[..PAGE], first[2 * PAGE..3 * PAGE]);
        assert_eq!(dst[PAGE..], second[..2 * PAGE]);
        assert_eq!((sums.pages, sums.row), (vec![12, 20, 21], None));

        // The last window carries the endpoint's row after its pages.
        let mut dst = vec![0u8; 2 * PAGE + ROW];
        let sums = bytes.fill(3, 2, true, &mut dst).unwrap();
        assert_eq!(dst, second);
        assert_eq!((sums.pages, sums.row), (vec![20, 21], Some(22)));

        // A row-only window after every page went in earlier ones.
        let mut dst = vec![0u8; ROW];
        let sums = bytes.fill(5, 0, true, &mut dst).unwrap();
        assert_eq!(dst, second[2 * PAGE..]);
        assert_eq!((sums.pages.len(), sums.row), (0, Some(22)));

        // A window the wrong size, or beyond the chain, is refused.
        assert!(bytes.fill(0, 1, false, &mut [0u8; PAGE + 1]).is_err());
        assert!(bytes.fill(4, 3, false, &mut [0u8; 3 * PAGE]).is_err());
        drop(bytes);
        assert_eq!(releases.drain(), [key(48, 1), key(80, 2)]);
    }

    #[test]
    fn mixed_raw_parent_and_packed_child_offers_packed_suffix_and_reverse_stays_raw() {
        use crate::memory::PackedMemory;
        use crate::shared::wire::{
            PackedUnit,
            packed_unit::{Kind, Representation},
        };
        const PAGE: usize = 64;
        const ROW: usize = 32;
        let raw_parent: Vec<u8> = (0..3 * PAGE + ROW).map(|i| i as u8).collect();
        let packed_p3 = vec![0xAAu8; 20];
        let packed_p4 = vec![0xBBu8; 24];
        let packed_row = vec![0xCCu8; 16];
        let mut packed_page_slab = vec![0u8; PAGE];
        packed_page_slab[..20].copy_from_slice(&packed_p3);
        packed_page_slab[20..44].copy_from_slice(&packed_p4);
        let mut packed_row_slab = vec![0u8; PAGE];
        packed_row_slab[..16].copy_from_slice(&packed_row);
        let make_pkt = |kind: Kind, idx: u64, data: &[u8], orig: u64| PackedUnit {
            kind: kind as i32,
            logical_index: idx,
            stored_rank: 0,
            representation: Representation::Packet as i32,
            profile: vec![1u8; 32],
            original_length: orig,
            stored_length: data.len() as u64,
            stored_crc: crate::shared::packed::crc32(data),
            original_crc: 1,
            raw_xxh3: 0,
            replicas: Vec::new(),
        };
        let packed_mem = Arc::new(PackedMemory {
            first_page: 3,
            page_count: 2,
            pages: vec![
                (make_pkt(Kind::Page, 3, &packed_p3, PAGE as u64), 0),
                (make_pkt(Kind::Page, 4, &packed_p4, PAGE as u64), 20),
            ],
            row: Some(vec![(make_pkt(Kind::Row, 0, &packed_row, ROW as u64), 0)]),
        });
        let layout = Arc::new(crate::Layout {
            page_tokens: 16,
            page_record_bytes: PAGE as u64,
            row_record_bytes: ROW as u64,
            ranks: vec![crate::geometry::RankLayout {
                page_offset: 0,
                plane_offsets: vec![0],
                plane_bytes: vec![64],
                row_bytes: 32,
                row_offset: 0,
            }],
        });

        // 1. Raw v3 parent (pages 0..3) + packed v4 child (pages 3..5 + row):
        //    `packed_offer()` must return the child's packed suffix starting at page 3,
        //    and `PinBytes::fill(3, 2, true, ...)` must stream the packed child cleanly.
        let (a, pp, pr) = (
            raw_parent.as_ptr() as usize,
            packed_page_slab.as_ptr() as usize,
            packed_row_slab.as_ptr() as usize,
        );
        let nodes = [(key(48, 1), 3), (key(80, 2), 2)];
        let pin = MemoryPin {
            chain: host_chain(16, 0, true, &nodes),
            spans: vec![
                MemorySpan {
                    pages: vec![a, a + PAGE, a + 2 * PAGE],
                    row: Some(vec![a + 3 * PAGE]),
                    sums: Arc::new(RecordSums {
                        pages: vec![10, 11, 12],
                        row: Some(13),
                    }),
                    packed: None,
                    slab_bytes: PAGE,
                },
                MemorySpan {
                    pages: vec![pp],
                    row: Some(vec![pr]),
                    sums: Arc::new(RecordSums::default()),
                    packed: Some(packed_mem.clone()),
                    slab_bytes: PAGE,
                },
            ],
            layout: layout.clone(),
            keys: vec![key(48, 1), key(80, 2)],
            releases: Releases::default(),
        };
        let offer = pin
            .packed_offer()
            .expect("raw parent + packed child must offer packed suffix");
        assert_eq!(offer.first_page, 3);
        assert_eq!(offer.units.len(), 3);
        let mut bytes = PinBytes {
            pin: Box::new(pin),
            page_bytes: PAGE,
            row_bytes: ROW,
            packed: true,
        };
        let mut dst = vec![0u8; 20 + 24 + 16];
        bytes.fill(3, 2, true, &mut dst).unwrap();
        assert_eq!(&dst[..20], packed_p3.as_slice());
        assert_eq!(&dst[20..44], packed_p4.as_slice());
        assert_eq!(&dst[44..], packed_row.as_slice());

        // 2. Reverse ordering (packed parent + raw child): `packed_offer()` returns None
        //    and raw suffix fill for the raw child succeeds cleanly.
        let raw_child: Vec<u8> = (0..2 * PAGE + ROW).map(|i| (i * 5 + 3) as u8).collect();
        let rc = raw_child.as_ptr() as usize;
        let rev_pin = MemoryPin {
            chain: host_chain(16, 0, true, &nodes),
            spans: vec![
                MemorySpan {
                    pages: vec![pp],
                    row: None,
                    sums: Arc::new(RecordSums::default()),
                    packed: Some(packed_mem),
                    slab_bytes: PAGE,
                },
                MemorySpan {
                    pages: vec![rc, rc + PAGE],
                    row: Some(vec![rc + 2 * PAGE]),
                    sums: Arc::new(RecordSums {
                        pages: vec![20, 21],
                        row: Some(22),
                    }),
                    packed: None,
                    slab_bytes: PAGE,
                },
            ],
            layout,
            keys: vec![key(48, 1), key(80, 2)],
            releases: Releases::default(),
        };
        assert!(rev_pin.packed_offer().is_none());
        let mut rev_bytes = PinBytes {
            pin: Box::new(rev_pin),
            page_bytes: PAGE,
            row_bytes: ROW,
            packed: false,
        };
        let mut raw_dst = vec![0u8; 2 * PAGE + ROW];
        let sums = rev_bytes.fill(3, 2, true, &mut raw_dst).unwrap();
        assert_eq!(raw_dst, raw_child);
        assert_eq!((sums.pages, sums.row), (vec![20, 21], Some(22)));
    }

    #[test]
    fn cold_manager_retires_mixed_raw_and_packed_nvme_chains_parent_first() {
        use crate::memory::PackedMemory;
        use crate::shared::wire::{
            PackedUnit,
            packed_unit::{Kind, Representation},
        };
        use crate::store::TierStore;
        use std::collections::HashSet;

        const PAGE: usize = 64;
        const ROW: usize = 32;
        let p_key = key(48, 1);
        let c_key = key(80, 2);
        let layout = Arc::new(crate::Layout {
            page_tokens: 16,
            page_record_bytes: PAGE as u64,
            row_record_bytes: ROW as u64,
            ranks: vec![crate::geometry::RankLayout {
                page_offset: 0,
                plane_offsets: vec![0],
                plane_bytes: vec![64],
                row_bytes: 32,
                row_offset: 0,
            }],
        });

        let make_pkt = |kind: Kind, idx: u64, data: &[u8], orig: u64| PackedUnit {
            kind: kind as i32,
            logical_index: idx,
            stored_rank: 0,
            representation: Representation::Packet as i32,
            profile: vec![1u8; 32],
            original_length: orig,
            stored_length: data.len() as u64,
            stored_crc: crate::shared::packed::crc32(data),
            original_crc: 1,
            raw_xxh3: 0,
            replicas: Vec::new(),
        };

        struct MixedDiskStore {
            layout: Arc<crate::Layout>,
            parent_packed: bool,
            raw_p: Vec<u8>,
            raw_c: Vec<u8>,
            packed_p_pages: Vec<u8>,
            packed_p_row: Vec<u8>,
            packed_c_pages: Vec<u8>,
            packed_c_row: Vec<u8>,
            packed_p_mem: Arc<PackedMemory>,
            packed_c_mem: Arc<PackedMemory>,
            retiring: Vec<Retiring>,
            retired: Vec<(PrefixKey, Retirement)>,
        }

        impl crate::store::HostStore for MixedDiskStore {
            fn lookup(&mut self, _: &dyn crate::History, _: u64, _: u64) -> Option<crate::HostHit> {
                None
            }
            fn store(&mut self, _: &crate::PublishedCheckpoint) -> crate::Result<crate::StoreOutcome> {
                unreachable!()
            }
            fn fetch(&mut self, _: crate::HostHit, _: crate::InboundTicket) -> crate::Result<()> {
                unreachable!()
            }
            fn poll(&mut self, _: &mut Vec<crate::StoreEvent>) -> crate::Result<()> {
                Ok(())
            }
            fn stats(&self) -> crate::StoreStats {
                crate::StoreStats::default()
            }
            fn close(&mut self) -> crate::Result<()> {
                Ok(())
            }
        }

        impl TierStore for MixedDiskStore {
            fn pin_export(
                &mut self,
                tier: Tier,
                target: PrefixKey,
                _: u64,
                want_row: bool,
            ) -> std::result::Result<Box<dyn HostPin>, crate::ExportRefusal> {
                assert_eq!(tier, Tier::Nvme);
                let p_key = key(48, 1);
                let c_key = key(80, 2);
                let raw_p_span = || MemorySpan {
                    pages: vec![
                        self.raw_p.as_ptr() as usize,
                        self.raw_p.as_ptr() as usize + PAGE,
                        self.raw_p.as_ptr() as usize + 2 * PAGE,
                    ],
                    row: want_row.then(|| vec![self.raw_p.as_ptr() as usize + 3 * PAGE]),
                    sums: Arc::new(RecordSums {
                        pages: vec![10, 11, 12],
                        row: want_row.then_some(13),
                    }),
                    packed: None,
                    slab_bytes: PAGE,
                };
                let packed_p_span = || MemorySpan {
                    pages: vec![self.packed_p_pages.as_ptr() as usize],
                    row: want_row.then(|| vec![self.packed_p_row.as_ptr() as usize]),
                    sums: Arc::new(RecordSums::default()),
                    packed: Some(self.packed_p_mem.clone()),
                    slab_bytes: PAGE,
                };
                let raw_c_span = || MemorySpan {
                    pages: vec![
                        self.raw_c.as_ptr() as usize,
                        self.raw_c.as_ptr() as usize + PAGE,
                    ],
                    row: want_row.then(|| vec![self.raw_c.as_ptr() as usize + 2 * PAGE]),
                    sums: Arc::new(RecordSums {
                        pages: vec![20, 21],
                        row: want_row.then_some(22),
                    }),
                    packed: None,
                    slab_bytes: PAGE,
                };
                let packed_c_span = || MemorySpan {
                    pages: vec![self.packed_c_pages.as_ptr() as usize],
                    row: want_row.then(|| vec![self.packed_c_row.as_ptr() as usize]),
                    sums: Arc::new(RecordSums::default()),
                    packed: Some(self.packed_c_mem.clone()),
                    slab_bytes: PAGE,
                };
                if target == p_key {
                    let span = if self.parent_packed { packed_p_span() } else { raw_p_span() };
                    Ok(Box::new(MemoryPin {
                        chain: host_chain(16, 0, want_row, &[(p_key, 3)]),
                        spans: vec![span],
                        layout: self.layout.clone(),
                        keys: vec![p_key],
                        releases: Releases::default(),
                    }))
                } else {
                    let (s0, s1) = if self.parent_packed {
                        (packed_p_span(), raw_c_span())
                    } else {
                        (raw_p_span(), packed_c_span())
                    };
                    Ok(Box::new(MemoryPin {
                        chain: host_chain(16, 0, want_row, &[(p_key, 3), (c_key, 2)]),
                        spans: vec![s0, s1],
                        layout: self.layout.clone(),
                        keys: vec![p_key, c_key],
                        releases: Releases::default(),
                    }))
                }
            }
            fn retirement_predecessor(&self, target: PrefixKey, _: bool) -> Option<Retiring> {
                (target == key(48, 1)).then(|| Retiring {
                    tier: Tier::Nvme,
                    key: target,
                    parent: None,
                    ancestor_keys: Vec::new(),
                    bucket: [0; 16],
                    tail_pages: 3,
                    row: true,
                    capture_role: crate::CaptureRole::Endpoint,
                    child_paths: 1,
                    descendant_row: true,
                })
            }
            fn take_retiring(&mut self) -> Vec<Retiring> {
                std::mem::take(&mut self.retiring)
            }
            fn retired(&mut self, k: PrefixKey, _: Tier, how: Retirement) {
                self.retired.push((k, how));
            }
        }

        // Cold manager simulator: rejects any full mixed root put, accepts parent at first_page=0,
        // and accepts child suffix at first_page=3 once parent is stored.
        struct ColdManager {
            next_id: u64,
            held: HashSet<PrefixKey>,
            force_root: bool,
            completed: Vec<(u64, PutOutcome)>,
            log: Vec<(PrefixKey, bool, u64)>,
        }
        impl PoolPort for ColdManager {
            fn begin_lookup(&self, _: &dyn crate::History, _: u64) -> crate::pool::LookupTicket {
                unreachable!()
            }
            fn fetch(&mut self, _: crate::pool::PoolFetch) -> Result<u64> {
                unreachable!()
            }
            fn poll(&mut self, _: &mut Vec<crate::pool::PoolEvent>) -> Result<()> {
                Ok(())
            }
            fn saved(&self, _: crate::pool::IndexEntry) {}
            fn evicted(&self, _: PrefixKey, _: Tier, _: Option<crate::pool::IndexEntry>) {}
            fn evictions_lost(&self) {}
            fn cancel_put(&mut self, _: u64) {}
            fn announcer(&self) -> crate::pool::Announcer {
                unreachable!()
            }
            fn host_pinned(&mut self, _: crate::ExportId, _: Box<dyn HostPin>) {}
            fn export_pinned(&mut self, _: crate::ExportId, _: crate::ExportChain) {}
            fn export_refused(&mut self, _: crate::ExportId, _: crate::ExportRefusal) {}
            fn close(&mut self) -> Result<()> {
                Ok(())
            }
            fn ready(&self) -> bool {
                true
            }
            fn put(&mut self, req: PutRequest) -> Result<u64> {
                self.next_id += 1;
                let id = self.next_id;
                let first_page =
                    if !self.force_root && req.parent.is_some_and(|p| self.held.contains(&p)) {
                        3
                    } else {
                        0
                    };
                let PutSource::Host(mut bytes) = req.source;
                if first_page < bytes.min_first_page() {
                    self.log.push((req.key, req.packed.is_some(), first_page));
                    self.completed
                        .push((id, PutOutcome::Refused(PutRefusal::Invalid)));
                    return Ok(id);
                }
                let page_count = (req.chain_pages - first_page) as u32;
                let is_packed = req
                    .packed
                    .as_ref()
                    .is_some_and(|o| o.first_page <= first_page);
                let dst_len = if is_packed {
                    req.packed
                        .as_ref()
                        .unwrap()
                        .units
                        .iter()
                        .filter(|u| {
                            u.kind == Kind::Row as i32 || u.logical_index >= first_page
                        })
                        .map(|u| u.stored_length as usize)
                        .sum()
                } else {
                    page_count as usize * PAGE + usize::from(req.row) * ROW
                };
                let mut buf = vec![0u8; dst_len];
                bytes
                    .fill(first_page, page_count, req.row, &mut buf)
                    .expect("cold manager fill must succeed without mixed-span read error");
                self.held.insert(req.key);
                self.log.push((req.key, is_packed, first_page));
                self.completed.push((
                    id,
                    PutOutcome::Stored {
                        bytes: dst_len as u64,
                        pages: u64::from(page_count),
                        linked: (!self.force_root)
                            .then(|| req.parent.filter(|p| self.held.contains(p)))
                            .flatten(),
                    },
                ));
                Ok(id)
            }
        }

        for (parent_packed, force_root) in [
            (false, false),
            (true, false),
            (false, true),
            (true, true),
        ] {
            let mut packed_p_pages = vec![0u8; PAGE];
            packed_p_pages[..12].fill(0x11);
            packed_p_pages[12..24].fill(0x22);
            packed_p_pages[24..36].fill(0x33);
            let mut packed_p_row = vec![0u8; PAGE];
            packed_p_row[..10].fill(0x44);
            let mut packed_c_pages = vec![0u8; PAGE];
            packed_c_pages[..14].fill(0x55);
            packed_c_pages[14..28].fill(0x66);
            let mut packed_c_row = vec![0u8; PAGE];
            packed_c_row[..12].fill(0x77);

            let packed_p_mem = Arc::new(PackedMemory {
                first_page: 0,
                page_count: 3,
                pages: vec![
                    (make_pkt(Kind::Page, 0, &packed_p_pages[..12], PAGE as u64), 0),
                    (make_pkt(Kind::Page, 1, &packed_p_pages[12..24], PAGE as u64), 12),
                    (make_pkt(Kind::Page, 2, &packed_p_pages[24..36], PAGE as u64), 24),
                ],
                row: Some(vec![(make_pkt(Kind::Row, 0, &packed_p_row[..10], ROW as u64), 0)]),
            });
            let packed_c_mem = Arc::new(PackedMemory {
                first_page: 3,
                page_count: 2,
                pages: vec![
                    (make_pkt(Kind::Page, 3, &packed_c_pages[..14], PAGE as u64), 0),
                    (make_pkt(Kind::Page, 4, &packed_c_pages[14..28], PAGE as u64), 14),
                ],
                row: Some(vec![(make_pkt(Kind::Row, 0, &packed_c_row[..12], ROW as u64), 0)]),
            });

            let mut store = MixedDiskStore {
                layout: layout.clone(),
                parent_packed,
                raw_p: vec![0xA1; 3 * PAGE + ROW],
                raw_c: vec![0xB2; 2 * PAGE + ROW],
                packed_p_pages,
                packed_p_row,
                packed_c_pages,
                packed_c_row,
                packed_p_mem,
                packed_c_mem,
                retiring: vec![Retiring {
                    tier: Tier::Nvme,
                    key: c_key,
                    parent: Some(p_key),
                    ancestor_keys: vec![p_key],
                    bucket: [0; 16],
                    tail_pages: 2,
                    row: true,
                    capture_role: crate::CaptureRole::Endpoint,
                    child_paths: 0,
                    descendant_row: false,
                }],
                retired: Vec::new(),
            };
            let mut mgr = ColdManager {
                next_id: 0,
                held: HashSet::new(),
                force_root,
                completed: Vec::new(),
                log: Vec::new(),
            };
            let mut cascade = Cascade::new(&layout);
            for _ in 0..4 {
                cascade.pump(&mut store, &mut mgr);
                for (id, outcome) in mgr.completed.drain(..) {
                    assert!(cascade.ended(&mut store, id, &outcome));
                }
            }
            if force_root {
                assert_eq!(
                    store.retired,
                    vec![(c_key, Retirement::Dropped)],
                    "forced mixed root (REROOT_DEPTH) must drop cleanly without retry loops"
                );
                assert_eq!(
                    mgr.log,
                    vec![(p_key, parent_packed, 0), (c_key, !parent_packed, 0)],
                );
            } else {
                assert_eq!(store.retired, vec![(c_key, Retirement::Moved)]);
                assert_eq!(
                    mgr.log,
                    vec![
                        (p_key, parent_packed, 0),
                        (c_key, !parent_packed, 3),
                    ],
                    "cold manager must retire parent first at page 0, then child suffix at page 3"
                );
            }
        }
    }
}
