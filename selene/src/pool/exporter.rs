/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! What a host tier pins for an export: a chain of records that fills windows.

use crate::{PrefixKey, Result};

/// One record of a pinned host chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostRecord {
    pub key: PrefixKey,
    /// Record-relative page where this export's pages start.
    pub first_page: u32,
    pub pages: u32,
    /// The record carries the endpoint row this export includes.
    pub row: bool,
}

/// What a host pin holds, in restore order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostChain {
    pub records: Vec<HostRecord>,
    pub total_pages: u64,
    pub row: bool,
}

/// A span of one record to copy into a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostWindow {
    pub record: u32,
    pub first_page: u32,
    pub page_count: u32,
    pub row: bool,
}

/// The checksums persisted with the records read, never hashes of the bytes
/// just read: a corrupted record must reach the requester as a mismatch.
pub use crate::shared::SliceSums as Checksums;

/// A host record chain held against eviction. Dropping it releases the hold.
pub trait HostPin: Send {
    fn chain(&self) -> &HostChain;
    /// Copy the span into `dst` (page records, then the row) and return its
    /// persisted checksums. May block on a device: it runs on a copy thread.
    fn read(&mut self, window: &HostWindow, dst: &mut [u8]) -> Result<Checksums>;
    /// Whether `record` in the pinned chain is stored packed (v4).
    fn is_packed(&self, record: usize) -> bool {
        self.packed_offer_for(record).is_some()
    }
    /// Optional packed unit directory for one record in the pinned chain.
    fn packed_offer_for(&self, _record: usize) -> Option<crate::shared::wire::PackedOffer> {
        None
    }
    /// Optional combined packed unit directory for the longest packed suffix ending at
    /// the pinned endpoint. Returns `None` if the endpoint itself is stored raw, allowing
    /// mixed raw-ancestor + packed-child chains to offer the child's packed suffix once
    /// the raw ancestor is already held by the manager.
    fn packed_offer(&self) -> Option<crate::shared::wire::PackedOffer> {
        let count = self.chain().records.len();
        if count == 0 {
            return None;
        }
        let mut offers = Vec::new();
        for i in (0..count).rev() {
            match self.packed_offer_for(i) {
                Some(offer) => offers.push(offer),
                None => break,
            }
        }
        if offers.is_empty() {
            return None;
        }
        offers.reverse();
        let first_page = offers[0].first_page;
        let units = offers.into_iter().flat_map(|o| o.units).collect();
        Some(crate::shared::wire::PackedOffer { first_page, units })
    }
    /// Chain-relative starting page of the longest homogeneous (all-raw or all-packed)
    /// suffix ending at the pinned endpoint. Returns `0` for homogeneous chains; for mixed
    /// chains, a manager forcing a new root below this page cannot stream both formats in one
    /// v3 or v4 record.
    fn homogeneous_suffix_first_page(&self) -> u64 {
        let records = &self.chain().records;
        let Some(last) = records.len().checked_sub(1) else {
            return 0;
        };
        let end_packed = self.is_packed(last);
        let mut start = last;
        while start > 0 && self.is_packed(start - 1) == end_packed {
            start -= 1;
        }
        records[..start].iter().map(|r| u64::from(r.pages)).sum()
    }
    /// Whether the pinned chain mixes raw (v3) and packed (v4) records across its spans.
    fn has_mixed_representations(&self) -> bool {
        let count = self.chain().records.len();
        if count <= 1 {
            return false;
        }
        let first = self.is_packed(0);
        (1..count).any(|i| self.is_packed(i) != first)
    }
    /// Copy packed bytes for `window` into `dst`, returning bytes written.
    fn read_packed(&mut self, _window: &HostWindow, _dst: &mut [u8]) -> Result<usize> {
        Err(crate::Error::new("host pin does not carry packed records"))
    }
}

/// Bytes a store or a probe supplies for a put, window by window.
pub trait PutBytes: Send {
    /// Earliest chain-relative page this source can serve in its offered format.
    /// Homogeneous chains return `0`; mixed raw/packed chains return the start of their
    /// homogeneous suffix so a forced root below that boundary is refused cleanly.
    fn min_first_page(&self) -> u64 {
        0
    }
    /// Fill `dst` with `page_count` page records starting at page `first_page`
    /// of the chain, then the row record when `row`, and return the checksums to
    /// commit with them: the persisted ones when the source has them, never
    /// hashes of what was just read.
    fn fill(
        &mut self,
        first_page: u64,
        page_count: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<Checksums>;
}
