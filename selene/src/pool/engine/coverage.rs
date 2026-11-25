/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exact coverage of a claim or export by its windows.

use super::{super::wire, requester::ExpectedRecord};

/// Tracks which pages and row the windows issued so far have covered. A window
/// is accepted only if it extends the coverage exactly: ordinal order, inside the
/// record being filled, the row only in the window that completes the last
/// record's pages or in a window of its own after them (the manager keeps a row
/// out of a window that its pages already fill), and a length that matches its
/// pages.
pub(crate) struct Coverage {
    pub(crate) expected: Vec<ExpectedRecord>,
    packed: Vec<Option<crate::shared::packed::Layout>>,
    pub(crate) channel_rank: Option<u32>,
    taken: Vec<u32>,
    cursor: usize,
    windows: u32,
}

impl Coverage {
    pub(crate) fn new(expected: Vec<ExpectedRecord>) -> Self {
        Self {
            packed: vec![None; expected.len()],
            channel_rank: None,
            taken: vec![0; expected.len()],
            expected,
            cursor: 0,
            windows: 0,
        }
    }

    pub(crate) fn select_packed(
        &mut self,
        records: &[wire::Record],
        caps: Option<&wire::CodecCaps>,
        page_tokens: u64,
        ranks: impl FnOnce() -> Vec<crate::record::RankSizes>,
    ) -> Result<(), &'static str> {
        if records.iter().all(|r| r.packed.is_none()) {
            return Ok(());
        }
        let caps = caps.ok_or("packed selection requires a negotiated profile")?;
        let ranks = ranks();
        for (i, r) in records.iter().enumerate() {
            let Some(offer) = &r.packed else { continue };
            let first = (self.expected[i].key.consumed_tokens / page_tokens)
                .checked_sub(r.pages.into())
                .ok_or("invalid packed first page")?;
            let layout = crate::shared::packed::Layout::new(offer.clone(), &ranks, first, r.pages)
                .map_err(|_| "invalid packed selection")?;
            if !crate::shared::packed::permits(Some(caps), &layout.required_profiles(r.row))
                || r.row != (layout.row_bytes() != 0)
            {
                return Err("incompatible packed selection");
            }
            self.packed[i] = Some(layout);
        }
        Ok(())
    }

    pub(crate) fn set_packed_layout(&mut self, record: usize, layout: crate::shared::packed::Layout) {
        if let Some(slot) = self.packed.get_mut(record) {
            *slot = Some(layout);
        }
    }

    pub(crate) fn pinned_units(&self, spec: &wire::WindowSpec) -> Option<Vec<wire::PackedUnit>> {
        let r = spec.record as usize;
        let layout = self.packed.get(r)?.as_ref()?;
        let first = if spec.page_count == 0 && spec.row {
            layout.page_count()
        } else {
            spec.first_page.checked_sub(self.expected[r].first_page)? as usize
        };
        Some(
            layout
                .units(first, spec.page_count as usize, spec.row)?
                .iter()
                .filter(|u| {
                    self.channel_rank
                        .is_none_or(|r| r == u.stored_rank || u.replicas.contains(&r))
                })
                .cloned()
                .collect(),
        )
    }

    /// Every record, page and the row has been covered once.
    pub(crate) fn complete(&self) -> bool {
        self.cursor == self.expected.len()
    }

    /// Accept `spec` as the next window and return the claim-order index of
    /// its first page.
    pub(crate) fn accept(
        &mut self,
        spec: &wire::WindowSpec,
        page_bytes: u64,
        row_bytes: u64,
    ) -> Result<u64, String> {
        if spec.index != self.windows {
            return Err(format!("window {} out of order or repeated", spec.index));
        }
        let record = spec.record as usize;
        if self.cursor >= self.expected.len() || record != self.cursor {
            return Err(format!(
                "window {} names record {record}, expected {}",
                spec.index, self.cursor
            ));
        }
        let expected = &self.expected[record];
        let taken = self.taken[record];
        let last = record + 1 == self.expected.len();
        let remaining = expected.pages - taken;
        // A window without pages carries only the row, once every page is in.
        let row_alone = spec.page_count == 0 && spec.row && last && expected.row && remaining == 0;
        // A row alone names no page, so no first page is checked for it.
        if (!row_alone && spec.first_page != expected.first_page + taken)
            || spec.page_count > remaining
            || (spec.page_count == 0 && !row_alone)
        {
            return Err(format!(
                "window {} does not extend the coverage",
                spec.index
            ));
        }
        let completes = spec.page_count == remaining;
        if spec.row && !(last && completes && expected.row) {
            return Err(format!("window {} carries the row wrongly", spec.index));
        }
        let row = if spec.row { row_bytes } else { 0 };
        let length = match (&self.packed[record], &spec.packed) {
            (None, None) => u64::from(spec.page_count) * page_bytes + row,
            (Some(_), Some(window)) => {
                let units = self
                    .pinned_units(spec)
                    .ok_or("window outside packed selection")?;
                if units != window.units {
                    return Err("window differs from pinned selection".into());
                }
                units.iter().map(|u| u.stored_length).sum()
            }
            _ => return Err("window representation differs from pinned selection".into()),
        };
        if spec.length != length {
            return Err(format!("window {} has the wrong length", spec.index));
        }
        let before: u64 = self.expected[..record]
            .iter()
            .map(|r| u64::from(r.pages))
            .sum();
        self.taken[record] += spec.page_count;
        // The record is covered when its pages are in and, for the last, its row too.
        if completes && (!(last && expected.row) || spec.row) {
            self.cursor += 1;
        }
        self.windows += 1;
        Ok(before + u64::from(taken))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PrefixKey;

    const PAGE: u64 = 100;
    const ROW: u64 = 40;

    fn record(first_page: u32, pages: u32, row: bool) -> ExpectedRecord {
        ExpectedRecord {
            key: PrefixKey {
                consumed_tokens: 64,
                digest: [1; 16],
            },
            first_page,
            pages,
            row,
        }
    }

    fn window(index: u32, record: u32, first_page: u32, pages: u32, row: bool) -> wire::WindowSpec {
        wire::WindowSpec {
            index,
            record,
            first_page,
            page_count: pages,
            row,
            length: u64::from(pages) * PAGE + if row { ROW } else { 0 },
            ..Default::default()
        }
    }

    #[test]
    fn the_row_rides_with_the_last_pages_or_follows_in_a_window_of_its_own() {
        let mut together = Coverage::new(vec![record(0, 2, true)]);
        together
            .accept(&window(0, 0, 0, 2, true), PAGE, ROW)
            .unwrap();
        assert!(together.complete());

        let mut apart = Coverage::new(vec![record(0, 2, true)]);
        apart.accept(&window(0, 0, 0, 2, false), PAGE, ROW).unwrap();
        assert!(!apart.complete());
        // The manager names no page for a window that carries only the row.
        apart.accept(&window(1, 0, 0, 0, true), PAGE, ROW).unwrap();
        assert!(apart.complete());
    }

    #[test]
    fn a_misplaced_row_or_an_empty_window_is_refused() {
        let mut early = Coverage::new(vec![record(0, 2, true)]);
        assert!(early.accept(&window(0, 0, 0, 1, true), PAGE, ROW).is_err());
        // Pages cannot follow a record's row, and an ancestor never carries one.
        let mut ancestor = Coverage::new(vec![record(0, 1, true), record(1, 1, true)]);
        assert!(
            ancestor
                .accept(&window(0, 0, 0, 1, true), PAGE, ROW)
                .is_err()
        );
        let mut empty = Coverage::new(vec![record(0, 2, true)]);
        assert!(empty.accept(&window(0, 0, 0, 0, true), PAGE, ROW).is_err());
        // A row-only window needs a row to carry.
        let mut rowless = Coverage::new(vec![record(0, 2, false)]);
        rowless
            .accept(&window(0, 0, 0, 2, false), PAGE, ROW)
            .unwrap();
        assert!(rowless.complete());
        let mut without = Coverage::new(vec![record(0, 1, false), record(1, 1, false)]);
        without
            .accept(&window(0, 0, 0, 1, false), PAGE, ROW)
            .unwrap();
        assert!(
            without
                .accept(&window(1, 1, 1, 0, true), PAGE, ROW)
                .is_err()
        );
    }

    #[test]
    fn a_row_only_tail_is_one_window_with_no_pages() {
        let mut tail = Coverage::new(vec![record(3, 0, true)]);
        tail.accept(&window(0, 0, 3, 0, true), PAGE, ROW).unwrap();
        assert!(tail.complete());
    }
}
