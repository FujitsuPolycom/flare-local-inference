/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Mutually exclusive ownership of snapshot aliases and KV page ids.

use super::*;

#[cfg(test)]
thread_local! {
    pub(super) static STORAGE_SIMULATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SnapshotQuote {
    pub(super) snapshots: Vec<SnapshotId>,
    victims: Vec<CheckpointId>,
    planned_rows: usize,
    witness: Vec<SnapshotId>,
    keep_pages: usize,
    rank: crate::cache::RetentionRank,
}

// Counts pages visited by select_snapshots; read only by the work-bound test.
#[cfg(test)]
thread_local! {
    static SNAPSHOT_PAGE_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Minimizes free-page backing for one or two aliases, with lowest-id ties.
/// Larger selections retain the greedy feasible witness used for future captures.
fn select_snapshots(
    backing: &[bool],
    protected_refs: &[u32],
    ranges: &[std::ops::Range<u32>],
    candidates: &BTreeSet<SnapshotId>,
    keep_pages: usize,
    count: usize,
) -> Option<Vec<SnapshotId>> {
    let available = backing.iter().filter(|free| **free).count();
    if available < keep_pages {
        return None;
    }
    if count == 0 {
        return Some(Vec::new());
    }
    if candidates.len() < count {
        return None;
    }
    let budget = available - keep_pages;
    let mut free = vec![0u32; backing.len() + 1];
    let mut blocked = vec![0u32; backing.len() + 1];
    // Aliases are contiguous ranges. Prefix sums count free pages and detect pages
    // occupied by KV rather than protected snapshot backing in constant time per range.
    for page in 0..backing.len() {
        #[cfg(test)]
        SNAPSHOT_PAGE_VISITS.set(SNAPSHOT_PAGE_VISITS.get() + 1);
        free[page + 1] = free[page] + u32::from(backing[page]);
        blocked[page + 1] = blocked[page] + u32::from(!backing[page] && protected_refs[page] == 0);
    }
    let span = |sum: &[u32], r: &std::ops::Range<u32>| {
        (sum[r.end as usize] - sum[r.start as usize]) as usize
    };
    let candidates: Vec<(SnapshotId, std::ops::Range<u32>, usize)> = candidates
        .iter()
        .filter_map(|&id| {
            let range = ranges.get(id.0 as usize).cloned().unwrap_or(0..0);
            (span(&blocked, &range) == 0).then(|| {
                let cost = span(&free, &range);
                (id, range, cost)
            })
        })
        .collect();
    if candidates.len() < count {
        return None;
    }
    if count == 1 {
        let mut best: Option<(usize, SnapshotId)> = None;
        for &(id, _, cost) in &candidates {
            if best.is_some_and(|(best_cost, _)| cost >= best_cost) || cost > budget {
                continue;
            }
            best = Some((cost, id));
            if cost == 0 {
                break;
            }
        }
        return best.map(|(_, id)| vec![id]);
    }
    if count > 2 {
        let mut selected = BTreeSet::<SnapshotId>::new();
        let mut excluded = vec![false; backing.len()];
        let mut excluded_pages = 0usize;
        let mut open = free.clone();
        for _ in 0..count {
            let (id, range, _) = candidates
                .iter()
                .filter(|(id, _, _)| !selected.contains(id))
                .min_by_key(|(id, range, _)| (span(&open, range), *id))?;
            for page in range.clone() {
                #[cfg(test)]
                SNAPSHOT_PAGE_VISITS.set(SNAPSHOT_PAGE_VISITS.get() + 1);
                if backing[page as usize] && !excluded[page as usize] {
                    excluded[page as usize] = true;
                    excluded_pages += 1;
                }
            }
            if excluded_pages > budget {
                return None;
            }
            selected.insert(*id);
            for page in 0..backing.len() {
                #[cfg(test)]
                SNAPSHOT_PAGE_VISITS.set(SNAPSHOT_PAGE_VISITS.get() + 1);
                open[page + 1] = open[page] + u32::from(backing[page] && !excluded[page]);
            }
        }
        return Some(selected.into_iter().collect());
    }
    // The union costs |A| + |B| - |A ∩ B|; the intersection is another range.
    let mut best: Option<(usize, usize, usize)> = None;
    for a in 0..candidates.len() {
        let (_, range_a, cost_a) = &candidates[a];
        // A pair costs at least its larger member, so a member at or above the best
        // cost cannot improve on it (first-found ties are already the lowest ids).
        if *cost_a > budget || best.is_some_and(|(cost, _, _)| *cost_a >= cost) {
            continue;
        }
        for b in a + 1..candidates.len() {
            let (_, range_b, cost_b) = &candidates[b];
            if best.is_some_and(|(cost, _, _)| *cost_b >= cost) {
                continue;
            }
            let (lo, hi) = (
                range_a.start.max(range_b.start),
                range_a.end.min(range_b.end),
            );
            let shared = if lo < hi {
                (free[hi as usize] - free[lo as usize]) as usize
            } else {
                0
            };
            let union = cost_a + cost_b - shared;
            if union <= budget && best.is_none_or(|(cost, _, _)| union < cost) {
                best = Some((union, a, b));
            }
        }
        if best.is_some_and(|(cost, _, _)| cost == 0) {
            break;
        }
    }
    best.map(|(_, a, b)| vec![candidates[a].0, candidates[b].0])
}

impl RequestScheduler {
    pub fn share_snapshot_pages(&mut self, ranges: &[std::ops::Range<u32>]) -> Result<()> {
        if self.submitted != Fence(0)
            || self.slots.iter().any(|s| s.request.is_some())
            || ranges.len() != self.limits.snapshots as usize
            || ranges
                .iter()
                .any(|r| r.start > r.end || r.end > self.limits.pages)
        {
            return Err(
                "snapshot aliases must cover valid KV ids and be configured before admission"
                    .into(),
            );
        }
        self.snapshot_pages = ranges.to_vec();
        Ok(())
    }

    pub(super) fn available_snapshot(&self, keep_pages: usize) -> Option<SnapshotId> {
        let backing = self.free_backing();
        select_snapshots(
            &backing,
            &self.page_snapshot_refs,
            &self.snapshot_pages,
            &self.free_snapshots,
            keep_pages,
            1,
        )
        .and_then(|ids| ids.first().copied())
    }

    fn reserve_snapshot_id(&mut self, id: SnapshotId) {
        assert!(self.free_snapshots.remove(&id));
        if let Some(range) = self.snapshot_pages.get(id.0 as usize) {
            for page in range.clone() {
                self.free_pages.remove(&PageId(page));
                if self.page_snapshot_refs[page as usize] == 0 {
                    self.stats.snapshot_backing_pages += 1;
                }
                self.page_snapshot_refs[page as usize] += 1;
            }
        }
    }

    pub(super) fn reserve_snapshot(&mut self, keep_pages: usize) -> Option<SnapshotId> {
        let id = self.available_snapshot(keep_pages)?;
        self.reserve_snapshot_id(id);
        Some(id)
    }

    pub(super) fn quote_snapshots(
        &self,
        rows: usize,
        keep_pages: usize,
        rank: crate::cache::RetentionRank,
    ) -> Option<SnapshotQuote> {
        self.quote_snapshots_preserving(rows, rows, keep_pages, rank)
    }

    pub(super) fn quote_snapshots_preserving(
        &self,
        rows: usize,
        planned_rows: usize,
        keep_pages: usize,
        rank: crate::cache::RetentionRank,
    ) -> Option<SnapshotQuote> {
        if rows > planned_rows {
            return None;
        }
        if rows == planned_rows {
            return self.simulate_snapshot_quote(rows, keep_pages, rank, None);
        }
        let future = self.simulate_snapshot_quote(planned_rows, keep_pages, rank, None)?;
        let candidates = future.snapshots.iter().copied().collect();
        let mut current =
            self.simulate_snapshot_quote(rows, keep_pages, rank, Some(&candidates))?;
        current.planned_rows = planned_rows;
        current.witness = future.snapshots;
        Some(current)
    }

    fn simulate_snapshot_quote(
        &self,
        rows: usize,
        keep_pages: usize,
        rank: crate::cache::RetentionRank,
        allowed: Option<&BTreeSet<SnapshotId>>,
    ) -> Option<SnapshotQuote> {
        #[cfg(test)]
        STORAGE_SIMULATIONS.set(STORAGE_SIMULATIONS.get() + 1);
        let keep_pages = keep_pages.max(self.decode_growth_pages());
        let mut cache = self.cache.clone();
        let mut backing = self.free_backing();
        let mut free_snapshots = self.free_snapshots.clone();
        let mut refs = self.page_snapshot_refs.clone();
        let mut victims = Vec::new();
        loop {
            let candidates =
                allowed.map(|allowed| free_snapshots.intersection(allowed).copied().collect());
            if let Some(snapshots) = select_snapshots(
                &backing,
                &refs,
                &self.snapshot_pages,
                candidates.as_ref().unwrap_or(&free_snapshots),
                keep_pages,
                rows,
            ) {
                return Some(SnapshotQuote {
                    witness: snapshots.clone(),
                    snapshots,
                    victims,
                    planned_rows: rows,
                    keep_pages,
                    rank,
                });
            }
            let id = cache.eviction_candidate(Some(rank), true)?;
            cache.evict_checkpoint(id).ok()?;
            victims.push(id);
            let reusable = cache.advance_completed(self.completed).ok()?.reusable;
            // Eviction only ever returns pages, so the array built once above stays exact
            // and no per-round rescan of the free set is needed.
            for page in reusable.pages {
                if let Some(slot) = backing.get_mut(page.0 as usize) {
                    *slot = true;
                }
            }
            for snapshot in reusable.snapshots {
                free_snapshots.insert(snapshot);
                if let Some(range) = self.snapshot_pages.get(snapshot.0 as usize) {
                    for page in range.clone() {
                        refs[page as usize] -= 1;
                        if refs[page as usize] == 0 {
                            if let Some(slot) = backing.get_mut(page as usize) {
                                *slot = true;
                            }
                        }
                    }
                }
            }
        }
    }

    /// `backing[page]` is true exactly where the free set holds that id. Filling from the
    /// set takes one pass over the free pages; asking the set about every id takes a tree
    /// search per page, which grows with the pool.
    fn free_backing(&self) -> Vec<bool> {
        let mut backing = vec![false; self.limits.pages as usize];
        for page in &self.free_pages {
            if let Some(slot) = backing.get_mut(page.0 as usize) {
                *slot = true;
            }
        }
        backing
    }

    pub(super) fn apply_snapshot_quote(
        &mut self,
        quote: &SnapshotQuote,
    ) -> Result<Option<Vec<SnapshotId>>> {
        if self
            .quote_snapshots_preserving(
                quote.snapshots.len(),
                quote.planned_rows,
                quote.keep_pages,
                quote.rank,
            )
            .as_ref()
            != Some(quote)
        {
            return Ok(None);
        }
        let mut report = EvictionReport::default();
        for id in &quote.victims {
            let retired = self.cache.evict_checkpoint(*id)?;
            report.pages_scheduled += retired.pages.len();
            report.snapshots_scheduled += 1;
            report.retired.push(retired);
        }
        self.record_evictions(&report);
        self.stats.snapshot_evictions += report.retired.len() as u64;
        let update = self.cache.advance_completed(self.completed)?;
        self.apply_completion(update);
        for id in &quote.snapshots {
            self.reserve_snapshot_id(*id);
        }
        Ok(Some(quote.snapshots.clone()))
    }

    pub(super) fn retain_snapshot(&mut self) -> Result<Option<SnapshotId>> {
        loop {
            if let Some(id) = self.reserve_snapshot(0) {
                return Ok(Some(id));
            }
            if self.revoke_unused_anchor_guards() {
                continue;
            }
            let evicted = self.evict(EvictionTarget {
                pages: 0,
                snapshots: 1,
            });
            self.stats.snapshot_evictions += evicted.retired.len() as u64;
            let update = self.cache.advance_completed(self.completed)?;
            self.apply_completion(update);
            if evicted.retired.is_empty() {
                return Ok(None);
            }
        }
    }

    /// Backing that is free or becomes free once every retired owner and `extra` reaches
    /// its fence, with the snapshots those owners release.
    pub(super) fn pending_backing(&self, extra: &Reclaimable) -> (Vec<bool>, Vec<SnapshotId>) {
        let mut available = vec![false; self.limits.pages as usize];
        let mut retiring_refs = vec![0; available.len()];
        let mut snapshots = extra.snapshots.clone();
        for page in self.free_pages.iter().chain(&extra.pages) {
            available[page.0 as usize] = true;
        }
        for retirement in self.cache.pending_retirement() {
            for page in &retirement.pages {
                available[page.0 as usize] = true;
            }
            snapshots.push(retirement.snapshot);
        }
        for retirement in &self.retired {
            for page in &retirement.pages {
                available[page.0 as usize] = true;
            }
            snapshots.extend(retirement.snapshots.iter().flatten().copied());
        }
        for snapshot in &snapshots {
            if let Some(range) = self.snapshot_pages.get(snapshot.0 as usize) {
                for page in range.clone() {
                    retiring_refs[page as usize] += 1;
                }
            }
        }
        for (page, refs) in retiring_refs.into_iter().enumerate() {
            if refs > 0 && refs == self.page_snapshot_refs[page] {
                available[page] = true;
            }
        }
        (available, snapshots)
    }

    /// Project free backing after every already-retired owner reaches its fence.
    pub(super) fn pending_page_count(&self) -> usize {
        let (available, _) = self.pending_backing(&Reclaimable::default());
        available.into_iter().filter(|free| *free).count()
    }

    pub(super) fn pending_resources(&self, pages: usize, snapshots: usize) -> bool {
        if self.cache.pending_retirement().is_empty() && self.retired.is_empty() {
            return false;
        }
        let (backing, retiring) = self.pending_backing(&Reclaimable::default());
        self.projected_fits(&backing, &retiring, pages, snapshots)
    }

    /// Whether `pages` pages and `snapshots` snapshot ids are obtainable once `backing` is
    /// free and the `retiring` snapshots are released, honouring snapshot aliases.
    pub(super) fn projected_fits(
        &self,
        backing: &[bool],
        retiring: &[SnapshotId],
        pages: usize,
        snapshots: usize,
    ) -> bool {
        let mut candidates = self.free_snapshots.clone();
        let mut refs = self.page_snapshot_refs.clone();
        let retiring: BTreeSet<_> = retiring.iter().copied().collect();
        for snapshot in retiring {
            candidates.insert(snapshot);
            if let Some(range) = self.snapshot_pages.get(snapshot.0 as usize) {
                for page in range.clone() {
                    refs[page as usize] -= 1;
                }
            }
        }
        select_snapshots(
            backing,
            &refs,
            &self.snapshot_pages,
            &candidates,
            pages,
            snapshots,
        )
        .is_some()
    }

    pub(super) fn release_snapshot(&mut self, id: SnapshotId) {
        assert!(self.free_snapshots.insert(id), "snapshot released twice");
        if let Some(range) = self.snapshot_pages.get(id.0 as usize) {
            for page in range.clone() {
                let count = &mut self.page_snapshot_refs[page as usize];
                *count = count.checked_sub(1).expect("snapshot alias ownership");
                if *count == 0 {
                    self.stats.snapshot_backing_pages -= 1;
                    assert!(self.free_pages.insert(PageId(page)));
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn capture_rank() -> crate::cache::RetentionRank {
        crate::cache::RetentionRank { use_epoch: 2, role: crate::cache::RetentionRole::Message, consumed_tokens: 2 }
    }

    #[test]
    fn snapshot_selection_minimizes_newly_excluded_pages_before_id() {
        let mut s = super::super::tests::scheduler(1, 4, 2);
        s.share_snapshot_pages(&[0..3, 2..3]).unwrap();
        assert_eq!(s.reserve_snapshot(1), Some(SnapshotId(1)));
        assert_eq!(s.free_page_count(), 3);
    }

    #[test]
    fn joint_snapshot_quote_checks_union_cost_without_evicting_impossible_pairs() {
        let mut s = super::super::tests::scheduler(1, 4, 2);
        s.share_snapshot_pages(&[0..3, 1..4]).unwrap();
        let (backing, retiring) = s.pending_backing(&Reclaimable::default());
        assert!(s.projected_fits(&backing, &retiring, 1, 1));
        assert!(!s.projected_fits(&backing, &retiring, 1, 2));
        assert!(s.quote_snapshots(2, 1, capture_rank()).is_none());
        assert_eq!(s.free_page_count(), 4);
        assert_eq!(s.free_snapshot_count(), 2);
    }

    #[test]
    fn impossible_quote_preserves_removable_checkpoints_and_aliases() {
        let mut s = super::super::tests::scheduler(1, 4, 2);
        s.share_snapshot_pages(&[0..2, 2..4]).unwrap();
        let a = s.reserve_snapshot(0).unwrap();
        let b = s.reserve_snapshot(0).unwrap();
        let mut history = PrefixHistory::new([9; 32]);
        history.append(&[1, 2], &[]).unwrap();
        let mut parent = None;
        for (endpoint, snapshot) in [(1, a), (2, b)] {
            parent = Some(s.cache.publish(&history, &Publication {
                parent, consumed_tokens: endpoint, tail_pages: Vec::new(),
                snapshot: StateSnapshot { id: snapshot, consumed_tokens: endpoint }, ready_after: Fence(0),
            }).unwrap().id());
        }
        assert!(s.quote_snapshots(2, 1, capture_rank()).is_none());
        assert_eq!(s.cache.stats().published_checkpoints, 2);
        assert_eq!(s.free_page_count(), 0);
        assert_eq!(s.stats().snapshot_backing_pages, 4);
    }

    #[test]
    fn large_future_capture_witness_reserves_only_the_immediate_pair() {
        let mut s = super::super::tests::scheduler(1, 64, 64);
        s.share_snapshot_pages(&(0..64).map(|page| page..page + 1).collect::<Vec<_>>()).unwrap();
        let quote = s.quote_snapshots_preserving(2, 20, 44, capture_rank()).unwrap();
        assert_eq!(quote.snapshots, [SnapshotId(0), SnapshotId(1)]);
        assert_eq!(quote.witness.len(), 20);
        assert_eq!(s.apply_snapshot_quote(&quote).unwrap(), Some(vec![SnapshotId(0), SnapshotId(1)]));
        assert_eq!(s.free_page_count(), 62);
        assert_eq!(s.free_snapshot_count(), 62);
    }

    #[test]
    fn future_capture_witness_does_not_evict_checkpoints_needed_only_by_future_rows() {
        let mut s = super::super::tests::scheduler(1, 4, 3);
        let snapshot = SnapshotId(2);
        s.reserve_snapshot_id(snapshot);
        let mut history = PrefixHistory::new([9; 32]);
        history.append(&[1], &[]).unwrap();
        let id = s.cache.publish(&history, &Publication {
            parent: None, consumed_tokens: 1, tail_pages: Vec::new(),
            snapshot: StateSnapshot { id: snapshot, consumed_tokens: 1 }, ready_after: Fence(0),
        }).unwrap().id();
        let quote = s.quote_snapshots_preserving(2, 3, 4, capture_rank()).unwrap();
        assert_eq!(quote.snapshots, [SnapshotId(0), SnapshotId(1)]);
        assert!(quote.victims.is_empty());
        s.apply_snapshot_quote(&quote).unwrap().unwrap();
        assert!(s.cache.checkpoint(id).is_ok());
        assert_eq!(s.stats().snapshot_evictions, 0);
    }

    #[test]
    fn future_capture_witness_is_revalidated_before_any_current_eviction() {
        let mut s = super::super::tests::scheduler(1, 4, 3);
        let snapshot = SnapshotId(2);
        s.reserve_snapshot_id(snapshot);
        let mut history = PrefixHistory::new([9; 32]);
        history.append(&[1], &[]).unwrap();
        let id = s.cache.publish(&history, &Publication {
            parent: None, consumed_tokens: 1, tail_pages: Vec::new(),
            snapshot: StateSnapshot { id: snapshot, consumed_tokens: 1 }, ready_after: Fence(0),
        }).unwrap().id();
        let quote = s.quote_snapshots_preserving(1, 3, 4, capture_rank()).unwrap();
        s.cache.pin(id).unwrap();
        assert!(s.apply_snapshot_quote(&quote).unwrap().is_none());
        assert_eq!(s.free_snapshot_count(), 2);
        assert!(s.cache.checkpoint(id).is_ok());
    }

    #[test]
    fn optional_quote_revalidates_live_decode_headroom_before_reserving_aliases() {
        let mut s = super::super::tests::scheduler(1, 4, 1);
        s.share_snapshot_pages(&[3..4]).unwrap();
        let mut admission = super::super::tests::spec(60, 400);
        admission.growable_output = true;
        let handle = s.admit(admission).unwrap();
        let quote = s.quote_snapshots(1, 0, capture_rank()).unwrap();
        let request = s.request_mut(handle).unwrap();
        request.phase = RequestPhase::Decode;
        request.scheduled_consumed = 191;
        assert!(s.apply_snapshot_quote(&quote).unwrap().is_none());
        assert_eq!(s.free_page_count(), 2);
        assert_eq!(s.free_snapshot_count(), 1);
        assert_eq!(s.stats().snapshot_backing_pages, 0);
    }

    #[test]
    fn joint_quote_reserves_overlapping_rows_and_rejects_stale_ownership() {
        let mut s = super::super::tests::scheduler(1, 4, 3);
        s.share_snapshot_pages(&[0..3, 2..3, 2..3]).unwrap();
        let quote = s.quote_snapshots(2, 3, capture_rank()).unwrap();
        assert_eq!(quote.snapshots, [SnapshotId(1), SnapshotId(2)]);
        assert_eq!(s.apply_snapshot_quote(&quote).unwrap().unwrap(), quote.snapshots);
        assert_eq!(s.free_page_count(), 3);
        assert!(s.apply_snapshot_quote(&quote).unwrap().is_none());
        s.release_snapshot(SnapshotId(1));
        assert_eq!(s.free_page_count(), 3);
        s.release_snapshot(SnapshotId(2));
        assert_eq!(s.free_page_count(), 4);
    }

    #[test]
    fn optional_quote_cannot_wait_for_gpu_reads_or_replace_higher_demand_rank() {
        for (use_epoch, fence) in [(1, Fence(3)), (3, Fence(0)), (1, Fence(0))] {
            let mut s = super::super::tests::scheduler(1, 4, 1);
            let snapshot = s.reserve_snapshot(0).unwrap();
            let mut history = PrefixHistory::new([9; 32]);
            history.append(&[1], &[]).unwrap();
            let id = s.cache.publish_with_retention(&history, &Publication {
                parent: None, consumed_tokens: 1, tail_pages: Vec::new(),
                snapshot: StateSnapshot { id: snapshot, consumed_tokens: 1 }, ready_after: Fence(0),
            }, crate::cache::RetentionRole::Endpoint, use_epoch).unwrap().id();
            s.cache.mark_used(id, fence).unwrap();
            let quote = s.quote_snapshots(1, 4, capture_rank());
            if use_epoch == 1 && fence == Fence(0) {
                let quote = quote.unwrap();
                assert!(s.cache.checkpoint(id).is_ok());
                s.cache.pin(id).unwrap();
                assert!(s.apply_snapshot_quote(&quote).unwrap().is_none());
                assert!(s.cache.checkpoint(id).is_ok());
                s.cache.unpin(id).unwrap();
                assert_eq!(s.apply_snapshot_quote(&quote).unwrap(), Some(vec![snapshot]));
                assert!(s.cache.checkpoint(id).is_err());
            } else {
                assert!(quote.is_none());
                assert!(s.cache.checkpoint(id).is_ok());
            }
        }
    }

    #[test]
    fn all_two_snapshot_alias_layouts_preserve_page_ownership_and_reservation_budget() {
        fn check(s: &RequestScheduler, ranges: &[std::ops::Range<u32>; 2], live: &[bool; 2]) {
            let refs: Vec<_> = (0..4)
                .map(|page| (0..2).filter(|&id| live[id] && ranges[id].contains(&page)).count())
                .collect();
            for (page, &count) in refs.iter().enumerate() {
                assert_eq!(s.page_snapshot_refs[page] as usize, count);
                assert_eq!(s.free_pages.contains(&PageId(page as u32)), count == 0);
            }
            assert_eq!(s.stats().snapshot_backing_pages as usize,
                refs.iter().filter(|&&count| count != 0).count());
            for (id, &allocated) in live.iter().enumerate() {
                assert_eq!(s.free_snapshots.contains(&SnapshotId(id as u32)), !allocated);
            }
        }

        fn reserve(s: &mut RequestScheduler, ranges: &[std::ops::Range<u32>; 2],
                   live: &mut [bool; 2], keep: usize) {
            let eligible: Vec<_> = (0..2).filter(|&id| {
                !live[id] && (0..4).filter(|page| {
                    !ranges[id].contains(page)
                        && !(0..2).any(|owner| live[owner] && ranges[owner].contains(page))
                }).count() >= keep
            }).collect();
            match s.reserve_snapshot(keep) {
                Some(id) => {
                    assert!(eligible.contains(&(id.0 as usize)));
                    live[id.0 as usize] = true;
                    assert!(s.free_page_count() >= keep);
                }
                None => assert!(eligible.is_empty()),
            }
            check(s, ranges, live);
        }

        let ranges: Vec<_> = (0..=4).flat_map(|start| (start..=4).map(move |end| start..end)).collect();
        for a in &ranges {
            for b in &ranges {
                let aliases = [a.clone(), b.clone()];
                for first_keep in 0..=5 {
                    for second_keep in 0..=5 {
                        for release_first in 0..2 {
                            let mut s = super::super::tests::scheduler(1, 4, 2);
                            s.share_snapshot_pages(&aliases).unwrap();
                            let mut live = [false; 2];
                            reserve(&mut s, &aliases, &mut live, first_keep);
                            reserve(&mut s, &aliases, &mut live, second_keep);
                            if live[release_first] {
                                s.release_snapshot(SnapshotId(release_first as u32));
                                live[release_first] = false;
                                check(&s, &aliases, &live);
                            }
                            reserve(&mut s, &aliases, &mut live, first_keep);
                            for id in [1 - release_first, release_first] {
                                if live[id] {
                                    s.release_snapshot(SnapshotId(id as u32));
                                    live[id] = false;
                                    check(&s, &aliases, &live);
                                }
                            }
                            assert_eq!(s.free_page_count(), 4);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn snapshots_in_execution_alignment_padding_do_not_exclude_pages() {
        let mut s = super::super::tests::scheduler(1, 16, 1);
        s.share_snapshot_pages(&[0..0]).unwrap();
        let snapshot = s.reserve_snapshot(16).unwrap();
        assert_eq!(s.free_page_count(), 16);
        assert_eq!(s.stats().snapshot_backing_pages, 0);
        s.release_snapshot(snapshot);
        assert_eq!(s.free_page_count(), 16);
    }

    #[test]
    fn overlapping_snapshot_aliases_release_pages_only_after_the_last_owner() {
        let mut s = super::super::tests::scheduler(2, 16, 2);
        s.share_snapshot_pages(&[8..12, 10..14]).unwrap();
        let a = s.reserve_snapshot(0).unwrap();
        let b = s.reserve_snapshot(0).unwrap();
        assert_eq!(s.free_page_count(), 10);
        assert_eq!(s.stats().snapshot_backing_pages, 6);
        s.release_snapshot(a);
        assert_eq!(s.free_page_count(), 12);
        assert_eq!(s.stats().snapshot_backing_pages, 4);
        assert!(!s.free_pages.contains(&PageId(10)));
        s.release_snapshot(b);
        assert_eq!(s.free_page_count(), 16);
        assert_eq!(s.stats().snapshot_backing_pages, 0);
    }

    #[test]
    fn request_page_reservations_exclude_snapshot_aliases() {
        let mut s = super::super::tests::scheduler(2, 16, 2);
        s.share_snapshot_pages(&[0..8, 8..16]).unwrap();
        let handle = s.admit(super::super::tests::spec(64 * 12, 1)).unwrap();
        assert_eq!(s.free_snapshot_count(), 2);
        assert!(s.reserve_snapshot(0).is_none());
        s.cancel(handle).unwrap();
        assert!(s.reserve_snapshot(8).is_some());
        assert!(s.reserve_snapshot(1).is_none());
    }

    #[test]
    fn cancellation_keeps_snapshot_backing_until_the_last_gpu_fence() {
        let mut s = super::super::tests::scheduler(1, 16, 1);
        s.share_snapshot_pages(&[8..16]).unwrap();
        let handle = s.admit(super::super::tests::spec(64, 1)).unwrap();
        let snapshot = s.reserve_snapshot(0).unwrap();
        s.request_mut(handle).unwrap().terminal_snapshot = Some(snapshot);
        let step = s.plan().unwrap();
        s.commit(&step, Fence(1)).unwrap();
        s.cancel(handle).unwrap();
        assert_eq!(s.stats().snapshot_backing_pages, 8);
        assert!(s.reserve_snapshot(0).is_none());
        s.complete(
            Fence(1),
            &vec![CompletionRecord::default(); REQUEST_LANES],
            Default::default(),
        )
        .unwrap();
        assert_eq!(s.stats().snapshot_backing_pages, 0);
        assert_eq!(s.free_page_count(), 16);
        assert_eq!(s.free_snapshot_count(), 1);
    }
    #[test]
    fn page_pressure_compacts_aliases_with_free_snapshot_ids_and_preserves_fences() {
        for delayed in [false, true] {
            let mut s = super::super::tests::scheduler(2, 16, 4);
            s.share_snapshot_pages(&[8..12, 10..14, 0..0, 0..0])
                .unwrap();
            let mut history = PrefixHistory::new([9; 32]);
            history.append(&(0..128).collect::<Vec<_>>(), &[]).unwrap();
            let (a, b) = (SnapshotId(0), SnapshotId(1));
            s.reserve_snapshot_id(a);
            s.reserve_snapshot_id(b);
            let mut parent = None;
            for (endpoint, page, snapshot) in [(64, 0, a), (128, 1, b)] {
                s.free_pages.remove(&PageId(page));
                parent = Some(
                    s.cache
                        .publish(
                            &history,
                            &Publication {
                                parent,
                                consumed_tokens: endpoint,
                                tail_pages: vec![PageId(page)],
                                snapshot: StateSnapshot {
                                    id: snapshot,
                                    consumed_tokens: endpoint,
                                },
                                ready_after: Fence(0),
                            },
                        )
                        .unwrap()
                        .id(),
                );
            }
            let child = parent.unwrap();
            s.cache.pin(child).unwrap();
            if delayed {
                s.cache.mark_used(child, Fence(7)).unwrap();
            }
            let held = std::mem::take(&mut s.free_pages);
            assert_eq!(s.free_snapshot_count(), 2);
            let mut admission = super::super::tests::spec(64, 1);
            admission.prompt.fill(99);
            let result = s.admit(admission);
            if delayed {
                assert!(result.is_err());
                assert_eq!(s.free_page_count(), 0);
                assert_eq!(s.pending_page_count(), 2);
                assert_eq!(s.stats().snapshot_backing_pages, 6);
                let update = s.cache.advance_completed(Fence(7)).unwrap();
                s.apply_completion(update);
            } else {
                s.cancel(result.unwrap()).unwrap();
            }
            assert_eq!(s.free_page_count(), 2);
            assert_eq!(s.stats().snapshot_backing_pages, 4);
            assert_eq!(s.page_snapshot_refs[10], 1);
            assert_eq!(s.cache.page_chain(child).unwrap(), [PageId(0), PageId(1)]);
            s.free_pages.extend(held);
        }
    }

    #[test]
    fn decoder_growth_waits_for_pending_alias_reclamation() {
        let mut s = super::super::tests::scheduler(1, 16, 4);
        s.share_snapshot_pages(&[8..12, 10..14, 0..0, 0..0])
            .unwrap();
        let mut spec = super::super::tests::spec(60, 400);
        spec.growable_output = true;
        let handle = s.admit(spec).unwrap();
        let request = s.request_mut(handle).unwrap();
        request.phase = RequestPhase::Decode;
        request.scheduled_consumed = 127;
        let mut history = PrefixHistory::new([9; 32]);
        history.append(&vec![7; 128], &[]).unwrap();
        let (a, b) = (SnapshotId(0), SnapshotId(1));
        s.reserve_snapshot_id(a);
        s.reserve_snapshot_id(b);
        let mut parent = None;
        for (endpoint, page, snapshot) in [(64, 2, a), (128, 3, b)] {
            s.free_pages.remove(&PageId(page));
            parent = Some(
                s.cache
                    .publish(
                        &history,
                        &Publication {
                            parent,
                            consumed_tokens: endpoint,
                            tail_pages: vec![PageId(page)],
                            snapshot: StateSnapshot {
                                id: snapshot,
                                consumed_tokens: endpoint,
                            },
                            ready_after: Fence(0),
                        },
                    )
                    .unwrap()
                    .id(),
            );
        }
        let child = parent.unwrap();
        s.cache.pin(child).unwrap();
        s.cache.mark_used(child, Fence(7)).unwrap();
        let held = std::mem::take(&mut s.free_pages);
        s.refill_reservations().unwrap();
        assert_eq!(s.stats().preemptions, 0);
        assert!(!s.fits_next_rounds(s.request(handle).unwrap()));
        assert_eq!(s.free_page_count(), 0);
        assert_eq!(s.pending_page_count(), 2);
        let update = s.cache.advance_completed(Fence(7)).unwrap();
        s.apply_completion(update);
        s.refill_reservations().unwrap();
        assert!(s.fits_next_rounds(s.request(handle).unwrap()));
        assert_eq!(s.stats().preemptions, 0);
        s.free_pages.extend(held);
    }

    // ------------------------------------------------------------ quote parity
    //
    // Independent implementations of the selection and quote rules, written in the direct
    // form: clone the free-page set, ask it about every page id, and enumerate candidate
    // subsets. The tests below compare the scheduler's results against these, because
    // `apply_snapshot_quote` re-quotes a stored quote and compares it, so agreement with
    // the selection rule is a runtime contract, not only a test convenience.

    fn reference_select(
        backing: &[bool],
        protected_refs: &[u32],
        ranges: &[std::ops::Range<u32>],
        candidates: &BTreeSet<SnapshotId>,
        keep_pages: usize,
        count: usize,
    ) -> Option<Vec<SnapshotId>> {
        let available = backing.iter().filter(|free| **free).count();
        if available < keep_pages {
            return None;
        }
        if count == 0 {
            return Some(Vec::new());
        }
        let candidates: Vec<_> = candidates
            .iter()
            .copied()
            .filter_map(|id| {
                let mut cost = Vec::new();
                if let Some(range) = ranges.get(id.0 as usize) {
                    for page in range.clone() {
                        if backing[page as usize] {
                            cost.push(page as usize);
                        } else if protected_refs[page as usize] == 0 {
                            return None;
                        }
                    }
                }
                Some((id, cost))
            })
            .collect();
        if count > 2 {
            let mut selected = BTreeSet::<SnapshotId>::new();
            let mut excluded = BTreeSet::<usize>::new();
            for _ in 0..count {
                let (id, pages) = candidates
                    .iter()
                    .filter(|(id, _)| !selected.contains(id))
                    .min_by_key(|(id, pages)| {
                        (
                            pages
                                .iter()
                                .filter(|page| !excluded.contains(*page))
                                .count(),
                            *id,
                        )
                    })?;
                excluded.extend(pages);
                if excluded.len() > available - keep_pages {
                    return None;
                }
                selected.insert(*id);
            }
            return Some(selected.into_iter().collect());
        }
        fn choose(
            candidates: &[(SnapshotId, Vec<usize>)],
            start: usize,
            count: usize,
            selected: &mut Vec<SnapshotId>,
            excluded: &BTreeSet<usize>,
            budget: usize,
            best: &mut Option<(usize, Vec<SnapshotId>)>,
        ) {
            if excluded.len() > budget {
                return;
            }
            if best.as_ref().is_some_and(|best| best.0 == 0) {
                return;
            }
            if count == 0 {
                let candidate = (excluded.len(), selected.clone());
                if best.as_ref().is_none_or(|best| candidate < *best) {
                    *best = Some(candidate);
                }
                return;
            }
            if candidates.len().saturating_sub(start) < count {
                return;
            }
            for index in start..=candidates.len() - count {
                let (id, pages) = &candidates[index];
                let mut union = excluded.clone();
                union.extend(pages);
                if best.as_ref().is_some_and(|best| union.len() > best.0) {
                    continue;
                }
                selected.push(*id);
                choose(
                    candidates,
                    index + 1,
                    count - 1,
                    selected,
                    &union,
                    budget,
                    best,
                );
                selected.pop();
            }
        }
        let mut best = None;
        choose(
            &candidates,
            0,
            count,
            &mut Vec::new(),
            &BTreeSet::new(),
            available - keep_pages,
            &mut best,
        );
        best.map(|(_, selected)| selected)
    }

    fn reference_simulate(
        s: &RequestScheduler,
        rows: usize,
        keep_pages: usize,
        rank: crate::cache::RetentionRank,
        allowed: Option<&BTreeSet<SnapshotId>>,
    ) -> Option<SnapshotQuote> {
        let keep_pages = keep_pages.max(s.decode_growth_pages());
        let mut cache = s.cache.clone();
        let mut free_pages = s.free_pages.clone();
        let mut free_snapshots = s.free_snapshots.clone();
        let mut refs = s.page_snapshot_refs.clone();
        let mut victims = Vec::new();
        loop {
            let backing: Vec<_> = (0..s.limits.pages)
                .map(|page| free_pages.contains(&PageId(page)))
                .collect();
            let candidates =
                allowed.map(|allowed| free_snapshots.intersection(allowed).copied().collect());
            if let Some(snapshots) = reference_select(
                &backing,
                &refs,
                &s.snapshot_pages,
                candidates.as_ref().unwrap_or(&free_snapshots),
                keep_pages,
                rows,
            ) {
                return Some(SnapshotQuote {
                    witness: snapshots.clone(),
                    snapshots,
                    victims,
                    planned_rows: rows,
                    keep_pages,
                    rank,
                });
            }
            let id = cache.eviction_candidate(Some(rank), true)?;
            cache.evict_checkpoint(id).ok()?;
            victims.push(id);
            let reusable = cache.advance_completed(s.completed).ok()?.reusable;
            free_pages.extend(reusable.pages);
            for snapshot in reusable.snapshots {
                free_snapshots.insert(snapshot);
                if let Some(range) = s.snapshot_pages.get(snapshot.0 as usize) {
                    for page in range.clone() {
                        refs[page as usize] -= 1;
                        if refs[page as usize] == 0 {
                            free_pages.insert(PageId(page));
                        }
                    }
                }
            }
        }
    }

    fn reference_quote_preserving(
        s: &RequestScheduler,
        rows: usize,
        planned_rows: usize,
        keep_pages: usize,
        rank: crate::cache::RetentionRank,
    ) -> Option<SnapshotQuote> {
        if rows > planned_rows {
            return None;
        }
        if rows == planned_rows {
            return reference_simulate(s, rows, keep_pages, rank, None);
        }
        let future = reference_simulate(s, planned_rows, keep_pages, rank, None)?;
        let candidates = future.snapshots.iter().copied().collect();
        let mut current = reference_simulate(s, rows, keep_pages, rank, Some(&candidates))?;
        current.planned_rows = planned_rows;
        current.witness = future.snapshots;
        Some(current)
    }

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 33
    }

    /// Every quote must match the reference implementation, including states with an empty
    /// free set, a full cache, and fragmented free pages, and including quotes that evict.
    #[test]
    fn snapshot_quote_matches_reference() {
        let rng = &mut 0x5eed_1234_9abc_def0u64;
        let (mut quoted, mut evicted, mut compared) = (0u32, 0u32, 0u32);
        for case in 0..120 {
            let pages = if case % 3 == 0 { 4_096u32 } else { 256 };
            let snapshots = if case % 3 == 0 { 32u32 } else { 6 };
            let mut s = super::super::tests::scheduler(1, pages, snapshots);
            let aliases: Vec<_> = (0..snapshots)
                .map(|_| {
                    let start = (lcg(rng) % u64::from(pages)) as u32;
                    let width = (lcg(rng) % 40) as u32;
                    start..(start + width).min(pages)
                })
                .collect();
            s.share_snapshot_pages(&aliases).unwrap();
            let mut history = PrefixHistory::new([9; 32]);
            history
                .append(&(0..512u32).collect::<Vec<_>>(), &[])
                .unwrap();
            let mut parent = None;
            for step in 0..(lcg(rng) % 4) {
                let free: Vec<_> = s.free_snapshots.iter().copied().collect();
                if free.is_empty() {
                    break;
                }
                let pick = free[(lcg(rng) as usize) % free.len()];
                s.reserve_snapshot_id(pick);
                let endpoint = (step + 1) * 64;
                parent = Some(
                    s.cache
                        .publish(
                            &history,
                            &Publication {
                                parent,
                                consumed_tokens: endpoint,
                                tail_pages: vec![PageId(step as u32 % pages)],
                                snapshot: StateSnapshot {
                                    id: pick,
                                    consumed_tokens: endpoint,
                                },
                                ready_after: Fence(0),
                            },
                        )
                        .unwrap()
                        .id(),
                );
            }
            let taken: Vec<PageId> = s
                .free_pages
                .iter()
                .copied()
                .filter(|_| lcg(rng) % 7 == 0)
                .collect();
            for page in &taken {
                s.free_pages.remove(page);
            }
            for _ in 0..(lcg(rng) % 8) {
                let page = PageId((lcg(rng) % u64::from(pages)) as u32);
                s.free_pages.insert(page);
            }
            if case % 11 == 0 {
                s.free_pages.clear();
            }
            if case % 13 == 0 {
                s.free_snapshots.clear();
                s.free_pages.clear();
            }
            for rows in 0..=4usize {
                for planned in [rows, rows + 1] {
                    for keep in [0usize, 1, 4] {
                        let rank = crate::cache::RetentionRank {
                            use_epoch: 3,
                            role: crate::cache::RetentionRole::Message,
                            consumed_tokens: 64,
                        };
                        let actual = s.quote_snapshots_preserving(rows, planned, keep, rank);
                        let reference = reference_quote_preserving(&s, rows, planned, keep, rank);
                        assert_eq!(
                            actual, reference,
                            "case {case} rows {rows} planned {planned} keep {keep}"
                        );
                        compared += 1;
                        if let Some(quote) = &actual {
                            quoted += 1;
                            evicted += u32::from(!quote.victims.is_empty());
                        }
                    }
                }
            }
        }
        println!(
            "SNAPSHOT_QUOTE_PARITY compared={compared} quoted={quoted} with_victims={evicted}"
        );
        assert!(
            quoted > compared / 10,
            "too few successful quotes to prove equality"
        );
        assert!(evicted > 0, "no state exercised the eviction path");
    }

    /// The selection paths alone, on random pools, against the enumerating reference.
    #[test]
    fn snapshot_selection_matches_reference() {
        let rng = &mut 0xfeed_0bad_c0de_0001u64;
        let (mut compared, mut picked) = (0u32, 0u32);
        for case in 0..300 {
            let pages = 1 + (lcg(rng) % 300) as usize;
            let snapshots = 1 + (lcg(rng) % 12) as usize;
            let backing: Vec<bool> = (0..pages).map(|_| lcg(rng) % 3 != 0).collect();
            let refs: Vec<u32> = (0..pages).map(|_| (lcg(rng) % 3) as u32).collect();
            let ranges: Vec<std::ops::Range<u32>> = (0..snapshots)
                .map(|_| {
                    let start = (lcg(rng) % pages as u64) as u32;
                    start..(start + (lcg(rng) % 40) as u32).min(pages as u32)
                })
                .collect();
            let candidates: BTreeSet<SnapshotId> = (0..snapshots)
                .filter(|_| lcg(rng) % 4 != 0)
                .map(|id| SnapshotId(id as u32))
                .collect();
            for count in 0..=4usize {
                for keep in [0usize, 1, 7, pages] {
                    let actual =
                        select_snapshots(&backing, &refs, &ranges, &candidates, keep, count);
                    let reference =
                        reference_select(&backing, &refs, &ranges, &candidates, keep, count);
                    assert_eq!(actual, reference, "case {case} count {count} keep {keep}");
                    compared += 1;
                    picked += u32::from(actual.as_ref().is_some_and(|ids| !ids.is_empty()));
                }
            }
        }
        println!("SNAPSHOT_SELECT_PARITY compared={compared} picked={picked}");
        assert!(picked > compared / 20);
    }

    /// Two rows on pools where candidates' alias windows overlap heavily.
    #[test]
    fn two_row_selection_matches_reference() {
        let rng = &mut 0x00d2_c0de_1234_5678u64;
        let (mut compared, mut picked) = (0u32, 0u32);
        for case in 0..30 {
            let pages = 200 + (lcg(rng) % 800) as usize;
            let snapshots = 40 + (lcg(rng) % 160) as usize;
            let backing: Vec<bool> = (0..pages).map(|_| lcg(rng) % 4 != 0).collect();
            let refs: Vec<u32> = (0..pages).map(|_| (lcg(rng) % 2) as u32).collect();
            let stride = 1 + (lcg(rng) % 12) as u32;
            let ranges: Vec<std::ops::Range<u32>> = (0..snapshots)
                .map(|row| {
                    let start = (row as u32 * stride) % pages as u32;
                    start..(start + 8 + (lcg(rng) % 24) as u32).min(pages as u32)
                })
                .collect();
            let candidates: BTreeSet<SnapshotId> = (0..snapshots)
                .filter(|_| lcg(rng) % 5 != 0)
                .map(|id| SnapshotId(id as u32))
                .collect();
            for keep in [0usize, 1, 40, 200, pages] {
                let actual = select_snapshots(&backing, &refs, &ranges, &candidates, keep, 2);
                let reference = reference_select(&backing, &refs, &ranges, &candidates, keep, 2);
                assert_eq!(actual, reference, "case {case} keep {keep}");
                compared += 1;
                picked += u32::from(actual.is_some());
            }
        }
        println!("SNAPSHOT_TWO_ROW_PARITY compared={compared} picked={picked}");
        assert!(
            picked > compared / 4,
            "too few two-row selections to prove equality"
        );
    }

    #[test]
    fn overlapping_snapshot_selection_matches_reference() {
        let rng = &mut 0x51ec_7100_abc0_1234u64;
        let mut compared = 0;
        for case in 0..500 {
            let pages = (lcg(rng) % 25) as usize;
            let snapshots = (lcg(rng) % 10) as usize;
            let backing: Vec<bool> = (0..pages)
                .map(|_| lcg(rng) % 3 != 0 || case % 3 == 0)
                .collect();
            let refs: Vec<u32> = (0..pages).map(|_| (lcg(rng) % 3) as u32).collect();
            let ranges: Vec<_> = (0..snapshots)
                .map(|_| {
                    let start = (lcg(rng) % (pages + 1) as u64) as u32;
                    let end = (lcg(rng) % (pages + 1) as u64) as u32;
                    start.min(end)..start.max(end)
                })
                .collect();
            // Missing ranges represent rows with no shared page backing.
            let candidates: BTreeSet<_> = (0..snapshots + case % 3)
                .filter(|_| lcg(rng) % 4 != 0)
                .map(|id| SnapshotId(id as u32))
                .collect();
            for count in 0..=5 {
                for keep in 0..=pages + 1 {
                    assert_eq!(
                        select_snapshots(&backing, &refs, &ranges, &candidates, keep, count),
                        reference_select(&backing, &refs, &ranges, &candidates, keep, count),
                        "case {case} count {count} keep {keep}"
                    );
                    compared += 1;
                }
            }
        }
        println!("OVERLAPPING_SNAPSHOT_PARITY compared={compared}");
    }

    #[test]
    fn nested_snapshot_selection_bounds_page_visits() {
        const PAGES: usize = 256;
        const ALIASES: usize = 128;
        let backing = vec![true; PAGES];
        let refs = vec![0; PAGES];
        let ranges: Vec<_> = (0..ALIASES).map(|id| id as u32..PAGES as u32).collect();
        let candidates = (0..ALIASES).map(|id| SnapshotId(id as u32)).collect();
        for count in [2, 4] {
            SNAPSHOT_PAGE_VISITS.set(0);
            let selected = select_snapshots(&backing, &refs, &ranges, &candidates, 0, count);
            let visits = SNAPSHOT_PAGE_VISITS.get();
            let expected = (ALIASES - count..ALIASES)
                .map(|id| SnapshotId(id as u32))
                .collect();
            assert_eq!(selected, Some(expected));
            // One prefix scan, then at most one selected range and one pool scan per
            // greedy row. Page work must not grow with the sum of pairwise overlaps.
            let bound = PAGES * (1 + 2 * count);
            println!("NESTED_SNAPSHOT_WORK count={count} visits={visits} bound={bound}");
            assert!(
                visits <= bound,
                "count {count}: {visits} page visits exceed {bound}"
            );
        }
    }

    #[test]
    fn large_overlapping_alias_pool_selects_lowest_ids() {
        const ALIASES: u32 = 65_537;
        const PAGES: u32 = 65_536;
        let backing = vec![true; PAGES as usize];
        let refs = vec![0; PAGES as usize];
        let ranges = vec![0..PAGES; ALIASES as usize];
        let candidates = (0..ALIASES).map(SnapshotId).collect();
        assert!(u64::from(ALIASES) * u64::from(PAGES) > u64::from(u32::MAX));
        assert_eq!(
            select_snapshots(&backing, &refs, &ranges, &candidates, 0, 2),
            Some(vec![SnapshotId(0), SnapshotId(1)])
        );
    }

    /// Cache benchmark fixture: 202,183 pages of 128 KV tokens and 2,457 snapshot rows.
    /// Aliases use contiguous windows with a 32-page overlap between neighbours,
    /// matching the layout shape constructed by `Layout`.
    const FIXTURE_PAGES: u32 = 202_183;
    const FIXTURE_SNAPSHOTS: u32 = 2_457;

    fn fixture_scheduler() -> RequestScheduler {
        let mut scheduler = RequestScheduler::new(
            [7; 32],
            SchedulerLimits {
                request_slots: 16,
                max_active: 1,
                max_sequence_tokens: 1_048_576,
                prefill_chunk_size: 8192,
                pages: FIXTURE_PAGES,
                snapshots: FIXTURE_SNAPSHOTS,
            },
            SchedulerCapabilities {
                completion_token_capacity: 16,
                vocabulary: 248_320,
                page_tokens: 128,
                vision: crate::model::VisionCapabilities {
                    feature_slots: 4,
                    features_per_request: 16384,
                    max_images: 4,
                    patches_per_feature: 4,
                    patch_capacities: vec![256, 1024, 4096, 16384],
                },
                draft_pages_per_request: 1,
                completion_inputs: vec![],
                speculative_allocations: vec![],
                state_rows_per_request: None,
            },
        )
        .unwrap();
        scheduler
            .measured_graph_durations(&super::super::tests::duration_graphs(1))
            .unwrap();
        let stride = FIXTURE_PAGES / FIXTURE_SNAPSHOTS;
        let aliases: Vec<_> = (0..FIXTURE_SNAPSHOTS)
            .map(|row| {
                let start = row * stride;
                start..(start + stride + 32).min(FIXTURE_PAGES)
            })
            .collect();
        scheduler.share_snapshot_pages(&aliases).unwrap();
        scheduler
    }

    /// Time quote and admission operations at the 202,183-page, 2,457-row fixture.
    /// Report medians of five optimized samples and one timed reference call per operation.
    /// Run explicitly in release mode with output enabled:
    /// `cargo test -p flare-helios --release --lib -- --ignored --nocapture snapshot_quote_cost`
    #[test]
    #[ignore = "measures cost; run explicitly in release"]
    fn snapshot_quote_cost_at_production_geometry() {
        let mut scheduler = fixture_scheduler();
        scheduler.reserve_snapshot(0).expect("anchor snapshot");
        assert_eq!(scheduler.free_page_count() + 114, FIXTURE_PAGES as usize);
        let rank = crate::cache::RetentionRank {
            use_epoch: 3,
            role: crate::cache::RetentionRole::Anchor,
            consumed_tokens: 16_213,
        };
        let rounds = 5;
        let median = |mut samples: Vec<std::time::Duration>| {
            samples.sort_unstable();
            samples[samples.len() / 2]
        };
        let mut clones = Vec::new();
        let mut scans = Vec::new();
        let mut single = Vec::new();
        let mut pair = Vec::new();
        let mut triple = Vec::new();
        let mut available = Vec::new();
        for _ in 0..rounds {
            let start = std::time::Instant::now();
            let copy = scheduler.free_pages.clone();
            clones.push(start.elapsed());
            std::hint::black_box(copy);

            let start = std::time::Instant::now();
            let backing: Vec<_> = (0..scheduler.limits.pages)
                .map(|page| scheduler.free_pages.contains(&PageId(page)))
                .collect();
            scans.push(start.elapsed());
            std::hint::black_box(backing);

            let start = std::time::Instant::now();
            let quote = scheduler.quote_snapshots_preserving(1, 1, 0, rank);
            single.push(start.elapsed());
            assert!(quote.is_some());

            let start = std::time::Instant::now();
            let quote = scheduler.quote_snapshots_preserving(1, 2, 0, rank);
            pair.push(start.elapsed());
            assert!(quote.is_some());

            let start = std::time::Instant::now();
            let quote = scheduler.quote_snapshots_preserving(1, 3, 0, rank);
            triple.push(start.elapsed());
            assert!(quote.is_some());

            let start = std::time::Instant::now();
            let id = scheduler.available_snapshot(0);
            available.push(start.elapsed());
            assert!(id.is_some());
        }
        let us = |duration: std::time::Duration| duration.as_secs_f64() * 1e6;
        println!(
            "SNAPSHOT_QUOTE_COST pages={} snapshots={} free_pages={}\n\
             free_pages.clone   median {:9.1} us\n\
             full backing scan  median {:9.1} us\n\
             quote rows=1/1     median {:9.1} us\n\
             quote rows=1/2     median {:9.1} us\n\
             quote rows=1/3     median {:9.1} us\n\
             available_snapshot median {:9.1} us",
            scheduler.limits.pages,
            scheduler.limits.snapshots,
            scheduler.free_page_count(),
            us(median(clones)),
            us(median(scans)),
            us(median(single)),
            us(median(pair)),
            us(median(triple)),
            us(median(available)),
        );

        // The same calls through the reference implementation, once each, for comparison.
        // Its two-row search takes seconds at this geometry.
        let start = std::time::Instant::now();
        assert!(reference_quote_preserving(&scheduler, 1, 1, 0, rank).is_some());
        let reference_single = start.elapsed();
        let start = std::time::Instant::now();
        assert!(reference_quote_preserving(&scheduler, 1, 2, 0, rank).is_some());
        let reference_pair = start.elapsed();
        let start = std::time::Instant::now();
        assert!(reference_quote_preserving(&scheduler, 1, 3, 0, rank).is_some());
        let reference_triple = start.elapsed();
        let start = std::time::Instant::now();
        let backing: Vec<_> = (0..scheduler.limits.pages)
            .map(|page| scheduler.free_pages.contains(&PageId(page)))
            .collect();
        let id = reference_select(
            &backing,
            &scheduler.page_snapshot_refs,
            &scheduler.snapshot_pages,
            &scheduler.free_snapshots,
            0,
            1,
        );
        let reference_available = start.elapsed();
        assert!(id.is_some());
        println!(
            "SNAPSHOT_QUOTE_REFERENCE quote rows=1/1 {:9.1} us\n\
             quote rows=1/2     {:9.1} us\n\
             quote rows=1/3     {:9.1} us\n\
             available_snapshot {:9.1} us",
            us(reference_single),
            us(reference_pair),
            us(reference_triple),
            us(reference_available),
        );
    }

    /// The quoted results themselves at production geometry, against the reference
    /// implementation. Ignored because the reference pair search takes seconds here.
    #[test]
    #[ignore = "reference search takes seconds at production geometry"]
    fn production_geometry_quote_matches_reference() {
        let mut scheduler = fixture_scheduler();
        scheduler.reserve_snapshot(0).expect("anchor snapshot");
        let rank = crate::cache::RetentionRank {
            use_epoch: 3,
            role: crate::cache::RetentionRole::Anchor,
            consumed_tokens: 16_213,
        };
        for (rows, planned) in [(1usize, 1usize), (1, 2), (1, 3)] {
            assert_eq!(
                scheduler.quote_snapshots_preserving(rows, planned, 0, rank),
                reference_quote_preserving(&scheduler, rows, planned, 0, rank),
                "rows {rows} planned {planned}"
            );
        }
        let backing = scheduler.free_backing();
        assert_eq!(
            scheduler.available_snapshot(0),
            reference_select(
                &backing,
                &scheduler.page_snapshot_refs,
                &scheduler.snapshot_pages,
                &scheduler.free_snapshots,
                0,
                1,
            )
            .and_then(|ids| ids.first().copied())
        );
    }
}
