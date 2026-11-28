/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exact-prefix capture selection and request-local predicted work accounting.

use super::*;

pub(super) type SelectionCache = HashMap<(RequestHandle, bool), (CheckpointAccounting, bool, bool)>;

pub(super) struct PlanningScope<'a>(&'a std::cell::RefCell<Option<SelectionCache>>);

impl Drop for PlanningScope<'_> {
    fn drop(&mut self) {
        *self.0.borrow_mut() = None;
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckpointAccounting {
    pub demand_epoch: u64,
    pub initial_prefill_ns: Option<u64>,
    pub productive_decode_ns: u64,
    pub extra_spent_ns: u64,
    baseline_interval: Option<(usize, usize)>,
    window_ends: Vec<usize>,
    reserved_ns: u64,
    cuts: Vec<Cut>,
}

impl CheckpointAccounting {
    pub(super) fn for_epoch(demand_epoch: u64, consumed: usize, prompt: usize) -> Self {
        Self {
            demand_epoch,
            baseline_interval: Some((consumed, prompt)),
            ..Self::default()
        }
    }
    pub(super) fn retain_window_ends(&mut self, ends: impl Iterator<Item = usize>) {
        self.window_ends.extend(ends);
        self.window_ends.sort_unstable();
        self.window_ends.dedup();
    }

    pub(super) fn clear_reservations(&mut self) {
        self.cuts.clear();
        self.reserved_ns = 0;
    }

    pub(super) fn refund_copy(&mut self, cost: u64) {
        self.reserved_ns = self.reserved_ns.saturating_sub(cost);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cut {
    endpoint: usize,
    role: RetentionRole,
    split_ns: u64,
    copy_ns: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ArmedCapture {
    pub snapshot: SnapshotId,
    pub endpoint: u64,
    pub role: RetentionRole,
    pub copy_ns: u64,
}

#[derive(Clone, Debug)]
pub(super) struct CapturePlan {
    handle: RequestHandle,
    accounting: CheckpointAccounting,
    anchor: usize,
    cut: Option<Cut>,
    quote: Option<cache_memory::SnapshotQuote>,
    reserve_guard: bool,
    unknown: bool,
    budget: bool,
}

fn allowance(work: u64, fraction: f64) -> u64 {
    ((work as f64) * fraction).floor().min(u64::MAX as f64) as u64
}

#[derive(Clone, Copy, Debug)]
struct PrefillCatalogCost {
    capacity: usize,
    has_terminal: bool,
    non_terminal_ns: Option<u64>,
    terminal_ns: Option<u64>,
}

/// Whether `work` tokens for `nanos` cost less per token than `best`; a tie prefers more work.
fn cheaper(work: usize, nanos: u64, best: Option<(usize, u64)>) -> bool {
    best.is_none_or(|(old_work, old_ns)| {
        u128::from(nanos) * (old_work as u128) < u128::from(old_ns) * work as u128
            || (u128::from(nanos) * old_work as u128 == u128::from(old_ns) * work as u128
                && work > old_work)
    })
}

impl LiveRequest {
    pub(super) fn checkpoint_anchor(&self) -> usize {
        self.message_boundaries
            .iter()
            .rev()
            .copied()
            .find(|&end| self.checkpoint_endpoint_eligible(end))
            .unwrap_or(self.prompt_tokens)
    }

    fn checkpoint_endpoint_eligible(&self, end: usize) -> bool {
        !self
            .prefill_spans
            .iter()
            .any(|span| span.start < end && end < span.end)
            && (end == self.prompt_tokens
                || self.checkpoints.window_ends.binary_search(&end).is_err())
    }
}

impl RequestScheduler {
    pub(super) fn checkpoint_planning_scope(&self) -> PlanningScope<'_> {
        *self.checkpoint_selections.borrow_mut() = Some(HashMap::new());
        PlanningScope(&self.checkpoint_selections)
    }
    pub(super) fn revoke_unused_anchor_guards(&mut self) -> bool {
        let handle = self
            .slots
            .iter()
            .filter_map(|slot| slot.request.as_ref())
            .filter(|request| request.anchor_guard.is_some())
            .min_by_key(|request| {
                (
                    request.checkpoints.demand_epoch,
                    request.handle.slot,
                    request.handle.generation,
                )
            })
            .map(|request| request.handle);
        let Some(handle) = handle else {
            return false;
        };
        let guard = self
            .request_mut(handle)
            .unwrap()
            .anchor_guard
            .take()
            .unwrap();
        self.release_snapshot(guard);
        true
    }

    /// Each prefill capacity this request can use, with its measured non-terminal and terminal
    /// costs when known and nonzero. Selection reads the costs once instead of once per chunk of
    /// every interval.
    fn prefill_catalog_costs(&self, request: &LiveRequest) -> Vec<PrefillCatalogCost> {
        let catalog = if request.prompt_logprobs {
            &self.scored_prefill_catalog
        } else {
            &self.prefill_catalog
        };
        let has_terminal_catalog =
            !request.prompt_logprobs && !self.terminal_prefill_catalog.is_empty();
        catalog
            .iter()
            .map(|&capacity| {
                // Prefill dispatch retains measured costs after expiry; aging alone
                // does not make its predicted work unknown.
                let prefill_ns = self
                    .costs
                    .duration(request.prefill_kind(), 1, capacity)
                    .filter(|&nanos| nanos != 0);
                if !has_terminal_catalog {
                    return PrefillCatalogCost {
                        capacity,
                        has_terminal: false,
                        non_terminal_ns: prefill_ns,
                        terminal_ns: prefill_ns,
                    };
                }
                let has_terminal = self.terminal_prefill_catalog.binary_search(&capacity).is_ok();
                let terminal_ns = if has_terminal {
                    self.costs
                        .duration(GraphKind::TerminalPrefill, 1, capacity)
                        .filter(|&nanos| nanos != 0)
                        .or(prefill_ns)
                } else {
                    prefill_ns
                };
                let non_terminal_ns = prefill_ns.or(terminal_ns);
                PrefillCatalogCost {
                    capacity,
                    has_terminal,
                    non_terminal_ns,
                    terminal_ns,
                }
            })
            .collect()
    }

    fn interval_prefill_ns(
        request: &LiveRequest,
        catalog: &[PrefillCatalogCost],
        terminal_window: usize,
        mut start: usize,
        end: usize,
    ) -> Option<u64> {
        let has_terminal_catalog = catalog.iter().any(|entry| entry.has_terminal);
        let uniform = terminal_window == 0 && !has_terminal_catalog;
        // With every cost known, the capacity cheapest per token is chosen whole at each
        // position whose chunk ends by the next window end, span start or interval end: a
        // smaller one fits unchanged, and a larger one only does worse when cut short.
        let stride = catalog
            .iter()
            .try_fold(None, |best, entry| {
                let nanos = entry.non_terminal_ns?;
                Some(if cheaper(entry.capacity, nanos, best) {
                    Some((entry.capacity, nanos))
                } else {
                    best
                })
            })
            .flatten();
        let mut total = 0u64;
        while start < end {
            let limit = request
                .checkpoints
                .window_ends
                .iter()
                .copied()
                .filter(|&limit| start < limit)
                .min()
                .unwrap_or(end)
                .min(end);
            if let Some((capacity, nanos)) = stride
                && !request
                    .prefill_spans
                    .iter()
                    .any(|span| span.start <= start && start < span.end)
            {
                let mut bound = request
                    .prefill_spans
                    .iter()
                    .map(|span| span.start)
                    .filter(|&span_start| start < span_start)
                    .fold(limit, usize::min);
                if !uniform {
                    // Stop before the terminal chunk and any penultimate chunk inside the
                    // SWA tail protection window so those chunks price with `terminal_ns`.
                    bound = bound.min(end.saturating_sub(terminal_window + 1));
                }
                let chunks = bound.saturating_sub(start) / capacity;
                if chunks > 0 {
                    #[cfg(test)]
                    tests::STRIDE_RUNS.set(tests::STRIDE_RUNS.get() + 1);
                    total = total.checked_add(nanos.checked_mul(chunks as u64)?)?;
                    start += chunks * capacity;
                    continue;
                }
            }
            let remaining = end.saturating_sub(start);
            let step_remaining = limit.saturating_sub(start);
            let target_capacity = catalog
                .iter()
                .map(|entry| entry.capacity)
                .find(|&capacity| capacity >= step_remaining)
                .or_else(|| catalog.last().map(|entry| entry.capacity));
            let terminal_only = has_terminal_catalog
                && limit >= end
                && catalog.last().is_some_and(|last| {
                    let mut through = start.saturating_add(last.capacity).min(limit);
                    if let Some(span) = request
                        .prefill_spans
                        .iter()
                        .find(|span| span.start < through && through < span.end)
                    {
                        through = span.start;
                    }
                    through >= end && remaining <= last.capacity
                })
                && catalog
                    .iter()
                    .find(|entry| entry.capacity >= remaining)
                    .is_some_and(|entry| entry.has_terminal);
            let mut best: Option<(usize, u64)> = None;
            let mut max_work = 0usize;
            for entry in catalog {
                let mut through = start.saturating_add(entry.capacity).min(limit);
                if let Some(span) = request
                    .prefill_spans
                    .iter()
                    .find(|span| span.start < through && through < span.end)
                {
                    through = span.start;
                }
                let work = through.saturating_sub(start);
                if work == 0 {
                    continue;
                }
                let is_terminal = entry.has_terminal
                    && (through >= end
                        || (terminal_window > 0 && end.saturating_sub(through) <= terminal_window));
                if terminal_only && !is_terminal {
                    continue;
                }
                let adds_work = work > max_work;
                max_work = max_work.max(work);
                let nanos = if is_terminal {
                    entry.terminal_ns
                } else {
                    entry.non_terminal_ns
                };
                let Some(nanos) = nanos else {
                    if !adds_work
                        || (has_terminal_catalog && Some(entry.capacity) != target_capacity)
                    {
                        continue;
                    }
                    return None;
                };
                if cheaper(work, nanos, best) {
                    best = Some((work, nanos));
                }
            }
            let (work, nanos) = best?;
            start += work;
            total = total.checked_add(nanos)?;
        }
        Some(total)
    }

    fn checkpoint_selection(&self, request: &LiveRequest) -> (CheckpointAccounting, bool, bool) {
        self.checkpoint_selection_cached(request, true)
    }

    fn checkpoint_selection_cached(
        &self,
        request: &LiveRequest,
        quote_storage: bool,
    ) -> (CheckpointAccounting, bool, bool) {
        if let Some(value) = self
            .checkpoint_selections
            .borrow()
            .as_ref()
            .and_then(|cache| cache.get(&(request.handle, quote_storage)))
        {
            return value.clone();
        }
        let selection = self.checkpoint_selection_uncached(request, quote_storage);
        if let Some(cache) = self.checkpoint_selections.borrow_mut().as_mut() {
            cache.insert((request.handle, quote_storage), selection.clone());
        }
        selection
    }

    fn checkpoint_selection_uncached(
        &self,
        request: &LiveRequest,
        quote_storage: bool,
    ) -> (CheckpointAccounting, bool, bool) {
        #[cfg(test)]
        tests::SELECTIONS.set(tests::SELECTIONS.get() + 1);
        let mut accounting = request.checkpoints.clone();
        if accounting
            .cuts
            .iter()
            .any(|cut| cut.role != RetentionRole::Anchor)
        {
            return (accounting, false, false);
        }
        accounting.cuts.clear();
        accounting.reserved_ns = 0;
        let anchor = request.checkpoint_anchor();
        let start = request.consumed_tokens as usize;
        let (baseline_start, baseline_end) = *accounting.baseline_interval.get_or_insert((
            request.admission_reused_tokens as usize,
            request.prompt_tokens,
        ));
        let catalog = self.prefill_catalog_costs(request);
        let terminal_window = if request.prompt_logprobs {
            0
        } else {
            self.terminal_prefill_window()
        };
        if accounting.initial_prefill_ns.is_none() {
            accounting.initial_prefill_ns = Self::interval_prefill_ns(
                request,
                &catalog,
                terminal_window,
                baseline_start,
                baseline_end,
            );
        }
        if anchor > start {
            accounting.cuts.push(Cut {
                endpoint: anchor,
                role: RetentionRole::Anchor,
                split_ns: 0,
                copy_ns: 0,
            });
        }
        let Some(prefill) = accounting.initial_prefill_ns else {
            return (accounting, true, false);
        };
        let Some(copy_ns) = self.costs.fresh_duration(GraphKind::Snapshot, 1) else {
            return (accounting, true, false);
        };
        let mut unknown = false;
        let mut budget = false;
        let available = allowance(prefill, self.policy.checkpoint_extra_work_fraction);
        let mut candidates: Vec<_> = request
            .message_boundaries
            .iter()
            .rev()
            .copied()
            .filter(|&end| start < end && end < anchor && request.checkpoint_endpoint_eligible(end))
            .map(|end| (end, RetentionRole::Message))
            .collect();
        if start < request.prompt_tokens && request.prompt_tokens != anchor {
            candidates.push((request.prompt_tokens, RetentionRole::Endpoint));
        }
        // The interval between accepted cuts is shared by every candidate inside it.
        let mut between: Option<((usize, usize), Option<u64>)> = None;
        for (endpoint, role) in candidates {
            let left = accounting
                .cuts
                .iter()
                .map(|cut| cut.endpoint)
                .filter(|&cut| cut < endpoint)
                .max()
                .unwrap_or(start);
            let right = accounting
                .cuts
                .iter()
                .map(|cut| cut.endpoint)
                .filter(|&cut| endpoint < cut)
                .min()
                .unwrap_or(request.prompt_tokens);
            let split = if endpoint == right {
                Some(0)
            } else {
                let whole = match between {
                    Some((interval, nanos)) if interval == (left, right) => nanos,
                    _ => {
                        let nanos = Self::interval_prefill_ns(
                            request,
                            &catalog,
                            terminal_window,
                            left,
                            right,
                        );
                        between = Some(((left, right), nanos));
                        nanos
                    }
                };
                whole.and_then(|whole| {
                    let before = Self::interval_prefill_ns(
                        request,
                        &catalog,
                        terminal_window,
                        left,
                        endpoint,
                    )?;
                    let after = Self::interval_prefill_ns(
                        request,
                        &catalog,
                        terminal_window,
                        endpoint,
                        right,
                    )?;
                    Some(before.saturating_add(after).saturating_sub(whole))
                })
            };
            let Some(split_ns) = split else {
                unknown = true;
                continue;
            };
            let charge = split_ns.saturating_add(copy_ns);
            if accounting
                .extra_spent_ns
                .saturating_add(accounting.reserved_ns)
                .saturating_add(charge)
                > available
            {
                budget = true;
                continue;
            }
            let rank = RetentionRank {
                use_epoch: accounting.demand_epoch,
                role,
                consumed_tokens: endpoint as u64,
            };
            let rows = 1 + accounting
                .cuts
                .iter()
                .filter(|cut| cut.endpoint > start)
                .count()
                - usize::from(request.anchor_guard.is_some() && start < anchor);
            let protected_budget = (role == RetentionRole::Endpoint)
                .then_some(endpoint.saturating_sub(left) as u64);
            if quote_storage
                && self
                    .quote_snapshots_preserving_with_budget(
                        1,
                        rows,
                        0,
                        rank,
                        protected_budget,
                    )
                    .is_none()
            {
                continue;
            }
            accounting.reserved_ns = accounting.reserved_ns.saturating_add(charge);
            accounting.cuts.push(Cut {
                endpoint,
                role,
                split_ns,
                copy_ns,
            });
        }
        accounting.cuts.sort_unstable_by_key(|cut| cut.endpoint);
        (accounting, unknown, budget)
    }

    pub(super) fn estimated_checkpoint_cut(&self, request: &LiveRequest) -> Option<usize> {
        let (accounting, _, _) = self.checkpoint_selection_cached(request, false);
        accounting
            .cuts
            .iter()
            .find(|cut| cut.endpoint > request.consumed_tokens as usize)
            .map(|cut| cut.endpoint)
    }

    pub(super) fn checkpoint_cut(&self, request: &LiveRequest) -> Option<usize> {
        let (accounting, _, _) = self.checkpoint_selection(request);
        accounting
            .cuts
            .iter()
            .find(|cut| cut.endpoint > request.consumed_tokens as usize)
            .map(|cut| cut.endpoint)
    }

    pub(super) fn plan_checkpoints(&self, step: &ScheduledStep) -> Vec<CapturePlan> {
        let mut plans = Vec::new();
        for (row, lane) in step.lanes.iter().enumerate() {
            let request = self.request(lane.request).unwrap();
            let kind = step.lane_kind(row);
            if !kind.is_prefill() {
                continue;
            }
            let anchor = request.checkpoint_anchor();
            let (accounting, unknown, budget) = self.checkpoint_selection(request);
            let end = lane.start_position as usize + lane.tokens;
            let cut = accounting
                .cuts
                .iter()
                .find(|cut| {
                    cut.endpoint == end && !self.is_non_terminal_prefill(kind, step.token_capacity)
                })
                .copied();
            let reserve_guard = cut.is_some_and(|cut| cut.endpoint < anchor)
                && request.anchor_guard.is_none()
                && anchor > request.consumed_tokens as usize;
            let rank = RetentionRank {
                use_epoch: accounting.demand_epoch,
                role: cut.map_or(RetentionRole::Endpoint, |cut| cut.role),
                consumed_tokens: end as u64,
            };
            let rows = usize::from(cut.is_some()) + usize::from(reserve_guard);
            let planned_rows = accounting
                .cuts
                .len()
                .saturating_sub(usize::from(request.anchor_guard.is_some()))
                .max(rows);
            let protected_budget = cut
                .filter(|cut| cut.role == RetentionRole::Endpoint)
                .map(|cut| (cut.endpoint as u64).saturating_sub(request.prefix_tokens));
            let quote = if cut.is_some_and(|cut| cut.role == RetentionRole::Anchor)
                && request.anchor_guard.is_some()
            {
                None
            } else if rows > 0 {
                self.quote_snapshots_preserving_with_budget(
                    rows,
                    planned_rows,
                    0,
                    rank,
                    protected_budget,
                )
            } else {
                None
            };
            plans.push(CapturePlan {
                handle: lane.request,
                accounting,
                anchor,
                cut,
                quote,
                reserve_guard,
                unknown,
                budget,
            });
        }
        plans
    }

    pub(super) fn checkpoint_anchor_waiting(&self, plans: &[CapturePlan]) -> bool {
        !self.submissions.is_empty()
            && plans.iter().any(|plan| {
                plan.cut
                    .is_some_and(|cut| cut.role == RetentionRole::Anchor)
                    && self
                        .request(plan.handle)
                        .is_ok_and(|request| request.anchor_guard.is_none())
                    && plan.quote.is_none()
                    && self.pending_resources(self.decode_growth_pages(), 1)
            })
    }

    pub(super) fn promote_request_anchor(&mut self, handle: RequestHandle) -> Result<()> {
        let request = self.request(handle)?;
        let key = request.history.key_at(request.checkpoint_anchor() as u64);
        let epoch = request.checkpoints.demand_epoch;
        if let Some(id) = key.and_then(|key| self.cache.ready_by_prefix(&key)) {
            let promoted = self.cache.checkpoint(id)?.retention_rank().role < RetentionRole::Anchor;
            self.cache.promote(id, RetentionRole::Anchor, epoch)?;
            if promoted {
                self.reoffer_publication(id);
            }
        }
        Ok(())
    }

    pub(super) fn commit_checkpoint_plans(&mut self, plans: &[CapturePlan]) -> Result<()> {
        for plan in plans {
            let handle = plan.handle;
            let mut accounting = plan.accounting.clone();
            self.stats.checkpoint_skip_unknown_cost += u64::from(plan.unknown);
            self.stats.checkpoint_skip_work_budget += u64::from(plan.budget);
            let Some(cut) = plan.cut else {
                self.request_mut(handle)?.checkpoints = accounting;
                continue;
            };
            {
                accounting
                    .cuts
                    .retain(|candidate| candidate.endpoint != cut.endpoint);
                accounting.reserved_ns = accounting.reserved_ns.saturating_sub(cut.split_ns);
                accounting.extra_spent_ns = accounting.extra_spent_ns.saturating_add(cut.split_ns);
                self.stats.checkpoint_extra_work_ns = self
                    .stats
                    .checkpoint_extra_work_ns
                    .saturating_add(cut.split_ns);
            }
            let mut destination = if cut.role == RetentionRole::Anchor {
                self.request_mut(handle)?.anchor_guard.take()
            } else {
                None
            };
            if destination.is_none() {
                let mut quote = plan.quote.clone();
                if cut.role == RetentionRole::Anchor {
                    let rank = RetentionRank {
                        use_epoch: accounting.demand_epoch,
                        role: cut.role,
                        consumed_tokens: cut.endpoint as u64,
                    };
                    quote = self.quote_snapshots(1, 0, rank);
                }
                if let Some(quote) = quote {
                    if let Some(mut rows) = self.apply_snapshot_quote(&quote)? {
                        if plan.reserve_guard {
                            let guard = rows.remove(0);
                            self.request_mut(handle)?.anchor_guard = Some(guard);
                        }
                        destination = rows.pop();
                    }
                }
            }
            if destination.is_none()
                && plan.reserve_guard
                && self.request(handle)?.anchor_guard.is_none()
            {
                let rank = RetentionRank {
                    use_epoch: accounting.demand_epoch,
                    role: RetentionRole::Anchor,
                    consumed_tokens: plan.anchor as u64,
                };
                if let Some(quote) = self.quote_snapshots(1, 0, rank) {
                    if let Some(rows) = self.apply_snapshot_quote(&quote)? {
                        self.request_mut(handle)?.anchor_guard = rows.first().copied();
                    }
                }
            }
            if let Some(snapshot) = destination {
                let capture = ArmedCapture {
                    snapshot,
                    endpoint: cut.endpoint as u64,
                    role: cut.role,
                    copy_ns: cut.copy_ns,
                };
                self.request_mut(handle)?.checkpoint_capture = Some(capture);
            } else {
                accounting.reserved_ns = accounting.reserved_ns.saturating_sub(cut.copy_ns);
                self.stats.checkpoint_skip_storage += 1;
            }
            self.request_mut(handle)?.checkpoints = accounting;
        }
        Ok(())
    }

    pub(super) fn checkpoint_copy_allowed(
        &self,
        handle: RequestHandle,
        capture: ArmedCapture,
    ) -> Result<bool> {
        if capture.role == RetentionRole::Anchor {
            return Ok(true);
        }
        if self.costs.fresh_duration(GraphKind::Snapshot, 1).is_none() {
            return Ok(false);
        }
        let accounting = &self.request(handle)?.checkpoints;
        let Some(prefill) = accounting.initial_prefill_ns else {
            return Ok(false);
        };
        Ok(accounting
            .extra_spent_ns
            .saturating_add(accounting.reserved_ns)
            <= allowance(prefill, self.policy.checkpoint_extra_work_fraction))
    }

    pub(super) fn credit_checkpoint_decode(
        &mut self,
        step: &ScheduledStep,
        useful: &[u64; REQUEST_LANES],
        generated: &[GeneratedToken],
        timing: crate::execution::ExecutionTiming,
    ) {
        let total: u128 = useful.iter().map(|&tokens| u128::from(tokens)).sum();
        if total == 0 {
            return;
        }
        let Some(gpu_ns) = timing
            .gpu_ns
            .into_iter()
            .flatten()
            .max()
            .filter(|&nanos| nanos > 0)
        else {
            return;
        };
        let work =
            gpu_ns.saturating_sub(timing.readiness.map_or(0, |readiness| readiness.wait_ns()));
        for (row, lane) in step.lanes.iter().enumerate() {
            if step.lane_kind(row).is_prefill() || useful[lane.request.slot as usize] == 0 {
                continue;
            }
            let output = generated
                .iter()
                .filter(|event| event.request == lane.request)
                .count();
            let credit = ((u128::from(work) * output as u128) / total) as u64;
            if let Ok(request) = self.request_mut(lane.request) {
                request.checkpoints.productive_decode_ns = request
                    .checkpoints
                    .productive_decode_ns
                    .saturating_add(credit);
            }
        }
    }

    pub(super) fn allocate_terminal_capture(
        &mut self,
        handle: RequestHandle,
    ) -> Result<Option<SnapshotId>> {
        let request = self.request(handle)?;
        let (mut accounting, _, _) = self.checkpoint_selection(request);
        accounting.cuts.clear();
        accounting.reserved_ns = 0;
        let Some(cost) = self.costs.fresh_duration(GraphKind::Snapshot, 1) else {
            self.stats.checkpoint_skip_unknown_cost += 1;
            return Ok(None);
        };
        let Some(prefill) = accounting.initial_prefill_ns else {
            self.stats.checkpoint_skip_unknown_cost += 1;
            return Ok(None);
        };
        if accounting.extra_spent_ns.saturating_add(cost)
            > allowance(
                prefill.saturating_add(accounting.productive_decode_ns),
                self.policy.checkpoint_extra_work_fraction,
            )
        {
            self.stats.checkpoint_skip_work_budget += 1;
            return Ok(None);
        }
        let rank = RetentionRank {
            use_epoch: accounting.demand_epoch,
            role: RetentionRole::Endpoint,
            consumed_tokens: request.consumed_tokens,
        };
        let gain_tokens = request.consumed_tokens.saturating_sub(request.prefix_tokens);
        let quote = self.quote_snapshots_with_budget(1, 0, rank, Some(gain_tokens));
        let rows = match quote {
            Some(quote) => self.apply_snapshot_quote(&quote)?,
            None => None,
        };
        let snapshot = rows.and_then(|rows| rows.first().copied());
        if snapshot.is_some() {
            accounting.reserved_ns = cost;
            let request = self.request_mut(handle)?;
            request.checkpoint_copy_ns = cost;
            request.checkpoints = accounting;
        } else {
            self.stats.checkpoint_skip_storage += 1;
        }
        Ok(snapshot)
    }

    pub(super) fn commit_checkpoint_copies(&mut self, step: &ScheduledStep) -> Result<()> {
        for (row, lane) in step.lanes.iter().enumerate() {
            if step.lane_kind(row) != GraphKind::Snapshot {
                continue;
            }
            let request = self.request_mut(lane.request)?;
            let cost = std::mem::take(&mut request.checkpoint_copy_ns);
            request.checkpoints.reserved_ns = request.checkpoints.reserved_ns.saturating_sub(cost);
            request.checkpoints.extra_spent_ns =
                request.checkpoints.extra_spent_ns.saturating_add(cost);
            self.stats.checkpoint_extra_work_ns =
                self.stats.checkpoint_extra_work_ns.saturating_add(cost);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    thread_local! {
        /// Uncached checkpoint selections on this thread.
        pub(super) static SELECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// Runs of whole cheapest chunks the interval walk summed at once.
        pub(super) static STRIDE_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    fn measure(scheduler: &mut RequestScheduler, kind: GraphKind, tokens: usize, nanos: u64) {
        scheduler.costs.set_measured(
            &ScheduledStep {
                draft_tokens: 0,
                kind,
                batch_capacity: 1,
                token_capacity: tokens,
                packed_token_capacity: 0,
                mixed: None,
                lanes: Vec::new(),
                requires_feedback: true,
            },
            nanos,
        );
    }

    fn linear_costs(scheduler: &mut RequestScheduler) {
        for capacity in scheduler.prefill_catalog.clone() {
            measure(
                scheduler,
                GraphKind::Prefill,
                capacity,
                capacity as u64 * 1000,
            );
        }
        measure(scheduler, GraphKind::Snapshot, 1, 1);
    }

    /// The interval walk as it was before catalog costs were resolved once per selection.
    fn reference_interval_prefill_ns(
        scheduler: &RequestScheduler,
        request: &LiveRequest,
        mut start: usize,
        end: usize,
    ) -> Option<u64> {
        let catalog = if request.prompt_logprobs {
            &scheduler.scored_prefill_catalog
        } else {
            &scheduler.prefill_catalog
        };
        let mut total = 0u64;
        while start < end {
            let limit = request
                .checkpoints
                .window_ends
                .iter()
                .copied()
                .filter(|&limit| start < limit)
                .min()
                .unwrap_or(end)
                .min(end);
            let mut best: Option<(usize, u64)> = None;
            let mut max_work = 0usize;
            for &capacity in catalog {
                let mut through = start.saturating_add(capacity).min(limit);
                if let Some(span) = request
                    .prefill_spans
                    .iter()
                    .find(|span| span.start < through && through < span.end)
                {
                    through = span.start;
                }
                let work = through.saturating_sub(start);
                if work == 0 {
                    continue;
                }
                let adds_work = work > max_work;
                max_work = max_work.max(work);
                let Some(nanos) = scheduler
                    .costs
                    .duration(request.prefill_kind(), 1, capacity)
                    .filter(|&nanos| nanos != 0)
                else {
                    if !adds_work {
                        continue;
                    }
                    return None;
                };
                if best.is_none_or(|(old_work, old_ns)| {
                    u128::from(nanos) * (old_work as u128) < u128::from(old_ns) * work as u128
                        || (u128::from(nanos) * old_work as u128
                            == u128::from(old_ns) * work as u128
                            && work > old_work)
                }) {
                    best = Some((work, nanos));
                }
            }
            let (work, nanos) = best?;
            start += work;
            total = total.checked_add(nanos)?;
        }
        Some(total)
    }

    /// Scored prefill and one mixed shape, which the shared duration fixture lacks.
    fn extra_graphs() -> Vec<crate::execution::GraphDescriptor> {
        let graph = |kind, batch, tokens, prefill| crate::execution::GraphDescriptor {
            tuned_bindings: vec![],
            confidence_width: 0,
            draft_tokens: 0,
            packed_token_capacity: 0,
            name: format!("{kind:?}-{batch}-{tokens}-{prefill}"),
            kind,
            sampling_mode: crate::execution::SamplingMode::General,
            handle: 1,
            batch_capacity: batch,
            token_capacity: tokens,
            prefill_capacity: prefill,
            estimated_duration_ns: 5_000,
            input_bindings: vec![],
        };
        let mut graphs: Vec<_> = [128, 256, 512]
            .into_iter()
            .map(|tokens| graph(GraphKind::ScoredPrefill, 1, tokens, 0))
            .collect();
        graphs.push(graph(GraphKind::Mixed, 2, 1, 512));
        graphs
    }

    fn walk_scheduler(scored: bool) -> RequestScheduler {
        let template = super::super::tests::scheduler(2, 128, 16);
        let mut scheduler = RequestScheduler::new(
            [9; 32],
            SchedulerLimits {
                max_sequence_tokens: 16_384,
                pages: 1024,
                ..template.limits
            },
            template.capabilities.clone(),
        )
        .unwrap();
        let mut graphs = super::super::tests::duration_graphs(2);
        graphs.extend(extra_graphs());
        scheduler.measured_graph_durations(&graphs).unwrap();
        scheduler.prefill_graphs(&[128, 256, 512]).unwrap();
        if scored {
            scheduler.scored_prefill_graphs(&[128, 256, 512]).unwrap();
        }
        scheduler
    }

    #[test]
    fn resolved_catalog_costs_walk_every_interval_as_the_lookups_did() {
        let profiles: [(&str, &[(usize, u64)]); 6] = [
            ("smallest cheapest", &[(128, 90_000), (256, 250_000), (512, 410_000)]),
            ("middle cheapest", &[(128, 120_000), (256, 200_000), (512, 450_000)]),
            ("ties take the largest", &[(128, 128_000), (256, 256_000), (512, 512_000)]),
            ("largest cheapest", &[(128, 200_000), (256, 300_000), (512, 400_000)]),
            ("one capacity unknown", &[(128, 90_000), (256, 0), (512, 410_000)]),
            ("sums overflow", &[(128, u64::MAX / 16), (256, u64::MAX / 8), (512, u64::MAX / 4)]),
        ];
        // A span longer than the 128 and 256 strides, two adjacent spans, and a window
        // ending at a span end. A span may be at most one chunk long.
        let spans = [1000..1500, 2500..2600, 2600..2700, 4000..4100];
        let windows = vec![1500, 3333, 5120];
        let prompt = 6000;
        let landmarks = [
            0, 999, 1000, 1001, 1500, 2500, 2600, 2700, 3333, 4000, 4100, 5120, prompt,
        ];
        let mut intervals: Vec<(usize, usize)> = (0..prompt)
            .step_by(97)
            .flat_map(|start| (start..=prompt).step_by(113).map(move |end| (start, end)))
            .collect();
        for &mark in &landmarks {
            for &other in &landmarks {
                intervals.push((mark.min(other), mark.max(other)));
            }
            for offset in [1, 63, 128, 511, 512, 513, 1500] {
                intervals.push((mark.saturating_sub(offset), mark));
                intervals.push((mark, (mark + offset).min(prompt)));
            }
        }
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..2000 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let a = (seed >> 33) as usize % (prompt + 1);
            let b = (seed >> 11) as usize % (prompt + 1);
            intervals.push((a.min(b), a.max(b)));
        }
        for scored in [false, true] {
            let kind = if scored {
                GraphKind::ScoredPrefill
            } else {
                GraphKind::Prefill
            };
            for (name, costs) in profiles {
                let mut scheduler = walk_scheduler(scored);
                for &(capacity, nanos) in costs {
                    measure(&mut scheduler, kind, capacity, nanos);
                }
                let mut spec = super::super::tests::spec(prompt, 1);
                spec.prompt_logprobs = scored;
                let handle = scheduler.admit(spec).unwrap();
                scheduler.set_prefill_spans(handle, &spans).unwrap();
                scheduler.request_mut(handle).unwrap().checkpoints.window_ends = windows.clone();
                let request = scheduler.request(handle).unwrap();
                assert_eq!(request.prefill_kind(), kind);
                let catalog = scheduler.prefill_catalog_costs(request);
                STRIDE_RUNS.set(0);
                let mut unknown = 0;
                for &(start, end) in &intervals {
                    let walked = RequestScheduler::interval_prefill_ns(
                        request,
                        &catalog,
                        0,
                        start,
                        end,
                    );
                    assert_eq!(
                        walked,
                        reference_interval_prefill_ns(&scheduler, request, start, end),
                        "{name}, scored {scored}: {start}..{end}"
                    );
                    unknown += usize::from(walked.is_none());
                }
                // The shortcut ran wherever every cost is known; unknown or overflowing
                // sums came back as None on both walks.
                let known = !costs.iter().any(|&(_, nanos)| nanos == 0);
                assert_eq!(STRIDE_RUNS.get() > 0, known, "{name}, scored {scored}");
                let expect_unknown = name == "one capacity unknown" || name == "sums overflow";
                if expect_unknown {
                    assert!(unknown > 100, "{name}, scored {scored}: {unknown} unknown");
                }
            }
        }
    }

    #[test]
    fn selections_over_long_agentic_prompts_look_costs_up_once_per_selection() {
        let template = super::super::tests::scheduler(4, 128, 16);
        let mut scheduler = RequestScheduler::new(
            [9; 32],
            SchedulerLimits {
                max_sequence_tokens: 262_144,
                prefill_chunk_size: 8192,
                pages: 16_384,
                ..template.limits
            },
            template.capabilities.clone(),
        )
        .unwrap();
        let mut graphs = super::super::tests::duration_graphs(4);
        graphs.extend(extra_graphs());
        scheduler.measured_graph_durations(&graphs).unwrap();
        scheduler.mixed_graphs(&[(2, 1, 512)]).unwrap();
        linear_costs(&mut scheduler);
        scheduler
            .configure_policy(PolicyConfig {
                checkpoint_extra_work_fraction: 0.05,
                ..Default::default()
            })
            .unwrap();
        // Four long agentic conversations in prefill, each with a boundary every 150 tokens.
        let mut handles = Vec::new();
        for _ in 0..4 {
            let mut spec = super::super::tests::spec(150_000, 1);
            spec.message_boundaries = (150..150_000).step_by(150).collect();
            handles.push(scheduler.admit(spec).unwrap());
        }
        let catalog = scheduler.prefill_catalog.len();
        super::super::cost::ENTRY_LOOKUPS.set(0);
        let started = std::time::Instant::now();
        for &handle in &handles {
            let request = scheduler.request(handle).unwrap();
            std::hint::black_box(scheduler.checkpoint_cut(request));
        }
        let elapsed = started.elapsed();
        let lookups = super::super::cost::ENTRY_LOOKUPS.get();
        let started = std::time::Instant::now();
        for &handle in &handles {
            let request = scheduler.request(handle).unwrap();
            std::hint::black_box(scheduler.estimated_checkpoint_cut(request));
        }
        println!(
            "LONG_PROMPT_ESTIMATE requests=4 elapsed={:.2} ms",
            started.elapsed().as_secs_f64() * 1e3
        );
        println!(
            "LONG_PROMPT_SELECTION requests=4 boundaries=999 catalog={catalog} \
             lookups={lookups} elapsed={:.1} ms",
            elapsed.as_secs_f64() * 1e3
        );
        // Each selection resolves the catalog once; the snapshot cost adds one more.
        assert!(lookups <= handles.len() * (catalog + 4), "{lookups} cost lookups");
        // Service rates price every prefill capacity and the mixed shape from one selection,
        // with the work each capacity would take unchanged.
        let request = scheduler.request(handles[0]).unwrap();
        let cut = scheduler.estimated_checkpoint_cut(request);
        let capacities = scheduler.prefill_catalog.iter().copied().chain([512]);
        for capacity in capacities {
            assert_eq!(
                scheduler.prefill_work_through(request, capacity, cut),
                scheduler.prefill_work(request, capacity),
                "{capacity}"
            );
        }
        assert!(scheduler.prefill_work(request, 512) > 0);
        SELECTIONS.set(0);
        std::hint::black_box(scheduler.service_rates(request));
        assert_eq!(SELECTIONS.get(), 1);
    }

    #[test]
    fn prefill_capacity_matches_submitted_work_without_snapshot_storage() {
        for (prompt_tokens, capacity, refresh) in [
            (200, 256, false),
            (300, 512, false),
            (1024, 512, false),
            (300, 256, true),
        ] {
            let mut scheduler = super::super::tests::scheduler(1, 128, 16);
            scheduler.prefill_graphs(&[128, 256, 512]).unwrap();
            linear_costs(&mut scheduler);
            scheduler
                .configure_policy(PolicyConfig {
                    checkpoint_extra_work_fraction: 0.5,
                    ..Default::default()
                })
                .unwrap();
            let mut spec = super::super::tests::spec(prompt_tokens, 1);
            spec.message_boundaries = vec![100, prompt_tokens];
            let handle = scheduler.admit(spec).unwrap();
            let records = |sequence: i64, consumed: usize| {
                let output = consumed == prompt_tokens;
                let mut records = vec![CompletionRecord::default(); 4];
                records[handle.slot as usize] = CompletionRecord {
                    generation: handle.generation as i64,
                    submission_sequence: sequence,
                    consumed_tokens: consumed as i64,
                    generated_tokens: i64::from(output),
                    token_id: if output { 42 } else { -1 },
                    status: 1,
                    ..Default::default()
                };
                records
            };
            let init = scheduler.plan().unwrap();
            assert_eq!(init.kind, GraphKind::Init);
            scheduler.commit(&init, Fence(1)).unwrap();
            scheduler
                .complete(Fence(1), &records(1, 0), Default::default())
                .unwrap();
            let origin = if refresh {
                scheduler
                    .advance_to(scheduler.policy.estimate_ttl_ns)
                    .unwrap();
                measure(&mut scheduler, GraphKind::Snapshot, 1, 1);
                scheduler.refresh.credit_ns = refresh::CREDIT_CAP_NS;
                let plan = scheduler.plan().unwrap();
                assert_eq!(plan.token_capacity, capacity);
                assert!(matches!(plan.origin, PlanOrigin::RefreshStart(_)));
                plan.origin
            } else {
                PlanOrigin::Ordinary
            };
            let mut held = Vec::new();
            while let Some(snapshot) = scheduler.reserve_snapshot(0) {
                held.push(snapshot);
            }
            assert_eq!(held.len(), 16);
            assert_eq!(scheduler.free_snapshots.len(), 0);
            let request = scheduler.request(handle).unwrap();
            assert_eq!(scheduler.estimated_checkpoint_cut(request), Some(100));
            assert_eq!(scheduler.checkpoint_cut(request), Some(prompt_tokens));
            let plan = scheduler.plan().unwrap();
            assert_eq!(plan.kind, GraphKind::Prefill);
            assert_eq!(
                (plan.token_capacity, plan.lanes[0].tokens),
                (capacity, prompt_tokens.min(capacity))
            );
            assert_eq!(plan.estimated_duration_ns(), capacity as u64 * 1000);
            assert_eq!(plan.selected_cost.retained_ns, capacity as u64 * 1000);
            assert_eq!(plan.origin, origin);
        }
    }

    #[test]
    fn missing_cost_retries_original_uncached_interval_after_progress() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 16);
        scheduler
            .graph_durations(&super::super::tests::duration_graphs(1))
            .unwrap();
        let mut spec = super::super::tests::spec(1024, 3);
        spec.message_boundaries = vec![128, 256, 512, 896];
        let handle = scheduler.admit(spec).unwrap();
        let (cold, unknown, _) = scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert!(unknown);
        assert_eq!(cold.initial_prefill_ns, None);
        assert_eq!(
            cold.cuts.iter().map(|cut| cut.endpoint).collect::<Vec<_>>(),
            [896]
        );
        scheduler.request_mut(handle).unwrap().checkpoints = cold;
        scheduler.request_mut(handle).unwrap().consumed_tokens = 128;
        linear_costs(&mut scheduler);
        let (warm, unknown, _) = scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert!(!unknown);
        assert_eq!(warm.initial_prefill_ns, Some(1_024_000));
        assert!(warm.cuts.len() > 1);
        assert!(warm.cuts.iter().all(|cut| cut.endpoint > 128));
    }

    #[test]
    fn unknown_prefill_or_stale_snapshot_cost_disables_extras() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 16);
        let mut spec = super::super::tests::spec(1024, 3);
        spec.message_boundaries = vec![128, 256, 512, 896];
        let handle = scheduler.admit(spec).unwrap();
        linear_costs(&mut scheduler);
        measure(&mut scheduler, GraphKind::Prefill, 512, 0);
        let (unknown, _, _) = scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert_eq!(unknown.initial_prefill_ns, None);
        assert_eq!(unknown.cuts.len(), 1);
        linear_costs(&mut scheduler);
        scheduler
            .advance_to(scheduler.policy.estimate_ttl_ns)
            .unwrap();
        let (stale, _, _) = scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert_eq!(stale.initial_prefill_ns, Some(1_024_000));
        assert_eq!(stale.cuts.len(), 1);
    }

    #[test]
    fn measured_prefill_costs_keep_extras_available_after_idle() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 16);
        linear_costs(&mut scheduler);
        scheduler.advance_to(scheduler.policy.estimate_ttl_ns).unwrap();
        measure(&mut scheduler, GraphKind::Snapshot, 1, 1);
        let mut spec = super::super::tests::spec(1024, 3);
        spec.message_boundaries = vec![128, 256, 512, 896];
        let handle = scheduler.admit(spec).unwrap();
        let (selected, unknown, budget) =
            scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert!(!unknown);
        assert!(!budget);
        assert_eq!(selected.initial_prefill_ns, Some(1_024_000));
        assert_eq!(
            selected.cuts.iter().map(|cut| cut.endpoint).collect::<Vec<_>>(),
            [128, 256, 512, 896, 1024]
        );
    }

    #[test]
    fn joint_storage_selection_keeps_supplementaries_nearest_anchor() {
        for snapshots in [2, 3] {
            let mut scheduler = super::super::tests::scheduler(1, 128, snapshots);
            linear_costs(&mut scheduler);
            let mut spec = super::super::tests::spec(1024, 3);
            spec.message_boundaries = (128..1024).step_by(128).collect();
            let handle = scheduler.admit(spec).unwrap();
            let (selected, _, _) =
                scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
            let endpoints: Vec<_> = selected.cuts.iter().map(|cut| cut.endpoint).collect();
            assert_eq!(
                endpoints,
                if snapshots == 2 {
                    vec![768, 896]
                } else {
                    vec![640, 768, 896]
                }
            );
        }
    }

    #[test]
    fn decode_attribution_excludes_prefill_and_uses_mixed_useful_denominator() {
        let mut scheduler = super::super::tests::scheduler(2, 128, 16);
        let prefill = scheduler.admit(super::super::tests::spec(128, 3)).unwrap();
        let decoder = scheduler.admit(super::super::tests::spec(128, 3)).unwrap();
        let step = ScheduledStep {
            draft_tokens: 0,
            kind: GraphKind::Mixed,
            batch_capacity: 2,
            token_capacity: 1,
            packed_token_capacity: 0,
            mixed: Some(MixedStep {
                prefill_capacity: 128,
                prefill_row: Some(0),
                speculative: false,
            }),
            lanes: vec![
                ScheduledLane {
                    request: prefill,
                    start_position: 0,
                    tokens: 128,
                },
                ScheduledLane {
                    request: decoder,
                    start_position: 128,
                    tokens: 1,
                },
            ],
            requires_feedback: true,
        };
        let mut useful = [0; REQUEST_LANES];
        useful[prefill.slot as usize] = 128;
        useful[decoder.slot as usize] = 1;
        let generated = [
            GeneratedToken {
                request: prefill,
                token: 1,
                finished: false,
            },
            GeneratedToken {
                request: decoder,
                token: 1,
                finished: false,
            },
        ];
        scheduler.credit_checkpoint_decode(&step, &useful, &generated, Default::default());
        assert_eq!(
            scheduler
                .request(decoder)
                .unwrap()
                .checkpoints
                .productive_decode_ns,
            0
        );
        let mut timing = crate::execution::ExecutionTiming::default();
        timing.gpu_ns[0] = Some(1290);
        timing.gpu_ns[1] = Some(2580);
        scheduler.credit_checkpoint_decode(&step, &useful, &generated, timing);
        assert_eq!(
            scheduler
                .request(prefill)
                .unwrap()
                .checkpoints
                .productive_decode_ns,
            0
        );
        assert_eq!(
            scheduler
                .request(decoder)
                .unwrap()
                .checkpoints
                .productive_decode_ns,
            20
        );
        let mut readiness = crate::execution::ReadinessIntervals::default();
        readiness.len = 1;
        readiness.intervals[0] = [1500, 210];
        timing.readiness = Some(readiness);
        scheduler.credit_checkpoint_decode(&step, &useful, &generated, timing);
        assert_eq!(
            scheduler
                .request(decoder)
                .unwrap()
                .checkpoints
                .productive_decode_ns,
            30
        );
    }

    #[test]
    fn joint_split_ledger_prices_interacting_cuts_and_keeps_spent_work_on_stale_quote() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 16);
        scheduler
            .configure_policy(PolicyConfig {
                checkpoint_extra_work_fraction: 0.5,
                ..Default::default()
            })
            .unwrap();
        for (tokens, nanos) in [(128, 100), (256, 110), (512, 200)] {
            measure(&mut scheduler, GraphKind::Prefill, tokens, nanos);
        }
        measure(&mut scheduler, GraphKind::Snapshot, 1, 1);
        let mut spec = super::super::tests::spec(1024, 3);
        spec.message_boundaries = (128..1024).step_by(128).collect();
        let handle = scheduler.admit(spec).unwrap();
        let (selected, _, _) = scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert_eq!(selected.initial_prefill_ns, Some(400));
        assert_eq!(
            selected
                .cuts
                .iter()
                .map(|cut| cut.endpoint)
                .collect::<Vec<_>>(),
            [256, 512, 640, 768, 896, 1024]
        );
        assert_eq!(selected.reserved_ns, 115);
        assert_eq!(
            selected
                .cuts
                .iter()
                .find(|cut| cut.endpoint == 256)
                .unwrap()
                .split_ns,
            20
        );
        assert_eq!(
            selected
                .cuts
                .iter()
                .find(|cut| cut.endpoint == 640)
                .unwrap()
                .split_ns,
            90
        );
        scheduler.request_mut(handle).unwrap().phase = RequestPhase::Prefill;
        let plan = scheduler.plan().unwrap();
        assert_eq!(plan.lanes[0].tokens, 256);
        let _held = scheduler.reserve_snapshot(0).unwrap();
        scheduler.commit(&plan, Fence(1)).unwrap();
        let accounting = scheduler.request(handle).unwrap().checkpoint_accounting();
        assert_eq!(accounting.extra_spent_ns, 20);
        assert_eq!(accounting.reserved_ns, 94);
        assert!(
            scheduler
                .request(handle)
                .unwrap()
                .checkpoint_capture
                .is_none()
        );
        assert!(scheduler.request(handle).unwrap().anchor_guard.is_some());
    }

    #[test]
    fn restored_prefix_promotes_exact_anchor_outside_its_parent_chain_without_rewind() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 16);
        let mut history = PrefixHistory::new([9; 32]);
        history.append(&(0..1024).collect::<Vec<_>>(), &[]).unwrap();
        let mut ids = Vec::new();
        for consumed in [1024u64, 512] {
            let snapshot = scheduler.reserve_snapshot(0).unwrap();
            let pages = (0..consumed / 64)
                .map(|_| scheduler.free_pages.pop_first().unwrap())
                .collect();
            ids.push(
                scheduler
                    .cache
                    .publish(
                        &history,
                        &Publication {
                            parent: None,
                            consumed_tokens: consumed,
                            tail_pages: pages,
                            snapshot: StateSnapshot {
                                id: snapshot,
                                consumed_tokens: consumed,
                            },
                            ready_after: Fence(0),
                        },
                    )
                    .unwrap()
                    .id(),
            );
        }
        let mut spec = super::super::tests::spec(1024, 3);
        spec.message_boundaries = vec![512];
        let handle = scheduler.admit(spec).unwrap();
        assert_eq!(scheduler.request(handle).unwrap().prefix, Some(ids[0]));
        let plan = scheduler.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Restore);
        scheduler.commit(&plan, Fence(1)).unwrap();
        assert_eq!(scheduler.request(handle).unwrap().consumed_tokens, 1024);
        assert_eq!(
            scheduler
                .cache
                .checkpoint(ids[1])
                .unwrap()
                .retention_rank()
                .role,
            RetentionRole::Anchor
        );
    }

    #[test]
    fn completed_conditioned_windows_preserve_anchor_eligibility_and_baseline_cuts() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 16);
        let mut spec = super::super::tests::spec(768, 3);
        spec.message_boundaries = vec![256, 400, 512];
        let handle = scheduler.admit(spec).unwrap();
        scheduler
            .request_mut(handle)
            .unwrap()
            .checkpoints
            .retain_window_ends([256, 512, 768].into_iter());
        assert!(
            scheduler
                .request(handle)
                .unwrap()
                .conditioned_windows
                .is_empty()
        );
        assert_eq!(scheduler.request(handle).unwrap().checkpoint_anchor(), 400);
        let (accounting, _, _) = scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert_eq!(accounting.initial_prefill_ns, Some(3000));
    }

    #[test]
    fn zero_fraction_never_reserves_or_charges_extra_copy_work() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 16);
        scheduler
            .configure_policy(PolicyConfig {
                checkpoint_extra_work_fraction: 0.0,
                ..Default::default()
            })
            .unwrap();
        linear_costs(&mut scheduler);
        let mut spec = super::super::tests::spec(1024, 3);
        spec.message_boundaries = vec![128, 512, 896];
        let handle = scheduler.admit(spec).unwrap();
        let (selected, _, _) = scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert_eq!(selected.cuts.len(), 1);
        assert_eq!(selected.cuts[0].role, RetentionRole::Anchor);
        assert!(
            scheduler
                .allocate_terminal_capture(handle)
                .unwrap()
                .is_none()
        );
        assert_eq!(scheduler.free_snapshot_count(), 16);
        assert_eq!(scheduler.stats().checkpoint_extra_work_ns, 0);
    }

    #[test]
    fn terminal_prefill_cut_pricing_accounts_for_extra_terminal_chunk_cost() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 16);
        let mut graphs = super::super::tests::duration_graphs(1);
        for tokens in [256, 512] {
            graphs.push(crate::execution::GraphDescriptor {
                tuned_bindings: vec![],
                confidence_width: 0,
                draft_tokens: 0,
                packed_token_capacity: 0,
                name: format!("terminal_prefill{tokens}"),
                kind: GraphKind::TerminalPrefill,
                sampling_mode: crate::execution::SamplingMode::General,
                handle: 1,
                batch_capacity: 1,
                token_capacity: tokens,
                prefill_capacity: 0,
                estimated_duration_ns: 0,
                input_bindings: vec![],
            });
        }
        scheduler.measured_graph_durations(&graphs).unwrap();
        scheduler.prefill_graphs(&[128, 256, 512]).unwrap();
        scheduler.terminal_prefill_graphs(&[256, 512]).unwrap();
        measure(&mut scheduler, GraphKind::Prefill, 128, 18_000);
        measure(&mut scheduler, GraphKind::Prefill, 256, 25_000);
        measure(&mut scheduler, GraphKind::Prefill, 512, 40_000);
        measure(&mut scheduler, GraphKind::TerminalPrefill, 256, 41_000);
        measure(&mut scheduler, GraphKind::TerminalPrefill, 512, 56_000);
        measure(&mut scheduler, GraphKind::Snapshot, 1, 500);
        scheduler
            .configure_policy(PolicyConfig {
                checkpoint_extra_work_fraction: 0.25,
                ..Default::default()
            })
            .unwrap();
        // 1024 tokens = 1 encoder-only Prefill(512) (40_000) + 1 TerminalPrefill(512) (56_000) = 96_000.
        // Cutting at 512 turns chunk 0 into TerminalPrefill(512) (+16_000 split cost) + 500 snapshot.
        let mut spec = super::super::tests::spec(1024, 3);
        spec.message_boundaries = vec![512, 1024];
        let handle = scheduler.admit(spec).unwrap();
        let (selected, _, _) = scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert_eq!(selected.initial_prefill_ns, Some(96_000));
        let cut = selected
            .cuts
            .iter()
            .find(|c| c.endpoint == 512)
            .expect("cut at 512 fits within 25% allowance (24_000)");
        assert_eq!(cut.split_ns, 16_000);
    }

    #[test]
    fn terminal_prefill_catalog_captures_extras_without_nonterminal_or_padded_costs() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 16);
        let caps = [128usize, 256, 512];
        scheduler.prefill_graphs(&caps).unwrap();
        scheduler.terminal_prefill_graphs(&caps).unwrap();
        let mut graphs = super::super::tests::duration_graphs(1);
        let terminal: Vec<_> = graphs
            .iter()
            .filter(|g| g.kind == GraphKind::Prefill && caps.contains(&g.token_capacity))
            .cloned()
            .map(|mut g| {
                g.kind = GraphKind::TerminalPrefill;
                g.name = format!("terminal-{}", g.token_capacity);
                g
            })
            .collect();
        graphs.extend(terminal);
        scheduler.graph_durations(&graphs).unwrap();
        // Only TerminalPrefill(128) and Snapshot(1) are measured; non-terminal Prefill
        // and larger padded capacities (256, 512) remain unmeasured (0 ns).
        measure(&mut scheduler, GraphKind::TerminalPrefill, 128, 100_000);
        measure(&mut scheduler, GraphKind::Snapshot, 1, 1_000);
        let mut spec = super::super::tests::spec(96, 3);
        spec.message_boundaries = vec![94];
        let handle = scheduler.admit(spec).unwrap();
        let (selected, unknown, budget) =
            scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert!(!unknown, "short terminal prefill must not report unknown_cost");
        assert!(!budget);
        assert_eq!(selected.initial_prefill_ns, Some(100_000));
        let endpoints: Vec<_> = selected.cuts.iter().map(|cut| cut.endpoint).collect();
        assert_eq!(endpoints, vec![94, 96]);

        // Partial terminal catalog [256, 512] (window 128): only TerminalPrefill(512)
        // and Snapshot(1) are measured; unmeasured non-terminal Prefill(128) and
        // non-target TerminalPrefill(256) must not fail a 300-token interval.
        let mut partial = super::super::tests::scheduler(1, 128, 16);
        partial.prefill_graphs(&caps).unwrap();
        partial.terminal_prefill_graphs(&[256, 512]).unwrap();
        partial.graph_durations(&graphs).unwrap();
        measure(&mut partial, GraphKind::TerminalPrefill, 512, 150_000);
        measure(&mut partial, GraphKind::Snapshot, 1, 1_000);
        let partial_handle = partial.admit(super::super::tests::spec(300, 3)).unwrap();
        let (partial_selected, partial_unknown, partial_budget) =
            partial.checkpoint_selection(partial.request(partial_handle).unwrap());
        assert!(
            !partial_unknown,
            "partial terminal catalog must not fail on unmeasured non-target capacities"
        );
        assert!(!partial_budget);
        assert_eq!(partial_selected.initial_prefill_ns, Some(150_000));
    }

    fn publish_test_checkpoint(
        scheduler: &mut RequestScheduler,
        parent: Option<CheckpointId>,
        token_fill: u32,
        consumed_tokens: u64,
        role: RetentionRole,
    ) -> CheckpointId {
        let mut history = PrefixHistory::new([9; 32]);
        if let Some(parent_id) = parent {
            let parent_len = scheduler
                .cache
                .checkpoint(parent_id)
                .unwrap()
                .prefix()
                .consumed_tokens as usize;
            history.append(&vec![token_fill; parent_len], &[]).unwrap();
            let extra = consumed_tokens as usize - parent_len;
            history.append(&vec![token_fill + 1; extra], &[]).unwrap();
        } else {
            history
                .append(&vec![token_fill; consumed_tokens as usize], &[])
                .unwrap();
        }
        let parent_pages = parent.map_or(0, |id| {
            scheduler
                .cache
                .checkpoint(id)
                .unwrap()
                .prefix()
                .consumed_tokens as usize
                / scheduler.capabilities.page_tokens
        });
        let total_pages = consumed_tokens as usize / scheduler.capabilities.page_tokens;
        let tail_pages: Vec<_> = (0..(total_pages - parent_pages))
            .map(|_| scheduler.free_pages.pop_first().unwrap())
            .collect();
        let snapshot = scheduler.reserve_snapshot(0).unwrap();
        let epoch = scheduler.cache.new_use_epoch().unwrap();
        scheduler
            .cache
            .publish_with_retention(
                &history,
                &Publication {
                    parent,
                    consumed_tokens,
                    tail_pages,
                    snapshot: StateSnapshot {
                        id: snapshot,
                        consumed_tokens,
                    },
                    ready_after: Fence(0),
                },
                role,
                epoch,
            )
            .unwrap()
            .id()
    }

    #[test]
    fn short_terminal_endpoint_preserves_larger_resident_anchor_leaves_for_all_prompts() {
        for message_boundaries in [vec![512], vec![]] {
            let mut scheduler = super::super::tests::scheduler(2, 64, 2);
            scheduler.enable_evictions();
            linear_costs(&mut scheduler);
            let doc_a =
                publish_test_checkpoint(&mut scheduler, None, 101, 512, RetentionRole::Anchor);
            let doc_b =
                publish_test_checkpoint(&mut scheduler, None, 202, 512, RetentionRole::Anchor);
            assert_eq!(scheduler.free_snapshot_count(), 0);

            let mut spec = super::super::tests::spec(544, 64);
            spec.prompt[..512].fill(202);
            spec.prompt[512..].fill(203);
            spec.message_boundaries = message_boundaries;
            let handle = scheduler.admit(spec).unwrap();
            assert_eq!(scheduler.request(handle).unwrap().prefix, Some(doc_b));
            {
                let req = scheduler.request_mut(handle).unwrap();
                req.consumed_tokens = 544;
                req.checkpoints.initial_prefill_ns = Some(500_000);
            }

            let capture = scheduler.allocate_terminal_capture(handle).unwrap();
            assert!(
                capture.is_none(),
                "short terminal endpoint must not evict a larger protected anchor leaf"
            );
            assert_eq!(scheduler.stats().checkpoint_skip_storage, 1);
            assert!(scheduler.take_evictions().is_empty());
            assert!(scheduler.cache.checkpoint(doc_a).is_ok());
            assert!(scheduler.cache.checkpoint(doc_b).is_ok());
            scheduler.cancel(handle).unwrap();

            // Once doc_a ages enough that `512 / (1 + age) <= 32`, a newer request with the
            // same 32-token terminal delta reaps doc_a freely.
            for _ in 0..16 {
                scheduler.cache.new_use_epoch().unwrap();
            }
            let mut newer_spec = super::super::tests::spec(544, 64);
            newer_spec.prompt[..512].fill(202);
            newer_spec.prompt[512..].fill(204);
            let newer = scheduler.admit(newer_spec).unwrap();
            assert_eq!(scheduler.request(newer).unwrap().prefix, Some(doc_b));
            {
                let req = scheduler.request_mut(newer).unwrap();
                req.consumed_tokens = 544;
                req.checkpoints.initial_prefill_ns = Some(500_000);
            }
            let row = scheduler
                .allocate_terminal_capture(newer)
                .unwrap()
                .expect("aged anchor leaf must be reaped freely by a newer terminal endpoint");
            assert_eq!(scheduler.take_evictions().len(), 1);
            assert!(scheduler.cache.checkpoint(doc_a).is_err());
            assert!(scheduler.cache.checkpoint(doc_b).is_ok());
            scheduler.release_snapshot(row);
            scheduler.cancel(newer).unwrap();
        }
    }

    #[test]
    fn short_terminal_endpoint_evicts_older_non_root_endpoint_keeping_parent_anchor_intact() {
        let mut scheduler = super::super::tests::scheduler(2, 64, 3);
        scheduler.enable_evictions();
        linear_costs(&mut scheduler);
        let doc_a = publish_test_checkpoint(&mut scheduler, None, 101, 512, RetentionRole::Anchor);
        let reply_a = publish_test_checkpoint(
            &mut scheduler,
            Some(doc_a),
            101,
            640,
            RetentionRole::Endpoint,
        );
        // Match real completion recency refresh: terminal Endpoint and its direct parent Anchor
        // share the completion epoch so Endpoint ranks below Anchor at that epoch.
        let reply_epoch = scheduler
            .cache
            .checkpoint(reply_a)
            .unwrap()
            .retention_rank()
            .use_epoch;
        scheduler.cache.touch(doc_a, reply_epoch).unwrap();
        let doc_b = publish_test_checkpoint(&mut scheduler, None, 202, 512, RetentionRole::Anchor);
        assert_eq!(scheduler.free_snapshot_count(), 0);

        let mut spec = super::super::tests::spec(544, 64);
        spec.prompt[..512].fill(202);
        spec.prompt[512..].fill(203);
        spec.message_boundaries = vec![512];
        let handle = scheduler.admit(spec).unwrap();
        assert_eq!(scheduler.request(handle).unwrap().prefix, Some(doc_b));
        {
            let req = scheduler.request_mut(handle).unwrap();
            // Marginal gain is 32 tokens, smaller than reply_a's 128-token delta; LRU among
            // non-root Endpoint leaves must still replace reply_a without touching doc_a.
            req.consumed_tokens = 544;
            req.checkpoints.initial_prefill_ns = Some(500_000);
        }

        let row = scheduler
            .allocate_terminal_capture(handle)
            .unwrap()
            .expect("older non-root endpoint leaf must remain LRU-evictable");
        assert_eq!(scheduler.stats().checkpoint_skip_storage, 0);
        assert_eq!(scheduler.take_evictions().len(), 1);
        assert!(scheduler.cache.checkpoint(reply_a).is_err());
        assert!(scheduler.cache.checkpoint(doc_a).is_ok());
        assert!(scheduler.cache.checkpoint(doc_b).is_ok());
        scheduler.release_snapshot(row);
        scheduler.cancel(handle).unwrap();
    }

    #[test]
    fn long_terminal_endpoint_evicts_smaller_older_anchor_leaf() {
        let mut scheduler = super::super::tests::scheduler(2, 64, 2);
        scheduler.enable_evictions();
        linear_costs(&mut scheduler);
        let short_doc =
            publish_test_checkpoint(&mut scheduler, None, 101, 128, RetentionRole::Anchor);
        let active_doc =
            publish_test_checkpoint(&mut scheduler, None, 202, 512, RetentionRole::Anchor);
        assert_eq!(scheduler.free_snapshot_count(), 0);

        let mut spec = super::super::tests::spec(512, 512);
        spec.prompt.fill(202);
        spec.message_boundaries = vec![512];
        let handle = scheduler.admit(spec).unwrap();
        assert_eq!(scheduler.request(handle).unwrap().prefix, Some(active_doc));
        {
            let req = scheduler.request_mut(handle).unwrap();
            // Marginal gain is 512 tokens >= short_doc's 128-token root anchor.
            req.consumed_tokens = 1024;
            req.checkpoints.initial_prefill_ns = Some(500_000);
        }

        let row = scheduler
            .allocate_terminal_capture(handle)
            .unwrap()
            .expect("terminal gain exceeding older anchor leaf must allow eviction");
        assert_eq!(scheduler.stats().checkpoint_skip_storage, 0);
        assert_eq!(scheduler.take_evictions().len(), 1);
        assert!(scheduler.cache.checkpoint(short_doc).is_err());
        assert!(scheduler.cache.checkpoint(active_doc).is_ok());
        scheduler.release_snapshot(row);
        scheduler.cancel(handle).unwrap();
    }

    #[test]
    fn multi_victim_terminal_capture_enforces_cumulative_protected_token_budget_atomically() {
        // Need 5 free pages for running decoder headroom; each anchor holds 3 pages (192 tokens),
        // so freeing enough pages requires evicting BOTH anchors (384 protected tokens total).
        let mut scheduler = super::super::tests::scheduler(3, 14, 3);
        scheduler.enable_evictions();
        linear_costs(&mut scheduler);
        let anchor_0 =
            publish_test_checkpoint(&mut scheduler, None, 101, 192, RetentionRole::Anchor);
        let anchor_1 =
            publish_test_checkpoint(&mut scheduler, None, 202, 192, RetentionRole::Anchor);
        let active =
            publish_test_checkpoint(&mut scheduler, None, 303, 128, RetentionRole::Anchor);
        assert_eq!(scheduler.free_snapshot_count(), 0);
        // 14 total pages - 3 - 3 - 2 = 6 free pages left.

        let mut growing = super::super::tests::spec(64, 400);
        growing.prompt.fill(404);
        growing.growable_output = true;
        let decoder = scheduler.admit(growing).unwrap();
        {
            let req = scheduler.request_mut(decoder).unwrap();
            req.phase = RequestPhase::Decode;
            // Needs 5 additional pages beyond its 2 reserved pages at decode_horizon=1.
            req.scheduled_consumed = 383;
        }
        assert_eq!(scheduler.decode_growth_pages(), 5);
        // Drain remaining free pages into the live request so 0 free pages remain before eviction.
        let mut spec = super::super::tests::spec(128, 256);
        spec.prompt.fill(303);
        spec.message_boundaries = vec![128];
        let handle = scheduler.admit(spec).unwrap();
        assert_eq!(scheduler.request(handle).unwrap().prefix, Some(active));
        assert_eq!(scheduler.free_pages.len(), 0);

        // Epochs: anchor_0=1 (age 4 -> 192/5 = 38), anchor_1=2 (age 3 -> 192/4 = 48),
        // cumulative age-discounted protected cost = 86 tokens.
        // Gain = 64 tokens covers anchor_0 (38) but not both (86 > 64): must decline atomically.
        {
            let req = scheduler.request_mut(handle).unwrap();
            req.consumed_tokens = 128 + 64;
            req.checkpoints.initial_prefill_ns = Some(500_000);
        }
        assert!(
            scheduler
                .allocate_terminal_capture(handle)
                .unwrap()
                .is_none()
        );
        assert_eq!(scheduler.stats().checkpoint_skip_storage, 1);
        assert!(scheduler.take_evictions().is_empty());
        assert!(scheduler.cache.checkpoint(anchor_0).is_ok());
        assert!(scheduler.cache.checkpoint(anchor_1).is_ok());

        // Gain = 86 tokens: covers the exact cumulative age-discounted cost (38 + 48) and frees 6 >= 5 pages.
        scheduler.request_mut(handle).unwrap().consumed_tokens = 128 + 86;
        let row = scheduler
            .allocate_terminal_capture(handle)
            .unwrap()
            .expect("cumulative gain covering both anchors must succeed");
        assert_eq!(scheduler.take_evictions().len(), 2);
        assert!(scheduler.cache.checkpoint(anchor_0).is_err());
        assert!(scheduler.cache.checkpoint(anchor_1).is_err());
        scheduler.release_snapshot(row);
        scheduler.cancel(handle).unwrap();
        scheduler.cancel(decoder).unwrap();
    }

    #[test]
    fn compacted_root_endpoint_leaf_is_protected_by_its_full_token_span() {
        let mut scheduler = super::super::tests::scheduler(2, 64, 3);
        scheduler.enable_evictions();
        linear_costs(&mut scheduler);
        let root_anchor =
            publish_test_checkpoint(&mut scheduler, None, 101, 512, RetentionRole::Anchor);
        let child_endpoint = publish_test_checkpoint(
            &mut scheduler,
            Some(root_anchor),
            101,
            576,
            RetentionRole::Endpoint,
        );
        // Compact root_anchor into child_endpoint so child_endpoint becomes a root Endpoint leaf
        // owning all 576 tokens (9 pages).
        let retired = scheduler.cache.evict_checkpoint(root_anchor).unwrap();
        let update = scheduler.cache.advance_completed(scheduler.completed).unwrap();
        scheduler.apply_completion(update);
        assert_eq!(retired.heir, Some(child_endpoint));
        assert_eq!(
            scheduler.cache.checkpoint(child_endpoint).unwrap().parent(),
            None
        );

        let active_doc =
            publish_test_checkpoint(&mut scheduler, None, 202, 512, RetentionRole::Anchor);
        // Occupy the remaining free snapshot row with a pinned dummy checkpoint or reserve it.
        let held = scheduler.reserve_snapshot(0).unwrap();
        assert_eq!(scheduler.free_snapshot_count(), 0);

        let mut spec = super::super::tests::spec(512, 64);
        spec.prompt.fill(202);
        let handle = scheduler.admit(spec).unwrap();
        assert_eq!(scheduler.request(handle).unwrap().prefix, Some(active_doc));
        {
            let req = scheduler.request_mut(handle).unwrap();
            req.consumed_tokens = 512 + 64;
            req.checkpoints.initial_prefill_ns = Some(500_000);
        }

        assert!(
            scheduler
                .allocate_terminal_capture(handle)
                .unwrap()
                .is_none(),
            "short terminal endpoint must not evict a compacted root endpoint holding 576 tokens"
        );
        assert!(scheduler.cache.checkpoint(child_endpoint).is_ok());
        scheduler.release_snapshot(held);
        scheduler.cancel(handle).unwrap();
    }

    #[test]
    fn prompt_end_endpoint_respects_marginal_gain_and_decode_headroom() {
        // Part A: a 16-token prompt-end Endpoint past a 512-token Anchor must not evict
        // another resident 512-token Anchor leaf, while keeping the Anchor cut itself.
        let mut scheduler = super::super::tests::scheduler(2, 64, 2);
        linear_costs(&mut scheduler);
        let doc_a = publish_test_checkpoint(&mut scheduler, None, 101, 512, RetentionRole::Anchor);
        let doc_b = publish_test_checkpoint(&mut scheduler, None, 202, 512, RetentionRole::Anchor);
        assert_eq!(scheduler.free_snapshot_count(), 0);

        let mut spec = super::super::tests::spec(528, 16);
        spec.prompt.fill(303);
        spec.message_boundaries = vec![512];
        let handle = scheduler.admit(spec).unwrap();
        let (selected, _, _) =
            scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        let endpoints: Vec<_> = selected
            .cuts
            .iter()
            .map(|cut| (cut.endpoint, cut.role))
            .collect();
        assert_eq!(
            endpoints,
            vec![(512, RetentionRole::Anchor)],
            "16-token prompt-end endpoint must be skipped when only 512-token anchors can be evicted"
        );
        assert!(scheduler.cache.checkpoint(doc_a).is_ok());
        assert!(scheduler.cache.checkpoint(doc_b).is_ok());
        scheduler.cancel(handle).unwrap();

        // Part B: terminal capture preserves running decode headroom on aliased free rows.
        let mut s = super::super::tests::scheduler(2, 5, 1);
        s.share_snapshot_pages(&[4..5]).unwrap();
        let mut req_spec = super::super::tests::spec(64, 1);
        req_spec.message_boundaries = vec![32];
        let pending = s.admit(req_spec).unwrap();
        let mut growing = super::super::tests::spec(60, 400);
        growing.growable_output = true;
        let decoder = s.admit(growing).unwrap();
        {
            let req = s.request_mut(decoder).unwrap();
            req.phase = RequestPhase::Decode;
            req.scheduled_consumed = 127;
        }
        assert_eq!(s.decode_growth_pages(), 1);
        assert_eq!(s.available_snapshot(0), Some(SnapshotId(0)));
        assert!(s.available_snapshot(1).is_none());
        assert_eq!(s.allocate_terminal_capture(pending).unwrap(), None);
        assert_eq!(s.stats().checkpoint_skip_storage, 1);
        assert!(s.free_pages.contains(&PageId(4)));
        s.cancel(decoder).unwrap();
        let row = s.allocate_terminal_capture(pending).unwrap().unwrap();
        assert_eq!(s.stats().checkpoint_skip_storage, 1);
        s.release_snapshot(row);
        s.cancel(pending).unwrap();
    }
}
