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

    fn checkpoint_headroom(&self) -> usize {
        self.slots
            .iter()
            .filter_map(|slot| slot.request.as_ref())
            .filter(|request| request.growable_output && request.phase == RequestPhase::Decode)
            .map(|request| {
                self.pages_through(
                    request,
                    request
                        .scheduled_consumed
                        .saturating_add(self.decode_horizon()),
                )
                .saturating_sub(request.reserved_pages.len())
            })
            .fold(0usize, usize::saturating_add)
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
        if self.limits.snapshots == 0 {
            return (accounting, false, false);
        }
        // Under a zero extra-work budget, an interior message boundary is kept
        // as the anchor only when splitting prefill at that boundary adds zero
        // predicted prefill work; otherwise fall back to the zero-split prompt
        // end so short prompts and chunk tails do not pay an extra prefill launch.
        let anchor = if anchor > start
            && anchor < request.prompt_tokens
            && self.policy.checkpoint_extra_work_fraction == 0.0
        {
            let whole = if (start, request.prompt_tokens) == (baseline_start, baseline_end) {
                accounting.initial_prefill_ns
            } else {
                Self::interval_prefill_ns(
                    request,
                    &catalog,
                    terminal_window,
                    start,
                    request.prompt_tokens,
                )
            };
            let split_ns = whole.and_then(|whole| {
                let before = Self::interval_prefill_ns(
                    request,
                    &catalog,
                    terminal_window,
                    start,
                    anchor,
                )?;
                let after = Self::interval_prefill_ns(
                    request,
                    &catalog,
                    terminal_window,
                    anchor,
                    request.prompt_tokens,
                )?;
                Some(before.saturating_add(after).saturating_sub(whole))
            });
            if split_ns == Some(0) {
                anchor
            } else {
                request.prompt_tokens
            }
        } else {
            anchor
        };
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
            if quote_storage && self
                .quote_snapshots_preserving(1, rows, self.checkpoint_headroom(), rank)
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
            let (accounting, unknown, budget) = self.checkpoint_selection(request);
            let anchor = accounting
                .cuts
                .iter()
                .find(|cut| cut.role == RetentionRole::Anchor)
                .map_or_else(|| request.checkpoint_anchor(), |cut| cut.endpoint);
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
            let quote = if cut.is_some_and(|cut| cut.role == RetentionRole::Anchor)
                && request.anchor_guard.is_some()
            {
                None
            } else if rows > 0 {
                self.quote_snapshots_preserving(
                    rows,
                    planned_rows,
                    self.checkpoint_headroom(),
                    rank,
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
                    && self.pending_resources(self.checkpoint_headroom(), 1)
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
                    quote = self.quote_snapshots(1, self.checkpoint_headroom(), rank);
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
                if let Some(quote) = self.quote_snapshots(1, self.checkpoint_headroom(), rank) {
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
        let quote = self.quote_snapshots(1, self.checkpoint_headroom(), rank);
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

    fn complete_test_step(
        scheduler: &mut RequestScheduler,
        step: &ScheduledPlan,
        fence: Fence,
        dur_ns: u64,
    ) {
        let mut records = vec![CompletionRecord::default(); scheduler.limits.request_capacity()];
        let mut produced_output = false;
        for lane in &step.lanes {
            let r = scheduler.request(lane.request).unwrap();
            let end = lane.start_position + lane.tokens as u64;
            let emitted = match step.kind {
                GraphKind::Prefill | GraphKind::TerminalPrefill => {
                    usize::from(end == r.prompt_tokens as u64)
                }
                GraphKind::Decode | GraphKind::Resume => 1,
                GraphKind::Speculative => 3
                    .min(lane.tokens)
                    .min((r.max_new_tokens - r.generated_tokens) as usize),
                _ => 0,
            };
            if emitted > 0 {
                produced_output = true;
            }
            let consumed = if step.kind == GraphKind::Speculative {
                r.consumed_tokens + emitted as u64
            } else {
                end
            };
            let generated = r.generated_tokens + emitted as u32;
            let mut record = CompletionRecord {
                generation: lane.request.generation as i64,
                submission_sequence: fence.0 as i64,
                consumed_tokens: consumed as i64,
                generated_tokens: generated as i64,
                token_id: if emitted > 0 { 42 } else { -1 },
                status: if emitted > 0 && generated == r.max_new_tokens {
                    2
                } else {
                    1
                },
                ..Default::default()
            };
            if step.kind == GraphKind::Speculative {
                let offered = lane.tokens.saturating_sub(1);
                let mut vector = crate::execution::TokenVector {
                    emitted_count: emitted as i64,
                    offered_drafts: offered as i64,
                    accepted_drafts: emitted.saturating_sub(1).min(offered) as i64,
                    verification_width: step.token_capacity as i64,
                    draft_cache_consumed: consumed as i64 - 1,
                    ..Default::default()
                };
                vector.tokens[..emitted].fill(42);
                record.vector = Some(vector);
            }
            records[lane.request.slot as usize] = record;
        }
        let mut timing = crate::execution::ExecutionTiming::default();
        if dur_ns > 0 {
            timing.gpu_ns[0] = Some(dur_ns);
        }
        scheduler.complete(fence, &records, timing).unwrap();
        if produced_output {
            for lane in &step.lanes {
                let _ = scheduler.published_output(lane.request, fence);
            }
        }
    }

    #[test]
    fn zero_snapshot_capacity_produces_no_cuts_and_never_splits_prefill() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 0);
        let mut spec = super::super::tests::spec(67, 2);
        spec.message_boundaries = vec![63];
        let handle = scheduler.admit(spec).unwrap();
        let (selected, unknown, budget) =
            scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert!(!unknown);
        assert!(!budget);
        assert!(selected.cuts.is_empty());
        assert_eq!(
            scheduler.checkpoint_cut(scheduler.request(handle).unwrap()),
            None
        );

        let mut prefill_launches = 0usize;
        let mut snapshot_launches = 0usize;
        for step_idx in 1..=8u64 {
            let Some(plan) = scheduler.plan() else {
                break;
            };
            if plan.kind.is_prefill() {
                prefill_launches += 1;
                assert_eq!(plan.lanes[0].tokens, 67);
            } else if plan.kind == GraphKind::Snapshot {
                snapshot_launches += 1;
            }
            let fence = Fence(step_idx);
            scheduler.commit(&plan, fence).unwrap();
            complete_test_step(&mut scheduler, &plan, fence, 1000);
            if scheduler.request(handle).unwrap().consumed_tokens >= 68 {
                break;
            }
        }
        assert_eq!(prefill_launches, 1);
        assert_eq!(snapshot_launches, 0);
    }

    #[test]
    fn zero_fraction_falls_back_to_prompt_end_when_interior_anchor_splits_prefill() {
        let template = super::super::tests::scheduler(1, 128, 16);
        let mut scheduler = RequestScheduler::new(
            [9; 32],
            SchedulerLimits {
                max_sequence_tokens: 262_144,
                prefill_chunk_size: 16_384,
                pages: 8192,
                snapshots: 16,
                ..template.limits
            },
            template.capabilities.clone(),
        )
        .unwrap();
        let prefill_caps = [128usize, 1024, 4096, 10_240, 16_384];
        scheduler.prefill_graphs(&prefill_caps).unwrap();
        let mut graphs = super::super::tests::duration_graphs(1);
        for cap in [10_240usize, 16_384] {
            graphs.push(crate::execution::GraphDescriptor {
                tuned_bindings: vec![],
                confidence_width: 0,
                draft_tokens: 0,
                packed_token_capacity: 0,
                name: format!("prefill{cap}"),
                kind: GraphKind::Prefill,
                sampling_mode: crate::execution::SamplingMode::General,
                handle: 1,
                batch_capacity: 1,
                token_capacity: cap,
                prefill_capacity: 0,
                estimated_duration_ns: 0,
                input_bindings: vec![],
            });
        }
        scheduler.measured_graph_durations(&graphs).unwrap();
        measure(&mut scheduler, GraphKind::Prefill, 128, 23_000_000);
        measure(&mut scheduler, GraphKind::Prefill, 1024, 32_000_000);
        measure(&mut scheduler, GraphKind::Prefill, 4096, 70_000_000);
        measure(&mut scheduler, GraphKind::Prefill, 10_240, 165_000_000);
        measure(&mut scheduler, GraphKind::Prefill, 16_384, 245_000_000);
        measure(&mut scheduler, GraphKind::Snapshot, 1, 300_000);
        scheduler
            .configure_policy(PolicyConfig {
                checkpoint_extra_work_fraction: 0.0,
                ..Default::default()
            })
            .unwrap();

        // Short prompt with interior message boundary 4 tokens before prompt end:
        // splitting at 63 would require two prefill128 launches, so fall back to 67.
        let mut short_spec = super::super::tests::spec(67, 2);
        short_spec.message_boundaries = vec![63];
        let short_handle = scheduler.admit(short_spec).unwrap();
        let (short_sel, _, _) =
            scheduler.checkpoint_selection(scheduler.request(short_handle).unwrap());
        assert_eq!(short_sel.cuts.len(), 1);
        assert_eq!(short_sel.cuts[0].role, RetentionRole::Anchor);
        assert_eq!(short_sel.cuts[0].endpoint, 67);
        scheduler.cancel(short_handle).unwrap();

        // Long 124,610-token prompt with interior boundary 4 tokens before prompt end:
        // splitting at 124,606 would add a 9th launch (prefill128), so fall back to 124,610.
        let mut long_spec = super::super::tests::spec(124_610, 2);
        long_spec.message_boundaries = vec![124_606];
        let long_handle = scheduler.admit(long_spec).unwrap();
        let (long_sel, _, _) =
            scheduler.checkpoint_selection(scheduler.request(long_handle).unwrap());
        assert_eq!(long_sel.cuts.len(), 1);
        assert_eq!(long_sel.cuts[0].role, RetentionRole::Anchor);
        assert_eq!(long_sel.cuts[0].endpoint, 124_610);
        scheduler.cancel(long_handle).unwrap();

        // Exact chunk-aligned interior boundary (7 * 16,384 = 114,688) has zero split cost
        // relative to 7 * prefill16384 + prefill10240, so the interior anchor is kept.
        let mut aligned_spec = super::super::tests::spec(124_610, 2);
        aligned_spec.message_boundaries = vec![114_688];
        let aligned_handle = scheduler.admit(aligned_spec).unwrap();
        let (aligned_sel, _, _) =
            scheduler.checkpoint_selection(scheduler.request(aligned_handle).unwrap());
        assert_eq!(aligned_sel.cuts.len(), 1);
        assert_eq!(aligned_sel.cuts[0].role, RetentionRole::Anchor);
        assert_eq!(aligned_sel.cuts[0].endpoint, 114_688);
    }

    #[test]
    fn balanced_c16_short_prompts_skip_interior_anchor_tails_and_reach_full_batch() {
        let template = super::super::tests::scheduler(16, 128, 64);
        let mut scheduler = RequestScheduler::new(
            [9; 32],
            SchedulerLimits {
                request_slots: 16,
                max_active: 16,
                max_sequence_tokens: 4096,
                prefill_chunk_size: 128,
                pages: 1024,
                snapshots: 64,
            },
            template.capabilities.clone(),
        )
        .unwrap();
        let mut graphs = super::super::tests::duration_graphs(16);
        let spec_dur_ns = |batch: usize| match batch {
            1 => 5_950_000,
            2 => 7_370_000,
            4 => 8_470_000,
            6 => 9_410_000,
            8 => 12_540_000,
            12 => 14_190_000,
            _ => 18_220_000,
        };
        for graph in &mut graphs {
            graph.estimated_duration_ns = match graph.kind {
                GraphKind::Prefill => 22_900_000,
                GraphKind::Decode => 5_500_000 + (graph.batch_capacity as u64) * 800_000,
                GraphKind::Speculative => spec_dur_ns(graph.batch_capacity),
                GraphKind::Snapshot => 300_000,
                GraphKind::Init => 50_000,
                _ => 500_000,
            };
        }
        scheduler.measured_graph_durations(&graphs).unwrap();
        scheduler.enable_speculation(&[1, 2, 4, 8]).unwrap();
        scheduler
            .configure_policy(PolicyConfig {
                checkpoint_extra_work_fraction: 0.0,
                ..Default::default()
            })
            .unwrap();

        let mut handles = Vec::new();
        for i in 0..16usize {
            let prompt_len = 64 + i;
            let mut spec = super::super::tests::spec(prompt_len, 512);
            spec.prompt = (0..prompt_len).map(|t| (i * 1000 + t + 1) as u32).collect();
            spec.message_boundaries = vec![prompt_len - 4];
            handles.push(scheduler.admit(spec).unwrap());
        }

        let mut now_ns = 0u64;
        let mut prefill_launches = 0usize;
        let mut max_decode_batch = 0usize;
        let mut decode_steps = 0usize;
        let mut all_prefills_done_ns = 0u64;
        for step_idx in 1..=500u64 {
            scheduler.advance_to(now_ns).unwrap();
            let Some(plan) = scheduler.plan() else {
                break;
            };
            let dur_ns = match plan.kind {
                GraphKind::Prefill => {
                    prefill_launches += 1;
                    22_900_000
                }
                GraphKind::Speculative => {
                    decode_steps += 1;
                    max_decode_batch = max_decode_batch.max(plan.lanes.len());
                    spec_dur_ns(plan.batch_capacity)
                }
                GraphKind::Decode => {
                    decode_steps += 1;
                    max_decode_batch = max_decode_batch.max(plan.lanes.len());
                    5_500_000 + (plan.batch_capacity as u64) * 800_000
                }
                GraphKind::Snapshot => 300_000,
                GraphKind::Init => 50_000,
                _ => 500_000,
            };
            let fence = Fence(step_idx);
            scheduler.commit(&plan, fence).unwrap();
            now_ns += dur_ns;
            scheduler.advance_to(now_ns).unwrap();
            complete_test_step(&mut scheduler, &plan, fence, dur_ns);
            if prefill_launches == 16 && all_prefills_done_ns == 0 {
                all_prefills_done_ns = now_ns;
            }
            if max_decode_batch == 16 && decode_steps >= 4 {
                break;
            }
        }
        assert_eq!(
            prefill_launches, 16,
            "each of the 16 short prompts should complete prefill in one launch"
        );
        assert_eq!(
            max_decode_batch, 16,
            "balanced scheduler should converge all 16 prompts into a full B=16 decode batch (all_prefills_done_ms={:.1}, end_ms={:.1})",
            all_prefills_done_ns as f64 / 1e6,
            now_ns as f64 / 1e6,
        );
        assert!(
            all_prefills_done_ns <= 2_000_000_000,
            "16 single-launch short prefills should complete without anchor-tail starvation (took {:.1} ms)",
            all_prefills_done_ns as f64 / 1e6,
        );
    }
}
