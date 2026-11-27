/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Completed speculative forecasts and unified captured generation rounds.

use super::{policy::{Candidate, bits}, utility::{LaneValue, ValueRate, best_cohort}, *};

const POPULATION_WEIGHT: f32 = 1.0 / 64.0;
const CALIBRATION_PRIOR_WEIGHT: f64 = 4.0;
const POPULATION_CALIBRATION_MAX_WEIGHT: f64 = 64.0;
const CALIBRATION_FACTOR_MAX: f64 = 2.0;
const REQUEST_CALIBRATION_PRIOR_WEIGHT: f64 = 2.0;
const REQUEST_CALIBRATION_MAX_WEIGHT: f64 = 32.0;
const REQUEST_CALIBRATION_HALF_LIFE: f64 = 128.0;
const REQUEST_CALIBRATION_ANCHOR_FLOOR: f64 = 0.05;
const SURPLUS_HISTORY_CAPACITY: usize = 256;
const SURPLUS_MIN_SAMPLES: usize = 4;
const SURPLUS_FACTOR_MAX: f64 = 2.0;

#[derive(Clone, Copy)]
struct DepthCalibration {
    realized: f64,
    predicted: f64,
    weight: f64,
}

impl DepthCalibration {
    fn new(prior: f64) -> Self {
        Self { realized: prior, predicted: prior, weight: CALIBRATION_PRIOR_WEIGHT }
    }

    fn observe(&mut self, realized: f64, predicted: f64) {
        debug_assert!(realized.is_finite() && (0.0..=1.0).contains(&realized));
        debug_assert!(predicted.is_finite() && (0.0..=1.0).contains(&predicted));
        self.weight = (self.weight + 1.0).min(POPULATION_CALIBRATION_MAX_WEIGHT);
        self.realized += (realized - self.realized) / self.weight;
        self.predicted += (predicted - self.predicted) / self.weight;
    }

    fn factor(self) -> f64 {
        if self.predicted > 1e-6 {
            (self.realized / self.predicted).clamp(0.0, CALIBRATION_FACTOR_MAX)
        } else {
            1.0
        }
    }
}

/// Request-local aged per-depth verification calibration shrunk toward the global prior.
#[derive(Clone, Copy)]
struct RequestCalibration {
    realized: f64,
    predicted: f64,
    weight: f64,
    observed_at: u32,
}

impl RequestCalibration {
    fn new(generated: u32) -> Self {
        Self { realized: 0.0, predicted: 0.0, weight: 0.0, observed_at: generated }
    }

    fn decay(generated: u32, observed_at: u32) -> f64 {
        (-f64::from(generated.saturating_sub(observed_at)) / REQUEST_CALIBRATION_HALF_LIFE).exp2()
    }

    fn weight_at(self, generated: u32) -> f64 {
        self.weight * Self::decay(generated, self.observed_at)
    }

    fn information_at(self, generated: u32) -> f64 {
        let base = if self.weight > 0.0 { self.weight } else { REQUEST_CALIBRATION_PRIOR_WEIGHT };
        base * Self::decay(generated, self.observed_at)
    }

    fn factor(self, global: DepthCalibration) -> f64 {
        if self.weight == 0.0 {
            global.factor()
        } else if self.predicted > 1e-6 {
            (self.realized / self.predicted).clamp(0.0, CALIBRATION_FACTOR_MAX)
        } else {
            1.0
        }
    }

    fn observe(&mut self, realized: f64, predicted: f64, generated: u32, global: DepthCalibration) {
        debug_assert!(realized.is_finite() && (0.0..=1.0).contains(&realized));
        debug_assert!(predicted.is_finite() && (0.0..=1.0).contains(&predicted));
        let aged = if self.weight == 0.0 {
            let anchor = predicted.max(REQUEST_CALIBRATION_ANCHOR_FLOOR);
            self.predicted = anchor;
            self.realized = (anchor * global.factor()).clamp(0.0, 1.0);
            REQUEST_CALIBRATION_PRIOR_WEIGHT * Self::decay(generated, self.observed_at)
        } else {
            self.weight_at(generated)
        };
        self.weight = (aged + 1.0).min(REQUEST_CALIBRATION_MAX_WEIGHT);
        self.realized += (realized - self.realized) / self.weight;
        self.predicted += (predicted - self.predicted) / self.weight;
        self.observed_at = generated;
    }
}

/// Rolling windows of paired `(actual, pre-step forecast)` calibrated prefix curves
/// for estimating the GPU causal prefix allocator's cross-lane order-statistic surplus.
/// Depth-1 (`width = 2`) pairs are stored in a dedicated 256-entry ring so sustained
/// `W=2` rounds keep depth-1 surplus fresh without evicting deeper (`depth >= 2`) curves.
struct SurplusHistory {
    deep: Vec<(Vec<f32>, Vec<f32>)>,
    deep_next: usize,
    depth_one: Vec<(f32, f32)>,
    depth_one_next: usize,
}

impl SurplusHistory {
    fn new() -> Self {
        Self {
            deep: Vec::with_capacity(SURPLUS_HISTORY_CAPACITY),
            deep_next: 0,
            depth_one: Vec::with_capacity(SURPLUS_HISTORY_CAPACITY),
            depth_one_next: 0,
        }
    }

    fn push(&mut self, actual: Vec<f32>, forecast: Vec<f32>) {
        debug_assert_eq!(actual.len(), forecast.len());
        if let (Some(&a0), Some(&f0)) = (actual.first(), forecast.first()) {
            if self.depth_one.len() < SURPLUS_HISTORY_CAPACITY {
                self.depth_one.push((a0, f0));
            } else {
                self.depth_one[self.depth_one_next] = (a0, f0);
                self.depth_one_next = (self.depth_one_next + 1) % SURPLUS_HISTORY_CAPACITY;
            }
        }
        if actual.len() >= 2 {
            if self.deep.len() < SURPLUS_HISTORY_CAPACITY {
                self.deep.push((actual, forecast));
            } else {
                self.deep[self.deep_next] = (actual, forecast);
                self.deep_next = (self.deep_next + 1) % SURPLUS_HISTORY_CAPACITY;
            }
        }
    }

    fn build_from_values(depth: usize, samples: usize, mut actual_vals: Vec<f64>, mut forecast_vals: Vec<f64>) -> SurplusTable {
        actual_vals.sort_unstable_by(|a, b| b.total_cmp(a));
        forecast_vals.sort_unstable_by(|a, b| b.total_cmp(a));
        let mut actual_prefix = Vec::with_capacity(actual_vals.len() + 1);
        let mut forecast_prefix = Vec::with_capacity(forecast_vals.len() + 1);
        actual_prefix.push(0.0);
        forecast_prefix.push(0.0);
        let (mut a_sum, mut f_sum) = (0.0, 0.0);
        for (a, f) in actual_vals.into_iter().zip(forecast_vals) {
            a_sum += a;
            f_sum += f;
            actual_prefix.push(a_sum);
            forecast_prefix.push(f_sum);
        }
        SurplusTable { depth, samples, actual_prefix, forecast_prefix }
    }

    fn build_depth_one(&self) -> Option<SurplusTable> {
        let samples = self.depth_one.len();
        if samples < SURPLUS_MIN_SAMPLES {
            return None;
        }
        let mut actual_vals = Vec::with_capacity(samples);
        let mut forecast_vals = Vec::with_capacity(samples);
        for &(a0, f0) in &self.depth_one {
            actual_vals.push(f64::from(a0).clamp(0.0, 1.0));
            forecast_vals.push(f64::from(f0).clamp(0.0, 1.0));
        }
        Some(Self::build_from_values(1, samples, actual_vals, forecast_vals))
    }

    fn build_table(&self, depth: usize) -> Option<SurplusTable> {
        if depth == 0 {
            return None;
        }
        if depth == 1 {
            return self.build_depth_one();
        }
        // Choose the deepest effective depth in 2..=depth with at least SURPLUS_MIN_SAMPLES
        // entries in `deep`, falling back conservatively to `depth_one` (eff_depth=1) only
        // when no deeper history exists yet, without extrapolating past observed depths.
        if self.deep.len() >= SURPLUS_MIN_SAMPLES {
            let mut counts = vec![0usize; depth];
            for (actual, _) in &self.deep {
                let capped = actual.len().min(depth);
                if capped >= 2 {
                    counts[capped - 1] += 1;
                }
            }
            let mut samples = 0usize;
            let mut eff_depth = 0usize;
            for d in (2..=depth).rev() {
                samples += counts[d - 1];
                if samples >= SURPLUS_MIN_SAMPLES {
                    eff_depth = d;
                    break;
                }
            }
            if eff_depth >= 2 {
                let mut actual_vals = Vec::with_capacity(samples * eff_depth);
                let mut forecast_vals = Vec::with_capacity(samples * eff_depth);
                for (actual, forecast) in &self.deep {
                    if actual.len() < eff_depth {
                        continue;
                    }
                    for d in 0..eff_depth {
                        actual_vals.push(f64::from(actual[d]).clamp(0.0, 1.0));
                        forecast_vals.push(f64::from(forecast[d]).clamp(0.0, 1.0));
                    }
                }
                return Some(Self::build_from_values(eff_depth, samples, actual_vals, forecast_vals));
            }
        }
        self.build_depth_one()
    }
}

pub(super) struct SurplusTable {
    depth: usize,
    samples: usize,
    actual_prefix: Vec<f64>,
    forecast_prefix: Vec<f64>,
}

impl SurplusTable {
    pub(super) fn factor(&self, lanes: usize, draft_budget: usize, alpha: f64) -> f64 {
        if lanes < 2 || draft_budget == 0 || draft_budget >= lanes * self.depth {
            return 1.0;
        }
        let quota = draft_budget as f64 / lanes as f64;
        let pos = (quota * self.samples as f64).clamp(0.0, (self.samples * self.depth) as f64);
        let actual = interp_prefix(&self.actual_prefix, pos);
        let forecast = interp_prefix(&self.forecast_prefix, pos);
        let actual_total = *self.actual_prefix.last().unwrap_or(&0.0);
        let forecast_total = *self.forecast_prefix.last().unwrap_or(&0.0);
        if forecast <= 1e-9 || actual_total <= 1e-9 || forecast_total <= 1e-9 {
            return 1.0;
        }
        // Normalized Lorenz concentration ratio over linear calibrated survival:
        // identically 1.0 at full quota, under pure mean shifts, and under pure
        // between-request heterogeneity. Dampen smoothly when alpha < 1.0 already
        // applies a concave boost to prefix survival.
        let raw_ratio = (actual * forecast_total) / (forecast * actual_total);
        let cohort_scale = (1.0 - 1.0 / lanes as f64) * alpha.min(1.0);
        (1.0 + cohort_scale * (raw_ratio - 1.0)).clamp(1.0, SURPLUS_FACTOR_MAX)
    }
}

fn interp_prefix(prefix: &[f64], pos: f64) -> f64 {
    let lo = (pos.floor() as usize).min(prefix.len() - 1);
    let hi = (lo + 1).min(prefix.len() - 1);
    let frac = pos - lo as f64;
    prefix[lo] + frac * (prefix[hi] - prefix[lo])
}

#[derive(Clone)]
struct Forecast {
    request: RequestHandle,
    prefixes: Vec<speculation::PrefixEstimate>,
    calibration: Vec<RequestCalibration>,
}

impl Forecast {
    fn new(request: RequestHandle, prior: &[f32], generated: u32) -> Self {
        Self {
            request,
            prefixes: prior.iter().map(|&p| speculation::PrefixEstimate::new(f64::from(p), generated)).collect(),
            calibration: vec![RequestCalibration::new(generated); prior.len()],
        }
    }

    fn calibration_factor(&self, depth: usize, global: &[DepthCalibration]) -> f64 {
        let global_entry = global.get(depth).copied().unwrap_or(DepthCalibration::new(1.0));
        self.calibration.get(depth).map_or_else(|| global_entry.factor(), |entry| entry.factor(global_entry))
    }

    fn calibration_information(&self, depth: usize, generated: u32) -> f64 {
        self.calibration.get(depth).map_or(REQUEST_CALIBRATION_PRIOR_WEIGHT, |entry| entry.information_at(generated))
    }

    fn observe_calibration(&mut self, depth: usize, realized: f64, predicted: f64, generated: u32, global: &[DepthCalibration]) {
        if let Some(entry) = self.calibration.get_mut(depth) {
            let global_entry = global.get(depth).copied().unwrap_or(DepthCalibration::new(1.0));
            entry.observe(realized, predicted, generated, global_entry);
        }
    }
}

pub(super) struct Forecasts {
    requests: [Option<Forecast>; REQUEST_LANES],
    pub(super) population: Vec<f32>,
    calibration: Vec<DepthCalibration>,
    surplus: SurplusHistory,
}

impl Forecasts {
    pub(super) fn new(token_capacity: usize) -> Self {
        let population: Vec<f32> = (0..token_capacity - 1)
            .map(|i| 0.5f32.powi(i as i32 + 1)).collect();
        let calibration = population.iter().map(|&p| DepthCalibration::new(f64::from(p))).collect();
        Self {
            requests: std::array::from_fn(|_| None),
            population,
            calibration,
            surplus: SurplusHistory::new(),
        }
    }

    /// Moves the mean survival of the prefix at `depth` toward one observed value.
    pub(super) fn observe_population(&mut self, depth: usize, prefix: f32) {
        debug_assert!(prefix.is_finite() && (0.0..=1.0).contains(&prefix));
        self.population[depth] += POPULATION_WEIGHT * (prefix - self.population[depth]);
    }

    /// Calibrates raw GPU prefix confidence against realized verification outcomes at `depth`.
    pub(super) fn observe_calibration(&mut self, depth: usize, realized: f64, predicted: f64) {
        if let Some(entry) = self.calibration.get_mut(depth) {
            entry.observe(realized, predicted);
        }
    }

    #[cfg(test)]
    pub(super) fn calibration_factor(&self, depth: usize) -> f64 {
        self.calibration.get(depth).map_or(1.0, |entry| entry.factor())
    }

    #[cfg(test)]
    pub(super) fn request_calibration_factor(&self, request: RequestHandle, depth: usize) -> f64 {
        self.requests[request.slot as usize]
            .as_ref()
            .filter(|forecast| forecast.request == request)
            .map_or_else(
                || self.calibration_factor(depth),
                |forecast| forecast.calibration_factor(depth, &self.calibration),
            )
    }

    pub(super) fn observe_surplus(&mut self, actual: Vec<f32>, forecast: Vec<f32>) {
        self.surplus.push(actual, forecast);
    }

    pub(super) fn surplus_table(&self, depth: usize) -> Option<SurplusTable> {
        self.surplus.build_table(depth)
    }

    #[cfg(test)]
    pub(super) fn surplus_deep_entries(&self) -> usize {
        self.surplus.deep.len()
    }

    #[cfg(test)]
    pub(super) fn surplus_depth_one_entries(&self) -> usize {
        self.surplus.depth_one.len()
    }
}

pub(super) struct HostPrefixRanking {
    probabilities: Vec<Vec<f64>>,
    ranks_by_slot: [Vec<usize>; REQUEST_LANES],
    slots_by_rank: Vec<usize>,
}

impl HostPrefixRanking {
    pub(super) fn extend_cohort(&self, slot: usize, lengths: &mut [usize; REQUEST_LANES],
        frontier: &mut std::collections::BinaryHeap<usize>, draft_budget: usize) {
        lengths[slot] = 1;
        for &rank in &self.ranks_by_slot[slot] {
            frontier.push(rank);
            lengths[slot] += 1;
        }
        while frontier.len() > draft_budget {
            lengths[self.slots_by_rank[frontier.pop().unwrap()]] -= 1;
        }
    }
}

fn rank_prefixes(positions: &mut [(f64, usize, usize)]) {
    // Survival is nonincreasing with depth; ties keep predecessors before successors.
    positions.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
}

impl RequestScheduler {
    pub(super) fn pending_ceiling(&self, request: &LiveRequest) -> usize {
        // Completion reconciles the scheduled cursor with the remaining submissions.
        let pending = (request.scheduled_consumed - request.consumed_tokens) as usize;
        debug_assert_eq!(pending, self.pending_token_ceiling(request.handle));
        pending
    }

    pub(super) fn pending_token_ceiling(&self, handle: RequestHandle) -> usize {
        self.submissions.values().map(|submission| submission.lanes.iter()
            .filter(|lane| lane.request == handle)
            .map(|lane| submission.lane_advance_ceiling(lane)).sum::<usize>()).sum()
    }

    pub(super) fn speculative_curve(&self, request: &LiveRequest, optimistic: bool) -> Vec<f64> {
        if self.proposal_source == speculation::ProposalSource::Mtp && self.gpu_confidence() {
            let forecast = self.forecasts.requests[request.handle.slot as usize].as_ref()
                .filter(|forecast| forecast.request == request.handle)
                .cloned().unwrap_or_else(|| Forecast::new(request.handle, &self.forecasts.population, request.generated_tokens));
            let mut bound = 1.0f64;
            forecast.prefixes.iter().enumerate().map(|(depth, estimate)| {
                let calibrated = estimate.mean() * forecast.calibration_factor(depth, &self.forecasts.calibration);
                let value = if optimistic {
                    let info = estimate.weight_at(request.generated_tokens)
                        .min(forecast.calibration_information(depth, request.generated_tokens));
                    calibrated + (20.0f64.ln() / (2.0 * info.max(1e-9))).sqrt()
                } else {
                    calibrated
                };
                bound = bound.min(value.clamp(0.0, 1.0));
                bound
            }).collect()
        } else {
            request.acceptance.curve(self.proposal_source, request.generated_tokens, optimistic)
        }
    }

    /// Whether the model reports per-depth confidence on the GPU rather than leaving it to host estimates.
    pub(super) fn gpu_confidence(&self) -> bool {
        self.capabilities.speculative_allocations.iter().any(|shape| shape.4 == crate::execution::PrefixAllocation::Gpu)
    }

    pub(super) fn gpu_prefix_allocation(&self, batch: usize, drafts: usize, width: usize, packed: usize) -> bool {
        self.capabilities.speculative_allocations.iter().any(|&(b, d, w, p, allocation)|
            (b, d, w, p) == (batch, drafts, width, packed) && allocation == crate::execution::PrefixAllocation::Gpu)
    }

    fn round_ceilings(&self, mask: u64, drafts: usize, width: usize) -> [usize; REQUEST_LANES] {
        std::array::from_fn(|slot| {
            if mask & (1 << slot) == 0 { 0 } else {
                self.speculative_tokens(self.slots[slot].request.as_ref().unwrap(), width.min(drafts + 1))
            }
        })
    }

    pub(super) fn round_lengths(&self, mask: u64, batch: usize, drafts: usize, width: usize, packed: usize) -> [usize; REQUEST_LANES] {
        self.round_lengths_optimistic(mask, batch, drafts, width, packed, false)
    }

    pub(super) fn round_lengths_optimistic(&self, mask: u64, batch: usize, drafts: usize, width: usize, packed: usize, optimistic: bool) -> [usize; REQUEST_LANES] {
        let ceilings = self.round_ceilings(mask, drafts, width);
        if self.gpu_prefix_allocation(batch, drafts, width, packed)
            || self.speculative_fixed_width.is_some() || ceilings.iter().sum::<usize>() <= packed {
            return ceilings;
        }
        let (values, _, _) = self.lane_values(self.window_remaining().unwrap_or(0));
        self.prefix_lengths_with_values(mask, batch, drafts, width, packed, true, &values, optimistic)
    }

    pub(super) fn estimated_round_lengths(&self, mask: u64, batch: usize, drafts: usize, width: usize, packed: usize, optimistic: bool) -> [usize; REQUEST_LANES] {
        let (values, _, _) = self.lane_values(self.window_remaining().unwrap_or(0));
        self.prefix_lengths_with_values(mask, batch, drafts, width, packed, false, &values, optimistic)
    }

    pub(super) fn prefix_lengths_with_values(&self, mask: u64, batch: usize, drafts: usize, width: usize, packed: usize,
        preserve_gpu_ceilings: bool, values: &[LaneValue; REQUEST_LANES], optimistic: bool) -> [usize; REQUEST_LANES] {
        let mut lengths = [0; REQUEST_LANES];
        let ceiling = self.round_ceilings(mask, drafts, width);
        let gpu = self.gpu_prefix_allocation(batch, drafts, width, packed);
        if (gpu && preserve_gpu_ceilings) || (!gpu && self.speculative_fixed_width.is_some())
            || ceiling.iter().sum::<usize>() <= packed { return ceiling; }
        for slot in bits(mask) { lengths[slot] = 1; }
        let remaining = packed.saturating_sub(mask.count_ones() as usize);
        if remaining == 0 { return lengths; }
        let probabilities: [Vec<f64>; REQUEST_LANES] = std::array::from_fn(|slot| {
            if mask & (1 << slot) == 0 { return Vec::new(); }
            let request = self.slots[slot].request.as_ref().unwrap();
            if gpu { self.speculative_curve(request, optimistic) }
            else { request.acceptance.curve(self.proposal_source, request.generated_tokens, optimistic)
                .into_iter().map(|probability| probability.powf(self.policy.adaptive_verification_alpha) * values[slot].output_token).collect() }
        });
        let mut positions = Vec::new();
        for slot in bits(mask) {
            for depth in 0..ceiling[slot].saturating_sub(1) {
                positions.push((probabilities[slot][depth], depth, slot));
            }
        }
        rank_prefixes(&mut positions);
        for (_, _, slot) in positions.into_iter().take(remaining) {
            lengths[slot] += 1;
        }
        lengths
    }

    pub(super) fn host_prefix_ranking(&self, available: &[usize], drafts: usize, width: usize,
        values: &[LaneValue; REQUEST_LANES], optimistic: bool) -> HostPrefixRanking {
        let mut probabilities = vec![vec![0.0; self.capabilities.completion_token_capacity - 1]; REQUEST_LANES];
        let mut scores = Vec::new();
        for &slot in available {
            let request = self.slots[slot].request.as_ref().unwrap();
            let limit = self.speculative_tokens(request, width.min(drafts + 1));
            probabilities[slot] = request.acceptance.curve(self.proposal_source, request.generated_tokens, optimistic)
                .into_iter().map(|probability| probability.powf(self.policy.adaptive_verification_alpha)).collect();
            for position in 0..limit.saturating_sub(1) {
                scores.push((probabilities[slot][position] * values[slot].output_token, position, slot));
            }
        }
        rank_prefixes(&mut scores);
        let mut ranks_by_slot = std::array::from_fn(|_| Vec::new());
        let mut slots_by_rank = Vec::with_capacity(scores.len());
        for (rank, &(_, _, slot)) in scores.iter().enumerate() {
            ranks_by_slot[slot].push(rank);
            slots_by_rank.push(slot);
        }
        HostPrefixRanking { probabilities, ranks_by_slot, slots_by_rank }
    }

    pub(super) fn speculative_information(&self, request: &LiveRequest, depth: usize) -> f64 {
        self.forecasts.requests[request.handle.slot as usize].as_ref().filter(|forecast| forecast.request == request.handle)
            .map_or_else(|| request.acceptance.information(self.proposal_source, depth, request.generated_tokens),
                |forecast| {
                    let info = forecast.prefixes[depth].weight_at(request.generated_tokens);
                    if self.proposal_source == speculation::ProposalSource::Mtp && self.gpu_confidence() {
                        info.min(forecast.calibration_information(depth, request.generated_tokens))
                    } else {
                        info
                    }
                })
    }

    pub(super) fn expected_speculative_tokens(&self, request: &LiveRequest, width: usize) -> f64 {
        self.proposal_forecast_tokens(request, width, false)
    }

    fn proposal_forecast_tokens(&self, request: &LiveRequest, width: usize, optimistic: bool) -> f64 {
        forecast_tokens(&self.powered_curve(request, optimistic), width)
    }

    /// Prefix survival raised to the adaptive verification exponent.
    pub(super) fn powered_curve(&self, request: &LiveRequest, optimistic: bool) -> Vec<f64> {
        self.speculative_curve(request, optimistic).into_iter().map(|p| p.powf(self.policy.adaptive_verification_alpha)).collect()
    }

    pub(super) fn speculative_duration(&self, request: &LiveRequest, batch: usize, width: usize) -> Option<u64> {
        if self.speculative_fixed_width.is_none() {
            // The full B x W graph is among the rounds this table minimizes over.
            return self.costs.packed_round_duration(batch, width);
        }
        let length = self.speculative_tokens(request, width);
        self.costs.packed_round_shapes().filter(|shape| shape.batch == batch && shape.width == width
            && shape.drafts + 1 >= length && shape.tokens >= length)
            .fold(None, |best, shape| cost::min_measured_duration(best, shape.nanos))
    }

    pub(super) fn observe_packed_forecasts(&mut self, step: &ScheduledStep,
        records: &[CompletionRecord], confidence: &[Vec<f32>],
        observed_realized: &[(usize, usize, f64)]) {
        // A model that does not report GPU confidence feeds the population from host estimates instead.
        let gpu_confidence = self.gpu_confidence();
        let mut round_cal = if gpu_confidence {
            vec![(0.0f64, 0.0f64, 0usize); self.forecasts.calibration.len()]
        } else {
            Vec::new()
        };
        let mut cursor = 0usize;
        for (row, lane) in step.lanes.iter().enumerate() {
            while cursor < observed_realized.len() && observed_realized[cursor].0 < row {
                cursor += 1;
            }
            let row_start = cursor;
            while cursor < observed_realized.len() && observed_realized[cursor].0 == row {
                cursor += 1;
            }
            let row_realized = &observed_realized[row_start..cursor];
            let record = &records[lane.request.slot as usize];
            let Some(vector) = record.vector.as_ref() else { continue; };
            if !matches!(record.status, 1 | 2) || vector.emitted_count == 0
                || self.request(lane.request).is_err() { continue; }
            let generated = (record.generated_tokens - vector.emitted_count) as u32;
            let mut forecast = self.forecasts.requests[lane.request.slot as usize].take()
                .filter(|forecast| forecast.request == lane.request)
                .unwrap_or_else(|| Forecast::new(lane.request, &self.forecasts.population, generated));
            let depth_limit = lane.tokens.saturating_sub(1).min(step.draft_tokens);
            let cal_factors = if gpu_confidence && depth_limit >= 1 {
                (0..depth_limit)
                    .map(|depth| forecast.calibration_factor(depth, &self.forecasts.calibration))
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let mut host_pred = if !cal_factors.is_empty() {
                let mut bound = 1.0f64;
                (0..depth_limit).map(|depth| {
                    let calibrated = forecast.prefixes[depth].mean() * cal_factors[depth];
                    bound = bound.min(calibrated.clamp(0.0, 1.0));
                    bound as f32
                }).collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let mut actual_cal = if !host_pred.is_empty() {
                Vec::with_capacity(depth_limit)
            } else {
                Vec::new()
            };
            let mut predicted_prefixes = Vec::with_capacity(depth_limit);
            let mut prefix = 1.0f32;
            let mut actual_bound = 1.0f64;
            for depth in 0..depth_limit {
                prefix *= confidence[row][depth];
                predicted_prefixes.push(prefix);
                if !host_pred.is_empty() {
                    let calibrated = f64::from(prefix) * cal_factors[depth];
                    actual_bound = actual_bound.min(calibrated.clamp(0.0, 1.0));
                    actual_cal.push(actual_bound as f32);
                }
                forecast.prefixes[depth].observe(f64::from(prefix), generated);
                if gpu_confidence { self.forecasts.observe_population(depth, prefix); }
            }
            if !host_pred.is_empty() {
                self.forecasts.observe_surplus(actual_cal, std::mem::take(&mut host_pred));
            }
            if gpu_confidence {
                for &(_, depth, realized) in row_realized {
                    if let Some(&predicted) = predicted_prefixes.get(depth) {
                        let pred = f64::from(predicted);
                        forecast.observe_calibration(depth, realized, pred, generated, &self.forecasts.calibration);
                        if let Some(acc) = round_cal.get_mut(depth) {
                            acc.0 += realized;
                            acc.1 += pred;
                            acc.2 += 1;
                        }
                    }
                }
            }
            self.forecasts.requests[lane.request.slot as usize] = Some(forecast);
        }
        // Update the global calibration prior once per completed round using the round's
        // mean realized and predicted survival across contributing lanes, so a single
        // correlated large-batch dip cannot overwrite the population prior in one step.
        if gpu_confidence {
            for (depth, (sum_real, sum_pred, count)) in round_cal.into_iter().enumerate() {
                if count > 0 {
                    let denom = count as f64;
                    self.forecasts.observe_calibration(depth, sum_real / denom, sum_pred / denom);
                }
            }
        }
    }
}

impl RequestScheduler {
    fn fixed_round_candidate(&self, shape: &cost::PackedRoundShape, prefixes: &[(usize, usize, f64)],
        values: &[LaneValue; REQUEST_LANES], age: f64) -> Option<Candidate> {
        let offers: Vec<_> = prefixes.iter().filter_map(|&(slot, length, expected)| {
            (length != 0 && length <= shape.tokens).then(||
                (slot, length, values[slot].decode(expected, shape.nanos as f64 / 1e6)))
        }).collect();
        let &(_, first_length, _) = offers.first()?;
        let scores: Vec<_> = offers.iter().map(|&(slot, _, value)| (slot, value)).collect();
        let unconstrained = best_cohort::<REQUEST_LANES>(&scores, shape.batch);
        let full_rows: usize = offers.iter().filter(|&&(slot, _, _)| unconstrained & (1 << slot) != 0)
            .map(|offer| offer.1).sum();
        let mask = if full_rows <= shape.tokens {
            unconstrained
        } else if offers.iter().all(|&(_, length, _)| length == first_length) {
            best_cohort::<REQUEST_LANES>(&scores, shape.batch.min(shape.tokens / first_length))
        } else {
            let budget = shape.tokens.min(offers.iter().map(|offer| offer.1).sum());
            let mut lengths: Vec<_> = offers.iter().map(|offer| offer.1).collect();
            lengths.sort_unstable();
            let mut minimum_rows = vec![0];
            for length in lengths {
                minimum_rows.push(minimum_rows.last().unwrap() + length);
            }
            let maximum_lanes = minimum_rows.iter().take_while(|&&rows| rows <= budget).count() - 1;
            let count = shape.batch.min(maximum_lanes);
            // A row budget that already bounds cohort size needs no cardinality dimension.
            let cardinality = maximum_lanes > shape.batch;
            let stride = budget + 1;
            let layers = if cardinality { count + 1 } else { 1 };
            let mut states = vec![None; layers * stride];
            states[0] = Some((0.0, 0u64));
            let maximum_length = offers.iter().map(|offer| offer.1).max().unwrap();
            for (index, &(slot, length, value)) in offers.iter().enumerate() {
                let first = usize::from(cardinality);
                let last = if cardinality { count.min(index + 1) } else { 0 };
                for lanes in (first..=last).rev() {
                    let previous = if cardinality { lanes - 1 } else { 0 };
                    let lower = length + if cardinality { minimum_rows[previous] } else { 0 };
                    let upper = budget.min(length + if cardinality { previous * maximum_length } else { budget });
                    for rows in (lower..=upper).rev() {
                        let Some((before, mask)) = states[previous * stride + rows - length] else { continue; };
                        let candidate = (before + value, mask | (1 << slot));
                        let destination = &mut states[lanes * stride + rows];
                        if destination.is_none_or(|old: (f64, u64)| candidate.0 > old.0
                            || candidate.0 == old.0 && (candidate.1.count_ones() > old.1.count_ones()
                                || candidate.1.count_ones() == old.1.count_ones() && candidate.1 < old.1)) {
                            *destination = Some(candidate);
                        }
                    }
                }
            }
            states.into_iter().flatten().max_by(|a, b| a.0.total_cmp(&b.0)
                .then(a.1.count_ones().cmp(&b.1.count_ones())).then(b.1.cmp(&a.1)))?.1
        };
        if mask == 0 { return None; }
        let value = offers.iter().filter(|&&(slot, _, _)| mask & (1 << slot) != 0)
            .map(|offer| offer.2).sum();
        let mut host_lengths = [0; REQUEST_LANES];
        for &(slot, length, _) in &offers {
            if mask & (1 << slot) != 0 { host_lengths[slot] = length; }
        }
        Some(Candidate { host_lengths: Some(host_lengths), draft_tokens: shape.drafts, kind: GraphKind::Speculative,
            mask, batch: shape.batch, capacity: shape.width, packed_token_capacity: shape.tokens,
            prefill_capacity: 0, prefill_slot: None, work: 0, duration_ns: shape.nanos,
            rate: ValueRate::new(shape.nanos, value, age), feedback: false })
    }

    pub(super) fn packed_round_candidates(&self, output: &mut Vec<Candidate>,
        values: &[LaneValue; REQUEST_LANES], age: f64, optimistic: bool) {
        let available: Vec<_> = self.slots.iter().enumerate().filter_map(|(slot, entry)| {
            let request = entry.request.as_ref()?;
            (request.phase == RequestPhase::Decode && self.eligible(request)).then_some(slot)
        }).collect();
        let mut forecasts = HashMap::new();
        let mut fixed_prefixes = HashMap::new();
        let mut host_rankings = HashMap::new();
        // Lane forecasts stay fixed while shapes compete; evaluate each at most once.
        let mut curves: [Option<Vec<f64>>; REQUEST_LANES] = std::array::from_fn(|_| None);
        let mut order = Vec::with_capacity(available.len());
        let mut frontier = std::collections::BinaryHeap::new();
        for shape in self.costs.packed_round_shapes() {
            if shape.batch > self.limits.request_slots
                || !self.speculative_widths_for(shape.batch).contains(&shape.width) { continue; }
            if self.speculative_fixed_width.is_some()
                && !self.gpu_prefix_allocation(shape.batch, shape.drafts, shape.width, shape.tokens) {
                let prefixes = fixed_prefixes.entry((shape.drafts, shape.width)).or_insert_with(|| {
                    available.iter().map(|&slot| {
                        let request = self.slots[slot].request.as_ref().unwrap();
                        let length = self.speculative_tokens(request, shape.width.min(shape.drafts + 1));
                        let expected = f64::from(length != 0) + request.acceptance.curve(
                            self.proposal_source, request.generated_tokens, false)[..length.saturating_sub(1)]
                            .iter().map(|&probability| {
                                if self.policy.adaptive_verification_alpha == 1.0 { probability }
                                else { probability.powf(self.policy.adaptive_verification_alpha) }
                            }).sum::<f64>();
                        (slot, length, expected)
                    }).collect::<Vec<_>>()
                });
                if let Some(candidate) = self.fixed_round_candidate(&shape, prefixes, values, age) {
                    output.push(candidate);
                }
                continue;
            }
            // Speculative positions ranked by marginal forecast tokens; shapes with equal
            // draft and width bounds share the ranking.
            let gpu_alloc = self.gpu_prefix_allocation(shape.batch, shape.drafts, shape.width, shape.tokens);
            let depth_cap = shape.drafts.min(shape.width.saturating_sub(1));
            let forecast = forecasts.entry((shape.drafts, shape.width)).or_insert_with(|| {
                let mut scores = Vec::new();
                for &slot in &available {
                    let request = self.slots[slot].request.as_ref().unwrap();
                    let limit = self.speculative_tokens(request, (shape.drafts + 1).min(shape.width));
                    let curve = curves[slot].get_or_insert_with(|| self.powered_curve(request, optimistic));
                    let mut before = 1.0;
                    for position in 0..limit.saturating_sub(1) {
                        let total = forecast_tokens(curve, position + 2);
                        scores.push((total - before, position, slot));
                        before = total;
                    }
                }
                scores.sort_unstable_by(|a, b|
                    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
                let gains: Vec<_> = scores.iter().map(|&(score, _, slot)|
                    score * values[slot].output_token).collect();
                let slot_prices: Vec<_> = scores.iter().map(|&(_, _, slot)|
                    values[slot].output_token).collect();
                let mut ranks: [Vec<usize>; REQUEST_LANES] = std::array::from_fn(|_| Vec::new());
                for (rank, &(_, _, slot)) in scores.iter().enumerate() { ranks[slot].push(rank); }
                let surplus = if self.proposal_source == speculation::ProposalSource::Mtp && self.gpu_confidence() {
                    self.forecasts.surplus_table(depth_cap)
                } else {
                    None
                };
                (gains, slot_prices, ranks, surplus)
            });
            let (gains, slot_prices, ranks, surplus_table) = (&forecast.0, &forecast.1, &forecast.2, &forecast.3);
            let host_ranking = if gpu_alloc {
                None
            } else {
                Some(host_rankings.entry((shape.drafts, shape.width)).or_insert_with(||
                    self.host_prefix_ranking(&available, shape.drafts, shape.width, values, optimistic)))
            };
            let nanos = shape.nanos;
            let mut base = [0.0; REQUEST_LANES];
            let mut full = [0.0; REQUEST_LANES];
            let mut limits = [0; REQUEST_LANES];
            for &slot in &available {
                let request = self.slots[slot].request.as_ref().unwrap();
                base[slot] = values[slot].decode(1.0, nanos as f64 / 1e6);
                full[slot] = base[slot] + ranks[slot].iter().take(shape.tokens - 1)
                    .map(|&rank| gains[rank]).sum::<f64>();
                limits[slot] = self.speculative_tokens(request, (shape.drafts + 1).min(shape.width));
            }
            let mut best: Option<(u64, f64, Option<[usize; REQUEST_LANES]>)> = None;
            // Alternate bounded orderings expose long useful prefixes, delivery
            // fairness, and short tails. Every cohort is priced by its causal
            // prefix allocation and the full measured physical B/D/W/P round.
            for ordering in 0..3 {
                order.clear();
                order.extend(available.iter().map(|&slot| (slot, match ordering {
                    0 => full[slot], 1 => base[slot], _ => full[slot] / limits[slot] as f64 })));
                // Only the leading cohort-size entries of the total order are used.
                let rank = |a: &(usize, f64), b: &(usize, f64)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
                let cohort = shape.batch.min(shape.tokens);
                if (1..order.len()).contains(&cohort) {
                    order.select_nth_unstable_by(cohort - 1, rank);
                    order.truncate(cohort);
                }
                order.sort_unstable_by(rank);
                frontier.clear();
                let mut host_lengths = [0; REQUEST_LANES];
                // A lane's value changes only when its host prefix length does.
                let mut lane_terms = [(0, 0.0); REQUEST_LANES];
                let (mut mask, mut pending_value, mut draft_value, mut draft_cap, mut total_ceiling_drafts) =
                    (0u64, 0., 0., 0., 0usize);
                for (count, &(slot, _)) in order.iter().take(shape.batch.min(shape.tokens)).enumerate() {
                    mask |= 1 << slot;
                    pending_value += base[slot];
                    let value = if let Some(ranking) = &host_ranking {
                        ranking.extend_cohort(slot, &mut host_lengths, &mut frontier, shape.tokens - count - 1);
                        bits(mask).map(|slot| {
                            if lane_terms[slot].0 != host_lengths[slot] {
                                let expected = 1.0 + ranking.probabilities[slot][..host_lengths[slot] - 1]
                                    .iter().sum::<f64>();
                                lane_terms[slot] = (host_lengths[slot], values[slot].decode(expected, nanos as f64 / 1e6));
                            }
                            lane_terms[slot].1
                        }).sum()
                    } else {
                        // The heap retains the best P-count speculative positions.
                        // Its capacity decreases as pending rows enter; discarded
                        // positions cannot re-enter at a subsequent cohort size.
                        for &rank in &ranks[slot] {
                            frontier.push(rank);
                            draft_value += gains[rank];
                            draft_cap += slot_prices[rank];
                        }
                        total_ceiling_drafts += ranks[slot].len();
                        let lanes = count + 1;
                        let draft_budget = shape.tokens - lanes;
                        while frontier.len() > draft_budget {
                            let evicted = frontier.pop().unwrap();
                            draft_value -= gains[evicted];
                            draft_cap -= slot_prices[evicted];
                        }
                        let effective_draft = if lanes >= 2 && draft_budget > 0 && draft_budget < total_ceiling_drafts {
                            let surplus = surplus_table.as_ref().map_or(1.0, |t|
                                t.factor(lanes, draft_budget, self.policy.adaptive_verification_alpha));
                            (draft_value * surplus).min(draft_cap)
                        } else {
                            draft_value
                        };
                        pending_value + effective_draft
                    };
                    if best.is_none_or(|(old_mask, old_value, _)|
                        value > old_value || (value == old_value &&
                            (mask.count_ones(), std::cmp::Reverse(mask)) >
                            (old_mask.count_ones(), std::cmp::Reverse(old_mask)))) {
                        best = Some((mask, value, host_ranking.as_ref().map(|_| host_lengths)));
                    }
                }
            }
            if let Some((mask, value, host_lengths)) = best {
                output.push(Candidate { host_lengths, draft_tokens: shape.drafts, kind: GraphKind::Speculative,
                    mask, batch: shape.batch, capacity: shape.width, packed_token_capacity: shape.tokens,
                    prefill_capacity: 0, prefill_slot: None, work: 0, duration_ns: nanos,
                    rate: ValueRate::new(nanos, value, age), feedback: false,
                    });
            }
        }
    }
}

/// Expected tokens from a round of `width` positions with powered prefix survival.
pub(super) fn forecast_tokens(powered: &[f64], width: usize) -> f64 {
    f64::from(width != 0) + powered[..width.saturating_sub(1)].iter().sum::<f64>()
}
