/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Retained duration estimates; zero denotes an unmeasured graph cost.

use super::ScheduledStep;
use crate::{
    Result,
    execution::{ExecutionTiming, GraphDescriptor, GraphKind},
};

#[cfg(test)]
thread_local! {
    /// Cost-table lookups on this thread, for tests that bound planning work.
    pub(super) static ENTRY_LOOKUPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(crate) fn observed_duration_ns(timing: ExecutionTiming) -> Option<u64> {
    timing
        .gpu_ns
        .into_iter()
        .flatten()
        .max()
        .filter(|ns| *ns > 0)
        .map(|ns| if timing.queued { ns } else { ns.max(timing.wall_ns) })
}

/// Prefers the shortest positive measured duration, yielding `0` only when every
/// aggregated shape is unmeasured.
pub(super) fn min_measured_duration(current: Option<u64>, nanos: u64) -> Option<u64> {
    Some(match current {
        None | Some(0) => nanos,
        Some(old) if nanos == 0 => old,
        Some(old) => old.min(nanos),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Key {
    pub drafts: usize,
    pub kind: GraphKind,
    pub batch: usize,
    pub tokens: usize,
    pub prefill: usize,
    pub packed: usize,
}

impl Key {
    pub fn from_step(step: &ScheduledStep) -> Self {
        Self { drafts: step.draft_tokens, kind: step.kind, batch: step.batch_capacity, tokens: step.token_capacity,
            prefill: step.prefill_capacity(), packed: step.packed_token_capacity }
    }
}

pub(super) struct PackedRoundShape {
    pub batch: usize,
    pub drafts: usize,
    pub width: usize,
    pub tokens: usize,
    pub nanos: u64,
}

struct Estimate {
    nanos: u64,
    replays: ReplayTimes,
    observed_at_ns: u64,
    last_attempt_ns: u64,
}

#[derive(Default)]
struct ReplayTimes {
    samples: [u64; 9],
    len: usize,
    next: usize,
}

impl ReplayTimes {
    fn observe(&mut self, sample: u64, fresh: bool) -> u64 {
        if !fresh {
            *self = Self::default();
        }
        self.samples[self.next] = sample;
        self.next = (self.next + 1) % self.samples.len();
        self.len = (self.len + 1).min(self.samples.len());
        let mut ordered = self.samples;
        ordered[..self.len].sort_unstable();
        let middle = self.len / 2;
        if self.len % 2 == 0 {
            ((u128::from(ordered[middle - 1]) + u128::from(ordered[middle]) + 1) / 2) as u64
        } else {
            ordered[middle]
        }
    }
}

impl Estimate {
    fn fresh(&self, now_ns: u64, ttl_ns: u64) -> bool {
        self.nanos != 0 && now_ns.saturating_sub(self.observed_at_ns) < ttl_ns
    }
}

fn kind_index(kind: GraphKind) -> usize {
    match kind {
        GraphKind::Init => 0,
        GraphKind::Restore => 1,
        GraphKind::Snapshot => 2,
        GraphKind::Prefill => 3,
        GraphKind::TerminalPrefill => 4,
        GraphKind::ScoredPrefill => 5,
        GraphKind::Decode => 6,
        GraphKind::Speculative => 7,
        GraphKind::Mixed => 8,
        GraphKind::Resume => 9,
        GraphKind::Vision => 10,
    }
}

#[derive(Default)]
pub(super) struct ActionCosts {
    entries: Vec<(Key, Estimate)>,
    index: std::collections::HashMap<Key, usize>,
    batches: [Vec<usize>; 11],
    packed_round_durations: Vec<Vec<Option<u64>>>,
    now_ns: u64,
    ttl_ns: u64,
}

impl ActionCosts {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub fn set_measured(&mut self, step: &ScheduledStep, nanos: u64) {
        let key = Key {
            drafts: step.draft_tokens,
            kind: step.kind,
            batch: step.batch_capacity,
            tokens: step.token_capacity,
            prefill: step.prefill_capacity(),
            packed: step.packed_token_capacity,
        };
        let estimate = &mut self.entries[self.index[&key]].1;
        estimate.nanos = nanos;
        estimate.replays.observe(nanos, false);
        estimate.observed_at_ns = self.now_ns;
        self.refresh_packed_durations();
    }

    pub fn prepared(graphs: &[GraphDescriptor], token_capacity: usize) -> Result<Self> {
        let mut entries = Vec::with_capacity(graphs.len());
        let mut index = std::collections::HashMap::with_capacity(graphs.len());
        for graph in graphs {
            let key = Key {
                drafts: graph.draft_tokens,
                kind: graph.kind,
                batch: graph.batch_capacity,
                tokens: graph.token_capacity,
                prefill: graph.prefill_capacity,
                packed: graph.packed_token_capacity,
            };
            if index.insert(key, entries.len()).is_some() {
                return Err("prepared graphs require unique shapes".into());
            }
            entries.push((
                key,
                Estimate {
                    nanos: 0,
                    replays: ReplayTimes::default(),
                    observed_at_ns: 0,
                    last_attempt_ns: 0,
                },
            ));
        }
        let index = entries.iter().enumerate().map(|(index, (key, _))| (*key, index)).collect();
        let mut batches: [Vec<usize>; 11] = Default::default();
        for (key, _) in &entries {
            batches[kind_index(key.kind)].push(key.batch);
        }
        for list in &mut batches {
            list.sort_unstable();
            list.dedup();
        }
        let mut costs = Self { entries, index, batches,
            packed_round_durations: (0..=super::REQUEST_LANES)
                .map(|_| vec![None; token_capacity + 1])
                .collect(),
            ttl_ns: super::PolicyConfig::default().estimate_ttl_ns, ..Default::default() };
        costs.refresh_packed_durations();
        Ok(costs)
    }

    pub fn batches(&self, kind: GraphKind) -> &[usize] {
        &self.batches[kind_index(kind)]
    }

    pub fn advance_to(&mut self, now_ns: u64, ttl_ns: u64) {
        self.now_ns = now_ns;
        self.ttl_ns = ttl_ns;
    }

    pub fn duration(&self, kind: GraphKind, batch: usize, tokens: usize) -> Option<u64> {
        self.shape_duration(kind, batch, tokens, 0)
    }

    fn refresh_packed_durations(&mut self) {
        for row in &mut self.packed_round_durations { row.fill(None); }
        for (key, estimate) in &self.entries {
            if key.kind != GraphKind::Speculative || key.packed == 0 { continue; }
            let Some(row) = self.packed_round_durations.get_mut(key.batch) else { continue; };
            let maximum = (key.drafts + 1).min(key.tokens).min(key.packed);
            let nanos = estimate.nanos;
            for duration in row.iter_mut().take(maximum + 1).skip(1) {
                *duration = min_measured_duration(*duration, nanos);
            }
        }
    }

    pub fn packed_round_duration(&self, batch: usize, width: usize) -> Option<u64> {
        self.packed_round_durations.get(batch)?.get(width).copied().flatten()
    }

    pub fn packed_round_shapes(&self) -> impl Iterator<Item = PackedRoundShape> + '_ {
        self.entries.iter().filter(|(key, _)| key.kind == GraphKind::Speculative && key.packed != 0)
            .map(|(key, estimate)| PackedRoundShape {
                batch: key.batch, drafts: key.drafts, width: key.tokens, tokens: key.packed,
                nanos: estimate.nanos,
            })
    }

    pub fn shape_duration(
        &self,
        kind: GraphKind,
        batch: usize,
        tokens: usize,
        prefill: usize,
    ) -> Option<u64> {
        self.exact_duration(kind, batch, tokens, prefill, 0)
    }

    fn exact_duration(&self, kind: GraphKind, batch: usize, tokens: usize, prefill: usize, packed: usize) -> Option<u64> {
        self.entry(Key {
            drafts: if kind == GraphKind::Speculative { tokens - 1 } else { 0 },
            kind,
            batch,
            tokens,
            prefill,
            packed: if kind == GraphKind::Speculative { batch * tokens } else { packed },
        })
        .map(|estimate| estimate.nanos)
    }

    pub fn duration_key(&self, key: Key) -> Option<u64> {
        self.entry(key).map(|estimate| estimate.nanos)
    }

    pub fn is_fresh(&self, key: Key) -> bool {
        self.entry(key).is_some_and(|estimate| estimate.fresh(self.now_ns, self.ttl_ns))
    }

    pub fn fresh_duration(&self, kind: GraphKind, tokens: usize) -> Option<u64> {
        let key = Key { drafts: 0, kind, batch: 1, tokens, prefill: 0, packed: 0 };
        self.entry(key).filter(|estimate| estimate.fresh(self.now_ns, self.ttl_ns))
            .map(|estimate| estimate.nanos)
    }

    pub fn stale_keys(&self) -> impl Iterator<Item = Key> + '_ {
        self.entries.iter().filter(|(_, estimate)| estimate.nanos != 0
            && !estimate.fresh(self.now_ns, self.ttl_ns)).map(|(key, _)| *key)
    }

    pub fn refresh_order(&self, key: Key) -> u64 {
        self.entry(key).map_or(0, |estimate| estimate.observed_at_ns.max(estimate.last_attempt_ns))
    }

    pub fn record_attempt(&mut self, key: Key) {
        if let Some(&index) = self.index.get(&key) {
            let estimate = &mut self.entries[index].1;
            estimate.last_attempt_ns = self.now_ns;
        }
    }

    pub fn observation(&self, key: Key) -> Option<(u64, Option<u64>, bool)> {
        self.entry(key).map(|estimate| (estimate.nanos,
            (estimate.nanos != 0).then_some(estimate.observed_at_ns),
            estimate.fresh(self.now_ns, self.ttl_ns)))
    }

    fn entry(&self, key: Key) -> Option<&Estimate> {
        #[cfg(test)]
        ENTRY_LOOKUPS.set(ENTRY_LOOKUPS.get() + 1);
        self.index.get(&key).map(|&index| &self.entries[index].1)
    }

    pub fn get(&self, step: &ScheduledStep) -> Option<u64> {
        self.duration_key(Key::from_step(step))
    }

    pub fn observe(&mut self, step: &ScheduledStep, timing: ExecutionTiming) {
        let Some(raw_sample) = timing.gpu_ns.into_iter().flatten().max().filter(|n| *n > 0) else {
            return;
        };
        let sample = raw_sample.saturating_sub(timing.readiness.map_or(0, |intervals| intervals.wait_ns()));
        if sample == 0 {
            return;
        }
        let key = Key {
            drafts: step.draft_tokens,
            kind: step.kind,
            batch: step.batch_capacity,
            tokens: step.token_capacity,
            prefill: step.prefill_capacity(),
            packed: step.packed_token_capacity,
        };
        if let Some(&index) = self.index.get(&key) {
            let estimate = &mut self.entries[index].1;
            let fresh = estimate.fresh(self.now_ns, self.ttl_ns);
            estimate.nanos = estimate.replays.observe(sample, fresh);
            estimate.observed_at_ns = self.now_ns;
            self.refresh_packed_durations();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ActionCosts, ReplayTimes, ScheduledStep};
    use crate::execution::{GraphDescriptor, GraphKind, SamplingMode};

    #[test]
    fn isolated_slow_replay_does_not_displace_the_steady_cost() {
        let mut replays = ReplayTimes::default();
        for _ in 0..9 {
            assert_eq!(replays.observe(8_000_000, true), 8_000_000);
        }
        assert_eq!(replays.observe(16_000_000, true), 8_000_000);
        for _ in 0..9 {
            assert_eq!(replays.observe(8_000_000, true), 8_000_000);
        }
    }

    #[test]
    fn sustained_cost_changes_replace_the_window_and_staleness_resets_it() {
        let mut replays = ReplayTimes::default();
        for _ in 0..9 {
            replays.observe(8, true);
        }
        for _ in 0..4 {
            assert_eq!(replays.observe(16, true), 8);
        }
        assert_eq!(replays.observe(16, true), 16);
        assert_eq!(replays.observe(32, false), 32);
        assert_eq!(replays.observe(16, true), 24);
    }

    #[test]
    fn even_sample_median_does_not_overflow() {
        let mut replays = ReplayTimes::default();
        assert_eq!(replays.observe(u64::MAX, true), u64::MAX);
        assert_eq!(replays.observe(u64::MAX - 1, true), u64::MAX);
    }

    #[test]
    fn unmeasured_packed_sibling_does_not_zero_measured_round_duration() {
        let make = |name: &str, drafts, width, packed| GraphDescriptor {
            name: name.into(),
            kind: GraphKind::Speculative,
            sampling_mode: SamplingMode::General,
            confidence_width: drafts,
            draft_tokens: drafts,
            handle: 1,
            batch_capacity: 16,
            token_capacity: width,
            packed_token_capacity: packed,
            prefill_capacity: 0,
            estimated_duration_ns: 0,
            input_bindings: vec![],
            tuned_bindings: vec![],
        };
        let step_for = |drafts, width, packed| ScheduledStep {
            draft_tokens: drafts,
            kind: GraphKind::Speculative,
            batch_capacity: 16,
            token_capacity: width,
            packed_token_capacity: packed,
            mixed: None,
            lanes: vec![],
            requires_feedback: false,
        };
        let graphs = vec![
            make("b16w4p64", 3, 4, 64),
            make("b16w6p39", 5, 6, 39),
        ];
        let mut costs = ActionCosts::prepared(&graphs, 8).unwrap();
        assert_eq!(costs.packed_round_duration(16, 4), Some(0));
        // Measured first entry must not be zeroed by trailing unmeasured sibling.
        costs.set_measured(&step_for(3, 4, 64), 24_000_000);
        assert_eq!(costs.packed_round_duration(16, 4), Some(24_000_000));
        assert_eq!(costs.packed_round_duration(16, 6), Some(0));
        // Measuring a faster second sibling replaces the positive minimum.
        costs.set_measured(&step_for(5, 6, 39), 18_000_000);
        assert_eq!(costs.packed_round_duration(16, 4), Some(18_000_000));
        assert_eq!(costs.packed_round_duration(16, 6), Some(18_000_000));

        // Reverse order: unmeasured first entry must not block a measured sibling.
        let mut reversed = ActionCosts::prepared(&graphs, 8).unwrap();
        reversed.set_measured(&step_for(5, 6, 39), 18_000_000);
        assert_eq!(reversed.packed_round_duration(16, 4), Some(18_000_000));
        assert_eq!(reversed.packed_round_duration(16, 6), Some(18_000_000));
    }
}
