/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fixed-capacity, four-way row storage for immutable embedding tables.

use crate::Result;

const WAYS: usize = 4;
const PINNED: u64 = 1 << 63;
const CLOCK_MASK: u64 = !PINNED;

#[derive(Clone, Copy)]
struct Entry {
    id: i64,
    accessed: u64,
}

pub(super) struct RowCache {
    entries: Vec<Entry>,
    payload: Vec<u8>,
    row_bytes: usize,
    clock: u64,
    frequency: Frequency,
}

/// Bounded approximate usage counts, including requests that miss the row cache.
struct Frequency {
    counters: Vec<u8>,
    samples: usize,
}

impl Frequency {
    fn indices(&self, id: i64) -> [usize; 4] {
        let hash = twox_hash::XxHash3_64::oneshot(&id.to_le_bytes());
        let width = self.counters.len() / 4;
        std::array::from_fn(|i| {
            i * width + (hash.rotate_right(i as u32 * 16) as usize & (width - 1))
        })
    }

    fn estimate(&self, id: i64) -> u8 {
        self.indices(id)
            .map(|i| self.counters[i])
            .into_iter()
            .min()
            .unwrap()
    }

    fn observe(&mut self, id: i64) {
        self.samples += 1;
        if self.samples >= self.counters.len() * 10 {
            for value in &mut self.counters {
                *value /= 2;
            }
            self.samples = 0;
        }
        for index in self.indices(id) {
            self.counters[index] = self.counters[index].saturating_add(1);
        }
    }
}

pub(super) struct Insertion {
    pub added: bool,
    pub evicted: bool,
}

impl RowCache {
    pub fn new(budget: usize, rows: usize, row_bytes: usize) -> Result<Self> {
        let history_bytes = (budget / 64).clamp(16, 1 << 20).next_power_of_two();
        let payload_budget = budget
            .checked_sub(history_bytes)
            .ok_or("embedding row cache budget cannot hold usage counters")?;
        let set_bytes = row_bytes
            .checked_add(size_of::<Entry>())
            .and_then(|n| n.checked_mul(WAYS))
            .ok_or("embedding row cache size overflow")?;
        let sets = (payload_budget / set_bytes).min(rows.div_ceil(WAYS));
        if sets == 0 {
            return Err("embedding row cache budget cannot hold one four-row set".into());
        }
        let capacity = sets * WAYS;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(capacity)
            .map_err(|e| format!("embedding row cache index: {e}"))?;
        entries.resize(
            capacity,
            Entry {
                id: -1,
                accessed: 0,
            },
        );
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(capacity * row_bytes)
            .map_err(|e| format!("embedding row cache payload: {e}"))?;
        payload.resize(capacity * row_bytes, 0);
        // Commit every host page before the integrated GPU's KV budget reads
        // MemAvailable. A lazily zeroed allocation would leave that budget too large.
        for page in payload.chunks_mut(4096) {
            unsafe { page.as_mut_ptr().write_volatile(0) };
        }
        let mut counters = Vec::new();
        counters
            .try_reserve_exact(history_bytes)
            .map_err(|e| format!("embedding row cache usage counters: {e}"))?;
        counters.resize(history_bytes, 0);
        for page in counters.chunks_mut(4096) {
            unsafe { page.as_mut_ptr().write_volatile(0) };
        }
        Ok(Self {
            entries,
            payload,
            row_bytes,
            clock: 0,
            frequency: Frequency {
                counters,
                samples: 0,
            },
        })
    }

    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    pub fn bytes(&self) -> usize {
        self.entries.capacity() * size_of::<Entry>()
            + self.payload.capacity()
            + self.frequency.counters.capacity()
    }

    fn set(&self, id: i64) -> std::ops::Range<usize> {
        let hash = twox_hash::XxHash3_64::oneshot(&id.to_le_bytes());
        let first = (hash as usize % (self.entries.len() / WAYS)) * WAYS;
        first..first + WAYS
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1) & CLOCK_MASK;
        self.clock
    }

    fn is_pinned(&self, entry: Entry) -> bool {
        entry.id >= 0
            && (entry.accessed & PINNED) != 0
            && self.clock.saturating_sub(entry.accessed & CLOCK_MASK) < self.entries.len() as u64
    }

    pub fn get(&mut self, id: i64) -> Option<&[u8]> {
        self.frequency.observe(id);
        let slot = self.set(id).find(|&slot| self.entries[slot].id == id)?;
        let now = self.tick();
        // Consuming a row clears any advisory pin so normal frequency/recency rules manage it.
        self.entries[slot].accessed = now;
        Some(&self.payload[slot * self.row_bytes..(slot + 1) * self.row_bytes])
    }

    /// Protect an already-resident row against eviction until consumed or one cache turnover passes.
    pub fn pin_if_present(&mut self, id: i64) -> bool {
        let Some(slot) = self.set(id).find(|&slot| self.entries[slot].id == id) else {
            return false;
        };
        let now = self.tick();
        self.entries[slot].accessed = now | PINNED;
        true
    }

    pub fn insert(&mut self, id: i64, weights: &[u8], scales: &[u8]) -> Insertion {
        debug_assert!(id >= 0);
        debug_assert_eq!(weights.len() + scales.len(), self.row_bytes);
        let now = self.tick();
        let Some(slot) = self.admit(id, now) else {
            return Insertion {
                added: false,
                evicted: false,
            };
        };
        let previous = self.entries[slot].id;
        self.entries[slot] = Entry { id, accessed: now };
        self.fill(id, weights, scales);
        Insertion {
            added: previous == -1,
            evicted: previous >= 0 && previous != id,
        }
    }

    /// Admit an advisory-prefetched row without requiring prior frequency, pinning it until
    /// consumed or aged out. Declines admission if every way in the target set is already pinned.
    pub fn insert_advisory(&mut self, id: i64, weights: &[u8], scales: &[u8]) -> Insertion {
        debug_assert!(id >= 0);
        debug_assert_eq!(weights.len() + scales.len(), self.row_bytes);
        let slot = self.victim(id);
        let previous = self.entries[slot];
        if previous.id >= 0 && previous.id != id && self.is_pinned(previous) {
            return Insertion {
                added: false,
                evicted: false,
            };
        }
        let now = self.tick();
        self.entries[slot] = Entry {
            id,
            accessed: now | PINNED,
        };
        self.fill(id, weights, scales);
        Insertion {
            added: previous.id == -1,
            evicted: previous.id >= 0 && previous.id != id,
        }
    }

    fn victim(&self, id: i64) -> usize {
        let set = self.set(id);
        set.clone()
            .find(|&slot| self.entries[slot].id == id)
            .or_else(|| set.clone().find(|&slot| self.entries[slot].id == -1))
            .unwrap_or_else(|| {
                set.min_by_key(|&slot| {
                    let entry = self.entries[slot];
                    (
                        self.is_pinned(entry),
                        self.frequency.estimate(entry.id),
                        entry.accessed & CLOCK_MASK,
                    )
                })
                .unwrap()
            })
    }

    fn admit(&self, id: i64, accessed: u64) -> Option<usize> {
        let slot = self.victim(id);
        let previous = self.entries[slot];
        if previous.id < 0 || previous.id == id {
            return Some(slot);
        }
        (!self.is_pinned(previous)
            && (self.frequency.estimate(id), accessed)
                >= (
                    self.frequency.estimate(previous.id),
                    previous.accessed & CLOCK_MASK,
                ))
        .then_some(slot)
    }

    fn fill(&mut self, id: i64, weights: &[u8], scales: &[u8]) {
        let slot = self
            .set(id)
            .find(|&slot| self.entries[slot].id == id)
            .unwrap();
        let row = &mut self.payload[slot * self.row_bytes..(slot + 1) * self.row_bytes];
        row[..weights.len()].copy_from_slice(weights);
        row[weights.len()..].copy_from_slice(scales);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collisions_evict_the_least_recent_row_and_keep_both_planes() {
        let mut cache = RowCache::new(16 + 4 * (size_of::<Entry>() + 3), 100, 3).unwrap();
        for id in 0..4 {
            let result = cache.insert(id, &[id as u8, 17], &[42]);
            assert!(result.added && !result.evicted);
        }
        // Admission compares usage before recency. Give the incoming row the
        // same frequency as an untouched resident to exercise the recency tie.
        assert!(cache.get(4).is_none());
        assert_eq!(cache.get(0), Some(&[0, 17, 42][..]));
        let result = cache.insert(4, &[4, 18], &[43]);
        assert!(!result.added && result.evicted);
        assert!(cache.get(1).is_none());
        assert_eq!(cache.get(0), Some(&[0, 17, 42][..]));
        assert_eq!(cache.get(4), Some(&[4, 18, 43][..]));
        let result = cache.insert(4, &[9, 19], &[44]);
        assert!(!result.added && !result.evicted);
        assert_eq!(cache.get(4), Some(&[9, 19, 44][..]));
    }

    #[test]
    fn payload_and_index_stay_within_the_budget() {
        for budget in [1024, 4096, 65536] {
            let cache = RowCache::new(budget, 100000, 90).unwrap();
            assert!(cache.bytes() <= budget);
            assert_eq!(cache.capacity() % WAYS, 0);
            assert!(cache.bytes() >= cache.capacity() * 106);
        }
        assert!(RowCache::new(423, 100, 90).is_err());
    }

    #[test]
    fn frequently_used_rows_survive_scans_and_repeated_misses_gain_admission() {
        // Four resident rows compete for one set, with enough history counters
        // to distinguish a scan from repeated use despite hash approximation.
        let mut cache = RowCache::new(16384, 1000, 2048).unwrap();
        let payload = [0; 2048];
        assert_eq!(cache.capacity(), 4);
        for id in 0..4 {
            cache.get(id);
            cache.insert(id, &payload, &[]);
        }
        for _ in 0..20 {
            for id in 0..4 {
                assert!(cache.get(id).is_some());
            }
        }
        for id in 4..12 {
            assert!(cache.get(id).is_none());
            assert!(!cache.insert(id, &payload, &[]).evicted);
        }
        for id in 0..4 {
            assert!(cache.get(id).is_some());
        }
        for _ in 0..100 {
            if cache.get(12).is_some() {
                break;
            }
            cache.insert(12, &payload, &[]);
        }
        assert!(cache.get(12).is_some());
    }

    #[test]
    fn usage_counts_age_as_requests_shift() {
        let mut frequency = Frequency {
            counters: vec![0; 1024],
            samples: 0,
        };
        for _ in 0..100 {
            frequency.observe(7);
        }
        let before = frequency.estimate(7);
        for _ in 0..frequency.counters.len() * 20 {
            frequency.observe(99);
        }
        assert!(frequency.estimate(7) < before);
    }

    #[test]
    fn advisory_insert_admits_into_full_cache_pins_until_consumed_and_ages_out() {
        let mut cache = RowCache::new(16384, 1000, 2048).unwrap();
        let payload = [7u8; 2048];
        let warm_payload = [42u8; 2048];
        assert_eq!(cache.capacity(), 4);
        // Fill all four ways with hot entries so normal frequency admission rejects cold rows.
        for id in 0..4 {
            cache.get(id);
            cache.insert(id, &payload, &[]);
        }
        for _ in 0..10 {
            for id in 0..4 {
                assert!(cache.get(id).is_some());
            }
        }
        // Advisory insert admits row 100 despite zero prior frequency and protects it from
        // immediate non-advisory eviction.
        let ins = cache.insert_advisory(100, &warm_payload, &[]);
        assert!(ins.evicted);
        assert!(cache.pin_if_present(100));
        // Consuming row 100 returns its exact bytes and clears the pin.
        assert_eq!(cache.get(100), Some(&warm_payload[..]));
        // An unconsumed advisory pin ages out once one full cache turnover of ticks passes.
        // Row 101 evicts row 100 (whose pin cleared on get above), while hot row 1 stays resident.
        let ins101 = cache.insert_advisory(101, &warm_payload, &[]);
        assert!(ins101.evicted);
        assert!(cache.get(100).is_none());
        for _ in 0..cache.capacity() {
            assert!(cache.get(1).is_some());
        }
        let slot = cache.set(101).find(|&s| cache.entries[s].id == 101).unwrap();
        assert!(!cache.is_pinned(cache.entries[slot]));
    }

    #[test]
    fn advisory_insert_does_not_evict_unconsumed_pinned_rows_when_set_is_full() {
        // Use a multi-set cache (capacity > WAYS) so pins do not age out after WAYS insertions.
        let mut cache = RowCache::new(65536, 1000, 2048).unwrap();
        assert!(cache.capacity() >= WAYS * 4);
        let same_set: Vec<i64> = (0i64..1000)
            .filter(|&id| cache.set(id).start == 0)
            .take(WAYS + 2)
            .collect();
        assert_eq!(same_set.len(), WAYS + 2);
        let payloads: [[u8; 2048]; 6] = std::array::from_fn(|i| [(i + 1) as u8; 2048]);
        for i in 0..WAYS {
            let ins = cache.insert_advisory(same_set[i], &payloads[i], &[]);
            assert!(ins.added && !ins.evicted);
        }
        // All four ways in set 0 are pinned: a fifth advisory insert into set 0 must not evict them.
        let overflow = cache.insert_advisory(same_set[WAYS], &payloads[WAYS], &[]);
        assert!(!overflow.added && !overflow.evicted);
        // A regular insert into the same set is also blocked while every way holds an unconsumed pin.
        let blocked = cache.insert(same_set[WAYS + 1], &payloads[WAYS + 1], &[]);
        assert!(!blocked.added && !blocked.evicted);
        // Consuming row 0 returns its exact bytes and clears its pin so normal admission resumes.
        assert_eq!(cache.get(same_set[0]), Some(&payloads[0][..]));
        for i in 1..WAYS {
            let id = same_set[i];
            let slot = cache.set(id).find(|&s| cache.entries[s].id == id).unwrap();
            assert!(cache.is_pinned(cache.entries[slot]));
        }
        cache.get(same_set[WAYS + 1]);
        let admitted = cache.insert(same_set[WAYS + 1], &payloads[WAYS + 1], &[]);
        assert!(admitted.evicted);
        assert_eq!(cache.get(same_set[WAYS + 1]), Some(&payloads[WAYS + 1][..]));
    }
}
