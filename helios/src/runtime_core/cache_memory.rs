/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fixed virtual aliases let immutable continuation rows borrow KV storage.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    ops::Range,
};

use super::{
    NativeResources, Storage,
    driver::{AllocationProperties, Location},
    format::{Core, Memory, invalid},
};
use crate::Result;
use flare_selene::CacheGeometry;

#[derive(Clone, Debug)]
struct Chunk {
    plane: usize,
    offset: u64,
    bytes: u64,
    row_offset: u64,
    row_bytes: u64,
}

impl Chunk {
    fn record_conflicts(
        &self,
        geometry: &CacheGeometry,
        execution: u64,
        aliases: &mut [Range<u32>],
    ) {
        if self.row_bytes == 0 {
            return;
        }
        let row = geometry.row.unit_bytes;
        let unit = geometry.planes[self.plane].unit_bytes;
        let plane_bytes = geometry.page_capacity * unit;
        let end = self.row_offset + self.row_bytes;
        for id in
            (self.row_offset / row).max(execution)..end.div_ceil(row).min(geometry.row_capacity)
        {
            let first = self.offset + (id * row).saturating_sub(self.row_offset);
            let last = (self.offset + ((id + 1) * row).min(end) - self.row_offset).min(plane_bytes);
            if first < last {
                let range = &mut aliases[(id - execution) as usize];
                range.start = range.start.min((first / unit) as u32);
                range.end = range.end.max(last.div_ceil(unit) as u32);
            }
        }
    }
}

#[derive(Debug)]
pub(super) struct Layout {
    alignment: u64,
    fixed_bytes: u64,
    chunks: Vec<Chunk>,
    pub aliases: Vec<Range<u32>>,
}

pub(super) fn alias_plane(unit: u64, largest: u64, row: u64, page: u64, alignment: u64) -> bool {
    // A tiny plane can exclude hundreds of pages for one mapping granule. Borrow
    // from planes whose granules span at most a row's page cost, including the densest.
    unit == largest || alignment.div_ceil(unit) <= row.div_ceil(page)
}

impl Layout {
    /// Backing handles and per-device address mappings created by this layout.
    fn handle_charge(&self, row_readers: u64, plane_readers: &[u64]) -> Result<u64> {
        self.chunks
            .iter()
            .try_fold(1 + row_readers, |total, chunk| {
                total
                    .checked_add(1)
                    .and_then(|n| n.checked_add(plane_readers[chunk.plane]))
                    .and_then(|n| n.checked_add(if chunk.row_bytes == 0 { 0 } else { row_readers }))
                    .ok_or_else(|| invalid("mapping handle count overflow"))
            })
    }

    pub fn new(
        geometry: &CacheGeometry,
        execution: u64,
        alignments: &[u64],
        row_alignment: u64,
    ) -> Result<Self> {
        let mut layout = Self::unmerged(geometry, execution, alignments, row_alignment)?;
        layout.pack_rows(geometry)?;
        Ok(layout)
    }

    fn unmerged(
        geometry: &CacheGeometry,
        execution: u64,
        alignments: &[u64],
        row_alignment: u64,
    ) -> Result<Self> {
        if alignments.len() != geometry.planes.len()
            || alignments
                .iter()
                .any(|&alignment| alignment != row_alignment)
        {
            return Err(invalid(
                "cache aliases require matching CUDA allocation granularity",
            ));
        }
        // Interleave short page runs so a snapshot borrows approximately its
        // own byte cost in logical pages. Requests allocate low page IDs first.
        let group = if geometry.row_capacity == execution {
            geometry.page_capacity
        } else {
            (geometry.row.unit_bytes / geometry.page_bytes() / 8).clamp(1, 32)
        };
        Self::grouped(geometry, execution, row_alignment, group)
    }

    fn grouped(
        geometry: &CacheGeometry,
        execution: u64,
        row_alignment: u64,
        group: u64,
    ) -> Result<Self> {
        let row_bytes = geometry.row.unit_bytes;
        let fixed_bytes = (execution * row_bytes).div_ceil(row_alignment) * row_alignment;
        let row_end = (geometry.row_capacity * row_bytes).div_ceil(row_alignment) * row_alignment;
        let snapshots = geometry.row_capacity - execution;
        let mut aliases = vec![u32::MAX..0; snapshots as usize];
        let mut cursor = fixed_bytes;
        let mut chunks = Vec::new();
        let largest = geometry.planes.iter().map(|p| p.unit_bytes).max().unwrap();
        let eligible: Vec<_> = geometry
            .planes
            .iter()
            .map(|p| {
                alias_plane(
                    p.unit_bytes,
                    largest,
                    row_bytes,
                    geometry.page_bytes(),
                    row_alignment,
                )
            })
            .collect();
        for (plane, p) in geometry.planes.iter().enumerate() {
            if !eligible[plane] {
                chunks.push(Chunk {
                    plane,
                    offset: 0,
                    bytes: (geometry.page_capacity * p.unit_bytes).div_ceil(row_alignment)
                        * row_alignment,
                    row_offset: cursor,
                    row_bytes: 0,
                });
            }
        }
        for group_id in (0..geometry.page_capacity.div_ceil(group)).rev() {
            let first = group_id * group;
            let last = (first + group).min(geometry.page_capacity);
            for (plane, p) in geometry.planes.iter().enumerate() {
                if !eligible[plane] {
                    continue;
                }
                let offset = first * p.unit_bytes / row_alignment * row_alignment;
                let end = if last == geometry.page_capacity {
                    (last * p.unit_bytes).div_ceil(row_alignment) * row_alignment
                } else {
                    last * p.unit_bytes / row_alignment * row_alignment
                };
                if end == offset {
                    continue;
                }
                let bytes = end - offset;
                let mapped = bytes.min(row_end.saturating_sub(cursor));
                let chunk = Chunk {
                    plane,
                    offset,
                    bytes: if mapped == 0 { bytes } else { mapped },
                    row_offset: cursor,
                    row_bytes: mapped,
                };
                chunk.record_conflicts(geometry, execution, &mut aliases);
                chunks.push(chunk);
                cursor += mapped;
                if mapped > 0 && mapped < bytes {
                    // NVIDIA Linux driver 615.71.10 rejects partial-handle cuMemMap aliases on
                    // Blackwell VMM allocations when a trailing unaliased tail shares the handle.
                    // Keep the unaliased plane tail as an independent physical handle.
                    chunks.push(Chunk {
                        plane,
                        offset: offset + mapped,
                        bytes: bytes - mapped,
                        row_offset: cursor,
                        row_bytes: 0,
                    });
                }
            }
            if cursor == row_end {
                for (plane, p) in geometry.planes.iter().enumerate() {
                    if !eligible[plane] {
                        continue;
                    }
                    let bytes = first * p.unit_bytes / row_alignment * row_alignment;
                    if bytes != 0 {
                        chunks.push(Chunk {
                            plane,
                            offset: 0,
                            bytes,
                            row_offset: cursor,
                            row_bytes: 0,
                        });
                    }
                }
                break;
            }
        }
        if cursor < row_end {
            return Err(invalid("KV backing cannot cover continuation aliases"));
        }
        for range in &mut aliases {
            if *range == (u32::MAX..0) {
                *range = 0..0;
            } else if range.start >= range.end {
                return Err(invalid("continuation alias lacks a KV conflict range"));
            }
        }
        Ok(Self {
            alignment: row_alignment,
            fixed_bytes,
            chunks,
            aliases,
        })
    }

    /// Permute equal page spans, then merge adjacent backing inside one row.
    /// Neither operation changes any row's conflicting KV pages. Chunks crossing
    /// a row boundary retain their row span and are never moved or merged.
    fn pack_rows(&mut self, geometry: &CacheGeometry) -> Result<()> {
        let row_bytes = geometry.row.unit_bytes;
        let contained_row = |chunk: &Chunk| {
            if chunk.row_bytes == 0 || chunk.row_bytes != chunk.bytes {
                return None;
            }
            let row = chunk.row_offset / row_bytes;
            let end = chunk.row_offset.checked_add(chunk.bytes)?;
            ((end - 1) / row_bytes == row).then_some(row)
        };
        let mut spans: BTreeMap<(u64, u64, u64), Vec<usize>> = BTreeMap::new();
        for (index, chunk) in self.chunks.iter().enumerate() {
            spans
                .entry((
                    geometry.planes[chunk.plane].unit_bytes,
                    chunk.offset,
                    chunk.bytes,
                ))
                .or_default()
                .push(index);
        }
        let mut spans: Vec<_> = spans.into_values().collect();
        spans.retain(|indices| {
            indices
                .iter()
                .any(|&i| contained_row(&self.chunks[i]).is_some())
        });
        spans.sort_by_key(|indices| {
            indices
                .iter()
                .filter_map(|&i| {
                    (self.chunks[i].row_bytes != 0).then_some(self.chunks[i].row_offset)
                })
                .min()
                .unwrap()
        });
        let mut previous: HashMap<(u64, usize), u64> = HashMap::new();
        for indices in spans {
            let mut available: BTreeSet<_> =
                indices.iter().map(|&i| self.chunks[i].plane).collect();
            let mut assignments = vec![None; indices.len()];
            // Favor the plane already present at the adjacent page span in this
            // row. Every candidate has identical byte and logical-page geometry.
            for (assignment, &index) in assignments.iter_mut().zip(&indices) {
                let chunk = &self.chunks[index];
                let Some(row) = contained_row(chunk) else {
                    continue;
                };
                if let Some(plane) = available.iter().copied().find(|&plane| {
                    previous.get(&(row, plane)).copied() == chunk.offset.checked_add(chunk.bytes)
                }) {
                    *assignment = Some(plane);
                    available.remove(&plane);
                }
            }
            let mut remaining = available.into_iter();
            for (&index, assignment) in indices.iter().zip(assignments) {
                let chunk = &mut self.chunks[index];
                chunk.plane = assignment.unwrap_or_else(|| remaining.next().unwrap());
                if let Some(row) = contained_row(chunk) {
                    previous.insert((row, chunk.plane), chunk.offset);
                }
            }
        }

        let mut rows: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (index, chunk) in self.chunks.iter().enumerate() {
            if let Some(row) = contained_row(chunk) {
                rows.entry(row).or_default().push(index);
            }
        }
        let mut retained = vec![true; self.chunks.len()];
        let mut replacements: HashMap<usize, Vec<Chunk>> = HashMap::new();
        for mut indices in rows.into_values() {
            indices.sort_by_key(|&i| self.chunks[i].row_offset);
            // Only move chunks within an uninterrupted interval of this row.
            let mut start = 0;
            while start < indices.len() {
                let mut end = start + 1;
                while end < indices.len() {
                    let previous = &self.chunks[indices[end - 1]];
                    if previous.row_offset.checked_add(previous.row_bytes)
                        != Some(self.chunks[indices[end]].row_offset)
                    {
                        break;
                    }
                    end += 1;
                }
                let interval = &indices[start..end];
                let mut chunks: Vec<_> = interval.iter().map(|&i| self.chunks[i].clone()).collect();
                chunks.sort_by_key(|c| (c.plane, c.offset));
                let mut merged: Vec<Chunk> = Vec::new();
                for chunk in chunks {
                    if let Some(last) = merged.last_mut().filter(|last| {
                        last.plane == chunk.plane
                            && last.offset.checked_add(last.bytes) == Some(chunk.offset)
                    }) {
                        last.bytes = last.bytes.checked_add(chunk.bytes).ok_or_else(|| {
                            invalid("merged cache backing overflows address space")
                        })?;
                        last.row_bytes = last.bytes;
                    } else {
                        merged.push(chunk);
                    }
                }
                let mut cursor = self.chunks[interval[0]].row_offset;
                for chunk in &mut merged {
                    chunk.row_offset = cursor;
                    cursor = cursor.checked_add(chunk.row_bytes).ok_or_else(|| {
                        invalid("merged continuation mapping overflows address space")
                    })?;
                }
                for &index in interval {
                    retained[index] = false;
                }
                replacements.insert(interval[0], merged);
                start = end;
            }
        }
        let chunks = std::mem::take(&mut self.chunks);
        for (index, chunk) in chunks.into_iter().enumerate() {
            if let Some(merged) = replacements.remove(&index) {
                self.chunks.extend(merged);
            } else if retained[index] {
                self.chunks.push(chunk);
            }
        }
        Ok(())
    }

    /// Use only pages already eligible for a live snapshot prefix. Group the
    /// longest contiguous run available in one plane into one physical handle.
    fn pack_prefixes(&self, geometry: &CacheGeometry, execution: u64) -> Option<Self> {
        let row_bytes = geometry.row.unit_bytes;
        let mut cursor = self.fixed_bytes;
        let mut row_end = cursor;
        let mut remaining = vec![0; geometry.planes.len()];
        let mut borrowed = vec![false; remaining.len()];
        for chunk in &self.chunks {
            remaining[chunk.plane] = remaining[chunk.plane].max(chunk.offset + chunk.bytes);
            borrowed[chunk.plane] |= chunk.row_bytes > 0;
            row_end += chunk.row_bytes;
        }
        let mut chunks = Vec::new();
        let mut aliases = vec![u32::MAX..0; self.aliases.len()];
        while cursor < row_end {
            let row = cursor / row_bytes - execution;
            let floor = u64::from(self.aliases[row as usize].start);
            let (length, _, std::cmp::Reverse(plane)) = geometry
                .planes
                .iter()
                .enumerate()
                .filter(|(plane, _)| borrowed[*plane])
                .filter_map(|(plane, p)| {
                    let low = (floor * p.unit_bytes).div_ceil(self.alignment) * self.alignment;
                    let length = remaining[plane].saturating_sub(low).min(row_end - cursor);
                    (length > 0).then_some((
                        length,
                        remaining[plane].div_ceil(p.unit_bytes),
                        std::cmp::Reverse(plane),
                    ))
                })
                .max()?;
            remaining[plane] -= length;
            let offset = remaining[plane];
            let end = cursor + length;
            let chunk = Chunk {
                plane,
                offset,
                bytes: length,
                row_offset: cursor,
                row_bytes: length,
            };
            chunk.record_conflicts(geometry, execution, &mut aliases);
            chunks.push(chunk);
            cursor = end;
        }
        // A compacted row can extend toward higher page IDs, where an earlier
        // snapshot already excludes those pages. Every old row still works
        // with the same occupied low KV prefix, including after other rows are freed.
        let mut covered = u32::MAX..0;
        for (planned, compact) in self.aliases.iter().zip(&mut aliases) {
            if planned.is_empty() {
                if !compact.is_empty() {
                    return None;
                }
                *compact = 0..0;
                continue;
            }
            if !covered.is_empty() && (planned.end < covered.start || planned.start > covered.end) {
                return None;
            }
            covered.start = covered.start.min(planned.start);
            covered.end = covered.end.max(planned.end);
            if compact.is_empty() || compact.start < planned.start || compact.end > covered.end {
                return None;
            }
        }
        for (plane, bytes) in remaining.into_iter().enumerate() {
            if bytes > 0 {
                chunks.push(Chunk {
                    plane,
                    offset: 0,
                    bytes,
                    row_offset: row_end,
                    row_bytes: 0,
                });
            }
        }
        Some(Self {
            alignment: self.alignment,
            fixed_bytes: self.fixed_bytes,
            chunks,
            aliases,
        })
    }
}

// Linux UVM uses one NVIDIA Resource Manager (RM) client for external GPU
// mappings. Its automatic handle range (RS_UNIQUE_HANDLE_RANGE) contains
// 524,288 entries shared by VMM handles and other driver objects.
pub(super) const RM_HANDLE_RANGE: u64 = 1 << 19;

pub(super) fn local_handle_limit(os: &str, clustered: bool) -> Option<u64> {
    (os == "linux" && !clustered).then_some(RM_HANDLE_RANGE)
}

pub(super) fn compact_cache_layouts(
    ranks: &[NativeResources],
    geometry: &[CacheGeometry],
    layouts: &mut [Layout],
    execution: u64,
    limit: Option<u64>,
) -> Result<(u64, u64)> {
    let rank_readers: Vec<_> = ranks
        .iter()
        .map(|rank| {
            let fixed: Vec<_> = rank
                .allocations
                .values()
                .map(|a| (a.address, a.mappings.len() as u64))
                .collect();
            (&rank.peer_readers, fixed)
        })
        .collect();
    compact_cache_layouts_with_readers(&rank_readers, geometry, layouts, execution, limit)
}

fn compact_cache_layouts_with_readers(
    ranks: &[(&HashMap<u64, Vec<(usize, i32)>>, Vec<(u64, u64)>)],
    geometry: &[CacheGeometry],
    layouts: &mut [Layout],
    execution: u64,
    limit: Option<u64>,
) -> Result<(u64, u64)> {
    // Multi-host agreement carries page and row capacities, not per-host VMM
    // charges. Its scheduler contract requires the same snapshot conflicts.
    let Some(limit) = limit else {
        return Ok((0, 0));
    };
    let mut before = 0u64;
    let mut counts = Vec::with_capacity(ranks.len());
    for (((peer_readers, fixed_allocs), g), layout) in
        ranks.iter().zip(geometry).zip(layouts.iter())
    {
        let readers = |address| peer_readers.get(&address).map_or(1, |r| r.len() as u64);
        let row = readers(g.row.address);
        let planes: Vec<_> = g.planes.iter().map(|p| readers(p.address)).collect();
        let fixed: u64 = fixed_allocs
            .iter()
            .map(|&(address, mappings)| mappings * (1 + readers(address)))
            .sum();
        let cache = layout.handle_charge(row, &planes)?;
        before = before
            .checked_add(fixed)
            .and_then(|n| n.checked_add(cache))
            .ok_or_else(|| invalid("mapping handle count overflow"))?;
        counts.push((row, planes, cache));
    }
    let after = plan_mapping_handles(geometry, layouts, execution, before, counts, limit)?;
    Ok((before, after))
}

fn plan_mapping_handles(
    geometry: &[CacheGeometry],
    layouts: &mut [Layout],
    execution: u64,
    mut total: u64,
    mut counts: Vec<(u64, Vec<u64>, u64)>,
    limit: u64,
) -> Result<u64> {
    // Preserve individual alias ranges when the normal plan fits, then
    // preserve all occupied low-page prefixes where longer runs suffice.
    // Only explore larger page groups if that plan still exhausts the range.
    // Stop modifying further ranks as soon as the aggregate drops below limit.
    let mut granules = 0u64;
    while total >= limit {
        let mut whole_planes = granules > 0;
        for ((layout, g), (row, planes, charge)) in
            layouts.iter_mut().zip(geometry).zip(&mut counts)
        {
            if total < limit {
                break;
            }
            let candidate = if granules == 0 {
                layout.pack_prefixes(g, execution)
            } else {
                let densest = g.planes.iter().map(|p| p.unit_bytes).max().unwrap();
                let group = granules
                    .saturating_mul(layout.alignment)
                    .div_ceil(densest)
                    .min(g.page_capacity);
                whole_planes &= group == g.page_capacity;
                let mut candidate = Layout::grouped(g, execution, layout.alignment, group)?;
                candidate.pack_rows(g)?;
                Some(candidate)
            };
            if let Some(candidate) = candidate {
                let cost = candidate.handle_charge(*row, planes)?;
                if cost < *charge {
                    total -= *charge - cost;
                    *charge = cost;
                    *layout = candidate;
                }
            }
        }
        if whole_planes {
            break;
        }
        granules = if granules == 0 {
            2
        } else {
            granules.saturating_mul(2)
        };
    }
    if total >= limit {
        let shortfall = total - limit + 1;
        return Err(invalid(format!(
            "estimated VMM mapping handles ({total}) exceed local handle limit ({limit}) by {shortfall} after compaction"
        )));
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(row_bytes: u64, units: &[u64], pages: u64, rows: u64) -> CacheGeometry {
        CacheGeometry {
            page_tokens: 64,
            page_capacity: pages,
            row_capacity: rows,
            row: flare_selene::Plane {
                name: "rows".into(),
                address: 0,
                unit_bytes: row_bytes,
            },
            planes: units
                .iter()
                .enumerate()
                .map(|(i, &unit)| flare_selene::Plane {
                    name: format!("plane-{i}"),
                    address: 0,
                    unit_bytes: unit,
                })
                .collect(),
            placement: Default::default(),
            codec_fields: Vec::new(),
            snapshot_pages: Vec::new(),
        }
    }

    fn conflicts(layout: &Layout, geometry: &CacheGeometry, row: u64) -> BTreeSet<u64> {
        let start = row * geometry.row.unit_bytes;
        let end = start + geometry.row.unit_bytes;
        let mut pages = BTreeSet::new();
        for chunk in &layout.chunks {
            let low = start.max(chunk.row_offset);
            let high = end.min(chunk.row_offset + chunk.row_bytes);
            if low < high {
                let unit = geometry.planes[chunk.plane].unit_bytes;
                pages.extend(
                    (chunk.offset + low - chunk.row_offset) / unit
                        ..(chunk.offset + high - chunk.row_offset)
                            .div_ceil(unit)
                            .min(geometry.page_capacity),
                );
            }
        }
        pages
    }

    #[test]
    fn row_packing_preserves_every_byte_and_exact_conflicts() {
        let alignment = 2 << 20;
        for (row, units, pages, rows, execution) in [
            (7 * alignment / 2, vec![alignment / 8; 8], 128, 7, 3),
            (
                35_450_112,
                [vec![73728; 24], vec![16896; 24]].concat(),
                2234,
                117,
                5,
            ),
            (
                35_450_112,
                [vec![73728; 24], vec![16896; 24]].concat(),
                3072,
                158,
                5,
            ),
            (81_464_576, vec![135168, 8448].repeat(12), 3844, 41, 9),
            (
                122_932_224,
                vec![65536, 65536, 4096].repeat(13),
                9056,
                41,
                9,
            ),
        ] {
            let geometry = geometry(row, &units, pages, rows);
            let alignments = vec![alignment; units.len()];
            let before = Layout::unmerged(&geometry, execution, &alignments, alignment).unwrap();
            let packed = Layout::new(&geometry, execution, &alignments, alignment).unwrap();
            assert_eq!(before.fixed_bytes, packed.fixed_bytes);
            assert_eq!(before.aliases, packed.aliases);
            assert!(packed.chunks.len() <= before.chunks.len());
            for row in 0..rows {
                assert_eq!(
                    conflicts(&before, &geometry, row),
                    conflicts(&packed, &geometry, row),
                    "row {row}"
                );
            }
            for (plane, &unit) in units.iter().enumerate() {
                let mut pieces: Vec<_> =
                    packed.chunks.iter().filter(|c| c.plane == plane).collect();
                pieces.sort_by_key(|c| c.offset);
                let mut end = 0;
                for piece in pieces {
                    assert_eq!(piece.offset, end);
                    assert_eq!(piece.bytes % alignment, 0);
                    assert_eq!(piece.row_offset % alignment, 0);
                    end += piece.bytes;
                }
                assert_eq!(end, (pages * unit).div_ceil(alignment) * alignment);
            }
            let mut aliases: Vec<_> = packed.chunks.iter().filter(|c| c.row_bytes != 0).collect();
            aliases.sort_by_key(|c| c.row_offset);
            let mut end = packed.fixed_bytes;
            for alias in aliases {
                assert_eq!(alias.row_offset, end);
                assert!(alias.row_bytes <= alias.bytes);
                end += alias.row_bytes;
            }
            assert_eq!(
                end.max(packed.fixed_bytes),
                (rows * row).div_ceil(alignment) * alignment
            );
            let readers: Vec<_> = (0..units.len()).map(|i| [1, 8, 16][i % 3]).collect();
            assert!(
                packed.handle_charge(1, &readers).unwrap()
                    <= before.handle_charge(1, &readers).unwrap()
            );
        }
    }

    #[test]
    fn packing_preserves_partial_row_boundaries_across_small_layouts() {
        let alignment = 65536;
        for row_units in [1, 2, 3, 5, 7, 9, 17] {
            for execution in [1, 3] {
                for snapshots in 1..=12 {
                    let geometry = geometry(
                        row_units * alignment / 2,
                        &[8192; 8],
                        128,
                        execution + snapshots,
                    );
                    let before =
                        Layout::unmerged(&geometry, execution, &[alignment; 8], alignment).unwrap();
                    let packed =
                        Layout::new(&geometry, execution, &[alignment; 8], alignment).unwrap();
                    assert_eq!(before.aliases, packed.aliases);
                    assert!(packed.chunks.len() <= before.chunks.len());
                    for row in 0..geometry.row_capacity {
                        assert_eq!(
                            conflicts(&before, &geometry, row),
                            conflicts(&packed, &geometry, row)
                        );
                    }
                    for plane in 0..8 {
                        let mut pieces: Vec<_> =
                            packed.chunks.iter().filter(|c| c.plane == plane).collect();
                        pieces.sort_by_key(|c| c.offset);
                        let mut end = 0;
                        for piece in pieces {
                            assert_eq!(piece.offset, end);
                            end += piece.bytes;
                        }
                        assert_eq!(end, 128 * 8192);
                    }
                }
            }
        }
    }

    #[test]
    fn tp16_cache_fits_mapping_resources_without_losing_kv_or_snapshots() {
        let units = [vec![73728; 24], vec![16896; 24]].concat();
        let alignment = 2 << 20;
        for (readers, pages, rows) in [(16, 2234, 117), (8, 3072, 158)] {
            let geometry = geometry(35_450_112, &units, pages, rows);
            let access = [vec![readers; 24], vec![1; 24]].concat();
            let before = Layout::unmerged(&geometry, 5, &[alignment; 48], alignment).unwrap();
            let packed = Layout::new(&geometry, 5, &[alignment; 48], alignment).unwrap();
            // Fixed allocations and their imports contribute 4,694 handles in
            // this TP16 fixture. UVM's shared RM client has 524,288 handle slots.
            let before_handles = 4694 + 16 * before.handle_charge(1, &access).unwrap();
            let packed_handles = 4694 + 16 * packed.handle_charge(1, &access).unwrap();
            assert!(
                packed_handles < 524288,
                "{readers} readers: {packed_handles}"
            );
            if readers == 16 {
                assert!(before_handles > 524288);
            }
            assert!(packed.chunks.len() < before.chunks.len());
            assert_eq!(packed.aliases.len() as u64, rows - 5);
            assert_eq!(
                packed.chunks.iter().map(|c| c.bytes).sum::<u64>(),
                before.chunks.iter().map(|c| c.bytes).sum::<u64>()
            );
        }
    }

    #[test]
    fn full_size_dcp16_cache_preserves_each_admissible_kv_prefix() {
        let units = [vec![73_728; 24], vec![16_896; 24]].concat();
        let readers = [vec![16; 24], vec![1; 24]].concat();
        let a = 2 << 20;
        for (pages, snapshots, fixed, expected) in
            [(2304, 116, 20320, 494576), (3103, 157, 3528, 647544)]
        {
            let g = geometry(35_122_432, &units, pages, snapshots + 2);
            let old = Layout::new(&g, 2, &[a; 48], a).unwrap();
            assert_eq!(
                fixed + 16 * old.handle_charge(1, &readers).unwrap(),
                expected
            );
            if expected < RM_HANDLE_RANGE {
                continue;
            }
            let candidate = old.pack_prefixes(&g, 2).unwrap();
            assert_eq!(
                fixed + 16 * candidate.handle_charge(1, &readers).unwrap(),
                518232
            );
            assert_eq!(candidate.aliases.len(), snapshots as usize);
            assert_eq!(candidate.fixed_bytes, old.fixed_bytes);
            let mut old_prefix = BTreeSet::new();
            let mut new_prefix = BTreeSet::new();
            for (index, (before, after)) in old.aliases.iter().zip(&candidate.aliases).enumerate() {
                let touched = conflicts(&candidate, &g, index as u64 + 2);
                assert_eq!(touched.first().copied(), Some(after.start as u64));
                assert_eq!(touched.last().copied(), Some((after.end - 1) as u64));
                assert!(after.start >= before.start);
                old_prefix.extend(before.clone());
                new_prefix.extend(after.clone());
                assert!(new_prefix.is_subset(&old_prefix));
            }
            for (plane, unit) in units.iter().enumerate() {
                let mut pieces: Vec<_> = candidate
                    .chunks
                    .iter()
                    .filter(|c| c.plane == plane)
                    .collect();
                pieces.sort_by_key(|c| c.offset);
                let mut offset = 0;
                for chunk in pieces {
                    assert_eq!(chunk.offset, offset);
                    assert_eq!(chunk.bytes % a, 0);
                    assert!(chunk.row_bytes == 0 || chunk.row_bytes == chunk.bytes);
                    offset += chunk.bytes;
                }
                assert_eq!(offset, (pages * unit).div_ceil(a) * a);
            }
            let mut cursor = candidate.fixed_bytes;
            for chunk in candidate.chunks.iter().filter(|c| c.row_bytes != 0) {
                assert_eq!(chunk.row_offset, cursor);
                cursor += chunk.bytes;
            }
            assert_eq!(cursor, (g.row_capacity * g.row.unit_bytes).div_ceil(a) * a);
        }
    }

    #[test]
    fn a_two_granule_plane_can_lend_one_complete_granule() {
        let a = 2 << 20;
        let g = geometry(a, &[2 * a], 1, 2);
        for layout in [
            Layout::unmerged(&g, 1, &[a], a).unwrap(),
            Layout::new(&g, 1, &[a], a).unwrap(),
        ] {
            assert_eq!(layout.chunks.len(), 2);
            let mut pieces: Vec<_> = layout
                .chunks
                .iter()
                .map(|c| (c.offset, c.bytes, c.row_bytes))
                .collect();
            pieces.sort_unstable();
            assert_eq!(pieces, [(0, a, a), (a, a, 0)]);
            assert_eq!(conflicts(&layout, &g, 1), BTreeSet::from([0]));
        }
    }

    #[test]
    fn larger_peer_dense_cache_searches_page_groups_before_allocating() {
        let a = 2 << 20;
        let units = [vec![73_728; 24], vec![16_896; 24]].concat();
        let readers = [vec![16; 24], vec![1; 24]].concat();
        let geometries: Vec<_> = (0..16)
            .map(|_| geometry(35_122_432, &units, 4096, 208))
            .collect();
        let mut layouts: Vec<_> = geometries
            .iter()
            .map(|g| Layout::new(g, 2, &[a; 48], a).unwrap())
            .collect();
        let prior = layouts[0].handle_charge(1, &readers).unwrap();
        let prefix = layouts[0].pack_prefixes(&geometries[0], 2).unwrap();
        let prefix_handles = 3528 + 16 * prefix.handle_charge(1, &readers).unwrap();
        assert!(prefix_handles >= RM_HANDLE_RANGE);
        let original = 3528 + 16 * prior;
        let counts = vec![(1, readers.clone(), prior); 16];
        let after = plan_mapping_handles(
            &geometries,
            &mut layouts,
            2,
            original,
            counts,
            RM_HANDLE_RANGE,
        )
        .unwrap();
        assert!(after < RM_HANDLE_RANGE);
        let recomputed: u64 = 3528
            + layouts
                .iter()
                .map(|l| l.handle_charge(1, &readers).unwrap())
                .sum::<u64>();
        assert_eq!(after, recomputed);
        for (g, layout) in geometries.iter().zip(layouts) {
            assert_eq!(layout.aliases.len(), 206);
            assert!(
                layout
                    .chunks
                    .iter()
                    .all(|c| c.row_bytes == 0 || c.row_bytes == c.bytes)
            );
            for (row, range) in layout.aliases.iter().enumerate() {
                let touched = conflicts(&layout, g, row as u64 + 2);
                assert_eq!(touched.first().copied(), Some(range.start as u64));
                assert_eq!(touched.last().copied(), Some((range.end - 1) as u64));
            }
            let total: u64 = units.iter().map(|u| (4096 * u).div_ceil(a) * a).sum();
            assert_eq!(layout.chunks.iter().map(|c| c.bytes).sum::<u64>(), total);
        }
    }

    #[test]
    fn plan_mapping_handles_selects_prefix_packing_and_stops_when_under_limit() {
        let a = 2 << 20;
        let units = [vec![73_728; 24], vec![16_896; 24]].concat();
        let readers = [vec![16; 24], vec![1; 24]].concat();
        let geometries: Vec<_> = (0..16)
            .map(|_| geometry(35_122_432, &units, 3103, 159))
            .collect();
        let mut layouts: Vec<_> = geometries
            .iter()
            .map(|g| Layout::new(g, 2, &[a; 48], a).unwrap())
            .collect();
        let canonical_charge = layouts[0].handle_charge(1, &readers).unwrap();
        let original = 3528 + 16 * canonical_charge;
        assert!(original >= RM_HANDLE_RANGE);
        let counts = vec![(1, readers.clone(), canonical_charge); 16];
        let after = plan_mapping_handles(
            &geometries,
            &mut layouts,
            2,
            original,
            counts,
            RM_HANDLE_RANGE,
        )
        .unwrap();
        assert!(after < RM_HANDLE_RANGE);
        let recomputed: u64 = 3528
            + layouts
                .iter()
                .map(|l| l.handle_charge(1, &readers).unwrap())
                .sum::<u64>();
        assert_eq!(after, recomputed);
        // Every compacted rank uses prefix packing (strictly fewer handles than canonical).
        assert!(layouts[0].handle_charge(1, &readers).unwrap() < canonical_charge);
    }

    #[test]
    fn plan_mapping_handles_rejects_over_limit_layout_with_shortfall() {
        let a = 2 << 20;
        let units = [vec![73_728; 24], vec![16_896; 24]].concat();
        let readers = [vec![16; 24], vec![1; 24]].concat();
        let g = geometry(35_122_432, &units, 3103, 159);
        let mut layouts = vec![Layout::new(&g, 2, &[a; 48], a).unwrap()];
        let charge = layouts[0].handle_charge(1, &readers).unwrap();
        let tight_limit = 64;
        let error = plan_mapping_handles(
            std::slice::from_ref(&g),
            &mut layouts,
            2,
            charge,
            vec![(1, readers, charge)],
            tight_limit,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("exceed local handle limit (64) by "), "{error}");
    }

    #[test]
    fn compact_cache_layouts_counts_peer_readers_and_honors_cluster_gate() {
        let a = 2 << 20;
        let units = [vec![73_728; 24], vec![16_896; 24]].concat();
        let mut geometries: Vec<_> = (0..16)
            .map(|_| geometry(35_122_432, &units, 3103, 159))
            .collect();
        let mut peer_readers = HashMap::new();
        let sixteen: Vec<(usize, i32)> = (0..16).map(|r| (r, r as i32)).collect();
        for g in &mut geometries {
            for (idx, plane) in g.planes.iter_mut().enumerate() {
                plane.address = 0x1000 + (idx as u64) * 0x100;
                if idx < 24 {
                    peer_readers.insert(plane.address, sixteen.clone());
                }
            }
        }
        let rank_specs: Vec<_> = (0..16).map(|_| (&peer_readers, vec![(0x9000u64, 110u64)])).collect();
        let make = || {
            geometries
                .iter()
                .map(|g| Layout::new(g, 2, &[a; 48], a).unwrap())
                .collect::<Vec<_>>()
        };
        let mut clustered = make();
        assert_eq!(
            compact_cache_layouts_with_readers(
                &rank_specs,
                &geometries,
                &mut clustered,
                2,
                local_handle_limit("linux", true),
            )
            .unwrap(),
            (0, 0)
        );
        let mut local = make();
        let (before, after) = compact_cache_layouts_with_readers(
            &rank_specs,
            &geometries,
            &mut local,
            2,
            local_handle_limit("linux", false),
        )
        .unwrap();
        assert!(before >= RM_HANDLE_RANGE);
        assert!(after < RM_HANDLE_RANGE);
        assert_ne!(local[0].aliases, clustered[0].aliases);
    }

    #[test]
    fn per_host_charges_cannot_change_cluster_or_non_linux_snapshot_ranges() {
        assert_eq!(local_handle_limit("linux", false), Some(RM_HANDLE_RANGE));
        let units = [vec![73_728; 24], vec![16_896; 24]].concat();
        let g = geometry(35_122_432, &units, 3103, 159);
        let make = || Layout::new(&g, 2, &[2 << 20; 48], 2 << 20).unwrap();
        let canonical = make().aliases;
        for (os, cluster) in [("linux", true), ("windows", false), ("windows", true)] {
            let limit = local_handle_limit(os, cluster);
            assert_eq!(limit, None);
            let mut layouts = [make()];
            assert_eq!(
                compact_cache_layouts(&[], std::slice::from_ref(&g), &mut layouts, 2, limit)
                    .unwrap(),
                (0, 0)
            );
            assert_eq!(layouts[0].aliases, canonical);
        }
    }

    #[test]
    fn alignment_padding_can_back_complete_snapshots_without_kv_aliases() {
        for snapshots in [0, 1, 7, 8, 16, 32] {
            let geometry = geometry(64 << 10, &[64 << 10, 4 << 10], 1024, 9 + snapshots);
            let layout = Layout::new(&geometry, 9, &[2 << 20; 2], 2 << 20).unwrap();
            assert_eq!(layout.aliases.len(), snapshots as usize);
            for (row, range) in layout.aliases.iter().enumerate() {
                assert_eq!(range.is_empty(), row < 23);
            }
        }
    }

    #[test]
    fn plane_granule_padding_backs_snapshots_without_kv_conflicts() {
        let alignment = 2 << 20;
        let geometry = geometry(256 << 10, &[64 << 10, 4 << 10], 1000, 8 + 16);
        let layout = Layout::new(&geometry, 8, &[alignment; 2], alignment).unwrap();
        assert_eq!(layout.aliases.len(), 16);
        let mut unconflicted = 0usize;
        for (snapshot, range) in layout.aliases.iter().enumerate() {
            let pages = conflicts(&layout, &geometry, 8 + snapshot as u64);
            if pages.is_empty() {
                assert_eq!(range, &(0..0));
                unconflicted += 1;
            } else {
                assert!(range.start <= range.end && range.end <= 1000);
                assert!((range.start as u64) <= *pages.first().unwrap());
                assert!((range.end as u64) > *pages.last().unwrap());
            }
        }
        assert!(unconflicted > 0);
    }

    #[test]
    #[ignore = "requires an assigned CUDA_VISIBLE_DEVICES GPU"]
    fn cuda_aliases_preserve_rows_while_unowned_pages_are_rewritten() {
        use super::super::{
            driver::Driver,
            format::{Allocation, Context},
        };
        let driver = Driver::load().unwrap();
        for equal_planes in [false, true] {
            unsafe {
                driver.check("cuInit", (driver.init)(0)).unwrap();
                let mut device = 0;
                driver
                    .check("cuDeviceGet", (driver.device)(&mut device, 0))
                    .unwrap();
                let mut architecture = [0; 2];
                for (value, attribute) in architecture.iter_mut().zip([75, 76]) {
                    driver
                        .check(
                            "cuDeviceGetAttribute",
                            (driver.device_attribute)(value, attribute, device),
                        )
                        .unwrap();
                }
                let mut core = Core {
                    resources: Default::default(),
                    events: vec![],
                    architecture,
                    modules: vec![],
                    functions: vec![],
                    graphs: vec![],
                    reservations: vec![],
                    execution: None,
                    context: Context::default(),
                    allocations: vec![],
                };
                let mut owner = NativeResources::initialize(&core, 0).unwrap();
                let properties = AllocationProperties {
                    kind: 1,
                    handles: 0,
                    location: Location { kind: 1, device },
                    windows: std::ptr::null_mut(),
                    flags: [0; 8],
                };
                let mut alignment = 0;
                driver
                    .check(
                        "cuMemGetAllocationGranularity",
                        (driver.granularity)(&mut alignment, &properties, 0),
                    )
                    .unwrap();
                let alignment = alignment as u64;
                let units = if equal_planes {
                    vec![alignment / 8; 8]
                } else {
                    vec![alignment / 8, alignment / 4, alignment / 64]
                };
                let mut geometry = geometry(
                    if equal_planes {
                        7 * alignment / 2
                    } else {
                        3 * alignment / 2
                    },
                    &units,
                    128,
                    7,
                );
                for plane in std::iter::once(&mut geometry.row).chain(&mut geometry.planes) {
                    let units = if plane.name == "rows" { 7 } else { 128 };
                    let bytes = (units * plane.unit_bytes).div_ceil(alignment) * alignment;
                    driver
                        .check(
                            "cuMemAddressReserve",
                            (driver.reserve)(&mut plane.address, bytes as usize, 0, 0, 0),
                        )
                        .unwrap();
                    owner.reservations.push((plane.address, bytes as usize));
                    core.allocations.push(Allocation {
                        name: plane.name.clone(),
                        nbytes: bytes as usize,
                        alignment: alignment as usize,
                        file: None,
                        data_spans: vec![],
                        sources: vec![],
                        relocations: vec![],
                        address: Some(plane.address),
                        memory: Memory::Device,
                        sizing: None,
                    });
                }
                let alignments = vec![alignment; geometry.planes.len()];
                let layout = Layout::new(&geometry, 3, &alignments, alignment).unwrap();
                if equal_planes {
                    assert!(
                        layout.chunks.len()
                            < Layout::unmerged(&geometry, 3, &alignments, alignment)
                                .unwrap()
                                .chunks
                                .len()
                    );
                }
                owner
                    .allocate_shared_cache(&core, &geometry, &layout)
                    .unwrap();
                // Distinct bytes within each row detect repeated or misplaced
                // backing, not just writes crossing snapshot boundaries.
                let expected: Vec<Vec<u8>> = (0..7)
                    .map(|row| {
                        (0..geometry.row.unit_bytes as usize)
                            .map(|i| {
                                ((i ^ (i >> 8) ^ (i >> 16) ^ (i >> 24)) as u8)
                                    .wrapping_add(row * 31)
                            })
                            .collect()
                    })
                    .collect();
                for (row, bytes) in expected.iter().enumerate() {
                    driver
                        .check(
                            "cuMemcpyHtoD_v2",
                            (driver.upload)(
                                geometry.row.address + row as u64 * geometry.row.unit_bytes,
                                bytes.as_ptr().cast(),
                                bytes.len(),
                            ),
                        )
                        .unwrap();
                }
                // Release snapshots one at a time, overwriting every page outside the remaining exclusions.
                for released in 0..=4 {
                    for page in 0..geometry.page_capacity {
                        if layout.aliases[released..]
                            .iter()
                            .any(|r| r.contains(&(page as u32)))
                        {
                            continue;
                        }
                        for plane in &geometry.planes {
                            driver
                                .check(
                                    "cuMemsetD8_v2",
                                    (driver.zero)(
                                        plane.address + page * plane.unit_bytes,
                                        0xA5,
                                        plane.unit_bytes as usize,
                                    ),
                                )
                                .unwrap();
                        }
                    }
                    driver
                        .check("cuCtxSynchronize", (driver.context_sync)())
                        .unwrap();
                    for row in (0..3).chain(3 + released as u64..7) {
                        let mut contents = vec![0; geometry.row.unit_bytes as usize];
                        driver
                            .check(
                                "cuMemcpyDtoH_v2",
                                (driver.download)(
                                    contents.as_mut_ptr().cast(),
                                    geometry.row.address + row * geometry.row.unit_bytes,
                                    contents.len(),
                                ),
                            )
                            .unwrap();
                        assert!(
                            contents == expected[row as usize],
                            "released {released}, row {row}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn unaligned_rows_and_multiplane_pages_have_complete_conflict_ranges() {
        for (row_bytes, units, pages, alignment) in [
            (81_464_576, vec![135_168, 8_448].repeat(12), 3844, 65536),
            (81_464_576, vec![135_168, 8_448].repeat(12), 3844, 2097152),
            (
                122_932_224,
                vec![65_536, 65_536, 4_096].repeat(13),
                9056,
                2097152,
            ),
        ] {
            let planes: Vec<_> = units
                .iter()
                .enumerate()
                .map(|(i, &unit)| flare_selene::Plane {
                    name: format!("plane-{i}"),
                    address: 0,
                    unit_bytes: unit,
                })
                .collect();
            let geometry = CacheGeometry {
                page_tokens: 64,
                page_capacity: pages,
                row_capacity: 41,
                row: flare_selene::Plane {
                    name: "rows".into(),
                    address: 0,
                    unit_bytes: row_bytes,
                },
                planes,
                placement: Default::default(),
                codec_fields: Vec::new(),
                snapshot_pages: Vec::new(),
            };
            let layout =
                Layout::new(&geometry, 9, &vec![alignment; units.len()], alignment).unwrap();
            assert_eq!(layout.aliases.len(), 32);
            let physical = layout.fixed_bytes + layout.chunks.iter().map(|c| c.bytes).sum::<u64>();
            let expected = layout.fixed_bytes
                + units
                    .iter()
                    .map(|u| (u * pages).div_ceil(alignment) * alignment)
                    .sum::<u64>();
            assert_eq!(physical, expected);
            for (plane, &unit) in units.iter().enumerate() {
                let mut pieces: Vec<_> =
                    layout.chunks.iter().filter(|c| c.plane == plane).collect();
                pieces.sort_by_key(|c| c.offset);
                let mut end = 0;
                for chunk in pieces {
                    assert_eq!(chunk.offset, end);
                    end += chunk.bytes;
                }
                assert_eq!(end, (unit * pages).div_ceil(alignment) * alignment);
            }
            for (snapshot, range) in layout.aliases.iter().enumerate() {
                let start = (snapshot as u64 + 9) * row_bytes;
                let end = start + row_bytes;
                for chunk in &layout.chunks {
                    let lo = start.max(chunk.row_offset);
                    let hi = end.min(chunk.row_offset + chunk.row_bytes);
                    if lo >= hi {
                        continue;
                    }
                    let unit = units[chunk.plane];
                    let first = (chunk.offset + lo - chunk.row_offset) / unit;
                    let last = (chunk.offset + hi - chunk.row_offset)
                        .div_ceil(unit)
                        .min(pages);
                    assert!(range.start as u64 <= first && range.end as u64 >= last);
                }
                // Interleaving prevents one row from excluding whole per-layer page pools.
                assert!(
                    (range.end - range.start) as u64
                        <= 2 * row_bytes.div_ceil(geometry.page_bytes()) + 16
                );
            }
        }
    }
}

impl NativeResources {
    pub(super) unsafe fn allocate_shared_cache(
        &mut self,
        core: &Core,
        geometry: &CacheGeometry,
        layout: &Layout,
    ) -> Result<()> {
        unsafe {
            self.driver
                .check("cuCtxSetCurrent", (self.driver.context_set)(self.context))?;
            for plane in std::iter::once(&geometry.row).chain(&geometry.planes) {
                let allocation = core
                    .allocations
                    .iter()
                    .find(|a| a.name == plane.name)
                    .unwrap();
                self.allocations.insert(
                    plane.name.clone(),
                    Storage {
                        base: 0,
                        address: plane.address,
                        nbytes: allocation.nbytes,
                        memory: Memory::Device,
                        fixed: true,
                        mappings: Vec::new(),
                    },
                );
            }
            self.map_cache_chunk(
                layout.fixed_bytes,
                &[(
                    geometry.row.name.as_str(),
                    geometry.row.address,
                    layout.fixed_bytes,
                )],
            )?;
            for chunk in &layout.chunks {
                let plane = &geometry.planes[chunk.plane];
                let mut mappings = vec![(
                    plane.name.as_str(),
                    plane.address + chunk.offset,
                    chunk.bytes,
                )];
                if chunk.row_bytes != 0 {
                    mappings.push((
                        geometry.row.name.as_str(),
                        geometry.row.address + chunk.row_offset,
                        chunk.row_bytes,
                    ));
                }
                self.map_cache_chunk(chunk.bytes, &mappings)?;
            }
            for plane in std::iter::once(&geometry.row).chain(&geometry.planes) {
                self.grant_storage_access(&plane.name)?;
            }
            // Zero each physical byte once, before any graph or transfer can read an alias.
            self.driver.check(
                "cuMemsetD8_v2",
                (self.driver.zero)(geometry.row.address, 0, layout.fixed_bytes as usize),
            )?;
            for plane in &geometry.planes {
                let bytes = self.allocations[&plane.name].nbytes;
                self.driver
                    .check("cuMemsetD8_v2", (self.driver.zero)(plane.address, 0, bytes))?;
            }
            self.driver
                .check("cuCtxSynchronize", (self.driver.context_sync)())?;
        }
        Ok(())
    }

    unsafe fn map_cache_chunk(&mut self, bytes: u64, mappings: &[(&str, u64, u64)]) -> Result<()> {
        let properties = AllocationProperties {
            kind: 1,
            handles: 0,
            location: Location {
                kind: 1,
                device: self.device,
            },
            windows: std::ptr::null_mut(),
            flags: [0; 8],
        };
        let mut physical = 0;
        unsafe {
            self.driver.check(
                "cuMemCreate",
                (self.driver.physical_create)(&mut physical, bytes as usize, &properties, 0),
            )?;
            let result = (|| {
                for &(name, address, length) in mappings {
                    self.driver.check(
                        "cuMemMap",
                        (self.driver.map)(address, length as usize, 0, physical, 0),
                    )?;
                    self.allocations
                        .get_mut(name)
                        .unwrap()
                        .mappings
                        .push((address, length as usize));
                }
                Ok(())
            })();
            (self.driver.physical_release)(physical);
            result
        }
    }
}
