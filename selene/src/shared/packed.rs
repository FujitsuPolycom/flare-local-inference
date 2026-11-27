/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! Physical layout of v4 records. The packet payload is opaque to Selene.

use super::{
    record::RankSizes,
    wire::packed_unit::{Kind, Representation},
    wire::{CodecCaps, PackedOffer, PackedUnit, RankPlanes},
};
use crate::{Result, geometry::invalid};
use prost::Message;

/// Header version byte for v4 packed records (raw records use `record::VERSION = 3`).
pub const REVISION: u32 = 4;
pub const MAX_DIRECTORY: usize = 2 * 1024 * 1024;
pub const MAX_UNITS: usize = 16384;
pub const MAX_RANKS: usize = 64;

/// Whether a record tail with `pages`, optional `row`, and `ranks` fits within `MAX_UNITS`.
pub(crate) fn fits_units(pages: usize, row: bool, ranks: usize) -> bool {
    pages
        .saturating_add(usize::from(row))
        .saturating_mul(ranks)
        <= MAX_UNITS
}

pub fn raw_reader() -> CodecCaps {
    CodecCaps {
        envelope_revision: REVISION,
        receive_profiles: Vec::new(),
    }
}

pub fn canonical_caps(caps: Option<&CodecCaps>) -> Result<Option<CodecCaps>> {
    let Some(caps) = caps else { return Ok(None) };
    if caps.receive_profiles.len() > 64 || caps.receive_profiles.iter().any(|p| p.len() != 32) {
        return Err(invalid("invalid codec profile set"));
    }
    let mut profiles = caps.receive_profiles.clone();
    profiles.sort_unstable();
    profiles.dedup();
    Ok(Some(CodecCaps {
        envelope_revision: caps.envelope_revision,
        receive_profiles: profiles,
    }))
}

pub fn permits<'a>(
    caps: Option<&CodecCaps>,
    profiles: impl IntoIterator<Item = &'a Vec<u8>>,
) -> bool {
    caps.is_some_and(|c| {
        c.envelope_revision == REVISION
            && !c.receive_profiles.is_empty()
            && profiles.into_iter().all(|p| c.receive_profiles.contains(p))
    })
}

pub fn permits_offer(caps: Option<&CodecCaps>, offer: &PackedOffer) -> bool {
    permits(caps, offer.units.iter().filter(|u| u.representation != Representation::Literal as i32).map(|u| &u.profile))
}

pub fn permits_holding(caps: Option<&CodecCaps>, packed: bool, profiles: &[Vec<u8>]) -> bool {
    (!packed && profiles.is_empty()) || permits(caps, profiles)
}

pub fn crc32(bytes: &[u8]) -> u32 {
    crc32fast::hash(bytes)
}

pub fn crc32_chunks(chunks: &[&[u8]]) -> u32 {
    let mut hash = crc32fast::Hasher::new();
    for chunk in chunks {
        hash.update(chunk);
    }
    hash.finalize()
}

/// Directory on disk, protected as one length-delimited blob.
#[derive(Clone, PartialEq, Message)]
pub(crate) struct Directory {
    #[prost(message, repeated, tag = "1")]
    pub ranks: Vec<RankPlanes>,
    #[prost(message, optional, tag = "2")]
    pub offer: Option<PackedOffer>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub(crate) offer: PackedOffer,
    // Offset of each unit, and one end sentinel. Relative to the payload.
    offsets: Vec<u64>,
    // Index of the first unit of each page, then the first row unit.
    pages: Vec<usize>,
}

impl Eq for Layout {}

impl std::hash::Hash for Layout {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Exhaustive destructures keep Hash aligned with PartialEq whenever
        // fields are added to Layout, PackedOffer, or PackedUnit.
        let Self {
            offer: PackedOffer { first_page, units },
            offsets,
            pages,
        } = self;
        first_page.hash(state);
        units.len().hash(state);
        for PackedUnit {
            kind,
            logical_index,
            stored_rank,
            representation,
            profile,
            original_length,
            stored_length,
            replicas,
            stored_crc,
            original_crc,
            raw_xxh3,
        } in units
        {
            kind.hash(state);
            logical_index.hash(state);
            stored_rank.hash(state);
            representation.hash(state);
            profile.hash(state);
            original_length.hash(state);
            stored_length.hash(state);
            replicas.hash(state);
            stored_crc.hash(state);
            original_crc.hash(state);
            raw_xxh3.hash(state);
        }
        offsets.hash(state);
        pages.hash(state);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Span {
    pub first: u32,
    pub pages: u32,
    pub row: bool,
    pub offset: u64,
    pub length: u64,
}

impl Layout {
    /// Windows contain whole pages. The row is one independent grant, even above the target size.
    pub fn windows(&self, skip: u32, row: bool, max: u64) -> Vec<Span> {
        let mut windows = Vec::new();
        let mut i = skip as usize;
        while i < self.page_count() {
            let mut end = i + 1;
            let offset = self.page_offset(i);
            while end < self.page_count()
                && end - i < 64
                && self.page_offset(end + 1) - offset <= max
            {
                end += 1
            }
            windows.push(Span {
                first: i as u32,
                pages: (end - i) as u32,
                row: false,
                offset,
                length: self.page_offset(end) - offset,
            });
            i = end;
        }
        if row && self.row_bytes() != 0 {
            windows.push(Span {
                first: self.page_count() as u32,
                pages: 0,
                row: true,
                offset: self.page_offset(self.page_count()),
                length: self.row_bytes(),
            });
        }
        windows
    }

    pub fn new(offer: PackedOffer, ranks: &[RankSizes], first: u64, tail: u32) -> Result<Self> {
        if ranks.is_empty()
            || ranks.len() > MAX_RANKS
            || offer.first_page != first
            || offer.units.is_empty()
            || offer.units.len() > MAX_UNITS
            || tail as usize > offer.units.len()
            || offer.encoded_len() > MAX_DIRECTORY
        {
            return Err(invalid(
                "packed directory exceeds bounds or has the wrong first page",
            ));
        }
        // Geometry from disk is untrusted. Check the padding before using the raw size accessors.
        use crate::geometry::padded_bytes;
        for rank in ranks {
            if rank.planes.is_empty()
                || rank.planes.contains(&0)
                || rank.row_bytes == 0
                || padded_bytes(rank.row_bytes).is_none()
                || rank
                    .planes
                    .iter()
                    .try_fold(0u64, |n, &p| n.checked_add(padded_bytes(p)?))
                    .is_none()
            {
                return Err(invalid("packed geometry exceeds address space"));
            }
        }
        let end = first
            .checked_add(tail.into())
            .ok_or_else(|| invalid("packed page overflow"))?;
        let mut offsets = vec![0u64];
        let mut pages = Vec::new();
        let mut i = 0;
        for (kind, index) in (first..end).map(|p| (Kind::Page, p)).chain(std::iter::once((Kind::Row, 0))) {
            pages.push(i);
            if kind == Kind::Row && i == offer.units.len() {
                break;
            }
            let mut seen = 0u64;
            let mut previous = None;
            while let Some(unit) = offer
                .units
                .get(i)
                .filter(|u| u.kind == kind as i32 && u.logical_index == index)
            {
                if previous.is_some_and(|p| p >= unit.stored_rank)
                    || unit.stored_length == 0
                    || unit.stored_length > unit.original_length
                    || unit.replicas.len() >= ranks.len()
                {
                    return Err(invalid("invalid packed unit order or size"));
                }
                previous = Some(unit.stored_rank);
                match Representation::try_from(unit.representation) {
                    Ok(Representation::Literal) if unit.profile.is_empty()
                        && unit.original_length == unit.stored_length
                        && unit.stored_crc == 0
                        && unit.original_crc == 0 => {}
                    Ok(Representation::Packet) if unit.profile.len() == 32 && unit.raw_xxh3 == 0 => {}
                    _ => return Err(invalid("unknown unit representation or checksum domain")),
                }
                let mut last = None;
                for rank in std::iter::once(&unit.stored_rank)
                    .chain(&unit.replicas)
                    .copied()
                {
                    let geometry = ranks
                        .get(rank as usize)
                        .ok_or_else(|| invalid("packed unit rank"))?;
                    let length = if kind == Kind::Page {
                        geometry.page_record_bytes()
                    } else {
                        geometry.row_record_bytes()
                    };
                    if last.is_some_and(|r| r >= rank)
                        || seen & (1u64 << rank) != 0
                        || length != unit.original_length
                    {
                        return Err(invalid("packed unit geometry or replica"));
                    }
                    last = Some(rank);
                    seen |= 1u64 << rank;
                }
                let next = offsets
                    .last()
                    .unwrap()
                    .checked_add(unit.stored_length)
                    .ok_or_else(|| invalid("packed length overflow"))?;
                offsets.push(next);
                i += 1;
            }
            if seen.count_ones() as usize != ranks.len() {
                return Err(invalid("incomplete packed rank coverage"));
            }
        }
        if i != offer.units.len() {
            return Err(invalid("unexpected trailing packed units"));
        }
        Ok(Self {
            offer,
            offsets,
            pages,
        })
    }

    pub fn offer(&self) -> &PackedOffer {
        &self.offer
    }
    pub fn physical_bytes(&self) -> u64 {
        *self.offsets.last().unwrap()
    }
    pub fn page_offset(&self, local: usize) -> u64 {
        self.offsets[self.pages[local]]
    }
    pub fn page_count(&self) -> usize {
        self.pages.len() - 1
    }
    pub fn row_bytes(&self) -> u64 {
        self.physical_bytes() - self.page_offset(self.page_count())
    }
    pub fn units(&self, first: usize, pages: usize, row: bool) -> Option<&[PackedUnit]> {
        let tail = self.page_count();
        let first = if pages == 0 && row { tail } else { first };
        let end = first.checked_add(pages)?;
        if end > tail || (row && end != tail) {
            return None;
        }
        let stop = if row {
            self.offer.units.len()
        } else {
            self.pages[end]
        };
        self.offer.units.get(self.pages[first]..stop)
    }
    pub fn required_profiles(&self, row: bool) -> Vec<Vec<u8>> {
        let mut profiles: Vec<_> = self
            .units(0, self.page_count(), row)
            .unwrap()
            .iter()
            .filter(|u| u.representation == Representation::Packet as i32)
            .map(|u| u.profile.clone())
            .collect();
        profiles.sort_unstable();
        profiles.dedup();
        profiles
    }
}

/// Authenticate the presented grant against the pinned selection before consuming it.
#[cfg(any(test, feature = "test-support"))]
pub fn verify_window(
    expected: &[PackedUnit],
    manifest: &super::wire::PackedWindow,
    bytes: &[u8],
) -> Result<()> {
    if expected != manifest.units {
        return Err(invalid("packed window differs from the pinned selection"));
    }
    verify_units(expected, bytes)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Checksum {
    Stored,
    Original,
}

impl std::fmt::Display for Checksum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Stored => "stored codec checksum mismatch",
            Self::Original => "original codec checksum mismatch",
        })
    }
}

impl std::error::Error for Checksum {}

// Check before handing the bytes to a decoder, including on shared-memory reads.
pub fn verify_units(units: &[PackedUnit], bytes: &[u8]) -> Result<()> {
    let mut rest = bytes;
    for unit in units {
        let n = usize::try_from(unit.stored_length).map_err(|_| invalid("packed unit length"))?;
        let (data, tail) = rest
            .split_at_checked(n)
            .ok_or_else(|| invalid("short packed window"))?;
        let valid = match Representation::try_from(unit.representation) {
            Ok(Representation::Literal) => super::record::checksum(data) == unit.raw_xxh3,
            Ok(Representation::Packet) => crc32(data) == unit.stored_crc,
            _ => false,
        };
        if !valid {
            return Err(crate::Error::carrying(Checksum::Stored));
        }
        rest = tail;
    }
    if !rest.is_empty() {
        return Err(invalid("extra packed window bytes"));
    }
    Ok(())
}

/// Select the rank's units while retaining their offsets in the shared extent.
pub fn leg_units(
    window: &super::wire::PackedWindow,
    rank: Option<u32>,
) -> impl Iterator<Item = (u64, &PackedUnit)> {
    window
        .units
        .iter()
        .scan(0u64, |offset, u| {
            let at = *offset;
            *offset = at.checked_add(u.stored_length)?;
            Some((at, u))
        })
        .filter(move |(_, u)| rank.is_none_or(|r| r == u.stored_rank || u.replicas.contains(&r)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::wire::PackedWindow;

    fn unit(rank: u32, data: &[u8], replicas: Vec<u32>) -> PackedUnit {
        PackedUnit {
            stored_rank: rank,
            replicas,
            representation: 1,
            profile: vec![4; 32],
            original_length: 4096,
            stored_length: data.len() as u64,
            stored_crc: crc32(data),
            original_crc: 7,
            ..Default::default()
        }
    }

    #[test]
    fn replica_legs_read_the_owner_bytes() {
        let a = unit(0, b"aaa", vec![2]);
        let b = unit(1, b"bb", vec![]);
        let window = PackedWindow { units: vec![a, b] };
        let ranks = vec![
            RankSizes {
                planes: vec![4096],
                row_bytes: 1
            };
            3
        ];
        let offer = PackedOffer {
            first_page: 0,
            units: window.units.clone(),
        };
        Layout::new(offer, &ranks, 0, 1).unwrap();
        for (rank, expected) in [
            (None, vec![(0, 3), (3, 2)]),
            (Some(1), vec![(3, 2)]),
            (Some(2), vec![(0, 3)]),
            (Some(0), vec![(0, 3)]),
        ] {
            let spans: Vec<_> = leg_units(&window, rank)
                .map(|(at, u)| (at, u.stored_length))
                .collect();
            assert_eq!(spans, expected);
        }
    }

    #[test]
    fn packed_receive_checks_selection_length_and_stored_crc() {
        let expected = vec![unit(0, b"abc", vec![])];
        let window = PackedWindow {
            units: expected.clone(),
        };
        let mut mutated = window.clone();
        mutated.units[0].original_crc ^= 1;
        assert!(verify_window(&expected, &mutated, b"abc").is_err());
        assert!(verify_window(&expected, &window, b"abd").is_err());
        assert!(verify_window(&expected, &window, b"ab").is_err());
        assert!(verify_window(&expected, &window, b"abcd").is_err());
        verify_window(&expected, &window, b"abc").unwrap();
    }

    #[test]
    fn a_raw_only_receiver_or_missing_ack_cannot_receive_a_packet() {
        let literal = PackedOffer {
            units: vec![PackedUnit::default()],
            ..Default::default()
        };
        assert!(!permits_offer(Some(&raw_reader()), &literal));
        let need = vec![vec![9; 32]];
        assert!(!permits(None, &need));
        assert!(!permits(Some(&raw_reader()), &need));
        let mut caps = CodecCaps {
            envelope_revision: 4,
            receive_profiles: vec![vec![9; 32], vec![5; 32], vec![9; 32]],
        };
        let negotiated_caps = canonical_caps(Some(&caps)).unwrap().unwrap();
        assert_eq!(
            negotiated_caps.receive_profiles,
            vec![vec![5; 32], vec![9; 32]]
        );
        assert!(permits(Some(&negotiated_caps), &need));
        assert!(permits_offer(Some(&negotiated_caps), &literal));
        caps.receive_profiles[1].pop();
        assert!(canonical_caps(Some(&caps)).is_err());
    }

    #[test]
    fn fits_units_enforces_max_units_boundary() {
        // TP4: 4,095 pages + 1 row = 4,096 * 4 = 16,384 (fits); 4,096 pages + 1 row = 16,388 (exceeds).
        assert!(fits_units(4095, true, 4));
        assert!(!fits_units(4096, true, 4));
        assert!(fits_units(4096, false, 4));
        assert!(!fits_units(4097, false, 4));
        assert!(!fits_units(usize::MAX, true, 4));
    }
}
