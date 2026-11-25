// SPDX-License-Identifier: Apache-2.0

//! Stored rank shares passed between a host store and its device copier.
use crate::shared::{packed, wire::PackedUnit};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodeStats {
    pub original_bytes: u64,
    pub stored_bytes: u64,
    pub transform_ns: u64,
    pub failed_windows: u64,
    pub stored_failures: u64,
    pub original_failures: u64,
}

pub type DecodeCounters = Arc<std::sync::Mutex<DecodeStats>>;

/// CPU encoder configuration shared with `MemoryStore` so `cpu` mode can copy
/// raw bytes out of VRAM on the CUDA transfer thread without blocking it on
/// CPU compression, then encode DRAM slabs asynchronously on a bounded host pool.
#[cfg(feature = "codec")]
#[derive(Clone)]
pub struct HostEncoder {
    pub profile: [u8; 32],
    pub page: Arc<crate::shared::codec::Plan>,
    pub row: Arc<crate::shared::codec::Plan>,
    pub counters: Option<DecodeCounters>,
}

#[derive(Clone, Debug)]
pub struct PinnedPacket {
    pub unit: PackedUnit,
    pub(crate) spans: Vec<(usize, usize)>,
}

impl PinnedPacket {
    /// # Safety
    /// The ranges must stay registered and immutable through the last copier fence.
    #[cfg(any(test, feature = "test-support"))]
    pub unsafe fn from_registered(unit: PackedUnit, spans: Vec<(usize, usize)>) -> Self {
        Self { unit, spans }
    }
    /// # Safety
    /// The copier must still own the immutable slab pin or transport window lease.
    pub unsafe fn chunks(&self) -> Vec<&[u8]> {
        self.spans
            .iter()
            .map(|&(address, length)| unsafe {
                std::slice::from_raw_parts(address as *const u8, length)
            })
            .collect()
    }
}

#[derive(Debug)]
pub struct CaptureSlot {
    pub unit: PackedUnit,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Default)]
struct DirectSlabState {
    page_offset: usize,
    row_offset: usize,
    page_dir: Vec<(PackedUnit, usize)>,
    row_dir: Vec<(PackedUnit, usize)>,
    overflowed: bool,
}

#[derive(Debug)]
struct DirectSlabTarget {
    page_addrs: Vec<usize>,
    row_addrs: Vec<usize>,
    slab_bytes: usize,
    state: std::sync::Mutex<DirectSlabState>,
}

#[derive(Debug, Default)]
pub struct CaptureWindow {
    pub units: std::sync::Mutex<Vec<CaptureSlot>>,
    direct: Option<DirectSlabTarget>,
}

impl CaptureWindow {
    pub fn with_slabs(page_addrs: Vec<usize>, row_addrs: Vec<usize>, slab_bytes: usize) -> Self {
        Self {
            units: std::sync::Mutex::new(Vec::new()),
            direct: Some(DirectSlabTarget {
                page_addrs,
                row_addrs,
                slab_bytes,
                state: std::sync::Mutex::new(DirectSlabState::default()),
            }),
        }
    }

    pub fn push(&self, unit: PackedUnit, bytes: Vec<u8>) {
        if self.try_push_direct(&unit, &bytes) {
            return;
        }
        self.units
            .lock()
            .unwrap()
            .push(CaptureSlot { unit, bytes });
    }

    /// Write an encoded unit directly into the target DRAM slabs on the calling
    /// transfer thread when a slab destination is attached (zero heap allocation
    /// and zero main-thread copy), falling back to a `Vec<u8>` slot otherwise.
    pub fn push_slice(&self, unit: PackedUnit, bytes: &[u8]) {
        if self.try_push_direct(&unit, bytes) {
            return;
        }
        self.units.lock().unwrap().push(CaptureSlot {
            unit,
            bytes: bytes.to_vec(),
        });
    }

    fn try_push_direct(&self, unit: &PackedUnit, bytes: &[u8]) -> bool {
        let Some(target) = &self.direct else {
            return false;
        };
        let is_page = unit.kind == crate::shared::wire::packed_unit::Kind::Page as i32;
        let addrs = if is_page {
            &target.page_addrs
        } else {
            &target.row_addrs
        };
        let cap = addrs.len().saturating_mul(target.slab_bytes);
        let start = {
            let mut st = target.state.lock().unwrap();
            if st.overflowed {
                return true;
            }
            let offset = if is_page {
                &mut st.page_offset
            } else {
                &mut st.row_offset
            };
            let Some(end) = offset.checked_add(bytes.len()) else {
                st.overflowed = true;
                return true;
            };
            if end > cap {
                st.overflowed = true;
                return true;
            }
            let start = *offset;
            *offset = end;
            if is_page {
                st.page_dir.push((unit.clone(), start));
            } else {
                st.row_dir.push((unit.clone(), start));
            }
            start
        };
        // SAFETY: exclusive non-overlapping byte range reserved inside pinned slabs
        // owned by an unpublished `MemoryStore` record until transfer completion.
        unsafe {
            write_to_slabs(addrs, target.slab_bytes, start, bytes);
        }
        true
    }

    pub fn take_direct(
        &self,
    ) -> Option<
        std::result::Result<
            (Vec<(PackedUnit, usize)>, usize, Vec<(PackedUnit, usize)>, usize),
            &'static str,
        >,
    > {
        let target = self.direct.as_ref()?;
        let mut st = target.state.lock().unwrap();
        if st.overflowed {
            return Some(Err("direct slab capture exceeded allocated slab capacity"));
        }
        let mut page_dir = std::mem::take(&mut st.page_dir);
        let mut row_dir = std::mem::take(&mut st.row_dir);
        page_dir.sort_by_key(|(u, _)| (u.kind, u.logical_index, u.stored_rank));
        row_dir.sort_by_key(|(u, _)| (u.kind, u.logical_index, u.stored_rank));
        let page_keep = st.page_offset.div_ceil(target.slab_bytes);
        let row_keep = st.row_offset.div_ceil(target.slab_bytes);
        Some(Ok((page_dir, page_keep, row_dir, row_keep)))
    }

    pub fn take_sorted(&self) -> Vec<CaptureSlot> {
        let mut units = std::mem::take(&mut *self.units.lock().unwrap());
        units.sort_by_key(|s| (s.unit.kind, s.unit.logical_index, s.unit.stored_rank));
        units
    }
}

#[derive(Debug)]
pub struct Window {
    pub first: u64,
    pub packets: Vec<PinnedPacket>,
    pub capture: Option<Arc<CaptureWindow>>,
}

impl PartialEq for Window {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}
impl Eq for Window {}

impl Window {
    pub fn restore_pinned(first: u64, packets: Vec<PinnedPacket>) -> Arc<Self> {
        Arc::new(Self {
            first,
            packets,
            capture: None,
        })
    }

    pub fn capture(first: u64) -> Arc<Self> {
        Arc::new(Self {
            first,
            packets: Vec::new(),
            capture: Some(Arc::new(CaptureWindow::default())),
        })
    }

    pub fn capture_shared(first: u64, capture: Arc<CaptureWindow>) -> Arc<Self> {
        Arc::new(Self {
            first,
            packets: Vec::new(),
            capture: Some(capture),
        })
    }
}

#[cfg(feature = "codec")]
pub fn profile(geometry: &[crate::CacheGeometry]) -> Option<[u8; 32]> {
    if geometry
        .iter()
        .any(|rank| rank.placement == crate::offload::KvCompression::Off)
    {
        return None;
    }
    checked_profile(geometry).ok()
}

#[cfg(feature = "codec")]
pub fn checked_profile(geometry: &[crate::CacheGeometry]) -> std::result::Result<[u8; 32], String> {
    let mut expected = None;
    for (rank, g) in geometry.iter().enumerate() {
        let actual = g
            .checked_codec_profile()
            .map_err(|e| format!("rank {rank}: {e}"))?;
        if expected.is_some_and(|p| p != actual) {
            return Err(format!(
                "codec field layout on rank {rank} differs from rank 0"
            ));
        }
        expected = Some(actual);
    }
    expected.ok_or_else(|| "no cache ranks".to_owned())
}

#[cfg(feature = "codec")]
pub fn receive_profiles(geometry: &[crate::CacheGeometry]) -> Vec<[u8; 32]> {
    if geometry.is_empty()
        || geometry
            .iter()
            .any(|rank| rank.placement == crate::offload::KvCompression::Off)
    {
        return Vec::new();
    }
    if checked_profile(geometry).is_err() {
        return Vec::new();
    }
    let g0 = &geometry[0];
    let mut out = Vec::with_capacity(2);
    if let Ok(cpu) =
        crate::shared::codec_layout::profile(g0, crate::shared::codec_layout::REVISION_CPU)
    {
        out.push(cpu);
    }
    if let Ok(gpu) =
        crate::shared::codec_layout::profile(g0, crate::shared::codec_layout::REVISION_GPU)
        && !out.contains(&gpu)
    {
        out.push(gpu);
    }
    out
}

#[cfg(not(feature = "codec"))]
pub fn receive_profiles(_geometry: &[crate::CacheGeometry]) -> Vec<[u8; 32]> {
    Vec::new()
}

/// Copy `bytes` into `addrs` starting at byte offset `dst_off` across fixed-size slabs.
///
/// # Safety
/// The caller must hold exclusive mutable access to `[dst_off, dst_off + bytes.len())`
/// inside `addrs`.
pub(crate) unsafe fn write_to_slabs(
    addrs: &[usize],
    slab_bytes: usize,
    mut dst_off: usize,
    bytes: &[u8],
) {
    let mut src_pos = 0usize;
    let mut remaining = bytes.len();
    while remaining > 0 {
        let idx = dst_off / slab_bytes;
        let within = dst_off % slab_bytes;
        let take = remaining.min(slab_bytes - within);
        let addr = addrs[idx] + within;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr().add(src_pos), addr as *mut u8, take);
        }
        dst_off += take;
        src_pos += take;
        remaining -= take;
    }
}

#[cfg(any(test, feature = "test-support"))]
pub fn reader_caps(profile: Option<[u8; 32]>) -> crate::shared::wire::CodecCaps {
    let slice: Vec<[u8; 32]> = profile.into_iter().collect();
    reader_caps_many(&slice)
}

pub fn reader_caps_many(profiles: &[[u8; 32]]) -> crate::shared::wire::CodecCaps {
    let mut caps = packed::raw_reader();
    for profile in profiles {
        if !caps
            .receive_profiles
            .iter()
            .any(|p| p.as_slice() == profile.as_slice())
        {
            caps.receive_profiles.push(profile.to_vec());
        }
    }
    caps
}
