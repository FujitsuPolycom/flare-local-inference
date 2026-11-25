// SPDX-License-Identifier: Apache-2.0

//! Plan layout, wire parser, and pure-Rust CPU decoder for the GPU-edge lossless checkpoint codec
//! (`GKV1/v2/64-pivots-rans32`).
//!
//! Keeping `GpuPlan` and `decode_checked_cpu` in `flare-selene` (without any CUDA Driver API
//! dependency) lets `cpu` mode and Helios share profile planning, workspace budgeting, and
//! bit-exact `GKV1/v2` CPU decoding while `helios` owns the GPU execution engine.

use super::codec::{
    Checksum, Encoding, MATRIX_BYTES, MATRIX_ELEMS, Plan, byte_shuffle_decode, crc32_chunks_par,
    crc32_par, page_split_decode, unpack_restore_f32_128,
};
use rayon::prelude::*;
use std::cell::RefCell;
use std::io::{Error, ErrorKind, Result};

pub const MAGIC: u32 = u32::from_le_bytes(*b"GKV1");
pub const VERSION: u32 = 2;
pub const HEADER_BYTES: usize = 24;
pub const RAW_FLAG: u32 = 1 << 31;
pub const ANS_CHUNK: usize = 65536;
pub const MAX_LANE_U32S: usize = 3968;
pub const LANE_STRIDE: usize = MAX_LANE_U32S + 2;

const RANS_LANES: usize = 32;
const RANS_PROB_BITS: u32 = 12;
const RANS_PROB_SCALE: u32 = 1 << RANS_PROB_BITS; // 4096
const RANS_L: u32 = 1 << 16; // 65536
const RANS_HDR_RESERVE: usize = 640;
const RANS_LANE_CAP_U16: usize = 1152;
pub const RANS_SLOT_STRIDE: usize = RANS_HDR_RESERVE + RANS_LANES * RANS_LANE_CAP_U16 * 2; // 74368 B
pub const CRC_BLOCKS: usize = 256;
pub const MAX_PAGE_BATCH: usize = 64;
const TARGET_PAGE_BATCH_BYTES: usize = 16 << 20;

/// One contiguous GPU device memory span belonging to a rank's page or row share.
#[derive(Clone, Copy, Debug)]
pub struct DeviceSpan {
    pub device: u64,
    pub offset: usize,
    pub length: usize,
}

fn invalid(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

#[derive(Clone, Copy, Debug)]
pub struct CopyOp {
    /// Offset in the transformed staging buffer.
    pub trans_offset: usize,
    /// Offset in the coalesced rANS batch buffer.
    pub ans_offset: usize,
    pub length: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct TransformOp {
    pub offset: usize,
    pub length: usize,
    pub encoding: Encoding,
}

/// Validated `GKV1/v2` packet header, chunk directory, and matrix lane table shared by CPU and
/// GPU decoders so both enforce identical wire invariants.
#[derive(Clone, Debug)]
pub struct ParsedHeader {
    pub chunk_pkt_offs: Vec<u32>,
    pub chunk_words: Vec<u32>,
    pub lane_offs: Vec<u32>,
    pub lane_u32s: Vec<u16>,
    pub low_offset: usize,
    pub low_bytes: usize,
}

/// Static GPU/CPU transform & coalesced 32-way rANS chunk plan derived from a validated [`Plan`].
#[derive(Clone, Debug)]
pub struct GpuPlan {
    id: u64,
    original_bytes: usize,
    num_mats: usize,
    matrix_offsets: Vec<u64>,
    transforms: Vec<TransformOp>,
    coalesce: Vec<CopyOp>,
    chunk_offsets: Vec<usize>,
    chunk_sizes: Vec<usize>,
    ans_total_bytes: usize,
}

impl GpuPlan {
    pub fn from_plan(plan: &Plan) -> Result<Self> {
        let original_bytes = plan.original_bytes();
        let fields = plan.fields();

        // Build complete non-overlapping interval list, tagging untyped alignment gaps separately
        // from explicit `Encoding::Bytes` fields so all-zero padding lands in its own RLE pool
        // instead of polluting high-entropy byte fields.
        let mut intervals: Vec<(TransformOp, bool)> = Vec::new();
        let mut end = 0usize;
        for &f in fields {
            if f.offset > end {
                intervals.push((
                    TransformOp {
                        offset: end,
                        length: f.offset - end,
                        encoding: Encoding::Bytes,
                    },
                    true,
                ));
            }
            intervals.push((
                TransformOp {
                    offset: f.offset,
                    length: f.length,
                    encoding: f.encoding,
                },
                false,
            ));
            end = f.offset + f.length;
        }
        if end < original_bytes {
            intervals.push((
                TransformOp {
                    offset: end,
                    length: original_bytes - end,
                    encoding: Encoding::Bytes,
                },
                true,
            ));
        }

        // Partition intervals into matrix fields and homogeneous lane pools so every rANS chunk
        // sees a single byte-lane distribution.
        let mut matrix_offsets = Vec::new();
        let mut transforms = Vec::new();
        let mut pools: std::collections::BTreeMap<(u32, u32), Vec<(usize, usize)>> =
            std::collections::BTreeMap::new();

        for (op, is_gap) in intervals {
            match op.encoding {
                Encoding::Float32Matrix128 => {
                    let count = op.length / MATRIX_BYTES;
                    for i in 0..count {
                        matrix_offsets.push((op.offset + i * MATRIX_BYTES) as u64);
                    }
                }
                Encoding::Bytes => {
                    transforms.push(op);
                    let key = if is_gap { (u32::MAX, 0) } else { (0, 0) };
                    pools.entry(key).or_default().push((op.offset, op.length));
                }
                Encoding::Bfloat16 | Encoding::Convolution20 | Encoding::Convolution32 => {
                    // Conv20 (10x bf16) and Conv32 (16x bf16) share BF16 2-byte lane structure.
                    transforms.push(TransformOp {
                        offset: op.offset,
                        length: op.length,
                        encoding: Encoding::Bfloat16,
                    });
                    let half = op.length / 2;
                    let tag = match op.encoding {
                        Encoding::Bfloat16 => 2,
                        Encoding::Convolution20 => 20,
                        _ => 32,
                    };
                    pools.entry((tag, 0)).or_default().push((op.offset, half));
                    pools
                        .entry((tag, 1))
                        .or_default()
                        .push((op.offset + half, half));
                }
                enc => {
                    let spec = enc
                        .page_spec()
                        .ok_or_else(|| invalid("unsupported GKV1 field encoding"))?;
                    transforms.push(op);
                    let nval = spec.rows * spec.cols;
                    let ns = spec.block - nval;
                    let nblocks = op.length / spec.block;
                    for b in 0..nblocks {
                        let base = op.offset + b * spec.block;
                        pools.entry((spec.kind, 0)).or_default().push((base, nval));
                        if ns > 0 {
                            if spec.scale_w > 1 {
                                let lane_len = ns / spec.scale_w as usize;
                                for l in 0..spec.scale_w {
                                    pools.entry((spec.kind, 1 + l)).or_default().push((
                                        base + nval + l as usize * lane_len,
                                        lane_len,
                                    ));
                                }
                            } else {
                                pools
                                    .entry((spec.kind, 1))
                                    .or_default()
                                    .push((base + nval, ns));
                            }
                        }
                    }
                }
            }
        }

        // Assign contiguous ranges inside `ans_in`:
        // 1. Matrix widths come first (at offset 0..num_mats * MATRIX_ELEMS).
        // 2. Each homogeneous non-matrix lane pool follows contiguously.
        let mut chunk_offsets = Vec::new();
        let mut chunk_sizes = Vec::new();
        let mut ans_pos = 0usize;

        let push_chunked =
            |start: usize, len: usize, offs: &mut Vec<usize>, szs: &mut Vec<usize>| {
                if len == 0 {
                    return;
                }
                // Balance chunks evenly across `ceil(len / ANS_CHUNK)` blocks so tail chunks avoid
                // small-chunk rANS table overhead and GPU CTAs stay load-balanced.
                let k = len.div_ceil(ANS_CHUNK);
                let mut p = 0;
                for i in 0..k {
                    let n = (len - p).div_ceil(k - i);
                    offs.push(start + p);
                    szs.push(n);
                    p += n;
                }
            };

        let num_mats = matrix_offsets.len();
        let widths_bytes = num_mats * MATRIX_ELEMS;
        if widths_bytes > 0 {
            push_chunked(0, widths_bytes, &mut chunk_offsets, &mut chunk_sizes);
            ans_pos = widths_bytes;
        }

        let mut coalesce = Vec::new();
        for spans in pools.values() {
            let pool_start = ans_pos;
            for &(trans_offset, length) in spans {
                coalesce.push(CopyOp {
                    trans_offset,
                    ans_offset: ans_pos,
                    length,
                });
                ans_pos += length;
            }
            push_chunked(
                pool_start,
                ans_pos - pool_start,
                &mut chunk_offsets,
                &mut chunk_sizes,
            );
        }

        // Deterministic fingerprint for caching uploaded static plan tables on GPU.
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x0100_0000_01b3);
        };
        mix(original_bytes as u64);
        mix(num_mats as u64);
        mix(chunk_sizes.len() as u64);
        mix(coalesce.len() as u64);
        mix(transforms.len() as u64);
        for t in &transforms {
            mix(t.offset as u64);
            mix(t.length as u64);
            mix(t.encoding as u64);
        }
        for &m in &matrix_offsets {
            mix(m);
        }
        for &o in &chunk_offsets {
            mix(o as u64);
        }
        for &s in &chunk_sizes {
            mix(s as u64);
        }
        for c in &coalesce {
            mix(c.trans_offset as u64);
            mix(c.ans_offset as u64);
            mix(c.length as u64);
        }

        Ok(Self {
            id: h,
            original_bytes,
            num_mats,
            matrix_offsets,
            transforms,
            coalesce,
            chunk_offsets,
            chunk_sizes,
            ans_total_bytes: ans_pos,
        })
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn original_bytes(&self) -> usize {
        self.original_bytes
    }

    pub fn num_chunks(&self) -> usize {
        self.chunk_sizes.len()
    }

    pub fn num_mats(&self) -> usize {
        self.num_mats
    }

    pub fn matrix_offsets(&self) -> &[u64] {
        &self.matrix_offsets
    }

    pub fn transforms(&self) -> &[TransformOp] {
        &self.transforms
    }

    pub fn coalesce(&self) -> &[CopyOp] {
        &self.coalesce
    }

    pub fn chunk_offsets(&self) -> &[usize] {
        &self.chunk_offsets
    }

    pub fn chunk_sizes(&self) -> &[usize] {
        &self.chunk_sizes
    }

    pub fn ans_total_bytes(&self) -> usize {
        self.ans_total_bytes
    }

    pub fn metadata_bytes(&self) -> usize {
        HEADER_BYTES + self.chunk_sizes.len() * 4 + self.num_mats * 8
    }

    pub fn max_packet(&self) -> usize {
        self.metadata_bytes() + self.original_bytes
    }

    /// Maximum number of non-matrix KV pages encoded or decoded together in one GPU wave.
    /// Fits within the row workspace on large-row models (0 extra VRAM) and caps small-page
    /// models near `TARGET_PAGE_BATCH_BYTES` per aliased buffer.
    pub fn page_batch_cap(page: &Self, row: &Self) -> usize {
        if page.original_bytes == 0 || page.num_mats > 0 || page.num_chunks() == 0 {
            return 1;
        }
        let page_slot = page
            .original_bytes
            .max(page.max_packet())
            .max(page.num_chunks() * RANS_SLOT_STRIDE);
        let target = row
            .original_bytes
            .max(row.max_packet())
            .max(TARGET_PAGE_BATCH_BYTES);
        (target / page_slot.max(1)).clamp(1, MAX_PAGE_BATCH)
    }

    /// Estimate fixed device workspace bytes needed for a `(page, row)` plan pair using
    /// the 3-buffer aliased workspace matching `GpuRankCodec::new` (plus cubin module headroom).
    pub fn workspace_bytes_estimate_pair(page: &Self, row: &Self) -> u64 {
        let batch = Self::page_batch_cap(page, row) as u64;
        let max_orig = (page.original_bytes as u64 * batch).max(row.original_bytes as u64);
        let max_pkt = (page.max_packet() as u64 * batch)
            .max(row.max_packet() as u64)
            .max(max_orig);
        let max_mats = page.num_mats.max(row.num_mats).max(1) as u64;
        let max_chunks = (page.num_chunks() as u64 * batch)
            .max(row.num_chunks() as u64)
            .max(1);
        let buf_a = max_orig
            .max(max_pkt)
            .max(max_chunks * RANS_SLOT_STRIDE as u64);
        let buf_b = max_orig.max(max_pkt);
        let buf_c = max_orig.max(max_mats * 4 * LANE_STRIDE as u64 * 4);
        // 3 aliased workspace buffers + small metadata/plan descriptors + 8 MiB cubin headroom.
        buf_a + buf_b + buf_c + (8 << 20)
    }

    /// Validate a `GKV1/v2` packet header, chunk directory, and matrix lane table.
    pub fn parse_packet_header(&self, pkt: &[u8]) -> Result<ParsedHeader> {
        let total = pkt.len();
        let meta = self.metadata_bytes();
        if total < meta || total > self.max_packet() {
            return Err(invalid("invalid GKV1 packet length"));
        }
        let word = |p: usize| u32::from_le_bytes(pkt[p..p + 4].try_into().unwrap());
        let nc = self.num_chunks();
        if word(0) != MAGIC
            || word(4) != VERSION
            || word(8) as usize != self.original_bytes
            || word(12) as usize != nc
            || word(16) as usize != self.num_mats
        {
            return Err(invalid("invalid GKV1 packet header"));
        }
        let low_bytes = word(20) as usize;
        let dir_end = HEADER_BYTES + nc * 4;
        let total_lanes = self.num_mats * 4;

        let mut chunk_pkt_offs = Vec::with_capacity(nc);
        let mut chunk_words = Vec::with_capacity(nc);
        let mut pos = meta;
        for i in 0..nc {
            let entry = word(HEADER_BYTES + i * 4);
            let is_raw = (entry & RAW_FLAG) != 0;
            let len = (entry & !RAW_FLAG) as usize;
            let expected_raw = self.chunk_sizes[i];
            if len > total - pos {
                return Err(invalid("truncated GKV1 chunk"));
            }
            if is_raw {
                if len != expected_raw {
                    return Err(invalid("invalid GKV1 raw chunk length"));
                }
            } else if len == 0 || len >= expected_raw {
                return Err(invalid("invalid GKV1 compressed chunk length"));
            }
            chunk_pkt_offs.push(pos as u32);
            chunk_words.push(entry);
            pos += len;
        }
        if pos + low_bytes != total {
            return Err(invalid("GKV1 payload length mismatch"));
        }

        let mut lane_offs = Vec::with_capacity(total_lanes);
        let mut lane_u32s = Vec::with_capacity(total_lanes);
        let mut expected_low = 0usize;
        for i in 0..total_lanes {
            let nw = u16::from_le_bytes([pkt[dir_end + i * 2], pkt[dir_end + i * 2 + 1]]);
            if usize::from(nw) > MAX_LANE_U32S {
                return Err(invalid("GKV1 matrix lane exceeds maximum words"));
            }
            lane_offs.push(expected_low as u32);
            lane_u32s.push(nw);
            expected_low += usize::from(nw) * 4;
        }
        if expected_low != low_bytes {
            return Err(invalid("GKV1 matrix low-bit byte count mismatch"));
        }

        Ok(ParsedHeader {
            chunk_pkt_offs,
            chunk_words,
            lane_offs,
            lane_u32s,
            low_offset: pos,
            low_bytes,
        })
    }

    /// Verify `stored_crc`, decode a `GKV1/v2` packet on CPU (optionally using Rayon `pool`),
    /// and verify `original_crc`. Enables `cpu` nodes to natively restore `gpu`-encoded shares.
    pub fn decode_checked_cpu(
        &self,
        chunks: &[&[u8]],
        out: &mut [u8],
        stored_crc: u32,
        original_crc: u32,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<()> {
        if out.len() != self.original_bytes {
            return Err(invalid("GKV1 CPU decode destination length mismatch"));
        }
        if crc32_chunks_par(chunks, pool) != stored_crc {
            return Err(Error::new(ErrorKind::InvalidData, Checksum::Stored));
        }

        // Reuse per-thread scratch buffers up to `MAX_RETAINED_DECODE_SCRATCH` to avoid
        // per-page allocation churn while releasing oversized row buffers immediately on exit.
        CPU_DECODE_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            struct TrimGuard<'a>(&'a mut CpuDecodeScratch);
            impl Drop for TrimGuard<'_> {
                fn drop(&mut self) {
                    self.0.trim();
                }
            }
            let guard = TrimGuard(&mut *scratch);
            let scratch = &mut *guard.0;
            let pkt: &[u8] = if chunks.len() == 1 {
                chunks[0]
            } else {
                let total: usize = chunks.iter().map(|c| c.len()).sum();
                scratch.pkt_buf.clear();
                scratch.pkt_buf.reserve(total);
                for c in chunks {
                    scratch.pkt_buf.extend_from_slice(c);
                }
                &scratch.pkt_buf
            };

            let hdr = self.parse_packet_header(pkt)?;
            let nc = self.num_chunks();

            // 1. Decode all rANS32 / raw chunks into reusable `ans_buf` (parallelized when `pool` is provided).
            if scratch.ans_buf.len() < self.ans_total_bytes {
                scratch.ans_buf.resize(self.ans_total_bytes, 0);
            }
            let ans_buf = &mut scratch.ans_buf[..self.ans_total_bytes];
            if nc > 0 {
                let mut slices: Vec<&mut [u8]> = Vec::with_capacity(nc);
                let mut rest = &mut *ans_buf;
                for &sz in &self.chunk_sizes {
                    let (head, tail) = rest.split_at_mut(sz);
                    slices.push(head);
                    rest = tail;
                }
                let decode_one = |(dst, (&off, &entry)): (&mut [u8], (&u32, &u32))| {
                    let is_raw = (entry & RAW_FLAG) != 0;
                    let len = (entry & !RAW_FLAG) as usize;
                    let src = &pkt[off as usize..off as usize + len];
                    if is_raw {
                        dst.copy_from_slice(src);
                        Ok(())
                    } else {
                        rans32_decode_chunk(src, dst)
                    }
                };
                if let Some(p) = pool.filter(|p| p.current_num_threads() > 1 && nc > 1) {
                    p.install(|| {
                        slices
                            .into_par_iter()
                            .zip(hdr.chunk_pkt_offs.par_iter().zip(hdr.chunk_words.par_iter()))
                            .try_for_each(decode_one)
                    })?;
                } else {
                    for pair in slices
                        .into_iter()
                        .zip(hdr.chunk_pkt_offs.iter().zip(hdr.chunk_words.iter()))
                    {
                        decode_one(pair)?;
                    }
                }
            }

            // 2. Reconstruct 128x128 float32 matrices via existing SIMD/scalar CPU inverse predictor.
            if self.num_mats > 0 {
                let widths = &ans_buf[..self.num_mats * MATRIX_ELEMS];
                let low_all = &pkt[hdr.low_offset..hdr.low_offset + hdr.low_bytes];
                if let Some(p) = pool.filter(|p| p.current_num_threads() > 1 && self.num_mats > 1) {
                    let mut mat_slices: Vec<(usize, &mut [u8])> = Vec::with_capacity(self.num_mats);
                    let mut cursor = 0usize;
                    let mut rest = &mut *out;
                    for (m, &mat_off) in self.matrix_offsets.iter().enumerate() {
                        let off = mat_off as usize;
                        let (_, tail) = rest.split_at_mut(off - cursor);
                        let (mat_dst, after) = tail.split_at_mut(MATRIX_BYTES);
                        mat_slices.push((m, mat_dst));
                        cursor = off + MATRIX_BYTES;
                        rest = after;
                    }
                    p.install(|| {
                        mat_slices.into_par_iter().try_for_each(|(m, mat_dst)| {
                            let mat_lanes = [
                                hdr.lane_u32s[m * 4],
                                hdr.lane_u32s[m * 4 + 1],
                                hdr.lane_u32s[m * 4 + 2],
                                hdr.lane_u32s[m * 4 + 3],
                            ];
                            let lb: usize = mat_lanes.iter().map(|&w| usize::from(w) * 4).sum();
                            let lstart = hdr.lane_offs[m * 4] as usize;
                            unpack_restore_f32_128(
                                mat_dst,
                                &widths[m * MATRIX_ELEMS..(m + 1) * MATRIX_ELEMS],
                                &low_all[lstart..lstart + lb],
                                &mat_lanes,
                            )
                        })
                    })?;
                } else {
                    for (m, &mat_off) in self.matrix_offsets.iter().enumerate() {
                        let off = mat_off as usize;
                        let mat_lanes = [
                            hdr.lane_u32s[m * 4],
                            hdr.lane_u32s[m * 4 + 1],
                            hdr.lane_u32s[m * 4 + 2],
                            hdr.lane_u32s[m * 4 + 3],
                        ];
                        let lb: usize = mat_lanes.iter().map(|&w| usize::from(w) * 4).sum();
                        let lstart = hdr.lane_offs[m * 4] as usize;
                        unpack_restore_f32_128(
                            &mut out[off..off + MATRIX_BYTES],
                            &widths[m * MATRIX_ELEMS..(m + 1) * MATRIX_ELEMS],
                            &low_all[lstart..lstart + lb],
                            &mat_lanes,
                        )?;
                    }
                }
            }

            // 3. Uncoalesce & invert non-matrix transforms (parallelized across transforms when `pool` is provided).
            if !self.transforms.is_empty() {
                if scratch.trans_buf.len() < self.original_bytes {
                    scratch.trans_buf.resize(self.original_bytes, 0);
                }
                let trans_buf = &mut scratch.trans_buf[..self.original_bytes];
                for c in &self.coalesce {
                    trans_buf[c.trans_offset..c.trans_offset + c.length]
                        .copy_from_slice(&ans_buf[c.ans_offset..c.ans_offset + c.length]);
                }
                let apply_op = |op: &TransformOp, dst: &mut [u8], tmp: &mut [u8; 4096]| -> Result<()> {
                    let src = &trans_buf[op.offset..op.offset + op.length];
                    match op.encoding {
                        Encoding::Bytes => {
                            dst.copy_from_slice(src);
                            Ok(())
                        }
                        Encoding::Bfloat16 => {
                            byte_shuffle_decode(dst, src, 2);
                            Ok(())
                        }
                        enc => {
                            let spec = enc.page_spec().unwrap();
                            page_split_decode(dst, src, spec.kind, tmp)
                        }
                    }
                };
                if let Some(p) =
                    pool.filter(|p| p.current_num_threads() > 1 && self.transforms.len() > 1)
                {
                    let mut op_slices: Vec<(&TransformOp, &mut [u8])> =
                        Vec::with_capacity(self.transforms.len());
                    let mut cursor = 0usize;
                    let mut rest = &mut *out;
                    for op in &self.transforms {
                        let (_, tail) = rest.split_at_mut(op.offset - cursor);
                        let (dst, after) = tail.split_at_mut(op.length);
                        op_slices.push((op, dst));
                        cursor = op.offset + op.length;
                        rest = after;
                    }
                    p.install(|| {
                        op_slices.into_par_iter().try_for_each_init(
                            || [0u8; 4096],
                            |tmp, (op, dst)| apply_op(op, dst, tmp),
                        )
                    })?;
                } else {
                    let mut tmp = [0u8; 4096];
                    for op in &self.transforms {
                        apply_op(op, &mut out[op.offset..op.offset + op.length], &mut tmp)?;
                    }
                }
            }

            if crc32_par(out, pool) != original_crc {
                return Err(Error::new(ErrorKind::InvalidData, Checksum::Original));
            }
            Ok(())
        })
    }
}

/// Maximum retained capacity per thread-local CPU decode scratch buffer (2 MiB).
/// Covers hot page restores without pinning multi-megabyte row buffers on worker threads.
const MAX_RETAINED_DECODE_SCRATCH: usize = 2 << 20;

#[derive(Default)]
struct CpuDecodeScratch {
    pkt_buf: Vec<u8>,
    ans_buf: Vec<u8>,
    trans_buf: Vec<u8>,
}

impl CpuDecodeScratch {
    fn trim(&mut self) {
        for buf in [&mut self.pkt_buf, &mut self.ans_buf, &mut self.trans_buf] {
            if buf.capacity() > MAX_RETAINED_DECODE_SCRATCH {
                *buf = Vec::new();
            }
        }
    }
}

thread_local! {
    static CPU_DECODE_SCRATCH: RefCell<CpuDecodeScratch> = RefCell::new(CpuDecodeScratch::default());
    static RANS_LUT_SCRATCH: RefCell<Box<[u32; RANS_PROB_SCALE as usize]>> =
        RefCell::new(Box::new([0u32; RANS_PROB_SCALE as usize]));
}

/// Pure-Rust decoder for one 32-way interleaved 12-bit static rANS (or RLE) chunk.
/// Matches `kv_rans32_encode_gpu` / `kv_rans32_decode_gpu` bit-for-bit with strict bounds checking.
pub(crate) fn rans32_decode_chunk(src: &[u8], dst: &mut [u8]) -> Result<()> {
    if src.len() < 2 {
        return Err(invalid("truncated GKV1 rANS chunk"));
    }
    match src[0] {
        1 => {
            if src.len() != 2 {
                return Err(invalid("invalid GKV1 RLE chunk length"));
            }
            dst.fill(src[1]);
            Ok(())
        }
        mode @ (2 | 3) => {
            let mut freq = [0u32; 256];
            let tbl_len;
            if mode == 2 {
                let k = usize::from(src[1]) + 1;
                let fbytes = (k * 12).div_ceil(8);
                tbl_len = 2 + k + fbytes;
                if !(2..=150).contains(&k) || src.len() < tbl_len + RANS_LANES * 6 {
                    return Err(invalid("invalid GKV1 sparse rANS table"));
                }
                let syms = &src[2..2 + k];
                let fp = &src[2 + k..tbl_len];
                let mut prev = -1i32;
                for i in 0..k {
                    let s = i32::from(syms[i]);
                    if s <= prev {
                        return Err(invalid("unsorted GKV1 sparse rANS symbols"));
                    }
                    prev = s;
                    let bi = (i >> 1) * 3;
                    let f = if (i & 1) == 0 {
                        u32::from(fp[bi]) | ((u32::from(fp[bi + 1]) & 0x0f) << 8)
                    } else {
                        (u32::from(fp[bi + 1]) >> 4) | (u32::from(fp[bi + 2]) << 4)
                    };
                    if f == 0 || f > 4095 {
                        return Err(invalid("invalid GKV1 sparse rANS frequency"));
                    }
                    freq[s as usize] = f;
                }
                if (k & 1) != 0 && (fp[(k >> 1) * 3 + 1] & 0xf0) != 0 {
                    return Err(invalid("non-zero GKV1 sparse rANS padding nibble"));
                }
            } else {
                tbl_len = 1 + 384;
                if src.len() < tbl_len + RANS_LANES * 6 {
                    return Err(invalid("invalid GKV1 dense rANS table"));
                }
                let mut active = 0usize;
                let fp = &src[1..tbl_len];
                for s in (0..256).step_by(2) {
                    let bi = (s >> 1) * 3;
                    let a = u32::from(fp[bi]) | ((u32::from(fp[bi + 1]) & 0x0f) << 8);
                    let b = (u32::from(fp[bi + 1]) >> 4) | (u32::from(fp[bi + 2]) << 4);
                    if a > 4095 || b > 4095 {
                        return Err(invalid("invalid GKV1 dense rANS frequency"));
                    }
                    freq[s] = a;
                    freq[s + 1] = b;
                    active += usize::from(a > 0) + usize::from(b > 0);
                }
                if active < 2 {
                    return Err(invalid("invalid GKV1 dense rANS active count"));
                }
            }

            let mut cdf = [0u32; 256];
            let mut sum = 0u32;
            for s in 0..256 {
                cdf[s] = sum;
                sum += freq[s];
            }
            if sum != RANS_PROB_SCALE {
                return Err(invalid("GKV1 rANS frequency sum mismatch"));
            }

            let mut states = [0u32; RANS_LANES];
            let mut p = tbl_len;
            for st in &mut states {
                let v = u32::from_le_bytes(src[p..p + 4].try_into().unwrap());
                if v < RANS_L {
                    return Err(invalid("invalid GKV1 rANS initial state"));
                }
                *st = v;
                p += 4;
            }
            let mut lane_bytes = [0usize; RANS_LANES];
            let mut payload_bytes = 0usize;
            for lb in &mut lane_bytes {
                let nb = usize::from(u16::from_le_bytes([src[p], src[p + 1]])) * 2;
                *lb = nb;
                payload_bytes += nb;
                p += 2;
            }
            if p + payload_bytes != src.len() {
                return Err(invalid("GKV1 rANS payload length mismatch"));
            }
            let mut rest_payload = &src[p..];
            let mut lanes: [&[u8]; RANS_LANES] = [&[]; RANS_LANES];
            for (slot, &nb) in lanes.iter_mut().zip(&lane_bytes) {
                let (head, tail) = rest_payload.split_at(nb);
                *slot = head;
                rest_payload = tail;
            }

            RANS_LUT_SCRATCH.with(|lut_cell| {
                let mut lut_ref = lut_cell.borrow_mut();
                let lut: &mut [u32; RANS_PROB_SCALE as usize] = &mut lut_ref;
                for s in 0..256 {
                    let f = freq[s];
                    if f > 0 {
                        let c = cdf[s] as usize;
                        let entry = (s as u32) | (f << 8) | ((c as u32) << 20);
                        lut[c..c + f as usize].fill(entry);
                    }
                }

                #[inline(always)]
                fn step_lane(
                    state: &mut u32,
                    lane: &mut &[u8],
                    lut: &[u32; RANS_PROB_SCALE as usize],
                ) -> Result<u8> {
                    let mut x = *state;
                    let slot = (x & 4095) as usize;
                    let entry = lut[slot];
                    let sym = (entry & 0xff) as u8;
                    let f = (entry >> 8) & 0xfff;
                    let c = entry >> 20;
                    x = f * (x >> RANS_PROB_BITS) + (slot as u32 - c);
                    if x < RANS_L {
                        let Some((word, tail)) = lane.split_first_chunk::<2>() else {
                            return Err(invalid("truncated GKV1 rANS lane stream"));
                        };
                        x = (x << 16) | u32::from(u16::from_le_bytes(*word));
                        *lane = tail;
                    }
                    *state = x;
                    Ok(sym)
                }

                let mut iter = dst.chunks_exact_mut(RANS_LANES);
                for group in iter.by_ref() {
                    for l in 0..RANS_LANES {
                        group[l] = step_lane(&mut states[l], &mut lanes[l], lut)?;
                    }
                }
                let tail = iter.into_remainder();
                for (l, byte) in tail.iter_mut().enumerate() {
                    *byte = step_lane(&mut states[l], &mut lanes[l], lut)?;
                }
                for l in 0..RANS_LANES {
                    if states[l] != RANS_L || !lanes[l].is_empty() {
                        return Err(invalid("corrupted GKV1 rANS terminal state"));
                    }
                }
                Ok(())
            })
        }
        _ => Err(invalid("unknown GKV1 rANS chunk mode")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::codec::Field;

    #[test]
    fn gpu_plan_builds_coalesced_chunks_for_all_encodings() {
        let mut off = 0usize;
        let mut fields = Vec::new();
        for (enc, len) in [
            (Encoding::Float32Matrix128, 65536 * 2),
            (Encoding::Bfloat16, 4096),
            (Encoding::Convolution20, 20 * 64),
            (Encoding::Convolution32, 32 * 64),
            (Encoding::Latent528, 135_168),
            (Encoding::Index132, 8_448),
            (Encoding::Fp8Page512, 32_768),
            (Encoding::WindowFp4, 33_792),
            (Encoding::LatentFp4, 18_432),
            (Encoding::Bytes, 512),
        ] {
            fields.push(Field {
                offset: off,
                length: len,
                encoding: enc,
            });
            off += len;
        }
        let total = off.next_multiple_of(4096);
        let plan = Plan::new(total, &fields).unwrap();
        let gplan = GpuPlan::from_plan(&plan).unwrap();
        assert_eq!(gplan.num_mats(), 2);
        assert!(gplan.num_chunks() > 0);
        assert!(gplan.max_packet() > total);
        assert!(GpuPlan::workspace_bytes_estimate_pair(&gplan, &gplan) > 3 * total as u64);
    }

    #[test]
    fn rans32_cpu_decoder_handles_rle_and_rejects_malformed_chunks() {
        let mut out = vec![0u8; 1024];
        rans32_decode_chunk(&[1u8, 0x5A], &mut out).unwrap();
        assert!(out.iter().all(|&b| b == 0x5A));
        assert!(rans32_decode_chunk(&[1u8, 0x5A, 0], &mut out).is_err());
        assert!(rans32_decode_chunk(&[9u8, 0], &mut out).is_err());
        assert!(rans32_decode_chunk(&[], &mut out).is_err());
    }

    #[test]
    fn cpu_decode_scratch_trims_oversized_buffers_after_large_restore() {
        // Build a synthetic valid GKV1/v2 RLE packet for a 3 MiB (> MAX_RETAINED_DECODE_SCRATCH)
        // unit and a 64 KiB small unit to verify oversized thread-local buffers are released on exit.
        let make_rle_pkt = |orig_bytes: usize, byte_val: u8| -> (GpuPlan, Vec<u8>, u32, u32) {
            let plan = Plan::new(
                orig_bytes,
                &[Field {
                    offset: 0,
                    length: orig_bytes,
                    encoding: Encoding::Bytes,
                }],
            )
            .unwrap();
            let gplan = GpuPlan::from_plan(&plan).unwrap();
            let nc = gplan.num_chunks();
            let mut pkt = Vec::new();
            pkt.extend_from_slice(&MAGIC.to_le_bytes());
            pkt.extend_from_slice(&VERSION.to_le_bytes());
            pkt.extend_from_slice(&(orig_bytes as u32).to_le_bytes());
            pkt.extend_from_slice(&(nc as u32).to_le_bytes());
            pkt.extend_from_slice(&0u32.to_le_bytes());
            pkt.extend_from_slice(&0u32.to_le_bytes());
            for _ in 0..nc {
                pkt.extend_from_slice(&2u32.to_le_bytes());
            }
            for _ in 0..nc {
                pkt.push(1u8);
                pkt.push(byte_val);
            }
            let sc = crc32_par(&pkt, None);
            let orig = vec![byte_val; orig_bytes];
            let oc = crc32_par(&orig, None);
            (gplan, pkt, sc, oc)
        };

        let (big_plan, big_pkt, big_sc, big_oc) = make_rle_pkt(3 << 20, 0x3C);
        let mut big_out = vec![0u8; 3 << 20];
        big_plan
            .decode_checked_cpu(&[&big_pkt], &mut big_out, big_sc, big_oc, None)
            .unwrap();
        assert!(big_out.iter().all(|&b| b == 0x3C));
        CPU_DECODE_SCRATCH.with(|cell| {
            let s = cell.borrow();
            assert_eq!(s.ans_buf.capacity(), 0, "oversized ans_buf must be released");
            assert_eq!(s.trans_buf.capacity(), 0, "oversized trans_buf must be released");
            assert!(s.pkt_buf.capacity() <= MAX_RETAINED_DECODE_SCRATCH);
        });

        let (sm_plan, sm_pkt, sm_sc, sm_oc) = make_rle_pkt(64 << 10, 0x5A);
        let mut sm_out = vec![0u8; 64 << 10];
        sm_plan
            .decode_checked_cpu(&[&sm_pkt], &mut sm_out, sm_sc, sm_oc, None)
            .unwrap();
        assert!(sm_out.iter().all(|&b| b == 0x5A));
        CPU_DECODE_SCRATCH.with(|cell| {
            let s = cell.borrow();
            assert!(s.ans_buf.capacity() >= (64 << 10));
            assert!(s.ans_buf.capacity() <= MAX_RETAINED_DECODE_SCRATCH);
            assert!(s.trans_buf.capacity() <= MAX_RETAINED_DECODE_SCRATCH);
        });
    }
}
