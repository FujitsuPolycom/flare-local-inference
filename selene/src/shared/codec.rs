// SPDX-License-Identifier: Apache-2.0

//! Lossless packets for one rank's share of a cache page or continuation row.
//!
//! Each packet partitions the padded rank share into independent frames:
//! - 128x128 float32 recurrent state matrices use a bit-exact 64-pivot low-rank
//!   predictor with 4-lane residual low-bit packing and standard Zstandard
//!   compression on the residual bit-widths.
//! - Other typed fields apply byte-lane shuffle or compound page value/scale
//!   splitting followed by standard Zstandard compression.
//! - Any frame whose encoded payload does not shrink is stored literally.

use super::packed::{crc32, crc32_chunks};
use rayon::prelude::*;
use std::cell::RefCell;
use std::io::{Error, ErrorKind, Result};

pub use super::packed::Checksum;

const MAGIC: u32 = u32::from_le_bytes(*b"ZKV1");
const VERSION: u32 = 1;
const HEADER: usize = 16;
const ENTRY: usize = 4;
const RAW_FLAG: u32 = 1 << 31;
const MAX_UNIT: usize = 256 << 20;
const MAX_FRAMES: usize = 65536;
pub(crate) const MATRIX_BYTES: usize = 65536;
pub(crate) const MATRIX_ELEMS: usize = 16384;
const MATRIX_GROUP: usize = 4;
const MATRIX_MAX_LOW_BYTES: usize = 3968 * 4 * 4;
const MAX_BYTE_FRAME: usize = 65_536;
const MAX_FRAME_BUF: usize = 135_168;
const PAGE_SCALE_TMP: usize = 4096;
const PAR_CRC_MIN: usize = 1 << 20;
const PAR_CRC_CHUNK: usize = 1 << 20;

pub fn cpu_pool() -> Result<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<std::result::Result<rayon::ThreadPool, String>> =
        std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).clamp(1, 24))
            .unwrap_or(4);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("flare-kv-codec-{i}"))
            .start_handler(|_| {
                #[cfg(unix)]
                // SAFETY: lowering codec worker thread priority keeps host encode/decode
                // from preempting the model worker or CUDA launch threads.
                unsafe {
                    libc::nice(10);
                }
            })
            .build()
            .map_err(|e| format!("cannot start KV codec pool: {e}"))
    })
    .as_ref()
    .map_err(|e| Error::other(e.clone()))
}

pub(crate) fn crc32_par(bytes: &[u8], pool: Option<&rayon::ThreadPool>) -> u32 {
    let Some(pool) = pool.filter(|p| p.current_num_threads() > 1 && bytes.len() >= PAR_CRC_MIN)
    else {
        return crc32(bytes);
    };
    let parts: Vec<crc32fast::Hasher> = pool.install(|| {
        bytes
            .par_chunks(PAR_CRC_CHUNK)
            .map(|chunk| {
                let mut h = crc32fast::Hasher::new();
                h.update(chunk);
                h
            })
            .collect()
    });
    let mut combined = crc32fast::Hasher::new();
    for part in parts {
        combined.combine(&part);
    }
    combined.finalize()
}

pub(crate) fn crc32_chunks_par(chunks: &[&[u8]], pool: Option<&rayon::ThreadPool>) -> u32 {
    if chunks.len() == 1 {
        return crc32_par(chunks[0], pool);
    }
    let total: usize = chunks.iter().map(|c| c.len()).sum();
    let Some(pool) = pool.filter(|p| p.current_num_threads() > 1 && total >= PAR_CRC_MIN) else {
        return crc32_chunks(chunks);
    };
    let parts: Vec<crc32fast::Hasher> = pool.install(|| {
        chunks
            .par_iter()
            .map(|chunk| {
                let mut h = crc32fast::Hasher::new();
                h.update(chunk);
                h
            })
            .collect()
    });
    let mut combined = crc32fast::Hasher::new();
    for part in parts {
        combined.combine(&part);
    }
    combined.finalize()
}

// Public Zstandard parameters tuned for 16 KiB per-block Huffman adaptation and L1-resident hashing.
const ZSTD_LEVEL: i32 = 1;
const ZSTD_WINDOW_LOG: u32 = 14;
const ZSTD_HASH_LOG: u32 = 10;
const ZSTD_MIN_MATCH: u32 = 7;

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

/// Each field describes a contiguous interval of the padded rank share.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub offset: usize,
    pub length: usize,
    pub encoding: Encoding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    Bytes,
    Bfloat16,
    Float32Matrix128,
    Convolution20,
    Convolution32,
    Latent528,
    Index132,
    Fp8Page512,
    WindowFp4,
    LatentFp4,
}

/// Physical block geometry for compound KV page encodings (shared by CPU and GPU codecs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageSpec {
    pub kind: u32,
    pub block: usize,
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
    pub scale_w: u32,
    pub planar: i32,
}

impl Encoding {
    pub fn page_spec(self) -> Option<PageSpec> {
        Some(match self {
            Self::Latent528 => PageSpec { kind: 40, block: 135_168, rows: 256, cols: 512, stride: 528, scale_w: 4, planar: 0 },
            Self::Index132 => PageSpec { kind: 41, block: 8_448, rows: 64, cols: 128, stride: 128, scale_w: 4, planar: 1 },
            Self::Fp8Page512 => PageSpec { kind: 42, block: 32_768, rows: 64, cols: 512, stride: 512, scale_w: 1, planar: 0 },
            Self::WindowFp4 => PageSpec { kind: 45, block: 33_792, rows: 64, cols: 512, stride: 528, scale_w: 1, planar: 0 },
            Self::LatentFp4 => PageSpec { kind: 47, block: 18_432, rows: 64, cols: 256, stride: 288, scale_w: 1, planar: 0 },
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transform {
    Raw,
    Shuffle(u32),
    PageSplit(u32),
    MatrixGroup(usize),
}

#[derive(Clone, Copy, Debug)]
pub struct Frame {
    offset: usize,
    raw_bytes: usize,
    transform: Transform,
}

impl Frame {
    fn raw_bytes(&self) -> usize {
        self.raw_bytes
    }
}

unsafe extern "C" {
    fn kv_predict_pack_f32_128(
        raw_mat: *const u8,
        widths_out: *mut u8,
        low_out: *mut u8,
        lane_u32s: *mut u16,
    ) -> usize;
    fn kv_unpack_restore_f32_128(
        raw_mat: *mut u8,
        widths_in: *const u8,
        low_in: *const u8,
        low_len: usize,
        lane_u32s: *const u16,
    ) -> i32;
    fn kv_byte_shuffle(dst: *mut u8, src: *const u8, n: usize, width: u32, decode: bool);
    fn kv_page_split_shuffle(
        dst: *mut u8,
        src: *const u8,
        n: usize,
        kind: u32,
        decode: bool,
        tmp: *mut u8,
    ) -> i32;
}

pub(crate) fn unpack_restore_f32_128(
    raw_mat: &mut [u8],
    widths_in: &[u8],
    low_in: &[u8],
    lane_u32s: &[u16; 4],
) -> Result<()> {
    if raw_mat.len() != MATRIX_BYTES || widths_in.len() != MATRIX_ELEMS {
        return Err(invalid("matrix decode buffer length mismatch"));
    }
    let rc = unsafe {
        kv_unpack_restore_f32_128(
            raw_mat.as_mut_ptr(),
            widths_in.as_ptr(),
            low_in.as_ptr(),
            low_in.len(),
            lane_u32s.as_ptr(),
        )
    };
    if rc < 0 {
        return Err(invalid("invalid residual data in matrix frame"));
    }
    Ok(())
}

pub(crate) fn byte_shuffle_decode(dst: &mut [u8], src: &[u8], width: u32) {
    assert_eq!(dst.len(), src.len());
    assert!(width > 0 && src.len().is_multiple_of(width as usize));
    unsafe {
        kv_byte_shuffle(dst.as_mut_ptr(), src.as_ptr(), src.len(), width, true);
    }
}

pub(crate) fn page_split_decode(
    dst: &mut [u8],
    src: &[u8],
    kind: u32,
    tmp: &mut [u8],
) -> Result<()> {
    if dst.len() != src.len() || tmp.len() < PAGE_SCALE_TMP {
        return Err(invalid("page split decode buffer length mismatch"));
    }
    let ok = unsafe {
        kv_page_split_shuffle(
            dst.as_mut_ptr(),
            src.as_ptr(),
            src.len(),
            kind,
            true,
            tmp.as_mut_ptr(),
        )
    };
    if ok < 0 {
        return Err(invalid("page split inverse transform failed"));
    }
    Ok(())
}

struct WorkerState {
    cctx: zstd_safe::CCtx<'static>,
    dctx: zstd_safe::DCtx<'static>,
    widths: Vec<u8>,
    low: Vec<u8>,
    shuffle: Vec<u8>,
    comp: Vec<u8>,
    tmp: Vec<u8>,
}

impl WorkerState {
    fn new() -> Self {
        let mut cctx = zstd_safe::CCtx::create();
        use zstd_safe::CParameter;
        cctx.set_parameter(CParameter::CompressionLevel(ZSTD_LEVEL))
            .expect("zstd compression level");
        cctx.set_parameter(CParameter::WindowLog(ZSTD_WINDOW_LOG))
            .expect("zstd window log");
        cctx.set_parameter(CParameter::HashLog(ZSTD_HASH_LOG))
            .expect("zstd hash log");
        cctx.set_parameter(CParameter::MinMatch(ZSTD_MIN_MATCH))
            .expect("zstd min match");
        cctx.set_parameter(CParameter::ChecksumFlag(false))
            .expect("zstd checksum flag");
        cctx.set_parameter(CParameter::ContentSizeFlag(false))
            .expect("zstd content size flag");
        let dctx = zstd_safe::DCtx::create();
        Self {
            cctx,
            dctx,
            widths: vec![0; MATRIX_GROUP * MATRIX_ELEMS],
            low: vec![0; MATRIX_GROUP * (8 + MATRIX_MAX_LOW_BYTES)],
            shuffle: vec![0; MAX_FRAME_BUF],
            comp: vec![0; zstd_safe::compress_bound(MAX_FRAME_BUF)],
            tmp: vec![0; PAGE_SCALE_TMP],
        }
    }

    fn compress(&mut self, src: &[u8]) -> Result<usize> {
        self.cctx
            .compress2(&mut self.comp, src)
            .map_err(|_| invalid("zstd frame compression failed"))
    }

    fn decompress(&mut self, dst: &mut [u8], src: &[u8]) -> Result<()> {
        let n = self
            .dctx
            .decompress(dst, src)
            .map_err(|_| invalid("invalid zstd data in codec frame"))?;
        if n != dst.len() {
            return Err(invalid("zstd frame decompressed to unexpected length"));
        }
        Ok(())
    }
}

thread_local! {
    static WORKER: RefCell<Option<WorkerState>> = const { RefCell::new(None) };
}

fn with_worker<T>(f: impl FnOnce(&mut WorkerState) -> T) -> T {
    WORKER.with(|cell| {
        let mut slot = cell.borrow_mut();
        let state = slot.get_or_insert_with(WorkerState::new);
        f(state)
    })
}

fn encode_frame(state: &mut WorkerState, frame: &Frame, source: &[u8], out: &mut [u8]) -> Result<u32> {
    let raw = &source[frame.offset..frame.offset + frame.raw_bytes];
    debug_assert_eq!(out.len(), frame.raw_bytes);
    match frame.transform {
        Transform::MatrixGroup(count) => {
            let mut lpos = 0usize;
            for i in 0..count {
                let mat = &raw[i * MATRIX_BYTES..(i + 1) * MATRIX_BYTES];
                let mut lanes = [0u16; 4];
                // widths covers MATRIX_GROUP * MATRIX_ELEMS; low covers MATRIX_GROUP * (8 + MATRIX_MAX_LOW_BYTES).
                let lb = unsafe {
                    kv_predict_pack_f32_128(
                        mat.as_ptr(),
                        state.widths.as_mut_ptr().add(i * MATRIX_ELEMS),
                        state.low.as_mut_ptr().add(lpos + 8),
                        lanes.as_mut_ptr(),
                    )
                };
                for (l, words) in lanes.iter().enumerate() {
                    state.low[lpos + 2 * l..lpos + 2 * l + 2].copy_from_slice(&words.to_le_bytes());
                }
                lpos += 8 + lb;
            }
            let wlen = count * MATRIX_ELEMS;
            let zlen = state
                .cctx
                .compress2(&mut state.comp, &state.widths[..wlen])
                .map_err(|_| invalid("zstd matrix width compression failed"))?;
            let total = 4 + zlen + lpos;
            if total >= frame.raw_bytes {
                out.copy_from_slice(raw);
                return Ok(RAW_FLAG | frame.raw_bytes as u32);
            }
            out[..4].copy_from_slice(&(zlen as u32).to_le_bytes());
            out[4..4 + zlen].copy_from_slice(&state.comp[..zlen]);
            out[4 + zlen..total].copy_from_slice(&state.low[..lpos]);
            Ok(total as u32)
        }
        Transform::Raw => {
            let zlen = state.compress(raw)?;
            if zlen >= frame.raw_bytes {
                out.copy_from_slice(raw);
                return Ok(RAW_FLAG | frame.raw_bytes as u32);
            }
            out[..zlen].copy_from_slice(&state.comp[..zlen]);
            Ok(zlen as u32)
        }
        Transform::Shuffle(width) => {
            let n = frame.raw_bytes;
            // shuffle holds MAX_BYTE_FRAME >= n; raw has length n.
            unsafe {
                kv_byte_shuffle(state.shuffle.as_mut_ptr(), raw.as_ptr(), n, width, false);
            }
            let zlen = state
                .cctx
                .compress2(&mut state.comp, &state.shuffle[..n])
                .map_err(|_| invalid("zstd frame compression failed"))?;
            if zlen >= n {
                out.copy_from_slice(raw);
                return Ok(RAW_FLAG | n as u32);
            }
            out[..zlen].copy_from_slice(&state.comp[..zlen]);
            Ok(zlen as u32)
        }
        Transform::PageSplit(kind) => {
            let n = frame.raw_bytes;
            // shuffle holds MAX_BYTE_FRAME >= n; tmp holds PAGE_SCALE_TMP.
            let ok = unsafe {
                kv_page_split_shuffle(
                    state.shuffle.as_mut_ptr(),
                    raw.as_ptr(),
                    n,
                    kind,
                    false,
                    state.tmp.as_mut_ptr(),
                )
            };
            if ok < 0 {
                return Err(invalid("page split transform failed"));
            }
            let zlen = state
                .cctx
                .compress2(&mut state.comp, &state.shuffle[..n])
                .map_err(|_| invalid("zstd frame compression failed"))?;
            if zlen >= n {
                out.copy_from_slice(raw);
                return Ok(RAW_FLAG | n as u32);
            }
            out[..zlen].copy_from_slice(&state.comp[..zlen]);
            Ok(zlen as u32)
        }
    }
}

fn decode_frame(
    state: &mut WorkerState,
    frame: &Frame,
    is_raw: bool,
    payload: &[u8],
    dst: &mut [u8],
) -> Result<()> {
    debug_assert_eq!(dst.len(), frame.raw_bytes);
    if is_raw {
        if payload.len() != frame.raw_bytes {
            return Err(invalid("raw codec frame length mismatch"));
        }
        dst.copy_from_slice(payload);
        return Ok(());
    }
    match frame.transform {
        Transform::MatrixGroup(count) => {
            if payload.len() < 4 {
                return Err(invalid("short matrix codec frame"));
            }
            let zlen = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
            let Some(rest) = payload.get(4..) else {
                return Err(invalid("short matrix codec frame"));
            };
            let Some((zwidths, low_all)) = rest.split_at_checked(zlen) else {
                return Err(invalid("matrix width stream exceeds frame"));
            };
            let wlen = count * MATRIX_ELEMS;
            let n = state
                .dctx
                .decompress(&mut state.widths[..wlen], zwidths)
                .map_err(|_| invalid("invalid zstd data in matrix width stream"))?;
            if n != wlen {
                return Err(invalid("matrix width stream length mismatch"));
            }
            let mut lpos = 0usize;
            for i in 0..count {
                let header = low_all
                    .get(lpos..lpos + 8)
                    .ok_or_else(|| invalid("truncated matrix lane header"))?;
                let mut lanes = [0u16; 4];
                let mut lb = 0usize;
                for l in 0..4 {
                    lanes[l] = u16::from_le_bytes([header[2 * l], header[2 * l + 1]]);
                    lb += usize::from(lanes[l]) * 4;
                }
                let lane_slice = low_all
                    .get(lpos + 8..lpos + 8 + lb)
                    .ok_or_else(|| invalid("truncated matrix low-bit lanes"))?;
                let mat = &mut dst[i * MATRIX_BYTES..(i + 1) * MATRIX_BYTES];
                // widths slice is MATRIX_ELEMS bytes; lane_slice has exact length lb.
                let rc = unsafe {
                    kv_unpack_restore_f32_128(
                        mat.as_mut_ptr(),
                        state.widths.as_ptr().add(i * MATRIX_ELEMS),
                        lane_slice.as_ptr(),
                        lb,
                        lanes.as_ptr(),
                    )
                };
                if rc < 0 {
                    return Err(invalid("invalid residual data in matrix frame"));
                }
                lpos += 8 + lb;
            }
            if lpos != low_all.len() {
                return Err(invalid("trailing bytes in matrix codec frame"));
            }
            Ok(())
        }
        Transform::Raw => state.decompress(dst, payload),
        Transform::Shuffle(width) => {
            let n = frame.raw_bytes;
            let got = state
                .dctx
                .decompress(&mut state.shuffle[..n], payload)
                .map_err(|_| invalid("invalid zstd data in codec frame"))?;
            if got != n {
                return Err(invalid("zstd frame decompressed to unexpected length"));
            }
            // shuffle holds n bytes; dst has length n.
            unsafe {
                kv_byte_shuffle(dst.as_mut_ptr(), state.shuffle.as_ptr(), n, width, true);
            }
            Ok(())
        }
        Transform::PageSplit(kind) => {
            let n = frame.raw_bytes;
            let got = state
                .dctx
                .decompress(&mut state.shuffle[..n], payload)
                .map_err(|_| invalid("invalid zstd data in codec frame"))?;
            if got != n {
                return Err(invalid("zstd frame decompressed to unexpected length"));
            }
            // shuffle holds n bytes; dst has length n; tmp holds PAGE_SCALE_TMP.
            let ok = unsafe {
                kv_page_split_shuffle(
                    dst.as_mut_ptr(),
                    state.shuffle.as_ptr(),
                    n,
                    kind,
                    true,
                    state.tmp.as_mut_ptr(),
                )
            };
            if ok < 0 {
                return Err(invalid("page split inverse transform failed"));
            }
            Ok(())
        }
    }
}

#[derive(Clone, Debug)]
pub struct Plan {
    fields: Vec<Field>,
    frames: Vec<Frame>,
    original_bytes: usize,
}

impl Plan {
    /// Padding and untyped intervals are included byte for byte.
    pub fn new(original_bytes: usize, fields: &[Field]) -> Result<Self> {
        if original_bytes == 0 || original_bytes > MAX_UNIT {
            return Err(invalid("codec unit size outside supported range"));
        }
        let mut plan = Self {
            fields: fields.to_vec(),
            frames: Vec::new(),
            original_bytes,
        };
        let mut end = 0;
        for field in fields {
            if field.offset < end
                || field.length == 0
                || field
                    .offset
                    .checked_add(field.length)
                    .is_none_or(|n| n > original_bytes)
            {
                return Err(invalid("invalid codec field extent"));
            }
            plan.bytes(end, field.offset - end)?;
            plan.field(*field)?;
            end = field.offset + field.length;
        }
        plan.bytes(end, original_bytes - end)?;
        Ok(plan)
    }

    fn push(&mut self, offset: usize, raw_bytes: usize, transform: Transform) -> Result<()> {
        if self.frames.len() >= MAX_FRAMES {
            return Err(invalid("too many codec frames"));
        }
        self.frames.push(Frame {
            offset,
            raw_bytes,
            transform,
        });
        Ok(())
    }

    fn bytes(&mut self, offset: usize, size: usize) -> Result<()> {
        for start in (0..size).step_by(MAX_BYTE_FRAME) {
            let n = (size - start).min(MAX_BYTE_FRAME);
            self.push(offset + start, n, Transform::Raw)?;
        }
        Ok(())
    }

    fn field(&mut self, field: Field) -> Result<()> {
        use Encoding::*;
        let Field {
            offset,
            length,
            encoding,
        } = field;
        let multiple = |size: usize| {
            if length.is_multiple_of(size) {
                Ok(())
            } else {
                Err(invalid("field length disagrees with codec geometry"))
            }
        };
        match encoding {
            Bytes => self.bytes(offset, length)?,
            Float32Matrix128 => {
                multiple(MATRIX_BYTES)?;
                if offset % 4 != 0 {
                    return Err(invalid("unaligned codec matrix"));
                }
                let total = length / MATRIX_BYTES;
                for m0 in (0..total).step_by(MATRIX_GROUP) {
                    let count = (total - m0).min(MATRIX_GROUP);
                    self.push(
                        offset + m0 * MATRIX_BYTES,
                        count * MATRIX_BYTES,
                        Transform::MatrixGroup(count),
                    )?;
                }
            }
            Bfloat16 | Convolution20 | Convolution32 => {
                let width = match encoding {
                    Bfloat16 => 2usize,
                    Convolution20 => 20,
                    _ => 32,
                };
                multiple(width)?;
                let chunk = (MAX_BYTE_FRAME / width) * width;
                for start in (0..length).step_by(chunk) {
                    let n = (length - start).min(chunk);
                    self.push(offset + start, n, Transform::Shuffle(width as u32))?;
                }
            }
            Latent528 | Index132 | Fp8Page512 | WindowFp4 | LatentFp4 => {
                let spec = encoding.page_spec().unwrap();
                multiple(spec.block)?;
                let chunk = (MAX_BYTE_FRAME / spec.block).max(1) * spec.block;
                for start in (0..length).step_by(chunk) {
                    let n = (length - start).min(chunk);
                    self.push(offset + start, n, Transform::PageSplit(spec.kind))?;
                }
            }
        }
        Ok(())
    }

    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    pub fn original_bytes(&self) -> usize {
        self.original_bytes
    }

    pub fn metadata(&self) -> usize {
        HEADER + self.frames.len() * ENTRY
    }

    pub fn max_packet(&self) -> usize {
        self.metadata() + self.original_bytes
    }

    fn entries(&self, packet: &[u8], total: usize) -> Result<Vec<(bool, std::ops::Range<usize>)>> {
        let word = |p| u32::from_le_bytes(packet[p..p + 4].try_into().unwrap());
        if packet.len() < self.metadata()
            || total < self.metadata()
            || total > self.max_packet()
            || word(0) != MAGIC
            || word(4) != VERSION
            || word(8) as usize != self.frames.len()
            || word(12) as usize != self.original_bytes
        {
            return Err(invalid("invalid codec packet header"));
        }
        let mut pos = self.metadata();
        let mut ranges = Vec::with_capacity(self.frames.len());
        for (frame, entry) in self
            .frames
            .iter()
            .zip(packet[HEADER..self.metadata()].chunks_exact(ENTRY))
        {
            let word = u32::from_le_bytes(entry.try_into().unwrap());
            let is_raw = (word & RAW_FLAG) != 0;
            let len = (word & !RAW_FLAG) as usize;
            if is_raw {
                if len != frame.raw_bytes() || len > total - pos {
                    return Err(invalid("invalid raw codec frame extent"));
                }
            } else if len == 0 || len >= frame.raw_bytes() || len > total - pos {
                return Err(invalid("invalid codec frame payload extent"));
            }
            ranges.push((is_raw, pos..pos + len));
            pos += len;
        }
        if pos != total {
            return Err(invalid("trailing codec packet bytes"));
        }
        Ok(ranges)
    }

    fn encode_in(
        &self,
        source: &[u8],
        out: &mut [u8],
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<usize> {
        if source.len() != self.original_bytes {
            return Err(invalid("codec input size mismatch"));
        }
        if out.len() < self.max_packet() {
            return Err(invalid("codec packet buffer too small"));
        }
        for (chunk, word) in out[..HEADER].chunks_exact_mut(4).zip([
            MAGIC,
            VERSION,
            self.frames.len() as u32,
            source.len() as u32,
        ]) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        let (directory, payload) = out.split_at_mut(self.metadata());
        let entries = &mut directory[HEADER..];
        let payload_base = payload.as_mut_ptr() as usize;
        let run_one = |(frame, entry): (&Frame, &mut [u8])| -> Result<()> {
            // Frames partition 0..original_bytes in ascending offset order, so each frame
            // owns the disjoint slice [frame.offset .. frame.offset + frame.raw_bytes].
            let slot = unsafe {
                std::slice::from_raw_parts_mut(
                    (payload_base + frame.offset) as *mut u8,
                    frame.raw_bytes,
                )
            };
            let word = with_worker(|state| encode_frame(state, frame, source, slot))?;
            entry.copy_from_slice(&word.to_le_bytes());
            Ok(())
        };
        if let Some(pool) =
            pool.filter(|p| p.current_num_threads() > 1 && self.frames.len() > 1)
        {
            pool.install(|| {
                self.frames
                    .par_iter()
                    .zip(entries.par_chunks_exact_mut(ENTRY))
                    .try_for_each(run_one)
            })?;
        } else {
            self.frames
                .iter()
                .zip(entries.chunks_exact_mut(ENTRY))
                .try_for_each(run_one)?;
        }
        let mut pos = 0usize;
        for (frame, entry) in self.frames.iter().zip(entries.chunks_exact(ENTRY)) {
            let word = u32::from_le_bytes(entry.try_into().unwrap());
            let len = (word & !RAW_FLAG) as usize;
            if frame.offset != pos {
                payload.copy_within(frame.offset..frame.offset + len, pos);
            }
            pos += len;
        }
        Ok(self.metadata() + pos)
    }

    fn decode_chunks(
        &self,
        chunks: &[&[u8]],
        dst: &mut [u8],
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<()> {
        if dst.len() != self.original_bytes {
            return Err(invalid("codec destination size mismatch"));
        }
        let input = Source::new(chunks)?;
        let ranges = self.entries(&input.range(0..self.metadata())?, input.len)?;
        let dst_base = dst.as_mut_ptr() as usize;
        let run_one = |(frame, (is_raw, range)): (&Frame, (bool, std::ops::Range<usize>))| -> Result<()> {
            let bytes = input.range(range)?;
            // Frames partition 0..original_bytes into disjoint destination slices.
            let target = unsafe {
                std::slice::from_raw_parts_mut(
                    (dst_base + frame.offset) as *mut u8,
                    frame.raw_bytes,
                )
            };
            with_worker(|state| decode_frame(state, frame, is_raw, &bytes, target))
        };
        if let Some(pool) =
            pool.filter(|p| p.current_num_threads() > 1 && self.frames.len() > 1)
        {
            pool.install(|| {
                self.frames
                    .par_iter()
                    .zip(ranges)
                    .try_for_each(run_one)
            })
        } else {
            self.frames.iter().zip(ranges).try_for_each(run_one)
        }
    }

    /// Encode a rank share during VRAM -> DRAM capture into `out` (`ZKV1` packet or `Literal` raw).
    /// Returns `(PackedUnit, bytes_written_to_out)`.
    pub fn encode_capture(
        &self,
        mut unit: super::wire::PackedUnit,
        profile: &[u8; 32],
        source: &[u8],
        out: &mut [u8],
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<(super::wire::PackedUnit, usize)> {
        if source.len() != self.original_bytes {
            return Err(invalid("codec input size mismatch"));
        }
        if out.len() < self.max_packet() {
            return Err(invalid("codec capture buffer too small"));
        }
        let used = self.encode_in(source, out, pool)?;
        unit.original_length = source.len() as u64;
        if used < source.len() {
            unit.representation = super::wire::packed_unit::Representation::Packet as i32;
            unit.profile = profile.to_vec();
            unit.stored_crc = crc32_par(&out[..used], pool);
            unit.original_crc = crc32_par(source, pool);
            unit.stored_length = used as u64;
            unit.raw_xxh3 = 0;
            Ok((unit, used))
        } else {
            out[..source.len()].copy_from_slice(source);
            unit.representation = super::wire::packed_unit::Representation::Literal as i32;
            unit.profile.clear();
            unit.stored_length = source.len() as u64;
            unit.stored_crc = 0;
            unit.original_crc = 0;
            unit.raw_xxh3 = super::record::checksum(source);
            Ok((unit, source.len()))
        }
    }

    /// Check the stored CRC before decoding and the original CRC before releasing the output.
    pub fn decode_checked(
        &self,
        chunks: &[&[u8]],
        dst: &mut [u8],
        stored_crc: u32,
        original_crc: u32,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<()> {
        if crc32_chunks_par(chunks, pool) != stored_crc {
            return Err(Error::new(ErrorKind::InvalidData, Checksum::Stored));
        }
        self.decode_chunks(chunks, dst, pool)?;
        if crc32_par(dst, pool) != original_crc {
            return Err(Error::new(ErrorKind::InvalidData, Checksum::Original));
        }
        Ok(())
    }
}

struct Source<'a> {
    chunks: &'a [&'a [u8]],
    ends: Vec<usize>,
    len: usize,
}

impl<'a> Source<'a> {
    fn new(chunks: &'a [&'a [u8]]) -> Result<Self> {
        let mut len = 0usize;
        let ends = chunks
            .iter()
            .map(|s| {
                len = len
                    .checked_add(s.len())
                    .ok_or_else(|| invalid("packet length overflow"))?;
                Ok(len)
            })
            .collect::<Result<_>>()?;
        Ok(Self { chunks, ends, len })
    }

    fn range(&self, range: std::ops::Range<usize>) -> Result<std::borrow::Cow<'a, [u8]>> {
        if range.end > self.len || range.is_empty() {
            return Err(invalid("short codec packet"));
        }
        let i = self.ends.partition_point(|&end| end <= range.start);
        let start = range.start - (self.ends[i] - self.chunks[i].len());
        let first = &self.chunks[i][start..];
        if first.len() >= range.len() {
            return Ok((&first[..range.len()]).into());
        }
        let mut frame = Vec::with_capacity(range.len());
        frame.extend_from_slice(first);
        for chunk in &self.chunks[i + 1..] {
            let n = chunk.len().min(range.len() - frame.len());
            frame.extend_from_slice(&chunk[..n]);
            if frame.len() == range.len() {
                break;
            }
        }
        Ok(frame.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" {
        fn kv_predict_pack_f32_128_scalar(
            raw_mat: *const u8,
            widths_out: *mut u8,
            low_out: *mut u8,
            lane_u32s: *mut u16,
        ) -> usize;
        fn kv_unpack_restore_f32_128_scalar(
            raw_mat: *mut u8,
            widths_in: *const u8,
            low_in: *const u8,
            low_len: usize,
            lane_u32s: *const u16,
        ) -> i32;
    }

    fn encode_checked_packet(
        plan: &Plan,
        raw: &[u8],
        dst: &mut [u8],
        pool: Option<&rayon::ThreadPool>,
    ) -> (super::super::wire::PackedUnit, Vec<u8>) {
        let mut out = vec![0u8; plan.max_packet()];
        let (unit, used) = plan
            .encode_capture(Default::default(), &[7; 32], raw, &mut out, pool)
            .unwrap();
        out.truncate(used);
        if unit.representation == super::super::wire::packed_unit::Representation::Packet as i32 {
            plan.decode_checked(&[&out], dst, unit.stored_crc, unit.original_crc, pool)
                .unwrap();
        } else {
            dst.copy_from_slice(&out);
        }
        assert_eq!(dst, raw);
        (unit, out)
    }

    #[test]
    fn fragmented_packet_checks_both_crcs_and_rejects_corruption_and_truncation() {
        let raw: Vec<u8> = (0..4096).map(|j| ((j / 8) % 2) as u8).collect();
        let plan = Plan::new(raw.len(), &[]).unwrap();
        let mut dst = vec![0u8; raw.len()];
        let (_, packet) = encode_checked_packet(&plan, &raw, &mut dst, None);
        assert!(packet.len() < raw.len());
        let stored = crc32(&packet);
        let original = crc32(&raw);
        let spans = packet.chunks(19).collect::<Vec<_>>();
        dst.fill(0xa5);
        plan.decode_checked(&spans, &mut dst, stored, original, None)
            .unwrap();
        assert_eq!(dst, raw);
        assert!(plan
            .decode_checked(&spans, &mut dst, stored ^ 1, original, None)
            .is_err());
        assert!(plan
            .decode_checked(&spans, &mut dst, stored, original ^ 1, None)
            .is_err());

        let mut damaged = packet.clone();
        damaged[plan.metadata()] ^= 0xff;
        assert!(plan
            .decode_checked(&[&damaged], &mut dst, crc32(&damaged), original, None)
            .is_err());
        for cut in 0..packet.len() {
            let truncated = &packet[..cut];
            assert!(plan
                .decode_checked(&[truncated], &mut dst, crc32(truncated), original, None)
                .is_err());
        }
    }

    #[test]
    fn partial_sizes_restore_only_requested_bytes() {
        for size in [1, 255, 256, 257, 1023, 4095, 4097, 16385, 65535, 65537, 262_145] {
            let raw: Vec<u8> = (0..size).map(|i| ((i / 3 + i / 29) % 7) as u8).collect();
            let plan = Plan::new(size, &[]).unwrap();
            let mut guarded = vec![0xff; size + 2];
            let (unit, packet) = encode_checked_packet(&plan, &raw, &mut guarded[1..=size], None);
            guarded[1..=size].fill(0xa5);
            if unit.representation == super::super::wire::packed_unit::Representation::Packet as i32 {
                plan.decode_checked(
                    &[&packet],
                    &mut guarded[1..=size],
                    crc32(&packet),
                    crc32(&raw),
                    None,
                )
                .unwrap();
            } else {
                guarded[1..=size].copy_from_slice(&packet);
            }
            assert_eq!(&guarded[1..=size], raw);
            assert_eq!((guarded[0], guarded[size + 1]), (0xff, 0xff));
        }
    }

    #[test]
    fn matrix_simd_matches_scalar_across_edge_cases_and_rejects_malformed_lanes() {
        let mut cases = Vec::new();
        // Structured floats with pivot ties and zero pivots.
        let mut tie_case = Vec::with_capacity(MATRIX_BYTES);
        for col in 0..128 {
            for row in 0..128 {
                let word = match row % 4 {
                    0 => 0,
                    1 => 0x3f80_0000 | (((col & 1) as u32) << 31),
                    2 => 1u32 << ((row + col) % 32),
                    _ => (1u32 << ((row + col) % 32)).wrapping_sub(1),
                };
                tie_case.extend_from_slice(&word.to_le_bytes());
            }
        }
        cases.push(tie_case);
        // Subnormals, -0, NaNs, infinities, and normal floats.
        let mut special = Vec::with_capacity(MATRIX_BYTES);
        let mut rng = 0xdead_beefu32;
        for i in 0..MATRIX_ELEMS {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let word = match i % 7 {
                0 => 0u32,
                1 => 0x8000_0000u32,
                2 => 1u32 << (i % 23),
                3 => 0x7f80_0000u32,
                4 => 0x7fc0_0001u32 | (rng & 0x3fffff),
                5 => (((rng & 0xffff) as i32 - 32768) as f32 / 32768.0).to_bits(),
                _ => rng,
            };
            special.extend_from_slice(&word.to_le_bytes());
        }
        cases.push(special);

        for raw in cases {
            let mut w_simd = vec![0u8; MATRIX_ELEMS];
            let mut l_simd = vec![0u8; MATRIX_MAX_LOW_BYTES];
            let mut lanes_simd = [0u16; 4];
            let n_simd = unsafe {
                kv_predict_pack_f32_128(
                    raw.as_ptr(),
                    w_simd.as_mut_ptr(),
                    l_simd.as_mut_ptr(),
                    lanes_simd.as_mut_ptr(),
                )
            };
            let mut w_ref = vec![0u8; MATRIX_ELEMS];
            let mut l_ref = vec![0u8; MATRIX_MAX_LOW_BYTES];
            let mut lanes_ref = [0u16; 4];
            let n_ref = unsafe {
                kv_predict_pack_f32_128_scalar(
                    raw.as_ptr(),
                    w_ref.as_mut_ptr(),
                    l_ref.as_mut_ptr(),
                    lanes_ref.as_mut_ptr(),
                )
            };
            assert_eq!(lanes_simd, lanes_ref);
            assert_eq!(n_simd, n_ref);
            assert_eq!(w_simd, w_ref);
            assert_eq!(&l_simd[..n_simd], &l_ref[..n_ref]);

            let mut out_simd = vec![0xa5u8; MATRIX_BYTES];
            let mut out_ref = vec![0xa5u8; MATRIX_BYTES];
            assert_eq!(
                unsafe {
                    kv_unpack_restore_f32_128(
                        out_simd.as_mut_ptr(),
                        w_simd.as_ptr(),
                        l_simd.as_ptr(),
                        n_simd,
                        lanes_simd.as_ptr(),
                    )
                },
                0
            );
            assert_eq!(
                unsafe {
                    kv_unpack_restore_f32_128_scalar(
                        out_ref.as_mut_ptr(),
                        w_ref.as_ptr(),
                        l_ref.as_ptr(),
                        n_ref,
                        lanes_ref.as_ptr(),
                    )
                },
                0
            );
            assert_eq!(out_simd, raw);
            assert_eq!(out_ref, raw);

            // Corrupt width > 32 must fail cleanly.
            let mut bad_w = w_simd.clone();
            bad_w[0] = 33;
            assert!(
                unsafe {
                    kv_unpack_restore_f32_128(
                        out_simd.as_mut_ptr(),
                        bad_w.as_ptr(),
                        l_simd.as_ptr(),
                        n_simd,
                        lanes_simd.as_ptr(),
                    )
                } < 0
            );

            // Shortening lane 0 by one word must fail cleanly.
            if lanes_simd[0] > 0 {
                let mut bad_lanes = lanes_simd;
                bad_lanes[0] -= 1;
                assert!(
                    unsafe {
                        kv_unpack_restore_f32_128(
                            out_simd.as_mut_ptr(),
                            w_simd.as_ptr(),
                            l_simd.as_ptr(),
                            n_simd - 4,
                            bad_lanes.as_ptr(),
                        )
                    } < 0
                );
            }
        }
    }

    #[test]
    fn typed_units_preserve_bits_including_padding_and_raw_fallback() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (kind, size) in [
            (Encoding::Float32Matrix128, MATRIX_BYTES * 2),
            (Encoding::Latent528, 135_168),
            (Encoding::Index132, 8_448),
            (Encoding::Fp8Page512, 32_768),
            (Encoding::Convolution20, 20_000),
            (Encoding::Convolution32, 32_768),
            (Encoding::Bfloat16, 6_144),
            (Encoding::WindowFp4, 33_792),
            (Encoding::LatentFp4, 18_432),
        ] {
            let mut state = 2026u32;
            let mut raw: Vec<u8> = (0..size + 16)
                .map(|_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    (state >> 24) as u8
                })
                .collect();
            raw[4..20].copy_from_slice(&[
                0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 128, 127, 37, 2, 192, 127,
            ]);
            let plan = Plan::new(
                raw.len(),
                &[Field {
                    offset: 4,
                    length: size,
                    encoding: kind,
                }],
            )
            .unwrap();
            let mut dst = vec![1; raw.len()];
            let (unit, packet) = encode_checked_packet(&plan, &raw, &mut dst, None);
            dst.fill(0xa5);
            let (par_unit, parallel) = encode_checked_packet(&plan, &raw, &mut dst, Some(&pool));
            assert_eq!(packet, parallel);
            assert_eq!(unit, par_unit);
            let profile = [7; 32];
            if unit.representation == super::super::wire::packed_unit::Representation::Packet as i32 {
                assert_eq!(unit.profile, profile);
                assert_eq!(unit.original_length, raw.len() as u64);
                assert_eq!(unit.stored_length, packet.len() as u64);
                assert_eq!(unit.stored_crc, crc32(&packet));
                assert_eq!(unit.original_crc, crc32(&raw));
            }
        }
    }

    #[test]
    fn large_zero_row_compresses_and_round_trips_in_parallel() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let raw = vec![0u8; 9 << 20];
        let mut dst = vec![1; raw.len()];
        let plan = Plan::new(raw.len(), &[]).unwrap();
        let (_, packet) = encode_checked_packet(&plan, &raw, &mut dst, Some(&pool));
        assert!(packet.len() < 8192);
        dst.fill(0xa5);
        plan.decode_checked(&[&packet], &mut dst, crc32(&packet), crc32(&raw), Some(&pool))
            .unwrap();
        assert_eq!(dst, raw);
    }
}
