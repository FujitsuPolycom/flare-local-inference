// SPDX-License-Identifier: Apache-2.0

//! Pure-CUDA Driver-API `GKV1/v2` checkpoint codec executed on each rank's lowest-priority
//! transfer stream with a compact 3-buffer aliased device workspace and zero external runtime
//! libraries beyond `libcuda.so.1`.

use crate::cuda::{DeviceBuffer, Event, LoadedModule, PinnedHost, Stream};
use flare_selene::shared::{
    codec::{Checksum, Encoding},
    gpu_codec::{
        CRC_BLOCKS, DeviceSpan, GpuPlan, LANE_STRIDE, MAX_PAGE_BATCH, RANS_SLOT_STRIDE,
    },
    packed::crc32_chunks,
    wire::{PackedUnit, packed_unit::Representation},
};
use std::io::{Error, ErrorKind, Result};

fn invalid(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

#[cfg(target_os = "linux")]
mod imp {
    use super::*;
    use std::ffi::{CStr, c_int, c_void};

    fn io_err(e: impl std::fmt::Display) -> Error {
        Error::other(e.to_string())
    }

    fn sync_stream(stream: &Stream, event: &Event) -> Result<()> {
        stream.record(event).map_err(io_err)?;
        event.synchronize().map_err(io_err)
    }

    fn launch_1d(
        func: usize,
        grid: u32,
        block: u32,
        smem: u32,
        stream: &Stream,
        args: &mut [*mut c_void],
        label: &str,
    ) -> Result<()> {
        unsafe {
            stream
                .launch_1d(func, grid, block, smem, args, label)
                .map_err(io_err)
        }
    }

    fn write_u64s(dst: &mut [u8], vals: &[u64]) {
        for (slot, &v) in dst.chunks_exact_mut(8).zip(vals) {
            slot.copy_from_slice(&v.to_le_bytes());
        }
    }

    fn write_u32s(dst: &mut [u8], vals: &[u32]) {
        for (slot, &v) in dst.chunks_exact_mut(4).zip(vals) {
            slot.copy_from_slice(&v.to_le_bytes());
        }
    }

    fn write_u16s(dst: &mut [u8], vals: &[u16]) {
        for (slot, &v) in dst.chunks_exact_mut(2).zip(vals) {
            slot.copy_from_slice(&v.to_le_bytes());
        }
    }

    const TRANSFORM_DESC_BYTES: usize = 48;
    const BATCH_OFF_PKT_OFFS: usize = 0;
    const BATCH_OFF_PKT_LENS: usize = MAX_PAGE_BATCH * 8;
    const BATCH_OFF_ORIG_CRC: usize = MAX_PAGE_BATCH * 16;
    const BATCH_OFF_STORED_CRC: usize = MAX_PAGE_BATCH * 20;
    const BATCH_OFF_TOTAL_BYTES: usize = MAX_PAGE_BATCH * 24;
    const BATCH_META_BYTES: usize = 2048;

    /// Pre-uploaded static plan tables on device (uploaded once per plan, zero per-call HtoD).
    struct DevicePlan {
        plan_id: u64,
        d_mat_offs: DeviceBuffer,
        d_chunk_offs: DeviceBuffer,
        d_chunk_sizes: DeviceBuffer,
        d_coal_trans: DeviceBuffer,
        d_coal_ans: DeviceBuffer,
        d_coal_len: DeviceBuffer,
        d_transforms: DeviceBuffer,
        num_transforms: usize,
    }

    impl DevicePlan {
        fn upload(
            plan: &GpuPlan,
            h_stage: &mut PinnedHost,
            stream: &Stream,
            event: &Event,
        ) -> Result<Self> {
            let d_mat_offs = stream.alloc_device(plan.num_mats().max(1) * 8).map_err(io_err)?;
            let d_chunk_offs = stream.alloc_device(plan.num_chunks().max(1) * 8).map_err(io_err)?;
            let d_chunk_sizes = stream.alloc_device(plan.num_chunks().max(1) * 8).map_err(io_err)?;
            let coal_n = plan.coalesce().len().max(1);
            let d_coal_trans = stream.alloc_device(coal_n * 8).map_err(io_err)?;
            let d_coal_ans = stream.alloc_device(coal_n * 8).map_err(io_err)?;
            let d_coal_len = stream.alloc_device(coal_n * 8).map_err(io_err)?;
            let tr_n = plan.transforms().len();
            let d_transforms = stream
                .alloc_device(tr_n.max(1) * TRANSFORM_DESC_BYTES)
                .map_err(io_err)?;

            let upload_u64 = |dst: &DeviceBuffer, vals: &[u64], stage: &mut PinnedHost| -> Result<()> {
                if vals.is_empty() {
                    return Ok(());
                }
                let bytes = vals.len() * 8;
                debug_assert!(bytes <= stage.bytes().len());
                write_u64s(&mut stage.bytes_mut()[..bytes], vals);
                unsafe {
                    stream
                        .copy_htod_async(dst.ptr(), stage.address() as _, bytes)
                        .map_err(io_err)?;
                }
                sync_stream(stream, event)
            };

            upload_u64(&d_mat_offs, plan.matrix_offsets(), h_stage)?;
            let cho: Vec<u64> = plan.chunk_offsets().iter().map(|&v| v as u64).collect();
            let chs: Vec<u64> = plan.chunk_sizes().iter().map(|&v| v as u64).collect();
            upload_u64(&d_chunk_offs, &cho, h_stage)?;
            upload_u64(&d_chunk_sizes, &chs, h_stage)?;
            let ct: Vec<u64> = plan.coalesce().iter().map(|c| c.trans_offset as u64).collect();
            let ca: Vec<u64> = plan.coalesce().iter().map(|c| c.ans_offset as u64).collect();
            let cl: Vec<u64> = plan.coalesce().iter().map(|c| c.length as u64).collect();
            upload_u64(&d_coal_trans, &ct, h_stage)?;
            upload_u64(&d_coal_ans, &ca, h_stage)?;
            upload_u64(&d_coal_len, &cl, h_stage)?;

            if tr_n > 0 {
                let bytes = tr_n * TRANSFORM_DESC_BYTES;
                debug_assert!(bytes <= h_stage.bytes().len());
                let dst = &mut h_stage.bytes_mut()[..bytes];
                for (i, op) in plan.transforms().iter().enumerate() {
                    let s = &mut dst[i * TRANSFORM_DESC_BYTES..(i + 1) * TRANSFORM_DESC_BYTES];
                    s[0..8].copy_from_slice(&(op.offset as u64).to_le_bytes());
                    s[8..16].copy_from_slice(&(op.length as u64).to_le_bytes());
                    let (kind, block, rows, cols, stride, scale_w, planar) = match op.encoding {
                        Encoding::Bytes => (0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32),
                        Encoding::Bfloat16 => (1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32),
                        enc => {
                            let spec = enc.page_spec().unwrap();
                            (
                                2u32,
                                spec.block as u32,
                                spec.rows as u32,
                                spec.cols as u32,
                                spec.stride as u32,
                                spec.scale_w,
                                spec.planar as u32,
                            )
                        }
                    };
                    s[16..20].copy_from_slice(&kind.to_le_bytes());
                    s[20..24].copy_from_slice(&block.to_le_bytes());
                    s[24..28].copy_from_slice(&rows.to_le_bytes());
                    s[28..32].copy_from_slice(&cols.to_le_bytes());
                    s[32..36].copy_from_slice(&stride.to_le_bytes());
                    s[36..40].copy_from_slice(&scale_w.to_le_bytes());
                    s[40..44].copy_from_slice(&planar.to_le_bytes());
                    s[44..48].copy_from_slice(&0u32.to_le_bytes());
                }
                unsafe {
                    stream
                        .copy_htod_async(d_transforms.ptr(), h_stage.address() as _, bytes)
                        .map_err(io_err)?;
                }
                sync_stream(stream, event)?;
            }

            Ok(Self {
                plan_id: plan.id(),
                d_mat_offs,
                d_chunk_offs,
                d_chunk_sizes,
                d_coal_trans,
                d_coal_ans,
                d_coal_len,
                d_transforms,
                num_transforms: tr_n,
            })
        }

        #[cfg(test)]
        fn allocated_bytes(&self) -> usize {
            self.d_mat_offs.bytes()
                + self.d_chunk_offs.bytes()
                + self.d_chunk_sizes.bytes()
                + self.d_coal_trans.bytes()
                + self.d_coal_ans.bytes()
                + self.d_coal_len.bytes()
                + self.d_transforms.bytes()
        }
    }

    #[derive(Clone, Copy)]
    struct BatchFns {
        fn_copy_batch_rel_strided: usize,
        fn_transform_batch: usize,
        fn_crc32_batch: usize,
        fn_rans_enc_batch: usize,
        fn_rans_dec_batch: usize,
        fn_build_plan_batch: usize,
        fn_compact_pkt_batch: usize,
    }

    pub struct GpuRankCodec {
        _module: LoadedModule,
        fn_pred_enc: usize,
        fn_pred_dec: usize,
        fn_pack: usize,
        fn_unpack: usize,
        fn_shuffle: usize,
        fn_page_split: usize,
        fn_copy_batch: usize,
        fn_copy_batch_rel: usize,
        fn_crc32_stage1: usize,
        fn_crc32_finish: usize,
        fn_rans_enc: usize,
        fn_rans_dec: usize,
        fn_build_plan: usize,
        fn_compact_pkt: usize,
        batch_fns: Option<BatchFns>,
        page_batch_cap: usize,
        // Compact 3-buffer aliased device workspace (zero steady-state allocation):
        // Encode: buf_a = z_mats/trans/rans_slots, buf_b = ans_in -> d_pkt, buf_c = gather_raw/low_pad
        // Decode: buf_a = d_pkt -> d_raw,          buf_b = ans_in,          buf_c = z_mats -> d_trans
        buf_a: DeviceBuffer,
        buf_b: DeviceBuffer,
        buf_c: DeviceBuffer,
        d_chunk_words: DeviceBuffer,
        d_chunk_pkt_offs: DeviceBuffer,
        d_lane_u32s: DeviceBuffer,
        d_lane_offs: DeviceBuffer,
        d_copy_dst: DeviceBuffer,
        d_copy_src: DeviceBuffer,
        d_copy_len: DeviceBuffer,
        d_crc_partials: DeviceBuffer,
        d_crc_part_lens: DeviceBuffer,
        // Packed scalar device slots:
        // [0..8] total_pkt (u64), [8..12] orig_crc (u32), [12..16] stored_crc (u32),
        // [16..20] err_flag (u32), [20..24] low_base (u32).
        d_scalars: DeviceBuffer,
        d_batch_meta: DeviceBuffer,
        dev_plans: Vec<DevicePlan>,
        h_pinned: PinnedHost,
        d2h_event: Event,
        h_meta: PinnedHost,
        h_scalars: PinnedHost,
        h_batch_meta: PinnedHost,
        max_spans: usize,
        max_chunks: usize,
        max_lanes: usize,
    }

    unsafe impl Send for GpuRankCodec {}

    impl GpuRankCodec {
        /// Load an embedded ELF cubin image and allocate a compact 3-buffer aliased per-rank
        /// workspace for `page_plan` and `row_plan` in the current CUDA context.
        pub fn new(
            page_plan: &GpuPlan,
            row_plan: &GpuPlan,
            max_spans: usize,
            cubin: &[u8],
            stream: &Stream,
            event: &Event,
        ) -> Result<Self> {
            if !cubin.starts_with(b"\x7fELF") {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "embedded GPU KV codec image is not a compiled ELF cubin",
                ));
            }
            let module = stream.load_module(cubin).map_err(io_err)?;
            let get_fn = |name: &CStr| -> Result<usize> { module.function(name).map_err(io_err) };

            let fn_pred_enc = get_fn(c"kv_predict_f32_128_enc")?;
            let fn_pred_dec = get_fn(c"kv_predict_f32_128_dec")?;
            let fn_pack = get_fn(c"kv_pack_4lane_gpu")?;
            let fn_unpack = get_fn(c"kv_unpack_4lane_gpu")?;
            let fn_shuffle = get_fn(c"kv_byte_shuffle_gpu")?;
            let fn_page_split = get_fn(c"kv_page_split_gpu")?;
            let fn_copy_batch = get_fn(c"kv_copy_batch_gpu")?;
            let fn_copy_batch_rel = get_fn(c"kv_copy_batch_rel_gpu")?;
            let fn_crc32_stage1 = get_fn(c"kv_crc32_stage1_gpu")?;
            let fn_crc32_finish = get_fn(c"kv_crc32_finish_gpu")?;
            let fn_rans_enc = get_fn(c"kv_rans32_encode_gpu")?;
            let fn_rans_dec = get_fn(c"kv_rans32_decode_gpu")?;
            let fn_build_plan = get_fn(c"kv_build_packet_plan_gpu")?;
            let fn_compact_pkt = get_fn(c"kv_compact_packet_gpu")?;

            // Optional batched page kernels: present in newly exported cubins; older exported
            // cores without them fall back to single-page execution cleanly.
            let batch_fns = (|| -> Option<BatchFns> {
                Some(BatchFns {
                    fn_copy_batch_rel_strided: module
                        .function(c"kv_copy_batch_rel_strided_gpu")
                        .ok()?,
                    fn_transform_batch: module.function(c"kv_transform_batch_gpu").ok()?,
                    fn_crc32_batch: module.function(c"kv_crc32_batch_gpu").ok()?,
                    fn_rans_enc_batch: module.function(c"kv_rans32_encode_batch_gpu").ok()?,
                    fn_rans_dec_batch: module.function(c"kv_rans32_decode_batch_gpu").ok()?,
                    fn_build_plan_batch: module
                        .function(c"kv_build_packet_plan_batch_gpu")
                        .ok()?,
                    fn_compact_pkt_batch: module
                        .function(c"kv_compact_packet_batch_gpu")
                        .ok()?,
                })
            })();
            let page_batch_cap = if batch_fns.is_some() {
                GpuPlan::page_batch_cap(page_plan, row_plan)
            } else {
                1
            };

            for f in [fn_pred_enc, fn_pred_dec] {
                module
                    .set_max_dynamic_shared_bytes(f, 65536)
                    .map_err(io_err)?;
            }

            let max_orig = (page_plan.original_bytes() * page_batch_cap)
                .max(row_plan.original_bytes());
            let max_pkt = (page_plan.max_packet() * page_batch_cap)
                .max(row_plan.max_packet())
                .max(max_orig);
            let max_mats = page_plan.num_mats().max(row_plan.num_mats()).max(1);
            let max_chunks = (page_plan.num_chunks() * page_batch_cap)
                .max(row_plan.num_chunks())
                .max(1);
            let max_lanes = max_mats * 4;
            let max_spans = (max_spans * page_batch_cap).max(64);
            let max_coal = page_plan.coalesce().len().max(row_plan.coalesce().len());
            let max_tr = page_plan
                .transforms()
                .len()
                .max(row_plan.transforms().len());

            let buf_a_bytes = max_orig
                .max(max_pkt)
                .max(max_chunks * RANS_SLOT_STRIDE);
            let buf_b_bytes = max_orig.max(max_pkt);
            let buf_c_bytes = max_orig.max(max_lanes * LANE_STRIDE * 4);
            let meta_bytes = (max_chunks * 8 + max_lanes * 6 + 64)
                .max(max_spans * 24 + 64)
                .max(max_coal * 8 + 64)
                .max(max_tr * TRANSFORM_DESC_BYTES + 64)
                .max(65536);

            let mut h_meta = stream.mapped_host_buffer(meta_bytes).map_err(io_err)?;
            let mut dev_plans = Vec::with_capacity(2);
            dev_plans.push(DevicePlan::upload(page_plan, &mut h_meta, stream, event)?);
            if row_plan.id() != page_plan.id() {
                dev_plans.push(DevicePlan::upload(row_plan, &mut h_meta, stream, event)?);
            }

            Ok(Self {
                _module: module,
                fn_pred_enc,
                fn_pred_dec,
                fn_pack,
                fn_unpack,
                fn_shuffle,
                fn_page_split,
                fn_copy_batch,
                fn_copy_batch_rel,
                fn_crc32_stage1,
                fn_crc32_finish,
                fn_rans_enc,
                fn_rans_dec,
                fn_build_plan,
                fn_compact_pkt,
                batch_fns,
                page_batch_cap,
                buf_a: stream.alloc_device(buf_a_bytes).map_err(io_err)?,
                buf_b: stream.alloc_device(buf_b_bytes).map_err(io_err)?,
                buf_c: stream.alloc_device(buf_c_bytes).map_err(io_err)?,
                d_chunk_words: stream.alloc_device(max_chunks * 4).map_err(io_err)?,
                d_chunk_pkt_offs: stream.alloc_device(max_chunks * 4).map_err(io_err)?,
                d_lane_u32s: stream.alloc_device(max_lanes * 2).map_err(io_err)?,
                d_lane_offs: stream.alloc_device(max_lanes * 4).map_err(io_err)?,
                d_copy_dst: stream.alloc_device(max_spans * 8).map_err(io_err)?,
                d_copy_src: stream.alloc_device(max_spans * 8).map_err(io_err)?,
                d_copy_len: stream.alloc_device(max_spans * 8).map_err(io_err)?,
                d_crc_partials: stream.alloc_device(CRC_BLOCKS * 4).map_err(io_err)?,
                d_crc_part_lens: stream.alloc_device(CRC_BLOCKS * 8).map_err(io_err)?,
                d_scalars: stream.alloc_device(64).map_err(io_err)?,
                d_batch_meta: stream.alloc_device(BATCH_META_BYTES).map_err(io_err)?,
                dev_plans,
                h_pinned: stream.mapped_host_buffer(max_pkt).map_err(io_err)?,
                d2h_event: stream.blocking_event().map_err(io_err)?,
                h_meta,
                h_scalars: stream.mapped_host_buffer(64).map_err(io_err)?,
                h_batch_meta: stream.mapped_host_buffer(BATCH_META_BYTES).map_err(io_err)?,
                max_spans,
                max_chunks,
                max_lanes,
            })
        }

        pub fn page_batch_cap(&self) -> usize {
            self.page_batch_cap
        }

        #[cfg(test)]
        fn allocated_device_bytes(&self) -> u64 {
            let plans_bytes: usize = self.dev_plans.iter().map(DevicePlan::allocated_bytes).sum();
            (self.buf_a.bytes()
                + self.buf_b.bytes()
                + self.buf_c.bytes()
                + self.d_chunk_words.bytes()
                + self.d_chunk_pkt_offs.bytes()
                + self.d_lane_u32s.bytes()
                + self.d_lane_offs.bytes()
                + self.d_copy_dst.bytes()
                + self.d_copy_src.bytes()
                + self.d_copy_len.bytes()
                + self.d_crc_partials.bytes()
                + self.d_crc_part_lens.bytes()
                + self.d_scalars.bytes()
                + self.d_batch_meta.bytes()
                + plans_bytes) as u64
        }

        fn device_plan_index(&self, plan: &GpuPlan) -> Result<usize> {
            self.dev_plans
                .iter()
                .position(|p| p.plan_id == plan.id())
                .ok_or_else(|| invalid("GPU codec plan was not pre-uploaded for this rank"))
        }

        fn transfer_spans(
            &mut self,
            base: u64,
            n: usize,
            spans: &[DeviceSpan],
            gather: bool,
            stream: &Stream,
        ) -> Result<()> {
            if gather {
                unsafe { stream.memset_d8_async(base, 0, n).map_err(io_err)? };
            }
            if spans.len() > 1 {
                let mut dsts = Vec::with_capacity(spans.len());
                let mut srcs = Vec::with_capacity(spans.len());
                let mut lens = Vec::with_capacity(spans.len());
                for s in spans {
                    if s.offset.checked_add(s.length).is_none_or(|e| e > n) {
                        return Err(invalid("GPU span exceeds unit size"));
                    }
                    if s.length > 0 {
                        let staged = base + s.offset as u64;
                        dsts.push(if gather { staged } else { s.device });
                        srcs.push(if gather { s.device } else { staged });
                        lens.push(s.length as u64);
                    }
                }
                let count = dsts.len();
                if count == 0 {
                    return Ok(());
                }
                if count > self.max_spans {
                    return Err(invalid(format!(
                        "GPU span copy count {count} exceeds capacity {}",
                        self.max_spans
                    )));
                }
                let stride = count * 8;
                let meta = self.h_meta.bytes_mut();
                write_u64s(&mut meta[0..stride], &dsts);
                write_u64s(&mut meta[stride..stride * 2], &srcs);
                write_u64s(&mut meta[stride * 2..stride * 3], &lens);
                let hptr = self.h_meta.address() as *const u8;
                for (dev, off) in [
                    (self.d_copy_dst.ptr(), 0),
                    (self.d_copy_src.ptr(), stride),
                    (self.d_copy_len.ptr(), stride * 2),
                ] {
                    unsafe {
                        stream
                            .copy_htod_async(dev, hptr.add(off) as _, stride)
                            .map_err(io_err)?;
                    }
                }
                let (mut dp, mut sp, mut lp, mut cnt) = (
                    self.d_copy_dst.ptr(),
                    self.d_copy_src.ptr(),
                    self.d_copy_len.ptr(),
                    count as c_int,
                );
                launch_1d(
                    self.fn_copy_batch,
                    count as u32,
                    128,
                    0,
                    stream,
                    &mut [
                        &mut dp as *mut _ as _,
                        &mut sp as *mut _ as _,
                        &mut lp as *mut _ as _,
                        &mut cnt as *mut _ as _,
                    ],
                    "launch(kv_copy_batch_gpu)",
                )?;
            } else if let Some(s) = spans.first() {
                if s.offset.checked_add(s.length).is_none_or(|e| e > n) {
                    return Err(invalid("GPU span exceeds unit size"));
                }
                if s.length > 0 {
                    let staged = base + s.offset as u64;
                    let (dst, src) = if gather {
                        (staged, s.device)
                    } else {
                        (s.device, staged)
                    };
                    unsafe { stream.copy_dtod_async(dst, src, s.length).map_err(io_err)? };
                }
            }
            Ok(())
        }

        fn launch_rel_copy_batch(
            &self,
            mut dst_base: u64,
            mut src_base: u64,
            mut dst_offs: u64,
            mut src_offs: u64,
            mut lens: u64,
            count: usize,
            stream: &Stream,
        ) -> Result<()> {
            if count == 0 {
                return Ok(());
            }
            let mut n = count as c_int;
            launch_1d(
                self.fn_copy_batch_rel,
                count as u32,
                128,
                0,
                stream,
                &mut [
                    &mut dst_base as *mut _ as _,
                    &mut src_base as *mut _ as _,
                    &mut dst_offs as *mut _ as _,
                    &mut src_offs as *mut _ as _,
                    &mut lens as *mut _ as _,
                    &mut n as *mut _ as _,
                ],
                "launch(kv_copy_batch_rel_gpu)",
            )
        }

        fn launch_gpu_crc32(
            &self,
            mut dev_ptr: u64,
            mut len_ptr: u64,
            mut fixed_len: u64,
            mut max_len: u64,
            mut crc_out_dev: u64,
            stream: &Stream,
        ) -> Result<()> {
            let mut parts = self.d_crc_partials.ptr();
            let mut plens = self.d_crc_part_lens.ptr();
            launch_1d(
                self.fn_crc32_stage1,
                CRC_BLOCKS as u32,
                128,
                0,
                stream,
                &mut [
                    &mut dev_ptr as *mut _ as _,
                    &mut len_ptr as *mut _ as _,
                    &mut fixed_len as *mut _ as _,
                    &mut max_len as *mut _ as _,
                    &mut parts as *mut _ as _,
                    &mut plens as *mut _ as _,
                ],
                "launch(kv_crc32_stage1_gpu)",
            )?;
            launch_1d(
                self.fn_crc32_finish,
                1,
                CRC_BLOCKS as u32,
                0,
                stream,
                &mut [
                    &mut parts as *mut _ as _,
                    &mut plens as *mut _ as _,
                    &mut crc_out_dev as *mut _ as _,
                ],
                "launch(kv_crc32_finish_gpu)",
            )
        }

        fn launch_non_matrix_transforms(
            &self,
            plan: &GpuPlan,
            dst_base: u64,
            src_base: u64,
            decode: i32,
            stream: &Stream,
        ) -> Result<()> {
            for op in plan.transforms() {
                let mut dst = dst_base + op.offset as u64;
                let mut src = src_base + op.offset as u64;
                let mut len = op.length;
                let grid = len.div_ceil(256) as u32;
                match op.encoding {
                    Encoding::Bytes => {
                        unsafe { stream.copy_dtod_async(dst, src, len).map_err(io_err)? };
                    }
                    Encoding::Bfloat16 => {
                        let (mut width, mut dec) = (2u32, decode);
                        launch_1d(
                            self.fn_shuffle,
                            grid,
                            256,
                            0,
                            stream,
                            &mut [
                                &mut dst as *mut _ as _,
                                &mut src as *mut _ as _,
                                &mut len as *mut _ as _,
                                &mut width as *mut _ as _,
                                &mut dec as *mut _ as _,
                            ],
                            "launch(kv_byte_shuffle_gpu)",
                        )?;
                    }
                    enc => {
                        let spec = enc.page_spec().unwrap();
                        let (mut block, mut rows, mut cols, mut stride, mut scale_w, mut planar, mut dec) = (
                            spec.block,
                            spec.rows,
                            spec.cols,
                            spec.stride,
                            spec.scale_w,
                            spec.planar,
                            decode,
                        );
                        launch_1d(
                            self.fn_page_split,
                            grid,
                            256,
                            0,
                            stream,
                            &mut [
                                &mut dst as *mut _ as _,
                                &mut src as *mut _ as _,
                                &mut len as *mut _ as _,
                                &mut block as *mut _ as _,
                                &mut rows as *mut _ as _,
                                &mut cols as *mut _ as _,
                                &mut stride as *mut _ as _,
                                &mut scale_w as *mut _ as _,
                                &mut planar as *mut _ as _,
                                &mut dec as *mut _ as _,
                            ],
                            "launch(kv_page_split_gpu)",
                        )?;
                    }
                }
            }
            Ok(())
        }

        /// Encode one unit directly from GPU cache plane spans and pass the resulting pinned DMA
        /// slice directly to `f` without heap allocation.
        pub fn encode_from_device_with<F, R>(
            &mut self,
            plan: &GpuPlan,
            mut unit: PackedUnit,
            profile: &[u8; 32],
            spans: &[DeviceSpan],
            stream: &Stream,
            event: &Event,
            f: F,
        ) -> Result<R>
        where
            F: FnOnce(PackedUnit, &[u8]) -> R,
        {
            let n = plan.original_bytes();
            let nc = plan.num_chunks();
            let nm_usize = plan.num_mats();
            if nc > self.max_chunks || nm_usize * 4 > self.max_lanes {
                return Err(invalid("GPU encode plan exceeds codec workspace capacity"));
            }
            let dp_idx = self.device_plan_index(plan)?;

            // 1. Zero-copy direct read for contiguous full-size spans; gather multi-plane pages into `buf_c`.
            let full_contiguous = spans.len() == 1 && spans[0].offset == 0 && spans[0].length == n;
            let raw_dev = if full_contiguous {
                spans[0].device
            } else {
                self.transfer_spans(self.buf_c.ptr(), n, spans, true, stream)?;
                self.buf_c.ptr()
            };

            // 2. Original CRC32 on GPU -> `d_scalars[8..12]`.
            self.launch_gpu_crc32(raw_dev, 0, n as u64, 0, self.d_scalars.ptr() + 8, stream)?;

            // 3. Non-matrix transforms (raw_dev -> buf_a) + coalesce (buf_a -> buf_b) before matrix pack reuses `buf_c`.
            if !plan.transforms().is_empty() {
                self.launch_non_matrix_transforms(plan, self.buf_a.ptr(), raw_dev, 0, stream)?;
                let dp = &self.dev_plans[dp_idx];
                self.launch_rel_copy_batch(
                    self.buf_b.ptr(),
                    self.buf_a.ptr(),
                    dp.d_coal_ans.ptr(),
                    dp.d_coal_trans.ptr(),
                    dp.d_coal_len.ptr(),
                    plan.coalesce().len(),
                    stream,
                )?;
            }

            // 4. Matrix predictor (raw_dev -> buf_a) + 4-lane pack (buf_a -> widths in buf_b, low_pad in buf_c).
            if nm_usize > 0 {
                let (mut raw_base, mut mat_offs, mut z_ptr, mut nm) = (
                    raw_dev,
                    self.dev_plans[dp_idx].d_mat_offs.ptr(),
                    self.buf_a.ptr(),
                    nm_usize as c_int,
                );
                launch_1d(
                    self.fn_pred_enc,
                    nm_usize as u32,
                    256,
                    65536,
                    stream,
                    &mut [
                        &mut raw_base as *mut _ as _,
                        &mut mat_offs as *mut _ as _,
                        &mut z_ptr as *mut _ as _,
                        &mut nm as *mut _ as _,
                    ],
                    "launch(kv_predict_f32_128_enc)",
                )?;
                let (mut z_ptr, mut w_ptr, mut low_ptr, mut lanes_ptr, mut nm) = (
                    self.buf_a.ptr(),
                    self.buf_b.ptr(),
                    self.buf_c.ptr(),
                    self.d_lane_u32s.ptr(),
                    nm_usize as c_int,
                );
                launch_1d(
                    self.fn_pack,
                    (nm_usize * 4).div_ceil(32) as u32,
                    32,
                    0,
                    stream,
                    &mut [
                        &mut z_ptr as *mut _ as _,
                        &mut w_ptr as *mut _ as _,
                        &mut low_ptr as *mut _ as _,
                        &mut lanes_ptr as *mut _ as _,
                        &mut nm as *mut _ as _,
                    ],
                    "launch(kv_pack_4lane_gpu)",
                )?;
            }

            // 5. Pure-CUDA 32-way rANS encode (`buf_b` -> `buf_a` slots).
            if nc > 0 {
                let (mut ans_in, mut cho, mut chs, mut slot_out, mut ch_words, mut n_chunks) = (
                    self.buf_b.ptr(),
                    self.dev_plans[dp_idx].d_chunk_offs.ptr(),
                    self.dev_plans[dp_idx].d_chunk_sizes.ptr(),
                    self.buf_a.ptr(),
                    self.d_chunk_words.ptr(),
                    nc as c_int,
                );
                launch_1d(
                    self.fn_rans_enc,
                    nc as u32,
                    256,
                    0,
                    stream,
                    &mut [
                        &mut ans_in as *mut _ as _,
                        &mut cho as *mut _ as _,
                        &mut chs as *mut _ as _,
                        &mut slot_out as *mut _ as _,
                        &mut ch_words as *mut _ as _,
                        &mut n_chunks as *mut _ as _,
                    ],
                    "launch(kv_rans32_encode_gpu)",
                )?;
            }

            // 6. On-GPU packet header/directory assembly & compaction (`buf_a` + `buf_c` -> `buf_b`)
            //    + stored CRC32 (`buf_b` -> `d_scalars[12..16]`).
            {
                let (
                    mut d_pkt,
                    mut ch_words,
                    mut ch_pkt_offs,
                    mut lane_u32s,
                    mut lane_offs,
                    mut total_pkt_ptr,
                    mut low_base_ptr,
                    mut orig_b,
                    mut n_ch,
                    mut n_mat,
                ) = (
                    self.buf_b.ptr(),
                    self.d_chunk_words.ptr(),
                    self.d_chunk_pkt_offs.ptr(),
                    self.d_lane_u32s.ptr(),
                    self.d_lane_offs.ptr(),
                    self.d_scalars.ptr(),
                    self.d_scalars.ptr() + 20,
                    n as u32,
                    nc as u32,
                    nm_usize as u32,
                );
                launch_1d(
                    self.fn_build_plan,
                    1,
                    1,
                    0,
                    stream,
                    &mut [
                        &mut d_pkt as *mut _ as _,
                        &mut ch_words as *mut _ as _,
                        &mut ch_pkt_offs as *mut _ as _,
                        &mut lane_u32s as *mut _ as _,
                        &mut lane_offs as *mut _ as _,
                        &mut total_pkt_ptr as *mut _ as _,
                        &mut low_base_ptr as *mut _ as _,
                        &mut orig_b as *mut _ as _,
                        &mut n_ch as *mut _ as _,
                        &mut n_mat as *mut _ as _,
                    ],
                    "launch(kv_build_packet_plan_gpu)",
                )?;
                let compact_blocks = nc + nm_usize * 4;
                if compact_blocks > 0 {
                    let (mut slot_out, mut low_pad) = (self.buf_a.ptr(), self.buf_c.ptr());
                    launch_1d(
                        self.fn_compact_pkt,
                        compact_blocks as u32,
                        128,
                        0,
                        stream,
                        &mut [
                            &mut d_pkt as *mut _ as _,
                            &mut slot_out as *mut _ as _,
                            &mut ch_words as *mut _ as _,
                            &mut ch_pkt_offs as *mut _ as _,
                            &mut low_pad as *mut _ as _,
                            &mut lane_u32s as *mut _ as _,
                            &mut lane_offs as *mut _ as _,
                            &mut total_pkt_ptr as *mut _ as _,
                            &mut low_base_ptr as *mut _ as _,
                            &mut orig_b as *mut _ as _,
                            &mut n_ch as *mut _ as _,
                            &mut n_mat as *mut _ as _,
                        ],
                        "launch(kv_compact_packet_gpu)",
                    )?;
                }
            }
            self.launch_gpu_crc32(
                self.buf_b.ptr(),
                self.d_scalars.ptr(),
                0,
                n as u64,
                self.d_scalars.ptr() + 12,
                stream,
            )?;

            // 7. Read back 16-byte pinned scalar summary: (total_pkt, orig_crc, stored_crc).
            unsafe {
                stream
                    .copy_dtoh_async(self.h_scalars.address() as _, self.d_scalars.ptr(), 16)
                    .map_err(io_err)?;
            }
            sync_stream(stream, event)?;
            let sc = &self.h_scalars.bytes()[..16];
            let total_pkt = u64::from_le_bytes(sc[0..8].try_into().unwrap()) as usize;
            let original_crc = u32::from_le_bytes(sc[8..12].try_into().unwrap());
            let stored_crc = u32::from_le_bytes(sc[12..16].try_into().unwrap());

            unit.original_length = n as u64;
            let used = if total_pkt >= n {
                // Incompressible: emit Literal raw copy via pinned DMA.
                let lit_src = if full_contiguous {
                    spans[0].device
                } else {
                    if nm_usize > 0 {
                        self.transfer_spans(self.buf_c.ptr(), n, spans, true, stream)?;
                    }
                    self.buf_c.ptr()
                };
                unsafe {
                    stream
                        .copy_dtoh_async(self.h_pinned.address() as _, lit_src, n)
                        .map_err(io_err)?;
                }
                sync_stream(stream, event)?;
                unit.representation = Representation::Literal as i32;
                unit.profile.clear();
                unit.stored_length = n as u64;
                unit.stored_crc = 0;
                unit.original_crc = 0;
                unit.raw_xxh3 = flare_selene::record::checksum(&self.h_pinned.bytes()[..n]);
                n
            } else {
                // 8. ONE exact-size pinned D2H of the finished `GKV1/v2` packet.
                unsafe {
                    stream
                        .copy_dtoh_async(self.h_pinned.address() as _, self.buf_b.ptr(), total_pkt)
                        .map_err(io_err)?;
                }
                sync_stream(stream, event)?;
                unit.representation = Representation::Packet as i32;
                unit.profile = profile.to_vec();
                unit.stored_length = total_pkt as u64;
                unit.stored_crc = stored_crc;
                unit.original_crc = original_crc;
                unit.raw_xxh3 = 0;
                total_pkt
            };
            Ok(f(unit, &self.h_pinned.bytes()[..used]))
        }

        /// Verify stored CRC32 on GPU, decode a `GKV1/v2` packet on GPU, verify original CRC32 on
        /// GPU, and scatter the verified bytes into the destination GPU cache planes.
        #[allow(clippy::too_many_arguments)]
        pub fn decode_to_device(
            &mut self,
            plan: &GpuPlan,
            chunks: &[&[u8]],
            spans: &[DeviceSpan],
            stored_crc: u32,
            original_crc: u32,
            stream: &Stream,
            event: &Event,
        ) -> Result<()> {
            let total: usize = chunks.iter().map(|c| c.len()).sum();
            if total < plan.metadata_bytes() || total > plan.max_packet() {
                if crc32_chunks(chunks) != stored_crc {
                    return Err(Error::new(ErrorKind::InvalidData, Checksum::Stored));
                }
                return Err(invalid("invalid GKV1 packet length"));
            }
            let nc = plan.num_chunks();
            let nm_usize = plan.num_mats();
            let total_lanes = nm_usize * 4;
            if nc > self.max_chunks || total_lanes > self.max_lanes {
                return Err(invalid("GPU decode plan exceeds codec workspace capacity"));
            }

            if crc32_chunks(chunks) != stored_crc {
                return Err(Error::new(ErrorKind::InvalidData, Checksum::Stored));
            }
            {
                let dst = &mut self.h_pinned.bytes_mut()[..total];
                let mut off = 0usize;
                for c in chunks {
                    dst[off..off + c.len()].copy_from_slice(c);
                    off += c.len();
                }
            }
            let hdr = plan.parse_packet_header(&self.h_pinned.bytes()[..total])?;
            let dp_idx = self.device_plan_index(plan)?;

            unsafe {
                stream
                    .copy_htod_async(self.buf_a.ptr(), self.h_pinned.address() as _, total)
                    .map_err(io_err)?;
            }

            let off_ch_words = nc * 4;
            let off_lane_offs = nc * 8;
            let off_lane_u32s = off_lane_offs + total_lanes * 4;
            {
                let meta = self.h_meta.bytes_mut();
                write_u32s(&mut meta[0..nc * 4], &hdr.chunk_pkt_offs);
                write_u32s(&mut meta[off_ch_words..off_lane_offs], &hdr.chunk_words);
                write_u32s(&mut meta[off_lane_offs..off_lane_u32s], &hdr.lane_offs);
                write_u16s(
                    &mut meta[off_lane_u32s..off_lane_u32s + total_lanes * 2],
                    &hdr.lane_u32s,
                );
            }

            unsafe {
                stream
                    .memset_d8_async(self.d_scalars.ptr() + 16, 0, 4)
                    .map_err(io_err)?;
            }
            let hptr = self.h_meta.address() as *const u8;
            if nc > 0 {
                unsafe {
                    stream
                        .copy_htod_async(self.d_chunk_pkt_offs.ptr(), hptr as _, nc * 4)
                        .map_err(io_err)?;
                    stream
                        .copy_htod_async(
                            self.d_chunk_words.ptr(),
                            hptr.add(off_ch_words) as _,
                            nc * 4,
                        )
                        .map_err(io_err)?;
                }
            }
            if total_lanes > 0 {
                unsafe {
                    stream
                        .copy_htod_async(
                            self.d_lane_offs.ptr(),
                            hptr.add(off_lane_offs) as _,
                            total_lanes * 4,
                        )
                        .map_err(io_err)?;
                    stream
                        .copy_htod_async(
                            self.d_lane_u32s.ptr(),
                            hptr.add(off_lane_u32s) as _,
                            total_lanes * 2,
                        )
                        .map_err(io_err)?;
                }
            }

            // 1. Decode all rANS32 + raw chunks (`buf_a` -> `buf_b`).
            if nc > 0 {
                let (
                    mut d_pkt,
                    mut ch_pkt_offs,
                    mut ch_words,
                    mut ans_offs,
                    mut ch_raw_szs,
                    mut ans_out,
                    mut n_chunks,
                    mut err_p,
                ) = (
                    self.buf_a.ptr(),
                    self.d_chunk_pkt_offs.ptr(),
                    self.d_chunk_words.ptr(),
                    self.dev_plans[dp_idx].d_chunk_offs.ptr(),
                    self.dev_plans[dp_idx].d_chunk_sizes.ptr(),
                    self.buf_b.ptr(),
                    nc as c_int,
                    self.d_scalars.ptr() + 16,
                );
                launch_1d(
                    self.fn_rans_dec,
                    nc as u32,
                    256,
                    0,
                    stream,
                    &mut [
                        &mut d_pkt as *mut _ as _,
                        &mut ch_pkt_offs as *mut _ as _,
                        &mut ch_words as *mut _ as _,
                        &mut ans_offs as *mut _ as _,
                        &mut ch_raw_szs as *mut _ as _,
                        &mut ans_out as *mut _ as _,
                        &mut n_chunks as *mut _ as _,
                        &mut err_p as *mut _ as _,
                    ],
                    "launch(kv_rans32_decode_gpu)",
                )?;
            }

            // 2. Matrix 4-lane unpack (`buf_b` + `buf_a + low_offset` -> `buf_c`) + inverse predictor (`buf_c` -> `buf_a`).
            if nm_usize > 0 {
                let (mut zp, mut wp, mut lc, mut lo, mut lu, mut nm, mut ef) = (
                    self.buf_c.ptr(),
                    self.buf_b.ptr(),
                    self.buf_a.ptr() + hdr.low_offset as u64,
                    self.d_lane_offs.ptr(),
                    self.d_lane_u32s.ptr(),
                    nm_usize as c_int,
                    self.d_scalars.ptr() + 16,
                );
                launch_1d(
                    self.fn_unpack,
                    total_lanes.div_ceil(32) as u32,
                    32,
                    0,
                    stream,
                    &mut [
                        &mut zp as *mut _ as _,
                        &mut wp as *mut _ as _,
                        &mut lc as *mut _ as _,
                        &mut lo as *mut _ as _,
                        &mut lu as *mut _ as _,
                        &mut nm as *mut _ as _,
                        &mut ef as *mut _ as _,
                    ],
                    "launch(kv_unpack_4lane_gpu)",
                )?;

                let (mut raw_base, mut mat_offs, mut z_ptr, mut nm, mut err_ptr) = (
                    self.buf_a.ptr(),
                    self.dev_plans[dp_idx].d_mat_offs.ptr(),
                    self.buf_c.ptr(),
                    nm_usize as c_int,
                    self.d_scalars.ptr() + 16,
                );
                launch_1d(
                    self.fn_pred_dec,
                    nm_usize as u32,
                    256,
                    65536,
                    stream,
                    &mut [
                        &mut raw_base as *mut _ as _,
                        &mut mat_offs as *mut _ as _,
                        &mut z_ptr as *mut _ as _,
                        &mut nm as *mut _ as _,
                        &mut err_ptr as *mut _ as _,
                    ],
                    "launch(kv_predict_f32_128_dec)",
                )?;
            }

            // 3. Uncoalesce non-matrix lanes (`buf_b` -> `buf_c`) + inverse transforms (`buf_c` -> `buf_a`).
            if !plan.transforms().is_empty() {
                let dp = &self.dev_plans[dp_idx];
                self.launch_rel_copy_batch(
                    self.buf_c.ptr(),
                    self.buf_b.ptr(),
                    dp.d_coal_trans.ptr(),
                    dp.d_coal_ans.ptr(),
                    dp.d_coal_len.ptr(),
                    plan.coalesce().len(),
                    stream,
                )?;
                self.launch_non_matrix_transforms(plan, self.buf_a.ptr(), self.buf_c.ptr(), 1, stream)?;
            }

            // 4. Verify `err_flag` and `original_crc` over `buf_a` BEFORE scattering into live GPU cache.
            let n = plan.original_bytes();
            self.launch_gpu_crc32(self.buf_a.ptr(), 0, n as u64, 0, self.d_scalars.ptr() + 8, stream)?;
            unsafe {
                stream
                    .copy_dtoh_async(
                        (self.h_scalars.address() + 8) as _,
                        self.d_scalars.ptr() + 8,
                        12,
                    )
                    .map_err(io_err)?;
            }
            sync_stream(stream, event)?;
            let sc = &self.h_scalars.bytes()[..20];
            let got_orig_crc = u32::from_le_bytes(sc[8..12].try_into().unwrap());
            let err_val = u32::from_le_bytes(sc[16..20].try_into().unwrap());
            if err_val != 0 {
                return Err(invalid("invalid compressed payload in GKV1 packet"));
            }
            if got_orig_crc != original_crc {
                return Err(Error::new(ErrorKind::InvalidData, Checksum::Original));
            }

            // 5. Scatter verified raw bytes from `buf_a` into destination GPU cache planes.
            self.transfer_spans(self.buf_a.ptr(), n, spans, false, stream)?;
            sync_stream(stream, event)
        }

        fn transfer_spans_batch(
            &mut self,
            base: u64,
            n: usize,
            spans_per_page: &[Vec<DeviceSpan>],
            gather: bool,
            stream: &Stream,
        ) -> Result<()> {
            let batch = spans_per_page.len();
            if gather {
                let has_gaps = spans_per_page.iter().any(|spans| {
                    let mut covered = 0usize;
                    for s in spans {
                        if s.offset != covered {
                            return true;
                        }
                        covered += s.length;
                    }
                    covered != n
                });
                if has_gaps {
                    unsafe {
                        stream
                            .memset_d8_async(base, 0, batch * n)
                            .map_err(io_err)?;
                    }
                }
            }
            let total_est: usize = spans_per_page.iter().map(|s| s.len()).sum();
            let mut dsts = Vec::with_capacity(total_est);
            let mut srcs = Vec::with_capacity(total_est);
            let mut lens = Vec::with_capacity(total_est);
            for (p, spans) in spans_per_page.iter().enumerate() {
                let page_base = base + (p * n) as u64;
                for s in spans {
                    if s.offset.checked_add(s.length).is_none_or(|e| e > n) {
                        return Err(invalid("GPU span exceeds unit size"));
                    }
                    if s.length > 0 {
                        let staged = page_base + s.offset as u64;
                        dsts.push(if gather { staged } else { s.device });
                        srcs.push(if gather { s.device } else { staged });
                        lens.push(s.length as u64);
                    }
                }
            }
            let count = dsts.len();
            if count == 0 {
                return Ok(());
            }
            if count > self.max_spans {
                return Err(invalid(format!(
                    "GPU batched span copy count {count} exceeds capacity {}",
                    self.max_spans
                )));
            }
            let stride = count * 8;
            let meta = self.h_meta.bytes_mut();
            write_u64s(&mut meta[0..stride], &dsts);
            write_u64s(&mut meta[stride..stride * 2], &srcs);
            write_u64s(&mut meta[stride * 2..stride * 3], &lens);
            let hptr = self.h_meta.address() as *const u8;
            for (dev, off) in [
                (self.d_copy_dst.ptr(), 0),
                (self.d_copy_src.ptr(), stride),
                (self.d_copy_len.ptr(), stride * 2),
            ] {
                unsafe {
                    stream
                        .copy_htod_async(dev, hptr.add(off) as _, stride)
                        .map_err(io_err)?;
                }
            }
            let (mut dp, mut sp, mut lp, mut cnt) = (
                self.d_copy_dst.ptr(),
                self.d_copy_src.ptr(),
                self.d_copy_len.ptr(),
                count as c_int,
            );
            launch_1d(
                self.fn_copy_batch,
                count as u32,
                128,
                0,
                stream,
                &mut [
                    &mut dp as *mut _ as _,
                    &mut sp as *mut _ as _,
                    &mut lp as *mut _ as _,
                    &mut cnt as *mut _ as _,
                ],
                "launch(kv_copy_batch_gpu)",
            )
        }

        /// Queue GPU kernels (gather, CRC32, transforms, rANS32, compaction, stored CRC32) and
        /// metadata D2H for one page batch on `stream` without waiting on host.
        #[allow(clippy::too_many_arguments)]
        fn launch_encode_pages_batch_kernels(
            &mut self,
            plan: &GpuPlan,
            spans_per_page: &[Vec<DeviceSpan>],
            stream: &Stream,
        ) -> Result<()> {
            let batch = spans_per_page.len();
            if batch == 0
                || batch > self.page_batch_cap
                || plan.num_mats() != 0
                || plan.num_chunks() == 0
            {
                return Err(invalid("GPU page batch exceeds codec batch capacity"));
            }
            let bf = self
                .batch_fns
                .ok_or_else(|| invalid("loaded GPU codec cubin lacks batched page kernels"))?;
            let n = plan.original_bytes();
            let nc = plan.num_chunks();
            let total_chunks = nc * batch;
            if total_chunks > self.max_chunks {
                return Err(invalid("GPU batched encode exceeds chunk workspace capacity"));
            }
            let dp_idx = self.device_plan_index(plan)?;

            // 1. Gather all `batch * planes` spans into `buf_c[0..batch * n]`.
            self.transfer_spans_batch(self.buf_c.ptr(), n, spans_per_page, true, stream)?;

            // 2. Original CRC32 for all `batch` pages in 1 launch (`buf_c` -> `d_batch_meta[ORIG_CRC]`).
            {
                let (mut base, mut offs, mut lens, mut fstride, mut flen, mut mlen, mut out, mut b) = (
                    self.buf_c.ptr(),
                    0u64,
                    0u64,
                    n as u64,
                    n as u64,
                    0u64,
                    self.d_batch_meta.ptr() + BATCH_OFF_ORIG_CRC as u64,
                    batch as c_int,
                );
                launch_1d(
                    bf.fn_crc32_batch,
                    batch as u32,
                    256,
                    0,
                    stream,
                    &mut [
                        &mut base as *mut _ as _,
                        &mut offs as *mut _ as _,
                        &mut lens as *mut _ as _,
                        &mut fstride as *mut _ as _,
                        &mut flen as *mut _ as _,
                        &mut mlen as *mut _ as _,
                        &mut out as *mut _ as _,
                        &mut b as *mut _ as _,
                    ],
                    "launch(kv_crc32_batch_gpu:orig)",
                )?;
            }

            // 3. Fused non-matrix transforms (`buf_c` -> `buf_a`) + strided coalesce (`buf_a` -> `buf_b`).
            let ans_stride = plan.ans_total_bytes();
            {
                let dp = &self.dev_plans[dp_idx];
                if dp.num_transforms > 0 {
                    let (mut dst, mut src, mut descs, mut nd, mut pstride, mut b, mut dec) = (
                        self.buf_a.ptr(),
                        self.buf_c.ptr(),
                        dp.d_transforms.ptr(),
                        dp.num_transforms as c_int,
                        n,
                        batch as c_int,
                        0i32,
                    );
                    launch_1d(
                        bf.fn_transform_batch,
                        (dp.num_transforms * batch) as u32,
                        256,
                        0,
                        stream,
                        &mut [
                            &mut dst as *mut _ as _,
                            &mut src as *mut _ as _,
                            &mut descs as *mut _ as _,
                            &mut nd as *mut _ as _,
                            &mut pstride as *mut _ as _,
                            &mut b as *mut _ as _,
                            &mut dec as *mut _ as _,
                        ],
                        "launch(kv_transform_batch_gpu:enc)",
                    )?;
                }
                let n_coal = plan.coalesce().len();
                if n_coal > 0 {
                    let (mut dst, mut src, mut doffs, mut soffs, mut lens, mut nops, mut dstr, mut sstr, mut b) = (
                        self.buf_b.ptr(),
                        self.buf_a.ptr(),
                        dp.d_coal_ans.ptr(),
                        dp.d_coal_trans.ptr(),
                        dp.d_coal_len.ptr(),
                        n_coal as c_int,
                        ans_stride,
                        n,
                        batch as c_int,
                    );
                    launch_1d(
                        bf.fn_copy_batch_rel_strided,
                        (n_coal * batch) as u32,
                        128,
                        0,
                        stream,
                        &mut [
                            &mut dst as *mut _ as _,
                            &mut src as *mut _ as _,
                            &mut doffs as *mut _ as _,
                            &mut soffs as *mut _ as _,
                            &mut lens as *mut _ as _,
                            &mut nops as *mut _ as _,
                            &mut dstr as *mut _ as _,
                            &mut sstr as *mut _ as _,
                            &mut b as *mut _ as _,
                        ],
                        "launch(kv_copy_batch_rel_strided_gpu:enc)",
                    )?;
                }
            }

            // 4. Batched 32-way rANS encode (`buf_b` -> `buf_a` slots).
            {
                let dp = &self.dev_plans[dp_idx];
                let (mut ans_in, mut cho, mut chs, mut slot_out, mut ch_words, mut n_ch, mut astr, mut b) = (
                    self.buf_b.ptr(),
                    dp.d_chunk_offs.ptr(),
                    dp.d_chunk_sizes.ptr(),
                    self.buf_a.ptr(),
                    self.d_chunk_words.ptr(),
                    nc as c_int,
                    ans_stride,
                    batch as c_int,
                );
                launch_1d(
                    bf.fn_rans_enc_batch,
                    total_chunks as u32,
                    256,
                    0,
                    stream,
                    &mut [
                        &mut ans_in as *mut _ as _,
                        &mut cho as *mut _ as _,
                        &mut chs as *mut _ as _,
                        &mut slot_out as *mut _ as _,
                        &mut ch_words as *mut _ as _,
                        &mut n_ch as *mut _ as _,
                        &mut astr as *mut _ as _,
                        &mut b as *mut _ as _,
                    ],
                    "launch(kv_rans32_encode_batch_gpu)",
                )?;
            }

            // 5. Build all `batch` packet headers/directories contiguously into `buf_b` and compact.
            let d_pkt_offs = self.d_batch_meta.ptr() + BATCH_OFF_PKT_OFFS as u64;
            let d_pkt_lens = self.d_batch_meta.ptr() + BATCH_OFF_PKT_LENS as u64;
            let d_total_bytes = self.d_batch_meta.ptr() + BATCH_OFF_TOTAL_BYTES as u64;
            {
                let (mut d_pkt, mut ch_w, mut ch_po, mut tot_arr, mut out_offs, mut tot_b, mut orig_b, mut n_ch, mut b) = (
                    self.buf_b.ptr(),
                    self.d_chunk_words.ptr(),
                    self.d_chunk_pkt_offs.ptr(),
                    d_pkt_lens,
                    d_pkt_offs,
                    d_total_bytes,
                    n as u32,
                    nc as u32,
                    batch as c_int,
                );
                launch_1d(
                    bf.fn_build_plan_batch,
                    1,
                    MAX_PAGE_BATCH as u32,
                    0,
                    stream,
                    &mut [
                        &mut d_pkt as *mut _ as _,
                        &mut ch_w as *mut _ as _,
                        &mut ch_po as *mut _ as _,
                        &mut tot_arr as *mut _ as _,
                        &mut out_offs as *mut _ as _,
                        &mut tot_b as *mut _ as _,
                        &mut orig_b as *mut _ as _,
                        &mut n_ch as *mut _ as _,
                        &mut b as *mut _ as _,
                    ],
                    "launch(kv_build_packet_plan_batch_gpu)",
                )?;
                let (mut slot_out, mut raw_c) = (self.buf_a.ptr(), self.buf_c.ptr());
                launch_1d(
                    bf.fn_compact_pkt_batch,
                    total_chunks as u32,
                    128,
                    0,
                    stream,
                    &mut [
                        &mut d_pkt as *mut _ as _,
                        &mut slot_out as *mut _ as _,
                        &mut raw_c as *mut _ as _,
                        &mut ch_w as *mut _ as _,
                        &mut ch_po as *mut _ as _,
                        &mut tot_arr as *mut _ as _,
                        &mut out_offs as *mut _ as _,
                        &mut orig_b as *mut _ as _,
                        &mut n_ch as *mut _ as _,
                        &mut b as *mut _ as _,
                    ],
                    "launch(kv_compact_packet_batch_gpu)",
                )?;
            }

            // 6. Stored CRC32 for all compressible packets in 1 launch (skips literal fallbacks where len >= n).
            {
                let (mut base, mut offs, mut lens, mut fstride, mut flen, mut mlen, mut out, mut b) = (
                    self.buf_b.ptr(),
                    d_pkt_offs,
                    d_pkt_lens,
                    0u64,
                    0u64,
                    n as u64,
                    self.d_batch_meta.ptr() + BATCH_OFF_STORED_CRC as u64,
                    batch as c_int,
                );
                launch_1d(
                    bf.fn_crc32_batch,
                    batch as u32,
                    256,
                    0,
                    stream,
                    &mut [
                        &mut base as *mut _ as _,
                        &mut offs as *mut _ as _,
                        &mut lens as *mut _ as _,
                        &mut fstride as *mut _ as _,
                        &mut flen as *mut _ as _,
                        &mut mlen as *mut _ as _,
                        &mut out as *mut _ as _,
                        &mut b as *mut _ as _,
                    ],
                    "launch(kv_crc32_batch_gpu:stored)",
                )?;
            }

            // 7. Queue compact batch metadata (`1544` bytes) D2H.
            unsafe {
                stream
                    .copy_dtoh_async(
                        self.h_batch_meta.address() as _,
                        self.d_batch_meta.ptr(),
                        BATCH_OFF_TOTAL_BYTES + 8,
                    )
                    .map_err(io_err)
            }
        }

        fn emit_encoded_batch<F>(
            bm: &[u8],
            pinned: &[u8],
            units: &[PackedUnit],
            n: usize,
            profile: &[u8; 32],
            f: &mut F,
        ) -> u64
        where
            F: FnMut(PackedUnit, &[u8]),
        {
            let mut stored_total = 0u64;
            for (p, base_unit) in units.iter().enumerate() {
                let pkt_off = u64::from_le_bytes(
                    bm[BATCH_OFF_PKT_OFFS + p * 8..BATCH_OFF_PKT_OFFS + (p + 1) * 8]
                        .try_into()
                        .unwrap(),
                ) as usize;
                let total_pkt = u64::from_le_bytes(
                    bm[BATCH_OFF_PKT_LENS + p * 8..BATCH_OFF_PKT_LENS + (p + 1) * 8]
                        .try_into()
                        .unwrap(),
                ) as usize;
                let orig_crc = u32::from_le_bytes(
                    bm[BATCH_OFF_ORIG_CRC + p * 4..BATCH_OFF_ORIG_CRC + (p + 1) * 4]
                        .try_into()
                        .unwrap(),
                );
                let stored_crc = u32::from_le_bytes(
                    bm[BATCH_OFF_STORED_CRC + p * 4..BATCH_OFF_STORED_CRC + (p + 1) * 4]
                        .try_into()
                        .unwrap(),
                );
                let mut unit = base_unit.clone();
                unit.original_length = n as u64;
                if total_pkt >= n {
                    let slice = &pinned[pkt_off..pkt_off + n];
                    unit.representation = Representation::Literal as i32;
                    unit.profile.clear();
                    unit.stored_length = n as u64;
                    unit.stored_crc = 0;
                    unit.original_crc = 0;
                    unit.raw_xxh3 = flare_selene::record::checksum(slice);
                    stored_total += n as u64;
                    f(unit, slice);
                } else {
                    let slice = &pinned[pkt_off..pkt_off + total_pkt];
                    unit.representation = Representation::Packet as i32;
                    unit.profile = profile.to_vec();
                    unit.stored_length = total_pkt as u64;
                    unit.stored_crc = stored_crc;
                    unit.original_crc = orig_crc;
                    unit.raw_xxh3 = 0;
                    stored_total += total_pkt as u64;
                    f(unit, slice);
                }
            }
            stored_total
        }

        /// Encode one or more page batches in an overlapped host/GPU pipeline:
        /// overlaps batch `N`'s CPU slab copy and literal hashing (`f`) with batch `N+1`'s GPU
        /// execution on `stream` using the single existing pinned host staging buffer
        /// (0 extra GPU VRAM and 0 extra pinned host RAM; batch `N+1`'s D2H is only queued after
        /// batch `N`'s synchronous host emission finishes).
        #[allow(clippy::too_many_arguments)]
        pub fn encode_pages_batches_with<F>(
            &mut self,
            plan: &GpuPlan,
            batches: &[(&[PackedUnit], &[Vec<DeviceSpan>])],
            profile: &[u8; 32],
            stream: &Stream,
            event: &Event,
            mut f: F,
        ) -> Result<u64>
        where
            F: FnMut(PackedUnit, &[u8]),
        {
            if batches.is_empty() {
                return Ok(0);
            }
            for &(units, spans) in batches {
                if units.len() != spans.len() {
                    return Err(invalid("GPU page batch unit/span count mismatch"));
                }
            }
            let n = plan.original_bytes();
            let mut stored_total = 0u64;
            let mut saved_meta = [0u8; BATCH_OFF_TOTAL_BYTES + 8];
            let pinned_cap = self.h_pinned.bytes().len();

            self.launch_encode_pages_batch_kernels(plan, batches[0].1, stream)?;
            for b in 0..batches.len() {
                sync_stream(stream, event)?;
                saved_meta.copy_from_slice(&self.h_batch_meta.bytes()[..BATCH_OFF_TOTAL_BYTES + 8]);
                let total_batch_bytes = u64::from_le_bytes(
                    saved_meta[BATCH_OFF_TOTAL_BYTES..BATCH_OFF_TOTAL_BYTES + 8]
                        .try_into()
                        .unwrap(),
                ) as usize;
                if total_batch_bytes > pinned_cap {
                    return Err(invalid("GPU batch output exceeds pinned staging buffer"));
                }
                if total_batch_bytes > 0 {
                    unsafe {
                        stream
                            .copy_dtoh_async(
                                self.h_pinned.address() as _,
                                self.buf_b.ptr(),
                                total_batch_bytes,
                            )
                            .map_err(io_err)?;
                    }
                    stream.record(&self.d2h_event).map_err(io_err)?;
                }
                // Queue next batch's GPU kernels immediately in stream order (executes right after
                // batch `b`'s D2H finishes reading `buf_b`), overlapping GPU work with CPU emission.
                // Batch `b+1` only writes device buffers and `d_batch_meta` until the next loop
                // iteration, so `self.h_pinned` stays exclusively owned by host emission below.
                if b + 1 < batches.len() {
                    self.launch_encode_pages_batch_kernels(plan, batches[b + 1].1, stream)?;
                }
                if total_batch_bytes > 0 {
                    self.d2h_event.synchronize().map_err(io_err)?;
                }
                let pinned = &self.h_pinned.bytes()[..total_batch_bytes];
                stored_total +=
                    Self::emit_encoded_batch(&saved_meta, pinned, batches[b].0, n, profile, &mut f);
            }
            Ok(stored_total)
        }

        /// Encode a batch of up to `page_batch_cap` non-matrix KV pages in one GPU wave
        /// (2 stream syncs and 1 contiguous D2H DMA for the entire batch).
        #[cfg(all(test, feature = "runtime-core"))]
        #[allow(clippy::too_many_arguments)]
        pub fn encode_pages_batch_with<F>(
            &mut self,
            plan: &GpuPlan,
            units: &[PackedUnit],
            profile: &[u8; 32],
            spans_per_page: &[Vec<DeviceSpan>],
            stream: &Stream,
            event: &Event,
            f: F,
        ) -> Result<u64>
        where
            F: FnMut(PackedUnit, &[u8]),
        {
            if units.is_empty() {
                return Ok(0);
            }
            self.encode_pages_batches_with(plan, &[(units, spans_per_page)], profile, stream, event, f)
        }

        /// Verify stored CRC32 on GPU, decode a batch of `GKV1/v2` non-matrix page packets on GPU,
        /// verify every original CRC32 on GPU, and scatter all verified pages into GPU KV cache.
        #[allow(clippy::too_many_arguments)]
        pub fn decode_pages_batch_to_device(
            &mut self,
            plan: &GpuPlan,
            packets: &[(&[&[u8]], u32, u32)],
            spans_per_page: &[Vec<DeviceSpan>],
            stream: &Stream,
            event: &Event,
        ) -> Result<()> {
            let batch = packets.len();
            if batch == 0 {
                return Ok(());
            }
            if batch != spans_per_page.len()
                || batch > self.page_batch_cap
                || plan.num_mats() != 0
                || plan.num_chunks() == 0
            {
                return Err(invalid("GPU page decode batch exceeds codec batch capacity"));
            }
            let bf = self
                .batch_fns
                .ok_or_else(|| invalid("loaded GPU codec cubin lacks batched page kernels"))?;
            let n = plan.original_bytes();
            let nc = plan.num_chunks();
            let total_chunks = nc * batch;
            if total_chunks > self.max_chunks {
                return Err(invalid("GPU batched decode exceeds chunk workspace capacity"));
            }
            let dp_idx = self.device_plan_index(plan)?;

            // 1. Verify stored CRC32 on CPU before touching PCIe/VRAM, copy all `batch` packets
            //    contiguously into `h_pinned`, and parse packet headers.
            let mut pkt_offs = Vec::with_capacity(batch);
            let mut pkt_lens = Vec::with_capacity(batch);
            let mut parsed = Vec::with_capacity(batch);
            let mut total_bytes = 0usize;
            for &(chunks, stored_crc, _) in packets {
                if crc32_chunks(chunks) != stored_crc {
                    return Err(Error::new(ErrorKind::InvalidData, Checksum::Stored));
                }
                let len: usize = chunks.iter().map(|c| c.len()).sum();
                if len < plan.metadata_bytes() || len > plan.max_packet() {
                    return Err(invalid("invalid GKV1 packet length"));
                }
                pkt_offs.push(total_bytes as u64);
                pkt_lens.push(len as u64);
                let dst = &mut self.h_pinned.bytes_mut()[total_bytes..total_bytes + len];
                let mut pos = 0usize;
                for c in chunks {
                    dst[pos..pos + c.len()].copy_from_slice(c);
                    pos += c.len();
                }
                parsed.push(plan.parse_packet_header(&self.h_pinned.bytes()[total_bytes..total_bytes + len])?);
                total_bytes += len;
            }
            {
                let bm = self.h_batch_meta.bytes_mut();
                write_u64s(&mut bm[BATCH_OFF_PKT_OFFS..BATCH_OFF_PKT_OFFS + batch * 8], &pkt_offs);
                write_u64s(&mut bm[BATCH_OFF_PKT_LENS..BATCH_OFF_PKT_LENS + batch * 8], &pkt_lens);
            }

            // 2. Upload all verified packets (`buf_a`) + offsets/lens (`d_batch_meta`).
            let d_pkt_offs = self.d_batch_meta.ptr() + BATCH_OFF_PKT_OFFS as u64;
            unsafe {
                stream
                    .copy_htod_async(self.buf_a.ptr(), self.h_pinned.address() as _, total_bytes)
                    .map_err(io_err)?;
                stream
                    .copy_htod_async(
                        self.d_batch_meta.ptr(),
                        self.h_batch_meta.address() as _,
                        BATCH_OFF_PKT_LENS + batch * 8,
                    )
                    .map_err(io_err)?;
            }

            // Pack all `batch * nc` chunk directories into `h_meta`.
            let off_ch_words = total_chunks * 4;
            {
                let meta = self.h_meta.bytes_mut();
                for (p, hdr) in parsed.into_iter().enumerate() {
                    let base = p * nc * 4;
                    write_u32s(&mut meta[base..base + nc * 4], &hdr.chunk_pkt_offs);
                    write_u32s(
                        &mut meta[off_ch_words + base..off_ch_words + base + nc * 4],
                        &hdr.chunk_words,
                    );
                }
            }

            unsafe {
                stream
                    .memset_d8_async(self.d_scalars.ptr() + 16, 0, 4)
                    .map_err(io_err)?;
                let hptr = self.h_meta.address() as *const u8;
                stream
                    .copy_htod_async(self.d_chunk_pkt_offs.ptr(), hptr as _, total_chunks * 4)
                    .map_err(io_err)?;
                stream
                    .copy_htod_async(
                        self.d_chunk_words.ptr(),
                        hptr.add(off_ch_words) as _,
                        total_chunks * 4,
                    )
                    .map_err(io_err)?;
            }

            // 3. Batched rANS decode (`buf_a` -> `buf_b`).
            let ans_stride = plan.ans_total_bytes();
            {
                let dp = &self.dev_plans[dp_idx];
                let (
                    mut d_pkt,
                    mut p_base,
                    mut ch_po,
                    mut ch_w,
                    mut ans_o,
                    mut ch_rsz,
                    mut ans_out,
                    mut n_ch,
                    mut astr,
                    mut b,
                    mut err_p,
                ) = (
                    self.buf_a.ptr(),
                    d_pkt_offs,
                    self.d_chunk_pkt_offs.ptr(),
                    self.d_chunk_words.ptr(),
                    dp.d_chunk_offs.ptr(),
                    dp.d_chunk_sizes.ptr(),
                    self.buf_b.ptr(),
                    nc as c_int,
                    ans_stride,
                    batch as c_int,
                    self.d_scalars.ptr() + 16,
                );
                launch_1d(
                    bf.fn_rans_dec_batch,
                    total_chunks as u32,
                    256,
                    0,
                    stream,
                    &mut [
                        &mut d_pkt as *mut _ as _,
                        &mut p_base as *mut _ as _,
                        &mut ch_po as *mut _ as _,
                        &mut ch_w as *mut _ as _,
                        &mut ans_o as *mut _ as _,
                        &mut ch_rsz as *mut _ as _,
                        &mut ans_out as *mut _ as _,
                        &mut n_ch as *mut _ as _,
                        &mut astr as *mut _ as _,
                        &mut b as *mut _ as _,
                        &mut err_p as *mut _ as _,
                    ],
                    "launch(kv_rans32_decode_batch_gpu)",
                )?;
            }

            // 4. Uncoalesce (`buf_b` -> `buf_c`) + inverse transforms (`buf_c` -> `buf_a`).
            {
                let dp = &self.dev_plans[dp_idx];
                let n_coal = plan.coalesce().len();
                if n_coal > 0 {
                    let (mut dst, mut src, mut doffs, mut soffs, mut lens, mut nops, mut dstr, mut sstr, mut b) = (
                        self.buf_c.ptr(),
                        self.buf_b.ptr(),
                        dp.d_coal_trans.ptr(),
                        dp.d_coal_ans.ptr(),
                        dp.d_coal_len.ptr(),
                        n_coal as c_int,
                        n,
                        ans_stride,
                        batch as c_int,
                    );
                    launch_1d(
                        bf.fn_copy_batch_rel_strided,
                        (n_coal * batch) as u32,
                        128,
                        0,
                        stream,
                        &mut [
                            &mut dst as *mut _ as _,
                            &mut src as *mut _ as _,
                            &mut doffs as *mut _ as _,
                            &mut soffs as *mut _ as _,
                            &mut lens as *mut _ as _,
                            &mut nops as *mut _ as _,
                            &mut dstr as *mut _ as _,
                            &mut sstr as *mut _ as _,
                            &mut b as *mut _ as _,
                        ],
                        "launch(kv_copy_batch_rel_strided_gpu:dec)",
                    )?;
                }
                if dp.num_transforms > 0 {
                    let (mut dst, mut src, mut descs, mut nd, mut pstride, mut b, mut dec) = (
                        self.buf_a.ptr(),
                        self.buf_c.ptr(),
                        dp.d_transforms.ptr(),
                        dp.num_transforms as c_int,
                        n,
                        batch as c_int,
                        1i32,
                    );
                    launch_1d(
                        bf.fn_transform_batch,
                        (dp.num_transforms * batch) as u32,
                        256,
                        0,
                        stream,
                        &mut [
                            &mut dst as *mut _ as _,
                            &mut src as *mut _ as _,
                            &mut descs as *mut _ as _,
                            &mut nd as *mut _ as _,
                            &mut pstride as *mut _ as _,
                            &mut b as *mut _ as _,
                            &mut dec as *mut _ as _,
                        ],
                        "launch(kv_transform_batch_gpu:dec)",
                    )?;
                }
            }

            // 5. Verify `err_flag` and all `batch` `original_crc` values BEFORE scattering to live VRAM.
            {
                let (mut base, mut offs, mut lens, mut fstride, mut flen, mut mlen, mut out, mut b) = (
                    self.buf_a.ptr(),
                    0u64,
                    0u64,
                    n as u64,
                    n as u64,
                    0u64,
                    self.d_batch_meta.ptr() + BATCH_OFF_ORIG_CRC as u64,
                    batch as c_int,
                );
                launch_1d(
                    bf.fn_crc32_batch,
                    batch as u32,
                    256,
                    0,
                    stream,
                    &mut [
                        &mut base as *mut _ as _,
                        &mut offs as *mut _ as _,
                        &mut lens as *mut _ as _,
                        &mut fstride as *mut _ as _,
                        &mut flen as *mut _ as _,
                        &mut mlen as *mut _ as _,
                        &mut out as *mut _ as _,
                        &mut b as *mut _ as _,
                    ],
                    "launch(kv_crc32_batch_gpu:dec_orig)",
                )?;
            }
            unsafe {
                stream
                    .copy_dtoh_async(
                        (self.h_batch_meta.address() + BATCH_OFF_ORIG_CRC) as _,
                        self.d_batch_meta.ptr() + BATCH_OFF_ORIG_CRC as u64,
                        batch * 4,
                    )
                    .map_err(io_err)?;
                stream
                    .copy_dtoh_async(
                        (self.h_scalars.address() + 16) as _,
                        self.d_scalars.ptr() + 16,
                        4,
                    )
                    .map_err(io_err)?;
            }
            sync_stream(stream, event)?;
            let err_val =
                u32::from_le_bytes(self.h_scalars.bytes()[16..20].try_into().unwrap());
            if err_val != 0 {
                return Err(invalid("invalid compressed payload in GKV1 packet"));
            }
            let bm = self.h_batch_meta.bytes();
            for (p, &(_, _, expected_orig_crc)) in packets.iter().enumerate() {
                let got = u32::from_le_bytes(
                    bm[BATCH_OFF_ORIG_CRC + p * 4..BATCH_OFF_ORIG_CRC + (p + 1) * 4]
                        .try_into()
                        .unwrap(),
                );
                if got != expected_orig_crc {
                    return Err(Error::new(ErrorKind::InvalidData, Checksum::Original));
                }
            }

            // 6. Scatter all verified pages from `buf_a` into destination GPU KV cache planes.
            self.transfer_spans_batch(self.buf_a.ptr(), n, spans_per_page, false, stream)?;
            sync_stream(stream, event)
        }
    }

    #[cfg(all(test, feature = "runtime-core"))]
    mod tests {
        use super::*;
        use crate::runtime_core::driver::Driver;
        use flare_selene::shared::codec::{Field, Plan};

        #[test]
        #[ignore = "exercises pure-CUDA GPU KV codec on real CUDA GPU"]
        fn gpu_codec_round_trips_all_encodings_and_rejects_corruption() {
            let cubin_path = std::env::var("FLARE_TEST_KV_CODEC_CUBIN")
                .expect("set FLARE_TEST_KV_CODEC_CUBIN to a compiled kv_codec_gpu.cubin");
            let cubin = std::fs::read(&cubin_path).expect("read cubin");

            let driver = Driver::load().unwrap();
            let mut ctx = std::ptr::null_mut();
            unsafe {
                driver.check("cuInit", (driver.init)(0)).unwrap();
                driver
                    .check(
                        "cuDevicePrimaryCtxRetain",
                        (driver.primary_ctx_retain)(&mut ctx, 0),
                    )
                    .unwrap();
            }
            struct PrimaryCtx<'a>(&'a Driver);
            impl Drop for PrimaryCtx<'_> {
                fn drop(&mut self) {
                    unsafe {
                        let _ = (self.0.primary_ctx_release)(0);
                    }
                }
            }
            let _guard = PrimaryCtx(&driver);
            let stream = Stream::open_lowest_priority(ctx as usize, 0).unwrap();
            let event = stream.blocking_event().unwrap();

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

            let mut src = vec![0u8; total];
            for m in 0..2 {
                for c in 0..128 {
                    for r in 0..128 {
                        let v = ((r as f32 * 0.013 + m as f32).sin() * (c as f32 * 0.017).cos())
                            .to_bits();
                        let idx = m * 65536 + (c * 128 + r) * 4;
                        src[idx..idx + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
            for (i, b) in src[65536 * 2..].iter_mut().enumerate() {
                *b = ((i / 17) % 19) as u8;
            }

            let mut src_mem = stream.mapped_host_buffer(total).unwrap();
            let mut dst_mem = stream.mapped_host_buffer(total).unwrap();
            src_mem.bytes_mut().copy_from_slice(&src);
            dst_mem.bytes_mut().fill(0xA5);
            let src_addr = stream.mapped_address(&src_mem).unwrap() as u64;
            let dst_addr = stream.mapped_address(&dst_mem).unwrap() as u64;

            let mut codec = GpuRankCodec::new(&gplan, &gplan, 8, &cubin, &stream, &event).unwrap();
            let est = GpuPlan::workspace_bytes_estimate_pair(&gplan, &gplan);
            assert!(codec.allocated_device_bytes() <= est);

            let profile = [0x42u8; 32];
            let src_spans = [DeviceSpan {
                device: src_addr,
                offset: 0,
                length: total,
            }];
            let dst_spans = [DeviceSpan {
                device: dst_addr,
                offset: 0,
                length: total,
            }];
            let (unit, pkt) = codec
                .encode_from_device_with(
                    &gplan,
                    PackedUnit::default(),
                    &profile,
                    &src_spans,
                    &stream,
                    &event,
                    |u, bytes| (u, bytes.to_vec()),
                )
                .unwrap();
            assert_eq!(unit.representation, Representation::Packet as i32);
            assert!(pkt.len() < total);

            // GPU decode bit-exact
            codec
                .decode_to_device(
                    &gplan,
                    &[&pkt],
                    &dst_spans,
                    unit.stored_crc,
                    unit.original_crc,
                    &stream,
                    &event,
                )
                .unwrap();
            assert_eq!(dst_mem.bytes(), &src[..]);

            // CPU cross-decode bit-exact
            let mut cpu_out = vec![0u8; total];
            gplan
                .decode_checked_cpu(&[&pkt], &mut cpu_out, unit.stored_crc, unit.original_crc, None)
                .unwrap();
            assert_eq!(cpu_out, src);

            // Corrupted stored CRC rejected before touching dst
            dst_mem.bytes_mut().fill(0x5A);
            let mut bad_pkt = pkt.clone();
            bad_pkt[pkt.len() / 2] ^= 0xFF;
            let err = codec
                .decode_to_device(
                    &gplan,
                    &[&bad_pkt],
                    &dst_spans,
                    unit.stored_crc,
                    unit.original_crc,
                    &stream,
                    &event,
                )
                .unwrap_err();
            assert!(err.to_string().contains("stored"));
            assert!(dst_mem.bytes().iter().all(|&b| b == 0x5A));

            // Incompressible random input falls back to Literal
            let mut x = 0x9e37_79b9_7f4a_7c15u64;
            for b in src_mem.bytes_mut().iter_mut() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *b = x as u8;
            }
            let (lit_unit, lit_bytes) = codec
                .encode_from_device_with(
                    &gplan,
                    PackedUnit::default(),
                    &profile,
                    &src_spans,
                    &stream,
                    &event,
                    |u, bytes| (u, bytes.to_vec()),
                )
                .unwrap();
            assert_eq!(lit_unit.representation, Representation::Literal as i32);
            assert_eq!(lit_bytes, src_mem.bytes());

            // Verify teardown from a thread with no current CUDA context succeeds cleanly.
            std::thread::spawn(move || drop(codec)).join().unwrap();

            // Multi-page batched encode/decode parity, literal fallback, and corruption rejection.
            let mut p_off = 0usize;
            let mut p_fields = Vec::new();
            for (enc, len) in [
                (Encoding::LatentFp4, 18_432),
                (Encoding::Fp8Page512, 32_768),
                (Encoding::Bfloat16, 4096),
                (Encoding::Bytes, 512),
            ] {
                p_fields.push(Field {
                    offset: p_off,
                    length: len,
                    encoding: enc,
                });
                p_off += len;
            }
            let p_total = p_off.next_multiple_of(4096);
            let p_plan = Plan::new(p_total, &p_fields).unwrap();
            let p_gplan = GpuPlan::from_plan(&p_plan).unwrap();
            let mut bcodec =
                GpuRankCodec::new(&p_gplan, &gplan, 4, &cubin, &stream, &event).unwrap();
            assert!(bcodec.page_batch_cap() >= 8);
            let b_est = GpuPlan::workspace_bytes_estimate_pair(&p_gplan, &gplan);
            assert!(bcodec.allocated_device_bytes() <= b_est);

            const N_PAGES: usize = 8;
            let mut b_src = stream.mapped_host_buffer(p_total * N_PAGES).unwrap();
            let mut b_dst = stream.mapped_host_buffer(p_total * N_PAGES).unwrap();
            let mut rng = 0x1234_5678_9abc_def0u64;
            for p in 0..N_PAGES {
                let slice = &mut b_src.bytes_mut()[p * p_total..(p + 1) * p_total];
                if p == 3 {
                    // Incompressible random page in the middle of the batch.
                    for b in slice.iter_mut() {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        *b = rng as u8;
                    }
                } else {
                    for (i, b) in slice.iter_mut().enumerate() {
                        *b = (((i + p * 31) / 11) % 23) as u8;
                    }
                }
            }
            b_dst.bytes_mut().fill(0xA5);
            let b_src_addr = stream.mapped_address(&b_src).unwrap() as u64;
            let b_dst_addr = stream.mapped_address(&b_dst).unwrap() as u64;
            let make_spans = |base: u64| -> Vec<Vec<DeviceSpan>> {
                (0..N_PAGES)
                    .map(|p| {
                        let page_base = base + (p * p_total) as u64;
                        // Split into 2 spans to exercise multi-plane gather/scatter.
                        let split = p_total / 2;
                        vec![
                            DeviceSpan {
                                device: page_base,
                                offset: 0,
                                length: split,
                            },
                            DeviceSpan {
                                device: page_base + split as u64,
                                offset: split,
                                length: p_total - split,
                            },
                        ]
                    })
                    .collect()
            };
            let b_src_spans = make_spans(b_src_addr);
            let b_dst_spans = make_spans(b_dst_addr);
            let b_units: Vec<PackedUnit> = (0..N_PAGES)
                .map(|p| PackedUnit {
                    logical_index: p as u64,
                    ..Default::default()
                })
                .collect();
            let mut encoded: Vec<(PackedUnit, Vec<u8>)> = Vec::with_capacity(N_PAGES);
            bcodec
                .encode_pages_batch_with(
                    &p_gplan,
                    &b_units,
                    &profile,
                    &b_src_spans,
                    &stream,
                    &event,
                    |u, bytes| encoded.push((u, bytes.to_vec())),
                )
                .unwrap();
            assert_eq!(encoded.len(), N_PAGES);
            // Multi-batch single-buffer pipeline must be bit-exact with single-batch output
            // across consecutive batches.
            let mut piped: Vec<(PackedUnit, Vec<u8>)> = Vec::with_capacity(N_PAGES);
            let split_batches = [
                (&b_units[0..3], &b_src_spans[0..3]),
                (&b_units[3..6], &b_src_spans[3..6]),
                (&b_units[6..8], &b_src_spans[6..8]),
            ];
            bcodec
                .encode_pages_batches_with(
                    &p_gplan,
                    &split_batches,
                    &profile,
                    &stream,
                    &event,
                    |u, bytes| piped.push((u, bytes.to_vec())),
                )
                .unwrap();
            assert_eq!(piped.len(), N_PAGES);
            for p in 0..N_PAGES {
                assert_eq!(piped[p].0.stored_crc, encoded[p].0.stored_crc);
                assert_eq!(piped[p].0.original_crc, encoded[p].0.original_crc);
                assert_eq!(piped[p].0.raw_xxh3, encoded[p].0.raw_xxh3);
                assert_eq!(piped[p].1, encoded[p].1);
            }
            for (p, (u, bytes)) in encoded.iter().enumerate() {
                if p == 3 {
                    assert_eq!(u.representation, Representation::Literal as i32);
                    assert_eq!(bytes, &b_src.bytes()[p * p_total..(p + 1) * p_total]);
                } else {
                    assert_eq!(u.representation, Representation::Packet as i32);
                    assert!(bytes.len() < p_total);
                    // Bit-exact parity with single-page GPU encode and CPU cross-decode!
                    let (su, sbytes) = bcodec
                        .encode_from_device_with(
                            &p_gplan,
                            b_units[p].clone(),
                            &profile,
                            &b_src_spans[p],
                            &stream,
                            &event,
                            |u, b| (u, b.to_vec()),
                        )
                        .unwrap();
                    assert_eq!(u.stored_crc, su.stored_crc);
                    assert_eq!(u.original_crc, su.original_crc);
                    assert_eq!(bytes, &sbytes);
                    let mut cpu_out = vec![0u8; p_total];
                    p_gplan
                        .decode_checked_cpu(&[bytes], &mut cpu_out, u.stored_crc, u.original_crc, None)
                        .unwrap();
                    assert_eq!(cpu_out, &b_src.bytes()[p * p_total..(p + 1) * p_total]);
                }
            }
            // Batched GPU decode of all compressible pages (skipping literal page 3).
            let comp_indices: Vec<usize> = (0..N_PAGES).filter(|&p| p != 3).collect();
            let comp_slices: Vec<&[u8]> =
                comp_indices.iter().map(|&p| encoded[p].1.as_slice()).collect();
            let comp_packets: Vec<(&[&[u8]], u32, u32)> = comp_indices
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    (
                        std::slice::from_ref(&comp_slices[i]),
                        encoded[p].0.stored_crc,
                        encoded[p].0.original_crc,
                    )
                })
                .collect();
            let comp_dst_spans: Vec<Vec<DeviceSpan>> =
                comp_indices.iter().map(|&p| b_dst_spans[p].clone()).collect();
            bcodec
                .decode_pages_batch_to_device(
                    &p_gplan,
                    &comp_packets,
                    &comp_dst_spans,
                    &stream,
                    &event,
                )
                .unwrap();
            for &p in &comp_indices {
                assert_eq!(
                    &b_dst.bytes()[p * p_total..(p + 1) * p_total],
                    &b_src.bytes()[p * p_total..(p + 1) * p_total]
                );
            }
            // Corrupting one packet in a batch must fail stored CRC before touching any destination.
            b_dst.bytes_mut().fill(0x5A);
            let mut bad_batch = comp_packets.clone();
            bad_batch[2].1 ^= 1;
            let err = bcodec
                .decode_pages_batch_to_device(
                    &p_gplan,
                    &bad_batch,
                    &comp_dst_spans,
                    &stream,
                    &event,
                )
                .unwrap_err();
            assert!(err.to_string().contains("stored"));
            assert!(b_dst.bytes().iter().all(|&b| b == 0x5A));
            drop(bcodec);
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::*;
    pub struct GpuRankCodec;
    impl GpuRankCodec {
        pub fn new(
            _: &GpuPlan,
            _: &GpuPlan,
            _: usize,
            _: &[u8],
            _: &Stream,
            _: &Event,
        ) -> Result<Self> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "GPU KV codec requires Linux with CUDA",
            ))
        }
        pub fn encode_from_device_with<F, R>(
            &mut self,
            _: &GpuPlan,
            _: PackedUnit,
            _: &[u8; 32],
            _: &[DeviceSpan],
            _: &Stream,
            _: &Event,
            _: F,
        ) -> Result<R>
        where
            F: FnOnce(PackedUnit, &[u8]) -> R,
        {
            Err(Error::new(
                ErrorKind::Unsupported,
                "GPU KV codec requires Linux",
            ))
        }
        #[allow(clippy::too_many_arguments)]
        pub fn decode_to_device(
            &mut self,
            _: &GpuPlan,
            _: &[&[u8]],
            _: &[DeviceSpan],
            _: u32,
            _: u32,
            _: &Stream,
            _: &Event,
        ) -> Result<()> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "GPU KV codec requires Linux",
            ))
        }
        pub fn page_batch_cap(&self) -> usize {
            1
        }
        #[allow(clippy::too_many_arguments)]
        pub fn encode_pages_batches_with<F>(
            &mut self,
            _: &GpuPlan,
            _: &[(&[PackedUnit], &[Vec<DeviceSpan>])],
            _: &[u8; 32],
            _: &Stream,
            _: &Event,
            _: F,
        ) -> Result<u64>
        where
            F: FnMut(PackedUnit, &[u8]),
        {
            Err(Error::new(
                ErrorKind::Unsupported,
                "GPU KV codec requires Linux",
            ))
        }
        #[allow(clippy::too_many_arguments)]
        pub fn decode_pages_batch_to_device(
            &mut self,
            _: &GpuPlan,
            _: &[(&[&[u8]], u32, u32)],
            _: &[Vec<DeviceSpan>],
            _: &Stream,
            _: &Event,
        ) -> Result<()> {
            Err(Error::new(
                ErrorKind::Unsupported,
                "GPU KV codec requires Linux",
            ))
        }
    }
}

pub(super) use imp::GpuRankCodec;
