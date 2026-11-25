// SPDX-License-Identifier: Apache-2.0

use super::{gpu_codec::GpuRankCodec, *};
use crate::cuda::PinnedHost;
use flare_selene::{
    codec_copy::{DecodeCounters, DecodeStats, Window},
    offload::KvCompression,
    shared::{
        codec::{Checksum, Plan},
        gpu_codec::{DeviceSpan, GpuPlan},
        packed,
        wire::{
            PackedUnit,
            packed_unit::{Kind, Representation},
        },
    },
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub(super) type SharedGpuCodecs = Arc<Mutex<HashMap<(usize, usize), Arc<Mutex<GpuRankCodec>>>>>;

fn cpu_pool() -> flare_selene::Result<&'static rayon::ThreadPool> {
    flare_selene::shared::codec::cpu_pool()
        .map_err(|e| flare_selene::geometry::invalid(e.to_string()))
}

struct CpuCodec {
    page: Plan,
    row: Plan,
    gpu_page: GpuPlan,
    gpu_row: GpuPlan,
    staging: PinnedHost,
    staging_alias: u64,
    encode_buf: Vec<u8>,
}

struct GpuCodec {
    codec: Arc<Mutex<GpuRankCodec>>,
    page_batch_cap: usize,
}

pub(super) struct RankCodec {
    counters: DecodeCounters,
    write_mode: KvCompression,
    cpu_profile: [u8; 32],
    gpu_profile: [u8; 32],
    cpu: CpuCodec,
    gpu: Option<GpuCodec>,
}

impl RankCodec {
    fn original_bytes(&self, row: bool) -> usize {
        if row {
            self.cpu.row.original_bytes()
        } else {
            self.cpu.page.original_bytes()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        rank: usize,
        context: usize,
        stream: &Stream,
        event: &Event,
        geometry: &CacheGeometry,
        counters: DecodeCounters,
        gpu_cubin: Option<&[u8]>,
        shared_gpu: Option<&SharedGpuCodecs>,
    ) -> Result<Self> {
        let ([page, row], cpu_profile) = flare_selene::shared::codec_layout::plans_and_profile(
            geometry,
            flare_selene::shared::codec_layout::REVISION_CPU,
        )
        .map_err(|e| invalid(e.to_string()))?;
        let gpu_profile = flare_selene::shared::codec_layout::profile(
            geometry,
            flare_selene::shared::codec_layout::REVISION_GPU,
        )
        .map_err(|e| invalid(e.to_string()))?;

        // Both Cpu and Gpu modes keep one CPU pinned staging buffer so Gpu mode can
        // transparently decode existing ZKV1 checkpoints and Literal shares.
        let bytes = page.original_bytes().max(row.original_bytes());
        let staging = stream.mapped_host_buffer(bytes)?;
        let staging_alias = stream.mapped_address(&staging)? as u64;
        let encode_buf = if geometry.placement == KvCompression::Cpu {
            vec![0; page.max_packet().max(row.max_packet())]
        } else {
            Vec::new()
        };

        // Always derive lightweight CPU-side `GpuPlan` metadata (0 GPU VRAM) so both
        // `cpu` and `gpu` modes can natively decode `GKV1/v2` checkpoints.
        let gpu_page = GpuPlan::from_plan(&page).map_err(|e| invalid(e.to_string()))?;
        let gpu_row = GpuPlan::from_plan(&row).map_err(|e| invalid(e.to_string()))?;

        // Only GPU mode loads the embedded GPU codec cubin and allocates the shared
        // per-rank GPU workspace; CPU mode allocates no GPU VRAM.
        let gpu = match geometry.placement {
            KvCompression::Gpu => {
                let cubin = gpu_cubin
                    .ok_or_else(|| invalid("missing embedded GPU codec module for rank"))?;
                let max_spans = geometry.planes.len().max(1);
                let make_codec = || -> Result<Arc<Mutex<GpuRankCodec>>> {
                    Ok(Arc::new(Mutex::new(
                        GpuRankCodec::new(&gpu_page, &gpu_row, max_spans, cubin, stream, event)
                            .map_err(|e| invalid(e.to_string()))?,
                    )))
                };
                let codec = if let Some(shared) = shared_gpu {
                    let mut map = shared.lock().unwrap();
                    if let Some(existing) = map.get(&(context, rank)) {
                        existing.clone()
                    } else {
                        let created = make_codec()?;
                        map.insert((context, rank), created.clone());
                        created
                    }
                } else {
                    make_codec()?
                };
                let page_batch_cap = codec.lock().unwrap().page_batch_cap();
                Some(GpuCodec {
                    codec,
                    page_batch_cap,
                })
            }
            KvCompression::Cpu | KvCompression::Off => None,
        };

        Ok(Self {
            counters,
            write_mode: geometry.placement,
            cpu_profile,
            gpu_profile,
            cpu: CpuCodec {
                page,
                row,
                gpu_page,
                gpu_row,
                staging,
                staging_alias,
                encode_buf,
            },
            gpu,
        })
    }

    pub(super) fn run(
        &mut self,
        rank: usize,
        geometry: &CacheGeometry,
        layout: &RankLayout,
        stream: &Stream,
        event: &Event,
        job: &Job,
        window: &Window,
    ) -> Result<u64> {
        let mut stats = DecodeStats::default();
        let result = match job.direction {
            Direction::In => {
                self.restore_window(rank, geometry, layout, stream, event, job, window, &mut stats)
            }
            Direction::Out => {
                self.capture_window(rank, geometry, layout, stream, event, job, window, &mut stats)
            }
        };
        {
            let mut totals = self.counters.lock().unwrap();
            totals.transform_ns += stats.transform_ns;
            totals.stored_bytes += stats.stored_bytes;
            match &result {
                Ok(_) => {
                    totals.original_bytes +=
                        self.original_bytes(false) as u64 * job.pages.len() as u64;
                    if job.row.is_some() {
                        totals.original_bytes += self.original_bytes(true) as u64;
                    }
                }
                Err(error) => {
                    totals.failed_windows += 1;
                    if let Some(cause) = error.carried() {
                        match cause.downcast_ref::<Checksum>() {
                            Some(Checksum::Stored) => totals.stored_failures += 1,
                            Some(Checksum::Original) => totals.original_failures += 1,
                            None => {}
                        }
                    }
                }
            }
        }
        if result.is_err() {
            stream.synchronize()?;
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_window(
        &mut self,
        rank: usize,
        geometry: &CacheGeometry,
        layout: &RankLayout,
        stream: &Stream,
        event: &Event,
        job: &Job,
        window: &Window,
        stats: &mut DecodeStats,
    ) -> Result<u64> {
        let capture = window
            .capture
            .as_ref()
            .ok_or_else(|| invalid("capture job missing capture target"))?;
        reject_duplicate_pages(job)?;
        let first = window.first;
        let mut transferred = 0u64;
        let mut stored = 0u64;
        match self.write_mode {
            KvCompression::Gpu => {
                let page_bytes = self.original_bytes(false);
                let page_plan = &self.cpu.gpu_page;
                let gpu = self
                    .gpu
                    .as_mut()
                    .ok_or_else(|| invalid("GPU codec unavailable"))?;
                let batch_cap = if page_plan.num_mats() == 0 && page_plan.num_chunks() > 0 {
                    gpu.page_batch_cap
                } else {
                    1
                };
                if batch_cap > 1 && job.pages.len() > 1 {
                    // Pipeline up to 4 batches per lock acquisition so host slab copies overlap
                    // with GPU encoding while bounding mutex hold time and span-list allocation
                    // so concurrent foreground restores can still acquire the codec between waves.
                    const MAX_PIPELINE_BATCHES: usize = 4;
                    let wave_pages = batch_cap * MAX_PIPELINE_BATCHES;
                    let mut page_offset = 0usize;
                    for wave in job.pages.chunks(wave_pages) {
                        let mut owned_batches: Vec<(Vec<PackedUnit>, Vec<Vec<DeviceSpan>>)> =
                            Vec::with_capacity(wave.len().div_ceil(batch_cap));
                        for chunk in wave.chunks(batch_cap) {
                            let mut units = Vec::with_capacity(chunk.len());
                            let mut spans_per_page = Vec::with_capacity(chunk.len());
                            for (i, (page, _)) in chunk.iter().enumerate() {
                                spans_per_page.push(device_spans(
                                    geometry,
                                    layout,
                                    false,
                                    u64::from(page.0),
                                )?);
                                units.push(PackedUnit {
                                    kind: Kind::Page as i32,
                                    logical_index: first + (page_offset + i) as u64,
                                    stored_rank: rank as u32,
                                    ..Default::default()
                                });
                            }
                            owned_batches.push((units, spans_per_page));
                            page_offset += chunk.len();
                        }
                        let start = Instant::now();
                        let batch_refs: Vec<(&[PackedUnit], &[Vec<DeviceSpan>])> = owned_batches
                            .iter()
                            .map(|(u, s)| (u.as_slice(), s.as_slice()))
                            .collect();
                        let stored_len = {
                            let mut guard = gpu.codec.lock().unwrap();
                            let res = guard.encode_pages_batches_with(
                                page_plan,
                                &batch_refs,
                                &self.gpu_profile,
                                stream,
                                event,
                                |unit, bytes| capture.push_slice(unit, bytes),
                            );
                            sync_on_err(stream, res)
                        }
                        .map_err(decode_error)?;
                        stats.transform_ns += start.elapsed().as_nanos() as u64;
                        stored += stored_len;
                    }
                    transferred += (page_bytes * job.pages.len()) as u64;
                } else {
                    for (i, (page, _)) in job.pages.iter().enumerate() {
                        let spans = device_spans(geometry, layout, false, u64::from(page.0))?;
                        let base_unit = PackedUnit {
                            kind: Kind::Page as i32,
                            logical_index: first + i as u64,
                            stored_rank: rank as u32,
                            ..Default::default()
                        };
                        let start = Instant::now();
                        let stored_len = {
                            let mut guard = gpu.codec.lock().unwrap();
                            let res = guard.encode_from_device_with(
                                page_plan,
                                base_unit,
                                &self.gpu_profile,
                                &spans,
                                stream,
                                event,
                                |unit, bytes| {
                                    let slen = unit.stored_length;
                                    capture.push_slice(unit, bytes);
                                    slen
                                },
                            );
                            sync_on_err(stream, res)
                        }
                        .map_err(decode_error)?;
                        stats.transform_ns += start.elapsed().as_nanos() as u64;
                        stored += stored_len;
                        transferred += page_bytes as u64;
                    }
                }
                if let Some((slot, _)) = &job.row {
                    let n = self.cpu.row.original_bytes();
                    let spans = device_spans(geometry, layout, true, u64::from(slot.0))?;
                    let base_unit = PackedUnit {
                        kind: Kind::Row as i32,
                        logical_index: 0,
                        stored_rank: rank as u32,
                        ..Default::default()
                    };
                    let start = Instant::now();
                    let stored_len = {
                        let mut guard = gpu.codec.lock().unwrap();
                        let res = guard.encode_from_device_with(
                            &self.cpu.gpu_row,
                            base_unit,
                            &self.gpu_profile,
                            &spans,
                            stream,
                            event,
                            |unit, bytes| {
                                let slen = unit.stored_length;
                                capture.push_slice(unit, bytes);
                                slen
                            },
                        );
                        sync_on_err(stream, res)
                    }
                    .map_err(decode_error)?;
                    stats.transform_ns += start.elapsed().as_nanos() as u64;
                    stored += stored_len;
                    transferred += n as u64;
                }
            }
            KvCompression::Cpu | KvCompression::Off => {
                let pool = cpu_pool()?;
                // Batch multiple pages into the existing pinned staging buffer (sized for
                // max(page, row)): one D2H burst + one sync, then parallel Rayon encode.
                let page_bytes = self.cpu.page.original_bytes();
                let batch_cap = (self.cpu.staging.bytes().len() / page_bytes.max(1)).max(1);
                let mut page_offset = 0usize;
                for chunk in job.pages.chunks(batch_cap) {
                    let mut batch_spans = Vec::with_capacity(chunk.len());
                    for (slot_idx, (page, _)) in chunk.iter().enumerate() {
                        let spans = device_spans(geometry, layout, false, u64::from(page.0))?;
                        let slot_start = slot_idx * page_bytes;
                        zero_span_gaps(
                            &mut self.cpu.staging.bytes_mut()[slot_start..slot_start + page_bytes],
                            &spans,
                        );
                        batch_spans.push((slot_start, spans));
                    }
                    copy_spans_batch(
                        stream,
                        event,
                        Direction::Out,
                        &batch_spans,
                        self.cpu.staging_alias,
                    )?;
                    let start = Instant::now();
                    let staging = self.cpu.staging.bytes();
                    let page_plan = &self.cpu.page;
                    let profile = &self.cpu_profile;
                    let encoded: Result<Vec<(PackedUnit, Vec<u8>)>> = if chunk.len() == 1 {
                        let base_unit = PackedUnit {
                            kind: Kind::Page as i32,
                            logical_index: first + page_offset as u64,
                            stored_rank: rank as u32,
                            ..Default::default()
                        };
                        let (unit, used) = page_plan
                            .encode_capture(
                                base_unit,
                                profile,
                                &staging[..page_bytes],
                                &mut self.cpu.encode_buf,
                                Some(pool),
                            )
                            .map_err(decode_error)?;
                        Ok(vec![(unit, self.cpu.encode_buf[..used].to_vec())])
                    } else {
                        let max_pkt = page_plan.max_packet();
                        pool.install(|| {
                            (0..chunk.len())
                                .into_par_iter()
                                .map(|slot_idx| {
                                    let base_unit = PackedUnit {
                                        kind: Kind::Page as i32,
                                        logical_index: first + (page_offset + slot_idx) as u64,
                                        stored_rank: rank as u32,
                                        ..Default::default()
                                    };
                                    let slot_start = slot_idx * page_bytes;
                                    let mut out = vec![0u8; max_pkt];
                                    let (unit, used) = page_plan
                                        .encode_capture(
                                            base_unit,
                                            profile,
                                            &staging[slot_start..slot_start + page_bytes],
                                            &mut out,
                                            None,
                                        )
                                        .map_err(decode_error)?;
                                    out.truncate(used);
                                    Ok((unit, out))
                                })
                                .collect()
                        })
                    };
                    let encoded = encoded?;
                    stats.transform_ns += start.elapsed().as_nanos() as u64;
                    for (unit, bytes) in encoded {
                        stored += unit.stored_length;
                        capture.push(unit, bytes);
                        transferred += page_bytes as u64;
                    }
                    page_offset += chunk.len();
                }
                if let Some((slot, _)) = &job.row {
                    let n = self.cpu.row.original_bytes();
                    let spans = device_spans(geometry, layout, true, u64::from(slot.0))?;
                    zero_span_gaps(&mut self.cpu.staging.bytes_mut()[..n], &spans);
                    copy_spans_batch(
                        stream,
                        event,
                        Direction::Out,
                        &[(0, spans)],
                        self.cpu.staging_alias,
                    )?;
                    let base_unit = PackedUnit {
                        kind: Kind::Row as i32,
                        logical_index: 0,
                        stored_rank: rank as u32,
                        ..Default::default()
                    };
                    let start = Instant::now();
                    let (unit, used) = self
                        .cpu
                        .row
                        .encode_capture(
                            base_unit,
                            &self.cpu_profile,
                            &self.cpu.staging.bytes()[..n],
                            &mut self.cpu.encode_buf,
                            Some(pool),
                        )
                        .map_err(decode_error)?;
                    stats.transform_ns += start.elapsed().as_nanos() as u64;
                    stored += unit.stored_length;
                    capture.push(unit, self.cpu.encode_buf[..used].to_vec());
                    transferred += n as u64;
                }
            }
        }
        stats.stored_bytes = stored;
        Ok(transferred)
    }

    fn decode_route(&self, u: &PackedUnit, representation: Representation) -> Result<DecodeRoute> {
        /// Minimum non-matrix unit size decoded on GPU during critical-path restores.
        /// Smaller non-matrix pages (< 256 KiB) amortize poorly across GPU stream syncs under
        /// foreground saturation and decode in milliseconds on CPU over async HtoD DMA without
        /// contending for SMs or the shared GPU codec mutex.
        const GPU_DECODE_MIN_BYTES: usize = 256 << 10;

        if representation == Representation::Literal {
            Ok(DecodeRoute::Literal)
        } else if u.profile.as_slice() == self.cpu_profile.as_slice() {
            Ok(DecodeRoute::Cpu)
        } else if u.profile.as_slice() == self.gpu_profile.as_slice() {
            let plan = if u.kind == Kind::Row as i32 {
                &self.cpu.gpu_row
            } else {
                &self.cpu.gpu_page
            };
            if self.gpu.is_some()
                && (plan.num_mats() > 0 || plan.original_bytes() >= GPU_DECODE_MIN_BYTES)
            {
                Ok(DecodeRoute::GpuOnDevice)
            } else {
                Ok(DecodeRoute::GpuOnCpu)
            }
        } else {
            Err(invalid("incompatible codec share"))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_pages_cpu_batch(
        &mut self,
        items: &[(&PackedUnit, DecodeRoute, Vec<&[u8]>)],
        batch_spans: &[(usize, Vec<DeviceSpan>)],
        stream: &Stream,
        event: &Event,
        stats: &mut DecodeStats,
    ) -> Result<()> {
        let pool = cpu_pool()?;
        let page_bytes = self.cpu.page.original_bytes();
        let staging_slice = &mut self.cpu.staging.bytes_mut()[..items.len() * page_bytes];
        let page_plan = &self.cpu.page;
        let gpu_page_plan = &self.cpu.gpu_page;
        let start = Instant::now();
        if items.len() == 1 {
            let (u, route, chunks) = &items[0];
            decode_unit_cpu(
                u,
                *route,
                chunks,
                staging_slice,
                page_plan,
                gpu_page_plan,
                Some(pool),
            )?;
        } else {
            pool.install(|| -> Result<()> {
                staging_slice
                    .par_chunks_mut(page_bytes)
                    .zip(items.par_iter())
                    .try_for_each(|(raw, (u, route, chunks))| -> Result<()> {
                        decode_unit_cpu(u, *route, chunks, raw, page_plan, gpu_page_plan, None)
                    })
            })?;
        }
        stats.transform_ns += start.elapsed().as_nanos() as u64;
        copy_spans_batch(
            stream,
            event,
            Direction::In,
            batch_spans,
            self.cpu.staging_alias,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn restore_window(
        &mut self,
        rank: usize,
        geometry: &CacheGeometry,
        layout: &RankLayout,
        stream: &Stream,
        event: &Event,
        job: &Job,
        window: &Window,
        stats: &mut DecodeStats,
    ) -> Result<u64> {
        reject_duplicate_pages(job)?;
        let first = window.first;
        let mut transferred = 0u64;
        let mut stored = 0u64;
        let page_bytes = self.cpu.page.original_bytes();
        let batch_cap = (self.cpu.staging.bytes().len() / page_bytes.max(1)).max(1);
        let all_pages_cpu_path = job.pages.len() > 1
            && job.pages.iter().enumerate().all(|(i, _)| {
                let idx = first + i as u64;
                window.packets.iter().any(|p| {
                    let u = &p.unit;
                    u.kind == Kind::Page as i32
                        && u.logical_index == idx
                        && (u.stored_rank == rank as u32 || u.replicas.contains(&(rank as u32)))
                        && Representation::try_from(u.representation)
                            .ok()
                            .and_then(|rep| self.decode_route(u, rep).ok())
                            .is_some_and(|route| !matches!(route, DecodeRoute::GpuOnDevice))
                })
            });

        let mut remaining_pages_offset = 0usize;
        if all_pages_cpu_path {
            for chunk in job.pages.chunks(batch_cap) {
                let mut items = Vec::with_capacity(chunk.len());
                let mut batch_spans = Vec::with_capacity(chunk.len());
                for (slot_idx, (page, _)) in chunk.iter().enumerate() {
                    let index = first + (remaining_pages_offset + slot_idx) as u64;
                    let spans = device_spans(geometry, layout, false, u64::from(page.0))?;
                    let (u, representation, chunks) =
                        select_packet(window, false, index, rank, page_bytes)?;
                    let route = self.decode_route(u, representation)?;
                    debug_assert!(!matches!(route, DecodeRoute::GpuOnDevice));
                    stored += u.stored_length;
                    items.push((u, route, chunks));
                    batch_spans.push((slot_idx * page_bytes, spans));
                }
                self.decode_pages_cpu_batch(&items, &batch_spans, stream, event, stats)?;
                transferred += (chunk.len() * page_bytes) as u64;
                remaining_pages_offset += chunk.len();
            }
        }

        // Batch contiguous runs of GKV1 pages on GPU (handles mixed GKV1 + Literal windows cleanly).
        if remaining_pages_offset < job.pages.len()
            && self.cpu.gpu_page.num_mats() == 0
            && self.cpu.gpu_page.num_chunks() > 0
            && self.gpu.is_some()
        {
            let gpu_batch_cap = self.gpu.as_ref().unwrap().page_batch_cap;
            if gpu_batch_cap > 1 {
                let mut i = remaining_pages_offset;
                while i < job.pages.len() {
                    let mut run_units: Vec<&PackedUnit> = Vec::new();
                    let mut run_chunks: Vec<Vec<&[u8]>> = Vec::new();
                    let mut run_spans: Vec<Vec<DeviceSpan>> = Vec::new();
                    let mut run_stored = 0u64;
                    let mut j = i;
                    while j < job.pages.len() && run_chunks.len() < gpu_batch_cap {
                        let index = first + j as u64;
                        let (u, representation, chunks) =
                            select_packet(window, false, index, rank, page_bytes)?;
                        if self.decode_route(u, representation)? != DecodeRoute::GpuOnDevice {
                            break;
                        }
                        let spans =
                            device_spans(geometry, layout, false, u64::from(job.pages[j].0.0))?;
                        run_stored += u.stored_length;
                        run_units.push(u);
                        run_chunks.push(chunks);
                        run_spans.push(spans);
                        j += 1;
                    }
                    if run_chunks.len() > 1 {
                        let start = Instant::now();
                        let gpu_ran = {
                            if let Ok(mut guard) = self.gpu.as_mut().unwrap().codec.try_lock() {
                                let packets: Vec<(&[&[u8]], u32, u32)> = run_chunks
                                    .iter()
                                    .zip(&run_units)
                                    .map(|(ch, u)| (ch.as_slice(), u.stored_crc, u.original_crc))
                                    .collect();
                                let res = guard.decode_pages_batch_to_device(
                                    &self.cpu.gpu_page,
                                    &packets,
                                    &run_spans,
                                    stream,
                                    event,
                                );
                                sync_on_err(stream, res).map_err(decode_error)?;
                                true
                            } else {
                                false
                            }
                        };
                        if gpu_ran {
                            stats.transform_ns += start.elapsed().as_nanos() as u64;
                        } else {
                            // Background GPU capture currently holds the codec mutex: decode this
                            // critical-path restore run on the bounded CPU pool instead of blocking.
                            let mut off = 0usize;
                            while off < run_chunks.len() {
                                let take = (run_chunks.len() - off).min(batch_cap);
                                let items: Vec<(&PackedUnit, DecodeRoute, Vec<&[u8]>)> =
                                    run_units[off..off + take]
                                        .iter()
                                        .zip(&run_chunks[off..off + take])
                                        .map(|(&u, ch)| (u, DecodeRoute::GpuOnCpu, ch.clone()))
                                        .collect();
                                let batch_spans: Vec<(usize, Vec<DeviceSpan>)> = run_spans
                                    [off..off + take]
                                    .iter()
                                    .enumerate()
                                    .map(|(s, sp)| (s * page_bytes, sp.clone()))
                                    .collect();
                                self.decode_pages_cpu_batch(
                                    &items,
                                    &batch_spans,
                                    stream,
                                    event,
                                    stats,
                                )?;
                                off += take;
                            }
                        }
                        stored += run_stored;
                        transferred += (page_bytes * run_chunks.len()) as u64;
                        i = j;
                    } else {
                        // Single GPU or non-GPU page: fall through to the per-unit loop below.
                        break;
                    }
                }
                remaining_pages_offset = i;
            }
        }

        let ids = job
            .pages
            .iter()
            .enumerate()
            .skip(remaining_pages_offset)
            .map(|(i, (page, _))| (false, first + i as u64, u64::from(page.0)))
            .chain(job.row.iter().map(|(slot, _)| (true, 0, u64::from(slot.0))));
        for (row, index, id) in ids {
            let n = self.original_bytes(row);
            let spans = device_spans(geometry, layout, row, id)?;
            let (u, representation, chunks) = select_packet(window, row, index, rank, n)?;
            let mut route = self.decode_route(u, representation)?;
            stored += u.stored_length;
            if route == DecodeRoute::GpuOnDevice {
                let plan = if row {
                    &self.cpu.gpu_row
                } else {
                    &self.cpu.gpu_page
                };
                let start = Instant::now();
                let gpu_ran = {
                    if let Ok(mut guard) = self.gpu.as_mut().unwrap().codec.try_lock() {
                        let res = guard.decode_to_device(
                            plan,
                            &chunks,
                            &spans,
                            u.stored_crc,
                            u.original_crc,
                            stream,
                            event,
                        );
                        sync_on_err(stream, res).map_err(decode_error)?;
                        true
                    } else {
                        false
                    }
                };
                if gpu_ran {
                    stats.transform_ns += start.elapsed().as_nanos() as u64;
                } else {
                    route = DecodeRoute::GpuOnCpu;
                }
            }
            if route != DecodeRoute::GpuOnDevice {
                let cpu_plan = if row { &self.cpu.row } else { &self.cpu.page };
                let gpu_plan = if row {
                    &self.cpu.gpu_row
                } else {
                    &self.cpu.gpu_page
                };
                let raw = &mut self.cpu.staging.bytes_mut()[..n];
                let pool = (route != DecodeRoute::Literal).then(cpu_pool).transpose()?;
                let start = Instant::now();
                decode_unit_cpu(u, route, &chunks, raw, cpu_plan, gpu_plan, pool)?;
                if route != DecodeRoute::Literal {
                    stats.transform_ns += start.elapsed().as_nanos() as u64;
                }
                copy_spans_batch(
                    stream,
                    event,
                    Direction::In,
                    &[(0, spans)],
                    self.cpu.staging_alias,
                )?;
            }
            transferred += n as u64;
        }
        stats.stored_bytes = stored;
        Ok(transferred)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeRoute {
    Literal,
    Cpu,
    GpuOnCpu,
    GpuOnDevice,
}

fn sync_on_err<T, E>(stream: &Stream, res: std::result::Result<T, E>) -> std::result::Result<T, E> {
    if res.is_err() {
        let _ = stream.synchronize();
    }
    res
}

fn select_packet<'a>(
    window: &'a Window,
    row: bool,
    index: u64,
    rank: usize,
    original_bytes: usize,
) -> Result<(&'a PackedUnit, Representation, Vec<&'a [u8]>)> {
    let kind = (if row { Kind::Row } else { Kind::Page }) as i32;
    let packet = window
        .packets
        .iter()
        .find(|p| {
            let u = &p.unit;
            u.kind == kind
                && u.logical_index == index
                && (u.stored_rank == rank as u32 || u.replicas.contains(&(rank as u32)))
        })
        .ok_or_else(|| invalid("missing codec rank share"))?;
    let u = &packet.unit;
    // The pool lease holds the source through the all-rank fence.
    let chunks = unsafe { packet.chunks() };
    let representation = Representation::try_from(u.representation)
        .map_err(|_| invalid("unknown codec share representation"))?;
    if u.original_length != original_bytes as u64
        || u.stored_length != chunks.iter().map(|c| c.len() as u64).sum::<u64>()
    {
        return Err(invalid("incompatible codec share"));
    }
    Ok((u, representation, chunks))
}

fn decode_unit_cpu(
    u: &PackedUnit,
    route: DecodeRoute,
    chunks: &[&[u8]],
    raw: &mut [u8],
    cpu_plan: &Plan,
    gpu_plan: &GpuPlan,
    pool: Option<&rayon::ThreadPool>,
) -> Result<()> {
    match route {
        DecodeRoute::Literal => {
            let mut pos = 0;
            for c in chunks {
                let end = pos + c.len();
                raw.get_mut(pos..end)
                    .ok_or_else(|| invalid("literal length mismatch"))?
                    .copy_from_slice(c);
                pos = end;
            }
            if pos != raw.len() {
                return Err(invalid("literal length mismatch"));
            }
            packed::verify_units(std::slice::from_ref(u), raw)?;
            Ok(())
        }
        DecodeRoute::Cpu => cpu_plan
            .decode_checked(chunks, raw, u.stored_crc, u.original_crc, pool)
            .map_err(decode_error),
        DecodeRoute::GpuOnCpu => gpu_plan
            .decode_checked_cpu(chunks, raw, u.stored_crc, u.original_crc, pool)
            .map_err(decode_error),
        DecodeRoute::GpuOnDevice => Err(invalid("GPU packet dispatched to CPU decoder")),
    }
}

fn reject_duplicate_pages(job: &Job) -> Result<()> {
    if job
        .pages
        .iter()
        .enumerate()
        .any(|(i, (page, _))| job.pages[..i].iter().any(|(prior, _)| prior == page))
    {
        return Err(invalid("codec window repeats a device page"));
    }
    Ok(())
}

fn device_spans(
    geometry: &CacheGeometry,
    layout: &RankLayout,
    row: bool,
    id: u64,
) -> Result<Vec<DeviceSpan>> {
    let capacity = if row {
        geometry.row_capacity
    } else {
        geometry.page_capacity
    };
    if id >= capacity {
        return Err(invalid("codec destination id outside cache"));
    }
    if row {
        Ok(vec![DeviceSpan {
            device: geometry.row.unit_address(id),
            offset: 0,
            length: geometry.row.unit_bytes as usize,
        }])
    } else {
        Ok(geometry
            .planes
            .iter()
            .zip(&layout.plane_offsets)
            .map(|(p, off)| DeviceSpan {
                device: p.unit_address(id),
                offset: *off as usize,
                length: p.unit_bytes as usize,
            })
            .collect())
    }
}

fn decode_error(error: std::io::Error) -> Error {
    if let Some(checksum) = error
        .get_ref()
        .and_then(|cause| cause.downcast_ref::<Checksum>())
    {
        Error::carrying(*checksum)
    } else {
        Error::carrying(error)
    }
}

fn zero_span_gaps(buf: &mut [u8], spans: &[DeviceSpan]) {
    let mut covered = 0usize;
    let mut contiguous = true;
    for s in spans {
        if s.offset != covered {
            contiguous = false;
            break;
        }
        covered += s.length;
    }
    if contiguous && covered == buf.len() {
        return;
    }
    buf.fill(0);
}

fn copy_spans_batch(
    stream: &Stream,
    event: &Event,
    direction: Direction,
    batch: &[(usize, Vec<DeviceSpan>)],
    host: u64,
) -> Result<()> {
    struct Fence<'a>(&'a Stream);
    impl Drop for Fence<'_> {
        fn drop(&mut self) {
            let _ = self.0.synchronize();
        }
    }
    let fence = Fence(stream);
    let mut dsts = Vec::new();
    let mut srcs = Vec::new();
    let mut lens = Vec::new();
    let mut queued = 0usize;
    let flush = |dsts: &mut Vec<u64>, srcs: &mut Vec<u64>, lens: &mut Vec<usize>, queued: &mut usize| -> Result<()> {
        if !lens.is_empty() {
            // SAFETY: Each span is inside this rank's cache and the pinned staging buffer stays mapped.
            unsafe {
                stream.copy_batch(dsts, srcs, lens)?;
            }
            dsts.clear();
            srcs.clear();
            lens.clear();
            *queued = 0;
        }
        Ok(())
    };
    for (slot_start, spans) in batch {
        let base = host + *slot_start as u64;
        for p in spans {
            for start in (0..p.length).step_by(BATCH_BYTES) {
                let n = (p.length - start).min(BATCH_BYTES);
                if !lens.is_empty() && (lens.len() >= BATCH_ENTRIES || queued + n > BATCH_BYTES) {
                    flush(&mut dsts, &mut srcs, &mut lens, &mut queued)?;
                }
                let dev = p.device + start as u64;
                let hst = base + p.offset as u64 + start as u64;
                let (dst, src) = match direction {
                    Direction::In => (dev, hst),
                    Direction::Out => (hst, dev),
                };
                dsts.push(dst);
                srcs.push(src);
                lens.push(n);
                queued += n;
            }
        }
    }
    flush(&mut dsts, &mut srcs, &mut lens, &mut queued)?;
    stream.record(event)?;
    event.synchronize()?;
    std::mem::forget(fence);
    Ok(())
}

#[cfg(all(test, target_os = "linux", feature = "runtime-core"))]
mod tests {
    use super::*;
    use crate::runtime_core::driver::Driver;
    use flare_selene::{
        PageId, StateSlotId,
        codec_copy::PinnedPacket,
        geometry::{CacheGeometry, CodecField, Plane},
        offload::KvCompression,
        shared::wire::packed_unit::Kind,
    };

    fn run_roundtrip_mode(placement: KvCompression) {
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
        let context = ctx as usize;
        let stream = Stream::open_lowest_priority(context, 0).unwrap();
        let (page_bytes, row_bytes) = (4096usize, 65536usize);
        let n_pages = 4usize;
        let mut page_mem = stream.mapped_host_buffer(page_bytes * n_pages * 2).unwrap();
        let mut row_mem = stream.mapped_host_buffer(row_bytes * 2).unwrap();
        let page_addr = stream.mapped_address(&page_mem).unwrap() as u64;
        let row_addr = stream.mapped_address(&row_mem).unwrap() as u64;
        let mut geometry = CacheGeometry {
            placement,
            codec_fields: vec![
                CodecField {
                    buffer: "kv".into(),
                    offset: 0,
                    length: page_bytes as u64,
                    encoding: 2, // Bfloat16
                },
                CodecField {
                    buffer: "state".into(),
                    offset: 0,
                    length: row_bytes as u64,
                    encoding: 1, // Float32Matrix128
                },
            ],
            snapshot_pages: Vec::new(),
            page_tokens: 64,
            page_capacity: (n_pages * 2) as u64,
            row_capacity: 2,
            planes: vec![Plane {
                name: "kv".into(),
                address: page_addr,
                unit_bytes: page_bytes as u64,
            }],
            row: Plane {
                name: "state".into(),
                address: row_addr,
                unit_bytes: row_bytes as u64,
            },
        };

        // Populate source pages 0..n_pages and row 0 in device-visible memory.
        let src_pages: Vec<Vec<u8>> = (0..n_pages)
            .map(|p| (0..page_bytes).map(|i| (((i + p * 17) / 7) % 13) as u8).collect())
            .collect();
        let mut row0 = vec![0u8; row_bytes];
        for c in 0..128 {
            for r in 0..128 {
                let v = ((r as f32 * 0.01).sin() * (c as f32 * 0.02).cos()).to_bits();
                let idx = (c * 128 + r) * 4;
                row0[idx..idx + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        for (p, data) in src_pages.iter().enumerate() {
            page_mem.bytes_mut()[p * page_bytes..(p + 1) * page_bytes].copy_from_slice(data);
        }
        row_mem.bytes_mut()[..row_bytes].copy_from_slice(&row0);
        page_mem.bytes_mut()[n_pages * page_bytes..].fill(0xa5);
        row_mem.bytes_mut()[row_bytes..].fill(0xa5);

        let mut transfers = if geometry.placement == KvCompression::Gpu {
            let path = std::env::var("FLARE_TEST_KV_CODEC_CUBIN")
                .expect("set FLARE_TEST_KV_CODEC_CUBIN to a compiled kv_codec_gpu.cubin");
            let cubin = Arc::new(std::fs::read(path).unwrap());
            Copiers::new(vec![cubin])
                .start(&[(context, 0)], vec![geometry.clone()], None)
                .unwrap()
        } else {
            Copiers::new(Vec::new())
                .start(&[(context, 0)], vec![geometry.clone()], None)
                .unwrap()
        };

        // 1. Capture VRAM -> packed host (Direction::Out) across multiple pages + row.
        let cap_win = Window::capture(0);
        transfers
            .submit(Job {
                id: flare_selene::JobId(1),
                window: 0,
                direction: Direction::Out,
                pages: (0..n_pages as u32).map(|p| (PageId(p), p as usize)).collect(),
                row: Some((StateSlotId(0), vec![0])),
                replication: Default::default(),
                codec: Some(cap_win.clone()),
            })
            .unwrap();
        let cap_reply = loop {
            let mut out = Vec::new();
            transfers.poll(&mut out);
            if let Some(reply) = out.pop() {
                break reply;
            }
            std::thread::yield_now();
        };
        cap_reply.result.unwrap();
        let slots = cap_win.capture.as_ref().unwrap().take_sorted();
        assert_eq!(slots.len(), n_pages + 1);
        for p in 0..n_pages {
            assert_eq!(slots[p].unit.kind, Kind::Page as i32);
            assert_eq!(slots[p].unit.representation, Representation::Packet as i32);
            assert!(slots[p].bytes.len() < page_bytes);
        }
        assert_eq!(slots[n_pages].unit.kind, Kind::Row as i32);
        assert_eq!(slots[n_pages].unit.representation, Representation::Packet as i32);
        assert!(slots[n_pages].bytes.len() < row_bytes);

        // 2. Restore packed host -> VRAM pages n_pages..2*n_pages and row 1 (Direction::In)
        let restore_win = Window::restore_pinned(
            0,
            slots
                .iter()
                .map(|s| unsafe {
                    PinnedPacket::from_registered(
                        s.unit.clone(),
                        vec![(s.bytes.as_ptr() as usize, s.bytes.len())],
                    )
                })
                .collect(),
        );
        transfers
            .submit(Job {
                id: flare_selene::JobId(2),
                window: 0,
                direction: Direction::In,
                pages: (0..n_pages as u32)
                    .map(|p| (PageId(n_pages as u32 + p), p as usize))
                    .collect(),
                row: Some((StateSlotId(1), vec![0])),
                replication: Default::default(),
                codec: Some(restore_win),
            })
            .unwrap();
        let res_reply = loop {
            let mut out = Vec::new();
            transfers.poll(&mut out);
            if let Some(reply) = out.pop() {
                break reply;
            }
            std::thread::yield_now();
        };
        res_reply.result.unwrap();
        for (p, data) in src_pages.iter().enumerate() {
            let dst_off = (n_pages + p) * page_bytes;
            assert_eq!(&page_mem.bytes()[dst_off..dst_off + page_bytes], data.as_slice());
        }
        assert_eq!(&row_mem.bytes()[row_bytes..row_bytes * 2], &row0);

        // 3. Corrupt stored CRC must fail cleanly and count a stored verification failure.
        let mut bad_u = slots[0].unit.clone();
        bad_u.stored_crc ^= 1;
        let bad_win = Window::restore_pinned(
            0,
            vec![unsafe {
                PinnedPacket::from_registered(
                    bad_u,
                    vec![(slots[0].bytes.as_ptr() as usize, slots[0].bytes.len())],
                )
            }],
        );
        transfers
            .submit(Job {
                id: flare_selene::JobId(3),
                window: 0,
                direction: Direction::In,
                pages: vec![(PageId(1), 0)],
                row: None,
                replication: Default::default(),
                codec: Some(bad_win),
            })
            .unwrap();
        let bad_reply = loop {
            let mut out = Vec::new();
            transfers.poll(&mut out);
            if let Some(reply) = out.pop() {
                break reply;
            }
            std::thread::yield_now();
        };
        assert!(bad_reply.result.is_err());
        let stats = *transfers.decode_counters().unwrap().lock().unwrap();
        assert_eq!(stats.stored_failures, 1);
        assert_eq!(stats.failed_windows, 1);

        geometry.placement = KvCompression::Off;
        let mut off = Copiers::new(Vec::new())
            .start(&[(context, 0)], vec![geometry], None)
            .unwrap();
        assert!(off.codec_profile().is_none());
        transfers.close().unwrap();
        off.close().unwrap();
    }

    #[test]
    #[ignore = "exercises CPU VRAM <-> DRAM capture and restore onto a GPU"]
    fn cpu_vram_capture_and_restore_copies_exact_bytes_and_rejects_corruption() {
        run_roundtrip_mode(KvCompression::Cpu);
    }

    #[test]
    #[ignore = "exercises GPU GKV1 VRAM <-> DRAM capture and restore onto a GPU"]
    fn gpu_vram_capture_and_restore_copies_exact_bytes_and_rejects_corruption() {
        run_roundtrip_mode(KvCompression::Gpu);
    }

    #[test]
    #[ignore = "exercises mixed CPU ZKV1 <-> GPU GKV1 cross-mode restore without abandoning cache"]
    fn mixed_cpu_and_gpu_modes_restore_each_others_packets_and_write_their_own() {
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
        let context = ctx as usize;
        let stream = Stream::open_lowest_priority(context, 0).unwrap();
        let (page_bytes, row_bytes) = (4096usize, 65536usize);
        let mut page_mem = stream.mapped_host_buffer(page_bytes * 2).unwrap();
        let mut row_mem = stream.mapped_host_buffer(row_bytes * 2).unwrap();
        let page_addr = stream.mapped_address(&page_mem).unwrap() as u64;
        let row_addr = stream.mapped_address(&row_mem).unwrap() as u64;
        let mut base_geo = CacheGeometry {
            placement: KvCompression::Cpu,
            codec_fields: vec![
                CodecField {
                    buffer: "kv".into(),
                    offset: 0,
                    length: page_bytes as u64,
                    encoding: 2,
                },
                CodecField {
                    buffer: "state".into(),
                    offset: 0,
                    length: row_bytes as u64,
                    encoding: 1,
                },
            ],
            snapshot_pages: Vec::new(),
            page_tokens: 64,
            page_capacity: 2,
            row_capacity: 2,
            planes: vec![Plane {
                name: "kv".into(),
                address: page_addr,
                unit_bytes: page_bytes as u64,
            }],
            row: Plane {
                name: "state".into(),
                address: row_addr,
                unit_bytes: row_bytes as u64,
            },
        };
        let page0: Vec<u8> = (0..page_bytes).map(|i| ((i / 7) % 13) as u8).collect();
        let mut row0 = vec![0u8; row_bytes];
        for c in 0..128 {
            for r in 0..128 {
                let v = ((r as f32 * 0.01).sin() * (c as f32 * 0.02).cos()).to_bits();
                let idx = (c * 128 + r) * 4;
                row0[idx..idx + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        page_mem.bytes_mut()[..page_bytes].copy_from_slice(&page0);
        row_mem.bytes_mut()[..row_bytes].copy_from_slice(&row0);

        let cpu_profile = flare_selene::shared::codec_layout::profile(
            &base_geo,
            flare_selene::shared::codec_layout::REVISION_CPU,
        )
        .unwrap();
        let gpu_profile = flare_selene::shared::codec_layout::profile(
            &base_geo,
            flare_selene::shared::codec_layout::REVISION_GPU,
        )
        .unwrap();
        assert_ne!(cpu_profile, gpu_profile);

        let test_cubin = || -> Vec<Arc<Vec<u8>>> {
            let path = std::env::var("FLARE_TEST_KV_CODEC_CUBIN")
                .expect("set FLARE_TEST_KV_CODEC_CUBIN to a compiled kv_codec_gpu.cubin");
            vec![Arc::new(std::fs::read(path).unwrap())]
        };
        let mut cpu_xfer = Copiers::new(Vec::new())
            .start(&[(context, 0)], vec![base_geo.clone()], None)
            .unwrap();
        base_geo.placement = KvCompression::Gpu;
        let gpu_copiers = Copiers::new(test_cubin());
        let mut gpu_xfer = gpu_copiers.start(&[(context, 0)], vec![base_geo], None).unwrap();

        // Both CPU and GPU modes advertise [ZKV1, GKV1] for full bidirectional pool interop.
        assert_eq!(cpu_xfer.codec_receive_profiles(), vec![cpu_profile, gpu_profile]);
        assert_eq!(gpu_xfer.codec_receive_profiles(), vec![cpu_profile, gpu_profile]);

        let run_one = |xfer: &mut Transfers, job: Job| {
            xfer.submit(job).unwrap();
            loop {
                let mut out = Vec::new();
                xfer.poll(&mut out);
                if let Some(reply) = out.pop() {
                    reply.result.unwrap();
                    return;
                }
                std::thread::yield_now();
            }
        };

        // 1. Capture in CPU mode (writes ZKV1 / cpu_profile)
        let cpu_cap = Window::capture(0);
        run_one(
            &mut cpu_xfer,
            Job {
                id: flare_selene::JobId(1),
                window: 0,
                direction: Direction::Out,
                pages: vec![(PageId(0), 0)],
                row: Some((StateSlotId(0), vec![0])),
                replication: Default::default(),
                codec: Some(cpu_cap.clone()),
            },
        );
        let cpu_slots = cpu_cap.capture.as_ref().unwrap().take_sorted();
        assert!(cpu_slots.iter().all(|s| s.unit.profile.as_slice() == cpu_profile.as_slice()));

        // 2. Restore CPU ZKV1 packets using GPU-mode Transfers (decodes on CPU -> VRAM)
        page_mem.bytes_mut()[page_bytes..].fill(0xaa);
        row_mem.bytes_mut()[row_bytes..].fill(0xaa);
        let zkv_win = Window::restore_pinned(
            0,
            cpu_slots
                .iter()
                .map(|s| unsafe {
                    PinnedPacket::from_registered(
                        s.unit.clone(),
                        vec![(s.bytes.as_ptr() as usize, s.bytes.len())],
                    )
                })
                .collect(),
        );
        run_one(
            &mut gpu_xfer,
            Job {
                id: flare_selene::JobId(2),
                window: 0,
                direction: Direction::In,
                pages: vec![(PageId(1), 0)],
                row: Some((StateSlotId(1), vec![0])),
                replication: Default::default(),
                codec: Some(zkv_win),
            },
        );
        assert_eq!(&page_mem.bytes()[page_bytes..], &page0);
        assert_eq!(&row_mem.bytes()[row_bytes..], &row0);

        // 3. Re-capture in GPU mode (writes GKV1 / gpu_profile)
        let gpu_cap = Window::capture(0);
        run_one(
            &mut gpu_xfer,
            Job {
                id: flare_selene::JobId(3),
                window: 0,
                direction: Direction::Out,
                pages: vec![(PageId(1), 0)],
                row: Some((StateSlotId(1), vec![0])),
                replication: Default::default(),
                codec: Some(gpu_cap.clone()),
            },
        );
        let gpu_slots = gpu_cap.capture.as_ref().unwrap().take_sorted();
        assert!(gpu_slots.iter().all(|s| s.unit.profile.as_slice() == gpu_profile.as_slice()));

        // 4. CPU-mode Transfers natively restores GKV1/v2 packets on CPU with zero GPU VRAM!
        page_mem.bytes_mut()[page_bytes..].fill(0x55);
        row_mem.bytes_mut()[row_bytes..].fill(0x55);
        let gkv_win = Window::restore_pinned(
            0,
            gpu_slots
                .iter()
                .map(|s| unsafe {
                    PinnedPacket::from_registered(
                        s.unit.clone(),
                        vec![(s.bytes.as_ptr() as usize, s.bytes.len())],
                    )
                })
                .collect(),
        );
        run_one(
            &mut cpu_xfer,
            Job {
                id: flare_selene::JobId(4),
                window: 0,
                direction: Direction::In,
                pages: vec![(PageId(1), 0)],
                row: Some((StateSlotId(1), vec![0])),
                replication: Default::default(),
                codec: Some(gkv_win),
            },
        );
        assert_eq!(&page_mem.bytes()[page_bytes..], &page0);
        assert_eq!(&row_mem.bytes()[row_bytes..], &row0);

        cpu_xfer.close().unwrap();
        gpu_xfer.close().unwrap();
    }
}
