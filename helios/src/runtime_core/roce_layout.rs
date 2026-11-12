/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! The proxy's device-independent region layout and core geometry gate.
//!
//! `roce_layout` is arithmetic over the core's recorded geometry: it opens no device, HCA or
//! allocation. The core loader can therefore validate a recorded geometry that describes a
//! different wire, a mismatched region or a foreign proxy ABI while it reads metadata, before any
//! CUDA context exists, and it can do so in builds without the native RoCE module (and without
//! libibverbs). The native transport keeps its own FFI call as a cross-check; tests assert both
//! produce the same seventeen words.

use super::model::RoceSpec;
use crate::{ExecutionError, Failure, Result};

/// The proxy ABI compiled into this runtime and required in live peer records.
///
/// Operation credits prevent a rank from overwriting receive rows while a peer still uses them.
/// Open lines of four or more ranks broadcast pipelined all-gather chunks. Connection records
/// carry the ring geometry so peers can refuse incompatible allocations before using addresses.
///
/// Core compatibility is checked separately against the recorded GPU-visible region layout.
pub(crate) const PROXY_ABI: u32 = 15;
/// The oldest core ABI whose captured region this runtime still addresses.
pub(crate) const OLDEST_PACKAGE_ABI: u32 = 9;
/// The ABI number that names two different wires: one with a single pipelined receive row per peer
/// and chunk, and one with two rows selected by `seq & 1`. Only the recorded layout tells them
/// apart.
pub(crate) const PARITY_PACKAGE_ABI: u32 = 12;
/// Rails whose flag rows `roce_layout` sizes even when a group uses fewer.
const LAYOUT_RAILS: u64 = 2;
/// Receive slots per peer and chunk, as the native proxy defines them.
const SLOTS: u64 = 2;
/// Bytes per ownership flag word.
const FLAG_STRIDE: u64 = 128;
/// Control bytes reserved after the send rows.
const CTRL_BYTES: u64 = 4096;
/// Chunks in the pipelined area.
const MAX_CHUNKS: u64 = 64;
/// Chunks in the variable-length area.
const VARLEN_MAX_CHUNKS: u64 = 512;
/// Largest peer count the native proxy accepts.
const MAX_PEERS: usize = 16;

fn failure(message: impl Into<String>) -> crate::Error {
    ExecutionError::Poisoned(Failure::new(message.into())).into()
}

/// `roce_layout`'s seventeen words for a geometry, or the refusal the native proxy gives for it.
///
/// Pure arithmetic: no device, HCA or allocation is touched. The arithmetic mirrors
/// `native/roce_proxy.c` including its alignment, range and overflow refusals; the RoCE tests
/// compare the result against the native function for the same geometries.
pub(crate) fn layout(
    world: usize,
    slot_bytes: u64,
    pipe_bytes: u64,
    varlen_bytes: u64,
) -> Result<[u64; 17]> {
    if !(2..=MAX_PEERS).contains(&world)
        || slot_bytes == 0
        || slot_bytes % 4096 != 0
        || pipe_bytes % 4096 != 0
        || varlen_bytes % 4096 != 0
        || varlen_bytes > 1 << 32
    {
        return Err(failure(
            "RoCE layout rejects the core's transport geometry",
        ));
    }
    if slot_bytes > 1 << 40 || pipe_bytes > 1 << 32 {
        return Err(failure(
            "RoCE layout rejects the core's transport geometry",
        ));
    }
    let overflow = || failure("RoCE layout overflows the core's transport geometry");
    let world = world as u64;
    let recv_bytes = world
        .checked_mul(SLOTS)
        .and_then(|n| n.checked_mul(slot_bytes))
        .ok_or_else(overflow)?;
    let flag_bytes = world
        .checked_mul(SLOTS)
        .and_then(|n| n.checked_mul(LAYOUT_RAILS))
        .and_then(|n| n.checked_mul(FLAG_STRIDE))
        .ok_or_else(overflow)?;
    let send_bytes = SLOTS.checked_mul(slot_bytes).ok_or_else(overflow)?;
    let send_off = recv_bytes.checked_add(flag_bytes).ok_or_else(overflow)?;
    let ctrl_off = send_off.checked_add(send_bytes).ok_or_else(overflow)?;
    let pipe_send_off = ctrl_off.checked_add(CTRL_BYTES).ok_or_else(overflow)?;
    // A shard holds at most ceil(packs / world) packs.
    let shard_cap = (pipe_bytes / 16 + world - 1) / world * 16;
    let shard_cap = (shard_cap + 4095) / 4096 * 4096;
    let pipe_recv_off = pipe_send_off.checked_add(pipe_bytes).ok_or_else(overflow)?;
    let pipe_recv_bytes = world
        .checked_mul(2)
        .and_then(|n| n.checked_mul(shard_cap))
        .ok_or_else(overflow)?;
    let pipe_flag_off = pipe_recv_off
        .checked_add(pipe_recv_bytes)
        .ok_or_else(overflow)?;
    let pipe_flag_bytes = if pipe_bytes == 0 {
        0
    } else {
        2u64.checked_mul(world)
            .and_then(|n| n.checked_mul(MAX_CHUNKS))
            .and_then(|n| n.checked_mul(LAYOUT_RAILS))
            .and_then(|n| n.checked_mul(FLAG_STRIDE))
            .ok_or_else(overflow)?
    };
    let varlen_recv_off = pipe_flag_off
        .checked_add(pipe_flag_bytes)
        .ok_or_else(overflow)?;
    let varlen_flag_off = varlen_recv_off
        .checked_add(
            SLOTS
                .checked_mul(world)
                .and_then(|n| n.checked_mul(varlen_bytes))
                .ok_or_else(overflow)?,
        )
        .ok_or_else(overflow)?;
    let varlen_flag_bytes = if varlen_bytes == 0 {
        0
    } else {
        SLOTS
            .checked_mul(world)
            .and_then(|n| n.checked_mul(VARLEN_MAX_CHUNKS))
            .and_then(|n| n.checked_mul(LAYOUT_RAILS))
            .and_then(|n| n.checked_mul(FLAG_STRIDE))
            .ok_or_else(overflow)?
    };
    let total = varlen_flag_off
        .checked_add(varlen_flag_bytes)
        .ok_or_else(overflow)?;
    Ok([
        0,
        recv_bytes,
        send_off,
        ctrl_off,
        total,
        FLAG_STRIDE,
        SLOTS,
        pipe_send_off,
        pipe_recv_off,
        pipe_flag_off,
        shard_cap,
        MAX_CHUNKS,
        pipe_bytes,
        varlen_recv_off,
        varlen_flag_off,
        varlen_bytes,
        VARLEN_MAX_CHUNKS,
    ])
}

/// The bytes a doubled-row ABI 12 core adds to `roce_layout`'s total for the same geometry.
///
/// Such a core multiplied the pipelined receive rows and flags by `SLOTS` and selected them by
/// `seq & 1`, so it recorded a larger region under ABI 12. The region size is the only evidence
/// that separates it from an ABI 12 core whose layout is this runtime's.
pub(crate) fn parity_pipe_excess(world: u64, layout: &[u64; 17]) -> u64 {
    let extra_slots = layout[6].saturating_sub(1);
    let recv = extra_slots
        .saturating_mul(world)
        .saturating_mul(2)
        .saturating_mul(layout[10]);
    let flags = extra_slots
        .saturating_mul(2)
        .saturating_mul(world)
        .saturating_mul(layout[11])
        .saturating_mul(LAYOUT_RAILS)
        .saturating_mul(layout[5]);
    recv.saturating_add(flags)
}

/// The lockstep core gate: this runtime addresses a core whose recorded region is the one
/// `roce_layout` computes for its geometry, whatever ABI number the export recorded.
///
/// ABI numbers 9 through 15 all describe one kernel-visible region (`roce_layout` is byte-for-byte
/// the same across them), so an older core runs on this runtime on every path its kernels know,
/// including ring and chain: the proxy posts peers' start-of-op credits itself, and the fanout is
/// proxy-internal. The single exception is an ABI 12 core that doubled the pipelined rows; it is
/// refused by its recorded geometry with a named message rather than by its number.
pub(crate) fn check_core_abi(
    core_abi: u32,
    runtime_abi: u32,
    spec: &RoceSpec,
    layout: &[u64; 17],
) -> Result<()> {
    // `open` rejects a proxy that reports anything but `PROXY_ABI`; repeating that here keeps the
    // forward direction (a newer core on an older runtime) testable without a device.
    if !(OLDEST_PACKAGE_ABI..=PROXY_ABI).contains(&core_abi) || runtime_abi != PROXY_ABI {
        return Err(failure(format!(
            "core RoCE kernels use proxy ABI {core_abi}, this runtime provides {runtime_abi}"
        )));
    }
    if spec.pipe_bytes == 0 {
        return Ok(());
    }
    let Some(excess) = spec.region_bytes.checked_sub(layout[4]) else {
        // A region smaller than the geometry needs fails the layout check with its own message.
        return Ok(());
    };
    if core_abi == PARITY_PACKAGE_ABI && excess == parity_pipe_excess(spec.world as u64, layout)
    {
        return Err(failure(
            "core RoCE pipelined rows were recorded with the ABI 12 sequence-parity layout, \
             which holds two rows per peer and chunk; this runtime provides ABI 15's single-copy \
             pipelined rows and storage and cannot address them — re-export from a proxy ABI 15 \
             build to serve this model here",
        ));
    }
    Ok(())
}

/// Validate a core's recorded transport geometry against this runtime's proxy, opening nothing.
///
/// This is pure arithmetic over the core's recorded numbers plus [`layout`]: no device, no HCA
/// and no allocation. The core loader runs it while it reads the transport metadata, before any
/// CUDA context, allocation, resource or cache exists, so a core this runtime cannot address —
/// an ABI 12 core with two pipelined rows per peer and chunk, a region that is not `roce_layout`
/// for its own geometry, or a proxy ABI this runtime does not provide — is refused by name
/// instead of after device work.
/// RoCE-capable transports repeat it as a defensive check for a spec built without the loader.
pub(crate) fn check_core_layout(spec: &RoceSpec) -> Result<()> {
    if spec.varlen_bytes != 0 && !(2..=16).contains(&spec.world) {
        return Err(failure(
            "core RoCE variable-length geometry needs 2 to 16 ranks",
        ));
    }
    let layout = layout(
        spec.world,
        spec.slot_bytes,
        spec.pipe_bytes,
        spec.varlen_bytes,
    )?;
    check_core_abi(spec.proxy_abi, PROXY_ABI, spec, &layout)?;
    if layout[3] != spec.ctrl_offset || layout[4] != spec.region_bytes {
        return Err(failure(
            "core RoCE region differs from the proxy's layout for its geometry",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specification(region_bytes: u64) -> RoceSpec {
        RoceSpec {
            proxy_abi: PROXY_ABI,
            world: 4,
            rank: 0,
            region_address: 0x7f00_0000_0000,
            region_bytes,
            counters_address: 0x7f00_1000_0000,
            counters_bytes: 64,
            slot_bytes: 4096,
            pipe_bytes: 4096,
            varlen_bytes: 4096,
            ctrl_offset: 43_008,
        }
    }

    #[test]
    fn pure_layout_matches_the_recorded_world_four_geometry() {
        let words = layout(4, 4096, 4096, 4096).unwrap();
        assert_eq!(
            words,
            [
                0, 32_768, 34_816, 43_008, 1_296_384, 128, 2, 47_104, 51_200, 83_968, 4096, 64,
                4096, 215_040, 247_808, 4096, 512,
            ]
        );
        // Without a pipelined or variable-length area the region is the receive, flag and send
        // rows plus the control record.
        let plain = layout(2, 4096, 0, 0).unwrap();
        assert_eq!(plain[4], 2 * 2 * 4096 + 2 * 2 * 2 * 128 + 2 * 4096 + 4096);
        assert_eq!(
            &plain[7..],
            &[
                plain[7], plain[7], plain[7], 0, 64, 0, plain[7], plain[7], 0, 512
            ]
        );
    }

    #[test]
    fn pure_layout_refuses_geometry_the_native_proxy_refuses() {
        for (world, slot, pipe, varlen) in [
            (1, 4096, 0, 0),
            (17, 4096, 0, 0),
            (4, 0, 0, 0),
            (4, 4097, 0, 0),
            (4, 4096, 1, 0),
            (4, 4096, 0, 1),
            (4, 4096, 0, (1 << 32) + 4096),
            (4, (1 << 40) + 4096, 0, 0),
            (4, 4096, (1 << 32) + 4096, 0),
        ] {
            assert!(
                layout(world, slot, pipe, varlen).is_err(),
                "world {world}, slot {slot}, pipe {pipe}, varlen {varlen}"
            );
        }
        // The ceilings are inclusive on both sides: the native proxy refuses only beyond them.
        for (world, slot, pipe, varlen) in [
            (2, 4096, 0, 0),
            (16, 4096, 0, 0),
            (4, 1 << 40, 0, 0),
            (4, 4096, 1 << 32, 0),
            (4, 4096, 0, 1 << 32),
        ] {
            assert!(
                layout(world, slot, pipe, varlen).is_ok(),
                "world {world}, slot {slot}, pipe {pipe}, varlen {varlen}"
            );
        }
    }

    #[test]
    fn the_metadata_gate_refuses_a2_and_mismatched_regions() {
        let words = layout(4, 4096, 4096, 4096).unwrap();
        let matching = specification(words[4]);
        check_core_layout(&matching).expect("the recorded geometry is this runtime's layout");
        let error = check_core_layout(&specification(words[4] + 4096))
            .expect_err("a shifted region must be refused")
            .to_string();
        assert!(error.contains("region differs"), "{error}");
        let mut parity = specification(words[4] + parity_pipe_excess(4, &words));
        parity.proxy_abi = PARITY_PACKAGE_ABI;
        let error = check_core_layout(&parity)
            .expect_err("the doubled-row ABI 12 layout must be refused")
            .to_string();
        assert!(error.contains("sequence-parity"), "{error}");
        assert!(error.contains("re-export from a proxy ABI 15"), "{error}");
        let mut foreign = specification(words[4]);
        foreign.proxy_abi = PROXY_ABI + 1;
        let error = check_core_layout(&foreign)
            .expect_err("a newer core ABI must be refused")
            .to_string();
        assert!(error.contains("proxy ABI 16"), "{error}");
    }
}
