/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{
    format::{Memory, Core, invalid},
    protobuf::wire,
};
use crate::{
    Result,
    engram::{Index, Plane, Table},
};
use std::collections::HashSet;

/// Payload ownership follows the table layout; absent partitions retain shared storage.
pub(super) fn spans(
    index: &wire::ModelIndex,
    selected: Option<usize>,
) -> Result<Vec<(Option<usize>, &wire::Span)>> {
    let world = index
        .cluster
        .as_ref()
        .map_or(index.ranks.len(), |c| c.world as usize);
    let mut result = Vec::new();
    for table in index.engram.iter().flat_map(|e| &e.tables) {
        validate_shards(table, world)?;
        if table.shards.is_empty() {
            for span in [&table.weights, &table.scales] {
                result.push((
                    None,
                    span.as_ref()
                        .ok_or_else(|| invalid("missing table plane"))?,
                ));
            }
        } else {
            for shard in &table.shards {
                if selected.is_none_or(|rank| rank == shard.rank as usize) {
                    for span in [&shard.weights, &shard.scales] {
                        result.push((
                            Some(shard.rank as usize),
                            span.as_ref()
                                .ok_or_else(|| invalid("missing table partition plane"))?,
                        ));
                    }
                }
            }
        }
    }
    Ok(result)
}

fn validate_shards(table: &wire::EngramTable, world: usize) -> Result<()> {
    if table.shards.is_empty() {
        return Ok(());
    }
    if table.weights.is_some()
        || table.scales.is_some()
        || table.request_buffer.is_empty()
        || table.shards.len() != world
    {
        return Err(invalid("invalid partitioned embedding storage"));
    }
    let mut end = 0;
    for (rank, shard) in table.shards.iter().enumerate() {
        if shard.rank as usize != rank || shard.first_row != end || shard.rows == 0 {
            return Err(invalid(
                "embedding partitions must cover table rows in rank order",
            ));
        }
        end = end
            .checked_add(shard.rows)
            .ok_or_else(|| invalid("embedding partition overflow"))?;
        // Sibling partitions absent from a rank-selective copy carry neither plane; the
        // owning rank still requires both (decode and `spans` reject a missing plane).
        if shard.weights.is_none() != shard.scales.is_none() {
            return Err(invalid("missing embedding partition plane"));
        }
    }
    if end != table.rows {
        return Err(invalid("embedding partitions must cover all table rows"));
    }
    Ok(())
}

pub(super) fn decode<F: super::payload::Files>(
    value: wire::Engram,
    archive: &F,
    cores: &[Core],
    rank: usize,
    world: usize,
) -> Result<Index> {
    if value.compressed_vocabulary == 0
        || value
            .token_map
            .iter()
            .any(|n| *n >= value.compressed_vocabulary)
        || value.pad_token_id
            >= if value.token_map.is_empty() {
                value.compressed_vocabulary
            } else {
                value.token_map.len() as u32
            }
        || value.tables.is_empty()
    {
        return Err(invalid("invalid Engram vocabulary or table catalog"));
    }
    let mut layers = HashSet::new();
    let mut buffers = HashSet::new();
    let mut events = HashSet::new();
    let mut tables = Vec::new();
    for table in value.tables {
        validate_shards(&table, world)?;
        let gpu = !table.request_buffer.is_empty();
        let encoding = wire::engram_table::Encoding::try_from(table.encoding)
            .map_err(|_| invalid("unknown n-gram encoding"))?;
        let (width, block, row_bytes, scale_bytes) = match encoding {
            wire::engram_table::Encoding::Fp8E4m3E8m0 => (256, 32, 256, 8),
            wire::engram_table::Encoding::Nvfp4E4m3Global
                if table.global_scale.is_finite() && table.global_scale > 0.0 =>
            {
                (160, 16, 80, 10)
            }
            _ => return Err(invalid("invalid n-gram encoding or global scale")),
        };
        let geometry = table
            .hash
            .as_ref()
            .ok_or_else(|| invalid("n-gram table lacks hash geometry"))?;
        let order = match wire::engram_hash::Method::try_from(geometry.method) {
            Ok(wire::engram_hash::Method::XorModulo) if !value.token_map.is_empty() => 4,
            Ok(wire::engram_hash::Method::EosXorModulo) if gpu && value.token_map.is_empty() => 3,
            _ => return Err(invalid("unsupported n-gram hash method or request source")),
        };
        let heads = (order - 1) * 8;
        if !layers.insert(table.layer)
            || table.rows == 0
            || table.embedding_width != width
            || table.scale_block != block
            || !events.insert(table.ready_event.clone())
            || table.ready_event.is_empty()
            || [
                &table.weight_buffer,
                &table.scale_buffer,
                &table.hash_buffer,
            ]
            .iter()
            .any(|name| name.is_empty() || !buffers.insert((*name).clone()))
        {
            return Err(invalid("invalid Engram table encoding or bindings"));
        }
        if gpu && !buffers.insert(table.request_buffer.clone()) {
            return Err(invalid("n-gram mailbox aliases another binding"));
        }
        let has_prefetch = !table.prefetch_buffer.is_empty();
        if has_prefetch {
            if !gpu {
                return Err(invalid("n-gram prefetch mailbox requires request delivery"));
            }
            if !buffers.insert(table.prefetch_buffer.clone()) {
                return Err(invalid("n-gram prefetch mailbox aliases another binding"));
            }
        }
        let geometry = table
            .hash
            .ok_or_else(|| invalid("Engram table lacks hash geometry"))?;
        if geometry.orders != (2..=order as u32).flat_map(|n| [n; 8]).collect::<Vec<_>>()
            || geometry.multipliers.len() != order
            || geometry.primes.len() != heads
            || geometry.offsets.len() != heads
            || geometry.multipliers.iter().any(|m| {
                *m == 0
                    || m.checked_mul(value.compressed_vocabulary as u64)
                        .is_none_or(|n| n > i64::MAX as u64)
            })
            || geometry.primes.iter().zip(&geometry.offsets).any(|(p, o)| {
                *p < 2
                    || o.checked_add(*p)
                        .is_none_or(|n| n > table.rows || n > i64::MAX as u64)
            })
        {
            return Err(invalid("invalid Engram hash geometry"));
        }
        let (stored_rows, weights, scales) = if table.shards.is_empty() {
            (table.rows, table.weights.clone(), table.scales.clone())
        } else {
            let shard = table
                .shards
                .get(rank)
                .ok_or_else(|| invalid("embedding partition rank is absent"))?;
            (shard.rows, shard.weights.clone(), shard.scales.clone())
        };
        let plane = |span: Option<wire::Span>, row_bytes: usize| -> Result<Plane> {
            let span = span.ok_or_else(|| invalid("Engram table lacks a payload plane"))?;
            if stored_rows.checked_mul(row_bytes as u64) != Some(span.nbytes) {
                return Err(invalid("Engram payload geometry mismatch"));
            }
            Ok(Plane {
                file: archive.source_index(&span)?,
                offset: archive.source(&span)?.1,
                nbytes: usize::try_from(span.nbytes)
                    .map_err(|_| invalid("Engram plane exceeds host range"))?,
                row_bytes,
            })
        };
        let planes = [plane(weights, row_bytes)?, plane(scales, scale_bytes)?];
        let compact_heads = (gpu && !table.shards.is_empty() && world > 1 && heads % world == 0)
            .then_some(heads / world);
        let mut table_heads = heads;
        let mut prefetch_max_rows = 0usize;
        for core in cores {
            if !core.events.contains(&table.ready_event) {
                return Err(invalid("rank lacks Engram readiness event"));
            }
            let execution = core
                .execution
                .as_ref()
                .ok_or_else(|| invalid("Engram rank lacks execution bindings"))?;
            let tokens = execution
                .graphs
                .iter()
                .filter(|g| {
                    matches!(
                        g.kind.as_str(),
                        "prefill" | "terminal_prefill" | "decode" | "speculative" | "mixed"
                    )
                })
                .map(|g| g.packed_tokens())
                .max()
                .ok_or_else(|| invalid("Engram catalog lacks consumers"))?;
            let hash_buffer = execution
                .buffers
                .iter()
                .find(|b| b.name == table.hash_buffer)
                .ok_or_else(|| invalid("rank lacks Engram input buffer"))?;
            let core_heads = match hash_buffer.shape.as_slice() {
                [_, columns] => usize::try_from(*columns)
                    .map_err(|_| invalid("invalid Engram staged head width"))?,
                _ => return Err(invalid("invalid Engram buffer layout")),
            };
            if core_heads != heads && compact_heads != Some(core_heads) {
                return Err(invalid("invalid Engram staged head width"));
            }
            table_heads = core_heads;
            let validate_buffer = |buffer: &wire::Buffer, dtype: &str, rows: u64, columns: u64| {
                if buffer.dtype != dtype
                    || buffer.shape.len() != 2
                    || buffer.shape[0] < rows
                    || buffer.shape[1] != columns
                    || buffer.strides != [columns, 1]
                    || buffer.nbytes
                        != buffer
                            .shape[0]
                            .checked_mul(columns * if dtype == "int64" { 8 } else { 1 })
                            .unwrap_or(0)
                {
                    return Err(invalid("invalid Engram buffer layout"));
                }
                Ok(())
            };
            validate_buffer(hash_buffer, "int64", tokens, core_heads as u64)?;
            for (name, dtype, columns) in [
                (&table.weight_buffer, "uint8", row_bytes as u64),
                (&table.scale_buffer, "uint8", scale_bytes as u64),
            ] {
                let buffer = execution
                    .buffers
                    .iter()
                    .find(|b| &b.name == name)
                    .ok_or_else(|| invalid("rank lacks Engram input buffer"))?;
                validate_buffer(buffer, dtype, tokens * core_heads as u64, columns)?;
            }
            let validate_mailbox = |name: &str, min_len: u64, max_len: Option<u64>, label: &str| {
                let mailbox = execution
                    .buffers
                    .iter()
                    .find(|b| b.name == name)
                    .ok_or_else(|| invalid(format!("rank lacks {label}")))?;
                let reference = mailbox
                    .reference
                    .as_ref()
                    .ok_or_else(|| invalid(format!("{label} lacks storage")))?;
                if mailbox.dtype != "int64"
                    || mailbox.shape.len() != 1
                    || mailbox.strides != [1]
                    || mailbox.shape[0] < min_len
                    || max_len.is_some_and(|limit| mailbox.shape[0] > limit)
                    || mailbox.nbytes != mailbox.shape[0] * 8
                    || !reference.offset.is_multiple_of(8)
                    || !core.allocations.iter().any(|a| {
                        a.name == reference.allocation && matches!(a.memory, Memory::MappedHost)
                    })
                {
                    return Err(invalid(format!("invalid mapped {label}")));
                }
                Ok(mailbox.shape[0])
            };
            let max_staged_rows = tokens
                .checked_mul(core_heads as u64)
                .ok_or_else(|| invalid("Engram staging row count overflow"))?;
            if gpu {
                validate_mailbox(
                    &table.request_buffer,
                    max_staged_rows + 1,
                    None,
                    "n-gram request mailbox",
                )?;
            }
            if has_prefetch {
                let len = validate_mailbox(
                    &table.prefetch_buffer,
                    3,
                    Some(max_staged_rows + 2),
                    "n-gram prefetch mailbox",
                )?;
                let cap = usize::try_from(len - 2)
                    .map_err(|_| invalid("n-gram prefetch mailbox exceeds host range"))?;
                if prefetch_max_rows != 0 && prefetch_max_rows != cap {
                    return Err(invalid("ranks declare different n-gram prefetch capacities"));
                }
                prefetch_max_rows = cap;
            }
            for graph in execution.graphs.iter().filter(|g| {
                matches!(
                    g.kind.as_str(),
                    "prefill" | "terminal_prefill" | "decode" | "speculative" | "mixed"
                )
            }) {
                if !gpu
                    && !graph.input_bindings.iter().any(|b| {
                        b.buffer_name == table.hash_buffer
                            && b.nbytes >= graph.packed_tokens() * core_heads as u64 * 8
                    })
                {
                    return Err(invalid("Engram graph lacks hash input binding"));
                }
            }
        }
        tables.push(Table {
            planes,
            heads: table_heads,
            weights: table.weight_buffer,
            scales: table.scale_buffer,
            hashes: table.hash_buffer,
            event: table.ready_event,
            request: gpu.then_some(table.request_buffer),
            prefetch: has_prefetch.then_some((table.prefetch_buffer, prefetch_max_rows)),
        });
    }
    Ok(Index { tables })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn partitioned() -> wire::EngramTable {
        wire::EngramTable {
            rows: 40,
            request_buffer: "requests".into(),
            shards: (0..4)
                .map(|rank| wire::EngramTableShard {
                    rank,
                    first_row: rank as u64 * 10,
                    rows: 10,
                    weights: Some(wire::Span {
                        nbytes: 2560,
                        ..Default::default()
                    }),
                    scales: Some(wire::Span {
                        nbytes: 80,
                        ..Default::default()
                    }),
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn table_partitions_cover_rows_exactly_once() {
        let table = partitioned();
        assert!(validate_shards(&table, 4).is_ok());
        assert!(validate_shards(&table, 2).is_err());
        for change in 0..7 {
            let mut invalid = table.clone();
            match change {
                0 => invalid.shards[1].first_row += 1,
                1 => invalid.shards[1].first_row -= 1,
                2 => invalid.shards[2].rank = 0,
                3 => invalid.shards[3].rows = 0,
                4 => invalid.weights = Some(wire::Span::default()),
                5 => invalid.shards[1].scales = None,
                _ => invalid.request_buffer.clear(),
            }
            assert!(validate_shards(&invalid, 4).is_err(), "mutation {change}");
        }
    }

    #[test]
    fn rank_selected_partitions_omit_sibling_planes_but_keep_their_rows() {
        let mut table = partitioned();
        for rank in [0, 2, 3] {
            table.shards[rank].weights = None;
            table.shards[rank].scales = None;
        }
        assert!(validate_shards(&table, 4).is_ok());
        let mut one_plane = table.clone();
        one_plane.shards[0].weights = Some(wire::Span::default());
        assert!(validate_shards(&one_plane, 4).is_err());
        let mut gap = table;
        gap.shards[2].first_row += 1;
        assert!(validate_shards(&gap, 4).is_err());
    }

    #[test]
    fn metadata_without_partitions_retains_shared_planes() {
        // EngramTable encoded before field 15 existed: rows=40, weight/scales spans.
        let bytes = [
            0x10, 0x28, 0x2a, 0x04, 0x08, 0x00, 0x10, 0x20, 0x32, 0x04, 0x08, 0x20, 0x10, 0x08,
        ];
        let table = wire::EngramTable::decode(bytes.as_slice()).unwrap();
        assert!(table.shards.is_empty());
        assert!(validate_shards(&table, 4).is_ok());
        let index = wire::ModelIndex {
            ranks: vec![wire::BundleIndex::default(); 4],
            engram: Some(wire::Engram {
                tables: vec![table],
                ..Default::default()
            }),
            ..Default::default()
        };
        for rank in 0..4 {
            let planes = spans(&index, Some(rank)).unwrap();
            assert_eq!(planes.len(), 2);
            assert!(planes.iter().all(|(owner, _)| owner.is_none()));
            assert_eq!((planes[0].1.offset, planes[0].1.nbytes), (0, 32));
            assert_eq!((planes[1].1.offset, planes[1].1.nbytes), (32, 8));
        }
    }

    struct StubFiles(super::super::archive::PayloadFile);

    impl StubFiles {
        fn new() -> Self {
            let file = std::fs::File::open(std::env::current_exe().unwrap()).unwrap();
            Self(super::super::archive::PayloadFile::new(
                std::sync::Arc::new(file),
                0,
                0,
            ))
        }
    }

    impl super::super::payload::Files for StubFiles {
        fn span<'a>(&self, _: &'a Core, _: &str) -> Result<&'a wire::Span> { unimplemented!() }
        fn source(&self, span: &wire::Span) -> Result<(&super::super::archive::PayloadFile, u64)> { Ok((&self.0, span.offset)) }
        fn read(&self, _: &Core, _: &str) -> Result<Vec<u8>> { unimplemented!() }
        fn records<'a>(&'a self, _: &Core, _: &str) -> Result<super::super::payload::RecordReader<'a>> { unimplemented!() }
        fn path(&self) -> &std::path::Path { std::path::Path::new("") }
        fn source_index(&self, _: &wire::Span) -> Result<usize> { Ok(0) }
        fn asset_path(&self, _: &wire::Span) -> Result<std::path::PathBuf> { unimplemented!() }
        fn release_page_cache(&self) {}
        fn retained_files(&self) -> Result<Vec<std::sync::Arc<std::fs::File>>> { Ok(vec![]) }
        fn direct_files(&self) -> Box<dyn Iterator<Item = Result<std::sync::Arc<std::fs::File>>> + '_> { Box::new(std::iter::empty()) }
        fn validate(&self, _: &Core) -> Result<()> { Ok(()) }
        fn resource_reader<'a>(&'a self, _: &Core, _: &str) -> Result<Box<dyn std::io::Read + 'a>> { unimplemented!() }
    }

    fn decode_fixture(world: usize, staged_heads: u64, partitioned: bool, gpu: bool) -> (wire::Engram, Core) {
        let tokens = 128u64;
        let rows_per_head = 4u64;
        let total_rows = 24 * rows_per_head;
        let mut table = wire::EngramTable {
            layer: 1,
            rows: total_rows,
            embedding_width: 256,
            scale_block: 32,
            hash: Some(wire::EngramHash {
                method: wire::engram_hash::Method::XorModulo as i32,
                orders: (2..=4u32).flat_map(|n| [n; 8]).collect(),
                multipliers: vec![1, 3, 5, 7],
                primes: vec![rows_per_head; 24],
                offsets: (0..24u64).map(|i| i * rows_per_head).collect(),
            }),
            weight_buffer: "engram_1_weights".into(),
            scale_buffer: "engram_1_scales".into(),
            hash_buffer: "engram_1_hashes".into(),
            ready_event: "engram_1_ready".into(),
            encoding: wire::engram_table::Encoding::Fp8E4m3E8m0 as i32,
            request_buffer: if gpu { "engram_1_requests".into() } else { String::new() },
            ..Default::default()
        };
        if partitioned {
            let shard_rows = total_rows / world as u64;
            table.shards = (0..world as u32)
                .map(|rank| wire::EngramTableShard {
                    rank,
                    first_row: rank as u64 * shard_rows,
                    rows: shard_rows,
                    weights: Some(wire::Span {
                        offset: 0,
                        nbytes: shard_rows * 256,
                        ..Default::default()
                    }),
                    scales: Some(wire::Span {
                        offset: shard_rows * 256,
                        nbytes: shard_rows * 8,
                        ..Default::default()
                    }),
                })
                .collect();
        } else {
            table.weights = Some(wire::Span {
                offset: 0,
                nbytes: total_rows * 256,
                ..Default::default()
            });
            table.scales = Some(wire::Span {
                offset: total_rows * 256,
                nbytes: total_rows * 8,
                ..Default::default()
            });
        }
        let catalog = wire::Engram {
            token_map: vec![0, 1, 2, 3],
            pad_token_id: 2,
            compressed_vocabulary: 4,
            tables: vec![table],
        };
        let mut buffers = vec![
            wire::Buffer {
                name: "engram_1_weights".into(),
                dtype: "uint8".into(),
                shape: vec![tokens * staged_heads, 256],
                strides: vec![256, 1],
                nbytes: tokens * staged_heads * 256,
                reference: None,
            },
            wire::Buffer {
                name: "engram_1_scales".into(),
                dtype: "uint8".into(),
                shape: vec![tokens * staged_heads, 8],
                strides: vec![8, 1],
                nbytes: tokens * staged_heads * 8,
                reference: None,
            },
            wire::Buffer {
                name: "engram_1_hashes".into(),
                dtype: "int64".into(),
                shape: vec![tokens, staged_heads],
                strides: vec![staged_heads, 1],
                nbytes: tokens * staged_heads * 8,
                reference: None,
            },
        ];
        let mut allocations = Vec::new();
        if gpu {
            buffers.push(wire::Buffer {
                name: "engram_1_requests".into(),
                dtype: "int64".into(),
                shape: vec![tokens * staged_heads + 1],
                strides: vec![1],
                nbytes: (tokens * staged_heads + 1) * 8,
                reference: Some(wire::Reference {
                    allocation: "host-0".into(),
                    offset: 0,
                }),
            });
            allocations.push(super::super::format::Allocation {
                name: "host-0".into(),
                nbytes: ((tokens * staged_heads + 1) * 8) as usize,
                alignment: 4096,
                file: None,
                data_spans: vec![],
                sources: vec![],
                relocations: vec![],
                address: None,
                memory: Memory::MappedHost,
                sizing: None,
            });
        }
        let mut input_bindings = Vec::new();
        if !gpu {
            input_bindings.push(wire::InputBinding {
                buffer_name: "engram_1_hashes".into(),
                host_offset: 0,
                nbytes: tokens * staged_heads * 8,
            });
        }
        let core = Core {
            resources: Default::default(),
            architecture: [12, 0],
            allocations,
            modules: vec![],
            functions: vec![],
            graphs: vec![],
            reservations: vec![],
            execution: Some(wire::Execution {
                abi_version: 1,
                completion_token_capacity: 8,
                max_active: 1,
                input_frame_bytes: 64,
                buffers,
                graphs: vec![wire::ExecutionGraph {
                    name: "decode128".into(),
                    kind: "decode".into(),
                    batch_capacity: 128,
                    token_capacity: 1,
                    packed_token_capacity: 128,
                    input_bindings,
                    ..Default::default()
                }],
                tuning_graphs: vec![],
                persistent_workers: vec![],
            }),
            context: Default::default(),
            events: vec!["engram_1_ready".into()],
        };
        (catalog, core)
    }

    #[test]
    fn partitioned_gpu_tables_accept_both_legacy_and_compact_local_head_staging() {
        let archive = StubFiles::new();
        let (legacy_catalog, legacy_core) = decode_fixture(4, 24, true, true);
        let legacy = decode(legacy_catalog, &archive, &[legacy_core], 0, 4).unwrap();
        assert_eq!(legacy.tables[0].heads, 24);
        let legacy_staging = crate::engram::staging_bytes(&legacy, 16384).unwrap();

        let (compact_catalog, compact_core) = decode_fixture(4, 6, true, true);
        let compact = decode(compact_catalog, &archive, &[compact_core], 0, 4).unwrap();
        assert_eq!(compact.tables[0].heads, 6);
        let compact_staging = crate::engram::staging_bytes(&compact, 16384).unwrap();
        // 4x fewer staged payload bytes per frame (99.0 MiB -> 24.75 MiB at 16,384 tokens).
        assert_eq!(legacy_staging, 2 * (103_809_024 + 4));
        assert_eq!(compact_staging, 2 * (25_952_256 + 4));

        for bad_heads in [0, 12] {
            let (bad_catalog, bad_core) = decode_fixture(4, bad_heads, true, true);
            assert!(
                decode(bad_catalog, &archive, &[bad_core], 0, 4).is_err(),
                "expected rejection for bad_heads={bad_heads}"
            );
        }

        // Unpartitioned and host-hashed tables still require all 24 canonical heads.
        let (unpart_catalog, unpart_core) = decode_fixture(4, 6, false, true);
        assert!(decode(unpart_catalog, &archive, &[unpart_core], 0, 4).is_err());
        let (host_catalog, host_core) = decode_fixture(4, 6, false, false);
        assert!(decode(host_catalog, &archive, &[host_core], 0, 1).is_err());
    }

    #[test]
    fn prefetch_mailbox_validation_enforces_workspace_bound_and_rank_uniformity() {
        let archive = StubFiles::new();
        let make = |prefetch_len: u64| {
            let (mut catalog, mut core) = decode_fixture(4, 6, true, true);
            catalog.tables[0].prefetch_buffer = "engram_1_prefetch".into();
            core.allocations.push(super::super::format::Allocation {
                name: "prefetch_host".into(),
                nbytes: (prefetch_len * 8) as usize,
                alignment: 4096,
                file: None,
                data_spans: vec![],
                sources: vec![],
                relocations: vec![],
                address: None,
                memory: Memory::MappedHost,
                sizing: None,
            });
            core.execution.as_mut().unwrap().buffers.push(wire::Buffer {
                name: "engram_1_prefetch".into(),
                dtype: "int64".into(),
                shape: vec![prefetch_len],
                strides: vec![1],
                nbytes: prefetch_len * 8,
                reference: Some(wire::Reference {
                    allocation: "prefetch_host".into(),
                    offset: 0,
                }),
            });
            (catalog, core)
        };
        // Fixture has tokens = 128, staged_heads = 6 -> max_staged_rows = 768 -> max len = 770.
        let (ok_cat, ok_core) = make(770);
        let decoded = decode(ok_cat, &archive, &[ok_core], 0, 4).unwrap();
        assert_eq!(
            decoded.tables[0].prefetch,
            Some(("engram_1_prefetch".into(), 768))
        );
        assert_eq!(crate::engram::prefetch_queue_bytes(&decoded).unwrap(), 768 * 8);

        // Below minimum (3) and above workspace ceiling (max_staged_rows + 2 = 770) must reject.
        for bad_len in [2, 771] {
            let (bad_cat, bad_core) = make(bad_len);
            assert!(
                decode(bad_cat, &archive, &[bad_core], 0, 4).is_err(),
                "expected rejection for prefetch_len={bad_len}"
            );
        }

        // Different prefetch capacities across ranks must reject.
        let (cat_a, core_a) = make(100);
        let (_, core_b) = make(200);
        assert!(decode(cat_a, &archive, &[core_a, core_b], 0, 4).is_err());
    }
}
