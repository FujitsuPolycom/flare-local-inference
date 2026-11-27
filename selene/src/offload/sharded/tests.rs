/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::{
    CacheGeometry, CheckpointId, History, HostStore, InboundTicket, PageId, PublishedCheckpoint,
    StateSlotId, StoreEvent, StoreOutcome,
    offload::{self, CopierFactory, PrefixCacheOffload, Preparation},
    testing::{GpuMemory, IdentityRegistrar, SimulatedGpu, TestHistory, geometry},
};
use std::sync::{Arc, Mutex, mpsc};

#[test]
fn service_shutdown_does_not_join_a_thread_that_has_not_finished() {
    let (release, blocked) = mpsc::channel();
    let (finished, outcome) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        blocked.recv().unwrap();
        let _ = finished.send(Ok(()));
        Ok(())
    });
    let service = Service {
        thread: Some(thread),
        finished: Some(outcome),
        pool: None,
        summary: String::new(),
    };
    let (closed, result) = mpsc::channel();
    let shutdown = std::thread::spawn(move || {
        closed
            .send(service.join_timeout(Duration::from_millis(10)))
            .unwrap();
    });
    let result = result.recv_timeout(Duration::from_secs(1));
    release.send(()).unwrap();
    shutdown.join().unwrap();
    assert!(result.unwrap().unwrap_err().to_string().contains("in time"));
}

#[derive(Default)]
struct Copiers(Mutex<std::collections::HashMap<usize, GpuMemory>>);
impl CopierFactory for Copiers {
    fn mapped(
        &self,
        contexts: &[(usize, i32)],
        geometry: Vec<CacheGeometry>,
        pool: &crate::slab::SlabPool,
    ) -> Result<crate::SharedCopier> {
        assert_eq!(
            pool.intake_slots(),
            offload::intake_slots(&crate::Layout::new(&geometry)?, 8)
        );
        assert_eq!(pool.slot_count(), pool.intake_slots());
        let (copier, memory) = SimulatedGpu::shared_mapped(&geometry)?;
        let rank = contexts[0].0;
        for page in 1..=2 {
            memory
                .pages
                .lock()
                .unwrap()
                .insert(page, vec![rank as u8 + page as u8; 4096]);
        }
        memory
            .rows
            .lock()
            .unwrap()
            .insert(1, vec![rank as u8 + 10; 3 * 4096]);
        self.0.lock().unwrap().insert(rank, memory);
        Ok(copier)
    }
    fn unmapped(
        &self,
        _: &[(usize, i32)],
        _: Vec<CacheGeometry>,
    ) -> Result<Box<dyn crate::PageCopier>> {
        panic!("this test has no manager pool")
    }
}

fn completed(store: &mut dyn HostStore) -> StoreEvent {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let mut events = Vec::new();
        store.poll(&mut events).unwrap();
        if !events.is_empty() {
            assert_eq!(events.len(), 1);
            return events.remove(0);
        }
        assert!(Instant::now() < deadline, "record transfer stalled");
        std::thread::yield_now();
    }
}

#[test]
fn sharded_export_and_restore_use_each_ranks_real_record_file() {
    let directory = crate::testing::test_directory("sharded", "real-records").unwrap();
    let factory = Arc::new(Copiers::default());
    let local = geometry(64, &[("kv", 4096)], 3 * 4096);
    let prepare = |rank| {
        let options = PrefixCacheOffload {
            memory_bytes: Some(0),
            disk: Some(directory.join(format!("rank-{rank}"))),
            disk_limit: Some(1 << 20),
            ..Default::default()
        };
        let preparation = offload::prepare_with_inputs(
            &options,
            [3; 32],
            vec![local.clone()],
            vec![(rank, 0)],
            offload::PrepareInputs {
                max_active: 8,
                input_staging: 0,
                shard: Some((rank, 2)),
                integrated: false,
            },
            &IdentityRegistrar,
        )
        .unwrap();
        let Preparation::Store(_) = &preparation else {
            panic!("no disk store");
        };
        (
            preparation.into_shared().unwrap(),
            prepared_directory(&options, rank, &local),
        )
    };
    let (zero, zero_directory) = prepare(0);
    let (peer, peer_directory) = prepare(1);
    let (offer_tx, offer_rx) = mpsc::channel();
    let (reply_tx, reply_rx) = mpsc::channel();
    let peer_factory = factory.clone();
    let service = std::thread::spawn(move || {
        join_peer(Some(&peer), 1, peer_factory.as_ref(), |record| {
            reply_tx.send(record.to_vec()).unwrap();
            let offer = offer_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            Ok(vec![offer, record.to_vec()])
        })
        .unwrap()
        .unwrap()
    });
    let address = "127.0.0.1:1".parse().unwrap();
    join_rank_zero(Some(&zero), &[(address, address)], |mut offers| {
        offer_tx.send(offers.remove(1)).unwrap();
        Ok(vec![
            vec![],
            reply_rx.recv_timeout(Duration::from_secs(10)).unwrap(),
        ])
    })
    .unwrap();
    let service = service.join().unwrap();
    let mut store = offload::open(zero.take().unwrap(), factory.as_ref()).unwrap();
    let intake = offload::intake_slots(&crate::Layout::new(&[local]).unwrap(), 8);
    let stats = store.stats();
    assert_eq!(stats.slots_used, 0);
    assert_eq!(stats.slots_total, intake);
    assert_eq!(stats.intake_slots, intake);
    let mut history = TestHistory::new([3; 32]);
    history.append(&[7; 129], &[]).unwrap();
    let key = history.key_at(128).unwrap();
    let checkpoint = PublishedCheckpoint {
        capture_role: crate::CaptureRole::Endpoint,
        id: CheckpointId::new(1),
        key,
        ancestors: Vec::new(),
        bucket: key.digest,
        chain_pages: vec![PageId(1), PageId(2)],
        tail_pages: 2,
        slot: StateSlotId(1),
    };
    assert_eq!(store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
    assert!(matches!(
        completed(store.as_mut()),
        StoreEvent::Stored { result: Ok(_), .. }
    ));
    for directory in [&zero_directory, &peer_directory] {
        assert!(directory.join(record::file_name(&key)).is_file());
    }
    let hit = store.lookup(&history, 0, u64::MAX).unwrap();
    store
        .fetch(
            hit,
            InboundTicket {
                key,
                pages: vec![PageId(3), PageId(4)],
                slot: StateSlotId(2),
            },
        )
        .unwrap();
    assert!(matches!(
        completed(store.as_mut()),
        StoreEvent::Fetched { result: Ok(_), .. }
    ));
    let memories = factory.0.lock().unwrap();
    for rank in [0, 1] {
        let memory = &memories[&rank];
        let pages = memory.pages.lock().unwrap();
        assert_eq!(pages[&3], vec![rank as u8 + 1; 4096]);
        assert_eq!(pages[&4], vec![rank as u8 + 2; 4096]);
        assert_eq!(
            memory.rows.lock().unwrap()[&2],
            vec![rank as u8 + 10; 3 * 4096]
        );
    }
    store.close().unwrap();
    drop(store);
    service.join().unwrap();
    drop(zero);
    std::fs::remove_dir_all(directory).unwrap();
}

fn prepared_directory(
    options: &PrefixCacheOffload,
    rank: usize,
    local: &CacheGeometry,
) -> std::path::PathBuf {
    options.disk.as_ref().unwrap().join(record::directory_name(
        &[3; 32],
        &crate::geometry::shard_fingerprint(local, rank, 2),
    ))
}

#[cfg(feature = "codec")]
#[test]
fn sharded_join_rejects_mismatched_compression_modes() {
    let directory = crate::testing::test_directory("sharded", "compression-mismatch").unwrap();
    let factory = Arc::new(Copiers::default());
    let mut local_cpu = geometry(64, &[("kv", 4096)], 3 * 4096);
    local_cpu.placement = offload::KvCompression::Cpu;
    for plane in local_cpu.planes.iter().chain([&local_cpu.row]) {
        local_cpu.codec_fields.push(crate::geometry::CodecField {
            buffer: plane.name.clone(),
            offset: 0,
            length: plane.unit_bytes,
            encoding: 0,
        });
    }
    let local_off = geometry(64, &[("kv", 4096)], 3 * 4096);
    let prepare = |rank: usize, geo: CacheGeometry, compression: offload::KvCompression| {
        let options = PrefixCacheOffload {
            compression,
            memory_bytes: Some(0),
            disk: Some(directory.join(format!("rank-{rank}"))),
            disk_limit: Some(1 << 20),
            ..Default::default()
        };
        offload::prepare_with_inputs(
            &options,
            [3; 32],
            vec![geo],
            vec![(rank, 0)],
            offload::PrepareInputs {
                max_active: 8,
                input_staging: 0,
                shard: Some((rank, 2)),
                integrated: false,
            },
            &IdentityRegistrar,
        )
        .unwrap()
        .into_shared()
        .unwrap()
    };
    let zero = prepare(0, local_off, offload::KvCompression::Off);
    let peer = prepare(1, local_cpu, offload::KvCompression::Cpu);
    let (offer_tx, offer_rx) = mpsc::channel();
    let (reply_tx, reply_rx) = mpsc::channel();
    let peer_factory = factory.clone();
    let service = std::thread::spawn(move || {
        join_peer(Some(&peer), 1, peer_factory.as_ref(), |record| {
            reply_tx.send(record.to_vec()).unwrap();
            let offer = offer_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            Ok(vec![offer, record.to_vec()])
        })
    });
    let address = "127.0.0.1:1".parse().unwrap();
    let zero_err = join_rank_zero(Some(&zero), &[(address, address)], |mut offers| {
        offer_tx.send(offers.remove(1)).unwrap();
        Ok(vec![
            vec![],
            reply_rx.recv_timeout(Duration::from_secs(10)).unwrap(),
        ])
    })
    .unwrap_err();
    let peer_err = service.join().unwrap().err().expect("peer must reject compression mismatch");
    assert!(
        zero_err.to_string().contains("compression"),
        "unexpected rank-0 error: {zero_err}"
    );
    assert!(
        peer_err.to_string().contains("compression"),
        "unexpected peer error: {peer_err}"
    );
    drop(zero);
    std::fs::remove_dir_all(directory).unwrap();
}

#[cfg(feature = "codec")]
#[test]
fn sharded_compressed_v4_store_fetch_truncate_and_restart_agree() {
    use crate::copier::PageCopier;

    #[derive(Default)]
    struct CodecCopiers(Mutex<std::collections::HashMap<usize, GpuMemory>>);
    impl CopierFactory for CodecCopiers {
        fn mapped(
            &self,
            contexts: &[(usize, i32)],
            geometry: Vec<CacheGeometry>,
            _pool: &crate::slab::SlabPool,
        ) -> Result<crate::SharedCopier> {
            let ([page_plan, row_plan], profile) =
                crate::shared::codec_layout::plans_and_profile(
                    &geometry[0],
                    crate::shared::codec_layout::REVISION_CPU,
                )
                .unwrap();
            let layout = crate::Layout::new(&geometry)?;
            let rank = contexts[0].0;
            let gpu = GpuMemory::default();
            // Give rank 0 highly compressible zeros and rank 1 patterned data so both
            // compress as v4 packets with DIFFERENT physical file sizes and row offsets.
            for page in 1..=2u32 {
                let data: Vec<u8> = if rank == 0 {
                    vec![page as u8; 4096]
                } else {
                    (0..4096usize)
                        .map(|i| ((i / 7 + page as usize * 13) % 19) as u8)
                        .collect()
                };
                gpu.pages.lock().unwrap().insert(page, data);
            }
            let row_data: Vec<u8> = if rank == 0 {
                vec![7u8; 3 * 4096]
            } else {
                (0..3 * 4096usize).map(|i| ((i / 5) % 23) as u8).collect()
            };
            gpu.rows.lock().unwrap().insert(1, row_data);
            self.0.lock().unwrap().insert(rank, gpu.clone());
            Ok(crate::testing::CodecCopier::new(
                layout,
                profile,
                Arc::new(page_plan),
                Arc::new(row_plan),
                gpu,
            )
            .shared())
        }
        fn unmapped(
            &self,
            _: &[(usize, i32)],
            _: Vec<CacheGeometry>,
        ) -> Result<Box<dyn PageCopier>> {
            panic!("no manager pool")
        }
    }

    let directory = crate::testing::test_directory("sharded", "v4-compressed").unwrap();
    let mut local = geometry(64, &[("kv", 4096)], 3 * 4096);
    local.placement = offload::KvCompression::Cpu;
    for plane in local.planes.iter().chain([&local.row]) {
        local.codec_fields.push(crate::geometry::CodecField {
            buffer: plane.name.clone(),
            offset: 0,
            length: plane.unit_bytes,
            encoding: 0,
        });
    }
    let prepare = |rank: usize| {
        let options = PrefixCacheOffload {
            compression: offload::KvCompression::Cpu,
            memory_bytes: Some(0),
            disk: Some(directory.join(format!("rank-{rank}"))),
            disk_limit: Some(1 << 20),
            ..Default::default()
        };
        let prep = offload::prepare_with_inputs(
            &options,
            [3; 32],
            vec![local.clone()],
            vec![(rank, 0)],
            offload::PrepareInputs {
                max_active: 8,
                input_staging: 0,
                shard: Some((rank, 2)),
                integrated: false,
            },
            &IdentityRegistrar,
        )
        .unwrap();
        (
            prep.into_shared().unwrap(),
            prepared_directory(&options, rank, &local),
        )
    };
    let join_pair = |zero: &Shared, peer: Arc<Shared>, factory: Arc<CodecCopiers>| {
        let (offer_tx, offer_rx) = mpsc::channel();
        let (reply_tx, reply_rx) = mpsc::channel();
        let service = std::thread::spawn(move || {
            join_peer(Some(&peer), 1, factory.as_ref(), |record| {
                reply_tx.send(record.to_vec()).unwrap();
                let offer = offer_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                Ok(vec![offer, record.to_vec()])
            })
            .unwrap()
            .unwrap()
        });
        let address = "127.0.0.1:1".parse().unwrap();
        join_rank_zero(Some(zero), &[(address, address)], |mut offers| {
            offer_tx.send(offers.remove(1)).unwrap();
            Ok(vec![
                vec![],
                reply_rx.recv_timeout(Duration::from_secs(10)).unwrap(),
            ])
        })
        .unwrap();
        service.join().unwrap()
    };

    let factory = Arc::new(CodecCopiers::default());
    let (zero, zero_dir) = prepare(0);
    let (peer, peer_dir) = prepare(1);
    let service = join_pair(&zero, peer, factory.clone());
    let mut store = offload::open(zero.take().unwrap(), factory.as_ref()).unwrap();

    let mut history = TestHistory::new([3; 32]);
    history.append(&[9; 129], &[]).unwrap();
    let key = history.key_at(128).unwrap();
    let checkpoint = PublishedCheckpoint {
        capture_role: crate::CaptureRole::Endpoint,
        id: CheckpointId::new(1),
        key,
        ancestors: Vec::new(),
        bucket: key.digest,
        chain_pages: vec![PageId(1), PageId(2)],
        tail_pages: 2,
        slot: StateSlotId(1),
    };
    assert_eq!(store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
    assert!(matches!(
        completed(store.as_mut()),
        StoreEvent::Stored { result: Ok(_), .. }
    ));

    // Verify both ranks wrote v4 packed files with distinct physical lengths.
    let zero_path = zero_dir.join(record::file_name(&key));
    let peer_path = peer_dir.join(record::file_name(&key));
    let zero_meta = record::Header::read_metadata(&mut std::fs::File::open(&zero_path).unwrap()).unwrap();
    let peer_meta = record::Header::read_metadata(&mut std::fs::File::open(&peer_path).unwrap()).unwrap();
    let zero_hdr = record::Header::decode(&zero_meta).unwrap();
    let peer_hdr = record::Header::decode(&peer_meta).unwrap();
    assert!(zero_hdr.packed.is_some() && peer_hdr.packed.is_some());
    assert_ne!(
        zero_hdr.row_segment, peer_hdr.row_segment,
        "test requires distinct per-rank compressed offsets"
    );

    // Fetch restores each rank's v4 file byte-identically.
    let hit = store.lookup(&history, 0, u64::MAX).unwrap();
    store
        .fetch(
            hit,
            InboundTicket {
                key,
                pages: vec![PageId(3), PageId(4)],
                slot: StateSlotId(2),
            },
        )
        .unwrap();
    assert!(matches!(
        completed(store.as_mut()),
        StoreEvent::Fetched { result: Ok(_), .. }
    ));
    {
        let memories = factory.0.lock().unwrap();
        for rank in [0, 1] {
            let mem = &memories[&rank];
            let pages = mem.pages.lock().unwrap();
            let rows = mem.rows.lock().unwrap();
            assert_eq!(pages[&3], pages[&1]);
            assert_eq!(pages[&4], pages[&2]);
            assert_eq!(rows[&2], rows[&1]);
        }
    }
    store.close().unwrap();
    drop(store);
    service.join().unwrap();

    // Restart both ranks and verify startup agreement retains the differing-length v4 files.
    let factory2 = Arc::new(CodecCopiers::default());
    let (zero2, _) = prepare(0);
    let (peer2, _) = prepare(1);
    let service2 = join_pair(&zero2, peer2, factory2.clone());
    let mut store2 = offload::open(zero2.take().unwrap(), factory2.as_ref()).unwrap();
    let hit2 = store2
        .lookup(&history, 0, u64::MAX)
        .expect("restarted sharded store must retain agreed v4 record");
    store2
        .fetch(
            hit2,
            InboundTicket {
                key,
                pages: vec![PageId(5), PageId(6)],
                slot: StateSlotId(3),
            },
        )
        .unwrap();
    assert!(matches!(
        completed(store2.as_mut()),
        StoreEvent::Fetched { result: Ok(_), .. }
    ));
    {
        let memories = factory2.0.lock().unwrap();
        for rank in [0, 1] {
            let mem = &memories[&rank];
            let pages = mem.pages.lock().unwrap();
            let rows = mem.rows.lock().unwrap();
            assert_eq!(pages[&5], pages[&1]);
            assert_eq!(pages[&6], pages[&2]);
            assert_eq!(rows[&3], rows[&1]);
        }
    }
    store2.close().unwrap();
    drop(store2);
    service2.join().unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn sharded_pool_with_compression_rejected_at_startup() {
    let local = geometry(64, &[("kv", 4096)], 3 * 4096);
    for compression in [offload::KvCompression::Cpu, offload::KvCompression::Gpu] {
        let options = PrefixCacheOffload {
            compression,
            pool: Some(crate::pool::PoolOptions {
                address: "127.0.0.1:9200".to_owned(),
                ca: None,
            }),
            ..Default::default()
        };
        let err = offload::prepare_with_inputs(
            &options,
            [1; 32],
            vec![local.clone()],
            vec![(0, 0)],
            offload::PrepareInputs {
                max_active: 4,
                input_staging: 0,
                shard: Some((0, 2)),
                integrated: false,
            },
            &IdentityRegistrar,
        )
        .err()
        .expect("sharded --ranks + --prefix-cache-pool + compression must be rejected");
        assert!(
            err.to_string().contains("multi-host --rank"),
            "unexpected error: {err}"
        );
    }
}
