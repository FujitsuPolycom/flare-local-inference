/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

fn inputs(max_active: usize) -> PrepareInputs {
    PrepareInputs {
        max_active,
        input_staging: 0,
        shard: None,
        integrated: false,
    }
}

fn prepare(
    options: &PrefixCacheOffload,
    identity: [u8; 32],
    geometry: Vec<CacheGeometry>,
    contexts: Vec<(usize, i32)>,
) -> Result<Preparation> {
    super::prepare_with_inputs(
        options,
        identity,
        geometry,
        contexts,
        inputs(1),
        &crate::testing::IdentityRegistrar,
    )
}
fn prepare_with_inputs(
    options: &PrefixCacheOffload,
    identity: [u8; 32],
    geometry: Vec<CacheGeometry>,
    contexts: Vec<(usize, i32)>,
    input_staging: u64,
    shard: Option<(usize, usize)>,
) -> Result<Preparation> {
    super::prepare_with_inputs(
        options,
        identity,
        geometry,
        contexts,
        PrepareInputs {
            max_active: 1,
            input_staging,
            shard,
            integrated: false,
        },
        &crate::testing::IdentityRegistrar,
    )
}

fn test_directory() -> PathBuf {
    static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "flare-offload-disabled-{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

fn pool(address: &str) -> PoolOptions {
    PoolOptions {
        address: address.into(),
        ca: None,
    }
}

#[test]
fn a_pool_with_shards_keeps_one_local_rank_and_refuses_memory_retention() {
    let mut options = PrefixCacheOffload {
        pool: Some(pool("unix:/run/selene.sock")),
        ..Default::default()
    };
    let local = crate::testing::geometry(64, &[("kv", 4096)], 4096);
    let prepare = |options: &PrefixCacheOffload| {
        prepare_with_inputs(
            options,
            [0; 32],
            vec![local.clone()],
            vec![(0, 0)],
            0,
            Some((1, 2)),
        )
    };
    let Preparation::Store(prepared) = prepare(&options).unwrap() else {
        panic!("no pool store")
    };
    assert_eq!(prepared.geometry.len(), 1);
    assert!(prepared.summary().contains("across rank processes: restores only"));
    assert!(prepared.memory_pool.is_none());
    assert_eq!(prepared.warm_bytes, 0);
    assert_eq!(
        prepared.fingerprint,
        geometry::shard_fingerprint(&local, 1, 2)
    );
    for pooled in [false, true] {
        for disk in [false, true] {
            options.pool = pooled.then(|| pool("unix:/run/selene.sock"));
            options.disk = disk.then(|| test_directory());
            options.memory_bytes = Some(1);
            assert!(
                prepare(&options)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("disk only")
            );
            if let Some(directory) = &options.disk {
                assert!(!directory.exists());
            }
        }
    }
}

fn qualified_rank() -> CacheGeometry {
    let mut g = crate::testing::geometry(64, &[("kv", 4096)], 4096);
    for name in ["kv", "state"] {
        g.codec_fields.push(crate::geometry::CodecField {
            buffer: name.into(),
            offset: 0,
            length: 4096,
            encoding: 0,
        });
    }
    g
}

#[test]
fn restore_placement_preserves_warm_capacity() {
    let memory = 4 << 20;
    let g = qualified_rank();
    for placement in [KvCompression::Off, KvCompression::Cpu] {
        let options = PrefixCacheOffload {
            memory_bytes: Some(memory),
            compression: placement,
            pool: Some(pool("unix:/run/selene.sock")),
            ..Default::default()
        };
        let Preparation::Store(p) = prepare(&options, [0; 32], vec![g.clone()], vec![(0, 0)])
            .unwrap()
        else {
            panic!("missing store")
        };
        let pool = p.memory_pool.unwrap();
        assert_eq!(p.warm_bytes, memory);
        let warm_slots = (memory / 4096) as usize;
        let warm = pool.allocate(warm_slots).unwrap();
        assert_eq!(pool.free_slots(), pool.intake_slots());
        assert_eq!(pool.bytes(), memory + pool.intake_slots() as u64 * 4096);
        pool.release(warm);
    }
}

#[test]
fn disk_only_restore_preserves_minimum_staging() {
    let directory = test_directory();
    let options = PrefixCacheOffload {
        memory_bytes: Some(0),
        disk: Some(directory.clone()),
        disk_limit: Some(4 << 20),
        ..Default::default()
    };
    for max_active in [1, 9] {
        let Preparation::Store(p) = super::prepare_with_inputs(
            &options,
            [0; 32],
            vec![qualified_rank()],
            vec![(0, 0)],
            inputs(max_active),
            &crate::testing::IdentityRegistrar,
        )
        .unwrap() else {
            panic!("missing disk store")
        };
        let (_, budget) = options
            .headroom_with_inputs(1, &p.layout, inputs(max_active), None)
            .unwrap();
        let pool = p.memory_pool.as_ref().unwrap();
        assert_eq!(pool.bytes(), budget);
        assert!(pool.intake_slots() >= intake_slots(&p.layout, 1));
        assert_eq!(pool.free_slots(), pool.intake_slots());
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn without_a_pool_nothing_for_it_is_budgeted_or_built() {
    let layout = Layout::new(&[crate::testing::geometry(64, &[("kv", 4096)], 4096)]).unwrap();
    assert_eq!(
        PrefixCacheOffload::default()
            .headroom_with_inputs(1, &layout, inputs(1), None)
            .unwrap(),
        (0, 0)
    );
    let tiers = PrefixCacheOffload {
        memory_bytes: Some(1 << 30),
        ..Default::default()
    };
    let pooled = PrefixCacheOffload {
        pool: Some(pool("host:7000")),
        ..tiers.clone()
    };
    let (device, host) = tiers
        .headroom_with_inputs(1, &layout, inputs(1), None)
        .unwrap();
    // Static headroom covers the copier; Hello's regions are budgeted at startup.
    assert_eq!(
        pooled
            .headroom_with_inputs(1, &layout, inputs(1), None)
            .unwrap(),
        (device + STORE_DEVICE_MARGIN_BYTES, host)
    );
    // A pool shares what the CPU reads, so its memory is mapped whole by the device
    // instead of split across ranks.
    let split = tiers
        .headroom_with_inputs(2, &layout, inputs(1), None)
        .unwrap()
        .0;
    let whole = pooled
        .headroom_with_inputs(2, &layout, inputs(1), None)
        .unwrap()
        .0;
    assert!(whole > split + STORE_DEVICE_MARGIN_BYTES, "{whole} {split}");
    let alone = PrefixCacheOffload {
        pool: Some(pool("host:7000")),
        ..Default::default()
    };
    assert!(alone.enabled());
    assert_eq!(
        alone
            .headroom_with_inputs(1, &layout, inputs(1), None)
            .unwrap(),
        (STORE_DEVICE_MARGIN_BYTES, 0)
    );
}

#[test]
fn disabled_preparation_requires_no_geometry_context_or_worker_binding() {
    for options in [
        PrefixCacheOffload::default(),
        PrefixCacheOffload {
            memory_bytes: Some(0),
            ..Default::default()
        },
        PrefixCacheOffload {
            memory_bytes: Some(0),
            disk_limit: Some(0),
            ..Default::default()
        },
    ] {
        let prepared = prepare(&options, [0; 32], Vec::new(), Vec::new()).unwrap();
        assert_eq!(prepared.summary(), "disabled");
        assert!(matches!(&prepared, Preparation::None(reason) if reason == "disabled"));
        assert!(prepared.into_shared().is_none());
    }
}

#[test]
fn zero_disk_limit_does_not_create_a_record_directory() {
    let directory = test_directory();
    let prepared = prepare(
        &PrefixCacheOffload {
            disk: Some(directory.clone()),
            disk_limit: Some(0),
            ..Default::default()
        },
        [0; 32],
        vec![crate::testing::geometry(64, &[("kv", 4096)], 4096)],
        Vec::new(),
    )
    .unwrap();
    let created = directory.exists();
    if created {
        std::fs::remove_dir_all(&directory).unwrap();
    }
    assert!(
        !created,
        "disabled disk store created {}",
        directory.display()
    );
    assert!(prepared.into_shared().is_none());
}

#[test]
fn zero_disk_limit_avoids_disk_setup_when_memory_is_enabled() {
    let directory = test_directory();
    let geometry = vec![crate::testing::geometry(64, &[("kv", 4096)], 4096)];
    let memory_bytes = 4096;
    let error = prepare(
        &PrefixCacheOffload {
            memory_bytes: Some(memory_bytes),
            disk: Some(directory.clone()),
            disk_limit: Some(0),
            ..Default::default()
        },
        [0; 32],
        geometry,
        Vec::new(),
    )
    .err()
    .expect("a memory store requires a CUDA context");
    let created = directory.exists();
    if created {
        std::fs::remove_dir_all(&directory).unwrap();
    }
    assert_eq!(
        error.to_string(),
        "the host prefix store requires at least one rank"
    );
    assert!(
        !created,
        "disabled disk store created {}",
        directory.display()
    );
}

#[test]
fn zero_disk_limit_does_not_lock_or_scan_existing_records() {
    let base = test_directory();
    let geometry = vec![crate::testing::geometry(64, &[("kv", 4096)], 4096)];
    let fingerprint = geometry::fingerprint(&geometry);
    let directory = base.join(record::directory_name(&[0; 32], &fingerprint));
    std::fs::create_dir_all(&directory).unwrap();
    let lock = crate::testing::directory_lock(&directory).unwrap();
    let active_write = directory.join("active-record.tmp");
    std::fs::write(&active_write, b"active writer payload").unwrap();
    let prepared = prepare(
        &PrefixCacheOffload {
            disk: Some(base.clone()),
            disk_limit: Some(0),
            ..Default::default()
        },
        [0; 32],
        geometry,
        Vec::new(),
    )
    .unwrap();
    assert!(prepared.into_shared().is_none());
    assert_eq!(
        std::fs::read(active_write).unwrap(),
        b"active writer payload"
    );
    drop(lock);
    std::fs::remove_dir_all(base).unwrap();
}

#[test]
fn sizes_parse_binary_suffixes_like_the_ignition_configuration() {
    assert_eq!(parse_size("16G").unwrap(), 16 << 30);
    assert_eq!(parse_size("1.5 GiB").unwrap(), 3 << 29);
    assert_eq!(parse_size("512m").unwrap(), 512 << 20);
    assert_eq!(parse_size("2TB").unwrap(), 2 << 40);
    assert_eq!(parse_size("4096").unwrap(), 4096);
    assert_eq!(parse_size("0").unwrap(), 0);
    for text in ["", "G", "16X", "1.2.3G", "-1G", "16 GB extra"] {
        assert!(parse_size(text).is_err(), "{text:?}");
    }
}

#[test]
fn memory_headroom_accounts_for_warm_slots_and_intake() {
    let layout = Layout {
        page_tokens: 64,
        ranks: vec![crate::geometry::RankLayout {
            page_offset: 0,
            plane_offsets: vec![0],
            plane_bytes: vec![32 << 20],
            row_bytes: 8 << 20,
            row_offset: 0,
        }],
        page_record_bytes: 32 << 20,
        row_record_bytes: 8 << 20,
    };
    let mut options = PrefixCacheOffload::default();
    assert_eq!(
        options
            .headroom_with_inputs(2, &layout, inputs(1), None)
            .unwrap(),
        (0, 0)
    );
    let intake = intake_slots(&layout, 1) as u64 * layout.page_record_bytes;
    options.memory_bytes = Some(16 << 30);
    assert_eq!(
        options
            .headroom_with_inputs(2, &layout, inputs(1), None)
            .unwrap(),
        (
            ((16u64 << 30) + intake).div_ceil(2).div_ceil(512) + STORE_DEVICE_MARGIN_BYTES,
            (16 << 30) + intake
        )
    );
    options.disk = Some("records".into());
    let host = (16u64 << 30) + intake;
    assert_eq!(
        options
            .headroom_with_inputs(2, &layout, inputs(1), None)
            .unwrap(),
        (host.div_ceil(512) + STORE_DEVICE_MARGIN_BYTES, host)
    );
    options.memory_bytes = Some(0);
    assert_eq!(
        options
            .headroom_with_inputs(2, &layout, inputs(1), None)
            .unwrap(),
        (intake.div_ceil(512) + STORE_DEVICE_MARGIN_BYTES, intake)
    );
    options.memory_bytes = Some(16 << 30);
    options.disk_limit = Some(0);
    assert_eq!(
        options
            .headroom_with_inputs(2, &layout, inputs(1), None)
            .unwrap()
            .1,
        (16 << 30) + intake
    );
}

#[test]
fn ordinary_input_share_preserves_mapping() {
    assert_eq!(input_memory_bytes(0, true), 0);
    assert_eq!(input_memory_bytes(8192, true), 0);
    assert_eq!(input_memory_bytes(1 << 30, true), 128 << 20);
    assert_eq!(input_memory_bytes(1 << 30, false), 0);
    let layout = Layout {
        page_tokens: 64,
        ranks: vec![crate::geometry::RankLayout {
            page_offset: 0,
            plane_offsets: vec![0],
            plane_bytes: vec![4096],
            row_bytes: 4096,
            row_offset: 0,
        }],
        page_record_bytes: 4096,
        row_record_bytes: 4096,
    };
    let options = PrefixCacheOffload {
        memory_bytes: Some(1 << 30),
        ..Default::default()
    };
    let ordinary = input_memory_bytes(1 << 30, true);
    let (device, host) = options
        .headroom_with_inputs(
            2,
            &layout,
            PrepareInputs {
                input_staging: 4096,
                ..inputs(1)
            },
            None,
        )
        .unwrap();
    let intake = intake_slots(&layout, 1) as u64 * layout.page_record_bytes;
    assert_eq!(host, (1 << 30) + intake);
    assert_eq!(
        device,
        ((1u64 << 30) - ordinary + intake).div_ceil(2).div_ceil(512) + STORE_DEVICE_MARGIN_BYTES
    );
}

#[test]
fn intake_warning_reports_shrinkage_and_the_required_memory() {
    let geometry = vec![crate::testing::geometry(64, &[("kv", 4096)], 4096)];
    let layout = Layout::new(&geometry).unwrap();
    let options = PrefixCacheOffload {
        memory_bytes: Some(layout.page_record_bytes * 128),
        pool: Some(pool("unix:/run/selene.sock")),
        ..Default::default()
    };
    let build = |options: &PrefixCacheOffload| {
        let Preparation::Store(prepared) = super::prepare_with_inputs(
            options,
            [0; 32],
            geometry.clone(),
            vec![(0, 0)],
            inputs(32),
            &crate::testing::IdentityRegistrar,
        )
        .unwrap() else {
            panic!("no store")
        };
        prepared
    };
    let mut prepared = build(&options);
    let required = intake_slots(&layout, 32);
    let bytes = required as u64 * layout.page_record_bytes;
    for warm in [0, prepared.warm_bytes] {
        prepared.warm_bytes = warm;
        for allocated in [
            required - 1,
            intake_slots(&layout, 1),
            required,
            required + 1,
        ] {
            prepared.memory_pool = Some(
                crate::slab::SlabPool::new(layout.page_record_bytes, 128 + allocated, allocated)
                    .unwrap(),
            );
            let warning = prepared.intake_warning();
            if allocated >= required {
                assert!(warning.is_none());
            } else {
                assert_eq!(
                    warning.unwrap(),
                    format!(
                        "host prefix intake shrunk to {allocated} slots to fit the host budget; \
                     full intake at max concurrency 32 needs {bytes} bytes ({}) of host memory",
                        gibibytes(bytes)
                    )
                );
            }
        }
    }
}

#[test]
fn pool_only_sizes_registered_capture_and_headroom_for_local_ranks() {
    let options = PrefixCacheOffload {
        pool: Some(pool("unix:/run/selene.sock")),
        ..Default::default()
    };
    for pages in [10, 512] {
        let mut ranks = vec![crate::testing::geometry(64, &[("kv", 4096)], 8192); 2];
        ranks[0].page_capacity = 8;
        ranks[1].page_capacity = pages;
        let registrar = crate::testing::TrackingRegistrar::default();
        let Preparation::Store(prepared) = super::prepare_with_inputs(
            &options,
            [0; 32],
            ranks,
            vec![(0, 0), (1, 1)],
            inputs(4),
            &registrar,
        )
        .unwrap() else {
            panic!("missing capture buffer");
        };
        let layout = &prepared.layout;
        let slots = intake_slots(layout, 4).max(pages as usize + layout.row_slots());
        let bytes = slots as u64 * layout.page_record_bytes;
        let buffer = prepared.memory_pool.as_ref().unwrap();
        assert_eq!((buffer.slot_count(), buffer.intake_slots()), (slots, 0));
        assert!(buffer.rank_parts().is_none());
        assert_eq!(registrar.calls.lock().unwrap().len(), 1);
        assert_eq!(prepared.warm_bytes, 0);
        assert!(prepared.intake_warning().is_none());
        assert!(prepared.summary().contains(&format!("pool only: capture buffer {}", gibibytes(bytes))));
        assert_eq!(
            options
                .headroom_with_inputs(2, layout, inputs(4), Some(pages))
                .unwrap(),
            (2 * STORE_DEVICE_MARGIN_BYTES + bytes.div_ceil(512), bytes)
        );
    }
}

struct RecordingFactory {
    maps: std::cell::RefCell<Vec<usize>>,
    closed: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
    fail: Option<usize>,
}
struct ClosingCopier {
    layout: Layout,
    id: usize,
    closed: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
    fail: bool,
}
impl crate::PageCopier for ClosingCopier {
    fn layout(&self) -> &Layout {
        &self.layout
    }
    fn submit(&mut self, _: crate::Job) -> Result<()> {
        Err("no copies expected".into())
    }
    fn poll(&mut self, _: &mut Vec<crate::Reply>) {}
    fn in_flight(&self) -> usize {
        0
    }
    fn close(&mut self) -> Result<()> {
        self.closed.lock().unwrap().push(self.id);
        if self.fail {
            Err("copier failed to drain".into())
        } else {
            Ok(())
        }
    }
}
impl CopierFactory for RecordingFactory {
    fn mapped(
        &self,
        _: &[(usize, i32)],
        geometry: Vec<CacheGeometry>,
        pool: &slab::SlabPool,
    ) -> Result<crate::SharedCopier> {
        let mut maps = self.maps.borrow_mut();
        let id = maps.len();
        maps.push(pool.slot_address(slab::Slot(0)));
        Ok(std::sync::Arc::new(std::sync::Mutex::new(ClosingCopier {
            layout: Layout::new(&geometry)?,
            id,
            closed: self.closed.clone(),
            fail: self.fail == Some(id),
        })))
    }
    fn unmapped(
        &self,
        _: &[(usize, i32)],
        _: Vec<CacheGeometry>,
    ) -> Result<Box<dyn crate::PageCopier>> {
        unreachable!()
    }
}

#[test]
fn disk_and_memory_share_one_registration_with_independent_copiers_and_safe_close() {
    for warm in [0, 16 * 4096] {
        for fail in [None, Some(0), Some(1)]
            .into_iter()
            .filter(|fail| warm != 0 || *fail != Some(1))
        {
            let directory = test_directory();
            let registrar = crate::testing::TrackingRegistrar::default();
            let options = PrefixCacheOffload {
                memory_bytes: Some(warm),
                disk: Some(directory.clone()),
                disk_limit: Some(1 << 30),
                ..Default::default()
            };
            let Preparation::Store(prepared) = super::prepare_with_inputs(
                &options,
                [0; 32],
                vec![crate::testing::geometry(64, &[("kv", 4096)], 4096)],
                vec![(0, 0)],
                inputs(1),
                &registrar,
            )
            .unwrap() else {
                panic!("no store")
            };
            assert_eq!(registrar.calls.lock().unwrap().len(), 1);
            let pool = prepared.memory_pool.as_ref().unwrap().clone();
            let slot = pool.allocate_page().unwrap();
            let factory = RecordingFactory {
                maps: Default::default(),
                closed: Default::default(),
                fail,
            };
            let mut store = open(*prepared, &factory).unwrap();
            let maps = factory.maps.borrow();
            assert_eq!(maps.len(), if warm == 0 { 1 } else { 2 });
            assert!(
                maps.iter()
                    .all(|&address| address == pool.slot_address(slab::Slot(0)))
            );
            assert_eq!(store.close().is_err(), fail.is_some());
            let order = factory.closed.lock().unwrap().clone();
            assert_eq!(
                order,
                if warm == 0 {
                    vec![0]
                } else if fail == Some(1) {
                    vec![1]
                } else {
                    vec![1, 0]
                }
            );
            drop(store);
            assert_eq!(*registrar.drops.lock().unwrap(), 0);
            pool.release_page(slot);
            if fail.is_some() {
                assert_eq!(pool.free_slots(), 0);
                assert_eq!(pool.allocate_page(), None);
            }
            drop(pool);
            assert_eq!(
                *registrar.drops.lock().unwrap(),
                usize::from(fail.is_none())
            );
            std::fs::remove_dir_all(directory).unwrap();
        }
    }
}

#[test]
fn intake_sizes_active_rows_and_rounds_warm_memory_down() {
    let geometry = vec![crate::testing::geometry(64, &[("kv", 4096)], 3 * 4096)];
    let layout = Layout::new(&geometry).unwrap();
    let active = 32;
    assert_eq!(
        intake_slots(&layout, active),
        2 * active * 3 + 2 * slab::WINDOW_PAGES
    );
    let options = PrefixCacheOffload {
        memory_bytes: Some(17 * 4096 + 1),
        pool: Some(pool("unix:/run/selene.sock")),
        ..Default::default()
    };
    let Preparation::Store(prepared) = super::prepare_with_inputs(
        &options,
        [0; 32],
        geometry,
        vec![(0, 0)],
        inputs(active),
        &crate::testing::IdentityRegistrar,
    )
    .unwrap() else {
        panic!("no store")
    };
    assert!(prepared.intake_warning().is_none());
    let pool = prepared.memory_pool.unwrap();
    assert_eq!(pool.intake_slots(), intake_slots(&layout, active));
    assert_eq!(pool.slot_count(), 17 + pool.intake_slots());
}

#[test]
fn intake_shrinks_to_two_full_windows_without_refusing() {
    let layout = Layout::new(&[crate::testing::geometry(64, &[("kv", 4096)], 3 * 4096)]).unwrap();
    let memory = u64::MAX - intake_slots(&layout, 1) as u64 * layout.page_record_bytes;
    let options = PrefixCacheOffload {
        memory_bytes: Some(memory),
        ..Default::default()
    };
    let (_, host) = options
        .headroom_with_inputs(1, &layout, inputs(64), None)
        .unwrap();
    assert_eq!(
        host - memory,
        intake_slots(&layout, 1) as u64 * layout.page_record_bytes
    );
}

#[test]
fn integrated_disk_only_bounds_intake_without_warning_and_preserves_other_modes() {
    let geometry = vec![crate::testing::geometry(64, &[("kv", 4096)], 5 * 4096)];
    let layout = Layout::new(&geometry).unwrap();
    let active = 32;
    let bounded = intake_slots(&layout, 1);
    let full = intake_slots(&layout, active);
    assert!(bounded < full);

    let directory = test_directory();
    let disk_only = PrefixCacheOffload {
        memory_bytes: Some(0),
        disk: Some(directory.clone()),
        disk_limit: Some(1 << 30),
        ..Default::default()
    };
    assert!(disk_only.disk_only());
    let integrated_inputs = PrepareInputs {
        integrated: true,
        ..inputs(active)
    };
    let (device, host) = disk_only
        .headroom_with_inputs(1, &layout, integrated_inputs, None)
        .unwrap();
    let bounded_bytes = bounded as u64 * layout.page_record_bytes;
    assert_eq!(host, bounded_bytes);
    assert_eq!(
        device,
        bounded_bytes.div_ceil(512) + STORE_DEVICE_MARGIN_BYTES
    );
    assert_eq!(
        disk_only
            .headroom_for_test_budget(1, &layout, integrated_inputs, None, 0)
            .unwrap()
            .1,
        bounded_bytes
    );

    // Integrated disk-only prepares at its two-window target (`sized_concurrency == 1`)
    // without a shrink warning, whereas discrete disk-only shrunk to the same floor warns
    // for the full `active` concurrency.
    let Preparation::Store(prepared) = super::prepare_with_inputs(
        &disk_only,
        [0; 32],
        geometry.clone(),
        vec![(0, 0)],
        integrated_inputs,
        &crate::testing::IdentityRegistrar,
    )
    .unwrap() else {
        panic!("no disk store")
    };
    assert_eq!(prepared.required_intake, bounded);
    assert_eq!(prepared.sized_concurrency, 1);
    assert!(prepared.intake_warning().is_none());
    let integrated_pool = prepared.memory_pool.as_ref().unwrap();
    assert_eq!(
        (integrated_pool.slot_count(), integrated_pool.intake_slots()),
        (bounded, bounded)
    );
    drop(prepared);

    let (_, shrunk_host) = disk_only
        .headroom_for_test_budget(1, &layout, inputs(active), None, 0)
        .unwrap();
    assert_eq!(shrunk_host, bounded_bytes);
    let Preparation::Store(mut discrete) = super::prepare_with_inputs(
        &disk_only,
        [0; 32],
        geometry,
        vec![(0, 0)],
        inputs(active),
        &crate::testing::IdentityRegistrar,
    )
    .unwrap() else {
        panic!("no discrete disk store")
    };
    assert_eq!(discrete.required_intake, full);
    assert_eq!(discrete.sized_concurrency, active);
    let shrunk_slots = (shrunk_host / layout.page_record_bytes) as usize;
    discrete.memory_pool = Some(
        crate::slab::SlabPool::new(layout.page_record_bytes, shrunk_slots, shrunk_slots).unwrap(),
    );
    let warning = discrete.intake_warning().unwrap();
    assert!(
        warning.contains(&format!("max concurrency {active}")),
        "{warning}"
    );
    drop(discrete);

    // Discrete GPUs, nonzero warm memory, disabled disk_limit = 0, and pool-only capture
    // keep concurrency-scaled or capture-chain sizing.
    let (_, discrete_host) = disk_only
        .headroom_with_inputs(1, &layout, inputs(active), None)
        .unwrap();
    assert_eq!(discrete_host, full as u64 * layout.page_record_bytes);

    let warm_bytes = 16 * layout.page_record_bytes;
    let warm_and_disk = PrefixCacheOffload {
        memory_bytes: Some(warm_bytes),
        disk: Some(directory.clone()),
        disk_limit: Some(1 << 30),
        ..Default::default()
    };
    assert!(!warm_and_disk.disk_only());
    let (_, warm_host) = warm_and_disk
        .headroom_with_inputs(1, &layout, integrated_inputs, None)
        .unwrap();
    assert_eq!(
        warm_host,
        warm_bytes + full as u64 * layout.page_record_bytes
    );

    let disabled_disk = PrefixCacheOffload {
        disk_limit: Some(0),
        ..disk_only
    };
    assert!(!disabled_disk.disk_only());
    assert_eq!(
        disabled_disk
            .headroom_with_inputs(1, &layout, integrated_inputs, None)
            .unwrap(),
        (0, 0)
    );

    let pool_only = PrefixCacheOffload {
        memory_bytes: Some(0),
        pool: Some(pool("unix:/run/selene.sock")),
        ..Default::default()
    };
    let pages = 32u64;
    let (_, pool_host) = pool_only
        .headroom_with_inputs(1, &layout, integrated_inputs, Some(pages))
        .unwrap();
    let expected_capture =
        full.max(pages as usize + layout.row_slots()) as u64 * layout.page_record_bytes;
    assert_eq!(pool_host, expected_capture);

    std::fs::remove_dir_all(directory).unwrap();
}
