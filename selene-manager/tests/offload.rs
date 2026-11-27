/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use flare_selene::{CacheGeometry, Result, offload::*, pool::PoolOptions, slab};

fn pool(address: &str) -> PoolOptions {
    PoolOptions { address: address.into(), ca: None }
}

#[test]
fn startup_fails_and_releases_registration_when_a_rank_rejects_the_pool() {
    use flare_selene::pool::{Budget, PoolBuilder};
    use flare_selene::testing::{TrackingRegistrar, geometry};
    use std::sync::Arc;

    let mut manager = selene_manager::testing::Manager::new();
    manager.run();
    for local_approved in [true, false] {
        let options = pool(&manager.address());
        let geometry = vec![geometry(64, &[("kv", 4096)], 4096)];
        let registrar = Arc::new(TrackingRegistrar::default());
        let mut startup = Startup::default();
        let mut agreed = false;
        let reason = startup
            .start_pool(
                PoolBuilder {
                    options: &options,
                    identity: [3; 32],
                    geometry: &geometry,
                    gpu_uuids: vec![[1; 16]],
                    registration_context: 0,
                    budget: Budget {
                        host: Ok(if local_approved { u64::MAX } else { 0 }),
                        devices: vec![(u64::MAX, false), (u64::MAX, true)],
                    },
                    peers: None,
                    disk: false,
                },
                registrar.clone(),
                |approved| {
                    agreed = true;
                    assert_eq!(approved, local_approved);
                    assert_eq!(
                        registrar.calls.lock().unwrap().len(),
                        if local_approved { 2 } else { 0 }
                    );
                    assert_eq!(*registrar.drops.lock().unwrap(), 0);
                    Ok(false)
                },
            )
            .err()
            .expect("a peer rejection must fail startup")
            .to_string();
        assert!(agreed);
        assert!(
            reason.contains(if local_approved {
                "a rank could not attach"
            } else {
                "host budget has 0 bytes left"
            }),
            "{reason}"
        );
        assert!(reason.contains("remove --prefix-cache-pool"), "{reason}");
        assert_eq!(
            *registrar.drops.lock().unwrap(),
            if local_approved { 2 } else { 0 }
        );
    }
}

#[test]
fn startup_keeps_registration_through_attachment_and_store_drain() {
    use flare_selene::pool::{Budget, PoolBuilder};
    use flare_selene::testing::{StoppedCopier, TrackingRegistrar, geometry};
    use std::sync::Arc;
    struct Copiers;
    impl CopierFactory for Copiers {
        fn mapped(
            &self,
            _: &[(usize, i32)],
            geometry: Vec<CacheGeometry>,
            _: &slab::SlabPool,
        ) -> Result<flare_selene::SharedCopier> {
            StoppedCopier::shared(&geometry)
        }
        fn unmapped(
            &self,
            _: &[(usize, i32)],
            geometry: Vec<CacheGeometry>,
        ) -> Result<Box<dyn flare_selene::PageCopier>> {
            Ok(Box::new(StoppedCopier::new(&geometry)?))
        }
    }
    let mut manager = selene_manager::testing::Manager::new();
    manager.run();
    let options = PrefixCacheOffload {
        pool: Some(pool(&manager.address())),
        ..Default::default()
    };
    let geometry = vec![geometry(64, &[("kv", 4096)], 4096)];
    let registrar = Arc::new(TrackingRegistrar::default());
    let mut startup = Startup::default();
    assert!(startup.binding(0).is_none());
    let (_, charges) = startup
        .start_pool(
            PoolBuilder {
                options: options.pool.as_ref().unwrap(),
                identity: [3; 32],
                geometry: &geometry,
                gpu_uuids: vec![[1; 16]],
                registration_context: 0,
                budget: Budget {
                    host: Ok(u64::MAX),
                    devices: vec![(u64::MAX, false)],
                },
                peers: None,
                disk: false,
            },
            registrar.clone(),
            |approved| {
                assert!(approved);
                Ok(true)
            },
        )
        .unwrap();
    assert!(charges[0] > 0);
    let prepared = prepare_with_inputs(
        &options,
        [3; 32],
        geometry,
        vec![(0, 0)],
        PrepareInputs {
            max_active: 1,
            input_staging: 0,
            shard: None,
            integrated: false,
        },
        &flare_selene::testing::IdentityRegistrar,
    )
    .unwrap();
    startup.attach(prepared, &Copiers).unwrap();
    let binding = startup.binding(0).unwrap();
    assert_eq!(*registrar.drops.lock().unwrap(), 0);
    let mut store = open(binding.take().unwrap(), &Copiers).unwrap();
    startup.reset();
    assert_eq!(*registrar.drops.lock().unwrap(), 0);
    store.close().unwrap();
    assert_eq!(*registrar.drops.lock().unwrap(), 2);
}
