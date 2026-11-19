/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use flare_selene::{CacheGeometry, Result, pool::{Budget, PoolBuilder, PoolPeers, PreparedPool}};
use flare_selene::{
    pool::{PoolOptions, PoolStatus},
    testing::{TestHistory, TrackingRegistrar as Registrar},
};
use selene_manager::testing::{Manager, wait};
use std::{sync::Arc, thread, time::Duration};
fn unlimited() -> Budget {
    Budget {
        host: Ok(u64::MAX),
        devices: vec![(u64::MAX, false)],
    }
}
fn start_with_registrar(
    options: &PoolOptions,
    identity: [u8; 32],
    geometry: &[CacheGeometry],
    gpu_uuids: Vec<[u8; 16]>,
    registration_context: usize,
    budget: Budget,
    registrar: Arc<Registrar>,
    peers: Option<PoolPeers>,
    disk: bool,
) -> Result<(PreparedPool, Vec<u64>)> {
    PoolBuilder {
        options,
        identity,
        geometry,
        gpu_uuids,
        registration_context,
        budget,
        peers,
        disk,
    }
    .start(registrar)
}
struct NoCopies;
impl flare_selene::offload::CopierFactory for NoCopies {
    fn mapped(
        &self,
        _: &[(usize, i32)],
        _: Vec<CacheGeometry>,
        _: &flare_selene::slab::SlabPool,
    ) -> Result<flare_selene::SharedCopier> {
        panic!("no copy engine should be constructed")
    }
    fn unmapped(
        &self,
        _: &[(usize, i32)],
        _: Vec<CacheGeometry>,
    ) -> Result<Box<dyn flare_selene::PageCopier>> {
        panic!("no copy engine should be constructed")
    }
}
fn start(
    manager: &Manager,
    registrar: Arc<Registrar>,
    budget: Budget,
) -> Result<(PreparedPool, Vec<u64>)> {
    let geometry = vec![flare_selene::testing::geometry(64, &[("kv", 4096)], 4096)];
    start_with_registrar(
        &PoolOptions { address: manager.address(), ca: None },
        [3; 32],
        &geometry,
        vec![[1; 16]],
        0,
        budget,
        registrar,
        None,
        false,
    )
}
#[test]
fn over_budget_startup_fails_before_registering_memory() {
    let mut manager = Manager::new();
    manager.run();
    for (budget, expected) in [
        (
            Budget {
                host: Ok(0),
                ..unlimited()
            },
            "host budget has 0 bytes left",
        ),
        (
            Budget {
                devices: vec![(1, false)],
                ..unlimited()
            },
            "budget has 1 bytes left",
        ),
    ] {
        let registrar = Arc::new(Registrar::default());
        let reason = start(&manager, registrar.clone(), budget)
            .err()
            .expect("an unapproved pool must fail startup")
            .to_string();
        for text in [
            "arena 8388608 bytes",
            "staging 8388608 bytes",
            "needs 16777216 host bytes",
            expected,
            "free host/device memory",
            "remove --prefix-cache-pool",
        ] {
            assert!(reason.contains(text), "{reason}");
        }
        assert!(registrar.calls.lock().unwrap().is_empty());
    }
}

#[test]
fn an_absent_manager_fails_startup_without_registering_memory() {
    let manager = Manager::new();
    let registrar = Arc::new(Registrar::default());
    let geometry = vec![flare_selene::testing::geometry(64, &[("kv", 4096)], 4096)];
    let options = PoolOptions {
        address: manager.address(),
        ca: None,
    };
    let reason = PoolBuilder {
        options: &options,
        identity: [3; 32],
        geometry: &geometry,
        gpu_uuids: vec![[1; 16]],
        registration_context: 0,
        budget: unlimited(),
        peers: None,
        disk: false,
    }
    .start_with_wait(registrar.clone(), Duration::from_millis(20))
    .err()
    .expect("a configured pool must attach at startup")
    .to_string();
    for text in [
        "did not answer within 20ms",
        "No such file or directory",
        "remove --prefix-cache-pool",
    ] {
        assert!(reason.contains(text), "{reason}");
    }
    assert!(registrar.calls.lock().unwrap().is_empty());
}

#[test]
fn startup_wait_precedes_readiness_and_restart_never_registers_again() {
    let mut manager = Manager::new();
    let registrar = Arc::new(Registrar::default());
    let options = PoolOptions { address: manager.address(), ca: None };
    let registered = registrar.clone();
    let (ready, readiness) = std::sync::mpsc::channel();
    let startup = thread::spawn(move || {
        let geometry = vec![flare_selene::testing::geometry(64, &[("kv", 4096)], 4096)];
        let result = start_with_registrar(
            &options,
            [3; 32],
            &geometry,
            vec![[1; 16]],
            0,
            unlimited(),
            registered,
            None,
            false,
        )
        .unwrap();
        ready.send(()).unwrap();
        result
    });
    assert!(registrar.calls.lock().unwrap().is_empty());
    assert!(readiness.recv_timeout(Duration::from_millis(20)).is_err());
    manager.run();
    readiness.recv_timeout(Duration::from_secs(5)).unwrap();
    let (mut prepared, charges) = startup.join().unwrap();
    assert_eq!(charges, [(16 << 20) / 512]);
    assert_eq!(registrar.calls.lock().unwrap().len(), 2);
    let changed = vec![flare_selene::testing::geometry(128, &[("kv", 4096)], 4096)];
    assert!(
        prepared
            .start_copies(&[], changed, &NoCopies)
            .unwrap_err()
            .to_string()
            .contains("layouts disagree")
    );
    wait("startup", || prepared.client.as_ref().unwrap().status() == PoolStatus::Ready);
    manager.stop();
    wait("the old session to disconnect", || prepared.client.as_ref().unwrap().status() != PoolStatus::Ready);
    manager.run();
    wait("startup", || {
        prepared.client.as_ref().unwrap().status() == PoolStatus::Ready
    });
    assert_eq!(registrar.calls.lock().unwrap().len(), 2);
    manager.stop();
    manager.config.staging *= 2;
    manager.run();
    wait("startup", || {
        matches!(
            prepared.client.as_ref().unwrap().status(),
            PoolStatus::Disabled(_)
        )
    });
    assert_eq!(registrar.calls.lock().unwrap().len(), 2);
    assert_eq!(*registrar.drops.lock().unwrap(), 0);
    prepared.client.as_mut().unwrap().close().unwrap();
    assert_eq!(*registrar.drops.lock().unwrap(), 2);
    let mut store = prepared.attach(None).unwrap();
    let mut history = TestHistory::new([3; 32]);
    history.append(&[7; 128], &[]).unwrap();
    assert!(store.lookup(&history, 0, u64::MAX).is_none());
    store.poll(&mut Vec::new()).unwrap();
    store.close().unwrap();
}
