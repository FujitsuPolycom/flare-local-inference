/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ranks of one model on separate hosts: rendezvous at rank zero and connect their RoCE transport.

pub use flare_helios::execution::roce_fabric::Shape;
use std::sync::Arc;

use flare_helios::{
    execution::{CollectiveHealth, RankStream},
    runtime_core::{ClusterRank, NativeModel},
    startup::StartupProgress,
};

type Failure = Box<dyn std::error::Error>;

const NO_TRANSPORT: &str = "this flare build has no RoCE transport for multi-host cores; on Linux x86_64, rebuild with --features roce";

/// Reject unsupported serving before loading resources or sizing caches.
pub fn require_transport() -> Result<(), Failure> {
    if cfg!(all(
        target_os = "linux",
        any(target_arch = "aarch64", feature = "roce")
    )) {
        Ok(())
    } else {
        Err(NO_TRANSPORT.into())
    }
}

#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
use flare_helios::execution::roce::RelayMode;
#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
pub use flare_helios::execution::roce::RoceTransport;

#[cfg(not(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce"))))]
pub struct RoceTransport;

/// A rank's connection to the rest of its cluster; the transport must outlive every replay.
pub struct Joined<S> {
    pub streams: S,
    pub transport: Arc<RoceTransport>,
    /// A peer's host prefix record service, which rank zero drives until it closes its store.
    pub records: Option<flare_helios::offload::shard::Service>,
}

/// The check every completed frame passes: a failed transport poisons the rank.
///
/// A failure here is terminal, not recoverable in place. The proxy refuses `roce_start` on a
/// context that failed or was stopped inside an operation, and in-process quiesce/reset/rejoin is
/// not implemented: a replacement would have to exchange records with peers that already hold the
/// failed one. A rank that reports this error must stop admitting replays, keep its package and
/// CUDA owners live until the proxy has destroyed every queue pair, and leave the process. The
/// group recovers by restarting every rank, which reloads the package — restoring the pinned region
/// and the device counters zeroed — and starts a fresh transport that publishes and exchanges its
/// own records. Serving never calls `start` on the same context again.
pub fn health(transport: &Arc<RoceTransport>) -> CollectiveHealth {
    #[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
    {
        let transport = transport.clone();
        CollectiveHealth(Arc::new(move || transport.health()))
    }
    #[cfg(not(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce"))))]
    {
        let _ = transport;
        unreachable!("no transport without RoCE support")
    }
}

/// Open this rank's transport over the fabric every rank discovered, from their endpoint lists in
/// rank order.
#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn open(
    owner: &NativeModel,
    fabric: &flare_helios::execution::roce_fabric::Fabric,
    endpoints: &[Vec<u8>],
    shape: Shape,
    relay: RelayMode,
) -> Result<RoceTransport, Failure> {
    use flare_helios::execution::roce_fabric;

    let spec = owner
        .roce(0)
        .ok_or("multi-host core rank has no RoCE transport")?;
    let endpoints = endpoints
        .iter()
        .map(|bytes| roce_fabric::decode(bytes))
        .collect::<flare_helios::Result<Vec<_>>>()?;
    let topology = fabric.topology(&endpoints, spec.rank, shape)?;
    // SAFETY: The core restored the region at its recorded address; owner outlives the
    // transport, and no graph runs before every rank connected.
    Ok(unsafe { RoceTransport::open(spec, &topology, relay)? })
}

#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn connected(transport: &RoceTransport) -> String {
    // A mesh writes every peer directly, so its line keeps its text; a ring or chain names how
    // another rank's all-gather chunk reaches this one.
    let relay = match transport.relay_mode() {
        RelayMode::Direct => "",
        RelayMode::Mailbox => ", all-gather chunks relay through the ring mailbox",
        RelayMode::Partner => ", all-gather chunks relay through both neighbors",
    };
    format!(
        "{} rails per {} over {}{relay}",
        transport.topology().rails,
        match transport.topology().shape {
            Shape::Mesh => "peer",
            Shape::Ring => "neighbor (experimental CPU relay ring)",
            Shape::Chain => "neighbor (experimental CPU relay chain)",
        },
        transport.topology().hcas.join(",")
    )
}

/// The transport shape and relay mode a rank was started with, as the ranks exchange it before
/// planning routes.
#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn shape_record(shape: Shape, relay: RelayMode) -> Vec<u8> {
    // One word per rank, so a disagreement is named by the record instead of surfacing as a
    // collective that hangs: "ring-mailbox" against "ring-partner" is a rank that joined a ring of
    // another size, where the topology gives it another relay.
    format!("{}-{}", shape.name(), relay.name()).into_bytes()
}

/// Every rank must select the same shape and relay mode: a chain rank plans no route to the far end
/// that a ring rank would wait for, and a rank that forwards all-gather chunks expects its peers to
/// wait for a forwarded chunk.
#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn agree_shape(records: &[Vec<u8>], local: &[u8]) -> Result<(), Failure> {
    if let Some(rank) = records.iter().position(|record| record != local) {
        return Err(format!(
            "rank {rank} selected RoCE transport {} where this rank selected {}",
            String::from_utf8_lossy(&records[rank]),
            String::from_utf8_lossy(local)
        )
        .into());
    }
    Ok(())
}

/// A rank's admitted connections to the rest of its cluster, made before it loads.
pub enum Membership {
    /// Rank zero's connection to each peer, ranks 1 onward.
    Zero(Vec<RankStream>),
    /// A peer's connection to rank zero.
    Peer(RankStream),
}

impl flare_helios::runtime_core::CacheAgreement for Membership {
    fn caches(&mut self, local: (u64, u64)) -> flare_helios::Result<(u64, u64)> {
        use flare_helios::execution::rendezvous;
        match self {
            Membership::Zero(peers) => rendezvous::agree_capacity(peers, local),
            Membership::Peer(stream) => rendezvous::propose_capacity(stream, local),
        }
    }

    fn pool(
        &mut self,
        offer: flare_helios::offload::shard::PoolOffer,
    ) -> flare_helios::Result<Option<flare_helios::offload::shard::PoolPeers>> {
        use flare_helios::offload::shard;
        match self {
            Membership::Zero(peers) => Ok(shard::pool_rank_zero(offer, peers)?),
            Membership::Peer(stream) => Ok(shard::pool_peer(offer, stream)?),
        }
    }

    fn pool_approved(&mut self, approved: bool, world: usize) -> flare_helios::Result<bool> {
        use flare_helios::execution::rendezvous;
        let records = match self {
            Membership::Zero(peers) => rendezvous::share_records(peers, vec![approved as u8])?,
            Membership::Peer(stream) => {
                rendezvous::exchange_record(stream, &[approved as u8], world)?
            }
        };
        Ok(records.iter().all(|record| record.as_slice() == [1]))
    }
}

/// Rank zero: exchange transport records with every admitted peer and wait until every rank's
/// transport is connected.
pub fn rank_zero(
    owner: &NativeModel,
    streams: Vec<RankStream>,
    progress: &StartupProgress,
    shape: Shape,
) -> Result<Joined<Vec<RankStream>>, Failure> {
    #[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
    {
        use flare_helios::execution::{rendezvous, roce, roce_fabric};
        let mut streams = streams;

        let phase = progress.phase("Connect RoCE transport", None);
        // Every rank learns every rank's devices and the shape, then plans the same routes from them.
        let fabric = roce_fabric::discover()?;
        // The relay mode is part of what the ranks agree on: a rank that forwards another rank's
        // all-gather chunk expects the same plan on every peer.
        let relay = roce::relay_mode(shape, streams.len() + 1);
        let local = shape_record(shape, relay);
        let shapes = rendezvous::share_records(&mut streams, local.clone())?;
        agree_shape(&shapes, &local)?;
        let endpoints =
            rendezvous::share_records(&mut streams, roce_fabric::encode(&fabric.endpoints))?;
        let mut transport = open(owner, &fabric, &endpoints, shape, relay)?;
        let records = rendezvous::share_records(&mut streams, transport.local_record()?)?;
        transport.connect(&records)?;
        rendezvous::await_connected(&mut streams)?;
        phase.finish(connected(&transport));
        let phase = progress.phase("Join host prefix stores", None);
        match flare_helios::offload::shard::join_rank_zero(owner.offload(), &mut streams)? {
            Some(summary) => phase.finish(summary),
            None => phase.finish("none"),
        }
        Ok(Joined {
            streams,
            transport: Arc::new(transport),
            records: None,
        })
    }
    #[cfg(not(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce"))))]
    {
        let _ = (owner, streams, progress, shape);
        Err(NO_TRANSPORT.into())
    }
}

/// A peer rank: exchange transport records with rank zero over its admitted stream, and connect.
pub fn peer(
    owner: &NativeModel,
    cluster: &ClusterRank,
    stream: RankStream,
    progress: &StartupProgress,
    shape: Shape,
) -> Result<Joined<RankStream>, Failure> {
    #[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
    {
        use flare_helios::execution::{rendezvous, roce, roce_fabric};
        let mut stream = stream;

        let phase = progress.phase("Connect RoCE transport", None);
        let fabric = roce_fabric::discover()?;
        let relay = roce::relay_mode(shape, cluster.world);
        let local = shape_record(shape, relay);
        let shapes = rendezvous::exchange_record(&mut stream, &local, cluster.world)?;
        agree_shape(&shapes, &local)?;
        let endpoints = rendezvous::exchange_record(
            &mut stream,
            &roce_fabric::encode(&fabric.endpoints),
            cluster.world,
        )?;
        let mut transport = open(owner, &fabric, &endpoints, shape, relay)?;
        let records =
            rendezvous::exchange_record(&mut stream, &transport.local_record()?, cluster.world)?;
        transport.connect(&records)?;
        rendezvous::report_connected(&mut stream)?;
        phase.finish(connected(&transport));
        let phase = progress.phase("Join host prefix stores", None);
        let records = flare_helios::offload::shard::join_peer(
            owner.offload(),
            &mut stream,
            cluster.rank,
            cluster.world,
        )?;
        phase.finish(match &records {
            Some(service) => service.summary.clone(),
            None => "none".to_owned(),
        });
        Ok(Joined {
            streams: stream,
            transport: Arc::new(transport),
            records,
        })
    }
    #[cfg(not(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce"))))]
    {
        let _ = (owner, cluster, stream, progress, shape);
        Err(NO_TRANSPORT.into())
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;
    use flare_helios::runtime_core::CacheAgreement;
    use std::net::{TcpListener, TcpStream};

    #[test]
    fn pool_approval_is_unanimous_and_waits_for_every_rank() {
        for peer_approved in [true, false] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            let zero = listener.accept().unwrap().0;
            let (ready, started) = std::sync::mpsc::channel();
            let zero = std::thread::spawn(move || {
                let approved = Membership::Zero(vec![RankStream::Tcp(zero)])
                    .pool_approved(true, 2)
                    .unwrap();
                ready.send(approved).unwrap();
                approved
            });
            assert!(
                started
                    .recv_timeout(std::time::Duration::from_millis(50))
                    .is_err()
            );
            let approved = Membership::Peer(RankStream::Tcp(peer))
                .pool_approved(peer_approved, 2)
                .unwrap();
            assert_eq!(approved, peer_approved);
            assert_eq!(started.recv().unwrap(), peer_approved);
            assert_eq!(zero.join().unwrap(), peer_approved);
        }
    }
}

#[cfg(all(
    test,
    target_os = "linux",
    any(target_arch = "aarch64", feature = "roce")
))]
mod shape_tests {
    use super::*;

    #[test]
    fn the_shape_record_names_the_shape_and_its_relay_and_a_mismatch_is_refused() {
        // One record per rank, so a peer's transport is named rather than inferred.
        let record = |shape, relay| String::from_utf8(shape_record(shape, relay)).unwrap();
        assert_eq!(record(Shape::Mesh, RelayMode::Direct), "mesh-direct");
        assert_eq!(record(Shape::Ring, RelayMode::Mailbox), "ring-mailbox");
        assert_eq!(record(Shape::Ring, RelayMode::Partner), "ring-partner");
        assert_eq!(record(Shape::Chain, RelayMode::Mailbox), "chain-mailbox");
        // Ranks that agree connect; a rank whose shape or group size gives it another relay is
        // refused at the rendezvous instead of leaving a collective to wait for a chunk nobody
        // forwards.
        let partner = shape_record(Shape::Ring, RelayMode::Partner);
        let agreed = [partner.clone(), partner.clone()];
        agree_shape(&agreed, &partner).unwrap();
        let error = agree_shape(&[partner.clone(), b"ring-mailbox".to_vec()], &partner)
            .expect_err("a relay disagreement must be refused")
            .to_string();
        assert!(
            error.contains("rank 1 selected RoCE transport ring-mailbox"),
            "{error}"
        );
        assert!(error.contains("this rank selected ring-partner"), "{error}");
    }
}

#[cfg(all(
    test,
    not(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))
))]
mod transport_tests {
    use super::*;

    #[test]
    fn unsupported_build_rejects_multi_host_serving() {
        assert!(
            require_transport()
                .unwrap_err()
                .to_string()
                .contains("--features roce")
        );
    }
}
