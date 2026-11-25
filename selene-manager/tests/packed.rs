/*
 * SPDX-License-Identifier: Apache-2.0
 */

mod support;

use flare_selene::shared::{codec::Plan, frame::ToInstance, packed, record::Header};
use selene_manager::proto::v2;
use support::{
    Host,
    channel::{self, Security},
    key,
    pool::{Ident, Pool, Session},
};
use std::sync::OnceLock;
use tokio::io::AsyncReadExt;

fn fixture_packets() -> &'static [(Vec<u8>, u32); 3] {
    static PACKETS: OnceLock<[(Vec<u8>, u32); 3]> = OnceLock::new();
    PACKETS.get_or_init(|| {
        let plan = Plan::new(4096, &[]).unwrap();
        std::array::from_fn(|i| {
            let raw: Vec<u8> = (0..4096)
                .map(|j| ((j / (8 + 2 * i) + i) % (2 + i)) as u8)
                .collect();
            let mut pkt = vec![0u8; plan.max_packet()];
            let (_, used) = plan
                .encode_capture(v2::PackedUnit::default(), &[1; 32], &raw, &mut pkt, None)
                .unwrap();
            pkt.truncate(used);
            (pkt, packed::crc32(&raw))
        })
    })
}

fn page0() -> &'static [u8] {
    &fixture_packets()[0].0
}

fn page1() -> &'static [u8] {
    &fixture_packets()[1].0
}

fn row() -> &'static [u8] {
    &fixture_packets()[2].0
}

fn payload(unit: &v2::PackedUnit) -> &'static [u8] {
    let packets = fixture_packets();
    packets
        .iter()
        .find(|(_, crc)| *crc == unit.original_crc)
        .map(|(pkt, _)| pkt.as_slice())
        .expect("unexpected unit")
}

fn offer() -> v2::PackedOffer {
    let packets = fixture_packets();
    let units = [(0, 0, packets[0].1), (0, 1, packets[1].1), (1, 0, packets[2].1)]
        .into_iter()
        .map(|(kind, logical_index, original_crc)| {
            let mut u = v2::PackedUnit {
                kind,
                logical_index,
                original_crc,
                representation: 1,
                profile: ident("fixture", 4096)
                    .codec
                    .unwrap()
                    .receive_profiles
                    .remove(0),
                original_length: 4096,
                ..Default::default()
            };
            let bytes = payload(&u);
            u.stored_crc = packed::crc32(bytes);
            u.stored_length = bytes.len() as u64;
            u
        })
        .collect();
    v2::PackedOffer {
        first_page: 0,
        units,
    }
}

fn ident(name: &str, row_bytes: u64) -> Ident {
    let mut ident = Ident::new(name);
    let mut g = flare_selene::testing::geometry(64, &[("kv", 4096)], row_bytes);
    for plane in g.planes.iter().chain([&g.row]) {
        g.codec_fields.push(flare_selene::geometry::CodecField {
            buffer: plane.name.clone(),
            offset: 0,
            length: plane.unit_bytes,
            encoding: 0,
        });
    }
    ident.codec = Some(flare_selene::codec_copy::reader_caps(g.codec_profile()));
    let geometry = ident.scope.geometry.as_mut().unwrap();
    geometry.ranks[0].row_bytes = row_bytes;
    geometry.row_record_bytes = row_bytes;
    ident
}

async fn open(pool: &mut Pool, name: &str, capable: bool) -> Session {
    let ident = if capable {
        ident(name, 4096)
    } else {
        Ident::new(name)
    };
    let mut session = pool.open(&ident).await;
    assert_eq!(session.opened.codec, ident.codec);
    session.snapshot(vec![]).await;
    session
}

async fn put_packed(
    writer: &mut Session,
    put_id: u64,
    tokens: u64,
    parent: Option<u64>,
    chain_pages: u32,
    tail_pages: u32,
    packed_offer: v2::PackedOffer,
    packed_payload: &[u8],
    row_bytes: u64,
) {
    writer
        .send(v2::client_message::M::PutBegin(v2::PutBegin {
            ancestor_keys: Vec::new(),
            put_id,
            key: Some(v2::PrefixKey {
                consumed_tokens: tokens,
                digest: key(tokens).digest.to_vec(),
            }),
            chain_pages: u64::from(chain_pages),
            capture_role: flare_selene::CaptureRole::Endpoint as i32,
            child_paths: 0,
            descendant_row: false,
            next_child: None,
            row: true,
            has_published_parent: parent.is_some(),
            published_parent: parent.map(|p| v2::PrefixKey {
                consumed_tokens: p,
                digest: key(p).digest.to_vec(),
            }),
            bucket: vec![4; 16],
            published_tail_pages: u64::from(tail_pages),
            mode: v2::CopyMode::SharedWindows as i32,
            packed: Some(packed_offer.clone()),
        }))
        .await;
    let accept = writer
        .take(|m| match m {
            v2::server_message::M::PutAccept(a) if a.put_id == put_id => Some(a.clone()),
            _ => None,
        })
        .await;
    assert!(accept.packed, "manager should accept packed put");
    let mut descriptor = support::descriptor();
    descriptor.ranks[0].row_bytes = row_bytes;
    let header = selene_manager::record::new_header(
        &descriptor,
        key(tokens),
        parent.map(key),
        [4; 16],
        tail_pages,
    )
    .unwrap()
    .with_capture_role(flare_selene::CaptureRole::Endpoint)
    .with_packed(packed_offer)
    .unwrap();
    let metadata = header.packed_metadata().unwrap();
    assert_eq!(accept.record_bytes, header.length(true));
    let mut filled = 0usize;
    while filled < packed_payload.len() {
        let grant = writer
            .take(|m| match m {
                v2::server_message::M::PutGrant(g) if g.put_id == put_id => {
                    Some(g.window.clone().unwrap())
                }
                _ => None,
            })
            .await;
        assert!(grant.packed.is_some());
        let len = grant.length as usize;
        writer
            .shm
            .as_ref()
            .unwrap()
            .write(&grant, &packed_payload[filled..filled + len]);
        filled += len;
        writer
            .send(v2::client_message::M::PutFilled(v2::PutFilled {
                put_id,
                index: grant.index,
                row_xxh3: 0,
                page_xxh3: Vec::new(),
                has_row: false,
                rank_sums: Vec::new(),
            }))
            .await;
    }
    writer
        .send(v2::client_message::M::PutCommit(v2::PutCommit {
            put_id,
            metadata_offset: 0,
            metadata,
        }))
        .await;
    let stored = writer
        .take(|m| match m {
            v2::server_message::M::Stored(s) if s.put_id == put_id => Some(s.bytes),
            _ => None,
        })
        .await;
    assert_eq!(stored, header.length(true));
    support::put::fence(writer, put_id).await;
}

async fn force_eviction(writer: &mut Session) {
    let mut descriptor = support::descriptor();
    descriptor.ranks[0].row_bytes = 4096;
    // 128 pages (512 KiB) + 4 KiB row = 516 KiB, which rounds up to a 1 MiB arena extent.
    let payload = vec![0x5Au8; 128 * 4096 + 4096];
    let outcome = support::put::put(
        writer,
        &descriptor,
        99,
        key(8192),
        None,
        support::store_client::PutOptions {
            role: flare_selene::CaptureRole::Endpoint,
            ..Default::default()
        },
        None,
        Some(&payload),
    )
    .await;
    assert!(matches!(outcome, support::put::Outcome::Stored { .. }));
    support::put::fence(writer, 99).await;
}

async fn read(reader: &mut Session, id: u64, tcp: bool) {
    let candidate = reader.manager_candidate(id, 128).await;
    let mut claim = reader.claim_for(id, &candidate, 64, true);
    if tcp {
        claim.mode = v2::CopyMode::DataChannel as i32
    }
    reader.claim(claim).await;
    let granted = reader.claim_granted(id).await;
    let selected = granted.records[0].packed.as_ref().unwrap();
    assert_eq!(selected.units, &offer().units[1..]);
    let mut conn = if tcp {
        Some(channel::dial(granted.channel.as_ref().unwrap(), &Security::Plain).await)
    } else {
        None
    };
    for expected in &selected.units {
        let (index, units, data) = if let Some(c) = &mut conn {
            let ToInstance::Window(window) = ToInstance::read(c, 64).await.unwrap() else {
                panic!("missing packed window")
            };
            let mut data = vec![0; window.shape.length as usize];
            c.read_exact(&mut data).await.unwrap();
            (window.shape.index, window.packed.unwrap().units, data)
        } else {
            let w = reader.window_ready(id).await.window.unwrap();
            let data = reader.shm.as_ref().unwrap().read(&w);
            packed::verify_window(
                std::slice::from_ref(expected),
                w.packed.as_ref().unwrap(),
                &data,
            )
            .unwrap();
            (w.index, w.packed.unwrap().units, data)
        };
        assert_eq!(units, std::slice::from_ref(expected));
        packed::verify_units(&units, &data).unwrap();
        assert_eq!(data, payload(expected));
        reader.released(id, index).await;
    }
    reader.copy_result(id, v2::CopyOutcome::CopyOk).await;
    assert_eq!(
        reader.claim_terminal(id).await.state,
        v2::ClaimState::Copied as i32
    );
    reader.fence(id, v2::FenceOutcome::Drained).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn canonical_packets_survive_shared_windows_tcp_drain_and_recovery() {
    let base = support::base("packed");
    let root = base.join("records");
    let mut config = selene_manager::store::Config::new(
        root.clone(),
        base.join("run/m.sock"),
        1 << 20,
        64 << 20,
    );
    config.name = support::instance_name("packed");
    let trace = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    config.faults.trace = Some(trace.clone());
    config.listen = Some(support::net::free_address());
    config.listen_data = Some(support::net::free_address());
    let mut host = Host::open_config(config.clone());
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut legacy = open(&mut pool, "legacy", false).await;
    let mut writer = open(&mut pool, "writer", true).await;
    put_packed(
        &mut writer,
        1,
        128,
        None,
        2,
        2,
        offer(),
        &[page0(), page1(), row()].concat(),
        4096,
    )
    .await;
    force_eviction(&mut writer).await;
    let mut rdma_ident = ident("rdma-reader", 4096);
    let rdma_profile = rdma_ident
        .codec
        .as_ref()
        .unwrap()
        .receive_profiles[0]
        .as_slice()
        .try_into()
        .unwrap();
    rdma_ident.codec = Some(flare_selene::codec_copy::reader_caps(Some(rdma_profile)));
    let mut rdma = pool.open(&rdma_ident).await;
    rdma.snapshot(vec![]).await;
    let rdma_hit = rdma
        .lookup_request(v2::LookupRequest {
            lookup_id: 8,
            boundaries: vec![v2::PrefixKey {
                consumed_tokens: 0,
                digest: vec![4; 16],
            }],
            before_tokens: 129,
            ..Default::default()
        })
        .await;
    assert_eq!(rdma_hit.candidates.len(), 1);
    drop(rdma);
    let mut reader = open(&mut pool, "reader", true).await;
    read(&mut reader, 1, false).await;
    let reply = legacy
        .lookup_request(v2::LookupRequest {
            lookup_id: 1,
            boundaries: vec![v2::PrefixKey {
                consumed_tokens: 0,
                digest: vec![4; 16],
            }],
            before_tokens: 129,
            ..Default::default()
        })
        .await;
    assert!(reply.candidates.is_empty());
    drop((legacy, writer, reader, pool));
    host.stop();
    let dir = root.join(flare_selene::record::directory_name(&[7; 32], &[3; 16]));
    let file = dir.join(flare_selene::record::file_name(&key(128)));
    let bytes = std::fs::read(&file).unwrap();
    let h = Header::decode(&bytes).unwrap();
    assert_eq!(bytes.len() as u64, h.length(true));
    assert_eq!(
        &bytes[h.page_segment as usize..],
        [page0(), page1(), row()].concat()
    );
    selene_manager::record::verify_record(&bytes, &h, true).unwrap();
    let mut second = Host::open_config(config);
    second.run();
    let mut pool = Pool::connect(&second.config.socket).await;
    let mut reader = open(&mut pool, "reopen", true).await;
    read(&mut reader, 2, true).await;
    drop((pool, reader));
    second.stop();
    let trace = trace.lock().unwrap();
    assert!(trace.contains(&b'w'), "no direct record write: {trace:?}");
    assert!(
        trace.iter().filter(|&&op| op == b'r').count() >= 3,
        "expected direct metadata, suffix and final row reads: {trace:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recovery_keeps_the_packed_directory_after_row_truncation() {
    let base = support::base("packed-cut");
    let root = base.join("records");
    let mut config = selene_manager::store::Config::new(
        root.clone(),
        base.join("run/m.sock"),
        1 << 20,
        64 << 20,
    );
    config.name = support::instance_name("packed-cut");
    let mut host = Host::open_config(config);
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut raw = open(&mut pool, "raw", false).await;
    assert!(matches!(
        raw.put_record(1, key(64), None, Default::default()).await,
        support::pool::PutResult::Stored(_)
    ));
    let mut writer = open(&mut pool, "packed", true).await;
    let mut child_offer = offer();
    child_offer.first_page = 1;
    for unit in &mut child_offer.units {
        if unit.kind == v2::packed_unit::Kind::Page as i32 {
            unit.logical_index += 1;
        }
    }
    put_packed(
        &mut writer,
        1,
        192,
        Some(64),
        3,
        2,
        child_offer,
        &[page0(), page1(), row()].concat(),
        4096,
    )
    .await;
    force_eviction(&mut writer).await;
    assert!(matches!(
        raw.put_record(2, key(256), Some(key(192)), Default::default())
            .await,
        support::pool::PutResult::Stored(_)
    ));
    drop((writer, raw, pool));
    host.stop();
    let dir = root.join(flare_selene::record::directory_name(&[7; 32], &[3; 16]));
    let path = dir.join(flare_selene::record::file_name(&key(192)));
    let bytes = std::fs::read(&path).unwrap();
    let h = Header::decode(&bytes).unwrap();
    let layout = h.packed.as_ref().unwrap();
    assert_eq!(layout.offer().first_page, 1);
    assert_eq!(h.length(true), bytes.len() as u64);
    assert_eq!(layout.row_bytes(), row().len() as u64);
    assert_eq!(
        &bytes[h.page_segment as usize..h.row_segment as usize],
        [page0(), page1()].concat()
    );
    assert_eq!(
        &bytes[..h.page_segment as usize],
        h.packed_metadata().unwrap()
    );
    let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(h.row_segment).unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        bytes[..h.row_segment as usize]
    );
    // A partial row also recovers as page-only while a child keeps the record reachable.
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(&row()[..37])
        .unwrap();
    let lock = flare_selene::record::DirectoryLock::acquire(&dir).unwrap();
    let records =
        flare_selene::record::scan(&lock, &[7; 32], &[3; 16], 64, &support::descriptor().ranks)
            .unwrap();
    assert_eq!(
        records
            .iter()
            .map(|r| (r.header.key, r.row))
            .collect::<Vec<_>>(),
        [
            (key(64), true),
            (key(192), false),
            (key(256), true),
            (key(8192), true)
        ]
    );
}

#[test]
fn directory_bounds_integrity_and_physical_window_limits() {
    let ranks = support::descriptor().ranks;
    let offer = offer();
    for mutate in [
        |o: &mut v2::PackedOffer| o.units[0].stored_length = u64::MAX,
        |o: &mut v2::PackedOffer| o.units[0].original_length = 17,
        |o: &mut v2::PackedOffer| o.units[0].replicas.push(0),
        |o: &mut v2::PackedOffer| o.units.swap(0, 2),
        |o: &mut v2::PackedOffer| o.units[0].profile.clear(),
        |o: &mut v2::PackedOffer| o.units[0].raw_xxh3 = 1,
    ] {
        let mut bad = offer.clone();
        mutate(&mut bad);
        assert!(packed::Layout::new(bad, &ranks, 0, 2).is_err());
    }
    let layout = packed::Layout::new(offer.clone(), &ranks, 0, 2).unwrap();
    let (p0, p1, r) = (page0().len() as u64, page1().len() as u64, row().len() as u64);
    let cap = p0.max(p1);
    assert!(cap < p0 + p1);
    assert_eq!(
        layout
            .windows(0, true, cap)
            .iter()
            .map(|w| w.length)
            .collect::<Vec<_>>(),
        [p0, p1, r]
    );
    assert_eq!(layout.windows(1, true, p0 + p1 + r)[0].offset, p0);
    let header =
        selene_manager::record::new_header(&support::descriptor(), key(128), None, [4; 16], 2)
            .unwrap()
            .with_packed(offer.clone())
            .unwrap();
    let metadata = header.packed_metadata().unwrap();
    assert!(Header::decode(&metadata[..metadata.len() - 1]).is_err());
    let mut corrupt = metadata.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    assert!(Header::decode(&corrupt).is_err());
    let mut huge = metadata;
    huge[181..185].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(Header::read_metadata(&*huge).is_err());
    let mut damaged = page1().to_vec();
    let mid = damaged.len() / 2;
    damaged[mid] ^= 1;
    assert!(packed::verify_units(&offer.units[1..2], &damaged).is_err());
    let mut literal = offer.units[0].clone();
    literal.representation = 0;
    literal.profile.clear();
    literal.stored_crc = 0;
    literal.original_crc = 0;
    literal.stored_length = 4096;
    let bytes = vec![17; 4096];
    literal.raw_xxh3 = flare_selene::record::checksum(&bytes);
    let mut mixed = offer;
    mixed.units[0] = literal;
    let layout = packed::Layout::new(mixed, &ranks, 0, 2).unwrap();
    assert!(packed::verify_units(layout.units(0, 1, false).unwrap(), &bytes).is_ok());
    assert!(packed::verify_units(layout.units(0, 1, false).unwrap(), &vec![18; 4096]).is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_row_larger_than_staging_uses_only_its_stored_length() {
    let raw = vec![0u8; 9 << 20];
    let plan = Plan::new(raw.len(), &[]).unwrap();
    let mut bytes = vec![0u8; plan.max_packet()];
    let (_, used) = plan
        .encode_capture(v2::PackedUnit::default(), &[1; 32], &raw, &mut bytes, None)
        .unwrap();
    bytes.truncate(used);
    let mut host = Host::with_storage(512 << 20, 128 << 20);
    host.config.staging = 8 << 20;
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let id = ident("large-row", 9 << 20);
    let row_profile = id.codec.as_ref().unwrap().receive_profiles[0].clone();
    let mut writer = pool.open(&id).await;
    writer.snapshot(vec![]).await;
    let large_offer = v2::PackedOffer {
        first_page: 0,
        units: vec![v2::PackedUnit {
            kind: 1,
            logical_index: 0,
            representation: 1,
            profile: row_profile,
            original_length: raw.len() as u64,
            stored_length: bytes.len() as u64,
            original_crc: packed::crc32(&raw),
            stored_crc: packed::crc32(&bytes),
            ..Default::default()
        }],
    };
    put_packed(&mut writer, 1, 17, None, 0, 0, large_offer, &bytes, 9 << 20).await;
    drop((writer, pool));
    host.restart();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut writer = pool.open(&id).await;
    writer.snapshot(vec![]).await;
    let candidate = writer.manager_candidate(4, 17).await;
    writer.claim(writer.claim_for(1, &candidate, 0, true)).await;
    let granted = writer.claim_granted(1).await;
    assert_eq!(granted.total_pages, 0);
    let w = writer.window_ready(1).await.window.unwrap();
    let data = writer.shm.as_ref().unwrap().read(&w);
    let expected = &granted.records[0].packed.as_ref().unwrap().units;
    packed::verify_window(expected, w.packed.as_ref().unwrap(), &data).unwrap();
    assert_eq!(data, bytes);
    writer.released(1, w.index).await;
    writer.copy_result(1, v2::CopyOutcome::CopyOk).await;
    writer.claim_terminal(1).await;
    writer.fence(1, v2::FenceOutcome::Drained).await;
    drop((writer, pool));
    host.stop();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn packed_put_stores_compressed_v4_directly_in_dram_and_flushes_verbatim_to_nvme() {
    let base = support::base("packed-put");
    let root = base.join("records");
    let mut config = selene_manager::store::Config::new(
        root.clone(),
        base.join("run/m.sock"),
        1 << 20,
        64 << 20,
    );
    config.name = support::instance_name("packed-put");
    let mut host = Host::open_config(config.clone());
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut writer = open(&mut pool, "writer", true).await;
    let packed_offer = offer();
    let packed_payload = [page0(), page1(), row()].concat();
    writer
        .send(v2::client_message::M::PutBegin(v2::PutBegin {
            ancestor_keys: Vec::new(),
            put_id: 1,
            key: Some(v2::PrefixKey {
                consumed_tokens: 128,
                digest: key(128).digest.to_vec(),
            }),
            chain_pages: 2,
            capture_role: flare_selene::CaptureRole::Endpoint as i32,
            child_paths: 0,
            descendant_row: false,
            next_child: None,
            row: true,
            has_published_parent: false,
            published_parent: None,
            bucket: vec![4; 16],
            published_tail_pages: 2,
            mode: v2::CopyMode::SharedWindows as i32,
            packed: Some(packed_offer.clone()),
        }))
        .await;
    let accept = writer
        .take(|m| match m {
            v2::server_message::M::PutAccept(a) if a.put_id == 1 => Some(a.clone()),
            _ => None,
        })
        .await;
    assert!(accept.packed, "manager should accept packed put");
    assert!(
        accept.record_bytes < 4096 + 3 * 4096,
        "packed record should occupy fewer arena bytes than raw: {}",
        accept.record_bytes
    );
    let mut descriptor = support::descriptor();
    descriptor.ranks[0].row_bytes = 4096;
    let header = selene_manager::record::new_header(
        &descriptor,
        key(128),
        None,
        [4; 16],
        2,
    )
    .unwrap()
    .with_capture_role(flare_selene::CaptureRole::Endpoint)
    .with_packed(packed_offer.clone())
    .unwrap();
    let metadata = header.packed_metadata().unwrap();
    assert_eq!(accept.record_bytes, header.length(true));
    let mut filled = 0usize;
    while filled < packed_payload.len() {
        let grant = writer
            .take(|m| match m {
                v2::server_message::M::PutGrant(g) if g.put_id == 1 => {
                    Some(g.window.clone().unwrap())
                }
                _ => None,
            })
            .await;
        assert!(grant.packed.is_some());
        let len = grant.length as usize;
        let slice = &packed_payload[filled..filled + len];
        writer.shm.as_ref().unwrap().write(&grant, slice);
        filled += len;
        writer
            .send(v2::client_message::M::PutFilled(v2::PutFilled {
                put_id: 1,
                index: grant.index,
                row_xxh3: 0,
                page_xxh3: Vec::new(),
                has_row: false,
                rank_sums: Vec::new(),
            }))
            .await;
    }
    writer
        .send(v2::client_message::M::PutCommit(v2::PutCommit {
            put_id: 1,
            metadata_offset: 0,
            metadata,
        }))
        .await;
    let stored = writer
        .take(|m| match m {
            v2::server_message::M::Stored(s) if s.put_id == 1 => Some(s.bytes),
            _ => None,
        })
        .await;
    assert_eq!(stored, header.length(true));
    support::put::fence(&mut writer, 1).await;

    // Claim directly from DRAM before NVMe eviction.
    let mut reader = open(&mut pool, "dram-reader", true).await;
    let candidate = reader.manager_candidate(1, 128).await;
    assert_eq!(candidate.tier, v2::Tier::Dram as i32);
    read(&mut reader, 1, false).await;

    // Evict to NVMe and verify the packed record flushes verbatim.
    force_eviction(&mut writer).await;
    drop((writer, reader, pool));
    host.stop();

    let dir = root.join(flare_selene::record::directory_name(&[7; 32], &[3; 16]));
    let file = dir.join(flare_selene::record::file_name(&key(128)));
    let bytes = std::fs::read(&file).unwrap();
    let on_disk = Header::decode(&bytes).unwrap();
    assert_eq!(bytes.len() as u64, on_disk.length(true));
    assert_eq!(&bytes[on_disk.page_segment as usize..], &packed_payload[..]);
    selene_manager::record::verify_record(&bytes, &on_disk, true).unwrap();

    // Restart manager and restore from NVMe.
    let mut second = Host::open_config(config);
    second.run();
    let mut pool = Pool::connect(&second.config.socket).await;
    let mut reader = open(&mut pool, "nvme-reader", true).await;
    read(&mut reader, 2, false).await;
    drop((reader, pool));
    second.stop();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mixed_cpu_and_gpu_packed_records_coexist_and_restore_by_dual_profile_reader() {
    let base = support::base("packed-mixed");
    let root = base.join("records");
    let mut config = selene_manager::store::Config::new(
        root.clone(),
        base.join("run/m.sock"),
        1 << 20,
        64 << 20,
    );
    config.name = support::instance_name("packed-mixed");
    let mut host = Host::open_config(config);
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;

    let mut g = flare_selene::testing::geometry(64, &[("kv", 4096)], 4096);
    for plane in g.planes.iter().chain([&g.row]) {
        g.codec_fields.push(flare_selene::geometry::CodecField {
            buffer: plane.name.clone(),
            offset: 0,
            length: plane.unit_bytes,
            encoding: 0,
        });
    }
    let cpu_profile = flare_selene::shared::codec_layout::profile(
        &g,
        flare_selene::shared::codec_layout::REVISION_CPU,
    )
    .unwrap();
    let gpu_profile = flare_selene::shared::codec_layout::profile(
        &g,
        flare_selene::shared::codec_layout::REVISION_GPU,
    )
    .unwrap();
    assert_ne!(cpu_profile, gpu_profile);

    let make_ident = |name: &str, profiles: &[[u8; 32]]| {
        let mut id = ident(name, 4096);
        id.codec = Some(flare_selene::codec_copy::reader_caps_many(profiles));
        id
    };
    let mut writer = pool.open(&make_ident("dual-writer", &[cpu_profile, gpu_profile])).await;
    writer.snapshot(vec![]).await;

    let mut descriptor = support::descriptor();
    descriptor.ranks[0].row_bytes = 4096;

    // Store parent (tokens=64, 1 page + row) using cpu_profile.
    let mut parent_offer = v2::PackedOffer {
        first_page: 0,
        units: vec![offer().units[0].clone(), offer().units[2].clone()],
    };
    for u in &mut parent_offer.units {
        u.profile = cpu_profile.to_vec();
    }
    let parent_payload = [page0(), row()].concat();
    writer
        .send(v2::client_message::M::PutBegin(v2::PutBegin {
            ancestor_keys: Vec::new(),
            put_id: 1,
            key: Some(v2::PrefixKey {
                consumed_tokens: 64,
                digest: key(64).digest.to_vec(),
            }),
            chain_pages: 1,
            capture_role: flare_selene::CaptureRole::Endpoint as i32,
            child_paths: 0,
            descendant_row: false,
            next_child: None,
            row: true,
            has_published_parent: false,
            published_parent: None,
            bucket: vec![4; 16],
            published_tail_pages: 1,
            mode: v2::CopyMode::SharedWindows as i32,
            packed: Some(parent_offer.clone()),
        }))
        .await;
    let accept1 = writer
        .take(|m| match m {
            v2::server_message::M::PutAccept(a) if a.put_id == 1 => Some(a.clone()),
            _ => None,
        })
        .await;
    assert!(accept1.packed);
    let h1 = selene_manager::record::new_header(&descriptor, key(64), None, [4; 16], 1)
        .unwrap()
        .with_capture_role(flare_selene::CaptureRole::Endpoint)
        .with_packed(parent_offer)
        .unwrap();
    let mut off = 0;
    while off < parent_payload.len() {
        let g = writer
            .take(|m| match m {
                v2::server_message::M::PutGrant(g) if g.put_id == 1 => g.window.clone(),
                _ => None,
            })
            .await;
        writer
            .shm
            .as_ref()
            .unwrap()
            .write(&g, &parent_payload[off..off + g.length as usize]);
        off += g.length as usize;
        writer
            .send(v2::client_message::M::PutFilled(v2::PutFilled {
                put_id: 1,
                index: g.index,
                ..Default::default()
            }))
            .await;
    }
    writer
        .send(v2::client_message::M::PutCommit(v2::PutCommit {
            put_id: 1,
            metadata_offset: 0,
            metadata: h1.packed_metadata().unwrap(),
        }))
        .await;
    writer
        .take(|m| match m {
            v2::server_message::M::Stored(s) if s.put_id == 1 => Some(()),
            _ => None,
        })
        .await;
    support::put::fence(&mut writer, 1).await;

    // Store child (tokens=128, page 1 + row) under parent using gpu_profile!
    let mut child_offer = v2::PackedOffer {
        first_page: 1,
        units: vec![offer().units[1].clone(), offer().units[2].clone()],
    };
    for u in &mut child_offer.units {
        u.profile = gpu_profile.to_vec();
    }
    let child_payload = [page1(), row()].concat();
    writer
        .send(v2::client_message::M::PutBegin(v2::PutBegin {
            ancestor_keys: Vec::new(),
            put_id: 2,
            key: Some(v2::PrefixKey {
                consumed_tokens: 128,
                digest: key(128).digest.to_vec(),
            }),
            chain_pages: 2,
            capture_role: flare_selene::CaptureRole::Endpoint as i32,
            child_paths: 0,
            descendant_row: false,
            next_child: None,
            row: true,
            has_published_parent: true,
            published_parent: Some(v2::PrefixKey {
                consumed_tokens: 64,
                digest: key(64).digest.to_vec(),
            }),
            bucket: vec![4; 16],
            published_tail_pages: 1,
            mode: v2::CopyMode::SharedWindows as i32,
            packed: Some(child_offer.clone()),
        }))
        .await;
    let accept2 = writer
        .take(|m| match m {
            v2::server_message::M::PutAccept(a) if a.put_id == 2 => Some(a.clone()),
            _ => None,
        })
        .await;
    assert!(accept2.packed);
    let h2 = selene_manager::record::new_header(&descriptor, key(128), Some(key(64)), [4; 16], 1)
        .unwrap()
        .with_capture_role(flare_selene::CaptureRole::Endpoint)
        .with_packed(child_offer)
        .unwrap();
    let mut off = 0;
    while off < child_payload.len() {
        let g = writer
            .take(|m| match m {
                v2::server_message::M::PutGrant(g) if g.put_id == 2 => g.window.clone(),
                _ => None,
            })
            .await;
        writer
            .shm
            .as_ref()
            .unwrap()
            .write(&g, &child_payload[off..off + g.length as usize]);
        off += g.length as usize;
        writer
            .send(v2::client_message::M::PutFilled(v2::PutFilled {
                put_id: 2,
                index: g.index,
                ..Default::default()
            }))
            .await;
    }
    writer
        .send(v2::client_message::M::PutCommit(v2::PutCommit {
            put_id: 2,
            metadata_offset: 0,
            metadata: h2.packed_metadata().unwrap(),
        }))
        .await;
    writer
        .take(|m| match m {
            v2::server_message::M::Stored(s) if s.put_id == 2 => Some(()),
            _ => None,
        })
        .await;
    support::put::fence(&mut writer, 2).await;

    // A CPU-only reader sees the CPU parent (64) but NOT the GPU child (128).
    let mut cpu_only = pool.open(&make_ident("cpu-only", &[cpu_profile])).await;
    cpu_only.snapshot(vec![]).await;
    let cpu_cand = cpu_only
        .lookup_request(v2::LookupRequest {
            lookup_id: 1,
            boundaries: vec![v2::PrefixKey {
                consumed_tokens: 0,
                digest: vec![4; 16],
            }],
            before_tokens: 129,
            ..Default::default()
        })
        .await;
    assert_eq!(cpu_cand.candidates.len(), 1);
    assert_eq!(
        cpu_cand.candidates[0].key.as_ref().unwrap().consumed_tokens,
        64
    );

    // A dual-profile reader sees AND claims the full mixed chain (CPU parent + GPU child)!
    let mut dual = pool.open(&make_ident("dual-reader", &[cpu_profile, gpu_profile])).await;
    dual.snapshot(vec![]).await;
    let cand = dual.manager_candidate(2, 128).await;
    dual.claim(dual.claim_for(1, &cand, 0, true)).await;
    let granted = dual.claim_granted(1).await;
    assert_eq!(granted.records.len(), 2);
    assert_eq!(
        granted.records[0].packed.as_ref().unwrap().units[0].profile,
        cpu_profile.to_vec()
    );
    assert_eq!(
        granted.records[1].packed.as_ref().unwrap().units[0].profile,
        gpu_profile.to_vec()
    );
    let expected_units: usize = granted
        .records
        .iter()
        .map(|r| r.packed.as_ref().unwrap().units.len())
        .sum();
    let mut seen_profiles = Vec::new();
    while seen_profiles.len() < expected_units {
        let w = dual.window_ready(1).await.window.unwrap();
        let units = w.packed.as_ref().unwrap().units.clone();
        let data = dual.shm.as_ref().unwrap().read(&w);
        packed::verify_units(&units, &data).unwrap();
        for u in units {
            seen_profiles.push(u.profile);
        }
        dual.released(1, w.index).await;
    }
    assert!(seen_profiles.contains(&cpu_profile.to_vec()));
    assert!(seen_profiles.contains(&gpu_profile.to_vec()));
    dual.copy_result(1, v2::CopyOutcome::CopyOk).await;
    assert_eq!(
        dual.claim_terminal(1).await.state,
        v2::ClaimState::Copied as i32
    );
    dual.fence(1, v2::FenceOutcome::Drained).await;
    drop((dual, cpu_only, writer, pool));
    host.stop();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn packed_peer_dram_export_relays_over_shm_and_tcp_and_isolates_raw_peers() {
    use flare_selene::shared::frame::ToManager;
    use tokio::io::AsyncWriteExt;

    let base = support::base("packed-peer-dram");
    let mut config = selene_manager::store::Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        1 << 20,
        64 << 20,
    );
    config.name = support::instance_name("packed-peer-dram");
    config.listen = Some(support::net::free_address());
    config.listen_data = Some(support::net::free_address());
    let mut host = Host::open_config(config);
    host.run();

    let mut pool = Pool::connect(&host.config.socket).await;
    let mut legacy = open(&mut pool, "legacy-peer", false).await;
    let mut source = open(&mut pool, "packed-source", true).await;
    let mut reader = open(&mut pool, "packed-reader", true).await;

    let full_offer = offer();
    let profile = full_offer.units[0].profile.clone();
    let stored_bytes: u64 = full_offer.units.iter().map(|u| u.stored_length).sum();
    let mut entry = support::pool::entry(128, None, v2::Tier::Dram);
    entry.bucket = vec![4; 16];
    entry.bytes = stored_bytes;
    entry.packed = true;
    entry.required_profiles = vec![profile];
    source
        .send(v2::client_message::M::Notices(v2::NoticeBatch {
            notices: vec![support::pool::saved(entry)],
        }))
        .await;

    // Wait until the codec-capable peer sees the packed DRAM holding.
    let mut first_lookup_id = 0u64;
    let cand = loop {
        first_lookup_id += 1;
        let reply = reader
            .lookup_request(v2::LookupRequest {
                lookup_id: first_lookup_id,
                boundaries: vec![v2::PrefixKey {
                    consumed_tokens: 0,
                    digest: vec![4; 16],
                }],
                before_tokens: 129,
                ..Default::default()
            })
            .await;
        if let Some(c) = reply.candidates.into_iter().next() {
            break c;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    };
    assert_eq!(cand.tier, v2::Tier::Dram as i32);
    assert!(
        !cand.source.as_ref().unwrap().instance_id.is_empty(),
        "candidate must come from the peer's DRAM, not the manager store"
    );

    // Raw-only peer must NOT see the packed peer DRAM holding.
    let legacy_reply = legacy
        .lookup_request(v2::LookupRequest {
            lookup_id: 2,
            boundaries: vec![v2::PrefixKey {
                consumed_tokens: 0,
                digest: vec![4; 16],
            }],
            before_tokens: 129,
            ..Default::default()
        })
        .await;
    assert!(legacy_reply.candidates.is_empty());

    // Claim the suffix [64, 128) + row from the peer's packed DRAM over both SHM and TCP.
    let suffix_offer = v2::PackedOffer {
        first_page: 1,
        units: full_offer.units[1..].to_vec(),
    };
    for (claim_id, tcp) in [(10u64, false), (11u64, true)] {
        let mut claim = reader.claim_for(claim_id, &cand, 64, true);
        if tcp {
            claim.mode = v2::CopyMode::DataChannel as i32;
        }
        reader.claim(claim).await;

        let req = source
            .take(|m| match m {
                v2::server_message::M::ExportRequest(r) => Some(r.clone()),
                _ => None,
            })
            .await;
        assert_eq!(req.tier, v2::Tier::Dram as i32);
        source
            .send(v2::client_message::M::PinAck(v2::PinAck {
                export_id: req.export_id,
                result: v2::PinResult::Pinned as i32,
                refusal: 0,
                records: vec![v2::Record {
                    key: req.key.clone(),
                    first_page: 1,
                    pages: 1,
                    row: true,
                    packed: Some(suffix_offer.clone()),
                }],
                total_pages: 1,
                row: true,
            }))
            .await;

        let granted = reader.claim_granted(claim_id).await;
        assert_eq!(granted.records[0].packed.as_ref(), Some(&suffix_offer));

        let fill = source
            .take(|m| match m {
                v2::server_message::M::ExportFill(f) if f.export_id == req.export_id => {
                    Some(f.clone())
                }
                _ => None,
            })
            .await;
        assert!(!fill.windows.is_empty());
        for w in &fill.windows {
            let mut buf = Vec::new();
            for u in &w.packed.as_ref().unwrap().units {
                buf.extend_from_slice(payload(u));
            }
            assert_eq!(buf.len() as u64, w.length);
            source.shm.as_ref().unwrap().write(w, &buf);
            source
                .send(v2::client_message::M::WindowFilled(v2::WindowFilled {
                    export_id: req.export_id,
                    index: w.index,
                    status: v2::FillStatus::FillOk as i32,
                    ..Default::default()
                }))
                .await;
        }

        let mut conn = if tcp {
            Some(channel::dial(granted.channel.as_ref().unwrap(), &Security::Plain).await)
        } else {
            None
        };
        let mut received_units = Vec::new();
        while received_units.len() < suffix_offer.units.len() {
            let (index, units, data) = if let Some(c) = &mut conn {
                let ToInstance::Window(window) = ToInstance::read(c, 64).await.unwrap() else {
                    panic!("missing packed window on TCP peer relay");
                };
                let mut data = vec![0; window.shape.length as usize];
                c.read_exact(&mut data).await.unwrap();
                (window.shape.index, window.packed.unwrap().units, data)
            } else {
                let w = reader.window_ready(claim_id).await.window.unwrap();
                let data = reader.shm.as_ref().unwrap().read(&w);
                (w.index, w.packed.unwrap().units, data)
            };
            packed::verify_units(&units, &data).unwrap();
            let mut off = 0usize;
            for u in units {
                let len = u.stored_length as usize;
                assert_eq!(&data[off..off + len], payload(&u));
                off += len;
                received_units.push(u);
            }
            if !tcp {
                reader.released(claim_id, index).await;
            }
        }
        assert_eq!(received_units, suffix_offer.units);
        reader.copy_result(claim_id, v2::CopyOutcome::CopyOk).await;
        assert_eq!(
            reader.claim_terminal(claim_id).await.state,
            v2::ClaimState::Copied as i32
        );
        reader.fence(claim_id, v2::FenceOutcome::Drained).await;
        source
            .send(v2::client_message::M::ExportEnd(v2::ExportEnd {
                export_id: req.export_id,
                status: v2::ExportEndStatus::ExportComplete as i32,
                fence: v2::FenceOutcome::Drained as i32,
            }))
            .await;
    }

    // Remote TCP source -> remote TCP requester packed peer-DRAM relay.
    drop((legacy, source, reader, pool));
    let tcp_addr = host.config.listen.unwrap();
    let tcp_ch = tonic::transport::Endpoint::from_shared(format!("http://{tcp_addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut remote_pool = Pool::over(tcp_ch, &host.config.socket).await;
    let remote_src_ident = ident("remote-packed-src", 4096);
    let mut remote_src = remote_pool.open(&remote_src_ident).await;
    remote_src.snapshot(vec![]).await;
    let mut remote_req = open(&mut remote_pool, "remote-packed-req", true).await;
    let mut remote_entry = support::pool::entry(128, None, v2::Tier::Dram);
    remote_entry.bucket = vec![4; 16];
    remote_entry.bytes = stored_bytes;
    remote_entry.packed = true;
    remote_entry.required_profiles = vec![full_offer.units[0].profile.clone()];
    remote_src
        .send(v2::client_message::M::Notices(v2::NoticeBatch {
            notices: vec![support::pool::saved(remote_entry)],
        }))
        .await;
    let mut lookup_id = 3u64;
    let remote_cand = loop {
        lookup_id += 1;
        let reply = remote_req
            .lookup_request(v2::LookupRequest {
                lookup_id,
                boundaries: vec![v2::PrefixKey {
                    consumed_tokens: 0,
                    digest: vec![4; 16],
                }],
                before_tokens: 129,
                ..Default::default()
            })
            .await;
        if let Some(c) = reply
            .candidates
            .into_iter()
            .find(|c| c.source.as_ref().unwrap().instance_id == remote_src_ident.instance_id)
        {
            break c;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    };
    let mut remote_claim = remote_req.claim_for(20, &remote_cand, 64, true);
    remote_claim.mode = v2::CopyMode::DataChannel as i32;
    remote_req.claim(remote_claim).await;
    let rreq = remote_src
        .take(|m| match m {
            v2::server_message::M::ExportRequest(r) => Some(r.clone()),
            _ => None,
        })
        .await;
    remote_src
        .send(v2::client_message::M::PinAck(v2::PinAck {
            export_id: rreq.export_id,
            result: v2::PinResult::Pinned as i32,
            refusal: 0,
            records: vec![v2::Record {
                key: rreq.key.clone(),
                first_page: 1,
                pages: 1,
                row: true,
                packed: Some(suffix_offer.clone()),
            }],
            total_pages: 1,
            row: true,
        }))
        .await;
    let rgranted = remote_req.claim_granted(20).await;
    let mut src_conn =
        channel::dial(rreq.channel.as_ref().unwrap(), &Security::Plain).await;
    let mut req_conn =
        channel::dial(rgranted.channel.as_ref().unwrap(), &Security::Plain).await;
    let mut sent_units = 0usize;
    while sent_units < suffix_offer.units.len() {
        let ToInstance::Grant(shape) = ToInstance::read(&mut src_conn, 64).await.unwrap() else {
            panic!("expected grant on remote source channel");
        };
        let unit = &suffix_offer.units[sent_units];
        let bytes = payload(unit);
        assert_eq!(bytes.len() as u64, shape.length);
        ToManager::Filled {
            index: shape.index,
            row_xxh3: None,
            page_xxh3: Vec::new(),
        }
        .write(&mut src_conn)
        .await
        .unwrap();
        src_conn.write_all(bytes).await.unwrap();
        src_conn.flush().await.unwrap();
        sent_units += 1;
    }
    let mut got_units = Vec::new();
    while got_units.len() < suffix_offer.units.len() {
        let ToInstance::Window(window) = ToInstance::read(&mut req_conn, 64).await.unwrap() else {
            panic!("expected packed window on remote requester channel");
        };
        let mut data = vec![0; window.shape.length as usize];
        req_conn.read_exact(&mut data).await.unwrap();
        let units = window.packed.unwrap().units;
        packed::verify_units(&units, &data).unwrap();
        got_units.extend(units);
    }
    assert_eq!(got_units, suffix_offer.units);
    remote_req.copy_result(20, v2::CopyOutcome::CopyOk).await;
    assert_eq!(
        remote_req.claim_terminal(20).await.state,
        v2::ClaimState::Copied as i32
    );
    remote_req.fence(20, v2::FenceOutcome::Drained).await;
    remote_src
        .send(v2::client_message::M::ExportEnd(v2::ExportEnd {
            export_id: rreq.export_id,
            status: v2::ExportEndStatus::ExportComplete as i32,
            fence: v2::FenceOutcome::Drained as i32,
        }))
        .await;

    drop((
        src_conn,
        req_conn,
        remote_src,
        remote_req,
        remote_pool,
    ));
    host.stop();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn all_literal_and_literal_only_suffix_offers_retire_and_restore_as_v4() {
    let mut host = Host::with_storage(64 << 20, 64 << 20);
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut writer = open(&mut pool, "literal-writer", true).await;

    let lit_p0 = vec![0x5Au8; 4096];
    let lit_p1 = vec![0xA5u8; 4096];
    let lit_row = vec![0x3Cu8; 4096];
    let make_lit = |kind: i32, logical_index: u64, data: &[u8]| v2::PackedUnit {
        kind,
        logical_index,
        stored_rank: 0,
        representation: v2::packed_unit::Representation::Literal as i32,
        original_length: data.len() as u64,
        stored_length: data.len() as u64,
        raw_xxh3: flare_selene::record::checksum(data),
        ..Default::default()
    };

    // 1. All-literal root checkpoint must be accepted as packed v4 (not rejected as raw).
    let all_lit_offer = v2::PackedOffer {
        first_page: 0,
        units: vec![
            make_lit(0, 0, &lit_p0),
            make_lit(0, 1, &lit_p1),
            make_lit(1, 0, &lit_row),
        ],
    };
    let all_lit_payload = [lit_p0.as_slice(), lit_p1.as_slice(), lit_row.as_slice()].concat();
    put_packed(
        &mut writer,
        1,
        128,
        None,
        2,
        2,
        all_lit_offer,
        &all_lit_payload,
        4096,
    )
    .await;

    // 2. Mixed parent (compressed page 0 + row) followed by a child whose full offer has
    //    compressed page 0 + literal suffix (page 1 + row): manager accepts suffix at first_page=1
    //    as packed v4 so the client's packed-mode stream succeeds.
    let parent_offer = v2::PackedOffer {
        first_page: 0,
        units: vec![offer().units[0].clone(), offer().units[2].clone()],
    };
    let parent_payload = [page0(), row()].concat();
    put_packed(&mut writer, 2, 64, None, 1, 1, parent_offer, &parent_payload, 4096).await;

    let child_full_offer = v2::PackedOffer {
        first_page: 0,
        units: vec![
            offer().units[0].clone(),
            make_lit(0, 1, &lit_p1),
            make_lit(1, 0, &lit_row),
        ],
    };
    let child_suffix_offer = v2::PackedOffer {
        first_page: 1,
        units: vec![make_lit(0, 1, &lit_p1), make_lit(1, 0, &lit_row)],
    };
    let child_suffix_payload = [lit_p1.as_slice(), lit_row.as_slice()].concat();
    writer
        .send(v2::client_message::M::PutBegin(v2::PutBegin {
            ancestor_keys: vec![v2::PrefixKey {
                consumed_tokens: 64,
                digest: key(64).digest.to_vec(),
            }],
            put_id: 3,
            key: Some(v2::PrefixKey {
                consumed_tokens: 129,
                digest: key(129).digest.to_vec(),
            }),
            chain_pages: 2,
            capture_role: flare_selene::CaptureRole::Endpoint as i32,
            child_paths: 0,
            descendant_row: false,
            next_child: None,
            row: true,
            has_published_parent: true,
            published_parent: Some(v2::PrefixKey {
                consumed_tokens: 64,
                digest: key(64).digest.to_vec(),
            }),
            bucket: vec![4; 16],
            published_tail_pages: 1,
            mode: v2::CopyMode::SharedWindows as i32,
            packed: Some(child_full_offer),
        }))
        .await;
    let accept = writer
        .take(|m| match m {
            v2::server_message::M::PutAccept(a) if a.put_id == 3 => Some(a.clone()),
            _ => None,
        })
        .await;
    assert!(accept.packed, "literal-only suffix must be accepted as packed v4");
    assert_eq!(accept.first_page, 1);
    let descriptor = support::descriptor();
    let header = selene_manager::record::new_header(
        &descriptor,
        key(129),
        Some(key(64)),
        [4; 16],
        1,
    )
    .unwrap()
    .with_capture_role(flare_selene::CaptureRole::Endpoint)
    .with_packed(child_suffix_offer)
    .unwrap();
    let metadata = header.packed_metadata().unwrap();
    assert_eq!(accept.record_bytes, header.length(true));
    let mut filled = 0usize;
    while filled < child_suffix_payload.len() {
        let grant = writer
            .take(|m| match m {
                v2::server_message::M::PutGrant(g) if g.put_id == 3 => Some(g.clone()),
                _ => None,
            })
            .await;
        let window = grant.window.unwrap();
        let end = filled + window.length as usize;
        writer
            .shm
            .as_ref()
            .unwrap()
            .write(&window, &child_suffix_payload[filled..end]);
        writer
            .send(v2::client_message::M::PutFilled(v2::PutFilled {
                put_id: 3,
                index: window.index,
                row_xxh3: 0,
                page_xxh3: Vec::new(),
                has_row: false,
                rank_sums: Vec::new(),
            }))
            .await;
        filled = end;
    }
    writer
        .send(v2::client_message::M::PutCommit(v2::PutCommit {
            put_id: 3,
            metadata_offset: 0,
            metadata,
        }))
        .await;
    let stored = writer
        .take(|m| match m {
            v2::server_message::M::Stored(s) if s.put_id == 3 => Some(s.bytes),
            _ => None,
        })
        .await;
    assert_eq!(stored, header.length(true));
    support::put::fence(&mut writer, 3).await;

    // Verify both records can be looked up and claimed back cleanly with checksum verification.
    let cand = writer.manager_candidate(4, 128).await;
    writer.claim(writer.claim_for(10, &cand, 0, true)).await;
    let _granted = writer.claim_granted(10).await;
    let mut got = Vec::new();
    while got.len() < all_lit_payload.len() {
        let w = writer.window_ready(10).await.window.unwrap();
        let data = writer.shm.as_ref().unwrap().read(&w);
        packed::verify_units(&w.packed.as_ref().unwrap().units, &data).unwrap();
        got.extend_from_slice(&data);
        writer.released(10, w.index).await;
    }
    assert_eq!(got, all_lit_payload);
    writer.copy_result(10, v2::CopyOutcome::CopyOk).await;
    writer.claim_terminal(10).await;
    writer.fence(10, v2::FenceOutcome::Drained).await;

    // Also claim and restore the 129-token child endpoint to verify full chain reconstruction:
    // parent compressed page 0 + child literal page 1 + child literal row.
    let child_cand = writer.manager_candidate(4, 129).await;
    writer.claim(writer.claim_for(11, &child_cand, 0, true)).await;
    let _child_granted = writer.claim_granted(11).await;
    let expected_chain = [page0(), lit_p1.as_slice(), lit_row.as_slice()].concat();
    let mut got_child = Vec::new();
    while got_child.len() < expected_chain.len() {
        let w = writer.window_ready(11).await.window.unwrap();
        let data = writer.shm.as_ref().unwrap().read(&w);
        packed::verify_units(&w.packed.as_ref().unwrap().units, &data).unwrap();
        got_child.extend_from_slice(&data);
        writer.released(11, w.index).await;
    }
    assert_eq!(got_child, expected_chain);
    writer.copy_result(11, v2::CopyOutcome::CopyOk).await;
    writer.claim_terminal(11).await;
    writer.fence(11, v2::FenceOutcome::Drained).await;

    drop((writer, pool));
    host.stop();
}
