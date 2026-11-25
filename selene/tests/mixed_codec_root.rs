/*
 * SPDX-License-Identifier: Apache-2.0
 */

use flare_selene::{
    PrefixKey,
    record::{self, DirectoryLock, Header, RankSizes},
    shared::{
        codec::Plan,
        wire::{PackedOffer, PackedUnit},
    },
};
use std::{fs, io::Write};

fn key(n: u64) -> PrefixKey {
    PrefixKey {
        consumed_tokens: n * 64,
        digest: [n as u8; 16],
    }
}

fn encode_packet(seed: usize) -> (Vec<u8>, u32, u32) {
    let raw: Vec<u8> = (0..4096)
        .map(|j| ((j / (8 + 2 * seed) + seed) % (2 + seed)) as u8)
        .collect();
    let plan = Plan::new(raw.len(), &[]).unwrap();
    let mut pkt = vec![0u8; plan.max_packet()];
    let (unit, used) = plan
        .encode_capture(PackedUnit::default(), &[7u8; 32], &raw, &mut pkt, None)
        .unwrap();
    pkt.truncate(used);
    assert_eq!(unit.representation, 1);
    let mut back = vec![0u8; raw.len()];
    plan.decode_checked(&[&pkt], &mut back, unit.stored_crc, unit.original_crc, None)
        .unwrap();
    assert_eq!(back, raw);
    (pkt, unit.stored_crc, unit.original_crc)
}

#[test]
fn scan_recovers_one_chain_spanning_v1_v2_v3_and_v4() {
    let directory =
        std::env::temp_dir().join(format!("selene-mixed-codec-root-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let lock = DirectoryLock::acquire(&directory).unwrap();
    let ranks = vec![RankSizes {
        planes: vec![4000],
        row_bytes: 1000,
    }];
    let header = |n: u64, p: Option<u64>| {
        Header::from_sizes(
            64,
            ranks.clone(),
            [1; 32],
            [2; 16],
            key(n),
            p.map(key),
            [3; 16],
            (n - p.unwrap_or(0)) as u32,
            0,
        )
        .unwrap()
    };
    for version in 1..=3u32 {
        let n = u64::from(version);
        let h = header(n, (n > 1).then_some(n - 1));
        let mut block = h.encode().unwrap();
        block[8..12].copy_from_slice(&version.to_le_bytes());
        if version <= 2 {
            if version == 2 {
                block[12..16].copy_from_slice(&(1u32 << 31).to_le_bytes())
            }
            block[181..189].copy_from_slice(&4000u64.to_le_bytes());
            block[189..197].copy_from_slice(&1000u64.to_le_bytes());
        }
        let mut f = fs::File::create(directory.join(record::file_name(&h.key))).unwrap();
        f.write_all(&block).unwrap();
        f.set_len(h.length(true)).unwrap();
    }
    let (page0, p0_stored, p0_orig) = encode_packet(0);
    let (page1, p1_stored, p1_orig) = encode_packet(1);
    let (row_pkt, row_stored, row_orig) = encode_packet(2);
    let profile = vec![7u8; 32];
    let offer = PackedOffer {
        first_page: 3,
        units: vec![
            PackedUnit {
                kind: 0,
                logical_index: 3,
                representation: 1,
                profile: profile.clone(),
                original_length: 4096,
                stored_length: page0.len() as u64,
                stored_crc: p0_stored,
                original_crc: p0_orig,
                ..Default::default()
            },
            PackedUnit {
                kind: 0,
                logical_index: 4,
                representation: 1,
                profile: profile.clone(),
                original_length: 4096,
                stored_length: page1.len() as u64,
                stored_crc: p1_stored,
                original_crc: p1_orig,
                ..Default::default()
            },
            PackedUnit {
                kind: 1,
                logical_index: 0,
                representation: 1,
                profile,
                original_length: 4096,
                stored_length: row_pkt.len() as u64,
                stored_crc: row_stored,
                original_crc: row_orig,
                ..Default::default()
            },
        ],
    };
    let h = header(5, Some(3)).with_packed(offer).unwrap();
    let metadata = h.packed_metadata().unwrap();
    let file = directory.join(record::file_name(&h.key));
    fs::write(&file, [&metadata[..], &page0, &page1, &row_pkt].concat()).unwrap();
    let result = record::scan(&lock, &[1; 32], &[2; 16], 64, &ranks).unwrap();
    let survived = file.is_file();
    let keys: Vec<_> = result.iter().map(|r| (r.header.key, r.row)).collect();
    drop(lock);
    fs::remove_dir_all(&directory).unwrap();
    assert!(survived, "the scanner deleted the v4 checkpoint");
    assert_eq!(
        keys,
        [
            (key(1), true),
            (key(2), true),
            (key(3), true),
            (key(5), true)
        ]
    );
}
