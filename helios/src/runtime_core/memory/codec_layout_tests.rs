// SPDX-License-Identifier: Apache-2.0

use super::{Core, cache_geometry, codec_fields};
use crate::model::contract::CacheContract;
use crate::runtime_core::protobuf::{self, wire};
use flare_selene::codec_copy::checked_profile;
use prost::Message;
use std::path::Path;

fn load(name: &str) -> (Core, CacheContract) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/codec_geometry");
    let read = |part| std::fs::read(dir.join(format!("{name}.{part}.pb"))).unwrap();
    let rank = wire::BundleIndex::decode(read("rank").as_slice()).unwrap();
    let serving = wire::ServingContract::decode(read("contract").as_slice()).unwrap();
    let contract = protobuf::serving_contract(serving, 1).unwrap();
    let cache = serde_json::from_value(contract["model_contract"]["cache"].clone()).unwrap();
    (protobuf::decode(rank).unwrap(), cache)
}

fn qualifies(name: &str, pages: usize, fields: usize) -> flare_selene::CacheGeometry {
    let (core, cache) = load(name);
    let geometry = cache_geometry(
        &core,
        cache.page_tokens as u64,
        2,
        2,
        cache.codec_layout.as_ref(),
    )
    .unwrap();
    let ranks = std::slice::from_ref(&geometry);
    checked_profile(ranks).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(geometry.planes.len(), pages);
    assert_eq!(geometry.codec_fields.len(), fields);
    geometry
}

#[test]
fn exported_glm53_tp4_mtp5_qualifies() {
    let (_, cache) = load("glm53-tp4-mtp5");
    // Older cores exported before embedded GPU codec cubins decode with gpu_module absent.
    assert!(cache.codec_layout.as_ref().unwrap().gpu_module.is_none());
    qualifies("glm53-tp4-mtp5", 24, 130);
}

#[test]
fn codec_layout_gpu_module_round_trips_through_protobuf() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/codec_geometry");
    let raw = std::fs::read(dir.join("glm53-tp4-mtp5.contract.pb")).unwrap();
    let mut serving = wire::ServingContract::decode(raw.as_slice()).unwrap();
    serving
        .model_contract
        .as_mut()
        .unwrap()
        .cache
        .as_mut()
        .unwrap()
        .codec_layout
        .as_mut()
        .unwrap()
        .gpu_module = Some("kv_codec_gpu".into());
    let encoded = serving.encode_to_vec();
    let decoded = wire::ServingContract::decode(encoded.as_slice()).unwrap();
    let contract = protobuf::serving_contract(decoded, 1).unwrap();
    let cache: CacheContract =
        serde_json::from_value(contract["model_contract"]["cache"].clone()).unwrap();
    assert_eq!(
        cache.codec_layout.as_ref().unwrap().gpu_module.as_deref(),
        Some("kv_codec_gpu")
    );
}

#[test]
fn exported_deepseek41_qualifies() {
    let geometry = qualifies("deepseek41", 8, 71);
    for plane in &geometry.planes {
        let field = geometry
            .codec_fields
            .iter()
            .find(|f| f.buffer == plane.name)
            .unwrap();
        assert_eq!(
            field.encoding,
            if plane.name.ends_with(".latent_page") {
                47
            } else {
                0
            }
        );
    }
    let windows = geometry
        .codec_fields
        .iter()
        .filter(|f| f.buffer == geometry.row.name && f.encoding == 45);
    assert_eq!(windows.count(), 43);
}

#[test]
fn exported_qwen38_tp1_qualifies() {
    qualifies("qwen38-tp1", 39, 212);
}

#[test]
fn exported_qwen38_tp2_qualifies() {
    qualifies("qwen38-tp2", 39, 212);
}

#[test]
fn fields_are_relative_to_the_view_and_must_fit_one_unit() {
    let (mut core, cache) = load("glm53-tp4-mtp5");
    let mut layout = cache.codec_layout.unwrap();
    layout
        .fields
        .retain(|field| field.buffer == "cache.11.index");
    let buffer = core
        .execution
        .as_mut()
        .unwrap()
        .buffers
        .iter_mut()
        .find(|b| b.name == "cache.11.index")
        .unwrap();
    buffer.reference.as_mut().unwrap().offset = 16;
    buffer.nbytes = 32;
    layout.fields[0].offset = 8;
    layout.fields[0].length = 24;
    let fields = codec_fields(&core, &layout).unwrap();
    assert_eq!(
        (fields[0].buffer.as_str(), fields[0].offset),
        ("kv-11-index", 24)
    );
    layout.fields[0].length = 25;
    assert!(codec_fields(&core, &layout).is_none());
    let buffer = core
        .execution
        .as_mut()
        .unwrap()
        .buffers
        .iter_mut()
        .find(|b| b.name == "cache.11.index")
        .unwrap();
    buffer.nbytes = 8448;
    buffer.reference.as_mut().unwrap().offset = 8440;
    assert!(codec_fields(&core, &layout).is_none());
    layout.fields[0].offset = u64::MAX;
    assert!(codec_fields(&core, &layout).is_none());
}
