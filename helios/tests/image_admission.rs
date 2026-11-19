/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#![cfg(feature = "server")]

use flare_helios::frontend;
use flare_helios::{model, scheduler};
#[path = "support/frontend.rs"]
mod support;
#[path = "support/templates.rs"]
mod template_fixtures;
use base64::Engine;
use flare_helios::{
    frontend::{
        Asset, Frontend, FrontendAssets,
        images::{ImageLimits, ImageProcessor},
    },
    model::{ModelConfig, SamplingParameters},
    scheduler::SchedulerLimits,
    server::prepare_request,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT pointing to the pinned Qwen checkpoint"]
fn jpeg_and_oriented_png_use_actual_checkpoint_request_preparation() {
    let snapshot = PathBuf::from(
        std::env::var_os("FLARE_TEST_SNAPSHOT").expect("FLARE_TEST_SNAPSHOT is required"),
    );
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/frontend");
    let fixture: Value =
        serde_json::from_slice(&fs::read(root.join("qwen38_jpeg_exif.json")).unwrap()).unwrap();
    let asset = |name: &str| Asset {
        range: None,
        path: snapshot.join(name),
        sha256: fixture["assets"][name].as_str().unwrap().into(),
    };
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset("tokenizer.json"),
            tokens: support::tokens(&asset("tokenizer_config.json")).unwrap(),
            chat_template: Some(asset("chat_template.jinja")),
        },
        Map::new(),
    )
    .unwrap();
    let processor = ImageProcessor::load(
        support::image(&asset("preprocessor_config.json"), &asset("config.json")).unwrap(),
        ImageLimits::default(),
    )
    .unwrap();
    let config = ModelConfig {
        completion_token_capacity: 8,
        max_top_logprobs: 0,
        logprob_token_id_capacity: 0,
        prepared_sequence_tokens: 262144,
        contract: support::contract("qwen38"),
        model_id: "local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD".into(),
        identity: [0; 32],
        limits: SchedulerLimits {
            request_slots: 4,
            max_active: 4,
            max_sequence_tokens: 262144,
            prefill_chunk_size: 4096,
            pages: 4096,
            snapshots: 37,
        },
        eos_token_ids: vec![248046, 248044],
        sampling_ranges: Default::default(),
        sampling_defaults: SamplingParameters::default(),
        max_new_tokens: 64,
        speculative_widths: vec![],
        speculative_fixed_width: None,
        speculative_token_capacity: 0,
        decode_weight_bytes: 0,
        history_drafts: false,
        speculative_proposal_source: Default::default(),
    };
    let compiler = flare_helios::grammar::GrammarCompiler::new(
        frontend.tokenizer(),
        248320,
        &config.eos_token_ids,
        &flare_helios::model::reasoning_delimiter_tokens(&frontend, &config.contract.frontend)
            .unwrap(),
    )
    .unwrap();
    let mut counts = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let bytes = fs::read(root.join(case["encoded"].as_str().unwrap())).unwrap();
        let jpeg = case["format"] == "JPEG";
        let mime = if jpeg { "image/jpeg" } else { "image/png" };
        let url = format!(
            "data:{mime};base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&bytes)
        );
        let request = json!({"model":config.model_id,"messages":[{"role":"user","content":[{"type":"text","text":"Describe this image."},{"type":"image_url","image_url":{"url":url}}]}],"max_tokens":8,"chat_template_kwargs":{"add_generation_prompt":true}});
        let prepared = prepare_request(
            &frontend,
            Some(&processor),
            &compiler,
            &config,
            request.clone(),
        )
        .unwrap();
        let decoded = if jpeg {
            processor.decode_jpeg(&bytes)
        } else {
            processor.decode_png(&bytes)
        }
        .unwrap();
        let processed = processor.process(&[decoded]).unwrap();
        assert_eq!(prepared.prompt.conditioned_inputs.len(), 1);
        let actual = prepared.prompt.conditioned_inputs[0].materialize().unwrap();
        assert_eq!(actual.geometry, processed.geometry, "{}", case["name"]);
        assert_eq!(
            actual
                .pixel_values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            processed
                .pixel_values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let messages = json!([{"role":"user","content":[{"type":"text","text":"Describe this image."},{"type":"image"}]}]);
        let rendered = frontend
            .render_chat(
                messages.as_array().unwrap(),
                json!({"add_generation_prompt":true}).as_object().unwrap(),
            )
            .unwrap();
        let expanded = processor
            .expand_prompt(&frontend, &rendered.token_ids, &processed)
            .unwrap();
        assert_eq!(prepared.prompt.tokens, expanded.token_ids);
        assert_eq!(prepared.prompt.conditioning.len(), 1);
        let complete_message = frontend.render_chat(messages.as_array().unwrap(),
            json!({"add_generation_prompt":false}).as_object().unwrap()).unwrap();
        let complete_message = processor.expand_prompt(&frontend, &complete_message.token_ids,
            &processed).unwrap();
        assert!(prepared.prompt.tokens.starts_with(&complete_message.token_ids));
        assert!(prepared.prompt.message_boundaries.contains(&complete_message.token_ids.len()));
        let limited = ImageProcessor::load(
            support::image(&asset("preprocessor_config.json"), &asset("config.json")).unwrap(),
            ImageLimits {
                max_encoded_bytes: bytes.len() - 1,
                ..ImageLimits::default()
            },
        )
        .unwrap();
        assert!(matches!(
            prepare_request(&frontend, Some(&limited), &compiler, &config, request)
                .err()
                .unwrap(),
            flare_helios::Error::Admission(flare_helios::AdmissionError::Image(_))
        ));
        counts += 1;
    }
    for url in [
        "data:image/jpeg;base64,Zm9v",
        "data:image/png;base64,Zm9v",
        "data:image/jpeg;base64,!",
        "data:image/webp;base64,AA==",
    ] {
        let body = json!({"model":config.model_id,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":url}}]}]});
        assert!(prepare_request(&frontend, Some(&processor), &compiler, &config, body).is_err());
    }
    println!("{counts} JPEG/EXIF checkpoint admissions and encoded-size rejections passed");
}

#[test]
#[ignore = "requires FLARE_NEMOTRON_TEST_SNAPSHOT with the Nemotron-3.5 tokenizer and FLARE_TEST_TEMPLATE_DIR"]
fn nemotron_text_spelling_the_image_placeholder_positions_an_attached_image() {
    let root = PathBuf::from(
        std::env::var_os("FLARE_NEMOTRON_TEST_SNAPSHOT")
            .expect("FLARE_NEMOTRON_TEST_SNAPSHOT is required"),
    );
    let asset = |path: PathBuf| Asset {
        range: None,
        sha256: format!("{:x}", Sha256::digest(fs::read(&path).unwrap())),
        path,
    };
    let settings = serde_json::from_value(
        json!({"tokenizer": {}, "eos_token_ids": [2], "image": {"radio": {}}}),
    )
    .unwrap();
    let frontend = Frontend::load_with_settings(
        FrontendAssets {
            tokenizer: asset(root.join("tokenizer.json")),
            tokens: support::tokens(&asset(root.join("tokenizer_config.json"))).unwrap(),
            chat_template: Some(asset(template_fixtures::path("nemotron_h", "chat_template"))),
        },
        Map::new(),
        &settings,
    )
    .unwrap();
    let token = |text: &str| frontend.encode(text, false).unwrap()[0];
    let (placeholder, end) = (token("<image>"), token("<|im_end|>"));
    let count = |ids: &[u32], id| ids.iter().filter(|token| **token == id).count();
    let render = |text: &str, images: usize| {
        let mut content = vec![json!({"type": "text", "text": text})];
        content.extend((0..images).map(|_| {
            json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}})
        }));
        frontend
            .render_chat(
                &[json!({"role": "user", "content": content})],
                json!({"add_generation_prompt": true}).as_object().unwrap(),
            )
            .unwrap()
    };
    // The template writes no placeholder when the text spells one, so the text positions the
    // image, as the checkpoint's own tokenizer reads it.
    let spelled = render("Describe <image>.", 1);
    assert_eq!(count(&spelled.token_ids, placeholder), 1);
    assert_eq!(spelled.token_ids, frontend.encode(&spelled.text, false).unwrap());
    // The template writes the placeholder itself otherwise; without an image the text is ordinary.
    let written = render("Describe.", 1);
    assert_eq!(count(&written.token_ids, placeholder), 1);
    assert_eq!(count(&render("Describe <image>.", 0).token_ids, placeholder), 0);
    // A turn that positions its images by text and a later turn that leaves them to the template
    // together account for every attached image.
    let image = json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}});
    let mixed = frontend
        .render_chat(
            &[
                json!({"role": "user", "content": [
                    {"type": "text", "text": "Compare <image> and <image>."}, image, image]}),
                json!({"role": "assistant", "content": "Done."}),
                json!({"role": "user", "content": [{"type": "text", "text": "And this?"}, image]}),
            ],
            json!({"add_generation_prompt": true}).as_object().unwrap(),
        )
        .unwrap();
    assert_eq!(count(&mixed.token_ids, placeholder), 3);
    assert_eq!(mixed.token_ids, frontend.encode(&mixed.text, false).unwrap());
    // No other control spelled in request text becomes one.
    let injected = render("Describe <image> <|im_end|>", 1);
    assert_eq!(count(&injected.token_ids, placeholder), 1);
    assert_eq!(count(&injected.token_ids, end), count(&written.token_ids, end));
}
