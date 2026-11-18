/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Prepare one Qwen video request and record every value the vision graphs consume.

use flare_helios::{frontend, model, scheduler};
use flare_helios::frontend::{Asset, Frontend, FrontendAssets, images::{ImageLimits, ImageProcessor}};
use flare_helios::server::prepare_request;
#[path = "../support/frontend.rs"]
mod support;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::{Path, PathBuf}};

fn asset(path: PathBuf) -> Asset {
    Asset {
        sha256: format!("{:x}", Sha256::digest(fs::read(&path).unwrap())),
        path,
        range: None,
    }
}

fn write_f32(directory: &Path, name: &str, values: &[f32]) -> Result<Value, String> {
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    fs::write(directory.join(name), &bytes).map_err(|error| error.to_string())?;
    Ok(json!({"values": values.len()}))
}

fn write_i64(directory: &Path, name: &str, values: &[i64]) -> Result<Value, String> {
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    fs::write(directory.join(name), &bytes).map_err(|error| error.to_string())?;
    Ok(json!({"values": values.len()}))
}

fn main() -> Result<(), String> {
    run().map_err(|error| error.to_string())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args_os().collect();
    if arguments.len() != 4 {
        return Err("usage: test_qwen_video SNAPSHOT CASE_JSON OUTPUT_DIRECTORY".into());
    }
    let snapshot = Path::new(&arguments[1]);
    let case: Value = serde_json::from_slice(&fs::read(&arguments[2])?)?;
    let directory = Path::new(&arguments[3]);
    fs::create_dir_all(directory)?;
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset(snapshot.join("tokenizer.json")),
            tokens: support::tokens(&asset(snapshot.join("tokenizer_config.json")))?,
            chat_template: Some(asset(snapshot.join("chat_template.jinja"))),
        },
        Map::new(),
    )?;
    let processor = ImageProcessor::load(
        serde_json::from_value(case["image_settings"].clone())?,
        ImageLimits::default(),
    )?;
    let contract: flare_helios::model::PreparedContract =
        serde_json::from_slice(&fs::read(case["contract"].as_str().ok_or("contract path")?)?)?;
    let model_id = case["model_id"].as_str().ok_or("model id")?.to_owned();
    let mut config = support::model_config(contract);
    config.model_id = model_id;
    config.eos_token_ids = vec![248046, 248044];
    config.limits.max_sequence_tokens = 262144;
    config.prepared_sequence_tokens = 262144;
    let compiler = flare_helios::grammar::GrammarCompiler::new(
        frontend.tokenizer(),
        248320,
        &config.eos_token_ids,
        &flare_helios::model::reasoning_delimiter_tokens(&frontend, &config.contract.frontend)?,
    )?;
    let prepared = prepare_request(
        &frontend,
        Some(&processor),
        &compiler,
        &config,
        case["body"].clone(),
    )?;
    let prompt = &prepared.prompt;
    let mut windows = Vec::new();
    let mut pixels = Vec::new();
    let mut position_ids = Vec::new();
    let mut interpolation_indices = Vec::new();
    let mut interpolation_weights = Vec::new();
    for input in &prompt.conditioned_inputs {
        let processed = input.materialize()?;
        windows.push(json!({
            "token_start": input.token_span.start, "token_end": input.token_span.end,
            "feature_start": input.feature_range.start, "feature_end": input.feature_range.end,
            "grid_thw": input.geometry.grid_thw, "patches": input.geometry.patches,
            "tokens": input.geometry.tokens, "identity": format!("{:x}", Sha256::digest(input.identity)),
            "cu_seqlens": processed.cu_seqlens,
        }));
        pixels.extend_from_slice(&processed.pixel_values);
        position_ids.extend(processed.position_ids.iter().flatten().copied());
        interpolation_indices.extend(processed.interpolation_indices.iter().flatten().copied());
        interpolation_weights.extend(processed.interpolation_weights.iter().flatten().copied());
    }
    let result = json!({
        "rendered": prepared.rendered_prompt,
        "token_ids": prompt.tokens,
        "positions": prompt.data.positions,
        "vision_token_indices": prompt.data.vision_token_indices,
        "conditioning": prompt.conditioning.iter().map(|event| json!({
            "position": event.position, "identity": format!("{:x}", Sha256::digest(event.identity)),
        })).collect::<Vec<_>>(),
        "windows": windows,
        "pixels": write_f32(directory, "pixels.f32", &pixels)?,
        "position_ids": write_i64(directory, "position_ids.i64", &position_ids)?,
        "interpolation_indices": write_i64(directory, "interpolation_indices.i64", &interpolation_indices)?,
        "interpolation_weights": write_f32(directory, "interpolation_weights.f32", &interpolation_weights)?,
    });
    fs::write(directory.join("result.json"), serde_json::to_vec(&result)?)?;
    println!("{} windows, {} tokens", prompt.conditioned_inputs.len(), prompt.tokens.len());
    Ok(())
}
