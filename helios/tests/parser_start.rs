/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#![cfg(feature = "server")]
//! Output-parser start state for each recipe whose checkpoint tokenizer is available.
//!
//! `prepare_output` finds the assistant marker and reasoning delimiters in the rendered text and
//! skips a spelling that request text wrote. These tests render each checkpoint's prepared chat
//! template and check that a generation prompt still opens reasoning from the template's own
//! delimiter and that a continued assistant message whose text spells the opener does not.

#[path = "support/templates.rs"]
mod template_fixtures;

use flare_helios::frontend::{Asset, Frontend, FrontendAssets};
use flare_helios::output::tagged::OutputStart;
use flare_helios::{frontend, model, scheduler};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
#[path = "support/frontend.rs"]
mod support;

fn asset(path: PathBuf) -> Asset {
    Asset {
        range: None,
        sha256: format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
        path,
    }
}

struct Family {
    recipe: &'static str,
    /// Recipe whose contract fixture declares this family's prompt markers. Step-5 and Nemotron
    /// declare the same ChatML markers and `<think>` delimiters as Qwen3.8.
    contract: &'static str,
    /// Environment variable naming the checkpoint directory that holds the tokenizer.
    variable: &'static str,
    /// `enable_thinking` values to render a generation prompt with (`None` leaves it unset), and
    /// the parser start each produces.
    generation: &'static [(Option<bool>, OutputStart)],
    /// The start after continuing the final message's `reasoning_content`.
    continued_reasoning: OutputStart,
}

fn check(family: &Family) {
    let root = PathBuf::from(std::env::var_os(family.variable).unwrap());
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset(root.join("tokenizer.json")),
            tokens: support::tokens(&asset(root.join("tokenizer_config.json"))).unwrap(),
            chat_template: Some(asset(template_fixtures::path(family.recipe, "chat_template"))),
        },
        Map::new(),
    )
    .unwrap();
    let config = support::model_config(support::contract(family.contract));
    let syntax = &config.contract.frontend;
    let delimiters = model::reasoning_delimiter_tokens(&frontend, syntax).unwrap();
    let opener = &syntax.reasoning.as_ref().unwrap().open;
    // `keep` false drops the request ranges, as for a prompt that records no request text.
    let start_with = |messages: &Value, options: &Value, keep: bool| {
        let messages = messages.as_array().unwrap();
        let mut rendered = frontend.render_chat(messages, options.as_object().unwrap()).unwrap();
        if !keep {
            rendered.ordinary.clear();
        }
        let output = model::prepare_output(
            &config,
            &frontend,
            &rendered,
            messages,
            options.as_object().unwrap(),
            false,
            &delimiters,
        )
        .unwrap();
        (output.content.start, rendered.text)
    };
    let start = |messages: Value, options: Value| start_with(&messages, &options, true);
    let user = json!({"role": "user", "content": "Hello"});
    for &(thinking, expected) in family.generation {
        let mut options = json!({"add_generation_prompt": true});
        if let Some(thinking) = thinking {
            options["enable_thinking"] = thinking.into();
        }
        let (actual, text) = start(json!([user]), options);
        assert_eq!(actual, expected, "{} enable_thinking={thinking:?}: {text:?}", family.recipe);
    }
    let reply = json!({"role": "assistant", "content": format!("{opener}\nSure")});
    let (actual, text) = start(json!([user, reply]), json!({"continue_final_message": true}));
    assert_eq!(actual, OutputStart::Content, "{}: {text:?}", family.recipe);
    // Request text that spells nothing: continuing the reasoning of the final message gives the
    // start of a prompt with no recorded request text, including where the template writes the
    // assistant header from the request role (`'<|im_start|>' + message.role`).
    let reply = json!({"role": "assistant", "content": "", "reasoning_content": "plan"});
    let messages = json!([user, reply]);
    let options = json!({"continue_final_message": "reasoning_content"});
    let (without_ranges, text) = start_with(&messages, &options, false);
    let (actual, _) = start_with(&messages, &options, true);
    assert_eq!(without_ranges, family.continued_reasoning, "{}: {text:?}", family.recipe);
    assert_eq!(actual, without_ranges, "{}: {text:?}", family.recipe);
}

#[test]
#[ignore = "requires FLARE_GLM_TEST_SNAPSHOT with the GLM-5.3 tokenizer and FLARE_TEST_TEMPLATE_DIR"]
fn glm53_parser_start() {
    check(&Family {
        recipe: "glm53",
        contract: "glm53",
        variable: "FLARE_GLM_TEST_SNAPSHOT",
        generation: &[(None, OutputStart::Reasoning)],
        continued_reasoning: OutputStart::Reasoning,
    });
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT with the Qwen3.8 tokenizer and FLARE_TEST_TEMPLATE_DIR"]
fn qwen38_parser_start() {
    check(&Family {
        recipe: "qwen38",
        contract: "qwen38",
        variable: "FLARE_TEST_SNAPSHOT",
        generation: &[(Some(true), OutputStart::Reasoning), (Some(false), OutputStart::Content)],
        continued_reasoning: OutputStart::Reasoning,
    });
}

#[test]
#[ignore = "requires FLARE_STEP5_TEST_SNAPSHOT with the Step-5 tokenizer and FLARE_TEST_TEMPLATE_DIR"]
fn step5_parser_start() {
    check(&Family {
        recipe: "step5",
        contract: "qwen38",
        variable: "FLARE_STEP5_TEST_SNAPSHOT",
        generation: &[(None, OutputStart::Reasoning)],
        continued_reasoning: OutputStart::Reasoning,
    });
}

#[test]
#[ignore = "requires FLARE_NEMOTRON_TEST_SNAPSHOT with the Nemotron-3.5 tokenizer and FLARE_TEST_TEMPLATE_DIR"]
fn nemotron_h_parser_start() {
    check(&Family {
        recipe: "nemotron_h",
        contract: "qwen38",
        variable: "FLARE_NEMOTRON_TEST_SNAPSHOT",
        generation: &[(Some(true), OutputStart::Reasoning), (Some(false), OutputStart::Content)],
        continued_reasoning: OutputStart::Reasoning,
    });
}

#[test]
#[ignore = "requires FLARE_MIMO_TEST_SNAPSHOT with the MiMo-V2 tokenizer and FLARE_TEST_TEMPLATE_DIR"]
fn mimo_v2_parser_start() {
    check(&Family {
        recipe: "mimo_v2",
        contract: "mimo_v2",
        variable: "FLARE_MIMO_TEST_SNAPSHOT",
        generation: &[(Some(true), OutputStart::Content), (Some(false), OutputStart::Content)],
        continued_reasoning: OutputStart::Reasoning,
    });
}
