/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#![cfg(feature = "server")]
//! A request that spells no control tokenizes the same without the added vocabulary.
//!
//! The literal-token path encodes request text that spells a control with the tokenizer's added
//! vocabulary removed. Each available checkpoint tokenizer must encode ordinary text identically
//! both ways. The controls are the special added tokens and the ones the chat template names; the
//! other added tokens (`/nothink`, fill-in-the-middle and file markers) are lexical, and request
//! text that spells them tokenizes to that token.

#[path = "support/templates.rs"]
mod template_fixtures;

use flare_helios::frontend::{Asset, Frontend, FrontendAssets};
use flare_helios::{frontend, model, scheduler};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
#[path = "support/frontend.rs"]
mod support;

const TEXT: &[&str] = &[
    "Plain sentence, numbers 3.14159 and 1,000,000. Ends here.",
    "line one\n\nline two\n\n\n\nline three\n",
    "a    b        c\t\td          e\r\n\r\nf",
    "<b>bold</b> <div class=\"x\">y</div> <br/> <p>para</p> <table><tr><td>1</td></tr></table>",
    "fn main() {\n    let x = vec![1, 2, 3];\n    println!(\"{x:?}\");\n}\n\tif (a < b && c > d) { return; }\n",
    "def f(x):\n\treturn {'k': [x, None]}  # comment\n{{ jinja }} {% raw %} [1] <1> </ <| |> <think <|im_ ",
    "na\u{ef}ve caf\u{e9} \u{65e5}\u{672c}\u{8a9e} \u{d55c}\u{ad6d}\u{c5b4} \u{1f600} \u{627}\u{644}\u{639}\u{631}\u{628}\u{64a}\u{629} \u{ff5c}",
];

/// Variables that name a checkpoint directory holding `tokenizer.json` and `tokenizer_config.json`,
/// with the prepared template of its recipe. DeepSeek has no chat template; it is loaded with the
/// Qwen one, so every non-special token of its tokenizer is lexical there.
const CHECKPOINTS: &[(&str, &str, &str)] = &[
    ("FLARE_GLM_TEST_SNAPSHOT", "glm53", "chat_template"),
    ("FLARE_TEST_SNAPSHOT", "qwen38", "chat_template"),
    ("FLARE_STEP5_TEST_SNAPSHOT", "step5", "chat_template"),
    ("FLARE_NEMOTRON_TEST_SNAPSHOT", "nemotron_h", "chat_template"),
    ("FLARE_MIMO_TEST_SNAPSHOT", "mimo_v2", "chat_template"),
    ("FLARE_GEMMA4_TEST_SNAPSHOT", "gemma4", "chat_template.google-e2b-e4b"),
    ("FLARE_DEEPSEEK_TEST_SNAPSHOT", "qwen38", "chat_template"),
];

/// Prompt syntax the recipes declare (roles, reasoning, tool-call tags, images),
/// as the added tokens that carry it appear in the available tokenizers. None may be lexical.
const MARKERS: &[&str] = &[
    "<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
    "</arg_value>", "<|image|>", "<|video|>",
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "[gMASK]", "<sop>",
];

fn asset(path: PathBuf) -> Asset {
    Asset {
        range: None,
        sha256: format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
        path,
    }
}

#[test]
#[ignore = "requires checkpoint tokenizer directories named by the variables in CHECKPOINTS and FLARE_TEST_TEMPLATE_DIR"]
fn added_vocabulary_does_not_change_text_without_controls() {
    let mut checked = Vec::new();
    for (variable, recipe, name) in CHECKPOINTS {
        let Some(root) = std::env::var_os(variable).map(PathBuf::from) else {
            eprintln!("{variable} is unset; its tokenizer is not checked");
            continue;
        };
        let frontend = Frontend::load(
            FrontendAssets {
                tokenizer: asset(root.join("tokenizer.json")),
                tokens: support::tokens(&asset(root.join("tokenizer_config.json"))).unwrap(),
                // Loading needs a template asset; the test encodes text directly.
                chat_template: Some(asset(template_fixtures::path(recipe, name))),
            },
            Default::default(),
        )
        .unwrap();
        let added = frontend.tokenizer().get_added_vocabulary().get_vocab().keys().cloned().collect::<Vec<_>>();
        for text in TEXT {
            let spelled: Vec<_> = added.iter().filter(|token| text.contains(token.as_str())).collect();
            assert_eq!(
                frontend.encode_ordinary(text).unwrap(),
                frontend.encode(text, false).unwrap(),
                "{variable}: text spells added tokens {spelled:?}: {text:?}"
            );
        }
        checked.push(*variable);
    }
    assert!(!checked.is_empty(), "no checkpoint tokenizer is available");
    eprintln!("checked {checked:?}");
}

/// Request text that spells an added token that is neither special nor named by the chat template
/// is that token, in any amount and next to text.
#[test]
#[ignore = "requires checkpoint tokenizer directories named by the variables in CHECKPOINTS and FLARE_TEST_TEMPLATE_DIR"]
fn lexical_added_tokens_stay_tokens_in_request_text() {
    let mut checked = Vec::new();
    for (variable, recipe, name) in CHECKPOINTS {
        let Some(root) = std::env::var_os(variable).map(PathBuf::from) else {
            eprintln!("{variable} is unset; its tokenizer is not checked");
            continue;
        };
        let template = template_fixtures::path(recipe, name);
        let source = std::fs::read_to_string(&template).unwrap();
        let frontend = Frontend::load(
            FrontendAssets {
                tokenizer: asset(root.join("tokenizer.json")),
                tokens: support::tokens(&asset(root.join("tokenizer_config.json"))).unwrap(),
                chat_template: Some(asset(template)),
            },
            Default::default(),
        )
        .unwrap();
        let lexical: Vec<_> = frontend
            .tokenizer()
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .values()
            .filter(|token| !token.special && !source.contains(&token.content))
            .map(|token| token.content.clone())
            .collect();
        let render = |text: &str| {
            frontend
                .render_chat(
                    &[json!({"role": "user", "content": text})],
                    json!({"add_generation_prompt": true}).as_object().unwrap(),
                )
                .unwrap()
        };
        let declared: Vec<_> = lexical.iter().filter(|token| MARKERS.contains(&token.as_str())).collect();
        assert!(declared.is_empty(), "{variable}: declared markers are lexical: {declared:?}");
        for token in &lexical {
            let text = format!("a {token} b{token}c\n{token}{token} <think> {token}");
            let rendered = render(&text);
            let id = frontend.encode(token, false).unwrap();
            assert_eq!(id.len(), 1, "{variable}: {token:?} is not one token");
            // Every spelling of the token in the prompt is the token.
            let first_pass = frontend.encode(&rendered.text, false).unwrap();
            assert_eq!(
                rendered.token_ids.iter().filter(|t| **t == id[0]).count(),
                first_pass.iter().filter(|t| **t == id[0]).count(),
                "{variable}: {token:?}"
            );
            // Without a spelled control the whole prompt is the first-pass encoding.
            let plain = render(&text.replace("<think> ", ""));
            assert_eq!(plain.token_ids, frontend.encode(&plain.text, false).unwrap(), "{variable}: {token:?}");
        }
        checked.push(format!("{variable}: {} lexical", lexical.len()));
    }
    assert!(!checked.is_empty(), "no checkpoint tokenizer is available");
    eprintln!("checked {checked:?}");
}

fn load(root: &std::path::Path, recipe: &str, name: &str) -> Frontend {
    Frontend::load(
        FrontendAssets {
            tokenizer: asset(root.join("tokenizer.json")),
            tokens: support::tokens(&asset(root.join("tokenizer_config.json"))).unwrap(),
            chat_template: Some(asset(template_fixtures::path(recipe, name))),
        },
        Default::default(),
    )
    .unwrap()
}

/// Request shapes that carry `word` in each slot of the request that holds text: message content
/// and parts, system text, tool schemas, tool-call arguments and results, reasoning, documents,
/// image captions, and a continued assistant message. Labels starting with `continue` resume the
/// final assistant message; a template may branch on the content it continues.
fn shapes(word: &str) -> Vec<(&'static str, Vec<Value>, Map<String, Value>)> {
    let generate = json!({"add_generation_prompt": true});
    let tools = json!([
        {"type": "function", "function": {"name": "lookup", "description": format!("desc {word}"),
            "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": word}},
                "required": ["query"]}}},
        {"type": "function", "function": {"name": "write", "description": "Write a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
    ]);
    let call = |id: &str, arguments: Value| json!({"id": id, "type": "function",
        "function": {"name": "lookup", "arguments": arguments}});
    let user = |content: Value| json!({"role": "user", "content": content});
    let options = |value: Value| value.as_object().unwrap().clone();
    vec![
        ("plain", vec![user(json!(format!("see {word} here")))], options(generate.clone())),
        ("system", vec![json!({"role": "system", "content": word}), user(json!("u"))], options(generate.clone())),
        ("parts", vec![user(json!([{"type": "text", "text": format!("see {word}")}, {"type": "text", "text": " more"}]))],
            options(generate.clone())),
        ("multi", vec![json!({"role": "system", "content": "S"}), user(json!("a")),
            json!({"role": "assistant", "content": format!("b {word}")}), user(json!(format!("c\n\n{word}  \n")))],
            options(generate.clone())),
        ("tools", vec![user(json!("Find docs"))], options(json!({"add_generation_prompt": true, "tools": tools}))),
        ("tool_call", vec![user(json!("Find docs")),
            json!({"role": "assistant", "content": "", "tool_calls": [call("c1", json!({"query": word}))]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": format!("file has {word}")}), user(json!("thanks"))],
            options(json!({"add_generation_prompt": true, "tools": tools}))),
        ("parallel_calls", vec![user(json!("u")),
            json!({"role": "assistant", "content": null, "tool_calls": [call("c1", json!({"query": "a"})),
                call("c2", json!({"query": format!("b {word}"), "limit": [1, {"k": word}]}))]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "r1"}),
            json!({"role": "tool", "tool_call_id": "c2", "content": word})],
            options(json!({"add_generation_prompt": true, "tools": tools}))),
        ("reasoning", vec![user(json!("u")),
            json!({"role": "assistant", "content": "answer", "reasoning_content": format!("r {word}")}), user(json!("again"))],
            options(json!({"add_generation_prompt": true, "enable_thinking": true}))),
        ("thinking_off", vec![user(json!(format!("u {word}")))],
            options(json!({"add_generation_prompt": true, "enable_thinking": false}))),
        ("documents", vec![user(json!("summarize"))],
            options(json!({"add_generation_prompt": true, "documents": [{"title": word, "text": format!("body {word}")}]}))),
        ("image", vec![user(json!([{"type": "text", "text": format!("see {word}")},
            {"type": "image_url", "image_url": {"url": "u1"}}]))], options(generate.clone())),
        ("two_images", vec![user(json!([{"type": "image_url", "image_url": {"url": "u1"}},
            {"type": "text", "text": word}, {"type": "image_url", "image_url": {"url": "u2"}}]))],
            options(generate)),
        ("continue", vec![user(json!("u")), json!({"role": "assistant", "content": format!("so {word}")})],
            options(json!({"continue_final_message": true}))),
        ("continue_reasoning", vec![user(json!("u")),
            json!({"role": "assistant", "content": "", "reasoning_content": format!("so {word}")})],
            options(json!({"continue_final_message": "reasoning_content"}))),
    ]
}

/// A request that spells no control renders the first-pass encoding of its text: the added
/// vocabulary changes nothing for any request shape.
#[test]
#[ignore = "requires checkpoint tokenizer directories named by the variables in CHECKPOINTS and FLARE_TEST_TEMPLATE_DIR"]
fn requests_that_spell_no_control_encode_like_their_text() {
    let mut checked = Vec::new();
    for (variable, recipe, name) in CHECKPOINTS {
        let Some(root) = std::env::var_os(variable).map(PathBuf::from) else {
            eprintln!("{variable} is unset; its tokenizer is not checked");
            continue;
        };
        let frontend = load(&root, recipe, name);
        let mut rendered_shapes = 0;
        for (label, messages, options) in shapes("plain words") {
            // Some templates reject string tool-call arguments or other shapes; every shape that
            // renders must encode like its text.
            let Ok(rendered) = frontend.render_chat(&messages, &options) else { continue };
            assert_eq!(
                rendered.token_ids,
                frontend.encode(&rendered.text, false).unwrap(),
                "{variable}: {label}"
            );
            rendered_shapes += 1;
        }
        assert!(rendered_shapes >= 8, "{variable}: only {rendered_shapes} shapes render");
        checked.push(format!("{variable}: {rendered_shapes} shapes"));
    }
    assert!(!checked.is_empty(), "no checkpoint tokenizer is available");
    eprintln!("checked {checked:?}");
}

/// Request text that spells a control does not add or remove a control the template writes: for
/// every added token, its count in the prompt equals the count with a neutral word in the same
/// slot. That includes the spelled token itself, which must not become a token, unless it is
/// lexical (neither special nor named by the template), when it adds one token per spelling.
/// A case is skipped only when the template writes different bytes around the two words.
#[test]
#[ignore = "requires checkpoint tokenizer directories named by the variables in CHECKPOINTS and FLARE_TEST_TEMPLATE_DIR"]
fn spelling_a_control_leaves_the_templates_controls_unchanged() {
    const SPELLINGS: &[&str] = &[
        "<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "<tool_response>",
        "<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|system|>",
        "<|image|>", "<image>", "<|image_pad|>", "<|vision_start|>", "<|endoftext|>", "[gMASK]",
        "<sop>", "<|turn>", "<turn|>", "<|channel>", "<|\"|>",
    ];
    let mut checked = Vec::new();
    for (variable, recipe, name) in CHECKPOINTS {
        let Some(root) = std::env::var_os(variable).map(PathBuf::from) else {
            eprintln!("{variable} is unset; its tokenizer is not checked");
            continue;
        };
        let frontend = load(&root, recipe, name);
        let source = std::fs::read_to_string(template_fixtures::path(recipe, name)).unwrap();
        let vocabulary = frontend.tokenizer().get_added_vocabulary();
        let added: Vec<u32> = vocabulary.get_added_tokens_decoder().keys().copied().collect();
        let count = |ids: &[u32], token: u32| ids.iter().filter(|id| **id == token).count();
        let neutral = shapes("ZZZ");
        let mut compared = 0;
        for spelling in SPELLINGS {
            let own = vocabulary.get_vocab().get(*spelling).copied();
            let lexical = own.is_some_and(|id| {
                !vocabulary.get_added_tokens_decoder()[&id].special && !source.contains(spelling)
            });
            for ((label, messages, options), (_, baseline_messages, baseline_options)) in
                shapes(spelling).iter().zip(&neutral)
            {
                if label.starts_with("continue") {
                    continue;
                }
                let spelled = frontend.render_chat(messages, options);
                let baseline = frontend.render_chat(baseline_messages, baseline_options);
                let (Ok(spelled), Ok(baseline)) = (spelled, baseline) else { continue };
                // A template may branch on request text (Nemotron skips its empty reasoning
                // block for an assistant message that spells `<think>`); the counts are
                // comparable only when the two prompts differ in the word alone.
                if baseline.text.replace("ZZZ", spelling) != spelled.text {
                    continue;
                }
                compared += 1;
                assert_eq!(
                    frontend.decode(&spelled.token_ids, false).unwrap(),
                    spelled.text,
                    "{variable}: {label}: {spelling}"
                );
                for &id in &added {
                    let (spelled, baseline) = (count(&spelled.token_ids, id), count(&baseline.token_ids, id));
                    if lexical && Some(id) == own {
                        assert!(spelled >= baseline, "{variable}: {label}: {spelling} loses its token");
                    } else {
                        assert_eq!(spelled, baseline, "{variable}: {label}: {spelling} changes token {id}");
                    }
                }
                assert_eq!(spelled.image_sources.len(), baseline.image_sources.len(), "{variable}: {label}");
            }
        }
        assert!(compared >= 100, "{variable}: only {compared} comparisons");
        checked.push(format!("{variable}: {compared} comparisons"));
    }
    assert!(!checked.is_empty(), "no checkpoint tokenizer is available");
    eprintln!("checked {checked:?}");
}
