/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#![cfg(feature = "server")]

#[path = "support/templates.rs"]
mod template_fixtures;

use flare_helios::{
    frontend::{Asset, Frontend, FrontendAssets, TemplateEncoding},
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};

fn frontend(template: &str, encoding: TemplateEncoding) -> Frontend {
    frontend_with_merges(template,encoding,vec![])
}

fn frontend_with_merges(template: &str, encoding: TemplateEncoding, merges: Vec<(String,String)>) -> Frontend {
    frontend_with_defaults(template, encoding, merges, Map::new())
}

fn frontend_with_defaults(
    template: &str,
    encoding: TemplateEncoding,
    merges: Vec<(String,String)>,
    defaults: Map<String, Value>,
) -> Frontend {
    frontend_with_strip(template, encoding, merges, defaults, false)
}

/// `strip` sets lstrip and rstrip on `<|open|>` and `<|close|>`, so each also covers the
/// whitespace beside it.
fn frontend_with_strip(
    template: &str,
    encoding: TemplateEncoding,
    merges: Vec<(String,String)>,
    defaults: Map<String, Value>,
    strip: bool,
) -> Frontend {
    static SERIAL: AtomicUsize = AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "flare-message-boundaries-{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&directory).unwrap();
    let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let mut vocabulary = alphabet.into_iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), i as u32))
        .collect::<tokenizers::models::bpe::Vocab>();
    for (a,b) in &merges {vocabulary.insert(format!("{a}{b}"),vocabulary.len() as u32);}
    let mut tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocabulary, merges)
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, false),
    ));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer.with_post_processor(Some(tokenizers::processors::byte_level::ByteLevel::new(false,true,false)));
    tokenizer
        .add_special_tokens(
            ["<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>", "<|image|>",
             "<|user|>", "<|endoftext|>", "[gMASK]", "</think>", "<|image_pad|>", "<|video_pad|>",
             "/nothink"]
            .map(|token| {
                let stripped = strip && matches!(token, "<|open|>" | "<|close|>");
                tokenizers::AddedToken::from(token, !matches!(token, "<|image|>" | "</think>" | "/nothink"))
                    .lstrip(stripped)
                    .rstrip(stripped)
            }),
        )
        .unwrap();
    let asset = |name: &str, bytes: &[u8]| {
        let path = directory.join(name);
        std::fs::write(&path, bytes).unwrap();
        Asset {
            path,
            range: None,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        }
    };
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset(
                "tokenizer.json",
                tokenizer.to_string(false).unwrap().as_bytes(),
            ),
            tokens: Default::default(),
            chat_template: Some(asset("chat.jinja", template.as_bytes())),
        },
        defaults,
    )
    .unwrap()
    .with_template_encoding(encoding);
    std::fs::remove_dir_all(directory).unwrap();
    frontend
}

#[test]
fn bpe_crossings_and_unicode_offsets_use_the_complete_encoding() {
    for (source,merges,expected) in [
        ("a{% mark %}b{% mark %}c",vec![("a".into(),"b".into())],vec![1]),
        ("a {% mark %}b",vec![("Ġ".into(),"b".into())],vec![]),
        ("é{% mark %}x",vec![],vec![2]),
    ] {
        let frontend=frontend_with_merges(source,TemplateEncoding::Text,merges);
        let rendered=frontend.render_chat(&[json!({"role":"user","content":"unused"})],&Map::new()).unwrap();
        assert_eq!(rendered.message_boundaries,expected, "{source}");
        assert_eq!(rendered.token_ids,frontend.encode(&rendered.text,false).unwrap());
    }
}

#[test]
fn continuation_clips_marks_and_segments_before_mapping() {
    let frontend=frontend("{% for m in messages %}{{m.content}}{% mark %}|{% endfor %}",TemplateEncoding::Text);
    let rendered=frontend.render_chat(&[json!({"role":"user","content":"A"}),json!({"role":"assistant","content":"café  "})],
        json!({"continue_final_message":true}).as_object().unwrap()).unwrap();
    assert_eq!(rendered.text,"A|café  ");
    assert_eq!(rendered.message_boundaries,vec![1]);
    assert_eq!(rendered.token_ids,frontend.encode(&rendered.text,false).unwrap());
}

fn check_messages(frontend: &Frontend, messages: &[Value]) -> (Vec<u32>, Vec<usize>) {
    let options = json!({"add_generation_prompt":true});
    let rendered = frontend
        .render_chat(messages, options.as_object().unwrap())
        .unwrap();
    let boundaries = rendered.message_boundaries;
    for count in 1..=messages.len() {
        let prefix = frontend
            .render_chat(&messages[..count], &Map::new())
            .unwrap();
        assert!(
            boundaries.contains(&prefix.token_ids.len()),
            "message {count} has no checkpoint"
        );
        assert_eq!(
            rendered.token_ids[..prefix.token_ids.len()],
            prefix.token_ids
        );
    }
    assert!(boundaries.windows(2).all(|pair| pair[0] < pair[1]));
    (rendered.token_ids, boundaries)
}

#[test]
fn every_role_transition_and_empty_message_has_a_boundary() {
    let frontend = frontend(
        concat!(
            "{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% mark %}{% endfor %}",
            "{% if add_generation_prompt %}<assistant>{% endif %}"
        ),
        TemplateEncoding::Text,
    );
    let messages = json!([
        {"role":"system","content":"Instructions"},
        {"role":"assistant","content":"Opening reply"},
        {"role":"user","content":"Question"},
        {"role":"assistant","content":"Answer"},
        {"role":"tool","content":"Result"},
        {"role":"developer","content":"More instructions"},
        {"role":"user","content":""},
        {"role":"user","content":"Follow-up"}
    ]);
    let (tokens, boundaries) = check_messages(&frontend, messages.as_array().unwrap());
    assert_eq!(boundaries.len(), 8);
    let mut changed = messages.as_array().unwrap().clone();
    changed[2]["content"] = "Different question".into();
    let (other, other_boundaries) = check_messages(&frontend, &changed);
    assert_eq!(&boundaries[..2], &other_boundaries[..2]);
    assert_eq!(&tokens[..boundaries[1]], &other[..other_boundaries[1]]);
    let complete = frontend.render_chat(messages.as_array().unwrap(), &Map::new()).unwrap();
    assert!(complete.message_boundaries.iter().all(|&end| end < complete.token_ids.len()));

}

#[test]
fn gemma_templates_preserve_size_specific_thinking_prefixes() {
    let messages = [json!({"role":"user","content":"Hi"})];
    for (template, empty_thought) in [
        (&template_fixtures::template("gemma4", "chat_template.google")[..], true),
        (&template_fixtures::template("gemma4", "chat_template.google-e2b-e4b")[..], false),
    ] {
        let frontend = frontend(template, TemplateEncoding::Text);
        for thinking in [false, true] {
            let options = json!({"bos_token":"<bos>","add_generation_prompt":true,"enable_thinking":thinking});
            let rendered = frontend.render_chat(&messages, options.as_object().unwrap()).unwrap();
            let mut expected = "<bos>".to_owned();
            if thinking {
                expected.push_str("<|turn>system\n<|think|>\n<turn|>\n");
            }
            expected.push_str("<|turn>user\nHi<turn|>\n<|turn>model\n");
            if empty_thought && !thinking {
                expected.push_str("<|channel>thought\n<channel|>");
            }
            assert_eq!(rendered.text, expected);
            let prefix = expected.find("<|turn>model").unwrap();
            let ids = frontend.encode(&expected[..prefix], false).unwrap();
            assert_eq!(rendered.message_boundaries.last(), Some(&ids.len()));
        }
    }
}

#[test]
fn gemma_templates_preserve_reasoning_and_assistant_continuations() {
    let messages = json!([
        {"role":"user","content":"Question"},
        {"role":"assistant","reasoning_content":"Reason","content":"A"},
        {"role":"assistant","content":"B"}
    ]);
    for template in [
        &template_fixtures::template("gemma4", "chat_template.google")[..],
        &template_fixtures::template("gemma4", "chat_template.google-e2b-e4b")[..],
    ] {
        let frontend = frontend(template, TemplateEncoding::Text);
        let rendered = frontend.render_chat(messages.as_array().unwrap(), &Map::new()).unwrap();
        assert_eq!(rendered.text, concat!(
            "<|turn>user\nQuestion<turn|>\n<|turn>model\n",
            "<|channel>thought\nReason\n<channel|>AB<turn|>\n"
        ));
    }
}

#[test]
fn gemma_templates_checkpoint_preamble_messages_and_tool_rounds() {
    let messages = json!([
        {"role":"system","content":"Instructions"},
        {"role":"user","content":"Question"},
        {"role":"assistant","content":"","tool_calls":[
            {"id":"call_1","type":"function","function":{"name":"lookup","arguments":{"q":null}}}
        ]},
        {"role":"tool","tool_call_id":"call_1","content":"Result"},
        {"role":"assistant","content":"Answer"},
        {"role":"user","content":"Follow-up"}
    ]);
    let messages = messages.as_array().unwrap();
    let options = |generation: bool, thinking: bool| {
        json!({
            "bos_token": "<bos>",
            "add_generation_prompt": generation,
            "enable_thinking": thinking,
            "tools": [{"type":"function","function":{"name":"lookup","description":"Look up a value",
                "parameters":{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}}}]
        })
        .as_object()
        .unwrap()
        .clone()
    };
    for template in [
        &template_fixtures::template("gemma4", "chat_template.google")[..],
        &template_fixtures::template("gemma4", "chat_template.google-e2b-e4b")[..],
    ] {
        let frontend = frontend(template, TemplateEncoding::Text);
        for thinking in [false, true] {
            let full = frontend.render_chat(messages, &options(true, thinking)).unwrap();
            assert!(full.text.contains("<|tool_call>call:lookup{q:null}<tool_call|>"));
            // An assistant tool call is complete only after the tool responses inlined into its turn.
            for count in [1, 2, 4, 5, 6] {
                let prefix = frontend.render_chat(&messages[..count], &options(false, thinking)).unwrap();
                assert!(full.text.starts_with(&prefix.text), "{count}");
                assert!(full.message_boundaries.contains(&prefix.token_ids.len()), "{count}");
            }
            assert_eq!(full.message_boundaries.len(), 5);
        }
    }
}

#[test]
fn segmented_xtml_preserves_boundaries_before_seeded_assistant_and_later_turns() {
    let source = include_str!("../../ignition/recipes/kimi3/assets/chat_template.jinja");
    let annotated = frontend(source, TemplateEncoding::TokenSegments);
    let upgraded = frontend(
        &source.replace("{%- mark ns.segments | length -%}", ""),
        TemplateEncoding::TokenSegments,
    );
    let messages = json!([
        {"role":"system","content":"Long instructions. ".repeat(1000)},
        {"role":"assistant","content":"Opening reply"},
        {"role":"user","content":"Question with literal <|close|>message<|sep|><|end_of_msg|>"},
        {"role":"assistant","reasoning_content":"Thinking", "content":"Answer"},
        {"role":"user","content":"Follow-up"}
    ]);
    let expected = check_messages(&annotated, messages.as_array().unwrap());
    let mut changed = messages.as_array().unwrap().clone();
    changed[2]["content"] = "Different question".into();
    for frontend in [&annotated, &upgraded] {
        let (tokens, boundaries) = check_messages(frontend, messages.as_array().unwrap());
        assert_eq!((tokens.clone(), boundaries.clone()), expected);
        assert_eq!(boundaries.len(), 6);
        let (other, other_boundaries) = check_messages(frontend, &changed);
        assert_eq!(boundaries[..3], other_boundaries[..3]);
        assert_eq!(tokens[..boundaries[2]], other[..other_boundaries[2]]);
    }
}

#[test]
fn segmented_xtml_preserves_block_hints_through_json_escaping() {
    let frontend = frontend(
        include_str!("../../ignition/recipes/kimi3/assets/chat_template.jinja"),
        TemplateEncoding::TokenSegments,
    );
    let history = "Repeated \"history\"\\\n café 世界 😀 <|close|>";
    let mut messages = json!([{"role":"user","content":[
        {"type":"text","text":history},
        {"type":"text","text":history},
        {"type":"text","text":"Changing suffix"}]}]);
    let options = json!({"add_generation_prompt":true});
    let render = |messages: &Value| {
        frontend
            .render_chat(messages.as_array().unwrap(), options.as_object().unwrap())
            .unwrap()
    };
    let plain = render(&messages);
    messages[0]["_checkpoint_hints"] = json!([2]);
    let marked = render(&messages);
    assert_eq!(marked.text, plain.text);
    assert_eq!(marked.token_ids, plain.token_ids);
    let cut = marked.text.find("Changing suffix").unwrap();
    // The fixture tokenizer emits one token per byte, except structural specials.
    let suffix_tokens = frontend.encode(&marked.text[cut..], false).unwrap().len();
    let cut = marked.token_ids.len() - suffix_tokens;
    assert!(!plain.message_boundaries.contains(&cut));
    assert!(marked.message_boundaries.contains(&cut));
    assert_eq!(
        marked.message_boundaries.len(),
        plain.message_boundaries.len() + 1
    );
    messages[0]["content"][2]["text"] = json!("Different suffix");
    let changed = render(&messages);
    assert!(changed.token_ids.starts_with(&marked.token_ids[..cut]));
    assert!(changed.message_boundaries.contains(&cut));
}

#[test]
fn segmented_tool_results_keep_images_and_checkpoints_in_rendered_order() {
    let frontend = frontend(
        include_str!("../../ignition/recipes/kimi31/assets/chat_template.jinja"),
        TemplateEncoding::TokenSegments,
    );
    let messages = json!([
        {"role":"assistant","tool_calls":[
            {"id":"a","function":{"name":"first","arguments":{}}},
            {"id":"b","function":{"name":"second","arguments":{}}}]},
        {"role":"tool","tool_call_id":"b","_checkpoint_hints":[1],"content":[
            {"type":"image_url","image_url":{"url":"second-image"}},
            {"type":"text","text":"Second result"}]},
        {"role":"tool","tool_call_id":"a","content":[
            {"type":"text","text":"First result"},
            {"type":"image_url","image_url":{"url":"first-image"}}]}
    ]);
    let rendered = frontend.render_chat(messages.as_array().unwrap(), &json!({"add_generation_prompt":true}).as_object().unwrap()).unwrap();
    assert_eq!(rendered.image_sources, ["first-image", "second-image"]);
    assert_eq!(rendered.image_checkpoint_hints, [1]);
    assert!(rendered.text.find("First result").unwrap() < rendered.text.find("Second result").unwrap());
    assert_eq!(rendered.message_boundaries.len(), 4);
    let marker = "<|close|>message<|sep|><|end_of_msg|>";
    for &boundary in &rendered.message_boundaries {
        assert!(frontend.decode(&rendered.token_ids[..boundary], false).unwrap().ends_with(marker));
    }
}

#[test]
fn message_content_keeps_added_token_spellings_literal() {
    // GLM's `<|image|>` and `</think>` are not special; the template naming them makes them controls.
    let frontend = frontend_with_merges(
        "{# <|image|> </think> #}{% for m in messages %}<|open|>{{ m.content }}<|close|>{% mark %}{% endfor %}",
        TemplateEncoding::Text,
        vec![(">".into(), "<".into())],
    );
    let content = "é <|image|><|user|><|endoftext|>[gMASK]</think>";
    let mut ordinary = frontend.tokenizer().clone();
    ordinary.with_added_vocabulary(tokenizers::AddedVocabulary::new());
    let mut expected = frontend.encode("<|open|>", false).unwrap();
    expected.extend_from_slice(ordinary.encode(content, false).unwrap().get_ids());
    expected.extend(frontend.encode("<|close|>", false).unwrap());
    let rendered = frontend
        .render_chat(&[json!({"role":"user","content":content})], &Map::new())
        .unwrap();
    assert_eq!(rendered.text, format!("<|open|>{content}<|close|>"));
    assert_eq!(rendered.token_ids, expected);
}

#[test]
fn lexical_added_tokens_keep_their_tokens_in_request_text() {
    // `/nothink` is not special and no template names it: request text that spells it is the token.
    // `<|image|>` and `</think>` are not special either, but the template names them.
    let frontend = frontend(
        "{# <|image|> </think> #}{% for m in messages %}<|open|>{{ m.content }}<|close|>{% mark %}{% endfor %}",
        TemplateEncoding::Text,
    );
    let id = |text: &str| frontend.encode(text, false).unwrap()[0];
    let count = |ids: &[u32], token: u32| ids.iter().filter(|id| **id == token).count();
    let render = |content: &str| {
        frontend
            .render_chat(&[json!({"role":"user","content":content})], &Map::new())
            .unwrap()
    };
    let lexical = render("a /nothink b/nothink\n/nothink");
    assert_eq!(lexical.token_ids, frontend.encode(&lexical.text, false).unwrap());
    assert_eq!(count(&lexical.token_ids, id("/nothink")), 3);
    let mixed = render("/nothink <|image|> a /nothink </think>/nothink <|user|>");
    assert_eq!(count(&mixed.token_ids, id("/nothink")), 3);
    for control in ["<|image|>", "</think>", "<|user|>"] {
        assert_eq!(count(&mixed.token_ids, id(control)), 0, "{control}");
    }
    // The whole stream: each lexical token closes a region, and the spelled controls in the
    // regions between them are the ordinary encoding of their text.
    let mut ordinary_tokenizer = frontend.tokenizer().clone();
    ordinary_tokenizer.with_added_vocabulary(tokenizers::AddedVocabulary::new());
    let mut expected = frontend.encode("<|open|>", false).unwrap();
    for (ordinary, last) in [("", false), (" <|image|> a ", false), (" </think>", false), (" <|user|>", true)] {
        expected.extend(ordinary_tokenizer.encode(ordinary, false).unwrap().get_ids());
        expected.extend(if last { frontend.encode("<|close|>", false).unwrap() } else { vec![id("/nothink")] });
    }
    assert_eq!(mixed.token_ids, expected);
}

#[test]
fn request_whitespace_beside_a_stripping_control_is_not_a_spelling() {
    // A control with lstrip or rstrip also covers the whitespace beside it. Whitespace that the
    // request wrote there does not make the template's own control a spelling.
    let frontend = frontend_with_strip(
        "{% for m in messages %}<|open|>{{ m.content }}<|close|>{% mark %}{% endfor %}",
        TemplateEncoding::Text,
        vec![],
        Map::new(),
        true,
    );
    let id = |text: &str| frontend.encode(text, false).unwrap()[0];
    let count = |ids: &[u32], token: u32| ids.iter().filter(|id| **id == token).count();
    for content in ["hi", " hi", "hi ", "  hi  "] {
        let rendered = frontend
            .render_chat(&[json!({"role":"user","content":content})], &Map::new())
            .unwrap();
        assert_eq!(rendered.token_ids, frontend.encode(&rendered.text, false).unwrap(), "{content:?}");
        assert_eq!(count(&rendered.token_ids, id("<|open|>")), 1, "{content:?}");
        assert_eq!(count(&rendered.token_ids, id("<|close|>")), 1, "{content:?}");
    }
    // Spelling the controls still makes them ordinary text.
    let rendered = frontend
        .render_chat(&[json!({"role":"user","content":" <|open|> hi <|close|> "})], &Map::new())
        .unwrap();
    assert_eq!(count(&rendered.token_ids, id("<|open|>")), 1);
    assert_eq!(count(&rendered.token_ids, id("<|close|>")), 1);
}

#[test]
fn tokens_named_by_a_template_default_are_controls() {
    // `/nothink` appears in no template text, only in a nested default string.
    let source = "{% for m in messages %}<|open|>{{ m.content }}<|close|>{% mark %}{% endfor %}";
    let defaults = json!({"tags": {"markers": ["x", "end: /nothink"]}});
    let spelled = |defaults: Map<String, Value>| {
        let frontend = frontend_with_defaults(source, TemplateEncoding::Text, vec![], defaults);
        let nothink = frontend.encode("/nothink", false).unwrap()[0];
        let rendered = frontend
            .render_chat(&[json!({"role":"user","content":"a /nothink b"})], &Map::new())
            .unwrap();
        rendered.token_ids.iter().filter(|id| **id == nothink).count()
    };
    assert_eq!(spelled(Map::new()), 1);
    assert_eq!(spelled(defaults.as_object().unwrap().clone()), 0);
}

#[test]
fn image_expansion_preserves_literal_markers_and_real_placeholders() {
    use flare_helios::frontend::images::{ImageLimits, ImageProcessor, ProcessedImages, RgbImage};
    type Settings = fn(u32, u32) -> Value;
    let layouts: [(&str, Settings); 2] = [
        ("<|image_pad|>", |image_id, video_id| {
            json!({"qwen":{"min_pixels":4096,"max_pixels":4096,"position_side":48,
                "image_token_id":image_id,"video_token_id":video_id}})
        }),
        ("<|image|>", |_, _| {
            json!({"patch_merge":{"patch_size":14,"min_pixels":3136,"max_pixels":451584,
                "image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0],
                "image_token":"<|image|>","start_token":"<|open|>","end_token":"<|close|>"}})
        }),
    ];
    for (marker, settings) in layouts {
        let source = format!("{{{{ messages[0].content }}}}<|open|>{marker}<|close|>{{% mark %}}");
        let frontend = frontend(&source, TemplateEncoding::Text);
        let image_id = frontend.encode(marker, false).unwrap()[0];
        let video_id = frontend.encode("<|video_pad|>", false).unwrap()[0];
        let settings = settings(image_id, video_id);
        let processor = ImageProcessor::load(
            serde_json::from_value(settings).unwrap(),
            ImageLimits {
                patches_per_feature: 4,
                ..ImageLimits::default()
            },
        )
        .unwrap();
        let content = format!("literal {marker} <|video_pad|>");
        let rendered = frontend
            .render_chat(&[json!({"role":"user","content":content})], &Map::new())
            .unwrap();
        let mut ordinary = frontend.tokenizer().clone();
        ordinary.with_added_vocabulary(tokenizers::AddedVocabulary::new());
        let literal = ordinary.encode(content, false).unwrap();
        let images = processor
            .process(&[RgbImage::new(64, 64, vec![0; 64 * 64 * 3]).unwrap()])
            .unwrap();
        let expanded = processor
            .expand_prompt(&frontend, &rendered.token_ids, &images)
            .unwrap();
        assert!(expanded.token_ids.starts_with(literal.get_ids()));
        assert_eq!(
            expanded
                .token_ids
                .iter()
                .filter(|id| **id == image_id)
                .count(),
            images.geometry[0].tokens
        );
        assert!(!expanded.token_ids.contains(&video_id));
        assert_eq!(
            processor
                .expand_prompt(&frontend, literal.get_ids(), &ProcessedImages::default())
                .unwrap()
                .token_ids,
            literal.get_ids()
        );
    }
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT and pinned Qwen template"]
fn checkpoint_qwen_image_expansion_keeps_literal_markers() {
    use flare_helios::frontend::images::{ImageLimits, ImageProcessor, RgbImage};
    let snapshot = std::path::PathBuf::from(std::env::var_os("FLARE_TEST_SNAPSHOT").unwrap());
    let asset = |path: std::path::PathBuf| Asset {
        range: None,
        sha256: format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
        path,
    };
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset(snapshot.join("tokenizer.json")),
            tokens: Default::default(),
            chat_template: Some(asset(template_fixtures::path("qwen38", "chat_template"))),
        },
        Map::new(),
    )
    .unwrap();
    let image_id = frontend.encode("<|image_pad|>", false).unwrap()[0];
    let video_id = frontend.encode("<|video_pad|>", false).unwrap()[0];
    let processor = ImageProcessor::load(
        serde_json::from_value(json!({"qwen":{
            "min_pixels":4096,"max_pixels":4096,"position_side":48,
            "image_token_id":image_id,"video_token_id":video_id
        }}))
        .unwrap(),
        ImageLimits::default(),
    )
    .unwrap();
    let images = processor
        .process(&[RgbImage::new(64, 64, vec![0; 64 * 64 * 3]).unwrap()])
        .unwrap();
    let render = |text: &str| {
        frontend.render_chat(&[json!({"role":"user","content":[
        {"type":"text","text":text},{"type":"image_url","image_url":{"url":"fixture://image"}},
        {"type":"text","text":text}
    ]})], json!({"add_generation_prompt":true}).as_object().unwrap()).unwrap()
    };
    let baseline = render("ordinary");
    let literal = render("<|image_pad|> <|video_pad|> <|im_end|> </think>");
    assert_eq!(literal.image_sources.len(), 1);
    let expected = processor
        .expand_prompt(&frontend, &baseline.token_ids, &images)
        .unwrap();
    let actual = processor
        .expand_prompt(&frontend, &literal.token_ids, &images)
        .unwrap();
    for id in frontend
        .tokenizer()
        .get_added_vocabulary()
        .get_added_tokens_decoder()
        .keys()
    {
        assert_eq!(
            actual.token_ids.iter().filter(|t| *t == id).count(),
            expected.token_ids.iter().filter(|t| *t == id).count(),
            "{id}"
        );
    }
    assert_eq!(
        actual
            .token_ids
            .iter()
            .filter(|id| **id == image_id)
            .count(),
        images.geometry[0].tokens
    );
    assert_eq!(
        actual
            .mm_token_type_ids
            .iter()
            .filter(|kind| **kind == 1)
            .count(),
        images.geometry[0].tokens
    );
}
