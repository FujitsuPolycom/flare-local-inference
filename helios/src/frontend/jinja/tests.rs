/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use serde_json::json;

fn oracle(source: &str, context: &Json) -> std::result::Result<String, minijinja::Error> {
    let mut environment = minijinja::Environment::new();
    environment.set_trim_blocks(true);
    environment.set_lstrip_blocks(true);
    environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    environment.add_filter("tojson", super::super::json::tojson as fn(_, _) -> _);
    environment.add_test("iterable", |v: &minijinja::Value| {
        !v.is_none() && v.try_iter().is_ok()
    });
    environment.add_test("sequence", |v: &minijinja::Value| {
        use minijinja::value::ValueKind;
        matches!(v.kind(), ValueKind::Seq | ValueKind::Map | ValueKind::String)
    });
    environment.set_formatter(|out, state, value| {
        if value.kind() == minijinja::value::ValueKind::Number && !value.is_integer() {
            if let Some(v) = f64::try_from(value.clone()).ok().filter(|v| v.is_finite()) {
                return out
                    .write_str(&super::super::json::python_float(v))
                    .map_err(|_| minijinja::Error::from(minijinja::ErrorKind::WriteFailure));
            }
        }
        minijinja::escape_formatter(out, state, value)
    });
    environment.add_function(
        "raise_exception",
        |v: minijinja::Value| -> std::result::Result<(), minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                v.to_string(),
            ))
        },
    );
    environment.add_template("chat", source)?;
    environment.get_template("chat")?.render(context)
}

fn compare(source: &str, context: Json) {
    let expected = oracle(source, &context).map_err(|e| e.to_string());
    let actual = CompiledTemplate::compile(source)
        .and_then(|t| t.render(context.as_object().unwrap(), &[]))
        .map(|r| r.text);
    match (expected, actual) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "{source}"),
        (Err(_), Err(_)) => {}
        (a, b) => panic!("{source}\noracle: {a:?}\nowned: {b:?}"),
    }
}

#[test]
fn expressions_and_python_methods_match_minijinja() {
    let context = json!({"s":"  café 👋  ","a":[1,2,3,4],"m":{"z":true,"a":null},"d":{"b":2,"B":3,"a":1},"f":[1e-6,-0.0,1e16,2.5,3.0],"none":null});
    for expr in [
        "1+2*3",
        "(1+2)*3",
        "2**3**2",
        "not 1 == 1",
        "1 if true else 2",
        "1 if false else 2 if true else 3",
        "s|trim",
        "s.strip()",
        "s.lstrip()",
        "s.rstrip()",
        "s.split()",
        "s.split(' ', 2)",
        "s.startswith(' ')",
        "s[-4:]",
        "s[::-1]",
        "a[::-1]",
        "a[1:3]",
        "a[-100:100:2]",
        "a|length",
        "a|join(':')",
        "m.items()",
        "m|items",
        "m.get('missing', 4)",
        "m.z",
        "m['a']",
        "m|tojson",
        "[1, 2]+[3]",
        "missing|default('x')",
        "none|default('x')",
        "none|default('x',true)",
        "missing is defined",
        "missing is iterable",
        "'x' in ['x','y']",
        "'x' not in []",
        "'z' in m",
        "false and missing()",
        "true or missing()",
        "m|tojson(indent=2)",
        "m|tojson(ensure_ascii=true,sort_keys=true,separators=(',',':'))",
        "18446744073709551615|tojson",
        "18446744073709551615",
        "9007199254740993 == 9007199254740992",
        "f[0]",
        "f[1]",
        "f[2]",
        "f[3]",
        "f[4]",
        "m is sequence",
        "s is sequence",
        "none is sequence",
        "d|dictsort",
        "d|dictsort(case_sensitive=true)",
        "d|dictsort(by='value',reverse=true)",
        "['a', 'A', 'b', 'a']|unique|list",
        "['a', 'A', 'b', 'a']|unique(case_sensitive=true)|list",
        "['a', 'A', 'b', 'a']|unique(case_sensitive=true)|list|length",
        "[3, 1, 3, 2, 1]|unique|list",
        "[1, 1]|unique(true)|list",
        "[{'id':'a'}, {'id':'A'}, {'id':'b'}]|unique(attribute='id')|list",
        "['b', 'B', 'a', 'A']|sort(case_sensitive=true)",
        "[{'id':'a'}, {'id':'A'}]|sort(attribute='id',case_sensitive=true)",
        "'a' \"b\" 'c'|upper",
        "['a' 'b', 'c']",
    ] {
        compare(&format!("{{{{ {expr} }}}}"), context.clone());
    }
}

#[test]
fn float_text_conversions_use_python_spelling() {
    let template = CompiledTemplate::compile("{{ f|join(' ') }}|{{ f[2]|string }}").unwrap();
    let context = json!({"f":[1e-6,-0.0,1e16,2.5,3.0]});
    assert_eq!(
        template.render(context.as_object().unwrap(), &[]).unwrap().text,
        "1e-06 -0.0 1e+16 2.5 3.0|1e+16"
    );
}

#[test]
fn scopes_macros_captures_and_control_flow_match_minijinja() {
    for source in [
        "{% set x=7 %}{% for i in range(3) %}{{ x }}{% set x=i %}{{ x }}{% endfor %}{{ x }}",
        "{% set ns=namespace(x=[]) %}{% for i in range(3) %}{% set ns.x=ns.x+[i] %}{% endfor %}{{ ns.x }}",
        "{% set ns=namespace(x=[1]) %}{% set alias=ns.x %}{% set ns.x=ns.x+[2] %}{{alias}}/{{ns.x}}",
        "{% for a,b in [('a',1),('b',2)] %}{{a}}:{{b}} {{loop.index}} {{loop.previtem}} {{loop.nextitem}}{% endfor %}",
        "{% for x in [] %}bad{% else %}empty{% endfor %}",
        "{% for x in range(6) if x % 2 == 0 %}{{x}}:{{loop.index}}/{{loop.length}}{% else %}empty{% endfor %}",
        "{% for x in range(4) %}{% if x==1 %}{% continue %}{% endif %}{% if x==3 %}{% break %}{% endif %}{{x}}{% else %}bad{% endfor %}",
        "{% macro f(x, y=3) %}{{x+y}}{% endmacro %}{{f(4)}} {{f(y=2,x=1)}}",
        "{% macro f(n) %}{% if n>0 %}{{n}}{{f(n-1)}}{% endif %}{% endmacro %}{{f(5)}}",
        "{% set x=3 %}{% macro f() %}{{x}}{% set x=4 %}{{x}}{% endmacro %}{{f()}}{{x}}",
        "{% set x=1 %}{% macro f() %}{{x}}{% endmacro %}{% set x=2 %}{{f()}}",
        "{% set x=1 %}{% macro f() %}{{x}}{% endmacro %}{% for x in [2,3] %}{{f()}}{% endfor %}",
        "{% macro f() %}{{g()}}{% endmacro %}{% macro g() %}hello{% endmacro %}{{f()}}",
        "{% set body %}hello {{3}}{% endset %}{{body|upper}}",
        "{% filter upper %}hello {{3}}{% endfilter %}",
        "a\n  {% if true %}\nb\n  {% endif %}\nc\n",
        "a\n {{- 'b' -}} \nc\n",
        "a {# ignored #} b\n",
        "a\r\nb\r\n",
    ] {
        compare(source, json!({}));
    }
}

#[test]
fn marks_are_zero_output_and_cannot_escape_captures() {
    let template = CompiledTemplate::compile(
        "é{% mark %}{% for s in ['a','👋'] %}{{s}}{% mark %}{% endfor %}{% mark 3 %}",
    )
    .unwrap();
    assert_eq!(
        template.render(&Map::new(), &[]).unwrap(),
        RenderedTemplate {
            text: "éa👋".into(),
            ordinary: vec![],
            marks: vec![
                Mark::Byte(2),
                Mark::Byte(3),
                Mark::Byte(7),
                Mark::Position(3)
            ]
        }
    );
    assert_eq!(template.annotation_spans().len(), 3);
    assert!(std::mem::size_of::<Instruction>() <= 16);
    for source in [
        "{% macro f() %}{% mark %}{% endmacro %}",
        "{% set x %}{% mark 1 %}{% endset %}",
        "{% filter trim %}{% mark %}{% endfilter %}",
    ] {
        assert!(
            CompiledTemplate::compile(source)
                .err()
                .unwrap()
                .message
                .contains("root output")
        );
    }
    for position in ["-1", "1.5", "true", "'2'", "missing"] {
        assert!(
            CompiledTemplate::compile(&format!("{{% mark {position} %}}"))
                .unwrap()
                .render(&Map::new(), &[])
                .is_err()
        );
    }
}

#[test]
fn request_state_is_isolated_and_template_is_shareable() {
    let template=CompiledTemplate::compile("{% set ns=namespace(n=0) %}{% for x in messages %}{% set ns.n=ns.n+1 %}{{x}}{% mark %}{% endfor %}{{ns.n}}").unwrap();
    std::thread::scope(|scope| {
        for n in 0..8 {
            let template = &template;
            scope.spawn(move || {
                for _ in 0..32 {
                    let context = json!({"messages":vec!["x";n]});
                    let result = template.render(context.as_object().unwrap(), &[]).unwrap();
                    assert_eq!(result.text, format!("{}{n}", "x".repeat(n)));
                    assert_eq!(result.marks.len(), n);
                }
            });
        }
    });
}

#[test]
fn malformed_inputs_do_not_panic() {
    let alphabet = b"{}%#'\"()[]:,.=+-*/ abc012\n";
    let mut state = 17u64;
    for n in 0..10_000 {
        let source: String = (0..n % 100)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                alphabet[(state >> 32) as usize % alphabet.len()] as char
            })
            .collect();
        let _ = CompiledTemplate::compile(&source);
        let _ = CompiledTemplate::compile(&format!("{{{{{source}}}}}"));
    }
}

#[test]
fn recipe_templates_match_on_messages_tools_images_and_options() {
    let sources = [
        (
            "qwen",
            &crate::template_fixtures::template("qwen38", "chat_template")[..],
        ),
        (
            "step",
            &crate::template_fixtures::template("step5", "chat_template")[..],
        ),
        (
            "glm",
            &crate::template_fixtures::template("glm53", "chat_template")[..],
        ),
        (
            "mimo",
            &crate::template_fixtures::template("mimo_v2", "chat_template")[..],
        ),
        (
            "nemotron",
            &crate::template_fixtures::template("nemotron_h", "chat_template")[..],
        ),
        (
            "kimi",
            include_str!("../../../../ignition/recipes/kimi3/assets/chat_template.jinja"),
        ),
        (
            "gemma",
            &crate::template_fixtures::template("gemma4", "chat_template.google")[..],
        ),
        (
            "gemma-e2b-e4b",
            &crate::template_fixtures::template("gemma4", "chat_template.google-e2b-e4b")[..],
        ),
    ];
    let user = json!({"role":"user","content":"  café 世界 👋 a\u{301}\r\n<|im_end|>  "});
    let tools = json!([{"type":"function","function":{"name":"lookup","description":" Find <items> & café. ","parameters":{"type":"object","properties":{"query":{"type":"string","description":"A query"},"count":{"type":"integer","enum":[1,2]}},"required":["query"],"additionalProperties":false}}}]);
    let calls = json!({"role":"assistant","content":"","reasoning_content":"Need two sources.","tool_calls":[
        {"id":"b","type":"function","function":{"name":"lookup","arguments":{"query":"café 世界","count":2,"options":{"x":true,"y":null,"z":[1e-6,-0.0,1e16]}}}},
        {"id":"a","type":"function","function":{"name":"lookup","arguments":{"query":"a\nb"}}}
    ]});
    let mut histories = vec![
        json!([]),
        json!([user]),
        json!([{"role":"system","content":"Be precise."},user]),
        json!([{"role":"system","content":" "},user]),
        json!([user,{"role":"assistant","content":null},user]),
        json!([user,{"role":"assistant","content":"Answer  ","reasoning_content":" Think "},user]),
        json!([user,{"role":"assistant","content":"<think>\nThink\n</think>\nAnswer"},user]),
        json!([user,{"role":"system","content":"Inline"},user]),
        json!([user,calls,{"role":"tool","tool_call_id":"a","name":"lookup","content":"second"},{"role":"tool","tool_call_id":"b","name":"lookup","content":"first"}]),
        json!([user,calls,{"role":"tool","name":"lookup","content":[{"id":"a","output":"second"},{"id":"b","output":"first"}]}]),
    ];
    for content in [
        Json::Null,
        json!(42),
        json!({"text":"map content"}),
        json!([{"type":"text","text":"A"},{"type":"text","text":"B"}]),
        json!([{"type":"image"},{"type":"text","text":"Describe <|open|>"},{"type":"image_url","image_url":{"url":"fixture://image"}}]),
        json!([{"type":"audio"}]),
        json!(["text", "blocks"]),
    ] {
        histories.push(json!([{"role":"user","content":content}]));
    }
    let options = [
        json!({}),
        json!({"tools":tools}),
        json!({"enable_thinking":false}),
        json!({"enable_thinking":true}),
        json!({"preserve_thinking":true}),
        json!({"preserve_thinking":false,"truncate_history_thinking":true,"clear_thinking":true}),
        json!({"reasoning_effort":"low"}),
        json!({"reasoning_effort":"max"}),
        json!({"reasoning_effort":"invalid"}),
        json!({"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"x":{"type":"integer"}}}}}}),
        json!({"tool_choice":"required","tools":tools}),
        json!({"tool_choice":"none"}),
        json!({"add_generation_prompt":false}),
    ];
    for (name, source) in sources {
        let compiled = CompiledTemplate::compile(source).unwrap();
        let mut unmarked = source.to_owned();
        for span in compiled.annotation_spans().iter().rev() {
            unmarked.replace_range(span.clone(), "");
        }
        for (h, history) in histories.iter().enumerate() {
            for (o, option) in options.iter().enumerate() {
                let mut context = json!({"messages":history,"tools":null,"documents":null,"bos_token":"<s>","eos_token":"</s>","add_generation_prompt":true});
                context
                    .as_object_mut()
                    .unwrap()
                    .extend(option.as_object().unwrap().clone());
                let expected = oracle(&unmarked, &context);
                let actual = compiled.render(context.as_object().unwrap(), &[]);
                match (expected, actual) {
                    (Ok(expected), Ok(actual)) => {
                        assert_eq!(actual.text, expected, "{name}: history {h}, options {o}")
                    }
                    (Err(_), Err(_)) => {}
                    (expected, actual) => panic!(
                        "{name}: history {h}, options {o}\noracle: {expected:?}\nowned: {actual:?}"
                    ),
                }
            }
        }
    }
}

#[test]
fn block_hint_marks_follow_string_output() {
    let context = json!({"messages":[{"content":[{"text":"history"},{"text":"action"}]}]});
    for (source, marked) in [
        (
            "{% for block in messages[0].content %}{{ block.text }}{% endfor %}",
            true,
        ),
        (
            "{% macro body() %}{{ messages[0].content[0].text }}{% endmacro %}{{ body() }}action",
            true,
        ),
        (
            "{% set body %}{{ messages[0].content[0].text }}{% endset %}{{ body }}action",
            true,
        ),
        (
            "{% macro body() %}{{ messages[0].content[0].text }}{% endmacro %}{% set text = body() %}{{ text }}action",
            true,
        ),
        (
            "{{ messages[0].content[0].text + messages[0].content[1].text }}",
            true,
        ),
        (
            "{{ messages[0].content[0].text|upper|lower }}action",
            false,
        ),
    ] {
        let rendered = CompiledTemplate::compile(source)
            .unwrap()
            .render(context.as_object().unwrap(), &[(0, 1)])
            .unwrap();
        assert_eq!(rendered.text, "historyaction");
        assert_eq!(
            rendered.marks,
            if marked { vec![Mark::Hint(7)] } else { vec![] }
        );
    }
}

#[test]
fn block_hints_preserve_byte_offsets_through_captures_trim_and_concatenation() {
    let context = json!({"messages":[{"content":[{"text":" \té你好  "},{"text":" tail "}]}]});
    let template = CompiledTemplate::compile(concat!(
        "{% macro body() %}{% for part in messages[0].content %}{{ part.text }}{% endfor %}{% endmacro %}",
        "{% set text = body()|trim %}{{ '<' + text + '>' }}",
    ))
    .unwrap();
    let rendered = template
        .render(context.as_object().unwrap(), &[(0, 1)])
        .unwrap();
    assert_eq!(rendered.text, "<é你好   tail>");
    assert_eq!(rendered.marks, [Mark::Hint("<é你好  ".len())]);
    let identical = json!({"messages":[{"content":[{"text":"same"},{"text":"same"}]}]});
    let rendered = template
        .render(identical.as_object().unwrap(), &[(0, 2)])
        .unwrap();
    assert_eq!(rendered.text, "<samesame>");
    assert_eq!(rendered.marks, [Mark::Hint("<samesame".len())]);
}

#[test]
fn checkpoint_hints_are_dropped_by_operations_that_return_part_of_a_string() {
    let hints = |content: &str, expression: &str| {
        let context = json!({"messages":[{"content":content}]});
        CompiledTemplate::compile(&format!("{{{{ {expression} }}}}"))
            .unwrap()
            .render(context.as_object().unwrap(), &[(0, 1)])
            .unwrap()
            .marks
            .iter()
            .filter(|mark| matches!(mark, Mark::Hint(_)))
            .count()
    };
    for (expression, expected) in [
        ("messages[0].content", 1),
        ("messages[0].content|trim", 1),
        ("messages[0].content.split('</think>')[0]", 0),
        ("messages[0].content.split('</think>')[-1]", 0),
        ("messages[0].content.rsplit('</think>', 1)[1]", 0),
        ("messages[0].content.split()[0]", 0),
        ("messages[0].content.strip()", 0),
        ("messages[0].content.lstrip()", 0),
        ("messages[0].content.rstrip()", 0),
        ("messages[0].content[0]", 0),
        ("messages[0].content[2:]", 0),
        ("messages[0].content|list|join", 0),
        ("[messages[0].content]|join(',')", 0),
    ] {
        let content = "reasoning</think>\n answer \n";
        assert_eq!(hints(content, expression), expected, "{expression}");
    }
    // A result that is the whole message string is not the hinted string.
    for expression in ["strip()", "split()[0]", "split(',')[0]", "[0]", "[0:]"] {
        let expression = format!("messages[0].content.{expression}").replace(".[", "[");
        assert_eq!(hints("a", &expression), 0, "{expression}");
    }
}

#[test]
fn recipe_defaults_are_controls_and_request_values_are_ordinary() {
    let context = json!({
        "bos_token": "<s>",
        "extras": ["<a>", {"key": "<b>"}],
        "overridden": "<c>",
        "messages": [{"content": "<d>"}],
    });
    let rendered = CompiledTemplate::compile(
        "{{ bos_token }}{{ extras[0] }}{% for key, text in extras[1].items() %}{{ text }}{% endfor %}\
         {{ overridden }}{{ messages[0].content }}",
    )
    .unwrap()
    .render_with_defaults(context.as_object().unwrap(), &[], &["bos_token", "extras"])
    .unwrap();
    assert_eq!(rendered.text, "<s><a><b><c><d>");
    assert_eq!(rendered.ordinary, [9..15]);
}

#[test]
fn qwen_content_macro_keeps_hints_only_when_its_result_is_emitted() {
    let context = json!({"messages":[{"role":"user","content":[
        {"type":"text","text":"  Earlier history.\n"},
        {"type":"text","text":"Newest action."}]}],
        "enable_thinking":false,"add_generation_prompt":true});
    let template = CompiledTemplate::compile(&crate::template_fixtures::template("qwen38", "chat_template")[..])
    .unwrap();
    let plain = template.render(context.as_object().unwrap(), &[]).unwrap();
    let marked = template
        .render(context.as_object().unwrap(), &[(0, 1)])
        .unwrap();
    assert_eq!(marked.text, plain.text);
    let hints: Vec<_> = marked
        .marks
        .into_iter()
        .filter(|mark| matches!(mark, Mark::Hint(_)))
        .collect();
    assert_eq!(hints, [Mark::Hint(plain.text.find("Newest action.").unwrap())]);
}

#[test]
fn huggingface_generation_annotations_preserve_text_and_message_marks() {
    let rendered = CompiledTemplate::compile("A{% generation %}{{ text }}{% endgeneration %}{% mark %}B")
        .unwrap().render(json!({"text":"café 世界"}).as_object().unwrap(), &[]).unwrap();
    assert_eq!(rendered.text, "Acafé 世界B");
    assert_eq!(rendered.marks, vec![super::Mark::Byte("Acafé 世界".len())]);
}

#[test]
fn boolean_identity_tests_accept_bare_and_parenthesized_arguments() {
    compare("{{ a is sameas false }} {{ b is not sameas false }} {{ a is sameas(false) }}",
        json!({"a":false,"b":true}));
}

#[test]
fn strftime_now_is_defined_and_uses_local_calendar_time() {
    let source = "{% if strftime_now is defined %}{{ strftime_now('%Y-%m-%d') }}{% endif %}";
    let rendered = CompiledTemplate::compile(source).unwrap().render(&serde_json::Map::new(), &[]).unwrap();
    assert_eq!(rendered.text.len(), 10);
    assert_eq!(&rendered.text[4..5], "-");
    assert_eq!(&rendered.text[7..8], "-");
}

#[test]
fn literal_replacement_maps_hints_and_drops_cuts_inside_replaced_spans() {
    for (first, second, expected) in [("a", "bc", None), ("abc", "tail", Some(1)), ("café", "abc", Some(5))] {
        let context=json!({"messages":[{"content":[{"type":"text","text":first},{"type":"text","text":second}]}]});
        let template=CompiledTemplate::compile("{% set body %}{{ messages[0].content[0].text }}{{ messages[0].content[1].text }}{% endset %}{{ body|replace('abc','X') }}").unwrap();
        let rendered=template.render(context.as_object().unwrap(), &[(0,1)]).unwrap();
        assert_eq!(rendered.text,format!("{first}{second}").replace("abc","X"));
        assert_eq!(rendered.marks,expected.map(super::Mark::Hint).into_iter().collect::<Vec<_>>());
    }
}

#[test]
fn json_object_members_preserves_literals_order_and_duplicate_keys() {
    let source = r#" {"z":1e2,"a":[1,  2],"z":-0,"obj":{ "b" : false },"s":"café\n\u4e16","nil":null} "#;
    let template = CompiledTemplate::compile("{{ json_object_members(source) | tojson(ensure_ascii=false) }}").unwrap();
    let context = json!({"source": source});
    let rendered = template.render(context.as_object().unwrap(), &[]).unwrap();
    let members: Json = serde_json::from_str(&rendered.text).unwrap();
    assert_eq!(members, json!([
        ["z",100.0,"1e2"], ["a",[1,2],"[1,  2]"], ["z",-0.0,"-0"],
        ["obj",{"b":false},"{ \"b\" : false }"], ["s","café\n世",r#""café\n\u4e16""#], ["nil",null,"null"]
    ]));
    for source in ["[]", "null", "1", "{", "{\"x\":1} trailing", "{\"x\":}"] {
        let context = json!({"source":source});
        assert!(template.render(context.as_object().unwrap(), &[]).is_err(), "{source}");
    }
    assert!(CompiledTemplate::compile("{{ json_object_members([]) }}").unwrap().render(&Map::new(), &[]).is_err());
}

/// Every occurrence of `marker` in the rendered text must lie in request text.
fn assert_ordinary(rendered: &RenderedTemplate, marker: &str, label: impl std::fmt::Display) {
    for (start, _) in rendered.text.match_indices(marker) {
        assert!(
            rendered.ordinary.iter().any(|r| start < r.end && start + marker.len() > r.start),
            "{label}, byte={start}: {:?}",
            &rendered.text[start.saturating_sub(30)..start + marker.len()]
        );
    }
}

#[test]
fn request_keys_and_builtin_results_keep_provenance() {
    let marker = "<|image|>";
    let context = json!({"key":marker,"object":{marker:"x"},"text":"{\"<|image|>\": {\"<|image|>\": \"x\"}}"});
    for expression in [
        "(object|items)[0][0]",
        "(object|dictsort)[0][0]",
        "object.keys()[0]",
        "object.items()[0][0]",
        "(text|fromjson|items)[0][0]",
        "((text|fromjson)[key]|items)[0][0]",
        "{object: 1}",
        "{object: 1}|tojson",
        "({object: 1}|items|first)[0]",
        "{[key]: 1}",
        "{key: none}|items",
        "{key: none}|list",
        "{key: none}|tojson",
        "namespace(object)",
        "[key, '<|image|>']|trim",
        "[key, '<|image|>']|string",
        "json_object_members(text)",
    ] {
        let rendered = CompiledTemplate::compile(&format!("{{{{ {expression} }}}}"))
            .unwrap()
            .render(context.as_object().unwrap(), &[])
            .unwrap();
        assert!(rendered.text.contains(marker), "{expression}");
        assert_ordinary(&rendered, marker, expression);
    }
}

#[test]
fn sliced_request_text_coalesces_ordinary_ranges() {
    let context = json!({"body":"x".repeat(1000)});
    let rendered = CompiledTemplate::compile("{{ body[:0] }}{{ body[1:] }}")
        .unwrap().render(context.as_object().unwrap(), &[]).unwrap();
    assert_eq!(rendered.ordinary, vec![0..999]);
}

/// A filter on the exact list in `filter` must return its operands' characters unchanged; each
/// other filter marks its whole result ordinary. A filter added to `filter_value` must be placed.
#[test]
fn every_filter_is_classified_for_provenance() {
    let source = include_str!("vm.rs");
    let end = source.find("fn method<").unwrap();
    let exact = &source[source.find("fn filter<").unwrap()..source.find("fn filter_value").unwrap()];
    let body = &source[source.find("fn filter_value").unwrap()..end];
    let transforms = ["length", "count", "capitalize", "lower", "upper", "replace", "tojson", "fromjson", "int"];
    for arm in body.lines().filter(|line| line.starts_with("        \"") && line.contains("=>")) {
        for name in arm.split("=>").next().unwrap().split('|').map(|n| n.trim().trim_matches('"')) {
            assert!(transforms.contains(&name) || exact.contains(&format!("\"{name}\"")), "unclassified filter {name}");
        }
    }
}

#[test]
fn exact_string_operations_preserve_template_controls() {
    let context = json!({"body":"  é<|image|>  "});
    for expression in [
        "['<control>', body|trim, '</control>']|join",
        "['<control>', body|trim, '<control>', '</control>']|unique|join",
        "['<control>', body|trim, '</control>']|sort|join",
        "('<control>' + body + '</control>').strip()",
        "('<control>' + body + '</control>').split('!')[0]",
        "('<control>' + body + '</control>')[:]",
        "dict(body=body, control='<control>').get('control') + body",
    ] {
        let rendered = CompiledTemplate::compile(&format!("{{{{ {expression} }}}}"))
            .unwrap()
            .render(context.as_object().unwrap(), &[])
            .unwrap();
        let marker = rendered.text.find("<|image|>").unwrap();
        assert!(
            rendered.ordinary.iter().any(|r| r.contains(&marker)),
            "{expression}"
        );
        assert!(
            !rendered
                .ordinary
                .iter()
                .any(|r| r.start < "<control>".len()),
            "{expression}: {rendered:?}"
        );
    }
}

#[test]
fn prepared_templates_mark_request_strings_and_object_keys_ordinary() {
    let marker = "<|image|>";
    fn mutations(value: &Json, marker: &str) -> Vec<Json> {
        let mut cases = Vec::new();
        match value {
            Json::String(_) => cases.push(json!(marker)),
            Json::Array(values) => {
                for (index, value) in values.iter().enumerate() {
                    for replacement in mutations(value, marker) {
                        let mut copy = values.clone();
                        copy[index] = replacement;
                        cases.push(Json::Array(copy));
                    }
                }
            }
            Json::Object(values) => {
                for (key, value) in values {
                    let mut copy = values.clone();
                    copy.remove(key);
                    copy.insert(marker.into(), value.clone());
                    cases.push(Json::Object(copy));
                    for replacement in mutations(value, marker) {
                        let mut copy = values.clone();
                        copy.insert(key.clone(), replacement);
                        cases.push(Json::Object(copy));
                    }
                }
            }
            _ => {}
        }
        cases
    }
    for (recipe, name, _) in PREPARED_TEMPLATES {
        let compiled =
            CompiledTemplate::compile(&crate::template_fixtures::template(recipe, name)).unwrap();
        for serialized in [false, true] {
            if serialized && !matches!(recipe, "step5" | "mimo_v2") {
                continue;
            }
            let context = prepared_context(marker, serialized);
            // Mutate structural fields too: invalid requests may fail template validation.
            for mutated in mutations(&context, marker) {
                if let Ok(rendered) = compiled.render(mutated.as_object().unwrap(), &[]) {
                    assert_ordinary(&rendered, marker, format!("{recipe}/{name}, context={mutated}"));
                }
            }
            let rendered = compiled.render(context.as_object().unwrap(), &[]).unwrap();
            assert_ordinary(&rendered, marker, format!("{recipe}/{name}, serialized={serialized}"));
            let count = rendered.text.matches(marker).count();
            assert!(count >= 5, "{recipe}/{name}: only {count} markers emitted");
        }
    }
}

/// Request text that spells no control leaves every control the template writes outside the
/// ordinary ranges, so an ordinary request tokenizes as it does without provenance. A builtin that
/// mixes template controls with request text and marks the whole result ordinary fails here.
#[test]
fn prepared_templates_keep_their_own_controls_for_ordinary_requests() {
    for (recipe, name, controls) in PREPARED_TEMPLATES {
        let compiled =
            CompiledTemplate::compile(&crate::template_fixtures::template(recipe, name)).unwrap();
        for serialized in [false, true] {
            if serialized && !matches!(recipe, "step5" | "mimo_v2") {
                continue;
            }
            let context = prepared_context("plain", serialized);
            let rendered = compiled.render(context.as_object().unwrap(), &[]).unwrap();
            assert!(!rendered.ordinary.is_empty(), "{recipe}/{name}: no request text recorded");
            let mut written = 0;
            for control in controls {
                for (at, _) in rendered.text.match_indices(control) {
                    written += 1;
                    assert!(
                        !crate::frontend::overlaps_ordinary(&rendered.ordinary, at, at + control.len()),
                        "{recipe}/{name}: {control} at {at} is ordinary in {rendered:?}"
                    );
                }
            }
            assert!(written >= 4, "{recipe}/{name}: only {written} controls written");
        }
    }
}

/// Prepared templates, with the control spellings each one writes for the roles, reasoning and
/// tool calls of `prepared_context` (the markers of the recipe contracts, and the other added
/// tokens the template emits).
const PREPARED_TEMPLATES: [(&str, &str, &[&str]); 9] = [
    ("glm53", "chat_template", &["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]),
    ("qwen38", "chat_template", &["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]),
    ("step5", "chat_template", &["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]),
    ("mimo_v2", "chat_template", &["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]),
    ("mimo_v2", "chat_template.flash-rl", &["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]),
    ("minimax_m3", "chat_template", &["]~b]", "]~!b[", "[e~[", "<mm:think>", "</mm:think>", "<tool_call>", "</tool_call>"]),
    ("nemotron_h", "chat_template", &["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]),
    ("gemma4", "chat_template.google", &["<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>"]),
    ("gemma4", "chat_template.google-e2b-e4b", &["<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>"]),
];

/// A conversation, tools and documents in which every request string is `marker`.
fn prepared_context(marker: &str, serialized: bool) -> Json {
    let tools = json!([{"type":"function","function":{"name":marker,"description":marker,
        "parameters":{"type":"object","properties":{marker:{"type":"string","description":marker,"enum":[marker]}},"required":[marker]}}}]);
    let mut arguments = json!({marker:marker,"nested":{marker:null}});
    if serialized {
        arguments = Json::String(arguments.to_string());
    }
    json!({"messages":[
        {"role":"system","content":marker},
        {"role":"user","content":marker},
        {"role":"assistant","content":marker,"reasoning_content":marker,"tool_calls":[
            {"id":marker,"type":"function","function":{"name":marker,"arguments":arguments}}]},
        {"role":"tool","tool_call_id":marker,"name":marker,"content":marker},
        {"role":"user","content":[{"type":"text","text":marker}]}],
        "tools":tools,"documents":[{"title":marker,"text":marker}],
        "bos_token":"<s>","eos_token":"</s>","add_generation_prompt":true,"preserve_thinking":true})
}
