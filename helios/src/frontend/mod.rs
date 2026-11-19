/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Native tokenization and rendering over verified Hugging Face assets.

mod deepseek;
mod boundaries;
mod tml;
mod templates;
#[cfg(feature = "images")]
pub mod images;
mod json;
mod jinja;

mod vocabulary;

use std::fs;
use std::path::PathBuf;

use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::{Error, Failure, FrontendError, Result};

pub(crate) fn validate_chat_template(source: &str) -> Result<()> {
    templates::compile(source).map(|_| ()).map_err(|error| {
        Error::frontend(FrontendError::Template(Failure::caused_by(error.to_string(), error)))
    })
}

pub struct Asset {
    pub path: PathBuf,
    pub sha256: String,
    pub range: Option<(u64, usize)>,
}

impl Asset {
    pub(crate) fn read(&self) -> Result<Vec<u8>> {
        if !self.path.is_absolute() {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                format!(
                    "frontend asset path must be absolute: {}",
                    self.path.display()
                ),
            ))));
        }
        if self.sha256.len() != 64 || !self.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                format!("invalid expected SHA256 for {}", self.path.display()),
            ))));
        }
        let bytes = (|| -> std::io::Result<Vec<u8>> {
            if let Some((offset, size)) = self.range {
                use crate::os::FileReadAt;
                let file = fs::File::open(&self.path)?;
                if offset
                    .checked_add(size as u64)
                    .is_none_or(|end| end > file.metadata().map_or(0, |m| m.len()))
                {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "frontend asset exceeds core",
                    ));
                }
                let mut bytes = vec![0; size];
                file.read_exact_at(&mut bytes, offset)?;
                Ok(bytes)
            } else {
                fs::read(&self.path)
            }
        })()
        .map_err(|error| {
            Error::frontend(FrontendError::Asset(Failure::caused_by(
                format!("cannot read {}: {error}", self.path.display()),
                error,
            )))
        })?;
        let actual = format!("{:x}", Sha256::digest(&bytes));
        if !actual.eq_ignore_ascii_case(&self.sha256) {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                format!(
                    "frontend asset SHA256 mismatch for {}: expected {}, received {actual}",
                    self.path.display(),
                    self.sha256
                ),
            ))));
        }
        Ok(bytes)
    }
}

pub struct FrontendAssets {
    pub tokenizer: Asset,
    pub tokens: TokenizerSettings,
    pub chat_template: Option<Asset>,
}

#[cfg(any(test, feature = "test-tools"))]
#[path = "../../tests/support/frontend.rs"]
pub mod test_support;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerSettings {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
    pub sep_token: Option<String>,
    pub pad_token: Option<String>,
    pub cls_token: Option<String>,
    pub mask_token: Option<String>,
    pub additional_special_tokens: Option<TokenList>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenList {
    pub values: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendSettings {
    #[serde(default)]
    pub prompt_encoder: PromptEncoder,
    #[serde(default)]
    pub image_placeholder_token_id: Option<u32>,
    pub tokenizer: TokenizerSettings,
    pub eos_token_ids: Vec<u32>,
    #[serde(default)]
    pub template_encoding: TemplateEncoding,
    #[cfg(feature = "images")]
    pub image: Option<images::ImageSettings>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "i32", into = "i32")]
pub enum PromptEncoder {
    #[default]
    Jinja = 0,
    DeepseekV4 = 1,
    DeepseekV41 = 2,
    /// TMLv0 message frames: role and content-kind control tokens around ordinary text.
    TmlV0 = 3,
}

impl PromptEncoder {
    /// Whether the encoder renders `tool_choice` and JSON response formats in its
    /// own syntax. Other encoders receive the shared instruction messages instead.
    pub fn renders_tool_choice(self) -> bool {
        matches!(self, Self::DeepseekV4 | Self::DeepseekV41)
    }

    /// Whether the encoder renders failed tool results in its own syntax. Other
    /// encoders receive them prefixed with a text part saying the tool failed.
    pub fn renders_tool_errors(self) -> bool {
        matches!(self, Self::TmlV0)
    }
}

impl From<PromptEncoder> for i32 {
    fn from(value: PromptEncoder) -> Self {
        value as i32
    }
}
impl TryFrom<i32> for PromptEncoder {
    type Error = &'static str;
    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Jinja),
            1 => Ok(Self::DeepseekV4),
            2 => Ok(Self::DeepseekV41),
            3 => Ok(Self::TmlV0),
            _ => Err("unsupported prompt encoder"),
        }
    }
}

#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "i32", into = "i32")]
pub enum TemplateEncoding {
    #[default]
    Text = 0,
    TokenSegments = 1,
}

impl From<TemplateEncoding> for i32 {
    fn from(value: TemplateEncoding) -> Self {
        value as i32
    }
}

impl TryFrom<i32> for TemplateEncoding {
    type Error = &'static str;
    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Text),
            1 => Ok(Self::TokenSegments),
            _ => Err("unsupported chat template encoding"),
        }
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenSegment {
    text: String,
    allow_special: bool,
    #[serde(default)]
    image_index: Option<usize>,
}

fn order_segment_images(
    segments: &[TokenSegment],
    sources: &mut Vec<String>,
    hints: &mut [usize],
) -> Result<()> {
    let order: Vec<_> = segments.iter().filter_map(|segment| segment.image_index).collect();
    if order.is_empty() {
        return Ok(());
    }
    if order.len() != sources.len() {
        return Err(deepseek::invalid("image segment indices must include each image source exactly once"));
    }
    let mut positions = vec![None; sources.len()];
    for (position, &index) in order.iter().enumerate() {
        let Some(slot) = positions.get_mut(index) else {
            return Err(deepseek::invalid("image segment index exceeds the image source count"));
        };
        if slot.replace(position).is_some() {
            return Err(deepseek::invalid("image segment indices must not repeat an image source"));
        }
    }
    for hint in hints {
        *hint = positions[*hint].unwrap();
    }
    let mut original: Vec<_> = std::mem::take(sources).into_iter().map(Some).collect();
    *sources = order.into_iter().map(|index| original[index].take().unwrap()).collect();
    Ok(())
}

fn token_segments(text: &str, hints: &mut Vec<usize>) -> serde_json::Result<Vec<TokenSegment>> {
    if hints.is_empty() {
        return serde_json::from_str(text);
    }
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawSegment<'a> {
        #[serde(borrow)]
        text: &'a serde_json::value::RawValue,
        allow_special: bool,
        #[serde(default)]
        image_index: Option<usize>,
    }
    let raw_segments: Vec<RawSegment<'_>> = serde_json::from_str(text)?;
    let mut segments = Vec::with_capacity(raw_segments.len());
    let mut positions = Vec::new();
    let mut byte_start = 0;
    for segment in raw_segments {
        let raw = segment.text.get();
        let start = raw.as_ptr() as usize - text.as_ptr() as usize;
        let decoded: String = serde_json::from_str(raw)?;
        for &hint in hints.iter() {
            if hint > start && hint < start + raw.len() {
                // The hint ends inside this JSON string, before its closing quote.
                let mut prefix = raw[..hint - start].to_owned();
                prefix.push('"');
                let prefix: String = serde_json::from_str(&prefix)?;
                positions.push(byte_start + prefix.len());
            }
        }
        byte_start += decoded.len();
        segments.push(TokenSegment {
            text: decoded,
            allow_special: segment.allow_special,
            image_index: segment.image_index,
        });
    }
    *hints = positions;
    Ok(segments)
}

/// A chat video part: frames the client extracted, in order, as image data URLs, and
/// the rate in frames per second at which it sampled them.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoPart {
    pub frames: Vec<String>,
    pub fps: f64,
}

/// Reads the shape of a video content part. Admission validates the frame count, the
/// rate, and each frame against the prepared video settings.
pub fn video_part(value: &Json) -> Result<VideoPart> {
    let invalid = |message: &str| Error::frontend(FrontendError::Input(Failure::new(message)));
    let frames = value
        .get("video")
        .and_then(Json::as_array)
        .ok_or_else(|| invalid("video must be an array of image data URLs"))?
        .iter()
        .map(|frame| {
            frame
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid("video frames must be image data URL strings"))
        })
        .collect::<Result<Vec<_>>>()?;
    let fps = value
        .get("fps")
        .and_then(Json::as_f64)
        .ok_or_else(|| invalid("video fps must be a number of frames per second"))?;
    Ok(VideoPart { frames, fps })
}

pub struct RenderedPrompt {
    pub image_sources: Vec<String>,
    /// Video parts in message order, each with the frames the client supplied.
    pub video_sources: Vec<VideoPart>,
    /// Indices in image_sources whose expanded spans end at requested checkpoints.
    pub image_checkpoint_hints: Vec<usize>,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub message_boundaries: Vec<usize>,
    /// Sorted, disjoint byte ranges of `text` that came from the request. A token whose bytes
    /// overlap one is request text; an output marker is skipped only when one range holds all of
    /// it. Empty for prompts that are not rendered from a Jinja text template.
    pub ordinary: Vec<std::ops::Range<usize>>,
}

/// Whether `ordinary`, sorted and disjoint, overlaps the byte range `start..end`.
pub(crate) fn overlaps_ordinary(ordinary: &[std::ops::Range<usize>], start: usize, end: usize) -> bool {
    let first = ordinary.partition_point(|range| range.end <= start);
    ordinary.get(first).is_some_and(|range| range.start < end)
}

/// Whether one range of `ordinary` holds all of the byte range `start..end`. Ranges are sorted,
/// disjoint and merged where they touch, so a match that the request spelled completely lies in
/// one range, and a match with any template-written byte does not.
#[cfg(feature = "server")]
pub(crate) fn within_ordinary(ordinary: &[std::ops::Range<usize>], start: usize, end: usize) -> bool {
    let first = ordinary.partition_point(|range| range.end < end);
    ordinary.get(first).is_some_and(|range| range.start <= start)
}

/// Whether `token` occurs in a string of `value`, nested arrays and objects included.
fn json_names(value: &Json, token: &str) -> bool {
    match value {
        Json::String(text) => text.contains(token),
        Json::Array(items) => items.iter().any(|item| json_names(item, token)),
        Json::Object(map) => map.values().any(|item| json_names(item, token)),
        _ => false,
    }
}

fn encode_text(tokenizer: &Tokenizer, text: &str, add_special_tokens: bool) -> Result<tokenizers::Encoding> {
    tokenizer.encode(text, add_special_tokens).map_err(|error| {
        Error::frontend(FrontendError::Tokenizer(Failure::boxed(
            error.to_string(),
            error,
        )))
    })
}

pub struct Frontend {
    prompt_encoder: PromptEncoder,
    image_placeholder_token_id: Option<u32>,
    tokenizer: Tokenizer,
    /// The tokenizer without its added vocabulary, for text that must not spell controls.
    ordinary_tokenizer: Tokenizer,
    /// The added tokens that request text may not spell: the special tokens and the ones the
    /// chat template or the image settings name. Other added tokens, such as `/nothink`, are
    /// lexical: request text that spells one tokenizes to that token.
    controls: std::collections::HashSet<u32>,
    /// The control that request text positions images with, when the image layout allows it.
    spelled_image_placeholder: Option<u32>,
    template_encoding: TemplateEncoding,
    template: Option<jinja::CompiledTemplate>,
    template_source: String,
    bos_token_id: Option<u32>,
    effort_levels: std::sync::OnceLock<Vec<String>>,
    vocabulary_bytes: std::sync::OnceLock<Result<vocabulary::VocabularyBytes>>,
    inline_system_messages: std::sync::OnceLock<bool>,
    defaults: Map<String, Json>,
}

impl Frontend {
    pub fn prompt_encoder(&self) -> PromptEncoder {
        self.prompt_encoder
    }
    pub fn with_template_encoding(mut self, encoding: TemplateEncoding) -> Self {
        self.template_encoding = encoding;
        self
    }
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Resolves advisory effort against the vocabulary accepted by the prepared template.
    pub fn resolve_effort<'a>(&self, candidates: &'a [&str]) -> Result<&'a str> {
        if self.prompt_encoder == PromptEncoder::TmlV0 {
            return candidates
                .iter()
                .copied()
                .find(|level| tml::effort_level(level))
                .ok_or_else(|| deepseek::invalid("unsupported reasoning effort"));
        }
        if self.prompt_encoder != PromptEncoder::Jinja {
            return candidates
                .iter()
                .copied()
                .find(|level| matches!(*level, "low" | "high" | "xhigh" | "max"))
                .ok_or_else(|| deepseek::invalid("unsupported reasoning effort"));
        }
        let accepted = self.effort_levels.get_or_init(|| {
            let mut context = self.defaults.clone();
            context.insert(
                "messages".into(),
                serde_json::json!([{"role":"user","content":"Hello"}]),
            );
            context.insert("enable_thinking".into(), true.into());
            context.insert("thinking_mode".into(), "thinking".into());
            context.insert("add_generation_prompt".into(), true.into());
            context.insert("continue_final_message".into(), false.into());
            ["low", "medium", "high", "xhigh", "max"]
                .into_iter()
                .filter(|effort| {
                    context.insert("reasoning_effort".into(), (*effort).into());
                    self.template
                        .as_ref()
                        .unwrap()
                        .render(&context, &[])
                        .is_ok()
                })
                .map(str::to_owned)
                .collect()
        });
        candidates
            .iter()
            .copied()
            .find(|candidate| accepted.iter().any(|level| level == candidate))
            .ok_or_else(|| {
                Error::admission(
                    crate::AdmissionError::Feature,
                    "prepared chat template does not accept the requested reasoning effort",
                )
            })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    pub fn template_default(&self, name: &str) -> Option<&Json> {
        self.defaults.get(name)
    }

    pub fn normalize_inline_system_messages(&self, messages: &mut [Json]) -> Result<()> {
        if self.prompt_encoder != PromptEncoder::Jinja {
            return Ok(());
        }
        if !messages.iter().skip(1).any(|m| m["role"] == "system") {
            return Ok(());
        }
        let supported = self.inline_system_messages.get_or_init(|| {
            let marker = "FLARE_INLINE_SYSTEM_PROBE";
            let mut context = self.defaults.clone();
            context.insert(
                "messages".into(),
                serde_json::json!([
                    {"role":"system","content":"Instructions"},
                    {"role":"user","content":"Hello"},
                    {"role":"system","content":marker},
                    {"role":"user","content":"Continue"}
                ]),
            );
            context.insert("add_generation_prompt".into(), false.into());
            context.insert("continue_final_message".into(), false.into());
            self.template
                .as_ref()
                .unwrap()
                .render(&context, &[])
                .is_ok_and(|rendered| rendered.text.contains(marker))
        });
        if *supported {
            return Ok(());
        }
        for message in messages.iter_mut().skip(1) {
            if message["role"] != "system" {
                continue;
            }
            let invalid = || {
                Error::frontend(FrontendError::Input(Failure::new(
                    "inline system messages require text content",
                )))
            };
            let text = match &message["content"] {
                Json::String(text) => text.clone(),
                Json::Array(parts) => parts
                    .iter()
                    .map(|part| {
                        if part["type"] != "text" {
                            return Err(invalid());
                        }
                        part["text"].as_str().ok_or_else(invalid)
                    })
                    .collect::<Result<Vec<_>>>()?
                    .join("\n\n"),
                _ => return Err(invalid()),
            };
            message["role"] = "user".into();
            message["content"] = format!("<system>{text}</system>").into();
        }
        Ok(())
    }

    pub fn load(assets: FrontendAssets, template_defaults: Map<String, Json>) -> Result<Self> {
        Self::load_encoder(assets, template_defaults, PromptEncoder::Jinja, None)
    }

    pub fn load_with_settings(
        assets: FrontendAssets,
        template_defaults: Map<String, Json>,
        settings: &FrontendSettings,
    ) -> Result<Self> {
        if settings.prompt_encoder != PromptEncoder::Jinja && assets.chat_template.is_some() {
            return Err(deepseek::invalid(
                "official prompt encoding does not use a chat template asset",
            ));
        }
        if settings.prompt_encoder != PromptEncoder::Jinja
            && matches!(settings.template_encoding, TemplateEncoding::TokenSegments)
        {
            return Err(deepseek::invalid(
                "official prompt encoding requires text template encoding",
            ));
        }
        Self::load_encoder(
            assets,
            template_defaults,
            settings.prompt_encoder,
            settings.image_placeholder_token_id,
        )
        .map(|frontend| {
            frontend
                .with_template_encoding(settings.template_encoding)
                .with_image_placeholder_text(settings)
        })
    }

    /// Records the token that request text may spell to position images.
    fn with_image_placeholder_text(mut self, settings: &FrontendSettings) -> Self {
        #[cfg(feature = "images")]
        let text = settings
            .image
            .as_ref()
            .and_then(images::ImageSettings::spelled_placeholder);
        #[cfg(not(feature = "images"))]
        let text = {
            let _ = settings;
            None::<&str>
        };
        self.spelled_image_placeholder = text.and_then(|text| self.tokenizer.token_to_id(text));
        self.controls.extend(self.spelled_image_placeholder);
        self
    }

    fn load_encoder(
        assets: FrontendAssets,
        template_defaults: Map<String, Json>,
        prompt_encoder: PromptEncoder,
        image_placeholder_token_id: Option<u32>,
    ) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_bytes(assets.tokenizer.read()?).map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                format!("invalid tokenizer: {error}"),
                error,
            )))
        })?;
        tokenizer.with_truncation(None).map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                error.to_string(),
                error,
            )))
        })?;
        tokenizer.with_padding(None);
        boundaries::preserve_offsets(&mut tokenizer);
        if prompt_encoder != PromptEncoder::Jinja
            && image_placeholder_token_id.is_some_and(|id| tokenizer.id_to_token(id).is_none())
        {
            return Err(deepseek::invalid(
                "image placeholder token ID is absent from the checkpoint tokenizer",
            ));
        }
        let template_bytes = match assets.chat_template {
            Some(asset) if prompt_encoder == PromptEncoder::Jinja => asset.read()?,
            _ if prompt_encoder != PromptEncoder::Jinja => Vec::new(),
            _ => {
                return Err(deepseek::invalid(
                    "Jinja prompt encoding requires a chat template asset",
                ));
            }
        };
        let template_source = String::from_utf8(template_bytes).map_err(|error| {
            Error::frontend(FrontendError::Template(Failure::caused_by(
                format!("chat template is not UTF-8: {error}"),
                error,
            )))
        })?;
        let template = if prompt_encoder == PromptEncoder::Jinja {
            Some(templates::compile(&template_source).map_err(|error| {
                Error::frontend(FrontendError::Template(Failure::caused_by(error.to_string(), error)))
            })?)
        } else { None };
        let bos_token_id = assets
            .tokens
            .bos_token
            .as_deref()
            .and_then(|token| tokenizer.token_to_id(token));
        let mut defaults = Map::new();
        for (name, value) in [
            ("bos_token", assets.tokens.bos_token),
            ("eos_token", assets.tokens.eos_token),
            ("unk_token", assets.tokens.unk_token),
            ("sep_token", assets.tokens.sep_token),
            ("pad_token", assets.tokens.pad_token),
            ("cls_token", assets.tokens.cls_token),
            ("mask_token", assets.tokens.mask_token),
        ] {
            if let Some(value) = value {
                defaults.insert(name.into(), Json::String(value));
            }
        }
        if let Some(tokens) = assets.tokens.additional_special_tokens {
            defaults.insert(
                "additional_special_tokens".into(),
                Json::Array(tokens.values.into_iter().map(Json::String).collect()),
            );
        }
        defaults.insert(
            "add_generation_prompt".into(),
            Json::Bool(prompt_encoder != PromptEncoder::Jinja),
        );
        defaults.insert("tools".into(), Json::Null);
        defaults.insert("documents".into(), Json::Null);
        defaults.extend(template_defaults);
        let mut ordinary_tokenizer = tokenizer.clone();
        ordinary_tokenizer.with_added_vocabulary(tokenizers::AddedVocabulary::new());
        // A token the template names in its text or writes through a default (a special-token
        // string of the tokenizer config, a recipe default, nested ones included) is a control.
        // Whitespace alone is never a marker, however often a template writes it.
        let controls = tokenizer
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .iter()
            .filter(|(_, token)| {
                token.special
                    || (!token.content.trim().is_empty()
                        && (template_source.contains(&token.content)
                            || defaults.values().any(|value| json_names(value, &token.content))))
            })
            .map(|(&id, _)| id)
            .chain(image_placeholder_token_id)
            .collect();
        let frontend = Self {
            prompt_encoder,
            image_placeholder_token_id,
            bos_token_id,
            vocabulary_bytes: Default::default(),
            effort_levels: Default::default(),
            inline_system_messages: Default::default(),
            tokenizer,
            ordinary_tokenizer,
            controls,
            spelled_image_placeholder: None,
            template_encoding: TemplateEncoding::Text,
            template,
            template_source,
            defaults,
        };
        match prompt_encoder {
            PromptEncoder::Jinja => {}
            PromptEncoder::TmlV0 => tml::validate_defaults(&frontend)?,
            PromptEncoder::DeepseekV4 | PromptEncoder::DeepseekV41 => deepseek::validate_defaults(&frontend)?,
        }
        Ok(frontend)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        encode_text(&self.tokenizer, text, add_special_tokens).map(|encoding| encoding.get_ids().to_vec())
    }

    /// Encodes `text` without the added vocabulary, so a spelled control stays ordinary text.
    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        encode_text(&self.ordinary_tokenizer, text, false).map(|encoding| encoding.get_ids().to_vec())
    }

    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                    error.to_string(),
                    error,
                )))
            })
    }

    pub fn render_chat(
        &self,
        messages: &[Json],
        options: &Map<String, Json>,
    ) -> Result<RenderedPrompt> {
        match self.prompt_encoder {
            PromptEncoder::Jinja => {}
            PromptEncoder::TmlV0 => return tml::render(self, messages, options),
            PromptEncoder::DeepseekV4 | PromptEncoder::DeepseekV41 => {
                return deepseek::render(self, messages, options);
            }
        }
        if messages.is_empty() {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "Cannot apply chat template to an empty conversation. Provide at least one message.",
            ))));
        }
        self.render_jinja(messages, options)
    }

    fn render_jinja(
        &self,
        messages: &[Json],
        options: &Map<String, Json>,
    ) -> Result<RenderedPrompt> {
        let mut context = self.defaults.clone();
        context.extend(options.clone());
        let continuation = match context.remove("continue_final_message") {
            None | Some(Json::Null | Json::Bool(false)) => None,
            Some(Json::Bool(true)) => Some("content".to_owned()),
            Some(Json::String(field)) if field.is_empty() => None,
            Some(Json::String(field)) => Some(field),
            _ => {
                return Err(Error::frontend(FrontendError::Input(Failure::new(
                    "continue_final_message must be a boolean or field name",
                ))));
            }
        };
        if continuation.is_some() && context.get("add_generation_prompt") == Some(&Json::Bool(true))
        {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead.",
            ))));
        }
        if let Some(tools) = context.get("tools").filter(|value| !value.is_null()) {
            if !tools
                .as_array()
                .is_some_and(|values| values.iter().all(Json::is_object))
            {
                return Err(Error::frontend(FrontendError::Input(Failure::new(
                    "Tools should either be a JSON schema, or a callable function with type hints and a docstring suitable for auto-conversion to a schema.",
                ))));
            }
        }
        if let Some(documents) = context.get("documents").filter(|value| !value.is_null()) {
            if !documents
                .as_array()
                .is_some_and(|values| values.iter().all(Json::is_object))
            {
                return Err(Error::frontend(FrontendError::Input(Failure::new(
                    "Documents should be a list of dicts with 'title' and 'text' keys!",
                ))));
            }
        }
        let mut messages = messages.to_vec();
        let mut image_sources = Vec::new();
        let mut video_sources = Vec::new();
        let mut image_checkpoint_hints = Vec::new();
        let mut checkpoint_hints = Vec::new();
        for (index, message) in messages.iter_mut().enumerate() {
            if !message.is_object() {
                return Err(deepseek::invalid("messages must contain objects"));
            }
            if let Some(Json::Array(hints)) =
                message.as_object_mut().unwrap().remove("_checkpoint_hints")
            {
                checkpoint_hints.extend(
                    hints
                        .iter()
                        .filter_map(|hint| hint.as_u64().map(|count| (index, count as usize))),
                );
            }
            if let Some(parts) = message.get_mut("content").and_then(Json::as_array_mut) {
                for (part_index, part) in parts.iter_mut().enumerate() {
                    let kind = part
                        .get("type")
                        .and_then(Json::as_str)
                        .ok_or_else(|| deepseek::invalid("message content part requires type"))?;
                    if kind == "text" && !part.get("text").is_some_and(Json::is_string) {
                        return Err(deepseek::invalid("text content must be a string"));
                    }
                    if kind == "image_url" {
                        let url = part["image_url"]["url"]
                            .as_str()
                            .ok_or_else(|| deepseek::invalid("image_url.url must be a string"))?;
                        if checkpoint_hints.contains(&(index, part_index + 1)) {
                            image_checkpoint_hints.push(image_sources.len());
                        }
                        let image_index = image_sources.len();
                        image_sources.push(url.to_owned());
                        *part = serde_json::json!({"type":"image", "image_index":image_index});
                    } else if kind == "video" {
                        let index = video_sources.len();
                        video_sources.push(video_part(part)?);
                        *part = serde_json::json!({"type":"video", "video_index":index});
                    }
                }
            }
        }
        let final_text = continuation
            .map(|field| self.mark_continuation(&mut messages, &field))
            .transpose()?;
        context.insert("messages".into(), Json::Array(messages));
        let default_names: Vec<_> = self
            .defaults
            .keys()
            .filter(|name| !options.contains_key(*name) && name.as_str() != "messages")
            .map(String::as_str)
            .collect();
        let rendered = self
            .template
            .as_ref()
            .unwrap()
            .render_with_defaults(&context, &checkpoint_hints, &default_names)
            .map_err(|error| {
                Error::frontend(FrontendError::Template(Failure::caused_by(
                    error.message.clone(),
                    error,
                )))
            })?;
        let ordinary = rendered.ordinary;
        let marks = rendered.marks;
        let rendered = rendered.text;
        let mut hint_bytes = marks
            .iter()
            .filter_map(|mark| match mark {
                jinja::Mark::Hint(byte) => Some(*byte),
                _ => None,
            })
            .collect();
        let segmented = matches!(self.template_encoding, TemplateEncoding::TokenSegments);
        let mut segments: Vec<TokenSegment> = if segmented {
            token_segments(&rendered, &mut hint_bytes).map_err(|error| {
                Error::frontend(FrontendError::Template(Failure::caused_by(
                    "chat template must render an array of text and allow_special segments",
                    error,
                )))
            })?
        } else {
            vec![TokenSegment {
                text: rendered,
                allow_special: true,
                image_index: None,
            }]
        };
        order_segment_images(&segments, &mut image_sources, &mut image_checkpoint_hints)?;
        let mut segment_ends = Vec::with_capacity(segments.len() + 1);
        segment_ends.push(0usize);
        for segment in &segments {
            segment_ends.push(segment_ends.last().unwrap() + segment.text.len());
        }
        let mut byte_boundaries = Vec::new();
        for mark in marks {
            match mark {
                jinja::Mark::Hint(_) => {}
                jinja::Mark::Byte(byte) if !segmented && byte <= *segment_ends.last().unwrap() => {
                    byte_boundaries.push(byte);
                }
                jinja::Mark::Position(count) if segmented => {
                    byte_boundaries.push(*segment_ends.get(count).ok_or_else(|| {
                        deepseek::invalid("mark position exceeds the rendered segment count")
                    })?);
                }
                _ => {
                    return Err(deepseek::invalid(
                        "text templates require byte marks; segmented templates require segment-count marks",
                    ));
                }
            }
        }
        let mut text: String = segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect();
        if let Some(final_text) = final_text {
            let marker = "CONTINUE_FINAL_MESSAGE_TAG";
            let index = text.rfind(marker).filter(|_| text.contains(final_text.trim())).ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new("continue_final_message is set but the final message does not appear in the chat after applying the chat template!"))))?;
            let preserved_space = text[index + marker.len()..].starts_with(' ');
            text.truncate(index);
            if !preserved_space {
                text.truncate(text.trim_end().len());
            }
        }
        let mut remaining = text.len();
        let mut token_ids = Vec::new();
        let mut offsets = Vec::new();
        let mut byte_start = 0;
        for segment in &mut segments {
            segment.text.truncate(remaining.min(segment.text.len()));
            remaining -= segment.text.len();
            let tokenizer = if segment.allow_special {
                &self.tokenizer
            } else {
                &self.ordinary_tokenizer
            };
            let encoding = encode_text(tokenizer, &segment.text, false)?;
            let first = token_ids.len();
            if segmented {
                token_ids.extend_from_slice(encoding.get_ids());
                offsets.extend_from_slice(encoding.get_offsets());
            } else {
                self.push_segment(
                    &segment.text,
                    &ordinary,
                    &encoding,
                    image_sources.len(),
                    &mut token_ids,
                    &mut offsets,
                )?;
            }
            for span in &mut offsets[first..] {
                *span = (byte_start + span.0, byte_start + span.1);
            }
            byte_start += segment.text.len();
            if remaining == 0 {
                break;
            }
        }
        let message_boundaries =
            boundaries::token_ends(&byte_boundaries, &hint_bytes, &offsets, text.len())?;
        Ok(RenderedPrompt {
            message_boundaries,
            text,
            token_ids,
            image_sources,
            video_sources,
            image_checkpoint_hints,
            // Segmented text is not the template output the ranges index.
            ordinary: if segmented { Vec::new() } else { ordinary },
        })
    }

    /// Appends the tokens of a prompt segment to `ids_out` and `offsets_out`, with byte spans
    /// relative to `text`, so that request text spelling a control stays ordinary. `encoding` is
    /// the segment encoded with the added vocabulary. Added tokens other than spelled controls
    /// (template controls and lexical tokens such as `/nothink`) delimit regions; only a region
    /// that holds a spelled control is encoded again, without the added vocabulary.
    fn push_segment(
        &self,
        text: &str,
        ordinary: &[std::ops::Range<usize>],
        encoding: &tokenizers::Encoding,
        images: usize,
        ids_out: &mut Vec<u32>,
        offsets_out: &mut Vec<(usize, usize)>,
    ) -> Result<()> {
        let added = self
            .tokenizer
            .get_added_vocabulary()
            .get_added_tokens_decoder();
        let (ids, offsets) = (encoding.get_ids(), encoding.get_offsets());
        // An added token with lstrip or rstrip also covers adjacent whitespace, which may be
        // request text. Only the token's own bytes decide whether the request spelled it.
        let spelled = |at: usize| {
            let (start, end) = offsets[at];
            let own = added.get(&ids[at]).and_then(|token| {
                let found = text.get(start..end)?.find(token.content.as_str())?;
                Some(start + found..start + found + token.content.len())
            });
            let own = own.unwrap_or(start..end);
            overlaps_ordinary(ordinary, own.start, own.end)
        };
        // The request positions images by spelling their placeholder when the spelled and
        // template-written placeholders together account for every attached image.
        let positional = self.spelled_image_placeholder.filter(|id| {
            let requested = (0..ids.len()).filter(|&at| ids[at] == *id && spelled(at)).count();
            let written = ids.iter().filter(|token| *token == id).count() - requested;
            requested > 0 && written + requested == images
        });
        // A token ends a region when it is a template control or a lexical added token, or the
        // image placeholder that positions an attached image.
        let closes = |at: usize| {
            let id = ids[at];
            (added.contains_key(&id) && !(self.controls.contains(&id) && spelled(at)))
                || positional == Some(id)
        };
        let (mut from, mut from_byte, mut dirty) = (0, 0, false);
        for at in 0..=ids.len() {
            if at < ids.len() && !closes(at) {
                dirty |= self.controls.contains(&ids[at]) && added.contains_key(&ids[at]);
                continue;
            }
            if dirty {
                let end = offsets.get(at).map_or(text.len(), |span| span.0);
                self.push_ordinary(text, from_byte..end, ids_out, offsets_out)?;
            } else {
                ids_out.extend_from_slice(&ids[from..at]);
                offsets_out.extend_from_slice(&offsets[from..at]);
            }
            if at < ids.len() {
                ids_out.push(ids[at]);
                offsets_out.push(offsets[at]);
                from_byte = offsets[at].1;
            }
            from = at + 1;
            dirty = false;
        }
        Ok(())
    }

    /// Appends the encoding of `text[range]` without the added vocabulary, with byte spans
    /// relative to `text`.
    fn push_ordinary(
        &self,
        text: &str,
        range: std::ops::Range<usize>,
        ids: &mut Vec<u32>,
        offsets: &mut Vec<(usize, usize)>,
    ) -> Result<()> {
        let encoding = encode_text(&self.ordinary_tokenizer, &text[range.clone()], false)?;
        ids.extend_from_slice(encoding.get_ids());
        offsets.extend(
            encoding
                .get_offsets()
                .iter()
                .map(|&(a, b)| (range.start + a, range.start + b)),
        );
        Ok(())
    }

    fn mark_continuation(&self, messages: &mut [Json], field: &str) -> Result<String> {
        let message = messages.last_mut().ok_or_else(|| {
            Error::frontend(FrontendError::Input(Failure::new(
                "list index out of range",
            )))
        })?;
        let value = message.get_mut(field).filter(|value| !value.is_null()).ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new(format!("continue_final_message is set but the final message has no \"{field}\" to continue!")))))?;
        if !self.template_source.contains(field) {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                format!(
                    "continue_final_message is set to \"{field}\" but this is not an accepted field in the chat_template"
                ),
            ))));
        }
        let text = if let Some(blocks) = value.as_array_mut() {
            blocks
                .iter_mut()
                .rev()
                .find_map(|block| block.get_mut("text"))
                .and_then(|value| value.as_str())
                .map(str::to_owned)
        } else {
            value.as_str().map(str::to_owned)
        };
        let original = text.ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new("continue_final_message is set but we could not find any text to continue in the final message!"))))?;
        let target = if let Some(blocks) = value.as_array_mut() {
            blocks
                .iter_mut()
                .rev()
                .find_map(|block| block.get_mut("text"))
                .unwrap()
        } else {
            value
        };
        *target = Json::String(format!("{original}CONTINUE_FINAL_MESSAGE_TAG "));
        Ok(original)
    }

    #[cfg(feature = "server")]
    pub(crate) fn prepare_token_bytes(&self) -> Result<()> {
        self.vocabulary_bytes.get_or_init(|| vocabulary::VocabularyBytes::new(&self.tokenizer))
            .as_ref().map(|_| ()).map_err(Clone::clone)
    }

    pub fn token_bytes(&self, token: u32) -> Result<&[u8]> {
        self.vocabulary_bytes.get_or_init(|| vocabulary::VocabularyBytes::new(&self.tokenizer))
            .as_ref().map_err(Clone::clone)?.get(token)
    }

    pub fn incremental_decoder(&self, skip_special_tokens: bool) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tokenizer: &self.tokenizer,
            skip_special_tokens,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            finished: false,
        }
    }
}

pub struct IncrementalDecoder<'a> {
    tokenizer: &'a Tokenizer,
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    finished: bool,
}

impl IncrementalDecoder<'_> {
    pub fn push(&mut self, token_ids: &[u32]) -> Result<Option<String>> {
        if self.finished {
            return Err(Error::frontend(FrontendError::Tokenizer(Failure::new(
                "incremental decoder is finished",
            ))));
        }
        if token_ids.is_empty() {
            return Ok(None);
        }
        tokenizers::tokenizer::step_decode_stream(
            self.tokenizer,
            token_ids.to_vec(),
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                error.to_string(),
                error,
            )))
        })
    }

    #[cfg(feature = "server")]
    pub(crate) fn pending_text(&self) -> Result<String> {
        let text = self.tokenizer.decode(&self.ids, self.skip_special_tokens).map_err(|e| e.to_string())?;
        Ok(text.strip_prefix(&self.prefix).ok_or("decoder changed its emitted prefix")?.to_owned())
    }

    pub fn finish(&mut self) -> Result<String> {
        if self.finished {
            return Ok(String::new());
        }
        let remainder = self
            .tokenizer
            .decode(&self.ids, self.skip_special_tokens)
            .map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                    error.to_string(),
                    error,
                )))
            })?;
        let suffix = remainder
            .strip_prefix(&self.prefix)
            .ok_or_else(|| {
                Error::frontend(FrontendError::Tokenizer(Failure::new(
                    "incremental decoder final text changed its emitted prefix",
                )))
            })?
            .to_owned();
        self.finished = true;
        Ok(suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontend_is_shareable_between_native_request_workers() {
        fn require_send_sync<T: Send + Sync>() {}
        require_send_sync::<Frontend>();
    }

    fn role_frontend(guard: &str) -> Frontend {
        let source = format!(
            "{{% for message in messages %}}{guard}{{{{ message.role }}}}:{{{{ message.content }}}}\n{{% endfor %}}"
        );
        part_frontend(&source)
    }

    fn part_frontend(source: &str) -> Frontend {
        Frontend {
            prompt_encoder: PromptEncoder::Jinja,
            image_placeholder_token_id: None,
            ordinary_tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            controls: Default::default(),
            spelled_image_placeholder: None,
            template_encoding: TemplateEncoding::Text,
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            template: Some(jinja::CompiledTemplate::compile(source).unwrap()),
            template_source: source.to_owned(),
            bos_token_id: None,
            vocabulary_bytes: Default::default(),
            effort_levels: Default::default(),
            inline_system_messages: Default::default(),
            defaults: Map::new(),
        }
    }

    #[test]
    fn video_parts_reach_the_template_without_their_frames() {
        let frontend = part_frontend(
            "{% for m in messages %}{% for part in m.content %}{% if part.type == 'video' %}{% if part.video is defined %}{{ raise_exception('video frames reached the template') }}{% endif %}<|vision_start|><|video_pad|><|vision_end|>{% endif %}{% endfor %}{% endfor %}",
        );
        let messages = vec![serde_json::json!({"role":"user","content":[{"type":"video",
            "video":["data:image/png;base64,AA==","data:image/png;base64,AQ=="],"fps":2.0}]})];
        let rendered = frontend.render_chat(&messages, &Map::new()).unwrap();
        assert_eq!(rendered.text, "<|vision_start|><|video_pad|><|vision_end|>");
        assert_eq!(rendered.video_sources.len(), 1);
        assert_eq!(rendered.video_sources[0].frames.len(), 2);
    }

    #[test]
    fn inline_system_roles_follow_template_capability() {
        let source = serde_json::json!([
            {"role":"system","content":"Initial instructions"},
            {"role":"user","content":"Hello"},
            {"role":"system","content":[
                {"type":"text","text":"Environment"},
                {"type":"text","text":"Use the repository"}
            ]},
            {"role":"assistant","content":"Ready"},
            {"role":"user","content":"Continue"}
        ]);
        let original = source.as_array().unwrap();
        for (guard, supported) in [
            ("", true),
            (
                "{% if message.role == 'system' and not loop.first %}{{ raise_exception('system must be first') }}{% endif %}",
                false,
            ),
            (
                "{% if message.role == 'system' and not loop.first %}{% continue %}{% endif %}",
                false,
            ),
        ] {
            let frontend = role_frontend(guard);
            let mut messages = original.clone();
            frontend
                .normalize_inline_system_messages(&mut messages)
                .unwrap();
            assert_eq!(&messages[..2], &original[..2]);
            assert_eq!(&messages[3..], &original[3..]);
            if supported {
                assert_eq!(messages, *original);
            } else {
                assert_eq!(
                    messages[2],
                    serde_json::json!({
                        "role":"user","content":"<system>Environment\n\nUse the repository</system>"
                    })
                );
                frontend.render_chat(&messages, &Map::new()).unwrap();
            }
            let once = messages.clone();
            frontend
                .normalize_inline_system_messages(&mut messages)
                .unwrap();
            assert_eq!(messages, once);
        }
    }

    #[test]
    fn inline_system_conversion_preserves_text_and_rejects_nontext() {
        let frontend = role_frontend(
            "{% if message.role == 'system' and not loop.first %}{{ raise_exception('system must be first') }}{% endif %}",
        );
        let mut messages = vec![
            serde_json::json!({"role":"user","content":"Hello"}),
            serde_json::json!({"role":"system","content":"\nUse <tag> & café.\n"}),
        ];
        frontend
            .normalize_inline_system_messages(&mut messages)
            .unwrap();
        assert_eq!(
            messages[1]["content"],
            "<system>\nUse <tag> & café.\n</system>"
        );
        messages[1] = serde_json::json!({"role":"system","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]});
        assert!(
            frontend
                .normalize_inline_system_messages(&mut messages)
                .is_err()
        );
    }

    #[test]
    fn advisory_effort_uses_the_prepared_templates_vocabulary() {
        for (levels, expected) in [
            ("['low','medium','xhigh']", ["low", "medium", "xhigh"]),
            ("['low','high','max']", ["low", "high", "max"]),
        ] {
            let source = format!(
                "{{% if reasoning_effort not in {levels} %}}{{{{ raise_exception('unsupported effort') }}}}{{% endif %}}"
            );
            let template = Some(jinja::CompiledTemplate::compile(&source).unwrap());
            let frontend = Frontend {
                prompt_encoder: PromptEncoder::Jinja,
                image_placeholder_token_id: None,
                ordinary_tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
                controls: Default::default(),
                spelled_image_placeholder: None,
                template_encoding: TemplateEncoding::Text,
                tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
                template,
                template_source: source,
                bos_token_id: None,
                vocabulary_bytes: Default::default(),
                effort_levels: Default::default(),
                inline_system_messages: Default::default(),
                defaults: Map::new(),
            };
            for (candidates, expected) in [
                (&["low"][..], expected[0]),
                (&["high", "medium"][..], expected[1]),
                (&["max", "xhigh", "high", "medium"][..], expected[2]),
            ] {
                assert_eq!(frontend.resolve_effort(candidates).unwrap(), expected);
            }
            assert!(frontend.resolve_effort(&["unsupported"]).is_err());
        }
    }

    #[test]
    fn missing_asset_retains_io_failure() {
        use std::error::Error as _;

        let asset = Asset {
            range: None,
            path: std::env::temp_dir()
                .join(format!("flare-missing-asset-{}", std::process::id()))
                .join("tokenizer.json"),
            sha256: "0".repeat(64),
        };
        let error = asset.read().unwrap_err();
        assert!(matches!(error, Error::Frontend(FrontendError::Asset(_))));
        let cause = error
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap();
        assert_eq!(cause.kind(), std::io::ErrorKind::NotFound);
        let cloned = error.clone();
        assert_eq!(cloned.to_string(), error.to_string());
        assert!(std::ptr::eq(
            error.source().unwrap(),
            cloned.source().unwrap()
        ));
    }

    #[test]
    fn render_failure_retains_template_cause_and_detail() {
        use std::error::Error as _;

        let source = "{{ raise_exception('fixture rejection') }}";
        let template = Some(jinja::CompiledTemplate::compile(source).unwrap());
        let frontend = Frontend {
            prompt_encoder: PromptEncoder::Jinja,
            image_placeholder_token_id: None,
            bos_token_id: None,
            vocabulary_bytes: Default::default(),
            effort_levels: Default::default(),
            inline_system_messages: Default::default(),
            ordinary_tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            controls: Default::default(),
            spelled_image_placeholder: None,
            template_encoding: TemplateEncoding::Text,
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            template,
            template_source: source.to_owned(),
            defaults: Map::new(),
        };
        let error = frontend
            .render_chat(
                &[serde_json::json!({"role":"user","content":"x"})],
                &Map::new(),
            )
            .err()
            .unwrap();
        assert!(matches!(error, Error::Frontend(FrontendError::Template(_))));
        assert_eq!(error.to_string(), "fixture rejection");
        let cause = error
            .source()
            .unwrap()
            .downcast_ref::<jinja::Error>()
            .unwrap();
        assert_eq!(cause.message, "fixture rejection");
        let input = frontend.render_chat(&[], &Map::new()).err().unwrap();
        assert!(matches!(input, Error::Frontend(FrontendError::Input(_))));
        assert!(input.source().is_none());
    }

    #[test]
    fn verified_asset_rejects_content_changed_after_fingerprinting() {
        let directory = std::env::temp_dir().join(format!("flare-asset-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("tokenizer_config.json");
        let original = b"{\"eos_token\":\"<eos>\"}";
        fs::write(&path, original).unwrap();
        let asset = Asset {
            range: None,
            path: path.clone(),
            sha256: format!("{:x}", Sha256::digest(original)),
        };
        assert_eq!(asset.read().unwrap(), original);
        fs::write(&path, b"{\"eos_token\":\"changed\"}").unwrap();
        assert!(matches!(
            asset.read().unwrap_err(),
            Error::Frontend(FrontendError::Asset(_))
        ));
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn image_segment_indices_reorder_sources_and_hints_and_reject_ambiguity() {
        let mut hints = Vec::new();
        let segments = token_segments(r#"[{"text":"second","allow_special":true,"image_index":1},{"text":"first","allow_special":true,"image_index":0}]"#, &mut hints).unwrap();
        let mut sources = vec!["first".into(), "second".into()];
        let mut checkpoints = vec![0];
        order_segment_images(&segments, &mut sources, &mut checkpoints).unwrap();
        assert_eq!(sources, ["second", "first"]);
        assert_eq!(checkpoints, [1]);
        for indices in [vec![0,0], vec![0,2], vec![1]] {
            let segments = indices.into_iter().map(|index| TokenSegment {
                text: "image".into(), allow_special:true, image_index:Some(index),
            }).collect::<Vec<_>>();
            assert!(order_segment_images(&segments, &mut sources, &mut checkpoints).is_err());
        }
        let segments = token_segments(r#"[{"text":"legacy","allow_special":true}]"#, &mut hints).unwrap();
        order_segment_images(&segments, &mut sources, &mut checkpoints).unwrap();
        assert_eq!(sources, ["second", "first"]);
        assert_eq!(checkpoints, [1]);
    }

    #[test]
    fn segment_hints_survive_json_escapes_key_order_and_nested_serialization() {
        use jinja::{CompiledTemplate, Mark};
        use serde_json::json;
        let context = json!({"messages":[{"content":[{"text":"é\n\"\\😀"}]}]});
        for ascii in ["true", "false"] {
            for sort in ["true", "false"] {
                let source = format!(
                    concat!(
                        "{{% set history = messages[0].content[0].text %}}",
                        "{{% set segments = [{{'text': 'prefix' + history + 'tail', 'allow_special': false}},",
                        "{{'text': history | tojson(ensure_ascii={0}), 'allow_special': true}}] %}}",
                        "{{{{ segments | tojson(ensure_ascii={0}, sort_keys={1}, indent=2) }}}}",
                    ),
                    ascii, sort
                );
                let template = CompiledTemplate::compile(&source).unwrap();
                let plain = template.render(context.as_object().unwrap(), &[]).unwrap();
                let marked = template
                    .render(context.as_object().unwrap(), &[(0, 1)])
                    .unwrap();
                assert_eq!(plain.text, marked.text);
                let mut hints = marked
                    .marks
                    .into_iter()
                    .map(|mark| match mark {
                        Mark::Hint(byte) => byte,
                        _ => panic!("unexpected mark"),
                    })
                    .collect();
                let segments = token_segments(&marked.text, &mut hints).unwrap();
                assert_eq!(segments[0].text, "prefixé\n\"\\😀tail");
                assert!(!segments[0].allow_special);
                assert!(segments[1].allow_special);
                assert_eq!(
                    hints,
                    [
                        "prefixé\n\"\\😀".len(),
                        segments[0].text.len() + segments[1].text.len() - 1
                    ]
                );
            }
        }
        let template = CompiledTemplate::compile(
            "{{ {'text': messages[0].content[0].text, 'text': 'x'} | tojson }}",
        )
        .unwrap();
        let rendered = template
            .render(context.as_object().unwrap(), &[(0, 1)])
            .unwrap();
        assert_eq!(rendered.text, "{\"text\": \"x\"}");
        assert!(rendered.marks.is_empty());
    }
}
