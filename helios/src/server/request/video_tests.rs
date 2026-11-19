/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! CPU admission and request preparation of video clips with a Qwen image profile.

use super::*;
use crate::frontend::VideoPart;
use crate::frontend::images::{
    ImageLimits, ImageSettings, ImageSource, QwenImageSettings, QwenVideoSettings,
};
use serde_json::json;

const IMAGE_TOKEN: u32 = 248_056;
const VIDEO_TOKEN: u32 = 248_057;

/// An RGB PNG whose samples all equal `value`.
fn png(width: u32, height: u32, value: u8) -> Vec<u8> {
    let mut info = png::Info::with_size(width, height);
    info.color_type = png::ColorType::Rgb;
    info.bit_depth = png::BitDepth::Eight;
    let mut bytes = vec![];
    let encoder = png::Encoder::with_info(&mut bytes, info).unwrap();
    let mut writer = encoder.write_header().unwrap();
    writer
        .write_image_data(&vec![value; width as usize * height as usize * 3])
        .unwrap();
    drop(writer);
    bytes
}

fn data_url(bytes: &[u8]) -> String {
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

/// A clip whose frame `index` has every sample equal to `index`.
fn clip(frames: &[(u32, u32)], fps: f64) -> VideoPart {
    VideoPart {
        frames: frames
            .iter()
            .enumerate()
            .map(|(index, (width, height))| data_url(&png(*width, *height, index as u8)))
            .collect(),
        fps,
    }
}

fn limits() -> ImageLimits {
    ImageLimits {
        max_patches_per_image: 4096,
        max_total_patches: 65_536,
        patches_per_feature: 4,
        ..ImageLimits::default()
    }
}

fn qwen(image_token_id: u32, video_token_id: u32, limits: ImageLimits, video: bool) -> ImageProcessor {
    ImageProcessor::load(
        ImageSettings::Qwen(QwenImageSettings {
            min_pixels: 4096,
            max_pixels: 65536,
            position_side: 48,
            image_token_id,
            video_token_id,
            video: video.then_some(QwenVideoSettings {
                max_frames: 768,
                min_pixels: 4096,
                max_pixels: 25_165_824,
                temporal_patch_size: 2,
            }),
        }),
        limits,
    )
    .unwrap()
}

fn video_processor(limits: ImageLimits) -> ImageProcessor {
    qwen(IMAGE_TOKEN, VIDEO_TOKEN, limits, true)
}

/// The message of a clip rejection, which must be an image admission error raised
/// before any frame is decoded.
fn rejection(processor: &ImageProcessor, clips: &[VideoPart], image_tokens: usize) -> String {
    let decoded = processor.preparation_counts().decoded_images;
    let error = describe_videos(clips, Some(processor), None, image_tokens).unwrap_err();
    assert!(matches!(error, Error::Admission(AdmissionError::Image(_))), "{error}");
    assert_eq!(processor.preparation_counts().decoded_images, decoded, "{error}");
    error.to_string()
}

#[test]
fn clips_are_planned_from_frame_headers_and_each_frame_is_decoded_once() {
    let processor = video_processor(limits());
    let videos = describe_videos(&[clip(&[(64, 64); 4], 2.0)], Some(&processor), None, 0).unwrap();
    assert_eq!(videos.len(), 1);
    assert_eq!(videos[0].frames.len(), 4);
    assert_eq!(videos[0].window.grid_thw, [1, 4, 4]);
    assert_eq!(videos[0].window.tokens, 4);
    assert_ne!(videos[0].frames[0].identity, videos[0].frames[1].identity);
    // Admission decodes each frame once, prepares no window, and leaves each decode with
    // its source under the decoded-source budget.
    let counts = processor.preparation_counts();
    assert_eq!((counts.decoded_images, counts.prepared_images), (4, 0));
    assert!(counts.decoded_source_bytes >= 4 * 64 * 64 * 3);
    for frame in &videos[0].frames {
        let ImageSource::Encoded { decoded, .. } = &frame.source else {
            panic!("a frame keeps its encoded source");
        };
        assert!(decoded.lock().unwrap().is_some());
    }
    drop(videos);
    assert_eq!(processor.preparation_counts().decoded_source_bytes, 0);
    // The final temporal patch of an odd clip names its last frame twice.
    let odd = describe_videos(&[clip(&[(64, 64); 5], 2.0)], Some(&processor), None, 0).unwrap();
    let (first, second) = odd[0].frames_of(2);
    assert!(std::ptr::eq(first, second));
    assert!(std::ptr::eq(first, &odd[0].frames[4]));
    assert_eq!(processor.preparation_counts().decoded_images, 9);
}

#[test]
fn sixteen_bit_png_frames_use_the_still_image_rgb_conversion() {
    use crate::frontend::images::RgbImage;

    let processor = video_processor(limits());
    let frames = vec![
        data_url(include_bytes!("../../../../tests/frontend/png-16bit-inputs/gray16.png")),
        data_url(include_bytes!("../../../../tests/frontend/png-16bit-inputs/rgb16.png")),
    ];
    let videos = describe_videos(
        &[VideoPart { frames, fps: 2.0 }], Some(&processor), None, 0,
    ).unwrap();
    assert_eq!(videos.len(), 1);
    assert_eq!(videos[0].frames.len(), 2);
    let (first, second) = videos[0].frames_of(0);
    // The fixture RGB bytes are the checkpoint processor's Pillow conversion.
    for (frame, rgb) in [
        (first, include_bytes!("../../../../tests/frontend/png-16bit-inputs/gray16.rgb")),
        (second, include_bytes!("../../../../tests/frontend/png-16bit-inputs/rgb16.rgb")),
    ] {
        let expected = RgbImage::new(2, 2, rgb.to_vec()).unwrap();
        assert_eq!(frame.identity, processor.semantic_identity(&expected));
    }
}

#[test]
fn clip_rejections_name_the_clip_and_decode_no_frame() {
    let processor = video_processor(limits());
    let one = |clip: VideoPart| rejection(&processor, &[clip], 0);
    assert_eq!(
        one(clip(&[(64, 64); 1], 2.0)),
        "video 1: clip has 1 frames; prepared video settings admit 2 through 768 frames"
    );
    // The frame count is checked before any frame is read.
    assert_eq!(
        one(VideoPart { frames: vec!["not a data URL".into(); 769], fps: 2.0 }),
        "video 1: clip has 769 frames; prepared video settings admit 2 through 768 frames"
    );
    assert_eq!(
        one(VideoPart { frames: vec![], fps: 2.0 }),
        "video 1: clip has 0 frames; prepared video settings admit 2 through 768 frames"
    );
    for fps in [0.0, -2.5, f64::MIN_POSITIVE / 2.0, 0.0005, 1000.5, f64::INFINITY, f64::NAN] {
        assert_eq!(
            one(VideoPart { frames: vec!["not a data URL".into(); 4], fps }),
            "video 1: frame rate must be between 0.001 and 1000 frames per second",
            "{fps}"
        );
    }
    for fps in [0.001, 1000.0] {
        assert!(describe_videos(&[clip(&[(64, 64); 2], fps)], Some(&processor), None, 0).is_ok());
    }
    assert_eq!(
        one(clip(&[(64, 64), (96, 64), (64, 64), (64, 64)], 2.0)),
        "video 1 frame 2 is 96x64 px; the frames of a clip share one size and its first frame \
         is 64x64 px"
    );
    assert_eq!(one(clip(&[(8000, 16); 2], 2.0)), "video 1: frame aspect ratio exceeds 200");
    assert_eq!(
        rejection(
            &processor,
            &[clip(&[(64, 64); 2], 2.0), clip(&[(64, 64); 1], 2.0)],
            0
        ),
        "video 2: clip has 1 frames; prepared video settings admit 2 through 768 frames"
    );
    let narrow = video_processor(ImageLimits { max_patches_per_image: 8, ..limits() });
    assert_eq!(
        rejection(&narrow, &[clip(&[(64, 64); 2], 2.0)], 0),
        "video 1: one temporal patch needs 4 video tokens (16 patches); \
         max_image_tokens_per_image is 2 (8 patches)"
    );
    let few = video_processor(ImageLimits { max_video_frames: 3, ..limits() });
    assert_eq!(
        rejection(&few, &[clip(&[(64, 64); 2], 2.0), clip(&[(64, 64); 2], 2.0)], 0),
        "request has 4 video frames; max_video_frames_per_prompt is 3"
    );
    assert!(describe_videos(&[clip(&[(64, 64); 3], 2.0)], Some(&few), None, 0).is_ok());
}

#[test]
fn frame_rejections_name_the_clip_and_frame() {
    let processor = video_processor(limits());
    let frame = data_url(&png(64, 64, 7));
    let with = |second: String| VideoPart { frames: vec![frame.clone(), second], fps: 2.0 };
    let message = rejection(&processor, &[with("data:image/png;base64,!!!!".into())], 0);
    assert!(message.starts_with("video 1 frame 2: invalid image base64"), "{message}");
    assert_eq!(
        rejection(&processor, &[with("https://example.invalid/frame.png".into())], 0),
        "video 1 frame 2: video frame requires a base64 PNG, JPEG, or WebP data URL"
    );
    let gif = format!(
        "data:image/gif;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(
            include_bytes!("../../../../tests/frontend/gif-inputs/frame-offset.gif")
        )
    );
    assert_eq!(
        rejection(&processor, &[with(gif)], 0),
        "video 1 frame 2: video frame requires a base64 PNG, JPEG, or WebP data URL"
    );
    // A frame cut off inside its image data fails the header pass of the second clip.
    let bytes = png(64, 64, 7);
    let truncated = data_url(&bytes[..bytes.len() - 20]);
    let message = rejection(&processor, &[with(frame.clone()), with(truncated)], 0);
    assert!(message.starts_with("video 2 frame 2: invalid PNG"), "{message}");
    // An animated container is rejected from its header.
    let animated = format!(
        "data:image/webp;base64,{}",
        base64::engine::general_purpose::STANDARD
            .encode(include_bytes!("../../../../tests/frontend/webp-inputs/animated.webp"))
    );
    assert_eq!(
        rejection(&processor, &[VideoPart { frames: vec![animated.clone(), animated], fps: 2.0 }], 0),
        "video 1 frame 1: animated WebP is unsupported; provide a still image"
    );
    let small = video_processor(ImageLimits { max_encoded_bytes: 16, ..limits() });
    assert_eq!(
        rejection(&small, &[with(frame.clone())], 0),
        "video 1 frame 1: encoded image exceeds the configured byte limit"
    );
    let few_pixels = video_processor(ImageLimits { max_input_pixels: 1000, ..limits() });
    assert_eq!(
        rejection(&few_pixels, &[with(frame.clone())], 0),
        "video 1 frame 1: image is 64x64 px; max_image_pixels is 1000"
    );
}

/// The CRC-32 of a PNG chunk's type and data.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB8_8320 & 0u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}

#[test]
fn a_frame_whose_image_data_is_shorter_than_its_header_is_rejected_by_frame() {
    // The header declares 64 rows and the image data holds eight. The header pass accepts
    // the frame, and decoding it fails after the clip was admitted.
    let mut bytes = png(64, 8, 3);
    bytes[20..24].copy_from_slice(&64u32.to_be_bytes());
    let crc = crc32(&bytes[12..29]);
    bytes[29..33].copy_from_slice(&crc.to_be_bytes());
    let processor = video_processor(limits());
    assert_eq!(
        processor
            .probe_video_frame(crate::frontend::images::ImageEncoding::Png, &bytes)
            .unwrap(),
        (64, 64)
    );
    let frame = data_url(&png(64, 64, 7));
    let error = describe_videos(
        &[VideoPart { frames: vec![frame, data_url(&bytes)], fps: 2.0 }],
        Some(&processor),
        None,
        0,
    )
    .unwrap_err();
    assert!(matches!(error, Error::Admission(AdmissionError::Image(_))));
    assert!(error.to_string().starts_with("video 1 frame 2: invalid PNG"), "{error}");
    // The first frame was decoded and retained; the rejection releases it.
    assert_eq!(processor.preparation_counts().decoded_source_bytes, 0);
}

#[test]
fn video_tokens_share_the_request_budget_with_image_tokens() {
    let image = data_url(&png(64, 64, 255));
    let admit = |patches: usize| {
        let processor = video_processor(ImageLimits { max_total_patches: patches, ..limits() });
        let images = describe_images(std::slice::from_ref(&image), Some(&processor), None).unwrap();
        let image_tokens: usize = images
            .iter()
            .map(|image| image.description.geometry.tokens)
            .sum();
        assert_eq!(image_tokens, 4);
        // The clip alone fits the budget.
        assert!(describe_videos(&[clip(&[(64, 64); 2], 2.0)], Some(&processor), None, 0).is_ok());
        let decoded = processor.preparation_counts().decoded_images;
        let admitted =
            describe_videos(&[clip(&[(64, 64); 2], 2.0)], Some(&processor), None, image_tokens);
        (admitted, processor.preparation_counts().decoded_images - decoded)
    };
    // Four image tokens and four video tokens against a budget of seven tokens.
    let (rejected, decoded) = admit(28);
    let error = rejected.unwrap_err();
    assert!(matches!(error, Error::Admission(AdmissionError::Image(_))));
    assert_eq!(
        error.to_string(),
        "request needs 8 image and video tokens; max_image_tokens_per_prompt is 7"
    );
    assert_eq!(decoded, 0);
    // A budget of eight tokens admits both.
    let (admitted, decoded) = admit(32);
    assert_eq!(admitted.unwrap().len(), 1);
    assert_eq!(decoded, 2);
}

#[test]
fn repeated_admission_takes_frame_identities_from_the_input_store() {
    use flare_selene::input::{DiskBudget, InputStore};

    let processor = video_processor(limits());
    let store = InputStore::start(2 << 20, None, DiskBudget::new(0, 0)).unwrap();
    let source = clip(&[(64, 64); 4], 2.0);
    let first =
        describe_videos(std::slice::from_ref(&source), Some(&processor), Some(&store), 0).unwrap();
    assert_eq!(processor.preparation_counts().decoded_images, 4);
    let second =
        describe_videos(std::slice::from_ref(&source), Some(&processor), Some(&store), 0).unwrap();
    assert_eq!(processor.preparation_counts().decoded_images, 4);
    for (first, second) in first[0].frames.iter().zip(&second[0].frames) {
        assert_eq!(first.identity, second.identity);
        // The repeated admission decoded nothing, so its sources retain nothing.
        let ImageSource::Encoded { decoded, .. } = &second.source else {
            panic!("a frame keeps its encoded source");
        };
        assert!(decoded.lock().unwrap().is_none());
    }
    // A picture described as a still image and as a frame has one record of each form.
    // Each use decodes it once, in either order, and neither replaces the other's record.
    let still = data_url(&png(64, 64, 200));
    let frames = [VideoPart { frames: vec![still.clone(), still.clone()], fps: 2.0 }];
    let image = std::slice::from_ref(&still);
    let decoded = || processor.preparation_counts().decoded_images;
    let images = describe_images(image, Some(&processor), Some(&store)).unwrap();
    assert_eq!(decoded(), 5);
    let videos = describe_videos(&frames, Some(&processor), Some(&store), 0).unwrap();
    assert_eq!(decoded(), 6);
    assert_eq!(videos[0].frames[0].identity, images[0].description.identity);
    describe_images(image, Some(&processor), Some(&store)).unwrap();
    describe_videos(&frames, Some(&processor), Some(&store), 0).unwrap();
    assert_eq!(decoded(), 6);
    let other = data_url(&png(64, 64, 201));
    let frames = [VideoPart { frames: vec![other.clone(), other.clone()], fps: 2.0 }];
    describe_videos(&frames, Some(&processor), Some(&store), 0).unwrap();
    assert_eq!(decoded(), 7);
    describe_images(std::slice::from_ref(&other), Some(&processor), Some(&store)).unwrap();
    assert_eq!(decoded(), 8);
    describe_videos(&frames, Some(&processor), Some(&store), 0).unwrap();
    assert_eq!(decoded(), 8);
}

fn frontend() -> (crate::frontend::Frontend, u32, u32, u32) {
    use crate::frontend::{Asset, FrontendAssets, TemplateEncoding};
    use sha2::{Digest, Sha256};
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SERIAL: AtomicUsize = AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "flare-video-requests-{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let mut alphabet: Vec<_> =
        tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let vocabulary: tokenizers::models::bpe::Vocab = alphabet
        .into_iter()
        .enumerate()
        .map(|(id, character)| (character.to_string(), id as u32))
        .collect();
    let mut tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocabulary, vec![])
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, false),
    ));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer.with_post_processor(Some(tokenizers::processors::byte_level::ByteLevel::new(
        false, true, false,
    )));
    tokenizer
        .add_special_tokens(
            [
                "<|im_start|>",
                "<|im_end|>",
                "<|image_pad|>",
                "<|video_pad|>",
                "<|vision_start|>",
                "<|vision_end|>",
                "<|endoftext|>",
            ]
            .map(|text| tokenizers::AddedToken::from(text, true)),
        )
        .unwrap();
    let image = tokenizer.token_to_id("<|image_pad|>").unwrap();
    let video = tokenizer.token_to_id("<|video_pad|>").unwrap();
    let eos = tokenizer.token_to_id("<|endoftext|>").unwrap();
    let template = b"{% for m in messages %}<|im_start|>{{ m.role }}\n{% for part in m.content %}{% if part.type == 'text' %}{{ part.text }}{% elif part.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% elif part.type == 'video' %}<|vision_start|><|video_pad|><|vision_end|>{% endif %}{% endfor %}<|im_end|>\n{% mark %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    let asset = |name: &str, bytes: &[u8]| {
        let path = directory.join(name);
        std::fs::write(&path, bytes).unwrap();
        Asset {
            path,
            range: None,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        }
    };
    let frontend = crate::frontend::Frontend::load(
        FrontendAssets {
            tokenizer: asset(
                "tokenizer.json",
                tokenizer.to_string(false).unwrap().as_bytes(),
            ),
            tokens: Default::default(),
            chat_template: Some(asset("chat.jinja", template)),
        },
        serde_json::Map::new(),
    )
    .unwrap()
    .with_template_encoding(TemplateEncoding::Text);
    std::fs::remove_dir_all(directory).unwrap();
    (frontend, image, video, eos)
}

/// A Qwen-layout model whose tokenizer and template render image and video parts.
struct Fixture {
    frontend: crate::frontend::Frontend,
    config: crate::model::ModelConfig,
    compiler: crate::grammar::GrammarCompiler,
    image_token: u32,
    video_token: u32,
}

impl Fixture {
    fn new() -> Self {
        use crate::model::ModelConfig;
        use crate::model::contract::{ImageLayoutKind, ImageTokenLayout};
        use crate::scheduler::SchedulerLimits;
        let (frontend, image_token, video_token, eos) = frontend();
        let mut config = ModelConfig {
            completion_token_capacity: 16,
            max_top_logprobs: 0,
            logprob_token_id_capacity: 0,
            prepared_sequence_tokens: 2048,
            contract: crate::model::testing::metadata(),
            identity: [0; 32],
            model_id: "video-requests".into(),
            limits: SchedulerLimits {
                request_slots: 4,
                max_active: 4,
                max_sequence_tokens: 2048,
                prefill_chunk_size: 512,
                pages: 64,
                snapshots: 40,
            },
            eos_token_ids: vec![eos],
            sampling_ranges: Default::default(),
            sampling_defaults: Default::default(),
            max_new_tokens: 16,
            speculative_widths: vec![],
            speculative_fixed_width: None,
            speculative_token_capacity: 0,
            decode_weight_bytes: 0,
            history_drafts: false,
            speculative_proposal_source: Default::default(),
        };
        let vocabulary = frontend.tokenizer().get_vocab_size(true) as usize;
        config.contract.vocabulary = vocabulary;
        config.contract.input_vocabulary = vocabulary;
        config.contract.vision = crate::model::VisionCapabilities {
            feature_slots: 4,
            features_per_request: 64,
            max_images: 8,
            patches_per_feature: 4,
            patch_capacities: vec![256],
        };
        let syntax = &mut config.contract.frontend;
        syntax.image_layout = Some(ImageLayoutKind::Qwen);
        syntax.image_preparation = Some(crate::model::contract::ImagePreparationKind::Qwen);
        syntax.position_dimensions = 3;
        syntax.image_tokens = Some(ImageTokenLayout {
            feature_token: image_token,
            video_feature_token: Some(video_token),
            span: None,
        });
        let compiler = crate::grammar::GrammarCompiler::new(
            frontend.tokenizer(),
            vocabulary,
            &[eos],
            &[],
        )
        .unwrap();
        Self { frontend, config, compiler, image_token, video_token }
    }

    fn processor(&self, video: bool) -> ImageProcessor {
        qwen(
            self.image_token,
            self.video_token,
            ImageLimits {
                max_images: 64,
                max_patches_per_image: 256,
                max_total_patches: 4096,
                patches_per_feature: 4,
                ..ImageLimits::default()
            },
            video,
        )
    }

    fn prepare(
        &self,
        processor: Option<&ImageProcessor>,
        messages: Vec<Value>,
        generation: bool,
    ) -> Result<PreparedRequest> {
        super::prepare(
            &self.frontend,
            processor,
            &self.compiler,
            &self.config,
            json!({
                "model": self.config.model_id,
                "messages": messages,
                "max_tokens": 8,
                "chat_template_kwargs": {"add_generation_prompt": generation},
            }),
        )
    }
}

fn part(clip: &VideoPart) -> Value {
    json!({"type": "video", "video": clip.frames, "fps": clip.fps})
}

#[test]
fn video_parts_are_unexpected_content_items_without_prepared_video_settings() {
    let fixture = Fixture::new();
    let image = json!({"type": "image_url", "image_url": {"url": data_url(&png(64, 64, 9))}});
    let video = part(&clip(&[(64, 64); 2], 2.0));
    let image_only = fixture.processor(false);
    let moonvit = ImageProcessor::load(ImageSettings::Moonvit {}, ImageLimits::default()).unwrap();
    for processor in [None, Some(&image_only), Some(&moonvit)] {
        for parts in [
            vec![image.clone(), video.clone()],
            vec![json!({"type": "video"})],
        ] {
            let error = fixture
                .prepare(processor, vec![json!({"role": "user", "content": parts})], true)
                .err()
                .unwrap();
            assert_eq!(error.to_string(), "Unexpected item type in content.");
            assert!(matches!(error, Error::Admission(AdmissionError::Feature(_))));
        }
        // The request is rejected before its image is decoded.
        if let Some(processor) = processor {
            assert_eq!(processor.preparation_counts().decoded_images, 0);
        }
    }
    // Prepared video settings admit the part and report its shape.
    let with_video = fixture.processor(true);
    let error = fixture
        .prepare(
            Some(&with_video),
            vec![json!({"role": "user", "content": [{"type": "video"}]})],
            true,
        )
        .err()
        .unwrap();
    assert_eq!(error.to_string(), "video must be an array of image data URLs");
    assert!(
        fixture
            .prepare(Some(&with_video), vec![json!({"role": "user", "content": [image, video]})], true)
            .is_ok()
    );
}

#[test]
fn windows_of_an_odd_clip_take_each_admission_decode_once() {
    let fixture = Fixture::new();
    let processor = fixture.processor(true);
    let source = clip(&[(64, 64); 5], 2.0);
    let prepared = fixture
        .prepare(Some(&processor), vec![json!({"role": "user", "content": [part(&source)]})], true)
        .unwrap();
    let inputs = &prepared.prompt.conditioned_inputs;
    assert_eq!(inputs.len(), 3);
    assert_eq!(processor.preparation_counts().decoded_images, 5);
    assert!(processor.preparation_counts().decoded_source_bytes >= 5 * 64 * 64 * 3);
    let windows: Vec<_> = inputs.iter().map(|input| input.materialize().unwrap()).collect();
    // Window preparation takes the admission decodes, the repeated last frame once, and
    // releases each when its window is packed.
    let counts = processor.preparation_counts();
    assert_eq!((counts.decoded_images, counts.prepared_images), (5, 3));
    assert_eq!(counts.decoded_source_bytes, 0);
    // Each patch row holds, per channel, the first frame's slice and then the second's.
    let slices = |window: &crate::frontend::images::ProcessedImages| {
        let row = &window.pixel_values[..3 * 2 * 16 * 16];
        (row[..256].to_vec(), row[256..512].to_vec())
    };
    let (first, second) = slices(&windows[0]);
    assert_ne!(first, second);
    let (first, second) = slices(&windows[2]);
    assert_eq!(first, second);
    // A window prepared again finds no retained decode and decodes its two frames.
    inputs[0].materialize().unwrap();
    assert_eq!(processor.preparation_counts().decoded_images, 7);
    assert_eq!(processor.preparation_counts().decoded_source_bytes, 0);
}

#[test]
fn a_request_rejected_after_frame_decoding_releases_its_frames() {
    let mut fixture = Fixture::new();
    // The clip is admitted and its frames decoded; the expanded prompt exceeds the context.
    fixture.config.limits.max_sequence_tokens = 32;
    fixture.config.prepared_sequence_tokens = 32;
    let processor = fixture.processor(true);
    let source = clip(&[(64, 64); 8], 2.0);
    assert!(
        fixture
            .prepare(Some(&processor), vec![json!({"role": "user", "content": [part(&source)]})], true)
            .is_err()
    );
    let counts = processor.preparation_counts();
    assert_eq!(counts.decoded_images, 8);
    assert_eq!(counts.decoded_source_bytes, 0);
}

#[test]
fn image_checkpoint_hints_name_still_images_beside_clips() {
    use crate::frontend::images::VisionSource;
    use crate::server::protocol::{Protocol, decode_with};

    let fixture = Fixture::new();
    let processor = fixture.processor(true);
    let frame = data_url(&png(64, 64, 255));
    let body = json!({"max_tokens": 8, "messages": [{"role": "user", "content": [
        {"type": "video", "video": [frame, frame], "fps": 2.0},
        {"type": "image_url", "image_url": {"url": frame}},
        {"type": "text", "text": "after"}]}]});
    let mut request = decode_with(Protocol::Chat, body, false, true).unwrap().request;
    // The hint ends at the second content part of the first message: the still image,
    // the request's first image and its second vision window.
    request.checkpoint_hints = vec![(0, 2)];
    let prepared = prepare_inference(
        &fixture.frontend,
        Some(&processor),
        &fixture.compiler,
        &fixture.config,
        &request,
        0,
        false,
    )
    .unwrap();
    let inputs = &prepared.prompt.conditioned_inputs;
    assert_eq!(inputs.len(), 2);
    assert!(matches!(inputs[0].source, VisionSource::Frames { .. }));
    assert!(matches!(inputs[1].source, VisionSource::Image(_)));
    let boundaries = &prepared.prompt.message_boundaries;
    assert!(boundaries.contains(&inputs[1].token_span.end), "{boundaries:?}");
    assert!(!boundaries.contains(&inputs[0].token_span.end), "{boundaries:?}");
}

#[test]
fn video_expansions_keep_later_message_boundaries_after_their_messages() {
    let fixture = Fixture::new();
    let processor = fixture.processor(true);
    let frame = data_url(&png(64, 64, 255));
    let clip = || json!({"type": "video", "video": [frame, frame], "fps": 2.0});
    let image = json!({"type": "image_url", "image_url": {"url": frame}});
    let text = |text: &str| json!({"type": "text", "text": text});
    let cases = [
        (
            "clip in the first message",
            vec![
                json!({"role": "user", "content": [text("first"), clip()]}),
                json!({"role": "assistant", "content": [text("second")]}),
            ],
            vec![json!({"role": "user", "content": [text("first"), clip()]})],
        ),
        (
            "clip followed by text",
            vec![json!({"role": "user", "content": [clip(), text("after")]})],
            vec![json!({"role": "user", "content": [clip(), text("after")]})],
        ),
        (
            "two clips",
            vec![json!({"role": "user", "content": [clip(), clip()]})],
            vec![json!({"role": "user", "content": [clip(), clip()]})],
        ),
        (
            "image and clip",
            vec![json!({"role": "user", "content": [text("look"), image, clip()]})],
            vec![json!({"role": "user", "content": [text("look"), image, clip()]})],
        ),
    ];
    for (name, messages, prefix) in cases {
        let prepared = fixture.prepare(Some(&processor), messages, true).unwrap();
        let complete = fixture.prepare(Some(&processor), prefix, false).unwrap();
        assert!(
            prepared
                .prompt
                .message_boundaries
                .contains(&complete.prompt.tokens.len()),
            "{name}: {} is not a message boundary in {:?}",
            complete.prompt.tokens.len(),
            prepared.prompt.message_boundaries
        );
    }
}
