/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! RGB8 decoding with explicit model preprocessing profiles.

#[cfg(test)]
mod decoding_tests;
mod deepseek4;
mod deepseek_row_major;
#[cfg(feature = "server")]
pub(crate) mod layouts;
mod mimo;
mod glm;
mod moonvit;
mod orientation;
mod conditioned;
mod padded_grid;
mod record;
pub use conditioned::{FrameDescription, ImageDescription, ImageEncoding, ImageInput, ImageSource, PreparedImageInput,
    VideoFrame, VideoInput, VisionSource};
pub(crate) mod packing;
mod patches;
mod patch_merge;
mod qwen;
mod radio;
mod resize;
mod step;
mod video;

use std::fmt::Write as _;
use std::io::Cursor;

use super::Frontend;
use crate::{Error, Failure, FrontendError, Result};

const PATCH: usize = 16;
const MERGE: usize = 2;
const FEATURES: usize = 3 * 2 * PATCH * PATCH;
const IMAGE_TOKEN: &str = "<|image_pad|>";
const VIDEO_TOKEN: &str = "<|video_pad|>";

/// Admission limits for one request's images. Patch counts are the encoder's unit;
/// `patches_per_feature` patches merge into one image token, the unit that the model card
/// (`max_image_tokens_per_prompt`, `max_image_tokens_per_image`), usage reports, and
/// rejection messages use.
#[derive(Clone, Copy)]
pub struct ImageLimits {
    pub max_input_pixels: usize,
    pub max_encoded_bytes: usize,
    pub max_images: usize,
    pub max_patches_per_image: usize,
    pub max_total_patches: usize,
    pub patches_per_feature: usize,
    /// Frames a request may carry across all of its video clips.
    pub max_video_frames: usize,
}

impl Default for ImageLimits {
    fn default() -> Self {
        Self {
            max_input_pixels: 64 * 1024 * 1024,
            max_encoded_bytes: 32 * 1024 * 1024,
            max_images: 16_384,
            max_patches_per_image: 16_384,
            max_total_patches: 16_384,
            patches_per_feature: 1,
            max_video_frames: 16_384,
        }
    }
}

/// The first image whose cost crosses a request's image-token budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageOverrun {
    /// One-based position among the request's images.
    pub ordinal: usize,
    pub width: usize,
    pub height: usize,
    pub tokens: usize,
}

impl ImageLimits {
    /// Image tokens for `patches`, rounding partial features up.
    pub fn tokens(&self, patches: usize) -> usize {
        patches.div_ceil(self.patches_per_feature.max(1))
    }

    pub fn source_retention_bytes(&self) -> Result<u64> {
        (self.max_input_pixels as u64).checked_mul(8)
            .ok_or_else(|| "decoded image source capacity overflows".into())
    }

    pub fn max_image_tokens_per_prompt(&self) -> usize {
        self.max_total_patches / self.patches_per_feature.max(1)
    }

    pub fn max_image_tokens_per_image(&self) -> usize {
        self.max_patches_per_image / self.patches_per_feature.max(1)
    }

    /// Checks the frames a request carries across all of its video clips.
    pub(crate) fn check_video_count(&self, frames: usize) -> Result<()> {
        if frames > self.max_video_frames {
            return Err(Error::frontend(FrontendError::Input(Failure::new(format!(
                "request has {frames} video frames; max_video_frames_per_prompt is {}",
                self.max_video_frames
            )))));
        }
        Ok(())
    }

    pub(crate) fn check_batch(&self, images: usize, patches: usize) -> Result<()> {
        self.check_budget(images, patches, None)
    }

    /// Checks the image count and the whole request's patch total. `patches` must be the
    /// complete total, so the message reports what the request needs rather than a running
    /// sum; `overrun` names the image that crossed the budget when the caller tracked it.
    pub(crate) fn check_budget(
        &self,
        images: usize,
        patches: usize,
        overrun: Option<ImageOverrun>,
    ) -> Result<()> {
        let message = if images == 0 {
            "image input must contain at least one image".to_owned()
        } else if images > self.max_images {
            format!("request has {images} images; max_images_per_prompt is {}", self.max_images)
        } else if patches > self.max_total_patches {
            let mut message = format!(
                "request needs {} image tokens ({patches} patches); max_image_tokens_per_prompt is {} ({} patches)",
                self.tokens(patches),
                self.max_image_tokens_per_prompt(),
                self.max_total_patches
            );
            if let Some(overrun) = overrun {
                let _ = write!(
                    message,
                    "; image {} of {images} ({}x{} px, {} tokens) crosses the budget",
                    overrun.ordinal, overrun.width, overrun.height, overrun.tokens
                );
            }
            message
        } else {
            return Ok(());
        };
        Err(Error::frontend(FrontendError::Input(Failure::new(message))))
    }
}

#[derive(Clone, Debug)]
pub struct RgbImage {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
    alpha: Option<Vec<u8>>,
    cmyk: Option<Vec<u8>>,
    nearest_resize: bool,
}

impl RgbImage {
    /// Pixels are row-major RGB8, with no orientation or color-profile conversion.
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|size| size.checked_mul(3));
        if width == 0 || height == 0 || expected != Some(pixels.len()) {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "RGB8 dimensions do not match pixel extent",
            ))));
        }
        Ok(Self {
            width,
            height,
            pixels,
            alpha: None,
            cmyk: None,
            nearest_resize: false,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ImageGeometry {
    pub height: usize,
    pub width: usize,
    pub grid_thw: [i64; 3],
    pub patches: usize,
    pub tokens: usize,
}

/// CPU metadata in the packed patch order consumed by the vision graph.
#[derive(Clone, Debug, Default)]
pub struct ProcessedImages {
    pub pixel_values: Vec<f32>,
    pub geometry: Vec<ImageGeometry>,
    pub position_ids: Vec<[i64; 2]>,
    pub interpolation_indices: Vec<[i64; 4]>,
    pub interpolation_weights: Vec<[f32; 4]>,
    pub cu_seqlens: Vec<i32>,
}

impl ProcessedImages {
    fn reserve_outputs(&mut self, patches: usize, pixel_width: usize, interpolation: bool,
        images: usize, sequences: usize) -> Result<()> {
        let pixels = patches.checked_mul(pixel_width).ok_or("image pixel tensor extent overflows")?;
        self.pixel_values.reserve_exact(pixels.saturating_sub(self.pixel_values.len()));
        self.position_ids.reserve_exact(patches.saturating_sub(self.position_ids.len()));
        self.geometry.reserve_exact(images.saturating_sub(self.geometry.len()));
        self.cu_seqlens.reserve_exact(sequences.saturating_sub(self.cu_seqlens.len()));
        if interpolation {
            self.interpolation_indices.reserve_exact(patches.saturating_sub(self.interpolation_indices.len()));
            self.interpolation_weights.reserve_exact(patches.saturating_sub(self.interpolation_weights.len()));
        }
        Ok(())
    }
}

pub struct ImagePrompt {
    pub token_ids: Vec<u32>,
    pub mm_token_type_ids: Vec<i32>,
    /// Text, temporal, height, and width position rows, each of length token_ids.len().
    pub position_ids: [Vec<i64>; 4],
    pub rope_delta: i64,
}

/// A prompt whose image and video placeholders are expanded.
pub struct MediaPrompt {
    pub prompt: ImagePrompt,
    /// Vision windows in prompt order: one still image or one video temporal patch each.
    pub windows: Vec<PromptWindow>,
    /// Tokens that replace each media placeholder, in prompt order: the tokens of an
    /// image, or the timestamp markers, vision delimiters and tokens of a clip.
    pub expansions: Vec<usize>,
}

/// The prepared input that fills one vision window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptWindowSource {
    /// Index into the request's still images.
    Image(usize),
    /// Clip and temporal patch among the request's video clips.
    Video { clip: usize, patch: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptWindow {
    /// Expanded prompt tokens whose features this window produces.
    pub span: std::ops::Range<usize>,
    pub geometry: ImageGeometry,
    pub source: PromptWindowSource,
}

#[derive(Default)]
struct PreparationCounters {
    decoded: std::sync::atomic::AtomicU64,
    prepared: std::sync::atomic::AtomicU64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImagePreparationCounts {
    pub decoded_images: u64,
    pub prepared_images: u64,
    pub decoded_source_capacity: u64,
    pub decoded_source_bytes: u64,
}

#[derive(Clone)]
pub struct ImageProcessor {
    limits: ImageLimits,
    profile: Profile,
    identity_domain: [u8; 32],
    counts: std::sync::Arc<PreparationCounters>,
    decoded_budget: std::sync::Arc<conditioned::DecodedBudget>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ImageSettings {
    Qwen(QwenImageSettings),
    Mimo(MimoImageSettings),
    PatchMerge(PatchMergeImageSettings),
    Deepseek4 {},
    DeepseekRowMajor(DeepseekRowMajorImageSettings),
    Moonvit {},
    Step {},
    Radio {},
    Glm {},
    PaddedGrid {},
}

/// The text that stands for an image in a RADIO prompt.
pub(crate) const RADIO_PLACEHOLDER: &str = "<image>";

impl ImageSettings {
    /// The text that positions an image when a request spells it and the chat template wrote
    /// none, for layouts whose template lets message text place the images. It comes from the
    /// layout because cores prepared for RADIO declare no image placeholder token ID.
    pub fn spelled_placeholder(&self) -> Option<&'static str> {
        matches!(self, Self::Radio {}).then_some(RADIO_PLACEHOLDER)
    }

    pub fn preparation_kind(&self) -> crate::model::contract::ImagePreparationKind {
        use crate::model::contract::ImagePreparationKind;
        match self {
            Self::Qwen(_) => ImagePreparationKind::Qwen,
            Self::Mimo(_) => ImagePreparationKind::Mimo,
            Self::PatchMerge(_) => ImagePreparationKind::PatchMerge,
            Self::Deepseek4 {} => ImagePreparationKind::Deepseek,
            Self::DeepseekRowMajor(_) => ImagePreparationKind::DeepseekRowMajor,
            Self::Moonvit {} => ImagePreparationKind::Moonvit,
            Self::Step {} => ImagePreparationKind::Step,
            Self::Radio {} => ImagePreparationKind::Radio,
            Self::Glm {} => ImagePreparationKind::Glm,
            Self::PaddedGrid {} => ImagePreparationKind::PaddedGrid,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeepseekRowMajorImageSettings {
    pub min_pixels: usize,
    pub max_tokens: usize,
    pub max_width_height_ratio: Option<usize>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenImageSettings {
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub position_side: usize,
    pub image_token_id: u32,
    pub video_token_id: u32,
    /// The checkpoint's video processor settings. A core without them rejects video
    /// parts, and its settings serialize without this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<QwenVideoSettings>,
}

/// The checkpoint video processor's clip limits: the largest frame count, the pixel
/// budget of a whole clip, and the frames in one temporal patch.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenVideoSettings {
    pub max_frames: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub temporal_patch_size: usize,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimoImageSettings {
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchMergeImageSettings {
    pub patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub image_token: String,
    pub start_token: String,
    pub end_token: String,
}

#[derive(Clone)]
enum Profile {
    Qwen(qwen::QwenProcessor),
    MiMo(patches::PatchProcessor),
    PatchMerge(patch_merge::Processor),
    DeepSeek4,
    DeepseekRowMajor(DeepseekRowMajorImageSettings),
    Moonvit,
    Step,
    Radio,
    Glm,
    PaddedGrid,
}

impl ImageProcessor {
    pub fn load(settings: ImageSettings, limits: ImageLimits) -> Result<Self> {
        if [
            limits.max_input_pixels,
            limits.max_encoded_bytes,
            limits.max_images,
            limits.max_patches_per_image,
            limits.max_total_patches,
            limits.patches_per_feature,
        ]
        .contains(&0)
            || limits.max_total_patches > i32::MAX as usize
        {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image limits must be positive and total patches must fit int32",
            ))));
        }
        let decoded_budget = std::sync::Arc::new(conditioned::DecodedBudget::new(limits.source_retention_bytes()?));
        use sha2::{Digest, Sha256};
        // Video settings shape clip windows and enter window identities through the
        // window geometry. Image and frame identities depend on the image settings alone.
        let identity_settings = match &settings {
            ImageSettings::Qwen(qwen) if qwen.video.is_some() => {
                Some(ImageSettings::Qwen(QwenImageSettings { video: None, ..qwen.clone() }))
            }
            _ => None,
        };
        let identity_domain = Sha256::digest(
            serde_json::to_vec(identity_settings.as_ref().unwrap_or(&settings)).map_err(|e| e.to_string())?,
        )
        .into();
        let profile = match settings {
            ImageSettings::Deepseek4 {} => Profile::DeepSeek4,
            ImageSettings::DeepseekRowMajor(settings) => {
                deepseek_row_major::validate(&settings)?;
                Profile::DeepseekRowMajor(settings)
            }
            ImageSettings::Moonvit {} => Profile::Moonvit,
            ImageSettings::Step {} => Profile::Step,
            ImageSettings::Radio {} => Profile::Radio,
            ImageSettings::Glm {} => Profile::Glm,
            ImageSettings::PaddedGrid {} => Profile::PaddedGrid,
            ImageSettings::Mimo(settings) => Profile::MiMo(mimo::load(&settings, limits)?),
            ImageSettings::PatchMerge(settings) => Profile::PatchMerge(patch_merge::Processor::load(settings, limits)?),
            ImageSettings::Qwen(settings) => {
                Profile::Qwen(qwen::QwenProcessor::load(&settings, limits)?)
            }
        };
        Ok(Self { limits, profile, identity_domain, counts: Default::default(), decoded_budget })
    }

    /// Identity of the decoded representation consumed by this preparation profile.
    pub fn semantic_identity(&self, image: &RgbImage) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"flare-conditioned-image-v2\0");
        hash.update(self.identity_domain);
        hash.update((image.width as u64).to_le_bytes());
        hash.update((image.height as u64).to_le_bytes());
        hash.update(&image.pixels);
        // MoonViT resizes retained alpha and CMYK planes; the padded grid composites alpha.
        if matches!(self.profile, Profile::Moonvit | Profile::PaddedGrid) {
            hash.update([u8::from(image.nearest_resize)]);
            for channel in [&image.alpha, &image.cmyk] {
                hash.update([u8::from(channel.is_some())]);
                if let Some(channel) = channel {
                    hash.update((channel.len() as u64).to_le_bytes());
                    hash.update(channel);
                }
            }
        }
        hash.finalize().into()
    }

    /// Encoded source lookup is separate from semantic prefix identity.
    pub fn source_identity(&self, encoding: ImageEncoding, bytes: &[u8]) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"flare-encoded-image-v1\0");
        hash.update([match encoding { ImageEncoding::Png => 0, ImageEncoding::Jpeg => 1, ImageEncoding::Webp => 2, ImageEncoding::Gif => 3 }]);
        hash.update(self.identity_domain);
        hash.update(bytes);
        hash.finalize().into()
    }

    /// Input-store key of a video frame's description. It differs from the key of the
    /// same encoded source described as a still image, whose record has another form.
    pub fn frame_source_identity(&self, encoding: ImageEncoding, bytes: &[u8]) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"flare-encoded-frame-v1\0");
        hash.update(self.source_identity(encoding, bytes));
        hash.finalize().into()
    }

    pub fn process_one(&self, image: &RgbImage) -> Result<ProcessedImages> {
        let images = self.process(std::slice::from_ref(image))?;
        self.counts.prepared.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(images)
    }

    /// Whether the prepared settings carry the checkpoint's video processor settings.
    pub fn has_video_settings(&self) -> bool {
        self.video().is_ok()
    }

    /// The Qwen profile and its video processor when video settings are prepared.
    fn video(&self) -> Result<(&qwen::QwenProcessor, &video::VideoProcessor)> {
        match &self.profile {
            Profile::Qwen(profile) => profile.video.as_ref().map(|video| (profile, video)),
            _ => None,
        }
        .ok_or_else(|| {
            Error::frontend(FrontendError::Input(Failure::new(
                "prepared model does not support video inputs",
            )))
        })
    }

    /// Checks a clip's frame count and source frame rate against the prepared bounds.
    pub fn check_video_clip(&self, frames: usize, fps: f64) -> Result<()> {
        let (_, video) = self.video()?;
        video.check_frames(frames)?;
        video.timestamps(frames, fps).map(drop)
    }

    /// The number of temporal patches of a clip and the vision window each of them fills.
    pub fn video_windows(
        &self,
        frames: usize,
        width: usize,
        height: usize,
    ) -> Result<(usize, ImageGeometry)> {
        let window = self.video()?.1.window(frames, width, height)?;
        Ok((frames.div_ceil(video::TEMPORAL_PATCH), window))
    }

    /// Packs the two frames of one temporal patch into the window's vision inputs.
    pub fn process_video_window(
        &self,
        first: &RgbImage,
        second: &RgbImage,
        geometry: &ImageGeometry,
    ) -> Result<ProcessedImages> {
        let result = self.video()?.0.process_pair(first, second, geometry)?;
        self.counts.prepared.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(result)
    }

    /// Expands the image and video placeholders of a prompt that carries video clips.
    pub fn expand_media_prompt(
        &self,
        frontend: &Frontend,
        prompt_token_ids: &[u32],
        images: &ProcessedImages,
        videos: &[VideoInput],
    ) -> Result<MediaPrompt> {
        let (profile, video) = self.video()?;
        profile.expand_media_prompt(video, frontend, prompt_token_ids, images, videos)
    }

    /// Identity of one temporal patch: both frames' identities and the window geometry.
    pub fn video_window_identity(
        &self,
        first: [u8; 32],
        second: [u8; 32],
        geometry: &ImageGeometry,
    ) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"flare-video-window-v1\0");
        hash.update(first);
        hash.update(second);
        hash.update((geometry.width as u64).to_le_bytes());
        hash.update((geometry.height as u64).to_le_bytes());
        hash.update(geometry.grid_thw[1].to_le_bytes());
        hash.update(geometry.grid_thw[2].to_le_bytes());
        hash.finalize().into()
    }

    pub fn preparation_counts(&self) -> ImagePreparationCounts {
        use std::sync::atomic::Ordering::Relaxed;
        ImagePreparationCounts { decoded_images: self.counts.decoded.load(Relaxed),
            prepared_images: self.counts.prepared.load(Relaxed),
            decoded_source_capacity: self.decoded_budget.capacity,
            decoded_source_bytes: self.decoded_budget.used.load(Relaxed) }
    }

    fn check_input_extent(&self, width: usize, height: usize) -> Result<()> {
        check_input_extent(self.limits, width, height)
    }

    pub fn max_encoded_bytes(&self) -> usize {
        self.limits.max_encoded_bytes
    }

    pub fn limits(&self) -> ImageLimits {
        self.limits
    }

    pub(crate) fn check_batch(&self, images: usize, patches: usize) -> Result<()> {
        self.limits.check_batch(images, patches)
    }

    pub(crate) fn check_budget(
        &self,
        images: usize,
        patches: usize,
        overrun: Option<ImageOverrun>,
    ) -> Result<()> {
        self.limits.check_budget(images, patches, overrun)
    }

    pub(crate) fn check_video_count(&self, frames: usize) -> Result<()> {
        self.limits.check_video_count(frames)
    }

    /// Decodes PNG with the prepared profile’s alpha and orientation semantics.
    /// Animated PNG is rejected. 16-bit samples follow Pillow's RGB conversion:
    /// RGB, RGBA, and grayscale-alpha keep the high byte, grayscale clamps, and
    /// 16-bit tRNS keys are ignored.
    pub fn decode_png(&self, bytes: &[u8]) -> Result<RgbImage> {
        if bytes.len() > self.limits.max_encoded_bytes {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "encoded image exceeds the configured byte limit",
            ))));
        }
        let mut decoder = png::Decoder::new(Cursor::new(bytes));
        decoder.set_transformations(png::Transformations::EXPAND);
        decoder.set_limits(png::Limits {
            bytes: self.limits.max_input_pixels.saturating_mul(8),
        });
        let mut reader = decoder.read_info().map_err(|e| {
            Error::frontend(FrontendError::ImageDecode(Failure::caused_by(
                format!("invalid PNG: {e}"),
                e,
            )))
        })?;
        let info = reader.info();
        self.check_input_extent(info.width as usize, info.height as usize)?;
        if info.animation_control.is_some() {
            return Err(Error::frontend(FrontendError::ImageDecode(Failure::new(
                "animated PNG is not supported",
            ))));
        }
        let source_gray = info.color_type == png::ColorType::Grayscale;
        // Pillow ignores 16-bit tRNS keys: convert("RGBA") is opaque, so drop the alpha that
        // EXPAND derives from them.
        let drop_trns_alpha = info.bit_depth == png::BitDepth::Sixteen
            && info.trns.is_some()
            && matches!(
                info.color_type,
                png::ColorType::Rgb | png::ColorType::Grayscale
            );
        // The padded grid composites every 8-bit PNG transparency form that Pillow reports,
        // including RGB and grayscale tRNS keys. Pillow ignores 16-bit tRNS keys, and MoonViT
        // ignores RGB tRNS keys.
        let retain_alpha = match self.profile {
            Profile::Moonvit => info.color_type != png::ColorType::Rgb,
            Profile::PaddedGrid => true,
            _ => false,
        };
        let nearest =
            info.color_type == png::ColorType::Indexed || info.bit_depth == png::BitDepth::One;
        let size = reader.output_buffer_size().ok_or_else(|| {
            Error::frontend(FrontendError::ImageDecode(Failure::new(
                "PNG decoded extent overflow",
            )))
        })?;
        let mut decoded = vec![0; size];
        let output = reader.next_frame(&mut decoded).map_err(|e| {
            Error::frontend(FrontendError::ImageDecode(Failure::caused_by(
                format!("invalid PNG: {e}"),
                e,
            )))
        })?;
        reader.finish().map_err(|e| {
            Error::frontend(FrontendError::ImageDecode(Failure::caused_by(
                format!("invalid PNG: {e}"),
                e,
            )))
        })?;
        let orientation = if matches!(self.profile, Profile::Moonvit | Profile::DeepseekRowMajor(_) | Profile::PaddedGrid) {
            1
        } else {
            reader
                .info()
                .exif_metadata
                .as_deref()
                .map(orientation::exif_orientation)
                .transpose()?
                .unwrap_or(1)
        };
        let mut rgb = Vec::with_capacity(output.width as usize * output.height as usize * 3);
        let mut alpha = Vec::new();
        let sixteen = output.bit_depth == png::BitDepth::Sixteen;
        let sample_bytes = if sixteen { 2 } else { 1 };
        let clamp_gray = sixteen && source_gray;
        let sample = |pixel: &[u8], channel: usize| -> u8 {
            if sixteen {
                let value = u16::from_be_bytes([pixel[channel * 2], pixel[channel * 2 + 1]]);
                if clamp_gray {
                    value.min(255) as u8
                } else {
                    pixel[channel * 2]
                }
            } else {
                pixel[channel]
            }
        };
        for pixel in decoded[..output.buffer_size()]
            .chunks_exact(output.color_type.samples() * sample_bytes)
        {
            if !drop_trns_alpha
                && matches!(
                    output.color_type,
                    png::ColorType::Rgba | png::ColorType::GrayscaleAlpha
                )
            {
                let channel = output.color_type.samples() - 1;
                alpha.push(if sixteen {
                    pixel[channel * 2]
                } else {
                    pixel[channel]
                });
            }
            match output.color_type {
                png::ColorType::Rgb | png::ColorType::Rgba => {
                    for channel in 0..3 {
                        rgb.push(sample(pixel, channel));
                    }
                }
                png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha => {
                    rgb.extend_from_slice(&[sample(pixel, 0); 3]);
                }
                png::ColorType::Indexed => {
                    return Err(Error::frontend(FrontendError::ImageDecode(Failure::new(
                        "PNG palette was not expanded",
                    ))));
                }
            }
        }
        let mut image = RgbImage::new(output.width as usize, output.height as usize, rgb)?;
        image.alpha = (retain_alpha && !alpha.is_empty()).then_some(alpha);
        if matches!(self.profile, Profile::Moonvit) {
            image.nearest_resize = nearest;
        }
        Ok(orientation::orient(image, orientation))
    }

    /// Decodes JPEG with accurate IDCT and smooth chroma upsampling, then applies EXIF.
    /// Color profiles are not applied, matching Pillow's RGB conversion.
    pub fn decode_jpeg(&self, bytes: &[u8]) -> Result<RgbImage> {
        if bytes.len() > self.limits.max_encoded_bytes {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "encoded image exceeds the configured byte limit",
            ))));
        }
        let orientation = if matches!(self.profile, Profile::Moonvit | Profile::DeepseekRowMajor(_) | Profile::PaddedGrid) {
            1
        } else {
            orientation::jpeg_orientation(bytes)?
        };
        let error = |e| {
            Error::frontend(FrontendError::ImageDecode(Failure::caused_by(
                format!("invalid JPEG: {e}"),
                e,
            )))
        };
        let mut decoder = turbojpeg::Decompressor::new().map_err(error)?;
        decoder.set_scan_limit(100).map_err(error)?;
        decoder.set_fast_upsample(false).map_err(error)?;
        let header = decoder.read_header(bytes).map_err(error)?;
        self.check_input_extent(header.width, header.height)?;
        if header.is_lossless || header.is_arithmetic {
            return Err(Error::frontend(FrontendError::ImageDecode(Failure::new(
                "JPEG requires baseline or progressive Huffman coding",
            ))));
        }
        let cmyk = matches!(
            header.colorspace,
            turbojpeg::Colorspace::CMYK | turbojpeg::Colorspace::YCCK
        );
        let channels = if cmyk { 4 } else { 3 };
        let length = header
            .width
            .checked_mul(header.height)
            .and_then(|n| n.checked_mul(channels))
            .ok_or_else(|| {
                Error::frontend(FrontendError::ImageDecode(Failure::new(
                    "JPEG decoded extent overflow",
                )))
            })?;
        let mut decoded = turbojpeg::Image {
            pixels: vec![0; length],
            width: header.width,
            height: header.height,
            pitch: header.width * channels,
            format: if cmyk {
                turbojpeg::PixelFormat::CMYK
            } else {
                turbojpeg::PixelFormat::RGB
            },
        };
        decoder
            .decompress(bytes, decoded.as_deref_mut())
            .map_err(error)?;
        let retained_cmyk = (cmyk && matches!(self.profile, Profile::Moonvit))
            .then(|| decoded.pixels.iter().map(|value| 255 - value).collect());
        let rgb = if cmyk {
            // JPEG CMYK samples are inverted; this matches Pillow's CMYK-to-RGB conversion.
            decoded
                .pixels
                .chunks_exact(4)
                .flat_map(|pixel| {
                    pixel[..3].iter().map(|channel| {
                        ((u32::from(*channel) * u32::from(pixel[3]) + 127) / 255) as u8
                    })
                })
                .collect()
        } else {
            decoded.pixels
        };
        let mut image = RgbImage::new(header.width, header.height, rgb)?;
        image.cmyk = retained_cmyk;
        Ok(orientation::orient(image, orientation))
    }

    /// Decodes static WebP with the prepared profile's alpha and orientation semantics.
    pub fn decode_webp(&self, bytes: &[u8]) -> Result<RgbImage> {
        if bytes.len() > self.limits.max_encoded_bytes {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "encoded image exceeds the configured byte limit",
            ))));
        }
        let error = |e| {
            Error::frontend(FrontendError::ImageDecode(Failure::caused_by(
                format!("invalid WebP: {e}"),
                e,
            )))
        };
        let mut decoder = image_webp::WebPDecoder::new(Cursor::new(bytes)).map_err(error)?;
        let (width, height) = decoder.dimensions();
        self.check_input_extent(width as usize, height as usize)?;
        decoder.set_memory_limit(self.limits.max_input_pixels.saturating_mul(8));
        if decoder.is_animated() {
            return Err(Error::frontend(FrontendError::ImageDecode(Failure::new(
                "animated WebP is unsupported; provide a still image",
            ))));
        }
        let orientation = if matches!(self.profile, Profile::Moonvit | Profile::DeepseekRowMajor(_) | Profile::PaddedGrid) {
            1
        } else {
            decoder
                .exif_metadata()
                .map_err(error)?
                .as_deref()
                .map(|data| orientation::exif_orientation(data.strip_prefix(b"Exif\0\0").unwrap_or(data)))
                .transpose()?
                .unwrap_or(1)
        };
        let size = decoder.output_buffer_size().ok_or_else(|| {
            Error::frontend(FrontendError::ImageDecode(Failure::new(
                "WebP decoded extent overflow",
            )))
        })?;
        let mut decoded = vec![0; size];
        decoder.read_image(&mut decoded).map_err(error)?;
        let (rgb, alpha) = if decoder.has_alpha() {
            let rgb = decoded.chunks_exact(4).flat_map(|pixel| pixel[..3].iter().copied()).collect();
            let alpha = matches!(self.profile, Profile::Moonvit | Profile::PaddedGrid)
                .then(|| decoded.chunks_exact(4).map(|pixel| pixel[3]).collect());
            (rgb, alpha)
        } else {
            (decoded, None)
        };
        let mut image = RgbImage::new(width as usize, height as usize, rgb)?;
        image.alpha = alpha;
        Ok(orientation::orient(image, orientation))
    }

    /// Decodes the first frame of a non-animated GIF as Pillow's RGB and RGBA conversions do.
    /// The frame sits on a canvas that covers the logical screen and the frame; pixels outside
    /// the frame take the transparent index, or index 0 without one. GIF has no EXIF orientation.
    pub fn decode_gif(&self, bytes: &[u8]) -> Result<RgbImage> {
        if bytes.len() > self.limits.max_encoded_bytes {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "encoded image exceeds the configured byte limit",
            ))));
        }
        let error = |e| {
            Error::frontend(FrontendError::ImageDecode(Failure::caused_by(
                format!("invalid GIF: {e}"),
                e,
            )))
        };
        // The decoder rejects a frame without a color table; Pillow reads such a frame as gray
        // levels. When the screen descriptor's packed byte (offset 10) lacks the global-table
        // flag 0x80, a copy gains a 256-entry (size bits 7) identity gray table after the
        // 13-byte header and screen descriptor.
        let patched;
        let bytes = if bytes.len() >= 13 && bytes[10] & 0x80 == 0 {
            let mut with_table = bytes[..10].to_vec();
            with_table.push(bytes[10] | 0x87);
            with_table.extend_from_slice(&bytes[11..13]);
            (0..=255u8).for_each(|level| with_table.extend_from_slice(&[level; 3]));
            with_table.extend_from_slice(&bytes[13..]);
            patched = with_table;
            &patched[..]
        } else {
            bytes
        };
        let mut options = gif::DecodeOptions::new();
        options.set_color_output(gif::ColorOutput::Indexed);
        let mut decoder = options.read_info(Cursor::new(bytes)).map_err(error)?;
        let (screen_width, screen_height) = (usize::from(decoder.width()), usize::from(decoder.height()));
        let global_table = decoder.global_palette().unwrap_or_default().to_vec();
        let Some(frame) = decoder.next_frame_info().map_err(error)? else {
            return Err(Error::frontend(FrontendError::ImageDecode(Failure::new(
                "invalid GIF: no image data",
            ))));
        };
        let (left, top) = (usize::from(frame.left), usize::from(frame.top));
        let (frame_width, frame_height) = (usize::from(frame.width), usize::from(frame.height));
        let (interlaced, transparent, local_table) = (frame.interlaced, frame.transparent, frame.palette.clone());
        // Rejects an empty frame, which the canvas extent admits. A zero-size screen is valid:
        // the canvas then has the frame's size.
        self.check_input_extent(frame_width, frame_height)?;
        let (width, height) = (screen_width.max(left + frame_width), screen_height.max(top + frame_height));
        self.check_input_extent(width, height)?;
        // The frame is read in one `fill_buffer` call, as the crate's non-interlaced read does,
        // which yields rows in stored order. Reading a frame in several calls can end it early
        // (a flat 31x19 frame read 31 pixels per call stops after 496 pixels), and the crate's
        // interlaced read takes one row per call, so interlaced rows are placed below.
        let mut frame_indices = vec![0; frame_width * frame_height];
        if !decoder.fill_buffer(&mut frame_indices).map_err(error)? {
            return Err(Error::frontend(FrontendError::ImageDecode(Failure::new(
                "invalid GIF: image truncated",
            ))));
        }
        // A readable second image makes the file animated. Pillow reads only the first frame,
        // so a missing trailer, a damaged second image or other bytes after it are accepted.
        if matches!(decoder.next_frame_info(), Ok(Some(_))) {
            return Err(Error::frontend(FrontendError::ImageDecode(Failure::new(
                "animated GIF is not supported; provide a still image",
            ))));
        }
        let mut indices = vec![transparent.unwrap_or(0); width * height];
        // An interlaced frame stores rows 0, 8, ...; 4, 12, ...; 2, 6, ...; then the odd rows.
        let rows: Vec<usize> = if interlaced {
            [(0, 8), (4, 8), (2, 4), (1, 2)]
                .into_iter()
                .flat_map(|(first, step)| (first..frame_height).step_by(step))
                .collect()
        } else {
            (0..frame_height).collect()
        };
        for (stored, row) in rows.into_iter().enumerate() {
            let start = (top + row) * width + left;
            indices[start..start + frame_width]
                .copy_from_slice(&frame_indices[stored * frame_width..(stored + 1) * frame_width]);
        }
        // Pillow (GifImagePlugin `_is_palette_needed` and `_seek`) reads a frame as mode P only
        // when its color table is not the identity gray ramp; otherwise the mode is L and a
        // pixel's level is its index. A local identity table over a global table still takes
        // the global colors. Pillow's resize samples only mode P images with nearest neighbors.
        let needed = |table: &[u8]| table.chunks_exact(3).enumerate().any(|(i, c)| c.iter().any(|v| usize::from(*v) != i));
        let global = needed(&global_table).then_some(global_table);
        let local = local_table.as_ref().map(|table| needed(table).then_some(table));
        let is_palette = match &local {
            Some(local) => local.is_some(),
            None => global.is_some(),
        };
        let table = local.flatten().or(global.as_ref()).map(|table| table.as_slice());
        let mut rgb = Vec::with_capacity(indices.len() * 3);
        for index in &indices {
            match table {
                Some(table) => {
                    let start = usize::from(*index) * 3;
                    rgb.extend_from_slice(table.get(start..start + 3).unwrap_or(&[0; 3]));
                }
                None => rgb.extend_from_slice(&[*index; 3]),
            }
        }
        let alpha = transparent
            .filter(|_| matches!(self.profile, Profile::Moonvit | Profile::PaddedGrid))
            .map(|transparent| {
                indices.iter().map(|index| if *index == transparent { 0 } else { 255 }).collect()
            });
        let mut image = RgbImage::new(width, height, rgb)?;
        image.alpha = alpha;
        image.nearest_resize = is_palette && matches!(self.profile, Profile::Moonvit);
        Ok(image)
    }

    pub fn geometry(&self, width: usize, height: usize) -> Result<ImageGeometry> {
        match &self.profile {
            Profile::Qwen(profile) => profile.patches.geometry(width, height),
            Profile::MiMo(profile) => profile.geometry(width, height),
            Profile::PatchMerge(profile) => profile.patches.geometry(width, height),
            Profile::DeepSeek4 => deepseek4::geometry(self.limits, width, height),
            Profile::DeepseekRowMajor(settings) => deepseek_row_major::geometry(settings, self.limits, width, height),
            Profile::Moonvit => moonvit::geometry(self.limits, width, height),
            Profile::Step => step::geometry(self.limits, width, height),
            Profile::Radio => radio::geometry(self.limits, width, height),
            Profile::Glm => glm::geometry(self.limits, width, height),
            Profile::PaddedGrid => padded_grid::geometry(self.limits, width, height),
        }
    }
    pub fn resized_rgb(&self, image: &RgbImage) -> Result<(ImageGeometry, Vec<u8>)> {
        match &self.profile {
            Profile::Qwen(profile) => profile.patches.resized_rgb(image),
            Profile::MiMo(profile) => profile.resized_rgb(image),
            Profile::PatchMerge(profile) => profile.patches.resized_rgb(image),
            Profile::DeepSeek4 => deepseek4::resized_rgb(self.limits, image),
            Profile::DeepseekRowMajor(settings) => deepseek_row_major::resized_rgb(settings, self.limits, image),
            Profile::Moonvit => moonvit::resized_rgb(self.limits, image),
            Profile::Step => step::resized_rgb(self.limits, image),
            Profile::Radio => Err("RADIO resizing produces float32 pixels; use process".into()),
            Profile::Glm => glm::resized_rgb(self.limits, image),
            Profile::PaddedGrid => Err("padded-grid preparation pads with values outside RGB8; use process".into()),
        }
    }
    pub fn process(&self, images: &[RgbImage]) -> Result<ProcessedImages> {
        match &self.profile {
            Profile::Qwen(profile) => profile.process(images),
            Profile::MiMo(profile) => profile.process(images),
            Profile::PatchMerge(profile) => profile.patches.process(images),
            Profile::DeepSeek4 => deepseek4::process(self.limits, images),
            Profile::DeepseekRowMajor(settings) => deepseek4::pack(
                self.limits,
                images,
                |image| deepseek_row_major::geometry(settings, self.limits, image.width, image.height),
                |image| deepseek_row_major::resized_rgb(settings, self.limits, image),
            ),
            Profile::Moonvit => moonvit::process(self.limits, images),
            Profile::Step => step::process(self.limits, images),
            Profile::Radio => radio::process(self.limits, images),
            Profile::Glm => glm::process(self.limits, images),
            Profile::PaddedGrid => padded_grid::process(self.limits, images),
        }
    }
    pub fn expand_prompt(
        &self,
        frontend: &Frontend,
        prompt_token_ids: &[u32],
        images: &ProcessedImages,
    ) -> Result<ImagePrompt> {
        match &self.profile {
            Profile::Qwen(profile) => profile.expand_prompt(frontend, prompt_token_ids, images),
            Profile::PatchMerge(profile) => {
                profile.expand_prompt(frontend, prompt_token_ids, images)
            }
            Profile::DeepSeek4 | Profile::DeepseekRowMajor(_) => Err("DeepSeek image spans require its model contract".into()),
            Profile::MiMo(_) => Err("MiMo image expansion requires its model contract".into()),
            Profile::Moonvit => Err("MoonViT image expansion requires its model contract".into()),
            Profile::Step => Err("Step image expansion requires its model contract".into()),
            Profile::Radio => Err("RADIO image expansion requires its model contract".into()),
            Profile::Glm => Err("GLM image expansion requires its prepared layout".into()),
            Profile::PaddedGrid => Err("padded-grid image expansion requires its prepared layout".into()),
        }
    }
}
fn check_input_extent(limits: ImageLimits, width: usize, height: usize) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(Error::frontend(FrontendError::Input(Failure::new(
            format!("image is {width}x{height} px; width and height must be positive"),
        ))));
    }
    if width
        .checked_mul(height)
        .is_none_or(|pixels| pixels > limits.max_input_pixels)
    {
        return Err(Error::frontend(FrontendError::Input(Failure::new(format!(
            "image is {width}x{height} px; max_image_pixels is {}",
            limits.max_input_pixels
        )))));
    }
    Ok(())
}

/// The per-image rejection shared by every profile: the processed image needs more patches
/// than one prepared vision graph holds.
fn check_image_patches(limits: ImageLimits, patches: usize) -> Result<()> {
    if patches > limits.max_patches_per_image {
        return Err(Error::frontend(FrontendError::Input(Failure::new(format!(
            "image needs {} image tokens ({patches} patches); max_image_tokens_per_image is {} ({} patches)",
            limits.tokens(patches),
            limits.max_image_tokens_per_image(),
            limits.max_patches_per_image
        )))));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn processor(limits: ImageLimits) -> ImageProcessor {
        ImageProcessor::load(
            ImageSettings::Qwen(QwenImageSettings {
                min_pixels: 65_536,
                max_pixels: 16_777_216,
                position_side: 48,
                image_token_id: 248056,
                video_token_id: 248057,
                video: None,
            }),
            limits,
        )
        .unwrap()
    }

    fn png(
        width: u32,
        height: u32,
        depth: png::BitDepth,
        color: png::ColorType,
        data: &[u8],
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, width, height);
            encoder.set_depth(depth);
            encoder.set_color(color);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(data).unwrap();
        }
        bytes
    }

    #[test]
    fn video_settings_stay_out_of_image_identities() {
        // The serialized settings are the identity domain of semantic identities, which
        // enter the prefix hash, and of encoded-source keys in the input store.
        let settings = |video| {
            ImageSettings::Qwen(QwenImageSettings {
                min_pixels: 65_536,
                max_pixels: 16_777_216,
                position_side: 48,
                image_token_id: 248_056,
                video_token_id: 248_057,
                video,
            })
        };
        assert_eq!(
            serde_json::to_string(&settings(None)).unwrap(),
            r#"{"qwen":{"min_pixels":65536,"max_pixels":16777216,"position_side":48,"image_token_id":248056,"video_token_id":248057}}"#
        );
        let image = ImageProcessor::load(settings(None), ImageLimits::default()).unwrap();
        let video = ImageProcessor::load(
            settings(Some(QwenVideoSettings {
                max_frames: 768,
                min_pixels: 4_096,
                max_pixels: 25_165_824,
                temporal_patch_size: 2,
            })),
            ImageLimits::default(),
        )
        .unwrap();
        assert!(video.has_video_settings() && !image.has_video_settings());
        let pixels = RgbImage::new(2, 2, (0..12).collect()).unwrap();
        assert_eq!(video.semantic_identity(&pixels), image.semantic_identity(&pixels));
        assert_eq!(
            video.source_identity(ImageEncoding::Png, b"source"),
            image.source_identity(ImageEncoding::Png, b"source")
        );
    }

    #[test]
    fn malformed_png_retains_decoder_cause() {
        use std::error::Error as _;

        let processor = processor(ImageLimits::default());
        let error = processor.decode_png(b"not a PNG").err().unwrap();
        assert!(matches!(
            error,
            Error::Frontend(FrontendError::ImageDecode(_))
        ));
        assert!(
            error
                .source()
                .unwrap()
                .downcast_ref::<png::DecodingError>()
                .is_some()
        );
        let input = processor.geometry(0, 1).unwrap_err();
        assert!(matches!(input, Error::Frontend(FrontendError::Input(_))));
        assert!(input.source().is_none());
    }

    #[test]
    fn rgb_extent_rejects_empty_truncated_and_overflow() {
        assert!(RgbImage::new(0, 1, vec![]).is_err());
        assert!(RgbImage::new(2, 2, vec![0; 11]).is_err());
        assert!(RgbImage::new(usize::MAX, 2, vec![]).is_err());
    }

    #[test]
    fn admission_checks_processed_capacity_without_changing_resize() {
        let processor = processor(ImageLimits::default());
        assert_eq!(processor.geometry(2033, 2033).unwrap().patches, 16_384);
        assert!(matches!(
            processor.geometry(2080, 2080).unwrap_err(),
            Error::Frontend(FrontendError::Input(_))
        ));
        assert!(processor.geometry(200, 1).is_ok());
        assert!(matches!(
            processor.geometry(201, 1).unwrap_err(),
            Error::Frontend(FrontendError::Input(_))
        ));
        assert!(processor.geometry(usize::MAX, 2).is_err());
        assert!(processor.geometry(0, 256).is_err());
    }

    #[test]
    fn image_profiles_accept_five_images_within_the_total_patch_budget() {
        let settings = [
            ImageSettings::Qwen(QwenImageSettings { min_pixels: 65536, max_pixels: 16777216,
                position_side: 48, image_token_id: 248056, video_token_id: 248057, video: None }),
            ImageSettings::Mimo(MimoImageSettings { min_pixels: 65536, max_pixels: 16777216,
                image_mean: [0.5; 3], image_std: [0.5; 3] }),
            ImageSettings::Deepseek4 {},
            ImageSettings::DeepseekRowMajor(DeepseekRowMajorImageSettings { min_pixels: 46656,
                max_tokens: 1024, max_width_height_ratio: None }),
            ImageSettings::Moonvit {}, ImageSettings::Step {}, ImageSettings::Radio {},
        ];
        for settings in settings {
            let limits = ImageLimits { max_patches_per_image: 65536, max_total_patches: 65536,
                ..ImageLimits::default() };
            let processor = ImageProcessor::load(settings.clone(), limits).unwrap();
            let patches = processor.geometry(64, 64).unwrap().patches;
            let processor = ImageProcessor::load(settings, ImageLimits {
                max_total_patches: 5 * patches, ..limits }).unwrap();
            let images: Vec<_> = (0..6).map(|_| RgbImage::new(64, 64, vec![127; 64 * 64 * 3]).unwrap()).collect();
            let processed = processor.process(&images[..5]).unwrap();
            assert_eq!(processed.geometry.len(), 5);
            assert_eq!(processed.geometry.iter().map(|g| g.patches).sum::<usize>(), 5 * patches);
            let message = processor.process(&images).err().unwrap().to_string();
            assert!(message.contains("max_image_tokens_per_prompt is"), "{message}");
            assert!(message.contains(&format!("request needs {} image tokens ({} patches)",
                limits.tokens(6 * patches), 6 * patches)), "{message}");
        }
    }

    #[test]
    fn image_count_and_total_patch_limits_are_independent() {
        let image = || RgbImage::new(1, 1, vec![0; 3]).unwrap();
        let processor = processor(ImageLimits {
            max_images: 2,
            max_total_patches: 256,
            ..ImageLimits::default()
        });
        assert!(processor.process(&[]).is_err());
        assert!(matches!(
            processor
                .process(&[image(), image(), image()])
                .err()
                .unwrap(),
            Error::Frontend(FrontendError::Input(_))
        ));
        assert!(matches!(
            processor.process(&[image(), image()]).err().unwrap(),
            Error::Frontend(FrontendError::Input(_))
        ));
        assert_eq!(
            processor.process(&[image()]).unwrap().pixel_values.len(),
            256 * FEATURES
        );
    }

    #[test]
    fn png_limits_apply_before_allocating_decoded_pixels() {
        let bytes = png(2, 1, png::BitDepth::Eight, png::ColorType::Rgb, &[1; 6]);
        let input_limit = processor(ImageLimits {
            max_input_pixels: 1,
            ..ImageLimits::default()
        });
        assert!(matches!(
            input_limit.decode_png(&bytes).err().unwrap(),
            Error::Frontend(FrontendError::Input(_))
        ));
        let byte_limit = processor(ImageLimits {
            max_encoded_bytes: bytes.len() - 1,
            ..ImageLimits::default()
        });
        assert!(matches!(
            byte_limit.decode_png(&bytes).err().unwrap(),
            Error::Frontend(FrontendError::Input(_))
        ));
    }

    #[test]
    fn png_drops_alpha_without_blending_and_rejects_truncated() {
        let processor = processor(ImageLimits::default());
        let bytes = png(
            2,
            1,
            png::BitDepth::Eight,
            png::ColorType::Rgba,
            &[5, 6, 7, 0, 8, 9, 10, 127],
        );
        assert_eq!(
            processor.decode_png(&bytes).unwrap().pixels(),
            &[5, 6, 7, 8, 9, 10]
        );
        assert!(processor.decode_png(&bytes[..bytes.len() - 12]).is_err());
    }
}
