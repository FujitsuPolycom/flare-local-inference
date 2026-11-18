/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Qwen image profile, learned position interpolation and multimodal prompt positions.

use super::*;

const VISION_START_TOKEN: &str = "<|vision_start|>";
const VISION_END_TOKEN: &str = "<|vision_end|>";

#[derive(Clone)]
pub(super) struct QwenProcessor {
    pub(super) patches: patches::PatchProcessor,
    pub(super) position_side: usize,
    pub(super) image_token_id: u32,
    pub(super) video_token_id: u32,
    pub(super) video: Option<video::VideoProcessor>,
}

impl QwenProcessor {
    pub(super) fn load(settings: &QwenImageSettings, limits: ImageLimits) -> Result<Self> {
        if settings.min_pixels == 0
            || settings.max_pixels < settings.min_pixels
            || settings.position_side != 48
            || settings.image_token_id == settings.video_token_id
        {
            return Err("invalid prepared Qwen image settings".into());
        }
        let video = settings
            .video
            .as_ref()
            .map(|video| video::VideoProcessor::load(video, limits))
            .transpose()?;
        Ok(Self {
            patches: patches::PatchProcessor {
                limits,
                patch_size: PATCH,
                min_pixels: settings.min_pixels,
                max_pixels: settings.max_pixels,
                normalization: patches::Normalization::Symmetric,
                resize: resize::resize,
            },
            position_side: settings.position_side,
            image_token_id: settings.image_token_id,
            video_token_id: settings.video_token_id,
            video,
        })
    }

    pub fn process(&self, images: &[RgbImage]) -> Result<ProcessedImages> {
        let mut result = self.patches.process(images)?;
        self.fill_interpolation(&mut result);
        Ok(result)
    }

    /// One temporal patch of a video clip: the vision window of a still image with one
    /// frame in each temporal slice.
    pub fn process_pair(
        &self,
        first: &RgbImage,
        second: &RgbImage,
        geometry: &ImageGeometry,
    ) -> Result<ProcessedImages> {
        let mut result = self.patches.pack_pair(first, second, geometry)?;
        self.fill_interpolation(&mut result);
        Ok(result)
    }

    fn fill_interpolation(&self, result: &mut ProcessedImages) {
        let side = self.position_side;
        let mut start = 0;
        result
            .interpolation_indices
            .reserve_exact(result.position_ids.len());
        result
            .interpolation_weights
            .reserve_exact(result.position_ids.len());
        for geometry in &result.geometry {
            let grid_h = geometry.height / PATCH;
            let grid_w = geometry.width / PATCH;
            for &[row, col] in &result.position_ids[start..start + geometry.patches] {
                let (yi, yw) = interpolation_axis(row as usize, grid_h, side);
                let (xi, xw) = interpolation_axis(col as usize, grid_w, side);
                result.interpolation_indices.push([
                    (yi[0] * side + xi[0]) as i64,
                    (yi[0] * side + xi[1]) as i64,
                    (yi[1] * side + xi[0]) as i64,
                    (yi[1] * side + xi[1]) as i64,
                ]);
                result.interpolation_weights.push([
                    yw[0] * xw[0],
                    yw[0] * xw[1],
                    yw[1] * xw[0],
                    yw[1] * xw[1],
                ]);
            }
            start += geometry.patches;
        }
    }

    /// Expands the image placeholders in already rendered chat text, in image order.
    /// Inputs contain no padding; the scheduler adds padding outside logical lengths.
    pub fn expand_prompt(
        &self,
        frontend: &Frontend,
        text: &str,
        images: &ProcessedImages,
    ) -> Result<ImagePrompt> {
        if frontend.encode(IMAGE_TOKEN, false)? != [self.image_token_id] {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image token ID does not match tokenizer",
            ))));
        }
        if text.matches(IMAGE_TOKEN).count() != images.geometry.len() {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image placeholder count does not match prepared images",
            ))));
        }
        let mut expanded = String::with_capacity(text.len());
        let mut parts = text.split(IMAGE_TOKEN);
        expanded.push_str(parts.next().unwrap_or_default());
        for (part, geometry) in parts.zip(&images.geometry) {
            for _ in 0..geometry.tokens {
                expanded.push_str(IMAGE_TOKEN);
            }
            expanded.push_str(part);
        }
        let token_ids = frontend.encode(&expanded, false)?;
        if token_ids.contains(&self.video_token_id) {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "video inputs are not supported by the image processor",
            ))));
        }
        let mm_token_type_ids: Vec<i32> = token_ids
            .iter()
            .map(|id| i32::from(*id == self.image_token_id))
            .collect();
        let mut positions: [Vec<i64>; 4] =
            std::array::from_fn(|_| Vec::with_capacity(token_ids.len()));
        positions[0].extend((0..token_ids.len()).map(|index| index as i64));
        let (mut offset, mut current, mut image_index) = (0, 0i64, 0);
        while offset < token_ids.len() {
            let modality = mm_token_type_ids[offset];
            let end = offset
                + mm_token_type_ids[offset..]
                    .iter()
                    .take_while(|value| **value == modality)
                    .count();
            if modality == 0 {
                for row in &mut positions[1..] {
                    row.extend((0..end - offset).map(|index| current + index as i64));
                }
                current += (end - offset) as i64;
            } else {
                let geometry = images.geometry.get(image_index).ok_or_else(|| {
                    Error::frontend(FrontendError::Input(Failure::new(
                        "unexpected image token group",
                    )))
                })?;
                if end - offset != geometry.tokens {
                    return Err(Error::frontend(FrontendError::Input(Failure::new(
                        "image token group does not match image grid",
                    ))));
                }
                let (height, width) = (
                    geometry.grid_thw[1] / MERGE as i64,
                    geometry.grid_thw[2] / MERGE as i64,
                );
                for y in 0..height {
                    for x in 0..width {
                        positions[1].push(current);
                        positions[2].push(current + y);
                        positions[3].push(current + x);
                    }
                }
                current += height.max(width);
                image_index += 1;
            }
            offset = end;
        }
        if image_index != images.geometry.len() {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "prepared image was not consumed by prompt",
            ))));
        }
        Ok(ImagePrompt {
            text: expanded,
            rope_delta: current - token_ids.len() as i64,
            token_ids,
            mm_token_type_ids,
            position_ids: positions,
        })
    }

    /// Expands the image and video placeholders of rendered chat text, in prompt order.
    /// A still image fills one vision window. A clip replaces its placeholder with one
    /// window per temporal patch, each preceded by its timestamp marker and enclosed in
    /// vision delimiters, which is the text the checkpoint processor writes.
    pub fn expand_media_prompt(
        &self,
        video: &video::VideoProcessor,
        frontend: &Frontend,
        text: &str,
        images: &ProcessedImages,
        videos: &[VideoInput],
    ) -> Result<MediaPrompt> {
        let invalid = |message: &str| Error::frontend(FrontendError::Input(Failure::new(message)));
        if frontend.encode(IMAGE_TOKEN, false)? != [self.image_token_id]
            || frontend.encode(VIDEO_TOKEN, false)? != [self.video_token_id]
        {
            return Err(invalid("image or video token ID does not match tokenizer"));
        }
        let (placeholders, _) = video::media_placeholders(text);
        let stills = placeholders.iter().filter(|(_, still)| *still).count();
        if stills != images.geometry.len() || placeholders.len() - stills != videos.len() {
            return Err(invalid(
                "image and video placeholder counts do not match prepared inputs",
            ));
        }
        // Each planned window with its input and grid, and the bytes of `expanded` that
        // replace each placeholder.
        let mut expanded = String::with_capacity(text.len());
        let mut plan = Vec::new();
        let mut replacements = Vec::with_capacity(images.geometry.len() + videos.len());
        let (mut copied, mut image, mut clip) = (0, 0, 0);
        for (at, still) in placeholders {
            expanded.push_str(&text[copied..at]);
            let start = expanded.len();
            if still {
                let geometry = images.geometry[image];
                for _ in 0..geometry.tokens {
                    expanded.push_str(IMAGE_TOKEN);
                }
                plan.push((PromptWindowSource::Image(image), geometry));
                image += 1;
                copied = at + IMAGE_TOKEN.len();
            } else {
                let input = &videos[clip];
                for (patch, seconds) in video
                    .timestamps(input.frames.len(), input.fps)?
                    .into_iter()
                    .enumerate()
                {
                    expanded.push_str(&video::timestamp_marker(seconds));
                    expanded.push_str(VISION_START_TOKEN);
                    for _ in 0..input.window.tokens {
                        expanded.push_str(VIDEO_TOKEN);
                    }
                    expanded.push_str(VISION_END_TOKEN);
                    plan.push((PromptWindowSource::Video { clip, patch }, input.window));
                }
                clip += 1;
                copied = at + VIDEO_TOKEN.len();
            }
            replacements.push(start..expanded.len());
        }
        expanded.push_str(&text[copied..]);
        let encoding = frontend
            .tokenizer()
            .encode(expanded.as_str(), false)
            .map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                    error.to_string(),
                    error,
                )))
            })?;
        let token_ids = encoding.get_ids().to_vec();
        let mm_token_type_ids: Vec<i32> = token_ids
            .iter()
            .map(|id| {
                if *id == self.image_token_id {
                    1
                } else if *id == self.video_token_id {
                    2
                } else {
                    0
                }
            })
            .collect();
        let mut positions: [Vec<i64>; 4] =
            std::array::from_fn(|_| Vec::with_capacity(token_ids.len()));
        positions[0].extend((0..token_ids.len()).map(|index| index as i64));
        let mut windows = Vec::with_capacity(plan.len());
        let mut planned = plan.into_iter();
        let (mut offset, mut current) = (0, 0i64);
        while offset < token_ids.len() {
            let modality = mm_token_type_ids[offset];
            let end = offset
                + mm_token_type_ids[offset..]
                    .iter()
                    .take_while(|value| **value == modality)
                    .count();
            if modality == 0 {
                for row in &mut positions[1..] {
                    row.extend((0..end - offset).map(|index| current + index as i64));
                }
                current += (end - offset) as i64;
            } else {
                let (source, geometry) = planned
                    .next()
                    .ok_or_else(|| invalid("unexpected vision token group"))?;
                let expected = match source {
                    PromptWindowSource::Image(_) => 1,
                    PromptWindowSource::Video { .. } => 2,
                };
                if modality != expected || end - offset != geometry.tokens {
                    return Err(invalid("vision token group does not match its prepared input"));
                }
                let (height, width) = (
                    geometry.grid_thw[1] / MERGE as i64,
                    geometry.grid_thw[2] / MERGE as i64,
                );
                for y in 0..height {
                    for x in 0..width {
                        positions[1].push(current);
                        positions[2].push(current + y);
                        positions[3].push(current + x);
                    }
                }
                current += height.max(width);
                windows.push(PromptWindow {
                    span: offset..end,
                    geometry,
                    source,
                });
            }
            offset = end;
        }
        if planned.next().is_some() {
            return Err(invalid("prepared vision input was not consumed by prompt"));
        }
        // Tokens of each replacement, from the tokenizer's byte offsets into `expanded`.
        let offsets = encoding.get_offsets();
        let expansions = replacements
            .iter()
            .map(|bytes| {
                offsets.partition_point(|(start, _)| *start < bytes.end)
                    - offsets.partition_point(|(_, end)| *end <= bytes.start)
            })
            .collect();
        Ok(MediaPrompt {
            prompt: ImagePrompt {
                text: expanded,
                rope_delta: current - token_ids.len() as i64,
                token_ids,
                mm_token_type_ids,
                position_ids: positions,
            },
            windows,
            expansions,
        })
    }
}

fn interpolation_axis(index: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    let source = index as f32 * (side - 1) as f32 / (size - 1).max(1) as f32;
    let floor = source.floor();
    let distance = source - floor;
    (
        [floor as usize, (floor as usize + 1).min(side - 1)],
        [1.0 - distance, 1.0 - (distance - 1.0).abs()],
    )
}
