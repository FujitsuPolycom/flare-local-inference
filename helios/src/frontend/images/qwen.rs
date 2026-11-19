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

    /// Expands the image placeholders in already rendered chat tokens, in image order.
    /// Inputs contain no padding; the scheduler adds padding outside logical lengths.
    pub fn expand_prompt(
        &self,
        frontend: &Frontend,
        prompt_token_ids: &[u32],
        images: &ProcessedImages,
    ) -> Result<ImagePrompt> {
        if frontend.encode(IMAGE_TOKEN, false)? != [self.image_token_id] {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image token ID does not match tokenizer",
            ))));
        }
        let placeholder_count = prompt_token_ids
            .iter()
            .filter(|id| **id == self.image_token_id)
            .count();
        if placeholder_count != images.geometry.len() {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image placeholder count does not match prepared images",
            ))));
        }
        if prompt_token_ids.contains(&self.video_token_id) {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "video inputs are not supported by the image processor",
            ))));
        }
        let mut token_ids = Vec::new();
        let mut geometry = images.geometry.iter();
        for &id in prompt_token_ids {
            if id == self.image_token_id {
                let tokens = geometry.next().expect("placeholder count was checked").tokens;
                token_ids.extend(std::iter::repeat_n(self.image_token_id, tokens));
            } else {
                token_ids.push(id);
            }
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
            rope_delta: current - token_ids.len() as i64,
            token_ids,
            mm_token_type_ids,
            position_ids: positions,
        })
    }

    /// Expands the image and video placeholders in rendered token IDs, in prompt order.
    /// A still image fills one vision window. A clip replaces its placeholder with one
    /// window per temporal patch, each preceded by its timestamp marker and enclosed in
    /// vision delimiters, which is the text the checkpoint processor writes.
    pub fn expand_media_prompt(
        &self,
        video: &video::VideoProcessor,
        frontend: &Frontend,
        prompt_token_ids: &[u32],
        images: &ProcessedImages,
        videos: &[VideoInput],
    ) -> Result<MediaPrompt> {
        let invalid = |message: &str| Error::frontend(FrontendError::Input(Failure::new(message)));
        if frontend.encode(IMAGE_TOKEN, false)? != [self.image_token_id]
            || frontend.encode(VIDEO_TOKEN, false)? != [self.video_token_id]
        {
            return Err(invalid("image or video token ID does not match tokenizer"));
        }
        let placeholders: Vec<_> = video::media_placeholders(
            prompt_token_ids, self.image_token_id, self.video_token_id,
        ).collect();
        let stills = placeholders.iter().filter(|(_, still)| *still).count();
        if stills != images.geometry.len() || placeholders.len() - stills != videos.len() {
            return Err(invalid(
                "image and video placeholder counts do not match prepared inputs",
            ));
        }
        let vision_start = frontend.encode(VISION_START_TOKEN, false)?;
        let vision_end = frontend.encode(VISION_END_TOKEN, false)?;
        let mut token_ids = Vec::with_capacity(prompt_token_ids.len());
        let mut plan = Vec::new();
        let mut expansions = Vec::with_capacity(placeholders.len());
        let (mut copied, mut image, mut clip) = (0, 0, 0);
        for (at, still) in placeholders {
            token_ids.extend_from_slice(&prompt_token_ids[copied..at]);
            let start = token_ids.len();
            if still {
                let geometry = images.geometry[image];
                token_ids.extend(std::iter::repeat_n(self.image_token_id, geometry.tokens));
                plan.push((PromptWindowSource::Image(image), geometry));
                image += 1;
            } else {
                let input = &videos[clip];
                for (patch, seconds) in video
                    .timestamps(input.frames.len(), input.fps)?
                    .into_iter()
                    .enumerate()
                {
                    token_ids.extend(frontend.encode(&video::timestamp_marker(seconds), false)?);
                    token_ids.extend_from_slice(&vision_start);
                    token_ids.extend(std::iter::repeat_n(self.video_token_id, input.window.tokens));
                    token_ids.extend_from_slice(&vision_end);
                    plan.push((PromptWindowSource::Video { clip, patch }, input.window));
                }
                clip += 1;
            }
            expansions.push(token_ids.len() - start);
            copied = at + 1;
        }
        token_ids.extend_from_slice(&prompt_token_ids[copied..]);
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
        Ok(MediaPrompt {
            prompt: ImagePrompt {
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
