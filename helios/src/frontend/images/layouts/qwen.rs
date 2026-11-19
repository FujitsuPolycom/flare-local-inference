/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::PromptData;
use crate::{
    Result,
    cache::ConditioningEvent,
    frontend::{
        Frontend, RenderedPrompt,
        images::{
            ImageDescription, ImageProcessor, ProcessedImages, PromptWindow, PromptWindowSource,
            VideoInput,
        },
    },
    model::PreparedInput,
};
pub(crate) fn prepare_input(
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    images: &[ImageDescription],
) -> Result<PreparedInput> {
    let placeholders = [
        frontend.encode("<|image_pad|>", false)?,
        frontend.encode("<|video_pad|>", false)?,
    ]
    .concat();
    let (tokens, positions, vision_token_indices, conditioning, images) = if images.is_empty() {
        if rendered
            .token_ids
            .iter()
            .any(|id| placeholders.contains(id))
        {
            return Err("image/video placeholders require matching supported image inputs".into());
        }
        let positions = (0..rendered.token_ids.len())
            .map(|position| [position as i64; 3])
            .collect();
        let indices = vec![-1; rendered.token_ids.len()];
        (
            rendered.token_ids.clone(),
            positions,
            indices,
            Vec::new(),
            None,
        )
    } else {
        let processor = processor.ok_or("image processing is unavailable")?;
        let processed = processor.describe_inputs(images)?;
        let expanded = processor.expand_prompt(frontend, &rendered.token_ids, &processed)?;
        let positions = (0..expanded.token_ids.len())
            .map(|position| {
                [
                    expanded.position_ids[1][position],
                    expanded.position_ids[2][position],
                    expanded.position_ids[3][position],
                ]
            })
            .collect();
        let mut conditioning = Vec::with_capacity(images.len());
        let mut image = 0;
        for (position, kind) in expanded.mm_token_type_ids.iter().enumerate() {
            if *kind == 1 && (position == 0 || expanded.mm_token_type_ids[position - 1] != 1) {
                let source = &images[image];
                conditioning.push(ConditioningEvent {
                    position: position as u64,
                    identity: source.identity,
                });
                image += 1;
            }
        }
        if image != images.len() {
            return Err("expanded image spans do not match input image count".into());
        }
        let mut index = 0;
        let indices = expanded
            .mm_token_type_ids
            .iter()
            .map(|kind| {
                if *kind == 1 {
                    let value = index;
                    index += 1;
                    value
                } else {
                    -1
                }
            })
            .collect();
        (
            expanded.token_ids,
            positions,
            indices,
            conditioning,
            Some(processed),
        )
    };
    let token_expansions = rendered.token_ids.iter().enumerate()
        .filter(|(_, token)| placeholders.contains(token))
        .zip(images.iter().flat_map(|images| &images.geometry))
        .map(|((index, _), geometry)| (index, geometry.tokens - 1))
        .collect();
    Ok(PreparedInput {
        tokens,
        token_expansions,
        data: PromptData {
            positions,
            vision_token_indices,
            spans: Vec::new(),
        },
        conditioning,
        conditioned_inputs: vec![],
    })
}

/// Prepares a prompt that carries video clips, with or without still images. Returns
/// the prompt's vision windows in prompt order beside the prepared input; conditioning
/// events follow the same order.
pub(crate) fn prepare_media_input(
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    images: &[ImageDescription],
    videos: &[VideoInput],
) -> Result<(PreparedInput, Vec<PromptWindow>)> {
    let processor = processor.ok_or("image processing is unavailable")?;
    let processed = if images.is_empty() {
        ProcessedImages::default()
    } else {
        processor.describe_inputs(images)?
    };
    let expanded = processor.expand_media_prompt(frontend, &rendered.token_ids, &processed, videos)?;
    let prompt = expanded.prompt;
    let positions = (0..prompt.token_ids.len())
        .map(|position| {
            [
                prompt.position_ids[1][position],
                prompt.position_ids[2][position],
                prompt.position_ids[3][position],
            ]
        })
        .collect();
    let conditioning = expanded
        .windows
        .iter()
        .map(|window| ConditioningEvent {
            position: window.span.start as u64,
            identity: match window.source {
                PromptWindowSource::Image(index) => images[index].identity,
                PromptWindowSource::Video { clip, patch } => {
                    let (first, second) = videos[clip].frames_of(patch);
                    processor.video_window_identity(first.identity, second.identity, &window.geometry)
                }
            },
        })
        .collect();
    let mut index = 0;
    let indices = prompt
        .mm_token_type_ids
        .iter()
        .map(|kind| {
            if *kind != 0 {
                let value = index;
                index += 1;
                value
            } else {
                -1
            }
        })
        .collect();
    let placeholders = [
        frontend.encode("<|image_pad|>", false)?,
        frontend.encode("<|video_pad|>", false)?,
    ]
    .concat();
    let token_expansions: Vec<_> = rendered
        .token_ids
        .iter()
        .enumerate()
        .filter(|(_, token)| placeholders.contains(token))
        .zip(&expanded.expansions)
        .map(|((index, _), tokens)| (index, tokens.saturating_sub(1)))
        .collect();
    // Every placeholder is one rendered token, so the expansions account for the whole
    // difference between the rendered and the expanded prompt.
    if token_expansions.len() != expanded.expansions.len()
        || rendered.token_ids.len()
            + token_expansions.iter().map(|(_, additional)| additional).sum::<usize>()
            != prompt.token_ids.len()
    {
        return Err("media placeholder expansions do not match the expanded prompt".into());
    }
    Ok((
        PreparedInput {
            tokens: prompt.token_ids,
            token_expansions,
            data: PromptData {
                positions,
                vision_token_indices: indices,
                ..Default::default()
            },
            conditioning,
            conditioned_inputs: vec![],
        },
        expanded.windows,
    ))
}
