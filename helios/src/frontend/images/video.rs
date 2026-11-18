/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Qwen video geometry: one resize target per clip and one vision window per temporal patch.

use super::*;

/// Frames in one temporal patch. The vision window packs two temporal slices, so prepared
/// video settings must declare this value.
pub(super) const TEMPORAL_PATCH: usize = 2;

/// Bounds of an accepted source frame rate, in frames per second. Every frame time
/// `index / fps` inside them is finite, and the timestamp marker of a temporal patch,
/// which prints one decimal, is at most `(max_frames - 1) / MIN_FPS` seconds.
pub(super) const MIN_FPS: f64 = 0.001;
pub(super) const MAX_FPS: f64 = 1000.0;

#[derive(Clone)]
pub(super) struct VideoProcessor {
    max_frames: usize,
    min_pixels: usize,
    max_pixels: usize,
    limits: ImageLimits,
}

fn invalid(message: String) -> Error {
    Error::frontend(FrontendError::Input(Failure::new(message)))
}

impl VideoProcessor {
    pub(super) fn load(settings: &QwenVideoSettings, limits: ImageLimits) -> Result<Self> {
        if settings.temporal_patch_size != TEMPORAL_PATCH
            || settings.min_pixels == 0
            || settings.max_pixels < settings.min_pixels
            || settings.max_frames < TEMPORAL_PATCH
        {
            return Err("invalid prepared Qwen video settings".into());
        }
        Ok(Self {
            max_frames: settings.max_frames,
            min_pixels: settings.min_pixels,
            max_pixels: settings.max_pixels,
            limits,
        })
    }

    /// A clip holds at least one temporal patch and at most the prepared frame count.
    /// The checkpoint processor's lower frame bound applies to frames it samples from a
    /// container; a client that supplies frames has sampled them.
    pub(super) fn check_frames(&self, frames: usize) -> Result<()> {
        if frames < TEMPORAL_PATCH || frames > self.max_frames {
            return Err(invalid(format!(
                "clip has {frames} frames; prepared video settings admit {TEMPORAL_PATCH} through {} frames",
                self.max_frames
            )));
        }
        Ok(())
    }

    /// The vision window of one temporal patch, by the checkpoint video processor's
    /// resize rule: one pixel budget covers the whole clip, and every frame is resized
    /// to the same target.
    pub(super) fn window(&self, frames: usize, width: usize, height: usize) -> Result<ImageGeometry> {
        check_input_extent(self.limits, width, height)?;
        self.check_frames(frames)?;
        let factor = (PATCH * MERGE) as f64;
        let (mut height, mut width) = (height, width);
        if (height as f64) < factor || (width as f64) < factor {
            let scale = (factor / height as f64).max(factor / width as f64);
            height = (height as f64 * scale) as usize;
            width = (width as f64 * scale) as usize;
        }
        if height.max(width) as f64 / height.min(width) as f64 > 200.0 {
            return Err(invalid("frame aspect ratio exceeds 200".into()));
        }
        let (h, w) = (height as f64, width as f64);
        let temporal = TEMPORAL_PATCH as f64;
        let mut target_h = (h / factor).round_ties_even() * factor;
        let mut target_w = (w / factor).round_ties_even() * factor;
        let target_t = (frames as f64 / temporal).round_ties_even() * temporal;
        let budget = target_t * target_h * target_w;
        if budget > self.max_pixels as f64 {
            let beta = ((frames as f64 * h * w) / self.max_pixels as f64).sqrt();
            target_h = ((h / beta / factor).floor() * factor).max(factor);
            target_w = ((w / beta / factor).floor() * factor).max(factor);
        } else if budget < self.min_pixels as f64 {
            let beta = (self.min_pixels as f64 / (frames as f64 * h * w)).sqrt();
            target_h = (h * beta / factor).ceil() * factor;
            target_w = (w * beta / factor).ceil() * factor;
        }
        if !(target_h.is_finite() && target_w.is_finite() && target_h >= factor && target_w >= factor) {
            return Err(invalid("clip has no valid resize target".into()));
        }
        let (height, width) = (target_h as usize, target_w as usize);
        let patches = (height / PATCH).saturating_mul(width / PATCH);
        if patches > self.limits.max_patches_per_image {
            return Err(invalid(format!(
                "one temporal patch needs {} video tokens ({patches} patches); \
                 max_image_tokens_per_image is {} ({} patches)",
                self.limits.tokens(patches),
                self.limits.max_image_tokens_per_image(),
                self.limits.max_patches_per_image
            )));
        }
        Ok(ImageGeometry {
            height,
            width,
            grid_thw: [1, (height / PATCH) as i64, (width / PATCH) as i64],
            patches,
            tokens: patches / (MERGE * MERGE),
        })
    }

    /// Source time of each temporal patch in seconds: the mean of its two frame times,
    /// where the final patch of an odd clip repeats the last frame.
    pub(super) fn timestamps(&self, frames: usize, fps: f64) -> Result<Vec<f64>> {
        if !(MIN_FPS..=MAX_FPS).contains(&fps) {
            return Err(invalid(format!(
                "frame rate must be between {MIN_FPS} and {MAX_FPS} frames per second"
            )));
        }
        Ok((0..frames.div_ceil(TEMPORAL_PATCH))
            .map(|patch| {
                let first = patch * TEMPORAL_PATCH;
                let last = (first + TEMPORAL_PATCH - 1).min(frames - 1);
                (first as f64 / fps + last as f64 / fps) / 2.0
            })
            .collect())
    }
}

/// The image and video placeholders of rendered text in prompt order: the byte offset
/// of each and whether it is a still image. Each placeholder kind is searched once
/// across the text. The second value counts the search steps: one for each placeholder
/// found and one for each kind when its search ends.
pub(super) fn media_placeholders(text: &str) -> (Vec<(usize, bool)>, usize) {
    let mut images = text.match_indices(IMAGE_TOKEN);
    let mut videos = text.match_indices(VIDEO_TOKEN);
    let mut searches = 0;
    let mut step = |matches: &mut std::str::MatchIndices<'_, &str>| {
        searches += 1;
        matches.next().map(|(at, _)| at)
    };
    let (mut image, mut video) = (step(&mut images), step(&mut videos));
    let mut placeholders = Vec::new();
    loop {
        match (image, video) {
            (Some(at), other) if other.is_none_or(|video| at < video) => {
                placeholders.push((at, true));
                image = step(&mut images);
            }
            (_, Some(at)) => {
                placeholders.push((at, false));
                video = step(&mut videos);
            }
            _ => break,
        }
    }
    (placeholders, searches)
}

/// The timestamp text that precedes one temporal patch's vision span.
pub(super) fn timestamp_marker(seconds: f64) -> String {
    format!("<{seconds:.1} seconds>")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video(max_pixels: usize) -> VideoProcessor {
        VideoProcessor::load(
            &QwenVideoSettings {
                max_frames: 768,
                min_pixels: 4096,
                max_pixels,
                temporal_patch_size: 2,
            },
            ImageLimits {
                max_patches_per_image: 16_384,
                patches_per_feature: 4,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn clip_windows_follow_the_clip_pixel_budget() {
        let processor = video(25_165_824);
        // 64x64 frames: a 4x4 patch grid per window, whatever the frame count.
        let window = processor.window(8, 64, 64).unwrap();
        assert_eq!((window.width, window.height), (64, 64));
        assert_eq!(window.grid_thw, [1, 4, 4]);
        assert_eq!((window.patches, window.tokens), (16, 4));
        assert_eq!(processor.window(5, 64, 64).unwrap(), window);
        // Frames below one merged patch scale up to it.
        let tiny = processor.window(4, 16, 8).unwrap();
        assert_eq!((tiny.width, tiny.height), (64, 32));
        // The budget's temporal extent is the frame count rounded half-even to whole
        // patches: four and five frames budget four, six frames budget six.
        let bounded = video(4 * 96 * 96);
        assert_eq!(bounded.window(4, 96, 96).unwrap().width, 96);
        assert_eq!(bounded.window(5, 96, 96).unwrap().width, 96);
        assert_eq!(bounded.window(6, 96, 96).unwrap().width, 64);
        // A long clip shrinks every frame.
        let long = processor.window(64, 1024, 1024).unwrap();
        assert_eq!((long.width, long.height, long.patches), (608, 608, 38 * 38));
    }

    #[test]
    fn clips_outside_the_prepared_bounds_are_rejected() {
        let processor = video(25_165_824);
        for frames in [0, 1, 769] {
            let error = processor.window(frames, 64, 64).unwrap_err().to_string();
            assert_eq!(
                error,
                format!("clip has {frames} frames; prepared video settings admit 2 through 768 frames")
            );
        }
        for frames in [2, 3, 768] {
            assert!(processor.window(frames, 64, 64).is_ok(), "{frames}");
        }
        assert_eq!(
            processor.window(4, 8000, 16).unwrap_err().to_string(),
            "frame aspect ratio exceeds 200"
        );
        assert!(processor.window(4, 0, 16).is_err());
        assert_eq!(
            processor.window(4, 3840, 2160).unwrap_err().to_string(),
            "one temporal patch needs 6032 video tokens (24128 patches); \
             max_image_tokens_per_image is 4096 (16384 patches)"
        );
        // A prepared temporal patch size other than two has no packing layout.
        assert!(
            VideoProcessor::load(
                &QwenVideoSettings {
                    max_frames: 768,
                    min_pixels: 4096,
                    max_pixels: 25_165_824,
                    temporal_patch_size: 4,
                },
                ImageLimits::default(),
            )
            .is_err()
        );
    }

    #[test]
    fn timestamps_average_the_frame_times_of_each_temporal_patch() {
        let processor = video(25_165_824);
        let markers = |frames, fps| {
            processor
                .timestamps(frames, fps)
                .unwrap()
                .into_iter()
                .map(timestamp_marker)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            markers(8, 2.0),
            ["<0.2 seconds>", "<1.2 seconds>", "<2.2 seconds>", "<3.2 seconds>"]
        );
        // The final patch of an odd clip repeats its last frame.
        assert_eq!(markers(5, 4.0), ["<0.1 seconds>", "<0.6 seconds>", "<1.0 seconds>"]);
        assert_eq!(processor.timestamps(2, 1.0).unwrap(), vec![0.5]);
        // The slowest rate at the prepared frame count gives the longest marker.
        assert_eq!(markers(768, MIN_FPS).last().unwrap(), "<766500.0 seconds>");
        assert_eq!(markers(768, MAX_FPS).last().unwrap(), "<0.8 seconds>");
    }

    #[test]
    fn placeholders_are_found_with_one_search_step_each() {
        // A search that restarts at every placeholder scans the rest of the text for the
        // kind that is absent there, and takes two steps for each placeholder.
        let text = format!("{}tail", "a<|video_pad|>b<|image_pad|><|video_pad|>".repeat(1000));
        let (placeholders, searches) = media_placeholders(&text);
        assert_eq!(placeholders.len(), 3000);
        assert_eq!(searches, placeholders.len() + 2);
        assert!(placeholders.windows(2).all(|pair| pair[0].0 < pair[1].0));
        for (index, (at, still)) in placeholders.iter().enumerate() {
            assert_eq!(*still, index % 3 == 1);
            assert!(text[*at..].starts_with(if *still { IMAGE_TOKEN } else { VIDEO_TOKEN }));
        }
        let (placeholders, searches) = media_placeholders(&IMAGE_TOKEN.repeat(500));
        assert_eq!((placeholders.len(), searches), (500, 502));
        assert!(placeholders.iter().all(|(_, still)| *still));
        assert_eq!(media_placeholders("no placeholder"), (vec![], 2));
    }

    #[test]
    fn frame_rates_outside_the_accepted_range_are_rejected() {
        let processor = video(25_165_824);
        for fps in [MIN_FPS, 1.0, 240.0, MAX_FPS] {
            assert!(processor.timestamps(4, fps).is_ok(), "{fps}");
        }
        for fps in [
            0.0,
            -1.0,
            f64::MIN_POSITIVE / 2.0,
            f64::MIN_POSITIVE,
            MIN_FPS / 2.0,
            MAX_FPS + 1.0,
            f64::INFINITY,
            f64::NAN,
        ] {
            let error = processor.timestamps(4, fps).unwrap_err().to_string();
            assert_eq!(error, "frame rate must be between 0.001 and 1000 frames per second", "{fps}");
        }
    }
}
