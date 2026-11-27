/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{ExecutionManifest, Fence, FrameCompletion, Result, VecDeque, invalid};
use crate::execution::GraphKind;

pub(crate) trait PendingFrame {
    fn fence(&self) -> Fence;
    fn graph(&self) -> usize;
    fn grammar(&self) -> bool;
    fn committed(&self) -> bool;
    fn ready(&self) -> &[bool];
    fn ready_mut(&mut self) -> &mut [bool];
    fn completions(&self) -> &[Option<FrameCompletion>];
    fn completions_mut(&mut self) -> &mut [Option<FrameCompletion>];
}

pub(crate) fn can_submit_pending<P: PendingFrame>(
    pending: &VecDeque<P>,
    barrier: bool,
    manifest: &ExecutionManifest,
) -> bool {
    pending.len() < 2
        && pending.back().is_none_or(|frame| frame.committed())
        && (!barrier || pending.is_empty())
        && !pending.iter().any(|frame| {
            frame.grammar()
                || !matches!(
                    manifest.graphs[frame.graph()].kind,
                    GraphKind::Decode | GraphKind::Speculative
                )
        })
}

pub(crate) fn record_ready<P: PendingFrame>(
    pending: &mut VecDeque<P>,
    peer: usize,
    fence: Fence,
    graph: usize,
) -> Result<()> {
    let frame = pending
        .iter_mut()
        .find(|frame| frame.fence() == fence)
        .ok_or_else(|| invalid("unexpected rank readiness fence"))?;
    if frame.graph() != graph || frame.ready()[peer] || frame.committed() {
        return Err(invalid("rank readiness identity or state differs"));
    }
    frame.ready_mut()[peer] = true;
    Ok(())
}

pub(crate) fn record_local_completion<P: PendingFrame>(
    pending: &mut VecDeque<P>,
    completion: FrameCompletion,
) -> Result<()> {
    let frame = pending
        .iter_mut()
        .find(|frame| frame.fence() == completion.fence)
        .ok_or_else(|| invalid("unexpected rank-zero completion fence"))?;
    if !frame.committed() || frame.graph() != completion.graph_index {
        return Err(invalid("rank-zero completion identity or state differs"));
    }
    if frame.completions()[0].is_some() {
        return Err(invalid("duplicate rank-zero completion"));
    }
    frame.completions_mut()[0] = Some(completion);
    Ok(())
}

pub(crate) fn record_peer_completion<P: PendingFrame>(
    pending: &mut VecDeque<P>,
    peer: usize,
    completion: FrameCompletion,
) -> Result<()> {
    let expected = pending
        .iter_mut()
        .find(|frame| frame.completions()[peer + 1].is_none())
        .ok_or_else(|| invalid("unexpected rank completion"))?;
    if !expected.committed()
        || expected.fence() != completion.fence
        || expected.graph() != completion.graph_index
    {
        return Err(invalid("rank completion order or identity differs"));
    }
    expected.completions_mut()[peer + 1] = Some(completion);
    Ok(())
}

pub(crate) fn ready_to_commit<P: PendingFrame>(frame: &P) -> bool {
    !frame.committed() && frame.ready().iter().all(|ready| *ready)
}

pub(crate) fn dispatch_commit<P>(
    peers: &mut [P],
    mut send: impl FnMut(&mut P) -> Result<()>,
) -> Result<()> {
    for peer in peers {
        send(peer)?;
    }
    Ok(())
}

pub(crate) fn pop_completed_front<P: PendingFrame>(pending: &mut VecDeque<P>) -> Option<P> {
    if !pending
        .front()
        .is_some_and(|frame| frame.completions().iter().all(Option::is_some))
    {
        return None;
    }
    pending.pop_front()
}
