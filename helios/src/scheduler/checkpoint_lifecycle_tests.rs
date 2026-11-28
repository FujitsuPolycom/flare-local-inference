/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

struct Observation {
    origin: u64,
    owners: [u64; 3],
    uses: [u64; 3],
    published: Vec<(CheckpointId, u64)>,
}

impl Observation {
    fn project(&mut self, scheduler: &RequestScheduler) -> [u64; 15] {
        let mut result = [0; 15];
        result[0] = scheduler.submitted.0 - self.origin;
        result[1] = scheduler.completed.0 - self.origin;
        for slot in &scheduler.slots {
            if let Some(request) = &slot.request {
                result[2] |= 1 << request.handle.slot;
                if let Some(prefix) = request.prefix {
                    for id in scheduler.cache.chain(prefix).unwrap() {
                        if !self.published.iter().any(|(known, _)| *known == id) {
                            self.published.push((id, request.handle.slot as u64 + 1));
                        }
                    }
                }
                let armed = request.checkpoint_capture.map(|capture| capture.snapshot);
                let copying = request.boundary_capture.map(|capture| capture.snapshot);
                for (snapshot, phase) in request
                    .anchor_guard
                    .map(|id| (id, 1))
                    .into_iter()
                    .chain(armed.map(|id| (id, 2)))
                    .chain(copying.map(|id| (id, if request.pending > 0 { 3 } else { 2 })))
                {
                    let row = snapshot.0 as usize;
                    self.owners[row] = request.handle.slot as u64 + 1;
                    let fence = if phase == 1 {
                        0
                    } else {
                        request.last_use.0 - self.origin
                    };
                    if phase != 1 {
                        self.uses[row] = self.uses[row].max(fence);
                    }
                    result[3 + row * 4..7 + row * 4].copy_from_slice(&[
                        phase,
                        self.owners[row],
                        fence,
                        self.uses[row],
                    ]);
                }
            }
        }
        for (id, owner) in &self.published {
            if let Ok(checkpoint) = scheduler.cache.checkpoint(*id) {
                let row = checkpoint.snapshot().0 as usize;
                result[3 + row * 4..7 + row * 4].copy_from_slice(&[
                    4,
                    *owner,
                    self.uses[row],
                    self.uses[row],
                ]);
            }
        }
        for retirement in &scheduler.retired {
            for snapshot in retirement.snapshots.into_iter().flatten() {
                let row = snapshot.0 as usize;
                result[3 + row * 4..7 + row * 4].copy_from_slice(&[
                    5,
                    self.owners[row],
                    retirement.through.0 - self.origin,
                    self.uses[row],
                ]);
            }
        }
        for row in 0..3 {
            result[6 + row * 4] = self.uses[row];
            if self.uses[row] > result[1] {
                assert!(
                    matches!(result[3 + row * 4], 2 | 3 | 5),
                    "outstanding checkpoint use lost destination ownership"
                );
                assert!(!scheduler.free_snapshots.contains(&SnapshotId(row as u32)));
            }
        }
        result
    }
}

fn complete_records(
    scheduler: &RequestScheduler,
    plan: &ScheduledPlan,
    fence: Fence,
) -> Vec<CompletionRecord> {
    let mut records = vec![CompletionRecord::default(); 2];
    for (row, lane) in plan.lanes.iter().enumerate() {
        let request = scheduler.request(lane.request).unwrap();
        let consumed = if plan.lane_kind(row).is_prefill() {
            lane.start_position + lane.tokens as u64
        } else {
            request.consumed_tokens
        };
        records[lane.request.slot as usize] = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence.0 as i64,
            status: 1,
            consumed_tokens: consumed as i64,
            generated_tokens: request.generated_tokens as i64,
            token_id: -1,
            ..Default::default()
        };
    }
    records
}

fn edge(before: [u64; 15], action: [u64; 3], after: [u64; 15]) {
    println!(
        "CHECKPOINT_RESERVATION_EDGE {}",
        format_args!("[{before:?},{action:?},{after:?}]")
    );
}

#[test]
fn checkpoint_reservations_match_lifecycle_model() {
    for cancellation in 0..5 {
        let mut scheduler = tests::scheduler(2, 64, 3);
        let handles = [0, 1].map(|slot| {
            let mut spec = tests::spec(64, 16);
            spec.prompt[0] += slot;
            spec.message_boundaries = vec![32];
            scheduler.admit(spec).unwrap()
        });
        let init = scheduler.plan().unwrap();
        assert_eq!(init.kind, GraphKind::Init);
        scheduler.commit(&init, Fence(1)).unwrap();
        let records = complete_records(&scheduler, &init, Fence(1));
        scheduler
            .complete(Fence(1), &records, Default::default())
            .unwrap();
        let mut observation = Observation {
            origin: 1,
            owners: [0; 3],
            uses: [0; 3],
            published: vec![],
        };
        let mut before = observation.project(&scheduler);
        let produce = scheduler.plan().unwrap();
        assert!(produce.kind.is_prefill());
        assert_eq!(produce.lanes.len(), 1);
        let handle = produce.lanes[0].request;
        assert!(handles.contains(&handle));
        scheduler.commit(&produce, Fence(2)).unwrap();
        let snapshot = scheduler
            .request(handle)
            .unwrap()
            .checkpoint_capture
            .unwrap()
            .snapshot;
        assert!(!scheduler.revoke_unused_anchor_guards());
        assert!(!scheduler.free_snapshots.contains(&snapshot));
        let request = handle.slot as u64 + 1;
        let row = snapshot.0 as u64 + 1;
        let mut after = observation.project(&scheduler);
        edge(before, [13, request, row], after);
        before = after;
        if cancellation == 0 {
            let records = complete_records(&scheduler, &produce, Fence(2));
            scheduler.cancel(handle).unwrap();
            after = observation.project(&scheduler);
            edge(before, [15, request, 0], after);
            scheduler
                .complete(Fence(2), &records, Default::default())
                .unwrap();
            edge(after, [16, 0, 0], observation.project(&scheduler));
            assert!(scheduler.free_snapshots.contains(&snapshot));
            continue;
        }
        let records = complete_records(&scheduler, &produce, Fence(2));
        scheduler
            .complete(Fence(2), &records, Default::default())
            .unwrap();
        after = observation.project(&scheduler);
        edge(before, [5, 0, 0], after);
        before = after;
        if cancellation == 1 {
            scheduler.cancel(handle).unwrap();
            edge(before, [15, request, 0], observation.project(&scheduler));
            continue;
        }
        let copy = scheduler.plan().unwrap();
        assert_eq!(copy.kind, GraphKind::Snapshot);
        scheduler.commit(&copy, Fence(3)).unwrap();
        after = observation.project(&scheduler);
        edge(before, [4, request, row], after);
        before = after;
        if cancellation == 2 {
            let records = complete_records(&scheduler, &copy, Fence(3));
            scheduler.cancel(handle).unwrap();
            after = observation.project(&scheduler);
            edge(before, [15, request, 0], after);
            scheduler
                .complete(Fence(3), &records, Default::default())
                .unwrap();
            edge(after, [16, 0, 0], observation.project(&scheduler));
            assert!(scheduler.free_snapshots.contains(&snapshot));
            continue;
        }
        let records = complete_records(&scheduler, &copy, Fence(3));
        scheduler
            .complete(Fence(3), &records, Default::default())
            .unwrap();
        after = observation.project(&scheduler);
        edge(before, [14, request, row], after);
        before = after;
        if cancellation == 3 {
            scheduler.cancel(handle).unwrap();
            edge(before, [15, request, 0], observation.project(&scheduler));
        }
        assert!(
            scheduler
                .cache
                .checkpoint(observation.published[0].0)
                .is_ok()
        );
    }
}

#[test]
fn unused_anchor_guards_revoke_oldest_first_without_cancelling_requests() {
    let mut scheduler = tests::scheduler(2, 64, 3);
    scheduler
        .share_snapshot_pages(&[40..44, 42..46, 48..52])
        .unwrap();
    let first = scheduler.admit(tests::spec(64, 16)).unwrap();
    let second = scheduler.admit(tests::spec(64, 16)).unwrap();
    let mut observation = Observation {
        origin: 0,
        owners: [0; 3],
        uses: [0; 3],
        published: vec![],
    };
    let mut before = observation.project(&scheduler);
    let one = scheduler.reserve_snapshot(0).unwrap();
    scheduler.request_mut(first).unwrap().anchor_guard = Some(one);
    let mut after = observation.project(&scheduler);
    edge(before, [1, first.slot as u64 + 1, one.0 as u64 + 1], after);
    before = after;
    let two = scheduler.reserve_snapshot(0).unwrap();
    scheduler.request_mut(second).unwrap().anchor_guard = Some(two);
    after = observation.project(&scheduler);
    edge(before, [1, second.slot as u64 + 1, two.0 as u64 + 1], after);
    before = after;
    assert!(scheduler.revoke_unused_anchor_guards());
    after = observation.project(&scheduler);
    edge(before, [17, first.slot as u64 + 1, one.0 as u64 + 1], after);
    assert!(scheduler.request(first).unwrap().anchor_guard.is_none());
    assert_eq!(scheduler.request(second).unwrap().anchor_guard, Some(two));
    assert!(scheduler.free_snapshots.contains(&one));
    assert!(!scheduler.free_snapshots.contains(&two));
    before = after;
    assert!(scheduler.revoke_unused_anchor_guards());
    edge(
        before,
        [17, second.slot as u64 + 1, two.0 as u64 + 1],
        observation.project(&scheduler),
    );
    assert!(scheduler.request(second).unwrap().anchor_guard.is_none());
    assert!(scheduler.free_snapshots.contains(&two));
    assert_eq!(scheduler.stats().snapshot_backing_pages, 0);
    assert!(!scheduler.revoke_unused_anchor_guards());
}

#[test]
fn terminal_reservations_match_lifecycle_model() {
    #[derive(Clone, Copy)]
    enum Case {
        SpareFreeRow,
        ProtectedAnchorRefused,
        AgedAnchorReaped,
        LargeGainEvictsAnchor,
        NonRootEndpointEvicted,
        EarlyFinish,
        Preempt,
    }

    for case in [
        Case::SpareFreeRow,
        Case::ProtectedAnchorRefused,
        Case::AgedAnchorReaped,
        Case::LargeGainEvictsAnchor,
        Case::NonRootEndpointEvicted,
        Case::EarlyFinish,
        Case::Preempt,
    ] {
        let mut s = tests::scheduler(2, 64, 3);
        s.enable_evictions();
        let mut observation = Observation {
            origin: 0,
            owners: [0; 3],
            uses: [0; 3],
            published: vec![],
        };

        let publish_row = |s: &mut RequestScheduler,
                           obs: &mut Observation,
                           parent: Option<CheckpointId>,
                           fill: u32,
                           tokens: u64,
                           role: RetentionRole| {
            let mut history = PrefixHistory::new([9; 32]);
            if let Some(parent_id) = parent {
                let parent_len = s.cache.checkpoint(parent_id).unwrap().prefix().consumed_tokens
                    as usize;
                history.append(&vec![fill; parent_len], &[]).unwrap();
                history
                    .append(&vec![fill + 1; tokens as usize - parent_len], &[])
                    .unwrap();
            } else {
                history.append(&vec![fill; tokens as usize], &[]).unwrap();
            }
            let parent_pages = parent.map_or(0, |id| {
                s.cache.checkpoint(id).unwrap().prefix().consumed_tokens as usize / 64
            });
            let pages: Vec<_> = (0..(tokens as usize / 64 - parent_pages))
                .map(|_| s.free_pages.pop_first().unwrap())
                .collect();
            let snapshot = s.reserve_snapshot(0).unwrap();
            let epoch = s.cache.new_use_epoch().unwrap();
            let id = s
                .cache
                .publish_with_retention(
                    &history,
                    &Publication {
                        parent,
                        consumed_tokens: tokens,
                        tail_pages: pages,
                        snapshot: StateSnapshot {
                            id: snapshot,
                            consumed_tokens: tokens,
                        },
                        ready_after: Fence(0),
                    },
                    role,
                    epoch,
                )
                .unwrap()
                .id();
            obs.published.push((id, 2));
            id
        };

        let active_anchor = match case {
            Case::SpareFreeRow => {
                publish_row(&mut s, &mut observation, None, 100, 128, RetentionRole::Anchor);
                publish_row(&mut s, &mut observation, None, 200, 128, RetentionRole::Anchor)
            }
            Case::NonRootEndpointEvicted => {
                let doc =
                    publish_row(&mut s, &mut observation, None, 100, 128, RetentionRole::Anchor);
                let reply = publish_row(
                    &mut s,
                    &mut observation,
                    Some(doc),
                    100,
                    192,
                    RetentionRole::Endpoint,
                );
                let reply_epoch = s
                    .cache
                    .checkpoint(reply)
                    .unwrap()
                    .retention_rank()
                    .use_epoch;
                s.cache.touch(doc, reply_epoch).unwrap();
                publish_row(&mut s, &mut observation, None, 200, 128, RetentionRole::Anchor)
            }
            Case::ProtectedAnchorRefused
            | Case::AgedAnchorReaped
            | Case::LargeGainEvictsAnchor
            | Case::EarlyFinish
            | Case::Preempt => {
                publish_row(&mut s, &mut observation, None, 100, 128, RetentionRole::Anchor);
                publish_row(&mut s, &mut observation, None, 150, 128, RetentionRole::Anchor);
                publish_row(&mut s, &mut observation, None, 200, 128, RetentionRole::Anchor)
            }
        };

        if matches!(case, Case::AgedAnchorReaped) {
            for _ in 0..8 {
                s.cache.new_use_epoch().unwrap();
            }
        }

        let mut spec = tests::spec(128, 128);
        spec.prompt.fill(200);
        spec.message_boundaries = vec![128];
        spec.growable_output = matches!(case, Case::EarlyFinish | Case::Preempt);
        let h = s.admit(spec).unwrap();
        assert_eq!(s.request(h).unwrap().prefix, Some(active_anchor));
        s.request_mut(h).unwrap().checkpoints.productive_decode_ns = 500_000;
        let before = observation.project(&s);
        let row = match case {
            Case::SpareFreeRow
            | Case::ProtectedAnchorRefused
            | Case::AgedAnchorReaped
            | Case::NonRootEndpointEvicted => {
                // Row 0 is 128 tokens: at age 3, cost = 128/4 = 32 > 16 (refused);
                // after aging 8 epochs (age 11), cost = 128/12 = 10 <= 16 (reaped).
                s.request_mut(h).unwrap().consumed_tokens = 128 + 16;
                s.allocate_terminal_capture(h).unwrap()
            }
            Case::LargeGainEvictsAnchor => {
                s.request_mut(h).unwrap().consumed_tokens = 128 + 64;
                s.allocate_terminal_capture(h).unwrap()
            }
            Case::EarlyFinish | Case::Preempt => {
                let req = s.request_mut(h).unwrap();
                req.phase = RequestPhase::Decode;
                req.consumed_tokens = 128 + 32;
                req.scheduled_consumed = 128 + 32;
                let prompt_page = req.reserved_pages.pop().unwrap();
                req.private_pages.push(prompt_page);
                if matches!(case, Case::Preempt) {
                    s.preempt_newest().unwrap();
                } else {
                    s.finish_early(h, false).unwrap();
                }
                s.request(h).unwrap().terminal_snapshot
            }
        };

        let expect_skip = matches!(case, Case::ProtectedAnchorRefused);
        let expect_evict = !matches!(case, Case::SpareFreeRow | Case::ProtectedAnchorRefused);
        assert_eq!(
            row.is_none(),
            expect_skip,
            "short terminal endpoint must not evict a larger protected anchor leaf"
        );
        assert_eq!(s.stats().checkpoint_skip_storage, u64::from(expect_skip));
        assert_eq!(s.take_evictions().len(), usize::from(expect_evict));

        let mut after = observation.project(&s);
        if let Some(id) = row {
            assert!(!s.free_snapshots.contains(&id));
            let start = 3 + 4 * id.0 as usize;
            assert_eq!(after[start], 0, "reserved row still has a previous owner");
            after[start..start + 4].copy_from_slice(&[2, h.slot as u64 + 1, 0, 0]);
        }
        edge(
            before,
            [
                if expect_evict { 19 } else { 18 },
                h.slot as u64 + 1,
                row.map_or(0, |id| id.0 as u64 + 1),
            ],
            after,
        );
    }
}
