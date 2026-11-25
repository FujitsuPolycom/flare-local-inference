/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lookup (PROTOCOL.md section 4): candidates from the index, never from a source.

use flare_selene::shared::Tier;

use super::{
    Checked, Violation,
    core::Core,
    index::{Entry, SourceId, walk_chain},
    key_from, key_to, tier_to,
};
use crate::{
    Key,
    proto::v2::{self, server_message::M},
};
use std::collections::VecDeque;

/// Entries scanned per lookup at most, so one lookup cannot hold the actor.
const SCAN_BUDGET: usize = 65_536;
/// Lookups answered per pass of the clock.
const PER_PASS: usize = 64;

pub(super) struct Queued {
    pub sid: u64,
    pub request: v2::LookupRequest,
}

fn rank(order: &[v2::Level], level: v2::Level) -> i32 {
    if level == v2::Level::ThisGpu {
        return -1;
    }
    order
        .iter()
        .position(|candidate| *candidate == level)
        .map_or(i32::MAX, |position| position as i32)
}

impl Core {
    pub(super) fn lookup(&mut self, sid: u64, request: v2::LookupRequest) -> Checked<()> {
        let limits = self.cfg.limits.clone();
        let Some(session) = self.sessions.get(&sid) else {
            return Ok(());
        };
        if !self.ready() {
            self.lookup_reply(
                sid,
                request.lookup_id,
                v2::LookupStatus::LookupNotReady,
                vec![],
                50,
            );
            return Ok(());
        }
        if session.lookups.contains(&request.lookup_id) {
            return Err(Violation::new("lookup_id reused while live"));
        }
        if session.lookups.len() >= limits.max_concurrent_lookups as usize {
            self.lookup_reply(
                sid,
                request.lookup_id,
                v2::LookupStatus::LookupOverloaded,
                vec![],
                50,
            );
            return Ok(());
        }
        let session = self.sessions.get_mut(&sid).expect("checked above");
        session.lookups.insert(request.lookup_id);
        self.lookup_queue.push_back(Queued { sid, request });
        Ok(())
    }

    pub(super) fn lookup_cancel(&mut self, sid: u64, lookup_id: u64) {
        let before = self.lookup_queue.len();
        self.lookup_queue
            .retain(|queued| !(queued.sid == sid && queued.request.lookup_id == lookup_id));
        if self.lookup_queue.len() != before {
            if let Some(session) = self.sessions.get_mut(&sid) {
                session.lookups.remove(&lookup_id);
            }
        }
    }

    /// Answers queued lookups. Called from the clock, so a burst arriving within one
    /// pass sees the concurrency bound.
    pub(super) fn lookup_pass(&mut self) {
        for _ in 0..PER_PASS {
            let Some(Queued { sid, request }) = self.lookup_queue.pop_front() else {
                break;
            };
            let Some(session) = self.sessions.get_mut(&sid) else {
                continue;
            };
            session.lookups.remove(&request.lookup_id);
            let lookup_id = request.lookup_id;
            let (status, candidates) = self.answer_lookup(sid, request);
            self.lookup_reply(sid, lookup_id, status, candidates, 0);
        }
    }

    fn lookup_reply(
        &mut self,
        sid: u64,
        lookup_id: u64,
        status: v2::LookupStatus,
        candidates: Vec<v2::Candidate>,
        retry_after_ms: u32,
    ) {
        self.count("lookups_total", format!("status=\"{status:?}\""));
        self.send(
            sid,
            M::LookupReply(v2::LookupReply {
                lookup_id,
                status: status as i32,
                candidates,
                retry_after_ms,
            }),
        );
    }

    fn answer_lookup(
        &mut self,
        sid: u64,
        request: v2::LookupRequest,
    ) -> (v2::LookupStatus, Vec<v2::Candidate>) {
        use v2::LookupStatus as S;
        let limits = &self.cfg.limits;
        let Some(session) = self.sessions.get(&sid) else {
            return (S::LookupInvalid, vec![]);
        };
        let Some(state) = self.scopes.get(&session.scope) else {
            return (S::LookupInvalid, vec![]);
        };
        let geometry = &state.valid.descriptor;

        // Validation.
        if request.boundaries.len() > limits.max_boundaries as usize {
            return (S::LookupInvalid, vec![]);
        }
        let mut boundaries: Vec<Key> = Vec::with_capacity(request.boundaries.len());
        for boundary in &request.boundaries {
            let Ok(key) = key_from(Some(boundary)) else {
                return (S::LookupInvalid, vec![]);
            };
            if boundaries
                .last()
                .is_some_and(|previous| previous.consumed_tokens >= key.consumed_tokens)
            {
                return (S::LookupInvalid, vec![]);
            }
            boundaries.push(key);
        }
        let order = [
            v2::Level::NeighbourGpu,
            v2::Level::HostDram,
            v2::Level::HostNvme,
            v2::Level::RemoteNvme,
        ];
        let max = limits.max_candidates as usize;

        // Records shorter than one page all live in the root bucket, whatever they are
        // records of, so for a history that has a block of its own they would fill the
        // candidate list with records that cannot be told from a match. Such a record
        // restores no page, so it is not offered then.
        let short = if boundaries.len() > 1 {
            geometry.page_tokens
        } else {
            0
        };

        // Candidates.
        let mut scanned = 0usize;
        let mut found: Vec<Ranked> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for boundary in &boundaries {
            for (source, key, tier) in state.index.in_bucket(&boundary.digest) {
                if scanned >= SCAN_BUDGET {
                    break;
                }
                scanned += 1;
                if !seen.insert((source, key, tier)) {
                    continue;
                }
                let Some(entry) = state.index.get(source, key, tier) else {
                    continue;
                };
                if !entry.row
                    || entry.key.consumed_tokens >= request.before_tokens
                    || entry.key.consumed_tokens < short
                {
                    continue;
                }
                let Some((mut level, source_ref)) =
                    self.reach(session.shm, session.instance, source, tier)
                else {
                    continue;
                };
                let begin = request.resident_tokens;
                if begin >= entry.key.consumed_tokens {
                    continue;
                }
                let start_page = begin / geometry.page_tokens;
                let Some(walk) = walk_chain(
                    &state.index,
                    source,
                    entry,
                    geometry.page_tokens,
                    start_page,
                    limits.max_chain_nodes as usize,
                ) else {
                    continue;
                };
                if walk.nodes.iter().any(|n| {
                    !flare_selene::shared::packed::permits_holding(
                        session.codec.as_ref(),
                        n.packed,
                        &n.required_profiles,
                    )
                }) {
                    *self
                        .counters
                        .entry(("codec_incompatible_total", "action=\"lookup\"".into()))
                        .or_default() += 1;
                    continue;
                }
                if source == SourceId::Manager
                    && level == v2::Level::HostDram
                    && walk
                        .nodes
                        .iter()
                        .any(|node| state.index.get(source, node.key, Tier::Dram).is_none())
                {
                    level = v2::Level::HostNvme;
                }
                found.push((
                    rank(&order, level),
                    std::cmp::Reverse(entry.key.consumed_tokens),
                    source,
                    tier,
                    candidate(entry, level, source_ref, begin, &walk, geometry),
                ));
            }
        }
        found.sort_by(|a, b| (a.1, a.0, a.2, a.3).cmp(&(b.1, b.0, b.2, b.3)));
        found.truncate(max);
        (S::LookupOk, found.into_iter().map(|item| item.4).collect())
    }

    /// The level a source's entry has for a requester, and the reference a claim
    /// names. `None` when the requester cannot use that source.
    pub(super) fn reach(
        &self,
        requester_shm: bool,
        requester: super::index::InstanceId,
        source: SourceId,
        tier: Tier,
    ) -> Option<(v2::Level, v2::SourceRef)> {
        match source {
            SourceId::Instance(id) => {
                if id == requester {
                    return None;
                }
                let sid = self.instances.get(&id)?.current?;
                let source_session = self.sessions.get(&sid)?;
                if source_session.export_slots == 0 {
                    return None;
                }
                // Without shared memory on either end the bytes go through the manager's
                // data channel; a manager with none cannot relay.
                if (!requester_shm || !source_session.shm) && self.cfg.channel.is_none() {
                    return None;
                }
                let level = match tier {
                    Tier::Gpu => v2::Level::NeighbourGpu,
                    Tier::Dram => v2::Level::HostDram,
                    Tier::Nvme => v2::Level::HostNvme,
                };
                Some((
                    level,
                    v2::SourceRef {
                        instance_id: id.to_vec(),
                        session_incarnation: sid,
                    },
                ))
            }
            SourceId::Manager => {
                let level = match (requester_shm, tier) {
                    (_, Tier::Gpu) => return None,
                    (true, Tier::Dram) => v2::Level::HostDram,
                    (true, Tier::Nvme) => v2::Level::HostNvme,
                    (false, _) => v2::Level::RemoteNvme,
                };
                Some((
                    level,
                    v2::SourceRef {
                        instance_id: Vec::new(),
                        session_incarnation: 0,
                    },
                ))
            }
        }
    }
}

fn candidate(
    entry: &Entry,
    level: v2::Level,
    source: v2::SourceRef,
    begin: u64,
    walk: &super::index::ChainWalk,
    geometry: &crate::record::Descriptor,
) -> v2::Candidate {
    v2::Candidate {
        level: level as i32,
        tier: tier_to(entry.tier) as i32,
        source: Some(source),
        key: Some(key_to(entry.key)),
        begin_tokens: begin,
        pages: walk.pages,
        bytes: walk.pages * geometry.page_bytes()
            + if entry.row { geometry.row_bytes() } else { 0 },
        row: entry.row,
        nodes: walk
            .nodes
            .iter()
            .map(|node| v2::ChainNode {
                key: Some(key_to(node.key)),
                tail_pages: node.tail_pages,
                row: node.row,
            })
            .collect(),
        skip_pages: walk.skip_pages,
    }
}

/// A candidate ordered by level, match length, source and tier.
type Ranked = (i32, std::cmp::Reverse<u64>, SourceId, Tier, v2::Candidate);

pub(super) type LookupQueue = VecDeque<Queued>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        arena::Arena,
        pool::core::{OpenPeer, PoolConfig},
    };
    use tokio::sync::mpsc;

    #[test]
    fn a_manager_chain_with_disk_pages_is_priced_as_nvme() {
        let (mut core, _handle, _events) = Core::new(
            PoolConfig::default(),
            v2::SharedMemory::default(),
            Arena::new(1 << 20).unwrap(),
            None,
        );
        let (tx, _rx) = mpsc::channel(64);
        let sid = core
            .open(
                OpenPeer {
                    unix: true,
                    pid: std::process::id(),
                    uid: core.euid,
                    gid: core.egid,
                    pidfd: None,
                },
                v2::OpenSession {
                    manager_incarnation: core.incarnation,
                    instance_id: vec![1; 16],
                    process_nonce: vec![2; 16],
                    scope: Some(v2::Scope {
                        model_identity: vec![1; 32],
                        layout_fingerprint: vec![2; 16],
                        geometry: Some(v2::Geometry {
                            page_tokens: 64,
                            ranks: vec![v2::RankPlanes {
                                plane_bytes: vec![4096],
                                row_bytes: 4096,
                            }],
                            page_record_bytes: 4096,
                            row_record_bytes: 4096,
                        }),
                    }),
                    ..Default::default()
                },
                tx,
            )
            .unwrap();
        let parent = Key {
            consumed_tokens: 128,
            digest: [3; 16],
        };
        let child = Key {
            consumed_tokens: 256,
            digest: [4; 16],
        };
        let scope = core.sessions[&sid].scope;
        let index = &mut core.scopes.get_mut(&scope).unwrap().index;
        for (key, parent, tier) in [
            (parent, None, Tier::Nvme),
            (child, Some(parent), Tier::Dram),
        ] {
            index.upsert(
                SourceId::Manager,
                Entry {
                    packed: false,
                    required_profiles: Default::default(),
                    key,
                    parent,
                    bucket: [9; 16],
                    tail_pages: 2,
                    row: true,
                    tier,
                    bytes: 3 * 4096,
                },
            );
        }
        let (status, candidates) = core.answer_lookup(
            sid,
            v2::LookupRequest {
                boundaries: vec![v2::PrefixKey {
                    consumed_tokens: 0,
                    digest: vec![9; 16],
                }],
                before_tokens: 257,
                ..Default::default()
            },
        );
        assert_eq!(status, v2::LookupStatus::LookupOk);
        let candidate = &candidates[0];
        assert_eq!(candidate.key, Some(key_to(child)));
        assert_eq!(candidate.nodes.len(), 2);
        assert_eq!(candidate.level, v2::Level::HostNvme as i32);
        assert_eq!(candidate.tier, v2::Tier::Dram as i32);
    }
}
