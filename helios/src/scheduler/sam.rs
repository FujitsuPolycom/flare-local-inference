/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Per-request prompt and generated-output suffix automaton for copy proposals.
//!
//! `PromptSAM` indexes an immutable prompt tail in linear time and compacts transitions
//! into a flat sorted edge table suitable for zero-copy sharing across requests via `Arc`.
//! `PromptOutputSAM` pairs an `Arc<PromptSAM>` with a lazy per-request dynamic automaton
//! over committed generated output tokens (seeded with a prompt-tail overlap so matches
//! straddling the prompt/output boundary are resolved seamlessly).
//!
//! Queries return the longest suffix match of at least `min_match_len` tokens that occurs
//! strictly before the current end of the sequence, breaking ties in favor of the most
//! recent occurrence so output edits override earlier prompt text.

use std::sync::Arc;

/// Default minimum context suffix length required to emit a copy proposal.
pub const DEFAULT_MIN_MATCH_LEN: usize = 6;

/// Maximum number of trailing prompt tokens indexed by default (128 Ki tokens).
pub const DEFAULT_MAX_PROMPT_TOKENS: usize = 131_072;

/// Maximum number of generated output tokens indexed per request by default.
pub const DEFAULT_MAX_OUTPUT_TOKENS: usize = 131_072;

/// Trailing prompt tokens copied into the dynamic output automaton so matches
/// spanning the prompt/output boundary are indexed directly.
pub const DEFAULT_BOUNDARY_OVERLAP: usize = 512;

const NONE: u32 = u32::MAX;
const ROOT: u32 = 0;
const INLINE_EDGES: usize = 4;
const LINEAR_SCAN_THRESHOLD: usize = 8;

/// Configuration bounds and matching threshold for `PromptOutputSAM`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamConfig {
    /// Minimum matched suffix length required before proposing continuations.
    pub min_match_len: usize,
    /// Maximum number of trailing prompt tokens indexed.
    pub max_prompt_tokens: usize,
    /// Maximum number of generated output tokens indexed per request.
    pub max_output_tokens: usize,
    /// Number of trailing prompt tokens mirrored into the dynamic output automaton
    /// to index matches straddling the prompt/output boundary.
    pub boundary_overlap: usize,
}

impl Default for SamConfig {
    fn default() -> Self {
        Self {
            min_match_len: DEFAULT_MIN_MATCH_LEN,
            max_prompt_tokens: DEFAULT_MAX_PROMPT_TOKENS,
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
            boundary_overlap: DEFAULT_BOUNDARY_OVERLAP,
        }
    }
}

/// Region containing the winning occurrence's ending position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchSource {
    Prompt,
    Output,
}

/// Candidate continuation proposed by a suffix-automaton match.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatchProposal {
    /// Proposed continuation tokens immediately following `endpos`.
    pub tokens: Vec<u32>,
    /// Length of the matched context suffix.
    pub match_len: usize,
    /// Global ending index of the matched occurrence in `indexed_prompt ++ output`.
    pub endpos: usize,
    /// Whether `endpos` lies in the prompt or in generated output.
    pub source: MatchSource,
}

impl MatchProposal {
    /// Converts the continuation into the `-1`-padded `i64` representation expected by
    /// Helios copy-input packing, truncating at `capacity` if shorter.
    pub fn into_padded_i64(self, capacity: usize) -> (Vec<i64>, usize) {
        let count = self.tokens.len().min(capacity);
        let mut padded = vec![-1i64; capacity];
        for (dst, &tok) in padded[..count].iter_mut().zip(&self.tokens[..count]) {
            *dst = i64::from(tok);
        }
        (padded, count)
    }
}

/// Compact per-state outgoing transition store for mutable automaton construction
/// and dynamic output updates.
///
/// States with at most four outgoing transitions store `(token, target)` inline without
/// heap allocation. Higher-degree states spill to a flat open-addressed power-of-two table
/// using `target == NONE` as the empty-slot sentinel so every `u32` token ID is valid.
#[derive(Clone)]
enum MutableEdges {
    Inline([(u32, u32); INLINE_EDGES]),
    Table(Box<[(u32, u32)]>),
}

impl Default for MutableEdges {
    fn default() -> Self {
        Self::Inline([(0, NONE); INLINE_EDGES])
    }
}

#[inline]
fn hash_token(token: u32, mask: usize) -> usize {
    (token.wrapping_mul(0x9E37_79B9) as usize) & mask
}

impl MutableEdges {
    #[inline]
    fn get(&self, count: u32, token: u32) -> Option<u32> {
        match self {
            Self::Inline(entries) => {
                let n = count as usize;
                for &(k, target) in &entries[..n] {
                    if k == token {
                        return Some(target);
                    }
                }
                None
            }
            Self::Table(table) => {
                let mask = table.len() - 1;
                let mut slot = hash_token(token, mask);
                loop {
                    let (k, target) = table[slot];
                    if target == NONE {
                        return None;
                    }
                    if k == token {
                        return Some(target);
                    }
                    slot = (slot + 1) & mask;
                }
            }
        }
    }

    #[inline]
    fn set(&mut self, count: &mut u32, token: u32, target: u32) {
        match self {
            Self::Inline(entries) => {
                let n = *count as usize;
                for entry in &mut entries[..n] {
                    if entry.0 == token {
                        entry.1 = target;
                        return;
                    }
                }
                if n < INLINE_EDGES {
                    entries[n] = (token, target);
                    *count += 1;
                    return;
                }
                let mut table = vec![(0u32, NONE); 8].into_boxed_slice();
                for &(k, v) in entries.iter() {
                    insert_table(&mut table, k, v);
                }
                insert_table(&mut table, token, target);
                *count += 1;
                *self = Self::Table(table);
            }
            Self::Table(table) => {
                let mask = table.len() - 1;
                let mut slot = hash_token(token, mask);
                loop {
                    let (k, existing) = table[slot];
                    if existing == NONE {
                        if ((*count as usize + 1) * 8) >= table.len() * 5 {
                            let mut grown = vec![(0u32, NONE); table.len() * 2].into_boxed_slice();
                            for &(ek, ev) in table.iter() {
                                if ev != NONE {
                                    insert_table(&mut grown, ek, ev);
                                }
                            }
                            insert_table(&mut grown, token, target);
                            *table = grown;
                        } else {
                            table[slot] = (token, target);
                        }
                        *count += 1;
                        return;
                    }
                    if k == token {
                        table[slot].1 = target;
                        return;
                    }
                    slot = (slot + 1) & mask;
                }
            }
        }
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Self::Inline(_) => 0,
            Self::Table(table) => table.len() * size_of::<(u32, u32)>(),
        }
    }

    fn push_all_into(&self, count: u32, dst: &mut Vec<(u32, u32)>) {
        match self {
            Self::Inline(entries) => dst.extend_from_slice(&entries[..count as usize]),
            Self::Table(table) => {
                for &(k, v) in table.iter() {
                    if v != NONE {
                        dst.push((k, v));
                    }
                }
            }
        }
    }
}

#[inline]
fn insert_table(table: &mut [(u32, u32)], token: u32, target: u32) {
    let mask = table.len() - 1;
    let mut slot = hash_token(token, mask);
    while table[slot].1 != NONE {
        slot = (slot + 1) & mask;
    }
    table[slot] = (token, target);
}

#[derive(Clone)]
struct MutableState {
    len: u32,
    link: u32,
    max_endpos: u32,
    edge_count: u32,
    edges: MutableEdges,
}

impl MutableState {
    fn root() -> Self {
        Self {
            len: 0,
            link: NONE,
            max_endpos: NONE,
            edge_count: 0,
            edges: MutableEdges::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FrozenState {
    len: u32,
    link: u32,
    max_endpos: u32,
    edge_start: u32,
    edge_len: u32,
}

/// Immutable suffix automaton over a prompt (or its trailing bounded window).
///
/// Compacted into a flat CSR edge array sorted by token ID per state so a 128 Ki-token
/// prompt occupies under 10 MiB and can be shared zero-copy across requests via `Arc`.
#[derive(Debug)]
pub struct PromptSAM {
    tokens: Vec<u32>,
    states: Vec<FrozenState>,
    edges: Vec<(u32, u32)>,
    last_state: u32,
    /// Sorted `(state_id, prev_endpos)` pairs for states whose `max_endpos == tokens.len() - 1`
    /// that also occurred strictly earlier in `tokens`. Used only when no output tokens exist yet.
    last_chain_prev: Vec<(u32, u32)>,
}

impl PromptSAM {
    /// Builds an immutable `PromptSAM` capped at `DEFAULT_MAX_PROMPT_TOKENS`.
    pub fn build(prompt: &[u32]) -> Arc<Self> {
        Self::build_capped(prompt, DEFAULT_MAX_PROMPT_TOKENS)
    }

    /// Builds an immutable `PromptSAM` indexing at most the last `max_prompt_tokens` of `prompt`.
    pub fn build_capped(prompt: &[u32], max_prompt_tokens: usize) -> Arc<Self> {
        let start = prompt.len().saturating_sub(max_prompt_tokens);
        let slice = &prompt[start..];
        let len = slice.len();
        if len == 0 {
            return Arc::new(Self {
                tokens: Vec::new(),
                states: vec![FrozenState {
                    len: 0,
                    link: NONE,
                    max_endpos: NONE,
                    edge_start: 0,
                    edge_len: 0,
                }],
                edges: Vec::new(),
                last_state: ROOT,
                last_chain_prev: Vec::new(),
            });
        }

        let mut states = Vec::with_capacity(len.saturating_mul(2).min(len + 65_536).max(2));
        states.push(MutableState::root());
        let mut last = ROOT;
        let mut total_edges: usize = 0;

        for (pos, &token) in slice.iter().enumerate() {
            let cur = states.len() as u32;
            let cur_len = states[last as usize].len + 1;
            states.push(MutableState {
                len: cur_len,
                link: ROOT,
                max_endpos: pos as u32,
                edge_count: 0,
                edges: MutableEdges::default(),
            });

            let mut p = last;
            while p != NONE && states[p as usize].edges.get(states[p as usize].edge_count, token).is_none() {
                let st = &mut states[p as usize];
                let prev_count = st.edge_count;
                st.edges.set(&mut st.edge_count, token, cur);
                total_edges += (st.edge_count - prev_count) as usize;
                p = st.link;
            }

            if p == NONE {
                states[cur as usize].link = ROOT;
            } else {
                let q = states[p as usize]
                    .edges
                    .get(states[p as usize].edge_count, token)
                    .unwrap();
                if states[p as usize].len + 1 == states[q as usize].len {
                    states[cur as usize].link = q;
                } else {
                    let clone = states.len() as u32;
                    let q_link = states[q as usize].link;
                    let q_edges = states[q as usize].edges.clone();
                    let q_edge_count = states[q as usize].edge_count;
                    total_edges += q_edge_count as usize;
                    states.push(MutableState {
                        len: states[p as usize].len + 1,
                        link: q_link,
                        max_endpos: NONE,
                        edge_count: q_edge_count,
                        edges: q_edges,
                    });
                    while p != NONE {
                        let st = &mut states[p as usize];
                        if st.edges.get(st.edge_count, token) != Some(q) {
                            break;
                        }
                        st.edges.set(&mut st.edge_count, token, clone);
                        p = st.link;
                    }
                    states[q as usize].link = clone;
                    states[cur as usize].link = clone;
                }
            }
            last = cur;
        }

        // Exact bottom-up topological propagation of `max_endpos` and `prev_before_last`
        // in O(L) time via counting sort on state length.
        let num_states = states.len();
        let max_len = len;
        let mut counts = vec![0u32; max_len + 1];
        for st in &states {
            counts[st.len as usize] += 1;
        }
        for i in 1..=max_len {
            counts[i] += counts[i - 1];
        }
        let mut order = vec![0u32; num_states];
        for (idx, st) in states.iter().enumerate() {
            let l = st.len as usize;
            counts[l] -= 1;
            order[counts[l] as usize] = idx as u32;
        }

        let last_pos = (len - 1) as u32;
        let mut prev_before_last = vec![NONE; num_states];
        for (idx, st) in states.iter().enumerate() {
            if st.max_endpos != NONE && st.max_endpos < last_pos {
                prev_before_last[idx] = st.max_endpos;
            }
        }

        for &state_id in order.iter().rev() {
            let parent = states[state_id as usize].link;
            if parent == NONE {
                continue;
            }
            let child_max = states[state_id as usize].max_endpos;
            if child_max != NONE {
                let p_max = &mut states[parent as usize].max_endpos;
                if *p_max == NONE || child_max > *p_max {
                    *p_max = child_max;
                }
            }
            let child_prev = prev_before_last[state_id as usize];
            if child_prev != NONE {
                let p_prev = &mut prev_before_last[parent as usize];
                if *p_prev == NONE || child_prev > *p_prev {
                    *p_prev = child_prev;
                }
            }
        }

        let mut last_chain_prev = Vec::new();
        let mut frozen_states = Vec::with_capacity(num_states);
        let mut frozen_edges = Vec::with_capacity(total_edges);
        for (idx, st) in states.iter().enumerate() {
            if st.max_endpos == last_pos && prev_before_last[idx] != NONE {
                last_chain_prev.push((idx as u32, prev_before_last[idx]));
            }
            let edge_start = frozen_edges.len() as u32;
            st.edges.push_all_into(st.edge_count, &mut frozen_edges);
            let edge_end = frozen_edges.len();
            let edge_len = edge_end - edge_start as usize;
            if edge_len > 1 {
                frozen_edges[edge_start as usize..edge_end].sort_unstable_by_key(|&(k, _)| k);
            }
            frozen_states.push(FrozenState {
                len: st.len,
                link: st.link,
                max_endpos: st.max_endpos,
                edge_start,
                edge_len: edge_len as u32,
            });
        }
        // `idx` increases monotonically so `last_chain_prev` is already sorted by state_id.

        Arc::new(Self {
            tokens: slice.to_vec(),
            states: frozen_states,
            edges: frozen_edges,
            last_state: last,
            last_chain_prev,
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn transition_count(&self) -> usize {
        self.edges.len()
    }

    /// Heap memory footprint of this immutable prompt automaton in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.tokens.capacity() * size_of::<u32>()
            + self.states.capacity() * size_of::<FrozenState>()
            + self.edges.capacity() * size_of::<(u32, u32)>()
            + self.last_chain_prev.capacity() * size_of::<(u32, u32)>()
    }

    #[inline]
    fn transition(&self, state: u32, token: u32) -> Option<u32> {
        let st = &self.states[state as usize];
        let start = st.edge_start as usize;
        let len = st.edge_len as usize;
        let slice = &self.edges[start..start + len];
        if len <= LINEAR_SCAN_THRESHOLD {
            for &(k, target) in slice {
                if k == token {
                    return Some(target);
                }
            }
            None
        } else {
            slice
                .binary_search_by_key(&token, |&(k, _)| k)
                .ok()
                .map(|i| slice[i].1)
        }
    }

    #[inline]
    fn step(&self, mut state: u32, mut matched_len: usize, token: u32) -> (u32, usize) {
        while state != ROOT && self.transition(state, token).is_none() {
            state = self.states[state as usize].link;
            matched_len = self.states[state as usize].len as usize;
        }
        if let Some(next) = self.transition(state, token) {
            (next, matched_len + 1)
        } else {
            (ROOT, 0)
        }
    }

    fn walk_context(&self, context: &[u32]) -> (u32, usize) {
        if self.tokens.is_empty() || context.is_empty() {
            return (ROOT, 0);
        }
        let mut state = ROOT;
        let mut matched_len = 0usize;
        let start = context.len().saturating_sub(self.tokens.len());
        for &token in &context[start..] {
            (state, matched_len) = self.step(state, matched_len, token);
        }
        (state, matched_len)
    }

    fn prev_before_last(&self, state: u32) -> Option<u32> {
        self.last_chain_prev
            .binary_search_by_key(&state, |&(s, _)| s)
            .ok()
            .map(|i| self.last_chain_prev[i].1)
    }

    /// Resolves `(state, matched_len)` to `(match_len, endpos)` obeying `endpos <= max_valid_endpos`.
    fn resolve_match(
        &self,
        mut state: u32,
        mut matched_len: usize,
        allow_last_pos: bool,
        min_match_len: usize,
    ) -> Option<(usize, usize)> {
        if self.tokens.is_empty() {
            return None;
        }
        let last_pos = (self.tokens.len() - 1) as u32;
        while state != ROOT && matched_len >= min_match_len {
            let max_end = self.states[state as usize].max_endpos;
            if max_end != NONE {
                if allow_last_pos || max_end < last_pos {
                    return Some((matched_len, max_end as usize));
                }
                if let Some(prev) = self.prev_before_last(state) {
                    return Some((matched_len, prev as usize));
                }
            }
            state = self.states[state as usize].link;
            if state == NONE {
                break;
            }
            matched_len = matched_len.min(self.states[state as usize].len as usize);
        }
        None
    }
}

/// Rooted Link-Cut Tree over the dynamic output automaton's suffix-link tree.
///
/// Maintains exact `max_endpos` across all ancestors of every appended state in
/// `O(log M)` amortized time (avoiding quadratic degradation on long repetitive outputs)
/// while allowing immutable `&self` point queries.
#[derive(Clone, Debug)]
struct LctNode {
    ch: [u32; 2],
    fa: u32,
    pos: u32,
    lazy: u32,
}

#[derive(Clone, Debug)]
struct RootedLct {
    nodes: Vec<LctNode>,
    splay_stack: Vec<u32>,
}

impl RootedLct {
    fn new() -> Self {
        Self {
            nodes: vec![LctNode {
                ch: [NONE, NONE],
                fa: NONE,
                pos: NONE,
                lazy: NONE,
            }],
            splay_stack: Vec::with_capacity(64),
        }
    }

    #[inline]
    fn push_node(&mut self, pos: u32) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(LctNode {
            ch: [NONE, NONE],
            fa: NONE,
            pos,
            lazy: NONE,
        });
        id
    }

    #[inline]
    fn is_root(&self, x: u32) -> bool {
        let fa = self.nodes[x as usize].fa;
        fa == NONE || (self.nodes[fa as usize].ch[0] != x && self.nodes[fa as usize].ch[1] != x)
    }

    #[inline]
    fn apply_lazy(&mut self, x: u32, val: u32) {
        if x != NONE {
            let node = &mut self.nodes[x as usize];
            node.pos = val;
            node.lazy = val;
        }
    }

    #[inline]
    fn push_down(&mut self, x: u32) {
        let lazy = self.nodes[x as usize].lazy;
        if lazy != NONE {
            let [left, right] = self.nodes[x as usize].ch;
            self.apply_lazy(left, lazy);
            self.apply_lazy(right, lazy);
            self.nodes[x as usize].lazy = NONE;
        }
    }

    #[inline]
    fn rotate(&mut self, x: u32) {
        let y = self.nodes[x as usize].fa;
        let z = self.nodes[y as usize].fa;
        let k = usize::from(self.nodes[y as usize].ch[1] == x);
        if !self.is_root(y) {
            let side = usize::from(self.nodes[z as usize].ch[1] == y);
            self.nodes[z as usize].ch[side] = x;
        }
        self.nodes[x as usize].fa = z;
        let w = self.nodes[x as usize].ch[k ^ 1];
        self.nodes[y as usize].ch[k] = w;
        if w != NONE {
            self.nodes[w as usize].fa = y;
        }
        self.nodes[x as usize].ch[k ^ 1] = y;
        self.nodes[y as usize].fa = x;
    }

    fn splay(&mut self, x: u32) {
        self.splay_stack.clear();
        let mut cur = x;
        self.splay_stack.push(cur);
        while !self.is_root(cur) {
            cur = self.nodes[cur as usize].fa;
            self.splay_stack.push(cur);
        }
        while let Some(node) = self.splay_stack.pop() {
            self.push_down(node);
        }
        while !self.is_root(x) {
            let y = self.nodes[x as usize].fa;
            let z = self.nodes[y as usize].fa;
            if !self.is_root(y) {
                let zig_zig =
                    (self.nodes[y as usize].ch[0] == x) == (self.nodes[z as usize].ch[0] == y);
                if zig_zig {
                    self.rotate(y);
                } else {
                    self.rotate(x);
                }
            }
            self.rotate(x);
        }
    }

    fn access(&mut self, x: u32) {
        let mut last_child = NONE;
        let mut cur = x;
        while cur != NONE {
            self.splay(cur);
            self.nodes[cur as usize].ch[1] = last_child;
            last_child = cur;
            cur = self.nodes[cur as usize].fa;
        }
        self.splay(x);
    }

    /// Detaches `x` from its current suffix-link parent and returns its exact `max_endpos`.
    fn cut_parent(&mut self, x: u32) -> u32 {
        self.access(x);
        let pos = self.nodes[x as usize].pos;
        let left = self.nodes[x as usize].ch[0];
        if left != NONE {
            self.nodes[left as usize].fa = NONE;
            self.nodes[x as usize].ch[0] = NONE;
        }
        pos
    }

    #[inline]
    fn link(&mut self, child: u32, parent: u32) {
        self.nodes[child as usize].fa = parent;
    }

    /// Sets `max_endpos = pos` on every state along the suffix-link path from `cur` to root.
    fn update_path_to_root(&mut self, cur: u32, pos: u32) {
        self.access(cur);
        self.apply_lazy(cur, pos);
    }

    /// Reads the exact `max_endpos` of `x` without mutating the tree.
    #[inline]
    fn get_pos_immutable(&self, x: u32) -> u32 {
        let mut best = self.nodes[x as usize].pos;
        let mut cur = x;
        loop {
            let node = &self.nodes[cur as usize];
            if node.lazy != NONE {
                best = node.lazy;
            }
            if self.is_root(cur) {
                break;
            }
            cur = node.fa;
        }
        best
    }

    fn memory_bytes(&self) -> usize {
        self.nodes.capacity() * size_of::<LctNode>()
            + self.splay_stack.capacity() * size_of::<u32>()
    }
}

/// Dynamic per-request suffix automaton covering the prompt-tail overlap plus
/// committed output tokens (lagged by one token so every state has a valid continuation).
#[derive(Clone)]
struct OutputSAM {
    base_offset: usize,
    indexed_len: usize,
    states: Vec<MutableState>,
    lct: RootedLct,
    last: u32,
}

impl OutputSAM {
    fn new(base_offset: usize, seed: &[u32]) -> Self {
        let mut sam = Self {
            base_offset,
            indexed_len: 0,
            states: vec![MutableState::root()],
            lct: RootedLct::new(),
            last: ROOT,
        };
        for &tok in seed {
            sam.extend(tok);
        }
        sam
    }

    fn extend(&mut self, token: u32) {
        let local_pos = self.indexed_len as u32;
        self.indexed_len += 1;

        let cur = self.states.len() as u32;
        let cur_len = self.states[self.last as usize].len + 1;
        self.states.push(MutableState {
            len: cur_len,
            link: ROOT,
            max_endpos: local_pos,
            edge_count: 0,
            edges: MutableEdges::default(),
        });
        let lct_cur = self.lct.push_node(local_pos);
        debug_assert_eq!(lct_cur, cur);

        let mut p = self.last;
        while p != NONE
            && self.states[p as usize]
                .edges
                .get(self.states[p as usize].edge_count, token)
                .is_none()
        {
            let st = &mut self.states[p as usize];
            st.edges.set(&mut st.edge_count, token, cur);
            p = st.link;
        }

        if p == NONE {
            self.states[cur as usize].link = ROOT;
            self.lct.link(cur, ROOT);
        } else {
            let q = self.states[p as usize]
                .edges
                .get(self.states[p as usize].edge_count, token)
                .unwrap();
            if self.states[p as usize].len + 1 == self.states[q as usize].len {
                self.states[cur as usize].link = q;
                self.lct.link(cur, q);
            } else {
                let clone = self.states.len() as u32;
                let old_parent = self.states[q as usize].link;
                let q_pos = self.lct.cut_parent(q);
                let q_edges = self.states[q as usize].edges.clone();
                let q_edge_count = self.states[q as usize].edge_count;
                self.states.push(MutableState {
                    len: self.states[p as usize].len + 1,
                    link: old_parent,
                    max_endpos: q_pos,
                    edge_count: q_edge_count,
                    edges: q_edges,
                });
                let lct_clone = self.lct.push_node(q_pos);
                debug_assert_eq!(lct_clone, clone);
                if old_parent != NONE {
                    self.lct.link(clone, old_parent);
                }
                self.lct.link(q, clone);
                self.lct.link(cur, clone);

                while p != NONE {
                    let st = &mut self.states[p as usize];
                    if st.edges.get(st.edge_count, token) != Some(q) {
                        break;
                    }
                    st.edges.set(&mut st.edge_count, token, clone);
                    p = st.link;
                }
                self.states[q as usize].link = clone;
                self.states[cur as usize].link = clone;
            }
        }
        self.lct.update_path_to_root(cur, local_pos);
        self.last = cur;
    }

    #[inline]
    fn transition(&self, state: u32, token: u32) -> Option<u32> {
        let st = &self.states[state as usize];
        st.edges.get(st.edge_count, token)
    }

    #[inline]
    fn step(&self, mut state: u32, mut matched_len: usize, token: u32) -> (u32, usize) {
        while state != ROOT && self.transition(state, token).is_none() {
            state = self.states[state as usize].link;
            matched_len = self.states[state as usize].len as usize;
        }
        if let Some(next) = self.transition(state, token) {
            (next, matched_len + 1)
        } else {
            (ROOT, 0)
        }
    }

    /// Finds the longest suffix of `indexed_tokens ++ [last_token]` already occurring in
    /// `self`, returning `(state, match_len)` in O(1) amortized time without scanning.
    #[inline]
    fn match_appended_token(&self, last_token: u32) -> (u32, usize) {
        let mut p = self.last;
        while p != NONE {
            if let Some(next) = self.transition(p, last_token) {
                return (next, self.states[p as usize].len as usize + 1);
            }
            p = self.states[p as usize].link;
        }
        (ROOT, 0)
    }

    fn walk_context(&self, context: &[u32]) -> (u32, usize) {
        if self.indexed_len == 0 || context.is_empty() {
            return (ROOT, 0);
        }
        let mut state = ROOT;
        let mut matched_len = 0usize;
        let start = context.len().saturating_sub(self.indexed_len);
        for &token in &context[start..] {
            (state, matched_len) = self.step(state, matched_len, token);
        }
        (state, matched_len)
    }

    #[inline]
    fn max_endpos_global(&self, state: u32) -> Option<usize> {
        if state == ROOT {
            return None;
        }
        let pos = self.lct.get_pos_immutable(state);
        (pos != NONE).then_some(self.base_offset + pos as usize)
    }

    fn state_count(&self) -> usize {
        self.states.len()
    }

    fn memory_bytes(&self) -> usize {
        let edges_heap: usize = self.states.iter().map(|s| s.edges.heap_bytes()).sum();
        self.states.capacity() * size_of::<MutableState>() + edges_heap + self.lct.memory_bytes()
    }
}

/// Split static-prompt + dynamic-output suffix automaton for one active request.
///
/// Combines a shared immutable `Arc<PromptSAM>` with a lazily initialized per-request
/// `OutputSAM` (lagged by one output token so every indexed state has an observed
/// continuation). Maintains an incremental cursor over `PromptSAM` so each decode step
/// updates and queries in sub-microsecond time regardless of prompt length.
#[derive(Clone)]
pub struct PromptOutputSAM {
    prompt: Arc<PromptSAM>,
    output: Vec<u32>,
    output_sam: Option<OutputSAM>,
    prompt_cursor_state: u32,
    prompt_cursor_len: usize,
    config: SamConfig,
}

impl PromptOutputSAM {
    /// Builds a `PromptOutputSAM` over `prompt` using `SamConfig::default()`.
    pub fn new(prompt: &[u32]) -> Self {
        Self::with_config(prompt, SamConfig::default())
    }

    /// Builds a `PromptOutputSAM` over `prompt` with explicit bounds and match threshold.
    pub fn with_config(prompt: &[u32], config: SamConfig) -> Self {
        let prompt_sam = PromptSAM::build_capped(prompt, config.max_prompt_tokens);
        Self::from_shared_prompt(prompt_sam, config)
    }

    /// Creates a per-request automaton sharing an existing immutable `Arc<PromptSAM>`.
    pub fn from_shared_prompt(prompt: Arc<PromptSAM>, mut config: SamConfig) -> Self {
        config.min_match_len = config.min_match_len.max(1);
        let cursor_state = prompt.last_state;
        let cursor_len = prompt.len();
        Self {
            prompt,
            output: Vec::new(),
            output_sam: None,
            prompt_cursor_state: cursor_state,
            prompt_cursor_len: cursor_len,
            config,
        }
    }

    pub fn prompt_sam(&self) -> &Arc<PromptSAM> {
        &self.prompt
    }

    pub fn config(&self) -> SamConfig {
        self.config
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt.len()
    }

    pub fn output_len(&self) -> usize {
        self.output.len()
    }

    pub fn total_len(&self) -> usize {
        self.prompt.len() + self.output.len()
    }

    /// Total automaton states across the shared prompt automaton and per-request output automaton.
    pub fn state_count(&self) -> usize {
        self.prompt.state_count()
            + self.output_sam.as_ref().map_or(0, OutputSAM::state_count)
    }

    /// Per-request dynamic memory in bytes (excluding the shared `Arc<PromptSAM>`).
    pub fn dynamic_memory_bytes(&self) -> usize {
        self.output.capacity() * size_of::<u32>()
            + self.output_sam.as_ref().map_or(0, OutputSAM::memory_bytes)
    }

    /// Combined memory footprint in bytes (shared `PromptSAM` plus dynamic per-request storage).
    pub fn memory_bytes(&self) -> usize {
        self.prompt.memory_bytes() + self.dynamic_memory_bytes()
    }

    #[inline]
    fn token_at(&self, global_pos: usize) -> u32 {
        let p_len = self.prompt.len();
        if global_pos < p_len {
            self.prompt.tokens[global_pos]
        } else {
            self.output[global_pos - p_len]
        }
    }

    /// Appends one committed output token in O(1) typical / O(log M) worst-case time.
    pub fn push_token(&mut self, token: u32) {
        if self.output.len() >= self.config.max_output_tokens {
            return;
        }
        // Maintain one-token lag in OutputSAM: before storing the new token, index the
        // previous output token so every state in OutputSAM has an observed continuation.
        if let Some(&prev_token) = self.output.last() {
            if self.output_sam.is_none() {
                let p_len = self.prompt.len();
                let overlap = p_len.min(self.config.boundary_overlap);
                let base_offset = p_len - overlap;
                let seed = &self.prompt.tokens[base_offset..p_len];
                self.output_sam = Some(OutputSAM::new(base_offset, seed));
            }
            if let Some(out_sam) = &mut self.output_sam {
                out_sam.extend(prev_token);
            }
        }
        self.output.push(token);
        if !self.prompt.is_empty() {
            (self.prompt_cursor_state, self.prompt_cursor_len) =
                self.prompt.step(self.prompt_cursor_state, self.prompt_cursor_len, token);
        }
    }

    /// Appends a slice of committed output tokens.
    pub fn push_slice(&mut self, tokens: &[u32]) {
        for &token in tokens {
            self.push_token(token);
        }
    }

    /// Proposes up to `max_draft_len` continuation tokens for the current sequence
    /// `indexed_prompt ++ output` in O(match_len_extension + draft_len) sub-microsecond time.
    pub fn propose(&self, max_draft_len: usize) -> Option<MatchProposal> {
        let total = self.total_len();
        if max_draft_len == 0 || total < 2 || total < self.config.min_match_len {
            return None;
        }

        let prompt_candidate = self.prompt.resolve_match(
            self.prompt_cursor_state,
            self.prompt_cursor_len,
            !self.output.is_empty(),
            self.config.min_match_len,
        );

        let output_candidate = self.output_sam.as_ref().and_then(|out_sam| {
            let last_tok = *self.output.last()?;
            let (state, mut match_len) = out_sam.match_appended_token(last_tok);
            let endpos = out_sam.max_endpos_global(state)?;
            // If the match reaches the start of the boundary-overlap window, extend leftward
            // directly against the earlier prompt prefix so long boundary matches stay exact.
            if out_sam.base_offset > 0 && endpos + 1 == out_sam.base_offset + match_len {
                let mut left_end = endpos + 1 - match_len;
                let mut query_left = total - match_len;
                while left_end > 0
                    && query_left > 0
                    && self.token_at(left_end - 1) == self.token_at(query_left - 1)
                {
                    left_end -= 1;
                    query_left -= 1;
                    match_len += 1;
                }
            }
            (match_len >= self.config.min_match_len).then_some((match_len, endpos))
        });

        self.build_proposal(prompt_candidate, output_candidate, max_draft_len)
    }

    /// Proposes up to `max_draft_len` continuation tokens for an arbitrary query `context`
    /// against this request's indexed `prompt ++ output`.
    pub fn propose_for_context(
        &self,
        context: &[u32],
        max_draft_len: usize,
    ) -> Option<MatchProposal> {
        let total = self.total_len();
        if max_draft_len == 0
            || total < 2
            || context.len() < self.config.min_match_len
        {
            return None;
        }

        let (p_state, p_len) = self.prompt.walk_context(context);
        let prompt_candidate = self.prompt.resolve_match(
            p_state,
            p_len,
            !self.output.is_empty(),
            self.config.min_match_len,
        );

        let output_candidate = self.output_sam.as_ref().and_then(|out_sam| {
            let (state, mut match_len) = out_sam.walk_context(context);
            let endpos = out_sam.max_endpos_global(state)?;
            if out_sam.base_offset > 0 && endpos + 1 == out_sam.base_offset + match_len {
                let mut left_end = endpos + 1 - match_len;
                let mut query_left = context.len() - match_len;
                while left_end > 0
                    && query_left > 0
                    && self.token_at(left_end - 1) == context[query_left - 1]
                {
                    left_end -= 1;
                    query_left -= 1;
                    match_len += 1;
                }
            }
            (match_len >= self.config.min_match_len).then_some((match_len, endpos))
        });

        self.build_proposal(prompt_candidate, output_candidate, max_draft_len)
    }

    /// Computes conditional copy proposals for every accepted-draft prefix length
    /// `a in 0..=active_drafts.len()` without rescanning the full sequence.
    ///
    /// Entry `a` matches the tentative suffix `current_sequence ++ active_drafts[..a]`
    /// against the committed prompt and output corpus, providing a building block for
    /// conditional copy lookup tables in 2-deep pipelined execution.
    pub fn propose_conditional_continuations(
        &self,
        active_drafts: &[u32],
        max_draft_len: usize,
    ) -> Vec<Option<MatchProposal>> {
        let mut results = Vec::with_capacity(active_drafts.len() + 1);
        let total = self.total_len();
        if max_draft_len == 0 || total < 2 {
            results.resize(active_drafts.len() + 1, None);
            return results;
        }

        let mut p_state = self.prompt_cursor_state;
        let mut p_len = self.prompt_cursor_len;

        let mut o_cursor = self.output_sam.as_ref().and_then(|out_sam| {
            let last_tok = *self.output.last()?;
            Some(out_sam.match_appended_token(last_tok))
        });

        for a in 0..=active_drafts.len() {
            if a > 0 {
                let tok = active_drafts[a - 1];
                if !self.prompt.is_empty() {
                    (p_state, p_len) = self.prompt.step(p_state, p_len, tok);
                }
                if let (Some(out_sam), Some((os, ol))) = (self.output_sam.as_ref(), o_cursor) {
                    o_cursor = Some(out_sam.step(os, ol, tok));
                }
            }

            let query_len = total + a;
            if query_len < self.config.min_match_len {
                results.push(None);
                continue;
            }

            let prompt_candidate = self.prompt.resolve_match(
                p_state,
                p_len,
                !self.output.is_empty(),
                self.config.min_match_len,
            );

            let output_candidate = self.output_sam.as_ref().and_then(|out_sam| {
                let (state, mut match_len) = o_cursor?;
                let endpos = out_sam.max_endpos_global(state)?;
                if out_sam.base_offset > 0 && endpos + 1 == out_sam.base_offset + match_len {
                    let mut left_end = endpos + 1 - match_len;
                    let mut query_left = query_len - match_len;
                    while left_end > 0 && query_left > 0 {
                        let q_tok = if query_left - 1 < total {
                            self.token_at(query_left - 1)
                        } else {
                            active_drafts[query_left - 1 - total]
                        };
                        if self.token_at(left_end - 1) != q_tok {
                            break;
                        }
                        left_end -= 1;
                        query_left -= 1;
                        match_len += 1;
                    }
                }
                (match_len >= self.config.min_match_len).then_some((match_len, endpos))
            });

            results.push(self.build_proposal(prompt_candidate, output_candidate, max_draft_len));
        }

        results
    }

    fn build_proposal(
        &self,
        prompt_candidate: Option<(usize, usize)>,
        output_candidate: Option<(usize, usize)>,
        max_draft_len: usize,
    ) -> Option<MatchProposal> {
        let best = match (prompt_candidate, output_candidate) {
            (Some(p), Some(o)) => Some(if o >= p { o } else { p }),
            (Some(p), None) => Some(p),
            (None, Some(o)) => Some(o),
            (None, None) => None,
        };
        let (match_len, endpos) = best?;
        let total = self.total_len();
        if endpos + 1 >= total {
            return None;
        }
        let count = max_draft_len.min(total - 1 - endpos);
        if count == 0 {
            return None;
        }
        let mut tokens = Vec::with_capacity(count);
        for offset in 0..count {
            tokens.push(self.token_at(endpos + 1 + offset));
        }
        let source = if endpos >= self.prompt.len() {
            MatchSource::Output
        } else {
            MatchSource::Prompt
        };
        Some(MatchProposal {
            tokens,
            match_len,
            endpos,
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force reference oracle: finds the longest suffix of `query` occurring
    /// strictly before `corpus.len() - 1`, breaking ties by the largest ending position.
    fn brute_force_oracle(
        corpus: &[u32],
        prompt_len: usize,
        query: &[u32],
        min_match_len: usize,
        max_draft_len: usize,
    ) -> Option<MatchProposal> {
        let n = corpus.len();
        let min_len = min_match_len.max(1);
        if max_draft_len == 0 || n < 2 || query.len() < min_len {
            return None;
        }
        let mut best: Option<(usize, usize)> = None;
        for endpos in 0..n - 1 {
            let max_k = (endpos + 1).min(query.len());
            let mut k = 0;
            while k < max_k && corpus[endpos - k] == query[query.len() - 1 - k] {
                k += 1;
            }
            if k >= min_len && best.is_none_or(|(bl, be)| (k, endpos) > (bl, be)) {
                best = Some((k, endpos));
            }
        }
        let (match_len, endpos) = best?;
        let count = max_draft_len.min(n - 1 - endpos);
        let tokens = corpus[endpos + 1..endpos + 1 + count].to_vec();
        let source = if endpos >= prompt_len {
            MatchSource::Output
        } else {
            MatchSource::Prompt
        };
        Some(MatchProposal {
            tokens,
            match_len,
            endpos,
            source,
        })
    }

    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
    }

    #[test]
    fn prompt_only_match_succeeds_with_zero_prior_output() {
        // Code-edit / RAG scenario: prompt contains a function definition, and the first
        // generated tokens begin quoting it with zero prior decoded output matches.
        assert_eq!(DEFAULT_MIN_MATCH_LEN, 6);
        let short_prompt = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 20, 30, 40, 50, 60];
        let short_sam = PromptOutputSAM::new(&short_prompt);
        assert_eq!(short_sam.propose(7), None, "5-token match must not trigger default min_match_len=6");

        let prompt = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
            20, 30, 40, 50, 60, 70,
        ];
        let mut sam = PromptOutputSAM::new(&prompt);
        // Even at output_len == 0, prompt self-match finds the earlier [20..=70].
        let p0 = sam.propose(7).expect("expected prompt match at output_len=0");
        assert_eq!(p0.match_len, 6);
        assert_eq!(p0.endpos, 6);
        assert_eq!(p0.source, MatchSource::Prompt);
        assert_eq!(p0.tokens, vec![80, 90, 100, 110, 120, 130, 140]);

        // After prefill emits the first output token `80`, suffix [20..=80] matches prompt.
        sam.push_token(80);
        let p1 = sam.propose(7).expect("expected prompt match after first output token");
        assert_eq!(p1.match_len, 7);
        assert_eq!(p1.endpos, 7);
        assert_eq!(p1.source, MatchSource::Prompt);
        assert_eq!(p1.tokens, vec![90, 100, 110, 120, 130, 140, 20]);
    }

    #[test]
    fn output_overrides_prompt_recency_on_tie() {
        // Prompt contains `fn foo() -> i32` = [1, 2, 3, 4, 5, 6, 10, 11, 12].
        // Output edits it to `fn foo() -> u64` = [1, 2, 3, 4, 5, 6, 99, 98, 97], then repeats [1..=6].
        let prompt = vec![1, 2, 3, 4, 5, 6, 10, 11, 12, 200, 201];
        let mut sam = PromptOutputSAM::new(&prompt);
        sam.push_slice(&[1, 2, 3, 4, 5, 6, 99, 98, 97, 300, 1, 2, 3, 4, 5, 6]);
        let proposal = sam.propose(5).expect("expected recency-updated output match");
        assert_eq!(proposal.source, MatchSource::Output);
        assert_eq!(proposal.match_len, 6);
        assert_eq!(proposal.tokens, vec![99, 98, 97, 300, 1]);
    }

    #[test]
    fn boundary_straddling_match_spans_prompt_into_output() {
        // Prompt ends with [10, 20, 30, 40, 50]; output starts with [60, 70, 80, 90, 999, 10..=60].
        // Longest suffix [10, 20, 30, 40, 50, 60] starts in prompt and ends in output!
        let prompt = vec![1, 2, 3, 10, 20, 30, 40, 50];
        let mut sam = PromptOutputSAM::new(&prompt);
        sam.push_slice(&[60, 70, 80, 90, 999, 10, 20, 30, 40, 50, 60]);
        let proposal = sam.propose(7).expect("expected boundary-straddling match");
        assert_eq!(proposal.match_len, 6);
        assert_eq!(proposal.endpos, 8);
        assert_eq!(proposal.source, MatchSource::Output);
        assert_eq!(proposal.tokens, vec![70, 80, 90, 999, 10, 20, 30]);

        // Also test boundary-straddling match exceeding a small boundary_overlap via leftward extension.
        let cfg = SamConfig {
            min_match_len: 3,
            max_prompt_tokens: 1024,
            max_output_tokens: 1024,
            boundary_overlap: 4,
        };
        let long_prompt: Vec<u32> = (1..=20).collect();
        let mut sam_small_overlap = PromptOutputSAM::with_config(&long_prompt, cfg);
        // Output continues with 21, 22, 23, then separator 999, then repeats 10..=21 (12 tokens straddling boundary!).
        let mut out = vec![21, 22, 23, 999];
        out.extend(10..=21);
        sam_small_overlap.push_slice(&out);
        let p = sam_small_overlap.propose(5).expect("expected leftward-extended boundary match");
        let mut corpus = long_prompt.clone();
        corpus.extend_from_slice(&out);
        let expected = brute_force_oracle(&corpus, long_prompt.len(), &corpus, 3, 5);
        assert_eq!(Some(p), expected);
    }

    #[test]
    fn edge_cases_empty_single_token_and_min_match_len() {
        let mut empty = PromptOutputSAM::new(&[]);
        assert_eq!(empty.propose(7), None);
        assert_eq!(empty.state_count(), 1);
        empty.push_token(42);
        assert_eq!(empty.propose(7), None);
        empty.push_slice(&[42, 42, 42, 42, 42]);
        assert_eq!(empty.propose(7), None, "5-token self-overlap is below default min_match_len=6");
        empty.push_token(42);
        let p = empty.propose(7).expect("expected output-only match on empty prompt");
        assert_eq!(p.match_len, 6);
        assert_eq!(p.tokens, vec![42]);

        let single = PromptOutputSAM::new(&[7]);
        assert_eq!(single.propose(7), None);
        assert!(single.state_count() <= 2);

        // min_match_len = 2 vs 3 vs 6 and max_draft_len = 0, 1, 23
        let prompt = vec![5, 6, 7, 8, 9, 10, 5, 6];
        let sam2 = PromptOutputSAM::with_config(
            &prompt,
            SamConfig { min_match_len: 2, ..Default::default() },
        );
        let sam3 = PromptOutputSAM::with_config(
            &prompt,
            SamConfig { min_match_len: 3, ..Default::default() },
        );
        assert_eq!(sam2.propose(0), None);
        assert_eq!(sam2.propose(23).unwrap().tokens, vec![7, 8, 9, 10, 5, 6]);
        assert_eq!(sam3.propose(23), None);

        // Token ID u32::MAX must work seamlessly in both Inline and Table edge stores.
        let max_tok_prompt = vec![u32::MAX, 1, 2, 3, 4, u32::MAX, 5, u32::MAX, 6, u32::MAX, 7, u32::MAX, 8, u32::MAX, 1, 2];
        let sam_max = PromptOutputSAM::with_config(
            &max_tok_prompt,
            SamConfig { min_match_len: 3, ..Default::default() },
        );
        let p_max = sam_max.propose(4).unwrap();
        assert_eq!(p_max.tokens, vec![3, 4, u32::MAX, 5]);
    }

    #[test]
    fn exact_oracle_equivalence_randomized_streams_and_repetitions() {
        let mut rng = Lcg(0xDEAD_BEEF_CAFE_1234);
        for trial in 0..120 {
            let alphabet = match trial % 5 {
                0 => 2,  // heavy repetition & deep suffix links
                1 => 4,  // high-degree overlap
                2 => 8,  // dense collisions
                3 => 16, // moderate vocabulary
                _ => 64, // sparse
            };
            let min_match_len = 1 + (trial % 4);
            let prompt_len = match trial % 6 {
                0 => 0,
                1 => 1,
                2 => 5,
                3 => 24,
                4 => 80,
                _ => 180,
            };
            let output_len = 40 + (rng.next_u32() as usize % 90);
            let prompt: Vec<u32> = (0..prompt_len).map(|_| rng.next_u32() % alphabet).collect();
            let config = SamConfig {
                min_match_len,
                max_prompt_tokens: 1024,
                max_output_tokens: 1024,
                boundary_overlap: 256,
            };
            let mut sam = PromptOutputSAM::with_config(&prompt, config);
            let mut corpus = prompt.clone();

            // Check output_len == 0
            for max_draft in [1, 7, 23] {
                let actual = sam.propose(max_draft);
                let expected = brute_force_oracle(&corpus, prompt.len(), &corpus, min_match_len, max_draft);
                assert_eq!(actual, expected, "trial={trial} step=0 max_draft={max_draft}");
            }

            for step in 0..output_len {
                // Inject periodic copied chunks and single-token bursts to stress LCT + cloning.
                let tok = if step % 17 == 0 && corpus.len() > 6 {
                    corpus[corpus.len() - 5]
                } else if trial % 10 == 0 && step % 25 < 10 {
                    0
                } else {
                    rng.next_u32() % alphabet
                };
                sam.push_token(tok);
                corpus.push(tok);

                let max_draft = [1, 7, 23][step % 3];
                let actual = sam.propose(max_draft);
                let expected = brute_force_oracle(&corpus, prompt.len(), &corpus, min_match_len, max_draft);
                assert_eq!(
                    actual, expected,
                    "propose mismatch trial={trial} step={step} min_match={min_match_len}"
                );

                if step % 9 == 0 {
                    // Verify arbitrary-context query matches oracle too.
                    let q_len = 1 + (rng.next_u32() as usize % 20);
                    let query: Vec<u32> = (0..q_len).map(|_| rng.next_u32() % alphabet).collect();
                    let actual_q = sam.propose_for_context(&query, 7);
                    let expected_q = brute_force_oracle(&corpus, prompt.len(), &query, min_match_len, 7);
                    assert_eq!(actual_q, expected_q, "context mismatch trial={trial} step={step}");
                }
            }
        }
    }

    #[test]
    fn adversarial_repeated_runs_stay_exact_under_lct() {
        // Exercises long A^N + B + A^K runs where naive suffix-link walking degrades or loses recency.
        let mut prompt = vec![1u32; 300];
        prompt.push(2);
        let mut sam = PromptOutputSAM::new(&prompt);
        let mut corpus = prompt.clone();
        let mut out = vec![1u32; 400];
        out.push(3);
        out.extend(std::iter::repeat_n(1u32, 50));
        for &tok in &out {
            sam.push_token(tok);
            corpus.push(tok);
        }
        let actual = sam.propose(7);
        let expected = brute_force_oracle(&corpus, prompt.len(), &corpus, DEFAULT_MIN_MATCH_LEN, 7);
        assert_eq!(actual, expected);
        assert_eq!(actual.unwrap().tokens[0], 3);
    }

    #[test]
    fn conditional_continuations_match_oracle_for_all_prefix_lengths() {
        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let mut sam = PromptOutputSAM::new(&prompt);
        sam.push_slice(&[100, 1, 2, 3, 4, 5, 6, 99, 98, 200, 1, 2, 3, 4, 5, 6]);
        let mut corpus = prompt.clone();
        corpus.extend_from_slice(&[100, 1, 2, 3, 4, 5, 6, 99, 98, 200, 1, 2, 3, 4, 5, 6]);

        // Active drafts branch across both output-matching (99, 98) and prompt-matching (7, 8, 9) paths.
        for active_drafts in [
            vec![7, 8, 9, 10, 11, 12, 13],
            vec![99, 98, 200, 1, 2, 3, 4],
            vec![777, 1, 2, 3, 4, 5, 6],
        ] {
            let cond = sam.propose_conditional_continuations(&active_drafts, 7);
            assert_eq!(cond.len(), active_drafts.len() + 1);
            for a in 0..=active_drafts.len() {
                let mut query = corpus.clone();
                query.extend_from_slice(&active_drafts[..a]);
                let expected = brute_force_oracle(&corpus, prompt.len(), &query, DEFAULT_MIN_MATCH_LEN, 7);
                assert_eq!(cond[a], expected, "conditional mismatch at a={a}");
            }
        }
    }

    #[test]
    fn shared_prompt_sam_isolates_per_request_outputs_and_bounds_states() {
        let prompt: Vec<u32> = (0..2048).map(|i| (i * 31 + 7) % 128).collect();
        let shared = PromptSAM::build(&prompt);
        assert!(shared.state_count() <= 2 * prompt.len());
        assert!(shared.transition_count() <= 3 * prompt.len());

        let mut req_a = PromptOutputSAM::from_shared_prompt(Arc::clone(&shared), SamConfig::default());
        let mut req_b = PromptOutputSAM::from_shared_prompt(Arc::clone(&shared), SamConfig::default());
        assert!(Arc::ptr_eq(req_a.prompt_sam(), req_b.prompt_sam()));
        assert_eq!(req_a.dynamic_memory_bytes(), 0);
        assert_eq!(req_b.dynamic_memory_bytes(), 0);

        req_a.push_slice(&[900, 901, 902, 903, 904, 905, 906, 907, 908, 900, 901, 902, 903, 904, 905]);
        req_b.push_slice(&[800, 801, 802, 803, 804, 805, 806, 807, 808, 900, 901, 902, 903, 904, 905]);

        let pa = req_a.propose(4).expect("req_a should match its own output");
        assert_eq!(pa.tokens, vec![906, 907, 908, 900]);
        assert_eq!(req_b.propose(4), None, "req_b must not see req_a outputs");

        // Verify prompt cap bounds indexed tokens and states for oversized prompts.
        let huge = vec![1u32; 4096];
        let capped = PromptSAM::build_capped(&huge, 512);
        assert_eq!(capped.len(), 512);
        assert!(capped.state_count() <= 2 * 512);
    }

    #[test]
    #[ignore = "CPU microbenchmark; run in release mode with --ignored --nocapture --test-threads=1"]
    fn benchmark_prompt_output_sam_build_memory_and_step_latency() {
        let mut rng = Lcg(0x1234_5678_9ABC_DEF0);
        for &prompt_len in &[8_192usize, 32_768, 131_072] {
            // Realistic code/text-like distribution: Zipf-like repeated lines + identifiers.
            let mut prompt = Vec::with_capacity(prompt_len);
            let mut line_pool: Vec<Vec<u32>> = (0..256)
                .map(|_| {
                    let len = 8 + (rng.next_u32() as usize % 24);
                    (0..len).map(|_| rng.next_u32() % 4096).collect()
                })
                .collect();
            while prompt.len() < prompt_len {
                if rng.next_u32() % 4 == 0 {
                    let line = &line_pool[rng.next_u32() as usize % line_pool.len()];
                    let take = line.len().min(prompt_len - prompt.len());
                    prompt.extend_from_slice(&line[..take]);
                } else {
                    prompt.push(rng.next_u32() % 32_768);
                }
            }
            // Add a few distinct motifs into line_pool for decode step queries.
            line_pool.push(prompt[prompt_len / 2..prompt_len / 2 + 32].to_vec());

            let build_start = std::time::Instant::now();
            let shared = PromptSAM::build(&prompt);
            let build_elapsed = build_start.elapsed();

            let mut sam = PromptOutputSAM::from_shared_prompt(Arc::clone(&shared), SamConfig::default());
            assert!(shared.state_count() <= 2 * prompt_len);

            // Simulate 1,000 decode steps alternating prompt quotes and fresh tokens.
            let steps = 1_000usize;
            let mut hits = 0usize;
            let mut total_proposed = 0usize;
            let step_start = std::time::Instant::now();
            for step in 0..steps {
                let tok = if step % 32 < 20 {
                    prompt[(prompt_len / 2 + step) % prompt_len]
                } else {
                    100_000 + step as u32
                };
                sam.push_token(tok);
                if let Some(proposal) = std::hint::black_box(sam.propose(7)) {
                    hits += 1;
                    total_proposed += proposal.tokens.len();
                }
            }
            let step_elapsed = step_start.elapsed();
            let ns_per_step = step_elapsed.as_nanos() as f64 / steps as f64;

            println!(
                "PromptOutputSAM L={prompt_len}: states={} ({:.2}x L) edges={} mem={:.2} MiB build={:.2} ms step_avg={:.0} ns/step hits={hits}/{steps} proposed={total_proposed}",
                shared.state_count(),
                shared.state_count() as f64 / prompt_len as f64,
                shared.transition_count(),
                shared.memory_bytes() as f64 / (1024.0 * 1024.0),
                build_elapsed.as_secs_f64() * 1e3,
                ns_per_step,
            );
        }
    }
}
