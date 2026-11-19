/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiled chat templates with zero-output position annotations.

mod compiler;
#[cfg(test)]
mod tests;
mod value;
mod vm;

use serde_json::{Map, Value as Json};

use compiler::{Call, Instruction, Loop, Macro};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message} (at byte {offset})")]
pub struct Error {
    pub message: String,
    pub offset: usize,
}

impl Error {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            offset: 0,
        }
    }
}

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mark {
    Byte(usize),
    Position(usize),
    Hint(usize),
}

#[derive(Debug, PartialEq, Eq)]
pub struct RenderedTemplate {
    pub text: String,
    pub marks: Vec<Mark>,
    pub ordinary: Vec<std::ops::Range<usize>>,
}

pub struct CompiledTemplate {
    code: Vec<Instruction>,
    constants: Vec<Json>,
    names: Vec<String>,
    calls: Vec<Call>,
    loops: Vec<Loop>,
    macros: Vec<Macro>,
    spans: Vec<usize>,
    annotations: Vec<std::ops::Range<usize>>,
    loop_symbol: u32,
}

impl CompiledTemplate {
    pub fn compile(source: &str) -> Result<Self> {
        compiler::compile(source)
    }

    pub fn render(
        &self,
        context: &Map<String, Json>,
        hints: &[(usize, usize)],
    ) -> Result<RenderedTemplate> {
        self.render_with_defaults(context, hints, &[])
    }

    /// Renders with the context variables named in `defaults` treated as template text: their
    /// values are recipe or tokenizer defaults the request did not override, so the strings in
    /// them are controls. Every other context value is request text and renders as ordinary
    /// text in `RenderedTemplate::ordinary`.
    pub fn render_with_defaults(
        &self,
        context: &Map<String, Json>,
        hints: &[(usize, usize)],
        defaults: &[&str],
    ) -> Result<RenderedTemplate> {
        vm::render(self, context, hints, defaults)
    }

    pub fn annotation_spans(&self) -> &[std::ops::Range<usize>] {
        &self.annotations
    }

    #[cfg(test)]
    #[cfg_attr(test, allow(dead_code))]
    pub fn bytecode_bytes(&self) -> usize {
        self.code.len() * std::mem::size_of::<Instruction>()
    }
}
