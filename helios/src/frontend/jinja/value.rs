/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use std::cell::{Cell, RefCell};
use std::fmt::Write;
use std::rc::Rc;

use serde_json::Value as Json;

use super::{Error, Result};

/// What a string value remembers about its characters: the end offsets of text a checkpoint hint
/// names, and the byte ranges that came from the request.
#[derive(Clone, Debug, Default)]
pub(super) struct Notes {
    pub hints: Vec<usize>,
    pub ordinary: Vec<std::ops::Range<usize>>,
}

impl Notes {
    pub fn hints(hints: Vec<usize>) -> Self {
        Self { hints, ..Self::default() }
    }
}

#[derive(Clone, Debug, Default)]
pub(super) enum Value<'a> {
    #[default]
    Undefined,
    None,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    /// Template text may use the checkpoint's added vocabulary.
    Str(&'a str),
    /// Request text must use ordinary tokens.
    Text(&'a str),
    String(Rc<(String, Notes)>),
    Array(&'a [Json]),
    Object(&'a serde_json::Map<String, Json>),
    List(Rc<Vec<Value<'a>>>),
    Map(Rc<Vec<(Value<'a>, Value<'a>)>>),
    Namespace(Rc<RefCell<Vec<(Value<'a>, Value<'a>)>>>),
    Loop(Rc<LoopValue<'a>>),
    Macro(u32),
    Builtin,
}

#[derive(Debug)]
pub(super) struct LoopValue<'a> {
    pub items: Vec<Value<'a>>,
    pub index: Cell<usize>,
}

impl<'a> Value<'a> {
    pub fn json(value: &'a Json) -> Self {
        match value {
            Json::Null => Self::None,
            Json::Bool(v) => Self::Bool(*v),
            Json::Number(v) => v
                .as_i64()
                .map(Self::Int)
                .or_else(|| v.as_u64().map(Self::UInt))
                .unwrap_or_else(|| Self::Float(v.as_f64().unwrap())),
            Json::String(v) => Self::Text(v),
            Json::Array(v) => Self::Array(v),
            Json::Object(v) => Self::Object(v),
        }
    }

    /// A value the recipe owns, such as a template default: every string in it is template text.
    pub fn constant(value: &'a Json) -> Self {
        match value {
            Json::String(text) => Self::Str(text),
            Json::Array(items) => Self::list(items.iter().map(Self::constant).collect()),
            Json::Object(items) => Self::Map(Rc::new(
                items
                    .iter()
                    .map(|(key, item)| (Self::Str(key), Self::constant(item)))
                    .collect(),
            )),
            _ => Self::json(value),
        }
    }

    pub fn owned(value: Json) -> Self {
        match value {
            Json::Null => Self::None,
            Json::Bool(v) => Self::Bool(v),
            Json::Number(v) => v
                .as_i64()
                .map(Self::Int)
                .or_else(|| v.as_u64().map(Self::UInt))
                .unwrap_or_else(|| Self::Float(v.as_f64().unwrap())),
            Json::String(v) => Self::string(v),
            Json::Array(v) => Self::list(v.into_iter().map(Self::owned).collect()),
            Json::Object(v) => Self::Map(Rc::new(
                v.into_iter()
                    .map(|(k, v)| (Self::string(k), Self::owned(v)))
                    .collect(),
            )),
        }
    }

    pub fn string(value: String) -> Self {
        Self::annotated(value, Notes::default())
    }
    pub fn hints(&self) -> &[usize] {
        match self {
            Self::String(s) => &s.1.hints,
            _ => &[],
        }
    }
    pub fn annotated(value: String, notes: Notes) -> Self {
        Self::String(Rc::new((value, notes)))
    }
    pub fn ordinary(&self) -> Vec<std::ops::Range<usize>> {
        match self {
            Self::Text(text) => vec![0..text.len()],
            Self::String(s) => s.1.ordinary.clone(),
            _ => Vec::new(),
        }
    }
    /// Whether the value may hold request text. An array or object from the request is ordinary
    /// even when empty, and no deeper check is made.
    pub fn has_ordinary(&self) -> bool {
        match self {
            Self::Text(_) | Self::Array(_) | Self::Object(_) => true,
            Self::String(s) => !s.1.ordinary.is_empty(),
            Self::List(values) => values.iter().any(Self::has_ordinary),
            Self::Map(values) => values
                .iter()
                .any(|(key, value)| key.has_ordinary() || value.has_ordinary()),
            Self::Namespace(values) => values
                .borrow()
                .iter()
                .any(|(key, value)| key.has_ordinary() || value.has_ordinary()),
            _ => false,
        }
    }
    pub fn into_ordinary(self) -> Self {
        match self {
            Self::Str(text) => Self::Text(text),
            Self::String(s) => {
                let (text, notes) = Rc::unwrap_or_clone(s);
                let end = text.len();
                let ordinary = vec![0..end];
                Self::annotated(text, Notes { ordinary, ..notes })
            }
            Self::List(values) => {
                Self::list(values.iter().cloned().map(Self::into_ordinary).collect())
            }
            Self::Map(values) => Self::Map(Rc::new(
                values
                    .iter()
                    .map(|(key, value)| {
                        (key.clone().into_ordinary(), value.clone().into_ordinary())
                    })
                    .collect(),
            )),
            other => other,
        }
    }
    pub fn append(&self, output: &mut String, notes: &mut Notes) {
        let start = output.len();
        notes
            .hints
            .extend(self.hints().iter().map(|end| start + end));
        self.write(output);
        let mut append_range = |range: std::ops::Range<usize>| {
            if range.is_empty() {
                return;
            }
            let range = start + range.start..start + range.end;
            if let Some(last) = notes.ordinary.last_mut().filter(|last| last.end == range.start) {
                last.end = range.end;
            } else {
                notes.ordinary.push(range);
            }
        };
        match self {
            Self::String(s) => s.1.ordinary.iter().cloned().for_each(append_range),
            _ if self.has_ordinary() => append_range(0..output.len() - start),
            _ => {}
        }
    }

    /// The text between two byte offsets with the ordinary ranges that cover it. Only `trim` keeps
    /// checkpoint hints (`hints`): one at the end of the string stays at the end of the result.
    /// Every other string operation yields a string without hints.
    fn substring(&self, start: usize, end: usize, hints: bool) -> Self {
        // A borrowed slice of the whole string would match its checkpoint hint by address.
        match self {
            Self::Text(text) if hints || end - start != text.len() => {
                return Self::Text(&text[start..end]);
            }
            Self::Str(text) if hints || end - start != text.len() => {
                return Self::Str(&text[start..end]);
            }
            _ => {}
        }
        let text = &self.as_str().unwrap()[start..end];
        let hints = self
            .hints()
            .iter()
            .filter(|hint| hints && **hint > start && !text.is_empty())
            .map(|hint| (hint - start).min(text.len()))
            .collect();
        let ordinary = self
            .ordinary()
            .into_iter()
            .filter_map(|range| {
                let low = range.start.max(start);
                let high = range.end.min(end);
                (low < high).then(|| low - start..high - start)
            })
            .collect();
        Self::annotated(text.to_owned(), Notes { hints, ordinary })
    }
    /// The part of this string that `part`, a slice borrowed from it, covers.
    pub fn piece(&self, part: &str, hints: bool) -> Self {
        let start = part.as_ptr() as usize - self.as_str().unwrap().as_ptr() as usize;
        self.substring(start, start + part.len(), hints)
    }
    pub fn list(value: Vec<Self>) -> Self {
        Self::List(Rc::new(value))
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) | Self::Text(s) => Some(s),
            Self::String(s) => Some(&s.0),
            _ => None,
        }
    }
    pub fn integer(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::UInt(v) => i64::try_from(*v).ok(),
            Self::Bool(v) => Some(*v as i64),
            _ => None,
        }
    }
    pub fn number(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::UInt(v) => Some(*v as f64),
            _ => self.integer().map(|v| v as f64),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Str(s) | Self::Text(s) => s.chars().count(),
            Self::String(s) => s.0.chars().count(),
            Self::Array(v) => v.len(),
            Self::Object(v) => v.len(),
            Self::List(v) => v.len(),
            Self::Map(v) => v.len(),
            Self::Namespace(v) => v.borrow().len(),
            _ => 0,
        }
    }
    pub fn truthy(&self) -> bool {
        match self {
            Self::Undefined | Self::None => false,
            Self::Bool(v) => *v,
            Self::Int(v) => *v != 0,
            Self::UInt(v) => *v != 0,
            Self::Float(v) => *v != 0.0,
            Self::Loop(_) | Self::Macro(_) | Self::Builtin | Self::Namespace(_) => true,
            _ => self.len() != 0,
        }
    }
    pub fn mapping(&self) -> bool {
        matches!(self, Self::Object(_) | Self::Map(_))
    }
    pub fn sequence(&self) -> bool {
        matches!(self, Self::Array(_) | Self::List(_)) || self.as_str().is_some()
    }
    pub fn iterable(&self) -> bool {
        self.mapping() || self.sequence() || matches!(self, Self::Undefined)
    }

    pub fn attr(&self, name: &str) -> Self {
        match self {
            Self::Object(v) => v.get(name).map(Self::json).unwrap_or_default(),
            Self::Map(v) => v
                .iter()
                .find(|(k, _)| k.as_str() == Some(name))
                .map(|(_, v)| v.clone())
                .unwrap_or_default(),
            Self::Namespace(v) => v
                .borrow()
                .iter()
                .find(|(k, _)| k.as_str() == Some(name))
                .map(|(_, v)| v.clone())
                .unwrap_or_default(),
            Self::Loop(v) => {
                let i = v.index.get();
                match name {
                    "index0" => Self::Int(i as i64),
                    "index" => Self::Int(i as i64 + 1),
                    "length" => Self::Int(v.items.len() as i64),
                    "revindex0" => Self::Int((v.items.len() - i - 1) as i64),
                    "revindex" => Self::Int((v.items.len() - i) as i64),
                    "first" => Self::Bool(i == 0),
                    "last" => Self::Bool(i + 1 == v.items.len()),
                    "previtem" => i
                        .checked_sub(1)
                        .and_then(|i| v.items.get(i))
                        .cloned()
                        .unwrap_or_default(),
                    "nextitem" => v.items.get(i + 1).cloned().unwrap_or_default(),
                    _ => Self::Undefined,
                }
            }
            _ => Self::Undefined,
        }
    }

    pub fn index(&self, key: &Self) -> Self {
        if let Some(name) = key.as_str() {
            return self.attr(name);
        }
        let Some(mut index) = key.integer() else {
            return Self::Undefined;
        };
        if index < 0 {
            index += self.len() as i64;
        }
        let Ok(index) = usize::try_from(index) else {
            return Self::Undefined;
        };
        match self {
            Self::Array(v) => v.get(index).map(Self::json).unwrap_or_default(),
            Self::List(v) => v.get(index).cloned().unwrap_or_default(),
            _ => self
                .as_str()
                .and_then(|s| s.char_indices().nth(index))
                .map(|(byte, c)| self.substring(byte, byte + c.len_utf8(), false))
                .unwrap_or_default(),
        }
    }

    pub fn iter(&self) -> Result<Vec<Self>> {
        match self {
            Self::Array(v) => Ok(v.iter().map(Self::json).collect()),
            Self::List(v) => Ok(v.as_ref().clone()),
            Self::Object(v) => Ok(v.keys().map(|s| Self::Text(s)).collect()),
            Self::Map(v) => Ok(v.iter().map(|(key, _)| key.clone()).collect()),
            Self::Undefined | Self::None => Ok(Vec::new()),
            _ if self.as_str().is_some() => Ok(self
                .as_str()
                .unwrap()
                .char_indices()
                .map(|(byte, c)| self.substring(byte, byte + c.len_utf8(), false))
                .collect()),
            _ => Err(Error::new("value is not iterable")),
        }
    }

    pub fn pairs(&self) -> Result<Vec<(Self, Self)>> {
        match self {
            Self::Object(v) => Ok(v
                .iter()
                .map(|(k, v)| (Self::Text(k), Self::json(v)))
                .collect()),
            Self::Map(v) => Ok(v.as_ref().clone()),
            Self::Namespace(v) => Ok(v.borrow().clone()),
            Self::Undefined => Ok(Vec::new()),
            _ => Err(Error::new("value is not a map")),
        }
    }

    pub fn write(&self, output: &mut String) {
        match self {
            Self::Undefined => {}
            Self::None => output.push_str("None"),
            Self::Bool(v) => output.push_str(if *v { "True" } else { "False" }),
            Self::Int(v) => {
                let _ = write!(output, "{v}");
            }
            Self::UInt(v) => {
                let _ = write!(output, "{v}");
            }
            Self::Float(v) if v.is_nan() => output.push_str("nan"),
            Self::Float(v) if v.is_infinite() => {
                output.push_str(if v.is_sign_negative() { "-inf" } else { "inf" })
            }
            Self::Float(v) => output.push_str(&super::super::json::python_float(*v)),
            Self::Str(s) | Self::Text(s) => output.push_str(s),
            Self::String(s) => output.push_str(&s.0),
            _ => self.write_repr(output),
        }
    }

    fn write_repr(&self, output: &mut String) {
        if let Some(s) = self.as_str() {
            output.push_str(&serde_json::to_string(s).unwrap());
        } else if self.mapping() || matches!(self, Self::Namespace(_)) {
            output.push('{');
            for (i, (k, v)) in self.pairs().unwrap().into_iter().enumerate() {
                if i != 0 {
                    output.push_str(", ");
                }
                output.push_str(&serde_json::to_string(&k.text()).unwrap());
                output.push_str(": ");
                v.write_repr(output);
            }
            output.push('}');
        } else if self.sequence() {
            output.push('[');
            for (i, v) in self.iter().unwrap().iter().enumerate() {
                if i != 0 {
                    output.push_str(", ");
                }
                v.write_repr(output);
            }
            output.push(']');
        } else if matches!(self, Self::Loop(_) | Self::Macro(_)) {
            output.push_str("<object>");
        } else {
            self.write(output);
        }
    }

    pub fn text(&self) -> String {
        let mut s = String::new();
        self.write(&mut s);
        s
    }

    pub fn to_json(&self) -> Result<Json> {
        Ok(match self {
            Self::None | Self::Undefined => Json::Null,
            Self::Bool(v) => (*v).into(),
            Self::Int(v) => (*v).into(),
            Self::Float(v) => (*v).into(),
            Self::UInt(v) => (*v).into(),
            Self::Str(s) | Self::Text(s) => (*s).into(),
            Self::String(s) => s.0.as_str().into(),
            Self::Array(v) => Json::Array(v.to_vec()),
            Self::Object(v) => Json::Object((*v).clone()),
            Self::List(v) => Json::Array(v.iter().map(Self::to_json).collect::<Result<_>>()?),
            Self::Map(_) | Self::Namespace(_) => Json::Object(
                self.pairs()?
                    .into_iter()
                    .map(|(k, v)| Ok((k.text(), v.to_json()?)))
                    .collect::<Result<_>>()?,
            ),
            _ => return Err(Error::new("cannot serialize template object")),
        })
    }

    pub fn json_hints<'j>(&self, json: &'j Json, hints: &mut Vec<(&'j str, Vec<usize>)>) {
        match (self, json) {
            (Self::String(value), Json::String(text)) if !value.1.hints.is_empty() => {
                hints.push((text, value.1.hints.clone()));
            }
            (Self::List(values), Json::Array(items)) => {
                for (value, item) in values.iter().zip(items) {
                    value.json_hints(item, hints);
                }
            }
            (Self::Map(values), Json::Object(items)) => {
                for (key, item) in items {
                    if let Some((_, value)) = values
                        .iter()
                        .rev()
                        .find(|(name, _)| name.as_str() == Some(key))
                    {
                        value.json_hints(item, hints);
                    }
                }
            }
            (Self::Namespace(values), Json::Object(items)) => {
                for (key, value) in values.borrow().iter() {
                    value.json_hints(&items[&key.text()], hints);
                }
            }
            _ => {}
        }
    }

    pub fn equals(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => return a == b,
            (Self::UInt(a), Self::UInt(b)) => return a == b,
            (Self::Int(a), Self::UInt(b)) | (Self::UInt(b), Self::Int(a)) => {
                return u64::try_from(*a).ok() == Some(*b);
            }
            _ => {}
        }
        if let (Some(a), Some(b)) = (self.as_str(), other.as_str()) {
            return a == b;
        }
        if let (Some(a), Some(b)) = (self.number(), other.number()) {
            return a == b;
        }
        if matches!(
            (self, other),
            (Self::None, Self::None) | (Self::Undefined, Self::Undefined)
        ) {
            return true;
        }
        if (self.sequence() && other.sequence()) || (self.mapping() && other.mapping()) {
            return self.to_json().ok() == other.to_json().ok();
        }
        false
    }
}
