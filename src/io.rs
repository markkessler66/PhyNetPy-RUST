//! ``phynetpy.IO``: the Newick, NEXUS, FASTA and VCF readers and writers.
//!
//! ``python/phynetpy/IO.py`` is a facade over the functions registered here.
//! The readers reproduce the libraries the Python module used before the
//! port, quirks included, because the test-suite pins that behaviour:
//!
//! * ``io_read_newick`` reproduces ``Bio.Phylo.read(StringIO(s), "newick")``
//!   (its tokeniser and clade processing) followed by the module's own
//!   ``_NewickTreeBuilder`` (breadth-first node creation, ``Internal<n>``
//!   labels, reticulation attributes, gamma pairing and completion).
//! * ``io_nexus_tree_lines`` reproduces python-nexus' block splitting and
//!   its ``TreeHandler`` tree-line detection.
//! * ``io_read_fasta_records`` reproduces ``Bio.SeqIO.parse(path, "fasta")``.
//!
//! Everything that returns a ``Network`` builds it through
//! ``Network::new_object`` so it is an instance of the Python class.

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use indexmap::{IndexMap, IndexSet};
use pyo3::exceptions::{
    PyAssertionError, PyFileNotFoundError, PyIndexError, PyKeyError, PyOSError, PyRuntimeError,
    PyTypeError, PyUnicodeDecodeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyList, PyType};
use rustc_hash::FxHashSet;

use crate::graph::newick::{complete_hybrid_gammas, py_float};
use crate::graph::{Edge, Network, Node};
use crate::util::{node_error, warn};

// ───────────────────────────── IOError ─────────────────────────────────

static IO_EXC: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// ``phynetpy.IO`` registers its ``IOError`` class here when it is imported.
#[pyfunction]
fn _register_io_error(py: Python<'_>, cls: Bound<'_, PyType>) {
    let _ = IO_EXC.set(py, cls.unbind());
}

/// An instance of the Python-defined ``phynetpy.IO.IOError``.
pub fn io_error(py: Python<'_>, msg: impl Into<String>) -> PyErr {
    if IO_EXC.get(py).is_none() {
        // Importing the module registers the class as a side effect.
        let _ = py.import("phynetpy.IO");
    }
    let msg = msg.into();
    match IO_EXC.get(py) {
        Some(t) => PyErr::from_type(t.bind(py).clone(), msg),
        None => PyOSError::new_err(msg),
    }
}

// ─────────────────────── Python string semantics ───────────────────────

/// ``str.isspace`` for one character (Rust's White_Space plus U+001C–U+001F).
fn py_is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python's ``\w``: alphanumerics and the underscore.
fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn py_strip(s: &str) -> &str {
    s.trim_matches(py_is_space)
}

fn py_rstrip(s: &str) -> &str {
    s.trim_end_matches(py_is_space)
}

/// ``str.split()`` with no separator.
fn py_split_ws(s: &str) -> Vec<&str> {
    s.split(py_is_space).filter(|p| !p.is_empty()).collect()
}

/// The lines of a text as iterating a Python text-mode file yields them
/// (universal newlines), without their terminators.
fn py_lines(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut lines = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\n' => {
                lines.push(&text[start..i]);
                i += 1;
                start = i;
            }
            b'\r' => {
                lines.push(&text[start..i]);
                i += 1;
                if i < bytes.len() && bytes[i] == b'\n' {
                    i += 1;
                }
                start = i;
            }
            _ => i += 1,
        }
    }
    if start < bytes.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// ``repr(text)`` for a str, as far as error messages need it.
fn py_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// ``int(text)`` through the interpreter, so every accepted spelling and
/// every error match Python's.
fn py_int_builtin<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("builtins")?.getattr("int")?.call1((text,))
}

/// ``int(text)`` for the small decimal fields a VCF carries (``None`` where
/// Python raises ``ValueError``).
fn py_int(text: &str) -> Option<i64> {
    let t = py_strip(text);
    let (neg, digits) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if digits.is_empty() || digits.starts_with('_') || digits.ends_with('_') || digits.contains("__") {
        return None;
    }
    let mut value: i64 = 0;
    for c in digits.chars() {
        if c == '_' {
            continue;
        }
        let d = c.to_digit(10)? as i64;
        value = value.checked_mul(10)?.checked_add(d)?;
    }
    Some(if neg { -value } else { value })
}

/// ``str(OSError)`` as Python formats it: ``[Errno 2] No such file or directory: 'path'``.
fn os_error_message(e: &io::Error, path: &str) -> String {
    match e.raw_os_error() {
        Some(code) => {
            let full = e.to_string();
            let text = match full.rfind(" (os error ") {
                Some(i) => full[..i].to_string(),
                None => full,
            };
            format!("[Errno {code}] {text}: {}", py_repr(path))
        }
        None => e.to_string(),
    }
}

/// Decode file bytes with Python's own codec so decoding errors are the
/// interpreter's (``UnicodeDecodeError`` with its message).
fn decode_with_python(py: Python<'_>, bytes: &[u8], encoding: &str) -> PyResult<String> {
    PyBytes::new(py, bytes).call_method1("decode", (encoding,))?.extract()
}

fn string_index_error() -> PyErr {
    PyIndexError::new_err("string index out of range")
}

// ──────────────────── Newick: the BioPython tokeniser ──────────────────

#[derive(Debug, Clone)]
enum NwTok {
    Open,
    Close,
    Comma,
    Semicolon,
    /// An unquoted label.
    Label(String),
    /// A quoted label, quotes included.
    Quoted(String),
    /// The text after the ``:`` of an edge length (may start with a space).
    Length(String),
    /// A comment body, brackets excluded.
    Comment(String),
}

impl NwTok {
    /// The text the tokeniser matched.
    fn text(&self) -> String {
        match self {
            NwTok::Open => "(".into(),
            NwTok::Close => ")".into(),
            NwTok::Comma => ",".into(),
            NwTok::Semicolon => ";".into(),
            NwTok::Label(s) | NwTok::Quoted(s) => s.clone(),
            NwTok::Length(s) => format!(":{s}"),
            NwTok::Comment(s) => format!("[{s}]"),
        }
    }
}

/// A character of ``[^\s\(\)\[\]\'\:\;\,]+``.
fn is_label_char(c: char) -> bool {
    !py_is_space(c) && !"()[]':;,".contains(c)
}

/// ``[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?`` at ``i``: the end of the match a
/// backtracking engine finds, if any.
fn match_num(chars: &[char], i: usize) -> Option<usize> {
    let n = chars.len();
    let mut j = i;
    while j < n && chars[j].is_ascii_digit() {
        j += 1;
    }
    let int_digits = j - i;
    let mut end = None;
    if j < n && chars[j] == '.' {
        let mut k = j + 1;
        while k < n && chars[k].is_ascii_digit() {
            k += 1;
        }
        if k > j + 1 {
            end = Some(k);
        }
    }
    let mut e = match end {
        Some(e) => e,
        None if int_digits > 0 => j,
        None => return None,
    };
    if e < n && (chars[e] == 'e' || chars[e] == 'E') {
        let mut k = e + 1;
        if k < n && (chars[k] == '+' || chars[k] == '-') {
            k += 1;
        }
        let digits_start = k;
        while k < n && chars[k].is_ascii_digit() {
            k += 1;
        }
        if k > digits_start {
            e = k;
        }
    }
    Some(e)
}

/// ``\:\ ?[+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?`` at ``i`` (``chars[i] == ':'``).
fn match_edge_length(chars: &[char], i: usize) -> Option<usize> {
    let n = chars.len();
    let mut j = i + 1;
    if j < n && chars[j] == ' ' {
        j += 1;
    }
    if j < n && (chars[j] == '+' || chars[j] == '-') {
        j += 1;
    }
    match_num(chars, j)
}

/// ``\[(\\.|[^\]])*\]`` (or the same with ``'``) at ``i``: the end of the
/// match, if any.  Greedy, with the backtracking fallback onto the last
/// escaped closing character.
fn match_bracketed(chars: &[char], i: usize, close: char) -> Option<usize> {
    let n = chars.len();
    let mut j = i + 1;
    let mut fallback = None;
    while j < n {
        let c = chars[j];
        if c == close {
            return Some(j + 1);
        }
        if c == '\\' && j + 1 < n && chars[j + 1] != '\n' {
            if chars[j + 1] == close {
                fallback = Some(j + 2);
            }
            j += 2;
        } else {
            j += 1;
        }
    }
    fallback
}

/// ``re.finditer(tokenizer, text)`` of ``Bio.Phylo.NewickIO``; characters no
/// alternative matches are skipped, as ``finditer`` skips them.
fn nw_tokenize(text: &str) -> Vec<NwTok> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        match c {
            '(' => {
                out.push(NwTok::Open);
                i += 1;
            }
            ')' => {
                out.push(NwTok::Close);
                i += 1;
            }
            ',' => {
                out.push(NwTok::Comma);
                i += 1;
            }
            ';' => {
                out.push(NwTok::Semicolon);
                i += 1;
            }
            ':' => match match_edge_length(&chars, i) {
                Some(end) => {
                    out.push(NwTok::Length(chars[i + 1..end].iter().collect()));
                    i = end;
                }
                None => i += 1,
            },
            '[' => match match_bracketed(&chars, i, ']') {
                Some(end) => {
                    out.push(NwTok::Comment(chars[i + 1..end - 1].iter().collect()));
                    i = end;
                }
                None => i += 1,
            },
            '\'' => match match_bracketed(&chars, i, '\'') {
                Some(end) => {
                    out.push(NwTok::Quoted(chars[i..end].iter().collect()));
                    i = end;
                }
                None => i += 1,
            },
            c if is_label_char(c) => {
                let mut j = i + 1;
                while j < n && is_label_char(chars[j]) {
                    j += 1;
                }
                out.push(NwTok::Label(chars[i..j].iter().collect()));
                i = j;
            }
            _ => i += 1,
        }
    }
    out
}

// ───────────────────── Newick: the BioPython parser ────────────────────

/// A ``Bio.Phylo.Newick.Clade`` as the parser fills it in.
pub(crate) struct Clade {
    pub(crate) name: Option<String>,
    pub(crate) branch_length: Option<f64>,
    comment: Option<String>,
    confidence: Option<f64>,
    pub(crate) clades: Vec<usize>,
    /// The parser's temporary parent reference (``del``-eted on processing).
    parent: Option<usize>,
}

pub(crate) struct ParsedTree {
    pub(crate) clades: Vec<Clade>,
    pub(crate) root: usize,
}

fn new_clade(arena: &mut Vec<Clade>, parent: Option<usize>) -> usize {
    arena.push(Clade {
        name: None,
        branch_length: None,
        comment: None,
        confidence: None,
        clades: Vec::new(),
        parent,
    });
    arena.len() - 1
}

/// ``NewickIO._parse_confidence``: an all-digit or float-like label is a
/// support value (only whether it parses matters here).
fn parse_confidence(text: &str) -> Option<f64> {
    if !text.is_empty() && text.chars().all(|c| c.is_ascii_digit()) {
        return Some(0.0);
    }
    py_float(text)
}

/// ``Parser.process_clade``: an internal node whose label is numeric loses
/// the label to its confidence; the clade is attached to its parent, which
/// is returned (``None`` for a clade without one).
fn process_clade(arena: &mut [Clade], idx: usize) -> Option<usize> {
    let clade = &mut arena[idx];
    let named = clade.name.as_deref().is_some_and(|s| !s.is_empty());
    if named && clade.confidence.is_none() && !clade.clades.is_empty() {
        clade.confidence = parse_confidence(clade.name.as_deref().unwrap_or(""));
        if clade.confidence.is_some() {
            clade.name = None;
        }
    }
    match clade.parent.take() {
        Some(p) => {
            arena[p].clades.push(idx);
            Some(p)
        }
        None => None,
    }
}

/// ``Parser._parse_tree``; the error text is BioPython's ``NewickError``.
fn parse_tree(text: &str) -> Result<ParsedTree, String> {
    let tokens = nw_tokenize(py_strip(text));
    let mut arena: Vec<Clade> = Vec::new();
    let mut root = new_clade(&mut arena, None);
    let mut current = root;
    let mut lp = 0usize;
    let mut rp = 0usize;
    let mut it = tokens.into_iter();
    for tok in it.by_ref() {
        match tok {
            NwTok::Quoted(t) => {
                let body_end = t.len() - 1;
                let cur = &mut arena[current];
                match &mut cur.name {
                    // Escaped quotes look like two adjacent quoted labels.
                    Some(name) if !name.is_empty() => name.push_str(&t[..body_end]),
                    _ => cur.name = Some(t[1..body_end].to_string()),
                }
            }
            NwTok::Comment(body) => arena[current].comment = Some(body),
            NwTok::Open => {
                current = new_clade(&mut arena, Some(current));
                lp += 1;
            }
            NwTok::Comma => {
                if current == root {
                    // Outer parentheses are missing: make a new root.
                    root = new_clade(&mut arena, None);
                    arena[current].parent = Some(root);
                }
                let parent = process_clade(&mut arena, current);
                current = new_clade(&mut arena, parent);
            }
            NwTok::Close => {
                match process_clade(&mut arena, current) {
                    Some(p) => current = p,
                    None => return Err("Parenthesis mismatch.".to_string()),
                }
                rp += 1;
            }
            NwTok::Semicolon => break,
            NwTok::Length(t) => arena[current].branch_length = Some(py_float(&t).unwrap_or(0.0)),
            NwTok::Label(t) => arena[current].name = Some(t),
        }
    }
    if lp != rp {
        return Err(format!("Mismatch, {lp} open vs {rp} close parentheses."));
    }
    if let Some(next) = it.next() {
        return Err(format!("Text after semicolon in Newick tree: {}", next.text()));
    }
    process_clade(&mut arena, current);
    process_clade(&mut arena, root);
    Ok(ParsedTree { clades: arena, root })
}

const MULTIPLE_TREES: &str = "There are multiple trees in this file; use parse() instead.";

/// ``Bio.Phylo.read(StringIO(text), "newick")``: lines are joined until one
/// ends with ``;``; exactly one tree must result.
pub(crate) fn phylo_read(text: &str) -> Result<ParsedTree, String> {
    let mut first: Option<ParsedTree> = None;
    let mut buf = String::new();
    for line in text.split('\n') {
        buf.push_str(py_rstrip(line));
        if buf.ends_with(';') {
            let tree = parse_tree(&buf)?;
            if first.is_some() {
                return Err(MULTIPLE_TREES.to_string());
            }
            first = Some(tree);
            buf.clear();
        }
    }
    if !buf.is_empty() {
        // A last tree missing its terminal ';' is fine.
        let tree = parse_tree(&buf)?;
        if first.is_some() {
            return Err(MULTIPLE_TREES.to_string());
        }
        first = Some(tree);
    }
    first.ok_or_else(|| "There are no trees in this file.".to_string())
}

// ─────────────────── Newick: the ``_NewickTreeBuilder`` ────────────────

/// A gamma entry of the builder's inheritance table: ``-1`` marks a parent
/// whose gamma is still unknown (a Python ``int``), a value is a ``float``.
#[derive(Clone, Copy)]
enum Gamma {
    Unknown,
    Val(f64),
}

impl Gamma {
    /// Python's ``info[0] == -1`` (true for the marker and for ``-1.0``).
    fn is_minus_one(self) -> bool {
        match self {
            Gamma::Unknown => true,
            Gamma::Val(v) => v == -1.0,
        }
    }

    fn value(self) -> f64 {
        match self {
            Gamma::Unknown => -1.0,
            Gamma::Val(v) => v,
        }
    }

    fn to_py<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            Gamma::Unknown => (-1i64).into_pyobject(py)?.into_any(),
            Gamma::Val(v) => v.into_pyobject(py)?.into_any(),
        })
    }
}

/// ``{parent_label: [gamma, branch_length]}``.
type GammaMap = IndexMap<String, (Gamma, Option<f64>)>;

/// The attribute dict ``_parse_comment`` / ``_merge_attributes`` produce.
struct Attr {
    event: Option<(String, Py<PyAny>)>,
    gamma: Option<GammaMap>,
    comment: Option<String>,
}

fn attr_to_dict<'py>(py: Python<'py>, attr: &Attr) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    if let Some((event, index)) = &attr.event {
        d.set_item("eventType", event)?;
        d.set_item("index", index.bind(py))?;
    }
    if let Some(map) = &attr.gamma {
        let g = PyDict::new(py);
        for (parent, (gamma, length)) in map {
            let entry = PyList::empty(py);
            entry.append(gamma.to_py(py)?)?;
            entry.append(*length)?;
            g.set_item(parent, entry)?;
        }
        d.set_item("gamma", g)?;
    }
    if let Some(c) = &attr.comment {
        d.set_item("comment", c)?;
    }
    Ok(d)
}

/// ``_parse_reticulation_attributes``: the event type and index encoded in
/// the part of a reticulation label after the ``#``.
fn parse_reticulation_attributes(py: Python<'_>, attr_str: &str) -> PyResult<(String, Py<PyAny>)> {
    if attr_str.is_empty() {
        return Err(io_error(py, "Reticulation event label formatting incorrect"));
    }
    let (event, skip) = match attr_str.chars().next().unwrap() {
        'R' => ("Recombination", 1),
        'H' => ("Hybridization", 1),
        'L' => {
            if attr_str.starts_with("LGT") {
                ("Lateral Gene Transfer", 3)
            } else {
                return Err(io_error(py, "Invalid reticulation label format (event error)"));
            }
        }
        _ => {
            // Not an extended-Newick event code ("#UID_6057", "#1", "#h1"):
            // a hybridization indexed by any trailing digits.
            let start = attr_str
                .rfind(|c: char| !c.is_ascii_digit())
                .map(|i| i + attr_str[i..].chars().next().unwrap().len_utf8())
                .unwrap_or(0);
            let digits = &attr_str[start..];
            let index = if digits.is_empty() {
                (-1i64).into_pyobject(py)?.into_any().unbind()
            } else {
                py_int_builtin(py, digits)?.unbind()
            };
            return Ok(("Hybridization".to_string(), index));
        }
    };
    match py_int_builtin(py, &attr_str[skip..]) {
        Ok(v) => Ok((event.to_string(), v.unbind())),
        Err(e) if e.is_instance_of::<PyValueError>(py) => {
            Err(io_error(py, "Invalid reticulation label format (number error)"))
        }
        Err(e) => Err(e),
    }
}

struct Builder<'py> {
    py: Python<'py>,
    net: Bound<'py, Network>,
    internal_count: usize,
    inheritance: IndexMap<String, GammaMap>,
    explicit: FxHashSet<u64>,
}

impl<'py> Builder<'py> {
    /// ``network.has_node_named(name)``.
    fn find(&self, name: &str) -> Option<Bound<'py, Node>> {
        let net = self.net.borrow();
        net.names.get(name).map(|u| net.nodes[u].obj.bind(self.py).clone())
    }

    /// Name an unnamed clade ``Internal<n>`` and return the clade's name.
    fn ensure_name(&mut self, tree: &mut ParsedTree, idx: usize) -> String {
        if tree.clades[idx].name.is_none() {
            tree.clades[idx].name = Some(format!("Internal{}", self.internal_count));
            self.internal_count += 1;
        }
        tree.clades[idx].name.clone().unwrap_or_default()
    }

    fn new_node(&self, name: &str) -> PyResult<Bound<'py, Node>> {
        Bound::new(self.py, Node::make(self.py, name.to_string(), false, None, None, None))
    }

    fn first_entry(&self, name: &str) -> Option<(String, (Gamma, Option<f64>))> {
        self.inheritance
            .get(name)
            .and_then(|m| m.iter().next().map(|(k, v)| (k.clone(), *v)))
    }

    /// ``_parse_comment``: the attributes of one occurrence of a clade under
    /// ``parent_label``, recording reticulation gammas in the inheritance
    /// table as it goes.
    fn parse_comment(&mut self, tree: &ParsedTree, idx: usize, parent_label: &str) -> PyResult<Attr> {
        let py = self.py;
        let clade = &tree.clades[idx];
        let mut attr = Attr {
            event: None,
            gamma: None,
            comment: None,
        };
        let name = match clade.name.as_deref() {
            Some("") => return Err(string_index_error()),
            Some(n) if n.starts_with('#') => n.to_string(),
            _ => {
                attr.comment = clade.comment.clone();
                return Ok(attr);
            }
        };
        let attr_str = name.split('#').nth(1).unwrap_or("").to_string();
        attr.event = Some(parse_reticulation_attributes(py, &attr_str)?);
        let length = clade.branch_length;
        match clade.comment.clone() {
            Some(comment) => {
                let mut parts = comment.split('=');
                if parts.next() == Some("&gamma") {
                    let text = parts
                        .next()
                        .ok_or_else(|| PyIndexError::new_err("list index out of range"))?;
                    let gamma = py_float(text).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "could not convert string to float: {}",
                            py_repr(text)
                        ))
                    })?;
                    let mut own = GammaMap::new();
                    own.insert(parent_label.to_string(), (Gamma::Val(gamma), length));
                    attr.gamma = Some(own);
                    match self.first_entry(&name) {
                        Some((par, info)) => {
                            let mut m = GammaMap::new();
                            if info.0.is_minus_one() {
                                // First parent had no explicit gamma: derive it.
                                m.insert(par, (Gamma::Val(1.0 - gamma), info.1));
                            } else {
                                if info.0.value() + gamma != 1.0 {
                                    return Err(io_error(
                                        py,
                                        "Gamma values provided in newick string do not add to 1",
                                    ));
                                }
                                m.insert(par, info);
                            }
                            m.insert(parent_label.to_string(), (Gamma::Val(gamma), length));
                            self.inheritance.insert(name, m);
                        }
                        None => {
                            if !self.inheritance.contains_key(&name) {
                                let mut m = GammaMap::new();
                                m.insert(parent_label.to_string(), (Gamma::Val(gamma), length));
                                self.inheritance.insert(name, m);
                            }
                        }
                    }
                } else {
                    attr.comment = Some(comment);
                }
            }
            None => match self.first_entry(&name) {
                Some((par, info)) => {
                    let mut m = GammaMap::new();
                    if info.0.is_minus_one() {
                        // Neither parent specified gamma: 0.5 / 0.5.
                        m.insert(par, (Gamma::Val(0.5), info.1));
                        m.insert(parent_label.to_string(), (Gamma::Val(0.5), length));
                    } else {
                        m.insert(par, info);
                        m.insert(
                            parent_label.to_string(),
                            (Gamma::Val(1.0 - info.0.value()), length),
                        );
                    }
                    self.inheritance.insert(name, m);
                }
                None => {
                    if !self.inheritance.contains_key(&name) {
                        let mut m = GammaMap::new();
                        m.insert(parent_label.to_string(), (Gamma::Unknown, length));
                        self.inheritance.insert(name, m);
                    }
                }
            },
        }
        Ok(attr)
    }

    /// ``_parse_parent``: the node of a clade that is a parent (the root the
    /// first time it is seen, otherwise an existing node).
    fn parse_parent(&mut self, tree: &mut ParsedTree, idx: usize) -> PyResult<Bound<'py, Node>> {
        if let Some(name) = tree.clades[idx].name.as_deref() {
            if let Some(found) = self.find(name) {
                return Ok(found);
            }
        }
        let name = self.ensure_name(tree, idx);
        let node = self.new_node(&name)?;
        Node::set_time(&node, 0.0)?;
        if name.is_empty() {
            return Err(string_index_error());
        }
        if name.starts_with('#') {
            Node::set_is_reticulation(&node, true);
        }
        self.net.borrow_mut().add_node(self.net.as_any(), &node)?;
        Ok(node)
    }

    /// ``_parse_child``: create (or revisit) the node of a clade and add the
    /// edge from ``parent`` to it.
    fn parse_child(&mut self, tree: &mut ParsedTree, idx: usize, parent: &Bound<'py, Node>) -> PyResult<()> {
        let py = self.py;
        let parent_label = parent.borrow().name.clone();
        let length = tree.clades[idx].branch_length;
        let existing = tree.clades[idx].name.as_deref().and_then(|n| self.find(n));
        let (node, gamma_map) = match existing {
            Some(node) => {
                let more = self.parse_comment(tree, idx, &parent_label)?;
                let label = node.borrow().name.clone();
                let merged = Attr {
                    event: more.event,
                    gamma: self.inheritance.get(&label).cloned(),
                    comment: None,
                };
                let dict = attr_to_dict(py, &merged)?;
                node.borrow_mut().attrs = dict.unbind();
                (node, merged.gamma)
            }
            None => {
                let name = self.ensure_name(tree, idx);
                let node = self.new_node(&name)?;
                if name.is_empty() {
                    return Err(string_index_error());
                }
                if name.starts_with('#') {
                    Node::set_is_reticulation(&node, true);
                }
                let parent_time = |py: Python<'py>| -> PyResult<f64> {
                    parent
                        .borrow()
                        .t
                        .ok_or_else(|| node_error(py, "No time has been set for this node!"))
                };
                match length {
                    Some(bl) => Node::set_time(&node, parent_time(py)? + bl)?,
                    None => {
                        warn(py, "No branch length provided for node; setting to 1.")?;
                        Node::set_time(&node, parent_time(py)? + 1.0)?;
                    }
                }
                let attr = self.parse_comment(tree, idx, &parent_label)?;
                let dict = attr_to_dict(py, &attr)?;
                node.borrow_mut().attrs = dict.unbind();
                self.net.borrow_mut().add_node(self.net.as_any(), &node)?;
                (node, attr.gamma)
            }
        };
        let edge = Bound::new(
            py,
            Edge::make(
                py,
                parent.clone().unbind(),
                node.clone().unbind(),
                Some(length.unwrap_or(1.0)),
                Some(0.0),
                Some(0.0),
                Some("no tag assigned yet".to_string()),
            ),
        )?;
        // -1 marks a parent whose gamma is still unknown; it is completed
        // once every occurrence has been read.
        if let Some(map) = &gamma_map {
            if let Some((g, _)) = map.get(&parent_label) {
                let g = g.value();
                if g >= 0.0 {
                    Edge::set_gamma(&edge, g)?;
                    self.explicit.insert(edge.borrow().uid);
                }
            }
        }
        self.net.borrow_mut().add_edge(self.net.as_any(), &edge)?;
        Ok(())
    }
}

/// ``_NewickTreeBuilder().build(tree)``: a ``Network`` with the parsed
/// topology, names, times and branch lengths.
fn build_network<'py>(py: Python<'py>, tree: &mut ParsedTree) -> PyResult<Bound<'py, Network>> {
    let net = Network::new_object(py, None)?;
    let mut builder = Builder {
        py,
        net: net.clone(),
        internal_count: 0,
        inheritance: IndexMap::new(),
        explicit: FxHashSet::default(),
    };
    // ``tree.find_clades(order="level")``: (child, parent) pairs in
    // breadth-first order.
    let mut pairs = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(tree.root);
    while let Some(v) = queue.pop_front() {
        for &c in &tree.clades[v].clades {
            pairs.push((c, v));
            queue.push_back(c);
        }
    }
    for (child, par) in pairs {
        let parent = builder.parse_parent(tree, par)?;
        builder.parse_child(tree, child, &parent)?;
    }
    // A reticulation whose first occurrence carried no gamma had that edge
    // created before the second occurrence supplied one.
    complete_hybrid_gammas(py, &net.borrow(), &builder.explicit)?;
    Ok(net)
}

/// ``IO.read_newick``: one newick / extended-newick string into a ``Network``.
#[pyfunction]
fn io_read_newick(py: Python<'_>, newick_str: &str) -> PyResult<Py<PyAny>> {
    let s = py_strip(newick_str);
    if s.is_empty() {
        return Err(io_error(py, "Cannot parse an empty newick string."));
    }
    let mut tree = phylo_read(s).map_err(|m| io_error(py, format!("Failed to parse newick string: {m}")))?;
    let net = build_network(py, &mut tree)?;
    Ok(net.into_any().unbind())
}

// ──────────────────────────────── NEXUS ────────────────────────────────

/// ``nexus.exceptions.NexusFormatException(msg)``.
fn nexus_format_error(py: Python<'_>, msg: String) -> PyErr {
    match py
        .import("nexus.exceptions")
        .and_then(|m| m.getattr("NexusFormatException"))
        .and_then(|c| c.cast_into::<PyType>().map_err(PyErr::from))
    {
        Ok(cls) => PyErr::from_type(cls, msg),
        Err(_) => PyRuntimeError::new_err(msg),
    }
}

fn lowercase_chars(chars: &[char]) -> Vec<char> {
    chars
        .iter()
        .map(|c| c.to_lowercase().next().unwrap_or(*c))
        .collect()
}

/// ``BEGIN_PATTERN.findall(line)[0][0]``: ``begin (\w+)(\s*|\[.*]);`` (case
/// insensitive, anywhere in the line).
fn begin_block_name(line: &str) -> Option<String> {
    let chars: Vec<char> = line.chars().collect();
    let lower = lowercase_chars(&chars);
    let n = chars.len();
    let pat: Vec<char> = "begin ".chars().collect();
    for i in 0..n {
        if i + pat.len() > n || lower[i..i + pat.len()] != pat[..] {
            continue;
        }
        let ws = i + pat.len();
        let mut j = ws;
        while j < n && is_word(chars[j]) {
            j += 1;
        }
        if j == ws {
            continue;
        }
        let name: String = chars[ws..j].iter().collect();
        // ``\s*;``
        let mut k = j;
        while k < n && py_is_space(chars[k]) {
            k += 1;
        }
        if k < n && chars[k] == ';' {
            return Some(name);
        }
        // ``\[.*];``
        if j < n && chars[j] == '[' && (j + 1..n.saturating_sub(1)).any(|p| chars[p] == ']' && chars[p + 1] == ';') {
            return Some(name);
        }
    }
    None
}

/// ``END_PATTERN.search(line)``: ``end\s*;`` (case insensitive).
fn has_block_end(line: &str) -> bool {
    let chars: Vec<char> = line.chars().collect();
    let lower = lowercase_chars(&chars);
    let n = chars.len();
    (0..n).any(|i| {
        if i + 3 > n || lower[i..i + 3] != ['e', 'n', 'd'] {
            return false;
        }
        let mut k = i + 3;
        while k < n && py_is_space(chars[k]) {
            k += 1;
        }
        k < n && chars[k] == ';'
    })
}

/// ``TreeHandler.is_tree.search(line)``: ``tree\s+.*=.*;`` (case insensitive).
fn is_tree_line(line: &str) -> bool {
    let chars: Vec<char> = line.chars().collect();
    let lower = lowercase_chars(&chars);
    let n = chars.len();
    (0..n).any(|i| {
        if i + 4 >= n || lower[i..i + 4] != ['t', 'r', 'e', 'e'] || !py_is_space(chars[i + 4]) {
            return false;
        }
        match (i + 5..n).find(|&p| chars[p] == '=') {
            Some(p) => (p + 1..n).any(|q| chars[q] == ';'),
            None => false,
        }
    })
}

/// ``GenericHandler.is_mesquite_attribute``: ``^TITLE\s+(.*);$`` or
/// ``^LINK\s+(.*?)\s+=\s+(.*);$`` (case insensitive).
fn is_mesquite_attribute(line: &str) -> bool {
    let chars: Vec<char> = line.chars().collect();
    let lower = lowercase_chars(&chars);
    let n = chars.len();
    if n == 0 || chars[n - 1] != ';' {
        return false;
    }
    if n > 5 && lower[..5] == ['t', 'i', 't', 'l', 'e'] && py_is_space(chars[5]) {
        return true;
    }
    if n > 4 && lower[..4] == ['l', 'i', 'n', 'k'] && py_is_space(chars[4]) {
        return (6..n.saturating_sub(2))
            .any(|p| chars[p] == '=' && py_is_space(chars[p - 1]) && py_is_space(chars[p + 1]));
    }
    false
}

/// ``translation_pattern.match(line)``: ``(\d+)\s(['"\w\d\*\.\_\-]+)[,;]?``.
fn translation_match(line: &str) -> Option<(String, String)> {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let mut j = 0;
    while j < n && chars[j].is_ascii_digit() {
        j += 1;
    }
    if j == 0 || j >= n || !py_is_space(chars[j]) {
        return None;
    }
    let start = j + 1;
    let mut k = start;
    while k < n && (matches!(chars[k], '\'' | '"' | '*' | '.' | '_' | '-') || is_word(chars[k])) {
        k += 1;
    }
    if k == start {
        return None;
    }
    Some((chars[..j].iter().collect(), chars[start..k].iter().collect()))
}

/// ``TreeHandler.__init__``: the tree lines of a TREES block (raising the
/// translate-table errors python-nexus raises).
fn trees_block_lines(py: Python<'_>, block: &[String]) -> PyResult<Vec<String>> {
    let mut trees = Vec::new();
    let mut translators: IndexMap<String, String> = IndexMap::new();
    let mut lost_in_translation = false;
    for line in block {
        if line.to_lowercase() == "translate" {
            lost_in_translation = true;
        } else if is_mesquite_attribute(line) {
            // A Mesquite TITLE/LINK attribute, not a tree.
        } else if lost_in_translation {
            if let Some((taxon_id, taxon)) = translation_match(line) {
                let taxon = taxon.trim_matches('\'').to_string();
                if translators.contains_key(&taxon_id) {
                    return Err(nexus_format_error(
                        py,
                        format!("Duplicate Taxa ID {taxon_id} in translate block"),
                    ));
                }
                if translators.values().any(|t| *t == taxon) {
                    return Err(nexus_format_error(
                        py,
                        format!("Duplicate Taxon {taxon} in translate block"),
                    ));
                }
                translators.insert(taxon_id, taxon);
            }
            if line.ends_with(';') {
                lost_in_translation = false;
            }
        } else if is_tree_line(line) {
            trees.push(line.clone());
        }
    }
    Ok(trees)
}

/// The text of a NEXUS file as ``NexusReader.from_file`` reads it
/// (``utf-8-sig``, gzip for ``.gz``).
fn read_nexus_text(py: Python<'_>, filepath: &str) -> PyResult<String> {
    let path = Path::new(filepath);
    if !(path.exists() && path.is_file()) {
        return Err(PyOSError::new_err(format!("Unable To Read File {filepath}")));
    }
    if path.extension().is_some_and(|e| e == "gz") {
        let kwargs = PyDict::new(py);
        kwargs.set_item("encoding", "utf-8-sig")?;
        let handle = py.import("gzip")?.call_method("open", (filepath, "rt"), Some(&kwargs))?;
        let text: String = handle.call_method0("read")?.extract()?;
        handle.call_method0("close")?;
        return Ok(text);
    }
    let bytes = fs::read(path).map_err(|e| PyOSError::new_err(os_error_message(&e, filepath)))?;
    decode_with_python(py, &bytes, "utf-8-sig")
}

/// ``NexusReader._iter_blocks``: ``(lower-case block name, stripped lines)``
/// in file order.  Blank lines and whole-line ``[...]`` comments are
/// dropped; ``end`` is optional.
fn nexus_blocks(text: &str) -> Vec<(String, Vec<String>)> {
    let mut blocks: Vec<(String, Vec<String>)> = Vec::new();
    let mut block: Option<String> = None;
    let mut lines: Vec<String> = Vec::new();
    for raw in py_lines(text) {
        let line = py_strip(raw);
        if line.is_empty() || (line.starts_with('[') && line.ends_with(']')) {
            continue;
        }
        if let Some(name) = begin_block_name(line) {
            if let Some(open) = block.take() {
                if !lines.is_empty() {
                    // "end" is optional.
                    blocks.push((open, std::mem::take(&mut lines)));
                }
            }
            block = Some(name.to_lowercase());
            lines = Vec::new();
        }
        if block.is_some() {
            lines.push(line.to_string());
        }
        if has_block_end(line) {
            if let Some(b) = block.take() {
                blocks.push((b, std::mem::take(&mut lines)));
            }
            block = None;
            lines = Vec::new();
        }
    }
    if let Some(b) = block {
        if !lines.is_empty() {
            blocks.push((b, lines));
        }
    }
    blocks
}

/// ``re.match(r"end\s*;", s)`` (case insensitive, anchored at the start).
fn end_pattern_at_start(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    let lower = lowercase_chars(&chars);
    if lower.len() < 4 || lower[..3] != ['e', 'n', 'd'] {
        return false;
    }
    let mut k = 3;
    while k < chars.len() && py_is_space(chars[k]) {
        k += 1;
    }
    k < chars.len() && chars[k] == ';'
}

/// ``QUOTED_PATTERN.sub('\\1', s)``: ``^["'](.*)["']$``.
fn nexus_dequote(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    if n >= 2 && matches!(chars[0], '"' | '\'') && matches!(chars[n - 1], '"' | '\'') && !chars[1..n - 1].contains(&'\n') {
        chars[1..n - 1].iter().collect()
    } else {
        s.to_string()
    }
}

/// ``GenericHandler.remove_comments``: ``re.sub(r"(\[.*?])", "", line)``.
fn nexus_remove_comments(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '[' {
            if let Some(off) = chars[i + 1..].iter().position(|c| *c == ']') {
                i += off + 2;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// ``re.split(r"\s+", line, 1)`` unpacked into two parts, or ``None`` when
/// the line holds no whitespace (python-nexus skips such lines).
fn split_first_ws(line: &str) -> Option<(&str, &str)> {
    let start = line.find(|c: char| py_is_space(c))?;
    let rest = &line[start..];
    let end = rest.find(|c: char| !py_is_space(c)).unwrap_or(rest.len());
    Some((&line[..start], &rest[end..]))
}

/// ``DataHandler._parse_sites``: one token per site, ``(..)`` multistate
/// groups kept as one token.
fn nexus_parse_sites(sites: &str) -> PyResult<Vec<String>> {
    let parsed: Vec<char> = sites.chars().filter(|c| *c != ' ' && *c != ';').collect();
    if !parsed.contains(&'(') {
        return Ok(parsed.iter().map(|c| c.to_string()).collect());
    }
    let mut out = Vec::new();
    let mut todo = parsed.into_iter();
    while let Some(site) = todo.next() {
        if site == ',' {
            continue;
        }
        if site == '(' {
            let mut multi = String::new();
            loop {
                match todo.next() {
                    Some(')') => break,
                    Some(c) => multi.push(c),
                    None => return Err(PyIndexError::new_err("pop from empty list")),
                }
            }
            out.push(multi);
        } else {
            out.push(site.to_string());
        }
    }
    Ok(out)
}

/// ``int(text)`` as an ``i64`` with Python's error message.
fn py_int_i64(py: Python<'_>, text: &str) -> PyResult<i64> {
    py_int_builtin(py, text)?.extract::<i64>()
}

/// ``DataHandler.parse_range`` for its ``int`` conversions (the labels
/// themselves are not needed, only the errors python-nexus raised).
fn nexus_parse_range(py: Python<'_>, siterange: &str) -> PyResult<()> {
    if siterange.contains(',') {
        for c in siterange.split(',') {
            py_int_i64(py, c)?;
        }
    } else if siterange.contains('-') {
        let parts: Vec<&str> = siterange.split('-').collect();
        if parts.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "too many values to unpack (expected 2{})",
                if parts.len() < 2 { format!(", got {}", parts.len()) } else { String::new() }
            )));
        }
        py_int_i64(py, parts[0])?;
        py_int_i64(py, parts[1])?;
    } else {
        py_int_i64(py, siterange)?;
    }
    Ok(())
}

/// ``re.findall(r"(.+?)=(.+?);", line)``: the second group of every match.
fn nexus_label_values(line: &str) -> Vec<String> {
    let chars: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + 1 < chars.len() {
        let Some(eq) = (pos + 1..chars.len()).find(|&i| chars[i] == '=') else { break };
        let Some(semi) = (eq + 2..chars.len()).find(|&i| chars[i] == ';') else { break };
        out.push(chars[eq + 1..semi].iter().collect());
        pos = semi + 1;
    }
    out
}

/// ``re.findall(r"<key>(\d+)", line, re.I)[0]``: the digits right after
/// ``key`` (case insensitive), or ``None``.
fn nexus_int_after(py: Python<'_>, line: &str, key: &str) -> PyResult<Option<i64>> {
    let lower = line.to_lowercase();
    let mut search = 0;
    while let Some(off) = lower[search..].find(key) {
        let start = search + off + key.len();
        let digits: String = line[start..].chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            return Ok(Some(py_int_i64(py, &digits)?));
        }
        search = start;
    }
    Ok(None)
}

/// ``DataHandler._parse_charstate_block``: strip ``charstatelabels ...;``
/// out of the block text; returns the text and the number of labels.
fn nexus_strip_charstatelabels(py: Python<'_>, text: &str) -> PyResult<(String, usize)> {
    let lower = text.to_lowercase();
    let key = "charstatelabels";
    let mut spans: Vec<(usize, usize, String)> = Vec::new();
    let mut search = 0;
    while let Some(off) = lower[search..].find(key) {
        let start = search + off;
        let body_start = start + key.len();
        match text[body_start..].find(';') {
            Some(semi) => {
                let end = body_start + semi + 1;
                spans.push((start, end, text[body_start..body_start + semi].to_string()));
                search = end;
            }
            None => break,
        }
    }
    let mut count = 0usize;
    if spans.len() == 1 {
        for label in spans[0].2.split(',') {
            if py_strip(label).is_empty() {
                warn(py, &format!("Encountered an empty character label at position {count}"))?;
                continue;
            }
            count += 1;
        }
    }
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for (s, e, _) in &spans {
        out.push_str(&text[last..*s]);
        last = *e;
    }
    out.push_str(&text[last..]);
    Ok((out, count))
}

/// ``nexus.handlers.data.iter_block``: ``(line, lower-stripped line,
/// in_matrix)`` for the lines that are not the BEGIN/END/FORMAT/MATRIX
/// commands.
fn nexus_iter_block<'a>(lines: &[&'a str]) -> Vec<(&'a str, String, bool)> {
    let mut seen_matrix = false;
    let mut out = Vec::new();
    for line in lines {
        let lline = line.to_lowercase();
        let lline = py_strip(&lline).to_string();
        if end_pattern_at_start(&lline) || lline.starts_with("format") {
            continue;
        }
        if begin_block_name(line).is_some() && line.to_lowercase().starts_with("begin ") {
            continue;
        }
        if lline.starts_with("matrix") {
            seen_matrix = true;
            continue;
        }
        out.push((*line, lline, seen_matrix));
    }
    out
}

/// python-nexus' ``DataHandler``: the ``(taxon, site tokens)`` rows of a
/// DATA or CHARACTERS block, with the warnings and errors it raised.
fn nexus_data_matrix(py: Python<'_>, block: &[String]) -> PyResult<Vec<(String, Vec<String>)>> {
    let joined = block.join("\n");
    let (joined, charlabel_count) = nexus_strip_charstatelabels(py, &joined)?;
    let lines: Vec<&str> = joined.split('\n').collect();
    let entries = nexus_iter_block(&lines);
    let mut dim_taxa: Option<i64> = None;
    let mut dim_chars: Option<i64> = None;
    let mut matrix: IndexMap<String, Vec<String>> = IndexMap::new();
    let mut read_data = false;
    for (line, lline, in_matrix) in &entries {
        if lline.starts_with("dimensions ") {
            if let Some(n) = nexus_int_after(py, line, "ntax=")? {
                dim_taxa = Some(n);
            }
            if let Some(n) = nexus_int_after(py, line, "nchar=")? {
                dim_chars = Some(n);
            }
        } else if is_mesquite_attribute(line) {
            // A Mesquite TITLE/LINK attribute.
        } else if *in_matrix {
            let line = nexus_remove_comments(line);
            let Some((taxon, sites)) = split_first_ws(&line) else { continue };
            read_data = true;
            let taxon = nexus_dequote(py_strip(taxon));
            let sites = nexus_parse_sites(py_strip(sites))?;
            matrix.entry(taxon).or_default().extend(sites);
        } else {
            for value in nexus_label_values(line) {
                nexus_parse_range(py, &value)?;
            }
        }
    }
    if !read_data {
        // A "wrapped" matrix: taxon on one line, its sites on the next.
        let mut taxon: Option<String> = None;
        let mut sites: Vec<String> = Vec::new();
        for (line, lline, in_matrix) in &entries {
            if !*in_matrix || lline.is_empty() {
                continue;
            }
            match &taxon {
                None => {
                    let t = py_strip(line);
                    if t.chars().any(py_is_space) {
                        return Err(PyAssertionError::new_err(()));
                    }
                    taxon = Some(nexus_dequote(t));
                }
                Some(t) => {
                    sites.extend(nexus_parse_sites(py_strip(line))?);
                    if dim_chars == Some(sites.len() as i64) {
                        matrix.entry(t.clone()).or_default().extend(std::mem::take(&mut sites));
                        taxon = None;
                    }
                }
            }
        }
    }
    let ntaxa = matrix.len() as i64;
    if let Some(dt) = dim_taxa {
        if ntaxa != dt {
            warn(py, &format!("Expected {ntaxa} taxa, got {dt}"))?;
        }
    }
    let nchar = match matrix.values().next() {
        Some(first) => first.len() as i64,
        None if charlabel_count > 0 => charlabel_count as i64,
        None => 0,
    };
    if let Some(dc) = dim_chars {
        if nchar != dc {
            warn(py, &format!("Expected {nchar} characters, got {dc}"))?;
        }
    }
    Ok(matrix.into_iter().collect())
}

/// ``re.sub(r"\btaxlabels\b", "", line)`` (case insensitive).
fn remove_taxlabels_words(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let lower = lowercase_chars(&chars);
    let key: Vec<char> = "taxlabels".chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let at_word = i + key.len() <= chars.len()
            && lower[i..i + key.len()] == key[..]
            && (i == 0 || !is_word(chars[i - 1]))
            && (i + key.len() == chars.len() || !is_word(chars[i + key.len()]));
        if at_word {
            i += key.len();
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// python-nexus' ``TaxaHandler``: only its consistency check matters here
/// (it raises when the taxa count and the DIMENSIONS line disagree).
fn nexus_taxa_check(py: Python<'_>, block: &[String]) -> PyResult<()> {
    let mut taxa: Vec<String> = Vec::new();
    let mut found: Option<i64> = None;
    let mut in_taxlabels = false;
    for raw in block {
        let line = nexus_dequote(raw);
        let lower = line.to_lowercase();
        // ``^dimensions\s*ntax\s*=\s*(\d+)``
        if let Some(rest) = lower.strip_prefix("dimensions") {
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix("ntax") {
                let rest = rest.trim_start();
                if let Some(rest) = rest.strip_prefix('=') {
                    let digits: String = rest.trim_start().chars().take_while(|c| c.is_ascii_digit()).collect();
                    if !digits.is_empty() {
                        found = Some(py_int_i64(py, &digits)?);
                        continue;
                    }
                }
            }
        }
        if lower.contains("begin taxa") || line == ";" || end_pattern_at_start(py_strip(&line)) || is_mesquite_attribute(&line) {
            continue;
        }
        let mut line = line;
        let starts_taxlabels = lower.starts_with("taxlabels")
            && lower.chars().nth("taxlabels".len()).map_or(true, |c| !is_word(c));
        if in_taxlabels || starts_taxlabels {
            in_taxlabels = true;
            line = remove_taxlabels_words(&line);
        }
        for piece in line.split(' ') {
            let token: String = piece.chars().filter(|c| *c != ';').collect();
            let mut token = py_strip(&token).to_string();
            // ``^\[.*\]\s+``: a leading comment.
            if token.starts_with('[') {
                if let Some(close) = token.rfind(']') {
                    let after = &token[close + 1..];
                    if !after.is_empty() && after.chars().all(py_is_space) {
                        token = String::new();
                    }
                }
            }
            // ``(.*)(\[.*\])$``: a trailing annotation.
            if token.ends_with(']') {
                if let Some(open) = token.rfind('[') {
                    token = token[..open].to_string();
                }
            }
            let token = nexus_dequote(&token);
            if !token.is_empty() && !taxa.contains(&token) {
                taxa.push(token);
            }
        }
    }
    if let Some(f) = found {
        if f != 0 && f != taxa.len() as i64 {
            return Err(nexus_format_error(
                py,
                format!("Number of found taxa ({}) doesn't match dimensions declaration ({f})", taxa.len()),
            ));
        }
    }
    Ok(())
}

/// What ``NexusReader.from_file`` exposes of a file: the TREES block's tree
/// lines and the DATA (else CHARACTERS) block's matrix rows.
struct NexusFile {
    trees: Option<Vec<String>>,
    data: Option<Vec<(String, Vec<String>)>>,
}

/// ``NexusReader._set_blocks``: every block's handler is built in file
/// order (raising what python-nexus raised), duplicates are an error.
fn nexus_read(py: Python<'_>, text: &str) -> PyResult<NexusFile> {
    let mut seen: FxHashSet<String> = FxHashSet::default();
    let mut trees = None;
    let mut data = None;
    let mut characters = None;
    for (name, lines) in nexus_blocks(text) {
        if !seen.insert(name.clone()) {
            return Err(nexus_format_error(py, format!("Duplicate Block {name}")));
        }
        match name.as_str() {
            "trees" => trees = Some(trees_block_lines(py, &lines)?),
            "data" => data = Some(nexus_data_matrix(py, &lines)?),
            "characters" => characters = Some(nexus_data_matrix(py, &lines)?),
            "taxa" => nexus_taxa_check(py, &lines)?,
            _ => {}
        }
    }
    if data.is_none() {
        data = characters;
    }
    Ok(NexusFile { trees, data })
}

/// ``NexusReader.from_file(filepath).trees``: the raw ``Tree ... = ...;``
/// lines of the TREES block, or ``None`` when the file has no such block.
/// Raises what python-nexus raises for duplicate blocks, translate tables
/// and malformed DATA / TAXA blocks.
#[pyfunction]
fn io_nexus_tree_lines(py: Python<'_>, filepath: &str) -> PyResult<Option<Vec<String>>> {
    let text = read_nexus_text(py, filepath)?;
    Ok(nexus_read(py, &text)?.trees)
}

// ───────────────────── NEXUS alignment matrix (MSA.parse) ─────────────────────
//
// ``MSA.parse`` read the DATA/CHARACTERS matrix with ``Bio.AlignIO.read(path,
// "nexus")`` and, when BioPython raised ``NexusError``, with python-nexus'
// ``NexusReader.from_file(path).data``.  Both are reproduced: the BioPython
// emulation covers its comment stripping, command splitting, DIMENSIONS /
// FORMAT / MATRIX handling (matchchar, interleaving, wrapped lines,
// parenthesised ambiguity codes, duplicate labels, character validation)
// and the conditions under which it raised ``NexusError`` (including the
// attribute errors its command dispatcher turned into one); those select
// the python-nexus emulation above.
//
// Deliberate differences: TREES blocks are not parsed by the BioPython
// emulation (BioPython raised ``TreeError`` for a Newick string it could
// not read); ``datatype=standard`` yields its codings as tokens (BioPython
// crashed with a ``RuntimeError`` while building the alignment); a missing
// file is ``FileNotFoundError`` (BioPython parsed the path as NEXUS text).

const BIO_WHITESPACE: &str = " \t\n";
const BIO_PUNCTUATION: &str = "()[]{}\\,;:=*'\"`+-<>";
const BIO_SPECIAL_COMMANDS: &[&str] = &[
    "charstatelabels", "charlabels", "taxlabels", "taxset", "charset", "charpartition",
    "taxpartition", "matrix", "tree", "utree", "translate", "codonposset", "title",
];
const BIO_KNOWN_BLOCKS: &[&str] = &["trees", "data", "characters", "taxa", "sets", "codons"];
/// Commands ``Bio.Nexus.Nexus`` has a ``_<command>`` method for.
const BIO_COMMANDS: &[&str] = &[
    "adjust_charlabels", "apply_block_structure", "changeset", "charlabels", "charpartition",
    "charset", "charstatelabels", "check_taxlabels", "codeset", "codonposset", "dimensions",
    "eliminate", "format", "get_indices", "get_nexus_block", "link", "matrix", "name_n_vector",
    "options", "parse_list", "parse_nexus_block", "resolve", "set", "statelabels", "stateset",
    "taxlabels", "taxpartition", "taxset", "title", "translate", "tree", "treepartition",
    "treeset", "unknown_nexus_block", "utree",
];
const IUPAC_DNA: &[(char, &str)] = &[
    ('A', "A"), ('C', "C"), ('G', "G"), ('T', "T"), ('M', "AC"), ('R', "AG"), ('W', "AT"),
    ('S', "CG"), ('Y', "CT"), ('K', "GT"), ('V', "ACG"), ('H', "ACT"), ('D', "AGT"),
    ('B', "CGT"), ('X', "GATC"), ('N', "GATC"),
];
const IUPAC_RNA: &[(char, &str)] = &[
    ('A', "A"), ('C', "C"), ('G', "G"), ('U', "U"), ('M', "AC"), ('R', "AG"), ('W', "AU"),
    ('S', "CG"), ('Y', "CU"), ('K', "GU"), ('V', "ACG"), ('H', "ACU"), ('D', "AGU"),
    ('B', "CGU"), ('X', "GAUC"), ('N', "GAUC"),
];
const PROTEIN_LETTERS: &str = "ACDEFGHIKLMNPQRSTVWY";

/// How the BioPython emulation fails: a ``NexusError`` (the caller falls
/// back to python-nexus) or another Python exception (propagated).
enum BioErr {
    /// The message BioPython would have raised (kept as a debugging aid).
    #[allow(dead_code)]
    Nexus(String),
    Py(PyErr),
}

impl From<PyErr> for BioErr {
    fn from(e: PyErr) -> Self {
        BioErr::Py(e)
    }
}

fn bio_nexus_err(msg: impl Into<String>) -> BioErr {
    BioErr::Nexus(msg.into())
}

fn bio_stop_iteration() -> BioErr {
    BioErr::Py(PyRuntimeError::new_err("generator raised StopIteration"))
}

/// ``_kill_comments_and_break_lines`` + ``_adjust_lines``: the command
/// lines of the file, comments removed (``[&...]`` kept), split on ``;``
/// outside quotes, newlines kept only inside MATRIX commands.
fn bio_commandlines(text: &str) -> Result<Vec<String>, BioErr> {
    let mut s = text.trim();
    if let Some(rest) = s.strip_prefix("#NEXUS") {
        s = rest;
    }
    let chars: Vec<char> = s.chars().collect();
    let mut pieces: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut special = false;
    let mut comm: i32 = 0;
    for i in 0..chars.len() {
        let t = chars[i];
        let t2 = chars.get(i + 1).copied();
        if Some(t) == quote && comm == 0 && !special {
            quote = None;
        } else if quote.is_none() && comm == 0 && !special && (t == '"' || t == '\'') {
            quote = Some(t);
        } else if quote.is_none() && t == '[' {
            if t2 == Some('&') && comm == 0 && !special {
                special = true;
            } else {
                comm += 1;
            }
        } else if quote.is_none() && t == ']' {
            if special {
                special = false;
            } else {
                comm -= 1;
                if comm < 0 {
                    return Err(bio_nexus_err("Nexus formatting error: unmatched ]"));
                }
                continue;
            }
        }
        if comm == 0 {
            if t == ';' && quote.is_none() {
                pieces.push(std::mem::take(&mut cur));
            } else {
                cur.push(t);
            }
        }
    }
    if !cur.is_empty() {
        pieces.push(cur);
    }
    if comm > 0 {
        return Err(bio_nexus_err("Nexus formatting error: unmatched ["));
    }
    let mut formatted: Vec<String> = Vec::new();
    for piece in pieces {
        let piece = piece.replace("\r\n", "\n").replace('\r', "\n");
        let line = piece.trim();
        if line.to_lowercase().starts_with("matrix") {
            formatted.push(line.to_string());
        } else {
            let flat = line.replace('\n', " ");
            if !flat.is_empty() {
                formatted.push(flat);
            }
        }
    }
    for cl in formatted.iter_mut() {
        if cl.get(..6).is_some_and(|h| h.eq_ignore_ascii_case("#NEXUS")) {
            *cl = cl[6..].trim().to_string();
        }
    }
    Ok(formatted)
}

/// ``Nexus._get_nexus_block``: ``(title, command lines)`` per block.
fn bio_blocks(commandlines: Vec<String>) -> Result<Vec<(String, Vec<String>)>, BioErr> {
    let mut blocks = Vec::new();
    let mut inblock = false;
    let mut title = String::new();
    let mut lines: Vec<String> = Vec::new();
    for cl in commandlines {
        let lower = cl.to_lowercase();
        if lower.starts_with("begin") {
            if inblock {
                return Err(bio_nexus_err(format!("Illegal block nesting in block {title}")));
            }
            inblock = true;
            title = cl
                .split_whitespace()
                .nth(1)
                .ok_or_else(|| BioErr::Py(PyIndexError::new_err("list index out of range")))?
                .to_lowercase();
        } else if lower.starts_with("end") {
            if !inblock {
                return Err(bio_nexus_err("Unmatched 'end'."));
            }
            inblock = false;
            blocks.push((std::mem::take(&mut title), std::mem::take(&mut lines)));
        } else if inblock {
            lines.push(cl);
        }
    }
    Ok(blocks)
}

/// ``Bio.Nexus.Nexus.Commandline``.
struct BioCommand {
    command: String,
    /// The raw option text of a SPECIAL_COMMANDS command (``matrix``).
    raw: String,
    /// ``key = value`` options of every other command (bare tokens map to ``None``).
    options: IndexMap<String, Option<String>>,
}

fn bio_commandline(line: &str) -> Result<BioCommand, BioErr> {
    let line = line.trim();
    let (command, options) = match line.split_once('\n') {
        Some((c, o)) => (c.trim().to_lowercase(), o.to_string()),
        None => {
            let mut words = line.split_whitespace();
            let c = words.next().unwrap_or("").to_lowercase();
            (c, words.collect::<Vec<_>>().join(" "))
        }
    };
    if BIO_SPECIAL_COMMANDS.contains(&command.as_str()) {
        return Ok(BioCommand { command, raw: options.trim().to_string(), options: IndexMap::new() });
    }
    let mut map: IndexMap<String, Option<String>> = IndexMap::new();
    if !options.is_empty() {
        let toks: Vec<String> = options.replace('=', " = ").split_whitespace().map(String::from).collect();
        let mut used = vec![false; toks.len()];
        for n in 0..toks.len() {
            if toks[n] == "=" && n != 0 {
                let value = toks
                    .get(n + 1)
                    .ok_or_else(|| BioErr::Py(PyIndexError::new_err("list index out of range")))?
                    .clone();
                used[n - 1] = true;
                used[n] = true;
                used[n + 1] = true;
                map.insert(toks[n - 1].to_lowercase(), Some(value));
            }
        }
        for n in 0..toks.len() {
            if !used[n] {
                map.insert(toks[n].to_lowercase(), None);
            }
        }
    }
    Ok(BioCommand { command, raw: String::new(), options: map })
}

/// ``options[key][0]``: the first character of an option value.
fn bio_first_char(value: &Option<String>) -> Result<char, BioErr> {
    match value {
        None => Err(BioErr::Py(PyTypeError::new_err("'NoneType' object is not subscriptable"))),
        Some(v) => v.chars().next().ok_or_else(|| BioErr::Py(PyIndexError::new_err("string index out of range"))),
    }
}

/// ``CharBuffer.next_word`` from ``pos``: the word and the position after it.
fn bio_next_word(chars: &[char], pos: usize) -> (Option<String>, usize) {
    let n = chars.len();
    let mut i = pos;
    while i < n && BIO_WHITESPACE.contains(chars[i]) {
        i += 1;
    }
    if i >= n {
        return (None, n);
    }
    let first = chars[i];
    i += 1;
    let mut word = vec![first];
    let quoted = matches!(first, '\'' | '"').then_some(first);
    if quoted.is_none() && BIO_PUNCTUATION.contains(first) {
        return (Some(first.to_string()), i);
    }
    loop {
        let c = chars.get(i).copied();
        if quoted.is_some() && c == quoted {
            word.push(chars[i]);
            i += 1;
            if chars.get(i).copied() == quoted {
                i += 1;
            } else {
                break;
            }
        } else if quoted.is_some() {
            match c {
                Some(ch) => {
                    word.push(ch);
                    i += 1;
                }
                None => break,
            }
        } else {
            match c {
                Some(ch) if !BIO_PUNCTUATION.contains(ch) && !BIO_WHITESPACE.contains(ch) => {
                    word.push(ch);
                    i += 1;
                }
                _ => break,
            }
        }
    }
    (Some(word.into_iter().collect()), i)
}

/// ``quotestrip``.
fn bio_quotestrip(word: &str) -> String {
    let mut w: Vec<char> = word.chars().collect();
    while w.len() >= 2
        && ((w[0] == '\'' && w[w.len() - 1] == '\'') || (w[0] == '"' && w[w.len() - 1] == '"'))
    {
        w = w[1..w.len() - 1].to_vec();
    }
    if w.len() == 1 && matches!(w[0], '\'' | '"') {
        // ``"'"[1:-1]`` is the empty string.
        w.clear();
    }
    w.into_iter().collect()
}

/// ``_unique_label``.
fn bio_unique_label<'a>(previous: impl Iterator<Item = &'a String>, label: &str) -> Result<String, BioErr> {
    let previous: FxHashSet<&str> = previous.map(|s| s.as_str()).collect();
    let mut label = label.to_string();
    while previous.contains(label.as_str()) {
        let parts: Vec<&str> = label.split('.').collect();
        let last = parts[parts.len() - 1];
        if let Some(num_text) = last.strip_prefix("copy") {
            let copy_num = if num_text.is_empty() {
                1
            } else {
                num_text.parse::<i64>().map_err(|_| {
                    BioErr::Py(PyValueError::new_err(format!(
                        "invalid literal for int() with base 10: {}",
                        py_repr(num_text)
                    )))
                })? + 1
            };
            label = format!("{}.copy{copy_num}", parts[..parts.len() - 1].join("."));
        } else {
            label.push_str(".copy");
        }
    }
    Ok(label)
}

/// ``_replace_parenthesized_ambigs``.
fn bio_replace_parenthesized(seq: &str, rev: &HashMap<String, char>) -> Result<Vec<char>, BioErr> {
    let mut seq: Vec<char> = seq.chars().collect();
    while let Some(opening) = seq.iter().position(|c| *c == '(') {
        let closing = seq
            .iter()
            .position(|c| *c == ')')
            .ok_or_else(|| bio_nexus_err(format!("Missing closing parenthesis in: {}", seq.iter().collect::<String>())))?;
        if closing < opening {
            return Err(bio_nexus_err(format!("Missing opening parenthesis in: {}", seq.iter().collect::<String>())));
        }
        let mut inner: Vec<char> = seq[opening + 1..closing].to_vec();
        inner.sort_unstable();
        let ambig: String = inner.iter().collect();
        let upper = ambig.to_uppercase();
        let mut code = *rev
            .get(&upper)
            .ok_or_else(|| BioErr::Py(PyKeyError::new_err(upper.clone())))?;
        if ambig != upper {
            code = code.to_lowercase().next().unwrap_or(code);
        }
        seq.splice(opening..=closing, std::iter::once(code));
    }
    Ok(seq)
}

/// ``StandardData``: ``(kind, symbols)`` per coding; kind is ``'s'``
/// (single), ``'m'`` (``(..)`` multistate) or ``'u'`` (``{..}`` uncertain).
fn bio_standard_data(chars: &str) -> Result<Vec<(char, Vec<char>)>, BioErr> {
    let mut data: Vec<(char, Vec<char>)> = Vec::new();
    let mut multi = false;
    let mut uncertain = false;
    let mut current: (char, Vec<char>) = ('s', Vec::new());
    for (pos, coding) in chars.chars().enumerate() {
        if multi {
            if coding == ')' {
                multi = false;
            } else {
                current.1.push(coding);
                continue;
            }
        } else if uncertain {
            if coding == '}' {
                uncertain = false;
            } else {
                current.1.push(coding);
                continue;
            }
        } else if coding == '(' {
            multi = true;
            current.0 = 'm';
            continue;
        } else if coding == '{' {
            uncertain = true;
            current.0 = 'u';
            continue;
        } else if coding == ')' || coding == '}' {
            return Err(bio_nexus_err(format!("Improper character {coding} at position {pos} of a coding sequence.")));
        } else {
            current.1.push(coding);
        }
        data.push(std::mem::replace(&mut current, ('s', Vec::new())));
    }
    Ok(data)
}

fn bio_standard_token(coding: &(char, Vec<char>)) -> String {
    let body: String = coding.1.iter().collect();
    match coding.0 {
        'm' => format!("({body})"),
        'u' => format!("{{{body}}}"),
        _ => body,
    }
}

/// The datatype-dependent tables ``Nexus._format`` creates.
struct BioType {
    ambiguous: IndexMap<char, String>,
    unambiguous: String,
    valid: String,
    rev: HashMap<String, char>,
}

/// The ``Bio.Nexus.Nexus`` state the DATA/CHARACTERS commands update.
struct BioNexus {
    ntax: i64,
    nchar: i64,
    datatype: String,
    respectcase: bool,
    missing: char,
    gap: char,
    symbols: Option<Vec<char>>,
    matchchar: Option<char>,
    interleave: bool,
    typed: Option<BioType>,
    have_matrix: bool,
    matrix: IndexMap<String, Vec<String>>,
    unaltered: Vec<String>,
    taxlabels: Vec<String>,
}

impl BioNexus {
    fn new() -> Self {
        BioNexus {
            ntax: 0,
            nchar: 0,
            datatype: "dna".to_string(),
            respectcase: false,
            missing: '?',
            gap: '-',
            symbols: None,
            matchchar: None,
            interleave: false,
            typed: None,
            have_matrix: false,
            matrix: IndexMap::new(),
            unaltered: Vec::new(),
            taxlabels: Vec::new(),
        }
    }

    fn dimensions(&mut self, py: Python<'_>, opts: &IndexMap<String, Option<String>>) -> Result<(), BioErr> {
        for (key, slot) in [("ntax", &mut self.ntax), ("nchar", &mut self.nchar)] {
            if let Some(value) = opts.get(key) {
                let value = value
                    .as_ref()
                    .ok_or_else(|| BioErr::Py(PyTypeError::new_err("int() can't convert non-string with explicit base")))?;
                *slot = py_int_i64(py, value)?;
            }
        }
        Ok(())
    }

    fn format(&mut self, opts: &IndexMap<String, Option<String>>) -> Result<(), BioErr> {
        // An AttributeError inside a command handler is reported by
        // BioPython as ``NexusError("Unknown command: format ")``.
        let attr_err = || bio_nexus_err("Unknown command: format ");
        if opts.contains_key("respectcase") {
            self.respectcase = true;
        }
        if let Some(value) = opts.get("symbols") {
            let value = value.as_ref().ok_or_else(attr_err)?;
            let mut s: Vec<char> = value.split_whitespace().collect::<String>().chars().collect();
            let n = s.len();
            if n >= 1 && ((s[0] == '"' && s[n - 1] == '"') || (s[0] == '\'' && s[n - 1] == '\'')) {
                s = if n >= 2 { s[1..n - 1].to_vec() } else { Vec::new() };
            }
            if !self.respectcase {
                s = s.iter().flat_map(|c| c.to_uppercase()).collect();
            }
            self.symbols = Some(s);
        }
        if let Some(value) = opts.get("datatype") {
            let value = value.as_ref().ok_or_else(attr_err)?;
            let dt = value.to_lowercase();
            let (ambiguous, unambiguous): (Vec<(char, String)>, String) = match dt.as_str() {
                "dna" | "nucleotide" => (IUPAC_DNA.iter().map(|(k, v)| (*k, v.to_string())).collect(), "GATC".to_string()),
                "rna" => (IUPAC_RNA.iter().map(|(k, v)| (*k, v.to_string())).collect(), "GAUC".to_string()),
                "protein" => (
                    vec![('B', "DN".to_string()), ('Z', "EQ".to_string()), ('X', PROTEIN_LETTERS.to_string())],
                    format!("{PROTEIN_LETTERS}*"),
                ),
                "standard" => {
                    let syms = self.symbols.as_ref().filter(|s| !s.is_empty()).ok_or_else(|| {
                        bio_nexus_err(
                            "Symbols must be defined when using standard datatype. Please remove any whitespace \
                             (spaces, tabs, etc.) between values for symbols as this confuses the Nexus parser.",
                        )
                    })?;
                    let mut u: String = syms.iter().collect();
                    if !self.respectcase {
                        u = format!("{u}{}", u.to_lowercase());
                    }
                    (Vec::new(), u)
                }
                _ => return Err(bio_nexus_err(format!("Unsupported datatype: {dt}"))),
            };
            self.datatype = dt;
            let mut valid: String = ambiguous.iter().map(|(k, _)| *k).collect();
            valid.push_str(&unambiguous);
            if !self.respectcase {
                valid = format!("{}{}", valid.to_lowercase(), valid.to_uppercase());
            }
            let mut rev: HashMap<String, char> = HashMap::new();
            for (k, v) in &ambiguous {
                if *k != 'X' {
                    let mut cs: Vec<char> = v.chars().collect();
                    cs.sort_unstable();
                    rev.insert(cs.into_iter().collect(), *k);
                }
            }
            self.typed = Some(BioType { ambiguous: ambiguous.into_iter().collect(), unambiguous, valid, rev });
        }
        if matches!(self.datatype.as_str(), "dna" | "nucleotide" | "rna") {
            let letters = if self.datatype == "rna" { "GAUCRYWSMKHBVDN" } else { "GATCRYWSMKHBVDN" };
            let t = self.typed.as_mut().ok_or_else(attr_err)?;
            self.symbols = Some(letters.chars().collect());
            if !t.ambiguous.contains_key(&self.missing) {
                let v = format!("{}{}", t.unambiguous, self.gap);
                t.ambiguous.insert(self.missing, v);
            }
            t.ambiguous.insert(self.gap, self.gap.to_string());
        }
        if let Some(value) = opts.get("missing") {
            self.missing = bio_first_char(value)?;
        }
        if let Some(value) = opts.get("gap") {
            self.gap = bio_first_char(value)?;
        }
        if let Some(value) = opts.get("matchchar") {
            self.matchchar = Some(bio_first_char(value)?);
        }
        if let Some(value) = opts.get("interleave") {
            if value.as_ref().is_none_or(|v| v.to_lowercase() == "yes") {
                self.interleave = true;
            }
        }
        Ok(())
    }

    fn check_taxlabels(&self, taxon: &str) -> Option<String> {
        let key = taxon.replace(' ', "_");
        let mut found = None;
        for t in &self.taxlabels {
            if t.replace(' ', "_") == key {
                found = Some(t.clone());
            }
        }
        found
    }

    fn matrix(&mut self, raw: &str) -> Result<(), BioErr> {
        if self.ntax == 0 || self.nchar == 0 {
            return Err(bio_nexus_err("Dimensions must be specified before matrix!"));
        }
        self.matrix.clear();
        self.have_matrix = true;
        let lines: Vec<&str> = raw.split('\n').map(str::trim).filter(|l| !l.is_empty()).collect();
        let remove_ws = |s: &str| -> String { s.chars().filter(|c| !py_is_space(*c)).collect() };
        let mut idx = 0usize;
        let mut taxcount: i64 = 0;
        let mut first_block = true;
        let mut refseq: Vec<char> = Vec::new();
        loop {
            let Some(line) = lines.get(idx) else {
                if taxcount < self.ntax {
                    return Err(bio_nexus_err("Not enough taxa in matrix."));
                } else if taxcount > self.ntax {
                    return Err(bio_nexus_err("Too many taxa in matrix."));
                }
                break;
            };
            idx += 1;
            taxcount += 1;
            if taxcount > self.ntax {
                if !self.interleave {
                    return Err(bio_nexus_err("Too many taxa in matrix - should matrix be interleaved?"));
                }
                taxcount = 1;
                first_block = false;
            }
            let lchars: Vec<char> = line.chars().collect();
            let (word, pos) = bio_next_word(&lchars, 0);
            let id = bio_quotestrip(&word.unwrap_or_default());
            let rest: String = lchars[pos.min(lchars.len())..].iter().collect();
            let rest = rest.trim();
            let mut chars: String;
            if self.interleave {
                if !rest.is_empty() {
                    chars = remove_ws(rest);
                } else {
                    let next = lines.get(idx).ok_or_else(bio_stop_iteration)?;
                    idx += 1;
                    chars = remove_ws(next);
                }
            } else {
                chars = remove_ws(rest);
                while (chars.chars().count() as i64) < self.nchar {
                    let next = lines.get(idx).ok_or_else(bio_stop_iteration)?;
                    idx += 1;
                    chars.push_str(&remove_ws(next));
                }
            }
            let typed = self.typed.as_ref().ok_or_else(|| bio_nexus_err("Unknown command: matrix "))?;
            let tokens: Vec<String> = if self.datatype != "standard" {
                let mut seq = bio_replace_parenthesized(&chars, &typed.rev)?;
                if taxcount == 1 {
                    refseq = seq.clone();
                } else if let Some(mc) = self.matchchar {
                    for p in 0..seq.len() {
                        if seq[p] == mc {
                            seq[p] = *refseq
                                .get(p)
                                .ok_or_else(|| BioErr::Py(PyIndexError::new_err("string index out of range")))?;
                        }
                    }
                }
                for &c in &seq {
                    if !typed.valid.contains(c) && c != self.gap && c != self.missing {
                        return Err(bio_nexus_err(format!(
                            "Taxon {id}: Illegal character {c} in sequence {} (check dimensions/interleaving)",
                            seq.iter().collect::<String>()
                        )));
                    }
                }
                seq.iter().map(|c| c.to_string()).collect()
            } else {
                let codings = bio_standard_data(&chars)?;
                for coding in &codings {
                    for &c in &coding.1 {
                        if !typed.valid.contains(c) && c != self.gap && c != self.missing {
                            let shown: String = codings.iter().map(bio_standard_token).collect();
                            return Err(bio_nexus_err(format!(
                                "Taxon {id}: Illegal character {c} in sequence {shown} (check dimensions/interleaving)"
                            )));
                        }
                    }
                }
                codings.iter().map(bio_standard_token).collect()
            };
            if first_block {
                self.unaltered.push(id.clone());
                let uid = bio_unique_label(self.matrix.keys(), &id)?;
                self.matrix.insert(uid.clone(), tokens);
                self.taxlabels.push(uid);
            } else {
                let uid = bio_unique_label(self.taxlabels[..(taxcount - 1) as usize].iter(), &id)?;
                match self.check_taxlabels(&uid).and_then(|t| self.matrix.get_mut(&t)) {
                    Some(existing) => existing.extend(tokens),
                    None => {
                        return Err(bio_nexus_err(format!(
                            "Taxon {uid} not in first block of interleaved matrix. Check matrix dimensions and interleave."
                        )))
                    }
                }
            }
        }
        for (taxon, seq) in &self.matrix {
            if seq.len() as i64 != self.nchar {
                return Err(bio_nexus_err(format!(
                    "Matrix Nchar {} does not match data length ({}) for taxon {taxon}",
                    self.nchar,
                    seq.len()
                )));
            }
        }
        let mut keys: Vec<&String> = self.matrix.keys().collect();
        keys.sort();
        let mut labels: Vec<&String> = self.taxlabels.iter().collect();
        labels.sort();
        if keys != labels {
            return Err(BioErr::Py(PyValueError::new_err(
                "ERROR: TAXLABELS must be identical with MATRIX. Please Report this as a bug, and send in data file.",
            )));
        }
        Ok(())
    }
}

/// ``Bio.AlignIO.read(text, "nexus")`` as ``(name, site tokens)`` rows.
fn bio_read_alignment(py: Python<'_>, text: &str) -> Result<Vec<(String, Vec<String>)>, BioErr> {
    let mut state = BioNexus::new();
    for (title, lines) in bio_blocks(bio_commandlines(text)?)? {
        if !BIO_KNOWN_BLOCKS.contains(&title.as_str()) {
            continue;
        }
        for line in lines {
            let cmd = bio_commandline(&line)?;
            match cmd.command.as_str() {
                "dimensions" => state.dimensions(py, &cmd.options)?,
                "format" => state.format(&cmd.options)?,
                "matrix" => state.matrix(&cmd.raw)?,
                "charstatelabels" => {
                    if state.symbols.as_ref().is_none_or(|s| s.is_empty()) {
                        return Err(bio_nexus_err("Symbols must be defined when using character states"));
                    }
                }
                other if BIO_COMMANDS.contains(&other) => {}
                other => return Err(bio_nexus_err(format!("Unknown command: {other} "))),
            }
        }
    }
    if !state.have_matrix || state.matrix.is_empty() {
        return Err(BioErr::Py(PyValueError::new_err("No records found in handle")));
    }
    let mut rows = Vec::with_capacity(state.unaltered.len());
    for (old, new) in state.unaltered.iter().zip(state.taxlabels.iter()) {
        let tokens = state.matrix.get(new).cloned().unwrap_or_default();
        rows.push((old.clone(), tokens));
    }
    Ok(rows)
}

/// ``MSA.parse``'s reader: the ``(taxon, site tokens)`` rows of a NEXUS
/// file's DATA/CHARACTERS matrix, as ``Bio.AlignIO.read(path, "nexus")``
/// produced them, or as python-nexus produced them when BioPython raised
/// ``NexusError``.
#[pyfunction]
fn io_read_nexus_alignment(py: Python<'_>, filepath: &str) -> PyResult<Vec<(String, Vec<String>)>> {
    let path = Path::new(filepath);
    if !(path.exists() && path.is_file()) {
        return Err(PyFileNotFoundError::new_err(format!("Nexus file not found: {filepath}")));
    }
    let nexus_text = read_nexus_text(py, filepath)?;
    // BioPython read the file with the default codec, keeping a byte-order mark.
    let bio_text = if path.extension().is_some_and(|e| e == "gz") {
        nexus_text.clone()
    } else {
        let bytes = fs::read(path).map_err(|e| PyOSError::new_err(os_error_message(&e, filepath)))?;
        decode_with_python(py, &bytes, "utf-8")?
    };
    match bio_read_alignment(py, &bio_text) {
        Ok(rows) => Ok(rows),
        Err(BioErr::Py(e)) => Err(e),
        Err(BioErr::Nexus(_)) => nexus_read(py, &nexus_text)?
            .data
            .ok_or_else(|| PyTypeError::new_err("'NoneType' object is not iterable")),
    }
}

/// The ``read_nexus`` loop: each tree line becomes a ``Network``; a line the
/// Newick parser rejects is skipped with a warning.
#[pyfunction]
fn io_networks_from_nexus_trees(py: Python<'_>, tree_lines: Vec<String>) -> PyResult<Vec<Py<PyAny>>> {
    let mut networks: Vec<Py<PyAny>> = Vec::new();
    for tree_str in &tree_lines {
        let (head, tail) = match tree_str.split_once('=') {
            Some((h, t)) => (h, Some(t)),
            None => (tree_str.as_str(), None),
        };
        let words = py_split_ws(head);
        let name = if words.len() > 1 {
            words[1].to_string()
        } else {
            format!("tree_{}", networks.len())
        };
        let newick_part = tail.unwrap_or(tree_str.as_str());
        let mut tree = match phylo_read(newick_part) {
            Ok(t) => t,
            Err(msg) => {
                warn(py, &format!("Failed to parse tree '{name}' in nexus file: {msg}"))?;
                continue;
            }
        };
        let net = build_network(py, &mut tree)?;
        networks.push(net.into_any().unbind());
    }
    Ok(networks)
}

// ──────────────────────────────── FASTA ────────────────────────────────

const FASTA_LEADING_COMMENT: &str = "This FASTA file contains comments at the beginning of the file, which are not\n\
allowed by the 'fasta' parser.\n\
\n\
To parse this file, you have three options:\n\
\n\
(1) Modify your FASTA file to remove such comments at the beginning of the\n\
file.\n\
\n\
(2) Use SeqIO.parse with the 'fasta-pearson' format instead of 'fasta'. This\n\
format is consistent with the FASTA format defined by William Pearson's FASTA\n\
aligner software. This format allows for comments before the first sequence;\n\
lines starting with the ';' character anywhere in the file are also regarded\n\
as comment lines and are ignored.\n\
\n\
(3) Use the 'fasta-blast' format. This format regards any lines \"starting with\n\
'!', '#', or ';' as comment lines. The 'fasta-blast' format may be safer than\n\
the 'fasta-pearson' format, as it explicitly indicates which lines are comments.\n";

/// ``Bio.SeqIO.parse(path, "fasta")`` as ``read_fasta_records`` consumes
/// it: ``(id, sequence)`` pairs.  The id is the first word of the title
/// (``seq_<i>`` when the title has none); whitespace inside the sequence
/// is dropped.
#[pyfunction]
fn io_read_fasta_records(py: Python<'_>, filepath: &str) -> PyResult<Vec<(String, String)>> {
    if !Path::new(filepath).exists() {
        return Err(PyFileNotFoundError::new_err(format!("FASTA file not found: {filepath}")));
    }
    let fail = |msg: String| io_error(py, format!("Failed to parse FASTA file '{filepath}': {msg}"));
    let bytes = fs::read(filepath).map_err(|e| fail(os_error_message(&e, filepath)))?;
    let text = decode_with_python(py, &bytes, "utf-8").map_err(|e| {
        let msg = e.value(py).str().map(|s| s.to_string()).unwrap_or_else(|_| e.to_string());
        fail(msg)
    })?;
    let lines = py_lines(&text);
    let mut records: Vec<(String, String)> = Vec::new();
    if let Some(first) = lines.first() {
        if !first.starts_with('>') {
            return Err(fail(FASTA_LEADING_COMMENT.to_string()));
        }
        let mut title: &str = py_rstrip(&first[1..]);
        let mut seq_parts: Vec<&str> = Vec::new();
        let finish = |title: &str, parts: &[&str], records: &mut Vec<(String, String)>| -> PyResult<()> {
            let joined: String = parts.concat();
            let mut seq: Vec<u8> = Vec::with_capacity(joined.len());
            for b in joined.bytes() {
                if !matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
                    seq.push(b);
                }
            }
            if let Some(pos) = seq.iter().position(|b| *b >= 0x80) {
                // ``str(Seq(bytes))`` decodes as ASCII.
                return Err(PyUnicodeDecodeError::new(
                    py,
                    c"ascii",
                    &seq,
                    pos..pos + 1,
                    c"ordinal not in range(128)",
                )?
                .into());
            }
            let seq = String::from_utf8(seq).unwrap_or_default();
            let first_word = py_split_ws(title).first().map(|w| w.to_string()).unwrap_or_default();
            let name = if first_word.is_empty() {
                format!("seq_{}", records.len())
            } else {
                first_word
            };
            records.push((name, seq));
            Ok(())
        };
        for line in &lines[1..] {
            if line.starts_with('>') {
                finish(title, &seq_parts, &mut records)?;
                seq_parts.clear();
                title = py_rstrip(&line[1..]);
            } else {
                seq_parts.push(line);
            }
        }
        finish(title, &seq_parts, &mut records)?;
    }
    if records.is_empty() {
        return Err(io_error(
            py,
            format!(
                "No valid FASTA sequences found in '{filepath}'. Ensure the file starts with '>' \
                 header lines followed by sequence data."
            ),
        ));
    }
    Ok(records)
}

/// ``textwrap.fill(text, width, break_on_hyphens=False, break_long_words=True)``.
/// Text without whitespace is cut into ``width``-character lines; anything
/// else goes through ``textwrap`` itself.
fn textwrap_fill(py: Python<'_>, text: &str, width: usize) -> PyResult<String> {
    if text.chars().any(py_is_space) {
        let kwargs = PyDict::new(py);
        kwargs.set_item("width", width)?;
        kwargs.set_item("break_on_hyphens", false)?;
        kwargs.set_item("break_long_words", true)?;
        return py.import("textwrap")?.call_method("fill", (text,), Some(&kwargs))?.extract();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + text.len() / width.max(1) + 1);
    for (i, chunk) in chars.chunks(width.max(1)).enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.extend(chunk.iter());
    }
    Ok(out)
}

/// The elements of ``rec.get_seq()`` as ``str(c)`` joined.
fn sequence_text(seq: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut out = String::new();
    for item in seq.try_iter()? {
        out.push_str(&item?.str()?.to_cow()?);
    }
    Ok(out)
}

fn create_file(py: Python<'_>, filepath: &str, what: &str) -> PyResult<BufWriter<File>> {
    File::create(filepath)
        .map(BufWriter::new)
        .map_err(|e| io_error(py, format!("Failed to write {what} file '{filepath}': {}", os_error_message(&e, filepath))))
}

fn write_text(py: Python<'_>, out: &mut BufWriter<File>, text: &str, filepath: &str, what: &str) -> PyResult<()> {
    out.write_all(text.as_bytes())
        .map_err(|e| io_error(py, format!("Failed to write {what} file '{filepath}': {}", os_error_message(&e, filepath))))
}

fn finish_file(py: Python<'_>, mut out: BufWriter<File>, filepath: &str, what: &str) -> PyResult<()> {
    out.flush()
        .map_err(|e| io_error(py, format!("Failed to write {what} file '{filepath}': {}", os_error_message(&e, filepath))))
}

/// ``IO.write_fasta``: one ``>name`` record per ``DataSequence`` of the MSA,
/// the sequence wrapped at ``line_width`` characters.
#[pyfunction]
fn io_write_fasta(py: Python<'_>, msa: &Bound<'_, PyAny>, filepath: &str, line_width: i64) -> PyResult<()> {
    if line_width < 1 {
        return Err(PyValueError::new_err("line_width must be >= 1"));
    }
    let records = msa.call_method0("get_records")?;
    if records.len()? == 0 {
        return Err(io_error(py, "Cannot write an empty MSA to FASTA: no records present."));
    }
    let mut out = create_file(py, filepath, "FASTA")?;
    for rec in records.try_iter()? {
        let rec = rec?;
        let name = rec.call_method0("get_name")?.str()?.to_cow()?.to_string();
        write_text(py, &mut out, &format!(">{name}\n"), filepath, "FASTA")?;
        let seq_str = sequence_text(&rec.call_method0("get_seq")?)?;
        let wrapped = textwrap_fill(py, &seq_str, line_width as usize)?;
        write_text(py, &mut out, &format!("{wrapped}\n"), filepath, "FASTA")?;
    }
    finish_file(py, out, filepath, "FASTA")
}

// ───────────────────────────────── VCF ─────────────────────────────────

/// ``_parse_vcf_genotype``: the ALT allele count of one genotype field, or
/// ``None`` for missing data.
fn parse_vcf_genotype(gt_str: &str) -> Option<i64> {
    let gt_field = gt_str.split(':').next().unwrap_or("");
    let alleles: Vec<&str> = if gt_field.contains('|') {
        gt_field.split('|').collect()
    } else if gt_field.contains('/') {
        gt_field.split('/').collect()
    } else {
        vec![gt_field]
    };
    if alleles.iter().any(|a| *a == ".") {
        return None;
    }
    let mut total = 0i64;
    for a in alleles {
        total = total.checked_add(py_int(a)?)?;
    }
    Some(total)
}

/// Read a VCF file as ``read_vcf`` does: ``(sample_names, {name: [counts]})``
/// with one entry per site, ``missing_value`` where the genotype is missing.
#[pyfunction]
fn io_read_vcf_records<'py>(
    py: Python<'py>,
    filepath: &str,
    missing_value: &Bound<'py, PyAny>,
) -> PyResult<(Vec<String>, Bound<'py, PyDict>)> {
    if !Path::new(filepath).exists() {
        return Err(PyFileNotFoundError::new_err(format!("VCF file not found: {filepath}")));
    }
    let bytes = fs::read(filepath)
        .map_err(|e| io_error(py, format!("Failed to read VCF file '{filepath}': {}", os_error_message(&e, filepath))))?;
    let text = decode_with_python(py, &bytes, "utf-8")?;

    let mut sample_names: Vec<String> = Vec::new();
    let genotypes = PyDict::new(py);
    let mut columns: IndexMap<String, Bound<'py, PyList>> = IndexMap::new();
    let mut num_variants = 0usize;

    for line in py_lines(&text) {
        if line.starts_with("##") {
            continue;
        }
        if line.starts_with("#CHROM") || line.starts_with("#chrom") {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 10 {
                return Err(io_error(
                    py,
                    "VCF header line has fewer than 10 columns. Expected at least: #CHROM POS ID REF \
                     ALT QUAL FILTER INFO FORMAT <sample1>",
                ));
            }
            sample_names = fields[9..].iter().map(|s| s.to_string()).collect();
            for name in &sample_names {
                let list = PyList::empty(py);
                genotypes.set_item(name, &list)?;
                columns.insert(name.clone(), list);
            }
            continue;
        }
        if py_strip(line).is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 10 {
            continue;
        }
        let fmt_fields: Vec<&str> = fields[8].split(':').collect();
        let Some(gt_index) = fmt_fields.iter().position(|f| *f == "GT") else {
            warn(
                py,
                &format!("VCF line at position {} has no GT field in FORMAT column; skipping.", fields[1]),
            )?;
            continue;
        };
        num_variants += 1;
        for (i, name) in sample_names.iter().enumerate() {
            let gt = if 9 + i < fields.len() {
                let sample_fmt: Vec<&str> = fields[9 + i].split(':').collect();
                if gt_index < sample_fmt.len() {
                    parse_vcf_genotype(sample_fmt[gt_index])
                } else {
                    None
                }
            } else {
                None
            };
            let column = &columns[name];
            match gt {
                Some(count) => column.append(count.to_string())?,
                None => column.append(missing_value)?,
            }
        }
    }

    if sample_names.is_empty() {
        return Err(io_error(
            py,
            format!(
                "No sample columns found in VCF file '{filepath}'. Ensure the file has a #CHROM \
                 header line with sample names."
            ),
        ));
    }
    if num_variants == 0 {
        return Err(io_error(py, format!("No variant records found in VCF file '{filepath}'.")));
    }
    Ok((sample_names, genotypes))
}

/// ``re.search(r'ID=([^,>]+)', line)``.
fn search_id(line: &str) -> Option<String> {
    let mut from = 0;
    while let Some(pos) = line[from..].find("ID=") {
        let start = from + pos + 3;
        let value: String = line[start..].chars().take_while(|c| *c != ',' && *c != '>').collect();
        if !value.is_empty() {
            return Some(value);
        }
        from = start;
    }
    None
}

/// ``IO.read_vcf_metadata``: the header of a VCF file as a dict.
#[pyfunction]
fn io_read_vcf_metadata<'py>(py: Python<'py>, filepath: &str) -> PyResult<Bound<'py, PyDict>> {
    if !Path::new(filepath).exists() {
        return Err(PyFileNotFoundError::new_err(format!("VCF file not found: {filepath}")));
    }
    let bytes = fs::read(filepath)
        .map_err(|e| io_error(py, format!("Failed to read VCF file '{filepath}': {}", os_error_message(&e, filepath))))?;
    let text = decode_with_python(py, &bytes, "utf-8")?;

    let mut fileformat = String::new();
    let mut metadata_lines: Vec<String> = Vec::new();
    let mut sample_names: Vec<String> = Vec::new();
    let mut info_fields: Vec<String> = Vec::new();
    let mut format_fields: Vec<String> = Vec::new();
    let mut filter_fields: Vec<String> = Vec::new();
    let mut contig_fields: Vec<String> = Vec::new();

    for line in py_lines(&text) {
        if line.starts_with("##fileformat") {
            fileformat = line
                .split_once('=')
                .map(|(_, v)| v.to_string())
                .ok_or_else(|| PyIndexError::new_err("list index out of range"))?;
            metadata_lines.push(line.to_string());
        } else if line.starts_with("##INFO") {
            metadata_lines.push(line.to_string());
            info_fields.extend(search_id(line));
        } else if line.starts_with("##FORMAT") {
            metadata_lines.push(line.to_string());
            format_fields.extend(search_id(line));
        } else if line.starts_with("##FILTER") {
            metadata_lines.push(line.to_string());
            filter_fields.extend(search_id(line));
        } else if line.starts_with("##contig") {
            metadata_lines.push(line.to_string());
            contig_fields.extend(search_id(line));
        } else if line.starts_with("##") {
            metadata_lines.push(line.to_string());
        } else if line.starts_with("#CHROM") || line.starts_with("#chrom") {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() >= 10 {
                sample_names = fields[9..].iter().map(|s| s.to_string()).collect();
            }
            break;
        }
    }

    let result = PyDict::new(py);
    result.set_item("fileformat", fileformat)?;
    result.set_item("metadata_lines", metadata_lines)?;
    result.set_item("sample_names", sample_names)?;
    result.set_item("info_fields", info_fields)?;
    result.set_item("format_fields", format_fields)?;
    result.set_item("filter_fields", filter_fields)?;
    result.set_item("contig_fields", contig_fields)?;
    Ok(result)
}

/// ``_allele_count_to_gt``: an allele count back to VCF genotype notation.
fn allele_count_to_gt(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    if value.is_none() {
        return Ok("./.".to_string());
    }
    for marker in ["?", ".", "-"] {
        if value.eq(marker)? {
            return Ok("./.".to_string());
        }
    }
    if value.eq("None")? {
        return Ok("./.".to_string());
    }
    let count = match py.import("builtins")?.getattr("int")?.call1((value,)) {
        Ok(v) => v,
        Err(e) if e.is_instance_of::<PyValueError>(py) || e.is_instance_of::<PyTypeError>(py) => {
            return Ok("./.".to_string());
        }
        Err(e) => return Err(e),
    };
    let count: i64 = count.extract()?;
    Ok(match count {
        0 => "0/0".to_string(),
        1 => "0/1".to_string(),
        2 => "1/1".to_string(),
        c if c < 0 => String::new(),
        c => vec!["1"; c as usize].join("/"),
    })
}

/// ``IO.write_vcf``: a minimal VCF with one record per site and one sample
/// column per ``DataSequence``.
#[pyfunction]
fn io_write_vcf(
    py: Python<'_>,
    msa: &Bound<'_, PyAny>,
    filepath: &str,
    chrom: String,
    start_pos: &Bound<'_, PyAny>,
    ref_allele: String,
    alt_allele: String,
) -> PyResult<()> {
    let records = msa.call_method0("get_records")?;
    if records.len()? == 0 {
        return Err(io_error(py, "Cannot write an empty MSA to VCF: no records present."));
    }
    let recs: Vec<Bound<'_, PyAny>> = records.try_iter()?.collect::<PyResult<_>>()?;
    let mut sample_names: Vec<String> = Vec::with_capacity(recs.len());
    for rec in &recs {
        sample_names.push(rec.call_method0("get_name")?.extract()?);
    }
    let seqs: Vec<Bound<'_, PyAny>> = recs.iter().map(|r| r.call_method0("get_seq")).collect::<PyResult<_>>()?;
    let num_sites = seqs[0].len()?;

    let mut out = create_file(py, filepath, "VCF")?;
    let mut header = String::from("##fileformat=VCFv4.1\n");
    header.push_str("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n");
    header.push_str("##source=PhyNetPy\n");
    let mut columns = vec!["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    columns.extend(sample_names.iter().cloned());
    header.push_str(&columns.join("\t"));
    header.push('\n');
    write_text(py, &mut out, &header, filepath, "VCF")?;

    for site_idx in 0..num_sites {
        let pos = start_pos.add(site_idx)?.str()?.to_cow()?.to_string();
        let mut fields: Vec<String> = vec![
            chrom.clone(),
            pos,
            ".".into(),
            ref_allele.clone(),
            alt_allele.clone(),
            ".".into(),
            "PASS".into(),
            ".".into(),
            "GT".into(),
        ];
        for seq in &seqs {
            if site_idx < seq.len()? {
                fields.push(allele_count_to_gt(py, &seq.get_item(site_idx)?)?);
            } else {
                fields.push("./.".into());
            }
        }
        write_text(py, &mut out, &format!("{}\n", fields.join("\t")), filepath, "VCF")?;
    }
    finish_file(py, out, filepath, "VCF")
}

// ───────────────────────────── Newick writers ──────────────────────────

/// ``IO.write_newick(net)``: the network's own ``newick()``.
fn newick_of(net: &Bound<'_, PyAny>) -> PyResult<String> {
    net.call_method1("newick", (false,))?.extract()
}

/// ``IO.write_newick_file``: one newick string per line.
#[pyfunction]
fn io_write_newick_file(py: Python<'_>, networks: &Bound<'_, PyAny>, filepath: &str) -> PyResult<()> {
    if !networks.is_truthy()? {
        return Err(io_error(py, "No networks provided to write."));
    }
    let mut out = create_file(py, filepath, "newick")?;
    for net in networks.try_iter()? {
        let nwk = newick_of(&net?)?;
        write_text(py, &mut out, &format!("{nwk}\n"), filepath, "newick")?;
    }
    finish_file(py, out, filepath, "newick")
}

/// ``Newick.get_labels``: the taxa labels of a newick string, in order of
/// first appearance.
fn newick_labels(newick_str: &str) -> IndexSet<String> {
    let mut labels = IndexSet::new();
    let mut cur = String::new();
    let clean = |label: &str| -> String {
        let before_colon = label.split(':').next().unwrap_or("");
        let before_bracket = before_colon.split('[').next().unwrap_or("");
        py_strip(before_bracket).to_string()
    };
    for c in newick_str.chars() {
        if c == ')' || c == '(' || c == ',' {
            if !cur.is_empty() {
                let cleaned = clean(&cur);
                if !cleaned.is_empty() {
                    labels.insert(cleaned);
                }
            }
            cur.clear();
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        let cleaned = clean(&cur).trim_end_matches(';').to_string();
        if !cleaned.is_empty() {
            labels.insert(cleaned);
        }
    }
    labels
}

/// ``IO.write_nexus``: TAXA and TREES blocks (plus an optional PHYLONET block).
#[pyfunction]
#[pyo3(signature = (networks, filepath, taxa, tree_prefix, overwrite, phylonet_cmds))]
fn io_write_nexus(
    py: Python<'_>,
    networks: &Bound<'_, PyAny>,
    filepath: &str,
    taxa: &Bound<'_, PyAny>,
    tree_prefix: &Bound<'_, PyAny>,
    overwrite: &Bound<'_, PyAny>,
    phylonet_cmds: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if !networks.is_truthy()? {
        return Err(io_error(py, "No networks provided to write."));
    }
    if !overwrite.is_truthy()? && Path::new(filepath).exists() {
        return Err(io_error(py, format!("File already exists: {filepath}")));
    }
    let mut newick_strings: Vec<String> = Vec::new();
    let mut inferred: IndexSet<String> = IndexSet::new();
    for net in networks.try_iter()? {
        let nwk = newick_of(&net?)?;
        inferred.extend(newick_labels(&nwk));
        newick_strings.push(nwk);
    }
    let mut all_taxa: Vec<String> = if taxa.is_none() {
        inferred.into_iter().collect()
    } else {
        taxa.try_iter()?.map(|t| t?.extract::<String>()).collect::<PyResult<_>>()?
    };
    all_taxa.sort();
    let prefix = tree_prefix.str()?.to_cow()?.to_string();

    let mut out = create_file(py, filepath, "nexus")?;
    let mut text = String::from("#NEXUS\n\n");
    // TAXA block: every label on one line so a NEXUS reader does not count
    // the terminating ';' as a taxon.
    text.push_str("BEGIN TAXA;\n");
    text.push_str(&format!("DIMENSIONS NTAX={};\n", all_taxa.len()));
    text.push_str(&format!("TAXLABELS {};\n", all_taxa.join(" ")));
    text.push_str("END;\n");
    text.push_str("BEGIN TREES;\n");
    for (idx, nwk) in newick_strings.iter().enumerate() {
        text.push_str(&format!("Tree {prefix}{} = {nwk}\n", idx + 1));
    }
    text.push_str("END;\n");
    write_text(py, &mut out, &text, filepath, "nexus")?;
    if phylonet_cmds.is_truthy()? {
        write_text(py, &mut out, "BEGIN PHYLONET;\n", filepath, "nexus")?;
        for cmd in phylonet_cmds.try_iter()? {
            let cmd: String = cmd?.extract()?;
            write_text(py, &mut out, &format!("{cmd}\n"), filepath, "nexus")?;
        }
        write_text(py, &mut out, "END;\n", filepath, "nexus")?;
    }
    finish_file(py, out, filepath, "nexus")
}

// ───────────────────── Newick standard conversion ──────────────────────

/// ``#[A-Za-z_]\w*`` at ``i``: the end of the reticulation name.
fn match_retic_name(chars: &[char], i: usize) -> Option<usize> {
    let n = chars.len();
    if chars[i] != '#' || i + 1 >= n || !(chars[i + 1].is_ascii_alphabetic() || chars[i + 1] == '_') {
        return None;
    }
    let mut j = i + 2;
    while j < n && is_word(chars[j]) {
        j += 1;
    }
    Some(j)
}

/// ``::NUM`` at ``i``: the gamma's span and the end of the match.
fn match_double_colon_num(chars: &[char], i: usize) -> Option<((usize, usize), usize)> {
    if i + 1 < chars.len() && chars[i] == ':' && chars[i + 1] == ':' {
        match_num(chars, i + 2).map(|e| ((i + 2, e), e))
    } else {
        None
    }
}

/// ``(:NUM)?::NUM`` at ``i``: the optional branch span, the gamma span and
/// the end of the match.
#[allow(clippy::type_complexity)]
fn match_phylonet_tail(chars: &[char], i: usize) -> Option<(Option<(usize, usize)>, (usize, usize), usize)> {
    if i < chars.len() && chars[i] == ':' {
        if let Some(e) = match_num(chars, i + 1) {
            if let Some((gamma, end)) = match_double_colon_num(chars, e) {
                return Some((Some((i, e)), gamma, end));
            }
        }
    }
    match_double_colon_num(chars, i).map(|(gamma, end)| (None, gamma, end))
}

fn starts_with_at(chars: &[char], i: usize, pat: &str) -> bool {
    let p: Vec<char> = pat.chars().collect();
    i + p.len() <= chars.len() && chars[i..i + p.len()] == p[..]
}

/// ``re.search(r'\[&[^\]]*gamma=', s)``.
fn has_gamma_annotation(chars: &[char]) -> bool {
    let n = chars.len();
    (0..n).any(|i| {
        if !starts_with_at(chars, i, "[&") {
            return false;
        }
        let end = (i + 2..n).find(|&p| chars[p] == ']').unwrap_or(n);
        (i + 2..end).any(|p| p + 6 <= end && starts_with_at(chars, p, "gamma="))
    })
}

/// ``IO.detect_newick_standard``.
#[pyfunction]
fn io_detect_newick_standard(newick_str: &str) -> String {
    let chars: Vec<char> = py_strip(newick_str).chars().collect();
    let n = chars.len();
    // PhyloNet: double-colon gamma on a reticulation node.
    if (0..n).any(|i| match_retic_name(&chars, i).is_some_and(|e| match_phylonet_tail(&chars, e).is_some())) {
        return "Phylonet".to_string();
    }
    // BEAST: [&R] or [&U] prefix.
    if starts_with_at(&chars, 0, "[&R]") || starts_with_at(&chars, 0, "[&U]") {
        return "Beast".to_string();
    }
    // PhyNetPy: [&gamma=X] comment annotation (also the default).
    let _ = has_gamma_annotation(&chars);
    "PhyNetPy".to_string()
}

/// ``re.search(r'gamma=(NUM)', metadata)``: the gamma value's span.
fn find_gamma_value(chars: &[char]) -> Option<(usize, usize)> {
    (0..chars.len()).find_map(|p| {
        if starts_with_at(chars, p, "gamma=") {
            match_num(chars, p + 6).map(|e| (p + 6, e))
        } else {
            None
        }
    })
}

/// ``re.sub(r',?\s*gamma=NUM', '', metadata)``.
fn remove_gamma_entries(chars: &[char]) -> String {
    let n = chars.len();
    let try_at = |j: usize| -> Option<usize> {
        let mut k = j;
        while k < n && py_is_space(chars[k]) {
            k += 1;
        }
        if starts_with_at(chars, k, "gamma=") {
            match_num(chars, k + 6)
        } else {
            None
        }
    };
    let mut out = String::new();
    let mut i = 0;
    while i < n {
        let hit = if chars[i] == ',' { try_at(i + 1).or_else(|| try_at(i)) } else { try_at(i) };
        match hit {
            Some(end) => i = end,
            None => {
                out.push(chars[i]);
                i += 1;
            }
        }
    }
    out
}

/// ``IO.convert_newick``: between the PhyNetPy, PhyloNet and BEAST gamma
/// conventions.
#[pyfunction]
fn io_convert_newick(py: Python<'_>, newick_str: &str, standard: &Bound<'_, PyAny>) -> PyResult<String> {
    let standard_name: Option<String> = standard.extract().ok();
    let target = match standard_name.as_deref() {
        Some(s @ ("PhyNetPy" | "Phylonet" | "Beast")) => s.to_string(),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid standard '{}'. Must be one of: Beast, PhyNetPy, Phylonet",
                standard.str()?.to_cow()?
            )))
        }
    };
    let s = py_strip(newick_str);
    if s.is_empty() {
        return Err(io_error(py, "Cannot convert an empty newick string."));
    }

    // Step 1: strip a BEAST [&R]/[&U] prefix.
    let s = if s.starts_with("[&R]") || s.starts_with("[&U]") {
        s[4..].trim_start_matches(py_is_space)
    } else {
        s
    };

    // Step 2: PhyloNet "#H:len::gamma" -> "#H[&gamma=g]:len".
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < n {
        if let Some(name_end) = match_retic_name(&chars, i) {
            if let Some((branch, (gs, ge), end)) = match_phylonet_tail(&chars, name_end) {
                out.extend(chars[i..name_end].iter());
                out.push_str("[&gamma=");
                out.extend(chars[gs..ge].iter());
                out.push(']');
                if let Some((bs, be)) = branch {
                    out.extend(chars[bs..be].iter());
                }
                i = end;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    let s = out;

    // Step 3: to the target standard.
    match target.as_str() {
        "PhyNetPy" => Ok(s),
        "Beast" => Ok(format!("[&R] {s}")),
        _ => {
            // "#H[&...gamma=g...]:len" -> "#H[&...]:len::g"
            let chars: Vec<char> = s.chars().collect();
            let n = chars.len();
            let mut out = String::with_capacity(s.len());
            let mut i = 0;
            while i < n {
                if let Some(name_end) = match_retic_name(&chars, i) {
                    if starts_with_at(&chars, name_end, "[&") {
                        let body_start = name_end + 2;
                        if let Some(close) = (body_start..n).find(|&p| chars[p] == ']') {
                            if close > body_start {
                                let mut end = close + 1;
                                let mut branch: Option<(usize, usize)> = None;
                                if end < n && chars[end] == ':' {
                                    if let Some(e) = match_num(&chars, end + 1) {
                                        branch = Some((end, e));
                                        end = e;
                                    }
                                }
                                let metadata = &chars[body_start..close];
                                match find_gamma_value(metadata) {
                                    None => {
                                        // No gamma: leave the node untouched.
                                        out.extend(chars[i..end].iter());
                                    }
                                    Some((gs, ge)) => {
                                        let gamma_val: String = metadata[gs..ge].iter().collect();
                                        let remaining = remove_gamma_entries(metadata);
                                        let remaining = py_strip(remaining.trim_matches(','));
                                        out.extend(chars[i..name_end].iter());
                                        if !remaining.is_empty() {
                                            out.push_str("[&");
                                            out.push_str(remaining);
                                            out.push(']');
                                        }
                                        if let Some((bs, be)) = branch {
                                            out.extend(chars[bs..be].iter());
                                        }
                                        out.push_str("::");
                                        out.push_str(&gamma_val);
                                    }
                                }
                                i = end;
                                continue;
                            }
                        }
                    }
                }
                out.push(chars[i]);
                i += 1;
            }
            Ok(out)
        }
    }
}

// ───────────────────────────── registration ────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_register_io_error, m)?)?;
    m.add_function(wrap_pyfunction!(io_read_newick, m)?)?;
    m.add_function(wrap_pyfunction!(io_nexus_tree_lines, m)?)?;
    m.add_function(wrap_pyfunction!(io_read_nexus_alignment, m)?)?;
    m.add_function(wrap_pyfunction!(io_networks_from_nexus_trees, m)?)?;
    m.add_function(wrap_pyfunction!(io_read_fasta_records, m)?)?;
    m.add_function(wrap_pyfunction!(io_write_fasta, m)?)?;
    m.add_function(wrap_pyfunction!(io_read_vcf_records, m)?)?;
    m.add_function(wrap_pyfunction!(io_read_vcf_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(io_write_vcf, m)?)?;
    m.add_function(wrap_pyfunction!(io_write_newick_file, m)?)?;
    m.add_function(wrap_pyfunction!(io_write_nexus, m)?)?;
    m.add_function(wrap_pyfunction!(io_detect_newick_standard, m)?)?;
    m.add_function(wrap_pyfunction!(io_convert_newick, m)?)?;
    Ok(())
}
