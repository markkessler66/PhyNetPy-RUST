//! The transaction journal shared between a Network and the nodes and edges
//! it owns.  Element setters record their inverse into the journal of the
//! owning network without ever borrowing the network object itself, which is
//! what makes rollback (which mutates the network while calling setters)
//! borrow-safe.

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use super::edge::Edge;
use super::node::Node;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Setter {
    Time,
    IsRetic,
    Gamma,
    Length,
    Tag,
    Weight,
}

/// The value a journaled setter is restoring.
#[derive(Clone, Debug)]
pub enum Old {
    Unset,
    F(f64),
    B(bool),
    S(String),
}

pub enum Element {
    Node(Py<Node>),
    Edge(Py<Edge>),
}

pub enum Entry {
    AddNode(Py<Node>),
    RemoveNode {
        node: Py<Node>,
        was_leaf: bool,
        was_root: bool,
    },
    AddEdge(Py<Edge>),
    RemoveEdge(Py<Edge>),
    Rename {
        node: Py<Node>,
        old: String,
    },
    Attr {
        element: Element,
        setter: Setter,
        old: Old,
    },
}

#[derive(Default)]
struct JournalInner {
    frames: Option<Vec<Vec<Entry>>>,
    replaying: bool,
}

#[derive(Clone, Default)]
pub struct Journal(Arc<Mutex<JournalInner>>);

impl Journal {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, JournalInner> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn begin(&self) {
        let mut g = self.lock();
        g.frames.get_or_insert_with(Vec::new).push(Vec::new());
    }

    pub fn in_transaction(&self) -> bool {
        self.lock().frames.is_some()
    }

    pub fn commit(&self) {
        let mut g = self.lock();
        if let Some(frames) = g.frames.as_mut() {
            let frame = frames.pop().unwrap_or_default();
            if frames.is_empty() {
                g.frames = None;
            } else if let Some(outer) = frames.last_mut() {
                outer.extend(frame);
            }
        }
    }

    /// Pop the innermost frame for replay; closes the journal when it was
    /// the last one.
    pub fn pop_for_rollback(&self) -> Option<Vec<Entry>> {
        let mut g = self.lock();
        let frames = g.frames.as_mut()?;
        let frame = frames.pop();
        if frames.is_empty() {
            g.frames = None;
        }
        frame
    }

    #[inline]
    pub fn is_recording(&self) -> bool {
        let g = self.lock();
        g.frames.is_some() && !g.replaying
    }

    pub fn record(&self, entry: Entry) {
        let mut g = self.lock();
        if g.replaying {
            return;
        }
        if let Some(frames) = g.frames.as_mut() {
            if let Some(frame) = frames.last_mut() {
                frame.push(entry);
            }
        }
    }

    pub fn set_replaying(&self, value: bool) {
        self.lock().replaying = value;
    }
}
