//! ``Network``: a directed graph of ``Node`` objects joined by ``Edge``
//! objects, with leaf/root bookkeeping and journaled transactions.
//!
//! Membership is keyed the way the Python reference keyed it: nodes by
//! label, edges by identity.  Internally every node and edge carries a
//! process-unique id, and all adjacency structures are kept on those ids so
//! the hot paths never hash a string or touch a Python object.

use indexmap::{IndexMap, IndexSet};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple, PyWeakrefReference};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::branch::Branch;
use super::edge::Edge;
use super::journal::{Element, Entry, Journal, Old, Setter};
use super::node::{flatten_nodes, Node, Owner};
use super::sets::{pick_edge, EdgeSet, NodeSet};
use crate::util::{
    coerce_unit, deepcopy, network_class, network_error, type_name, warn,
};

pub struct NodeEntry {
    pub obj: Py<Node>,
    pub in_edges: IndexSet<u64, FxBuildHasher>,
    pub out_edges: IndexSet<u64, FxBuildHasher>,
}

pub struct EdgeEntry {
    pub obj: Py<Edge>,
    pub su: u64,
    pub du: u64,
}

#[pyclass(name = "Network", module = "phynetpy.Network", subclass, weakref)]
pub struct Network {
    pub nodes: IndexMap<u64, NodeEntry, FxBuildHasher>,
    pub names: FxHashMap<String, u64>,
    pub edges: IndexMap<u64, EdgeEntry, FxBuildHasher>,
    pub pair: FxHashMap<(u64, u64), SmallVec<[u64; 2]>>,
    pub leaves: IndexSet<u64, FxBuildHasher>,
    pub roots: IndexSet<u64, FxBuildHasher>,
    pub items: Py<PyDict>,
    pub uid: u64,
    pub unit: Py<PyAny>,
    pub journal: Journal,
    pub weak_self: Option<Py<PyWeakrefReference>>,
}

// ── Rust-level API ─────────────────────────────────────────────────────

impl Network {
    pub fn empty(py: Python<'_>, unit: Py<PyAny>) -> Network {
        Network {
            nodes: IndexMap::default(),
            names: FxHashMap::default(),
            edges: IndexMap::default(),
            pair: FxHashMap::default(),
            leaves: IndexSet::default(),
            roots: IndexSet::default(),
            items: PyDict::new(py).unbind(),
            uid: 0,
            unit,
            journal: Journal::new(),
            weak_self: None,
        }
    }

    /// A new, empty network object.  Goes through the Python-level class
    /// when one is registered, so Rust-made networks are the same type users
    /// construct directly.
    pub fn new_object<'py>(
        py: Python<'py>,
        unit: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Network>> {
        if let Some(cls) = network_class(py) {
            let kwargs = PyDict::new(py);
            kwargs.set_item("branch_length_unit", unit)?;
            let obj = cls.call((), Some(&kwargs))?;
            Ok(obj.cast_into::<Network>()?)
        } else {
            let unit = coerce_unit(py, unit)?.unbind();
            Bound::new(py, Network::empty(py, unit))
        }
    }

    pub fn weak(&mut self, slf: &Bound<'_, PyAny>) -> PyResult<Py<PyWeakrefReference>> {
        if self.weak_self.is_none() {
            self.weak_self = Some(PyWeakrefReference::new(slf)?.unbind());
        }
        Ok(self.weak_self.as_ref().unwrap().clone_ref(slf.py()))
    }

    pub fn owner(&mut self, slf: &Bound<'_, PyAny>) -> PyResult<Owner> {
        Ok(Owner {
            net: self.weak(slf)?,
            journal: self.journal.clone(),
        })
    }

    // ── lookups ──

    #[inline]
    pub fn uid_of_ref(&self, n: &Node) -> Option<u64> {
        if self.nodes.contains_key(&n.uid) {
            Some(n.uid)
        } else {
            self.names.get(&n.name).copied()
        }
    }

    #[inline]
    pub fn uid_of(&self, py: Python<'_>, node: &Py<Node>) -> Option<u64> {
        self.uid_of_ref(&node.borrow(py))
    }

    #[inline]
    pub fn uid_of_bound(&self, node: &Bound<'_, Node>) -> Option<u64> {
        self.uid_of_ref(&node.borrow())
    }

    #[inline]
    pub fn uid_of_any(&self, obj: &Bound<'_, PyAny>) -> Option<u64> {
        obj.cast::<Node>().ok().and_then(|n| self.uid_of_bound(n))
    }

    #[inline]
    pub fn node_obj(&self, uid: u64) -> &Py<Node> {
        &self.nodes[&uid].obj
    }

    #[inline]
    pub fn edge_obj(&self, euid: u64) -> &Py<Edge> {
        &self.edges[&euid].obj
    }

    pub fn label(&self, py: Python<'_>, uid: u64) -> String {
        self.nodes[&uid].obj.borrow(py).name.clone()
    }

    #[inline]
    pub fn in_deg(&self, uid: u64) -> usize {
        self.nodes.get(&uid).map(|e| e.in_edges.len()).unwrap_or(0)
    }

    #[inline]
    pub fn out_deg(&self, uid: u64) -> usize {
        self.nodes.get(&uid).map(|e| e.out_edges.len()).unwrap_or(0)
    }

    pub fn in_edge_uids(&self, uid: u64) -> Vec<u64> {
        self.nodes
            .get(&uid)
            .map(|e| e.in_edges.iter().copied().collect())
            .unwrap_or_default()
    }

    pub fn out_edge_uids(&self, uid: u64) -> Vec<u64> {
        self.nodes
            .get(&uid)
            .map(|e| e.out_edges.iter().copied().collect())
            .unwrap_or_default()
    }

    pub fn children(&self, uid: u64) -> Vec<u64> {
        self.out_edge_uids(uid).into_iter().map(|e| self.edges[&e].du).collect()
    }

    pub fn parents(&self, uid: u64) -> Vec<u64> {
        self.in_edge_uids(uid).into_iter().map(|e| self.edges[&e].su).collect()
    }

    /// Roots with at least one outgoing edge.
    pub fn root_uids(&self) -> Vec<u64> {
        self.roots
            .iter()
            .copied()
            .filter(|&u| self.out_deg(u) != 0)
            .collect()
    }

    /// Leaves with at least one incoming edge.
    pub fn leaf_uids(&self) -> Vec<u64> {
        self.leaves
            .iter()
            .copied()
            .filter(|&u| self.in_deg(u) != 0)
            .collect()
    }

    /// The single root; warns when there are several, errors when none.
    pub fn root_uid(&self, py: Python<'_>) -> PyResult<u64> {
        let roots = self.root_uids();
        if roots.len() > 1 {
            warn(
                py,
                "Asked for singular root, but there are more than one. Returning the \
                 first one, but double check to make sure this is the root you intended \
                 to get!",
            )?;
        }
        match roots.first() {
            Some(&r) => Ok(r),
            None => Err(network_error(
                py,
                "There are no roots in this network. There is either a cycle, or \
                 nothing has been added and this is an empty network.",
            )),
        }
    }

    /// The edge from ``su`` to ``du``, disambiguating bubbles by gamma/tag.
    pub fn edge_between(
        &self,
        py: Python<'_>,
        su: u64,
        du: u64,
        gamma: Option<f64>,
        tag: Option<&str>,
    ) -> PyResult<Py<Edge>> {
        let Some(uids) = self.pair.get(&(su, du)) else {
            return Err(PyValueError::new_err("No matching edges found"));
        };
        let cands: Vec<&Py<Edge>> = uids.iter().map(|u| &self.edges[u].obj).collect();
        pick_edge(py, &cands, gamma, tag)
    }

    /// ``get_edge(src, dest)`` without a gamma: the first edge of a bubble,
    /// with the reference implementation's warning.
    pub fn first_edge_between(&self, py: Python<'_>, su: u64, du: u64) -> PyResult<u64> {
        let Some(uids) = self.pair.get(&(su, du)) else {
            return Err(PyValueError::new_err("No matching edges found"));
        };
        match uids.len() {
            1 => Ok(uids[0]),
            2 => {
                warn(py, "Bubble lookup without gamma - returning first edge")?;
                Ok(uids[0])
            }
            _ => Err(PyValueError::new_err(
                "More than 2 edges found - invalid topology",
            )),
        }
    }

    // ── mutation core ──

    fn reclassify(&mut self, uid: u64, is_par: bool, is_addition: bool) {
        if is_addition {
            if is_par {
                if self.out_deg(uid) == 1 {
                    self.leaves.swap_remove(&uid);
                }
                if self.in_deg(uid) == 0 {
                    self.roots.insert(uid);
                }
            } else {
                if self.in_deg(uid) == 1 {
                    self.roots.swap_remove(&uid);
                }
                if self.out_deg(uid) == 0 {
                    self.leaves.insert(uid);
                }
            }
        } else if is_par {
            if self.out_deg(uid) == 0 {
                self.leaves.insert(uid);
            }
        } else if self.in_deg(uid) == 0 {
            self.roots.insert(uid);
        }
    }

    /// Add one node.  Returns ``false`` when a node with that label (or this
    /// very object) is already present.
    pub fn add_node(&mut self, slf: &Bound<'_, PyAny>, node: &Bound<'_, Node>) -> PyResult<bool> {
        let (uid, name) = {
            let n = node.borrow();
            (n.uid, n.name.clone())
        };
        if self.nodes.contains_key(&uid) || self.names.contains_key(&name) {
            return Ok(false);
        }
        let owner = self.owner(slf)?;
        node.borrow_mut().owner = Some(owner);
        self.nodes.insert(
            uid,
            NodeEntry {
                obj: node.clone().unbind(),
                in_edges: IndexSet::default(),
                out_edges: IndexSet::default(),
            },
        );
        self.names.insert(name, uid);
        if self.journal.is_recording() {
            self.journal.record(Entry::AddNode(node.clone().unbind()));
        }
        Ok(true)
    }

    /// Add one edge; both endpoints must already be in V.
    pub fn add_edge(&mut self, slf: &Bound<'_, PyAny>, edge: &Bound<'_, Edge>) -> PyResult<()> {
        let py = slf.py();
        let (euid, su, du) = {
            let e = edge.borrow();
            (e.uid, self.uid_of(py, &e.src), self.uid_of(py, &e.dest))
        };
        let (su, du) = match (su, du) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                return Err(network_error(
                    py,
                    "Tried to add an edge between two nodes, at least one of which does \
                     not belong to this network.",
                ))
            }
        };
        if self.edges.contains_key(&euid) {
            return Ok(());
        }
        let owner = self.owner(slf)?;
        edge.borrow_mut().owner = Some(owner);
        self.edges.insert(
            euid,
            EdgeEntry {
                obj: edge.clone().unbind(),
                su,
                du,
            },
        );
        self.pair.entry((su, du)).or_default().push(euid);
        self.nodes.get_mut(&su).unwrap().out_edges.insert(euid);
        self.nodes.get_mut(&du).unwrap().in_edges.insert(euid);
        if self.journal.is_recording() {
            self.journal.record(Entry::AddEdge(edge.clone().unbind()));
        }
        self.reclassify(su, true, true);
        self.reclassify(du, false, true);
        Ok(())
    }

    pub fn remove_edge_uid(&mut self, py: Python<'_>, euid: u64) {
        let Some(EdgeEntry { obj, su, du }) = self.edges.swap_remove(&euid) else {
            return;
        };
        if let Some(v) = self.pair.get_mut(&(su, du)) {
            v.retain(|x| *x != euid);
            if v.is_empty() {
                self.pair.remove(&(su, du));
            }
        }
        if let Some(n) = self.nodes.get_mut(&su) {
            n.out_edges.swap_remove(&euid);
        }
        if let Some(n) = self.nodes.get_mut(&du) {
            n.in_edges.swap_remove(&euid);
        }
        obj.borrow_mut(py).owner = None;
        if self.journal.is_recording() {
            self.journal.record(Entry::RemoveEdge(obj.clone_ref(py)));
        }
        self.reclassify(su, true, false);
        self.reclassify(du, false, false);
    }

    pub fn remove_node_uid(&mut self, py: Python<'_>, uid: u64) {
        let Some(entry) = self.nodes.get(&uid) else {
            return;
        };
        let ins: Vec<u64> = entry.in_edges.iter().copied().collect();
        let outs: Vec<u64> = entry.out_edges.iter().copied().collect();
        for e in ins {
            self.remove_edge_uid(py, e);
        }
        for e in outs {
            self.remove_edge_uid(py, e);
        }
        let was_leaf = self.leaves.contains(&uid);
        let was_root = self.roots.contains(&uid);
        let entry = self.nodes.swap_remove(&uid).unwrap();
        let name = entry.obj.borrow(py).name.clone();
        if self.names.get(&name) == Some(&uid) {
            self.names.remove(&name);
        }
        if self.journal.is_recording() {
            self.journal.record(Entry::RemoveNode {
                node: entry.obj.clone_ref(py),
                was_leaf,
                was_root,
            });
        }
        entry.obj.borrow_mut(py).owner = None;
        self.leaves.swap_remove(&uid);
        self.roots.swap_remove(&uid);
    }

    pub fn rename_uid(&mut self, py: Python<'_>, uid: u64, new_name: String) {
        let obj = self.nodes[&uid].obj.clone_ref(py);
        let old = obj.borrow(py).name.clone();
        if self.journal.is_recording() {
            self.journal.record(Entry::Rename {
                node: obj.clone_ref(py),
                old: old.clone(),
            });
        }
        if self.names.get(&old) == Some(&uid) {
            self.names.remove(&old);
        }
        obj.borrow_mut(py).set_name_inner(new_name.clone());
        self.names.insert(new_name, uid);
    }

    /// A label of the form ``{prefix}{n}`` not used by any node.
    pub fn fresh_label(&self, py: Python<'_>, prefix: &str) -> String {
        let used: FxHashSet<String> = self
            .nodes
            .values()
            .map(|e| e.obj.borrow(py).name.clone())
            .collect();
        let mut n = 0usize;
        loop {
            let label = format!("{prefix}{n}");
            if !used.contains(&label) {
                return label;
            }
            n += 1;
        }
    }

    // ── bulk operations that need the Python object (owners) ──

    fn recompute_caches(&mut self) {
        let uids: Vec<u64> = self.nodes.keys().copied().collect();
        self.leaves = uids.iter().copied().filter(|&u| self.out_deg(u) == 0).collect();
        self.roots = uids.iter().copied().filter(|&u| self.in_deg(u) == 0).collect();
    }

    /// ``Network.copy``: an independent structural copy plus old -> new map.
    pub fn copy_impl<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, Network>, Vec<(Py<Node>, Py<Node>)>)> {
        let py = slf.py();
        let unit = slf.borrow().unit.clone_ref(py);
        let new = Network::new_object(py, Some(unit.bind(py)))?;
        let mut mapping: Vec<(Py<Node>, Py<Node>)> = Vec::new();
        let mut by_uid: FxHashMap<u64, Py<Node>> = FxHashMap::default();
        let (node_objs, edge_objs, uid_count) = {
            let this = slf.borrow();
            (
                this.nodes
                    .iter()
                    .map(|(u, e)| (*u, e.obj.clone_ref(py)))
                    .collect::<Vec<_>>(),
                this.edges
                    .values()
                    .map(|e| (e.obj.clone_ref(py), e.su, e.du))
                    .collect::<Vec<_>>(),
                this.uid,
            )
        };
        {
            let mut n = new.borrow_mut();
            for (uid, obj) in node_objs {
                let copy = obj.borrow(py).copy(py)?;
                n.add_node(new.as_any(), copy.bind(py))?;
                by_uid.insert(uid, copy.clone_ref(py));
                mapping.push((obj, copy));
            }
            for (obj, su, du) in edge_objs {
                let e = obj.borrow(py);
                let ns = by_uid[&su].clone_ref(py);
                let nd = by_uid[&du].clone_ref(py);
                let copy = e.copy(py, Some(ns), Some(nd))?;
                n.add_edge(new.as_any(), copy.bind(py))?;
            }
            n.uid = uid_count;
        }
        Ok((new, mapping))
    }

    /// ``Network.subnet``: copy of the subgraph under ``start`` with
    /// ``_copyN`` labels.
    pub fn subnet_impl<'py>(slf: &Bound<'py, Self>, start: u64) -> PyResult<Bound<'py, Network>> {
        let py = slf.py();
        let this = slf.borrow();
        let new = Network::new_object(py, None)?;
        let mut suffix = 0usize;
        let mut map: FxHashMap<u64, Py<Node>> = FxHashMap::default();
        let mk = |label: &str, suffix: &mut usize| -> PyResult<Py<Node>> {
            let name = format!("{label}_copy{suffix}");
            *suffix += 1;
            Py::new(py, Node::make(py, name, false, None, None, None))
        };
        {
            let mut n = new.borrow_mut();
            let first = mk(&this.label(py, start), &mut suffix)?;
            n.add_node(new.as_any(), first.bind(py))?;
            map.insert(start, first);
            let mut stack = vec![start];
            while let Some(cur) = stack.pop() {
                for nb in this.children(cur) {
                    let nn = mk(&this.label(py, nb), &mut suffix)?;
                    n.add_node(new.as_any(), nn.bind(py))?;
                    let edge = Py::new(
                        py,
                        Edge::make(
                            py,
                            map[&cur].clone_ref(py),
                            nn.clone_ref(py),
                            Some(1.0),
                            Some(0.0),
                            Some(0.0),
                            Some("no tag assigned yet".to_string()),
                        ),
                    )?;
                    map.insert(nb, nn);
                    n.add_edge(new.as_any(), edge.bind(py))?;
                    stack.push(nb);
                }
            }
        }
        Ok(new)
    }

    /// ``Network.clean``.
    pub fn clean_impl(slf: &Bound<'_, Self>, options: [bool; 3]) -> PyResult<()> {
        let py = slf.py();
        if options[0] {
            let floaters: Vec<u64> = {
                let this = slf.borrow();
                this.nodes
                    .keys()
                    .copied()
                    .filter(|&u| this.in_deg(u) == 0 && this.out_deg(u) == 0)
                    .collect()
            };
            let mut this = slf.borrow_mut();
            for f in floaters {
                this.remove_node_uid(py, f);
            }
        }
        if options[1] {
            let mut this = slf.borrow_mut();
            let root = this.root_uid(py)?;
            if this.out_deg(root) == 1 {
                let child = this.children(root)[0];
                let e = this.first_edge_between(py, root, child)?;
                this.remove_edge_uid(py, e);
                this.remove_node_uid(py, root);
            }
        }
        if options[2] {
            loop {
                let spacer = {
                    let this = slf.borrow();
                    this.nodes
                        .keys()
                        .copied()
                        .find(|&u| this.in_deg(u) == 1 && this.out_deg(u) == 1)
                };
                let Some(cur) = spacer else { break };
                let (par, child, len_in, len_out, gamma_out, e_in, e_out) = {
                    let this = slf.borrow();
                    let par = this.parents(cur)[0];
                    let child = this.children(cur)[0];
                    let e_in = this.first_edge_between(py, par, cur)?;
                    let e_out = this.first_edge_between(py, cur, child)?;
                    let len_in = this.edges[&e_in].obj.borrow(py).length.unwrap_or(0.0);
                    let (len_out, gamma_out) = {
                        let e = this.edges[&e_out].obj.borrow(py);
                        (e.length.unwrap_or(0.0), e.gamma)
                    };
                    (par, child, len_in, len_out, gamma_out, e_in, e_out)
                };
                let len_in = if len_in != 0.0 { len_in } else { 0.0 };
                let len_out = if len_out != 0.0 { len_out } else { 0.0 };
                let merged = {
                    let this = slf.borrow();
                    Py::new(
                        py,
                        Edge::make(
                            py,
                            this.node_obj(par).clone_ref(py),
                            this.node_obj(child).clone_ref(py),
                            Some(len_in + len_out),
                            Some(0.0),
                            Some(0.0),
                            Some("no tag assigned yet".to_string()),
                        ),
                    )?
                };
                {
                    let mut this = slf.borrow_mut();
                    this.remove_edge_uid(py, e_in);
                    this.remove_edge_uid(py, e_out);
                    this.remove_node_uid(py, cur);
                }
                if let Some(g) = gamma_out {
                    if g != 0.0 {
                        Edge::set_gamma(merged.bind(py), g)?;
                    }
                }
                slf.borrow_mut().add_edge(slf.as_any(), merged.bind(py))?;
            }
        }
        Ok(())
    }

    /// Copy ``subtree`` into ``slf`` with fresh internal labels; returns the
    /// image of the subtree's root.
    pub fn absorb_subtree(slf: &Bound<'_, Self>, subtree: &Bound<'_, Network>) -> PyResult<Py<Node>> {
        let py = slf.py();
        let mut used: FxHashSet<String> = {
            let this = slf.borrow();
            this.nodes.values().map(|e| e.obj.borrow(py).name.clone()).collect()
        };
        let sub = subtree.borrow();
        let mut mapping: FxHashMap<u64, Py<Node>> = FxHashMap::default();
        for (&uid, entry) in sub.nodes.iter() {
            let (label, is_retic) = {
                let n = entry.obj.borrow(py);
                (n.name.clone(), n.is_retic)
            };
            let is_leaf = sub.out_deg(uid) == 0;
            let image = if is_leaf {
                if used.contains(&label) {
                    return Err(network_error(
                        py,
                        format!("Cannot graft leaf '{label}': it is already in the network."),
                    ));
                }
                Py::new(py, Node::make(py, label, false, None, None, None))?
            } else {
                let mut l = label;
                if l.is_empty() || used.contains(&l) {
                    l = slf.borrow().fresh_label(py, "I");
                    let mut k = 0usize;
                    while used.contains(&l) {
                        k += 1;
                        l = format!("I{k}");
                    }
                }
                Py::new(py, Node::make(py, l, is_retic, None, None, None))?
            };
            used.insert(image.borrow(py).name.clone());
            slf.borrow_mut().add_node(slf.as_any(), image.bind(py))?;
            mapping.insert(uid, image);
        }
        for entry in sub.edges.values() {
            let (length, gamma) = {
                let e = entry.obj.borrow(py);
                (e.length, e.gamma)
            };
            let edge = Py::new(
                py,
                Edge::make(
                    py,
                    mapping[&entry.su].clone_ref(py),
                    mapping[&entry.du].clone_ref(py),
                    Some(length.unwrap_or(1.0)),
                    Some(0.0),
                    Some(0.0),
                    Some("no tag assigned yet".to_string()),
                ),
            )?;
            if let Some(g) = gamma {
                if g != 0.0 {
                    Edge::set_gamma(edge.bind(py), g)?;
                }
            }
            slf.borrow_mut().add_edge(slf.as_any(), edge.bind(py))?;
        }
        let roots: Vec<u64> = sub
            .nodes
            .keys()
            .copied()
            .filter(|&u| sub.in_deg(u) == 0)
            .collect();
        if roots.len() != 1 {
            return Err(network_error(
                py,
                "A grafted subtree must have exactly one root.",
            ));
        }
        Ok(mapping[&roots[0]].clone_ref(py))
    }

    fn default_edge(py: Python<'_>, src: Py<Node>, dest: Py<Node>, length: Option<f64>) -> PyResult<Py<Edge>> {
        Py::new(
            py,
            Edge::make(
                py,
                src,
                dest,
                Some(length.unwrap_or(1.0)),
                Some(0.0),
                Some(0.0),
                Some("no tag assigned yet".to_string()),
            ),
        )
    }

    fn node_set<'py>(&self, py: Python<'py>, uids: impl IntoIterator<Item = u64>) -> PyResult<Bound<'py, PySet>> {
        PySet::new(py, uids.into_iter().map(|u| self.nodes[&u].obj.bind(py)))
    }

    fn node_list<'py>(&self, py: Python<'py>, uids: impl IntoIterator<Item = u64>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, uids.into_iter().map(|u| self.nodes[&u].obj.bind(py)))
    }

    fn edge_list<'py>(&self, py: Python<'py>, uids: impl IntoIterator<Item = u64>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, uids.into_iter().map(|u| self.edges[&u].obj.bind(py)))
    }

    fn require_uid(&self, py: Python<'_>, node: &Bound<'_, PyAny>, msg: &str) -> PyResult<u64> {
        self.uid_of_any(node).ok_or_else(|| network_error(py, msg))
    }
}

// ── Python API ─────────────────────────────────────────────────────────

#[pymethods]
impl Network {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn py_new(
        py: Python<'_>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Ok(Network::empty(py, coerce_unit(py, None)?.unbind()))
    }

    /// The real constructor body, called by ``Network.__init__``.
    #[pyo3(signature = (edges=None, nodes=None, branch_length_unit=None))]
    fn _init(
        slf: &Bound<'_, Self>,
        edges: Option<Bound<'_, PyAny>>,
        nodes: Option<Bound<'_, PyAny>>,
        branch_length_unit: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let py = slf.py();
        let unit = coerce_unit(py, branch_length_unit.as_ref())?.unbind();
        let edge_objs: Vec<Bound<'_, Edge>> = match &edges {
            None => Vec::new(),
            Some(e) => {
                if let Ok(es) = e.cast::<EdgeSet>() {
                    es.borrow().edge_objects(py).into_iter().map(|p| p.into_bound(py)).collect()
                } else {
                    let mut v = Vec::new();
                    for item in e.try_iter()? {
                        let item = item?;
                        match item.cast::<Edge>() {
                            Ok(ed) => v.push(ed.clone()),
                            Err(_) => {
                                return Err(PyTypeError::new_err(format!(
                                    "Networks are directed and take Edge objects. Got a {}.",
                                    type_name(&item)
                                )))
                            }
                        }
                    }
                    v
                }
            }
        };
        let node_objs: Option<Vec<Bound<'_, Node>>> = match &nodes {
            None => None,
            Some(n) => Some(if let Ok(ns) = n.cast::<NodeSet>() {
                ns.borrow().node_objects(py).into_iter().map(|p| p.into_bound(py)).collect()
            } else {
                let mut v = Vec::new();
                for item in n.try_iter()? {
                    let item = item?;
                    match item.cast::<Node>() {
                        Ok(nd) => v.push(nd.clone()),
                        Err(_) => {
                            return Err(PyTypeError::new_err(format!(
                                "Expected Node objects; got {}.",
                                type_name(&item)
                            )))
                        }
                    }
                }
                v
            }),
        };
        let mut this = slf.borrow_mut();
        this.unit = unit;
        match node_objs {
            Some(ns) => {
                for n in ns {
                    this.add_node(slf.as_any(), &n)?;
                }
            }
            None => {
                for e in &edge_objs {
                    let (s, d) = {
                        let b = e.borrow();
                        (b.src.clone_ref(py), b.dest.clone_ref(py))
                    };
                    this.add_node(slf.as_any(), s.bind(py))?;
                    this.add_node(slf.as_any(), d.bind(py))?;
                }
            }
        }
        for e in &edge_objs {
            let ready = {
                let b = e.borrow();
                this.uid_of(py, &b.src).is_some() && this.uid_of(py, &b.dest).is_some()
            };
            if !ready {
                return Err(PyValueError::new_err(
                    "Edge contains node not in network. Add node first!",
                ));
            }
            this.add_edge(slf.as_any(), e)?;
        }
        this.recompute_caches();
        Ok(())
    }

    // ── transactions ──

    fn begin_transaction(&self) {
        self.journal.begin();
    }

    fn in_transaction(&self) -> bool {
        self.journal.in_transaction()
    }

    fn commit_transaction(&self) {
        self.journal.commit();
    }

    fn rollback_transaction(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let journal = slf.borrow().journal.clone();
        let Some(frame) = journal.pop_for_rollback() else {
            return Ok(());
        };
        journal.set_replaying(true);
        let result = (|| -> PyResult<()> {
            for entry in frame.into_iter().rev() {
                match entry {
                    Entry::AddNode(node) => {
                        let mut this = slf.borrow_mut();
                        let uid = node.borrow(py).uid;
                        if let Some(e) = this.nodes.get(&uid) {
                            let ins: Vec<u64> = e.in_edges.iter().copied().collect();
                            let outs: Vec<u64> = e.out_edges.iter().copied().collect();
                            for x in ins.into_iter().chain(outs) {
                                this.remove_edge_uid(py, x);
                            }
                            let e = this.nodes.swap_remove(&uid).unwrap();
                            let name = e.obj.borrow(py).name.clone();
                            if this.names.get(&name) == Some(&uid) {
                                this.names.remove(&name);
                            }
                        }
                        this.leaves.swap_remove(&uid);
                        this.roots.swap_remove(&uid);
                        node.borrow_mut(py).owner = None;
                    }
                    Entry::RemoveNode {
                        node,
                        was_leaf,
                        was_root,
                    } => {
                        let mut this = slf.borrow_mut();
                        let (uid, name) = {
                            let n = node.borrow(py);
                            (n.uid, n.name.clone())
                        };
                        let owner = this.owner(slf.as_any())?;
                        node.borrow_mut(py).owner = Some(owner);
                        this.nodes.insert(
                            uid,
                            NodeEntry {
                                obj: node,
                                in_edges: IndexSet::default(),
                                out_edges: IndexSet::default(),
                            },
                        );
                        this.names.insert(name, uid);
                        if was_leaf {
                            this.leaves.insert(uid);
                        }
                        if was_root {
                            this.roots.insert(uid);
                        }
                    }
                    Entry::AddEdge(edge) => {
                        let mut this = slf.borrow_mut();
                        let euid = edge.borrow(py).uid;
                        this.remove_edge_uid(py, euid);
                    }
                    Entry::RemoveEdge(edge) => {
                        let mut this = slf.borrow_mut();
                        this.add_edge(slf.as_any(), edge.bind(py))?;
                    }
                    Entry::Rename { node, old } => {
                        let mut this = slf.borrow_mut();
                        let uid = node.borrow(py).uid;
                        if this.nodes.contains_key(&uid) {
                            this.rename_uid(py, uid, old);
                        } else {
                            node.borrow_mut(py).set_name_inner(old);
                        }
                    }
                    Entry::Attr {
                        element,
                        setter,
                        old,
                    } => {
                        let f = |o: &Old| match o {
                            Old::F(v) => Some(*v),
                            _ => None,
                        };
                        match element {
                            Element::Node(n) => {
                                let mut nb = n.borrow_mut(py);
                                match setter {
                                    Setter::Time => nb.t = f(&old),
                                    Setter::IsRetic => {
                                        if let Old::B(b) = old {
                                            nb.is_retic = b;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            Element::Edge(e) => {
                                let mut eb = e.borrow_mut(py);
                                match setter {
                                    Setter::Gamma => eb.gamma = f(&old),
                                    Setter::Length => eb.length = f(&old),
                                    Setter::Weight => eb.weight = f(&old),
                                    Setter::Tag => {
                                        eb.tag = match old {
                                            Old::S(s) => Some(s),
                                            _ => None,
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        })();
        journal.set_replaying(false);
        result
    }

    // ── membership and structure ──

    fn __contains__(&self, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
        if obj.is_exact_instance_of::<Edge>() {
            let uid = obj.cast::<Edge>()?.borrow().uid;
            Ok(self.edges.contains_key(&uid))
        } else if obj.is_exact_instance_of::<Node>() {
            Ok(self.uid_of_any(obj).is_some())
        } else {
            Err(PyTypeError::new_err(
                "Networks only contain Node and Edge objects.",
            ))
        }
    }

    #[pyo3(signature = (*nodes))]
    fn add_nodes(slf: &Bound<'_, Self>, nodes: &Bound<'_, PyTuple>) -> PyResult<()> {
        let flat = flatten_nodes(nodes)?;
        let mut this = slf.borrow_mut();
        for n in flat {
            this.add_node(slf.as_any(), &n)?;
        }
        Ok(())
    }

    #[pyo3(signature = (node=None))]
    fn add_uid_node(slf: &Bound<'_, Self>, node: Option<Bound<'_, Node>>) -> PyResult<Py<Node>> {
        let py = slf.py();
        match node {
            None => {
                let name = format!("UID_{}", slf.borrow().uid);
                let n = Py::new(py, Node::make(py, name, false, None, None, None))?;
                let mut this = slf.borrow_mut();
                this.add_node(slf.as_any(), n.bind(py))?;
                this.uid += 1;
                Ok(n)
            }
            Some(n) => {
                let mut this = slf.borrow_mut();
                if this.uid_of_bound(&n).is_none() {
                    this.add_node(slf.as_any(), &n)?;
                }
                let uid = this.uid_of_bound(&n).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Node {} not found in NodeSet",
                        n.borrow().name
                    ))
                })?;
                let name = format!("UID_{}", this.uid);
                this.rename_uid(py, uid, name);
                this.uid += 1;
                Ok(n.unbind())
            }
        }
    }

    fn uid_count(&self) -> u64 {
        self.uid
    }

    fn get_branch_length_unit(&self, py: Python<'_>) -> Py<PyAny> {
        self.unit.clone_ref(py)
    }

    fn set_branch_length_unit(&mut self, py: Python<'_>, unit: Bound<'_, PyAny>) -> PyResult<()> {
        self.unit = coerce_unit(py, Some(&unit))?.unbind();
        Ok(())
    }

    fn set_uid_count(&mut self, value: u64) {
        self.uid = value;
    }

    #[pyo3(name = "V")]
    fn v_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.nodes.values().map(|e| e.obj.bind(py)))
    }

    #[pyo3(name = "E")]
    fn e_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.edges.values().map(|e| e.obj.bind(py)))
    }

    fn get_item<'py>(&self, py: Python<'py>, key: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        match self.items.bind(py).get_item(&key)? {
            Some(v) => Ok(v),
            None => Err(PyKeyError::new_err(key.unbind())),
        }
    }

    fn put_item(&self, py: Python<'_>, key: Bound<'_, PyAny>, item: Bound<'_, PyAny>) -> PyResult<()> {
        let items = self.items.bind(py);
        if !items.contains(&key)? {
            items.set_item(key, item)?;
        }
        Ok(())
    }

    fn has_node_named(&self, py: Python<'_>, name: Bound<'_, PyAny>) -> Option<Py<Node>> {
        let name: String = name.extract().ok()?;
        self.names.get(&name).map(|u| self.nodes[u].obj.clone_ref(py))
    }

    fn in_degree(&self, py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<usize> {
        match self.uid_of_any(node) {
            Some(u) => Ok(self.in_deg(u)),
            None => {
                warn(py, "Attempting to get the in-degree of a node that is not in the graph-- returning 0")?;
                Ok(0)
            }
        }
    }

    fn out_degree(&self, py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<usize> {
        match self.uid_of_any(node) {
            Some(u) => Ok(self.out_deg(u)),
            None => {
                warn(py, "Attempting to get the out-degree of a node that is not in the graph-- returning 0")?;
                Ok(0)
            }
        }
    }

    fn in_edges<'py>(&self, py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        match self.uid_of_any(node) {
            Some(u) => Ok(PySet::new(py, self.nodes[&u].in_edges.iter().map(|e| self.edges[e].obj.bind(py)))?.into_any()),
            None => {
                warn(py, "Attempting to get the in-edges of a node that is not in the graph-- returning an empty list")?;
                Ok(PyList::empty(py).into_any())
            }
        }
    }

    fn out_edges<'py>(&self, py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        match self.uid_of_any(node) {
            Some(u) => Ok(PySet::new(py, self.nodes[&u].out_edges.iter().map(|e| self.edges[e].obj.bind(py)))?.into_any()),
            None => {
                warn(py, "Attempting to get the out-edges of a node that is not in the graph-- returning an empty list")?;
                Ok(PyList::empty(py).into_any())
            }
        }
    }

    fn add_edges(slf: &Bound<'_, Self>, edges: Bound<'_, PyAny>) -> PyResult<()> {
        if edges.is_exact_instance_of::<PyList>() {
            let mut this = slf.borrow_mut();
            for item in edges.try_iter()? {
                let item = item?;
                let e = item.cast::<Edge>().map_err(|_| {
                    PyTypeError::new_err(format!(
                        "Networks are directed and hold only Edge objects. Got a {}.",
                        type_name(&item)
                    ))
                })?;
                this.add_edge(slf.as_any(), e)?;
            }
        } else if edges.is_exact_instance_of::<Edge>() {
            let e = edges.cast::<Edge>()?;
            slf.borrow_mut().add_edge(slf.as_any(), e)?;
        }
        Ok(())
    }

    fn remove_nodes(slf: &Bound<'_, Self>, node: &Bound<'_, PyAny>) {
        let py = slf.py();
        let mut this = slf.borrow_mut();
        if let Some(u) = this.uid_of_any(node) {
            this.remove_node_uid(py, u);
        }
    }

    #[pyo3(signature = (edge, gamma=None))]
    fn remove_edge(slf: &Bound<'_, Self>, edge: Bound<'_, PyAny>, gamma: Option<f64>) -> PyResult<()> {
        let py = slf.py();
        let edge = if edge.is_exact_instance_of::<PyList>() {
            let l = edge.cast::<PyList>()?;
            if l.len() != 2 {
                return Err(network_error(
                    py,
                    "Please provide a list of two nodes, in the format [src, dest]",
                ));
            }
            let a = l.get_item(0)?;
            let b = l.get_item(1)?;
            let this = slf.borrow();
            let (su, du) = (this.uid_of_any(&a), this.uid_of_any(&b));
            let (Some(su), Some(du)) = (su, du) else {
                return Err(PyValueError::new_err("No matching edges found"));
            };
            this.edge_between(py, su, du, gamma, None)?.into_bound(py).into_any()
        } else {
            edge
        };
        if edge.is_exact_instance_of::<Edge>() {
            let uid = edge.cast::<Edge>()?.borrow().uid;
            let mut this = slf.borrow_mut();
            if this.edges.contains_key(&uid) {
                this.remove_edge_uid(py, uid);
            }
            Ok(())
        } else {
            Err(network_error(
                py,
                format!(
                    "Tried to remove a {} from a Network, which holds only Edge objects.",
                    type_name(&edge)
                ),
            ))
        }
    }

    fn update_node_name(slf: &Bound<'_, Self>, node: &Bound<'_, Node>, name: String) -> PyResult<()> {
        let py = slf.py();
        let mut this = slf.borrow_mut();
        let Some(uid) = this.uid_of_bound(node) else {
            return Err(PyValueError::new_err(format!(
                "Node {} not found in NodeSet",
                node.borrow().name
            )));
        };
        this.rename_uid(py, uid, name);
        Ok(())
    }

    #[pyo3(signature = (n1, n2, gamma=None, tag=None))]
    fn get_edge(
        &self,
        py: Python<'_>,
        n1: &Bound<'_, PyAny>,
        n2: &Bound<'_, PyAny>,
        gamma: Option<f64>,
        tag: Option<String>,
    ) -> PyResult<Py<Edge>> {
        let (Some(su), Some(du)) = (self.uid_of_any(n1), self.uid_of_any(n2)) else {
            return Err(PyValueError::new_err("No matching edges found"));
        };
        self.edge_between(py, su, du, gamma, tag.as_deref())
    }

    fn root(&self, py: Python<'_>) -> PyResult<Py<Node>> {
        let r = self.root_uid(py)?;
        Ok(self.nodes[&r].obj.clone_ref(py))
    }

    fn roots<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.node_list(py, self.root_uids())
    }

    fn get_leaves<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.node_list(py, self.leaf_uids())
    }

    fn get_parents<'py>(&self, py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
        match self.uid_of_any(node) {
            Some(u) => self.node_list(py, self.parents(u)),
            None => Ok(PyList::empty(py)),
        }
    }

    fn get_children<'py>(&self, py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
        match self.uid_of_any(node) {
            Some(u) => self.node_list(py, self.children(u)),
            None => Ok(PyList::empty(py)),
        }
    }

    fn get_branches<'py>(&self, py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let (ins, outs) = match self.uid_of_any(node) {
            Some(u) => (self.in_edge_uids(u), self.out_edge_uids(u)),
            None => (Vec::new(), Vec::new()),
        };
        let mk = |uids: Vec<u64>| -> Vec<Branch> {
            uids.into_iter()
                .map(|e| {
                    let eb = self.edges[&e].obj.borrow(py);
                    let parent = eb.src.borrow(py).name.clone();
                    Branch::make(eb.length, eb.gamma, Some(parent))
                })
                .collect()
        };
        d.set_item("parent_branches", mk(ins))?;
        d.set_item("child_branches", mk(outs))?;
        Ok(d)
    }

    #[pyo3(signature = (options=None))]
    fn clean(slf: &Bound<'_, Self>, options: Option<Vec<bool>>) -> PyResult<()> {
        let o = options.unwrap_or_else(|| vec![true, true, true]);
        let get = |i: usize| o.get(i).copied().unwrap_or(false);
        Network::clean_impl(slf, [get(0), get(1), get(2)])
    }

    fn mrca(&self, py: Python<'_>, set_of_nodes: Bound<'_, PyAny>) -> PyResult<Py<Node>> {
        let mut targets: Vec<u64> = Vec::new();
        let mut seen: FxHashSet<u64> = FxHashSet::default();
        for item in set_of_nodes.try_iter()? {
            let item = item?;
            let uid = if item.is_exact_instance_of::<pyo3::types::PyString>() {
                let name: String = item.extract()?;
                match self.names.get(&name) {
                    Some(&u) => u,
                    None => return Err(network_error(py, "A node in 'set_of_nodes' is not in the graph")),
                }
            } else if item.is_exact_instance_of::<Node>() {
                match self.uid_of_any(&item) {
                    Some(u) => u,
                    None => return Err(network_error(py, "A node in 'set_of_nodes' is not in the graph")),
                }
            } else {
                return Err(network_error(
                    py,
                    "Wrong type for parameter set_of_nodes. Expected set[Node] or set[str].",
                ));
            };
            if seen.insert(uid) {
                targets.push(uid);
            }
        }
        let m = self.mrca_uids(py, &targets)?;
        Ok(self.nodes[&m].obj.clone_ref(py))
    }

    fn leaf_descendants<'py>(&self, py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PySet>> {
        let u = self.require_uid(py, node, "Node not found in graph.")?;
        self.node_set(py, self.leaf_descendants_uid(u))
    }

    fn diff_subtree_edges<'py>(&self, py: Python<'py>, rng: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
        let all: Vec<u64> = self.edges.keys().copied().collect();
        let pick = |n: usize| -> PyResult<usize> { rng.call_method1("integers", (0usize, n))?.extract::<usize>() };
        let first = all[pick(all.len())?];
        let first_leaves: FxHashSet<u64> = self.leaf_descendants_uid(self.edges[&first].du).into_iter().collect();
        let valid: Vec<u64> = all
            .iter()
            .copied()
            .filter(|e| {
                self.leaf_descendants_uid(self.edges[e].du)
                    .iter()
                    .all(|l| !first_leaves.contains(l))
            })
            .collect();
        let second = valid[pick(valid.len())?];
        self.edge_list(py, [first, second])
    }

    fn subgenome_count(&self, py: Python<'_>, n: &Bound<'_, PyAny>) -> PyResult<usize> {
        let u = self.require_uid(py, n, "Input node is not in the graph")?;
        self.subgenome_count_uid(py, u)
    }

    fn edges_downstream_of_node<'py>(&self, py: Python<'py>, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
        let u = self.require_uid(py, n, "Input node is not in the graph.")?;
        self.edge_list(py, self.edges_downstream(u))
    }

    fn edges_upstream_of_node<'py>(&self, py: Python<'py>, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
        let u = self.require_uid(py, n, "Input node is not in the graph.")?;
        self.edge_list(py, self.edges_upstream(u))
    }

    #[pyo3(signature = (downstream_node=None, delta=f64::INFINITY, start_node=None))]
    fn edges_to_subgenome_count<'py>(
        &self,
        py: Python<'py>,
        downstream_node: Option<Bound<'py, PyAny>>,
        delta: f64,
        start_node: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let buckets = self.subgenome_buckets(py, downstream_node.as_ref(), delta, start_node.as_ref())?;
        let d = PyDict::new(py);
        for (count, edges) in buckets {
            d.set_item(count, self.edge_list(py, edges)?)?;
        }
        Ok(d)
    }

    #[pyo3(signature = (downstream_node=None, delta=f64::INFINITY, start_node=None))]
    fn subgenome_ct_edges<'py>(
        &self,
        py: Python<'py>,
        downstream_node: Option<Bound<'py, PyAny>>,
        delta: f64,
        start_node: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let buckets = self.subgenome_buckets(py, downstream_node.as_ref(), delta, start_node.as_ref())?;
        let d = PyDict::new(py);
        for (count, edges) in buckets {
            for e in edges {
                d.set_item(self.edges[&e].obj.bind(py), count)?;
            }
        }
        Ok(d)
    }

    fn leaf_descendants_all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let map = self.leaf_descendants_all_map(py)?;
        let d = PyDict::new(py);
        for (u, leaves) in map {
            d.set_item(self.nodes[&u].obj.bind(py), self.node_set(py, leaves)?)?;
        }
        Ok(d)
    }

    #[pyo3(signature = (annotate=false))]
    fn newick(&self, py: Python<'_>, annotate: bool) -> PyResult<String> {
        self.newick_string(py, annotate)
    }

    fn is_acyclic(&self) -> bool {
        self.is_acyclic_impl()
    }

    #[pyo3(signature = (start_node=None, dfs=false, is_connected=false, accumulator=None, accumulated=None))]
    fn bfs_dfs<'py>(
        slf: &Bound<'py, Self>,
        start_node: Option<Bound<'py, PyAny>>,
        dfs: bool,
        is_connected: bool,
        accumulator: Option<Bound<'py, PyAny>>,
        accumulated: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyAny>)> {
        let py = slf.py();
        let (order, dist_map, objs, unvisited) = {
            let this = slf.borrow();
            let start = match &start_node {
                Some(n) => this.require_uid(py, n, "Start node is not in the graph.")?,
                None => this.root_uid(py)?,
            };
            let (order, dist) = this.bfs_order(start, dfs);
            let objs: FxHashMap<u64, Py<Node>> = order.iter().map(|&u| (u, this.nodes[&u].obj.clone_ref(py))).collect();
            let unvisited = this.nodes.keys().filter(|u| !dist.contains_key(*u)).count();
            (order, dist, objs, unvisited)
        };
        let dist = PyDict::new(py);
        for (u, d) in dist_map.iter() {
            dist.set_item(objs[u].bind(py), *d)?;
        }
        let mut acc = accumulated.unwrap_or_else(|| py.None().into_bound(py));
        if let Some(f) = accumulator {
            if !acc.is_none() {
                for u in &order {
                    acc = f.call1((objs[u].bind(py), acc))?;
                }
            }
        }
        if is_connected {
            let print = py.import("builtins")?.getattr("print")?;
            if unvisited != 0 {
                print.call1(("GRAPH HAS MORE THAN 1 CONNECTED COMPONENT",))?;
            } else {
                print.call1(("GRAPH IS FULLY CONNECTED",))?;
            }
        }
        Ok((dist, acc))
    }

    fn subnet<'py>(slf: &Bound<'py, Self>, retic_node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Network>> {
        let py = slf.py();
        let u = slf.borrow().require_uid(py, retic_node, "Input node is not in the graph.")?;
        Network::subnet_impl(slf, u)
    }

    fn copy<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, Network>, Bound<'py, PyDict>)> {
        let py = slf.py();
        let (new, mapping) = Network::copy_impl(slf)?;
        let d = PyDict::new(py);
        for (old, nw) in mapping {
            d.set_item(old.bind(py), nw.bind(py))?;
        }
        Ok((new, d))
    }

    #[pyo3(signature = (subtree, *, sibling_of=None, under=None))]
    fn graft_subtree(
        slf: &Bound<'_, Self>,
        subtree: &Bound<'_, Network>,
        sibling_of: Option<Bound<'_, PyAny>>,
        under: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<Node>> {
        let py = slf.py();
        if sibling_of.is_none() == under.is_none() {
            return Err(network_error(py, "Pass exactly one of sibling_of or under."));
        }
        let anchor = sibling_of.clone().or_else(|| under.clone()).unwrap();
        let anchor_uid = match slf.borrow().uid_of_any(&anchor) {
            Some(u) => u,
            None => return Err(network_error(py, "Graft anchor is not in the network.")),
        };
        let image = Network::absorb_subtree(slf, subtree)?;
        if under.is_some() {
            let edge = {
                let this = slf.borrow();
                Network::default_edge(py, this.node_obj(anchor_uid).clone_ref(py), image.clone_ref(py), None)?
            };
            slf.borrow_mut().add_edge(slf.as_any(), edge.bind(py))?;
            return Ok(image);
        }
        let parents = slf.borrow().parents(anchor_uid);
        if parents.is_empty() {
            let (label, anchor_obj) = {
                let this = slf.borrow();
                (this.fresh_label(py, "R"), this.node_obj(anchor_uid).clone_ref(py))
            };
            let new_root = Py::new(py, Node::make(py, label, false, None, None, None))?;
            let e1 = Network::default_edge(py, new_root.clone_ref(py), anchor_obj, None)?;
            let e2 = Network::default_edge(py, new_root.clone_ref(py), image.clone_ref(py), None)?;
            let mut this = slf.borrow_mut();
            this.add_node(slf.as_any(), new_root.bind(py))?;
            this.add_edge(slf.as_any(), e1.bind(py))?;
            this.add_edge(slf.as_any(), e2.bind(py))?;
            return Ok(image);
        }
        let parent = parents[0];
        let (euid, length, gamma, child_is_retic, parent_obj, anchor_obj) = {
            let this = slf.borrow();
            let euid = this.first_edge_between(py, parent, anchor_uid)?;
            let (l, g) = {
                let e = this.edges[&euid].obj.borrow(py);
                (e.length, e.gamma)
            };
            let child_is_retic = this.node_obj(anchor_uid).borrow(py).is_retic;
            let parent_obj = this.node_obj(parent).clone_ref(py);
            let anchor_obj = this.node_obj(anchor_uid).clone_ref(py);
            (euid, l, g, child_is_retic, parent_obj, anchor_obj)
        };
        slf.borrow_mut().remove_edge_uid(py, euid);
        let mid_label = slf.borrow().fresh_label(py, "I");
        let mid = Py::new(py, Node::make(py, mid_label, false, None, None, None))?;
        let upper = Network::default_edge(py, parent_obj, mid.clone_ref(py), length)?;
        let lower = Network::default_edge(py, mid.clone_ref(py), anchor_obj, length)?;
        if child_is_retic {
            if let Some(g) = gamma {
                if g != 0.0 {
                    Edge::set_gamma(lower.bind(py), g)?;
                }
            }
        }
        let link = Network::default_edge(py, mid.clone_ref(py), image.clone_ref(py), None)?;
        let mut this = slf.borrow_mut();
        this.add_node(slf.as_any(), mid.bind(py))?;
        this.add_edge(slf.as_any(), upper.bind(py))?;
        this.add_edge(slf.as_any(), lower.bind(py))?;
        this.add_edge(slf.as_any(), link.bind(py))?;
        Ok(image)
    }

    fn _fresh_label(&self, py: Python<'_>, prefix: &str) -> String {
        self.fresh_label(py, prefix)
    }

    fn _absorb_subtree(slf: &Bound<'_, Self>, subtree: &Bound<'_, Network>) -> PyResult<Py<Node>> {
        Network::absorb_subtree(slf, subtree)
    }

    fn to_networkx<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let nx = py.import("networkx")?;
        let g = nx.getattr("MultiDiGraph")?.call0()?;
        let labels: Vec<String> = self.nodes.values().map(|e| e.obj.borrow(py).name.clone()).collect();
        g.call_method1("add_nodes_from", (labels,))?;
        let pairs: Vec<(String, String)> = self
            .edges
            .values()
            .map(|e| {
                let eb = e.obj.borrow(py);
                let s = eb.src.borrow(py).name.clone();
                let d = eb.dest.borrow(py).name.clone();
                (s, d)
            })
            .collect();
        g.call_method1("add_edges_from", (pairs,))?;
        Ok(g)
    }

    fn rooted_triplet_distance(&self, py: Python<'_>, other: &Bound<'_, Network>) -> PyResult<f64> {
        let o = other.borrow();
        self.triplet_distance_impl(py, &o)
    }

    fn get_subtree_at<'py>(&self, py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PySet>> {
        match self.uid_of_any(node) {
            Some(u) => self.node_set(py, self.subtree_uids(u)),
            None => PySet::empty(py),
        }
    }

    fn dist_from_root(&self, py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<i64> {
        let root = self.root_uid(py)?;
        let (_, dist) = self.bfs_order(root, false);
        match self.uid_of_any(node).and_then(|u| dist.get(&u).copied()) {
            Some(d) => Ok(d),
            None => Err(PyKeyError::new_err(node.clone().unbind())),
        }
    }

    fn topological_order<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let order = self.topological_order_uids(py)?;
        self.node_list(py, order)
    }

    #[pyo3(signature = (node, use_time=true))]
    fn distance_from_root(&self, py: Python<'_>, node: &Bound<'_, PyAny>, use_time: bool) -> PyResult<f64> {
        let u = self.require_uid(py, node, "Node not found in network")?;
        self.distance_from_root_impl(py, u, use_time)
    }

    fn set_node_times_from_root(&self, py: Python<'_>) -> PyResult<()> {
        self.set_times_impl(py, false)
    }

    fn set_node_times_by_edge_count(&self, py: Python<'_>) -> PyResult<()> {
        self.set_times_impl(py, true)
    }

    // ── copy / pickle protocol ──

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::empty(py)
    }

    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let this = slf.borrow();
        let _ = py;
        let d = PyDict::new(py);
        let index: FxHashMap<u64, usize> = this.nodes.keys().enumerate().map(|(i, u)| (*u, i)).collect();
        d.set_item("nodes", this.node_list(py, this.nodes.keys().copied())?)?;
        d.set_item("edges", this.edge_list(py, this.edges.keys().copied())?)?;
        d.set_item("leaves", this.leaves.iter().map(|u| index[u]).collect::<Vec<usize>>())?;
        d.set_item("roots", this.roots.iter().map(|u| index[u]).collect::<Vec<usize>>())?;
        d.set_item("items", this.items.bind(py))?;
        d.set_item("uid", this.uid)?;
        d.set_item("unit", this.unit.bind(py))?;
        if let Ok(dict) = slf.getattr("__dict__") {
            d.set_item("__dict__", dict)?;
        }
        Ok(d)
    }

    fn __setstate__(slf: &Bound<'_, Self>, state: Bound<'_, PyDict>) -> PyResult<()> {
        let nodes: Vec<Bound<'_, Node>> = state
            .get_item("nodes")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_default();
        let edges: Vec<Bound<'_, Edge>> = state
            .get_item("edges")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_default();
        let leaves: Vec<usize> = state.get_item("leaves")?.map(|v| v.extract()).transpose()?.unwrap_or_default();
        let roots: Vec<usize> = state.get_item("roots")?.map(|v| v.extract()).transpose()?.unwrap_or_default();
        {
            let mut this = slf.borrow_mut();
            if let Some(items) = state.get_item("items")? {
                this.items = items.cast_into::<PyDict>()?.unbind();
            }
            if let Some(uid) = state.get_item("uid")? {
                this.uid = uid.extract()?;
            }
            if let Some(unit) = state.get_item("unit")? {
                this.unit = unit.unbind();
            }
            let mut uids = Vec::with_capacity(nodes.len());
            for n in &nodes {
                this.add_node(slf.as_any(), n)?;
                uids.push(n.borrow().uid);
            }
            for e in &edges {
                this.add_edge(slf.as_any(), e)?;
            }
            this.leaves = leaves.iter().filter_map(|&i| uids.get(i).copied()).collect();
            this.roots = roots.iter().filter_map(|&i| uids.get(i).copied()).collect();
        }
        if let Some(dict) = state.get_item("__dict__")? {
            if let Ok(target) = slf.getattr("__dict__") {
                target.call_method1("update", (dict,))?;
            }
        }
        Ok(())
    }

    fn __deepcopy__<'py>(slf: &Bound<'py, Self>, memo: Bound<'py, PyDict>) -> PyResult<Bound<'py, Network>> {
        let py = slf.py();
        let (unit, node_objs, edge_objs, leaves, roots, items, uid) = {
            let this = slf.borrow();
            (
                this.unit.clone_ref(py),
                this.nodes.iter().map(|(u, e)| (*u, e.obj.clone_ref(py))).collect::<Vec<_>>(),
                this.edges.values().map(|e| e.obj.clone_ref(py)).collect::<Vec<_>>(),
                this.leaves.iter().copied().collect::<Vec<_>>(),
                this.roots.iter().copied().collect::<Vec<_>>(),
                this.items.clone_ref(py),
                this.uid,
            )
        };
        let new = Network::new_object(py, Some(unit.bind(py)))?;
        memo.set_item(slf.as_any().as_ptr() as usize, new.as_any())?;
        let mut by_uid: FxHashMap<u64, u64> = FxHashMap::default();
        {
            let mut n = new.borrow_mut();
            for (old_uid, obj) in node_objs {
                let copy = deepcopy(py, obj.bind(py).as_any(), &memo)?.cast_into::<Node>()?;
                let new_uid = copy.borrow().uid;
                n.add_node(new.as_any(), &copy)?;
                by_uid.insert(old_uid, new_uid);
            }
            for obj in edge_objs {
                let copy = deepcopy(py, obj.bind(py).as_any(), &memo)?.cast_into::<Edge>()?;
                n.add_edge(new.as_any(), &copy)?;
            }
            n.items = deepcopy(py, items.bind(py).as_any(), &memo)?.cast_into::<PyDict>()?.unbind();
            n.uid = uid;
            n.leaves = leaves.iter().filter_map(|u| by_uid.get(u).copied()).collect();
            n.roots = roots.iter().filter_map(|u| by_uid.get(u).copied()).collect();
        }
        if let Ok(src_dict) = slf.getattr("__dict__") {
            if let Ok(dst_dict) = new.getattr("__dict__") {
                let copied = deepcopy(py, &src_dict, &memo)?;
                dst_dict.call_method1("update", (copied,))?;
            }
        }
        Ok(new)
    }
}

impl Network {
    /// Shared body of ``edges_to_subgenome_count`` / ``subgenome_ct_edges``:
    /// ``(count, [edge ids])`` buckets in first-seen order.
    fn subgenome_buckets(
        &self,
        py: Python<'_>,
        downstream_node: Option<&Bound<'_, PyAny>>,
        delta: f64,
        start_node: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<(usize, Vec<u64>)>> {
        let start = match start_node {
            Some(n) => self.require_uid(py, n, "Start node is not in the graph.")?,
            None => self.root_uid(py)?,
        };
        let mut counts: IndexMap<u64, usize, FxBuildHasher> =
            self.edges.keys().map(|&e| (e, 0usize)).collect();
        let mut stack = vec![start];
        while let Some(cur) = stack.pop() {
            for nb in self.children(cur) {
                let e = self.first_edge_between(py, cur, nb)?;
                *counts.get_mut(&e).unwrap() += 1;
                stack.push(nb);
            }
        }
        let mut partition: IndexMap<usize, Vec<u64>, FxBuildHasher> = IndexMap::default();
        for (e, c) in counts.iter() {
            partition.entry(*c).or_default().push(*e);
        }
        let excluded: FxHashSet<u64> = match downstream_node {
            Some(n) => {
                let u = self.require_uid(py, n, "Input node is not in the graph.")?;
                self.edges_downstream(u).into_iter().collect()
            }
            None => FxHashSet::default(),
        };
        let mut out = Vec::new();
        for (c, edges) in partition {
            if (c as f64) > delta {
                continue;
            }
            let kept: Vec<u64> = edges.into_iter().filter(|e| !excluded.contains(e)).collect();
            if downstream_node.is_some() && kept.is_empty() {
                continue;
            }
            out.push((c, kept));
        }
        Ok(out)
    }
}
