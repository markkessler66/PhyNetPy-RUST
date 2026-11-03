//! GraphUtils: topology analysis, manipulation and comparison of
//! ``Network`` objects.  The Python module ``phynetpy.GraphUtils`` is a
//! facade over the ``_gu_*`` functions registered here.
//!
//! Every function follows its Python reference step for step, including
//! the order in which nodes and edges are visited, copied and removed: the
//! ``Network`` containers are insertion-ordered (removal swaps the last
//! element into the hole), so the order of mutations decides the order of
//! ``V()``, ``E()`` and of the children in a newick string.

use std::borrow::Borrow;
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeSet, BinaryHeap, VecDeque};

use indexmap::{IndexMap, IndexSet};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyString, PyTuple};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use super::edge::Edge;
use super::network::Network;
use super::node::Node;
use crate::util::{network_error, warn};

/// A leaf label, shared between the clusters that contain it.
type Label = std::rc::Rc<str>;
/// ``frozenset[str]``: a cluster of leaf labels.
type LabelSet = BTreeSet<Label>;
/// ``frozenset[frozenset[str]]``: the cluster set identifying a tree topology.
type TopologyKey = BTreeSet<LabelSet>;
/// A canonical tripartition ``(desc(c1), desc(c2), rest)``.
type Tripartition = (LabelSet, LabelSet, LabelSet);

const DEFAULT_TAG: &str = "no tag assigned yet";

// ── Small helpers ──────────────────────────────────────────────────────

/// Python strings for labels, created once per distinct label.
struct Strs<'py> {
    py: Python<'py>,
    cache: FxHashMap<Label, Bound<'py, PyString>>,
}

impl<'py> Strs<'py> {
    fn new(py: Python<'py>) -> Self {
        Strs { py, cache: FxHashMap::default() }
    }

    fn get(&mut self, label: &Label) -> Bound<'py, PyString> {
        self.cache
            .entry(label.clone())
            .or_insert_with(|| PyString::new(self.py, label))
            .clone()
    }

    fn frozenset(&mut self, labels: &LabelSet) -> PyResult<Bound<'py, PyFrozenSet>> {
        let items: Vec<Bound<'py, PyString>> = labels.iter().map(|l| self.get(l)).collect();
        PyFrozenSet::new(self.py, items)
    }

    fn frozenset_of_sets(&mut self, key: &TopologyKey) -> PyResult<Bound<'py, PyFrozenSet>> {
        let items: Vec<Bound<'py, PyFrozenSet>> = key.iter().map(|s| self.frozenset(s)).collect::<PyResult<_>>()?;
        PyFrozenSet::new(self.py, items)
    }

    fn tuple(&mut self, sets: &[LabelSet]) -> PyResult<Bound<'py, PyTuple>> {
        let items: Vec<Bound<'py, PyFrozenSet>> = sets.iter().map(|s| self.frozenset(s)).collect::<PyResult<_>>()?;
        PyTuple::new(self.py, items)
    }
}

fn frozenset_of_labels<'py>(py: Python<'py>, labels: &LabelSet) -> PyResult<Bound<'py, PyFrozenSet>> {
    Strs::new(py).frozenset(labels)
}

fn set_of_label_sets<'py, I>(py: Python<'py>, sets: I) -> PyResult<Bound<'py, PySet>>
where
    I: IntoIterator,
    I::Item: std::borrow::Borrow<LabelSet>,
{
    let mut strs = Strs::new(py);
    let out = PySet::empty(py)?;
    for s in sets {
        out.add(strs.frozenset(s.borrow())?)?;
    }
    Ok(out)
}

/// Labels of a network's nodes, each read once per call.
struct Labels<'a, 'py> {
    py: Python<'py>,
    net: &'a Network,
    cache: FxHashMap<u64, Label>,
}

impl<'a, 'py> Labels<'a, 'py> {
    fn new(py: Python<'py>, net: &'a Network) -> Self {
        Labels { py, net, cache: FxHashMap::default() }
    }

    fn get(&mut self, uid: u64) -> Label {
        let (py, net) = (self.py, self.net);
        self.cache
            .entry(uid)
            .or_insert_with(|| Label::from(net.label(py, uid).as_str()))
            .clone()
    }

    fn set(&mut self, uids: impl IntoIterator<Item = u64>) -> LabelSet {
        uids.into_iter().map(|u| self.get(u)).collect()
    }
}

/// ``repr(sorted(labels))`` exactly as Python prints it.
fn sorted_list_repr(py: Python<'_>, labels: &LabelSet) -> PyResult<String> {
    let list = PyList::new(py, labels.iter().map(|s| &**s))?;
    Ok(list.repr()?.to_string())
}

/// Every string of a Python iterable.
fn collect_strings(items: &Bound<'_, PyAny>) -> PyResult<FxHashSet<String>> {
    let mut out = FxHashSet::default();
    for item in items.try_iter()? {
        out.insert(item?.extract::<String>()?);
    }
    Ok(out)
}

/// Sort key for a gamma: ``-0.0`` and ``0.0`` compare equal, as in Python.
fn gamma_key(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x
    }
}

fn new_edge(py: Python<'_>, src: Py<Node>, dest: Py<Node>, gamma: f64, tag: &str) -> PyResult<Py<Edge>> {
    Py::new(
        py,
        Edge::make(py, src, dest, Some(1.0), Some(gamma), Some(0.0), Some(tag.to_string())),
    )
}

/// ``Network.add_uid_node()`` with no argument.
fn add_uid_node(py: Python<'_>, net: &Bound<'_, Network>) -> PyResult<Py<Node>> {
    let name = format!("UID_{}", net.borrow().uid);
    let node = Py::new(py, Node::make(py, name, false, None, None, None))?;
    let mut this = net.borrow_mut();
    this.add_node(net.as_any(), node.bind(py))?;
    this.uid += 1;
    Ok(node)
}

/// ``Network.remove_edge(edge)`` for an Edge object: a no-op when absent.
fn remove_edge_obj(py: Python<'_>, net: &Bound<'_, Network>, edge: &Bound<'_, Edge>) {
    let uid = edge.borrow().uid;
    let mut this = net.borrow_mut();
    if this.edges.contains_key(&uid) {
        this.remove_edge_uid(py, uid);
    }
}

/// The argument checks of ``Network.mrca``: Node objects or labels, each
/// resolved to a node of ``net``, duplicates dropped.
fn collect_target_uids(py: Python<'_>, net: &Network, items: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    let mut out = Vec::new();
    let mut seen: FxHashSet<u64> = FxHashSet::default();
    for item in items.try_iter()? {
        let item = item?;
        let uid = if item.is_exact_instance_of::<PyString>() {
            let name: String = item.extract()?;
            match net.names.get(&name) {
                Some(&u) => u,
                None => return Err(network_error(py, "A node in 'set_of_nodes' is not in the graph")),
            }
        } else if item.is_exact_instance_of::<Node>() {
            match net.uid_of_any(&item) {
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
            out.push(uid);
        }
    }
    Ok(out)
}

/// The iteration order CPython gives a ``set`` built from an empty set by
/// adding distinct objects with these hashes, in this order.
///
/// The reference implementation iterated Python sets in three places
/// (``Network.in_edges`` in :func:`dominant_tree` and the displayed-tree
/// enumeration, and the blob node sets in :func:`tree_of_blobs`).  Node and
/// Edge hashes are process-independent, so those orders are deterministic
/// and show in the results (which parallel edge is kept when gammas tie,
/// the ``V()`` order of a blob).  This reproduces the table layout of
/// ``Objects/setobject.c`` (CPython 3.9-3.13: open addressing, nine linear
/// probes, perturbation shift 5, growth at three fifths full) so the port
/// returns exactly what the reference did.
fn cpython_set_order(hashes: &[u64]) -> Vec<usize> {
    const LINEAR_PROBES: usize = 9;
    const PERTURB_SHIFT: u32 = 5;
    const MINSIZE: usize = 8;

    fn next_probe(i: usize, perturb: u64, mask: usize) -> usize {
        (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb as usize)) & mask
    }

    fn insert_clean(table: &mut [Option<(u64, usize)>], hash: u64, idx: usize) {
        let mask = table.len() - 1;
        let mut perturb = hash;
        let mut i = (hash as usize) & mask;
        loop {
            if table[i].is_none() {
                table[i] = Some((hash, idx));
                return;
            }
            if i + LINEAR_PROBES <= mask {
                for j in 1..=LINEAR_PROBES {
                    if table[i + j].is_none() {
                        table[i + j] = Some((hash, idx));
                        return;
                    }
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = next_probe(i, perturb, mask);
        }
    }

    let mut table: Vec<Option<(u64, usize)>> = vec![None; MINSIZE];
    let mut used = 0usize;
    for (idx, &hash) in hashes.iter().enumerate() {
        let mask = table.len() - 1;
        let mut perturb = hash;
        let mut i = (hash as usize) & mask;
        let slot = 'probe: loop {
            let probes = if i + LINEAR_PROBES <= mask { LINEAR_PROBES } else { 0 };
            for j in 0..=probes {
                if table[i + j].is_none() {
                    break 'probe i + j;
                }
                // Occupied by a distinct object (keys never compare equal
                // here, and an insert-only table has no dummies): go on.
            }
            perturb >>= PERTURB_SHIFT;
            i = next_probe(i, perturb, mask);
        };
        table[slot] = Some((hash, idx));
        used += 1;
        // fill == used without deletions.
        if used * 5 >= mask * 3 {
            let minused = if used > 50000 { used * 2 } else { used * 4 };
            let mut newsize = MINSIZE;
            while newsize <= minused {
                newsize <<= 1;
            }
            let old = std::mem::replace(&mut table, vec![None; newsize]);
            for (hash, idx) in old.into_iter().flatten() {
                insert_clean(&mut table, hash, idx);
            }
        }
    }
    table.into_iter().flatten().map(|(_, idx)| idx).collect()
}

/// The in-edges of ``uid`` in the order the reference saw them: the
/// Python set ``Network.in_edges`` returns, built in in-edge order.
pub(crate) fn in_edges_in_set_order(py: Python<'_>, net: &Network, uid: u64) -> Vec<u64> {
    let edges = net.in_edge_uids(uid);
    let hashes: Vec<u64> = edges.iter().map(|e| net.edges[e].obj.borrow(py).hash).collect();
    cpython_set_order(&hashes).into_iter().map(|i| edges[i]).collect()
}

fn count_retics(py: Python<'_>, net: &Network) -> usize {
    net.nodes.values().filter(|e| e.obj.borrow(py).is_retic).count()
}

fn leaf_label_set(py: Python<'_>, net: &Network) -> LabelSet {
    Labels::new(py, net).set(net.leaf_uids())
}

/// ``_leaf_labels_below``: every node reachable from the root mapped to the
/// labels of the leaves below it, in ``leaf_descendants_all`` order.
fn labels_below(py: Python<'_>, net: &Network) -> PyResult<IndexMap<u64, LabelSet, FxBuildHasher>> {
    let map = net.leaf_descendants_all_map(py)?;
    let mut labels = Labels::new(py, net);
    let mut out: IndexMap<u64, LabelSet, FxBuildHasher> = IndexMap::default();
    for (u, leaves) in map.iter() {
        out.insert(*u, labels.set(leaves.iter().copied()));
    }
    Ok(out)
}

/// ``_hardwired_clusters_by_label``.
fn hardwired_clusters(py: Python<'_>, net: &Network, include_trivial: bool) -> PyResult<FxHashSet<LabelSet>> {
    let mut out: FxHashSet<LabelSet> = FxHashSet::default();
    for labels in labels_below(py, net)?.into_values() {
        if labels.len() > 1 || (include_trivial && labels.len() == 1) {
            out.insert(labels);
        }
    }
    Ok(out)
}

/// ``_tree_topology_key``.
fn topology_key(py: Python<'_>, tree: &Network) -> PyResult<TopologyKey> {
    Ok(hardwired_clusters(py, tree, true)?.into_iter().collect())
}

fn require_same_leaves(py: Python<'_>, net1: &Network, net2: &Network, caller: &str) -> PyResult<LabelSet> {
    let (l1, l2) = (leaf_label_set(py, net1), leaf_label_set(py, net2));
    if l1 != l2 {
        return Err(network_error(
            py,
            format!(
                "{caller} requires identical leaf label sets. Got {} vs {}",
                sorted_list_repr(py, &l1)?,
                sorted_list_repr(py, &l2)?
            ),
        ));
    }
    Ok(l1)
}

/// ``_symmetric_difference_distance`` on Rust feature sets: an ``int``, or a
/// ``float`` in [0, 1] when normalised.
fn symmetric_difference<T>(py: Python<'_>, s1: &FxHashSet<T>, s2: &FxHashSet<T>, normalize: bool) -> PyResult<Py<PyAny>>
where
    T: Eq + std::hash::Hash,
{
    let sym = s1.symmetric_difference(s2).count();
    if normalize {
        let union = s1.union(s2).count();
        let value = if union > 0 { sym as f64 / union as f64 } else { 0.0 };
        Ok(value.into_pyobject(py)?.into_any().unbind())
    } else {
        Ok(sym.into_pyobject(py)?.into_any().unbind())
    }
}

// ── Clusters ───────────────────────────────────────────────────────────

/// ``get_all_clusters``: clusters as frozensets of this network's nodes.
#[pyfunction]
#[pyo3(name = "_gu_get_all_clusters", signature = (net, include_trivial=false))]
fn get_all_clusters<'py>(py: Python<'py>, net: &Bound<'py, Network>, include_trivial: bool) -> PyResult<Bound<'py, PySet>> {
    let this = net.borrow();
    let root = this.root_uid(py)?;
    let map = this.leaf_descendants_all_map(py)?;
    let out = PySet::empty(py)?;
    for (uid, below) in map.iter() {
        if *uid != root && below.len() > 1 {
            out.add(PyFrozenSet::new(py, below.iter().map(|u| this.node_obj(*u).bind(py)))?)?;
        }
    }
    if include_trivial {
        for leaf in this.leaf_uids() {
            out.add(PyFrozenSet::new(py, [this.node_obj(leaf).bind(py)])?)?;
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_gu_leaf_label_set")]
fn leaf_label_set_py<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PyFrozenSet>> {
    frozenset_of_labels(py, &leaf_label_set(py, &net.borrow()))
}

#[pyfunction]
#[pyo3(name = "_gu_require_same_leaves")]
fn require_same_leaves_py<'py>(
    py: Python<'py>,
    net1: &Bound<'py, Network>,
    net2: &Bound<'py, Network>,
    caller: &str,
) -> PyResult<Bound<'py, PyFrozenSet>> {
    let labels = require_same_leaves(py, &net1.borrow(), &net2.borrow(), caller)?;
    frozenset_of_labels(py, &labels)
}

#[pyfunction]
#[pyo3(name = "_gu_hardwired_clusters_by_label", signature = (net, include_trivial=false))]
fn hardwired_clusters_by_label<'py>(py: Python<'py>, net: &Bound<'py, Network>, include_trivial: bool) -> PyResult<Bound<'py, PySet>> {
    set_of_label_sets(py, hardwired_clusters(py, &net.borrow(), include_trivial)?)
}

#[pyfunction]
#[pyo3(name = "_gu_leaf_labels_below")]
fn leaf_labels_below<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PyDict>> {
    let this = net.borrow();
    let out = PyDict::new(py);
    let mut strs = Strs::new(py);
    for (uid, labels) in labels_below(py, &this)? {
        out.set_item(this.node_obj(uid).bind(py), strs.frozenset(&labels)?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_gu_network_clusters")]
fn network_clusters<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PySet>> {
    let this = net.borrow();
    let leaf_count = this.leaf_uids().len();
    let clusters = hardwired_clusters(py, &this, false)?;
    set_of_label_sets(py, clusters.iter().filter(|c| c.len() < leaf_count))
}

fn softwired_clusters<'py>(py: Python<'py>, net: &Bound<'py, Network>, include_trivial: bool) -> PyResult<FxHashSet<LabelSet>> {
    if count_retics(py, &net.borrow()) == 0 {
        return hardwired_clusters(py, &net.borrow(), include_trivial);
    }
    let mut out: FxHashSet<LabelSet> = FxHashSet::default();
    for tree in displayed_trees(py, net, true)? {
        out.extend(hardwired_clusters(py, &tree.borrow(), include_trivial)?);
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_gu_softwired_clusters_by_label", signature = (net, include_trivial=false))]
fn softwired_clusters_by_label<'py>(py: Python<'py>, net: &Bound<'py, Network>, include_trivial: bool) -> PyResult<Bound<'py, PySet>> {
    set_of_label_sets(py, softwired_clusters(py, net, include_trivial)?)
}

#[pyfunction]
#[pyo3(name = "_gu_tree_topology_key")]
fn tree_topology_key<'py>(py: Python<'py>, tree: &Bound<'py, Network>) -> PyResult<Bound<'py, PyFrozenSet>> {
    Strs::new(py).frozenset_of_sets(&topology_key(py, &tree.borrow())?)
}

// ── Induced subnetworks ────────────────────────────────────────────────

/// ``_induced_subnetwork``: the part of ``net`` at or below MRCA(targets)
/// that leads to a target, copied into a fresh network.
fn induced_subnetwork_impl<'py>(py: Python<'py>, net: &Bound<'py, Network>, targets: &[u64]) -> PyResult<Bound<'py, Network>> {
    let this = net.borrow();
    let sub_root = this.mrca_uids(py, targets)?;
    let target_set: FxHashSet<u64> = targets.iter().copied().collect();
    let sub = Network::new_object(py, None)?;
    let mut old_new: FxHashMap<u64, Py<Node>> = FxHashMap::default();
    let root_copy = this.node_obj(sub_root).borrow(py).copy(py)?;
    sub.borrow_mut().add_node(sub.as_any(), root_copy.bind(py))?;
    old_new.insert(sub_root, root_copy);
    let mut queue: VecDeque<u64> = VecDeque::new();
    queue.push_back(sub_root);
    while let Some(cur) = queue.pop_front() {
        for child in this.children(cur) {
            if !this.leaf_descendants_uid(child).iter().any(|l| target_set.contains(l)) {
                continue;
            }
            if !old_new.contains_key(&child) {
                let copy = this.node_obj(child).borrow(py).copy(py)?;
                sub.borrow_mut().add_node(sub.as_any(), copy.bind(py))?;
                old_new.insert(child, copy);
                queue.push_back(child);
            }
            let old_edge = this.first_edge_between(py, cur, child)?;
            let (gamma, length) = {
                let e = this.edges[&old_edge].obj.borrow(py);
                (e.gamma, e.length)
            };
            let edge = new_edge(py, old_new[&cur].clone_ref(py), old_new[&child].clone_ref(py), 0.0, DEFAULT_TAG)?;
            match gamma {
                Some(g) => Edge::set_gamma(edge.bind(py), g)?,
                None => {
                    return Err(PyTypeError::new_err(
                        "argument 'gamma': must be real number, not NoneType",
                    ))
                }
            }
            Edge::set_length(edge.bind(py), length, false, false)?;
            sub.borrow_mut().add_edge(sub.as_any(), edge.bind(py))?;
        }
    }
    Ok(sub)
}

#[pyfunction]
#[pyo3(name = "_gu_induced_subnetwork")]
fn induced_subnetwork<'py>(py: Python<'py>, net: &Bound<'py, Network>, targets: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Network>> {
    let uids = collect_target_uids(py, &net.borrow(), targets)?;
    induced_subnetwork_impl(py, net, &uids)
}

#[pyfunction]
#[pyo3(name = "_gu_subnet_given_leaves")]
fn subnet_given_leaves<'py>(py: Python<'py>, net: &Bound<'py, Network>, leaf_set: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Network>> {
    let uids = collect_target_uids(py, &net.borrow(), leaf_set)?;
    let sub = induced_subnetwork_impl(py, net, &uids)?;
    if sub.borrow().nodes.len() > 1 {
        Network::clean_impl(&sub, [true, true, true])?;
    }
    Ok(sub)
}

#[pyfunction]
#[pyo3(name = "_gu_induced_subnetwork_by_taxa")]
fn induced_subnetwork_by_taxa<'py>(py: Python<'py>, net: &Bound<'py, Network>, taxa: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Network>> {
    let targets = {
        let this = net.borrow();
        let mut out = Vec::new();
        let mut seen: FxHashSet<u64> = FxHashSet::default();
        for item in taxa.try_iter()? {
            let item = item?;
            let uid = item
                .extract::<String>()
                .ok()
                .and_then(|name| this.names.get(&name).copied());
            match uid {
                Some(u) => {
                    if seen.insert(u) {
                        out.push(u);
                    }
                }
                None => {
                    return Err(network_error(
                        py,
                        format!("Leaf '{}' not found in network", item.str()?),
                    ))
                }
            }
        }
        out
    };
    let sub = induced_subnetwork_impl(py, net, &targets)?;
    Network::clean_impl(&sub, [false, false, true])?;
    Ok(sub)
}

// ── Displayed trees ────────────────────────────────────────────────────

/// ``_suppress_childless_nodes``: delete out-degree-0 nodes that stand for
/// no taxon, cascading to parents left childless.
fn suppress_childless_nodes_impl(py: Python<'_>, tree: &Bound<'_, Network>, taxa: &FxHashSet<String>) -> PyResult<()> {
    let mut childless: Vec<u64> = {
        let this = tree.borrow();
        this.nodes
            .keys()
            .copied()
            .filter(|&u| this.out_deg(u) == 0 && !taxa.contains(&this.label(py, u)))
            .collect()
    };
    let mut pending: FxHashSet<u64> = childless.iter().copied().collect();
    while let Some(node) = childless.pop() {
        pending.remove(&node);
        let parents = tree.borrow().parents(node);
        tree.borrow_mut().remove_node_uid(py, node);
        let this = tree.borrow();
        for parent in parents {
            if !pending.contains(&parent) && this.out_deg(parent) == 0 && !taxa.contains(&this.label(py, parent)) {
                pending.insert(parent);
                childless.push(parent);
            }
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "_gu_suppress_childless_nodes")]
fn suppress_childless_nodes(py: Python<'_>, tree: &Bound<'_, Network>, taxa: &Bound<'_, PyAny>) -> PyResult<()> {
    let taxa = collect_strings(taxa)?;
    suppress_childless_nodes_impl(py, tree, &taxa)
}

/// ``_displayed_tree``: copy every node and every edge outside ``dropped``,
/// then reduce the result to a tree on ``taxa``.
fn displayed_tree_impl<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
    dropped: &FxHashSet<u64>,
    taxa: &FxHashSet<String>,
) -> PyResult<Bound<'py, Network>> {
    let tree = {
        let this = net.borrow();
        let unit = this.unit.clone_ref(py);
        let tree = Network::new_object(py, Some(unit.bind(py)))?;
        let mut old_new: FxHashMap<u64, Py<Node>> = FxHashMap::default();
        {
            let mut t = tree.borrow_mut();
            for (uid, entry) in this.nodes.iter() {
                let copy = entry.obj.borrow(py).copy(py)?;
                // set_is_reticulation(False): the copy has no owner yet, so
                // nothing is journaled.
                copy.borrow_mut(py).is_retic = false;
                t.add_node(tree.as_any(), copy.bind(py))?;
                old_new.insert(*uid, copy);
            }
            for (euid, entry) in this.edges.iter() {
                if dropped.contains(euid) {
                    continue;
                }
                let copy = entry.obj.borrow(py).copy(
                    py,
                    Some(old_new[&entry.su].clone_ref(py)),
                    Some(old_new[&entry.du].clone_ref(py)),
                )?;
                t.add_edge(tree.as_any(), copy.bind(py))?;
            }
            t.uid = this.uid;
        }
        tree
    };
    suppress_childless_nodes_impl(py, &tree, taxa)?;
    Network::clean_impl(&tree, [true, true, true])?;
    Ok(tree)
}

#[pyfunction]
#[pyo3(name = "_gu_displayed_tree")]
fn displayed_tree<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
    dropped: &Bound<'py, PyAny>,
    taxa: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, Network>> {
    let mut dropped_uids: FxHashSet<u64> = FxHashSet::default();
    for item in dropped.try_iter()? {
        let item = item?;
        dropped_uids.insert(item.cast::<Edge>()?.borrow().uid);
    }
    let taxa = collect_strings(taxa)?;
    displayed_tree_impl(py, net, &dropped_uids, &taxa)
}

/// In-edges of a reticulation by gamma, highest first, then by source label.
fn sorted_in_edges(py: Python<'_>, net: &Network, uid: u64) -> Vec<u64> {
    let mut edges: Vec<(f64, String, u64)> = in_edges_in_set_order(py, net, uid)
        .into_iter()
        .map(|e| {
            let eb = net.edges[&e].obj.borrow(py);
            let src_label = eb.src.borrow(py).name.clone();
            (gamma_key(-eb.gamma.unwrap_or(0.0)), src_label, e)
        })
        .collect();
    edges.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    edges.into_iter().map(|t| t.2).collect()
}

/// ``_displayed_trees_with_probs``: one tree per hybrid-edge choice, with
/// the product of the kept gammas; reticulations by label, in-edges by
/// gamma, the last reticulation varying fastest.
fn displayed_trees_with_probs_impl<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Vec<(Bound<'py, Network>, f64)>> {
    let (retics, options, taxa) = {
        let this = net.borrow();
        let mut retics: Vec<(String, u64)> = this
            .nodes
            .iter()
            .filter(|(_, e)| e.obj.borrow(py).is_retic)
            .map(|(u, e)| (e.obj.borrow(py).name.clone(), *u))
            .collect();
        retics.sort_by(|a, b| a.0.cmp(&b.0));
        let options: Vec<Vec<u64>> = retics.iter().map(|(_, u)| sorted_in_edges(py, &this, *u)).collect();
        let taxa: FxHashSet<String> = this.leaf_uids().into_iter().map(|u| this.label(py, u)).collect();
        (retics, options, taxa)
    };
    if retics.is_empty() {
        let (copy, _) = Network::copy_impl(net)?;
        return Ok(vec![(copy, 1.0)]);
    }
    if options.iter().any(|o| o.is_empty()) {
        return Ok(Vec::new());
    }
    let mut index = vec![0usize; retics.len()];
    let mut results = Vec::new();
    loop {
        let (dropped, prob) = {
            let this = net.borrow();
            let mut prob = 1.0_f64;
            let mut dropped: FxHashSet<u64> = FxHashSet::default();
            for (r, (_, retic)) in retics.iter().enumerate() {
                let kept = options[r][index[r]];
                let gamma = this.edges[&kept].obj.borrow(py).gamma;
                match gamma {
                    Some(g) if g > 0.0 => prob *= g,
                    _ => {
                        let k = this.in_deg(*retic);
                        prob *= if k > 0 { 1.0 / k as f64 } else { 1.0 };
                    }
                }
                for e in this.in_edge_uids(*retic) {
                    if e != kept {
                        dropped.insert(e);
                    }
                }
            }
            (dropped, prob)
        };
        results.push((displayed_tree_impl(py, net, &dropped, &taxa)?, prob));
        let mut pos = retics.len();
        loop {
            if pos == 0 {
                return Ok(results);
            }
            pos -= 1;
            index[pos] += 1;
            if index[pos] < options[pos].len() {
                break;
            }
            index[pos] = 0;
        }
    }
}

#[pyfunction]
#[pyo3(name = "_gu_displayed_trees_with_probs")]
fn displayed_trees_with_probs<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for (tree, prob) in displayed_trees_with_probs_impl(py, net)? {
        out.append((tree, prob))?;
    }
    Ok(out)
}

/// ``get_displayed_trees``: the displayed trees, one per distinct topology
/// when ``unique`` (first occurrence kept).
fn displayed_trees<'py>(py: Python<'py>, net: &Bound<'py, Network>, unique: bool) -> PyResult<Vec<Bound<'py, Network>>> {
    let trees = displayed_trees_with_probs_impl(py, net)?;
    if !unique {
        return Ok(trees.into_iter().map(|(t, _)| t).collect());
    }
    let mut seen: FxHashSet<TopologyKey> = FxHashSet::default();
    let mut distinct = Vec::new();
    for (tree, _) in trees {
        let key = topology_key(py, &tree.borrow())?;
        if seen.insert(key) {
            distinct.push(tree);
        }
    }
    Ok(distinct)
}

#[pyfunction]
#[pyo3(name = "_gu_get_displayed_trees", signature = (net, unique=true))]
fn get_displayed_trees<'py>(py: Python<'py>, net: &Bound<'py, Network>, unique: bool) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, displayed_trees(py, net, unique)?)
}

#[pyfunction]
#[pyo3(name = "_gu_dominant_tree")]
fn dominant_tree<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, Network>> {
    let dom = {
        let this = net.borrow();
        let dom = Network::new_object(py, None)?;
        // Only the maximum-gamma in-edge of each reticulation survives.
        let mut removed: FxHashSet<u64> = FxHashSet::default();
        for (uid, entry) in this.nodes.iter() {
            if !entry.obj.borrow(py).is_retic {
                continue;
            }
            let mut edges: Vec<(Option<f64>, u64)> = in_edges_in_set_order(py, &this, *uid)
                .into_iter()
                .map(|e| (this.edges[&e].obj.borrow(py).gamma, e))
                .collect();
            if edges.len() > 1 && edges.iter().any(|(g, _)| g.is_none()) {
                return Err(PyTypeError::new_err(
                    "'<' not supported between instances of 'NoneType' and 'float'",
                ));
            }
            edges.sort_by(|a, b| gamma_key(a.0.unwrap_or(0.0)).total_cmp(&gamma_key(b.0.unwrap_or(0.0))));
            let keep = edges.len().saturating_sub(1);
            removed.extend(edges.iter().take(keep).map(|(_, e)| *e));
        }
        let mut old_new: FxHashMap<u64, Py<Node>> = FxHashMap::default();
        let mut d = dom.borrow_mut();
        for (uid, entry) in this.nodes.iter() {
            let copy = entry.obj.borrow(py).copy(py)?;
            d.add_node(dom.as_any(), copy.bind(py))?;
            old_new.insert(*uid, copy);
        }
        for (euid, entry) in this.edges.iter() {
            if removed.contains(euid) {
                continue;
            }
            let copy = entry.obj.borrow(py).copy(
                py,
                Some(old_new[&entry.su].clone_ref(py)),
                Some(old_new[&entry.du].clone_ref(py)),
            )?;
            d.add_edge(dom.as_any(), copy.bind(py))?;
        }
        drop(d);
        dom
    };
    Network::clean_impl(&dom, [true, true, true])?;
    Ok(dom)
}

#[pyfunction]
#[pyo3(name = "_gu_count_displayed_trees", signature = (net, exact=false))]
fn count_displayed_trees<'py>(py: Python<'py>, net: &Bound<'py, Network>, exact: bool) -> PyResult<Py<PyAny>> {
    if exact {
        let n = displayed_trees(py, net, true)?.len();
        return Ok(n.into_pyobject(py)?.into_any().unbind());
    }
    let this = net.borrow();
    // A Python int: the product can exceed 64 bits.
    let mut prod = 1usize.into_pyobject(py)?.into_any();
    for (uid, entry) in this.nodes.iter() {
        if entry.obj.borrow(py).is_retic {
            prod = prod.mul(this.in_deg(*uid).max(1))?;
        }
    }
    Ok(prod.unbind())
}

// ── Reticulations ──────────────────────────────────────────────────────

/// ``_ancestor_depths``: every ancestor of ``start`` (inclusive) with its
/// minimum edge distance, in discovery order.
fn ancestor_depths(net: &Network, start: u64) -> IndexMap<u64, usize, FxBuildHasher> {
    let mut dist: IndexMap<u64, usize, FxBuildHasher> = IndexMap::default();
    dist.insert(start, 0);
    let mut frontier = vec![start];
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for u in frontier {
            let du = dist[&u];
            for p in net.parents(u) {
                if !dist.contains_key(&p) {
                    dist.insert(p, du + 1);
                    next.push(p);
                }
            }
        }
        frontier = next;
    }
    dist
}

pub(crate) fn reticulation_cycle_size_impl(net: &Network, retic: u64) -> usize {
    let parents = net.parents(retic);
    if parents.len() < 2 {
        return 0;
    }
    let (p1, p2) = (parents[0], parents[1]);
    if p1 == p2 {
        return 2;
    }
    let d1 = ancestor_depths(net, p1);
    let d2 = ancestor_depths(net, p2);
    let mut best: Option<(usize, u64)> = None;
    for (n, &a) in d1.iter() {
        if let Some(&b) = d2.get(n) {
            if best.map_or(true, |(s, _)| a + b < s) {
                best = Some((a + b, *n));
            }
        }
    }
    match best {
        None => 0,
        // r + (path p1..lca) + (path p2..lca), lca shared once.
        Some((_, lca)) => d1[&lca] + d2[&lca] + 2,
    }
}

#[pyfunction]
#[pyo3(name = "_gu_reticulation_cycle_size")]
fn reticulation_cycle_size(net: &Bound<'_, Network>, retic: &Bound<'_, PyAny>) -> usize {
    let this = net.borrow();
    match this.uid_of_any(retic) {
        Some(u) => reticulation_cycle_size_impl(&this, u),
        None => 0,
    }
}

#[pyfunction]
#[pyo3(name = "_gu_count_reticulations", signature = (net, min_cycle_size=None))]
fn count_reticulations(py: Python<'_>, net: &Bound<'_, Network>, min_cycle_size: Option<f64>) -> usize {
    let this = net.borrow();
    match min_cycle_size {
        None => count_retics(py, &this),
        Some(min) => this
            .nodes
            .iter()
            .filter(|(u, e)| e.obj.borrow(py).is_retic && reticulation_cycle_size_impl(&this, **u) as f64 >= min)
            .count(),
    }
}

#[pyfunction]
#[pyo3(name = "_gu_is_tree")]
fn is_tree(py: Python<'_>, net: &Bound<'_, Network>) -> bool {
    count_retics(py, &net.borrow()) == 0
}

#[pyfunction]
#[pyo3(name = "_gu_add_hybrid", signature = (net, source, destination, t_src=None, t_dest=None))]
fn add_hybrid(
    py: Python<'_>,
    net: &Bound<'_, Network>,
    source: &Bound<'_, Edge>,
    destination: &Bound<'_, Edge>,
    t_src: Option<f64>,
    t_dest: Option<f64>,
) -> PyResult<()> {
    let edges: Vec<Py<Edge>> = if source.as_ptr() == destination.as_ptr() {
        // Bubble / genome-duplication special case.
        warn(py, "Source and destination edges are the same. Bubble edge will be created.")?;
        let (n1, n2) = {
            let e = source.borrow();
            (e.src.clone_ref(py), e.dest.clone_ref(py))
        };
        let n3 = add_uid_node(py, net)?;
        let n4 = add_uid_node(py, net)?;
        if let Some(t) = t_src {
            Node::set_time(n3.bind(py), t)?;
        }
        if let Some(t) = t_dest {
            Node::set_time(n4.bind(py), t)?;
        }
        remove_edge_obj(py, net, source);
        vec![
            new_edge(py, n3.clone_ref(py), n4.clone_ref(py), 0.5, "right")?,
            new_edge(py, n3.clone_ref(py), n4.clone_ref(py), 0.5, "left")?,
            new_edge(py, n1, n3, 0.0, DEFAULT_TAG)?,
            new_edge(py, n4, n2, 0.0, DEFAULT_TAG)?,
        ]
    } else {
        let (a, b) = {
            let e = source.borrow();
            (e.src.clone_ref(py), e.dest.clone_ref(py))
        };
        let (x, y) = {
            let e = destination.borrow();
            (e.src.clone_ref(py), e.dest.clone_ref(py))
        };
        let n1 = add_uid_node(py, net)?;
        let n2 = add_uid_node(py, net)?;
        Node::set_is_reticulation(n2.bind(py), true);
        if let Some(t) = t_src {
            Node::set_time(n1.bind(py), t)?;
        }
        if let Some(t) = t_dest {
            Node::set_time(n2.bind(py), t)?;
        }
        remove_edge_obj(py, net, source);
        remove_edge_obj(py, net, destination);
        vec![
            new_edge(py, a, n1.clone_ref(py), 0.0, DEFAULT_TAG)?,
            new_edge(py, n1.clone_ref(py), b, 0.0, DEFAULT_TAG)?,
            new_edge(py, x, n2.clone_ref(py), 0.0, DEFAULT_TAG)?,
            new_edge(py, n2.clone_ref(py), y, 0.0, DEFAULT_TAG)?,
            new_edge(py, n1, n2, 0.0, DEFAULT_TAG)?,
        ]
    };
    let mut this = net.borrow_mut();
    for e in edges {
        this.add_edge(net.as_any(), e.bind(py))?;
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "_gu_valid_network_degrees")]
fn valid_network_degrees(py: Python<'_>, net: &Bound<'_, Network>) -> bool {
    let this = net.borrow();
    let mut root_count = 0usize;
    for (uid, entry) in this.nodes.iter() {
        let indeg = this.in_deg(*uid);
        let outdeg = this.out_deg(*uid);
        if indeg == 0 {
            root_count += 1;
            if outdeg < 2 {
                return false;
            }
        } else if outdeg == 0 {
            if indeg != 1 {
                return false;
            }
        } else if entry.obj.borrow(py).is_retic {
            if indeg != 2 || outdeg != 1 {
                return false;
            }
        } else if indeg != 1 || outdeg < 2 {
            return false;
        }
    }
    root_count == 1
}

// ── Undirected view: distances, bridges, blobs ─────────────────────────

#[derive(Clone, Copy, PartialEq)]
struct Dist(f64);

impl Eq for Dist {}

impl PartialOrd for Dist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Dist {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

fn dijkstra(adj: &[Vec<(usize, f64)>], start: usize) -> Vec<f64> {
    let mut dist = vec![f64::INFINITY; adj.len()];
    dist[start] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(Reverse((Dist(0.0), start)));
    while let Some(Reverse((Dist(d), u))) = heap.pop() {
        if d > dist[u] {
            continue;
        }
        for &(v, w) in &adj[u] {
            let nd = d + w;
            if nd < dist[v] {
                dist[v] = nd;
                heap.push(Reverse((Dist(nd), v)));
            }
        }
    }
    dist
}

/// ``pairwise_leaf_distance``: shortest undirected paths between leaves,
/// keyed by the sorted label pair, in leaf order.
fn pairwise_leaf_distance_impl(py: Python<'_>, net: &Network, use_branch_lengths: bool) -> Vec<((String, String), f64)> {
    let index: FxHashMap<u64, usize> = net.nodes.keys().enumerate().map(|(i, u)| (*u, i)).collect();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); net.nodes.len()];
    for entry in net.edges.values() {
        // An edge with no length counts as one unit, as the newick readers default it.
        let w = if use_branch_lengths {
            entry.obj.borrow(py).length.unwrap_or(1.0)
        } else {
            1.0
        };
        let (s, d) = (index[&entry.su], index[&entry.du]);
        adj[s].push((d, w));
        adj[d].push((s, w));
    }
    let leaves = net.leaf_uids();
    let labels: Vec<String> = leaves.iter().map(|&u| net.label(py, u)).collect();
    let mut out = Vec::new();
    for i in 0..leaves.len() {
        let dist = dijkstra(&adj, index[&leaves[i]]);
        for j in (i + 1)..leaves.len() {
            let key = if labels[i] <= labels[j] {
                (labels[i].clone(), labels[j].clone())
            } else {
                (labels[j].clone(), labels[i].clone())
            };
            out.push((key, dist[index[&leaves[j]]]));
        }
    }
    out
}

fn pair_dict<'py>(py: Python<'py>, pairs: Vec<((String, String), f64)>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for ((a, b), d) in pairs {
        out.set_item(PyTuple::new(py, [a, b])?, d)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_gu_pairwise_leaf_distance", signature = (net, use_branch_lengths=true))]
fn pairwise_leaf_distance<'py>(py: Python<'py>, net: &Bound<'py, Network>, use_branch_lengths: bool) -> PyResult<Bound<'py, PyDict>> {
    pair_dict(py, pairwise_leaf_distance_impl(py, &net.borrow(), use_branch_lengths))
}

/// ``_undirected_adjacency``: distinct neighbours in first-encountered
/// order, so a bubble does not read as a cycle.
fn undirected_adjacency_impl(net: &Network) -> IndexMap<u64, Vec<u64>, FxBuildHasher> {
    let mut adj: IndexMap<u64, Vec<u64>, FxBuildHasher> = net.nodes.keys().map(|&u| (u, Vec::new())).collect();
    let mut seen: FxHashMap<u64, FxHashSet<u64>> = FxHashMap::default();
    for entry in net.edges.values() {
        let (s, d) = (entry.su, entry.du);
        if seen.entry(s).or_default().insert(d) {
            adj[&s].push(d);
        }
        if seen.entry(d).or_default().insert(s) {
            adj[&d].push(s);
        }
    }
    adj
}

#[pyfunction]
#[pyo3(name = "_gu_undirected_adjacency")]
fn undirected_adjacency<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PyDict>> {
    let this = net.borrow();
    let out = PyDict::new(py);
    for (u, neighbours) in undirected_adjacency_impl(&this) {
        let list = PyList::new(py, neighbours.iter().map(|n| this.node_obj(*n).bind(py)))?;
        out.set_item(this.node_obj(u).bind(py), list)?;
    }
    Ok(out)
}

#[derive(Clone, Copy)]
struct DfsFrame {
    node: u64,
    next: usize,
    child_count: usize,
    is_art: bool,
}

/// Tarjan's bridges and articulation points, iteratively but in the exact
/// visiting order of the recursive reference.
fn bridges_and_articulations_impl(py: Python<'_>, net: &Network) -> (Vec<(String, String)>, Vec<String>) {
    let adj = undirected_adjacency_impl(net);
    let mut time = 0usize;
    let mut disc: FxHashMap<u64, usize> = FxHashMap::default();
    let mut low: FxHashMap<u64, usize> = FxHashMap::default();
    let mut parent: FxHashMap<u64, u64> = FxHashMap::default();
    let mut visited: FxHashSet<u64> = FxHashSet::default();
    let mut bridges: BTreeSet<(String, String)> = BTreeSet::new();
    let mut arts: BTreeSet<String> = BTreeSet::new();

    for &start in adj.keys() {
        if visited.contains(&start) {
            continue;
        }
        visited.insert(start);
        disc.insert(start, time);
        low.insert(start, time);
        time += 1;
        let mut stack = vec![DfsFrame { node: start, next: 0, child_count: 0, is_art: false }];
        while let Some(&DfsFrame { node: u, next, .. }) = stack.last() {
            let neighbours = &adj[&u];
            if next < neighbours.len() {
                let v = neighbours[next];
                stack.last_mut().unwrap().next += 1;
                if !visited.contains(&v) {
                    parent.insert(v, u);
                    stack.last_mut().unwrap().child_count += 1;
                    visited.insert(v);
                    disc.insert(v, time);
                    low.insert(v, time);
                    time += 1;
                    stack.push(DfsFrame { node: v, next: 0, child_count: 0, is_art: false });
                } else if parent.get(&u) != Some(&v) {
                    // Back edge: update the low-link to reflect the cycle.
                    let dv = disc[&v];
                    if dv < low[&u] {
                        low.insert(u, dv);
                    }
                }
            } else {
                let finished = stack.pop().unwrap();
                let mut is_art = finished.is_art;
                // A DFS root is an articulation point iff it has >1 DFS children.
                if !parent.contains_key(&u) && finished.child_count > 1 {
                    is_art = true;
                }
                if is_art {
                    arts.insert(net.label(py, u));
                }
                if let Some(pf) = stack.last_mut() {
                    let pu = pf.node;
                    let lu = low[&u];
                    if lu < low[&pu] {
                        low.insert(pu, lu);
                    }
                    // Bridge: no back-edge from u's subtree reaches pu or above.
                    if lu > disc[&pu] {
                        let (a, b) = (net.label(py, pu), net.label(py, u));
                        bridges.insert(if a <= b { (a, b) } else { (b, a) });
                    }
                    // Articulation (non-root): removing pu disconnects u's subtree.
                    if parent.contains_key(&pu) && lu >= disc[&pu] {
                        pf.is_art = true;
                    }
                }
            }
        }
    }
    (bridges.into_iter().collect(), arts.into_iter().collect())
}

#[pyfunction]
#[pyo3(name = "_gu_bridges_and_articulations")]
fn bridges_and_articulations(py: Python<'_>, net: &Bound<'_, Network>) -> (Vec<(String, String)>, Vec<String>) {
    bridges_and_articulations_impl(py, &net.borrow())
}

/// Pop edges until ``until`` (in either orientation); the nodes seen, in
/// first-seen order.
fn pop_component(edge_stack: &mut Vec<(u64, u64)>, until: (u64, u64)) -> IndexSet<u64, FxBuildHasher> {
    let mut comp: IndexSet<u64, FxBuildHasher> = IndexSet::default();
    while let Some((a, b)) = edge_stack.pop() {
        comp.insert(a);
        comp.insert(b);
        if (a, b) == until || (b, a) == until {
            break;
        }
    }
    comp
}

/// Tarjan's biconnected components of the undirected view, in the order
/// the reference completes them.
fn blobs_impl(net: &Network) -> Vec<IndexSet<u64, FxBuildHasher>> {
    let adj = undirected_adjacency_impl(net);
    let mut time = 0i64;
    let mut disc: FxHashMap<u64, i64> = adj.keys().map(|&u| (u, -1)).collect();
    let mut low: FxHashMap<u64, i64> = adj.keys().map(|&u| (u, -1)).collect();
    let mut parent: FxHashMap<u64, u64> = FxHashMap::default();
    let mut edge_stack: Vec<(u64, u64)> = Vec::new();
    let mut components = Vec::new();

    for &start in adj.keys() {
        if disc[&start] != -1 {
            continue;
        }
        disc.insert(start, time);
        low.insert(start, time);
        time += 1;
        let mut stack: Vec<(u64, usize)> = vec![(start, 0)];
        while let Some(&(u, next)) = stack.last() {
            let neighbours = &adj[&u];
            if next < neighbours.len() {
                let v = neighbours[next];
                stack.last_mut().unwrap().1 += 1;
                if disc[&v] == -1 {
                    parent.insert(v, u);
                    edge_stack.push((u, v));
                    disc.insert(v, time);
                    low.insert(v, time);
                    time += 1;
                    stack.push((v, 0));
                } else if parent.get(&u) != Some(&v) && disc[&v] < disc[&u] {
                    // Back edge: part of the current component.
                    edge_stack.push((u, v));
                    let dv = disc[&v];
                    if dv < low[&u] {
                        low.insert(u, dv);
                    }
                }
            } else {
                stack.pop();
                if let Some(&(pu, _)) = stack.last() {
                    let lu = low[&u];
                    if lu < low[&pu] {
                        low.insert(pu, lu);
                    }
                    // Articulation detected: u's subtree cannot reach above pu.
                    if lu >= disc[&pu] {
                        let comp = pop_component(&mut edge_stack, (pu, u));
                        if !comp.is_empty() {
                            components.push(comp);
                        }
                    }
                }
            }
        }
        // Flush any remaining edges into a final component.
        if let Some(&last) = edge_stack.last() {
            let comp = pop_component(&mut edge_stack, last);
            if !comp.is_empty() {
                components.push(comp);
            }
        }
    }
    components
}

#[pyfunction]
#[pyo3(name = "_gu_blobs")]
fn blobs<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PyList>> {
    let this = net.borrow();
    let out = PyList::empty(py);
    for comp in blobs_impl(&this) {
        out.append(PySet::new(py, comp.iter().map(|u| this.node_obj(*u).bind(py)))?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_gu_tree_of_blobs")]
fn tree_of_blobs<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PyList>> {
    let this = net.borrow();
    let out = PyList::empty(py);
    for comp in blobs_impl(&this) {
        let blob = Network::new_object(py, None)?;
        let mut old_new: FxHashMap<u64, Py<Node>> = FxHashMap::default();
        // ``for node in comp_nodes``: the blob's node set in the order the
        // reference iterated it.
        let hashes: Vec<u64> = comp.iter().map(|u| this.node_obj(*u).borrow(py).hash).collect();
        let ordered: Vec<u64> = cpython_set_order(&hashes).into_iter().map(|i| comp[i]).collect();
        {
            let mut b = blob.borrow_mut();
            for &u in ordered.iter() {
                let copy = this.node_obj(u).borrow(py).copy(py)?;
                b.add_node(blob.as_any(), copy.bind(py))?;
                old_new.insert(u, copy);
            }
            for entry in this.edges.values() {
                if comp.contains(&entry.su) && comp.contains(&entry.du) {
                    let copy = entry.obj.borrow(py).copy(
                        py,
                        Some(old_new[&entry.su].clone_ref(py)),
                        Some(old_new[&entry.du].clone_ref(py)),
                    )?;
                    b.add_edge(blob.as_any(), copy.bind(py))?;
                }
            }
        }
        out.append(blob)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_gu_level")]
pub(crate) fn level(py: Python<'_>, net: &Bound<'_, Network>) -> usize {
    let this = net.borrow();
    if this.nodes.is_empty() {
        return 0;
    }
    blobs_impl(&this)
        .iter()
        .map(|comp| comp.iter().filter(|u| this.nodes[*u].obj.borrow(py).is_retic).count())
        .max()
        .unwrap_or(0)
}

// ── Node heights, edges, clones ────────────────────────────────────────

#[pyfunction]
#[pyo3(name = "_gu_edge_between")]
fn edge_between(py: Python<'_>, net: &Bound<'_, Network>, parent: &Bound<'_, PyAny>, child: &Bound<'_, PyAny>) -> PyResult<Py<Edge>> {
    let this = net.borrow();
    let (Some(su), Some(du)) = (this.uid_of_any(parent), this.uid_of_any(child)) else {
        return Err(PyValueError::new_err("No matching edges found"));
    };
    this.edge_between(py, su, du, None, None)
}

/// ``_node_height``: memoised post-order height above the present, read
/// from and written to the caller's ``cache`` dict (keyed by Node).
fn node_height_rec(
    py: Python<'_>,
    net: &Network,
    uid: u64,
    cache: &Bound<'_, PyDict>,
    memo: &mut FxHashMap<u64, f64>,
) -> PyResult<f64> {
    if let Some(&h) = memo.get(&uid) {
        return Ok(h);
    }
    let obj = net.node_obj(uid).bind(py);
    if let Some(cached) = cache.get_item(obj)? {
        if !cached.is_none() {
            let h: f64 = cached.extract()?;
            memo.insert(uid, h);
            return Ok(h);
        }
    }
    let mut best = 0.0_f64;
    for child in net.children(uid) {
        let euid = net.first_edge_between(py, uid, child)?;
        let length = net.edges[&euid].obj.borrow(py).length.unwrap_or(0.0);
        let h = node_height_rec(py, net, child, cache, memo)? + length;
        if h > best {
            best = h;
        }
    }
    cache.set_item(obj, best)?;
    memo.insert(uid, best);
    Ok(best)
}

#[pyfunction]
#[pyo3(name = "_gu_node_height")]
fn node_height(py: Python<'_>, net: &Bound<'_, Network>, node: &Bound<'_, PyAny>, cache: &Bound<'_, PyDict>) -> PyResult<f64> {
    let this = net.borrow();
    match this.uid_of_any(node) {
        Some(uid) => {
            let mut memo: FxHashMap<u64, f64> = FxHashMap::default();
            node_height_rec(py, &this, uid, cache, &mut memo)
        }
        None => {
            // Not in the graph: no children, so height 0.
            if let Some(cached) = cache.get_item(node)? {
                if !cached.is_none() {
                    return cached.extract();
                }
            }
            cache.set_item(node, 0.0)?;
            Ok(0.0)
        }
    }
}

#[pyfunction]
#[pyo3(name = "_gu_node_heights")]
fn node_heights<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PyDict>> {
    let this = net.borrow();
    let cache = PyDict::new(py);
    let mut memo: FxHashMap<u64, f64> = FxHashMap::default();
    let uids: Vec<u64> = this.nodes.keys().copied().collect();
    for uid in uids {
        node_height_rec(py, &this, uid, &cache, &mut memo)?;
    }
    Ok(cache)
}

#[pyfunction]
#[pyo3(name = "_gu_clone_net")]
fn clone_net<'py>(net: &Bound<'py, Network>) -> PyResult<Bound<'py, Network>> {
    Ok(Network::copy_impl(net)?.0)
}

// ── Isomorphism ────────────────────────────────────────────────────────

/// Non-trivial clusters with leaves renamed ``L1..Ln`` by label rank.
fn normalized_clusters(py: Python<'_>, net: &Network) -> PyResult<FxHashSet<BTreeSet<usize>>> {
    let mut leaves: Vec<(String, u64)> = net.leaf_uids().into_iter().map(|u| (net.label(py, u), u)).collect();
    leaves.sort_by(|a, b| a.0.cmp(&b.0));
    let rank: FxHashMap<u64, usize> = leaves.iter().enumerate().map(|(i, (_, u))| (*u, i)).collect();
    let root = net.root_uid(py)?;
    let map = net.leaf_descendants_all_map(py)?;
    let mut out: FxHashSet<BTreeSet<usize>> = FxHashSet::default();
    for (uid, below) in map.iter() {
        if *uid != root && below.len() > 1 {
            out.insert(below.iter().map(|u| rank[u]).collect());
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_gu_is_isomorphic")]
fn is_isomorphic(py: Python<'_>, net1: &Bound<'_, Network>, net2: &Bound<'_, Network>) -> PyResult<bool> {
    let (a, b) = (net1.borrow(), net2.borrow());
    if a.leaf_uids().len() != b.leaf_uids().len()
        || a.nodes.len() != b.nodes.len()
        || a.edges.len() != b.edges.len()
        || count_retics(py, &a) != count_retics(py, &b)
    {
        return Ok(false);
    }
    Ok(normalized_clusters(py, &a)? == normalized_clusters(py, &b)?)
}

// ── Distances ──────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(name = "_gu_symmetric_difference_distance")]
fn symmetric_difference_distance(py: Python<'_>, s1: &Bound<'_, PyAny>, s2: &Bound<'_, PyAny>, normalize: bool) -> PyResult<Py<PyAny>> {
    let sym = s1.call_method1("symmetric_difference", (s2,))?.len()?;
    if normalize {
        let union = s1.call_method1("union", (s2,))?.len()?;
        let value = if union > 0 { sym as f64 / union as f64 } else { 0.0 };
        return Ok(value.into_pyobject(py)?.into_any().unbind());
    }
    Ok(sym.into_pyobject(py)?.into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "_gu_hardwired_cluster_distance", signature = (net1, net2, normalize=false))]
fn hardwired_cluster_distance(py: Python<'_>, net1: &Bound<'_, Network>, net2: &Bound<'_, Network>, normalize: bool) -> PyResult<Py<PyAny>> {
    let s1 = hardwired_clusters(py, &net1.borrow(), false)?;
    let s2 = hardwired_clusters(py, &net2.borrow(), false)?;
    symmetric_difference(py, &s1, &s2, normalize)
}

#[pyfunction]
#[pyo3(name = "_gu_softwired_cluster_distance", signature = (net1, net2, normalize=false))]
fn softwired_cluster_distance<'py>(py: Python<'py>, net1: &Bound<'py, Network>, net2: &Bound<'py, Network>, normalize: bool) -> PyResult<Py<PyAny>> {
    let s1 = softwired_clusters(py, net1, false)?;
    let s2 = softwired_clusters(py, net2, false)?;
    symmetric_difference(py, &s1, &s2, normalize)
}

/// ``_mu_vectors``: path counts to every leaf, one vector per node that
/// would survive reduction, in ``V()`` order.
fn mu_vectors_impl(py: Python<'_>, net: &Network, leaf_order: &[String]) -> PyResult<Vec<Vec<i64>>> {
    let leaves: FxHashSet<u64> = net.leaf_uids().into_iter().collect();
    let mut leaf_index: FxHashMap<&str, usize> = FxHashMap::default();
    for (i, label) in leaf_order.iter().enumerate() {
        leaf_index.insert(label.as_str(), i);
    }
    let n = leaf_order.len();
    let topo = net.topological_order_uids(py)?;
    let mut counts: FxHashMap<u64, Vec<i64>> = FxHashMap::default();
    for &node in topo.iter().rev() {
        let mut vec = vec![0i64; n];
        if leaves.contains(&node) {
            if let Some(&i) = leaf_index.get(net.label(py, node).as_str()) {
                vec[i] = 1;
            }
        }
        for child in net.children(node) {
            if let Some(cv) = counts.get(&child) {
                for i in 0..n {
                    vec[i] += cv[i];
                }
            }
        }
        counts.insert(node, vec);
    }
    let mut reduced = Vec::new();
    for (uid, entry) in net.nodes.iter() {
        if !entry.obj.borrow(py).is_retic && net.in_deg(*uid) == 1 && net.out_deg(*uid) == 1 {
            continue; // this node would be suppressed in a reduced network
        }
        reduced.push(counts[uid].clone());
    }
    Ok(reduced)
}

#[pyfunction]
#[pyo3(name = "_gu_mu_vectors", signature = (net, leaf_order=None))]
fn mu_vectors<'py>(py: Python<'py>, net: &Bound<'py, Network>, leaf_order: Option<Vec<String>>) -> PyResult<Bound<'py, PyList>> {
    let this = net.borrow();
    let order = match leaf_order {
        Some(o) => o,
        None => {
            let mut labels: Vec<String> = this.leaf_uids().into_iter().map(|u| this.label(py, u)).collect();
            labels.sort();
            labels
        }
    };
    let vectors = mu_vectors_impl(py, &this, &order)?;
    let out = PyList::empty(py);
    for v in vectors {
        out.append(PyTuple::new(py, v)?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_gu_mu_distance")]
fn mu_distance(py: Python<'_>, net1: &Bound<'_, Network>, net2: &Bound<'_, Network>) -> PyResult<i64> {
    let (a, b) = (net1.borrow(), net2.borrow());
    let leaf_order: Vec<String> = require_same_leaves(py, &a, &b, "mu_distance")?
        .iter()
        .map(|l| l.to_string())
        .collect();
    let mut counter: FxHashMap<Vec<i64>, i64> = FxHashMap::default();
    for v in mu_vectors_impl(py, &a, &leaf_order)? {
        *counter.entry(v).or_insert(0) += 1;
    }
    for v in mu_vectors_impl(py, &b, &leaf_order)? {
        *counter.entry(v).or_insert(0) -= 1;
    }
    Ok(counter.values().map(|c| c.abs()).sum())
}

fn hybrid_ancestry_impl(py: Python<'_>, net: &Network) -> PyResult<LabelSet> {
    let below = labels_below(py, net)?;
    let mut out = LabelSet::new();
    for (uid, entry) in net.nodes.iter() {
        if !entry.obj.borrow(py).is_retic {
            continue;
        }
        match below.get(uid) {
            Some(labels) => out.extend(labels.iter().cloned()),
            // A reticulation unreachable from the root: KeyError, as the
            // reference's ``below[v]`` raises.
            None => return Err(PyKeyError::new_err(entry.obj.clone_ref(py).into_any())),
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_gu_hybrid_ancestry")]
fn hybrid_ancestry<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PyFrozenSet>> {
    frozenset_of_labels(py, &hybrid_ancestry_impl(py, &net.borrow())?)
}

#[pyfunction]
#[pyo3(name = "_gu_hybrid_ancestry_error")]
fn hybrid_ancestry_error(py: Python<'_>, estimated: &Bound<'_, Network>, true_net: &Bound<'_, Network>) -> PyResult<f64> {
    let (est, truth) = (estimated.borrow(), true_net.borrow());
    require_same_leaves(py, &est, &truth, "hybrid_ancestry_error")?;
    let b = hybrid_ancestry_impl(py, &est)?;
    let a = hybrid_ancestry_impl(py, &truth)?;
    let fpr = if b.is_empty() {
        0.0
    } else {
        b.difference(&a).count() as f64 / b.len() as f64
    };
    let fnr = if a.is_empty() {
        0.0
    } else {
        a.difference(&b).count() as f64 / a.len() as f64
    };
    Ok((fpr + fnr) / 2.0)
}

/// ``_tripartitions``: ``(desc(c1), desc(c2), rest)`` at every tree node
/// with exactly two children, each in canonical (sorted) form.
fn tripartitions_impl(py: Python<'_>, net: &Network) -> FxHashSet<Tripartition> {
    let mut labels = Labels::new(py, net);
    let all_leaves = labels.set(net.leaf_uids());
    let mut out: FxHashSet<Tripartition> = FxHashSet::default();
    for (uid, entry) in net.nodes.iter() {
        let children = net.children(*uid);
        if children.len() != 2 || entry.obj.borrow(py).is_retic {
            continue;
        }
        let desc1 = labels.set(net.leaf_descendants_uid(children[0]));
        let desc2 = labels.set(net.leaf_descendants_uid(children[1]));
        let rest: LabelSet = all_leaves
            .iter()
            .filter(|l| !desc1.contains(*l) && !desc2.contains(*l))
            .cloned()
            .collect();
        let mut parts = vec![desc1, desc2, rest];
        parts.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        let mut it = parts.into_iter();
        out.insert((it.next().unwrap(), it.next().unwrap(), it.next().unwrap()));
    }
    out
}

#[pyfunction]
#[pyo3(name = "_gu_tripartitions")]
fn tripartitions<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PySet>> {
    let out = PySet::empty(py)?;
    let mut strs = Strs::new(py);
    for (a, b, c) in tripartitions_impl(py, &net.borrow()) {
        out.add(strs.tuple(&[a, b, c])?)?;
    }
    Ok(out)
}

fn displayed_tree_topology_set_impl<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<FxHashSet<TopologyKey>> {
    let mut out: FxHashSet<TopologyKey> = FxHashSet::default();
    if count_retics(py, &net.borrow()) == 0 {
        out.insert(topology_key(py, &net.borrow())?);
        return Ok(out);
    }
    for tree in displayed_trees(py, net, true)? {
        out.insert(topology_key(py, &tree.borrow())?);
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_gu_displayed_tree_topology_set")]
fn displayed_tree_topology_set<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PySet>> {
    let out = PySet::empty(py)?;
    let mut strs = Strs::new(py);
    for key in displayed_tree_topology_set_impl(py, net)? {
        out.add(strs.frozenset_of_sets(&key)?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_gu_tripartition_distance", signature = (net1, net2, normalize=false))]
fn tripartition_distance(py: Python<'_>, net1: &Bound<'_, Network>, net2: &Bound<'_, Network>, normalize: bool) -> PyResult<Py<PyAny>> {
    let s1 = tripartitions_impl(py, &net1.borrow());
    let s2 = tripartitions_impl(py, &net2.borrow());
    symmetric_difference(py, &s1, &s2, normalize)
}

#[pyfunction]
#[pyo3(name = "_gu_displayed_tree_distance", signature = (net1, net2, normalize=false))]
fn displayed_tree_distance<'py>(py: Python<'py>, net1: &Bound<'py, Network>, net2: &Bound<'py, Network>, normalize: bool) -> PyResult<Py<PyAny>> {
    let s1 = displayed_tree_topology_set_impl(py, net1)?;
    let s2 = displayed_tree_topology_set_impl(py, net2)?;
    symmetric_difference(py, &s1, &s2, normalize)
}

#[pyfunction]
#[pyo3(name = "_gu_robinson_foulds_distance", signature = (net1, net2, normalize=false))]
fn robinson_foulds_distance(py: Python<'_>, net1: &Bound<'_, Network>, net2: &Bound<'_, Network>, normalize: bool) -> PyResult<Py<PyAny>> {
    let (a, b) = (net1.borrow(), net2.borrow());
    let (l1, l2) = (leaf_label_set(py, &a), leaf_label_set(py, &b));
    if l1 != l2 {
        return Err(network_error(
            py,
            format!(
                "Robinson-Foulds distance requires identical leaf label sets. Got {} vs {}",
                sorted_list_repr(py, &l1)?,
                sorted_list_repr(py, &l2)?
            ),
        ));
    }
    let n = l1.len();
    let nontrivial = |clusters: FxHashSet<LabelSet>| -> FxHashSet<LabelSet> {
        clusters.into_iter().filter(|c| 1 < c.len() && c.len() < n).collect()
    };
    let hc1 = nontrivial(hardwired_clusters(py, &a, false)?);
    let hc2 = nontrivial(hardwired_clusters(py, &b, false)?);
    let sym = hc1.symmetric_difference(&hc2).count();
    if normalize {
        let denom = hc1.len() + hc2.len();
        let value = if denom > 0 { sym as f64 / denom as f64 } else { 0.0 };
        return Ok(value.into_pyobject(py)?.into_any().unbind());
    }
    Ok(sym.into_pyobject(py)?.into_any().unbind())
}

/// ``_average_pairwise_matrix``: leaf distances averaged over the displayed
/// trees, uniformly or by tree probability, keyed by sorted label pair.
fn average_pairwise_impl<'py>(py: Python<'py>, net: &Bound<'py, Network>, weighted: bool) -> PyResult<Vec<((String, String), f64)>> {
    let labels: Vec<String> = {
        let this = net.borrow();
        let mut labels: Vec<String> = this.leaf_uids().into_iter().map(|u| this.label(py, u)).collect();
        labels.sort();
        labels
    };
    let mut pairs: Vec<(String, String)> = Vec::new();
    for i in 0..labels.len() {
        for j in (i + 1)..labels.len() {
            pairs.push(if labels[i] <= labels[j] {
                (labels[i].clone(), labels[j].clone())
            } else {
                (labels[j].clone(), labels[i].clone())
            });
        }
    }
    let mut sums = vec![0.0_f64; pairs.len()];
    let mut weight_sum = 0.0_f64;
    for (tree, prob) in displayed_trees_with_probs_impl(py, net)? {
        let w = if weighted { prob } else { 1.0 };
        weight_sum += w;
        let dist: FxHashMap<(String, String), f64> =
            pairwise_leaf_distance_impl(py, &tree.borrow(), true).into_iter().collect();
        for (k, pair) in pairs.iter().enumerate() {
            if let Some(d) = dist.get(pair) {
                sums[k] += w * d;
            }
        }
    }
    if weight_sum > 0.0 {
        for s in sums.iter_mut() {
            *s /= weight_sum;
        }
    }
    Ok(pairs.into_iter().zip(sums).collect())
}

#[pyfunction]
#[pyo3(name = "_gu_average_pairwise_matrix", signature = (net, weighted=false))]
fn average_pairwise_matrix<'py>(py: Python<'py>, net: &Bound<'py, Network>, weighted: bool) -> PyResult<Bound<'py, PyDict>> {
    pair_dict(py, average_pairwise_impl(py, net, weighted)?)
}

/// Shared body of the (weighted) average path distance.
fn path_distance<'py>(
    py: Python<'py>,
    net1: &Bound<'py, Network>,
    net2: &Bound<'py, Network>,
    normalize: bool,
    weighted: bool,
    caller: &str,
) -> PyResult<Py<PyAny>> {
    let (l1, l2) = (leaf_label_set(py, &net1.borrow()), leaf_label_set(py, &net2.borrow()));
    if l1 != l2 {
        return Err(network_error(
            py,
            format!(
                "{caller} requires identical leaf label sets. Got {} vs {}",
                sorted_list_repr(py, &l1)?,
                sorted_list_repr(py, &l2)?
            ),
        ));
    }
    let m1 = average_pairwise_impl(py, net1, weighted)?;
    let m2: FxHashMap<(String, String), f64> = average_pairwise_impl(py, net2, weighted)?.into_iter().collect();
    let mut total = 0.0_f64;
    for (pair, v1) in &m1 {
        total += (v1 - m2.get(pair).copied().unwrap_or(0.0)).abs();
    }
    if normalize {
        let n_pairs = m1.len();
        let value = if n_pairs > 0 { total / n_pairs as f64 } else { 0.0 };
        return Ok(value.into_pyobject(py)?.into_any().unbind());
    }
    if m1.is_empty() {
        // ``sum`` of nothing is the int 0.
        return Ok(0usize.into_pyobject(py)?.into_any().unbind());
    }
    Ok(total.into_pyobject(py)?.into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "_gu_average_path_distance", signature = (net1, net2, normalize=false))]
fn average_path_distance<'py>(py: Python<'py>, net1: &Bound<'py, Network>, net2: &Bound<'py, Network>, normalize: bool) -> PyResult<Py<PyAny>> {
    path_distance(py, net1, net2, normalize, false, "Average path distance")
}

#[pyfunction]
#[pyo3(name = "_gu_weighted_average_path_distance", signature = (net1, net2, normalize=false))]
fn weighted_average_path_distance<'py>(py: Python<'py>, net1: &Bound<'py, Network>, net2: &Bound<'py, Network>, normalize: bool) -> PyResult<Py<PyAny>> {
    path_distance(py, net1, net2, normalize, true, "Weighted average path distance")
}

// ── Reticulation identity and rooted triples ───────────────────────────

/// ``_direct_branch_leaf_sets``: the leaf sets of the clades directly
/// below ``uid``; a leaf is its own one-element branch.
fn direct_branch_leaf_sets(labels: &mut Labels<'_, '_>, uid: u64) -> Vec<LabelSet> {
    let net = labels.net;
    let children = net.children(uid);
    if children.is_empty() {
        return vec![LabelSet::from([labels.get(uid)])];
    }
    let mut branches = Vec::new();
    for child in children {
        let leaves = labels.set(net.leaf_descendants_uid(child));
        if !leaves.is_empty() {
            branches.push(leaves);
        }
    }
    branches
}

#[pyfunction]
#[pyo3(name = "_gu_direct_branch_leaf_sets")]
fn direct_branch_leaf_sets_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let this = net.borrow();
    let sets = match this.uid_of_any(node) {
        Some(uid) => direct_branch_leaf_sets(&mut Labels::new(py, &this), uid),
        // Not in the graph: no children, so the node's own label.
        None => vec![LabelSet::from([Label::from(node.getattr("label")?.extract::<String>()?.as_str())])],
    };
    let mut strs = Strs::new(py);
    let items: Vec<Bound<'py, PyFrozenSet>> = sets.iter().map(|s| strs.frozenset(s)).collect::<PyResult<_>>()?;
    PyList::new(py, items)
}

#[pyfunction]
#[pyo3(name = "_gu_reticulation_signature")]
fn reticulation_signature<'py>(py: Python<'py>, net: &Bound<'py, Network>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let this = net.borrow();
    let Some(uid) = this.uid_of_any(node) else {
        return Err(network_error(py, "reticulation node is not in the network"));
    };
    let mut labels = Labels::new(py, &this);
    let mut strs = Strs::new(py);
    // Child clade is recorded for placement. Equality does not use it.
    let children = this.children(uid);
    let child_branches: Vec<LabelSet> = match children.first() {
        Some(&child) => direct_branch_leaf_sets(&mut labels, child),
        None => vec![LabelSet::from([labels.get(uid)])],
    };
    // Siblings are the parent's other children, by object identity.
    let node_ptr = node.as_ptr();
    let mut parent_branches: Vec<Bound<'py, PyTuple>> = Vec::new();
    for parent in this.parents(uid) {
        let mut side: Vec<LabelSet> = Vec::new();
        for sibling in this.children(parent) {
            if this.node_obj(sibling).as_ptr() == node_ptr {
                continue;
            }
            side.extend(direct_branch_leaf_sets(&mut labels, sibling));
        }
        parent_branches.push(strs.tuple(&side)?);
    }
    let child_tuple = strs.tuple(&child_branches)?;
    let parent_tuple = PyTuple::new(py, parent_branches)?;
    PyTuple::new(py, [child_tuple.into_any(), parent_tuple.into_any()])
}

/// ``_signature_parents``: the parent-side branch lists of a signature, a
/// dict carrying one under ``"signature"``, or an object with that attribute.
fn signature_parents<'py>(sig: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let sig = match sig.cast::<PyDict>() {
        Ok(d) => match d.get_item("signature")? {
            Some(v) => v,
            None => sig.clone(),
        },
        Err(_) => sig.clone(),
    };
    let sig = if sig.hasattr("signature")? { sig.getattr("signature")? } else { sig };
    sig.get_item(1)
}

#[pyfunction]
#[pyo3(name = "_gu_signature_parents")]
fn signature_parents_py<'py>(sig: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    signature_parents(sig)
}

/// ``_branches_overlap``: some branch on ``side_a`` shares a leaf with some
/// branch on ``side_b``.
fn branches_overlap(py: Python<'_>, side_a: &Bound<'_, PyAny>, side_b: &Bound<'_, PyAny>) -> PyResult<bool> {
    let frozenset = py.get_type::<PyFrozenSet>();
    for branch_a in side_a.try_iter()? {
        let branch_a = branch_a?;
        for branch_b in side_b.try_iter()? {
            let branch_b = branch_b?;
            let fa = frozenset.call1((&branch_a,))?;
            let fb = frozenset.call1((&branch_b,))?;
            if fa.call_method1("intersection", (fb,))?.is_truthy()? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

#[pyfunction]
#[pyo3(name = "_gu_branches_overlap")]
fn branches_overlap_py(py: Python<'_>, side_a: &Bound<'_, PyAny>, side_b: &Bound<'_, PyAny>) -> PyResult<bool> {
    branches_overlap(py, side_a, side_b)
}

fn same_reticulation_impl(py: Python<'_>, sig_a: &Bound<'_, PyAny>, sig_b: &Bound<'_, PyAny>) -> PyResult<bool> {
    let parents_a = signature_parents(sig_a)?;
    let parents_b = signature_parents(sig_b)?;
    if parents_a.len()? != 2 || parents_b.len()? != 2 {
        return Ok(false);
    }
    let (a0, a1) = (parents_a.get_item(0)?, parents_a.get_item(1)?);
    let (b0, b1) = (parents_b.get_item(0)?, parents_b.get_item(1)?);
    let straight = branches_overlap(py, &a0, &b0)? && branches_overlap(py, &a1, &b1)?;
    let crossed = branches_overlap(py, &a0, &b1)? && branches_overlap(py, &a1, &b0)?;
    Ok(straight || crossed)
}

#[pyfunction]
#[pyo3(name = "_gu_same_reticulation")]
fn same_reticulation(py: Python<'_>, sig_a: &Bound<'_, PyAny>, sig_b: &Bound<'_, PyAny>) -> PyResult<bool> {
    same_reticulation_impl(py, sig_a, sig_b)
}

#[pyfunction]
#[pyo3(name = "_gu_group_reticulations")]
fn group_reticulations<'py>(py: Python<'py>, records: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    let mut groups: Vec<Bound<'py, PyList>> = Vec::new();
    let mut representatives: Vec<Bound<'py, PyAny>> = Vec::new();
    for record in records.try_iter()? {
        let record = record?;
        let mut matched = None;
        for (index, rep) in representatives.iter().enumerate() {
            if same_reticulation_impl(py, &record, rep)? {
                matched = Some(index);
                break;
            }
        }
        match matched {
            None => {
                let group = PyList::new(py, [&record])?;
                out.append(&group)?;
                groups.push(group);
                representatives.push(record);
            }
            Some(index) => groups[index].append(record)?,
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_gu_rooted_triples")]
fn rooted_triples<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PySet>> {
    let this = net.borrow();
    // Leaves are numbered in label order, so index order is label order and
    // a triple is three small integers until it is written out.
    let mut leaves: Vec<(String, u64)> = this.leaf_uids().into_iter().map(|u| (this.label(py, u), u)).collect();
    leaves.sort();
    let n = leaves.len();
    let index: FxHashMap<u64, usize> = leaves.iter().enumerate().map(|(i, (_, u))| (*u, i)).collect();
    // One descendant-leaf map, then emit each triple at the MRCA of its pair.
    let desc: FxHashMap<u64, Vec<usize>> = this
        .leaf_descendants_all_map(py)?
        .iter()
        .map(|(u, ls)| (*u, ls.iter().map(|l| index[l]).collect()))
        .collect();
    let mut triples: FxHashSet<(usize, usize, usize)> = FxHashSet::default();
    let mut in_below = vec![false; n];
    for (uid, entry) in this.nodes.iter() {
        let kids: Vec<u64> = this
            .children(*uid)
            .into_iter()
            .filter(|c| desc.get(c).map_or(false, |s| !s.is_empty()))
            .collect();
        if kids.len() < 2 {
            continue;
        }
        let below = match desc.get(uid) {
            Some(b) => b,
            None => return Err(PyKeyError::new_err(entry.obj.clone_ref(py).into_any())),
        };
        // The root's descendant set is every leaf, so a resolved clade is
        // always a proper subset. Skipping the full set keeps a polytomy
        // at the root from inventing triples.
        if below.len() < 2 || below.len() == n {
            continue;
        }
        in_below.iter_mut().for_each(|b| *b = false);
        for &l in below {
            in_below[l] = true;
        }
        let outside: Vec<usize> = (0..n).filter(|i| !in_below[*i]).collect();
        for i in 0..kids.len() {
            let left = &desc[&kids[i]];
            for j in (i + 1)..kids.len() {
                for &a in left {
                    for &b in &desc[&kids[j]] {
                        let (x, y) = if a < b { (a, b) } else { (b, a) };
                        for &c in &outside {
                            triples.insert((x, y, c));
                        }
                    }
                }
            }
        }
    }
    let strs: Vec<Bound<'py, PyString>> = leaves.iter().map(|(l, _)| PyString::new(py, l)).collect();
    let out = PySet::empty(py)?;
    for (x, y, c) in triples {
        out.add(PyTuple::new(py, [&strs[x], &strs[y], &strs[c]])?)?;
    }
    Ok(out)
}

// ── Registration ───────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_all_clusters, m)?)?;
    m.add_function(wrap_pyfunction!(leaf_label_set_py, m)?)?;
    m.add_function(wrap_pyfunction!(require_same_leaves_py, m)?)?;
    m.add_function(wrap_pyfunction!(hardwired_clusters_by_label, m)?)?;
    m.add_function(wrap_pyfunction!(leaf_labels_below, m)?)?;
    m.add_function(wrap_pyfunction!(network_clusters, m)?)?;
    m.add_function(wrap_pyfunction!(softwired_clusters_by_label, m)?)?;
    m.add_function(wrap_pyfunction!(tree_topology_key, m)?)?;
    m.add_function(wrap_pyfunction!(induced_subnetwork, m)?)?;
    m.add_function(wrap_pyfunction!(subnet_given_leaves, m)?)?;
    m.add_function(wrap_pyfunction!(induced_subnetwork_by_taxa, m)?)?;
    m.add_function(wrap_pyfunction!(suppress_childless_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(displayed_tree, m)?)?;
    m.add_function(wrap_pyfunction!(displayed_trees_with_probs, m)?)?;
    m.add_function(wrap_pyfunction!(get_displayed_trees, m)?)?;
    m.add_function(wrap_pyfunction!(dominant_tree, m)?)?;
    m.add_function(wrap_pyfunction!(count_displayed_trees, m)?)?;
    m.add_function(wrap_pyfunction!(reticulation_cycle_size, m)?)?;
    m.add_function(wrap_pyfunction!(count_reticulations, m)?)?;
    m.add_function(wrap_pyfunction!(is_tree, m)?)?;
    m.add_function(wrap_pyfunction!(add_hybrid, m)?)?;
    m.add_function(wrap_pyfunction!(valid_network_degrees, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_leaf_distance, m)?)?;
    m.add_function(wrap_pyfunction!(undirected_adjacency, m)?)?;
    m.add_function(wrap_pyfunction!(bridges_and_articulations, m)?)?;
    m.add_function(wrap_pyfunction!(blobs, m)?)?;
    m.add_function(wrap_pyfunction!(tree_of_blobs, m)?)?;
    m.add_function(wrap_pyfunction!(level, m)?)?;
    m.add_function(wrap_pyfunction!(edge_between, m)?)?;
    m.add_function(wrap_pyfunction!(node_height, m)?)?;
    m.add_function(wrap_pyfunction!(node_heights, m)?)?;
    m.add_function(wrap_pyfunction!(clone_net, m)?)?;
    m.add_function(wrap_pyfunction!(is_isomorphic, m)?)?;
    m.add_function(wrap_pyfunction!(symmetric_difference_distance, m)?)?;
    m.add_function(wrap_pyfunction!(hardwired_cluster_distance, m)?)?;
    m.add_function(wrap_pyfunction!(softwired_cluster_distance, m)?)?;
    m.add_function(wrap_pyfunction!(mu_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(mu_distance, m)?)?;
    m.add_function(wrap_pyfunction!(hybrid_ancestry, m)?)?;
    m.add_function(wrap_pyfunction!(hybrid_ancestry_error, m)?)?;
    m.add_function(wrap_pyfunction!(tripartitions, m)?)?;
    m.add_function(wrap_pyfunction!(displayed_tree_topology_set, m)?)?;
    m.add_function(wrap_pyfunction!(tripartition_distance, m)?)?;
    m.add_function(wrap_pyfunction!(displayed_tree_distance, m)?)?;
    m.add_function(wrap_pyfunction!(robinson_foulds_distance, m)?)?;
    m.add_function(wrap_pyfunction!(average_pairwise_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(average_path_distance, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_average_path_distance, m)?)?;
    m.add_function(wrap_pyfunction!(direct_branch_leaf_sets_py, m)?)?;
    m.add_function(wrap_pyfunction!(reticulation_signature, m)?)?;
    m.add_function(wrap_pyfunction!(signature_parents_py, m)?)?;
    m.add_function(wrap_pyfunction!(branches_overlap_py, m)?)?;
    m.add_function(wrap_pyfunction!(same_reticulation, m)?)?;
    m.add_function(wrap_pyfunction!(group_reticulations, m)?)?;
    m.add_function(wrap_pyfunction!(rooted_triples, m)?)?;
    Ok(())
}
