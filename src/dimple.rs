//! DIMPLE: divide a tree of blobs, infer each piece, stitch.
//!
//! The Python module ``phynetpy._dimple`` is a facade over the ``_dp_*``
//! functions registered here.  The records (dataclasses), the error class,
//! the subset-chain orchestration that calls the MCMC machinery and the
//! ``infer_dimple`` driver stay in Python; every graph, combinatorial and
//! numerical step of the pipeline runs here.
//!
//! Every function follows its Python reference step for step, including
//! the order in which nodes and edges are visited, copied and removed, and
//! every tie-break the reference made.  Where the reference depended on
//! CPython's ``set`` iteration order of small integers (networkx's
//! ``find_cliques``), that order is reproduced by a small simulator of
//! CPython's set layout, so the networkx dependency is gone without
//! changing which clique wins.

use std::collections::{BTreeSet, VecDeque};

use indexmap::{IndexMap, IndexSet};
use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{
    PyAssertionError, PyIndexError, PyKeyError, PyRuntimeError, PyStopIteration, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyString, PyTuple, PyType};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::graph::{Edge, Network, Node};
use crate::util::{network_error, warn};

/// ``frozenset[str]``: a set of leaf labels.  Sorted iteration for free.
type LabelSet = BTreeSet<String>;
/// ``(a, b, c)`` with ``a < b``: the rooted triple ``(a, b | c)``.
type Triple = (String, String, String);

const DEFAULT_TAG: &str = "no tag assigned yet";
const OUT: &str = "OUT";

// These mirror the module constants of ``phynetpy._dimple``; the Python
// module documents them and the tests pin the values.
const REDUNDANCY_THRESHOLD: f64 = 0.02;
const OUTGROUP_SCALE: f64 = 2.0;
const ENUM_MAX_ITEMS: usize = 12;
const SELECTION_PARAMS_PER_RETICULATION: f64 = 3.0;
const SELECTION_GAIN_THRESHOLD: f64 = 50.0;

// ── Python-side lookups ────────────────────────────────────────────────

static DIMPLE_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static GENE_TREES_CLS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static COALESCENT_UNIT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static GU_DISPLAYED_TREES: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static GU_ADD_HYBRID: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static GU_SIGNATURE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static GU_GROUP: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static GU_COUNT_RETICS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

fn cached<'py>(
    py: Python<'py>,
    cell: &'static PyOnceLock<Py<PyAny>>,
    module: &str,
    name: &str,
) -> PyResult<&'py Bound<'py, PyAny>> {
    cell.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py.import(module)?.getattr(name)?.unbind())
    })
    .map(|o| o.bind(py))
}

/// ``phynetpy._dimple.DimpleError(msg)``.
fn dimple_error(py: Python<'_>, msg: impl Into<String>) -> PyErr {
    let ty = DIMPLE_ERROR.get_or_try_init(py, || -> PyResult<Py<PyType>> {
        Ok(py
            .import("phynetpy._dimple")?
            .getattr("DimpleError")?
            .cast_into::<PyType>()?
            .unbind())
    });
    match ty {
        Ok(t) => PyErr::from_type(t.bind(py).clone(), msg.into()),
        Err(e) => e,
    }
}

fn gene_trees_class(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    cached(py, &GENE_TREES_CLS, "phynetpy.data", "GeneTrees")
}

fn coalescent_unit(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    COALESCENT_UNIT
        .get_or_try_init(py, || -> PyResult<Py<PyAny>> {
            Ok(py
                .import("phynetpy._units")?
                .getattr("BranchLengthUnit")?
                .getattr("COALESCENT_2N")?
                .unbind())
        })
        .map(|o| o.bind(py))
}

/// ``GraphUtils.get_all_subtrees``: one tree per hybrid-edge choice.
fn get_all_subtrees<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Vec<Bound<'py, Network>>> {
    let f = cached(py, &GU_DISPLAYED_TREES, "phynetpy._core", "_gu_get_displayed_trees")?;
    let out = f.call1((net, false))?;
    let mut trees = Vec::new();
    for item in out.try_iter()? {
        trees.push(net_of(&item?)?);
    }
    Ok(trees)
}

// ── Conversions ────────────────────────────────────────────────────────

fn net_of<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Network>> {
    obj.cast::<Network>()
        .map(|b| b.clone())
        .map_err(|_| PyTypeError::new_err(format!("expected a Network, got {}", crate::util::type_name(obj))))
}

fn labels_from_any(obj: &Bound<'_, PyAny>) -> PyResult<LabelSet> {
    let mut out = LabelSet::new();
    for item in obj.try_iter()? {
        out.insert(item?.extract::<String>()?);
    }
    Ok(out)
}

fn label_vec_from_any(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(item?.extract::<String>()?);
    }
    Ok(out)
}

fn frozenset<'py>(py: Python<'py>, labels: &LabelSet) -> PyResult<Bound<'py, PyFrozenSet>> {
    PyFrozenSet::new(py, labels.iter().map(|l| PyString::new(py, l)))
}

fn str_list<'py>(py: Python<'py>, labels: impl IntoIterator<Item = impl AsRef<str>>) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, labels.into_iter().map(|l| PyString::new(py, l.as_ref())))
}

fn intersect(a: &LabelSet, b: &LabelSet) -> LabelSet {
    a.intersection(b).cloned().collect()
}

fn minus(a: &LabelSet, b: &LabelSet) -> LabelSet {
    a.difference(b).cloned().collect()
}

fn overlaps(a: &LabelSet, b: &LabelSet) -> bool {
    a.iter().any(|l| b.contains(l))
}

/// ``repr(sorted(labels))`` as Python prints a list of strings.
fn repr_sorted(py: Python<'_>, labels: &LabelSet) -> PyResult<String> {
    Ok(str_list(py, labels.iter())?.repr()?.to_string())
}

fn join_first(labels: impl IntoIterator<Item = impl AsRef<str>>, limit: usize) -> String {
    labels
        .into_iter()
        .take(limit)
        .map(|l| l.as_ref().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

// ── Small graph helpers ────────────────────────────────────────────────

fn leaf_labels(py: Python<'_>, net: &Network) -> LabelSet {
    net.leaf_uids().into_iter().map(|u| net.label(py, u)).collect()
}

fn labels_under(py: Python<'_>, net: &Network, uid: u64) -> LabelSet {
    net.leaf_descendants_uid(uid).into_iter().map(|u| net.label(py, u)).collect()
}

/// ``_label_map``: leaf labels under every node reachable from the root.
fn label_map(py: Python<'_>, net: &Network) -> PyResult<IndexMap<u64, LabelSet, FxBuildHasher>> {
    let mut out: IndexMap<u64, LabelSet, FxBuildHasher> = IndexMap::default();
    for (uid, leaves) in net.leaf_descendants_all_map(py)? {
        out.insert(uid, leaves.into_iter().map(|l| net.label(py, l)).collect());
    }
    Ok(out)
}

fn copy_net<'py>(net: &Bound<'py, Network>) -> PyResult<Bound<'py, Network>> {
    Ok(Network::copy_impl(net)?.0)
}

/// ``Network.copy`` plus the old-uid -> new-uid map.
fn copy_with_map<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<(Bound<'py, Network>, FxHashMap<u64, u64>)> {
    let (copy, pairs) = Network::copy_impl(net)?;
    let mut map: FxHashMap<u64, u64> = FxHashMap::default();
    for (old, new) in pairs {
        map.insert(old.borrow(py).uid, new.borrow(py).uid);
    }
    Ok((copy, map))
}

fn node_by_name(net: &Network, name: &str) -> Option<u64> {
    net.names.get(name).copied()
}

fn count_retic_flags(py: Python<'_>, net: &Network) -> usize {
    net.nodes.values().filter(|e| e.obj.borrow(py).is_retic).count()
}

fn edge_length(py: Python<'_>, net: &Network, euid: u64) -> Option<f64> {
    net.edge_obj(euid).borrow(py).length
}

fn edge_gamma(py: Python<'_>, net: &Network, euid: u64) -> Option<f64> {
    net.edge_obj(euid).borrow(py).gamma
}

/// ``Network.get_edge(src, dest)``: the edge of a bubble picked without a
/// gamma, with the reference's lookup error when there is none.
fn edge_uid_between(py: Python<'_>, net: &Network, su: u64, du: u64) -> PyResult<u64> {
    Ok(net.edge_between(py, su, du, None, None)?.borrow(py).uid)
}

fn make_node(py: Python<'_>, name: String) -> PyResult<Py<Node>> {
    Py::new(py, Node::make(py, name, false, None, None, None))
}

/// ``Edge(src, dest, length=length)`` as the Python constructor builds it.
fn make_edge(py: Python<'_>, src: Py<Node>, dest: Py<Node>, length: Option<f64>) -> PyResult<Py<Edge>> {
    Py::new(
        py,
        Edge::make(py, src, dest, Some(length.unwrap_or(1.0)), Some(0.0), Some(0.0), Some(DEFAULT_TAG.to_string())),
    )
}

fn add_node(net: &Bound<'_, Network>, node: &Py<Node>) -> PyResult<()> {
    let py = net.py();
    net.borrow_mut().add_node(net.as_any(), node.bind(py))?;
    Ok(())
}

fn add_edge(net: &Bound<'_, Network>, edge: &Py<Edge>) -> PyResult<()> {
    let py = net.py();
    net.borrow_mut().add_edge(net.as_any(), edge.bind(py))
}

fn remove_edge(net: &Bound<'_, Network>, euid: u64) {
    let py = net.py();
    let mut this = net.borrow_mut();
    if this.edges.contains_key(&euid) {
        this.remove_edge_uid(py, euid);
    }
}

fn remove_node(net: &Bound<'_, Network>, uid: u64) {
    let py = net.py();
    let mut this = net.borrow_mut();
    if this.nodes.contains_key(&uid) {
        this.remove_node_uid(py, uid);
    }
}

fn set_unit_coalescent(net: &Bound<'_, Network>) -> PyResult<()> {
    let py = net.py();
    net.call_method1("set_branch_length_unit", (coalescent_unit(py)?,))?;
    Ok(())
}

/// ``_mcmc_gt._populate_default_branch_lengths``: missing or non-positive
/// lengths become 1, a missing gamma into a reticulation becomes 0.5.
fn populate_default_branch_lengths(py: Python<'_>, net: &Bound<'_, Network>) -> PyResult<()> {
    let edges: Vec<Py<Edge>> = net.borrow().edges.values().map(|e| e.obj.clone_ref(py)).collect();
    for edge in edges {
        let bound = edge.bind(py);
        let (length, gamma, dest_retic) = {
            let e = bound.borrow();
            let dest_retic = e.dest.borrow(py).is_retic;
            (e.length, e.gamma, dest_retic)
        };
        if length.map_or(true, |l| l <= 0.0) {
            Edge::set_length(bound, Some(1.0), false, false)?;
        }
        if gamma.is_none() && dest_retic {
            Edge::set_gamma(bound, 0.5)?;
        }
    }
    Ok(())
}

/// ``Network.graft_subtree(subtree, sibling_of=anchor)`` or ``under=anchor``;
/// returns the image of the subtree's root.
fn graft_subtree(net: &Bound<'_, Network>, subtree: &Bound<'_, Network>, anchor: u64, as_sibling: bool) -> PyResult<Py<Node>> {
    let py = net.py();
    if !net.borrow().nodes.contains_key(&anchor) {
        return Err(network_error(py, "Graft anchor is not in the network."));
    }
    let image = Network::absorb_subtree(net, subtree)?;
    if !as_sibling {
        let edge = {
            let this = net.borrow();
            make_edge(py, this.node_obj(anchor).clone_ref(py), image.clone_ref(py), None)?
        };
        add_edge(net, &edge)?;
        return Ok(image);
    }
    let parents = net.borrow().parents(anchor);
    if parents.is_empty() {
        let (label, anchor_obj) = {
            let this = net.borrow();
            (this.fresh_label(py, "R"), this.node_obj(anchor).clone_ref(py))
        };
        let new_root = make_node(py, label)?;
        let e1 = make_edge(py, new_root.clone_ref(py), anchor_obj, None)?;
        let e2 = make_edge(py, new_root.clone_ref(py), image.clone_ref(py), None)?;
        add_node(net, &new_root)?;
        add_edge(net, &e1)?;
        add_edge(net, &e2)?;
        return Ok(image);
    }
    let parent = parents[0];
    let (euid, length, gamma, child_is_retic, parent_obj, anchor_obj) = {
        let this = net.borrow();
        let euid = this.first_edge_between(py, parent, anchor)?;
        let (l, g) = {
            let e = this.edges[&euid].obj.borrow(py);
            (e.length, e.gamma)
        };
        let child_is_retic = this.node_obj(anchor).borrow(py).is_retic;
        (
            euid,
            l,
            g,
            child_is_retic,
            this.node_obj(parent).clone_ref(py),
            this.node_obj(anchor).clone_ref(py),
        )
    };
    remove_edge(net, euid);
    let mid_label = net.borrow().fresh_label(py, "I");
    let mid = make_node(py, mid_label)?;
    let upper = make_edge(py, parent_obj, mid.clone_ref(py), length)?;
    let lower = make_edge(py, mid.clone_ref(py), anchor_obj, length)?;
    if child_is_retic {
        if let Some(g) = gamma {
            if g != 0.0 {
                Edge::set_gamma(lower.bind(py), g)?;
            }
        }
    }
    let link = make_edge(py, mid.clone_ref(py), image.clone_ref(py), None)?;
    add_node(net, &mid)?;
    add_edge(net, &upper)?;
    add_edge(net, &lower)?;
    add_edge(net, &link)?;
    Ok(image)
}

/// ``GraphUtils._induced_subnetwork``: the part of ``net`` at or below
/// MRCA(targets) that leads to a target, copied into a fresh network.
fn induced_impl<'py>(py: Python<'py>, net: &Bound<'py, Network>, targets: &[u64]) -> PyResult<Bound<'py, Network>> {
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
            let edge = Py::new(
                py,
                Edge::make(
                    py,
                    old_new[&cur].clone_ref(py),
                    old_new[&child].clone_ref(py),
                    Some(1.0),
                    Some(0.0),
                    Some(0.0),
                    Some(DEFAULT_TAG.to_string()),
                ),
            )?;
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

/// ``GraphUtils.induced_subnetwork_by_taxa(net, taxa)``.
fn induced_by_taxa<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
    taxa: impl IntoIterator<Item = impl AsRef<str>>,
) -> PyResult<Bound<'py, Network>> {
    let targets = {
        let this = net.borrow();
        let mut out = Vec::new();
        let mut seen: FxHashSet<u64> = FxHashSet::default();
        for name in taxa {
            match this.names.get(name.as_ref()) {
                Some(&u) => {
                    if seen.insert(u) {
                        out.push(u);
                    }
                }
                None => {
                    return Err(network_error(
                        py,
                        format!("Leaf '{}' not found in network", name.as_ref()),
                    ))
                }
            }
        }
        out
    };
    let sub = induced_impl(py, net, &targets)?;
    Network::clean_impl(&sub, [false, false, true])?;
    Ok(sub)
}

/// ``_is_binary_tree``.
fn is_binary_tree(py: Python<'_>, net: &Network) -> bool {
    if !net.is_acyclic_impl() {
        return false;
    }
    for (&uid, entry) in net.nodes.iter() {
        if net.in_deg(uid) > 1 || entry.obj.borrow(py).is_retic {
            return false;
        }
        let out = net.out_deg(uid);
        if out == 0 {
            continue;
        }
        if out != 2 {
            return false;
        }
    }
    true
}

/// ``_blobs``: polytomies, in vertex order.
fn blob_uids(net: &Network) -> Vec<u64> {
    net.nodes.keys().copied().filter(|&u| net.out_deg(u) > 2).collect()
}

/// ``_node_with_leaves``: the node whose descendant leaf set is exactly
/// ``leaves``, in the order ``leaf_descendants_all`` lists nodes.
fn node_with_leaves(py: Python<'_>, net: &Network, leaves: &LabelSet) -> PyResult<Option<u64>> {
    for (uid, below) in label_map(py, net)? {
        if &below == leaves {
            return Ok(Some(uid));
        }
    }
    Ok(None)
}

/// ``_below_retics``: walk down through stacked reticulations.
fn below_retics(net: &Network, node: u64) -> u64 {
    let mut current = node;
    while net.in_deg(current) > 1 {
        let children = net.children(current);
        if children.len() != 1 {
            break;
        }
        current = children[0];
    }
    current
}

/// ``_detach``: cut ``parent -> child`` and return the clade under ``child``.
fn detach<'py>(py: Python<'py>, net: &Bound<'py, Network>, parent: u64, child: u64) -> PyResult<Bound<'py, Network>> {
    let (cut_labels, doomed, euid) = {
        let this = net.borrow();
        (labels_under(py, &this, child), this.subtree_uids(child), edge_uid_between(py, &this, parent, child)?)
    };
    let clade = induced_by_taxa(py, net, cut_labels.iter())?;
    remove_edge(net, euid);
    for uid in doomed {
        remove_node(net, uid);
    }
    Network::clean_impl(net, [false, false, true])?;
    Ok(clade)
}

/// ``_ensure_internal_labels``: name every unlabeled node ``B<n>``.
fn ensure_internal_labels(py: Python<'_>, net: &Bound<'_, Network>) -> PyResult<()> {
    let uids: Vec<u64> = net.borrow().nodes.keys().copied().collect();
    for uid in uids {
        let (present, empty) = {
            let this = net.borrow();
            match this.nodes.get(&uid) {
                Some(e) => (true, e.obj.borrow(py).name.is_empty()),
                None => (false, false),
            }
        };
        if present && empty {
            let label = net.borrow().fresh_label(py, "B");
            net.borrow_mut().rename_uid(py, uid, label);
        }
    }
    Ok(())
}

/// ``_mrca_or_none``: the MRCA of ``leaves`` on ``net``, or None when a
/// label is absent.  Every other failure propagates, as the reference's
/// ``except NetworkError`` let it.
fn mrca_or_none(py: Python<'_>, net: &Network, leaves: &LabelSet) -> PyResult<Option<u64>> {
    if leaves.is_empty() {
        return Ok(None);
    }
    let mut targets = Vec::with_capacity(leaves.len());
    for label in leaves {
        match node_by_name(net, label) {
            Some(u) => targets.push(u),
            None => return Ok(None),
        }
    }
    Ok(Some(net.mrca_uids(py, &targets)?))
}

/// ``_clade_node``: the exact clade node, or the MRCA.
fn clade_node(py: Python<'_>, net: &Network, leaves: &LabelSet) -> PyResult<Option<u64>> {
    if let Some(node) = node_with_leaves(py, net, leaves)? {
        return Ok(Some(node));
    }
    mrca_or_none(py, net, leaves)
}

/// ``Network.dist_from_root``.
fn dist_from_root(py: Python<'_>, net: &Network, uid: u64) -> PyResult<i64> {
    let root = net.root_uid(py)?;
    let (_, dist) = net.bfs_order(root, false);
    match dist.get(&uid) {
        Some(d) => Ok(*d),
        None => Err(PyKeyError::new_err(net.node_obj(uid).clone_ref(py).into_any())),
    }
}

/// ``GraphUtils.rooted_triples``: every ``(a, b | c)`` with ``a < b``,
/// emitted at the MRCA of the pair.
fn rooted_triples(py: Python<'_>, net: &Network) -> PyResult<FxHashSet<Triple>> {
    let mut leaves: Vec<(String, u64)> = net.leaf_uids().into_iter().map(|u| (net.label(py, u), u)).collect();
    leaves.sort();
    let n = leaves.len();
    let index: FxHashMap<u64, usize> = leaves.iter().enumerate().map(|(i, (_, u))| (*u, i)).collect();
    let desc: FxHashMap<u64, Vec<usize>> = net
        .leaf_descendants_all_map(py)?
        .iter()
        .map(|(u, ls)| (*u, ls.iter().map(|l| index[l]).collect()))
        .collect();
    let mut triples: FxHashSet<(usize, usize, usize)> = FxHashSet::default();
    let mut in_below = vec![false; n];
    for (uid, entry) in net.nodes.iter() {
        let kids: Vec<u64> = net
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
    Ok(triples
        .into_iter()
        .map(|(x, y, c)| (leaves[x].0.clone(), leaves[y].0.clone(), leaves[c].0.clone()))
        .collect())
}

/// Read ``{(a, b, c): count}`` from a Python mapping.
fn triple_counts(obj: &Bound<'_, PyAny>) -> PyResult<IndexMap<Triple, i64, FxBuildHasher>> {
    let mut out: IndexMap<Triple, i64, FxBuildHasher> = IndexMap::default();
    let dict = obj.cast::<PyDict>()?;
    for (key, value) in dict.iter() {
        let triple: (String, String, String) = key.extract()?;
        out.insert(triple, value.extract::<i64>()?);
    }
    Ok(out)
}

fn triple_dict<'py>(py: Python<'py>, counts: &IndexMap<Triple, i64, FxBuildHasher>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for ((a, b, c), count) in counts {
        out.set_item(PyTuple::new(py, [a, b, c])?, *count)?;
    }
    Ok(out)
}

// ── Small graph helpers exposed to Python ──────────────────────────────

#[pyfunction]
#[pyo3(name = "_dp_labels", signature = (net, node=None))]
fn labels_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, node: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyFrozenSet>> {
    let this = net.borrow();
    let labels = match node {
        None => leaf_labels(py, &this),
        Some(node) => {
            let uid = this.uid_of_any(node).ok_or_else(|| network_error(py, "Node not found in graph."))?;
            labels_under(py, &this, uid)
        }
    };
    frozenset(py, &labels)
}

#[pyfunction]
#[pyo3(name = "_dp_label_map")]
fn label_map_py<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PyDict>> {
    let this = net.borrow();
    let out = PyDict::new(py);
    for (uid, labels) in label_map(py, &this)? {
        out.set_item(this.node_obj(uid).bind(py), frozenset(py, &labels)?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_dp_copy")]
fn copy_py<'py>(net: &Bound<'py, Network>) -> PyResult<Bound<'py, Network>> {
    copy_net(net)
}

#[pyfunction]
#[pyo3(name = "_dp_blobs")]
fn blobs_py<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PyList>> {
    let this = net.borrow();
    PyList::new(py, blob_uids(&this).into_iter().map(|u| this.node_obj(u).bind(py)))
}

#[pyfunction]
#[pyo3(name = "_dp_is_binary_tree")]
fn is_binary_tree_py(py: Python<'_>, net: &Bound<'_, Network>) -> bool {
    is_binary_tree(py, &net.borrow())
}

#[pyfunction]
#[pyo3(name = "_dp_node_with_leaves")]
fn node_with_leaves_py(py: Python<'_>, net: &Bound<'_, Network>, leaves: &Bound<'_, PyAny>) -> PyResult<Option<Py<Node>>> {
    let this = net.borrow();
    Ok(node_with_leaves(py, &this, &labels_from_any(leaves)?)?.map(|u| this.node_obj(u).clone_ref(py)))
}

#[pyfunction]
#[pyo3(name = "_dp_below_retics")]
fn below_retics_py(py: Python<'_>, net: &Bound<'_, Network>, node: &Bound<'_, PyAny>) -> PyResult<Py<Node>> {
    let this = net.borrow();
    match this.uid_of_any(node) {
        // A node outside the graph has in-degree 0 there: it is returned as is.
        None => Ok(node.cast::<Node>()?.clone().unbind()),
        Some(uid) => Ok(this.node_obj(below_retics(&this, uid)).clone_ref(py)),
    }
}

#[pyfunction]
#[pyo3(name = "_dp_detach")]
fn detach_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, parent: &Bound<'py, PyAny>, child: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Network>> {
    let (p, c) = {
        let this = net.borrow();
        (
            this.uid_of_any(parent).ok_or_else(|| network_error(py, "Node not found in graph."))?,
            this.uid_of_any(child).ok_or_else(|| network_error(py, "Node not found in graph."))?,
        )
    };
    detach(py, net, p, c)
}

#[pyfunction]
#[pyo3(name = "_dp_ensure_internal_labels")]
fn ensure_internal_labels_py(py: Python<'_>, net: &Bound<'_, Network>) -> PyResult<()> {
    ensure_internal_labels(py, net)
}

#[pyfunction]
#[pyo3(name = "_dp_mrca_or_none")]
fn mrca_or_none_py(py: Python<'_>, net: &Bound<'_, Network>, leaves: &Bound<'_, PyAny>) -> PyResult<Option<Py<Node>>> {
    let this = net.borrow();
    Ok(mrca_or_none(py, &this, &labels_from_any(leaves)?)?.map(|u| this.node_obj(u).clone_ref(py)))
}

#[pyfunction]
#[pyo3(name = "_dp_clade_node")]
fn clade_node_py(py: Python<'_>, net: &Bound<'_, Network>, leaves: &Bound<'_, PyAny>) -> PyResult<Option<Py<Node>>> {
    let this = net.borrow();
    Ok(clade_node(py, &this, &labels_from_any(leaves)?)?.map(|u| this.node_obj(u).clone_ref(py)))
}

#[pyfunction]
#[pyo3(name = "_dp_strip_outgroup")]
fn strip_outgroup<'py>(py: Python<'py>, net: &Bound<'py, Network>, outgroup: Option<String>) -> PyResult<Bound<'py, Network>> {
    let Some(outgroup) = outgroup else { return copy_net(net) };
    let keep: Vec<String> = {
        let this = net.borrow();
        if node_by_name(&this, &outgroup).is_none() {
            drop(this);
            return copy_net(net);
        }
        this.leaf_uids()
            .into_iter()
            .map(|u| this.label(py, u))
            .filter(|l| l != &outgroup)
            .collect()
    };
    if keep.len() < 3 {
        return Err(dimple_error(
            py,
            format!("Removing outgroup '{outgroup}' leaves fewer than 3 taxa."),
        ));
    }
    let stripped = induced_by_taxa(py, net, keep.iter())?;
    Network::clean_impl(&stripped, [true, true, true])?;
    Ok(stripped)
}

// ── Distances ──────────────────────────────────────────────────────────

/// ``_leaf_path_lengths``: one post-order pass; a missing edge length is 1.
fn leaf_path_lengths(py: Python<'_>, tree: &Network, wanted: &LabelSet) -> PyResult<Vec<((String, String), f64)>> {
    let root = match tree.nodes.keys().copied().find(|&u| tree.in_deg(u) == 0) {
        Some(r) => r,
        None => return Err(PyStopIteration::new_err(())),
    };
    let mut order: Vec<u64> = Vec::new();
    let mut stack: Vec<(u64, bool)> = vec![(root, false)];
    while let Some((node, done)) = stack.pop() {
        if done {
            order.push(node);
            continue;
        }
        stack.push((node, true));
        for child in tree.children(node) {
            stack.push((child, false));
        }
    }
    let mut below: FxHashMap<u64, Vec<(String, f64)>> = FxHashMap::default();
    let mut lengths: Vec<((String, String), f64)> = Vec::new();
    for node in order {
        let children = tree.children(node);
        if children.is_empty() {
            let label = tree.label(py, node);
            below.insert(node, if wanted.contains(&label) { vec![(label, 0.0)] } else { vec![] });
            continue;
        }
        let mut merged: Vec<(String, f64)> = Vec::new();
        for child in children {
            let euid = edge_uid_between(py, tree, node, child)?;
            let length = edge_length(py, tree, euid).unwrap_or(1.0);
            let lifted: Vec<(String, f64)> = below
                .remove(&child)
                .unwrap_or_default()
                .into_iter()
                .map(|(label, dist)| (label, dist + length))
                .collect();
            for (left, left_dist) in &merged {
                for (right, right_dist) in &lifted {
                    let key = if left < right {
                        (left.clone(), right.clone())
                    } else {
                        (right.clone(), left.clone())
                    };
                    lengths.push((key, left_dist + right_dist));
                }
            }
            merged.extend(lifted);
        }
        below.insert(node, merged);
    }
    Ok(lengths)
}

#[pyfunction]
#[pyo3(name = "_dp_leaf_path_lengths")]
fn leaf_path_lengths_py<'py>(py: Python<'py>, tree: &Bound<'py, Network>, wanted: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for ((a, b), d) in leaf_path_lengths(py, &tree.borrow(), &labels_from_any(wanted)?)? {
        out.set_item(PyTuple::new(py, [a, b])?, d)?;
    }
    Ok(out)
}

/// Every gene tree of a ``GeneTrees`` object, in its iteration order.
fn gene_tree_list<'py>(gene_trees: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, Network>>> {
    let mut out = Vec::new();
    for tree in gene_trees.getattr("trees")?.try_iter()? {
        out.push(net_of(&tree?)?);
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_dp_gene_tree_distance_matrix", signature = (gene_trees, taxa, strict=true))]
fn gene_tree_distance_matrix<'py>(
    py: Python<'py>,
    gene_trees: &Bound<'py, PyAny>,
    taxa: &Bound<'py, PyAny>,
    strict: bool,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyArray2<f64>>)> {
    let wanted = labels_from_any(taxa)?;
    let labels: Vec<String> = wanted.iter().cloned().collect();
    let index: FxHashMap<&str, usize> = labels.iter().enumerate().map(|(i, l)| (l.as_str(), i)).collect();
    let size = labels.len();
    let mut total = vec![0.0_f64; size * size];
    let mut counts = vec![0.0_f64; size * size];
    for tree in gene_tree_list(gene_trees)? {
        let this = tree.borrow();
        let present = intersect(&leaf_labels(py, &this), &wanted);
        if present.len() < 2 {
            continue;
        }
        for ((left, right), distance) in leaf_path_lengths(py, &this, &wanted)? {
            let (i, j) = (index[left.as_str()], index[right.as_str()]);
            total[i * size + j] += distance;
            total[j * size + i] += distance;
            counts[i * size + j] += 1.0;
            counts[j * size + i] += 1.0;
        }
    }
    let mut never = vec![false; size * size];
    let mut never_count = 0usize;
    for i in 0..size {
        for j in 0..size {
            if i != j && counts[i * size + j] == 0.0 {
                never[i * size + j] = true;
                never_count += 1;
            }
        }
    }
    if strict && never_count > 0 {
        return Err(dimple_error(
            py,
            format!(
                "{} taxon pair(s) never co-occur in a gene tree; their distance is undefined.",
                never_count / 2
            ),
        ));
    }
    let mut matrix = vec![0.0_f64; size * size];
    for k in 0..size * size {
        matrix[k] = total[k] / counts[k].max(1.0);
        if never[k] {
            matrix[k] = f64::NAN;
        }
    }
    for i in 0..size {
        matrix[i * size + i] = 0.0;
    }
    let arr = Array2::from_shape_vec((size, size), matrix)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((str_list(py, labels.iter())?, PyArray2::from_owned_array(py, arr)))
}

/// numpy's pairwise summation of a contiguous float64 vector, so a mean
/// over many values rounds exactly as ``np.mean`` does.
fn numpy_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0_f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        numpy_sum(&a[..n2]) + numpy_sum(&a[n2..])
    }
}

/// ``numpy.asarray(matrix, dtype=float)`` read-only.
fn float_matrix<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<PyReadonlyArray2<'py, f64>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "float64")?;
    let arr = np.getattr("asarray")?.call((matrix,), Some(&kwargs))?;
    arr.extract::<PyReadonlyArray2<'py, f64>>()
}

#[pyfunction]
#[pyo3(name = "_dp_outgroup_distances", signature = (labels, matrix_labels, matrix, siblings, scale=None))]
fn outgroup_distances<'py>(
    py: Python<'py>,
    labels: &Bound<'py, PyAny>,
    matrix_labels: &Bound<'py, PyAny>,
    matrix: &Bound<'py, PyAny>,
    siblings: &Bound<'py, PyAny>,
    scale: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let labels = label_vec_from_any(labels)?;
    let matrix_labels = label_vec_from_any(matrix_labels)?;
    let index: FxHashMap<&str, usize> = matrix_labels.iter().enumerate().map(|(i, l)| (l.as_str(), i)).collect();
    // Sibling order is the caller's iteration order, as the reference read it.
    let sibs: Vec<usize> = label_vec_from_any(siblings)?
        .iter()
        .filter_map(|l| index.get(l.as_str()).copied())
        .collect();
    let matrix = float_matrix(py, matrix)?;
    let view = matrix.as_array();
    let out = PyDict::new(py);
    for label in &labels {
        let Some(&row) = index.get(label.as_str()) else { continue };
        if sibs.is_empty() {
            continue;
        }
        let values: Vec<f64> = sibs.iter().map(|&s| view[[row, s]]).collect();
        if !values.is_empty() && values.iter().all(|v| v.is_finite()) {
            let factor = scale.unwrap_or(OUTGROUP_SCALE);
            let mean = numpy_sum(&values) / values.len() as f64;
            out.set_item(label, factor * mean)?;
        }
    }
    Ok(out)
}

// ── Compatibility ──────────────────────────────────────────────────────

fn clade_sets(py: Python<'_>, net: &Network) -> PyResult<Vec<LabelSet>> {
    Ok(label_map(py, net)?.into_values().collect())
}

fn splits(clades: &[LabelSet], keep: &LabelSet) -> Vec<LabelSet> {
    let mut found = Vec::new();
    for clade in clades {
        let below = intersect(clade, keep);
        if 2 <= below.len() && below.len() + 2 <= keep.len() {
            found.push(below);
        }
    }
    found
}

fn trees_incompatible_impl(left_leaves: &LabelSet, right_leaves: &LabelSet, left_clades: &[LabelSet], right_clades: &[LabelSet]) -> bool {
    let shared = intersect(left_leaves, right_leaves);
    if shared.len() < 4 {
        return false;
    }
    let right_splits = splits(right_clades, &shared);
    for left_split in splits(left_clades, &shared) {
        let left_other = minus(&shared, &left_split);
        for right_split in &right_splits {
            let right_other = minus(&shared, right_split);
            if overlaps(&left_split, right_split)
                && overlaps(&left_split, &right_other)
                && overlaps(&left_other, right_split)
                && overlaps(&left_other, &right_other)
            {
                return true;
            }
        }
    }
    false
}

fn clades_from_any(obj: &Bound<'_, PyAny>) -> PyResult<Vec<LabelSet>> {
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(labels_from_any(&item?)?);
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_dp_clade_sets")]
fn clade_sets_py<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PyList>> {
    let sets = clade_sets(py, &net.borrow())?;
    PyList::new(py, sets.iter().map(|s| frozenset(py, s)).collect::<PyResult<Vec<_>>>()?)
}

#[pyfunction]
#[pyo3(name = "_dp_splits")]
fn splits_py<'py>(py: Python<'py>, clades: &Bound<'py, PyAny>, keep: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let found = splits(&clades_from_any(clades)?, &labels_from_any(keep)?);
    PyList::new(py, found.iter().map(|s| frozenset(py, s)).collect::<PyResult<Vec<_>>>()?)
}

#[pyfunction]
#[pyo3(name = "_dp_trees_incompatible", signature = (left, right, left_clades=None, right_clades=None))]
fn trees_incompatible<'py>(
    py: Python<'py>,
    left: &Bound<'py, Network>,
    right: &Bound<'py, Network>,
    left_clades: Option<&Bound<'py, PyAny>>,
    right_clades: Option<&Bound<'py, PyAny>>,
) -> PyResult<bool> {
    let (l, r) = (left.borrow(), right.borrow());
    let left_clades = match left_clades {
        Some(c) if !c.is_none() => clades_from_any(c)?,
        _ => clade_sets(py, &l)?,
    };
    let right_clades = match right_clades {
        Some(c) if !c.is_none() => clades_from_any(c)?,
        _ => clade_sets(py, &r)?,
    };
    Ok(trees_incompatible_impl(&leaf_labels(py, &l), &leaf_labels(py, &r), &left_clades, &right_clades))
}

fn tree_list<'py>(trees: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, Network>>> {
    let mut out = Vec::new();
    for item in trees.try_iter()? {
        out.push(net_of(&item?)?);
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_dp_greedy_compatible")]
fn greedy_compatible<'py>(py: Python<'py>, trees: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let trees = tree_list(trees)?;
    let leaves: Vec<LabelSet> = trees.iter().map(|t| leaf_labels(py, &t.borrow())).collect();
    let clades: Vec<Vec<LabelSet>> = trees.iter().map(|t| clade_sets(py, &t.borrow())).collect::<PyResult<_>>()?;
    let mut kept: Vec<usize> = Vec::new();
    for index in 0..trees.len() {
        if kept
            .iter()
            .all(|&earlier| !trees_incompatible_impl(&leaves[index], &leaves[earlier], &clades[index], &clades[earlier]))
        {
            kept.push(index);
        }
    }
    PyList::new(py, kept.into_iter().map(|i| trees[i].clone()))
}

/// A CPython 3.11 ``set`` of small non-negative integers, laid out exactly
/// as CPython lays it out (open addressing with linear probes and
/// perturbation, the same resize policy, dummies, and the ``pop`` finger),
/// so that iteration, ``pop`` and ``max`` visit elements in the order the
/// reference implementation did.
#[derive(Clone)]
struct PySetSim {
    table: Vec<Slot>,
    mask: usize,
    used: usize,
    fill: usize,
    finger: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    Empty,
    Dummy,
    /// ``(key, hash)``: ints hash to themselves; other objects carry their hash.
    Key(usize, usize),
}

const SET_MINSIZE: usize = 8;
const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;

impl PySetSim {
    fn new() -> Self {
        PySetSim { table: vec![Slot::Empty; SET_MINSIZE], mask: SET_MINSIZE - 1, used: 0, fill: 0, finger: 0 }
    }

    fn from_iter(keys: impl IntoIterator<Item = usize>) -> Self {
        let mut s = PySetSim::new();
        for k in keys {
            s.add(k);
        }
        s
    }

    fn len(&self) -> usize {
        self.used
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.table.iter().filter_map(|s| match s {
            Slot::Key(k, _) => Some(*k),
            _ => None,
        })
    }

    fn insert_clean(table: &mut [Slot], mask: usize, key: usize, hash: usize) {
        let mut perturb = hash;
        let mut i = hash & mask;
        loop {
            if table[i] == Slot::Empty {
                table[i] = Slot::Key(key, hash);
                return;
            }
            if i + LINEAR_PROBES <= mask {
                for j in 1..=LINEAR_PROBES {
                    if table[i + j] == Slot::Empty {
                        table[i + j] = Slot::Key(key, hash);
                        return;
                    }
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        }
    }

    fn resize(&mut self, minused: usize) {
        let mut newsize = SET_MINSIZE;
        while newsize <= minused {
            newsize <<= 1;
        }
        let old = std::mem::replace(&mut self.table, vec![Slot::Empty; newsize]);
        self.mask = newsize - 1;
        for slot in old {
            if let Slot::Key(k, h) = slot {
                PySetSim::insert_clean(&mut self.table, self.mask, k, h);
            }
        }
        self.fill = self.used;
    }

    /// ``set.add(key)`` for a small int: its hash is itself.
    fn add(&mut self, key: usize) {
        self.add_hashed(key, key);
    }

    /// ``set.add(obj)`` for an object with CPython hash ``hash``.
    fn add_hashed(&mut self, key: usize, hash: usize) {
        let mask = self.mask;
        let mut i = hash & mask;
        let mut perturb = hash;
        let mut freeslot: Option<usize> = None;
        loop {
            let probes = if i + LINEAR_PROBES <= mask { LINEAR_PROBES } else { 0 };
            let mut idx = i;
            for _ in 0..=probes {
                match self.table[idx] {
                    Slot::Empty => {
                        if let Some(f) = freeslot {
                            self.used += 1;
                            self.table[f] = Slot::Key(key, hash);
                            return;
                        }
                        self.fill += 1;
                        self.used += 1;
                        self.table[idx] = Slot::Key(key, hash);
                        if self.fill * 5 < mask * 3 {
                            return;
                        }
                        let minused = if self.used > 50000 { self.used * 2 } else { self.used * 4 };
                        self.resize(minused);
                        return;
                    }
                    Slot::Key(k, h) if h == hash && k == key => return,
                    Slot::Dummy => {
                        if freeslot.is_none() {
                            freeslot = Some(idx);
                        }
                    }
                    Slot::Key(_, _) => {}
                }
                idx += 1;
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        }
    }

    fn lookup(&self, key: usize) -> Option<usize> {
        let mask = self.mask;
        let mut i = key & mask;
        let mut perturb = key;
        loop {
            let probes = if i + LINEAR_PROBES <= mask { LINEAR_PROBES } else { 0 };
            let mut idx = i;
            for _ in 0..=probes {
                match self.table[idx] {
                    Slot::Empty => return None,
                    Slot::Key(k, h) if h == key && k == key => return Some(idx),
                    _ => {}
                }
                idx += 1;
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        }
    }

    fn contains(&self, key: usize) -> bool {
        self.lookup(key).is_some()
    }

    fn discard(&mut self, key: usize) {
        if let Some(idx) = self.lookup(key) {
            self.table[idx] = Slot::Dummy;
            self.used -= 1;
        }
    }

    fn pop(&mut self) -> Option<usize> {
        if self.used == 0 {
            return None;
        }
        let mut e = self.finger & self.mask;
        loop {
            if let Slot::Key(k, _) = self.table[e] {
                self.table[e] = Slot::Dummy;
                self.used -= 1;
                self.finger = e + 1;
                return Some(k);
            }
            e += 1;
            if e > self.mask {
                e = 0;
            }
        }
    }

    /// ``set.copy()``: ``set_merge`` into a fresh set.
    fn copy(&self) -> Self {
        let mut new = PySetSim::new();
        if self.used == 0 {
            return new;
        }
        if (new.fill + self.used) * 5 >= new.mask * 3 {
            new.resize((new.used + self.used) * 2);
        }
        if new.fill == 0 && new.mask == self.mask && self.fill == self.used {
            new.table = self.table.clone();
            new.fill = self.fill;
            new.used = self.used;
            return new;
        }
        new.fill = self.used;
        new.used = self.used;
        for slot in &self.table {
            if let Slot::Key(k, h) = slot {
                PySetSim::insert_clean(&mut new.table, new.mask, *k, *h);
            }
        }
        new
    }

    /// ``a & b``: iterate the smaller operand (the right one on a tie).
    fn intersection(&self, other: &PySetSim) -> Self {
        let (big, small) = if other.used > self.used { (other, self) } else { (self, other) };
        let mut result = PySetSim::new();
        for key in small.iter() {
            if big.contains(key) {
                result.add(key);
            }
        }
        result
    }

    fn common_count(&self, other: &PySetSim) -> usize {
        let (big, small) = if other.used > self.used { (other, self) } else { (self, other) };
        small.iter().filter(|k| big.contains(*k)).count()
    }

    /// ``a - b``.
    fn difference(&self, other: &PySetSim) -> Self {
        if (self.used >> 2) > other.used {
            let mut result = self.copy();
            for key in other.iter() {
                result.discard(key);
            }
            if result.fill - result.used > result.mask / 4 {
                let minused = if result.used > 50000 { result.used * 2 } else { result.used * 4 };
                result.resize(minused);
            }
            return result;
        }
        let mut result = PySetSim::new();
        for key in self.iter() {
            if !other.contains(key) {
                result.add(key);
            }
        }
        result
    }
}

/// ``networkx.find_cliques`` on a graph whose nodes are ``0..n`` and whose
/// adjacency sets were built in increasing neighbour order; yields the
/// maximal cliques in the order networkx reports them.
fn nx_find_cliques(n: usize, adj: &[PySetSim]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    if n == 0 {
        return out;
    }
    let mut cand = PySetSim::from_iter(0..n);
    if cand.len() == 0 {
        out.push(Vec::new());
        return out;
    }
    let pivot = |subg: &PySetSim, cand: &PySetSim| -> usize {
        let mut best: Option<(usize, usize)> = None;
        for u in subg.iter() {
            let score = cand.common_count(&adj[u]);
            if best.map_or(true, |(_, b)| score > b) {
                best = Some((u, score));
            }
        }
        best.map(|(u, _)| u).unwrap_or(0)
    };
    let mut subg = cand.copy();
    let mut stack: Vec<(PySetSim, PySetSim, PySetSim)> = Vec::new();
    let mut q: Vec<Option<usize>> = vec![None];
    let u = pivot(&subg, &cand);
    let mut ext_u = cand.difference(&adj[u]);
    loop {
        if ext_u.len() > 0 {
            let qn = ext_u.pop().expect("non-empty set pops");
            cand.discard(qn);
            if let Some(last) = q.last_mut() {
                *last = Some(qn);
            }
            let adj_q = &adj[qn];
            let subg_q = subg.intersection(adj_q);
            if subg_q.len() == 0 {
                out.push(q.iter().flatten().copied().collect());
            } else {
                let cand_q = cand.intersection(adj_q);
                if cand_q.len() > 0 {
                    stack.push((subg, cand, ext_u));
                    q.push(None);
                    subg = subg_q;
                    cand = cand_q;
                    let u = pivot(&subg, &cand);
                    ext_u = cand.difference(&adj[u]);
                }
            }
        } else {
            q.pop();
            match stack.pop() {
                Some((s, c, e)) => {
                    subg = s;
                    cand = c;
                    ext_u = e;
                }
                None => break,
            }
        }
    }
    out
}

/// The largest clique of a compatibility graph: the first of maximal size
/// that ``nx.find_cliques`` reports, sorted.
fn max_clique_indices(n: usize, compatible: &dyn Fn(usize, usize) -> bool) -> Vec<usize> {
    let mut adj: Vec<PySetSim> = (0..n).map(|_| PySetSim::new()).collect();
    for left in 0..n {
        for right in (left + 1)..n {
            if compatible(left, right) {
                adj[left].add(right);
                adj[right].add(left);
            }
        }
    }
    let cliques = nx_find_cliques(n, &adj);
    let mut best: Vec<usize> = Vec::new();
    for clique in cliques {
        if clique.len() > best.len() {
            best = clique;
        }
    }
    best.sort_unstable();
    best
}

#[pyfunction]
#[pyo3(name = "_dp_max_clique_indices")]
fn max_clique_indices_py(n: usize, edges: Vec<(usize, usize)>) -> Vec<usize> {
    let set: FxHashSet<(usize, usize)> = edges
        .into_iter()
        .map(|(a, b)| if a < b { (a, b) } else { (b, a) })
        .collect();
    max_clique_indices(n, &|a, b| set.contains(&(a.min(b), a.max(b))))
}

#[pyfunction]
#[pyo3(name = "_dp_max_clique_compatible")]
fn max_clique_compatible<'py>(py: Python<'py>, trees: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let trees = tree_list(trees)?;
    if trees.len() <= 1 {
        return PyList::new(py, trees);
    }
    let leaves: Vec<LabelSet> = trees.iter().map(|t| leaf_labels(py, &t.borrow())).collect();
    let clades: Vec<Vec<LabelSet>> = trees.iter().map(|t| clade_sets(py, &t.borrow())).collect::<PyResult<_>>()?;
    let n = trees.len();
    let mut compatible = vec![vec![false; n]; n];
    for left in 0..n {
        for right in (left + 1)..n {
            let ok = !trees_incompatible_impl(&leaves[left], &leaves[right], &clades[left], &clades[right]);
            compatible[left][right] = ok;
            compatible[right][left] = ok;
        }
    }
    let best = max_clique_indices(n, &|a, b| compatible[a][b]);
    PyList::new(py, best.into_iter().map(|i| trees[i].clone()))
}

// ── Neighbor-joining constraint ────────────────────────────────────────

type NameSet = IndexSet<String, FxBuildHasher>;

/// Undirected, degree-2-free tree of current cluster labels.
///
/// A cherry is two labels that share a neighbour. Neighbor-joining may join
/// two labels that both occur in this tree only when they are a cherry.
#[pyclass(name = "_Constraint", module = "phynetpy._dimple")]
pub struct Constraint {
    leaves: NameSet,
    adj: IndexMap<String, NameSet, FxBuildHasher>,
    broken: usize,
}

impl Constraint {
    fn from_network(py: Python<'_>, net: &Network) -> Self {
        let mut c = Constraint { leaves: NameSet::default(), adj: IndexMap::default(), broken: 0 };
        c.build(py, net);
        c
    }

    /// Copy ``net`` as an undirected graph and suppress every non-leaf of
    /// degree <= 2.
    fn build(&mut self, py: Python<'_>, net: &Network) {
        let mut names: FxHashMap<u64, String> = FxHashMap::default();
        let mut adj: IndexMap<String, NameSet, FxBuildHasher> = IndexMap::default();
        for (&uid, entry) in net.nodes.iter() {
            let name = if net.out_deg(uid) == 0 {
                let label = entry.obj.borrow(py).name.clone();
                self.leaves.insert(label.clone());
                label
            } else {
                format!("u{uid}")
            };
            adj.entry(name.clone()).or_default();
            names.insert(uid, name);
        }
        for entry in net.edges.values() {
            let (s, d) = (&names[&entry.su], &names[&entry.du]);
            adj.get_mut(s).map(|set| set.insert(d.clone()));
            adj.get_mut(d).map(|set| set.insert(s.clone()));
        }
        self.adj = adj;
        let mut changed = true;
        while changed {
            changed = false;
            let keys: Vec<String> = self.adj.keys().cloned().collect();
            for node in keys {
                if self.leaves.contains(&node) || self.adj[&node].len() > 2 {
                    continue;
                }
                self.suppress(&node);
                changed = true;
                break;
            }
        }
    }

    fn neighbours(&self, node: &str) -> NameSet {
        self.adj.get(node).cloned().unwrap_or_default()
    }

    /// Every unordered pair of leaves that currently shares a neighbour.
    fn cherry_pairs(&self) -> FxHashSet<(String, String)> {
        let mut pairs: FxHashSet<(String, String)> = FxHashSet::default();
        let key = |a: &str, b: &str| -> (String, String) {
            if a < b {
                (a.to_string(), b.to_string())
            } else {
                (b.to_string(), a.to_string())
            }
        };
        if self.leaves.len() == 2 {
            let mut it = self.leaves.iter();
            let (a, b) = (it.next().unwrap(), it.next().unwrap());
            pairs.insert(key(a, b));
        }
        for (node, neighbours) in &self.adj {
            let leaf_neighbours: Vec<&String> = neighbours.iter().filter(|n| self.leaves.contains(*n)).collect();
            if self.leaves.contains(node) {
                for leaf in leaf_neighbours {
                    pairs.insert(key(node, leaf));
                }
            } else {
                for i in 0..leaf_neighbours.len() {
                    for j in (i + 1)..leaf_neighbours.len() {
                        pairs.insert(key(leaf_neighbours[i], leaf_neighbours[j]));
                    }
                }
            }
        }
        pairs
    }

    /// Splice out a non-leaf of degree 2, or delete one of degree <= 1.
    fn suppress(&mut self, node: &str) {
        let Some(neighbours) = self.adj.get(node).cloned() else { return };
        if self.leaves.contains(node) {
            return;
        }
        if neighbours.len() == 2 {
            let left = neighbours[0].clone();
            let right = neighbours[1].clone();
            if let Some(set) = self.adj.get_mut(&left) {
                set.shift_remove(node);
            }
            if let Some(set) = self.adj.get_mut(&right) {
                set.shift_remove(node);
            }
            self.adj.entry(left.clone()).or_default().insert(right.clone());
            self.adj.entry(right).or_default().insert(left);
            self.adj.shift_remove(node);
        } else if neighbours.len() <= 1 {
            for neighbour in neighbours.iter() {
                if let Some(set) = self.adj.get_mut(neighbour) {
                    set.shift_remove(node);
                }
            }
            self.adj.shift_remove(node);
        }
    }

    fn rename_impl(&mut self, old: &str, new: &str) {
        if !self.leaves.contains(old) {
            return;
        }
        let neighbours = self.adj.shift_remove(old).unwrap_or_default();
        self.adj.insert(new.to_string(), neighbours.clone());
        for neighbour in neighbours.iter() {
            if let Some(set) = self.adj.get_mut(neighbour) {
                set.shift_remove(old);
                set.insert(new.to_string());
            }
        }
        self.leaves.shift_remove(old);
        self.leaves.insert(new.to_string());
    }

    fn remove_leaf_impl(&mut self, label: &str) {
        if !self.leaves.contains(label) {
            return;
        }
        let neighbours = self.adj.shift_remove(label).unwrap_or_default();
        self.leaves.shift_remove(label);
        for neighbour in neighbours.iter() {
            if let Some(set) = self.adj.get_mut(neighbour) {
                set.shift_remove(label);
            }
        }
        let mut sorted: Vec<String> = neighbours.into_iter().collect();
        sorted.sort();
        for neighbour in sorted {
            self.suppress(&neighbour);
        }
    }

    fn contract_impl(&mut self, left: &str, right: &str, new: &str) -> bool {
        let left_adj = self.neighbours(left);
        let right_adj = self.neighbours(right);
        let shared = left_adj.iter().find(|n| right_adj.contains(*n)).cloned();
        if let Some(parent) = shared {
            if let Some(set) = self.adj.get_mut(&parent) {
                set.shift_remove(left);
                set.shift_remove(right);
            }
            self.adj.shift_remove(left);
            self.adj.shift_remove(right);
            self.leaves.shift_remove(left);
            self.leaves.shift_remove(right);
            let mut only = NameSet::default();
            only.insert(parent.clone());
            self.adj.insert(new.to_string(), only);
            self.adj.entry(parent.clone()).or_default().insert(new.to_string());
            self.leaves.insert(new.to_string());
            self.suppress(&parent);
            return true;
        }
        if left_adj.contains(right) {
            self.adj.shift_remove(left);
            self.adj.shift_remove(right);
            self.leaves.shift_remove(left);
            self.leaves.shift_remove(right);
            self.adj.insert(new.to_string(), NameSet::default());
            self.leaves.insert(new.to_string());
            return true;
        }
        false
    }

    fn join_impl(&mut self, left: &str, right: &str, new: &str) {
        let has_left = self.leaves.contains(left);
        let has_right = self.leaves.contains(right);
        if has_left && has_right {
            if !self.contract_impl(left, right, new) {
                self.remove_leaf_impl(left);
                self.remove_leaf_impl(right);
                self.broken += 1;
            }
        } else if has_left {
            self.rename_impl(left, new);
        } else if has_right {
            self.rename_impl(right, new);
        }
    }
}

#[pymethods]
impl Constraint {
    #[new]
    fn py_new(py: Python<'_>, net: &Bound<'_, PyAny>) -> PyResult<Self> {
        let net = net_of(net)?;
        let this = net.borrow();
        Ok(Constraint::from_network(py, &this))
    }

    #[getter]
    fn leaves<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        PySet::new(py, self.leaves.iter().map(|l| PyString::new(py, l)))
    }

    #[getter]
    fn adj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (node, neighbours) in &self.adj {
            out.set_item(node, PySet::new(py, neighbours.iter().map(|l| PyString::new(py, l)))?)?;
        }
        Ok(out)
    }

    #[getter]
    fn broken(&self) -> usize {
        self.broken
    }

    fn _build(&mut self, py: Python<'_>, net: &Bound<'_, PyAny>) -> PyResult<()> {
        let net = net_of(net)?;
        let this = net.borrow();
        self.build(py, &this);
        Ok(())
    }

    /// Pairs of labels that currently share a neighbour.
    fn cherries<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let out = PySet::empty(py)?;
        for (a, b) in self.cherry_pairs() {
            out.add(PyFrozenSet::new(py, [PyString::new(py, &a), PyString::new(py, &b)])?)?;
        }
        Ok(out)
    }

    fn _suppress(&mut self, node: &str) {
        self.suppress(node);
    }

    /// Replace a leaf label. Used when only one side of a join is in this tree.
    fn rename(&mut self, old: &str, new: &str) {
        self.rename_impl(old, new);
    }

    /// Delete a leaf and suppress the internal node it leaves behind.
    fn remove_leaf(&mut self, label: &str) {
        self.remove_leaf_impl(label);
    }

    /// Replace a cherry with ``new``. Return False when the pair is not a cherry.
    fn contract(&mut self, left: &str, right: &str, new: &str) -> bool {
        self.contract_impl(left, right, new)
    }

    /// Apply a neighbor-joining cluster merge to this constraint.
    fn join(&mut self, left: &str, right: &str, new: &str) {
        self.join_impl(left, right, new);
    }
}

// ── Neighbor-joining ───────────────────────────────────────────────────

/// ``_leaf_network``: a one-node network.
fn leaf_network<'py>(py: Python<'py>, label: &str) -> PyResult<Bound<'py, Network>> {
    let net = Network::new_object(py, None)?;
    add_node(&net, &make_node(py, label.to_string())?)?;
    Ok(net)
}

/// ``_join_clusters``: a new root with ``left`` and ``right`` as its clades.
fn join_clusters<'py>(py: Python<'py>, left: &Bound<'py, Network>, right: &Bound<'py, Network>) -> PyResult<Bound<'py, Network>> {
    let joined = Network::new_object(py, None)?;
    let label = joined.borrow().fresh_label(py, "J");
    let root = make_node(py, label)?;
    add_node(&joined, &root)?;
    let root_uid = root.borrow(py).uid;
    graft_subtree(&joined, left, root_uid, false)?;
    graft_subtree(&joined, right, root_uid, false)?;
    Ok(joined)
}

#[pyfunction]
#[pyo3(name = "_dp_leaf_network")]
fn leaf_network_py<'py>(py: Python<'py>, label: &str) -> PyResult<Bound<'py, Network>> {
    leaf_network(py, label)
}

#[pyfunction]
#[pyo3(name = "_dp_join_clusters")]
fn join_clusters_py<'py>(py: Python<'py>, left: &Bound<'py, Network>, right: &Bound<'py, Network>) -> PyResult<Bound<'py, Network>> {
    join_clusters(py, left, right)
}

/// ``_reverse_upward``: reverse the edges from ``start`` up to the root.
fn reverse_upward(py: Python<'_>, net: &Bound<'_, Network>, start: u64, ignore: Option<u64>) -> PyResult<()> {
    let chain: Vec<(u64, u64)> = {
        let this = net.borrow();
        let mut chain = Vec::new();
        let mut current = start;
        let mut seen: FxHashSet<u64> = FxHashSet::default();
        while seen.insert(current) {
            let parents: Vec<u64> = this.parents(current).into_iter().filter(|p| Some(*p) != ignore).collect();
            let Some(&parent) = parents.first() else { break };
            chain.push((parent, current));
            current = parent;
        }
        chain
    };
    for (parent, child) in chain {
        let (euid, length, child_obj, parent_obj) = {
            let this = net.borrow();
            let euid = edge_uid_between(py, &this, parent, child)?;
            (euid, edge_length(py, &this, euid), this.node_obj(child).clone_ref(py), this.node_obj(parent).clone_ref(py))
        };
        remove_edge(net, euid);
        add_edge(net, &make_edge(py, child_obj, parent_obj, length)?)?;
    }
    Ok(())
}

/// ``_reroot_on_edge``: a new root halfway along ``parent -> child``.
fn reroot_on_edge(py: Python<'_>, net: &Bound<'_, Network>, parent: u64, child: u64) -> PyResult<()> {
    let (euid, length, child_obj, parent_obj, label) = {
        let this = net.borrow();
        let euid = edge_uid_between(py, &this, parent, child)?;
        (
            euid,
            edge_length(py, &this, euid),
            this.node_obj(child).clone_ref(py),
            this.node_obj(parent).clone_ref(py),
            this.fresh_label(py, "R"),
        )
    };
    remove_edge(net, euid);
    let root = make_node(py, label)?;
    add_node(net, &root)?;
    let half = length.map(|l| l / 2.0);
    add_edge(net, &make_edge(py, root.clone_ref(py), child_obj, half)?)?;
    add_edge(net, &make_edge(py, root.clone_ref(py), parent_obj, half)?)?;
    let root_uid = root.borrow(py).uid;
    reverse_upward(py, net, parent, Some(root_uid))?;
    Network::clean_impl(net, [true, true, true])
}

/// ``_reroot_above``: make the parent of leaf ``label`` the root.
fn reroot_above(py: Python<'_>, net: &Bound<'_, Network>, label: &str) -> PyResult<()> {
    let parent = {
        let this = net.borrow();
        let Some(leaf) = node_by_name(&this, label) else { return Ok(()) };
        let parents = this.parents(leaf);
        let Some(&parent) = parents.first() else { return Ok(()) };
        parent
    };
    reverse_upward(py, net, parent, None)
}

/// ``_smaller_side``: sorted labels of the smaller side of the edge into ``child``.
fn smaller_side(py: Python<'_>, net: &Network, child: u64) -> Vec<String> {
    let below: Vec<String> = labels_under(py, net, child).into_iter().collect();
    let below_set: FxHashSet<&str> = below.iter().map(|s| s.as_str()).collect();
    let universe: Vec<String> = leaf_labels(py, net).into_iter().collect();
    let complement: Vec<String> = universe.into_iter().filter(|l| !below_set.contains(l.as_str())).collect();
    if (below.len(), &below) <= (complement.len(), &complement) {
        below
    } else {
        complement
    }
}

/// ``_root_by_triples``: copy of ``tree`` rooted on the edge that displays
/// the most observed triples.
fn root_by_triples<'py>(
    py: Python<'py>,
    tree: &Bound<'py, Network>,
    counts: &IndexMap<Triple, i64, FxBuildHasher>,
) -> PyResult<Bound<'py, Network>> {
    if counts.is_empty() || counts.values().all(|c| *c == 0) {
        return Ok(tree.clone());
    }
    let (n_leaves, skip, edges) = {
        let this = tree.borrow();
        let n_leaves = this.leaf_uids().len();
        let root = this.root_uid(py)?;
        let mut under_root: Vec<(String, u64)> = this
            .out_edge_uids(root)
            .into_iter()
            .map(|e| (this.label(py, this.edges[&e].du), e))
            .collect();
        under_root.sort();
        let skip: FxHashSet<u64> = under_root.iter().skip(1).map(|(_, e)| *e).collect();
        let edges: Vec<(u64, u64, u64)> = this.edges.iter().map(|(e, entry)| (*e, entry.su, entry.du)).collect();
        (n_leaves, skip, edges)
    };
    if n_leaves > 60 {
        warn(py, &format!("Rooting a {n_leaves}-leaf blob by rooted triples scores every edge."))?;
    }
    let mut best_score: i64 = 0;
    let mut best_side: Vec<String> = Vec::new();
    let mut best_copy: Option<Bound<'py, Network>> = None;
    for (euid, su, du) in edges {
        if skip.contains(&euid) {
            continue;
        }
        let side = smaller_side(py, &tree.borrow(), du);
        let (copy, mapping) = copy_with_map(py, tree)?;
        reroot_on_edge(py, &copy, mapping[&su], mapping[&du])?;
        let displayed = rooted_triples(py, &copy.borrow())?;
        let score: i64 = displayed.iter().map(|t| counts.get(t).copied().unwrap_or(0)).sum();
        if best_copy.is_none() || score > best_score || (score == best_score && side < best_side) {
            best_score = score;
            best_side = side;
            best_copy = Some(copy);
        }
    }
    match best_copy {
        Some(copy) if best_score != 0 => Ok(copy),
        _ => Ok(tree.clone()),
    }
}

#[pyfunction]
#[pyo3(name = "_dp_reverse_upward", signature = (net, start, ignore=None))]
fn reverse_upward_py(py: Python<'_>, net: &Bound<'_, Network>, start: &Bound<'_, PyAny>, ignore: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let (s, i) = {
        let this = net.borrow();
        (
            this.uid_of_any(start).ok_or_else(|| network_error(py, "Node not found in graph."))?,
            ignore.and_then(|n| this.uid_of_any(n)),
        )
    };
    reverse_upward(py, net, s, i)
}

#[pyfunction]
#[pyo3(name = "_dp_reroot_on_edge")]
fn reroot_on_edge_py(py: Python<'_>, net: &Bound<'_, Network>, parent: &Bound<'_, PyAny>, child: &Bound<'_, PyAny>) -> PyResult<()> {
    let (p, c) = {
        let this = net.borrow();
        (
            this.uid_of_any(parent).ok_or_else(|| network_error(py, "Node not found in graph."))?,
            this.uid_of_any(child).ok_or_else(|| network_error(py, "Node not found in graph."))?,
        )
    };
    reroot_on_edge(py, net, p, c)
}

#[pyfunction]
#[pyo3(name = "_dp_reroot_above")]
fn reroot_above_py(py: Python<'_>, net: &Bound<'_, Network>, label: &str) -> PyResult<()> {
    reroot_above(py, net, label)
}

#[pyfunction]
#[pyo3(name = "_dp_smaller_side")]
fn smaller_side_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, child: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let this = net.borrow();
    let uid = this.uid_of_any(child).ok_or_else(|| network_error(py, "Node not found in graph."))?;
    PyTuple::new(py, smaller_side(py, &this, uid))
}

#[pyfunction]
#[pyo3(name = "_dp_root_by_triples")]
fn root_by_triples_py<'py>(py: Python<'py>, tree: &Bound<'py, Network>, counts: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Network>> {
    root_by_triples(py, tree, &triple_counts(counts)?)
}

/// ``overlap_nj`` proper, on a prepared distance table.
fn overlap_nj_impl<'py>(
    py: Python<'py>,
    trees: &[Bound<'py, Network>],
    names: Vec<String>,
    dist: FxHashMap<(usize, usize), f64>,
    ids: FxHashMap<String, usize>,
) -> PyResult<(Bound<'py, Network>, usize)> {
    let mut ids = ids;
    let mut dist = dist;
    let distance = |dist: &FxHashMap<(usize, usize), f64>, left: usize, right: usize| -> PyResult<f64> {
        if left == right {
            return Ok(0.0);
        }
        dist.get(&(left, right))
            .copied()
            .ok_or_else(|| PyKeyError::new_err(format!("({left}, {right})")))
    };
    let mut constraints: Vec<Constraint> = trees.iter().map(|t| Constraint::from_network(py, &t.borrow())).collect();
    let mut clusters: FxHashMap<usize, Bound<'py, Network>> = FxHashMap::default();
    for name in &names {
        clusters.insert(ids[name], leaf_network(py, name)?);
    }
    let mut labels: Vec<String> = names.clone();
    let mut active: Vec<usize> = names.iter().map(|n| ids[n]).collect();
    let mut relaxations = 0usize;
    let mut counter = 0usize;
    while active.len() > 1 {
        let count = active.len();
        let mut row_sum: FxHashMap<usize, f64> = FxHashMap::default();
        for &label in &active {
            let mut total = 0.0;
            for &other in &active {
                if other != label {
                    total += distance(&dist, label, other)?;
                }
            }
            row_sum.insert(label, total);
        }
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (i, &left) in active.iter().enumerate() {
            for &right in &active[i + 1..] {
                let q = (count as f64 - 2.0) * distance(&dist, left, right)? - row_sum[&left] - row_sum[&right];
                pairs.push((q, left, right));
            }
        }
        pairs.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| labels[a.1].cmp(&labels[b.1]))
                .then_with(|| labels[a.2].cmp(&labels[b.2]))
        });
        let mut cherries: Vec<Option<FxHashSet<(String, String)>>> = vec![None; constraints.len()];
        let mut chosen: Option<(usize, usize)> = None;
        for &(_, left, right) in &pairs {
            let (l, r) = (&labels[left], &labels[right]);
            let mut violated = false;
            for (k, constraint) in constraints.iter().enumerate() {
                if constraint.leaves.contains(l) && constraint.leaves.contains(r) {
                    let pair = if l < r { (l.clone(), r.clone()) } else { (r.clone(), l.clone()) };
                    let set = cherries[k].get_or_insert_with(|| constraint.cherry_pairs());
                    if !set.contains(&pair) {
                        violated = true;
                        break;
                    }
                }
            }
            if !violated {
                chosen = Some((left, right));
                break;
            }
        }
        let (left, right) = match chosen {
            Some(pair) => pair,
            None => {
                relaxations += 1;
                (pairs[0].1, pairs[0].2)
            }
        };
        let new_label = format!("NJ{counter}");
        counter += 1;
        let new_id = labels.len();
        labels.push(new_label.clone());
        ids.insert(new_label.clone(), new_id);
        for &other in &active {
            if other == left || other == right {
                continue;
            }
            let updated = 0.5
                * (distance(&dist, left, other)? + distance(&dist, right, other)? - distance(&dist, left, right)?);
            dist.insert((new_id, other), updated);
            dist.insert((other, new_id), updated);
        }
        for constraint in constraints.iter_mut() {
            constraint.join_impl(&labels[left], &labels[right], &new_label);
        }
        let joined = join_clusters(py, &clusters[&left], &clusters[&right])?;
        clusters.insert(new_id, joined);
        active.retain(|&l| l != left && l != right);
        active.push(new_id);
    }
    let merged = clusters.remove(&active[0]).expect("the last cluster");
    let out_uid = node_by_name(&merged.borrow(), OUT);
    if let Some(out_uid) = out_uid {
        reroot_above(py, &merged, OUT)?;
        remove_node(&merged, out_uid);
        Network::clean_impl(&merged, [true, true, true])?;
    }
    Ok((merged, relaxations))
}

#[pyfunction]
#[pyo3(name = "_dp_overlap_nj", signature = (trees, labels, matrix, outgroup_distances=None))]
fn overlap_nj<'py>(
    py: Python<'py>,
    trees: &Bound<'py, PyAny>,
    labels: &Bound<'py, PyAny>,
    matrix: &Bound<'py, PyAny>,
    outgroup_distances: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, Network>, usize)> {
    let trees = tree_list(trees)?;
    let mut names = label_vec_from_any(labels)?;
    if names.is_empty() {
        return Err(dimple_error(py, "Neighbor-joining was given no taxa."));
    }
    if names.len() == 1 {
        return Ok((leaf_network(py, &names[0])?, 0));
    }
    let shape: Vec<usize> = match matrix.getattr("shape") {
        Ok(s) => s.extract().unwrap_or_default(),
        Err(_) => Vec::new(),
    };
    if shape != vec![names.len(), names.len()] {
        return Err(dimple_error(py, "Distance matrix shape does not match the taxon list."));
    }
    let matrix = float_matrix(py, matrix)?;
    let view = matrix.as_array();
    if view.iter().any(|v| !v.is_finite()) {
        return Err(dimple_error(
            py,
            "A taxon pair this blob needs never co-occurs in a gene tree.",
        ));
    }
    let mut ids: FxHashMap<String, usize> = FxHashMap::default();
    for (i, name) in names.iter().enumerate() {
        ids.insert(name.clone(), i);
    }
    let mut dist: FxHashMap<(usize, usize), f64> = FxHashMap::default();
    for i in 0..names.len() {
        for j in 0..names.len() {
            if i < j {
                let value = view[[i, j]];
                dist.insert((i, j), value);
                dist.insert((j, i), value);
            }
        }
    }
    if let Some(outgroup) = outgroup_distances.filter(|o| !o.is_none()) {
        let missing: Vec<&String> = names
            .iter()
            .filter(|name| !outgroup.contains(name.as_str()).unwrap_or(false))
            .collect();
        if !missing.is_empty() {
            return Err(dimple_error(
                py,
                format!("Outgroup distances are missing for {}.", join_first(missing, 5)),
            ));
        }
        let out_id = names.len();
        for (i, name) in names.iter().enumerate() {
            let value: f64 = outgroup.get_item(name.as_str())?.extract()?;
            dist.insert((i, out_id), value);
            dist.insert((out_id, i), value);
        }
        ids.insert(OUT.to_string(), out_id);
        names.push(OUT.to_string());
    }
    overlap_nj_impl(py, &trees, names, dist, ids)
}

// ── Displayed trees ────────────────────────────────────────────────────

/// ``_miss_fraction``: fraction of observed triples on ``keep`` that
/// ``tree`` does not display.
fn miss_fraction<'py>(
    py: Python<'py>,
    tree: &Bound<'py, Network>,
    keep: &LabelSet,
    observed: &IndexMap<Triple, i64, FxBuildHasher>,
) -> PyResult<f64> {
    let present = leaf_labels(py, &tree.borrow());
    let used = intersect(&present, keep);
    if used.len() < 3 {
        return Ok(1.0);
    }
    let displayed = if used == present {
        rooted_triples(py, &tree.borrow())?
    } else {
        let viewed = induced_by_taxa(py, tree, used.iter())?;
        rooted_triples(py, &viewed.borrow())?
    };
    let mut total: i64 = 0;
    let mut matched: i64 = 0;
    for ((a, b, c), count) in observed {
        if !used.contains(a) || !used.contains(b) || !used.contains(c) {
            continue;
        }
        total += count;
        let (x, y) = if a < b { (a, b) } else { (b, a) };
        if displayed.contains(&(x.clone(), y.clone(), c.clone())) {
            matched += count;
        }
    }
    if total == 0 {
        return Ok(1.0);
    }
    Ok(1.0 - (matched as f64 / total as f64))
}

#[pyfunction]
#[pyo3(name = "_dp_miss_fraction")]
fn miss_fraction_py<'py>(py: Python<'py>, tree: &Bound<'py, Network>, keep: &Bound<'py, PyAny>, observed: &Bound<'py, PyAny>) -> PyResult<f64> {
    miss_fraction(py, tree, &labels_from_any(keep)?, &triple_counts(observed)?)
}

/// In-edges of ``uid`` by descending gamma, then source label.
fn sorted_in_edges(py: Python<'_>, net: &Network, uid: u64) -> Vec<u64> {
    let mut edges: Vec<(u64, String, u64)> = net
        .in_edge_uids(uid)
        .into_iter()
        .map(|e| (e, net.label(py, net.edges[&e].su), e))
        .collect();
    edges.sort_by(|a, b| {
        let ga = edge_gamma(py, net, a.0).unwrap_or(0.0);
        let gb = edge_gamma(py, net, b.0).unwrap_or(0.0);
        gb.partial_cmp(&ga).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.cmp(&b.1))
    });
    edges.into_iter().map(|(e, _, _)| e).collect()
}

/// ``_displayed_tree_choices``: the kept in-edge per reticulation of each
/// tree ``get_all_subtrees`` enumerates, as (node uid, edge uid) pairs.
fn displayed_tree_choices(py: Python<'_>, net: &Network) -> Vec<Vec<(u64, u64)>> {
    let mut retics: Vec<(String, u64)> = net
        .nodes
        .iter()
        .filter(|(_, e)| e.obj.borrow(py).is_retic)
        .map(|(u, e)| (e.obj.borrow(py).name.clone(), *u))
        .collect();
    retics.sort();
    let options: Vec<Vec<u64>> = retics.iter().map(|(_, u)| sorted_in_edges(py, net, *u)).collect();
    let mut combos: Vec<Vec<(u64, u64)>> = vec![Vec::new()];
    for (k, opts) in options.iter().enumerate() {
        let mut next = Vec::new();
        for combo in &combos {
            for &edge in opts {
                let mut extended = combo.clone();
                extended.push((retics[k].1, edge));
                next.push(extended);
            }
        }
        combos = next;
    }
    combos
}

fn choice_dict<'py>(py: Python<'py>, net: &Network, choice: &[(u64, u64)]) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (node, edge) in choice {
        out.set_item(net.node_obj(*node).bind(py), net.edge_obj(*edge).bind(py))?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_dp_displayed_tree_choices")]
fn displayed_tree_choices_py<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PyList>> {
    let this = net.borrow();
    let choices = displayed_tree_choices(py, &this);
    PyList::new(py, choices.iter().map(|c| choice_dict(py, &this, c)).collect::<PyResult<Vec<_>>>()?)
}

/// ``select_displayed_trees_with_choices``.
fn select_displayed_trees_impl<'py>(
    py: Python<'py>,
    networks: &[Bound<'py, Network>],
    observed: Option<&IndexMap<Triple, i64, FxBuildHasher>>,
) -> PyResult<(Vec<Bound<'py, Network>>, Vec<Option<Bound<'py, PyDict>>>)> {
    let leaf_sets: Vec<LabelSet> = networks.iter().map(|n| leaf_labels(py, &n.borrow())).collect();
    let mut chosen: Vec<Bound<'py, Network>> = Vec::new();
    let mut choices: Vec<Option<Bound<'py, PyDict>>> = Vec::new();
    for (index, net) in networks.iter().enumerate() {
        if count_retic_flags(py, &net.borrow()) == 0 {
            chosen.push(net.clone());
            choices.push(None);
            continue;
        }
        let candidates = get_all_subtrees(py, net)?;
        let mut options: Vec<Option<Bound<'py, PyDict>>> = {
            let this = net.borrow();
            displayed_tree_choices(py, &this)
                .iter()
                .map(|c| choice_dict(py, &this, c).map(Some))
                .collect::<PyResult<Vec<_>>>()?
        };
        if options.len() != candidates.len() {
            options = vec![None; candidates.len()];
        }
        if candidates.len() <= 1 {
            chosen.push(candidates.first().cloned().unwrap_or_else(|| net.clone()));
            choices.push(options.first().cloned().flatten());
            continue;
        }
        let Some(observed) = observed else {
            return Err(dimple_error(py, "Displayed-tree selection needs rooted-triple counts."));
        };
        let mut others = LabelSet::new();
        for (other_index, leaves) in leaf_sets.iter().enumerate() {
            if other_index != index {
                others.extend(leaves.iter().cloned());
            }
        }
        let shared = intersect(&leaf_sets[index], &others);
        let mut best_index = 0usize;
        let mut best_score = miss_fraction(py, &candidates[0], &shared, observed)?;
        for (position, candidate) in candidates.iter().enumerate().skip(1) {
            let score = miss_fraction(py, candidate, &shared, observed)?;
            if score < best_score {
                best_index = position;
                best_score = score;
            }
        }
        chosen.push(candidates[best_index].clone());
        choices.push(options[best_index].clone());
    }
    Ok((chosen, choices))
}

#[pyfunction]
#[pyo3(name = "_dp_select_displayed_trees_with_choices", signature = (networks, observed=None))]
fn select_displayed_trees_with_choices<'py>(
    py: Python<'py>,
    networks: &Bound<'py, PyAny>,
    observed: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let networks = tree_list(networks)?;
    let counts = match observed {
        Some(o) if !o.is_none() => Some(triple_counts(o)?),
        _ => None,
    };
    let (chosen, choices) = select_displayed_trees_impl(py, &networks, counts.as_ref())?;
    let choice_list = PyList::empty(py);
    for choice in choices {
        match choice {
            Some(d) => choice_list.append(d)?,
            None => choice_list.append(py.None())?,
        }
    }
    Ok((PyList::new(py, chosen)?, choice_list))
}

/// ``observed_triples``: counts of rooted triples of the gene trees on ``taxa``.
fn observed_triples_impl<'py>(
    py: Python<'py>,
    gene_trees: &Bound<'py, PyAny>,
    wanted: &LabelSet,
) -> PyResult<IndexMap<Triple, i64, FxBuildHasher>> {
    let mut counts: IndexMap<Triple, i64, FxBuildHasher> = IndexMap::default();
    for tree in gene_tree_list(gene_trees)? {
        let all = leaf_labels(py, &tree.borrow());
        let present = intersect(&all, wanted);
        if present.len() < 3 {
            continue;
        }
        let triples = if present != all {
            let viewed = induced_by_taxa(py, &tree, present.iter())?;
            rooted_triples(py, &viewed.borrow())?
        } else {
            rooted_triples(py, &tree.borrow())?
        };
        // Set iteration order only decides first-insertion order of the
        // counter; the reference's set order was arbitrary too.
        let mut sorted: Vec<Triple> = triples.into_iter().collect();
        sorted.sort();
        for triple in sorted {
            if wanted.contains(&triple.0) && wanted.contains(&triple.1) && wanted.contains(&triple.2) {
                *counts.entry(triple).or_insert(0) += 1;
            }
        }
    }
    Ok(counts)
}

#[pyfunction]
#[pyo3(name = "_dp_observed_triples")]
fn observed_triples<'py>(py: Python<'py>, gene_trees: &Bound<'py, PyAny>, taxa: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    triple_dict(py, &observed_triples_impl(py, gene_trees, &labels_from_any(taxa)?)?)
}

// ── Subset restriction ─────────────────────────────────────────────────

/// ``restrict_gene_trees``.
fn restrict_gene_trees_impl<'py>(py: Python<'py>, gene_trees: &Bound<'py, PyAny>, wanted: &LabelSet) -> PyResult<Bound<'py, PyAny>> {
    let mut kept: Vec<Bound<'py, Network>> = Vec::new();
    for tree in gene_tree_list(gene_trees)? {
        let (present, all): (Vec<String>, LabelSet) = {
            let this = tree.borrow();
            let leaves: Vec<String> = this.leaf_uids().into_iter().map(|u| this.label(py, u)).collect();
            (
                leaves.iter().filter(|l| wanted.contains(*l)).cloned().collect(),
                leaves.into_iter().collect(),
            )
        };
        if present.len() < 3 {
            continue;
        }
        let present_set: LabelSet = present.iter().cloned().collect();
        if present_set == all {
            kept.push(tree);
        } else {
            kept.push(induced_by_taxa(py, &tree, present.iter())?);
        }
    }
    if kept.is_empty() {
        return Err(dimple_error(
            py,
            format!(
                "No gene tree has three taxa in common with the subset {}.",
                repr_sorted(py, wanted)?
            ),
        ));
    }
    let mut remaining = LabelSet::new();
    for tree in &kept {
        remaining.extend(leaf_labels(py, &tree.borrow()));
    }
    let parent = gene_trees.call_method0("resolved_mapping")?;
    let mapping = PyDict::new(py);
    for (species, alleles) in parent.cast::<PyDict>()?.iter() {
        let kept_alleles = PyList::empty(py);
        for allele in alleles.try_iter()? {
            let allele = allele?;
            if remaining.contains(&allele.extract::<String>()?) {
                kept_alleles.append(allele)?;
            }
        }
        if kept_alleles.len() > 0 {
            mapping.set_item(species, kept_alleles)?;
        }
    }
    if mapping.len() == 0 {
        for label in &remaining {
            mapping.set_item(label, PyList::new(py, [label])?)?;
        }
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("trees", PyList::new(py, &kept)?)?;
    kwargs.set_item("mapping", mapping)?;
    gene_trees_class(py)?.call((), Some(&kwargs))
}

#[pyfunction]
#[pyo3(name = "_dp_restrict_gene_trees")]
fn restrict_gene_trees<'py>(py: Python<'py>, gene_trees: &Bound<'py, PyAny>, taxa: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    restrict_gene_trees_impl(py, gene_trees, &labels_from_any(taxa)?)
}

#[pyfunction]
#[pyo3(name = "_dp_restrict_start")]
fn restrict_start<'py>(py: Python<'py>, start_tree: &Bound<'py, Network>, taxa: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Network>> {
    let taxa = label_vec_from_any(taxa)?;
    let mut missing: Vec<String> = {
        let this = start_tree.borrow();
        taxa.iter().filter(|l| node_by_name(&this, l).is_none()).cloned().collect()
    };
    if !missing.is_empty() {
        missing.sort();
        return Err(dimple_error(
            py,
            format!("The start tree is missing subset taxa: {}.", join_first(missing, 5)),
        ));
    }
    let induced = induced_by_taxa(py, start_tree, taxa.iter())?;
    if !is_binary_tree(py, &induced.borrow()) {
        return Err(dimple_error(py, "The start tree restricted to the subset is not binary."));
    }
    set_unit_coalescent(&induced)?;
    populate_default_branch_lengths(py, &induced)?;
    Ok(induced)
}

#[pyfunction]
#[pyo3(name = "_dp_true_reticulation_bound")]
fn true_reticulation_bound<'py>(py: Python<'py>, true_network: &Bound<'py, Network>, taxa: &Bound<'py, PyAny>) -> PyResult<usize> {
    let taxa = label_vec_from_any(taxa)?;
    let mut missing: Vec<String> = {
        let this = true_network.borrow();
        taxa.iter().filter(|l| node_by_name(&this, l).is_none()).cloned().collect()
    };
    if !missing.is_empty() {
        missing.sort();
        return Err(dimple_error(
            py,
            format!("The true network lacks subset taxa: {}.", join_first(missing, 5)),
        ));
    }
    let sorted: LabelSet = taxa.into_iter().collect();
    let induced = induced_by_taxa(py, true_network, sorted.iter())?;
    let count = cached(py, &GU_COUNT_RETICS, "phynetpy._core", "_gu_count_reticulations")?;
    count.call1((induced, 4.0))?.extract::<usize>()
}

#[pyfunction]
#[pyo3(name = "_dp_selection_min_gain")]
fn selection_min_gain(selection: Option<&str>, data_size: i64) -> f64 {
    match selection {
        Some("gain") => SELECTION_GAIN_THRESHOLD,
        Some("aic") => SELECTION_PARAMS_PER_RETICULATION,
        Some("bic") => 0.5 * SELECTION_PARAMS_PER_RETICULATION * (data_size.max(1) as f64).ln(),
        _ => 0.0,
    }
}

#[pyfunction]
#[pyo3(name = "_dp_populate_default_branch_lengths")]
fn populate_default_branch_lengths_py(py: Python<'_>, net: &Bound<'_, Network>) -> PyResult<()> {
    populate_default_branch_lengths(py, net)
}

// ── Partition ──────────────────────────────────────────────────────────

/// A non-blob clade cut off a blob, and where it has to be put back
/// (``PrunedSubtree`` before it is handed to Python).
struct Prune<'py> {
    leaves: LabelSet,
    anchor_leaves: LabelSet,
    parent_leaves: LabelSet,
    subtree: Bound<'py, Network>,
    source_type: String,
    index: usize,
}

impl<'py> Prune<'py> {
    fn to_py(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let cls = py.import("phynetpy._dimple")?.getattr("PrunedSubtree")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("leaves", frozenset(py, &self.leaves)?)?;
        kwargs.set_item("anchor_leaves", frozenset(py, &self.anchor_leaves)?)?;
        kwargs.set_item("parent_leaves", frozenset(py, &self.parent_leaves)?)?;
        kwargs.set_item("subtree", &self.subtree)?;
        kwargs.set_item("source_type", &self.source_type)?;
        kwargs.set_item("index", self.index)?;
        cls.call((), Some(&kwargs))
    }

    /// Read a Python ``PrunedSubtree``.
    fn from_py(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Prune {
            leaves: labels_from_any(&obj.getattr("leaves")?)?,
            anchor_leaves: labels_from_any(&obj.getattr("anchor_leaves")?)?,
            parent_leaves: labels_from_any(&obj.getattr("parent_leaves")?)?,
            subtree: net_of(&obj.getattr("subtree")?)?,
            source_type: obj.getattr("source_type")?.extract()?,
            index: obj.getattr("index")?.extract()?,
        })
    }
}

type LabelPair = (String, String);

/// ``_forbidden``: original out-edges of every polytomy, as label pairs.
fn forbidden_pairs(py: Python<'_>, net: &Network, blobs: &[u64]) -> FxHashSet<LabelPair> {
    let mut banned: FxHashSet<LabelPair> = FxHashSet::default();
    for &blob in blobs {
        for child in net.children(blob) {
            banned.insert((net.label(py, blob), net.label(py, child)));
        }
    }
    banned
}

/// ``_mega_groups``: union blobs joined by a direct parent-child edge.
fn mega_groups(net: &Network, blobs: &[u64]) -> Vec<Vec<u64>> {
    let position: FxHashMap<u64, usize> = blobs.iter().enumerate().map(|(i, u)| (*u, i)).collect();
    let mut parent: Vec<usize> = (0..blobs.len()).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for (i, &blob) in blobs.iter().enumerate() {
        for child in net.children(blob) {
            if let Some(&j) = position.get(&child) {
                let (ra, rb) = (find(&mut parent, i), find(&mut parent, j));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }
    let mut groups: IndexMap<usize, Vec<u64>, FxBuildHasher> = IndexMap::default();
    for (i, &blob) in blobs.iter().enumerate() {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(blob);
    }
    groups.into_values().collect()
}

/// ``_top_member``: the member no other member is an ancestor of.
fn top_member(net: &Network, members: &[u64]) -> u64 {
    for &blob in members {
        let under_other = members.iter().any(|&other| other != blob && net.subtree_uids(other).contains(&blob));
        if !under_other {
            return blob;
        }
    }
    members[0]
}

/// ``_candidate_cuts``: legal cuts as ``(leaf count, edge uid)``.
fn candidate_cuts(
    py: Python<'_>,
    net: &Network,
    forbidden: &FxHashSet<LabelPair>,
    blob_set: &FxHashSet<u64>,
    max_subset_size: usize,
) -> Vec<(usize, u64)> {
    let total = net.leaf_uids().len();
    let mut found = Vec::new();
    for (&euid, entry) in net.edges.iter() {
        if forbidden.contains(&(net.label(py, entry.su), net.label(py, entry.du))) {
            continue;
        }
        if blob_set.contains(&entry.su) {
            continue;
        }
        if !blob_set.is_empty() {
            let below = net.subtree_uids(entry.du);
            if blob_set.iter().any(|b| below.contains(b)) {
                continue;
            }
        }
        let count = net.leaf_descendants_uid(entry.du).len();
        if count < 3 || total - count < 3 || count > max_subset_size {
            continue;
        }
        found.push((count, euid));
    }
    found
}

/// Largest clade first; equal sizes break toward the lexicographically
/// smaller leaf set.
fn sort_cuts(py: Python<'_>, net: &Network, cuts: &mut [(usize, u64)]) {
    let mut keyed: Vec<((usize, u64), Vec<String>)> = cuts
        .iter()
        .map(|&(count, euid)| ((count, euid), labels_under(py, net, net.edges[&euid].du).into_iter().collect()))
        .collect();
    keyed.sort_by(|a, b| b.0 .0.cmp(&a.0 .0).then_with(|| a.1.cmp(&b.1)));
    for (slot, (item, _)) in cuts.iter_mut().zip(keyed) {
        *slot = item;
    }
}

/// ``_oversized``: blobs whose children's leaf counts sum to more than S.
fn oversized(net: &Network, blobs: &[u64], max_subset_size: usize) -> Vec<u64> {
    let mut too_big = Vec::new();
    for &blob in blobs {
        if !net.nodes.contains_key(&blob) {
            continue;
        }
        let size: usize = net.children(blob).iter().map(|&c| net.leaf_descendants_uid(c).len()).sum();
        if size > max_subset_size {
            too_big.push(blob);
        }
    }
    too_big
}

/// ``_prune_piece``: cut the largest legal clade until every blob fits in S.
fn prune_piece<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
    blobs: &[u64],
    forbidden: &FxHashSet<LabelPair>,
    max_subset_size: usize,
) -> PyResult<Vec<Prune<'py>>> {
    let mut pruned: Vec<Prune<'py>> = Vec::new();
    let blob_set: FxHashSet<u64> = blobs.iter().copied().collect();
    loop {
        let chosen = {
            let this = net.borrow();
            let pending = oversized(&this, blobs, max_subset_size);
            if pending.is_empty() {
                break;
            }
            let mut cuts = candidate_cuts(py, &this, forbidden, &blob_set, max_subset_size);
            if cuts.is_empty() {
                break;
            }
            sort_cuts(py, &this, &mut cuts);
            let pending_subtrees: Vec<Vec<u64>> = pending.iter().map(|&b| this.subtree_uids(b)).collect();
            let mut chosen: Option<(u64, u64, LabelSet, LabelSet)> = None;
            for &(_, euid) in &cuts {
                let (su, du) = (this.edges[&euid].su, this.edges[&euid].du);
                if pending_subtrees.iter().any(|s| s.contains(&du)) {
                    chosen = Some((su, du, labels_under(py, &this, su), labels_under(py, &this, du)));
                    break;
                }
            }
            chosen
        };
        let Some((src, dest, parent_leaves, cut_leaves)) = chosen else { break };
        let siblings = minus(&parent_leaves, &cut_leaves);
        let clade = detach(py, net, src, dest)?;
        let index = pruned.len();
        pruned.push(Prune {
            leaves: cut_leaves,
            anchor_leaves: siblings,
            parent_leaves,
            subtree: clade,
            source_type: "blob_item".to_string(),
            index,
        });
    }
    Ok(pruned)
}

/// ``_split_clade``: split a leaf set larger than S by the same size rules.
fn split_clade<'py>(py: Python<'py>, net: &Bound<'py, Network>, leaves: &LabelSet, max_subset_size: usize) -> PyResult<Vec<LabelSet>> {
    if leaves.len() <= max_subset_size {
        return Ok(vec![leaves.clone()]);
    }
    let piece = induced_by_taxa(py, net, leaves.iter())?;
    let mut parts: Vec<LabelSet> = Vec::new();
    let empty_pairs: FxHashSet<LabelPair> = FxHashSet::default();
    let empty_blobs: FxHashSet<u64> = FxHashSet::default();
    loop {
        let (src, dest, cut_leaves) = {
            let this = piece.borrow();
            let n_leaves = this.leaf_uids().len();
            if n_leaves <= max_subset_size {
                break;
            }
            let mut cuts = candidate_cuts(py, &this, &empty_pairs, &empty_blobs, max_subset_size);
            if cuts.is_empty() {
                return Err(dimple_error(
                    py,
                    format!(
                        "A clade of {n_leaves} leaves has no legal cut of size at most {max_subset_size}."
                    ),
                ));
            }
            sort_cuts(py, &this, &mut cuts);
            let euid = cuts[0].1;
            let (su, du) = (this.edges[&euid].su, this.edges[&euid].du);
            (su, du, labels_under(py, &this, du))
        };
        detach(py, &piece, src, dest)?;
        parts.push(cut_leaves);
    }
    let rest = leaf_labels(py, &piece.borrow());
    if !rest.is_empty() {
        parts.push(rest);
    }
    Ok(parts)
}

/// ``_enumerate_partitions``: unique bin partitions in restricted-growth
/// order, each bin in ``[min_size, max_size]``.
fn enumerate_partitions(weights: &[usize], max_size: usize, min_size: usize, cap: usize) -> Vec<Vec<Vec<usize>>> {
    struct State<'a> {
        weights: &'a [usize],
        max_size: usize,
        min_size: usize,
        cap: usize,
        results: Vec<Vec<Vec<usize>>>,
        bins: Vec<(usize, Vec<usize>)>,
    }
    fn backtrack(st: &mut State<'_>, index: usize) -> bool {
        if st.results.len() >= st.cap {
            return true;
        }
        if index == st.weights.len() {
            if !st.bins.is_empty() && st.bins.iter().all(|b| b.0 >= st.min_size) {
                st.results.push(st.bins.iter().map(|b| b.1.clone()).collect());
            }
            return false;
        }
        let weight = st.weights[index];
        for k in 0..st.bins.len() {
            if st.bins[k].0 + weight <= st.max_size {
                st.bins[k].0 += weight;
                st.bins[k].1.push(index);
                let stop = backtrack(st, index + 1);
                st.bins[k].0 -= weight;
                st.bins[k].1.pop();
                if stop {
                    return true;
                }
            }
        }
        st.bins.push((weight, vec![index]));
        let stop = backtrack(st, index + 1);
        st.bins.pop();
        stop
    }
    let mut st = State { weights, max_size, min_size, cap, results: Vec::new(), bins: Vec::new() };
    if !weights.is_empty() {
        backtrack(&mut st, 0);
    }
    st.results
}

/// ``rng.shuffle(list(range(n)))``: the permutation Python's shuffle draws.
fn shuffle_permutation(py: Python<'_>, rng: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<usize>> {
    let list = PyList::new(py, 0..n)?;
    rng.call_method1("shuffle", (&list,))?;
    list.extract::<Vec<usize>>()
}

/// ``rng.choice(range(n))``: the index Python's choice draws.
fn choice_index(py: Python<'_>, rng: &Bound<'_, PyAny>, n: usize) -> PyResult<usize> {
    rng.call_method1("choice", (PyList::new(py, 0..n)?,))?.extract::<usize>()
}

fn python_random<'py>(py: Python<'py>, seed: i64) -> PyResult<Bound<'py, PyAny>> {
    py.import("random")?.getattr("Random")?.call1((seed,))
}

/// ``_random_pack``: one seeded first-fit packing with DIMPLE's repair of
/// an undersized last bin.
fn random_pack(
    py: Python<'_>,
    weights: &[usize],
    max_size: usize,
    min_size: usize,
    rng: &Bound<'_, PyAny>,
    items: Option<&[usize]>,
) -> PyResult<Vec<Vec<usize>>> {
    let base: Vec<usize> = match items {
        Some(items) => items.to_vec(),
        None => (0..weights.len()).collect(),
    };
    let perm = shuffle_permutation(py, rng, base.len())?;
    let order: Vec<usize> = perm.into_iter().map(|p| base[p]).collect();
    let mut bins: Vec<Vec<usize>> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();
    for index in order {
        let mut placed = false;
        for bin_index in 0..sizes.len() {
            if sizes[bin_index] + weights[index] <= max_size {
                bins[bin_index].push(index);
                sizes[bin_index] += weights[index];
                placed = true;
                break;
            }
        }
        if !placed {
            bins.push(vec![index]);
            sizes.push(weights[index]);
        }
    }
    if bins.len() > 1 && sizes[sizes.len() - 1] < min_size {
        let last = bins.len() - 1;
        let last_size = sizes[last];
        for bin_index in 0..last {
            if sizes[bin_index] + last_size <= max_size {
                let tail = bins.pop().expect("last bin");
                sizes.pop();
                bins[bin_index].extend(tail);
                sizes[bin_index] += last_size;
                return Ok(bins);
            }
        }
        while sizes[last] < min_size {
            let mut movable: Vec<(usize, usize)> = Vec::new();
            for bin_index in 0..last {
                if sizes[bin_index] > min_size {
                    for &index in &bins[bin_index] {
                        if sizes[bin_index] - weights[index] >= min_size && sizes[last] + weights[index] <= max_size {
                            movable.push((bin_index, index));
                        }
                    }
                }
            }
            if movable.is_empty() {
                break;
            }
            let (bin_index, index) = movable[choice_index(py, rng, movable.len())?];
            if let Some(pos) = bins[bin_index].iter().position(|&i| i == index) {
                bins[bin_index].remove(pos);
            }
            sizes[bin_index] -= weights[index];
            bins[last].push(index);
            sizes[last] += weights[index];
        }
    }
    Ok(bins)
}

/// ``_smallest_first_pack``: DIMPLE's ``smallest_first`` strategy.
fn smallest_first_pack(
    py: Python<'_>,
    weights: &[usize],
    max_size: usize,
    min_size: usize,
    rng: &Bound<'_, PyAny>,
) -> PyResult<Vec<Vec<usize>>> {
    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_by_key(|&i| weights[i]);
    let mut first: Vec<usize> = Vec::new();
    let mut first_size = 0usize;
    let mut rest: Vec<usize> = Vec::new();
    for index in order {
        if first_size + weights[index] <= max_size {
            first.push(index);
            first_size += weights[index];
        } else {
            rest.push(index);
        }
    }
    if first_size < min_size {
        return random_pack(py, weights, max_size, min_size, rng, None);
    }
    let mut bins: Vec<Vec<usize>> = if first.is_empty() { Vec::new() } else { vec![first] };
    if !rest.is_empty() {
        bins.extend(random_pack(py, weights, max_size, min_size, rng, Some(&rest))?);
    }
    Ok(bins)
}

type Item = (String, LabelSet);

/// ``_bins_to_division``: item-index bins to sorted taxon sets.
fn bins_to_division(bins: &[Vec<usize>], items: &[Item]) -> Vec<LabelSet> {
    let mut division: Vec<LabelSet> = Vec::new();
    for bin in bins {
        let mut leaves = LabelSet::new();
        for &index in bin {
            leaves.extend(items[index].1.iter().cloned());
        }
        if !leaves.is_empty() {
            division.push(leaves);
        }
    }
    division.sort();
    division
}

/// ``_allocations``: up to ``n_divisions`` distinct partitions of ``items``.
fn allocations(
    py: Python<'_>,
    items: &[Item],
    max_subset_size: usize,
    n_divisions: usize,
    seed: i64,
) -> PyResult<Vec<Vec<LabelSet>>> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let mut items: Vec<Item> = items.to_vec();
    items.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let weights: Vec<usize> = items.iter().map(|(_, leaves)| leaves.len()).collect();
    if weights.iter().sum::<usize>() <= max_subset_size {
        let mut everything = LabelSet::new();
        for (_, leaves) in &items {
            everything.extend(leaves.iter().cloned());
        }
        return Ok(vec![vec![everything]]);
    }
    let min_size = 3usize;
    let mut chosen: Vec<Vec<LabelSet>> = Vec::new();
    let mut seen: FxHashSet<BTreeSet<LabelSet>> = FxHashSet::default();
    let mut accept = |bins: &[Vec<usize>], floor: usize, chosen: &mut Vec<Vec<LabelSet>>| {
        let division = bins_to_division(bins, &items);
        if division.iter().any(|s| s.len() < floor || s.len() > max_subset_size) {
            return;
        }
        let key: BTreeSet<LabelSet> = division.iter().cloned().collect();
        if key.is_empty() || seen.contains(&key) {
            return;
        }
        seen.insert(key);
        chosen.push(division);
    };
    let mut packings: Vec<Vec<Vec<usize>>> = Vec::new();
    for attempt in 0..n_divisions {
        let rng = python_random(py, seed + attempt as i64)?;
        if attempt == 0 {
            packings.push(smallest_first_pack(py, &weights, max_subset_size, min_size, &rng)?);
        } else {
            packings.push(random_pack(py, &weights, max_subset_size, min_size, &rng, None)?);
        }
    }
    for bins in &packings {
        accept(bins, min_size, &mut chosen);
    }
    if chosen.len() < n_divisions && items.len() <= ENUM_MAX_ITEMS {
        let mut floor = min_size;
        let mut parts = enumerate_partitions(&weights, max_subset_size, min_size, 20000);
        if parts.is_empty() {
            floor = 1;
            parts = enumerate_partitions(&weights, max_subset_size, 1, 20000);
        }
        let perm = shuffle_permutation(py, &python_random(py, seed)?, parts.len())?;
        for &p in &perm {
            if chosen.len() >= n_divisions {
                break;
            }
            accept(&parts[p], floor, &mut chosen);
        }
    }
    if chosen.is_empty() {
        for bins in &packings {
            accept(bins, 1, &mut chosen);
        }
    }
    if chosen.is_empty() {
        return Err(dimple_error(
            py,
            format!("No partition puts every subset in [1, max_subset_size] ({max_subset_size})."),
        ));
    }
    Ok(chosen)
}

/// ``_items_of``: each non-member child clade of a member blob, with the
/// leaves of nested foreign blobs subtracted and oversized clades split.
fn items_of<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
    members: &[u64],
    other_blob_labels: &FxHashSet<String>,
    max_subset_size: usize,
) -> PyResult<Vec<Item>> {
    let this = net.borrow();
    let member_labels: FxHashSet<String> = members.iter().map(|&u| this.label(py, u)).collect();
    let present: Vec<u64> = members.iter().copied().filter(|u| this.nodes.contains_key(u)).collect();
    if present.is_empty() {
        return Ok(Vec::new());
    }
    let root = top_member(&this, &present);
    let others: Vec<u64> = this
        .nodes
        .keys()
        .copied()
        .filter(|&u| other_blob_labels.contains(&this.label(py, u)))
        .collect();
    let mut items: Vec<Item> = Vec::new();
    let mut stack: Vec<u64> = vec![root];
    // Explicit pre-order walk; children are pushed reversed so the visit
    // order is the reference's recursion order.
    while let Some(node) = stack.pop() {
        let label = this.label(py, node);
        if member_labels.contains(&label) {
            let children = this.children(node);
            for &child in children.iter().rev() {
                stack.push(child);
            }
            continue;
        }
        if other_blob_labels.contains(&label) {
            continue;
        }
        let mut leaves = labels_under(py, &this, node);
        let below = this.subtree_uids(node);
        for &other in &others {
            if other == node {
                leaves.clear();
                break;
            }
            if below.contains(&other) {
                leaves = minus(&leaves, &labels_under(py, &this, other));
            }
        }
        if leaves.is_empty() {
            continue;
        }
        let parts = split_clade(py, net, &leaves, max_subset_size)?;
        for (index, part) in parts.into_iter().enumerate() {
            let name = if index == 0 { label.clone() } else { format!("{label}:{index}") };
            items.push((name, part));
        }
    }
    Ok(items)
}

/// ``_classify_prunes``: a prune whose anchor already left the residual
/// blob waits for another prune.
fn classify_prunes(pruned: &mut [Prune<'_>], residual: &LabelSet) {
    for prune in pruned.iter_mut() {
        if !prune.anchor_leaves.is_empty() && !overlaps(&prune.anchor_leaves, residual) {
            prune.source_type = "prune".to_string();
        }
    }
}

fn divisions_to_py<'py>(py: Python<'py>, divisions: &[Vec<LabelSet>]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for division in divisions {
        out.append(PyList::new(py, division.iter().map(|s| frozenset(py, s)).collect::<PyResult<Vec<_>>>()?)?)?;
    }
    Ok(out)
}

fn blob_job<'py>(
    py: Python<'py>,
    name: &str,
    tob_leaves: &LabelSet,
    external_siblings: &LabelSet,
    divisions: &[Vec<LabelSet>],
    pruned: &[Prune<'py>],
    is_root: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let cls = py.import("phynetpy._dimple")?.getattr("BlobJob")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("name", name)?;
    kwargs.set_item("tob_leaves", frozenset(py, tob_leaves)?)?;
    kwargs.set_item("external_siblings", frozenset(py, external_siblings)?)?;
    kwargs.set_item("divisions", divisions_to_py(py, divisions)?)?;
    kwargs.set_item("pruned", PyList::new(py, pruned.iter().map(|p| p.to_py(py)).collect::<PyResult<Vec<_>>>()?)?)?;
    kwargs.set_item("is_root", is_root)?;
    cls.call((), Some(&kwargs))
}

fn uid_list(py: Python<'_>, net: &Network, nodes: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    let mut out = Vec::new();
    for item in nodes.try_iter()? {
        let item = item?;
        match net.uid_of_any(&item) {
            Some(u) => out.push(u),
            None => return Err(network_error(py, "Node not found in graph.")),
        }
    }
    Ok(out)
}

fn pairs_from_any(obj: &Bound<'_, PyAny>) -> PyResult<FxHashSet<LabelPair>> {
    let mut out: FxHashSet<LabelPair> = FxHashSet::default();
    for item in obj.try_iter()? {
        out.insert(item?.extract::<(String, String)>()?);
    }
    Ok(out)
}

fn items_from_any(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Item>> {
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        let item = item?;
        let name: String = item.get_item(0)?.extract()?;
        out.push((name, labels_from_any(&item.get_item(1)?)?));
    }
    Ok(out)
}

fn items_to_py<'py>(py: Python<'py>, items: &[Item]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for (name, leaves) in items {
        out.append(PyTuple::new(py, [PyString::new(py, name).into_any(), frozenset(py, leaves)?.into_any()])?)?;
    }
    Ok(out)
}

fn bins_to_py<'py>(py: Python<'py>, bins: &[Vec<usize>]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for bin in bins {
        out.append(PyList::new(py, bin.iter().copied())?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_dp_forbidden")]
fn forbidden_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, blobs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PySet>> {
    let this = net.borrow();
    let pairs = forbidden_pairs(py, &this, &uid_list(py, &this, blobs)?);
    let out = PySet::empty(py)?;
    for (a, b) in pairs {
        out.add(PyTuple::new(py, [a, b])?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_dp_mega_groups")]
fn mega_groups_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, blobs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let this = net.borrow();
    let groups = mega_groups(&this, &uid_list(py, &this, blobs)?);
    let out = PyList::empty(py);
    for group in groups {
        out.append(PyList::new(py, group.iter().map(|u| this.node_obj(*u).bind(py)))?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_dp_top_member")]
fn top_member_py(py: Python<'_>, net: &Bound<'_, Network>, members: &Bound<'_, PyAny>) -> PyResult<Py<Node>> {
    let this = net.borrow();
    let members = uid_list(py, &this, members)?;
    if members.is_empty() {
        return Err(PyIndexError::new_err("list index out of range"));
    }
    Ok(this.node_obj(top_member(&this, &members)).clone_ref(py))
}

#[pyfunction]
#[pyo3(name = "_dp_candidate_cuts")]
fn candidate_cuts_py<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
    forbidden: &Bound<'py, PyAny>,
    blob_set: &Bound<'py, PyAny>,
    max_subset_size: usize,
) -> PyResult<Bound<'py, PyList>> {
    let this = net.borrow();
    let blobs: FxHashSet<u64> = uid_list(py, &this, blob_set)?.into_iter().collect();
    let cuts = candidate_cuts(py, &this, &pairs_from_any(forbidden)?, &blobs, max_subset_size);
    let out = PyList::empty(py);
    for (count, euid) in cuts {
        out.append(PyTuple::new(py, [count.into_pyobject(py)?.into_any(), this.edge_obj(euid).bind(py).clone().into_any()])?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_dp_oversized")]
fn oversized_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, blobs: &Bound<'py, PyAny>, max_subset_size: usize) -> PyResult<Bound<'py, PyList>> {
    let this = net.borrow();
    // A blob that has left the graph is skipped, as the reference did.
    let mut uids = Vec::new();
    for item in blobs.try_iter()? {
        if let Some(u) = this.uid_of_any(&item?) {
            uids.push(u);
        }
    }
    PyList::new(py, oversized(&this, &uids, max_subset_size).into_iter().map(|u| this.node_obj(u).bind(py)))
}

#[pyfunction]
#[pyo3(name = "_dp_prune_piece")]
fn prune_piece_py<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
    blobs: &Bound<'py, PyAny>,
    forbidden: &Bound<'py, PyAny>,
    max_subset_size: usize,
) -> PyResult<Bound<'py, PyList>> {
    let blobs = uid_list(py, &net.borrow(), blobs)?;
    let pruned = prune_piece(py, net, &blobs, &pairs_from_any(forbidden)?, max_subset_size)?;
    PyList::new(py, pruned.iter().map(|p| p.to_py(py)).collect::<PyResult<Vec<_>>>()?)
}

#[pyfunction]
#[pyo3(name = "_dp_split_clade")]
fn split_clade_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, leaves: &Bound<'py, PyAny>, max_subset_size: usize) -> PyResult<Bound<'py, PyList>> {
    let parts = split_clade(py, net, &labels_from_any(leaves)?, max_subset_size)?;
    PyList::new(py, parts.iter().map(|s| frozenset(py, s)).collect::<PyResult<Vec<_>>>()?)
}

#[pyfunction]
#[pyo3(name = "_dp_enumerate_partitions", signature = (weights, max_size, min_size, cap=20000))]
fn enumerate_partitions_py<'py>(py: Python<'py>, weights: Vec<usize>, max_size: usize, min_size: usize, cap: usize) -> PyResult<Bound<'py, PyList>> {
    let parts = enumerate_partitions(&weights, max_size, min_size, cap);
    let out = PyList::empty(py);
    for bins in &parts {
        out.append(bins_to_py(py, bins)?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_dp_random_pack", signature = (weights, max_size, min_size, rng, items=None))]
fn random_pack_py<'py>(
    py: Python<'py>,
    weights: Vec<usize>,
    max_size: usize,
    min_size: usize,
    rng: &Bound<'py, PyAny>,
    items: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyList>> {
    let bins = random_pack(py, &weights, max_size, min_size, rng, items.as_deref())?;
    bins_to_py(py, &bins)
}

#[pyfunction]
#[pyo3(name = "_dp_smallest_first_pack")]
fn smallest_first_pack_py<'py>(py: Python<'py>, weights: Vec<usize>, max_size: usize, min_size: usize, rng: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let bins = smallest_first_pack(py, &weights, max_size, min_size, rng)?;
    bins_to_py(py, &bins)
}

#[pyfunction]
#[pyo3(name = "_dp_bins_to_division")]
fn bins_to_division_py<'py>(py: Python<'py>, bins: Vec<Vec<usize>>, items: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let division = bins_to_division(&bins, &items_from_any(items)?);
    PyList::new(py, division.iter().map(|s| frozenset(py, s)).collect::<PyResult<Vec<_>>>()?)
}

#[pyfunction]
#[pyo3(name = "_dp_allocations")]
fn allocations_py<'py>(py: Python<'py>, items: &Bound<'py, PyAny>, max_subset_size: usize, n_divisions: usize, seed: i64) -> PyResult<Bound<'py, PyList>> {
    divisions_to_py(py, &allocations(py, &items_from_any(items)?, max_subset_size, n_divisions, seed)?)
}

#[pyfunction]
#[pyo3(name = "_dp_items_of")]
fn items_of_py<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
    members: &Bound<'py, PyAny>,
    other_blob_labels: &Bound<'py, PyAny>,
    max_subset_size: usize,
) -> PyResult<Bound<'py, PyList>> {
    let members: Vec<u64> = {
        let this = net.borrow();
        let mut out = Vec::new();
        for item in members.try_iter()? {
            let item = item?;
            // Members may have left the graph; those are skipped by ``_items_of``.
            match this.uid_of_any(&item) {
                Some(u) => out.push(u),
                None => {}
            }
        }
        out
    };
    let others: FxHashSet<String> = labels_from_any(other_blob_labels)?.into_iter().collect();
    items_to_py(py, &items_of(py, net, &members, &others, max_subset_size)?)
}

#[pyfunction]
#[pyo3(name = "_dp_classify_prunes")]
fn classify_prunes_py(py: Python<'_>, pruned: &Bound<'_, PyAny>, residual: &Bound<'_, PyAny>) -> PyResult<()> {
    let residual = labels_from_any(residual)?;
    for prune in pruned.try_iter()? {
        let prune = prune?;
        let anchor = labels_from_any(&prune.getattr("anchor_leaves")?)?;
        if !anchor.is_empty() && !overlaps(&anchor, &residual) {
            prune.setattr("source_type", PyString::new(py, "prune"))?;
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "_dp_partition_scaffold", signature = (net, max_subset_size=12, n_divisions=15, seed=0, outgroup=None))]
fn partition_scaffold<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
    max_subset_size: usize,
    n_divisions: usize,
    seed: i64,
    outgroup: Option<String>,
) -> PyResult<Bound<'py, PyList>> {
    if max_subset_size < 4 {
        return Err(dimple_error(py, "max_subset_size must be at least 4."));
    }
    if n_divisions < 1 {
        return Err(dimple_error(py, "n_divisions must be at least 1."));
    }
    {
        let this = net.borrow();
        for (&uid, entry) in this.nodes.iter() {
            if entry.obj.borrow(py).is_retic || this.in_deg(uid) > 1 {
                return Err(dimple_error(
                    py,
                    "The tree of blobs contains a reticulation. A blob is a polytomy on a tree.",
                ));
            }
        }
    }
    let work = strip_outgroup(py, net, outgroup)?;
    ensure_internal_labels(py, &work)?;
    let blobs = blob_uids(&work.borrow());
    if blobs.is_empty() {
        let leaves = leaf_labels(py, &work.borrow());
        if leaves.len() > max_subset_size {
            return Err(dimple_error(
                py,
                "The scaffold has no polytomy and more leaves than max_subset_size.",
            ));
        }
        let divisions: Vec<Vec<LabelSet>> = if leaves.len() >= 3 { vec![vec![leaves.clone()]] } else { Vec::new() };
        let job = blob_job(py, "tree", &leaves, &LabelSet::new(), &divisions, &[], true)?;
        return PyList::new(py, [job]);
    }
    let (forbidden, groups) = {
        let this = work.borrow();
        (forbidden_pairs(py, &this, &blobs), mega_groups(&this, &blobs))
    };
    let mut jobs: Vec<(String, Bound<'py, PyAny>)> = Vec::new();
    for group in groups {
        let (member_labels, other_labels, full_leaves, siblings, is_root) = {
            let this = work.borrow();
            let top = top_member(&this, &group);
            let member_labels: FxHashSet<String> = group.iter().map(|&u| this.label(py, u)).collect();
            let other_labels: FxHashSet<String> = blobs
                .iter()
                .map(|&u| this.label(py, u))
                .filter(|l| !member_labels.contains(l))
                .collect();
            let full_leaves = labels_under(py, &this, top);
            let parents = this.parents(top);
            let (siblings, is_root) = match parents.first() {
                Some(&parent) => (minus(&labels_under(py, &this, parent), &full_leaves), false),
                None => (LabelSet::new(), true),
            };
            (member_labels, other_labels, full_leaves, siblings, is_root)
        };
        let piece = induced_by_taxa(py, &work, full_leaves.iter())?;
        ensure_internal_labels(py, &piece)?;
        let nested: Vec<u64> = {
            let this = piece.borrow();
            this.nodes
                .keys()
                .copied()
                .filter(|&u| other_labels.contains(&this.label(py, u)) && this.out_deg(u) > 2)
                .collect()
        };
        for nested_blob in nested {
            let parent = {
                let this = piece.borrow();
                if !this.nodes.contains_key(&nested_blob) {
                    continue;
                }
                match this.parents(nested_blob).first() {
                    Some(&p) => p,
                    None => continue,
                }
            };
            detach(py, &piece, parent, nested_blob)?;
        }
        let piece_blobs: Vec<u64> = {
            let this = piece.borrow();
            this.nodes
                .keys()
                .copied()
                .filter(|&u| member_labels.contains(&this.label(py, u)) && this.out_deg(u) > 2)
                .collect()
        };
        let mut pruned = prune_piece(py, &piece, &piece_blobs, &forbidden, max_subset_size)?;
        let residual = leaf_labels(py, &piece.borrow());
        classify_prunes(&mut pruned, &residual);
        let mut items = items_of(py, &piece, &piece_blobs, &other_labels, max_subset_size)?;
        if items.is_empty() && !residual.is_empty() {
            items = vec![("residual".to_string(), residual.clone())];
        }
        let divisions = allocations(py, &items, max_subset_size, n_divisions, seed)?;
        let mut sorted_members: Vec<&String> = member_labels.iter().collect();
        sorted_members.sort();
        let name = sorted_members.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("+");
        let job = blob_job(py, &name, &full_leaves, &siblings, &divisions, &pruned, is_root)?;
        jobs.push((name, job));
    }
    jobs.sort_by(|a, b| a.0.cmp(&b.0));
    PyList::new(py, jobs.into_iter().map(|(_, job)| job))
}

// ── Reticulation records ───────────────────────────────────────────────

/// The iteration order of a CPython ``set`` built by adding objects with
/// these hashes in this order (``Network.in_edges`` returns such a set).
fn cpython_set_order(hashes: &[u64]) -> Vec<usize> {
    let mut set = PySetSim::new();
    for (index, &hash) in hashes.iter().enumerate() {
        set.add_hashed(index, hash as usize);
    }
    set.iter().collect()
}

/// ``list(net.in_edges(node))``: the in-edges in Python set order.
fn in_edges_set_order(py: Python<'_>, net: &Network, uid: u64) -> Vec<u64> {
    let edges = net.in_edge_uids(uid);
    let hashes: Vec<u64> = edges.iter().map(|e| net.edge_obj(*e).borrow(py).hash).collect();
    cpython_set_order(&hashes).into_iter().map(|i| edges[i]).collect()
}

/// ``_major_edges``: the in-edge the major tree keeps at every node of
/// in-degree two or more, in vertex order.
fn major_edges(py: Python<'_>, net: &Network) -> IndexMap<u64, u64, FxBuildHasher> {
    let mut kept: IndexMap<u64, u64, FxBuildHasher> = IndexMap::default();
    for &uid in net.nodes.keys() {
        if net.in_deg(uid) < 2 {
            continue;
        }
        let incoming = sorted_in_edges(py, net, uid);
        kept.insert(uid, incoming[0]);
    }
    kept
}

/// ``_view``: a copy of ``net`` without the edges in ``drop``, with
/// childless non-leaf nodes pruned upward.  Returns the copy and the
/// old-uid -> new-uid map.
fn view_without<'py>(py: Python<'py>, net: &Bound<'py, Network>, drop: &[u64]) -> PyResult<(Bound<'py, Network>, FxHashMap<u64, u64>)> {
    let (view, node_map) = copy_with_map(py, net)?;
    let leaves = leaf_labels(py, &net.borrow());
    for &euid in drop {
        let (su, du) = {
            let this = net.borrow();
            match this.edges.get(&euid) {
                Some(e) => (e.su, e.du),
                None => continue,
            }
        };
        let target = {
            let v = view.borrow();
            edge_uid_between(py, &v, node_map[&su], node_map[&du])?
        };
        remove_edge(&view, target);
    }
    let mut pruning = true;
    while pruning {
        pruning = false;
        let snapshot: Vec<u64> = view.borrow().nodes.keys().copied().collect();
        for uid in snapshot {
            let childless = {
                let v = view.borrow();
                match v.nodes.get(&uid) {
                    Some(e) => v.out_deg(uid) == 0 && !leaves.contains(&e.obj.borrow(py).name),
                    None => false,
                }
            };
            if childless {
                remove_node(&view, uid);
                pruning = true;
            }
        }
    }
    Ok((view, node_map))
}

/// ``_attachment_leaves``: clade under the point a removed hybrid edge
/// attaches to, and its height.
fn attachment_leaves<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
    kept: &IndexMap<u64, u64, FxBuildHasher>,
    drop: &[u64],
    start: u64,
    taxa: &LabelSet,
) -> PyResult<(LabelSet, i64)> {
    let (view, node_map) = view_without(py, net, drop)?;
    let v = view.borrow();
    let this = net.borrow();
    let mut node = start;
    while !v.nodes.contains_key(&node_map[&node]) {
        if this.in_deg(node) >= 2 && kept.contains_key(&node) {
            node = this.edges[&kept[&node]].su;
        } else {
            let parents = this.parents(node);
            match parents.first() {
                Some(&p) => node = p,
                None => return Ok((LabelSet::new(), 0)),
            }
        }
    }
    let anchor = node_map[&node];
    let mut height = 0i64;
    let mut current = anchor;
    while v.out_deg(current) == 1 {
        current = v.children(current)[0];
        if v.out_deg(current) == 1 {
            height += 1;
        }
    }
    Ok((intersect(&labels_under(py, &v, anchor), taxa), height))
}

/// ``_chain_depth``: reticulations stacked directly under ``node``.
fn chain_depth(net: &Network, node: u64) -> i64 {
    let mut depth = 0i64;
    let mut current = node;
    loop {
        let children = net.children(current);
        if children.len() != 1 || net.in_deg(children[0]) < 2 {
            return depth;
        }
        depth += 1;
        current = children[0];
    }
}

/// One reticulation record, as the Python dict and as parsed fields.
struct Record<'py> {
    dict: Bound<'py, PyDict>,
    retic_leaves: LabelSet,
    kept: LabelSet,
    minor: LabelSet,
    kept_height: i64,
    minor_height: i64,
    major_gamma: f64,
    minor_gamma: f64,
    network_taxa: LabelSet,
    retic_depth: i64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Kept,
    Minor,
}

impl<'py> Record<'py> {
    fn from_dict(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let dict = obj.cast::<PyDict>()?.clone();
        let get = |key: &str| -> PyResult<Bound<'py, PyAny>> {
            dict.get_item(key)?.ok_or_else(|| PyKeyError::new_err(key.to_string()))
        };
        let get_int = |key: &str| -> PyResult<i64> {
            match dict.get_item(key)? {
                Some(v) => v.extract::<i64>(),
                None => Ok(0),
            }
        };
        Ok(Record {
            retic_leaves: labels_from_any(&get("retic_leaves")?)?,
            kept: labels_from_any(&get("kept_sibling_leaves")?)?,
            minor: labels_from_any(&get("minor_sibling_leaves")?)?,
            kept_height: get_int("kept_height")?,
            minor_height: get_int("minor_height")?,
            major_gamma: get("major_gamma")?.extract()?,
            minor_gamma: get("minor_gamma")?.extract()?,
            network_taxa: labels_from_any(&get("network_taxa")?)?,
            retic_depth: get_int("retic_depth")?,
            dict,
        })
    }

    fn side(&self, side: Side) -> &LabelSet {
        match side {
            Side::Kept => &self.kept,
            Side::Minor => &self.minor,
        }
    }

    /// The record's own frozenset for ``side``, so identity still tells
    /// the major side from the minor side on the Python side.
    fn side_object(&self, side: Side) -> PyResult<Bound<'py, PyAny>> {
        let key = match side {
            Side::Kept => "kept_sibling_leaves",
            Side::Minor => "minor_sibling_leaves",
        };
        self.dict.get_item(key)?.ok_or_else(|| PyKeyError::new_err(key.to_string()))
    }
}

fn records_from_any<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<Record<'py>>> {
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(Record::from_dict(&item?)?);
    }
    Ok(out)
}

/// ``kept`` / ``references`` entries: ``dict[Node, Edge]`` to uids on ``net``.
fn kept_map_from_any(py: Python<'_>, net: &Network, obj: &Bound<'_, PyAny>) -> PyResult<IndexMap<u64, u64, FxBuildHasher>> {
    let mut out: IndexMap<u64, u64, FxBuildHasher> = IndexMap::default();
    let dict = obj.cast::<PyDict>()?;
    for (node, edge) in dict.iter() {
        let (Some(n), Ok(e)) = (net.uid_of_any(&node), edge.cast::<Edge>()) else { continue };
        let euid = e.borrow().uid;
        if net.edges.contains_key(&euid) {
            out.insert(n, euid);
        }
    }
    let _ = py;
    Ok(out)
}

/// ``reticulation_records``.
fn reticulation_records_impl<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
    kept: Option<&IndexMap<u64, u64, FxBuildHasher>>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let this = net.borrow();
    let mut reference = major_edges(py, &this);
    if let Some(kept) = kept {
        if !kept.is_empty() {
            for (node, edge) in kept {
                if reference.contains_key(node) {
                    reference.insert(*node, *edge);
                }
            }
        }
    }
    let taxa = leaf_labels(py, &this);
    let mut retics: Vec<(String, u64)> = this
        .nodes
        .keys()
        .copied()
        .filter(|&u| this.in_deg(u) == 2)
        .map(|u| (this.label(py, u), u))
        .collect();
    retics.sort();
    let signature_fn = cached(py, &GU_SIGNATURE, "phynetpy._core", "_gu_reticulation_signature")?;
    let mut records = Vec::new();
    for (_, node) in retics {
        let mut incoming: Vec<u64> = in_edges_set_order(py, &this, node);
        incoming.sort_by(|a, b| {
            let ga = edge_gamma(py, &this, *a).unwrap_or(0.0);
            let gb = edge_gamma(py, &this, *b).unwrap_or(0.0);
            gb.partial_cmp(&ga)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| this.label(py, this.edges[a].su).cmp(&this.label(py, this.edges[b].su)))
        });
        let (major_edge, minor_edge) = (incoming[0], incoming[1]);
        let mut local = reference.clone();
        let mut current = node;
        loop {
            let children = this.children(current);
            if children.len() != 1 || this.in_deg(children[0]) < 2 {
                break;
            }
            let euid = edge_uid_between(py, &this, current, children[0])?;
            local.insert(children[0], euid);
            current = children[0];
        }
        let mut others: Vec<u64> = Vec::new();
        for (&other, &kept_edge) in &local {
            if other == node {
                continue;
            }
            for edge in in_edges_set_order(py, &this, other) {
                if edge != kept_edge {
                    others.push(edge);
                }
            }
        }
        let (base, base_map) = view_without(py, net, &others)?;
        let base_node = base_map[&node];
        if !base.borrow().nodes.contains_key(&base_node) {
            continue;
        }
        let mut with_major = others.clone();
        with_major.push(major_edge);
        let (kept_side, kept_height) = attachment_leaves(py, net, &local, &with_major, this.edges[&major_edge].su, &taxa)?;
        let mut with_minor = others.clone();
        with_minor.push(minor_edge);
        let (minor_side, minor_height) = attachment_leaves(py, net, &local, &with_minor, this.edges[&minor_edge].su, &taxa)?;
        let signature = {
            let b = base.borrow();
            let node_obj = b.node_obj(base_node).clone_ref(py);
            drop(b);
            signature_fn.call1((&base, node_obj))?
        };
        let retic_leaves = labels_under(py, &base.borrow(), base_node);
        let record = PyDict::new(py);
        record.set_item("signature", signature)?;
        record.set_item("retic_leaves", frozenset(py, &retic_leaves)?)?;
        record.set_item("kept_sibling_leaves", frozenset(py, &kept_side)?)?;
        record.set_item("minor_sibling_leaves", frozenset(py, &minor_side)?)?;
        record.set_item("kept_height", kept_height)?;
        record.set_item("minor_height", minor_height)?;
        record.set_item("major_gamma", edge_gamma(py, &this, major_edge).unwrap_or(0.0))?;
        record.set_item("minor_gamma", edge_gamma(py, &this, minor_edge).unwrap_or(0.0))?;
        record.set_item("network_taxa", frozenset(py, &taxa)?)?;
        record.set_item("retic_depth", chain_depth(&this, node))?;
        records.push(record);
    }
    Ok(records)
}

#[pyfunction]
#[pyo3(name = "_dp_reticulation_records", signature = (net, kept=None))]
fn reticulation_records<'py>(py: Python<'py>, net: &Bound<'py, Network>, kept: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyList>> {
    let kept_map = match kept {
        Some(k) if !k.is_none() => Some(kept_map_from_any(py, &net.borrow(), k)?),
        _ => None,
    };
    PyList::new(py, reticulation_records_impl(py, net, kept_map.as_ref())?)
}

/// ``round(x, 6)`` as Python computes it.
fn py_round6(py: Python<'_>, x: f64) -> PyResult<f64> {
    py.import("builtins")?.getattr("round")?.call1((x, 6))?.extract()
}

/// ``collect_reticulation_candidates``.
type KeptMaps = Vec<Option<IndexMap<u64, u64, FxBuildHasher>>>;

/// ``references``: per network, the displayed tree's kept in-edge per
/// reticulation (``dict[Node, Edge]`` or None), resolved to uids.
fn refs_from_any<'py>(py: Python<'py>, networks: &[Bound<'py, Network>], references: Option<&Bound<'py, PyAny>>) -> PyResult<Option<KeptMaps>> {
    let Some(r) = references.filter(|r| !r.is_none()) else { return Ok(None) };
    let items: Vec<Bound<'py, PyAny>> = r.try_iter()?.collect::<PyResult<_>>()?;
    let mut out: KeptMaps = Vec::with_capacity(networks.len());
    for (index, net) in networks.iter().enumerate() {
        let item = items
            .get(index)
            .ok_or_else(|| PyIndexError::new_err("list index out of range"))?;
        out.push(if item.is_none() { None } else { Some(kept_map_from_any(py, &net.borrow(), item)?) });
    }
    Ok(Some(out))
}

fn collect_candidates_impl<'py>(
    py: Python<'py>,
    networks: &[Bound<'py, Network>],
    references: Option<&[Option<IndexMap<u64, u64, FxBuildHasher>>]>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let mut records: Vec<Bound<'py, PyDict>> = Vec::new();
    for (index, net) in networks.iter().enumerate() {
        let kept = match references {
            Some(list) => list
                .get(index)
                .ok_or_else(|| PyIndexError::new_err("list index out of range"))?
                .as_ref(),
            None => None,
        };
        records.extend(reticulation_records_impl(py, net, kept)?);
    }
    let group = cached(py, &GU_GROUP, "phynetpy._core", "_gu_group_reticulations")?;
    let groups = group.call1((PyList::new(py, &records)?,))?;
    let mut representatives: Vec<Bound<'py, PyDict>> = Vec::new();
    for group in groups.try_iter()? {
        let mut best: Option<(usize, Bound<'py, PyDict>)> = None;
        for record in group?.try_iter()? {
            let record = record?.cast_into::<PyDict>()?;
            let size = record
                .get_item("retic_leaves")?
                .ok_or_else(|| PyKeyError::new_err("retic_leaves"))?
                .len()?;
            if best.as_ref().map_or(true, |(b, _)| size > *b) {
                best = Some((size, record));
            }
        }
        if let Some((_, record)) = best {
            representatives.push(record);
        }
    }
    let mut found: Vec<Bound<'py, PyDict>> = Vec::new();
    let mut seen: FxHashSet<(LabelSet, LabelSet, LabelSet, LabelSet, u64, u64)> = FxHashSet::default();
    for record in representatives {
        let r = Record::from_dict(record.as_any())?;
        let key = (
            r.network_taxa.clone(),
            r.retic_leaves.clone(),
            r.minor.clone(),
            r.kept.clone(),
            py_round6(py, r.major_gamma)?.to_bits(),
            py_round6(py, r.minor_gamma)?.to_bits(),
        );
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        found.push(record);
    }
    Ok(found)
}

#[pyfunction]
#[pyo3(name = "_dp_collect_reticulation_candidates", signature = (networks, references=None))]
fn collect_reticulation_candidates<'py>(py: Python<'py>, networks: &Bound<'py, PyAny>, references: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyList>> {
    let networks = tree_list(networks)?;
    let refs = refs_from_any(py, &networks, references)?;
    PyList::new(py, collect_candidates_impl(py, &networks, refs.as_deref())?)
}

/// ``_child_clade``: MRCA of the reticulation's leaves, if it is their clade.
fn child_clade(py: Python<'_>, net: &Network, record: &Record<'_>) -> PyResult<Option<u64>> {
    let Some(node) = mrca_or_none(py, net, &record.retic_leaves)? else { return Ok(None) };
    let below = intersect(&labels_under(py, net, node), &record.network_taxa);
    Ok(if below == record.retic_leaves { Some(node) } else { None })
}

/// ``_side_leaves``: leaves beside ``child`` under ``parent``, within ``taxa``.
fn side_leaves(py: Python<'_>, net: &Network, child: u64, parent: u64, taxa: &LabelSet) -> LabelSet {
    let mut current = LabelSet::new();
    for sibling in net.children(parent) {
        if sibling != child {
            current.extend(labels_under(py, net, sibling));
        }
    }
    intersect(&current, taxa)
}

/// ``choose_inserted_side``.
fn choose_inserted_side_impl(py: Python<'_>, net: &Network, record: &Record<'_>) -> PyResult<Side> {
    let Some(child) = child_clade(py, net, record)? else { return Ok(Side::Minor) };
    let parents = net.parents(child);
    if parents.is_empty() || record.kept.is_empty() {
        return Ok(Side::Minor);
    }
    let mut sorted_parents: Vec<(String, u64)> = parents.into_iter().map(|p| (net.label(py, p), p)).collect();
    sorted_parents.sort();
    let mut decided: Option<Side> = None;
    let mut best = -1.0_f64;
    for (_, parent) in sorted_parents {
        let current = side_leaves(py, net, child, parent, &record.network_taxa);
        if current.is_empty() {
            continue;
        }
        let in_minor = current.is_subset(&record.minor);
        let in_kept = current.is_subset(&record.kept);
        let (quality, choice) = if in_minor && !in_kept {
            (1.0, Side::Kept)
        } else if in_kept && !in_minor {
            (1.0, Side::Minor)
        } else {
            let kept_overlap = intersect(&current, &record.kept).len() as f64
                / (current.union(&record.kept).count().max(1)) as f64;
            let minor_overlap = intersect(&current, &record.minor).len() as f64
                / (current.union(&record.minor).count().max(1)) as f64;
            if kept_overlap >= minor_overlap {
                (kept_overlap, Side::Minor)
            } else {
                (minor_overlap, Side::Kept)
            }
        };
        if quality > best {
            best = quality;
            decided = Some(choice);
        }
    }
    Ok(decided.unwrap_or(Side::Minor))
}

/// ``_destination_edge``: edge above ``child`` on the side the backbone holds.
fn destination_edge(py: Python<'_>, net: &Network, child: u64, held: &LabelSet, taxa: &LabelSet) -> Option<u64> {
    let incoming = in_edges_set_order(py, net, child);
    if incoming.is_empty() {
        return None;
    }
    if incoming.len() == 1 {
        return Some(incoming[0]);
    }
    let mut best: Option<(usize, String, u64)> = None;
    for euid in incoming {
        let src = net.edges[&euid].su;
        let covered = side_leaves(py, net, child, src, taxa);
        let key = (intersect(&covered, held).len(), net.label(py, src));
        if best.as_ref().map_or(true, |(c, l, _)| (key.0, &key.1) > (*c, l)) {
            best = Some((key.0, key.1, euid));
        }
    }
    best.map(|(_, _, e)| e)
}

/// ``candidate_edges``: edges where the missing parental side can attach.
fn candidate_edges_impl(py: Python<'_>, net: &Network, record: &Record<'_>) -> PyResult<Vec<u64>> {
    let Some(child) = child_clade(py, net, record)? else { return Ok(Vec::new()) };
    let side = choose_inserted_side_impl(py, net, record)?;
    let target = record.side(side);
    let excluded: FxHashSet<u64> = net.subtree_uids(child).into_iter().collect();
    let labels = label_map(py, net)?;
    let mut exact: Vec<u64> = Vec::new();
    let mut partial: Vec<(usize, String, u64)> = Vec::new();
    for (&euid, entry) in net.edges.iter() {
        if excluded.contains(&entry.du) {
            continue;
        }
        let below = labels
            .get(&entry.du)
            .ok_or_else(|| PyKeyError::new_err(net.node_obj(entry.du).clone_ref(py).into_any()))?;
        let covered = intersect(below, &record.network_taxa);
        if covered.is_empty() || !covered.is_subset(target) {
            continue;
        }
        if entry.du == child {
            continue;
        }
        if &covered == target {
            exact.push(euid);
        } else {
            partial.push((covered.len(), covered.iter().cloned().collect::<String>(), euid));
        }
    }
    if !exact.is_empty() {
        let mut depth: FxHashMap<u64, i64> = FxHashMap::default();
        let mut frontier: Vec<(u64, i64)> = net.root_uids().into_iter().map(|r| (r, 0)).collect();
        while let Some((node, level)) = frontier.pop() {
            if depth.contains_key(&node) {
                continue;
            }
            depth.insert(node, level);
            frontier.extend(net.children(node).into_iter().map(|c| (c, level + 1)));
        }
        exact.sort_by(|a, b| {
            let (ea, eb) = (&net.edges[a], &net.edges[b]);
            let da = depth.get(&ea.du).copied().unwrap_or(0);
            let db = depth.get(&eb.du).copied().unwrap_or(0);
            db.cmp(&da)
                .then_with(|| net.label(py, ea.su).cmp(&net.label(py, eb.su)))
                .then_with(|| net.label(py, ea.du).cmp(&net.label(py, eb.du)))
        });
        let height = match side {
            Side::Kept => record.kept_height,
            Side::Minor => record.minor_height,
        };
        if height > 0 && (height as usize) < exact.len() {
            let picked = exact.remove(height as usize);
            exact.insert(0, picked);
        }
        return Ok(exact);
    }
    partial.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    Ok(partial.into_iter().map(|(_, _, e)| e).collect())
}

/// ``_edge_distance``: undirected distance, or ``limit + 1`` when farther.
fn edge_distance(net: &Network, start: u64, goal: u64, limit: usize) -> usize {
    if start == goal {
        return 0;
    }
    let mut seen: FxHashSet<u64> = FxHashSet::default();
    seen.insert(start);
    let mut frontier = vec![start];
    for distance in 1..=limit {
        let mut next = Vec::new();
        for node in frontier {
            let mut neighbours = net.children(node);
            neighbours.extend(net.parents(node));
            for neighbour in neighbours {
                if neighbour == goal {
                    return distance;
                }
                if seen.insert(neighbour) {
                    next.push(neighbour);
                }
            }
        }
        frontier = next;
    }
    limit + 1
}

/// ``_sort_records``: Kahn order on "i's child leaves meet j's inserted side".
fn sort_records<'py>(py: Python<'py>, records: Vec<Record<'py>>, net: &Network) -> PyResult<Vec<Record<'py>>> {
    let mut sides: Vec<LabelSet> = Vec::with_capacity(records.len());
    for record in &records {
        let side = choose_inserted_side_impl(py, net, record)?;
        sides.push(record.side(side).clone());
    }
    let count = records.len();
    let mut dependencies: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); count];
    for i in 0..count {
        let child = &records[i].retic_leaves;
        for j in 0..count {
            if i == j {
                continue;
            }
            if overlaps(child, &sides[j]) {
                dependencies[j].insert(i);
            }
            if overlaps(child, &records[j].retic_leaves) && records[i].retic_depth > records[j].retic_depth {
                dependencies[j].insert(i);
            }
        }
    }
    let mut indegree: Vec<usize> = dependencies.iter().map(|d| d.len()).collect();
    let mut queue: VecDeque<usize> = (0..count).filter(|&i| indegree[i] == 0).collect();
    let mut order: Vec<usize> = Vec::new();
    while let Some(index) = queue.pop_front() {
        order.push(index);
        for other in 0..count {
            if dependencies[other].remove(&index) {
                indegree[other] -= 1;
                if indegree[other] == 0 {
                    queue.push_back(other);
                }
            }
        }
    }
    for index in 0..count {
        if !order.contains(&index) {
            order.push(index);
        }
    }
    let mut slots: Vec<Option<Record<'py>>> = records.into_iter().map(Some).collect();
    Ok(order.into_iter().map(|i| slots[i].take().expect("each index once")).collect())
}

/// ``_insert_hybrid``: split ``source`` and ``destination`` and set the
/// inheritance probabilities.  Returns the labels of the new hybrid edge's
/// source node and of the reticulation node.
fn insert_hybrid(py: Python<'_>, net: &Bound<'_, Network>, source: u64, destination: u64, major_gamma: f64, minor_gamma: f64) -> PyResult<(String, String)> {
    let (dest_gamma, src_gamma, dest_child_retic, src_child_retic, before, source_obj, dest_obj) = {
        let this = net.borrow();
        let (s, d) = (&this.edges[&source], &this.edges[&destination]);
        let dest_child_retic = this.node_obj(d.du).borrow(py).is_retic;
        let src_child_retic = this.node_obj(s.du).borrow(py).is_retic;
        let dest_gamma = if dest_child_retic { edge_gamma(py, &this, destination) } else { None };
        let src_gamma = if src_child_retic { edge_gamma(py, &this, source) } else { None };
        let before: FxHashSet<u64> = this.nodes.keys().copied().collect();
        (dest_gamma, src_gamma, dest_child_retic, src_child_retic, before, s.obj.clone_ref(py), d.obj.clone_ref(py))
    };
    let add_hybrid = cached(py, &GU_ADD_HYBRID, "phynetpy._core", "_gu_add_hybrid")?;
    add_hybrid.call1((net, source_obj, dest_obj, py.None(), py.None()))?;
    let (retic, minor, retic_in, retic_out, minor_out) = {
        let this = net.borrow();
        let created: Vec<u64> = this.nodes.keys().copied().filter(|u| !before.contains(u)).collect();
        let retic = created
            .iter()
            .copied()
            .find(|&u| this.node_obj(u).borrow(py).is_retic)
            .ok_or_else(|| PyStopIteration::new_err(()))?;
        let minor = created
            .iter()
            .copied()
            .find(|&u| u != retic)
            .ok_or_else(|| PyStopIteration::new_err(()))?;
        (retic, minor, this.in_edge_uids(retic), this.out_edge_uids(retic), this.out_edge_uids(minor))
    };
    for euid in retic_in {
        let (obj, su) = {
            let this = net.borrow();
            (this.edge_obj(euid).clone_ref(py), this.edges[&euid].su)
        };
        Edge::set_gamma(obj.bind(py), if su == minor { minor_gamma } else { major_gamma })?;
    }
    if dest_child_retic {
        if let Some(g) = dest_gamma {
            for euid in retic_out {
                let obj = net.borrow().edge_obj(euid).clone_ref(py);
                Edge::set_gamma(obj.bind(py), g)?;
            }
        }
    }
    if src_child_retic {
        if let Some(g) = src_gamma {
            for euid in minor_out {
                let (obj, du) = {
                    let this = net.borrow();
                    (this.edge_obj(euid).clone_ref(py), this.edges[&euid].du)
                };
                if du != retic {
                    Edge::set_gamma(obj.bind(py), g)?;
                }
            }
        }
    }
    populate_default_branch_lengths(py, net)?;
    let this = net.borrow();
    Ok((this.label(py, minor), this.label(py, retic)))
}

/// ``_placement_view``: ``net`` without the hybrid edges placed so far.
fn placement_view<'py>(py: Python<'py>, net: &Bound<'py, Network>, inserted: &FxHashSet<LabelPair>) -> PyResult<Bound<'py, Network>> {
    if inserted.is_empty() {
        return Ok(net.clone());
    }
    let drop: Vec<u64> = {
        let this = net.borrow();
        this.edges
            .iter()
            .filter(|(_, e)| inserted.contains(&(this.label(py, e.su), this.label(py, e.du))))
            .map(|(e, _)| *e)
            .collect()
    };
    Ok(view_without(py, net, &drop)?.0)
}

fn is_network_error(py: Python<'_>, err: &PyErr) -> bool {
    match py.import("phynetpy.Network").and_then(|m| m.getattr("NetworkError")) {
        Ok(ty) => err.is_instance(py, &ty),
        Err(_) => false,
    }
}

/// ``_edge_by_labels``: the edge of ``net`` between the nodes named like
/// ``(src_label, dest_label)``, or None.
fn edge_by_labels(py: Python<'_>, net: &Network, src_label: &str, dest_label: &str) -> PyResult<Option<u64>> {
    let (Some(src), Some(dest)) = (node_by_name(net, src_label), node_by_name(net, dest_label)) else {
        return Ok(None);
    };
    match net.edge_between(py, src, dest, None, None) {
        Ok(edge) => Ok(Some(edge.borrow(py).uid)),
        Err(err) => {
            if err.is_instance_of::<PyAssertionError>(py)
                || err.is_instance_of::<PyKeyError>(py)
                || is_network_error(py, &err)
            {
                Ok(None)
            } else {
                Err(err)
            }
        }
    }
}

/// ``_near_accepted``: whether ``anchor`` is within two edges of a
/// reticulation whose child overlaps ``child``.
fn near_accepted(py: Python<'_>, net: &Network, anchor: u64, child: &LabelSet, accepted: &[LabelSet]) -> PyResult<bool> {
    for existing in accepted {
        if !overlaps(child, existing) {
            continue;
        }
        if let Some(retic) = mrca_or_none(py, net, existing)? {
            if edge_distance(net, anchor, retic, 2) <= 2 {
                return Ok(true);
            }
        }
    }
    let mut labels: Option<IndexMap<u64, LabelSet, FxBuildHasher>> = None;
    for &node in net.nodes.keys() {
        if net.in_deg(node) < 2 {
            continue;
        }
        if labels.is_none() {
            labels = Some(label_map(py, net)?);
        }
        let below = labels
            .as_ref()
            .and_then(|m| m.get(&node))
            .ok_or_else(|| PyKeyError::new_err(net.node_obj(node).clone_ref(py).into_any()))?;
        if overlaps(child, below) && edge_distance(net, anchor, node, 2) <= 2 {
            return Ok(true);
        }
    }
    Ok(false)
}

fn retic_trial<'py>(
    py: Python<'py>,
    child: &LabelSet,
    edge_leaves: &LabelSet,
    accepted: bool,
    before: f64,
    after: f64,
    reason: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let cls = py.import("phynetpy._dimple")?.getattr("ReticTrial")?;
    cls.call1((frozenset(py, child)?, frozenset(py, edge_leaves)?, accepted, before, after, reason))
}

/// ``place_reticulations``.
fn place_reticulations_impl<'py>(
    py: Python<'py>,
    tree: &Bound<'py, Network>,
    networks: &[Bound<'py, Network>],
    score_fn: &Bound<'py, PyAny>,
    min_gain: f64,
    references: Option<&[Option<IndexMap<u64, u64, FxBuildHasher>>]>,
) -> PyResult<(Bound<'py, Network>, Vec<Bound<'py, PyAny>>)> {
    let mut current = tree.clone();
    let base: f64 = score_fn.call1((&current,))?.extract()?;
    if !base.is_finite() {
        return Err(PyRuntimeError::new_err(format!(
            "Pseudo-likelihood of the base tree is not finite ({}).",
            crate::util::py_float_repr(py, base)?
        )));
    }
    let record_dicts = collect_candidates_impl(py, networks, references)?;
    let records: Vec<Record<'py>> = record_dicts.iter().map(|d| Record::from_dict(d.as_any())).collect::<PyResult<_>>()?;
    let mut inserted: FxHashSet<LabelPair> = FxHashSet::default();
    let mut view = placement_view(py, &current, &inserted)?;
    let mut pending: VecDeque<Record<'py>> = sort_records(py, records, &view.borrow())?.into_iter().collect();
    let mut trials: Vec<Bound<'py, PyAny>> = Vec::new();
    let mut accepted: Vec<LabelSet> = Vec::new();
    let mut score = base;
    let empty = LabelSet::new();
    while let Some(record) = pending.pop_front() {
        let child = record.retic_leaves.clone();
        let (child_node, child_label) = {
            let v = view.borrow();
            match child_clade(py, &v, &record)? {
                Some(node) => (node, v.label(py, node)),
                None => {
                    drop(v);
                    trials.push(retic_trial(py, &child, &empty, false, score, score, "child_not_clade")?);
                    continue;
                }
            }
        };
        let mut edges = candidate_edges_impl(py, &view.borrow(), &record)?;
        {
            let cur = current.borrow();
            if let Some(anchor) = node_by_name(&cur, &child_label) {
                let below: FxHashSet<String> = cur.subtree_uids(anchor).into_iter().map(|u| cur.label(py, u)).collect();
                let v = view.borrow();
                edges.retain(|e| !below.contains(&v.label(py, v.edges[e].du)));
            }
        }
        if edges.is_empty() {
            trials.push(retic_trial(py, &child, &empty, false, score, score, "no_candidate_edge")?);
            continue;
        }
        let side = choose_inserted_side_impl(py, &view.borrow(), &record)?;
        let (major_gamma, minor_gamma, held) = match side {
            Side::Kept => (record.minor_gamma, record.major_gamma, &record.minor),
            Side::Minor => (record.major_gamma, record.minor_gamma, &record.kept),
        };
        let destination = destination_edge(py, &view.borrow(), child_node, held, &record.network_taxa);
        let Some(destination) = destination else {
            trials.push(retic_trial(py, &child, &empty, false, score, score, "no_destination")?);
            continue;
        };
        let mut best_score = f64::NEG_INFINITY;
        let mut best_trial: Option<Bound<'py, Network>> = None;
        let mut best_edge_leaves = LabelSet::new();
        let mut best_near = false;
        let mut best_pair: Option<LabelPair> = None;
        for &edge in &edges {
            if edge == destination {
                continue;
            }
            let trial = copy_net(&current)?;
            let (edge_src, edge_dst, dest_src, dest_dst) = {
                let v = view.borrow();
                let (e, d) = (&v.edges[&edge], &v.edges[&destination]);
                (v.label(py, e.su), v.label(py, e.du), v.label(py, d.su), v.label(py, d.du))
            };
            let (source, dest) = {
                let t = trial.borrow();
                (edge_by_labels(py, &t, &edge_src, &edge_dst)?, edge_by_labels(py, &t, &dest_src, &dest_dst)?)
            };
            let (Some(source), Some(dest)) = (source, dest) else { continue };
            {
                let t = trial.borrow();
                let (s, d) = (&t.edges[&source], &t.edges[&dest]);
                if source == dest || (s.su == d.su && s.du == d.du) {
                    continue;
                }
            }
            let pair = insert_hybrid(py, &trial, source, dest, major_gamma, minor_gamma)?;
            if !trial.borrow().is_acyclic_impl() {
                continue;
            }
            let trial_score: f64 = score_fn.call1((&trial,))?.extract()?;
            if trial_score > best_score {
                best_score = trial_score;
                best_trial = Some(trial);
                best_pair = Some(pair);
                let v = view.borrow();
                best_edge_leaves = labels_under(py, &v, v.edges[&edge].du);
                let cur = current.borrow();
                best_near = match node_by_name(&cur, &edge_dst) {
                    Some(anchor) => near_accepted(py, &cur, anchor, &child, &accepted)?,
                    None => false,
                };
            }
        }
        let threshold = if best_near { REDUNDANCY_THRESHOLD } else { 0.0 };
        let gain = best_score - score;
        let relative = if score != 0.0 { gain / score.abs() } else { gain };
        let keep = best_trial.is_some() && best_score.is_finite() && best_score > score + min_gain && relative >= threshold;
        trials.push(retic_trial(
            py,
            &child,
            &best_edge_leaves,
            keep,
            score,
            if best_score.is_finite() { best_score } else { score },
            "",
        )?);
        if keep {
            if let Some(trial) = best_trial {
                current = trial;
                score = best_score;
                accepted.push(child);
                if let Some(pair) = best_pair {
                    inserted.insert(pair);
                }
                view = placement_view(py, &current, &inserted)?;
                if !pending.is_empty() {
                    let remaining: Vec<Record<'py>> = pending.drain(..).collect();
                    pending = sort_records(py, remaining, &view.borrow())?.into_iter().collect();
                }
            }
        }
    }
    Ok((current, trials))
}

#[pyfunction]
#[pyo3(name = "_dp_place_reticulations", signature = (tree, networks, score_fn, min_gain=0.0, references=None))]
fn place_reticulations<'py>(
    py: Python<'py>,
    tree: &Bound<'py, Network>,
    networks: &Bound<'py, PyAny>,
    score_fn: &Bound<'py, PyAny>,
    min_gain: f64,
    references: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, Network>, Bound<'py, PyList>)> {
    let networks = tree_list(networks)?;
    let refs = refs_from_any(py, &networks, references)?;
    let (net, trials) = place_reticulations_impl(py, tree, &networks, score_fn, min_gain, refs.as_deref())?;
    Ok((net, PyList::new(py, trials)?))
}

fn node_arg(py: Python<'_>, net: &Network, node: &Bound<'_, PyAny>) -> PyResult<u64> {
    net.uid_of_any(node).ok_or_else(|| network_error(py, "Node not found in graph."))
}

fn edge_arg(py: Python<'_>, net: &Network, edge: &Bound<'_, PyAny>) -> PyResult<u64> {
    let uid = edge.cast::<Edge>()?.borrow().uid;
    if net.edges.contains_key(&uid) {
        Ok(uid)
    } else {
        Err(network_error(py, "Edge not found in graph."))
    }
}

#[pyfunction]
#[pyo3(name = "_dp_major_edges")]
fn major_edges_py<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PyDict>> {
    let this = net.borrow();
    let out = PyDict::new(py);
    for (node, edge) in major_edges(py, &this) {
        out.set_item(this.node_obj(node).bind(py), this.edge_obj(edge).bind(py))?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_dp_view")]
fn view_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, drop: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, Network>, Bound<'py, PyDict>)> {
    let edges: Vec<u64> = {
        let this = net.borrow();
        let mut out = Vec::new();
        for item in drop.try_iter()? {
            out.push(edge_arg(py, &this, &item?)?);
        }
        out
    };
    let (view, map) = view_without(py, net, &edges)?;
    let out = PyDict::new(py);
    {
        let this = net.borrow();
        let v = view.borrow();
        for (old, new) in map {
            // Nodes pruned from the view are still mapped to their copies.
            if let (Some(o), Some(n)) = (this.nodes.get(&old), v.nodes.get(&new)) {
                out.set_item(o.obj.bind(py), n.obj.bind(py))?;
            }
        }
    }
    Ok((view, out))
}

#[pyfunction]
#[pyo3(name = "_dp_attachment_leaves")]
fn attachment_leaves_py<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
    kept: &Bound<'py, PyAny>,
    drop: &Bound<'py, PyAny>,
    start: &Bound<'py, PyAny>,
    taxa: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyFrozenSet>, i64)> {
    let (kept_map, edges, start_uid) = {
        let this = net.borrow();
        let mut out = Vec::new();
        for item in drop.try_iter()? {
            out.push(edge_arg(py, &this, &item?)?);
        }
        (kept_map_from_any(py, &this, kept)?, out, node_arg(py, &this, start)?)
    };
    let (leaves, height) = attachment_leaves(py, net, &kept_map, &edges, start_uid, &labels_from_any(taxa)?)?;
    Ok((frozenset(py, &leaves)?, height))
}

#[pyfunction]
#[pyo3(name = "_dp_chain_depth")]
fn chain_depth_py(py: Python<'_>, net: &Bound<'_, Network>, node: &Bound<'_, PyAny>) -> PyResult<i64> {
    let this = net.borrow();
    Ok(chain_depth(&this, node_arg(py, &this, node)?))
}

#[pyfunction]
#[pyo3(name = "_dp_child_clade")]
fn child_clade_py(py: Python<'_>, net: &Bound<'_, Network>, record: &Bound<'_, PyAny>) -> PyResult<Option<Py<Node>>> {
    let this = net.borrow();
    Ok(child_clade(py, &this, &Record::from_dict(record)?)?.map(|u| this.node_obj(u).clone_ref(py)))
}

#[pyfunction]
#[pyo3(name = "_dp_side_leaves")]
fn side_leaves_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, child: &Bound<'py, PyAny>, parent: &Bound<'py, PyAny>, taxa: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PySet>> {
    let this = net.borrow();
    let leaves = side_leaves(py, &this, node_arg(py, &this, child)?, node_arg(py, &this, parent)?, &labels_from_any(taxa)?);
    PySet::new(py, leaves.iter().map(|l| PyString::new(py, l)))
}

#[pyfunction]
#[pyo3(name = "_dp_choose_inserted_side")]
fn choose_inserted_side_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, record: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let record = Record::from_dict(record)?;
    let side = choose_inserted_side_impl(py, &net.borrow(), &record)?;
    record.side_object(side)
}

#[pyfunction]
#[pyo3(name = "_dp_destination_edge")]
fn destination_edge_py(py: Python<'_>, net: &Bound<'_, Network>, child: &Bound<'_, PyAny>, held: &Bound<'_, PyAny>, taxa: &Bound<'_, PyAny>) -> PyResult<Option<Py<Edge>>> {
    let this = net.borrow();
    let found = destination_edge(py, &this, node_arg(py, &this, child)?, &labels_from_any(held)?, &labels_from_any(taxa)?);
    Ok(found.map(|e| this.edge_obj(e).clone_ref(py)))
}

#[pyfunction]
#[pyo3(name = "_dp_candidate_edges")]
fn candidate_edges_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, record: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let this = net.borrow();
    let edges = candidate_edges_impl(py, &this, &Record::from_dict(record)?)?;
    PyList::new(py, edges.into_iter().map(|e| this.edge_obj(e).bind(py)))
}

#[pyfunction]
#[pyo3(name = "_dp_edge_distance", signature = (net, start, goal, limit=2))]
fn edge_distance_py(py: Python<'_>, net: &Bound<'_, Network>, start: &Bound<'_, PyAny>, goal: &Bound<'_, PyAny>, limit: usize) -> PyResult<usize> {
    let this = net.borrow();
    // The reference walked by object identity: a node of another graph
    // (a stale copy with the same label) is never met.
    let start_uid = start.cast::<Node>()?.borrow().uid;
    let goal_uid = goal.cast::<Node>()?.borrow().uid;
    if start_uid == goal_uid {
        return Ok(0);
    }
    if !this.nodes.contains_key(&start_uid) || !this.nodes.contains_key(&goal_uid) {
        return Ok(limit + 1);
    }
    let _ = py;
    Ok(edge_distance(&this, start_uid, goal_uid, limit))
}

#[pyfunction]
#[pyo3(name = "_dp_sort_records")]
fn sort_records_py<'py>(py: Python<'py>, records: &Bound<'py, PyAny>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PyList>> {
    let sorted = sort_records(py, records_from_any(records)?, &net.borrow())?;
    PyList::new(py, sorted.into_iter().map(|r| r.dict))
}

#[pyfunction]
#[pyo3(name = "_dp_insert_hybrid")]
fn insert_hybrid_py(py: Python<'_>, net: &Bound<'_, Network>, source: &Bound<'_, PyAny>, destination: &Bound<'_, PyAny>, major_gamma: f64, minor_gamma: f64) -> PyResult<(String, String)> {
    let (s, d) = {
        let this = net.borrow();
        (edge_arg(py, &this, source)?, edge_arg(py, &this, destination)?)
    };
    insert_hybrid(py, net, s, d, major_gamma, minor_gamma)
}

#[pyfunction]
#[pyo3(name = "_dp_placement_view")]
fn placement_view_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, inserted: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Network>> {
    placement_view(py, net, &pairs_from_any(inserted)?)
}

#[pyfunction]
#[pyo3(name = "_dp_edge_by_labels")]
fn edge_by_labels_py(py: Python<'_>, net: &Bound<'_, Network>, edge: &Bound<'_, Edge>) -> PyResult<Option<Py<Edge>>> {
    let (src, dest) = {
        let e = edge.borrow();
        let src = e.src.borrow(py).name.clone();
        let dest = e.dest.borrow(py).name.clone();
        (src, dest)
    };
    let this = net.borrow();
    Ok(edge_by_labels(py, &this, &src, &dest)?.map(|e| this.edge_obj(e).clone_ref(py)))
}

#[pyfunction]
#[pyo3(name = "_dp_near_accepted")]
fn near_accepted_py(py: Python<'_>, net: &Bound<'_, Network>, anchor: &Bound<'_, PyAny>, child: &Bound<'_, PyAny>, accepted: &Bound<'_, PyAny>) -> PyResult<bool> {
    let this = net.borrow();
    let sets = clades_from_any(accepted)?;
    near_accepted(py, &this, node_arg(py, &this, anchor)?, &labels_from_any(child)?, &sets)
}

// ── Prune reattachment ─────────────────────────────────────────────────

type Rank = (i64, i64, usize);

/// ``_attachment_rank``: higher means closer to the anchor, attached later.
fn attachment_rank(py: Python<'_>, scaffold: Option<&Network>, prune: &Prune<'_>) -> PyResult<Rank> {
    let fallback = (0, 0, prune.index);
    let Some(scaffold) = scaffold else { return Ok(fallback) };
    if prune.anchor_leaves.is_empty() || prune.parent_leaves.is_empty() {
        return Ok(fallback);
    }
    let parent = clade_node(py, scaffold, &prune.parent_leaves)?;
    let anchor = mrca_or_none(py, scaffold, &prune.anchor_leaves)?;
    let (Some(parent), Some(anchor)) = (parent, anchor) else { return Ok(fallback) };
    let targets: Vec<u64> = if parent == anchor { vec![parent] } else { vec![parent, anchor] };
    let common = match scaffold.mrca_uids(py, &targets) {
        Ok(c) => c,
        Err(err) => {
            if is_network_error(py, &err) {
                return Ok(fallback);
            }
            return Err(err);
        }
    };
    Ok((dist_from_root(py, scaffold, common)?, dist_from_root(py, scaffold, parent)?, prune.index))
}

/// ``attach`` of ``reattach_prunes``: graft ``subtree`` as the sibling of
/// ``anchor`` moved below any reticulation chain.
fn attach_prune<'py>(host: &Bound<'py, Network>, subtree: &Bound<'py, Network>, anchor: u64) -> PyResult<()> {
    let anchor = below_retics(&host.borrow(), anchor);
    graft_subtree(host, subtree, anchor, true)?;
    Ok(())
}

/// ``_place_stuck_prune``: DIMPLE's host search for a pass that placed
/// nothing.  Returns the blob and the remaining prunes with their original
/// indices into ``waiting``; a host whose clade grew carries the new copy.
fn place_stuck_prune<'py>(
    py: Python<'py>,
    net: Bound<'py, Network>,
    waiting: Vec<Prune<'py>>,
    attach: &mut dyn FnMut(usize, Bound<'py, Network>, u64) -> PyResult<Bound<'py, Network>>,
) -> PyResult<(Bound<'py, Network>, Vec<(usize, Prune<'py>)>)> {
    let blob_labels = leaf_labels(py, &net.borrow());
    let host_labels: Vec<LabelSet> = waiting.iter().map(|p| leaf_labels(py, &p.subtree.borrow())).collect();
    let mut best_count = 0usize;
    let mut best: Option<(usize, Option<usize>)> = None;
    for (i, prune) in waiting.iter().enumerate() {
        let mut hosts: Vec<(Option<usize>, &LabelSet)> = vec![(None, &blob_labels)];
        for (j, labels) in host_labels.iter().enumerate() {
            if j != i {
                hosts.push((Some(j), labels));
            }
        }
        for (host, labels) in hosts {
            let count = intersect(&prune.anchor_leaves, labels).len();
            if count > best_count {
                best_count = count;
                best = Some((i, host));
            }
        }
    }
    let Some((chosen, host)) = best else {
        return Err(dimple_error(
            py,
            format!(
                "{} pruned subtree(s) could not be attached: no host shares a leaf with their anchors.",
                waiting.len()
            ),
        ));
    };
    match host {
        None => {
            let available = intersect(&waiting[chosen].anchor_leaves, &blob_labels);
            let anchor = mrca_or_none(py, &net.borrow(), &available)?
                .ok_or_else(|| dimple_error(py, "A stuck pruned subtree has no MRCA on the blob."))?;
            let net = attach(chosen, net, anchor)?;
            let rest = waiting.into_iter().enumerate().filter(|(i, _)| *i != chosen).collect();
            Ok((net, rest))
        }
        Some(host) => {
            let host_tree = copy_net(&waiting[host].subtree)?;
            let available = intersect(&waiting[chosen].anchor_leaves, &leaf_labels(py, &host_tree.borrow()));
            let anchor = mrca_or_none(py, &host_tree.borrow(), &available)?
                .ok_or_else(|| dimple_error(py, "A stuck pruned subtree has no MRCA on its host clade."))?;
            let host_tree = attach(chosen, host_tree, anchor)?;
            let mut rest = Vec::new();
            for (i, mut prune) in waiting.into_iter().enumerate() {
                if i == chosen {
                    continue;
                }
                if i == host {
                    prune.subtree = host_tree.clone();
                }
                rest.push((i, prune));
            }
            Ok((net, rest))
        }
    }
}

/// ``reattach_prunes``.
fn reattach_prunes_impl<'py>(
    py: Python<'py>,
    blob: &Bound<'py, Network>,
    prunes: Vec<Prune<'py>>,
    scaffold: Option<&Bound<'py, Network>>,
) -> PyResult<Bound<'py, Network>> {
    let mut net = copy_net(blob)?;
    if prunes.is_empty() {
        return Ok(net);
    }
    let mut owned = LabelSet::new();
    for prune in &prunes {
        owned.extend(prune.leaves.iter().cloned());
    }
    let overlap = intersect(&leaf_labels(py, &net.borrow()), &owned);
    for label in &overlap {
        let uid = node_by_name(&net.borrow(), label);
        if let Some(uid) = uid {
            remove_node(&net, uid);
        }
    }
    Network::clean_impl(&net, [true, true, true])?;
    let initial = leaf_labels(py, &net.borrow());
    let mut ranked: Vec<(Rank, Prune<'py>)> = Vec::with_capacity(prunes.len());
    for prune in prunes {
        let rank = match scaffold {
            Some(s) => attachment_rank(py, Some(&s.borrow()), &prune)?,
            None => attachment_rank(py, None, &prune)?,
        };
        ranked.push((rank, prune));
    }
    ranked.sort_by(|a, b| a.0.cmp(&b.0));
    let mut remaining: Vec<Prune<'py>> = ranked.into_iter().map(|(_, p)| p).collect();
    while !remaining.is_empty() {
        let mut progress = false;
        let mut waiting: Vec<Prune<'py>> = Vec::new();
        for prune in remaining {
            if prune.anchor_leaves.is_empty() {
                return Err(dimple_error(
                    py,
                    "A pruned subtree has an empty anchor. The division must record the cut's siblings.",
                ));
            }
            if prune.source_type == "blob_item" {
                if !overlaps(&prune.anchor_leaves, &initial) {
                    return Err(dimple_error(
                        py,
                        "A pruned subtree's anchor shares no leaf with the blob, so it cannot be attached.",
                    ));
                }
                let target = intersect(&prune.anchor_leaves, &initial);
                let anchor = mrca_or_none(py, &net.borrow(), &target)?;
                let Some(anchor) = anchor else {
                    waiting.push(prune);
                    continue;
                };
                attach_prune(&net, &prune.subtree, anchor)?;
                progress = true;
                continue;
            }
            if !prune.anchor_leaves.is_subset(&leaf_labels(py, &net.borrow())) {
                waiting.push(prune);
                continue;
            }
            let anchor = mrca_or_none(py, &net.borrow(), &prune.anchor_leaves)?;
            let Some(anchor) = anchor else {
                waiting.push(prune);
                continue;
            };
            attach_prune(&net, &prune.subtree, anchor)?;
            progress = true;
        }
        if !progress {
            let subtrees: Vec<Bound<'py, Network>> = waiting.iter().map(|p| p.subtree.clone()).collect();
            let mut attach = |index: usize, host: Bound<'py, Network>, anchor: u64| -> PyResult<Bound<'py, Network>> {
                attach_prune(&host, &subtrees[index], anchor)?;
                Ok(host)
            };
            let (next_net, rest) = place_stuck_prune(py, net, waiting, &mut attach)?;
            net = next_net;
            waiting = rest.into_iter().map(|(_, p)| p).collect();
        }
        remaining = waiting;
    }
    Ok(net)
}

fn prunes_from_any<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<Prune<'py>>> {
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(Prune::from_py(&item?)?);
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_dp_attachment_rank")]
fn attachment_rank_py<'py>(py: Python<'py>, scaffold: Option<&Bound<'py, Network>>, prune: &Bound<'py, PyAny>) -> PyResult<Rank> {
    let prune = Prune::from_py(prune)?;
    match scaffold {
        Some(s) => attachment_rank(py, Some(&s.borrow()), &prune),
        None => attachment_rank(py, None, &prune),
    }
}

#[pyfunction]
#[pyo3(name = "_dp_place_stuck_prune")]
fn place_stuck_prune_py<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
    waiting: &Bound<'py, PyAny>,
    attach: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, Network>, Bound<'py, PyList>)> {
    let originals: Vec<Bound<'py, PyAny>> = waiting.try_iter()?.collect::<PyResult<_>>()?;
    let prunes: Vec<Prune<'py>> = originals.iter().map(Prune::from_py).collect::<PyResult<_>>()?;
    let mut call = |index: usize, host: Bound<'py, Network>, anchor: u64| -> PyResult<Bound<'py, Network>> {
        let node = host.borrow().node_obj(anchor).clone_ref(py);
        net_of(&attach.call1((&originals[index], &host, node))?)
    };
    let (result, rest) = place_stuck_prune(py, net.clone(), prunes, &mut call)?;
    let replace = py.import("dataclasses")?.getattr("replace")?;
    let out = PyList::empty(py);
    for (index, prune) in rest {
        let original = &originals[index];
        let same = original.getattr("subtree")?.is(&prune.subtree);
        if same {
            out.append(original)?;
        } else {
            let kwargs = PyDict::new(py);
            kwargs.set_item("subtree", &prune.subtree)?;
            out.append(replace.call((original,), Some(&kwargs))?)?;
        }
    }
    Ok((result, out))
}

#[pyfunction]
#[pyo3(name = "_dp_reattach_prunes", signature = (blob, prunes, scaffold=None))]
fn reattach_prunes<'py>(
    py: Python<'py>,
    blob: &Bound<'py, Network>,
    prunes: &Bound<'py, PyAny>,
    scaffold: Option<&Bound<'py, Network>>,
) -> PyResult<Bound<'py, Network>> {
    reattach_prunes_impl(py, blob, prunes_from_any(prunes)?, scaffold)
}

// ── Scaffold graft ─────────────────────────────────────────────────────

/// A combined blob ready to replace its scaffold clade (``GraftPiece``).
struct Piece<'py> {
    tob_leaves: LabelSet,
    network: Bound<'py, Network>,
    anchor_leaves: LabelSet,
}

impl<'py> Piece<'py> {
    fn from_py(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Piece {
            tob_leaves: labels_from_any(&obj.getattr("tob_leaves")?)?,
            network: net_of(&obj.getattr("network")?)?,
            anchor_leaves: labels_from_any(&obj.getattr("anchor_leaves")?)?,
        })
    }
}

fn pieces_from_any<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<Piece<'py>>> {
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(Piece::from_py(&item?)?);
    }
    Ok(out)
}

/// ``_remove_clade``: delete the clade whose leaf set is ``leaves``.
fn remove_clade(py: Python<'_>, net: &Bound<'_, Network>, leaves: &LabelSet) -> PyResult<()> {
    let (parent_edge, doomed) = {
        let this = net.borrow();
        let Some(node) = node_with_leaves(py, &this, leaves)? else { return Ok(()) };
        let parents = this.parents(node);
        let edge = match parents.first() {
            Some(&p) => Some(edge_uid_between(py, &this, p, node)?),
            None => None,
        };
        (edge, this.subtree_uids(node))
    };
    if let Some(euid) = parent_edge {
        remove_edge(net, euid);
    }
    for uid in doomed {
        remove_node(net, uid);
    }
    Network::clean_impl(net, [false, false, true])
}

/// ``_rehang_inner``: put detached inner blobs back inside a freshly
/// grafted outer blob, most-available anchor first.
fn rehang_inner<'py>(py: Python<'py>, result: &Bound<'py, Network>, graft_root: u64, saved: &[&Piece<'py>]) -> PyResult<()> {
    let mut remaining: Vec<&Piece<'py>> = saved.to_vec();
    while !remaining.is_empty() {
        let present = leaf_labels(py, &result.borrow());
        remaining.sort_by_key(|inner| std::cmp::Reverse(intersect(&inner.anchor_leaves, &present).len()));
        let inner = remaining.remove(0);
        let available = intersect(&inner.anchor_leaves, &present);
        let anchor = if available.is_empty() { None } else { mrca_or_none(py, &result.borrow(), &available)? };
        if let Some(anchor) = anchor {
            let below = below_retics(&result.borrow(), anchor);
            graft_subtree(result, &inner.network, below, true)?;
            continue;
        }
        let (root_present, children) = {
            let this = result.borrow();
            if this.nodes.contains_key(&graft_root) {
                (true, this.children(graft_root))
            } else {
                (false, Vec::new())
            }
        };
        if let Some(&first) = children.first() {
            let below = below_retics(&result.borrow(), first);
            graft_subtree(result, &inner.network, below, true)?;
        } else if root_present {
            graft_subtree(result, &inner.network, graft_root, false)?;
        } else {
            return Err(dimple_error(
                py,
                "An inner blob has no anchor leaf on the outer blob and the graft root is gone, so it cannot be reattached.",
            ));
        }
    }
    Ok(())
}

/// ``graft_blobs``.
fn graft_blobs_impl<'py>(py: Python<'py>, scaffold: &Bound<'py, Network>, pieces: &[Piece<'py>]) -> PyResult<Bound<'py, Network>> {
    let mut result = copy_net(scaffold)?;
    let mut depths: Vec<(i64, usize)> = Vec::with_capacity(pieces.len());
    for (i, piece) in pieces.iter().enumerate() {
        let depth = {
            let s = scaffold.borrow();
            match clade_node(py, &s, &piece.tob_leaves)? {
                Some(node) => dist_from_root(py, &s, node)?,
                None => 0,
            }
        };
        depths.push((depth, i));
    }
    depths.sort_by(|a, b| b.0.cmp(&a.0));
    let mut placed: Vec<usize> = Vec::new();
    for (_, index) in depths {
        let piece = &pieces[index];
        let mut node = clade_node(py, &result.borrow(), &piece.tob_leaves)?.ok_or_else(|| {
            let sample: Vec<&String> = piece.tob_leaves.iter().take(5).collect();
            let text = str_list(py, sample).and_then(|l| Ok(l.repr()?.to_string())).unwrap_or_default();
            dimple_error(py, format!("No scaffold clade has the leaves of blob {text}."))
        })?;
        let mut saved: Vec<usize> = Vec::new();
        for &earlier in &placed {
            let tob = &pieces[earlier].tob_leaves;
            if !(tob.is_subset(&piece.tob_leaves) && tob != &piece.tob_leaves) {
                continue;
            }
            let this = result.borrow();
            let Some(inner) = node_with_leaves(py, &this, tob)? else { continue };
            if this.subtree_uids(node).contains(&inner) {
                saved.push(earlier);
            }
        }
        for &earlier in &saved {
            remove_clade(py, &result, &pieces[earlier].tob_leaves)?;
        }
        let (present, remaining) = {
            let this = result.borrow();
            if this.nodes.contains_key(&node) {
                (true, labels_under(py, &this, node))
            } else {
                (false, LabelSet::new())
            }
        };
        if !present || remaining.is_empty() {
            let target = if saved.is_empty() {
                piece.tob_leaves.clone()
            } else {
                let mut union = LabelSet::new();
                for &earlier in &saved {
                    union.extend(pieces[earlier].tob_leaves.iter().cloned());
                }
                minus(&piece.tob_leaves, &union)
            };
            node = node_with_leaves(py, &result.borrow(), &target)?
                .ok_or_else(|| dimple_error(py, "The blob clade disappeared while detaching an inner graft."))?;
        }
        let (parent, doomed, parent_edge) = {
            let this = result.borrow();
            let parents = this.parents(node);
            let parent = parents.first().copied();
            let edge = match parent {
                Some(p) => Some(edge_uid_between(py, &this, p, node)?),
                None => None,
            };
            (parent, this.subtree_uids(node), edge)
        };
        if let Some(euid) = parent_edge {
            remove_edge(&result, euid);
        }
        for uid in doomed {
            remove_node(&result, uid);
        }
        let graft_root = if parent.is_none() && result.borrow().nodes.is_empty() {
            result = copy_net(&piece.network)?;
            let this = result.borrow();
            this.nodes
                .keys()
                .copied()
                .find(|&u| this.in_deg(u) == 0)
                .ok_or_else(|| PyStopIteration::new_err(()))?
        } else {
            let Some(parent) = parent else {
                return Err(dimple_error(
                    py,
                    "Replacing the root left other nodes behind. The blob does not cover the scaffold.",
                ));
            };
            let collision = intersect(&leaf_labels(py, &piece.network.borrow()), &leaf_labels(py, &result.borrow()));
            if !collision.is_empty() {
                return Err(dimple_error(
                    py,
                    format!(
                        "Grafting would duplicate taxa already on the scaffold: {}.",
                        join_first(collision.iter(), 5)
                    ),
                ));
            }
            graft_subtree(&result, &piece.network, parent, false)?.borrow(py).uid
        };
        let saved_pieces: Vec<&Piece<'py>> = saved.iter().map(|&i| &pieces[i]).collect();
        rehang_inner(py, &result, graft_root, &saved_pieces)?;
        placed.push(index);
    }
    Network::clean_impl(&result, [false, false, true])?;
    Ok(result)
}

#[pyfunction]
#[pyo3(name = "_dp_remove_clade")]
fn remove_clade_py(py: Python<'_>, net: &Bound<'_, Network>, leaves: &Bound<'_, PyAny>) -> PyResult<()> {
    remove_clade(py, net, &labels_from_any(leaves)?)
}

#[pyfunction]
#[pyo3(name = "_dp_rehang_inner")]
fn rehang_inner_py<'py>(py: Python<'py>, result: &Bound<'py, Network>, graft_root: &Bound<'py, PyAny>, saved: &Bound<'py, PyAny>) -> PyResult<()> {
    let root = {
        let this = result.borrow();
        // A graft root that already left the graph only matters when no
        // anchor is found; an unknown uid stands for "gone".
        this.uid_of_any(graft_root).unwrap_or(u64::MAX)
    };
    let pieces = pieces_from_any(saved)?;
    let refs: Vec<&Piece<'py>> = pieces.iter().collect();
    rehang_inner(py, result, root, &refs)
}

#[pyfunction]
#[pyo3(name = "_dp_graft_blobs")]
fn graft_blobs<'py>(py: Python<'py>, scaffold: &Bound<'py, Network>, pieces: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Network>> {
    graft_blobs_impl(py, scaffold, &pieces_from_any(pieces)?)
}

// ── Per-blob merge ─────────────────────────────────────────────────────

/// ``gene_tree_distance_matrix`` without the numpy wrapping.
fn distance_matrix_impl(py: Python<'_>, gene_trees: &Bound<'_, PyAny>, wanted: &LabelSet) -> PyResult<(Vec<String>, Vec<f64>)> {
    let labels: Vec<String> = wanted.iter().cloned().collect();
    let index: FxHashMap<&str, usize> = labels.iter().enumerate().map(|(i, l)| (l.as_str(), i)).collect();
    let size = labels.len();
    let mut total = vec![0.0_f64; size * size];
    let mut counts = vec![0.0_f64; size * size];
    for tree in gene_tree_list(gene_trees)? {
        let this = tree.borrow();
        let present = intersect(&leaf_labels(py, &this), wanted);
        if present.len() < 2 {
            continue;
        }
        for ((left, right), distance) in leaf_path_lengths(py, &this, wanted)? {
            let (i, j) = (index[left.as_str()], index[right.as_str()]);
            total[i * size + j] += distance;
            total[j * size + i] += distance;
            counts[i * size + j] += 1.0;
            counts[j * size + i] += 1.0;
        }
    }
    let mut matrix = vec![0.0_f64; size * size];
    for i in 0..size {
        for j in 0..size {
            let k = i * size + j;
            matrix[k] = total[k] / counts[k].max(1.0);
            if i != j && counts[k] == 0.0 {
                matrix[k] = f64::NAN;
            }
        }
        matrix[i * size + i] = 0.0;
    }
    Ok((labels, matrix))
}

/// ``_outgroup_distances`` on a flat matrix; ``sibling_order`` is the
/// caller's iteration order of the sibling set.
fn outgroup_distances_impl(
    labels: &[String],
    matrix_labels: &[String],
    matrix: &[f64],
    sibling_order: &[String],
    scale: Option<f64>,
) -> IndexMap<String, f64, FxBuildHasher> {
    let size = matrix_labels.len();
    let index: FxHashMap<&str, usize> = matrix_labels.iter().enumerate().map(|(i, l)| (l.as_str(), i)).collect();
    let sibs: Vec<usize> = sibling_order.iter().filter_map(|l| index.get(l.as_str()).copied()).collect();
    let mut out: IndexMap<String, f64, FxBuildHasher> = IndexMap::default();
    for label in labels {
        let Some(&row) = index.get(label.as_str()) else { continue };
        if sibs.is_empty() {
            continue;
        }
        let values: Vec<f64> = sibs.iter().map(|&s| matrix[row * size + s]).collect();
        if !values.is_empty() && values.iter().all(|v| v.is_finite()) {
            let factor = scale.unwrap_or(OUTGROUP_SCALE);
            out.insert(label.clone(), factor * (numpy_sum(&values) / values.len() as f64));
        }
    }
    out
}

/// ``references`` for the candidate collection, from the displayed-tree choices.
fn kept_maps<'py>(py: Python<'py>, networks: &[Bound<'py, Network>], choices: &[Option<Bound<'py, PyDict>>]) -> PyResult<Vec<Option<IndexMap<u64, u64, FxBuildHasher>>>> {
    let mut out = Vec::with_capacity(choices.len());
    for (net, choice) in networks.iter().zip(choices) {
        out.push(match choice {
            Some(d) => Some(kept_map_from_any(py, &net.borrow(), d.as_any())?),
            None => None,
        });
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_dp_merge_blob_networks", signature = (networks, gene_trees, external_siblings, required_taxa=None, model_selection=None, trace=None, tree_sources=None, backbone=None))]
fn merge_blob_networks<'py>(
    py: Python<'py>,
    networks: &Bound<'py, PyAny>,
    gene_trees: &Bound<'py, PyAny>,
    external_siblings: &Bound<'py, PyAny>,
    required_taxa: Option<&Bound<'py, PyAny>>,
    model_selection: Option<String>,
    trace: Option<&Bound<'py, PyDict>>,
    tree_sources: Option<&Bound<'py, PyAny>>,
    backbone: Option<&Bound<'py, Network>>,
) -> PyResult<(Bound<'py, Network>, usize, Bound<'py, PyList>)> {
    let networks = tree_list(networks)?;
    let sibling_order = label_vec_from_any(external_siblings)?;
    let siblings: LabelSet = sibling_order.iter().cloned().collect();
    let required = match required_taxa {
        Some(r) if !r.is_none() => labels_from_any(r)?,
        _ => LabelSet::new(),
    };
    let mut taxa = LabelSet::new();
    for net in &networks {
        taxa.extend(leaf_labels(py, &net.borrow()));
    }
    if networks.len() == 1 && required.is_subset(&taxa) {
        return Ok((networks[0].clone(), 0, PyList::empty(py)));
    }
    taxa.extend(required.iter().cloned());
    let blob_trees = restrict_gene_trees_impl(py, gene_trees, &taxa)?;
    let counts = observed_triples_impl(py, &blob_trees, &taxa)?;
    let data_size = blob_trees.getattr("trees")?.len()? as i64;
    let min_gain = match model_selection.as_deref() {
        Some("gain") | Some("aic") | Some("bic") => selection_min_gain(model_selection.as_deref(), data_size),
        _ => 0.0,
    };
    let scorer_fn = py.import("phynetpy._dimple")?.getattr("_msc_scorer")?;
    if let Some(backbone) = backbone {
        if leaf_labels(py, &backbone.borrow()) != taxa {
            return Err(dimple_error(py, "The supplied backbone is not on the blob's taxa."));
        }
        let merged = copy_net(backbone)?;
        if let Some(trace) = trace {
            trace.set_item("displayed", PyList::empty(py))?;
            trace.set_item("compatible", PyList::empty(py))?;
            trace.set_item("backbone", copy_net(&merged)?)?;
            trace.set_item("candidates", PyList::new(py, collect_candidates_impl(py, &networks, None)?)?)?;
        }
        let scorer = scorer_fn.call1((&blob_trees, frozenset(py, &taxa)?))?;
        let (with_retics, trials) = place_reticulations_impl(py, &merged, &networks, &scorer, min_gain, None)?;
        return Ok((with_retics, 0, PyList::new(py, trials)?));
    }
    let (displayed, references): (Vec<Bound<'py, Network>>, Option<KeptMaps>) =
        match tree_sources {
            Some(sources) if !sources.is_none() => (tree_list(sources)?, None),
            _ => {
                let (chosen, choices) = select_displayed_trees_impl(py, &networks, Some(&counts))?;
                let maps = kept_maps(py, &networks, &choices)?;
                (chosen, Some(maps))
            }
        };
    let compatible = {
        let leaves: Vec<LabelSet> = displayed.iter().map(|t| leaf_labels(py, &t.borrow())).collect();
        let clades: Vec<Vec<LabelSet>> = displayed.iter().map(|t| clade_sets(py, &t.borrow())).collect::<PyResult<_>>()?;
        let n = displayed.len();
        if n <= 1 {
            displayed.clone()
        } else {
            let mut compatible = vec![vec![false; n]; n];
            for left in 0..n {
                for right in (left + 1)..n {
                    let ok = !trees_incompatible_impl(&leaves[left], &leaves[right], &clades[left], &clades[right]);
                    compatible[left][right] = ok;
                    compatible[right][left] = ok;
                }
            }
            max_clique_indices(n, &|a, b| compatible[a][b]).into_iter().map(|i| displayed[i].clone()).collect()
        }
    };
    if compatible.is_empty() {
        return Err(PyAssertionError::new_err("max_clique_compatible returned no trees"));
    }
    if let Some(trace) = trace {
        trace.set_item("displayed", PyList::new(py, &displayed)?)?;
        trace.set_item("compatible", PyList::new(py, &compatible)?)?;
    }
    let mut needed = taxa.clone();
    needed.extend(siblings.iter().cloned());
    let (labels, matrix) = distance_matrix_impl(py, gene_trees, &needed)?;
    let size = labels.len();
    let blob_labels: Vec<String> = labels.iter().filter(|l| taxa.contains(*l)).cloned().collect();
    let positions: Vec<usize> = blob_labels.iter().map(|l| labels.iter().position(|x| x == l).expect("label present")).collect();
    let mut outgroup: Option<IndexMap<String, f64, FxBuildHasher>> = None;
    if !siblings.is_empty() {
        let distances = outgroup_distances_impl(&blob_labels, &labels, &matrix, &sibling_order, None);
        if blob_labels.iter().any(|l| !distances.contains_key(l)) {
            return Err(dimple_error(
                py,
                "Cannot place the virtual outgroup: a blob taxon has no distance to the scaffold siblings.",
            ));
        }
        outgroup = Some(distances);
    }
    // ``overlap_nj`` on the blob submatrix.
    let mut names = blob_labels.clone();
    let (merged, relaxations) = if names.is_empty() {
        return Err(dimple_error(py, "Neighbor-joining was given no taxa."));
    } else if names.len() == 1 {
        (leaf_network(py, &names[0])?, 0)
    } else {
        let sub: Vec<f64> = positions
            .iter()
            .flat_map(|&i| positions.iter().map(move |&j| (i, j)))
            .map(|(i, j)| matrix[i * size + j])
            .collect();
        if sub.iter().any(|v| !v.is_finite()) {
            return Err(dimple_error(
                py,
                "A taxon pair this blob needs never co-occurs in a gene tree.",
            ));
        }
        let n = names.len();
        let mut ids: FxHashMap<String, usize> = FxHashMap::default();
        for (i, name) in names.iter().enumerate() {
            ids.insert(name.clone(), i);
        }
        let mut dist: FxHashMap<(usize, usize), f64> = FxHashMap::default();
        for i in 0..n {
            for j in 0..n {
                if i < j {
                    dist.insert((i, j), sub[i * n + j]);
                    dist.insert((j, i), sub[i * n + j]);
                }
            }
        }
        if let Some(outgroup) = &outgroup {
            let out_id = n;
            for (i, name) in names.iter().enumerate() {
                dist.insert((i, out_id), outgroup[name]);
                dist.insert((out_id, i), outgroup[name]);
            }
            ids.insert(OUT.to_string(), out_id);
            names.push(OUT.to_string());
        }
        overlap_nj_impl(py, &compatible, names, dist, ids)?
    };
    let merged = if outgroup.is_none() { root_by_triples(py, &merged, &counts)? } else { merged };
    if let Some(trace) = trace {
        trace.set_item("backbone", copy_net(&merged)?)?;
        trace.set_item(
            "candidates",
            PyList::new(py, collect_candidates_impl(py, &networks, references.as_deref())?)?,
        )?;
    }
    let blob_set: LabelSet = blob_labels.iter().cloned().collect();
    let scorer = scorer_fn.call1((&blob_trees, frozenset(py, &blob_set)?))?;
    let (with_retics, trials) =
        place_reticulations_impl(py, &merged, &networks, &scorer, min_gain, references.as_deref())?;
    Ok((with_retics, relaxations, PyList::new(py, trials)?))
}

// ── Registration ───────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Constraint>()?;
    m.add_function(wrap_pyfunction!(labels_py, m)?)?;
    m.add_function(wrap_pyfunction!(label_map_py, m)?)?;
    m.add_function(wrap_pyfunction!(copy_py, m)?)?;
    m.add_function(wrap_pyfunction!(blobs_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_binary_tree_py, m)?)?;
    m.add_function(wrap_pyfunction!(node_with_leaves_py, m)?)?;
    m.add_function(wrap_pyfunction!(below_retics_py, m)?)?;
    m.add_function(wrap_pyfunction!(detach_py, m)?)?;
    m.add_function(wrap_pyfunction!(ensure_internal_labels_py, m)?)?;
    m.add_function(wrap_pyfunction!(mrca_or_none_py, m)?)?;
    m.add_function(wrap_pyfunction!(clade_node_py, m)?)?;
    m.add_function(wrap_pyfunction!(strip_outgroup, m)?)?;
    m.add_function(wrap_pyfunction!(leaf_path_lengths_py, m)?)?;
    m.add_function(wrap_pyfunction!(gene_tree_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(outgroup_distances, m)?)?;
    m.add_function(wrap_pyfunction!(clade_sets_py, m)?)?;
    m.add_function(wrap_pyfunction!(splits_py, m)?)?;
    m.add_function(wrap_pyfunction!(trees_incompatible, m)?)?;
    m.add_function(wrap_pyfunction!(greedy_compatible, m)?)?;
    m.add_function(wrap_pyfunction!(max_clique_indices_py, m)?)?;
    m.add_function(wrap_pyfunction!(max_clique_compatible, m)?)?;
    m.add_function(wrap_pyfunction!(leaf_network_py, m)?)?;
    m.add_function(wrap_pyfunction!(join_clusters_py, m)?)?;
    m.add_function(wrap_pyfunction!(reverse_upward_py, m)?)?;
    m.add_function(wrap_pyfunction!(reroot_on_edge_py, m)?)?;
    m.add_function(wrap_pyfunction!(reroot_above_py, m)?)?;
    m.add_function(wrap_pyfunction!(smaller_side_py, m)?)?;
    m.add_function(wrap_pyfunction!(root_by_triples_py, m)?)?;
    m.add_function(wrap_pyfunction!(overlap_nj, m)?)?;
    m.add_function(wrap_pyfunction!(miss_fraction_py, m)?)?;
    m.add_function(wrap_pyfunction!(displayed_tree_choices_py, m)?)?;
    m.add_function(wrap_pyfunction!(select_displayed_trees_with_choices, m)?)?;
    m.add_function(wrap_pyfunction!(observed_triples, m)?)?;
    m.add_function(wrap_pyfunction!(restrict_gene_trees, m)?)?;
    m.add_function(wrap_pyfunction!(restrict_start, m)?)?;
    m.add_function(wrap_pyfunction!(true_reticulation_bound, m)?)?;
    m.add_function(wrap_pyfunction!(selection_min_gain, m)?)?;
    m.add_function(wrap_pyfunction!(populate_default_branch_lengths_py, m)?)?;
    m.add_function(wrap_pyfunction!(forbidden_py, m)?)?;
    m.add_function(wrap_pyfunction!(mega_groups_py, m)?)?;
    m.add_function(wrap_pyfunction!(top_member_py, m)?)?;
    m.add_function(wrap_pyfunction!(candidate_cuts_py, m)?)?;
    m.add_function(wrap_pyfunction!(oversized_py, m)?)?;
    m.add_function(wrap_pyfunction!(prune_piece_py, m)?)?;
    m.add_function(wrap_pyfunction!(split_clade_py, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_partitions_py, m)?)?;
    m.add_function(wrap_pyfunction!(random_pack_py, m)?)?;
    m.add_function(wrap_pyfunction!(smallest_first_pack_py, m)?)?;
    m.add_function(wrap_pyfunction!(bins_to_division_py, m)?)?;
    m.add_function(wrap_pyfunction!(allocations_py, m)?)?;
    m.add_function(wrap_pyfunction!(items_of_py, m)?)?;
    m.add_function(wrap_pyfunction!(classify_prunes_py, m)?)?;
    m.add_function(wrap_pyfunction!(partition_scaffold, m)?)?;
    m.add_function(wrap_pyfunction!(reticulation_records, m)?)?;
    m.add_function(wrap_pyfunction!(collect_reticulation_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(place_reticulations, m)?)?;
    m.add_function(wrap_pyfunction!(major_edges_py, m)?)?;
    m.add_function(wrap_pyfunction!(view_py, m)?)?;
    m.add_function(wrap_pyfunction!(attachment_leaves_py, m)?)?;
    m.add_function(wrap_pyfunction!(chain_depth_py, m)?)?;
    m.add_function(wrap_pyfunction!(child_clade_py, m)?)?;
    m.add_function(wrap_pyfunction!(side_leaves_py, m)?)?;
    m.add_function(wrap_pyfunction!(choose_inserted_side_py, m)?)?;
    m.add_function(wrap_pyfunction!(destination_edge_py, m)?)?;
    m.add_function(wrap_pyfunction!(candidate_edges_py, m)?)?;
    m.add_function(wrap_pyfunction!(edge_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(sort_records_py, m)?)?;
    m.add_function(wrap_pyfunction!(insert_hybrid_py, m)?)?;
    m.add_function(wrap_pyfunction!(placement_view_py, m)?)?;
    m.add_function(wrap_pyfunction!(edge_by_labels_py, m)?)?;
    m.add_function(wrap_pyfunction!(near_accepted_py, m)?)?;
    m.add_function(wrap_pyfunction!(attachment_rank_py, m)?)?;
    m.add_function(wrap_pyfunction!(place_stuck_prune_py, m)?)?;
    m.add_function(wrap_pyfunction!(reattach_prunes, m)?)?;
    m.add_function(wrap_pyfunction!(remove_clade_py, m)?)?;
    m.add_function(wrap_pyfunction!(rehang_inner_py, m)?)?;
    m.add_function(wrap_pyfunction!(graft_blobs, m)?)?;
    m.add_function(wrap_pyfunction!(merge_blob_networks, m)?)?;
    Ok(())
}
