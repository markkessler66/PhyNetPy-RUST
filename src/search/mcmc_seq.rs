//! MCMC_SEQ: priors, starting trees and the reversible-jump operators of
//! the sequence-data sampler (``phynetpy._mcmc_seq``).
//!
//! The Python module keeps the dataclasses, the likelihood-cache shell,
//! ``SeqState``, the proposal kernel's dispatch and the drivers; everything
//! numerical and every network/gene-tree manipulation lives here.  Each
//! operator mirrors the reference step for step -- the same copy-and-swap
//! strategy (the move is built on a copy of the affected network, then
//! swapped into the state), the same order of draws from the caller's
//! ``numpy.random.Generator`` (including ``Generator.choice`` with the
//! normalised weight vector), the same candidate orderings -- so a seeded
//! chain proposes the same sequence of states.  Operators return
//! ``(log_hastings_ratio, undo)`` where ``undo`` is a [`SeqUndo`] object:
//! calling it swaps the captured pre-move objects (network, heights, gene
//! trees) back into the state and restores the likelihood caches, exactly
//! like the reference's closure.
//!
//! Scoring goes through the compiled kernels behind
//! ``phynetpy._msnc_density`` and ``phynetpy._seq_likelihood``; they are
//! called through their Python facades so the cache keyed by the state's
//! engine and the ``branch_thetas`` resolution stay identical.

use std::collections::BTreeSet;

use indexmap::IndexMap;
use numpy::PyArray1;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PyTuple};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::graph::utils::{in_edges_in_set_order, level as network_level, reticulation_cycle_size_impl};
use crate::graph::{Edge, Network, Node};
use crate::util::network_error;

/// Symmetric-under-reciprocal grid of gene-tree height-scale factors
/// (exact powers of two; see the reference for why that matters).
const HEIGHT_SCALE_GRID: [f64; 5] = [0.25, 0.5, 1.0, 2.0, 4.0];

type Heights = FxHashMap<u64, f64>;
/// ``_gt_signature``: ``{(sorted leaf labels, round(height, 12))}``.
type Sig = BTreeSet<(Vec<String>, u64)>;
type LabelSet = BTreeSet<String>;

// ═══════════════════════════════════════════════════════════════════════
// Cached Python callables
// ═══════════════════════════════════════════════════════════════════════

static MATH_LGAMMA: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static MSNC_MODULE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static NUMPY_MODULE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static SUBST_UNIT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

fn math_lgamma(py: Python<'_>, x: f64) -> PyResult<f64> {
    let f = MATH_LGAMMA.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py.import("math")?.getattr("lgamma")?.unbind())
    })?;
    f.bind(py).call1((x,))?.extract()
}

fn msnc_module(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    let m = MSNC_MODULE.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py.import("phynetpy._msnc_density")?.into_any().unbind())
    })?;
    Ok(m.bind(py))
}

fn numpy_module(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    let m = NUMPY_MODULE.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py.import("numpy")?.into_any().unbind())
    })?;
    Ok(m.bind(py))
}

fn substitution_unit(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    let u = SUBST_UNIT.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py
            .import("phynetpy._units")?
            .getattr("BranchLengthUnit")?
            .getattr("SUBSTITUTIONS_PER_SITE")?
            .unbind())
    })?;
    Ok(u.bind(py))
}

/// The Python-level ``Network`` class.
fn network_class(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    match crate::util::network_class(py) {
        Some(c) => Ok(c.clone().into_any()),
        None => py.import("phynetpy.Network")?.getattr("Network"),
    }
}

/// ``math.log`` with Python's domain error.
fn pylog(x: f64) -> PyResult<f64> {
    if x.is_nan() {
        return Ok(x);
    }
    if x <= 0.0 {
        return Err(PyValueError::new_err("math domain error"));
    }
    Ok(x.ln())
}

// ═══════════════════════════════════════════════════════════════════════
// Prior densities
// ═══════════════════════════════════════════════════════════════════════

fn log_poisson_pmf(py: Python<'_>, k: i64, mean: f64) -> PyResult<f64> {
    if mean <= 0.0 {
        return Ok(if k == 0 { 0.0 } else { f64::NEG_INFINITY });
    }
    Ok(k as f64 * mean.ln() - mean - math_lgamma(py, (k + 1) as f64)?)
}

fn log_gamma_pdf(py: Python<'_>, x: f64, shape: f64, mean: f64) -> PyResult<f64> {
    if x <= 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    let scale = mean / shape;
    Ok((shape - 1.0) * pylog(x)? - x / scale - shape * pylog(scale)? - math_lgamma(py, shape)?)
}

fn log_beta_pdf(py: Python<'_>, x: f64, alpha: f64, beta: f64) -> PyResult<f64> {
    if x <= 0.0 || x >= 1.0 {
        if alpha == 1.0 && beta == 1.0 {
            return Ok(0.0);
        }
        return Ok(f64::NEG_INFINITY);
    }
    Ok((alpha - 1.0) * pylog(x)? + (beta - 1.0) * (-x).ln_1p()
        - (math_lgamma(py, alpha)? + math_lgamma(py, beta)? - math_lgamma(py, alpha + beta)?))
}

/// The hyperparameters of ``MCMCSeqPriors``, read once per call.
struct Priors {
    poisson_mean: f64,
    max_reticulations: i64,
    max_level: Option<i64>,
    theta_shape: f64,
    theta_prior_mean: f64,
    diameter_mean: f64,
    use_diameter_prior: bool,
    gamma_alpha: f64,
    gamma_beta: f64,
    min_cycle_size: i64,
    min_branch_length: f64,
    branch_length_mean: f64,
    use_branch_length_prior: bool,
}

impl Priors {
    fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let max_level = {
            let v = obj.getattr("max_level")?;
            if v.is_none() {
                None
            } else {
                Some(v.extract::<i64>()?)
            }
        };
        Ok(Priors {
            poisson_mean: obj.getattr("poisson_mean")?.extract()?,
            max_reticulations: obj.getattr("max_reticulations")?.extract()?,
            max_level,
            theta_shape: obj.getattr("theta_shape")?.extract()?,
            theta_prior_mean: obj.getattr("theta_prior_mean")?.extract()?,
            diameter_mean: obj.getattr("diameter_mean")?.extract()?,
            use_diameter_prior: obj.getattr("use_diameter_prior")?.is_truthy()?,
            gamma_alpha: obj.getattr("gamma_alpha")?.extract()?,
            gamma_beta: obj.getattr("gamma_beta")?.extract()?,
            min_cycle_size: obj.getattr("min_cycle_size")?.extract()?,
            min_branch_length: obj.getattr("min_branch_length")?.extract()?,
            branch_length_mean: obj.getattr("branch_length_mean")?.extract()?,
            use_branch_length_prior: obj.getattr("use_branch_length_prior")?.is_truthy()?,
        })
    }
}

fn log_prior_seq_impl(py: Python<'_>, net: &Bound<'_, Network>, theta: f64, pr: &Priors) -> PyResult<f64> {
    let n = net.borrow();
    let reticulations = retic_uids(py, &n);
    let n_ret = reticulations.len() as i64;
    if n_ret > pr.max_reticulations {
        return Ok(f64::NEG_INFINITY);
    }
    if n_ret > 0 {
        for &r in &reticulations {
            if pr.min_cycle_size > 0 {
                let csize = reticulation_cycle_size_impl(&n, r) as i64;
                if csize < pr.min_cycle_size {
                    return Ok(f64::NEG_INFINITY);
                }
            }
            if pr.min_branch_length > 0.0 {
                for e in n.in_edge_uids(r) {
                    let length = edge_len(py, &n, e).ok_or_else(|| {
                        PyTypeError::new_err("float() argument must be a string or a real number, not 'NoneType'")
                    })?;
                    if length < pr.min_branch_length {
                        return Ok(f64::NEG_INFINITY);
                    }
                }
            }
        }
    }
    let mut lp = log_poisson_pmf(py, n_ret, pr.poisson_mean)?;
    let n_leaves = n.leaf_uids().len() as i64;
    let mut edges = 2 * n_leaves - 2;
    for _ in 0..n_ret {
        if edges > 1 {
            lp -= ((edges * (edges - 1)) as f64).ln();
        }
        edges += 3;
    }
    lp += log_gamma_pdf(py, theta, pr.theta_shape, pr.theta_prior_mean)?;

    let mut heights: Option<Heights> = None;
    if pr.use_branch_length_prior && pr.branch_length_mean > 0.0 {
        let bl_rate = 1.0 / pr.branch_length_mean;
        let log_bl_rate = bl_rate.ln();
        let h = node_heights_uid(py, &n)?;
        for e in n.edges.values() {
            let mut length = hget(py, &n, &h, e.su)? - hget(py, &n, &h, e.du)?;
            if length < 0.0 {
                length = 0.0;
            }
            lp += log_bl_rate - bl_rate * length;
        }
        heights = Some(h);
    }

    if n_ret > 0 && !(pr.gamma_alpha == 1.0 && pr.gamma_beta == 1.0) {
        for &r in &reticulations {
            let in_e = in_edges_in_set_order(py, &n, r);
            if let Some(&first) = in_e.first() {
                if let Some(g) = edge_gamma(py, &n, first) {
                    lp += log_beta_pdf(py, g, pr.gamma_alpha, pr.gamma_beta)?;
                }
            }
        }
    }

    if pr.use_diameter_prior && n_ret > 0 {
        let h = match heights {
            Some(h) => h,
            None => node_heights_uid(py, &n)?,
        };
        let rate = 1.0 / pr.diameter_mean;
        for &r in &reticulations {
            let parents = n.parents(r);
            if parents.len() >= 2 {
                let h0 = h.get(&parents[0]).copied().unwrap_or(0.0);
                let h1 = h.get(&parents[1]).copied().unwrap_or(0.0);
                let diam = (h0 - h1).abs();
                lp += pylog(rate)? - rate * diam;
            }
        }
    }
    Ok(lp)
}

// ═══════════════════════════════════════════════════════════════════════
// Network helpers (uid based; tolerant of stale ids)
// ═══════════════════════════════════════════════════════════════════════

fn label_of(py: Python<'_>, net: &Network, uid: u64) -> String {
    match net.nodes.get(&uid) {
        Some(e) => e.obj.borrow(py).name.clone(),
        None => String::new(),
    }
}

fn is_retic(py: Python<'_>, net: &Network, uid: u64) -> bool {
    net.nodes.get(&uid).map(|e| e.obj.borrow(py).is_retic).unwrap_or(false)
}

fn retic_uids(py: Python<'_>, net: &Network) -> Vec<u64> {
    net.nodes
        .iter()
        .filter(|(_, e)| e.obj.borrow(py).is_retic)
        .map(|(u, _)| *u)
        .collect()
}

fn edge_len(py: Python<'_>, net: &Network, euid: u64) -> Option<f64> {
    net.edges.get(&euid).and_then(|e| e.obj.borrow(py).length)
}

fn edge_gamma(py: Python<'_>, net: &Network, euid: u64) -> Option<f64> {
    net.edges.get(&euid).and_then(|e| e.obj.borrow(py).gamma)
}

fn edge_ends(net: &Network, euid: u64) -> Option<(u64, u64)> {
    net.edges.get(&euid).map(|e| (e.su, e.du))
}

fn node_py(py: Python<'_>, net: &Network, uid: u64) -> Option<Py<Node>> {
    net.nodes.get(&uid).map(|e| e.obj.clone_ref(py))
}

fn set_retic(py: Python<'_>, net: &Bound<'_, Network>, uid: u64, value: bool) {
    let obj = match net.borrow().nodes.get(&uid) {
        Some(e) => e.obj.clone_ref(py),
        None => return,
    };
    Node::set_is_reticulation(obj.bind(py), value);
}

/// ``edge.set_gamma(g)``; ``None`` is the TypeError the Rust setter raises.
fn set_gamma(py: Python<'_>, net: &Bound<'_, Network>, euid: u64, gamma: Option<f64>) -> PyResult<()> {
    let obj = match net.borrow().edges.get(&euid) {
        Some(e) => e.obj.clone_ref(py),
        None => return Ok(()),
    };
    match gamma {
        Some(g) => Edge::set_gamma(obj.bind(py), g),
        None => Err(PyTypeError::new_err("argument 'gamma': must be real number, not NoneType")),
    }
}

fn set_len(py: Python<'_>, net: &Bound<'_, Network>, euid: u64, length: f64) -> PyResult<()> {
    let obj = match net.borrow().edges.get(&euid) {
        Some(e) => e.obj.clone_ref(py),
        None => return Ok(()),
    };
    Edge::set_length(obj.bind(py), Some(length), false, false)
}

/// ``Network.add_uid_node()``: the node is returned even when a node of
/// that name already exists (then it was not added, as in the reference).
fn add_uid_node(py: Python<'_>, net: &Bound<'_, Network>) -> PyResult<u64> {
    let name = format!("UID_{}", net.borrow().uid);
    let node = Py::new(py, Node::make(py, name, false, None, None, None))?;
    let uid = node.borrow(py).uid;
    let mut n = net.borrow_mut();
    n.add_node(net.as_any(), node.bind(py))?;
    n.uid += 1;
    Ok(uid)
}

/// ``net.add_edges(Edge(src, dest, length=..., gamma=...))``.
fn new_edge(
    py: Python<'_>,
    net: &Bound<'_, Network>,
    su: u64,
    du: u64,
    length: Option<f64>,
    gamma: Option<f64>,
) -> PyResult<u64> {
    let gamma = gamma.unwrap_or(0.0);
    if !(0.0..=1.0).contains(&gamma) {
        return Err(PyValueError::new_err(
            "Please provide a probabilistic value for gamma (between 0 and 1, inclusive)!",
        ));
    }
    let (s, d) = {
        let n = net.borrow();
        match (n.nodes.get(&su), n.nodes.get(&du)) {
            (Some(a), Some(b)) => (a.obj.clone_ref(py), b.obj.clone_ref(py)),
            _ => {
                return Err(network_error(
                    py,
                    "Tried to add an edge between two nodes, at least one of which does \
                     not belong to this network.",
                ))
            }
        }
    };
    let edge = Py::new(
        py,
        Edge::make(
            py,
            s,
            d,
            Some(length.unwrap_or(1.0)),
            Some(gamma),
            Some(0.0),
            Some("no tag assigned yet".to_string()),
        ),
    )?;
    let euid = edge.borrow(py).uid;
    net.borrow_mut().add_edge(net.as_any(), edge.bind(py))?;
    Ok(euid)
}

fn remove_edge(py: Python<'_>, net: &Bound<'_, Network>, euid: u64) {
    net.borrow_mut().remove_edge_uid(py, euid);
}

fn remove_node(py: Python<'_>, net: &Bound<'_, Network>, uid: u64) {
    net.borrow_mut().remove_node_uid(py, uid);
}

/// ``GraphUtils._edge_between(net, parent, child)``.
fn edge_between(py: Python<'_>, net: &Network, su: u64, du: u64) -> PyResult<u64> {
    if !net.nodes.contains_key(&su) || !net.nodes.contains_key(&du) {
        return Err(PyValueError::new_err("No matching edges found"));
    }
    let e = net.edge_between(py, su, du, None, None)?;
    let uid = e.borrow(py).uid;
    Ok(uid)
}

/// ``_clone_net``: an independent copy plus the old -> new node id map.
fn clone_net<'py>(net: &Bound<'py, Network>) -> PyResult<(Bound<'py, Network>, FxHashMap<u64, u64>)> {
    let py = net.py();
    let (new, mapping) = Network::copy_impl(net)?;
    let map = mapping
        .iter()
        .map(|(o, n)| (o.borrow(py).uid, n.borrow(py).uid))
        .collect();
    Ok((new, map))
}

fn has_parallel_edges(net: &Network) -> bool {
    net.pair.values().any(|v| v.len() > 1)
}

/// ``_find_edge_by_labels``: ``(edge, src, dest)`` of the first edge whose
/// endpoint labels match.
fn find_edge_by_labels(py: Python<'_>, net: &Network, src: &str, dest: &str) -> Option<(u64, u64, u64)> {
    for (euid, e) in net.edges.iter() {
        if label_of(py, net, e.su) == src && label_of(py, net, e.du) == dest {
            return Some((*euid, e.su, e.du));
        }
    }
    None
}

// ── Heights ────────────────────────────────────────────────────────────

fn hget(py: Python<'_>, net: &Network, h: &Heights, uid: u64) -> PyResult<f64> {
    h.get(&uid)
        .copied()
        .ok_or_else(|| PyKeyError::new_err(label_of(py, net, uid)))
}

fn height_rec(py: Python<'_>, net: &Network, uid: u64, memo: &mut Heights) -> PyResult<f64> {
    if let Some(&h) = memo.get(&uid) {
        return Ok(h);
    }
    let mut best = 0.0_f64;
    for child in net.children(uid) {
        let euid = net.first_edge_between(py, uid, child)?;
        let length = edge_len(py, net, euid).unwrap_or(0.0);
        let h = height_rec(py, net, child, memo)? + length;
        if h > best {
            best = h;
        }
    }
    memo.insert(uid, best);
    Ok(best)
}

/// ``GraphUtils._node_heights`` keyed by uid.
fn node_heights_uid(py: Python<'_>, net: &Network) -> PyResult<Heights> {
    let mut memo = Heights::default();
    let uids: Vec<u64> = net.nodes.keys().copied().collect();
    for uid in uids {
        height_rec(py, net, uid, &mut memo)?;
    }
    Ok(memo)
}

/// A ``dict[Node, float]`` read into a uid map (nodes of ``net`` only).
fn heights_from_py(py: Python<'_>, net: &Network, d: &Bound<'_, PyAny>) -> PyResult<Heights> {
    let d = d.cast::<PyDict>()?;
    let mut out = Heights::default();
    for (uid, entry) in net.nodes.iter() {
        if let Some(v) = d.get_item(entry.obj.bind(py))? {
            out.insert(*uid, v.extract::<f64>()?);
        }
    }
    Ok(out)
}

/// A uid map written as ``dict[Node, float]`` in node order.
fn heights_to_py<'py>(py: Python<'py>, net: &Network, h: &Heights) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (uid, entry) in net.nodes.iter() {
        if let Some(v) = h.get(uid) {
            d.set_item(entry.obj.bind(py), *v)?;
        }
    }
    Ok(d)
}

fn scaled(h: &Heights, s: f64) -> Heights {
    h.iter().map(|(k, v)| (*k, v * s)).collect()
}

/// ``_sync_lengths``: every edge length becomes ``height(src) - height(dest)``.
fn sync_lengths(py: Python<'_>, net: &Bound<'_, Network>, h: &Heights) -> PyResult<()> {
    let edges: Vec<(u64, u64, Py<Edge>)> = {
        let n = net.borrow();
        n.edges.values().map(|e| (e.su, e.du, e.obj.clone_ref(py))).collect()
    };
    for (su, du, obj) in edges {
        let mut length = {
            let n = net.borrow();
            hget(py, &n, h, su)? - hget(py, &n, h, du)?
        };
        if length < 0.0 {
            length = 0.0;
        }
        Edge::set_length(obj.bind(py), Some(length), false, false)?;
    }
    Ok(())
}

/// ``_descendant_species``: species labels at or below every node.
fn descendant_species(py: Python<'_>, net: &Network, species_of: &FxHashMap<String, String>) -> FxHashMap<u64, LabelSet> {
    fn rec(
        py: Python<'_>,
        net: &Network,
        v: u64,
        species_of: &FxHashMap<String, String>,
        memo: &mut FxHashMap<u64, LabelSet>,
        stack: &mut FxHashSet<u64>,
    ) -> LabelSet {
        if let Some(s) = memo.get(&v) {
            return s.clone();
        }
        if !stack.insert(v) {
            // A cycle: the reference would recurse forever; stop instead.
            return LabelSet::new();
        }
        let kids = net.children(v);
        let r = if kids.is_empty() {
            let label = label_of(py, net, v);
            let mut s = LabelSet::new();
            s.insert(species_of.get(&label).cloned().unwrap_or(label));
            s
        } else {
            let mut acc = LabelSet::new();
            for c in kids {
                acc.extend(rec(py, net, c, species_of, memo, stack));
            }
            acc
        };
        stack.remove(&v);
        memo.insert(v, r.clone());
        r
    }
    let mut memo = FxHashMap::default();
    let mut stack = FxHashSet::default();
    let uids: Vec<u64> = net.nodes.keys().copied().collect();
    for v in uids {
        rec(py, net, v, species_of, &mut memo, &mut stack);
    }
    memo
}

fn species_map(species_of: &Bound<'_, PyAny>) -> PyResult<FxHashMap<String, String>> {
    let mut out = FxHashMap::default();
    if species_of.is_none() {
        return Ok(out);
    }
    for (k, v) in species_of.cast::<PyDict>()?.iter() {
        out.insert(k.extract::<String>()?, v.extract::<String>()?);
    }
    Ok(out)
}

fn leaf_desc_labels(py: Python<'_>, net: &Network, uid: u64) -> LabelSet {
    if !net.nodes.contains_key(&uid) {
        return LabelSet::new();
    }
    net.leaf_descendants_uid(uid).into_iter().map(|l| label_of(py, net, l)).collect()
}

fn node_by_leafset(py: Python<'_>, net: &Network, target: &LabelSet) -> Option<u64> {
    let uids: Vec<u64> = net.nodes.keys().copied().collect();
    uids.into_iter().find(|&v| leaf_desc_labels(py, net, v) == *target)
}

/// Python's ``round(x, 12)``: correctly rounded to 12 decimals.
fn round12(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.12}").parse::<f64>().unwrap_or(x)
}

fn sig_key(x: f64) -> u64 {
    let r = round12(x);
    if r == 0.0 {
        0.0f64.to_bits()
    } else {
        r.to_bits()
    }
}

/// ``_gt_signature``: ``(leaf-label set, rounded height)`` of every
/// internal node reachable from the root.
fn gt_signature(py: Python<'_>, net: &Network, h: &Heights) -> PyResult<Sig> {
    let map = net.leaf_descendants_all_map(py)?;
    let mut out = Sig::new();
    for (uid, leaves) in map.iter() {
        if net.out_deg(*uid) == 0 {
            continue;
        }
        let set: LabelSet = leaves.iter().map(|l| label_of(py, net, *l)).collect();
        let labels: Vec<String> = set.into_iter().collect();
        let hh = h.get(uid).copied().unwrap_or(0.0);
        out.insert((labels, sig_key(hh)));
    }
    Ok(out)
}

fn sig_to_py<'py>(py: Python<'py>, sig: &Sig) -> PyResult<Bound<'py, PyFrozenSet>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(sig.len());
    for (labels, bits) in sig {
        let fs = PyFrozenSet::new(py, labels.iter())?;
        let h = f64::from_bits(*bits);
        items.push(PyTuple::new(py, [fs.into_any(), h.into_pyobject(py)?.into_any()])?.into_any());
    }
    PyFrozenSet::new(py, items)
}

fn labelset_to_py<'py>(py: Python<'py>, s: &LabelSet) -> PyResult<Bound<'py, PyFrozenSet>> {
    PyFrozenSet::new(py, s.iter())
}

/// The reference's ``_logsumexp`` (``-inf`` for an empty list).
fn logsumexp_ref(values: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &v in values {
        if v > m {
            m = v;
        }
    }
    if m == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let mut acc = 0.0;
    for &v in values {
        acc += (v - m).exp();
    }
    m + acc.ln()
}

// ═══════════════════════════════════════════════════════════════════════
// RNG bridge and state access
// ═══════════════════════════════════════════════════════════════════════

struct Rng<'py>(Bound<'py, PyAny>);

impl<'py> Rng<'py> {
    fn random(&self) -> PyResult<f64> {
        self.0.call_method0("random")?.extract()
    }
    /// ``int(rng.integers(n))``.
    fn integers(&self, n: usize) -> PyResult<usize> {
        let v: i64 = self.0.call_method1("integers", (n,))?.extract()?;
        if v < 0 || v as usize >= n {
            return Err(PyIndexError::new_err("list index out of range"));
        }
        Ok(v as usize)
    }
    /// ``int(rng.choice(n, p=probs))``.
    fn choice(&self, n: usize, probs: &Bound<'py, PyAny>) -> PyResult<usize> {
        let kwargs = PyDict::new(self.0.py());
        kwargs.set_item("p", probs)?;
        let v: i64 = self.0.call_method("choice", (n,), Some(&kwargs))?.extract()?;
        if v < 0 || v as usize >= n {
            return Err(PyIndexError::new_err("list index out of range"));
        }
        Ok(v as usize)
    }
}

/// ``exp(s - lse)`` for every score, normalised by numpy exactly as the
/// reference does (``probs /= probs.sum()``).
fn selection_probs<'py>(py: Python<'py>, scores: &[f64], lse: f64) -> PyResult<Bound<'py, PyAny>> {
    let vals: Vec<f64> = scores.iter().map(|s| (s - lse).exp()).collect();
    let arr = PyArray1::from_vec(py, vals);
    let total = arr.as_any().call_method0("sum")?;
    arr.as_any().div(total)
}

/// A ``SeqState`` seen from Rust.
struct St<'py> {
    py: Python<'py>,
    obj: Bound<'py, PyAny>,
}

impl<'py> St<'py> {
    fn new(obj: Bound<'py, PyAny>) -> Self {
        St { py: obj.py(), obj }
    }
    fn get(&self, name: &str) -> PyResult<Bound<'py, PyAny>> {
        self.obj.getattr(name)
    }
    fn net(&self) -> PyResult<Bound<'py, Network>> {
        Ok(self.get("species_net")?.cast_into::<Network>()?)
    }
    fn net_heights_obj(&self) -> PyResult<Bound<'py, PyAny>> {
        self.get("net_heights")
    }
    fn gene_trees(&self) -> PyResult<Bound<'py, PyList>> {
        Ok(self.get("gene_trees")?.cast_into::<PyList>()?)
    }
    fn gt_heights(&self) -> PyResult<Bound<'py, PyList>> {
        Ok(self.get("gt_heights")?.cast_into::<PyList>()?)
    }
    fn gene_tree(&self, i: usize) -> PyResult<Bound<'py, Network>> {
        Ok(self.gene_trees()?.get_item(i)?.cast_into::<Network>()?)
    }
    fn theta(&self) -> PyResult<f64> {
        self.get("theta")?.extract()
    }
    fn engine(&self) -> PyResult<Bound<'py, PyAny>> {
        self.get("_engine")
    }
    fn n_loci(&self) -> PyResult<usize> {
        self.engine()?.getattr("n_loci")?.extract()
    }
    fn priors(&self) -> PyResult<Priors> {
        Priors::from_py(&self.get("priors")?)
    }
    fn num_reticulations(&self) -> PyResult<i64> {
        let net = self.net()?;
        let n = net.borrow();
        Ok(retic_uids(self.py, &n).len() as i64)
    }
    fn snapshot(&self) -> PyResult<Py<PyAny>> {
        Ok(self.engine()?.call_method0("clone_caches")?.unbind())
    }
    fn invalidate_network(&self) -> PyResult<()> {
        self.engine()?.call_method0("invalidate_network")?;
        Ok(())
    }
    fn invalidate_locus(&self, i: usize) -> PyResult<()> {
        self.engine()?.call_method1("invalidate_locus", (i,))?;
        Ok(())
    }
    fn invalidate_all_loci(&self) -> PyResult<()> {
        for i in 0..self.n_loci()? {
            self.invalidate_locus(i)?;
        }
        Ok(())
    }
    fn set(&self, name: &str, value: &Bound<'py, PyAny>) -> PyResult<()> {
        self.obj.setattr(name, value)
    }
    /// Install a new species network + heights.
    fn swap_net(&self, net: &Bound<'py, Network>, h: &Heights) -> PyResult<()> {
        let d = heights_to_py(self.py, &net.borrow(), h)?;
        self.set("species_net", net.as_any())?;
        self.set("net_heights", d.as_any())
    }
    /// Install new per-locus gene trees + heights (whole lists).
    fn swap_gene_trees(&self, gts: &[Bound<'py, Network>], hs: &[Heights]) -> PyResult<()> {
        let nets = PyList::new(self.py, gts.iter().map(|g| g.as_any()))?;
        let mut dicts = Vec::with_capacity(hs.len());
        for (g, h) in gts.iter().zip(hs) {
            dicts.push(heights_to_py(self.py, &g.borrow(), h)?);
        }
        let dl = PyList::new(self.py, dicts)?;
        self.set("gene_trees", nets.as_any())?;
        self.set("gt_heights", dl.as_any())
    }
}

/// The undo handle an operator returns: calling it swaps the captured
/// pre-move objects back into the state and restores the likelihood
/// caches, exactly like the reference's ``undo`` closure.
#[pyclass(name = "_SeqUndo", module = "phynetpy._mcmc_seq")]
pub struct SeqUndo {
    state: Py<PyAny>,
    snap: Py<PyAny>,
    theta: Option<Py<PyAny>>,
    species_net: Option<Py<PyAny>>,
    net_heights: Option<Py<PyAny>>,
    gene_trees: Option<Py<PyAny>>,
    gt_heights: Option<Py<PyAny>>,
    locus: Option<(usize, Py<PyAny>, Py<PyAny>)>,
}

impl SeqUndo {
    fn blank(st: &St<'_>, snap: Py<PyAny>) -> Self {
        SeqUndo {
            state: st.obj.clone().unbind(),
            snap,
            theta: None,
            species_net: None,
            net_heights: None,
            gene_trees: None,
            gt_heights: None,
            locus: None,
        }
    }
    fn with_net(st: &St<'_>, snap: Py<PyAny>, net: Bound<'_, Network>, heights: Bound<'_, PyAny>) -> Self {
        let mut u = Self::blank(st, snap);
        u.species_net = Some(net.into_any().unbind());
        u.net_heights = Some(heights.unbind());
        u
    }
    fn with_locus(st: &St<'_>, snap: Py<PyAny>, i: usize, gt: Bound<'_, PyAny>, h: Bound<'_, PyAny>) -> Self {
        let mut u = Self::blank(st, snap);
        u.locus = Some((i, gt.unbind(), h.unbind()));
        u
    }
    fn with_all(
        st: &St<'_>,
        snap: Py<PyAny>,
        net: Bound<'_, Network>,
        heights: Bound<'_, PyAny>,
        gts: Bound<'_, PyAny>,
        hs: Bound<'_, PyAny>,
    ) -> Self {
        let mut u = Self::with_net(st, snap, net, heights);
        u.gene_trees = Some(gts.unbind());
        u.gt_heights = Some(hs.unbind());
        u
    }
}

#[pymethods]
impl SeqUndo {
    fn __call__(&self, py: Python<'_>) -> PyResult<()> {
        let st = self.state.bind(py);
        if let Some(t) = &self.theta {
            st.setattr("theta", t.bind(py))?;
        }
        if let Some(n) = &self.species_net {
            st.setattr("species_net", n.bind(py))?;
        }
        if let Some(h) = &self.net_heights {
            st.setattr("net_heights", h.bind(py))?;
        }
        if let Some(g) = &self.gene_trees {
            st.setattr("gene_trees", g.bind(py))?;
        }
        if let Some(h) = &self.gt_heights {
            st.setattr("gt_heights", h.bind(py))?;
        }
        if let Some((i, gt, h)) = &self.locus {
            st.getattr("gene_trees")?.set_item(*i, gt.bind(py))?;
            st.getattr("gt_heights")?.set_item(*i, h.bind(py))?;
        }
        st.getattr("_engine")?.call_method1("restore_caches", (self.snap.bind(py),))?;
        Ok(())
    }

    fn __repr__(&self) -> &'static str {
        "<MCMC_SEQ undo handle>"
    }
}

type OpResult = PyResult<Option<(f64, Py<SeqUndo>)>>;

fn op_ok(py: Python<'_>, loghr: f64, undo: SeqUndo) -> OpResult {
    Ok(Some((loghr, Py::new(py, undo)?)))
}

// ═══════════════════════════════════════════════════════════════════════
// Embedding consistency and UPGMA starting trees
// ═══════════════════════════════════════════════════════════════════════

/// ``_enforce_embedding_consistency``: scales ``net_heights`` (the Python
/// dict, in place) and returns the factor.
fn enforce_embedding_consistency(
    py: Python<'_>,
    species_net: &Bound<'_, Network>,
    net_heights: &Bound<'_, PyAny>,
    gene_trees: &Bound<'_, PyAny>,
    gt_heights: &Bound<'_, PyAny>,
    species_of: &Bound<'_, PyAny>,
    eps: f64,
) -> PyResult<f64> {
    let sp = species_net.borrow();
    let empty: FxHashMap<String, String> = FxHashMap::default();
    let sp_desc = descendant_species(py, &sp, &empty);
    let sp_h = heights_from_py(py, &sp, net_heights)?;
    let nodes: Vec<u64> = sp.nodes.keys().copied().collect();
    let mut keyed: Vec<(f64, u64)> = Vec::with_capacity(nodes.len());
    for &v in &nodes {
        keyed.push((hget(py, &sp, &sp_h, v)?, v));
    }
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let nodes_by_h: Vec<u64> = keyed.into_iter().map(|(_, v)| v).collect();
    let mut bounds: IndexMap<u64, f64, FxBuildHasher> = IndexMap::default();
    for &v in &nodes {
        bounds.insert(v, f64::INFINITY);
    }
    let species = species_map(species_of)?;
    let gts: Vec<Bound<'_, PyAny>> = gene_trees.try_iter()?.collect::<PyResult<_>>()?;
    for (gi, gt_obj) in gts.iter().enumerate() {
        let gt_b = gt_obj.cast::<Network>()?;
        let gt = gt_b.borrow();
        let gh = heights_from_py(py, &gt, &gt_heights.get_item(gi)?)?;
        let gdesc = descendant_species(py, &gt, &species);
        let gnodes: Vec<u64> = gt.nodes.keys().copied().collect();
        for u in gnodes {
            if gt.out_deg(u) == 0 {
                continue;
            }
            let su = match gdesc.get(&u) {
                Some(s) => s,
                None => continue,
            };
            if su.len() < 2 {
                continue;
            }
            let hu = hget(py, &gt, &gh, u)?;
            for &v in &nodes_by_h {
                let covers = sp_desc.get(&v).map(|d| su.is_subset(d)).unwrap_or(false);
                if covers {
                    if let Some(b) = bounds.get_mut(&v) {
                        if hu < *b {
                            *b = hu;
                        }
                    }
                    break;
                }
            }
        }
    }
    let mut scale = 1.0;
    for (v, upper) in bounds.iter() {
        let h = hget(py, &sp, &sp_h, *v)?;
        if upper.is_finite() && h > 0.0 {
            let ratio = (1.0 - eps) * upper / h;
            if ratio < scale {
                scale = ratio;
            }
        }
    }
    if scale < 1.0 {
        let d = net_heights.cast::<PyDict>()?;
        let items: Vec<(Bound<'_, PyAny>, f64)> = d
            .iter()
            .map(|(k, v)| v.extract::<f64>().map(|x| (k, x)))
            .collect::<PyResult<_>>()?;
        for (k, x) in items {
            d.set_item(k, x * scale)?;
        }
    }
    Ok(scale)
}

fn jc_distance(s1: &str, s2: &str) -> f64 {
    let mut n = 0i64;
    let mut diff = 0i64;
    for (a, b) in s1.chars().zip(s2.chars()) {
        let a = a.to_uppercase().next().unwrap_or(a);
        let b = b.to_uppercase().next().unwrap_or(b);
        if "ACGT".contains(a) && "ACGT".contains(b) {
            n += 1;
            if a != b {
                diff += 1;
            }
        }
    }
    if n == 0 {
        return 1.0;
    }
    let p = diff as f64 / n as f64;
    if p >= 0.75 {
        return 3.0;
    }
    -0.75 * (1.0 - (4.0 / 3.0) * p).ln()
}

struct Cluster {
    newick: String,
    height: f64,
    size: i64,
}

fn build_upgma_gene_tree<'py>(py: Python<'py>, alignment: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let mut labels: Vec<String> = Vec::new();
    let mut seqs: Vec<String> = Vec::new();
    for key in alignment.call_method0("keys")?.try_iter()? {
        let key = key?;
        let value = alignment.get_item(&key)?;
        labels.push(key.extract::<String>()?);
        seqs.push(value.extract::<String>()?);
    }
    let unit = substitution_unit(py)?;
    let cls = network_class(py)?;
    if labels.len() == 1 {
        let kwargs = PyDict::new(py);
        kwargs.set_item("branch_length_unit", unit)?;
        let net = cls.call((), Some(&kwargs))?;
        let only = Py::new(py, Node::make(py, labels[0].clone(), false, None, None, None))?;
        net.call_method1("add_nodes", (only,))?;
        return Ok(net);
    }
    let n = labels.len();
    let mut clusters: Vec<Cluster> = labels
        .iter()
        .map(|lab| Cluster { newick: lab.clone(), height: 0.0, size: 1 })
        .collect();
    let mut dist: FxHashMap<(usize, usize), f64> = FxHashMap::default();
    for i in 0..n {
        for j in (i + 1)..n {
            dist.insert((i, j), jc_distance(&seqs[i], &seqs[j]));
        }
    }
    let key = |a: usize, b: usize| (a.min(b), a.max(b));
    let mut active: Vec<usize> = (0..n).collect();
    let mut next_id = n;
    while active.len() > 1 {
        let mut best: Option<(usize, usize)> = None;
        let mut best_d = f64::INFINITY;
        for a in 0..active.len() {
            for b in (a + 1)..active.len() {
                let d = dist
                    .get(&key(active[a], active[b]))
                    .copied()
                    .ok_or_else(|| PyKeyError::new_err("distance"))?;
                if d < best_d {
                    best_d = d;
                    best = Some((a, b));
                }
            }
        }
        let Some((a, b)) = best else {
            return Err(PyTypeError::new_err("cannot unpack non-iterable NoneType object"));
        };
        let (ia, ib) = (active[a], active[b]);
        let new_height = best_d / 2.0;
        let la = (new_height - clusters[ia].height).max(1e-9);
        let lb = (new_height - clusters[ib].height).max(1e-9);
        let new_newick = format!(
            "({}:{:.10},{}:{:.10})I{}",
            clusters[ia].newick, la, clusters[ib].newick, lb, next_id
        );
        let (size_a, size_b) = (clusters[ia].size, clusters[ib].size);
        clusters.push(Cluster { newick: new_newick, height: new_height, size: size_a + size_b });
        let new_idx = clusters.len() - 1;
        let remaining: Vec<usize> = active
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != a && *k != b)
            .map(|(_, x)| *x)
            .collect();
        for &ic in &remaining {
            let d_ac = dist.get(&key(ia, ic)).copied().ok_or_else(|| PyKeyError::new_err("distance"))?;
            let d_bc = dist.get(&key(ib, ic)).copied().ok_or_else(|| PyKeyError::new_err("distance"))?;
            let avg = (size_a as f64 * d_ac + size_b as f64 * d_bc) / (size_a + size_b) as f64;
            dist.insert(key(new_idx, ic), avg);
        }
        active = remaining;
        active.push(new_idx);
        next_id += 1;
    }
    let Some(&root) = active.first() else {
        return Err(PyIndexError::new_err("list index out of range"));
    };
    let newick = format!("{};", clusters[root].newick);
    let kwargs = PyDict::new(py);
    kwargs.set_item("branch_length_unit", unit)?;
    cls.call_method("from_newick", (newick,), Some(&kwargs))
}

// ═══════════════════════════════════════════════════════════════════════
// Simple operators
// ═══════════════════════════════════════════════════════════════════════

fn internal_nodes(net: &Network) -> Vec<u64> {
    net.nodes
        .keys()
        .copied()
        .filter(|&u| net.out_deg(u) > 0 && net.in_deg(u) > 0)
        .collect()
}

fn op_change_theta(st: &St<'_>, rng: &Rng<'_>, window: f64) -> OpResult {
    let py = st.py;
    let old_obj = st.get("theta")?;
    let old: f64 = old_obj.extract()?;
    let snap = st.snapshot()?;
    let f = (window * (rng.random()? - 0.5)).exp();
    let new_theta = (old * f).into_pyobject(py)?.into_any();
    st.set("theta", &new_theta)?;
    st.invalidate_network()?;
    let mut undo = SeqUndo::blank(st, snap);
    undo.theta = Some(old_obj.unbind());
    op_ok(py, f.ln(), undo)
}

fn op_change_gamma(st: &St<'_>, rng: &Rng<'_>, window: f64) -> OpResult {
    let py = st.py;
    let old_net = st.net()?;
    if retic_uids(py, &old_net.borrow()).is_empty() {
        return Ok(None);
    }
    let old_heights = st.net_heights_obj()?;
    let snap = st.snapshot()?;
    let (new_net, _) = clone_net(&old_net)?;
    let new_heights = node_heights_uid(py, &new_net.borrow())?;
    let rets = retic_uids(py, &new_net.borrow());
    let r = rets[rng.integers(rets.len())?];
    let in_edges = in_edges_in_set_order(py, &new_net.borrow(), r);
    if in_edges.len() < 2 {
        return Ok(None);
    }
    let (e0, e1) = (in_edges[0], in_edges[1]);
    let g = edge_gamma(py, &new_net.borrow(), e0).ok_or_else(|| {
        PyTypeError::new_err("unsupported operand type(s) for +: 'NoneType' and 'float'")
    })?;
    let mut g_new = g + window * (rng.random()? - 0.5);
    let mut guard = 0;
    while g_new <= 0.0 || g_new >= 1.0 {
        if g_new <= 0.0 {
            g_new = -g_new;
        }
        if g_new >= 1.0 {
            g_new = 2.0 - g_new;
        }
        guard += 1;
        if guard > 10_000 {
            return Ok(None);
        }
    }
    set_gamma(py, &new_net, e0, Some(g_new))?;
    set_gamma(py, &new_net, e1, Some(1.0 - g_new))?;
    sync_lengths(py, &new_net, &new_heights)?;
    st.swap_net(&new_net, &new_heights)?;
    st.invalidate_network()?;
    op_ok(py, 0.0, SeqUndo::with_net(st, snap, old_net, old_heights))
}

/// ``_slide_height_move`` on a uid map; ``None`` when no valid move exists.
fn slide_height_move(py: Python<'_>, net: &Network, h: &mut Heights, rng: &Rng<'_>, root_scale_window: f64) -> PyResult<Option<f64>> {
    let mut candidates = internal_nodes(net);
    let root = net.root_uid(py)?;
    candidates.push(root);
    let node = candidates[rng.integers(candidates.len())?];
    let children = net.children(node);
    let mut lower: Option<f64> = None;
    for c in children {
        let hc = hget(py, net, h, c)?;
        lower = Some(match lower {
            None => hc,
            Some(m) => {
                if hc > m {
                    hc
                } else {
                    m
                }
            }
        });
    }
    let lower = lower.unwrap_or(0.0);
    if node == root || net.in_deg(node) == 0 {
        let f = (root_scale_window * (rng.random()? - 0.5)).exp();
        let new_h = hget(py, net, h, node)? * f;
        if new_h <= lower {
            return Ok(None);
        }
        h.insert(node, new_h);
        return Ok(Some(f.ln()));
    }
    let parents = net.parents(node);
    let mut upper: Option<f64> = None;
    for p in parents {
        let hp = hget(py, net, h, p)?;
        upper = Some(match upper {
            None => hp,
            Some(m) => {
                if hp < m {
                    hp
                } else {
                    m
                }
            }
        });
    }
    let upper = upper.ok_or_else(|| PyValueError::new_err("min() arg is an empty sequence"))?;
    if upper <= lower {
        return Ok(None);
    }
    h.insert(node, lower + (upper - lower) * rng.random()?);
    Ok(Some(0.0))
}

fn op_net_node_height(st: &St<'_>, rng: &Rng<'_>, root_scale_window: f64) -> OpResult {
    let py = st.py;
    let old_net = st.net()?;
    let old_heights = st.net_heights_obj()?;
    let snap = st.snapshot()?;
    let (new_net, _) = clone_net(&old_net)?;
    let mut new_heights = node_heights_uid(py, &new_net.borrow())?;
    let loghr = slide_height_move(py, &new_net.borrow(), &mut new_heights, rng, root_scale_window)?;
    let Some(loghr) = loghr else { return Ok(None) };
    sync_lengths(py, &new_net, &new_heights)?;
    st.swap_net(&new_net, &new_heights)?;
    st.invalidate_network()?;
    op_ok(py, loghr, SeqUndo::with_net(st, snap, old_net, old_heights))
}

fn op_gene_node_height(st: &St<'_>, rng: &Rng<'_>, root_scale_window: f64) -> OpResult {
    let py = st.py;
    let n_loci = st.n_loci()?;
    if n_loci == 0 {
        return Ok(None);
    }
    let i = rng.integers(n_loci)?;
    let old_gt = st.gene_tree(i)?;
    let old_h = st.gt_heights()?.get_item(i)?;
    let snap = st.snapshot()?;
    let (new_gt, _) = clone_net(&old_gt)?;
    let mut new_h = node_heights_uid(py, &new_gt.borrow())?;
    let loghr = slide_height_move(py, &new_gt.borrow(), &mut new_h, rng, root_scale_window)?;
    let Some(loghr) = loghr else { return Ok(None) };
    sync_lengths(py, &new_gt, &new_h)?;
    let d = heights_to_py(py, &new_gt.borrow(), &new_h)?;
    st.gene_trees()?.set_item(i, new_gt.as_any())?;
    st.gt_heights()?.set_item(i, d.as_any())?;
    st.invalidate_locus(i)?;
    op_ok(py, loghr, SeqUndo::with_locus(st, snap, i, old_gt.into_any(), old_h))
}

fn op_gene_tree_nni(st: &St<'_>, rng: &Rng<'_>) -> OpResult {
    let py = st.py;
    let n_loci = st.n_loci()?;
    if n_loci == 0 {
        return Ok(None);
    }
    let i = rng.integers(n_loci)?;
    let old_gt = st.gene_tree(i)?;
    let old_h = st.gt_heights()?.get_item(i)?;
    let snap = st.snapshot()?;
    let (new_gt, _) = clone_net(&old_gt)?;
    let new_h = node_heights_uid(py, &new_gt.borrow())?;

    let mut eligible: Vec<(u64, u64, u64, u64)> = Vec::new();
    {
        let g = new_gt.borrow();
        let internal: Vec<u64> = g.nodes.keys().copied().filter(|&u| g.out_deg(u) >= 2).collect();
        for v in internal {
            let parents = g.parents(v);
            let Some(&p) = parents.first() else { continue };
            let sibs: Vec<u64> = g.children(p).into_iter().filter(|&c| c != v).collect();
            let Some(&s) = sibs.first() else { continue };
            let v_children = g.children(v);
            if v_children.len() < 2 {
                continue;
            }
            let c = v_children[rng.integers(v_children.len())?];
            eligible.push((p, v, s, c));
        }
    }
    if eligible.is_empty() {
        return Ok(None);
    }
    let (p, v, s, c) = eligible[rng.integers(eligible.len())?];
    {
        let g = new_gt.borrow();
        if hget(py, &g, &new_h, v)? <= hget(py, &g, &new_h, s)? {
            return Ok(None);
        }
    }
    let (e_ps, e_vc) = {
        let g = new_gt.borrow();
        (edge_between(py, &g, p, s)?, edge_between(py, &g, v, c)?)
    };
    remove_edge(py, &new_gt, e_ps);
    remove_edge(py, &new_gt, e_vc);
    new_edge(py, &new_gt, p, c, None, None)?;
    new_edge(py, &new_gt, v, s, None, None)?;
    sync_lengths(py, &new_gt, &new_h)?;

    let d = heights_to_py(py, &new_gt.borrow(), &new_h)?;
    st.gene_trees()?.set_item(i, new_gt.as_any())?;
    st.gt_heights()?.set_item(i, d.as_any())?;
    st.invalidate_locus(i)?;
    op_ok(py, 0.0, SeqUndo::with_locus(st, snap, i, old_gt.into_any(), old_h))
}

// ═══════════════════════════════════════════════════════════════════════
// Reticulation geometry
// ═══════════════════════════════════════════════════════════════════════

/// ``_split_edge``: a new degree-2 node at height ``t`` on ``(parent, child)``.
fn split_edge(py: Python<'_>, net: &Bound<'_, Network>, h: &mut Heights, parent: u64, child: u64, t: f64) -> PyResult<u64> {
    let e = edge_between(py, &net.borrow(), parent, child)?;
    remove_edge(py, net, e);
    let mid = add_uid_node(py, net)?;
    h.insert(mid, t);
    new_edge(py, net, parent, mid, None, None)?;
    new_edge(py, net, mid, child, None, None)?;
    Ok(mid)
}

/// ``_suppress_degree2``: merge the two edges of a degree-2 node; returns
/// the resulting edge count.
fn suppress_degree2(py: Python<'_>, net: &Bound<'_, Network>, h: &mut Heights, node: u64) -> PyResult<usize> {
    let (in_e, out_e) = {
        let n = net.borrow();
        if n.in_deg(node) != 1 || n.out_deg(node) != 1 {
            return Ok(n.edges.len());
        }
        (n.in_edge_uids(node)[0], n.out_edge_uids(node)[0])
    };
    let (parent, child, gamma) = {
        let n = net.borrow();
        let parent = edge_ends(&n, in_e).map(|(s, _)| s).unwrap_or(u64::MAX);
        let child = edge_ends(&n, out_e).map(|(_, d)| d).unwrap_or(u64::MAX);
        (parent, child, edge_gamma(py, &n, out_e))
    };
    remove_edge(py, net, in_e);
    remove_edge(py, net, out_e);
    remove_node(py, net, node);
    h.remove(&node);
    let child_retic = is_retic(py, &net.borrow(), child);
    if child_retic {
        // ``merged.set_gamma(gamma)`` before the edge is added.
        match gamma {
            Some(g) => new_edge(py, net, parent, child, None, Some(g))?,
            None => return Err(PyTypeError::new_err("argument 'gamma': must be real number, not NoneType")),
        };
    } else {
        new_edge(py, net, parent, child, None, None)?;
    }
    Ok(net.borrow().edges.len())
}

fn float_max(a: f64, b: f64) -> f64 {
    // Python's ``max(a, b)``: the first argument unless the second is larger.
    if b > a {
        b
    } else {
        a
    }
}

/// A scored single-reticulation placement.
struct Placement {
    donor: (String, String),
    retic: (String, String),
    l2: f64,
    s_rep: f64,
}

/// ``_geometric_placements``: ``(donor (src,dest), retic (src,dest))``.
fn geometric_placements(py: Python<'_>, net: &Network, h: &Heights) -> PyResult<Vec<((u64, u64), (u64, u64))>> {
    let edges: Vec<(u64, u64)> = net.edges.values().map(|e| (e.su, e.du)).collect();
    let mut out = Vec::new();
    for (i, &(v3, v4)) in edges.iter().enumerate() {
        let (h3, h4) = (hget(py, net, h, v3)?, hget(py, net, h, v4)?);
        if h3 - h4 <= 0.0 {
            continue;
        }
        for (j, &(v5, v6)) in edges.iter().enumerate() {
            if j == i {
                continue;
            }
            let (h5, h6) = (hget(py, net, h, v5)?, hget(py, net, h, v6)?);
            if h5 - h6 <= 0.0 {
                continue;
            }
            if h3 <= float_max(h4, h6) {
                continue;
            }
            out.push(((v3, v4), (v5, v6)));
        }
    }
    Ok(out)
}

/// ``_add_reticulation_placements``: every valid placement with its
/// representative midpoint score.
fn add_reticulation_placements(st: &St<'_>, net: &Bound<'_, Network>, h: &Heights) -> PyResult<Vec<Placement>> {
    let py = st.py;
    let edges: Vec<(u64, u64)> = net.borrow().edges.values().map(|e| (e.su, e.du)).collect();
    let labels: Vec<(String, String)> = {
        let n = net.borrow();
        edges.iter().map(|&(s, d)| (label_of(py, &n, s), label_of(py, &n, d))).collect()
    };
    let mut placements = Vec::new();
    for (di, &(v3, v4)) in edges.iter().enumerate() {
        let (h3, h4) = {
            let n = net.borrow();
            (hget(py, &n, h, v3)?, hget(py, &n, h, v4)?)
        };
        let l1 = h3 - h4;
        if l1 <= 0.0 {
            continue;
        }
        for (ri, &(v5, v6)) in edges.iter().enumerate() {
            if ri == di {
                continue;
            }
            let (h5, h6) = {
                let n = net.borrow();
                (hget(py, &n, h, v5)?, hget(py, &n, h, v6)?)
            };
            let l2 = h5 - h6;
            if l2 <= 0.0 {
                continue;
            }
            let t2 = h6 + 0.5 * l2;
            let lo = float_max(h4, t2);
            let hi = h3;
            if hi <= lo {
                continue;
            }
            let t1 = 0.5 * (lo + hi);
            let (cand, _) = clone_net(net)?;
            let mut ch = node_heights_uid(py, &cand.borrow())?;
            let (cd, cr) = {
                let c = cand.borrow();
                (
                    find_edge_by_labels(py, &c, &labels[di].0, &labels[di].1),
                    find_edge_by_labels(py, &c, &labels[ri].0, &labels[ri].1),
                )
            };
            let (Some((_, d_src, d_dst)), Some((_, r_src, r_dst))) = (cd, cr) else { continue };
            let built = (|| -> PyResult<bool> {
                let cv1 = split_edge(py, &cand, &mut ch, d_src, d_dst, t1)?;
                let cv2 = split_edge(py, &cand, &mut ch, r_src, r_dst, t2)?;
                let in_e = cand.borrow().in_edge_uids(cv2);
                if in_e.len() != 1 {
                    return Ok(false);
                }
                set_gamma(py, &cand, in_e[0], Some(0.5))?;
                set_retic(py, &cand, cv2, true);
                new_edge(py, &cand, cv1, cv2, None, Some(0.5))?;
                Ok(true)
            })();
            match built {
                Ok(true) => {}
                _ => continue,
            }
            if has_parallel_edges(&cand.borrow()) {
                continue;
            }
            if !cand.borrow().is_acyclic_impl() {
                continue;
            }
            sync_lengths(py, &cand, &ch)?;
            let s_rep = network_only_log_score(st, &cand)?;
            if !s_rep.is_finite() {
                continue;
            }
            placements.push(Placement {
                donor: labels[di].clone(),
                retic: labels[ri].clone(),
                l2,
                s_rep,
            });
        }
    }
    Ok(placements)
}

/// ``_network_only_log_score``: MSNC (height-scale marginalised) of the
/// current gene trees under ``net`` plus the network prior.
fn network_only_log_score(st: &St<'_>, net: &Bound<'_, Network>) -> PyResult<f64> {
    let py = st.py;
    let msnc = msnc_module(py)?;
    let built = msnc.getattr("build_network_msnc_index")?.call1((net,))?;
    let net_idx = built.get_item(0)?;
    let sp_heights = built.get_item(1)?;
    if sp_heights.is_none() {
        return Ok(f64::NEG_INFINITY);
    }
    let eng = st.engine()?;
    let n_loci = st.n_loci()?;
    let gene_trees = st.gene_trees()?;
    let theta = st.theta()?;
    let branch_thetas = st.get("branch_thetas")?;
    let prebuilt = msnc.getattr("msnc_log_density_prebuilt")?;
    let gene_index = eng.getattr("_gene_index")?;
    let mut total = 0.0;
    for i in 0..n_loci {
        let gt = gene_trees.get_item(i)?;
        let pair = gene_index.call1((i, gt))?;
        let gti = pair.get_item(0)?;
        let events = pair.get_item(1)?;
        let ev: Vec<(f64, Bound<'_, PyAny>, Bound<'_, PyAny>, Bound<'_, PyAny>)> = events.extract()?;
        let mut terms: Vec<f64> = Vec::new();
        for &s in &HEIGHT_SCALE_GRID {
            let sev = if s == 1.0 {
                events.clone()
            } else {
                let mut items: Vec<Bound<'_, PyAny>> = Vec::with_capacity(ev.len());
                for (t, a, b, c) in &ev {
                    items.push(
                        PyTuple::new(py, [(t * s).into_pyobject(py)?.into_any(), a.clone(), b.clone(), c.clone()])?.into_any(),
                    );
                }
                PyList::new(py, items)?.into_any()
            };
            let d: f64 = prebuilt
                .call1((&net_idx, &sp_heights, &gti, sev, theta, &branch_thetas))?
                .extract()?;
            if d.is_finite() {
                terms.push(d);
            }
        }
        if terms.is_empty() {
            return Ok(f64::NEG_INFINITY);
        }
        total += logsumexp_ref(&terms);
    }
    total += log_prior_seq_impl(py, net, theta, &st.priors()?)?;
    if !total.is_finite() {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(total)
}

/// The network built by an add: the copy, its heights and the birth
/// Jacobian pieces ``l2`` / ``win``.
struct AddBuild<'py> {
    net: Bound<'py, Network>,
    heights: Heights,
    l2: f64,
    win: f64,
}

/// Copy ``base``, draw ``t2``, ``t1`` and ``gamma`` and attach a hybrid on
/// the chosen placement.  ``None`` is a declined proposal.
fn build_add<'py>(
    py: Python<'py>,
    rng: &Rng<'py>,
    base: &Bound<'py, Network>,
    donor: &(String, String),
    retic: &(String, String),
) -> PyResult<Option<AddBuild<'py>>> {
    let (new_net, _) = clone_net(base)?;
    let mut h = node_heights_uid(py, &new_net.borrow())?;
    let (d, r) = {
        let n = new_net.borrow();
        (find_edge_by_labels(py, &n, &donor.0, &donor.1), find_edge_by_labels(py, &n, &retic.0, &retic.1))
    };
    let (Some((_, v3, v4)), Some((_, v5, v6))) = (d, r) else { return Ok(None) };
    let (h3, h4, h5, h6) = {
        let n = new_net.borrow();
        (hget(py, &n, &h, v3)?, hget(py, &n, &h, v4)?, hget(py, &n, &h, v5)?, hget(py, &n, &h, v6)?)
    };
    let l2 = h5 - h6;
    if l2 <= 0.0 {
        return Ok(None);
    }
    let t2 = h6 + l2 * rng.random()?;
    let lo = float_max(h4, t2);
    let hi = h3;
    let win = hi - lo;
    if win <= 0.0 {
        return Ok(None);
    }
    let t1 = lo + win * rng.random()?;
    let g = rng.random()?;
    let built = (|| -> PyResult<bool> {
        let v1 = split_edge(py, &new_net, &mut h, v3, v4, t1)?;
        let v2 = split_edge(py, &new_net, &mut h, v5, v6, t2)?;
        let in_e = new_net.borrow().in_edge_uids(v2);
        if in_e.len() != 1 {
            return Ok(false);
        }
        set_gamma(py, &new_net, in_e[0], Some(1.0 - g))?;
        set_retic(py, &new_net, v2, true);
        new_edge(py, &new_net, v1, v2, None, Some(g))?;
        Ok(true)
    })();
    match built {
        Ok(true) => {}
        _ => return Ok(None),
    }
    if has_parallel_edges(&new_net.borrow()) {
        return Ok(None);
    }
    if !new_net.borrow().is_acyclic_impl() {
        return Ok(None);
    }
    Ok(Some(AddBuild { net: new_net, heights: h, l2, win }))
}

/// Pick a placement with probability ``exp(s_rep - lse)``; returns
/// ``(index, lse)``.
fn choose_placement(py: Python<'_>, rng: &Rng<'_>, placements: &[Placement]) -> PyResult<(usize, f64)> {
    let s_reps: Vec<f64> = placements.iter().map(|p| p.s_rep).collect();
    let lse = logsumexp_ref(&s_reps);
    let probs = selection_probs(py, &s_reps, lse)?;
    let k = rng.choice(placements.len(), &probs)?;
    Ok((k, lse))
}

/// The shared informed-add body.  ``level_cap`` applies the level guard
/// (coupled / decoupled variants); ``coupled`` re-proposes the gene trees.
fn add_reticulation_generic(st: &St<'_>, rng: &Rng<'_>, level_cap: bool, coupled: bool) -> OpResult {
    let py = st.py;
    let r_count = st.num_reticulations()?;
    let priors = st.priors()?;
    if r_count >= priors.max_reticulations {
        return Ok(None);
    }
    let old_net = st.net()?;
    let old_heights_obj = st.net_heights_obj()?;
    let old_heights = heights_from_py(py, &old_net.borrow(), &old_heights_obj)?;
    let old_gts = st.get("gene_trees")?;
    let old_gt_h = st.get("gt_heights")?;

    let placements = add_reticulation_placements(st, &old_net, &old_heights)?;
    if placements.is_empty() {
        return Ok(None);
    }
    let (k, lse) = choose_placement(py, rng, &placements)?;
    let chosen = &placements[k];
    let s_star = chosen.s_rep;

    let snap = st.snapshot()?;
    let Some(built) = build_add(py, rng, &old_net, &chosen.donor, &chosen.retic)? else {
        return Ok(None);
    };
    if level_cap {
        if let Some(cap) = priors.max_level {
            if network_level(py, &built.net) as i64 > cap {
                return Ok(None);
            }
        }
    }
    let mut loghr = lse - s_star - (2.0 * (r_count + 1) as f64).ln() + built.l2.ln() + built.win.ln();
    sync_lengths(py, &built.net, &built.heights)?;

    if coupled {
        let Some(repro) = coupled_gene_tree_reproposal(st, &built.net, &old_net, rng)? else {
            return Ok(None);
        };
        loghr += repro.log_qr - repro.log_qf;
        st.swap_net(&built.net, &built.heights)?;
        st.swap_gene_trees(&repro.gene_trees, &repro.heights)?;
        st.invalidate_network()?;
        st.invalidate_all_loci()?;
        return op_ok(py, loghr, SeqUndo::with_all(st, snap, old_net, old_heights_obj, old_gts, old_gt_h));
    }
    st.swap_net(&built.net, &built.heights)?;
    st.invalidate_network()?;
    op_ok(py, loghr, SeqUndo::with_net(st, snap, old_net, old_heights_obj))
}

/// The reticulation / parent pick of a delete and the reverse-add geometry.
struct DeletePick {
    v2: u64,
    v1: u64,
    l2: f64,
    win: f64,
    rev_donor: (String, String),
    rev_retic: (String, String),
}

fn pick_delete(py: Python<'_>, rng: &Rng<'_>, old_net: &Network, h: &Heights) -> PyResult<Option<DeletePick>> {
    let rets = retic_uids(py, old_net);
    if rets.is_empty() {
        return Ok(None);
    }
    let v2 = rets[rng.integers(rets.len())?];
    let parents = old_net.parents(v2);
    if parents.len() != 2 {
        return Ok(None);
    }
    let v1 = parents[rng.integers(2)?];
    let v5 = if parents[0] == v1 { parents[1] } else { parents[0] };
    if is_retic(py, old_net, v1) || old_net.in_deg(v1) == 0 {
        return Ok(None);
    }
    let v3s = old_net.parents(v1);
    if v3s.len() != 1 {
        return Ok(None);
    }
    let v3 = v3s[0];
    let other_children: Vec<u64> = old_net.children(v1).into_iter().filter(|&c| c != v2).collect();
    if other_children.len() != 1 {
        return Ok(None);
    }
    let v4 = other_children[0];
    let v6s = old_net.children(v2);
    if v6s.len() != 1 {
        return Ok(None);
    }
    let v6 = v6s[0];
    if old_net.parents(v4).contains(&v3) {
        return Ok(None);
    }
    if old_net.parents(v6).contains(&v5) {
        return Ok(None);
    }
    if v3 == v5 && v4 == v6 {
        return Ok(None);
    }
    let l2 = hget(py, old_net, h, v5)? - hget(py, old_net, h, v6)?;
    if l2 <= 0.0 {
        return Ok(None);
    }
    let t2 = hget(py, old_net, h, v2)?;
    let lo = float_max(hget(py, old_net, h, v4)?, t2);
    let hi = hget(py, old_net, h, v3)?;
    let win = hi - lo;
    if win <= 0.0 {
        return Ok(None);
    }
    Ok(Some(DeletePick {
        v2,
        v1,
        l2,
        win,
        rev_donor: (label_of(py, old_net, v3), label_of(py, old_net, v4)),
        rev_retic: (label_of(py, old_net, v5), label_of(py, old_net, v6)),
    }))
}

/// Copy ``old_net`` without the hybrid edge ``v1 -> v2`` (both degree-2
/// nodes suppressed).  ``None`` is a declined proposal.
fn apply_delete<'py>(
    py: Python<'py>,
    old_net: &Bound<'py, Network>,
    pick: &DeletePick,
) -> PyResult<Option<(Bound<'py, Network>, Heights)>> {
    let (new_net, _) = clone_net(old_net)?;
    let mut h = node_heights_uid(py, &new_net.borrow())?;
    let (v2_label, v1_label) = {
        let o = old_net.borrow();
        (label_of(py, &o, pick.v2), label_of(py, &o, pick.v1))
    };
    let (cv2, cv1) = {
        let n = new_net.borrow();
        let cv2 = n
            .nodes
            .keys()
            .copied()
            .find(|&v| is_retic(py, &n, v) && label_of(py, &n, v) == v2_label);
        let Some(cv2) = cv2 else { return Ok(None) };
        let cv1 = n.parents(cv2).into_iter().find(|&p| label_of(py, &n, p) == v1_label);
        let Some(cv1) = cv1 else { return Ok(None) };
        (cv2, cv1)
    };
    let done = (|| -> PyResult<()> {
        let del_edge = edge_between(py, &new_net.borrow(), cv1, cv2)?;
        remove_edge(py, &new_net, del_edge);
        set_retic(py, &new_net, cv2, false);
        suppress_degree2(py, &new_net, &mut h, cv1)?;
        suppress_degree2(py, &new_net, &mut h, cv2)?;
        Ok(())
    })();
    if done.is_err() {
        return Ok(None);
    }
    if new_net.borrow().edges.len() < 2 {
        return Ok(None);
    }
    sync_lengths(py, &new_net, &h)?;
    Ok(Some((new_net, h)))
}

/// The reverse-add normaliser on the post-delete network: ``(s_rev, lse)``.
fn reverse_add_terms(st: &St<'_>, net: &Bound<'_, Network>, h: &Heights, pick: &DeletePick) -> PyResult<Option<(f64, f64)>> {
    let placements = add_reticulation_placements(st, net, h)?;
    if placements.is_empty() {
        return Ok(None);
    }
    let s_reps: Vec<f64> = placements.iter().map(|p| p.s_rep).collect();
    let lse = logsumexp_ref(&s_reps);
    let s_rev = placements
        .iter()
        .find(|p| p.donor == pick.rev_donor && p.retic == pick.rev_retic)
        .map(|p| p.s_rep);
    Ok(s_rev.map(|s| (s, lse)))
}

/// The shared informed-delete body; ``coupled`` re-proposes the gene trees
/// (and scores the reverse add with the re-proposed trees installed).
fn delete_reticulation_generic(st: &St<'_>, rng: &Rng<'_>, coupled: bool) -> OpResult {
    let py = st.py;
    let r_count = st.num_reticulations()?;
    if r_count == 0 {
        return Ok(None);
    }
    let old_net = st.net()?;
    let old_heights_obj = st.net_heights_obj()?;
    let old_heights = heights_from_py(py, &old_net.borrow(), &old_heights_obj)?;
    let old_gts = st.get("gene_trees")?;
    let old_gt_h = st.get("gt_heights")?;

    let Some(pick) = pick_delete(py, rng, &old_net.borrow(), &old_heights)? else {
        return Ok(None);
    };
    let snap = st.snapshot()?;
    let Some((new_net, new_heights)) = apply_delete(py, &old_net, &pick)? else {
        return Ok(None);
    };

    let base = (2.0 * r_count as f64).ln() - pick.l2.ln() - pick.win.ln();
    if coupled {
        let Some(repro) = coupled_gene_tree_reproposal(st, &new_net, &old_net, rng)? else {
            return Ok(None);
        };
        // The reverse add enumerates placements with the post-delete gene
        // trees installed.
        st.swap_gene_trees(&repro.gene_trees, &repro.heights)?;
        let terms = reverse_add_terms(st, &new_net, &new_heights, &pick)?;
        let Some((s_rev, lse)) = terms else {
            st.set("gene_trees", &old_gts)?;
            st.set("gt_heights", &old_gt_h)?;
            return Ok(None);
        };
        let loghr = s_rev - lse + base + (repro.log_qr - repro.log_qf);
        st.swap_net(&new_net, &new_heights)?;
        st.invalidate_network()?;
        st.invalidate_all_loci()?;
        return op_ok(py, loghr, SeqUndo::with_all(st, snap, old_net, old_heights_obj, old_gts, old_gt_h));
    }
    let Some((s_rev, lse)) = reverse_add_terms(st, &new_net, &new_heights, &pick)? else {
        return Ok(None);
    };
    let loghr = s_rev - lse + base;
    st.swap_net(&new_net, &new_heights)?;
    st.invalidate_network()?;
    op_ok(py, loghr, SeqUndo::with_net(st, snap, old_net, old_heights_obj))
}

// ═══════════════════════════════════════════════════════════════════════
// Coupled gene-tree re-proposal
// ═══════════════════════════════════════════════════════════════════════

/// ``_gene_tree_nni_neighbors``: height-preserving single-NNI rearrangements.
fn gene_tree_nni_neighbors<'py>(
    py: Python<'py>,
    gt: &Bound<'py, Network>,
    h: &Heights,
) -> PyResult<Vec<(Bound<'py, Network>, Heights)>> {
    let mut quads: Vec<(LabelSet, LabelSet)> = Vec::new();
    {
        let g = gt.borrow();
        let uids: Vec<u64> = g.nodes.keys().copied().collect();
        for v in uids {
            let kids = g.children(v);
            if kids.len() < 2 {
                continue;
            }
            let parents = g.parents(v);
            let Some(&p) = parents.first() else { continue };
            let sibs: Vec<u64> = g.children(p).into_iter().filter(|&c| c != v).collect();
            if sibs.len() != 1 {
                continue;
            }
            let s = sibs[0];
            if hget(py, &g, h, v)? <= hget(py, &g, h, s)? {
                continue;
            }
            let s_ls = leaf_desc_labels(py, &g, s);
            for c in kids {
                quads.push((s_ls.clone(), leaf_desc_labels(py, &g, c)));
            }
        }
    }
    let mut neighbors = Vec::new();
    let mut seen: FxHashSet<Sig> = FxHashSet::default();
    for (s_ls, c_ls) in quads {
        let (cand, _) = clone_net(gt)?;
        let ch = node_heights_uid(py, &cand.borrow())?;
        let (cp, cs, cv, cc) = {
            let c = cand.borrow();
            let (Some(cs), Some(cc)) = (node_by_leafset(py, &c, &s_ls), node_by_leafset(py, &c, &c_ls)) else {
                continue;
            };
            let cp_list = c.parents(cs);
            let cv_list = c.parents(cc);
            let (Some(&cp), Some(&cv)) = (cp_list.first(), cv_list.first()) else { continue };
            (cp, cs, cv, cc)
        };
        if cp == cv {
            continue;
        }
        let swapped = (|| -> PyResult<()> {
            let e1 = edge_between(py, &cand.borrow(), cp, cs)?;
            remove_edge(py, &cand, e1);
            let e2 = edge_between(py, &cand.borrow(), cv, cc)?;
            remove_edge(py, &cand, e2);
            new_edge(py, &cand, cp, cc, None, None)?;
            new_edge(py, &cand, cv, cs, None, None)?;
            Ok(())
        })();
        if swapped.is_err() {
            continue;
        }
        sync_lengths(py, &cand, &ch)?;
        let sig = gt_signature(py, &cand.borrow(), &ch)?;
        if !seen.insert(sig) {
            continue;
        }
        neighbors.push((cand, ch));
    }
    Ok(neighbors)
}

/// ``_scaled_gene_tree``: a copy with every height multiplied by ``s``.
fn scaled_gene_tree<'py>(py: Python<'py>, gt: &Bound<'py, Network>, s: f64) -> PyResult<(Bound<'py, Network>, Heights)> {
    let (cand, _) = clone_net(gt)?;
    let mut ch = node_heights_uid(py, &cand.borrow())?;
    if s != 1.0 {
        ch = scaled(&ch, s);
        sync_lengths(py, &cand, &ch)?;
    }
    Ok((cand, ch))
}

/// ``_candidate_set``: ``({gt} + NNI(gt)) x HEIGHT_SCALE_GRID``, de-duplicated.
fn candidate_set<'py>(py: Python<'py>, gt: &Bound<'py, Network>, h: &Heights) -> PyResult<Vec<(Bound<'py, Network>, Heights)>> {
    let mut bases: Vec<(Bound<'py, Network>, Heights)> = vec![(gt.clone(), h.clone())];
    bases.extend(gene_tree_nni_neighbors(py, gt, h)?);
    let mut out = Vec::new();
    let mut seen: FxHashSet<Sig> = FxHashSet::default();
    for (b_gt, _) in &bases {
        for &s in &HEIGHT_SCALE_GRID {
            let (cg, ch) = scaled_gene_tree(py, b_gt, s)?;
            let sig = gt_signature(py, &cg.borrow(), &ch)?;
            if !seen.insert(sig) {
                continue;
            }
            out.push((cg, ch));
        }
    }
    Ok(out)
}

/// Everything ``_enumerate_scored_candidates`` returns: signatures, log
/// weights and the descriptors needed to rebuild a chosen candidate.
struct Enumerated<'py> {
    sigs: Vec<Sig>,
    logw: Vec<f64>,
    desc: Vec<(usize, f64)>,
    bases: Vec<Bound<'py, Network>>,
}

impl<'py> Enumerated<'py> {
    /// ``rebuild(k)``: an independent ``(network, heights)`` for candidate ``k``.
    fn rebuild(&self, py: Python<'py>, k: usize) -> PyResult<(Bound<'py, Network>, Heights)> {
        let (bi, s) = self.desc.get(k).copied().ok_or_else(|| PyIndexError::new_err("list index out of range"))?;
        let (cg, _) = clone_net(&self.bases[bi])?;
        let mut ch = node_heights_uid(py, &cg.borrow())?;
        if s != 1.0 {
            ch = scaled(&ch, s);
        }
        sync_lengths(py, &cg, &ch)?;
        Ok((cg, ch))
    }
}

/// Score ``gt`` and its NNI neighbours at every height scale under one
/// network, cloning only the distinct base topologies.
#[allow(clippy::too_many_arguments)]
fn enumerate_scored_candidates<'py>(
    st: &St<'py>,
    i: usize,
    net_idx: &Bound<'py, PyAny>,
    sp_heights: &Bound<'py, PyAny>,
    theta: f64,
    gt: &Bound<'py, Network>,
    h: &Heights,
) -> PyResult<Enumerated<'py>> {
    let py = st.py;
    let eng = st.engine()?;
    let calc = eng.getattr("_calcs")?.get_item(i)?;
    let model = st.get("model")?;
    let species_of = eng.getattr("_species_of")?;
    let branch_thetas = st.get("branch_thetas")?;
    let msnc = msnc_module(py)?;
    let topo_index = msnc.getattr("build_gene_tree_topology_index")?;
    let events_fn = msnc.getattr("gene_tree_events")?;
    let prebuilt = msnc.getattr("msnc_log_density_prebuilt")?;

    let mut bases: Vec<(Bound<'py, Network>, Heights)> = vec![(gt.clone(), h.clone())];
    bases.extend(gene_tree_nni_neighbors(py, gt, h)?);

    let mut sigs: Vec<Sig> = Vec::new();
    let mut logw: Vec<f64> = Vec::new();
    let mut desc: Vec<(usize, f64)> = Vec::new();
    let mut seen: FxHashSet<Sig> = FxHashSet::default();
    for (bi, (b_gt, b_h)) in bases.iter().enumerate() {
        let base_lengths: Vec<(u64, Option<f64>)> = {
            let n = b_gt.borrow();
            n.edges.keys().map(|&e| (e, edge_len(py, &n, e))).collect()
        };
        let restore = |net: &Bound<'py, Network>| -> PyResult<()> {
            for &(e, l) in &base_lengths {
                set_len(py, net, e, l.unwrap_or(0.0))?;
            }
            Ok(())
        };
        let mut gti: Option<Bound<'py, PyAny>> = None;
        for &s in &HEIGHT_SCALE_GRID {
            let scaled_h: Heights;
            let sh: &Heights = if s != 1.0 {
                scaled_h = scaled(b_h, s);
                sync_lengths(py, b_gt, &scaled_h)?;
                &scaled_h
            } else {
                b_h
            };
            let sig = gt_signature(py, &b_gt.borrow(), sh)?;
            if seen.contains(&sig) {
                if s != 1.0 {
                    restore(b_gt)?;
                }
                continue;
            }
            let fels: f64 = calc.call_method1("log_likelihood", (b_gt, &model))?.extract()?;
            let mut w = f64::NEG_INFINITY;
            if fels.is_finite() {
                if gti.is_none() {
                    gti = Some(topo_index.call1((b_gt, &species_of))?);
                }
                let g = gti.as_ref().unwrap();
                let hd = heights_to_py(py, &b_gt.borrow(), sh)?;
                let events = events_fn.call1((b_gt, g, hd))?;
                let msnc_d: f64 = prebuilt
                    .call1((net_idx, sp_heights, g, events, theta, &branch_thetas))?
                    .extract()?;
                if msnc_d.is_finite() {
                    w = fels + msnc_d;
                }
            }
            if s != 1.0 {
                restore(b_gt)?;
            }
            seen.insert(sig.clone());
            sigs.push(sig);
            logw.push(w);
            desc.push((bi, s));
        }
    }
    Ok(Enumerated {
        sigs,
        logw,
        desc,
        bases: bases.into_iter().map(|(n, _)| n).collect(),
    })
}

struct Reproposal<'py> {
    gene_trees: Vec<Bound<'py, Network>>,
    heights: Vec<Heights>,
    log_qf: f64,
    log_qr: f64,
}

/// ``_coupled_gene_tree_reproposal``: guided joint re-draw of every locus.
fn coupled_gene_tree_reproposal<'py>(
    st: &St<'py>,
    target_net: &Bound<'py, Network>,
    reverse_net: &Bound<'py, Network>,
    rng: &Rng<'py>,
) -> PyResult<Option<Reproposal<'py>>> {
    let py = st.py;
    let msnc = msnc_module(py)?;
    let build = msnc.getattr("build_network_msnc_index")?;
    let tgt = build.call1((target_net,))?;
    let rev = build.call1((reverse_net,))?;
    let (tgt_idx, tgt_h) = (tgt.get_item(0)?, tgt.get_item(1)?);
    let (rev_idx, rev_h) = (rev.get_item(0)?, rev.get_item(1)?);
    if tgt_h.is_none() || rev_h.is_none() {
        return Ok(None);
    }
    let np = numpy_module(py)?;
    let np_exp = np.getattr("exp")?;
    let theta = st.theta()?;
    let n_loci = st.n_loci()?;
    let gene_trees = st.gene_trees()?;
    let gt_heights = st.gt_heights()?;

    let mut new_gts = Vec::with_capacity(n_loci);
    let mut new_hs = Vec::with_capacity(n_loci);
    let mut log_qf = 0.0;
    let mut log_qr = 0.0;
    for i in 0..n_loci {
        let g_i = gene_trees.get_item(i)?.cast_into::<Network>()?;
        let h_i = heights_from_py(py, &g_i.borrow(), &gt_heights.get_item(i)?)?;
        let target_sig = gt_signature(py, &g_i.borrow(), &h_i)?;

        let fwd = enumerate_scored_candidates(st, i, &tgt_idx, &tgt_h, theta, &g_i, &h_i)?;
        let fwd_lse = logsumexp_ref(&fwd.logw);
        if !fwd_lse.is_finite() {
            return Ok(None);
        }
        let lw = PyArray1::from_vec(py, fwd.logw.clone());
        let probs = np_exp.call1((lw.as_any().sub(fwd_lse)?,))?;
        let total: f64 = probs.call_method0("sum")?.extract()?;
        if !total.is_finite() || total <= 0.0 {
            return Ok(None);
        }
        let probs = probs.div(total)?;
        let j = rng.choice(fwd.sigs.len(), &probs)?;
        log_qf += fwd.logw[j] - fwd_lse;

        let (chosen_gt, chosen_h) = fwd.rebuild(py, j)?;

        let bwd = enumerate_scored_candidates(st, i, &rev_idx, &rev_h, theta, &chosen_gt, &chosen_h)?;
        let rev_lse = logsumexp_ref(&bwd.logw);
        if !rev_lse.is_finite() {
            return Ok(None);
        }
        let Some(m) = bwd.sigs.iter().position(|s| *s == target_sig) else {
            return Ok(None);
        };
        if !bwd.logw[m].is_finite() {
            return Ok(None);
        }
        log_qr += bwd.logw[m] - rev_lse;

        new_gts.push(chosen_gt);
        new_hs.push(chosen_h);
    }
    Ok(Some(Reproposal { gene_trees: new_gts, heights: new_hs, log_qf, log_qr }))
}

fn relocation_loghr(l2_new: f64, win_new: f64, l2_old: f64, win_old: f64, log_qf: f64, log_qr: f64) -> f64 {
    l2_new.ln() + win_new.ln() - l2_old.ln() - win_old.ln() + (log_qr - log_qf)
}

fn op_relocate_reticulation_coupled(st: &St<'_>, rng: &Rng<'_>) -> OpResult {
    let py = st.py;
    let r_count = st.num_reticulations()?;
    if r_count == 0 {
        return Ok(None);
    }
    let priors = st.priors()?;
    let old_net = st.net()?;
    let old_heights_obj = st.net_heights_obj()?;
    let old_heights = heights_from_py(py, &old_net.borrow(), &old_heights_obj)?;
    let old_gts = st.get("gene_trees")?;
    let old_gt_h = st.get("gt_heights")?;

    let Some(pick) = pick_delete(py, rng, &old_net.borrow(), &old_heights)? else {
        return Ok(None);
    };
    let snap = st.snapshot()?;
    let Some((n0, n0_h)) = apply_delete(py, &old_net, &pick)? else {
        return Ok(None);
    };

    let placements = geometric_placements(py, &n0.borrow(), &n0_h)?;
    if placements.is_empty() {
        return Ok(None);
    }
    let ((d3, d4), (r5, r6)) = placements[rng.integers(placements.len())?];
    let (donor, retic) = {
        let n = n0.borrow();
        (
            (label_of(py, &n, d3), label_of(py, &n, d4)),
            (label_of(py, &n, r5), label_of(py, &n, r6)),
        )
    };
    let Some(built) = build_add(py, rng, &n0, &donor, &retic)? else {
        return Ok(None);
    };
    if let Some(cap) = priors.max_level {
        if network_level(py, &built.net) as i64 > cap {
            return Ok(None);
        }
    }
    sync_lengths(py, &built.net, &built.heights)?;

    let Some(repro) = coupled_gene_tree_reproposal(st, &built.net, &old_net, rng)? else {
        return Ok(None);
    };
    let loghr = relocation_loghr(built.l2, built.win, pick.l2, pick.win, repro.log_qf, repro.log_qr);

    st.swap_net(&built.net, &built.heights)?;
    st.swap_gene_trees(&repro.gene_trees, &repro.heights)?;
    st.invalidate_network()?;
    st.invalidate_all_loci()?;
    op_ok(py, loghr, SeqUndo::with_all(st, snap, old_net, old_heights_obj, old_gts, old_gt_h))
}

// ═══════════════════════════════════════════════════════════════════════
// Posterior summaries
// ═══════════════════════════════════════════════════════════════════════

type TopoSig = (usize, BTreeSet<LabelSet>);

/// ``_topology_signature``: ``(reticulation count, non-trivial clades)``
/// of a newick string, or ``None`` when it does not parse.
fn topology_signature(py: Python<'_>, newick: &str) -> PyResult<Option<TopoSig>> {
    let cls = network_class(py)?;
    let parsed = match cls.call_method1("from_newick", (newick,)) {
        Ok(n) => n,
        Err(_) => return Ok(None),
    };
    let Ok(net_b) = parsed.cast_into::<Network>() else { return Ok(None) };
    let net = net_b.borrow();
    let leaves: LabelSet = net.leaf_uids().into_iter().map(|l| label_of(py, &net, l)).collect();
    let n = leaves.len();
    let empty: FxHashMap<String, String> = FxHashMap::default();
    let desc = descendant_species(py, &net, &empty);
    let mut clades: BTreeSet<LabelSet> = BTreeSet::new();
    let mut n_ret = 0usize;
    for v in net.nodes.keys() {
        if is_retic(py, &net, *v) {
            n_ret += 1;
        }
        if let Some(ds) = desc.get(v) {
            if 1 < ds.len() && ds.len() < n {
                clades.insert(ds.clone());
            }
        }
    }
    Ok(Some((n_ret, clades)))
}

fn topo_sig_to_py<'py>(py: Python<'py>, sig: &TopoSig) -> PyResult<Bound<'py, PyTuple>> {
    let sets: Vec<Bound<'py, PyFrozenSet>> = sig.1.iter().map(|s| labelset_to_py(py, s)).collect::<PyResult<_>>()?;
    let fs = PyFrozenSet::new(py, sets)?;
    PyTuple::new(py, [sig.0.into_pyobject(py)?.into_any(), fs.into_any()])
}

fn count_free_parameters(num_leaves: i64, num_reticulations: i64) -> i64 {
    num_leaves + 3 * num_reticulations
}

fn information_criteria<'py>(py: Python<'py>, log_likelihood: f64, k: i64, n: Option<i64>) -> PyResult<Bound<'py, PyDict>> {
    let aic = 2.0 * k as f64 - 2.0 * log_likelihood;
    let out = PyDict::new(py);
    out.set_item("log_likelihood", log_likelihood)?;
    out.set_item("k", k)?;
    out.set_item("AIC", aic)?;
    match n {
        Some(n) if n > 0 => {
            out.set_item("n", n)?;
            out.set_item("BIC", k as f64 * (n as f64).ln() - 2.0 * log_likelihood)?;
            let denom = n - k - 1;
            let aicc = if denom > 0 {
                aic + (2.0 * k as f64 * (k + 1) as f64 / denom as f64)
            } else {
                aic
            };
            out.set_item("AICc", aicc)?;
        }
        _ => {
            out.set_item("BIC", f64::NAN)?;
            out.set_item("AICc", aic)?;
        }
    }
    Ok(out)
}

fn reticulation_posterior<'py>(py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let items: Vec<Bound<'py, PyAny>> = samples.try_iter()?.collect::<PyResult<_>>()?;
    if items.is_empty() {
        return Ok(out);
    }
    let mut counts: std::collections::BTreeMap<i64, i64> = std::collections::BTreeMap::new();
    for s in &items {
        let k: i64 = s.getattr("num_reticulations")?.extract()?;
        *counts.entry(k).or_insert(0) += 1;
    }
    let tot = items.len() as f64;
    for (k, c) in counts {
        out.set_item(k, c as f64 / tot)?;
    }
    Ok(out)
}

fn topology_posterior<'py>(py: Python<'py>, samples: &Bound<'py, PyAny>, top_n: Option<i64>) -> PyResult<Vec<(String, f64)>> {
    let mut sig_count: IndexMap<TopoSig, usize> = IndexMap::new();
    let mut sig_rep: FxHashMap<TopoSig, String> = FxHashMap::default();
    let mut n = 0usize;
    for s in samples.try_iter()? {
        let s = s?;
        let newick: String = s.getattr("network_newick")?.extract()?;
        let Some(sig) = topology_signature(py, &newick)? else { continue };
        n += 1;
        *sig_count.entry(sig.clone()).or_insert(0) += 1;
        sig_rep.entry(sig).or_insert(newick);
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut ranked: Vec<(&TopoSig, usize)> = sig_count.iter().map(|(s, c)| (s, *c)).collect();
    ranked.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    let out: Vec<(String, f64)> = ranked
        .into_iter()
        .map(|(sig, cnt)| (sig_rep.get(sig).cloned().unwrap_or_default(), cnt as f64 / n as f64))
        .collect();
    Ok(match top_n {
        Some(k) if k > 0 => out.into_iter().take(k as usize).collect(),
        Some(k) if k < 0 => {
            let keep = out.len().saturating_sub(k.unsigned_abs() as usize);
            out.into_iter().take(keep).collect()
        }
        _ => out,
    })
}

// ═══════════════════════════════════════════════════════════════════════
// Python-visible functions
// ═══════════════════════════════════════════════════════════════════════

fn node_arg(net: &Network, node: &Bound<'_, PyAny>) -> PyResult<u64> {
    net.uid_of_any(node).ok_or_else(|| {
        PyKeyError::new_err(format!("node {} is not in the network", node.str().map(|s| s.to_string()).unwrap_or_default()))
    })
}

fn placements_to_py<'py>(py: Python<'py>, placements: &[Placement]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for p in placements {
        let d = PyDict::new(py);
        d.set_item("donor_labels", (p.donor.0.as_str(), p.donor.1.as_str()))?;
        d.set_item("retic_labels", (p.retic.0.as_str(), p.retic.1.as_str()))?;
        d.set_item("l2", p.l2)?;
        d.set_item("s_rep", p.s_rep)?;
        out.append(d)?;
    }
    Ok(out)
}

fn pair_to_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, h: &Heights) -> PyResult<Bound<'py, PyTuple>> {
    let d = heights_to_py(py, &net.borrow(), h)?;
    PyTuple::new(py, [net.as_any().clone(), d.into_any()])
}

#[pyfunction]
#[pyo3(name = "_seq_log_poisson_pmf")]
fn log_poisson_pmf_py(py: Python<'_>, k: i64, mean: f64) -> PyResult<f64> {
    log_poisson_pmf(py, k, mean)
}

#[pyfunction]
#[pyo3(name = "_seq_log_gamma_pdf")]
fn log_gamma_pdf_py(py: Python<'_>, x: f64, shape: f64, mean: f64) -> PyResult<f64> {
    log_gamma_pdf(py, x, shape, mean)
}

#[pyfunction]
#[pyo3(name = "_seq_log_beta_pdf")]
fn log_beta_pdf_py(py: Python<'_>, x: f64, alpha: f64, beta: f64) -> PyResult<f64> {
    log_beta_pdf(py, x, alpha, beta)
}

#[pyfunction]
#[pyo3(name = "_seq_log_prior_seq")]
fn log_prior_seq_py(py: Python<'_>, species_net: &Bound<'_, Network>, theta: f64, priors: &Bound<'_, PyAny>) -> PyResult<f64> {
    log_prior_seq_impl(py, species_net, theta, &Priors::from_py(priors)?)
}

#[pyfunction]
#[pyo3(name = "_seq_sync_lengths")]
fn sync_lengths_py(py: Python<'_>, net: &Bound<'_, Network>, heights: &Bound<'_, PyAny>) -> PyResult<()> {
    let h = heights_from_py(py, &net.borrow(), heights)?;
    sync_lengths(py, net, &h)
}

#[pyfunction]
#[pyo3(name = "_seq_descendant_species")]
fn descendant_species_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, species_of: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let n = net.borrow();
    let map = descendant_species(py, &n, &species_map(species_of)?);
    let out = PyDict::new(py);
    for (uid, entry) in n.nodes.iter() {
        if let Some(s) = map.get(uid) {
            out.set_item(entry.obj.bind(py), labelset_to_py(py, s)?)?;
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_seq_enforce_embedding_consistency", signature = (species_net, net_heights, gene_trees, gt_heights, species_of, eps=0.05))]
fn enforce_embedding_consistency_py(
    py: Python<'_>,
    species_net: &Bound<'_, Network>,
    net_heights: &Bound<'_, PyAny>,
    gene_trees: &Bound<'_, PyAny>,
    gt_heights: &Bound<'_, PyAny>,
    species_of: &Bound<'_, PyAny>,
    eps: f64,
) -> PyResult<f64> {
    enforce_embedding_consistency(py, species_net, net_heights, gene_trees, gt_heights, species_of, eps)
}

#[pyfunction]
#[pyo3(name = "_seq_jc_distance")]
fn jc_distance_py(s1: &str, s2: &str) -> f64 {
    jc_distance(s1, s2)
}

#[pyfunction]
#[pyo3(name = "_seq_build_upgma_gene_tree")]
fn build_upgma_gene_tree_py<'py>(py: Python<'py>, alignment: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    build_upgma_gene_tree(py, alignment)
}

#[pyfunction]
#[pyo3(name = "_seq_internal_nodes")]
fn internal_nodes_py<'py>(py: Python<'py>, net: &Bound<'py, Network>) -> PyResult<Bound<'py, PyList>> {
    let n = net.borrow();
    let nodes: Vec<Py<Node>> = internal_nodes(&n).into_iter().filter_map(|u| node_py(py, &n, u)).collect();
    PyList::new(py, nodes)
}

#[pyfunction]
#[pyo3(name = "_seq_slide_height_move")]
fn slide_height_move_py(
    py: Python<'_>,
    net: &Bound<'_, Network>,
    heights: &Bound<'_, PyAny>,
    rng: Bound<'_, PyAny>,
    root_scale_window: f64,
) -> PyResult<Option<f64>> {
    let n = net.borrow();
    let mut h = heights_from_py(py, &n, heights)?;
    let before = h.clone();
    let r = slide_height_move(py, &n, &mut h, &Rng(rng), root_scale_window)?;
    let d = heights.cast::<PyDict>()?;
    for (uid, v) in &h {
        if before.get(uid) != Some(v) {
            if let Some(obj) = node_py(py, &n, *uid) {
                d.set_item(obj, *v)?;
            }
        }
    }
    Ok(r)
}

#[pyfunction]
#[pyo3(name = "_seq_op_change_theta", signature = (state, rng, window=0.4))]
fn op_change_theta_py(state: Bound<'_, PyAny>, rng: Bound<'_, PyAny>, window: f64) -> OpResult {
    op_change_theta(&St::new(state), &Rng(rng), window)
}

#[pyfunction]
#[pyo3(name = "_seq_op_change_gamma", signature = (state, rng, window=0.2))]
fn op_change_gamma_py(state: Bound<'_, PyAny>, rng: Bound<'_, PyAny>, window: f64) -> OpResult {
    op_change_gamma(&St::new(state), &Rng(rng), window)
}

#[pyfunction]
#[pyo3(name = "_seq_op_net_node_height", signature = (state, rng, root_scale_window=0.3))]
fn op_net_node_height_py(state: Bound<'_, PyAny>, rng: Bound<'_, PyAny>, root_scale_window: f64) -> OpResult {
    op_net_node_height(&St::new(state), &Rng(rng), root_scale_window)
}

#[pyfunction]
#[pyo3(name = "_seq_op_gene_node_height", signature = (state, rng, root_scale_window=0.3))]
fn op_gene_node_height_py(state: Bound<'_, PyAny>, rng: Bound<'_, PyAny>, root_scale_window: f64) -> OpResult {
    op_gene_node_height(&St::new(state), &Rng(rng), root_scale_window)
}

#[pyfunction]
#[pyo3(name = "_seq_op_gene_tree_nni")]
fn op_gene_tree_nni_py(state: Bound<'_, PyAny>, rng: Bound<'_, PyAny>) -> OpResult {
    op_gene_tree_nni(&St::new(state), &Rng(rng))
}

#[pyfunction]
#[pyo3(name = "_seq_split_edge")]
fn split_edge_py(
    py: Python<'_>,
    net: &Bound<'_, Network>,
    heights: &Bound<'_, PyAny>,
    parent: &Bound<'_, PyAny>,
    child: &Bound<'_, PyAny>,
    t: f64,
) -> PyResult<Py<Node>> {
    let (p, c) = {
        let n = net.borrow();
        (node_arg(&n, parent)?, node_arg(&n, child)?)
    };
    let mut h = Heights::default();
    let mid = split_edge(py, net, &mut h, p, c, t)?;
    let n = net.borrow();
    let obj = node_py(py, &n, mid).ok_or_else(|| PyKeyError::new_err("mid"))?;
    heights.cast::<PyDict>()?.set_item(obj.bind(py), t)?;
    Ok(obj)
}

#[pyfunction]
#[pyo3(name = "_seq_has_parallel_edges")]
fn has_parallel_edges_py(net: &Bound<'_, Network>) -> bool {
    has_parallel_edges(&net.borrow())
}

#[pyfunction]
#[pyo3(name = "_seq_logsumexp")]
fn logsumexp_py(values: Vec<f64>) -> f64 {
    logsumexp_ref(&values)
}

#[pyfunction]
#[pyo3(name = "_seq_network_only_log_score")]
fn network_only_log_score_py(state: Bound<'_, PyAny>, net: &Bound<'_, Network>) -> PyResult<f64> {
    network_only_log_score(&St::new(state), net)
}

#[pyfunction]
#[pyo3(name = "_seq_find_edge_by_labels")]
fn find_edge_by_labels_py(py: Python<'_>, net: &Bound<'_, Network>, src_label: &str, dest_label: &str) -> Option<Py<Edge>> {
    let n = net.borrow();
    find_edge_by_labels(py, &n, src_label, dest_label).and_then(|(e, _, _)| n.edges.get(&e).map(|x| x.obj.clone_ref(py)))
}

#[pyfunction]
#[pyo3(name = "_seq_add_reticulation_placements")]
fn add_reticulation_placements_py<'py>(
    py: Python<'py>,
    state: Bound<'py, PyAny>,
    net: &Bound<'py, Network>,
    heights: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let h = heights_from_py(py, &net.borrow(), heights)?;
    let placements = add_reticulation_placements(&St::new(state), net, &h)?;
    placements_to_py(py, &placements)
}

#[pyfunction]
#[pyo3(name = "_seq_geometric_placements")]
fn geometric_placements_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, heights: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let n = net.borrow();
    let h = heights_from_py(py, &n, heights)?;
    let out = PyList::empty(py);
    for ((d3, d4), (r5, r6)) in geometric_placements(py, &n, &h)? {
        let d = PyDict::new(py);
        d.set_item("donor_labels", (label_of(py, &n, d3), label_of(py, &n, d4)))?;
        d.set_item("retic_labels", (label_of(py, &n, r5), label_of(py, &n, r6)))?;
        out.append(d)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_seq_op_add_reticulation")]
fn op_add_reticulation_py(state: Bound<'_, PyAny>, rng: Bound<'_, PyAny>) -> OpResult {
    add_reticulation_generic(&St::new(state), &Rng(rng), false, false)
}

#[pyfunction]
#[pyo3(name = "_seq_op_delete_reticulation")]
fn op_delete_reticulation_py(state: Bound<'_, PyAny>, rng: Bound<'_, PyAny>) -> OpResult {
    delete_reticulation_generic(&St::new(state), &Rng(rng), false)
}

#[pyfunction]
#[pyo3(name = "_seq_op_add_reticulation_decoupled")]
fn op_add_reticulation_decoupled_py(state: Bound<'_, PyAny>, rng: Bound<'_, PyAny>) -> OpResult {
    add_reticulation_generic(&St::new(state), &Rng(rng), true, false)
}

#[pyfunction]
#[pyo3(name = "_seq_op_delete_reticulation_decoupled")]
fn op_delete_reticulation_decoupled_py(state: Bound<'_, PyAny>, rng: Bound<'_, PyAny>) -> OpResult {
    delete_reticulation_generic(&St::new(state), &Rng(rng), false)
}

#[pyfunction]
#[pyo3(name = "_seq_op_add_reticulation_coupled")]
fn op_add_reticulation_coupled_py(state: Bound<'_, PyAny>, rng: Bound<'_, PyAny>) -> OpResult {
    add_reticulation_generic(&St::new(state), &Rng(rng), true, true)
}

#[pyfunction]
#[pyo3(name = "_seq_op_delete_reticulation_coupled")]
fn op_delete_reticulation_coupled_py(state: Bound<'_, PyAny>, rng: Bound<'_, PyAny>) -> OpResult {
    delete_reticulation_generic(&St::new(state), &Rng(rng), true)
}

#[pyfunction]
#[pyo3(name = "_seq_op_relocate_reticulation_coupled")]
fn op_relocate_reticulation_coupled_py(state: Bound<'_, PyAny>, rng: Bound<'_, PyAny>) -> OpResult {
    op_relocate_reticulation_coupled(&St::new(state), &Rng(rng))
}

#[pyfunction]
#[pyo3(name = "_seq_suppress_degree2")]
fn suppress_degree2_py(py: Python<'_>, net: &Bound<'_, Network>, heights: &Bound<'_, PyAny>, node: &Bound<'_, PyAny>) -> PyResult<usize> {
    let uid = {
        let n = net.borrow();
        match n.uid_of_any(node) {
            Some(u) => u,
            None => return Ok(n.edges.len()),
        }
    };
    let d = heights.cast::<PyDict>()?;
    let mut h = Heights::default();
    let count = suppress_degree2(py, net, &mut h, uid)?;
    // ``heights.pop(node, None)``
    if d.contains(node)? {
        d.del_item(node)?;
    }
    Ok(count)
}

#[pyfunction]
#[pyo3(name = "_seq_leaf_descendants")]
fn leaf_descendants_py<'py>(py: Python<'py>, net: &Bound<'py, Network>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyFrozenSet>> {
    let n = net.borrow();
    let set = match n.uid_of_any(node) {
        Some(u) => leaf_desc_labels(py, &n, u),
        None => LabelSet::new(),
    };
    labelset_to_py(py, &set)
}

#[pyfunction]
#[pyo3(name = "_seq_gt_signature")]
fn gt_signature_py<'py>(py: Python<'py>, gt: &Bound<'py, Network>, heights: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyFrozenSet>> {
    let n = gt.borrow();
    let h = heights_from_py(py, &n, heights)?;
    sig_to_py(py, &gt_signature(py, &n, &h)?)
}

#[pyfunction]
#[pyo3(name = "_seq_node_by_leafset")]
fn node_by_leafset_py(py: Python<'_>, net: &Bound<'_, Network>, target: &Bound<'_, PyAny>) -> PyResult<Option<Py<Node>>> {
    let mut set = LabelSet::new();
    for item in target.try_iter()? {
        set.insert(item?.extract::<String>()?);
    }
    let n = net.borrow();
    Ok(node_by_leafset(py, &n, &set).and_then(|u| node_py(py, &n, u)))
}

#[pyfunction]
#[pyo3(name = "_seq_gene_tree_nni_neighbors")]
fn gene_tree_nni_neighbors_py<'py>(py: Python<'py>, gt: &Bound<'py, Network>, heights: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let h = heights_from_py(py, &gt.borrow(), heights)?;
    let out = PyList::empty(py);
    for (cand, ch) in gene_tree_nni_neighbors(py, gt, &h)? {
        out.append(pair_to_py(py, &cand, &ch)?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_seq_scaled_gene_tree")]
fn scaled_gene_tree_py<'py>(py: Python<'py>, gt: &Bound<'py, Network>, s: f64) -> PyResult<Bound<'py, PyTuple>> {
    let (cand, ch) = scaled_gene_tree(py, gt, s)?;
    pair_to_py(py, &cand, &ch)
}

#[pyfunction]
#[pyo3(name = "_seq_candidate_set")]
fn candidate_set_py<'py>(py: Python<'py>, gt: &Bound<'py, Network>, heights: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let h = heights_from_py(py, &gt.borrow(), heights)?;
    let out = PyList::empty(py);
    for (cand, ch) in candidate_set(py, gt, &h)? {
        out.append(pair_to_py(py, &cand, &ch)?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "_seq_candidate_logweights")]
fn candidate_logweights_py<'py>(
    py: Python<'py>,
    state: Bound<'py, PyAny>,
    i: usize,
    net_idx: &Bound<'py, PyAny>,
    sp_heights: &Bound<'py, PyAny>,
    theta: f64,
    candidates: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let st = St::new(state);
    let eng = st.engine()?;
    let calc = eng.getattr("_calcs")?.get_item(i)?;
    let model = st.get("model")?;
    let species_of = eng.getattr("_species_of")?;
    let branch_thetas = st.get("branch_thetas")?;
    let msnc = msnc_module(py)?;
    let build_gt = msnc.getattr("build_gene_tree_msnc_index")?;
    let prebuilt = msnc.getattr("msnc_log_density_prebuilt")?;
    let mut logw: Vec<f64> = Vec::new();
    for item in candidates.try_iter()? {
        let item = item?;
        let gt = item.get_item(0)?;
        let mut w = f64::NEG_INFINITY;
        let fels: f64 = calc.call_method1("log_likelihood", (&gt, &model))?.extract()?;
        if fels.is_finite() {
            let pair = build_gt.call1((&gt, &species_of))?;
            let msnc_d: f64 = prebuilt
                .call1((net_idx, sp_heights, pair.get_item(0)?, pair.get_item(1)?, theta, &branch_thetas))?
                .extract()?;
            if msnc_d.is_finite() {
                w = fels + msnc_d;
            }
        }
        logw.push(w);
    }
    Ok(PyArray1::from_vec(py, logw))
}

/// The ``rebuild(k)`` callable ``_enumerate_scored_candidates`` returns.
#[pyclass(name = "_SeqRebuild", module = "phynetpy._mcmc_seq")]
pub struct SeqRebuild {
    bases: Vec<Py<Network>>,
    desc: Vec<(usize, f64)>,
}

#[pymethods]
impl SeqRebuild {
    fn __call__<'py>(&self, py: Python<'py>, k: usize) -> PyResult<Bound<'py, PyTuple>> {
        let (bi, s) = self.desc.get(k).copied().ok_or_else(|| PyIndexError::new_err("list index out of range"))?;
        let base = self.bases[bi].bind(py);
        let (cg, _) = clone_net(base)?;
        let mut ch = node_heights_uid(py, &cg.borrow())?;
        if s != 1.0 {
            ch = scaled(&ch, s);
        }
        sync_lengths(py, &cg, &ch)?;
        pair_to_py(py, &cg, &ch)
    }
}

#[pyfunction]
#[pyo3(name = "_seq_enumerate_scored_candidates")]
#[allow(clippy::too_many_arguments)]
fn enumerate_scored_candidates_py<'py>(
    py: Python<'py>,
    state: Bound<'py, PyAny>,
    i: usize,
    net_idx: &Bound<'py, PyAny>,
    sp_heights: &Bound<'py, PyAny>,
    theta: f64,
    gt: &Bound<'py, Network>,
    heights: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let st = St::new(state);
    let h = heights_from_py(py, &gt.borrow(), heights)?;
    let e = enumerate_scored_candidates(&st, i, net_idx, sp_heights, theta, gt, &h)?;
    let sigs: Vec<Bound<'py, PyFrozenSet>> = e.sigs.iter().map(|s| sig_to_py(py, s)).collect::<PyResult<_>>()?;
    let sigs = PyList::new(py, sigs)?;
    let logw = PyArray1::from_vec(py, e.logw.clone());
    let rebuild = Py::new(
        py,
        SeqRebuild {
            bases: e.bases.iter().map(|b| b.clone().unbind()).collect(),
            desc: e.desc.clone(),
        },
    )?;
    PyTuple::new(py, [sigs.into_any(), logw.into_any(), rebuild.into_bound(py).into_any()])
}

#[pyfunction]
#[pyo3(name = "_seq_coupled_gene_tree_reproposal")]
fn coupled_gene_tree_reproposal_py<'py>(
    py: Python<'py>,
    state: Bound<'py, PyAny>,
    target_net: &Bound<'py, Network>,
    reverse_net: &Bound<'py, Network>,
    rng: Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let st = St::new(state);
    let Some(r) = coupled_gene_tree_reproposal(&st, target_net, reverse_net, &Rng(rng))? else {
        return Ok(None);
    };
    let nets = PyList::new(py, r.gene_trees.iter().map(|g| g.as_any()))?;
    let mut dicts = Vec::with_capacity(r.heights.len());
    for (g, h) in r.gene_trees.iter().zip(&r.heights) {
        dicts.push(heights_to_py(py, &g.borrow(), h)?);
    }
    let hs = PyList::new(py, dicts)?;
    Ok(Some(PyTuple::new(
        py,
        [
            nets.into_any(),
            hs.into_any(),
            r.log_qf.into_pyobject(py)?.into_any(),
            r.log_qr.into_pyobject(py)?.into_any(),
        ],
    )?))
}

#[pyfunction]
#[pyo3(name = "_seq_relocation_loghr")]
fn relocation_loghr_py(l2_new: f64, win_new: f64, l2_old: f64, win_old: f64, log_qf: f64, log_qr: f64) -> f64 {
    relocation_loghr(l2_new, win_new, l2_old, win_old, log_qf, log_qr)
}

#[pyfunction]
#[pyo3(name = "_seq_topology_signature")]
fn topology_signature_py<'py>(py: Python<'py>, newick: &str) -> PyResult<Option<Bound<'py, PyTuple>>> {
    match topology_signature(py, newick)? {
        Some(sig) => Ok(Some(topo_sig_to_py(py, &sig)?)),
        None => Ok(None),
    }
}

#[pyfunction]
#[pyo3(name = "_seq_count_free_parameters")]
fn count_free_parameters_py(num_leaves: i64, num_reticulations: i64) -> i64 {
    count_free_parameters(num_leaves, num_reticulations)
}

#[pyfunction]
#[pyo3(name = "_seq_information_criteria", signature = (log_likelihood, k, n=None))]
fn information_criteria_py<'py>(py: Python<'py>, log_likelihood: f64, k: i64, n: Option<i64>) -> PyResult<Bound<'py, PyDict>> {
    information_criteria(py, log_likelihood, k, n)
}

#[pyfunction]
#[pyo3(name = "_seq_reticulation_posterior")]
fn reticulation_posterior_py<'py>(py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    reticulation_posterior(py, samples)
}

#[pyfunction]
#[pyo3(name = "_seq_topology_posterior", signature = (samples, top_n=None))]
fn topology_posterior_py(py: Python<'_>, samples: &Bound<'_, PyAny>, top_n: Option<i64>) -> PyResult<Vec<(String, f64)>> {
    topology_posterior(py, samples, top_n)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SeqUndo>()?;
    m.add_class::<SeqRebuild>()?;
    m.add_function(wrap_pyfunction!(log_poisson_pmf_py, m)?)?;
    m.add_function(wrap_pyfunction!(log_gamma_pdf_py, m)?)?;
    m.add_function(wrap_pyfunction!(log_beta_pdf_py, m)?)?;
    m.add_function(wrap_pyfunction!(log_prior_seq_py, m)?)?;
    m.add_function(wrap_pyfunction!(sync_lengths_py, m)?)?;
    m.add_function(wrap_pyfunction!(descendant_species_py, m)?)?;
    m.add_function(wrap_pyfunction!(enforce_embedding_consistency_py, m)?)?;
    m.add_function(wrap_pyfunction!(jc_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_upgma_gene_tree_py, m)?)?;
    m.add_function(wrap_pyfunction!(internal_nodes_py, m)?)?;
    m.add_function(wrap_pyfunction!(slide_height_move_py, m)?)?;
    m.add_function(wrap_pyfunction!(op_change_theta_py, m)?)?;
    m.add_function(wrap_pyfunction!(op_change_gamma_py, m)?)?;
    m.add_function(wrap_pyfunction!(op_net_node_height_py, m)?)?;
    m.add_function(wrap_pyfunction!(op_gene_node_height_py, m)?)?;
    m.add_function(wrap_pyfunction!(op_gene_tree_nni_py, m)?)?;
    m.add_function(wrap_pyfunction!(split_edge_py, m)?)?;
    m.add_function(wrap_pyfunction!(has_parallel_edges_py, m)?)?;
    m.add_function(wrap_pyfunction!(logsumexp_py, m)?)?;
    m.add_function(wrap_pyfunction!(network_only_log_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(find_edge_by_labels_py, m)?)?;
    m.add_function(wrap_pyfunction!(add_reticulation_placements_py, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_placements_py, m)?)?;
    m.add_function(wrap_pyfunction!(op_add_reticulation_py, m)?)?;
    m.add_function(wrap_pyfunction!(op_delete_reticulation_py, m)?)?;
    m.add_function(wrap_pyfunction!(op_add_reticulation_decoupled_py, m)?)?;
    m.add_function(wrap_pyfunction!(op_delete_reticulation_decoupled_py, m)?)?;
    m.add_function(wrap_pyfunction!(op_add_reticulation_coupled_py, m)?)?;
    m.add_function(wrap_pyfunction!(op_delete_reticulation_coupled_py, m)?)?;
    m.add_function(wrap_pyfunction!(op_relocate_reticulation_coupled_py, m)?)?;
    m.add_function(wrap_pyfunction!(suppress_degree2_py, m)?)?;
    m.add_function(wrap_pyfunction!(leaf_descendants_py, m)?)?;
    m.add_function(wrap_pyfunction!(gt_signature_py, m)?)?;
    m.add_function(wrap_pyfunction!(node_by_leafset_py, m)?)?;
    m.add_function(wrap_pyfunction!(gene_tree_nni_neighbors_py, m)?)?;
    m.add_function(wrap_pyfunction!(scaled_gene_tree_py, m)?)?;
    m.add_function(wrap_pyfunction!(candidate_set_py, m)?)?;
    m.add_function(wrap_pyfunction!(candidate_logweights_py, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_scored_candidates_py, m)?)?;
    m.add_function(wrap_pyfunction!(coupled_gene_tree_reproposal_py, m)?)?;
    m.add_function(wrap_pyfunction!(relocation_loghr_py, m)?)?;
    m.add_function(wrap_pyfunction!(topology_signature_py, m)?)?;
    m.add_function(wrap_pyfunction!(count_free_parameters_py, m)?)?;
    m.add_function(wrap_pyfunction!(information_criteria_py, m)?)?;
    m.add_function(wrap_pyfunction!(reticulation_posterior_py, m)?)?;
    m.add_function(wrap_pyfunction!(topology_posterior_py, m)?)?;
    Ok(())
}
