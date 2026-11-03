//! MCMC_GT kernels: the computational half of ``phynetpy._mcmc_gt``.
//!
//! * the network prior (``_log_exp_pdf``, ``_log_beta_pdf``,
//!   ``_log_poisson_pmf``, ``log_prior_network``), evaluated with CPython's
//!   own ``lgamma`` algorithm so the densities agree with the Python
//!   reference to the last bit;
//! * ``_DisplayedTree``: one displayed tree of a species network, built by
//!   edge contraction, together with the int-indexed view the species-tree
//!   MSC DP consumes;
//! * ``_GTLikelihoodEngine``: the MSNC gene-tree likelihood engine (a
//!   ``MSCBranchKernel`` subclass) with its network / gene-tree indices,
//!   the signature-keyed score memo and the dirty-set invalidation
//!   protocol;
//! * structural helpers (``_is_valid_network``,
//!   ``_populate_default_branch_lengths``, ``_safe_add``) and the adaptive
//!   kernel's Robbins-Monro / weight-scaler / SPR-decay arithmetic.
//!
//! The chain driver, the proposal kernel's dispatch over move classes, the
//! result containers and the multiprocessing pool stay in Python.  No
//! random numbers are drawn here: the kernel's draws from its
//! ``numpy.random.Generator`` remain in the Python ``generate`` method.

use std::collections::VecDeque;
use std::f64::consts::PI;
use std::hash::Hash;

use indexmap::IndexMap;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyOverflowError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::graph::{Edge, Network, Node};
use crate::kernels::common::LOG_FLOOR;
use crate::kernels::msnc::{
    msc_log_prob_tree_py, msnc_log_prob_network_py, DisplayedTreeIndex, GeneTreeIndex,
    MSCBranchKernel, NetworkIndex,
};
use crate::util::network_error;

type FxIndexMap<K, V> = IndexMap<K, V, FxBuildHasher>;

// ═══════════════════════════════════════════════════════════════════════
// lgamma, exactly as CPython's ``math.lgamma`` computes it
// ═══════════════════════════════════════════════════════════════════════

const LANCZOS_N: usize = 13;
const LANCZOS_G: f64 = 6.024680040776729583740234375;
const LANCZOS_NUM: [f64; LANCZOS_N] = [
    23531376880.410759688572007674451636754734846804940,
    42919803642.649098768957899047001988850926355848959,
    35711959237.355668049440185451547166705960488635843,
    17921034426.037209699919755754458931112671403265390,
    6039542586.3520280050642916443072979210699388420708,
    1439720407.3117216736632230727949123939715485786772,
    248874557.86205415651146038641322942321632125127801,
    31426415.585400194380614231628318205362874684987640,
    2876370.6289353724412254090516208496135991145378768,
    186056.26539522349504029498971604569928220784236328,
    8071.6720023658162106380029022722506138218516325024,
    210.82427775157934587250973392071336271166969580291,
    2.5066282746310002701649081771338373386264310793408,
];
const LANCZOS_DEN: [f64; LANCZOS_N] = [
    0.0, 39916800.0, 120543840.0, 150917976.0, 105258076.0, 45995730.0, 13339535.0, 2637558.0,
    357423.0, 32670.0, 1925.0, 66.0, 1.0,
];
const LOG_PI: f64 = 1.144729885849400174143427351353058711647;

fn lanczos_sum(x: f64) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    if x < 5.0 {
        for i in (0..LANCZOS_N).rev() {
            num = num * x + LANCZOS_NUM[i];
            den = den * x + LANCZOS_DEN[i];
        }
    } else {
        for i in 0..LANCZOS_N {
            num = num / x + LANCZOS_NUM[i];
            den = den / x + LANCZOS_DEN[i];
        }
    }
    num / den
}

fn sinpi(x: f64) -> f64 {
    let y = x.abs() % 2.0;
    let n = (2.0 * y).round() as i32;
    let r = match n {
        0 => (PI * y).sin(),
        1 => (PI * (y - 0.5)).cos(),
        2 => (PI * (1.0 - y)).sin(),
        3 => -(PI * (y - 1.5)).cos(),
        _ => (PI * (y - 2.0)).sin(),
    };
    1.0f64.copysign(x) * r
}

/// ``math.lgamma(x)`` with CPython's Lanczos evaluation and its error cases.
pub fn lgamma(x: f64) -> PyResult<f64> {
    if !x.is_finite() {
        return Ok(if x.is_nan() { x } else { f64::INFINITY });
    }
    if x == x.floor() && x <= 2.0 {
        if x <= 0.0 {
            return Err(PyValueError::new_err("math domain error"));
        }
        return Ok(0.0);
    }
    let absx = x.abs();
    if absx < 1e-20 {
        return Ok(-absx.ln());
    }
    let mut r = lanczos_sum(absx).ln() - LANCZOS_G;
    r += (absx - 0.5) * ((absx + LANCZOS_G - 0.5).ln() - 1.0);
    if x < 0.0 {
        r = LOG_PI - sinpi(absx).abs().ln() - absx.ln() - r;
    }
    if r.is_infinite() {
        return Err(PyOverflowError::new_err("math range error"));
    }
    Ok(r)
}

/// ``math.log(v)``: the domain error for non-positive finite arguments.
fn py_log(v: f64) -> PyResult<f64> {
    if v.is_nan() || v > 0.0 {
        Ok(v.ln())
    } else {
        Err(PyValueError::new_err("math domain error"))
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Priors
// ═══════════════════════════════════════════════════════════════════════

/// Log-density of ``Exp(rate)`` at ``x``; ``LOG_FLOOR`` off the support.
pub fn log_exp_pdf(x: Option<f64>, rate: f64) -> PyResult<f64> {
    match x {
        Some(v) if v.is_finite() && v > 0.0 => Ok(py_log(rate)? - rate * v),
        _ => Ok(LOG_FLOOR),
    }
}

/// Log-density of ``Beta(alpha, beta)`` at ``x``; ``LOG_FLOOR`` outside (0, 1).
pub fn log_beta_pdf(x: Option<f64>, alpha: f64, beta: f64) -> PyResult<f64> {
    let v = match x {
        Some(v) if v.is_finite() && v > 0.0 && v < 1.0 => v,
        _ => return Ok(LOG_FLOOR),
    };
    let log_beta_fn = lgamma(alpha)? + lgamma(beta)? - lgamma(alpha + beta)?;
    Ok((alpha - 1.0) * v.ln() + (beta - 1.0) * (1.0 - v).ln() - log_beta_fn)
}

/// Log-pmf of ``Poisson(mean)`` at ``k``.
pub fn log_poisson_pmf(k: f64, mean: f64) -> PyResult<f64> {
    if k < 0.0 || mean <= 0.0 {
        return Ok(LOG_FLOOR);
    }
    Ok(k * mean.ln() - mean - lgamma(k + 1.0)?)
}

#[pyfunction]
#[pyo3(name = "_gt_lgamma")]
fn lgamma_py(x: f64) -> PyResult<f64> {
    lgamma(x)
}

#[pyfunction]
#[pyo3(name = "_gt_log_exp_pdf")]
fn log_exp_pdf_py(x: Option<f64>, rate: f64) -> PyResult<f64> {
    log_exp_pdf(x, rate)
}

#[pyfunction]
#[pyo3(name = "_gt_log_beta_pdf")]
fn log_beta_pdf_py(x: Option<f64>, alpha: f64, beta: f64) -> PyResult<f64> {
    log_beta_pdf(x, alpha, beta)
}

#[pyfunction]
#[pyo3(name = "_gt_log_poisson_pmf")]
fn log_poisson_pmf_py(k: f64, mean: f64) -> PyResult<f64> {
    log_poisson_pmf(k, mean)
}

/// ``log_prior_network(net, priors)``: topology term, ``Exp`` on every
/// branch length (edge order), ``Beta`` on one gamma per reticulation
/// (node order), ``Poisson`` on the reticulation count -- summed in that
/// order.
#[pyfunction]
#[pyo3(name = "_gt_log_prior_network")]
fn log_prior_network_py(net: &Bound<'_, PyAny>, priors: &Bound<'_, PyAny>) -> PyResult<f64> {
    let py = net.py();
    let mut total = 0.0;

    let topo_fn = priors.getattr("topology_prior_fn")?;
    let topo_weight: f64 = priors.getattr("topology_weight")?.extract()?;
    if !topo_fn.is_none() && topo_weight != 0.0 {
        let v: f64 = topo_fn.call1((net,))?.extract()?;
        total += topo_weight * v;
    }
    let rate: f64 = priors.getattr("branch_length_rate")?.extract()?;
    let alpha: f64 = priors.getattr("gamma_alpha")?.extract()?;
    let beta: f64 = priors.getattr("gamma_beta")?.extract()?;
    let mean: f64 = priors.getattr("retic_count_mean")?.extract()?;

    let network = net.cast::<Network>()?;
    let n = network.try_borrow()?;
    for entry in n.edges.values() {
        let length = entry.obj.borrow(py).length;
        total += log_exp_pdf(length, rate)?;
    }
    let mut retic_count = 0usize;
    for (&uid, entry) in n.nodes.iter() {
        if !entry.obj.borrow(py).is_retic {
            continue;
        }
        retic_count += 1;
        let parents = n.parents(uid);
        if parents.len() != 2 {
            continue;
        }
        let e1 = n.edge_between(py, parents[0], uid, None, None)?;
        let gamma = e1.borrow(py).gamma.unwrap_or(0.5);
        total += log_beta_pdf(Some(gamma), alpha, beta)?;
    }
    total += log_poisson_pmf(retic_count as f64, mean)?;
    Ok(total)
}

// ═══════════════════════════════════════════════════════════════════════
// Small helpers
// ═══════════════════════════════════════════════════════════════════════

/// Add two possibly-missing branch lengths; ``None`` absorbs.
#[pyfunction]
#[pyo3(name = "_gt_safe_add")]
fn safe_add_py(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a + b),
        _ => None,
    }
}

/// ``GraphUtils._valid_network_degrees``: one root (out >= 2), leaves with
/// in-degree 1, reticulations in 2 / out 1, other internals in 1 / out >= 2.
fn valid_network_degrees(py: Python<'_>, n: &Network) -> bool {
    let mut root_count = 0usize;
    for (uid, entry) in n.nodes.iter() {
        let indeg = n.in_deg(*uid);
        let outdeg = n.out_deg(*uid);
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

/// Structural-validity gate: degree invariants plus acyclicity.
#[pyfunction]
#[pyo3(name = "_gt_is_valid_network")]
fn is_valid_network_py(py: Python<'_>, net: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
    let Some(net) = net else { return Ok(false) };
    if net.is_none() {
        return Ok(false);
    }
    let net = net.cast::<Network>()?;
    let n = net.try_borrow()?;
    if !valid_network_degrees(py, &n) {
        return Ok(false);
    }
    Ok(n.is_acyclic_impl())
}

/// Fill missing / non-positive branch lengths with ``default`` and missing
/// gammas on reticulation in-edges with 0.5.
#[pyfunction]
#[pyo3(name = "_gt_populate_default_branch_lengths", signature = (net, default=1.0))]
fn populate_default_branch_lengths_py(net: &Bound<'_, Network>, default: f64) -> PyResult<()> {
    let py = net.py();
    let edges: Vec<Py<Edge>> = net
        .try_borrow()?
        .edges
        .values()
        .map(|e| e.obj.clone_ref(py))
        .collect();
    for e in edges {
        let b = e.bind(py);
        let (length, gamma, dest_retic) = {
            let r = b.try_borrow()?;
            let dest_retic = r.dest.borrow(py).is_retic;
            (r.length, r.gamma, dest_retic)
        };
        let needs_length = match length {
            None => true,
            Some(x) => x <= 0.0,
        };
        if needs_length {
            Edge::set_length(b, Some(default), false, false)?;
        }
        if gamma.is_none() && dest_retic {
            Edge::set_gamma(b, 0.5)?;
        }
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// Adaptive-kernel arithmetic
// ═══════════════════════════════════════════════════════════════════════

/// One Robbins-Monro step of ``log(sigma)`` toward the target acceptance.
#[pyfunction]
#[pyo3(name = "_gt_tune_sigma")]
#[allow(clippy::too_many_arguments)]
fn tune_sigma_py(
    current: f64,
    n_obs: i64,
    accepted: &Bound<'_, PyAny>,
    sigma_min: f64,
    sigma_max: f64,
    target: f64,
    delay: f64,
    exponent: f64,
    shift: f64,
) -> PyResult<f64> {
    if (n_obs as f64) < delay {
        return Ok(current);
    }
    let step = 1.0 / (n_obs as f64 + shift).powf(exponent);
    let indicator = if accepted.is_truthy()? { 1.0 } else { 0.0 };
    let log_adjust = step * (indicator - target);
    let new_sigma = current * log_adjust.exp();
    Ok(sigma_min.max(sigma_max.min(new_sigma)))
}

/// Scale base weights by the accept / improve blend of each move's
/// sliding window; moves with fewer than ``warmup`` observations keep
/// their base weight.
#[pyfunction]
#[pyo3(name = "_gt_adapt_weights")]
#[allow(clippy::too_many_arguments)]
fn adapt_weights_py<'py>(
    base_weights: Vec<f64>,
    histories: Vec<Vec<(Bound<'py, PyAny>, f64)>>,
    warmup: usize,
    min_weight: f64,
    acc_weight: f64,
    eff_weight: f64,
    eps: f64,
) -> PyResult<Vec<f64>> {
    let floor = min_weight;
    let mut out = Vec::with_capacity(base_weights.len());
    for (base_w, hist) in base_weights.iter().zip(histories.iter()) {
        if hist.len() < warmup {
            out.push(*base_w);
            continue;
        }
        let n = hist.len() as f64;
        let mut n_acc = 0usize;
        let mut n_imp = 0usize;
        for (accepted, delta) in hist {
            if accepted.is_truthy()? {
                n_acc += 1;
                if *delta > 0.0 {
                    n_imp += 1;
                }
            }
        }
        let acc_rate = n_acc as f64 / n;
        let imp_rate = n_imp as f64 / n;
        let efficiency = (imp_rate / acc_rate.max(eps)).min(1.0);
        let composite = acc_weight * acc_rate + eff_weight * efficiency;
        let scale = floor + (2.0 - floor) * composite;
        out.push(base_w * scale);
    }
    Ok(out)
}

/// SPR ``distance_decay`` for the current stagnation level.
#[pyfunction]
#[pyo3(name = "_gt_spr_decay")]
fn spr_decay_py(adaptive: bool, stagnation: f64, phase_patience: f64, decay_max: f64, decay_min: f64) -> f64 {
    if !adaptive || phase_patience <= 0.0 {
        return decay_max;
    }
    let stuckness = (stagnation / phase_patience).min(1.0);
    let span = decay_max - decay_min;
    decay_max - stuckness * span
}

// ═══════════════════════════════════════════════════════════════════════
// Displayed trees
// ═══════════════════════════════════════════════════════════════════════

/// Merge degree-2 (in 1, out 1) chains into single edges, summing lengths.
/// Works in place; insertion orders follow the reference's dict semantics.
fn collapse_passthroughs_core<K: Copy + Eq + Hash>(
    root: K,
    children: &mut FxIndexMap<K, Vec<K>>,
    parent: &mut FxIndexMap<K, K>,
    length: &mut FxIndexMap<(K, K), Option<f64>>,
) -> K {
    let mut changed = true;
    while changed {
        changed = false;
        let interior: Vec<K> = children
            .keys()
            .copied()
            .filter(|&k| {
                k != root
                    && children.get(&k).map(|v| v.len()).unwrap_or(0) == 1
                    && parent.contains_key(&k)
            })
            .collect();
        for n in interior {
            let Some(&p) = parent.get(&n) else { continue };
            let child = match children.get(&n) {
                Some(kids) if kids.len() == 1 => kids[0],
                _ => continue,
            };
            let top_len = length.get(&(p, n)).copied().flatten();
            let bot_len = length.get(&(n, child)).copied().flatten();
            let combined = match (top_len, bot_len) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            };
            let mut parent_kids: Vec<K> = children.get(&p).cloned().unwrap_or_default();
            parent_kids.retain(|&c| c != n);
            parent_kids.push(child);
            children.insert(p, parent_kids);
            parent.insert(child, p);
            length.shift_remove(&(p, n));
            length.shift_remove(&(n, child));
            length.insert((p, child), combined);
            children.shift_remove(&n);
            parent.shift_remove(&n);
            changed = true;
        }
    }
    root
}

/// Post-order traversal (children pushed in order, so visited last-first).
fn compute_post_order_core<K: Copy + Eq + Hash>(root: K, children: &FxIndexMap<K, Vec<K>>) -> Vec<K> {
    let mut post = Vec::new();
    let mut stack = vec![(root, false)];
    while let Some((node, processed)) = stack.pop() {
        if processed {
            post.push(node);
            continue;
        }
        stack.push((node, true));
        if let Some(kids) = children.get(&node) {
            for &c in kids {
                stack.push((c, false));
            }
        }
    }
    post
}

/// Interns arbitrary Python objects (by Python equality) to dense indices
/// so the generic static helpers can run the Rust cores on them.
struct Interner<'py> {
    table: Bound<'py, PyDict>,
    objs: Vec<Bound<'py, PyAny>>,
}

impl<'py> Interner<'py> {
    fn new(py: Python<'py>) -> Self {
        Interner {
            table: PyDict::new(py),
            objs: Vec::new(),
        }
    }

    fn intern(&mut self, obj: &Bound<'py, PyAny>) -> PyResult<usize> {
        if let Some(i) = self.table.get_item(obj)? {
            return i.extract();
        }
        let i = self.objs.len();
        self.table.set_item(obj, i)?;
        self.objs.push(obj.clone());
        Ok(i)
    }

    fn get(&self, i: usize) -> PyResult<&Bound<'py, PyAny>> {
        self.objs
            .get(i)
            .ok_or_else(|| PyKeyError::new_err("displayed-tree node index out of range"))
    }
}

/// Python's ``seq[i]`` on a two-element list.
fn py_index2(parents: &[u64], choice: i64) -> PyResult<u64> {
    let len = parents.len() as i64;
    let idx = if choice < 0 { choice + len } else { choice };
    if idx < 0 || idx >= len {
        return Err(PyIndexError::new_err("list index out of range"));
    }
    Ok(parents[idx as usize])
}

fn node_of<'py>(py: Python<'py>, n: &Network, uid: u64) -> PyResult<Bound<'py, Node>> {
    n.nodes
        .get(&uid)
        .map(|e| e.obj.bind(py).clone())
        .ok_or_else(|| network_error(py, "displayed tree refers to a node that left the network"))
}

/// One displayed tree of a species network.
///
/// Constructed by ``_GTLikelihoodEngine._build_displayed_trees`` via edge
/// contraction: for each reticulation ``r`` the retained parent edge is
/// kept and the other is deleted, then the resulting degree-2 passthrough
/// nodes are merged with their surviving neighbours into a single edge
/// whose length is the sum of the originals.  This preserves the MSC
/// evaluation semantics while reducing the graph to a strict rooted
/// binary tree.
///
/// Instances expose the minimal data the MSC DP needs -- a leaf label ->
/// node map, a parent map, a children map, an edge-length map and a
/// post-order node list -- plus the int-indexed view (``_idx_*``) the
/// compiled tree DP consumes.
#[pyclass(name = "_DisplayedTree", module = "phynetpy._mcmc_gt", subclass)]
pub struct DisplayedTree {
    log_weight: f64,
    root: Py<PyAny>,
    leaves: Py<PyAny>,
    leaf_species: Py<PyAny>,
    children: Py<PyAny>,
    parent: Py<PyAny>,
    edge_length: Py<PyAny>,
    post_order: Py<PyAny>,
    signature: Py<PyTuple>,
    idx: Py<DisplayedTreeIndex>,
}

impl DisplayedTree {
    /// ``_DisplayedTree.from_network`` on uids; ``choice`` maps a
    /// reticulation to the index (0 / 1) of the parent whose edge is kept.
    pub fn from_network_impl<'py>(
        cls: &Bound<'py, PyType>,
        network: &Bound<'py, Network>,
        choice: &FxHashMap<u64, i64>,
        log_weight: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = network.py();
        let n = network.try_borrow()?;
        let root = n.root_uid(py)?;
        let is_retic = |u: u64| n.nodes.get(&u).map(|e| e.obj.borrow(py).is_retic).unwrap_or(false);

        let mut children_map: FxIndexMap<u64, Vec<u64>> = IndexMap::default();
        let mut parent_map: FxIndexMap<u64, u64> = IndexMap::default();
        let mut length_map: FxIndexMap<(u64, u64), Option<f64>> = IndexMap::default();
        let mut leaves: Vec<u64> = Vec::new();
        let mut leaf_species: Vec<(u64, String)> = Vec::new();

        let mut visited: FxHashSet<u64> = FxHashSet::default();
        let mut queue: VecDeque<u64> = VecDeque::new();
        queue.push_back(root);
        while let Some(cur) = queue.pop_front() {
            if !visited.insert(cur) {
                continue;
            }
            if n.out_deg(cur) == 0 {
                leaves.push(cur);
                leaf_species.push((cur, n.label(py, cur)));
                children_map.insert(cur, Vec::new());
                continue;
            }
            for child in n.children(cur) {
                let edge = n.edge_between(py, cur, child, None, None)?;
                let length = edge.borrow(py).length;
                if is_retic(child) {
                    let parents = n.parents(child);
                    let kept = if parents.len() != 2 {
                        cur
                    } else {
                        py_index2(&parents, choice.get(&child).copied().unwrap_or(0))?
                    };
                    if kept != cur {
                        continue;
                    }
                }
                children_map.entry(cur).or_default().push(child);
                parent_map.insert(child, cur);
                length_map.insert((cur, child), length);
                queue.push_back(child);
            }
        }
        let root_out = collapse_passthroughs_core(root, &mut children_map, &mut parent_map, &mut length_map);
        let post_order = compute_post_order_core(root_out, &children_map);
        let kept_leaves: Vec<u64> = leaves
            .iter()
            .copied()
            .filter(|l| parent_map.contains_key(l) || *l == root_out)
            .collect();

        // Materialise the Python-side maps keyed by the live Node objects.
        let children_d = PyDict::new(py);
        for (k, kids) in children_map.iter() {
            let items: Vec<Bound<'py, Node>> =
                kids.iter().map(|&c| node_of(py, &n, c)).collect::<PyResult<_>>()?;
            children_d.set_item(node_of(py, &n, *k)?, PyList::new(py, items)?)?;
        }
        let parent_d = PyDict::new(py);
        for (c, p) in parent_map.iter() {
            parent_d.set_item(node_of(py, &n, *c)?, node_of(py, &n, *p)?)?;
        }
        let length_d = PyDict::new(py);
        for ((a, b), l) in length_map.iter() {
            let key = PyTuple::new(py, [node_of(py, &n, *a)?, node_of(py, &n, *b)?])?;
            length_d.set_item(key, *l)?;
        }
        let species_d = PyDict::new(py);
        for (l, sp) in &leaf_species {
            species_d.set_item(node_of(py, &n, *l)?, sp)?;
        }
        let leaves_l: Vec<Bound<'py, Node>> =
            kept_leaves.iter().map(|&l| node_of(py, &n, l)).collect::<PyResult<_>>()?;
        let post_l: Vec<Bound<'py, Node>> =
            post_order.iter().map(|&u| node_of(py, &n, u)).collect::<PyResult<_>>()?;
        let root_obj = node_of(py, &n, root_out)?;
        drop(n);

        let kwargs = PyDict::new(py);
        kwargs.set_item("log_weight", log_weight)?;
        kwargs.set_item("root", root_obj)?;
        kwargs.set_item("leaves", PyList::new(py, leaves_l)?)?;
        kwargs.set_item("leaf_species", species_d)?;
        kwargs.set_item("children", children_d)?;
        kwargs.set_item("parent", parent_d)?;
        kwargs.set_item("edge_length", length_d)?;
        kwargs.set_item("post_order", PyList::new(py, post_l)?)?;
        cls.call((), Some(&kwargs))
    }
}

#[pymethods]
impl DisplayedTree {
    /// Store the pre-computed topology and build the int-indexed view.
    #[new]
    #[pyo3(signature = (*, log_weight, root, leaves, leaf_species, children, parent, edge_length, post_order))]
    #[allow(clippy::too_many_arguments)]
    fn py_new<'py>(
        py: Python<'py>,
        log_weight: f64,
        root: Bound<'py, PyAny>,
        leaves: Bound<'py, PyAny>,
        leaf_species: Bound<'py, PyAny>,
        children: Bound<'py, PyAny>,
        parent: Bound<'py, PyAny>,
        edge_length: Bound<'py, PyAny>,
        post_order: Bound<'py, PyAny>,
    ) -> PyResult<Self> {
        let post: Vec<Bound<'py, PyAny>> = post_order.try_iter()?.collect::<PyResult<_>>()?;
        let leaf_list: Vec<Bound<'py, PyAny>> = leaves.try_iter()?.collect::<PyResult<_>>()?;
        let parent_d = parent.cast::<PyDict>()?;
        let children_d = children.cast::<PyDict>()?;
        let length_d = edge_length.cast::<PyDict>()?;
        let species_d = leaf_species.cast::<PyDict>()?;

        let length_of = |a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>| -> PyResult<Option<f64>> {
            match length_d.get_item((a, b))? {
                Some(v) => v.extract::<Option<f64>>(),
                None => Ok(None),
            }
        };

        // Cache key: (id(node), length of the edge feeding it) in post-order.
        let mut pairs: Vec<Bound<'py, PyTuple>> = Vec::with_capacity(post.len());
        for node in &post {
            let length = match parent_d.get_item(node)? {
                Some(p) if !p.is_none() => length_of(&p, node)?,
                _ => None,
            };
            pairs.push((node.as_ptr() as usize, length).into_pyobject(py)?);
        }
        let signature = PyTuple::new(py, pairs)?;

        let idx_of = PyDict::new(py);
        for (i, node) in post.iter().enumerate() {
            idx_of.set_item(node, i)?;
        }
        let n_total = post.len();
        let mut idx_children: Vec<Vec<usize>> = vec![Vec::new(); n_total];
        let mut idx_lengths: Vec<Vec<Option<f64>>> = vec![Vec::new(); n_total];
        for (i, node) in post.iter().enumerate() {
            if let Some(kids) = children_d.get_item(node)? {
                for child in kids.try_iter()? {
                    let child = child?;
                    let ci: usize = match idx_of.get_item(&child)? {
                        Some(v) => v.extract()?,
                        None => return Err(PyKeyError::new_err(child.repr()?.to_string())),
                    };
                    idx_children[i].push(ci);
                    idx_lengths[i].push(length_of(node, &child)?);
                }
            }
        }
        let mut idx_leaves: Vec<usize> = Vec::new();
        let mut idx_species: FxHashMap<usize, String> = FxHashMap::default();
        for leaf in &leaf_list {
            if let Some(i) = idx_of.get_item(leaf)? {
                let i: usize = i.extract()?;
                idx_leaves.push(i);
                if let Some(sp) = species_d.get_item(leaf)? {
                    idx_species.insert(i, sp.extract::<String>()?);
                }
            }
        }
        let idx_root: i64 = match idx_of.get_item(&root)? {
            Some(v) => v.extract()?,
            None => -1,
        };
        let idx = Py::new(
            py,
            DisplayedTreeIndex {
                n: n_total,
                children: idx_children,
                edge_lengths: idx_lengths,
                leaves: idx_leaves,
                leaf_species: idx_species,
                root: idx_root,
            },
        )?;
        Ok(DisplayedTree {
            log_weight,
            root: root.unbind(),
            leaves: leaves.unbind(),
            leaf_species: leaf_species.unbind(),
            children: children.unbind(),
            parent: parent.unbind(),
            edge_length: edge_length.unbind(),
            post_order: post_order.unbind(),
            signature: signature.unbind(),
            idx,
        })
    }

    /// Construct a displayed tree from a network + retic choices
    /// (``{retic_node: 0-or-1}`` selecting the retained parent).
    #[classmethod]
    fn from_network<'py>(
        cls: &Bound<'py, PyType>,
        net: &Bound<'py, PyAny>,
        retic_choice: &Bound<'py, PyAny>,
        log_weight: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let network = net.cast::<Network>()?;
        let mut choice: FxHashMap<u64, i64> = FxHashMap::default();
        if !retic_choice.is_none() {
            let d = retic_choice.cast::<PyDict>()?;
            let n = network.try_borrow()?;
            for (k, v) in d.iter() {
                if let Some(uid) = n.uid_of_any(&k) {
                    choice.insert(uid, v.extract::<i64>()?);
                }
            }
        }
        Self::from_network_impl(cls, network, &choice, log_weight)
    }

    /// Merge degree-2 (in=1, out=1) chains into single edges, in place on
    /// the supplied maps; returns the (unchanged) root.
    #[staticmethod]
    #[pyo3(signature = (*, root, children_map, parent_map, length_map, leaves, is_retic))]
    #[allow(unused_variables)]
    fn _collapse_passthroughs<'py>(
        root: Bound<'py, PyAny>,
        children_map: Bound<'py, PyDict>,
        parent_map: Bound<'py, PyDict>,
        length_map: Bound<'py, PyDict>,
        leaves: Bound<'py, PyAny>,
        is_retic: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = root.py();
        let mut it = Interner::new(py);
        let root_i = it.intern(&root)?;
        let mut children: FxIndexMap<usize, Vec<usize>> = IndexMap::default();
        for (k, v) in children_map.iter() {
            let ki = it.intern(&k)?;
            let mut kids = Vec::new();
            for c in v.try_iter()? {
                kids.push(it.intern(&c?)?);
            }
            children.insert(ki, kids);
        }
        let mut parent: FxIndexMap<usize, usize> = IndexMap::default();
        for (k, v) in parent_map.iter() {
            let ki = it.intern(&k)?;
            let vi = it.intern(&v)?;
            parent.insert(ki, vi);
        }
        let mut length: FxIndexMap<(usize, usize), Option<f64>> = IndexMap::default();
        for (k, v) in length_map.iter() {
            let (a, b): (Bound<'py, PyAny>, Bound<'py, PyAny>) = k.extract()?;
            let key = (it.intern(&a)?, it.intern(&b)?);
            length.insert(key, v.extract::<Option<f64>>()?);
        }
        let out_root = collapse_passthroughs_core(root_i, &mut children, &mut parent, &mut length);

        children_map.clear();
        for (k, kids) in children.iter() {
            let items: Vec<&Bound<'py, PyAny>> = kids.iter().map(|&c| it.get(c)).collect::<PyResult<_>>()?;
            children_map.set_item(it.get(*k)?, PyList::new(py, items)?)?;
        }
        parent_map.clear();
        for (k, p) in parent.iter() {
            parent_map.set_item(it.get(*k)?, it.get(*p)?)?;
        }
        length_map.clear();
        for ((a, b), l) in length.iter() {
            let key = PyTuple::new(py, [it.get(*a)?, it.get(*b)?])?;
            length_map.set_item(key, *l)?;
        }
        Ok(it.get(out_root)?.clone())
    }

    /// Return a post-order traversal of the displayed tree.
    #[staticmethod]
    fn _compute_post_order<'py>(
        root: Bound<'py, PyAny>,
        children_map: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyList>> {
        let py = root.py();
        let mut it = Interner::new(py);
        let root_i = it.intern(&root)?;
        let mut children: FxIndexMap<usize, Vec<usize>> = IndexMap::default();
        for (k, v) in children_map.iter() {
            let ki = it.intern(&k)?;
            let mut kids = Vec::new();
            for c in v.try_iter()? {
                kids.push(it.intern(&c?)?);
            }
            children.insert(ki, kids);
        }
        let post = compute_post_order_core(root_i, &children);
        let items: Vec<&Bound<'py, PyAny>> = post.iter().map(|&i| it.get(i)).collect::<PyResult<_>>()?;
        PyList::new(py, items)
    }

    #[getter]
    fn get_log_weight(&self) -> f64 {
        self.log_weight
    }

    #[setter]
    fn set_log_weight(&mut self, value: f64) {
        self.log_weight = value;
    }

    #[getter]
    fn root(&self, py: Python<'_>) -> Py<PyAny> {
        self.root.clone_ref(py)
    }

    #[getter]
    fn leaves(&self, py: Python<'_>) -> Py<PyAny> {
        self.leaves.clone_ref(py)
    }

    #[getter]
    fn leaf_species(&self, py: Python<'_>) -> Py<PyAny> {
        self.leaf_species.clone_ref(py)
    }

    #[getter]
    fn children(&self, py: Python<'_>) -> Py<PyAny> {
        self.children.clone_ref(py)
    }

    #[getter]
    fn parent(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.clone_ref(py)
    }

    #[getter]
    fn edge_length(&self, py: Python<'_>) -> Py<PyAny> {
        self.edge_length.clone_ref(py)
    }

    #[getter]
    fn post_order(&self, py: Python<'_>) -> Py<PyAny> {
        self.post_order.clone_ref(py)
    }

    #[getter]
    fn signature(&self, py: Python<'_>) -> Py<PyTuple> {
        self.signature.clone_ref(py)
    }

    #[getter]
    fn _rs_idx(&self, py: Python<'_>) -> Py<DisplayedTreeIndex> {
        self.idx.clone_ref(py)
    }

    #[getter]
    fn _idx_n(&self, py: Python<'_>) -> usize {
        self.idx.borrow(py).n
    }

    #[getter]
    fn _idx_post_order(&self, py: Python<'_>) -> Vec<usize> {
        (0..self.idx.borrow(py).n).collect()
    }

    #[getter]
    fn _idx_children(&self, py: Python<'_>) -> Vec<Vec<usize>> {
        self.idx.borrow(py).children.clone()
    }

    #[getter]
    fn _idx_edge_lengths(&self, py: Python<'_>) -> Vec<Vec<Option<f64>>> {
        self.idx.borrow(py).edge_lengths.clone()
    }

    #[getter]
    fn _idx_leaves(&self, py: Python<'_>) -> Vec<usize> {
        self.idx.borrow(py).leaves.clone()
    }

    #[getter]
    fn _idx_leaf_species(&self, py: Python<'_>) -> FxHashMap<usize, String> {
        self.idx.borrow(py).leaf_species.clone()
    }

    #[getter]
    fn _idx_root(&self, py: Python<'_>) -> i64 {
        self.idx.borrow(py).root
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Likelihood engine
// ═══════════════════════════════════════════════════════════════════════

/// Per-edge ``(src uid, dest uid, length bits, gamma bits)`` in edge order:
/// everything the AC DP reads off the network.
type Signature = Vec<(u64, u64, Option<u64>, Option<u64>)>;

/// Full MSNC gene-tree-topology likelihood for a fixed species network.
///
/// Computes ``log P(g | Psi)`` by the ancestral-configurations DP of Yu,
/// Degnan & Nakhleh (2012): the DP walks the species network in
/// bottom-up topological order maintaining a joint distribution over all
/// currently-open species-network edges, applying the Kingman branch
/// coalescent on every traversed edge and splitting each lineage
/// independently at every reticulation.  Reduces to the standard MSC
/// tree DP when the network has no reticulations (that case is routed
/// through the displayed-tree DP, which gives identical results).
///
/// Caching: the branch kernel's ``_log_gij`` / ``_log_denom`` memos
/// survive every move; the ``_NetworkIndex`` view is rebuilt only on
/// topology changes (``update(None)``); the per-gene-tree
/// ``_GeneTreeIndex`` views live for the engine's lifetime; and the
/// ``_score_cache`` keyed by ``(id(gene_tree), network signature)`` lets
/// MH chains skip the DP entirely on revisits to an evaluated state.
#[pyclass(
    name = "_GTLikelihoodEngine",
    module = "phynetpy._mcmc_gt",
    extends = MSCBranchKernel,
    subclass,
    dict
)]
pub struct GTLikelihoodEngine {
    network: Py<PyAny>,
    net_index: Option<Py<NetworkIndex>>,
    displayed_trees: Option<Vec<Py<PyAny>>>,
    gene_tree_log_prob: FxHashMap<usize, f64>,
    gti_cache: FxHashMap<usize, Py<GeneTreeIndex>>,
    cached_signature: Option<Signature>,
    score_cache: FxHashMap<Signature, FxHashMap<usize, f64>>,
    gt_by_id: FxHashMap<usize, Py<PyAny>>,
}

fn species_count(g: &GeneTreeIndex) -> usize {
    g.topo
        .leaves
        .iter()
        .filter(|&&l| g.topo.leaf_species.get(l).map(|s| s.is_some()).unwrap_or(false))
        .count()
}

impl GTLikelihoodEngine {
    fn fresh(network: Py<PyAny>) -> Self {
        GTLikelihoodEngine {
            network,
            net_index: None,
            displayed_trees: None,
            gene_tree_log_prob: FxHashMap::default(),
            gti_cache: FxHashMap::default(),
            cached_signature: None,
            score_cache: FxHashMap::default(),
            gt_by_id: FxHashMap::default(),
        }
    }

    fn network_bound<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Network>> {
        let py = slf.py();
        let net = slf.try_borrow()?.network.clone_ref(py);
        net.into_bound(py).cast_into::<Network>().map_err(PyErr::from)
    }

    /// Recompute the network signature if the memo was invalidated.
    fn ensure_signature(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        if slf.try_borrow()?.cached_signature.is_some() {
            return Ok(());
        }
        let net = Self::network_bound(slf)?;
        let sig: Signature = {
            let n = net.try_borrow()?;
            n.edges
                .values()
                .map(|e| {
                    let b = e.obj.borrow(py);
                    (e.su, e.du, b.length.map(f64::to_bits), b.gamma.map(f64::to_bits))
                })
                .collect()
        };
        slf.try_borrow_mut()?.cached_signature = Some(sig);
        Ok(())
    }

    fn ensure_net_index(slf: &Bound<'_, Self>) -> PyResult<Py<NetworkIndex>> {
        let py = slf.py();
        if let Some(idx) = slf.try_borrow()?.net_index.as_ref() {
            return Ok(idx.clone_ref(py));
        }
        let net = Self::network_bound(slf)?;
        let idx = py
            .get_type::<NetworkIndex>()
            .call1((net,))?
            .cast_into::<NetworkIndex>()?
            .unbind();
        slf.try_borrow_mut()?.net_index = Some(idx.clone_ref(py));
        Ok(idx)
    }

    fn build_displayed_trees_impl(slf: &Bound<'_, Self>) -> PyResult<Vec<Py<PyAny>>> {
        let py = slf.py();
        let net = Self::network_bound(slf)?;
        let retics: Vec<u64> = {
            let n = net.try_borrow()?;
            n.nodes
                .iter()
                .filter(|(_, e)| e.obj.borrow(py).is_retic)
                .map(|(u, _)| *u)
                .collect()
        };
        let k = retics.len();
        if k >= 64 {
            return Err(PyValueError::new_err(format!(
                "cannot enumerate the displayed trees of a network with {k} reticulations"
            )));
        }
        let cls = py.get_type::<DisplayedTree>();
        let mut trees = Vec::new();
        for mask in 0..(1u64 << k) {
            let mut keep_choice: FxHashMap<u64, i64> = FxHashMap::default();
            let mut log_weight = 0.0;
            let mut valid = true;
            {
                let n = net.try_borrow()?;
                for (i, &r) in retics.iter().enumerate() {
                    let parents = n.parents(r);
                    if parents.len() != 2 {
                        valid = false;
                        break;
                    }
                    let choice = ((mask >> i) & 1) as i64;
                    keep_choice.insert(r, choice);
                    let kept_edge = n.edge_between(py, parents[choice as usize], r, None, None)?;
                    let gamma = kept_edge.borrow(py).gamma;
                    match gamma {
                        Some(g) if !(g <= 0.0) => log_weight += g.ln(),
                        _ => log_weight += LOG_FLOOR,
                    }
                }
            }
            if !valid {
                continue;
            }
            let dt = DisplayedTree::from_network_impl(&cls, &net, &keep_choice, log_weight)?;
            trees.push(dt.unbind());
        }
        Ok(trees)
    }

    fn ensure_displayed_trees_impl(slf: &Bound<'_, Self>) -> PyResult<Vec<Py<PyAny>>> {
        let py = slf.py();
        if let Some(trees) = slf.try_borrow()?.displayed_trees.as_ref() {
            return Ok(trees.iter().map(|t| t.clone_ref(py)).collect());
        }
        let trees = Self::build_displayed_trees_impl(slf)?;
        slf.try_borrow_mut()?.displayed_trees = Some(trees.iter().map(|t| t.clone_ref(py)).collect());
        Ok(trees)
    }

    /// The first displayed tree (the whole tree when there is no retic).
    fn first_displayed_tree(slf: &Bound<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        if let Some(trees) = slf.try_borrow()?.displayed_trees.as_ref() {
            return Ok(trees.first().map(|t| t.clone_ref(py)));
        }
        let trees = Self::ensure_displayed_trees_impl(slf)?;
        Ok(trees.first().map(|t| t.clone_ref(py)))
    }

    fn get_gti_impl(
        slf: &Bound<'_, Self>,
        gene_tree: &Bound<'_, PyAny>,
        species_of: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<GeneTreeIndex>> {
        let py = slf.py();
        let gid = gene_tree.as_ptr() as usize;
        let existing = slf.try_borrow()?.gti_cache.get(&gid).map(|g| g.clone_ref(py));
        match existing {
            None => {
                let sp_arg: Bound<'_, PyAny> = match species_of {
                    Some(s) => s.clone(),
                    None => py.None().into_bound(py),
                };
                let gti = py
                    .get_type::<GeneTreeIndex>()
                    .call1((gene_tree, sp_arg))?
                    .cast_into::<GeneTreeIndex>()?
                    .unbind();
                let mut me = slf.try_borrow_mut()?;
                me.gti_cache.insert(gid, gti.clone_ref(py));
                me.gt_by_id.insert(gid, gene_tree.clone().unbind());
                Ok(gti)
            }
            Some(gti) => {
                if let Some(sp) = species_of {
                    if !sp.is_none() {
                        let empty = {
                            let g = gti.try_borrow(py)?;
                            species_count(&g) == 0
                        };
                        if empty {
                            gti.bind(py).call_method1("refresh_species", (gene_tree, sp))?;
                        }
                    }
                }
                Ok(gti)
            }
        }
    }

    fn log_prob_impl(
        slf: &Bound<'_, Self>,
        gene_tree: &Bound<'_, PyAny>,
        species_of: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        let py = slf.py();
        let key = gene_tree.as_ptr() as usize;
        if let Some(v) = slf.try_borrow()?.gene_tree_log_prob.get(&key) {
            return Ok(*v);
        }

        let net_idx = Self::ensure_net_index(slf)?;
        let (n_nodes, root, has_retic) = {
            let b = net_idx.try_borrow(py)?;
            (b.n_nodes, b.root, b.is_retic.iter().any(|&r| r))
        };
        if n_nodes == 0 || root < 0 {
            slf.try_borrow_mut()?.gene_tree_log_prob.insert(key, LOG_FLOOR);
            return Ok(LOG_FLOOR);
        }

        let gti = Self::get_gti_impl(slf, gene_tree, species_of)?;
        if let Some(sp) = species_of {
            if !sp.is_none() {
                let stale = {
                    let g = gti.try_borrow(py)?;
                    species_count(&g) != g.topo.leaves.len()
                };
                if stale {
                    gti.bind(py).call_method1("refresh_species", (gene_tree, sp))?;
                }
            }
        }

        Self::ensure_signature(slf)?;
        let cached_score = {
            let me = slf.try_borrow()?;
            let sig = me
                .cached_signature
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("network signature missing"))?;
            me.score_cache.get(sig).and_then(|m| m.get(&key)).copied()
        };
        let score = match cached_score {
            Some(s) => s,
            None => {
                // The DP wrappers borrow the kernel (our base) mutably, so no
                // borrow of ``slf`` may be live across these calls.
                let mut s = if !has_retic {
                    match Self::first_displayed_tree(slf)? {
                        Some(dt) => msc_log_prob_tree_py(dt.bind(py), gti.bind(py), slf.as_any())?,
                        None => msnc_log_prob_network_py(net_idx.bind(py), gti.bind(py), slf.as_any())?,
                    }
                } else {
                    msnc_log_prob_network_py(net_idx.bind(py), gti.bind(py), slf.as_any())?
                };
                if !s.is_finite() {
                    s = LOG_FLOOR;
                }
                let mut me = slf.try_borrow_mut()?;
                let sig = me
                    .cached_signature
                    .clone()
                    .ok_or_else(|| PyValueError::new_err("network signature missing"))?;
                me.score_cache.entry(sig).or_default().insert(key, s);
                s
            }
        };
        slf.try_borrow_mut()?.gene_tree_log_prob.insert(key, score);
        Ok(score)
    }
}

fn signature_to_py<'py>(py: Python<'py>, sig: &Signature) -> PyResult<Bound<'py, PyTuple>> {
    let items: Vec<Bound<'py, PyTuple>> = sig
        .iter()
        .map(|(s, d, l, g)| (*s, *d, l.map(f64::from_bits), g.map(f64::from_bits)).into_pyobject(py))
        .collect::<PyResult<_>>()?;
    PyTuple::new(py, items)
}

#[pymethods]
impl GTLikelihoodEngine {
    /// Bind the engine to a species network (``theta`` starts at 2.0).
    /// Heavy caching happens lazily on the first ``log_prob`` call.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn py_new(
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let mut network: Option<Bound<'_, PyAny>> = args.get_item(0).ok();
        if let Some(kw) = kwargs {
            if let Some(n) = kw.get_item("network")? {
                network = Some(n);
            }
        }
        let network = network.map(|b| b.unbind()).unwrap_or_else(|| py.None());
        Ok(PyClassInitializer::from(MSCBranchKernel::with_theta(2.0))
            .add_subclass(GTLikelihoodEngine::fresh(network)))
    }

    /// Legacy displayed-tree warning threshold (diagnostic only).
    #[classattr]
    #[allow(non_snake_case)]
    fn _DISPLAYED_TREE_WARN_THRESHOLD() -> i64 {
        7
    }

    #[getter]
    fn get_network(&self, py: Python<'_>) -> Py<PyAny> {
        self.network.clone_ref(py)
    }

    #[setter]
    fn set_network(&mut self, value: Bound<'_, PyAny>) {
        self.network = value.unbind();
    }

    #[getter]
    fn _net_index(&self, py: Python<'_>) -> Option<Py<NetworkIndex>> {
        self.net_index.as_ref().map(|i| i.clone_ref(py))
    }

    #[getter]
    fn _displayed_trees(&self, py: Python<'_>) -> Option<Vec<Py<PyAny>>> {
        self.displayed_trees
            .as_ref()
            .map(|v| v.iter().map(|t| t.clone_ref(py)).collect())
    }

    #[getter]
    fn _gene_tree_log_prob(&self) -> FxHashMap<usize, f64> {
        self.gene_tree_log_prob.clone()
    }

    #[getter]
    fn _gti_cache(&self, py: Python<'_>) -> FxHashMap<usize, Py<GeneTreeIndex>> {
        self.gti_cache.iter().map(|(k, v)| (*k, v.clone_ref(py))).collect()
    }

    #[getter]
    fn _gt_by_id(&self, py: Python<'_>) -> FxHashMap<usize, Py<PyAny>> {
        self.gt_by_id.iter().map(|(k, v)| (*k, v.clone_ref(py))).collect()
    }

    #[getter]
    fn _cached_signature<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        match &self.cached_signature {
            Some(sig) => Ok(Some(signature_to_py(py, sig)?)),
            None => Ok(None),
        }
    }

    #[getter]
    fn _score_cache<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (sig, per_gt) in self.score_cache.iter() {
            let sig_t = signature_to_py(py, sig)?;
            for (gid, score) in per_gt.iter() {
                d.set_item((*gid, sig_t.clone()), *score)?;
            }
        }
        Ok(d)
    }

    /// Enumerate every displayed tree of ``self.network`` (one per choice of
    /// retained parent edge at each reticulation; weight = product of the
    /// retained gammas).
    fn _build_displayed_trees(slf: &Bound<'_, Self>) -> PyResult<Vec<Py<PyAny>>> {
        Self::build_displayed_trees_impl(slf)
    }

    /// Return cached displayed trees; rebuild if missing.
    fn _ensure_displayed_trees(slf: &Bound<'_, Self>) -> PyResult<Vec<Py<PyAny>>> {
        Self::ensure_displayed_trees_impl(slf)
    }

    /// Invalidate caches affected by the last batch of moves: ``None``
    /// (topology changed) drops the network index, the displayed trees,
    /// the per-call memo and the score cache; a non-empty set of dirty
    /// nodes keeps the index but drops the displayed trees, the per-call
    /// memo and the memoised signature; an empty set is a no-op.
    fn update(slf: &Bound<'_, Self>, dirty_nodes: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let full = match &dirty_nodes {
            None => true,
            Some(d) => d.is_none(),
        };
        if full {
            let mut me = slf.try_borrow_mut()?;
            me.net_index = None;
            me.displayed_trees = None;
            me.gene_tree_log_prob.clear();
            me.score_cache.clear();
            me.cached_signature = None;
            return Ok(());
        }
        let Some(dirty) = dirty_nodes else { return Ok(()) };
        if !dirty.is_truthy()? {
            return Ok(());
        }
        let mut me = slf.try_borrow_mut()?;
        me.displayed_trees = None;
        me.gene_tree_log_prob.clear();
        me.cached_signature = None;
        Ok(())
    }

    /// Compact tuple of ``(src uid, dest uid, length, gamma)`` per edge:
    /// the cache key for the per-(gene-tree, network) score memo, memoised
    /// between two ``update`` calls.
    fn _network_signature<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        Self::ensure_signature(slf)?;
        let me = slf.try_borrow()?;
        match me.cached_signature.as_ref() {
            Some(sig) => signature_to_py(py, sig),
            None => PyTuple::new(py, Vec::<Bound<'py, PyTuple>>::new()),
        }
    }

    /// Lazily build (or fetch) the bit-indexed view of ``gene_tree``.
    #[pyo3(signature = (gene_tree, species_of=None))]
    fn _get_gti(
        slf: &Bound<'_, Self>,
        gene_tree: &Bound<'_, PyAny>,
        species_of: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<GeneTreeIndex>> {
        Self::get_gti_impl(slf, gene_tree, species_of)
    }

    /// ``log P(gene_tree | self.network)`` by the MSNC ancestral-
    /// configurations DP, floored at ``_LOG_FLOOR`` for incompatible
    /// configurations.  ``species_of`` maps gene-copy label -> species.
    #[pyo3(signature = (gene_tree, species_of=None))]
    fn log_prob(
        slf: &Bound<'_, Self>,
        gene_tree: &Bound<'_, PyAny>,
        species_of: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        Self::log_prob_impl(slf, gene_tree, species_of)
    }

    /// Sum of ``log_prob`` over a batch of gene trees, in order.
    #[pyo3(signature = (gene_trees, species_of=None))]
    fn score_many(
        slf: &Bound<'_, Self>,
        gene_trees: &Bound<'_, PyAny>,
        species_of: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        let mut total = 0.0;
        for g in gene_trees.try_iter()? {
            let g = g?;
            total += Self::log_prob_impl(slf, &g, species_of)?;
        }
        Ok(total)
    }

    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let d = PyDict::new(py);
        d.set_item("theta", slf.getattr("theta")?)?;
        d.set_item("network", slf.try_borrow()?.network.clone_ref(py))?;
        if let Ok(dict) = slf.getattr("__dict__") {
            d.set_item("__dict__", dict)?;
        }
        Ok(d)
    }

    fn __setstate__(slf: &Bound<'_, Self>, state: Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(t) = state.get_item("theta")? {
            slf.setattr("theta", t)?;
        }
        if let Some(n) = state.get_item("network")? {
            slf.try_borrow_mut()?.network = n.unbind();
        }
        if let Some(dict) = state.get_item("__dict__")? {
            if let Ok(target) = slf.getattr("__dict__") {
                target.call_method1("update", (dict,))?;
            }
        }
        Ok(())
    }

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::empty(py)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DisplayedTree>()?;
    m.add_class::<GTLikelihoodEngine>()?;
    m.add_function(wrap_pyfunction!(lgamma_py, m)?)?;
    m.add_function(wrap_pyfunction!(log_exp_pdf_py, m)?)?;
    m.add_function(wrap_pyfunction!(log_beta_pdf_py, m)?)?;
    m.add_function(wrap_pyfunction!(log_poisson_pmf_py, m)?)?;
    m.add_function(wrap_pyfunction!(log_prior_network_py, m)?)?;
    m.add_function(wrap_pyfunction!(safe_add_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_valid_network_py, m)?)?;
    m.add_function(wrap_pyfunction!(populate_default_branch_lengths_py, m)?)?;
    m.add_function(wrap_pyfunction!(tune_sigma_py, m)?)?;
    m.add_function(wrap_pyfunction!(adapt_weights_py, m)?)?;
    m.add_function(wrap_pyfunction!(spr_decay_py, m)?)?;
    Ok(())
}
