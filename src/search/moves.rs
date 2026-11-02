//! Network proposal moves for MCMC and simulated-annealing search.
//!
//! Each move mirrors the reference Python implementation step for step,
//! including the order of its random draws from the model's
//! ``numpy.random.Generator`` and the stable (label-sorted) candidate
//! orderings, so a seeded search proposes the same sequence of networks.
//! Topology moves open a transaction on the network and leave it open;
//! the caller commits or rolls back.  Every edit goes through the
//! network's journaled operations.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::{Edge, Network, Node};

const SPR_MIN_LEN: f64 = 1e-9;
const LOG_SQRT_2PI: f64 = 0.9189385332046727;

// ═══════════════════════════════════════════════════════════════════════
// Proposal math (shared with the sequence sampler)
// ═══════════════════════════════════════════════════════════════════════

pub fn add_reticulation_log_hastings(e: i64, r: i64, l1: f64, l2: f64) -> PyResult<f64> {
    if e < 2 || l1 <= 0.0 || l2 <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "add_reticulation_log_hastings requires E >= 2 and positive l1, l2 (got E={e}, l1={l1}, l2={l2})."
        )));
    }
    let pda = if r == 0 { 0.5f64 } else { 1.0 };
    Ok(pda.ln() + l1.ln() + l2.ln() + (e as f64).ln() + ((e - 1) as f64).ln() - (2.0 * (r + 1) as f64).ln())
}

pub fn remove_reticulation_log_hastings(ep: i64, r: i64, l1: f64, l2: f64) -> PyResult<f64> {
    if ep < 2 || r < 1 || l1 <= 0.0 || l2 <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "remove_reticulation_log_hastings requires E' >= 2, R >= 1 and positive l1, l2 (got E'={ep}, R={r}, l1={l1}, l2={l2})."
        )));
    }
    let pad = if r == 1 { 2.0f64 } else { 1.0 };
    Ok(pad.ln() + (2.0 * r as f64).ln() - l1.ln() - l2.ln() - (ep as f64).ln() - ((ep - 1) as f64).ln())
}

pub fn normal_log_pdf(x: f64, mu: f64, sigma: f64) -> PyResult<f64> {
    if sigma <= 0.0 {
        return Err(PyValueError::new_err(format!("sigma must be positive (got {sigma}).")));
    }
    let z = (x - mu) / sigma;
    Ok(-0.5 * z * z - sigma.ln() - LOG_SQRT_2PI)
}

pub fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
}

pub fn truncated_normal_log_norm(mu: f64, sigma: f64, lo: f64, hi: f64) -> PyResult<f64> {
    if sigma <= 0.0 {
        return Err(PyValueError::new_err(format!("sigma must be positive (got {sigma}).")));
    }
    if hi <= lo {
        return Err(PyValueError::new_err(format!("require hi > lo (got lo={lo}, hi={hi}).")));
    }
    let z = standard_normal_cdf((hi - mu) / sigma) - standard_normal_cdf((lo - mu) / sigma);
    Ok(z.max(1e-300).ln())
}

pub fn random_walk_truncated_log_hastings(old: f64, new: f64, sigma: f64, lo: f64, hi: f64) -> PyResult<f64> {
    Ok(truncated_normal_log_norm(old, sigma, lo, hi)? - truncated_normal_log_norm(new, sigma, lo, hi)?)
}

#[allow(clippy::too_many_arguments)]
pub fn gaussian_delta_log_hastings(
    delta: f64,
    sigma_fwd: f64,
    sigma_rev: f64,
    lo_fwd: f64,
    hi_fwd: f64,
    lo_rev: f64,
    hi_rev: f64,
) -> PyResult<f64> {
    let log_q_rev = normal_log_pdf(delta, 0.0, sigma_rev)? - truncated_normal_log_norm(0.0, sigma_rev, lo_rev, hi_rev)?;
    let log_q_fwd = normal_log_pdf(delta, 0.0, sigma_fwd)? - truncated_normal_log_norm(0.0, sigma_fwd, lo_fwd, hi_fwd)?;
    Ok(log_q_rev - log_q_fwd)
}

pub fn weighted_choice_log_hastings(fwd_weight: f64, fwd_total: f64, rev_weight: f64, rev_total: f64) -> PyResult<f64> {
    if fwd_weight <= 0.0 || rev_weight <= 0.0 || fwd_total <= 0.0 || rev_total <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "weighted_choice_log_hastings requires strictly positive weights and totals (got fwd={fwd_weight}/{fwd_total}, rev={rev_weight}/{rev_total})."
        )));
    }
    Ok(rev_weight.ln() - rev_total.ln() - fwd_weight.ln() + fwd_total.ln())
}

// ═══════════════════════════════════════════════════════════════════════
// RNG bridge and network helpers
// ═══════════════════════════════════════════════════════════════════════

struct Rng<'py>(Bound<'py, PyAny>);

impl<'py> Rng<'py> {
    fn integers(&self, n: usize) -> PyResult<usize> {
        let v: i64 = self.0.call_method1("integers", (0usize, n))?.extract()?;
        Ok(v as usize)
    }
    fn random(&self) -> PyResult<f64> {
        self.0.call_method0("random")?.extract()
    }
    fn normal(&self, mu: f64, sigma: f64) -> PyResult<f64> {
        self.0.call_method1("normal", (mu, sigma))?.extract()
    }
    fn exponential(&self, scale: f64) -> PyResult<f64> {
        self.0.call_method1("exponential", (scale,))?.extract()
    }
}

static LEVEL_FN: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// ``phynetpy.GraphUtils.level(net)``.
fn network_level(py: Python<'_>, net: &Bound<'_, Network>) -> PyResult<i64> {
    let f = LEVEL_FN.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py.import("phynetpy.GraphUtils")?.getattr("level")?.unbind())
    })?;
    f.bind(py).call1((net,))?.extract()
}

/// Everything a move needs: the network object, its Python handle, and the
/// generator.  All graph access is scoped so no borrow outlives a call.
struct Ctx<'a, 'py> {
    py: Python<'py>,
    net: &'a Bound<'py, Network>,
    rng: Rng<'py>,
}

/// The reference's ``x or default`` on an optional length.
#[inline]
fn or_default(v: Option<f64>, default: f64) -> f64 {
    match v {
        Some(x) if x != 0.0 => x,
        _ => default,
    }
}

impl<'a, 'py> Ctx<'a, 'py> {
    fn b(&self) -> PyRef<'py, Network> {
        self.net.borrow()
    }
    fn bm(&self) -> PyRefMut<'py, Network> {
        self.net.borrow_mut()
    }

    fn all_edges(&self) -> Vec<u64> {
        self.b().edges.keys().copied().collect()
    }
    fn all_nodes(&self) -> Vec<u64> {
        self.b().nodes.keys().copied().collect()
    }
    // Lookups tolerate ids that are no longer in the network (the
    // reference reads stale Node/Edge objects there); they return neutral
    // values instead of panicking.
    fn label(&self, uid: u64) -> String {
        let n = self.b();
        match n.nodes.get(&uid) {
            Some(e) => e.obj.borrow(self.py).name.clone(),
            None => String::new(),
        }
    }
    fn edge_ends(&self, euid: u64) -> (u64, u64) {
        let n = self.b();
        match n.edges.get(&euid) {
            Some(e) => (e.su, e.du),
            None => (u64::MAX, u64::MAX),
        }
    }
    fn edge_len(&self, euid: u64) -> Option<f64> {
        let n = self.b();
        n.edges.get(&euid).and_then(|e| e.obj.borrow(self.py).length)
    }
    fn edge_gamma(&self, euid: u64) -> Option<f64> {
        let n = self.b();
        n.edges.get(&euid).and_then(|e| e.obj.borrow(self.py).gamma)
    }
    fn is_retic(&self, uid: u64) -> bool {
        let n = self.b();
        n.nodes.get(&uid).map(|e| e.obj.borrow(self.py).is_retic).unwrap_or(false)
    }
    fn in_deg(&self, uid: u64) -> usize {
        self.b().in_deg(uid)
    }
    fn out_deg(&self, uid: u64) -> usize {
        self.b().out_deg(uid)
    }
    fn contains(&self, uid: u64) -> bool {
        self.b().nodes.contains_key(&uid)
    }
    fn edge_present(&self, euid: u64) -> bool {
        self.b().edges.contains_key(&euid)
    }
    fn children(&self, uid: u64) -> Vec<u64> {
        self.b().children(uid)
    }
    fn parents(&self, uid: u64) -> Vec<u64> {
        self.b().parents(uid)
    }
    fn in_edges(&self, uid: u64) -> Vec<u64> {
        self.b().in_edge_uids(uid)
    }
    fn out_edges(&self, uid: u64) -> Vec<u64> {
        self.b().out_edge_uids(uid)
    }
    fn root(&self) -> PyResult<u64> {
        self.b().root_uid(self.py)
    }
    fn count_retics(&self) -> usize {
        let n = self.b();
        n.nodes.values().filter(|e| e.obj.borrow(self.py).is_retic).count()
    }

    /// ``Network.get_edge(src, dest)`` (first edge of a bubble, with the warning).
    fn get_edge(&self, su: u64, du: u64) -> PyResult<u64> {
        self.b().first_edge_between(self.py, su, du)
    }

    /// Edges sorted by ``(src.label, dest.label)`` (stable), as ``_stable_sorted``.
    fn pick_edge(&self, cands: &[u64]) -> PyResult<Option<u64>> {
        if cands.is_empty() {
            return Ok(None);
        }
        let mut keyed: Vec<((String, String), u64)> = {
            let n = self.b();
            cands
                .iter()
                .filter_map(|&e| {
                    let ent = n.edges.get(&e)?;
                    Some(((n.label(self.py, ent.su), n.label(self.py, ent.du)), e))
                })
                .collect()
        };
        if keyed.is_empty() {
            return Ok(None);
        }
        keyed.sort_by(|a, b| a.0.cmp(&b.0));
        let i = self.rng.integers(keyed.len())?;
        Ok(Some(keyed[i].1))
    }

    /// Nodes sorted by label (stable).
    fn pick_node(&self, cands: &[u64]) -> PyResult<Option<u64>> {
        if cands.is_empty() {
            return Ok(None);
        }
        let mut keyed: Vec<(String, u64)> = cands.iter().map(|&u| (self.label(u), u)).collect();
        keyed.sort_by(|a, b| a.0.cmp(&b.0));
        let i = self.rng.integers(keyed.len())?;
        Ok(Some(keyed[i].1))
    }

    /// Unsorted uniform choice (``SwitchParentage._random_choice``).
    fn choice<T: Copy>(&self, cands: &[T]) -> PyResult<Option<T>> {
        if cands.is_empty() {
            return Ok(None);
        }
        let i = self.rng.integers(cands.len())?;
        Ok(Some(cands[i]))
    }

    fn begin(&self) {
        self.b().journal.begin();
    }
    fn rollback(&self) -> PyResult<()> {
        Network::rollback_transaction(self.net)
    }
    fn in_transaction(&self) -> bool {
        self.b().journal.in_transaction()
    }

    /// ``Network.add_uid_node()``.
    fn add_uid_node(&self) -> PyResult<u64> {
        let name = format!("UID_{}", self.b().uid);
        let node = Py::new(self.py, Node::make(self.py, name, false, None, None, None))?;
        let uid = node.borrow(self.py).uid;
        let mut n = self.bm();
        n.add_node(self.net.as_any(), node.bind(self.py))?;
        n.uid += 1;
        Ok(uid)
    }

    fn set_retic(&self, uid: u64, value: bool) {
        let obj = match self.b().nodes.get(&uid) {
            Some(e) => e.obj.clone_ref(self.py),
            None => return,
        };
        Node::set_is_reticulation(obj.bind(self.py), value);
    }

    /// ``Edge(a, b, length=..., gamma=...)`` added to the network.
    fn add_edge(&self, su: u64, du: u64, length: Option<f64>, gamma: Option<f64>) -> PyResult<u64> {
        let (s, d) = {
            let n = self.b();
            match (n.nodes.get(&su), n.nodes.get(&du)) {
                (Some(a), Some(b)) => (a.obj.clone_ref(self.py), b.obj.clone_ref(self.py)),
                _ => {
                    return Err(crate::util::network_error(
                        self.py,
                        "Tried to add an edge between two nodes, at least one of which does \
                         not belong to this network.",
                    ))
                }
            }
        };
        let gamma = gamma.unwrap_or(0.0);
        if !(0.0..=1.0).contains(&gamma) {
            return Err(PyValueError::new_err(
                "Please provide a probabilistic value for gamma (between 0 and 1, inclusive)!",
            ));
        }
        let edge = Py::new(
            self.py,
            Edge::make(
                self.py,
                s,
                d,
                Some(length.unwrap_or(1.0)),
                Some(gamma),
                Some(0.0),
                Some("no tag assigned yet".to_string()),
            ),
        )?;
        let euid = edge.borrow(self.py).uid;
        self.bm().add_edge(self.net.as_any(), edge.bind(self.py))?;
        Ok(euid)
    }

    fn remove_edge(&self, euid: u64) {
        self.bm().remove_edge_uid(self.py, euid);
    }
    fn remove_node(&self, uid: u64) {
        self.bm().remove_node_uid(self.py, uid);
    }
    fn set_len(&self, euid: u64, v: f64) -> PyResult<()> {
        let obj = match self.b().edges.get(&euid) {
            Some(e) => e.obj.clone_ref(self.py),
            None => return Ok(()),
        };
        Edge::set_length(obj.bind(self.py), Some(v), false, false)
    }
    fn set_gamma(&self, euid: u64, g: f64) -> PyResult<()> {
        let obj = match self.b().edges.get(&euid) {
            Some(e) => e.obj.clone_ref(self.py),
            None => return Ok(()),
        };
        Edge::set_gamma(obj.bind(self.py), g)
    }

    /// ``insert_node_in_edge(edge, node, net)``.
    fn insert_node_in_edge(&self, euid: u64, node: u64) -> PyResult<()> {
        let (a, b) = self.edge_ends(euid);
        let saved_gamma = self.edge_gamma(euid);
        self.remove_edge(euid);
        self.add_edge(a, node, None, None)?;
        if saved_gamma.is_some() && self.is_retic(b) {
            self.add_edge(node, b, None, saved_gamma)?;
        } else {
            self.add_edge(node, b, None, None)?;
        }
        Ok(())
    }

    /// ``_suppress_deg2(net, node)``.
    fn suppress_deg2(&self, uid: u64) -> PyResult<()> {
        if self.in_deg(uid) != 1 || self.out_deg(uid) != 1 {
            return Ok(());
        }
        let parent = self.parents(uid)[0];
        let child = self.children(uid)[0];
        let e_up = self.get_edge(parent, uid)?;
        let e_dn = self.get_edge(uid, child)?;
        let len_up = self.edge_len(e_up).unwrap_or(0.0);
        let len_dn = self.edge_len(e_dn).unwrap_or(0.0);
        let g_dn = self.edge_gamma(e_dn);
        let new_gamma = match g_dn {
            Some(g) if g != 0.0 => Some(g),
            _ => self.edge_gamma(e_up),
        };
        self.remove_edge(e_up);
        self.remove_edge(e_dn);
        self.remove_node(uid);
        self.add_edge(parent, child, Some(len_up + len_dn), new_gamma)?;
        Ok(())
    }

    /// ``_prune_orphan_chain(net, node)``.
    fn prune_orphan_chain(&self, start: u64) -> PyResult<()> {
        let mut stack = vec![start];
        while let Some(n) = stack.pop() {
            if !self.contains(n) {
                continue;
            }
            if self.in_deg(n) == 0 {
                continue;
            }
            if self.out_deg(n) != 0 {
                continue;
            }
            let parents = self.parents(n);
            for e in self.in_edges(n) {
                self.remove_edge(e);
            }
            self.remove_node(n);
            for p in parents {
                if !self.contains(p) {
                    continue;
                }
                if self.in_deg(p) > 0 && self.out_deg(p) == 0 {
                    stack.push(p);
                } else if self.in_deg(p) == 1 && self.out_deg(p) == 1 {
                    self.suppress_deg2(p)?;
                }
            }
        }
        Ok(())
    }

    /// ``Move._revert_if_exceeds_level``: returns true when reverted.
    fn revert_if_exceeds_level(&self, max_level: Option<i64>) -> PyResult<bool> {
        let Some(cap) = max_level else { return Ok(false) };
        if network_level(self.py, self.net)? <= cap {
            return Ok(false);
        }
        if self.in_transaction() {
            self.rollback()?;
        }
        Ok(true)
    }
}

fn ctx<'a, 'py>(net: &'a Bound<'py, Network>, rng: Bound<'py, PyAny>) -> Ctx<'a, 'py> {
    Ctx {
        py: net.py(),
        net,
        rng: Rng(rng),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Moves.  Each returns (log_hastings, update_network) unless noted.
// ═══════════════════════════════════════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (net, rng, max_reticulations=None, max_level=None))]
fn mv_add_reticulation(
    net: &Bound<'_, Network>,
    rng: Bound<'_, PyAny>,
    max_reticulations: Option<usize>,
    max_level: Option<i64>,
) -> PyResult<(f64, bool)> {
    let c = ctx(net, rng);
    if let Some(cap) = max_reticulations {
        if c.count_retics() >= cap {
            return Ok((0.0, false));
        }
    }
    c.begin();
    let all_edges = c.all_edges();
    let edge_count_pre = all_edges.len() as i64;
    let retic_count_pre = c.count_retics() as i64;
    let Some(src_e) = c.pick_edge(&all_edges)? else {
        return Ok((0.0, true));
    };
    let (a, b) = c.edge_ends(src_e);
    let avoid: FxHashSet<u64> = c.b().edges_upstream(a).into_iter().collect();
    let eligible: Vec<u64> = all_edges.iter().copied().filter(|e| !avoid.contains(e)).collect();
    if eligible.is_empty() {
        return Ok((0.0, true));
    }
    let dest_e = c.pick_edge(&eligible)?.unwrap();
    let (x, y) = c.edge_ends(dest_e);
    let len_ab = or_default(c.edge_len(src_e), 1.0);
    let len_xy = or_default(c.edge_len(dest_e), 1.0);

    let z = c.add_uid_node()?;
    let cn = c.add_uid_node()?;
    c.set_retic(cn, true);

    let split_ab = c.rng.random()?;
    let split_xy = c.rng.random()?;
    let retic_len = c.rng.exponential((0.1 * len_ab.min(len_xy)).max(1e-6))?;

    if a == x && b == y {
        let e_ab = c.get_edge(a, b)?;
        c.insert_node_in_edge(e_ab, z)?;
        c.set_len(c.get_edge(a, z)?, len_ab * split_ab)?;
        let remaining = len_ab * (1.0 - split_ab);
        c.set_len(c.get_edge(z, b)?, remaining)?;
        let e_zb = c.get_edge(z, b)?;
        c.insert_node_in_edge(e_zb, cn)?;
        let tree_zc = c.get_edge(z, cn)?;
        c.set_len(tree_zc, remaining * split_xy)?;
        c.set_gamma(tree_zc, 0.5)?;
        c.set_len(c.get_edge(cn, b)?, remaining * (1.0 - split_xy))?;
        c.add_edge(z, cn, Some(retic_len), Some(0.5))?;
        c.set_retic(cn, true);
    } else {
        let e_xy = c.get_edge(x, y)?;
        c.insert_node_in_edge(e_xy, cn)?;
        c.set_len(c.get_edge(x, cn)?, len_xy * split_xy)?;
        c.set_len(c.get_edge(cn, y)?, len_xy * (1.0 - split_xy))?;
        c.set_gamma(c.get_edge(x, cn)?, 0.5)?;
        let e_ab = c.get_edge(a, b)?;
        c.insert_node_in_edge(e_ab, z)?;
        c.set_len(c.get_edge(a, z)?, len_ab * split_ab)?;
        c.set_len(c.get_edge(z, b)?, len_ab * (1.0 - split_ab))?;
        c.add_edge(z, cn, Some(retic_len), Some(0.5))?;
        c.set_retic(cn, true);
    }
    let log_hr = add_reticulation_log_hastings(edge_count_pre, retic_count_pre, len_ab, len_xy).unwrap_or(0.0);
    if c.revert_if_exceeds_level(max_level)? {
        return Ok((0.0, true));
    }
    Ok((log_hr, true))
}

fn merged_length_if_deg2(c: &Ctx<'_, '_>, uid: u64) -> Option<f64> {
    if c.in_deg(uid) != 1 || c.out_deg(uid) != 1 {
        return None;
    }
    let in_e = c.in_edges(uid)[0];
    let out_e = c.out_edges(uid)[0];
    Some(c.edge_len(in_e).unwrap_or(0.0) + c.edge_len(out_e).unwrap_or(0.0))
}

#[pyfunction]
fn mv_remove_reticulation(net: &Bound<'_, Network>, rng: Bound<'_, PyAny>) -> PyResult<(f64, bool)> {
    let c = ctx(net, rng);
    c.begin();
    let retic_nodes: Vec<u64> = c
        .all_nodes()
        .into_iter()
        .filter(|&u| c.is_retic(u) && c.in_deg(u) == 2)
        .collect();
    if retic_nodes.is_empty() {
        return Ok((0.0, false));
    }
    let retic_count_pre = c.count_retics() as i64;
    let cn = c.pick_node(&retic_nodes)?.unwrap();
    let parents = c.parents(cn);
    if parents.len() != 2 {
        return Ok((0.0, false));
    }
    let drop_parent = c.pick_node(&parents)?.unwrap();
    let drop_edge = c.get_edge(drop_parent, cn)?;
    c.remove_edge(drop_edge);
    c.set_retic(cn, false);
    let l1 = merged_length_if_deg2(&c, cn);
    c.suppress_deg2(cn)?;
    let l2 = merged_length_if_deg2(&c, drop_parent);
    c.suppress_deg2(drop_parent)?;
    let edge_count_post = c.all_edges().len() as i64;
    let mut log_hr = 0.0;
    if let (Some(l1), Some(l2)) = (l1, l2) {
        if l1 > 0.0 && l2 > 0.0 {
            log_hr = remove_reticulation_log_hastings(edge_count_post, retic_count_pre, l1, l2).unwrap_or(0.0);
        }
    }
    Ok((log_hr, true))
}

#[pyfunction]
#[pyo3(signature = (net, rng, max_level=None))]
fn mv_flip_reticulation(net: &Bound<'_, Network>, rng: Bound<'_, PyAny>, max_level: Option<i64>) -> PyResult<(f64, bool)> {
    let c = ctx(net, rng);
    c.begin();
    let root = c.root()?;
    let retic_edges: Vec<u64> = c
        .all_edges()
        .into_iter()
        .filter(|&e| {
            let (s, d) = c.edge_ends(e);
            c.is_retic(d) && s != root && !c.is_retic(s) && c.in_deg(s) <= 1
        })
        .collect();
    if retic_edges.is_empty() {
        return Ok((0.0, false));
    }
    let retic_edge = c.pick_edge(&retic_edges)?.unwrap();
    let (z, cn) = c.edge_ends(retic_edge);
    let saved_gamma = c.edge_gamma(retic_edge);
    let saved_length = c.edge_len(retic_edge);
    c.remove_edge(retic_edge);
    c.add_edge(cn, z, saved_length, saved_gamma)?;
    if c.in_deg(cn) <= 1 {
        c.set_retic(cn, false);
    }
    if c.in_deg(z) > 1 {
        c.set_retic(z, true);
        let sg = saved_gamma.unwrap_or(0.5);
        let other = (1.0 - sg).clamp(0.0, 1.0);
        for in_e in c.in_edges(z) {
            let (s, _) = c.edge_ends(in_e);
            if s == cn {
                continue;
            }
            let g = c.edge_gamma(in_e);
            let bad = match g {
                None => true,
                Some(g) => g == 0.0 || !(0.0 < g && g < 1.0),
            };
            if bad {
                c.set_gamma(in_e, other)?;
                break;
            }
        }
    }
    if !c.b().is_acyclic_impl() {
        c.rollback()?;
        return Ok((0.0, true));
    }
    if c.revert_if_exceeds_level(max_level)? {
        return Ok((0.0, true));
    }
    Ok((0.0, true))
}

#[pyfunction]
fn mv_spr(net: &Bound<'_, Network>, rng: Bound<'_, PyAny>) -> PyResult<(f64, bool)> {
    let c = ctx(net, rng);
    c.begin();
    let prunable = |c: &Ctx<'_, '_>| -> Vec<u64> {
        c.all_edges()
            .into_iter()
            .filter(|&e| {
                let (u, v) = c.edge_ends(e);
                !c.is_retic(v) && c.in_deg(u) == 1 && c.out_deg(u) == 2
            })
            .collect()
    };
    let pr = prunable(&c);
    if pr.is_empty() {
        return Ok((0.0, false));
    }
    let p_before = pr.len() as f64;
    let prune_edge = c.pick_edge(&pr)?.unwrap();
    let (prune_parent, subtree_root) = c.edge_ends(prune_edge);
    let prune_len = or_default(c.edge_len(prune_edge), 1.0);
    let subtree: FxHashSet<u64> = c.b().subtree_uids(subtree_root).into_iter().collect();
    c.remove_edge(prune_edge);
    let ins = c.in_edges(prune_parent);
    let outs = c.out_edges(prune_parent);
    let mut e_star_len = 1.0;
    if ins.len() == 1 && outs.len() == 1 {
        e_star_len = c.edge_len(ins[0]).unwrap_or(0.0) + c.edge_len(outs[0]).unwrap_or(0.0);
    }
    let e_star_len = e_star_len.max(SPR_MIN_LEN);
    c.suppress_deg2(prune_parent)?;
    let eligible: Vec<u64> = c
        .all_edges()
        .into_iter()
        .filter(|&e| {
            let (s, d) = c.edge_ends(e);
            !subtree.contains(&s) && !subtree.contains(&d)
        })
        .collect();
    if eligible.is_empty() {
        c.rollback()?;
        return Ok((0.0, true));
    }
    let target_edge = c.pick_edge(&eligible)?.unwrap();
    let (a, b) = c.edge_ends(target_edge);
    let target_len = or_default(c.edge_len(target_edge), 1.0).max(SPR_MIN_LEN);
    let new_node = c.add_uid_node()?;
    let split = c.rng.random()?;
    c.insert_node_in_edge(target_edge, new_node)?;
    c.set_len(c.get_edge(a, new_node)?, target_len * split)?;
    c.set_len(c.get_edge(new_node, b)?, target_len * (1.0 - split))?;
    c.add_edge(new_node, subtree_root, Some(prune_len), None)?;
    let p_after = prunable(&c).len() as f64;
    let log_hr = p_before.ln() - p_after.ln() + target_len.ln() - e_star_len.ln();
    Ok((log_hr, true))
}

/// Returns ``(log_hr, touched_label, changes)``; an empty ``touched_label``
/// means the move was a no-op that only invalidates the model.
#[pyfunction]
fn mv_change_node_height(
    net: &Bound<'_, Network>,
    rng: Bound<'_, PyAny>,
    sigma_frac: f64,
) -> PyResult<(f64, Option<String>, Vec<(String, String, f64, f64)>)> {
    const EPS: f64 = 1e-6;
    let c = ctx(net, rng);
    let candidates: Vec<u64> = c
        .all_nodes()
        .into_iter()
        .filter(|&u| c.out_deg(u) > 0 && c.in_deg(u) > 0)
        .collect();
    if candidates.is_empty() {
        return Ok((0.0, None, Vec::new()));
    }
    let node = c.pick_node(&candidates)?.unwrap();
    let parent_edges = c.in_edges(node);
    let child_edges: Vec<u64> = c
        .out_edges(node)
        .into_iter()
        .filter(|&e| c.out_deg(c.edge_ends(e).1) != 0)
        .collect();
    if parent_edges.is_empty() {
        return Ok((0.0, None, Vec::new()));
    }
    let len_of = |e: u64| -> PyResult<f64> {
        c.edge_len(e).ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err("'<' not supported between instances of 'NoneType' and 'float'")
        })
    };
    let mut min_parent_len = f64::INFINITY;
    for &e in &parent_edges {
        min_parent_len = min_parent_len.min(len_of(e)?);
    }
    let has_child = !child_edges.is_empty();
    let mut min_child_len = f64::INFINITY;
    for &e in &child_edges {
        min_child_len = min_child_len.min(len_of(e)?);
    }
    let lower = -(min_parent_len - EPS);
    let upper = if has_child { min_child_len - EPS } else { min_parent_len - EPS };
    if lower >= upper {
        return Ok((0.0, None, Vec::new()));
    }
    let half_range = 0.5 * (upper - lower);
    let sigma = sigma_frac * half_range;
    let mut delta = c.rng.normal(0.0, sigma)?;
    if !(lower <= delta && delta <= upper) {
        let mut ok = false;
        for _ in 0..3 {
            delta = c.rng.normal(0.0, sigma)?;
            if lower <= delta && delta <= upper {
                ok = true;
                break;
            }
        }
        if !ok {
            delta = delta.clamp(lower, upper);
        }
    }
    let min_parent_post = min_parent_len + delta;
    let lower_rev = -(min_parent_post - EPS);
    let upper_rev = if has_child { (min_child_len - delta) - EPS } else { min_parent_post - EPS };
    let sigma_rev = sigma_frac * 0.5 * (upper_rev - lower_rev);
    let mut log_hr = 0.0;
    if upper_rev > lower_rev && sigma > 0.0 && sigma_rev > 0.0 {
        log_hr = gaussian_delta_log_hastings(delta, sigma, sigma_rev, lower, upper, lower_rev, upper_rev).unwrap_or(0.0);
    }
    let mut changes = Vec::new();
    for &e in &parent_edges {
        let (s, d) = c.edge_ends(e);
        let old = len_of(e)?;
        let new = old + delta;
        changes.push((c.label(s), c.label(d), old, new));
        c.set_len(e, new)?;
    }
    for &e in &child_edges {
        let (s, d) = c.edge_ends(e);
        let old = len_of(e)?;
        let new = old - delta;
        changes.push((c.label(s), c.label(d), old, new));
        c.set_len(e, new)?;
    }
    Ok((log_hr, Some(c.label(node)), changes))
}

/// Returns ``(log_hr, touched_label, undo_info, same_move_info)``.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn mv_change_inheritance_prob(
    net: &Bound<'_, Network>,
    rng: Bound<'_, PyAny>,
    sigma: f64,
) -> PyResult<(
    f64,
    Option<String>,
    Option<(String, String, Option<f64>, String, String, Option<f64>)>,
    Option<(String, String, f64, String, String, f64)>,
)> {
    const EPS: f64 = 0.01;
    let c = ctx(net, rng);
    let retic_nodes: Vec<u64> = c
        .all_nodes()
        .into_iter()
        .filter(|&u| c.in_deg(u) == 2 && c.is_retic(u))
        .collect();
    if retic_nodes.is_empty() {
        return Ok((0.0, None, None, None));
    }
    let node = c.pick_node(&retic_nodes)?.unwrap();
    let parent_edges = c.in_edges(node);
    if parent_edges.len() != 2 {
        return Ok((0.0, None, None, None));
    }
    let (e1, e2) = (parent_edges[0], parent_edges[1]);
    let old_g1 = c.edge_gamma(e1);
    let old_g2 = c.edge_gamma(e2);
    let current = old_g1.unwrap_or(0.5);
    let (lo, hi) = (EPS, 1.0 - EPS);
    let mut new_g1 = c.rng.normal(current, sigma)?;
    if !(lo < new_g1 && new_g1 < hi) {
        let mut ok = false;
        for _ in 0..8 {
            new_g1 = c.rng.normal(current, sigma)?;
            if lo < new_g1 && new_g1 < hi {
                ok = true;
                break;
            }
        }
        if !ok {
            return Ok((0.0, None, None, None));
        }
    }
    c.set_gamma(e1, new_g1)?;
    c.set_gamma(e2, 1.0 - new_g1)?;
    let log_hr = random_walk_truncated_log_hastings(current, new_g1, sigma, lo, hi).unwrap_or(0.0);
    let (s1, d1) = c.edge_ends(e1);
    let (s2, d2) = c.edge_ends(e2);
    let (ls1, ld1, ls2, ld2) = (c.label(s1), c.label(d1), c.label(s2), c.label(d2));
    Ok((
        log_hr,
        Some(c.label(node)),
        Some((ls1.clone(), ld1.clone(), old_g1, ls2.clone(), ld2.clone(), old_g2)),
        Some((ls1, ld1, new_g1, ls2, ld2, 1.0 - new_g1)),
    ))
}

#[pyfunction]
#[pyo3(signature = (net, rng, max_level=None))]
fn mv_change_retic_source(net: &Bound<'_, Network>, rng: Bound<'_, PyAny>, max_level: Option<i64>) -> PyResult<(f64, bool)> {
    let c = ctx(net, rng);
    c.begin();
    let retic_edges: Vec<u64> = c.all_edges().into_iter().filter(|&e| c.is_retic(c.edge_ends(e).1)).collect();
    if retic_edges.is_empty() {
        return Ok((0.0, true));
    }
    let retic_edge = c.pick_edge(&retic_edges)?.unwrap();
    let (z, cn) = c.edge_ends(retic_edge);
    let saved_gamma = c.edge_gamma(retic_edge);
    let saved_length = c.edge_len(retic_edge);
    c.remove_edge(retic_edge);
    if c.in_deg(z) > 0 && c.out_deg(z) == 0 {
        c.prune_orphan_chain(z)?;
    } else if c.in_deg(z) == 1 && c.out_deg(z) == 1 {
        c.suppress_deg2(z)?;
    }
    let mut forbidden: FxHashSet<u64> = c.b().edges_downstream(cn).into_iter().collect();
    forbidden.extend(c.in_edges(cn));
    forbidden.extend(c.out_edges(cn));
    let eligible: Vec<u64> = c.all_edges().into_iter().filter(|e| !forbidden.contains(e)).collect();
    if eligible.is_empty() {
        c.rollback()?;
        return Ok((0.0, true));
    }
    let host = c.pick_edge(&eligible)?.unwrap();
    let (a, b) = c.edge_ends(host);
    let z_new = c.add_uid_node()?;
    let host_len = c.edge_len(host).unwrap_or(1.0);
    let split = c.rng.random()?;
    c.remove_edge(host);
    c.add_edge(a, z_new, Some(host_len * split), None)?;
    c.add_edge(z_new, b, Some(host_len * (1.0 - split)), None)?;
    c.add_edge(z_new, cn, saved_length, saved_gamma)?;
    if c.in_deg(cn) > 1 {
        c.set_retic(cn, true);
    }
    if c.revert_if_exceeds_level(max_level)? {
        return Ok((0.0, true));
    }
    Ok((0.0, true))
}

#[pyfunction]
#[pyo3(signature = (net, rng, max_level=None))]
fn mv_relocate_reticulation(net: &Bound<'_, Network>, rng: Bound<'_, PyAny>, max_level: Option<i64>) -> PyResult<(f64, bool)> {
    let c = ctx(net, rng);
    c.begin();
    let retic_nodes: Vec<u64> = c.all_nodes().into_iter().filter(|&u| c.is_retic(u)).collect();
    if retic_nodes.is_empty() {
        return Ok((0.0, true));
    }
    let retic = c.pick_node(&retic_nodes)?.unwrap();
    let in_edges = c.in_edges(retic);
    if in_edges.len() != 2 {
        return Ok((0.0, true));
    }
    let children = c.children(retic);
    if children.is_empty() {
        return Ok((0.0, true));
    }
    let child = children[0];
    let e_child = c.get_edge(retic, child)?;
    let saved_child_len = c.edge_len(e_child);
    let saved_gammas: Vec<Option<f64>> = in_edges.iter().map(|&e| c.edge_gamma(e)).collect();
    let saved_lengths: Vec<Option<f64>> = in_edges.iter().map(|&e| c.edge_len(e)).collect();
    let old_parents: Vec<u64> = in_edges.iter().map(|&e| c.edge_ends(e).0).collect();
    c.remove_edge(e_child);
    for &e in &in_edges {
        c.remove_edge(e);
    }
    c.set_retic(retic, false);
    c.remove_node(retic);
    let mut seen: FxHashSet<u64> = FxHashSet::default();
    for &p in &old_parents {
        if seen.contains(&p) || !c.contains(p) {
            continue;
        }
        seen.insert(p);
        if c.in_deg(p) > 0 && c.out_deg(p) == 0 {
            c.prune_orphan_chain(p)?;
        } else if c.in_deg(p) == 1 && c.out_deg(p) == 1 {
            c.suppress_deg2(p)?;
        }
    }
    if c.in_deg(child) == 1 && c.out_deg(child) == 1 && !c.is_retic(child) {
        c.suppress_deg2(child)?;
    }
    let forbidden: FxHashSet<u64> = c.b().edges_downstream(child).into_iter().collect();
    let eligible: Vec<u64> = c.all_edges().into_iter().filter(|e| !forbidden.contains(e)).collect();
    if eligible.len() < 2 {
        c.rollback()?;
        return Ok((0.0, true));
    }
    let host1 = c.pick_edge(&eligible)?.unwrap();
    let (a1, b1) = c.edge_ends(host1);
    let eligible2: Vec<u64> = eligible.iter().copied().filter(|&e| e != host1).collect();
    if eligible2.is_empty() {
        c.rollback()?;
        return Ok((0.0, true));
    }
    let host2 = c.pick_edge(&eligible2)?.unwrap();
    let (a2, b2) = c.edge_ends(host2);
    let new_retic = c.add_uid_node()?;
    c.set_retic(new_retic, true);
    let z1 = c.add_uid_node()?;
    let h1_len = c.edge_len(host1).unwrap_or(0.0);
    let split1 = c.rng.random()?;
    c.remove_edge(host1);
    c.add_edge(a1, z1, Some(h1_len * split1), None)?;
    c.add_edge(z1, b1, Some(h1_len * (1.0 - split1)), None)?;
    let z2 = c.add_uid_node()?;
    let h2_len = c.edge_len(host2).unwrap_or(0.0);
    let split2 = c.rng.random()?;
    if !c.edge_present(host2) {
        c.rollback()?;
        return Ok((0.0, true));
    }
    c.remove_edge(host2);
    c.add_edge(a2, z2, Some(h2_len * split2), None)?;
    c.add_edge(z2, b2, Some(h2_len * (1.0 - split2)), None)?;
    c.add_edge(z1, new_retic, saved_lengths[0], saved_gammas[0])?;
    c.add_edge(z2, new_retic, saved_lengths[1], saved_gammas[1])?;
    c.add_edge(new_retic, child, saved_child_len, None)?;
    if c.revert_if_exceeds_level(max_level)? {
        return Ok((0.0, true));
    }
    Ok((0.0, true))
}

#[pyfunction]
#[pyo3(signature = (net, rng, max_level=None))]
fn mv_change_retic_dest(net: &Bound<'_, Network>, rng: Bound<'_, PyAny>, max_level: Option<i64>) -> PyResult<(f64, bool)> {
    let c = ctx(net, rng);
    c.begin();
    let retic_edges: Vec<u64> = c.all_edges().into_iter().filter(|&e| c.is_retic(c.edge_ends(e).1)).collect();
    if retic_edges.is_empty() {
        return Ok((0.0, true));
    }
    let retic_edge = c.pick_edge(&retic_edges)?.unwrap();
    let (z, cn) = c.edge_ends(retic_edge);
    let saved_gamma = c.edge_gamma(retic_edge);
    let saved_length = c.edge_len(retic_edge);
    c.remove_edge(retic_edge);
    if c.in_deg(cn) <= 1 {
        c.set_retic(cn, false);
    }
    if c.in_deg(cn) == 1 && c.out_deg(cn) == 1 {
        c.suppress_deg2(cn)?;
    }
    let mut forbidden: FxHashSet<u64> = c.b().edges_upstream(z).into_iter().collect();
    forbidden.extend(c.in_edges(z));
    forbidden.extend(c.out_edges(z));
    let eligible: Vec<u64> = c.all_edges().into_iter().filter(|e| !forbidden.contains(e)).collect();
    if eligible.is_empty() {
        c.rollback()?;
        return Ok((0.0, true));
    }
    let host = c.pick_edge(&eligible)?.unwrap();
    let (a, b) = c.edge_ends(host);
    let c_new = c.add_uid_node()?;
    c.set_retic(c_new, true);
    let host_len = c.edge_len(host).unwrap_or(1.0);
    let split = c.rng.random()?;
    let sg = saved_gamma.unwrap_or(0.5);
    let gamma_a = (1.0 - sg).clamp(0.0, 1.0);
    c.remove_edge(host);
    c.add_edge(a, c_new, Some(host_len * split), Some(gamma_a))?;
    c.add_edge(c_new, b, Some(host_len * (1.0 - split)), None)?;
    c.add_edge(z, c_new, saved_length, Some(sg))?;
    if c.revert_if_exceeds_level(max_level)? {
        return Ok((0.0, true));
    }
    Ok((0.0, true))
}

// ── SwitchParentage (PSPP) ────────────────────────────────────────────

fn sp_cleanup_node(c: &Ctx<'_, '_>, node: u64) -> PyResult<()> {
    let in_deg = c.in_deg(node);
    let out_deg = c.out_deg(node);
    if let Ok(root) = c.root() {
        if node == root {
            return Ok(());
        }
    } else {
        return Err(crate::util::network_error(
            c.py,
            "There are no roots in this network. There is either a cycle, or nothing has been added and this is an empty network.",
        ));
    }
    if in_deg == 1 && out_deg == 1 {
        let parent = c.parents(node)[0];
        let child = c.children(node)[0];
        let pe = c.get_edge(parent, node)?;
        let ce = c.get_edge(node, child)?;
        c.remove_edge(pe);
        c.remove_edge(ce);
        c.remove_node(node);
        c.add_edge(parent, child, None, None)?;
        if c.is_retic(parent) && c.in_deg(parent) == 1 {
            c.set_retic(parent, false);
            sp_cleanup_node(c, parent)?;
        }
    } else if in_deg == 0 && out_deg == 1 {
        let child = c.children(node)[0];
        let ce = c.get_edge(node, child)?;
        c.remove_edge(ce);
        c.remove_node(node);
    } else if in_deg == 0 && out_deg == 0 {
        c.remove_node(node);
    } else if in_deg >= 1 && out_deg == 0 {
        let parents = c.parents(node);
        for &p in &parents {
            let e = c.get_edge(p, node)?;
            c.remove_edge(e);
        }
        c.set_retic(node, false);
        c.remove_node(node);
        for &p in &parents {
            if c.is_retic(p) && c.in_deg(p) == 1 {
                c.set_retic(p, false);
            }
            sp_cleanup_node(c, p)?;
        }
    }
    Ok(())
}

fn sp_ploidy(c: &Ctx<'_, '_>) -> PyResult<FxHashMap<String, usize>> {
    let leaves = c.b().leaf_uids();
    let mut out = FxHashMap::default();
    for l in leaves {
        let count = c.b().subgenome_count_uid(c.py, l)?;
        out.insert(c.label(l), count);
    }
    Ok(out)
}

#[pyfunction]
fn mv_switch_parentage(net: &Bound<'_, Network>, rng: Bound<'_, PyAny>) -> PyResult<()> {
    let c = ctx(net, rng);
    c.begin();
    let pre_ploidy = sp_ploidy(&c)?;

    let root = c.root()?;
    let non_root: Vec<u64> = c.all_nodes().into_iter().filter(|&u| u != root).collect();
    if non_root.is_empty() {
        return Ok(());
    }
    let node_2_change = c.choice(&non_root)?.unwrap();
    let node_pars = c.parents(node_2_change);
    if node_pars.len() == 1 && node_pars[0] == root {
        let root_kids = c.children(root);
        let other: Vec<u64> = root_kids.into_iter().filter(|&n| n != node_2_change).collect();
        if let Some(&first) = other.first() {
            if c.out_deg(first) == 0 {
                return Ok(());
            }
        }
    }
    let target = c.b().subgenome_count_uid(c.py, node_2_change)?;
    let in_edges = c.in_edges(node_2_change);
    if in_edges.is_empty() {
        return Ok(());
    }
    let edge_2_remove = c.choice(&in_edges)?.unwrap();
    // _delete_parent_edge
    {
        let (parent_node, target_node) = c.edge_ends(edge_2_remove);
        c.remove_edge(edge_2_remove);
        if c.is_retic(target_node) && c.in_deg(target_node) == 1 {
            c.set_retic(target_node, false);
        }
        sp_cleanup_node(&c, parent_node)?;
    }
    let original = node_2_change;
    let mut cur_ct = if c.in_deg(original) >= 1 {
        c.b().subgenome_count_uid(c.py, original)?
    } else {
        0
    };
    // (edge id, src, dest): entries may outlive the edge in the network, as
    // the reference keeps stale Edge objects whose endpoints stay readable.
    let mut valid_attachment: Vec<(u64, u64, u64)> = Vec::new();
    let mut first_iter = true;
    for _ in 0..100 {
        if cur_ct == target {
            break;
        }
        let downstream;
        if !first_iter {
            if valid_attachment.is_empty() {
                break;
            }
            let Some((branch, bs, bd)) = c.choice(&valid_attachment)? else { break };
            let intermediate = c.add_uid_node()?;
            c.remove_edge(branch);
            valid_attachment.retain(|&(e, _, _)| e != branch);
            let e1 = c.add_edge(bs, intermediate, None, None)?;
            let e2 = c.add_edge(intermediate, bd, None, None)?;
            valid_attachment.push((e1, bs, intermediate));
            valid_attachment.push((e2, intermediate, bd));
            downstream = intermediate;
        } else {
            downstream = original;
        }
        let mut bfs_starts: Vec<u64> = c
            .all_nodes()
            .into_iter()
            .filter(|&n| c.in_deg(n) == 0 && c.out_deg(n) != 0)
            .collect();
        if bfs_starts.len() > 1 && bfs_starts.contains(&original) {
            bfs_starts.retain(|&n| n != original);
        }
        if bfs_starts.is_empty() {
            break;
        }
        let delta = target as f64 - cur_ct as f64;
        let buckets = {
            let n = c.b();
            let ds = n.node_obj(downstream).clone_ref(c.py);
            let st = n.node_obj(bfs_starts[0]).clone_ref(c.py);
            n.subgenome_buckets(c.py, Some(ds.bind(c.py).as_any()), delta, Some(st.bind(c.py).as_any()))?
        };
        if buckets.is_empty() {
            break;
        }
        let keys: Vec<usize> = buckets.iter().map(|(k, _)| *k).collect();
        let Some(key) = c.choice(&keys)? else { break };
        let edge_list: Vec<u64> = buckets.iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone()).unwrap_or_default();
        if edge_list.is_empty() {
            break;
        }
        let Some(new_edge) = c.choice(&edge_list)? else { break };
        let connector = c.add_uid_node()?;
        let (ns, nd) = c.edge_ends(new_edge);
        let e_to_child = c.add_edge(connector, nd, None, None)?;
        let _e_from_parent = c.add_edge(ns, connector, None, None)?;
        let e_to_target = c.add_edge(connector, downstream, None, None)?;
        let _ = e_to_child;
        valid_attachment.push((e_to_target, connector, downstream));
        c.remove_edge(new_edge);
        if c.in_deg(downstream) > 1 {
            c.set_retic(downstream, true);
        }
        cur_ct = c.b().subgenome_count_uid(c.py, original)?;
        first_iter = false;
    }
    Network::clean_impl(c.net, [true, true, true])?;
    for u in c.all_nodes() {
        let r = c.in_deg(u) > 1;
        c.set_retic(u, r);
    }
    let post = sp_ploidy(&c);
    let preserved = match post {
        Ok(p) => p == pre_ploidy,
        Err(_) => false,
    };
    if !preserved {
        c.rollback()?;
    }
    Ok(())
}

// ── Python-visible helpers ────────────────────────────────────────────

#[pyfunction]
fn insert_node_in_edge(edge: &Bound<'_, Edge>, node: &Bound<'_, Node>, net: &Bound<'_, Network>) -> PyResult<()> {
    let py = net.py();
    let euid = edge.borrow().uid;
    let node_uid = net.borrow().uid_of_bound(node).ok_or_else(|| {
        crate::util::network_error(py, "Tried to add an edge between two nodes, at least one of which does not belong to this network.")
    })?;
    let rng = py.None().into_bound(py);
    let c = ctx(net, rng);
    if !c.edge_present(euid) {
        // The reference removes a missing edge silently, then adds the two
        // new edges; mirror that.
        let (a, b) = {
            let e = edge.borrow();
            let n = net.borrow();
            (n.uid_of(py, &e.src), n.uid_of(py, &e.dest))
        };
        let (Some(a), Some(b)) = (a, b) else {
            return Err(crate::util::network_error(py, "Tried to add an edge between two nodes, at least one of which does not belong to this network."));
        };
        let saved_gamma = edge.borrow().gamma;
        c.add_edge(a, node_uid, None, None)?;
        if saved_gamma.is_some() && c.is_retic(b) {
            c.add_edge(node_uid, b, None, saved_gamma)?;
        } else {
            c.add_edge(node_uid, b, None, None)?;
        }
        return Ok(());
    }
    c.insert_node_in_edge(euid, node_uid)
}

#[pyfunction]
#[pyo3(name = "_suppress_deg2")]
fn suppress_deg2_py(net: &Bound<'_, Network>, node: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = net.py();
    let Some(uid) = net.borrow().uid_of_any(node) else { return Ok(()) };
    let c = ctx(net, py.None().into_bound(py));
    c.suppress_deg2(uid)
}

#[pyfunction]
#[pyo3(name = "_prune_orphan_chain")]
fn prune_orphan_chain_py(net: &Bound<'_, Network>, node: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = net.py();
    let Some(uid) = net.borrow().uid_of_any(node) else { return Ok(()) };
    let c = ctx(net, py.None().into_bound(py));
    c.prune_orphan_chain(uid)
}

#[pyfunction]
fn add_reticulation_log_hastings_py(edge_count_pre: i64, retic_count_pre: i64, l1: f64, l2: f64) -> PyResult<f64> {
    add_reticulation_log_hastings(edge_count_pre, retic_count_pre, l1, l2)
}

#[pyfunction]
fn remove_reticulation_log_hastings_py(edge_count_post: i64, retic_count_pre: i64, l1: f64, l2: f64) -> PyResult<f64> {
    remove_reticulation_log_hastings(edge_count_post, retic_count_pre, l1, l2)
}

#[pyfunction]
fn normal_log_pdf_py(x: f64, mu: f64, sigma: f64) -> PyResult<f64> {
    normal_log_pdf(x, mu, sigma)
}

#[pyfunction]
fn standard_normal_cdf_py(x: f64) -> f64 {
    standard_normal_cdf(x)
}

#[pyfunction]
fn truncated_normal_log_norm_py(mu: f64, sigma: f64, lo: f64, hi: f64) -> PyResult<f64> {
    truncated_normal_log_norm(mu, sigma, lo, hi)
}

#[pyfunction]
fn random_walk_truncated_log_hastings_py(old: f64, new: f64, sigma: f64, lo: f64, hi: f64) -> PyResult<f64> {
    random_walk_truncated_log_hastings(old, new, sigma, lo, hi)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn gaussian_delta_log_hastings_py(
    delta: f64,
    sigma_fwd: f64,
    sigma_rev: f64,
    lo_fwd: f64,
    hi_fwd: f64,
    lo_rev: f64,
    hi_rev: f64,
) -> PyResult<f64> {
    gaussian_delta_log_hastings(delta, sigma_fwd, sigma_rev, lo_fwd, hi_fwd, lo_rev, hi_rev)
}

#[pyfunction]
fn weighted_choice_log_hastings_py(fwd_weight: f64, fwd_total: f64, rev_weight: f64, rev_total: f64) -> PyResult<f64> {
    weighted_choice_log_hastings(fwd_weight, fwd_total, rev_weight, rev_total)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mv_add_reticulation, m)?)?;
    m.add_function(wrap_pyfunction!(mv_remove_reticulation, m)?)?;
    m.add_function(wrap_pyfunction!(mv_flip_reticulation, m)?)?;
    m.add_function(wrap_pyfunction!(mv_spr, m)?)?;
    m.add_function(wrap_pyfunction!(mv_change_node_height, m)?)?;
    m.add_function(wrap_pyfunction!(mv_change_inheritance_prob, m)?)?;
    m.add_function(wrap_pyfunction!(mv_change_retic_source, m)?)?;
    m.add_function(wrap_pyfunction!(mv_relocate_reticulation, m)?)?;
    m.add_function(wrap_pyfunction!(mv_change_retic_dest, m)?)?;
    m.add_function(wrap_pyfunction!(mv_switch_parentage, m)?)?;
    m.add_function(wrap_pyfunction!(insert_node_in_edge, m)?)?;
    m.add_function(wrap_pyfunction!(suppress_deg2_py, m)?)?;
    m.add_function(wrap_pyfunction!(prune_orphan_chain_py, m)?)?;
    m.add_function(wrap_pyfunction!(add_reticulation_log_hastings_py, m)?)?;
    m.add_function(wrap_pyfunction!(remove_reticulation_log_hastings_py, m)?)?;
    m.add_function(wrap_pyfunction!(normal_log_pdf_py, m)?)?;
    m.add_function(wrap_pyfunction!(standard_normal_cdf_py, m)?)?;
    m.add_function(wrap_pyfunction!(truncated_normal_log_norm_py, m)?)?;
    m.add_function(wrap_pyfunction!(random_walk_truncated_log_hastings_py, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_delta_log_hastings_py, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_choice_log_hastings_py, m)?)?;
    Ok(())
}
