//! Coalescent simulation of gene trees under the multispecies network
//! coalescent (MSNC) and sequence evolution down gene trees: the compiled
//! core behind ``phynetpy._sim_seq``.
//!
//! Every random draw goes back to the caller's ``numpy.random.Generator``
//! in the order the reference implementation drew it -- ``exponential`` and
//! ``choice(u, 2, replace=False)`` per coalescence, ``random`` per lineage
//! routed through a reticulation, ``choice(4, size, p=pi)`` for a root
//! sequence and one vectorised ``random(n_sites)`` per branch -- so a seeded
//! run reproduces the Python stream exactly.
//!
//! A genealogy is built as a compact arena ([`GTree`]) that renders the
//! reference's Newick string (branch lengths formatted ``:.10f``); the
//! gene-tree simulator parses that string into a ``Network`` exactly as the
//! reference did, while the marker simulator (``sim::snp``) walks the arena
//! directly instead of materialising a ``Network`` per site.

use std::cmp::Ordering;
use std::fmt::Write as _;

use numpy::{AllowTypeChange, PyArray1, PyArrayLike1, PyArrayLike2};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyTuple};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::newick::parse_newick;
use crate::graph::{Network, Node};
use crate::util::{network_class, network_error, py_float_repr, warn};

/// Nucleotide order shared with the likelihood kernels: ``A,C,G,T``.
const DNA: &[u8; 4] = b"ACGT";

// ═══════════════════════════════════════════════════════════════════════
// Branch-length units (phynetpy._units)
// ═══════════════════════════════════════════════════════════════════════

static UNITS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

fn units_module<'py>(py: Python<'py>) -> PyResult<&'py Bound<'py, PyAny>> {
    let m = UNITS.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py.import("phynetpy._units")?.into_any().unbind())
    })?;
    Ok(m.bind(py))
}

/// ``BranchLengthUnit.SUBSTITUTIONS_PER_SITE``.
pub(crate) fn subst_unit<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    units_module(py)?
        .getattr("BranchLengthUnit")?
        .getattr("SUBSTITUTIONS_PER_SITE")
}

/// ``require_branch_length_unit(net, SUBSTITUTIONS_PER_SITE, context=...)``.
pub(crate) fn require_subst_unit(py: Python<'_>, net: &Bound<'_, PyAny>, context: &str) -> PyResult<()> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("context", context)?;
    units_module(py)?
        .getattr("require_branch_length_unit")?
        .call((net, subst_unit(py)?), Some(&kwargs))?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// RNG bridge
// ═══════════════════════════════════════════════════════════════════════

fn int_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    match obj.extract::<PyArrayLike1<i64, AllowTypeChange>>() {
        Ok(a) => Ok(a.as_array().iter().copied().collect()),
        Err(e) => obj.extract::<Vec<i64>>().map_err(|_| e),
    }
}

fn float_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    match obj.extract::<PyArrayLike1<f64, AllowTypeChange>>() {
        Ok(a) => Ok(a.as_array().iter().copied().collect()),
        Err(e) => obj.extract::<Vec<f64>>().map_err(|_| e),
    }
}

/// The caller's ``numpy.random.Generator`` (or any object exposing the same
/// method names), called in the reference order so seeded runs reproduce.
pub(crate) struct PyRng<'py>(Bound<'py, PyAny>);

impl<'py> PyRng<'py> {
    pub fn new(obj: Bound<'py, PyAny>) -> Self {
        PyRng(obj)
    }

    /// ``rng.random()``.
    pub fn random(&self) -> PyResult<f64> {
        self.0.call_method0("random")?.extract()
    }

    /// ``rng.exponential(scale)``.
    pub fn exponential(&self, scale: f64) -> PyResult<f64> {
        self.0.call_method1("exponential", (scale,))?.extract()
    }

    /// ``rng.uniform(lo, hi)``.
    pub fn uniform(&self, lo: f64, hi: f64) -> PyResult<f64> {
        self.0.call_method1("uniform", (lo, hi))?.extract()
    }

    /// ``rng.choice(n, size=2, replace=False)`` as two indices into ``0..n``.
    pub fn choice_pair(&self, n: usize) -> PyResult<(usize, usize)> {
        let py = self.0.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item("size", 2)?;
        kwargs.set_item("replace", false)?;
        let out = self.0.call_method("choice", (n,), Some(&kwargs))?;
        let v = int_vec(&out)?;
        let (Some(&i), Some(&j)) = (v.first(), v.get(1)) else {
            return Err(PyValueError::new_err(
                "rng.choice(n, size=2, replace=False) returned fewer than two indices.",
            ));
        };
        let to_index = |x: i64| -> PyResult<usize> {
            if x < 0 || x as usize >= n {
                Err(PyIndexError::new_err(format!(
                    "rng.choice returned index {x}, outside 0..{n}."
                )))
            } else {
                Ok(x as usize)
            }
        };
        Ok((to_index(i)?, to_index(j)?))
    }

    /// ``rng.random(n)``.
    pub fn random_vec(&self, n: usize) -> PyResult<Vec<f64>> {
        let out = self.0.call_method1("random", (n,))?;
        let v = float_vec(&out)?;
        if v.len() != n {
            return Err(PyValueError::new_err(format!(
                "rng.random({n}) returned {} values.",
                v.len()
            )));
        }
        Ok(v)
    }

    /// ``rng.choice(4, size=n, p=p)``.
    pub fn choice4(&self, n: usize, p: &Bound<'py, PyAny>) -> PyResult<Vec<i64>> {
        let py = self.0.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item("size", n)?;
        kwargs.set_item("p", p)?;
        let out = self.0.call_method("choice", (4usize,), Some(&kwargs))?;
        let v = int_vec(&out)?;
        if v.len() != n {
            return Err(PyValueError::new_err(format!(
                "rng.choice(4, size={n}) returned {} values.",
                v.len()
            )));
        }
        Ok(v)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Per-branch population mutation rates (phynetpy._units.resolve_branch_theta)
// ═══════════════════════════════════════════════════════════════════════

enum ThetaVal {
    Ok(f64),
    /// A value ``_positive_theta`` rejects; the error is raised when (and
    /// only when) the key is looked up, as in the reference.
    Bad(String),
}

type ThetaKey = (Option<String>, String);

/// ``resolve_branch_theta`` with the user's mapping converted once.
///
/// Keys are child labels or ``(parent_label, child_label)`` pairs; stable
/// edge keys take precedence over child keys, and the population above the
/// root resolves through ``("__root__", label)``, ``"__root__"`` and finally
/// the root label itself.
pub(crate) struct ThetaResolver {
    default: f64,
    table: Vec<(ThetaKey, ThetaVal)>,
}

fn to_float(py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<f64> {
    match v.extract::<f64>() {
        Ok(x) => Ok(x),
        Err(_) => py.import("builtins")?.getattr("float")?.call1((v,))?.extract(),
    }
}

fn theta_key(k: &Bound<'_, PyAny>) -> Option<ThetaKey> {
    if let Ok(s) = k.extract::<String>() {
        return Some((None, s));
    }
    if let Ok(t) = k.cast::<PyTuple>() {
        if t.len() == 2 {
            let a = t.get_item(0).ok()?.extract::<String>().ok()?;
            let b = t.get_item(1).ok()?.extract::<String>().ok()?;
            return Some((Some(a), b));
        }
    }
    None
}

impl ThetaResolver {
    pub fn new(py: Python<'_>, theta: f64, branch_thetas: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if !theta.is_finite() || theta <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "default theta must be finite and positive; got {}.",
                py_float_repr(py, theta)?
            )));
        }
        let mut table = Vec::new();
        if let Some(bt) = branch_thetas {
            if !bt.is_none() && bt.is_truthy()? {
                for item in bt.call_method0("items")?.try_iter()? {
                    let (k, v): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
                    let Some(key) = theta_key(&k) else {
                        continue;
                    };
                    let name = format!("branch theta for {}", k.repr()?);
                    let val = match to_float(py, &v) {
                        Ok(x) if x.is_finite() && x > 0.0 => ThetaVal::Ok(x),
                        Ok(x) => ThetaVal::Bad(format!(
                            "{name} must be finite and positive; got {}.",
                            py_float_repr(py, x)?
                        )),
                        Err(_) => ThetaVal::Bad(format!("{name} must be numeric; got {}.", v.repr()?)),
                    };
                    table.push((key, val));
                }
            }
        }
        Ok(ThetaResolver { default: theta, table })
    }

    fn lookup(&self, parent: Option<&str>, child: &str) -> Option<&ThetaVal> {
        self.table
            .iter()
            .find(|((p, c), _)| p.as_deref() == parent && c == child)
            .map(|(_, v)| v)
    }

    /// Theta for the population branch above ``child``.
    pub fn resolve(&self, child: &str, parent: Option<&str>) -> PyResult<f64> {
        if self.table.is_empty() {
            return Ok(self.default);
        }
        let candidates: Vec<(Option<&str>, &str)> = match parent {
            Some(p) => vec![(Some(p), child), (None, child)],
            None => vec![(Some("__root__"), child), (None, "__root__"), (None, child)],
        };
        for (p, c) in candidates {
            if let Some(v) = self.lookup(p, c) {
                return match v {
                    ThetaVal::Ok(x) => Ok(*x),
                    ThetaVal::Bad(msg) => Err(PyValueError::new_err(msg.clone())),
                };
            }
        }
        Ok(self.default)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Genealogy arena
// ═══════════════════════════════════════════════════════════════════════

struct GNode {
    label: String,
    height: f64,
    children: Option<(usize, usize)>,
}

/// A simulated genealogy: every lineage the coalescent created, as an arena.
pub(crate) struct GTree {
    nodes: Vec<GNode>,
}

impl GTree {
    fn new() -> Self {
        GTree { nodes: Vec::new() }
    }

    fn leaf(&mut self, label: String, height: f64) -> usize {
        self.nodes.push(GNode {
            label,
            height,
            children: None,
        });
        self.nodes.len() - 1
    }

    fn merge(&mut self, a: usize, b: usize, t: f64, label: String) -> usize {
        self.nodes.push(GNode {
            label,
            height: t,
            children: Some((a, b)),
        });
        self.nodes.len() - 1
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn label(&self, i: usize) -> &str {
        self.nodes.get(i).map(|n| n.label.as_str()).unwrap_or("")
    }

    pub fn children(&self, i: usize) -> Option<(usize, usize)> {
        self.nodes.get(i).and_then(|n| n.children)
    }

    /// The branch length above ``child`` exactly as the reference's
    /// ``:.10f`` Newick round trip yields it.
    pub fn branch_len(&self, parent: usize, child: usize) -> f64 {
        let (Some(p), Some(c)) = (self.nodes.get(parent), self.nodes.get(child)) else {
            return 0.0;
        };
        let d = p.height - c.height;
        format!("{d:.10}").parse::<f64>().unwrap_or(d)
    }

    fn write_newick(&self, i: usize, out: &mut String) {
        let Some(n) = self.nodes.get(i) else {
            return;
        };
        match n.children {
            None => out.push_str(&n.label),
            Some((a, b)) => {
                out.push('(');
                self.write_newick(a, out);
                let _ = write!(out, ":{:.10},", n.height - self.nodes[a].height);
                self.write_newick(b, out);
                let _ = write!(out, ":{:.10})", n.height - self.nodes[b].height);
                out.push_str(&n.label);
            }
        }
    }

    /// The subtree below ``root`` in the reference's Newick form (no ``;``).
    pub fn newick(&self, root: usize) -> String {
        let mut s = String::new();
        self.write_newick(root, &mut s);
        s
    }
}

/// The coalescent on ``pool`` over ``[t_start, t_end)``: each pair merges at
/// rate ``2/theta``; ``t_end = None`` is the infinite root branch.
/// ``merge(a, b, t, label)`` builds the parent lineage.  Draw for draw the
/// reference's ``_coalesce_within``.
fn coalesce_within(
    pool_in: &[usize],
    t_start: f64,
    t_end: Option<f64>,
    theta: f64,
    rng: &PyRng<'_>,
    counter: &mut usize,
    merge: &mut dyn FnMut(usize, usize, f64, String) -> PyResult<usize>,
) -> PyResult<Vec<usize>> {
    let mut pool = pool_in.to_vec();
    let mut t = t_start;
    while pool.len() > 1 {
        let u = pool.len();
        let rate = (u * (u - 1)) as f64 / theta;
        let wait = rng.exponential(1.0 / rate)?;
        if let Some(end) = t_end {
            if t + wait >= end {
                break;
            }
        }
        t += wait;
        let (i, j) = rng.choice_pair(u)?;
        let (a, b) = (pool[i], pool[j]);
        let label = format!("i{}", *counter);
        *counter += 1;
        let merged = merge(a, b, t, label)?;
        // Remove the two children (higher index first) and add the parent.
        let (hi, lo) = if i > j { (i, j) } else { (j, i) };
        pool.remove(hi);
        if lo != hi {
            pool.remove(lo);
        }
        pool.push(merged);
    }
    Ok(pool)
}

// ═══════════════════════════════════════════════════════════════════════
// Species-network sweep
// ═══════════════════════════════════════════════════════════════════════

fn node_label(py: Python<'_>, net: &Network, uid: u64) -> String {
    net.nodes
        .get(&uid)
        .map(|e| e.obj.borrow(py).name.clone())
        .unwrap_or_default()
}

/// Ultrametric height of every node from child branch lengths (missing
/// lengths count as 0), as ``GraphUtils._node_heights`` computes them.
pub(crate) fn node_heights(py: Python<'_>, net: &Network) -> PyResult<FxHashMap<u64, f64>> {
    fn rec(
        py: Python<'_>,
        net: &Network,
        uid: u64,
        memo: &mut FxHashMap<u64, f64>,
        active: &mut FxHashSet<u64>,
    ) -> PyResult<f64> {
        if let Some(&h) = memo.get(&uid) {
            return Ok(h);
        }
        if !active.insert(uid) {
            return Err(network_error(
                py,
                "Cannot compute node heights: the network contains a cycle.",
            ));
        }
        let mut best = 0.0_f64;
        for child in net.children(uid) {
            let euid = net.first_edge_between(py, uid, child)?;
            let length = net
                .edges
                .get(&euid)
                .and_then(|e| e.obj.borrow(py).length)
                .unwrap_or(0.0);
            let h = rec(py, net, child, memo, active)? + length;
            if h > best {
                best = h;
            }
        }
        active.remove(&uid);
        memo.insert(uid, best);
        Ok(best)
    }
    let mut memo = FxHashMap::default();
    let mut active = FxHashSet::default();
    let uids: Vec<u64> = net.nodes.keys().copied().collect();
    for uid in uids {
        rec(py, net, uid, &mut memo, &mut active)?;
    }
    Ok(memo)
}

/// The two parent edges of reticulation ``uid`` (sorted by source label)
/// with normalised inheritance probabilities; see
/// ``_sim_seq._reticulation_parent_gammas``.
pub(crate) fn retic_parent_gammas(py: Python<'_>, net: &Network, uid: u64) -> PyResult<[(u64, f64); 2]> {
    let mut ins = net.in_edge_uids(uid);
    ins.sort_by_cached_key(|e| net.edges.get(e).map(|x| node_label(py, net, x.su)).unwrap_or_default());
    if ins.len() != 2 {
        let label = node_label(py, net, uid).into_pyobject(py)?.repr()?.to_string();
        return Err(PyValueError::new_err(format!(
            "Reticulation {label} must have exactly two parent edges."
        )));
    }
    let gamma = |e: u64| net.edges.get(&e).and_then(|x| x.obj.borrow(py).gamma).unwrap_or(0.0);
    let (e0, e1) = (ins[0], ins[1]);
    let (g0, g1) = (gamma(e0), gamma(e1));
    let total = g0 + g1;
    if (total - 1.0).abs() < 1e-9 {
        return Ok([(e0, g0), (e1, g1)]);
    }
    if total <= 0.0 {
        return Ok([(e0, 0.5), (e1, 0.5)]);
    }
    // One edge annotated (the other left at the 0.0 default): treat the set
    // value as gamma and give its complement to the sibling edge.
    if g1 == 0.0 {
        return Ok([(e0, g0), (e1, 1.0 - g0)]);
    }
    if g0 == 0.0 {
        return Ok([(e0, 1.0 - g1), (e1, g1)]);
    }
    // Both set but not normalised: renormalise.
    Ok([(e0, g0 / total), (e1, g1 / total)])
}

enum Exit {
    /// The root: coalesce the survivors on the infinite branch.
    Root { theta: f64 },
    /// A reticulation: each lineage ascends ``e_a`` with probability
    /// ``gamma_a``, else ``e_b``.
    Retic { e_a: u64, gamma_a: f64, e_b: u64 },
    /// A tree node: everything enters the single parent branch.
    Tree { parent_edge: u64 },
}

struct Step {
    height: f64,
    /// ``(edge uid, child height, theta on that branch)`` per child, in
    /// child-label order.
    children: Vec<(u64, f64, f64)>,
    exit: Exit,
}

struct Seed {
    labels: Vec<String>,
    /// The in-edge the sampled lineages wait at, or ``None`` for the
    /// degenerate single-node network (coalesce straight to the root).
    target: Option<u64>,
    theta_root: Option<f64>,
}

/// Everything about one ``simulate_gene_tree`` call that does not depend on
/// the random draws: the validated sampling scheme, node heights, the
/// youngest-to-oldest visiting order and the per-branch thetas.  Built once
/// and replayed per locus or marker site.
pub(crate) struct GenePlan {
    seeds: Vec<Seed>,
    steps: Vec<Step>,
}

fn label_of(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    match obj.extract::<String>() {
        Ok(s) => Ok(s),
        Err(_) => Ok(obj.str()?.to_string()),
    }
}

impl GenePlan {
    pub fn new(
        py: Python<'_>,
        net: &Bound<'_, Network>,
        alleles: &Bound<'_, PyAny>,
        theta: f64,
        branch_thetas: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        require_subst_unit(py, net.as_any(), "MSNC gene-tree simulation")?;
        let this = net.borrow();
        let lbl = |u: u64| node_label(py, &this, u);

        let mut leaves = this.leaf_uids();
        leaves.sort_by_cached_key(|u| lbl(*u));
        if leaves.is_empty() && this.nodes.len() == 1 {
            leaves = this.nodes.keys().copied().collect();
        }
        let leaf_names: FxHashSet<String> = leaves.iter().map(|u| lbl(*u)).collect();

        // The sampling scheme, validated as the reference validates it.
        let mut entries: Vec<(String, Bound<'_, PyAny>)> = Vec::new();
        for item in alleles.call_method0("items")?.try_iter()? {
            let (k, v): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
            entries.push((label_of(&k)?, v));
        }
        let mut unknown: Vec<&str> = entries
            .iter()
            .map(|(k, _)| k.as_str())
            .filter(|k| !leaf_names.contains(*k))
            .collect();
        if !unknown.is_empty() {
            unknown.sort_unstable();
            unknown.dedup();
            let shown = PyList::new(py, &unknown)?.repr()?;
            return Err(PyValueError::new_err(format!(
                "allele mapping contains unknown species: {shown}."
            )));
        }
        for (_, v) in &entries {
            if !v.is_truthy()? {
                return Err(PyValueError::new_err(
                    "every mapped species must have at least one allele.",
                ));
            }
        }
        let mut allele_map: FxHashMap<String, Vec<String>> = FxHashMap::default();
        let mut flat: Vec<String> = Vec::new();
        for (k, v) in &entries {
            let mut labels = Vec::new();
            for lab in v.try_iter()? {
                labels.push(label_of(&lab?)?);
            }
            flat.extend(labels.iter().cloned());
            allele_map.insert(k.clone(), labels);
        }
        let distinct: FxHashSet<&String> = flat.iter().collect();
        if distinct.len() != flat.len() {
            return Err(PyValueError::new_err(
                "allele labels must be unique across species.",
            ));
        }

        let heights = node_heights(py, &this)?;
        let height = |u: u64| heights.get(&u).copied().unwrap_or(0.0);
        let thetas = ThetaResolver::new(py, theta, branch_thetas)?;

        // Seed the leaves.
        let mut seeds = Vec::with_capacity(leaves.len());
        for &leaf in &leaves {
            let label = lbl(leaf);
            let labels = allele_map
                .get(&label)
                .cloned()
                .unwrap_or_else(|| vec![label.clone()]);
            let mut ins = this.in_edge_uids(leaf);
            ins.sort_by_cached_key(|e| this.edges.get(e).map(|x| lbl(x.su)).unwrap_or_default());
            let (target, theta_root) = match ins.first() {
                Some(&e) => (Some(e), None),
                None => (None, Some(thetas.resolve(&label, None)?)),
            };
            seeds.push(Seed {
                labels,
                target,
                theta_root,
            });
        }

        // Every non-leaf node from youngest to oldest.
        let mut internal: Vec<u64> = this
            .nodes
            .keys()
            .copied()
            .filter(|&u| this.out_deg(u) > 0)
            .collect();
        internal.sort_by(|a, b| {
            height(*a)
                .partial_cmp(&height(*b))
                .unwrap_or(Ordering::Equal)
                .then_with(|| lbl(*a).cmp(&lbl(*b)))
        });

        let mut steps = Vec::with_capacity(internal.len());
        for &v in &internal {
            let v_label = lbl(v);
            let mut kids = this.children(v);
            kids.sort_by_cached_key(|c| lbl(*c));
            let mut children = Vec::with_capacity(kids.len());
            for c in kids {
                let edge = this.edge_between(py, v, c, None, None)?;
                let euid = edge.borrow(py).uid;
                children.push((euid, height(c), thetas.resolve(&lbl(c), Some(&v_label))?));
            }
            let mut parents = this.parents(v);
            parents.sort_by_cached_key(|p| lbl(*p));
            let exit = if parents.is_empty() {
                Exit::Root {
                    theta: thetas.resolve(&v_label, None)?,
                }
            } else if this.node_obj(v).borrow(py).is_retic {
                let [(e_a, gamma_a), (e_b, _)] = retic_parent_gammas(py, &this, v)?;
                Exit::Retic { e_a, gamma_a, e_b }
            } else {
                let edge = this.edge_between(py, parents[0], v, None, None)?;
                let parent_edge = edge.borrow(py).uid;
                Exit::Tree { parent_edge }
            };
            steps.push(Step {
                height: height(v),
                children,
                exit,
            });
        }
        Ok(GenePlan { seeds, steps })
    }

    /// Draw one genealogy.  Returns the arena and the index of its root.
    pub fn run(&self, rng: &PyRng<'_>) -> PyResult<(GTree, usize)> {
        let mut tree = GTree::new();
        let mut counter = 0usize;
        let mut bottom: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
        let mut root: Option<usize> = None;

        for seed in &self.seeds {
            let lineages: Vec<usize> = seed
                .labels
                .iter()
                .map(|l| tree.leaf(l.clone(), 0.0))
                .collect();
            match seed.target {
                Some(e) => bottom.entry(e).or_default().extend(lineages),
                None => {
                    let theta = seed.theta_root.unwrap_or(0.0);
                    let fin = coalesce_within(&lineages, 0.0, None, theta, rng, &mut counter, &mut |a, b, t, l| {
                        Ok(tree.merge(a, b, t, l))
                    })?;
                    root = fin.first().copied();
                }
            }
        }

        for step in &self.steps {
            let mut arriving: Vec<usize> = Vec::new();
            for &(e, h_child, theta) in &step.children {
                let pool = bottom.get(&e).cloned().unwrap_or_default();
                let top = coalesce_within(&pool, h_child, Some(step.height), theta, rng, &mut counter, &mut |a, b, t, l| {
                    Ok(tree.merge(a, b, t, l))
                })?;
                arriving.extend(top);
            }
            match step.exit {
                Exit::Root { theta } => {
                    let fin = coalesce_within(&arriving, step.height, None, theta, rng, &mut counter, &mut |a, b, t, l| {
                        Ok(tree.merge(a, b, t, l))
                    })?;
                    root = fin.first().copied();
                }
                Exit::Retic { e_a, gamma_a, e_b } => {
                    for lineage in arriving {
                        let target = if rng.random()? < gamma_a { e_a } else { e_b };
                        bottom.entry(target).or_default().push(lineage);
                    }
                }
                Exit::Tree { parent_edge } => bottom.entry(parent_edge).or_default().extend(arriving),
            }
        }

        match root {
            Some(r) => Ok((tree, r)),
            None => Err(PyRuntimeError::new_err(
                "Coalescent simulation produced no root lineage.",
            )),
        }
    }
}

/// ``Network.from_newick(newick + ";", SUBSTITUTIONS_PER_SITE)``, or the
/// trivial one-node network when a single lineage was sampled.
pub(crate) fn tree_to_network(py: Python<'_>, tree: &GTree, root: usize) -> PyResult<Py<PyAny>> {
    let nwk = tree.newick(root);
    let unit = subst_unit(py)?;
    if !nwk.contains('(') {
        let net = Network::new_object(py, Some(&unit))?;
        let node = Py::new(py, Node::make(py, nwk, false, None, None, None))?;
        net.borrow_mut().add_node(net.as_any(), node.bind(py))?;
        return Ok(net.into_any().unbind());
    }
    let cls = match network_class(py) {
        Some(c) => c.clone(),
        None => py.get_type::<Network>(),
    };
    parse_newick(py, cls, &format!("{nwk};"), Some(unit))
}

// ═══════════════════════════════════════════════════════════════════════
// Sequence evolution
// ═══════════════════════════════════════════════════════════════════════

fn matrix4_of(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<[[f64; 4]; 4]> {
    let bad = || PyValueError::new_err(format!("{what} must be a 4x4 matrix."));
    let mut out = [[0.0; 4]; 4];
    if let Ok(a) = obj.extract::<PyArrayLike2<f64, AllowTypeChange>>() {
        let v = a.as_array();
        if v.shape() != [4, 4] {
            return Err(bad());
        }
        for i in 0..4 {
            for j in 0..4 {
                out[i][j] = v[[i, j]];
            }
        }
        return Ok(out);
    }
    let rows: Vec<Vec<f64>> = obj.extract().map_err(|_| bad())?;
    if rows.len() != 4 || rows.iter().any(|r| r.len() != 4) {
        return Err(bad());
    }
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = rows[i][j];
        }
    }
    Ok(out)
}

/// ``np.cumsum(P, axis=1)``.
fn row_cdf(p: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut cdf = [[0.0; 4]; 4];
    for i in 0..4 {
        let mut acc = 0.0;
        for j in 0..4 {
            acc += p[i][j];
            cdf[i][j] = acc;
        }
    }
    cdf
}

/// A nucleotide index as numpy would index ``cdf[parent_states]``.
fn state_index(s: i64) -> PyResult<u8> {
    match s {
        0..=3 => Ok(s as u8),
        -4..=-1 => Ok((s + 4) as u8),
        _ => Err(PyIndexError::new_err(format!(
            "index {s} is out of bounds for axis 0 with size 4"
        ))),
    }
}

/// ``(u[:, None] < cdf[parent]).argmax(axis=1)``: inverse-CDF sampling of
/// every site's child state.
fn evolve_states(parent: &[u8], cdf: &[[f64; 4]; 4], u: &[f64]) -> Vec<u8> {
    parent
        .iter()
        .zip(u)
        .map(|(&s, &x)| {
            let row = &cdf[(s & 3) as usize];
            row.iter().position(|&c| x < c).unwrap_or(0) as u8
        })
        .collect()
}

/// The reference's pre-order traversal of a gene tree as a list of
/// ``(parent slot, child slot, branch length)`` in visiting order, plus the
/// leaves (label, slot) in ``get_leaves()`` order and the slot count.
#[allow(clippy::type_complexity)]
fn sequence_plan(
    py: Python<'_>,
    net: &Network,
) -> PyResult<(Vec<(usize, usize, f64)>, Vec<(String, Option<usize>)>, usize)> {
    if !net.is_acyclic_impl() {
        return Err(network_error(
            py,
            "sequence simulation requires an acyclic gene tree.",
        ));
    }
    let root = net.root_uid(py)?;
    let mut slot: FxHashMap<u64, usize> = FxHashMap::default();
    slot.insert(root, 0);
    let mut edges = Vec::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let ps = slot.get(&node).copied().unwrap_or(0);
        for child in net.children(node) {
            let edge = net.edge_between(py, node, child, None, None)?;
            let t = edge.borrow(py).length.unwrap_or(0.0);
            let next = slot.len();
            let cs = *slot.entry(child).or_insert(next);
            edges.push((ps, cs, t));
            stack.push(child);
        }
    }
    let leaves = net
        .leaf_uids()
        .into_iter()
        .map(|u| (node_label(py, net, u), slot.get(&u).copied()))
        .collect();
    Ok((edges, leaves, slot.len()))
}

fn sim_sequences_impl<'py>(
    py: Python<'py>,
    gene_tree: &Bound<'py, Network>,
    model: &Bound<'py, PyAny>,
    n_sites: i64,
    rng: &PyRng<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    require_subst_unit(py, gene_tree.as_any(), "sequence simulation")?;
    if n_sites <= 0 {
        return Err(PyValueError::new_err("n_sites must be positive."));
    }
    let n = n_sites as usize;
    let (edges, leaves, n_slots) = sequence_plan(py, &gene_tree.borrow())?;

    // The root sequence is drawn i.i.d. from the stationary frequencies.
    let pi = model.getattr("pi")?;
    let root: Vec<u8> = rng
        .choice4(n, &pi)?
        .into_iter()
        .map(state_index)
        .collect::<PyResult<_>>()?;
    let mut states: Vec<Option<Vec<u8>>> = (0..n_slots).map(|_| None).collect();
    if let Some(s) = states.get_mut(0) {
        *s = Some(root);
    }

    // Pre-order: propagate each parent's states to its children.
    for (ps, cs, t) in edges {
        let p = matrix4_of(&model.call_method1("p_matrix", (t,))?, "p_matrix(t)")?;
        let cdf = row_cdf(&p);
        let u = rng.random_vec(n)?;
        let child = {
            let parent = states.get(ps).and_then(|s| s.as_ref()).ok_or_else(|| {
                PyRuntimeError::new_err("sequence simulation reached a node before its parent.")
            })?;
            evolve_states(parent, &cdf, &u)
        };
        if let Some(s) = states.get_mut(cs) {
            *s = Some(child);
        }
    }

    let out = PyDict::new(py);
    for (label, slot) in leaves {
        let seq = slot
            .and_then(|s| states.get(s))
            .and_then(|s| s.as_ref())
            .ok_or_else(|| PyKeyError::new_err(label.clone()))?;
        let text: String = seq.iter().map(|&i| DNA[(i & 3) as usize] as char).collect();
        out.set_item(label, text)?;
    }
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════
// Python entry points
// ═══════════════════════════════════════════════════════════════════════

/// One ancestral gene lineage during the coalescent sweep.
///
/// Attributes:
///     newick: Newick string of the subtree subtended by this lineage.
///     height: Height (substitution units) of the lineage's youngest open end.
#[pyclass(name = "_Lineage", module = "phynetpy._sim_seq")]
pub struct Lineage {
    #[pyo3(get, set)]
    pub newick: String,
    #[pyo3(get, set)]
    pub height: f64,
}

#[pymethods]
impl Lineage {
    /// Create a new ``_Lineage``.
    ///
    /// Args:
    ///     newick: Newick string of the subtree subtended by this lineage.
    ///     height: Height (substitution units) of its youngest open end.
    #[new]
    fn py_new(newick: String, height: f64) -> Self {
        Lineage { newick, height }
    }
}

/// ``_reticulation_parent_gammas(net, retic)``: the two parent edges of a
/// reticulation with normalised inheritance probabilities.
#[pyfunction]
fn sim_reticulation_parent_gammas<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
    retic: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let this = net.borrow();
    let Some(uid) = this.uid_of_any(retic) else {
        // ``net.in_edges`` of a foreign node warns and yields nothing.
        warn(
            py,
            "Attempting to get the in-edges of a node that is not in the graph-- returning an empty list",
        )?;
        let label = retic.getattr("label")?.repr()?;
        return Err(PyValueError::new_err(format!(
            "Reticulation {label} must have exactly two parent edges."
        )));
    };
    let pairs = retic_parent_gammas(py, &this, uid)?;
    let out = PyList::empty(py);
    for (e, g) in pairs {
        let edge = this
            .edges
            .get(&e)
            .map(|x| x.obj.clone_ref(py))
            .ok_or_else(|| PyValueError::new_err("No matching edges found"))?;
        out.append((edge, g))?;
    }
    Ok(out)
}

/// ``_coalesce_within(lineages, t_start, t_end, theta, rng, counter)`` on
/// ``_Lineage`` objects; ``counter[0]`` is advanced past the labels used.
#[pyfunction]
#[pyo3(signature = (lineages, t_start, t_end, theta, rng, counter))]
fn sim_coalesce_within<'py>(
    py: Python<'py>,
    lineages: &Bound<'py, PyAny>,
    t_start: f64,
    t_end: Option<f64>,
    theta: f64,
    rng: Bound<'py, PyAny>,
    counter: &Bound<'py, PyAny>,
) -> PyResult<Vec<Py<Lineage>>> {
    let mut arena: Vec<Py<Lineage>> = Vec::new();
    for item in lineages.try_iter()? {
        arena.push(item?.cast_into::<Lineage>()?.unbind());
    }
    let pool: Vec<usize> = (0..arena.len()).collect();
    let mut count: usize = counter.get_item(0)?.extract()?;
    let rng = PyRng::new(rng);
    let out = coalesce_within(&pool, t_start, t_end, theta, &rng, &mut count, &mut |a, b, t, label| {
        let (an, ah) = {
            let x = arena[a].borrow(py);
            (x.newick.clone(), x.height)
        };
        let (bn, bh) = {
            let x = arena[b].borrow(py);
            (x.newick.clone(), x.height)
        };
        let merged = Py::new(
            py,
            Lineage {
                newick: format!("({an}:{:.10},{bn}:{:.10}){label}", t - ah, t - bh),
                height: t,
            },
        )?;
        arena.push(merged);
        Ok(arena.len() - 1)
    })?;
    counter.set_item(0, count)?;
    Ok(out.into_iter().map(|i| arena[i].clone_ref(py)).collect())
}

/// ``simulate_gene_tree(species_net, alleles_per_species, theta, rng,
/// branch_thetas=None)``: one genealogy under the MSNC as a ``Network``.
#[pyfunction]
#[pyo3(signature = (species_net, alleles_per_species, theta, rng, branch_thetas=None))]
fn sim_gene_tree(
    py: Python<'_>,
    species_net: &Bound<'_, Network>,
    alleles_per_species: &Bound<'_, PyAny>,
    theta: f64,
    rng: Bound<'_, PyAny>,
    branch_thetas: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let plan = GenePlan::new(py, species_net, alleles_per_species, theta, branch_thetas.as_ref())?;
    let rng = PyRng::new(rng);
    let (tree, root) = plan.run(&rng)?;
    tree_to_network(py, &tree, root)
}

/// ``_evolve_states(parent_states, P, rng)``: child states for every site.
#[pyfunction]
#[pyo3(name = "sim_evolve_states", signature = (parent_states, p, rng))]
fn sim_evolve_states<'py>(
    py: Python<'py>,
    parent_states: &Bound<'py, PyAny>,
    p: &Bound<'py, PyAny>,
    rng: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<isize>>> {
    let parents: Vec<u8> = int_vec(parent_states)?
        .into_iter()
        .map(state_index)
        .collect::<PyResult<_>>()?;
    let cdf = row_cdf(&matrix4_of(p, "P")?);
    let rng = PyRng::new(rng);
    let u = rng.random_vec(parents.len())?;
    let out: Vec<isize> = evolve_states(&parents, &cdf, &u)
        .into_iter()
        .map(|x| x as isize)
        .collect();
    Ok(PyArray1::from_vec(py, out))
}

/// ``simulate_sequences(gene_tree, model, n_sites, rng)``: evolve an
/// alignment down ``gene_tree`` under ``model``.
#[pyfunction]
fn sim_sequences<'py>(
    py: Python<'py>,
    gene_tree: &Bound<'py, Network>,
    model: &Bound<'py, PyAny>,
    n_sites: i64,
    rng: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let rng = PyRng::new(rng);
    sim_sequences_impl(py, gene_tree, model, n_sites, &rng)
}

/// The per-locus loop of ``simulate_multilocus``: ``(loci, gene_trees)``.
#[pyfunction]
#[pyo3(signature = (species_net, mapping, n_loci, seq_length, theta, model, branch_thetas, rng))]
#[allow(clippy::too_many_arguments)]
fn sim_multilocus<'py>(
    py: Python<'py>,
    species_net: &Bound<'py, Network>,
    mapping: &Bound<'py, PyAny>,
    n_loci: i64,
    seq_length: i64,
    theta: f64,
    model: &Bound<'py, PyAny>,
    branch_thetas: Option<Bound<'py, PyAny>>,
    rng: Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let plan = GenePlan::new(py, species_net, mapping, theta, branch_thetas.as_ref())?;
    let rng = PyRng::new(rng);
    let loci = PyList::empty(py);
    let trees = PyList::empty(py);
    for _ in 0..n_loci.max(0) {
        let (tree, root) = plan.run(&rng)?;
        let gt = tree_to_network(py, &tree, root)?;
        let gt = gt.into_bound(py);
        let aln = sim_sequences_impl(py, gt.cast::<Network>()?, model, seq_length, &rng)?;
        trees.append(&gt)?;
        loci.append(aln)?;
    }
    Ok((loci, trees))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Lineage>()?;
    m.add_function(wrap_pyfunction!(sim_reticulation_parent_gammas, m)?)?;
    m.add_function(wrap_pyfunction!(sim_coalesce_within, m)?)?;
    m.add_function(wrap_pyfunction!(sim_gene_tree, m)?)?;
    m.add_function(wrap_pyfunction!(sim_evolve_states, m)?)?;
    m.add_function(wrap_pyfunction!(sim_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(sim_multilocus, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newick_round_trip_matches_fixed_precision() {
        let mut t = GTree::new();
        let a = t.leaf("A".into(), 0.0);
        let b = t.leaf("B".into(), 0.0);
        let r = t.merge(a, b, 0.123456789012345, "i0".into());
        assert_eq!(t.newick(r), "(A:0.1234567890,B:0.1234567890)i0");
        assert_eq!(t.branch_len(r, a), 0.1234567890);
    }

    #[test]
    fn inverse_cdf_picks_first_exceeding_column() {
        let p = [[0.7, 0.1, 0.1, 0.1]; 4];
        let cdf = row_cdf(&p);
        let out = evolve_states(&[0, 1, 2, 3], &cdf, &[0.5, 0.75, 0.85, 0.99]);
        assert_eq!(out, vec![0, 1, 2, 3]);
        // ``argmax`` of an all-False row is 0.
        assert_eq!(evolve_states(&[0], &cdf, &[1.5]), vec![0]);
    }
}
