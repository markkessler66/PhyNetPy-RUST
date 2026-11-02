//! Network comparison metrics: the reticulation-aware dissimilarity of
//! ``phynetpy.ReticulationComparison`` (``_rc_*`` functions) and the joint
//! reticulation-and-tree dissimilarity of ``phynetpy.JointDissimilarity``
//! (``_jd_*`` functions).  The two Python modules are facades over the
//! functions registered here; the small result dataclasses
//! (``ReticulationTripartition``, ``NetworkComparison``,
//! ``JointDissimilarityResult``) stay in Python and are filled from the
//! tuples these functions return.
//!
//! Every computation follows its Python reference step for step.  The
//! padded minimum-weight matchings are solved with a port of SciPy's
//! ``linear_sum_assignment`` (Crouse 2016, shortest augmenting path) so the
//! optimal matching -- not just its value -- is the one the reference
//! produced, and the matched costs are summed in NumPy's pairwise order.

use std::cell::OnceCell;
use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyFrozenSet, PyList, PyString};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::Network;
use crate::util::network_error;

const JACCARD: &str = "jaccard";
const SYMMETRIC: &str = "symmetric";
/// The maximum of one ``delta`` under the Jaccard ground distance: three
/// block comparisons, each in ``[0, 1]``.  Also the real--dummy cost of the
/// joint dissimilarity.
const DELTA_MAX_JACCARD: f64 = 3.0;

// ── Assignment and summation, as SciPy and NumPy do them ──────────────

/// One augmenting path of the shortest-augmenting-path LSAP solver
/// (``scipy/optimize/rectangular_lsap/rectangular_lsap.cpp``).
#[allow(clippy::too_many_arguments)]
fn augmenting_path(
    n: usize,
    cost: &[f64],
    u: &[f64],
    v: &[f64],
    path: &mut [usize],
    row4col: &[usize],
    shortest: &mut [f64],
    mut i: usize,
    sr: &mut [bool],
    sc: &mut [bool],
    remaining: &mut [usize],
) -> Option<(usize, f64)> {
    let mut min_val = 0.0;
    let mut num_remaining = n;
    // Filled in reverse so a constant cost matrix is solved by the identity.
    for (it, slot) in remaining.iter_mut().enumerate() {
        *slot = n - it - 1;
    }
    sr.fill(false);
    sc.fill(false);
    shortest.fill(f64::INFINITY);
    loop {
        let mut index: Option<usize> = None;
        let mut lowest = f64::INFINITY;
        sr[i] = true;
        for it in 0..num_remaining {
            let j = remaining[it];
            let r = min_val + cost[i * n + j] - u[i] - v[j];
            if r < shortest[j] {
                path[j] = i;
                shortest[j] = r;
            }
            // Among equal costs prefer a column that gives a new sink.
            if shortest[j] < lowest || (shortest[j] == lowest && row4col[j] == usize::MAX) {
                lowest = shortest[j];
                index = Some(it);
            }
        }
        min_val = lowest;
        if min_val == f64::INFINITY {
            return None;
        }
        let idx = index?;
        let j = remaining[idx];
        let sink = if row4col[j] == usize::MAX {
            Some(j)
        } else {
            i = row4col[j];
            None
        };
        sc[j] = true;
        num_remaining -= 1;
        remaining[idx] = remaining[num_remaining];
        if let Some(s) = sink {
            return Some((s, min_val));
        }
    }
}

/// ``scipy.optimize.linear_sum_assignment`` on a square ``n x n`` cost
/// matrix (row-major): the column assigned to every row.  ``None`` when the
/// matrix is infeasible (non-finite entries).
fn linear_sum_assignment(n: usize, cost: &[f64]) -> Option<Vec<usize>> {
    let mut u = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut shortest = vec![0.0; n];
    let mut path = vec![usize::MAX; n];
    let mut col4row = vec![usize::MAX; n];
    let mut row4col = vec![usize::MAX; n];
    let mut sr = vec![false; n];
    let mut sc = vec![false; n];
    let mut remaining = vec![0usize; n];
    for cur_row in 0..n {
        let (sink, min_val) = augmenting_path(
            n, cost, &u, &v, &mut path, &row4col, &mut shortest, cur_row, &mut sr, &mut sc, &mut remaining,
        )?;
        u[cur_row] += min_val;
        for i in 0..n {
            if sr[i] && i != cur_row {
                if let Some(&c) = col4row.get(i).filter(|&&c| c != usize::MAX) {
                    u[i] += min_val - shortest[c];
                }
            }
        }
        for j in 0..n {
            if sc[j] {
                v[j] -= min_val - shortest[j];
            }
        }
        let mut j = sink;
        loop {
            let i = path[j];
            if i == usize::MAX {
                return None;
            }
            row4col[j] = i;
            std::mem::swap(&mut col4row[i], &mut j);
            if i == cur_row {
                break;
            }
        }
    }
    if col4row.iter().any(|&c| c == usize::MAX) {
        return None;
    }
    Some(col4row)
}

/// NumPy's pairwise summation of a contiguous float64 vector.
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += a[i + k];
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
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// ``float(numpy_array.sum())``: the add-reduction starts from the identity
/// and adds the pairwise sum of the whole vector.
fn numpy_sum(a: &[f64]) -> f64 {
    0.0 + pairwise_sum(a)
}

/// The value of the optimal padded assignment and the column of every row.
fn solve_assignment(m: usize, cost: &[f64]) -> PyResult<(f64, Vec<usize>)> {
    let cols = linear_sum_assignment(m, cost)
        .ok_or_else(|| PyValueError::new_err("cost matrix is infeasible"))?;
    let picked: Vec<f64> = cols.iter().enumerate().map(|(i, &j)| cost[i * m + j]).collect();
    Ok((numpy_sum(&picked), cols))
}

// ── Labels and label sets ─────────────────────────────────────────────

/// Leaf labels interned to small integers, shared by the networks of one
/// comparison so equal labels get equal ids.
#[derive(Default)]
struct Interner {
    ids: FxHashMap<String, u32>,
    names: Vec<String>,
}

impl Interner {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.ids.get(s) {
            return id;
        }
        let id = self.names.len() as u32;
        self.ids.insert(s.to_string(), id);
        self.names.push(s.to_string());
        id
    }

    fn name(&self, id: u32) -> &str {
        self.names.get(id as usize).map(String::as_str).unwrap_or("")
    }
}

/// A set of interned labels: sorted and duplicate-free.
type IdSet = Vec<u32>;

fn ids_union(a: &[u32], b: &[u32]) -> IdSet {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

fn ids_difference(a: &[u32], b: &[u32]) -> IdSet {
    let mut out = Vec::with_capacity(a.len());
    let mut j = 0;
    for &x in a {
        while j < b.len() && b[j] < x {
            j += 1;
        }
        if j < b.len() && b[j] == x {
            continue;
        }
        out.push(x);
    }
    out
}

fn ids_intersection(a: &[u32], b: &[u32]) -> IdSet {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// ``(|P ^ Q|, |P | Q|)`` for two sorted id sets.
fn ids_symdiff_union(a: &[u32], b: &[u32]) -> (usize, usize) {
    let common = ids_intersection(a, b).len();
    (a.len() + b.len() - 2 * common, a.len() + b.len() - common)
}

/// ``set_distance`` on sizes: the symmetric-difference cardinality or the
/// Jaccard distance (0 when both sets are empty).
fn set_distance_sizes(symdiff: usize, union: usize, jaccard: bool) -> f64 {
    if !jaccard {
        return symdiff as f64;
    }
    if union == 0 {
        return 0.0;
    }
    symdiff as f64 / union as f64
}

/// ``repr(sorted(labels))`` as Python prints it.
fn sorted_repr(py: Python<'_>, labels: &mut Vec<&str>) -> PyResult<String> {
    labels.sort_unstable();
    Ok(PyList::new(py, labels.iter().copied())?.repr()?.to_string())
}

/// ``frozenset`` of the names behind a set of ids.
fn frozenset_of_ids<'py>(py: Python<'py>, interner: &Interner, ids: &[u32]) -> PyResult<Bound<'py, PyFrozenSet>> {
    PyFrozenSet::new(py, ids.iter().map(|&i| PyString::new(py, interner.name(i))))
}

/// The ``(|P ^ Q|, |P | Q|)`` of two arbitrary Python sets.  Sets of
/// strings are compared in Rust; anything else goes through Python's own
/// ``frozenset`` operations so any hashable element works.
fn py_set_sizes(py: Python<'_>, p: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>) -> PyResult<(usize, usize)> {
    fn strings(obj: &Bound<'_, PyAny>) -> PyResult<Option<FxHashSet<String>>> {
        let mut out = FxHashSet::default();
        for item in obj.try_iter()? {
            let item = item?;
            if !item.is_instance_of::<PyString>() {
                return Ok(None);
            }
            out.insert(item.extract::<String>()?);
        }
        Ok(Some(out))
    }
    if let (Some(a), Some(b)) = (strings(p)?, strings(q)?) {
        let common = a.iter().filter(|x| b.contains(*x)).count();
        return Ok((a.len() + b.len() - 2 * common, a.len() + b.len() - common));
    }
    let fp = PyFrozenSet::new(py, p.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
    let fq = PyFrozenSet::new(py, q.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
    let symdiff = fp.call_method1("symmetric_difference", (&fq,))?.len()?;
    let union = fp.call_method1("union", (&fq,))?.len()?;
    Ok((symdiff, union))
}

/// Resolve a ground-distance name; ``NetworkError`` otherwise.
fn ground_distance(py: Python<'_>, kind: &Bound<'_, PyAny>) -> PyResult<bool> {
    let name = kind.extract::<String>().ok();
    match name.as_deref() {
        Some(SYMMETRIC) => Ok(false),
        Some(JACCARD) => Ok(true),
        _ => Err(network_error(
            py,
            format!("Unknown set distance '{}'. Use '{}' or '{}'.", kind.str()?, JACCARD, SYMMETRIC),
        )),
    }
}

// ── ReticulationComparison: tripartitions ─────────────────────────────

/// ``(A_h, B_h, C_h)`` of one reticulation, as sorted label-id sets.
struct Tri {
    a: IdSet,
    b: IdSet,
    c: IdSet,
    hybrid: String,
}

/// ``_leaf_descendant_sets``: the hardwired cluster of every node in one
/// reverse-topological pass.
fn leaf_descendant_sets(py: Python<'_>, net: &Network, interner: &mut Interner) -> PyResult<FxHashMap<u64, IdSet>> {
    let leaves: FxHashSet<u64> = net.leaf_uids().into_iter().collect();
    let order = net.topological_order_uids(py)?;
    let mut desc: FxHashMap<u64, IdSet> = FxHashMap::default();
    for &uid in order.iter().rev() {
        let mut acc: IdSet = Vec::new();
        if leaves.contains(&uid) {
            acc.push(interner.intern(&net.label(py, uid)));
        }
        for child in net.children(uid) {
            if let Some(d) = desc.get(&child) {
                acc = ids_union(&acc, d);
            }
        }
        desc.insert(uid, acc);
    }
    Ok(desc)
}

/// ``reticulation_tripartitions``: one tripartition per node flagged as a
/// reticulation, in ``V()`` order.
fn tripartitions(py: Python<'_>, net: &Network, interner: &mut Interner) -> PyResult<Vec<Tri>> {
    let desc = leaf_descendant_sets(py, net, interner)?;
    let empty: IdSet = Vec::new();
    let mut out = Vec::new();
    for (&uid, entry) in net.nodes.iter() {
        if !entry.obj.borrow(py).is_retic {
            continue;
        }
        let parents = net.parents(uid);
        let label = net.label(py, uid);
        if parents.len() != 2 {
            return Err(network_error(
                py,
                format!(
                    "Reticulation node '{}' has {} parent(s); the tripartition measure requires binary reticulations with exactly 2 parents.",
                    label,
                    parents.len()
                ),
            ));
        }
        let b = desc.get(&uid).unwrap_or(&empty).clone();
        let a = ids_difference(desc.get(&parents[0]).unwrap_or(&empty), &b);
        let c = ids_difference(desc.get(&parents[1]).unwrap_or(&empty), &b);
        out.push(Tri { a, b, c, hybrid: label });
    }
    Ok(out)
}

// ── ReticulationComparison: the global metric D ───────────────────────

/// A canonical signature of a node: a leaf's label or the sorted class keys
/// of its children.
#[derive(Hash, PartialEq, Eq)]
enum Sig {
    Leaf(String),
    Node(Vec<u32>),
}

/// ``_canonical_key_multiset``: bottom-up canonical labels, interned in a
/// table shared by the two networks being compared.
fn canonical_key_multiset(py: Python<'_>, net: &Network, interner: &mut FxHashMap<Sig, u32>) -> PyResult<FxHashMap<u32, i64>> {
    let order = net.topological_order_uids(py)?;
    let mut key_of: FxHashMap<u64, u32> = FxHashMap::default();
    let mut counts: FxHashMap<u32, i64> = FxHashMap::default();
    for &uid in order.iter().rev() {
        let children = net.children(uid);
        let sig = if children.is_empty() {
            Sig::Leaf(net.label(py, uid))
        } else {
            let mut keys: Vec<u32> = children.iter().filter_map(|c| key_of.get(c).copied()).collect();
            keys.sort_unstable();
            Sig::Node(keys)
        };
        let next = interner.len() as u32;
        let key = *interner.entry(sig).or_insert(next);
        key_of.insert(uid, key);
        *counts.entry(key).or_insert(0) += 1;
    }
    Ok(counts)
}

/// ``sum_k |c1[k] - c2[k]|`` over the union of keys.
fn multiset_symdiff<K: std::hash::Hash + Eq>(c1: &FxHashMap<K, i64>, c2: &FxHashMap<K, i64>) -> i64 {
    let mut total = 0;
    for (k, &a) in c1 {
        total += (a - c2.get(k).copied().unwrap_or(0)).abs();
    }
    for (k, &b) in c2 {
        if !c1.contains_key(k) {
            total += b.abs();
        }
    }
    total
}

/// ``nakhleh_metric``: half the symmetric difference of the two multisets of
/// canonical keys; optionally normalised by ``|V1| + |V2|``.
fn nakhleh_metric_impl(py: Python<'_>, net1: &Network, net2: &Network, normalize: bool) -> PyResult<f64> {
    let mut interner: FxHashMap<Sig, u32> = FxHashMap::default();
    let counts1 = canonical_key_multiset(py, net1, &mut interner)?;
    let counts2 = canonical_key_multiset(py, net2, &mut interner)?;
    let distance = 0.5 * multiset_symdiff(&counts1, &counts2) as f64;
    if normalize {
        let denom = net1.nodes.len() + net2.nodes.len();
        return Ok(if denom > 0 { 2.0 * distance / denom as f64 } else { 0.0 });
    }
    Ok(distance)
}

// ── ReticulationComparison: block distance, delta and the matching ────

/// The block distance ``d`` (``alpha = 1``) or its topology-aware
/// refinement ``d*`` (``alpha < 1``) of ``_block_distance_factory``.
struct BlockDistance<'a, 'py> {
    py: Python<'py>,
    jaccard: bool,
    alpha: f64,
    net_ref: &'a Bound<'py, Network>,
    net_inf: &'a Bound<'py, Network>,
    interner: &'a Interner,
    /// ``_core._gu_induced_subnetwork_by_taxa``, fetched once when needed.
    induced: Option<Bound<'py, PyAny>>,
}

impl<'a, 'py> BlockDistance<'a, 'py> {
    fn new(
        py: Python<'py>,
        jaccard: bool,
        alpha: f64,
        net_ref: &'a Bound<'py, Network>,
        net_inf: &'a Bound<'py, Network>,
        interner: &'a Interner,
    ) -> PyResult<Self> {
        let induced = if alpha >= 1.0 {
            None
        } else {
            Some(py.import("phynetpy._core")?.getattr("_gu_induced_subnetwork_by_taxa")?)
        };
        Ok(BlockDistance { py, jaccard, alpha, net_ref, net_inf, interner, induced })
    }

    fn d(&self, p: &[u32], q: &[u32]) -> PyResult<f64> {
        let (symdiff, union) = ids_symdiff_union(p, q);
        let base = set_distance_sizes(symdiff, union, self.jaccard);
        let induced = match &self.induced {
            None => return Ok(base),
            Some(f) => f,
        };
        let shared = ids_intersection(p, q);
        let d_top = if shared.len() < 2 {
            // A single shared leaf (or none) induces a trivial, identical
            // substructure, so the topological term contributes nothing.
            0.0
        } else {
            let py = self.py;
            let mut taxa: Vec<&str> = shared.iter().map(|&i| self.interner.name(i)).collect();
            taxa.sort_unstable();
            let taxa_list = PyList::new(py, taxa.iter().copied())?;
            let sub_ref = induced.call1((self.net_ref, &taxa_list))?.cast_into::<Network>()?;
            let sub_inf = induced.call1((self.net_inf, &taxa_list))?.cast_into::<Network>()?;
            nakhleh_metric_impl(py, &sub_ref.borrow(), &sub_inf.borrow(), false)?
        };
        Ok(self.alpha * base + (1.0 - self.alpha) * d_top)
    }

    /// ``_delta``: the hybrid blocks compared directly, the outer blocks
    /// under the cheaper of the two parent correspondences.
    fn delta(&self, t: &Tri, s: &Tri) -> PyResult<f64> {
        let middle = self.d(&t.b, &s.b)?;
        let direct = self.d(&t.a, &s.a)? + self.d(&t.c, &s.c)?;
        let swapped = self.d(&t.a, &s.c)? + self.d(&t.c, &s.a)?;
        Ok(middle + if swapped < direct { swapped } else { direct })
    }
}

/// One entry of the optimal matching: ``(ref_index, inf_index, cost)`` with
/// ``None`` for the padding symbol.
type Match = (Option<usize>, Option<usize>, f64);

/// ``_assign``: the padded minimum-weight matching of Equation 4.
fn assign(bd: &BlockDistance<'_, '_>, tri_ref: &[Tri], tri_inf: &[Tri], rho: f64) -> PyResult<(f64, Vec<Match>)> {
    let (r1, r2) = (tri_ref.len(), tri_inf.len());
    let m = r1.max(r2);
    if m == 0 {
        return Ok((0.0, Vec::new()));
    }
    let mut cost = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            cost[i * m + j] = if i < r1 && j < r2 {
                bd.delta(&tri_ref[i], &tri_inf[j])?
            } else if i < r1 || j < r2 {
                rho
            } else {
                0.0
            };
        }
    }
    let (total, cols) = solve_assignment(m, &cost)?;
    let mut matching = Vec::new();
    for (i, &j) in cols.iter().enumerate() {
        let ref_idx = if i < r1 { Some(i) } else { None };
        let inf_idx = if j < r2 { Some(j) } else { None };
        if ref_idx.is_none() && inf_idx.is_none() {
            continue;
        }
        matching.push((ref_idx, inf_idx, cost[i * m + j]));
    }
    Ok((total, matching))
}

/// ``(D, D_hat, D_ret, D_ret_hat)``.
type Scores = (f64, f64, f64, f64);
/// ``(precision, recall, f1)``.
type PrecRec = (f64, f64, f64);
/// ``(r_reference, r_inferred, recovered)``.
type Counts = (usize, usize, usize);
/// Everything ``compare_networks`` computes: the scores, the recovery
/// statistics, the matching, and ``(rho, delta_max)`` as actually used.
type Comparison = (Scores, PrecRec, Counts, Vec<Match>, (f64, f64));

/// A Python number as ``f64`` (the caller prints the original object in
/// its error messages, as the reference's f-strings did).
fn number(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    obj.extract::<f64>()
}

/// ``compare_networks``: validation, tripartitions, the padded matching,
/// precision / recall / F1 and the global metric.
fn compare_impl<'py>(
    py: Python<'py>,
    reference: &Bound<'py, Network>,
    inferred: &Bound<'py, Network>,
    distance: &Bound<'py, PyAny>,
    rho: Option<&Bound<'py, PyAny>>,
    tolerance: &Bound<'py, PyAny>,
    alpha: &Bound<'py, PyAny>,
) -> PyResult<Comparison> {
    let jaccard = ground_distance(py, distance)?;
    let alpha_v = number(alpha)?;
    if !(0.0 <= alpha_v && alpha_v <= 1.0) {
        return Err(network_error(py, format!("alpha must lie in [0, 1]; got {}.", alpha.str()?)));
    }
    let tol = number(tolerance)?;
    if tol < 0.0 {
        return Err(network_error(py, format!("tolerance (tau) must be >= 0; got {}.", tolerance.str()?)));
    }

    let net_ref = reference.borrow();
    let net_inf = inferred.borrow();

    // ``_leaf_universe_size``: the combined taxon set of the two networks.
    let universe_size = {
        let mut labels: FxHashSet<String> = FxHashSet::default();
        for u in net_ref.leaf_uids() {
            labels.insert(net_ref.label(py, u));
        }
        for u in net_inf.leaf_uids() {
            labels.insert(net_inf.label(py, u));
        }
        labels.len()
    };
    let max_delta = if jaccard { DELTA_MAX_JACCARD } else { 3.0 * universe_size as f64 };
    let rho_v = match rho {
        None => max_delta,
        Some(r) => {
            let r_v = number(r)?;
            if r_v < 0.0 {
                return Err(network_error(py, format!("rho must be >= 0; got {}.", r.str()?)));
            }
            r_v
        }
    };
    let delta_max = if rho_v > max_delta { rho_v } else { max_delta };

    let mut interner = Interner::default();
    let tri_ref = tripartitions(py, &net_ref, &mut interner)?;
    let tri_inf = tripartitions(py, &net_inf, &mut interner)?;
    let (r1, r2) = (tri_ref.len(), tri_inf.len());
    let m = r1.max(r2);

    let bd = BlockDistance::new(py, jaccard, alpha_v, reference, inferred, &interner)?;
    let (d_ret, matching) = assign(&bd, &tri_ref, &tri_inf, rho_v)?;
    let d_ret_hat = if m > 0 { d_ret / (m as f64 * delta_max) } else { 0.0 };

    let recovered = matching
        .iter()
        .filter(|(r, i, c)| r.is_some() && i.is_some() && *c <= tol)
        .count();
    let precision = if r2 == 0 { 1.0 } else { recovered as f64 / r2 as f64 };
    let recall = if r1 == 0 { 1.0 } else { recovered as f64 / r1 as f64 };
    let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };

    let d_global = nakhleh_metric_impl(py, &net_ref, &net_inf, false)?;
    let denom = net_ref.nodes.len() + net_inf.nodes.len();
    let d_global_hat = if denom > 0 { 2.0 * d_global / denom as f64 } else { 0.0 };

    Ok((
        (d_global, d_global_hat, d_ret, d_ret_hat),
        (precision, recall, f1),
        (r1, r2, recovered),
        matching,
        (rho_v, delta_max),
    ))
}

// ── ReticulationComparison: Python entry points ───────────────────────

/// ``set_distance(P, Q, kind)``.
#[pyfunction]
#[pyo3(name = "_rc_set_distance")]
fn rc_set_distance(py: Python<'_>, p: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>, kind: &Bound<'_, PyAny>) -> PyResult<f64> {
    let jaccard = ground_distance(py, kind)?;
    let (symdiff, union) = py_set_sizes(py, p, q)?;
    Ok(set_distance_sizes(symdiff, union, jaccard))
}

/// ``reticulation_tripartitions(net)`` as ``(A, B, C, hybrid)`` tuples.
#[pyfunction]
#[pyo3(name = "_rc_reticulation_tripartitions")]
fn rc_reticulation_tripartitions<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
) -> PyResult<Vec<(Bound<'py, PyFrozenSet>, Bound<'py, PyFrozenSet>, Bound<'py, PyFrozenSet>, String)>> {
    let mut interner = Interner::default();
    let tris = tripartitions(py, &net.borrow(), &mut interner)?;
    tris.into_iter()
        .map(|t| {
            Ok((
                frozenset_of_ids(py, &interner, &t.a)?,
                frozenset_of_ids(py, &interner, &t.b)?,
                frozenset_of_ids(py, &interner, &t.c)?,
                t.hybrid,
            ))
        })
        .collect()
}

/// ``tripartition_dissimilarity(theta, theta_prime, kind, normalize)`` on
/// two ``ReticulationTripartition`` objects.
#[pyfunction]
#[pyo3(name = "_rc_tripartition_dissimilarity")]
fn rc_tripartition_dissimilarity(
    py: Python<'_>,
    theta: &Bound<'_, PyAny>,
    theta_prime: &Bound<'_, PyAny>,
    kind: &Bound<'_, PyAny>,
    normalize: bool,
) -> PyResult<f64> {
    let d = |p: &Bound<'_, PyAny>, q: &Bound<'_, PyAny>| -> PyResult<f64> {
        let jaccard = ground_distance(py, kind)?;
        let (symdiff, union) = py_set_sizes(py, p, q)?;
        Ok(set_distance_sizes(symdiff, union, jaccard))
    };
    let (a, b, c) = (theta.getattr("A")?, theta.getattr("B")?, theta.getattr("C")?);
    let (e, f, g) = (theta_prime.getattr("A")?, theta_prime.getattr("B")?, theta_prime.getattr("C")?);
    let middle = d(&b, &f)?;
    let direct = d(&a, &e)? + d(&c, &g)?;
    let swapped = d(&a, &g)? + d(&c, &e)?;
    let value = middle + if swapped < direct { swapped } else { direct };
    Ok(if normalize { value / DELTA_MAX_JACCARD } else { value })
}

/// ``nakhleh_metric(net1, net2, normalize)``.
#[pyfunction]
#[pyo3(name = "_rc_nakhleh_metric")]
fn rc_nakhleh_metric(py: Python<'_>, net1: &Bound<'_, Network>, net2: &Bound<'_, Network>, normalize: bool) -> PyResult<f64> {
    nakhleh_metric_impl(py, &net1.borrow(), &net2.borrow(), normalize)
}

/// ``compare_networks(...)`` as the tuple ``NetworkComparison`` is built
/// from: ``(D, D_hat, D_ret, D_ret_hat, precision, recall, f1, r_reference,
/// r_inferred, recovered, matching, rho, delta_max)``.
#[pyfunction]
#[pyo3(name = "_rc_compare_networks")]
fn rc_compare_networks<'py>(
    py: Python<'py>,
    reference: &Bound<'py, Network>,
    inferred: &Bound<'py, Network>,
    distance: &Bound<'py, PyAny>,
    rho: Option<&Bound<'py, PyAny>>,
    tolerance: &Bound<'py, PyAny>,
    alpha: &Bound<'py, PyAny>,
) -> PyResult<Comparison> {
    compare_impl(py, reference, inferred, distance, rho, tolerance, alpha)
}

/// ``reticulation_dissimilarity(net1, net2, distance, rho, alpha, normalize)``.
#[pyfunction]
#[pyo3(name = "_rc_reticulation_dissimilarity")]
fn rc_reticulation_dissimilarity<'py>(
    py: Python<'py>,
    net1: &Bound<'py, Network>,
    net2: &Bound<'py, Network>,
    distance: &Bound<'py, PyAny>,
    rho: Option<&Bound<'py, PyAny>>,
    alpha: &Bound<'py, PyAny>,
    normalize: bool,
) -> PyResult<f64> {
    let zero = 0.0f64.into_pyobject(py)?;
    let (scores, ..) = compare_impl(py, net1, net2, distance, rho, zero.as_any(), alpha)?;
    Ok(if normalize { scores.3 } else { scores.2 })
}

/// ``reticulation_precision_recall(...)`` as ``(precision, recall, f1)``.
#[pyfunction]
#[pyo3(name = "_rc_reticulation_precision_recall")]
fn rc_reticulation_precision_recall<'py>(
    py: Python<'py>,
    reference: &Bound<'py, Network>,
    inferred: &Bound<'py, Network>,
    distance: &Bound<'py, PyAny>,
    rho: Option<&Bound<'py, PyAny>>,
    tolerance: &Bound<'py, PyAny>,
    alpha: &Bound<'py, PyAny>,
) -> PyResult<(f64, f64, f64)> {
    let (_, prec_rec, ..) = compare_impl(py, reference, inferred, distance, rho, tolerance, alpha)?;
    Ok(prec_rec)
}

/// ``combined_dissimilarity(net1, net2, lam, distance, rho, alpha, normalize)``.
#[pyfunction]
#[pyo3(name = "_rc_combined_dissimilarity")]
fn rc_combined_dissimilarity<'py>(
    py: Python<'py>,
    net1: &Bound<'py, Network>,
    net2: &Bound<'py, Network>,
    lam: &Bound<'py, PyAny>,
    distance: &Bound<'py, PyAny>,
    rho: Option<&Bound<'py, PyAny>>,
    alpha: &Bound<'py, PyAny>,
    normalize: bool,
) -> PyResult<f64> {
    let lam_v = number(lam)?;
    if !(0.0 <= lam_v && lam_v <= 1.0) {
        return Err(network_error(py, format!("lambda must lie in [0, 1]; got {}.", lam.str()?)));
    }
    let zero = 0.0f64.into_pyobject(py)?;
    let ((d, d_hat, d_ret, d_ret_hat), ..) = compare_impl(py, net1, net2, distance, rho, zero.as_any(), alpha)?;
    Ok(if normalize {
        lam_v * d_ret_hat + (1.0 - lam_v) * d_hat
    } else {
        lam_v * d_ret + (1.0 - lam_v) * d
    })
}

// ── JointDissimilarity: the light rooted DAG ──────────────────────────

/// A leaf label of a block or remainder: a taxon, or the marker hung for a
/// reticulation (labelled by the taxa below it).  The two namespaces never
/// collide, as ``("taxon", name)`` / ``("marker", taxa)`` in the reference.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Lab {
    Taxon(u32),
    Marker(Vec<u32>),
}

/// A cluster record: the multiset of labels below a vertex, as sorted
/// ``(label, multiplicity)`` pairs.
type Record = Vec<(Lab, u32)>;

/// Nakhleh's vertex signature on a ``Graph``.
#[derive(Hash, PartialEq, Eq)]
enum GSig {
    Leaf(Lab),
    Node(Vec<u32>),
}

/// ``_Graph``: a rooted DAG with labelled leaves and small-integer ids.
/// ``children[v]`` has no repeats; two leaves may share a label.  ``root``
/// is ``None`` only for the empty graph.  Derived views are cached lazily.
struct Graph {
    root: Option<usize>,
    children: FxHashMap<usize, Vec<usize>>,
    label: FxHashMap<usize, Lab>,
    leaves: OnceCell<FxHashSet<Lab>>,
    order: OnceCell<Vec<usize>>,
    clusters: OnceCell<FxHashMap<Record, i64>>,
}

impl Graph {
    fn new(root: Option<usize>, children: FxHashMap<usize, Vec<usize>>, label: FxHashMap<usize, Lab>) -> Graph {
        Graph { root, children, label, leaves: OnceCell::new(), order: OnceCell::new(), clusters: OnceCell::new() }
    }

    fn empty() -> Graph {
        Graph::new(None, FxHashMap::default(), FxHashMap::default())
    }

    fn kids(&self, v: usize) -> &[usize] {
        self.children.get(&v).map(Vec::as_slice).unwrap_or(&[])
    }

    /// The set of distinct leaf labels ``L(S)``.
    fn leaves(&self) -> &FxHashSet<Lab> {
        self.leaves.get_or_init(|| self.label.values().cloned().collect())
    }

    /// Vertices root first, every vertex before its children (Kahn).
    fn order(&self) -> &Vec<usize> {
        self.order.get_or_init(|| {
            let mut indeg: FxHashMap<usize, usize> = FxHashMap::default();
            for cs in self.children.values() {
                for &c in cs {
                    *indeg.entry(c).or_insert(0) += 1;
                }
            }
            let mut out = Vec::with_capacity(self.children.len());
            let mut ready: Vec<usize> = self.root.into_iter().collect();
            while let Some(v) = ready.pop() {
                out.push(v);
                for &c in self.kids(v) {
                    if let Some(d) = indeg.get_mut(&c) {
                        *d -= 1;
                        if *d == 0 {
                            ready.push(c);
                        }
                    }
                }
            }
            out
        })
    }

    /// Hardwired clusters of the FULL structure, one record per vertex: the
    /// multiset of labels of the leaf vertices reachable from the vertex.
    fn cluster_multiset(&self) -> &FxHashMap<Record, i64> {
        self.clusters.get_or_init(|| {
            let mut reach: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
            let mut out: FxHashMap<Record, i64> = FxHashMap::default();
            for &v in self.order().iter().rev() {
                let r: Vec<usize> = if self.label.contains_key(&v) {
                    vec![v]
                } else {
                    let mut acc: Vec<usize> = Vec::new();
                    for c in self.kids(v) {
                        if let Some(rc) = reach.get(c) {
                            acc.extend_from_slice(rc);
                        }
                    }
                    acc.sort_unstable();
                    acc.dedup();
                    acc
                };
                let mut counts: BTreeMap<&Lab, u32> = BTreeMap::new();
                for leaf in &r {
                    if let Some(l) = self.label.get(leaf) {
                        *counts.entry(l).or_insert(0) += 1;
                    }
                }
                let record: Record = counts.into_iter().map(|(l, n)| (l.clone(), n)).collect();
                *out.entry(record).or_insert(0) += 1;
                reach.insert(v, r);
            }
            out
        })
    }

    /// Nakhleh's vertex-equivalence classes of the FULL structure, interned
    /// in ``interner`` so two graphs keyed through the same table share ids.
    fn node_classes(&self, interner: &mut FxHashMap<GSig, u32>) -> FxHashMap<u32, i64> {
        let mut key: FxHashMap<usize, u32> = FxHashMap::default();
        let mut out: FxHashMap<u32, i64> = FxHashMap::default();
        for &v in self.order().iter().rev() {
            let sig = match self.label.get(&v) {
                Some(l) => GSig::Leaf(l.clone()),
                None => {
                    let mut ks: Vec<u32> = self.kids(v).iter().filter_map(|c| key.get(c).copied()).collect();
                    ks.sort_unstable();
                    GSig::Node(ks)
                }
            };
            let next = interner.len() as u32;
            let k = *interner.entry(sig).or_insert(next);
            key.insert(v, k);
            *out.entry(k).or_insert(0) += 1;
        }
        out
    }
}

/// ``_restrict``: the graph induced on ``keep``, rooted at ``root``.
fn restrict(g: &Graph, root: usize, keep: &FxHashSet<usize>) -> Graph {
    let mut children: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    let mut label: FxHashMap<usize, Lab> = FxHashMap::default();
    for &v in keep {
        children.insert(v, g.kids(v).iter().copied().filter(|c| keep.contains(c)).collect());
        if let Some(l) = g.label.get(&v) {
            label.insert(v, l.clone());
        }
    }
    Graph::new(Some(root), children, label)
}

/// ``_subgraph``: everything reachable from ``root`` without entering a
/// vertex of ``exclude``; the empty graph if ``root`` is excluded.
fn subgraph(g: &Graph, root: usize, exclude: &FxHashSet<usize>) -> Graph {
    if exclude.contains(&root) {
        return Graph::empty();
    }
    let mut seen: FxHashSet<usize> = FxHashSet::default();
    seen.insert(root);
    let mut stack = vec![root];
    while let Some(v) = stack.pop() {
        for &c in g.kids(v) {
            if !exclude.contains(&c) && seen.insert(c) {
                stack.push(c);
            }
        }
    }
    restrict(g, root, &seen)
}

/// ``_parent_block``: the vertices on a path from ``root`` to a taxon leaf
/// once ``exclude`` is deleted (nothing is suppressed).
fn parent_block(g: &Graph, root: usize, exclude: &FxHashSet<usize>) -> Graph {
    let s = subgraph(g, root, exclude);
    let s_root = match s.root {
        None => return Graph::empty(),
        Some(r) => r,
    };
    let mut keep: FxHashSet<usize> = FxHashSet::default();
    for &v in s.order().iter().rev() {
        if s.label.contains_key(&v) || s.kids(v).iter().any(|c| keep.contains(c)) {
            keep.insert(v);
        }
    }
    if keep.contains(&s_root) {
        restrict(&s, s_root, &keep)
    } else {
        Graph::empty()
    }
}

/// ``_remainder``: delete the blocks below all reticulations and hang one
/// marker leaf per surviving parent edge, labelled by the taxa below the
/// reticulation.  Unary vertices are kept.
fn remainder(g: &Graph, below: &[(usize, Graph)]) -> Graph {
    let mut deleted: FxHashSet<usize> = FxHashSet::default();
    let mut marker: FxHashMap<usize, Lab> = FxHashMap::default();
    for (h, b) in below {
        deleted.extend(b.children.keys().copied());
        let mut taxa: Vec<u32> = b
            .leaves()
            .iter()
            .filter_map(|l| match l {
                Lab::Taxon(t) => Some(*t),
                Lab::Marker(_) => None,
            })
            .collect();
        taxa.sort_unstable();
        marker.insert(*h, Lab::Marker(taxa));
    }
    let mut children: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    let mut label: FxHashMap<usize, Lab> = FxHashMap::default();
    let mut next_id = g.children.len();
    for &v in g.order() {
        if deleted.contains(&v) {
            continue;
        }
        let mut kids = Vec::new();
        for &c in g.kids(v) {
            match marker.get(&c) {
                Some(m) => {
                    children.insert(next_id, Vec::new());
                    label.insert(next_id, m.clone());
                    kids.push(next_id);
                    next_id += 1;
                }
                None => kids.push(c),
            }
        }
        children.insert(v, kids);
        if let Some(l) = g.label.get(&v) {
            label.insert(v, l.clone());
        }
    }
    Graph::new(g.root, children, label)
}

// ── JointDissimilarity: the three instantiations of d(S, S') ──────────

#[derive(Clone, Copy)]
enum Kind {
    Leaf,
    Cluster,
    Nakhleh,
}

/// ``sum_k |C1[k] - C2[k]| / sum_k max(C1[k], C2[k])``; 0 when both empty.
fn multiset_dissimilarity<K: std::hash::Hash + Eq>(c1: &FxHashMap<K, i64>, c2: &FxHashMap<K, i64>) -> f64 {
    let mut union = 0i64;
    let mut diff = 0i64;
    for (k, &a) in c1 {
        let b = c2.get(k).copied().unwrap_or(0);
        union += a.max(b);
        diff += (a - b).abs();
    }
    for (k, &b) in c2 {
        if !c1.contains_key(k) {
            union += b.max(0);
            diff += b.abs();
        }
    }
    if union != 0 {
        diff as f64 / union as f64
    } else {
        0.0
    }
}

impl Kind {
    fn d(self, s: &Graph, t: &Graph) -> f64 {
        match self {
            Kind::Leaf => {
                let (p, q) = (s.leaves(), t.leaves());
                if p.is_empty() && q.is_empty() {
                    return 0.0;
                }
                let common = p.iter().filter(|l| q.contains(*l)).count();
                let symdiff = p.len() + q.len() - 2 * common;
                let union = p.len() + q.len() - common;
                symdiff as f64 / union as f64
            }
            Kind::Cluster => multiset_dissimilarity(s.cluster_multiset(), t.cluster_multiset()),
            Kind::Nakhleh => {
                let mut interner: FxHashMap<GSig, u32> = FxHashMap::default();
                let cs = s.node_classes(&mut interner);
                let ct = t.node_classes(&mut interner);
                multiset_dissimilarity(&cs, &ct)
            }
        }
    }
}

// ── JointDissimilarity: from a Network to tripartitions and remainder ─

/// The reduction of one network: its tripartitions ``(A, B, C)``, its
/// remainder ``T(N)`` and its taxon set.
struct Prepared {
    tri: Vec<[Graph; 3]>,
    rem: Graph,
    taxa: FxHashSet<u32>,
}

/// ``_validate`` + ``_network_graph`` + ``_prepare``.
fn prepare(py: Python<'_>, net: &Network, interner: &mut Interner) -> PyResult<Prepared> {
    if net.nodes.is_empty() {
        return Err(network_error(py, "joint_dissimilarity: empty network"));
    }
    let order = net.topological_order_uids(py)?; // raises on a cycle
    let roots = net.nodes.keys().filter(|&&u| net.in_deg(u) == 0).count();
    if roots != 1 {
        return Err(network_error(
            py,
            format!("joint_dissimilarity: network has {} roots; a rooted network has exactly one", roots),
        ));
    }
    let labels: Vec<String> = net.nodes.keys().filter(|&&u| net.out_deg(u) == 0).map(|&u| net.label(py, u)).collect();
    if labels.iter().any(|l| l.is_empty()) {
        return Err(network_error(py, "joint_dissimilarity: network has an unlabelled leaf"));
    }
    let mut counts: FxHashMap<&str, usize> = FxHashMap::default();
    for l in &labels {
        *counts.entry(l.as_str()).or_insert(0) += 1;
    }
    let mut dup: Vec<&str> = counts.iter().filter(|(_, &c)| c > 1).map(|(&l, _)| l).collect();
    if !dup.is_empty() {
        return Err(network_error(
            py,
            format!("joint_dissimilarity: duplicate leaf labels {}", sorted_repr(py, &mut dup)?),
        ));
    }
    // The tripartition of a reticulation needs two distinct parent blocks
    // and one block below it.
    for &h in net.nodes.keys() {
        if net.in_deg(h) < 2 {
            continue;
        }
        let parents = net.parents(h);
        if parents.len() != 2 || parents[0] == parents[1] {
            return Err(network_error(
                py,
                format!(
                    "joint_dissimilarity: reticulation {} has {} parent edges; exactly two distinct parents are required",
                    net.label(py, h),
                    parents.len()
                ),
            ));
        }
        if net.out_deg(h) != 1 {
            return Err(network_error(
                py,
                format!(
                    "joint_dissimilarity: reticulation {} has {} children; exactly one is required",
                    net.label(py, h),
                    net.out_deg(h)
                ),
            ));
        }
    }

    // ``_network_graph``: small-integer ids in topological order, 0 the root.
    let id_of: FxHashMap<u64, usize> = order.iter().enumerate().map(|(i, &u)| (u, i)).collect();
    let mut children: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    let mut label: FxHashMap<usize, Lab> = FxHashMap::default();
    for (i, &u) in order.iter().enumerate() {
        let mut kids: Vec<usize> = Vec::new();
        for c in net.children(u) {
            if let Some(&cid) = id_of.get(&c) {
                if !kids.contains(&cid) {
                    kids.push(cid);
                }
            }
        }
        children.insert(i, kids);
        if net.out_deg(u) == 0 {
            label.insert(i, Lab::Taxon(interner.intern(&net.label(py, u))));
        }
    }
    let g = Graph::new(Some(0), children, label);

    // ``_prepare``: one tripartition per reticulation, in topological order.
    let mut retics: Vec<(usize, [usize; 2])> = Vec::new();
    for &u in &order {
        if net.in_deg(u) >= 2 {
            let ps = net.parents(u);
            match (id_of.get(&u), id_of.get(&ps[0]), id_of.get(&ps[1])) {
                (Some(&h), Some(&a), Some(&b)) => retics.push((h, [a, b])),
                _ => continue,
            }
        }
    }
    let no_exclude = FxHashSet::default();
    let below: Vec<(usize, Graph)> = retics.iter().map(|&(h, _)| (h, subgraph(&g, h, &no_exclude))).collect();
    let rem = remainder(&g, &below);
    let mut tri = Vec::with_capacity(retics.len());
    for ((h, [u, v]), (_, b)) in retics.into_iter().zip(below.into_iter()) {
        let excl: FxHashSet<usize> = std::iter::once(h).collect();
        tri.push([parent_block(&g, u, &excl), b, parent_block(&g, v, &excl)]);
    }
    let taxa: FxHashSet<u32> = g
        .leaves()
        .iter()
        .filter_map(|l| match l {
            Lab::Taxon(t) => Some(*t),
            Lab::Marker(_) => None,
        })
        .collect();
    Ok(Prepared { tri, rem, taxa })
}

/// ``_d_ret``: pad the shorter list with dummies (real--dummy cost 3,
/// dummy--dummy 0) and take the minimum-cost bijection.
fn d_ret(t1: &[[Graph; 3]], t2: &[[Graph; 3]], kind: Kind) -> PyResult<f64> {
    let m = t1.len().max(t2.len());
    if m == 0 {
        return Ok(0.0);
    }
    let mut cost = vec![DELTA_MAX_JACCARD; m * m];
    for i in t1.len()..m {
        for j in t2.len()..m {
            cost[i * m + j] = 0.0;
        }
    }
    for (i, [a, b, c]) in t1.iter().enumerate() {
        for (j, [d, e, f]) in t2.iter().enumerate() {
            let direct = kind.d(a, d) + kind.d(c, f);
            let swapped = kind.d(a, f) + kind.d(c, d);
            cost[i * m + j] = kind.d(b, e) + if swapped < direct { swapped } else { direct };
        }
    }
    let (total, _) = solve_assignment(m, &cost)?;
    Ok(total)
}

/// ``joint_dissimilarity(net1, net2, kind)`` as ``(r1, r2, D_ret, D_tree)``.
#[pyfunction]
#[pyo3(name = "_jd_joint_dissimilarity")]
fn jd_joint_dissimilarity(
    py: Python<'_>,
    net1: &Bound<'_, Network>,
    net2: &Bound<'_, Network>,
    kind: &Bound<'_, PyAny>,
) -> PyResult<(usize, usize, f64, f64)> {
    let kind_v = match kind.extract::<String>().ok().as_deref() {
        Some("leaf") => Kind::Leaf,
        Some("cluster") => Kind::Cluster,
        Some("nakhleh") => Kind::Nakhleh,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown instantiation {}; choose from ('leaf', 'cluster', 'nakhleh')",
                kind.repr()?
            )))
        }
    };
    let mut interner = Interner::default();
    let p1 = prepare(py, &net1.borrow(), &mut interner)?;
    let p2 = prepare(py, &net2.borrow(), &mut interner)?;
    if p1.taxa != p2.taxa {
        let mut x1: Vec<&str> = p1.taxa.iter().map(|&t| interner.name(t)).collect();
        let mut x2: Vec<&str> = p2.taxa.iter().map(|&t| interner.name(t)).collect();
        return Err(network_error(
            py,
            format!(
                "joint_dissimilarity requires identical leaf label sets. Got {} vs {}",
                sorted_repr(py, &mut x1)?,
                sorted_repr(py, &mut x2)?
            ),
        ));
    }
    let ret = d_ret(&p1.tri, &p2.tri, kind_v)?;
    let tree = kind_v.d(&p1.rem, &p2.rem);
    Ok((p1.tri.len(), p2.tri.len(), ret, tree))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rc_set_distance, m)?)?;
    m.add_function(wrap_pyfunction!(rc_reticulation_tripartitions, m)?)?;
    m.add_function(wrap_pyfunction!(rc_tripartition_dissimilarity, m)?)?;
    m.add_function(wrap_pyfunction!(rc_nakhleh_metric, m)?)?;
    m.add_function(wrap_pyfunction!(rc_compare_networks, m)?)?;
    m.add_function(wrap_pyfunction!(rc_reticulation_dissimilarity, m)?)?;
    m.add_function(wrap_pyfunction!(rc_reticulation_precision_recall, m)?)?;
    m.add_function(wrap_pyfunction!(rc_combined_dissimilarity, m)?)?;
    m.add_function(wrap_pyfunction!(jd_joint_dissimilarity, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assignment_matches_hand_solutions() {
        // Identity on a constant matrix (scipy #11602).
        assert_eq!(linear_sum_assignment(3, &[1.0; 9]), Some(vec![0, 1, 2]));
        // A unique optimum.
        let cost = [4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0];
        assert_eq!(linear_sum_assignment(3, &cost), Some(vec![1, 0, 2]));
        // scipy's tie-breaking on [[0, 1], [1, 2]]: the diagonal.
        assert_eq!(linear_sum_assignment(2, &[0.0, 1.0, 1.0, 2.0]), Some(vec![0, 1]));
    }

    #[test]
    fn pairwise_sum_is_a_sum() {
        let a: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        assert_eq!(numpy_sum(&a), 210.0);
        assert_eq!(numpy_sum(&[]), 0.0);
    }

    #[test]
    fn id_set_algebra() {
        assert_eq!(ids_union(&[1, 3, 5], &[2, 3, 6]), vec![1, 2, 3, 5, 6]);
        assert_eq!(ids_difference(&[1, 3, 5], &[3]), vec![1, 5]);
        assert_eq!(ids_intersection(&[1, 3, 5], &[3, 5, 7]), vec![3, 5]);
        assert_eq!(ids_symdiff_union(&[1, 2], &[2, 3]), (2, 3));
    }
}
