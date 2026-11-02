//! Multispecies (network) coalescent DPs (Yu, Degnan & Nakhleh 2012).
//!
//! Three dynamic programmes over lineage bitmasks, each generic in the mask
//! width so gene trees of any size run in native integers:
//!
//! * the topology-only frontier DP on a species network
//!   (``_msnc_log_prob_network_int``), used by MCMC_GT;
//! * the timed (event-based) frontier DP (``_msnc_log_density_timed``),
//!   used by MCMC_SEQ;
//! * the species-*tree* DP on a displayed tree (``_msc_log_prob_tree_int``).
//!
//! The Python classes ``MSCBranchKernel``, ``_GeneTreeIndex`` and
//! ``_NetworkIndex`` keep their upstream attribute surface.

use indexmap::IndexMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PyTuple};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::common::{gij_clamped, ln_factorial, logaddexp, logsumexp, LOG_FLOOR};
use super::mask::{Mask, Words};
use crate::graph::{Edge, Network, Node};

// ═══════════════════════════════════════════════════════════════════════
// Branch kernel
// ═══════════════════════════════════════════════════════════════════════

/// Kingman branch-coalescent kernel with explicit ``theta`` (``4 N mu``).
#[pyclass(name = "MSCBranchKernel", module = "phynetpy._msnc_density", subclass)]
pub struct MSCBranchKernel {
    theta: f64,
    gij_cache: FxHashMap<(u64, u32, u32), f64>,
    denom_cache: FxHashMap<(u32, u32), f64>,
}

impl MSCBranchKernel {
    pub fn with_theta(theta: f64) -> Self {
        MSCBranchKernel {
            theta,
            gij_cache: FxHashMap::default(),
            denom_cache: FxHashMap::default(),
        }
    }

    #[inline]
    fn scaled(&self, length: Option<f64>) -> Option<f64> {
        length.map(|l| l * 2.0 / self.theta)
    }

    /// ``log g_{ij}(length)`` with lengths in substitution units scaled by
    /// ``2 / theta``.
    #[inline]
    pub fn log_gij(&mut self, length: Option<f64>, i: u32, j: u32) -> f64 {
        let key = (length.map(f64::to_bits).unwrap_or(u64::MAX), i, j);
        if let Some(v) = self.gij_cache.get(&key) {
            return *v;
        }
        let val = gij_clamped(self.scaled(length), i, j);
        let lv = if val <= 0.0 { LOG_FLOOR } else { val.ln() };
        self.gij_cache.insert(key, lv);
        lv
    }

    /// ``log prod_{i=1..k} C(n-i+1, 2)``.
    #[inline]
    pub fn log_denom(&mut self, n: u32, k: u32) -> f64 {
        if k == 0 {
            return 0.0;
        }
        if let Some(v) = self.denom_cache.get(&(n, k)) {
            return *v;
        }
        let mut acc = 0.0;
        let mut zero = false;
        for i in 1..=k {
            let a = n as i64 - i as i64 + 1;
            let b = n as i64 - i as i64;
            let term = (a * b) / 2;
            if term <= 0 {
                zero = true;
                break;
            }
            acc += (term as f64).ln();
        }
        let v = if zero { LOG_FLOOR } else { acc };
        self.denom_cache.insert((n, k), v);
        v
    }
}

#[pymethods]
impl MSCBranchKernel {
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn py_new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut theta = 2.0;
        if let Some(kw) = kwargs {
            if let Some(t) = kw.get_item("theta")? {
                theta = t.extract::<f64>()?;
            }
        } else if let Ok(first) = args.get_item(0) {
            if let Ok(t) = first.extract::<f64>() {
                theta = t;
            }
        }
        Ok(MSCBranchKernel::with_theta(theta))
    }

    #[getter]
    fn get_theta(&self) -> f64 {
        self.theta
    }

    #[setter]
    fn set_theta(&mut self, value: f64) {
        if value != self.theta {
            self.gij_cache.clear();
            self.denom_cache.clear();
        }
        self.theta = value;
    }

    fn _scaled_length(&self, length: Option<f64>) -> Option<f64> {
        self.scaled(length)
    }

    #[staticmethod]
    fn _fact_range(start: i64, end: i64) -> f64 {
        super::common::fact_range(start, end)
    }

    #[staticmethod]
    fn _gij_raw(length: Option<f64>, i: u32, j: u32) -> f64 {
        gij_clamped(length, i, j)
    }

    fn _log_gij(&mut self, length: Option<f64>, i: u32, j: u32) -> f64 {
        self.log_gij(length, i, j)
    }

    fn _log_denom(&mut self, n: u32, k: u32) -> f64 {
        self.log_denom(n, k)
    }

    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let d = PyDict::new(py);
        d.set_item("theta", slf.borrow().theta)?;
        if let Ok(dict) = slf.getattr("__dict__") {
            d.set_item("__dict__", dict)?;
        }
        Ok(d)
    }

    fn __setstate__(slf: &Bound<'_, Self>, state: Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(t) = state.get_item("theta")? {
            slf.borrow_mut().theta = t.extract()?;
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

// ═══════════════════════════════════════════════════════════════════════
// Gene-tree index
// ═══════════════════════════════════════════════════════════════════════

/// One pre-computed coarsening of an active lineage set.
#[derive(Clone, Copy, Debug)]
pub struct Row<M: Mask> {
    pub cfg_out: M,
    pub merge: M,
    pub m_out: u32,
    pub k: u32,
    pub log_le: f64,
}

#[derive(Default)]
pub struct Caches<M: Mask> {
    coarse: FxHashMap<M, Vec<Row<M>>>,
    log_le: FxHashMap<M, f64>,
}

pub enum MaskCaches {
    U64(Caches<u64>),
    U128(Caches<u128>),
    W4(Caches<Words<4>>),
    W8(Caches<Words<8>>),
    W16(Caches<Words<16>>),
    W32(Caches<Words<32>>),
}

impl MaskCaches {
    fn for_bits(n: usize) -> PyResult<MaskCaches> {
        Ok(if n <= 64 {
            MaskCaches::U64(Caches::default())
        } else if n <= 128 {
            MaskCaches::U128(Caches::default())
        } else if n <= 256 {
            MaskCaches::W4(Caches::default())
        } else if n <= 512 {
            MaskCaches::W8(Caches::default())
        } else if n <= 1024 {
            MaskCaches::W16(Caches::default())
        } else if n <= 2048 {
            MaskCaches::W32(Caches::default())
        } else {
            return Err(PyValueError::new_err(format!(
                "gene tree has {n} nodes; the coalescent DP supports at most 2048"
            )));
        })
    }
}

/// Run ``$body`` with ``$c`` bound to the typed cache and ``$m`` to the
/// mask type of a gene-tree index.
macro_rules! with_mask {
    ($caches:expr, |$c:ident, $m:ident| $body:expr) => {
        match $caches {
            MaskCaches::U64($c) => {
                type $m = u64;
                $body
            }
            MaskCaches::U128($c) => {
                type $m = u128;
                $body
            }
            MaskCaches::W4($c) => {
                type $m = Words<4>;
                $body
            }
            MaskCaches::W8($c) => {
                type $m = Words<8>;
                $body
            }
            MaskCaches::W16($c) => {
                type $m = Words<16>;
                $body
            }
            MaskCaches::W32($c) => {
                type $m = Words<32>;
                $body
            }
        }
    };
}

/// The topology of one gene tree, indexed by bit.
pub struct GtTopo {
    pub n_total: usize,
    pub children: Vec<Option<(usize, usize)>>,
    pub parent: Vec<Option<usize>>,
    pub root_bit: i64,
    pub leaves: Vec<usize>,
    pub leaf_species: Vec<Option<String>>,
}

impl GtTopo {
    fn size_within<M: Mask>(&self, root_bit: usize, merge: &M) -> u32 {
        let mut count = 0;
        let mut stack = vec![root_bit];
        while let Some(cur) = stack.pop() {
            if merge.test(cur) {
                count += 1;
            }
            if let Some((l, r)) = self.children[cur] {
                stack.push(l);
                stack.push(r);
            }
        }
        count
    }

    pub fn log_le<M: Mask>(&self, merge: M, cache: &mut FxHashMap<M, f64>) -> f64 {
        if let Some(v) = cache.get(&merge) {
            return *v;
        }
        let bits = merge.bits();
        let v = if bits.len() <= 1 {
            0.0
        } else {
            let k = bits.len() as u32;
            let denom: f64 = bits
                .iter()
                .map(|&b| (self.size_within(b, &merge) as f64).ln())
                .sum();
            ln_factorial(k) - denom
        };
        cache.insert(merge, v);
        v
    }

    /// Every reachable coarsening of ``active``: BFS over (active, merged)
    /// states in the reference implementation's order.
    pub fn coarsenings<'a, M: Mask>(&self, caches: &'a mut Caches<M>, active: M) -> &'a Vec<Row<M>> {
        if !caches.coarse.contains_key(&active) {
            let mut states: Vec<(M, M)> = Vec::new();
            let mut seen: FxHashSet<(M, M)> = FxHashSet::default();
            let mut frontier: Vec<(M, M)> = vec![(active, M::zero())];
            while !frontier.is_empty() {
                let mut next: Vec<(M, M)> = Vec::new();
                for state in frontier {
                    if !seen.insert(state) {
                        continue;
                    }
                    states.push(state);
                    let (act, merged) = state;
                    // Group active bits by parent, first-seen order.
                    let mut groups: IndexMap<usize, SmallVec<[usize; 2]>, FxBuildHasher> =
                        IndexMap::default();
                    let mut m = act;
                    while !m.is_zero() {
                        let bit = m.lowest();
                        m = m.clear_lowest();
                        if let Some(p) = self.parent[bit] {
                            groups.entry(p).or_default().push(bit);
                        }
                    }
                    for (p, kids) in groups.iter() {
                        if merged.test(*p) {
                            continue;
                        }
                        let Some((lb, rb)) = self.children[*p] else { continue };
                        if kids.len() < 2 {
                            continue;
                        }
                        if kids.contains(&lb) && kids.contains(&rb) {
                            let new_active = act.andnot(M::bit(lb).or(M::bit(rb))).or(M::bit(*p));
                            let new_merged = merged.or(M::bit(*p));
                            let nxt = (new_active, new_merged);
                            if !seen.contains(&nxt) {
                                next.push(nxt);
                            }
                        }
                    }
                }
                frontier = next;
            }
            let n_in = active.popcount();
            let mut rows = Vec::with_capacity(states.len());
            for (out, merge) in states {
                let m_out = out.popcount();
                let log_le = self.log_le(merge, &mut caches.log_le);
                rows.push(Row {
                    cfg_out: out,
                    merge,
                    m_out,
                    k: n_in - m_out,
                    log_le,
                });
            }
            caches.coarse.insert(active, rows);
        }
        &caches.coarse[&active]
    }

    /// Species label -> mask of gene leaves mapped there.
    pub fn species_to_bits<M: Mask>(&self) -> FxHashMap<String, M> {
        let mut out: FxHashMap<String, M> = FxHashMap::default();
        for &leaf in &self.leaves {
            if let Some(sp) = &self.leaf_species[leaf] {
                let e = out.entry(sp.clone()).or_insert_with(M::zero);
                *e = e.or(M::bit(leaf));
            }
        }
        out
    }
}

/// Per-gene-tree static-poset cache used by the bitmask MSC DP.
#[pyclass(name = "_GeneTreeIndex", module = "phynetpy._msnc_density")]
pub struct GeneTreeIndex {
    pub topo: GtTopo,
    bit_of: Py<PyDict>,
    labels: Vec<String>,
    pub caches: MaskCaches,
}

impl GeneTreeIndex {
    fn species_map(species_of: Option<&Bound<'_, PyAny>>) -> PyResult<FxHashMap<String, String>> {
        let mut out = FxHashMap::default();
        if let Some(s) = species_of {
            if !s.is_none() {
                for (k, v) in s.cast::<PyDict>()?.iter() {
                    out.insert(k.extract::<String>()?, v.extract::<String>()?);
                }
            }
        }
        Ok(out)
    }
}

#[pymethods]
impl GeneTreeIndex {
    #[new]
    #[pyo3(signature = (gene_tree, species_of=None))]
    fn py_new(
        py: Python<'_>,
        gene_tree: &Bound<'_, Network>,
        species_of: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let species = Self::species_map(species_of.as_ref())?;
        let net = gene_tree.borrow();
        let uids: Vec<u64> = net.nodes.keys().copied().collect();
        let n = uids.len();
        let index: FxHashMap<u64, usize> = uids.iter().enumerate().map(|(i, u)| (*u, i)).collect();
        let bit_of = PyDict::new(py);
        let mut labels = Vec::with_capacity(n);
        let mut children: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut leaves = Vec::new();
        let mut leaf_species: Vec<Option<String>> = vec![None; n];
        for (i, &uid) in uids.iter().enumerate() {
            let obj = net.node_obj(uid);
            bit_of.set_item(obj.bind(py), i)?;
            let label = obj.borrow(py).name.clone();
            let kids = net.children(uid);
            if kids.is_empty() {
                leaves.push(i);
                leaf_species[i] = species.get(&label).cloned();
            } else if kids.len() >= 2 {
                let lb = index[&kids[0]];
                let rb = index[&kids[1]];
                children[i] = Some((lb, rb));
                parent[lb] = Some(i);
                parent[rb] = Some(i);
            }
            labels.push(label);
        }
        let root_bit = match net.root_uids().first() {
            Some(r) => index.get(r).map(|&i| i as i64).unwrap_or(-1),
            None => -1,
        };
        let root_bit = if net.root_uids().is_empty() {
            -1
        } else {
            match net.root_uid(py) {
                Ok(r) => index.get(&r).map(|&i| i as i64).unwrap_or(-1),
                Err(_) => root_bit,
            }
        };
        Ok(GeneTreeIndex {
            topo: GtTopo {
                n_total: n,
                children,
                parent,
                root_bit,
                leaves,
                leaf_species,
            },
            bit_of: bit_of.unbind(),
            labels,
            caches: MaskCaches::for_bits(n)?,
        })
    }

    #[getter]
    fn n_total(&self) -> usize {
        self.topo.n_total
    }

    #[getter]
    fn bit_of(&self, py: Python<'_>) -> Py<PyDict> {
        self.bit_of.clone_ref(py)
    }

    #[getter]
    fn leaves(&self) -> Vec<usize> {
        self.topo.leaves.clone()
    }

    #[getter]
    fn leaf_species_of(&self) -> FxHashMap<usize, String> {
        self.topo
            .leaves
            .iter()
            .filter_map(|&l| self.topo.leaf_species[l].clone().map(|s| (l, s)))
            .collect()
    }

    #[getter]
    fn internals_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let bits: Vec<usize> = (0..self.topo.n_total).filter(|&i| self.topo.children[i].is_some()).collect();
        bits_to_py_int(py, &bits)
    }

    #[getter]
    fn children(&self) -> FxHashMap<usize, (usize, usize)> {
        (0..self.topo.n_total)
            .filter_map(|i| self.topo.children[i].map(|c| (i, c)))
            .collect()
    }

    #[getter]
    fn parent(&self) -> FxHashMap<usize, usize> {
        (0..self.topo.n_total)
            .filter_map(|i| self.topo.parent[i].map(|p| (i, p)))
            .collect()
    }

    #[getter]
    fn root_bit(&self) -> i64 {
        self.topo.root_bit
    }

    #[pyo3(signature = (gene_tree, species_of=None))]
    fn refresh_species(
        &mut self,
        gene_tree: &Bound<'_, PyAny>,
        species_of: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let _ = gene_tree;
        let species = Self::species_map(species_of.as_ref())?;
        for &leaf in &self.topo.leaves.clone() {
            self.topo.leaf_species[leaf] = species.get(&self.labels[leaf]).cloned();
        }
        Ok(())
    }

    fn _cached_leaves_set(&self) -> FxHashSet<usize> {
        self.topo.leaves.iter().copied().collect()
    }

    /// Exact ``|L(F)|`` (hook-length formula) as a Python integer.
    fn linear_extensions<'py>(&self, py: Python<'py>, merge_mask: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let bits = py_int_bits(&merge_mask)?;
        if bits.len() <= 1 {
            return Ok(1i64.into_pyobject(py)?.into_any());
        }
        let math = py.import("math")?;
        let k = bits.len();
        let mut denom = 1i64.into_pyobject(py)?.into_any();
        for &v in &bits {
            // size_within on the bit set
            let mut count = 0i64;
            let mut stack = vec![v];
            while let Some(cur) = stack.pop() {
                if bits.contains(&cur) {
                    count += 1;
                }
                if let Some((l, r)) = self.topo.children[cur] {
                    stack.push(l);
                    stack.push(r);
                }
            }
            denom = denom.call_method1("__mul__", (count,))?;
        }
        let num = math.getattr("factorial")?.call1((k,))?;
        num.call_method1("__floordiv__", (denom,))
    }

    /// Pre-computed coarsenings as ``(cfg_out, merge_mask, m_out, k, log_le)`` rows.
    fn coarsenings<'py>(&mut self, py: Python<'py>, active_mask: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
        let topo = &self.topo;
        with_mask!(&mut self.caches, |c, M| {
            let active = M::from_py(&active_mask)?;
            let rows = topo.coarsenings::<M>(c, active).clone();
            let items: Vec<Bound<'py, PyAny>> = rows
                .iter()
                .map(|r| -> PyResult<Bound<'py, PyAny>> {
                    Ok(PyTuple::new(
                        py,
                        [
                            r.cfg_out.to_py(py)?,
                            r.merge.to_py(py)?,
                            r.m_out.into_pyobject(py)?.into_any(),
                            r.k.into_pyobject(py)?.into_any(),
                            r.log_le.into_pyobject(py)?.into_any(),
                        ],
                    )?
                    .into_any())
                })
                .collect::<PyResult<_>>()?;
            PyTuple::new(py, items)
        })
    }
}

fn py_int_bits(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let int = obj.cast::<PyInt>()?;
    let nbits: usize = int.call_method0("bit_length")?.extract()?;
    let nbytes = nbits.div_ceil(8).max(1);
    let bytes = int
        .call_method1("to_bytes", (nbytes, "little"))?
        .cast_into::<pyo3::types::PyBytes>()?;
    let mut out = Vec::new();
    for (i, b) in bytes.as_bytes().iter().enumerate() {
        for j in 0..8 {
            if (b >> j) & 1 == 1 {
                out.push(i * 8 + j);
            }
        }
    }
    Ok(out)
}

fn bits_to_py_int<'py>(py: Python<'py>, bits: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    let mut acc = 0i64.into_pyobject(py)?.into_any();
    let one = 1i64.into_pyobject(py)?.into_any();
    for &b in bits {
        let bit = one.call_method1("__lshift__", (b,))?;
        acc = acc.call_method1("__or__", (bit,))?;
    }
    Ok(acc)
}

// ═══════════════════════════════════════════════════════════════════════
// Network index
// ═══════════════════════════════════════════════════════════════════════

/// Int-indexed view of a species network for the AC DP.
#[pyclass(name = "_NetworkIndex", module = "phynetpy._msnc_density")]
pub struct NetworkIndex {
    #[pyo3(get)]
    pub n_nodes: usize,
    #[pyo3(get)]
    pub n_edges: usize,
    #[pyo3(get)]
    pub is_retic: Vec<bool>,
    #[pyo3(get)]
    pub is_leaf: Vec<bool>,
    pub leaf_label: Vec<Option<String>>,
    #[pyo3(get)]
    pub down_edges: Vec<Vec<usize>>,
    #[pyo3(get)]
    pub up_edges: Vec<Vec<usize>>,
    #[pyo3(get)]
    pub edge_src: Vec<usize>,
    #[pyo3(get)]
    pub edge_dst: Vec<usize>,
    #[pyo3(get)]
    pub topo_order: Vec<usize>,
    #[pyo3(get)]
    pub root: i64,
    node_objs: Vec<Py<Node>>,
    edge_objs: Vec<Py<Edge>>,
}

/// Branch parameters read live from the network's edges at DP time.
pub struct EdgeParams {
    pub length: Vec<Option<f64>>,
    pub gamma: Vec<Option<f64>>,
}

impl NetworkIndex {
    pub fn params(&self, py: Python<'_>) -> EdgeParams {
        let mut length = Vec::with_capacity(self.n_edges);
        let mut gamma = Vec::with_capacity(self.n_edges);
        for e in &self.edge_objs {
            let b = e.borrow(py);
            length.push(b.length);
            gamma.push(b.gamma);
        }
        EdgeParams { length, gamma }
    }
}

#[pymethods]
impl NetworkIndex {
    #[new]
    fn py_new(py: Python<'_>, net: &Bound<'_, Network>) -> Self {
        let net = net.borrow();
        let uids: Vec<u64> = net.nodes.keys().copied().collect();
        let n = uids.len();
        let index: FxHashMap<u64, usize> = uids.iter().enumerate().map(|(i, u)| (*u, i)).collect();
        let node_objs: Vec<Py<Node>> = uids.iter().map(|u| net.node_obj(*u).clone_ref(py)).collect();
        let is_retic: Vec<bool> = node_objs.iter().map(|o| o.borrow(py).is_retic).collect();
        let is_leaf: Vec<bool> = uids.iter().map(|&u| net.out_deg(u) == 0).collect();
        let leaf_label: Vec<Option<String>> = (0..n)
            .map(|i| if is_leaf[i] { Some(node_objs[i].borrow(py).name.clone()) } else { None })
            .collect();
        let mut down_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut up_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut edge_src = Vec::new();
        let mut edge_dst = Vec::new();
        let mut edge_objs = Vec::new();
        for (j, entry) in net.edges.values().enumerate() {
            let s = index[&entry.su];
            let d = index[&entry.du];
            edge_src.push(s);
            edge_dst.push(d);
            down_edges[s].push(j);
            up_edges[d].push(j);
            edge_objs.push(entry.obj.clone_ref(py));
        }
        let n_edges = edge_objs.len();
        // Bottom-up Kahn order.
        let mut remaining: Vec<usize> = (0..n).map(|v| down_edges[v].len()).collect();
        let mut ready: std::collections::VecDeque<usize> =
            (0..n).filter(|&v| remaining[v] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(v) = ready.pop_front() {
            order.push(v);
            for &j in &up_edges[v] {
                let p = edge_src[j];
                remaining[p] -= 1;
                if remaining[p] == 0 {
                    ready.push_back(p);
                }
            }
        }
        let root = (0..n).find(|&v| up_edges[v].is_empty()).map(|v| v as i64).unwrap_or(-1);
        NetworkIndex {
            n_nodes: n,
            n_edges,
            is_retic,
            is_leaf,
            leaf_label,
            down_edges,
            up_edges,
            edge_src,
            edge_dst,
            topo_order: order,
            root,
            node_objs,
            edge_objs,
        }
    }

    #[getter]
    fn leaf_label(&self) -> FxHashMap<usize, String> {
        self.leaf_label
            .iter()
            .enumerate()
            .filter_map(|(i, l)| l.clone().map(|s| (i, s)))
            .collect()
    }

    #[getter]
    fn _node_objs(&self, py: Python<'_>) -> Vec<Py<Node>> {
        self.node_objs.iter().map(|o| o.clone_ref(py)).collect()
    }

    #[getter]
    fn _edge_objs(&self, py: Python<'_>) -> Vec<Py<Edge>> {
        self.edge_objs.iter().map(|o| o.clone_ref(py)).collect()
    }

    fn edge_length(&self, py: Python<'_>, edge_id: usize) -> Option<f64> {
        self.edge_objs[edge_id].borrow(py).length
    }

    fn edge_gamma(&self, py: Python<'_>, edge_id: usize) -> Option<f64> {
        self.edge_objs[edge_id].borrow(py).gamma
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Displayed-tree index
// ═══════════════════════════════════════════════════════════════════════

/// Int-indexed view of one displayed (species) tree.
#[pyclass(name = "_DisplayedTreeIndex", module = "phynetpy._msnc_density")]
pub struct DisplayedTreeIndex {
    pub n: usize,
    pub children: Vec<Vec<usize>>,
    pub edge_lengths: Vec<Vec<Option<f64>>>,
    pub leaves: Vec<usize>,
    pub leaf_species: FxHashMap<usize, String>,
    pub root: i64,
}

#[pymethods]
impl DisplayedTreeIndex {
    #[new]
    fn py_new(
        n: usize,
        children: Vec<Vec<usize>>,
        edge_lengths: Vec<Vec<Option<f64>>>,
        leaves: Vec<usize>,
        leaf_species: FxHashMap<usize, String>,
        root: i64,
    ) -> Self {
        DisplayedTreeIndex {
            n,
            children,
            edge_lengths,
            leaves,
            leaf_species,
            root,
        }
    }
}

fn displayed_index_from(dt: &Bound<'_, PyAny>) -> PyResult<Py<DisplayedTreeIndex>> {
    let py = dt.py();
    if let Ok(d) = dt.cast::<DisplayedTreeIndex>() {
        return Ok(d.clone().unbind());
    }
    if let Ok(cached) = dt.getattr("_rs_idx") {
        if let Ok(d) = cached.cast::<DisplayedTreeIndex>() {
            return Ok(d.clone().unbind());
        }
    }
    let idx = DisplayedTreeIndex {
        n: dt.getattr("_idx_n")?.extract()?,
        children: dt.getattr("_idx_children")?.extract()?,
        edge_lengths: dt.getattr("_idx_edge_lengths")?.extract()?,
        leaves: dt.getattr("_idx_leaves")?.extract()?,
        leaf_species: dt.getattr("_idx_leaf_species")?.extract()?,
        root: dt.getattr("_idx_root")?.extract()?,
    };
    let obj = Py::new(py, idx)?;
    let _ = dt.setattr("_rs_idx", obj.bind(py));
    Ok(obj)
}

// ═══════════════════════════════════════════════════════════════════════
// DP cores (generic over the mask width)
// ═══════════════════════════════════════════════════════════════════════

type Config<M> = FxHashMap<M, f64>;
type Key<M> = SmallVec<[(i32, M); 4]>;

#[inline]
fn acc<M: Mask>(out: &mut Config<M>, cfg: M, lp: f64) {
    match out.get_mut(&cfg) {
        Some(v) => *v = logaddexp(*v, lp),
        None => {
            out.insert(cfg, lp);
        }
    }
}

/// Coalescent transition over one species branch for every entry of
/// ``config_in``, streamed into ``out``.
fn apply_branch<M: Mask>(
    config_in: impl IntoIterator<Item = (M, f64)>,
    length: Option<f64>,
    topo: &GtTopo,
    caches: &mut Caches<M>,
    kernel: &mut MSCBranchKernel,
    out: &mut Config<M>,
) {
    for (cfg_in, lp_in) in config_in {
        let n = cfg_in.popcount();
        let rows = topo.coarsenings(caches, cfg_in);
        // `rows` borrows `caches`; `kernel` is separate so this is fine.
        for row in rows.iter() {
            let mut log_branch = kernel.log_gij(length, n, row.m_out);
            if log_branch <= LOG_FLOOR {
                continue;
            }
            if row.k > 0 {
                log_branch += row.log_le - kernel.log_denom(n, row.k);
            }
            acc(out, row.cfg_out, lp_in + log_branch);
        }
    }
}

fn combine<M: Mask>(left: &Config<M>, right: &Config<M>) -> Config<M> {
    let mut out: Config<M> = FxHashMap::default();
    for (&cl, &ll) in left.iter() {
        for (&cr, &lr) in right.iter() {
            if cl.overlaps(&cr) {
                continue;
            }
            acc(&mut out, cl.or(cr), ll + lr);
        }
    }
    out
}

#[inline]
fn key_insert<M: Mask>(base: &Key<M>, eid: i32, mask: M) -> Key<M> {
    let mut out = base.clone();
    let pos = out.iter().position(|(e, _)| *e >= eid).unwrap_or(out.len());
    out.insert(pos, (eid, mask));
    out
}

#[inline]
fn gamma_logs(g1: Option<f64>, g2: Option<f64>) -> (f64, f64) {
    let (gamma1, gamma2) = match (g1, g2) {
        (None, None) => (0.5, 0.5),
        (None, Some(b)) => ((1.0 - b).max(0.0), b),
        (Some(a), None) => (a, (1.0 - a).max(0.0)),
        (Some(a), Some(b)) => (a, b),
    };
    let lg = |g: f64| if g > 0.0 { g.ln() } else { LOG_FLOOR };
    (lg(gamma1), lg(gamma2))
}

/// Pull ``v``'s down-edges off a frontier key, OR-ing their masks.
#[inline]
fn pop_down_edges<M: Mask>(key: &Key<M>, down: &[usize]) -> Option<(Key<M>, M)> {
    let mut rest = key.clone();
    let mut mask = M::zero();
    for &de in down {
        let pos = rest.iter().position(|(e, _)| *e == de as i32)?;
        mask = mask.or(rest[pos].1);
        rest.remove(pos);
    }
    Some((rest, mask))
}

fn finish_frontier<M: Mask>(frontier: &FxHashMap<Key<M>, f64>, root_bit: usize) -> f64 {
    let target = M::bit(root_bit);
    let mut terms = Vec::new();
    for (key, lp) in frontier.iter() {
        if key.len() != 1 {
            continue;
        }
        let (eid, mask) = key[0];
        if eid != -1 {
            continue;
        }
        if mask == target || (mask.popcount() == 1 && mask.test(root_bit)) {
            terms.push(*lp);
        }
    }
    if terms.is_empty() {
        LOG_FLOOR
    } else {
        logsumexp(&terms)
    }
}

/// Topology-only MSNC frontier DP on a species network.
pub fn msnc_log_prob_network<M: Mask>(
    net: &NetworkIndex,
    params: &EdgeParams,
    topo: &GtTopo,
    caches: &mut Caches<M>,
    kernel: &mut MSCBranchKernel,
) -> f64 {
    if topo.root_bit < 0 || topo.leaves.is_empty() {
        return 0.0;
    }
    if net.n_nodes == 0 || net.root < 0 {
        return LOG_FLOOR;
    }
    let species_to_bits = topo.species_to_bits::<M>();
    let mut frontier: FxHashMap<Key<M>, f64> = FxHashMap::default();
    frontier.insert(SmallVec::new(), 0.0);
    let mut scratch: Config<M> = FxHashMap::default();
    let mut scratch2: Config<M> = FxHashMap::default();

    for &v in &net.topo_order {
        let down = &net.down_edges[v];
        let up = &net.up_edges[v];
        let mut new_frontier: FxHashMap<Key<M>, f64> = FxHashMap::default();

        for (key, &lp) in frontier.iter() {
            let (base, mask_at_v) = if down.is_empty() {
                let m = net.leaf_label[v]
                    .as_ref()
                    .and_then(|sp| species_to_bits.get(sp).copied())
                    .unwrap_or_else(M::zero);
                (key.clone(), m)
            } else {
                match pop_down_edges(key, down) {
                    Some(x) => x,
                    None => continue,
                }
            };

            if up.is_empty() {
                scratch.clear();
                apply_branch([(mask_at_v, 0.0)], None, topo, caches, kernel, &mut scratch);
                for (&top, &tlp) in scratch.iter() {
                    let nk = key_insert(&base, -1, top);
                    match new_frontier.get_mut(&nk) {
                        Some(x) => *x = logaddexp(*x, lp + tlp),
                        None => {
                            new_frontier.insert(nk, lp + tlp);
                        }
                    }
                }
                continue;
            }

            if up.len() >= 2 && net.is_retic[v] {
                let (e1, e2) = (up[0], up[1]);
                let (log_g1, log_g2) = gamma_logs(params.gamma[e1], params.gamma[e2]);
                let (len1, len2) = (params.length[e1], params.length[e2]);
                let n_total = mask_at_v.popcount();
                let mut s = mask_at_v;
                loop {
                    let k_s = s.popcount();
                    let factor = k_s as f64 * log_g1 + (n_total - k_s) as f64 * log_g2;
                    let a_s = mask_at_v.xor(s);
                    scratch.clear();
                    apply_branch([(s, 0.0)], len1, topo, caches, kernel, &mut scratch);
                    scratch2.clear();
                    apply_branch([(a_s, 0.0)], len2, topo, caches, kernel, &mut scratch2);
                    for (&t1, &l1) in scratch.iter() {
                        let k1 = key_insert(&base, e1 as i32, t1);
                        for (&t2, &l2) in scratch2.iter() {
                            let nk = key_insert(&k1, e2 as i32, t2);
                            let val = lp + factor + l1 + l2;
                            match new_frontier.get_mut(&nk) {
                                Some(x) => *x = logaddexp(*x, val),
                                None => {
                                    new_frontier.insert(nk, val);
                                }
                            }
                        }
                    }
                    if s.is_zero() {
                        break;
                    }
                    s = s.pred_and(mask_at_v);
                }
                continue;
            }

            let e_up = up[0];
            scratch.clear();
            apply_branch([(mask_at_v, 0.0)], params.length[e_up], topo, caches, kernel, &mut scratch);
            for (&top, &tlp) in scratch.iter() {
                let nk = key_insert(&base, e_up as i32, top);
                match new_frontier.get_mut(&nk) {
                    Some(x) => *x = logaddexp(*x, lp + tlp),
                    None => {
                        new_frontier.insert(nk, lp + tlp);
                    }
                }
            }
        }
        frontier = new_frontier;
    }
    finish_frontier(&frontier, topo.root_bit as usize)
}

/// A gene-tree coalescence: ``(time, parent_bit, child0_bit, child1_bit)``.
pub type Event = (f64, usize, usize, usize);

/// Event-based coalescent density along one species branch for one
/// configuration (the result is deterministic).
#[inline]
fn apply_branch_timed_single<M: Mask>(
    cfg: M,
    lp_in: f64,
    tau_low: f64,
    tau_high: Option<f64>,
    theta: f64,
    events: &[Event],
) -> (M, f64) {
    let inv_theta = 1.0 / theta;
    let log_two_over_theta = (2.0 * inv_theta).ln();
    let mut cur = tau_low;
    let mut u = cfg.popcount() as f64;
    let mut cfg_cur = cfg;
    let mut lp = lp_in;
    for &(t, pid, c0, c1) in events {
        if t < tau_low {
            continue;
        }
        if let Some(h) = tau_high {
            if t >= h {
                break;
            }
        }
        if cfg_cur.test(c0) && cfg_cur.test(c1) {
            lp += -(t - cur) * u * (u - 1.0) * inv_theta + log_two_over_theta;
            cur = t;
            cfg_cur = cfg_cur.andnot(M::bit(c0)).andnot(M::bit(c1)).or(M::bit(pid));
            u -= 1.0;
        }
    }
    if let Some(h) = tau_high {
        if u > 1.0 {
            lp += -(h - cur) * u * (u - 1.0) * inv_theta;
        }
    }
    (cfg_cur, lp)
}

/// Timed (event-based) MSNC frontier DP.
pub fn msnc_log_density_timed<M: Mask>(
    net: &NetworkIndex,
    params: &EdgeParams,
    topo: &GtTopo,
    events: &[Event],
    sp_heights: &[f64],
    theta: f64,
    edge_thetas: Option<&[f64]>,
    root_theta: Option<f64>,
) -> f64 {
    if topo.root_bit < 0 || topo.leaves.is_empty() {
        return 0.0;
    }
    if net.n_nodes == 0 || net.root < 0 {
        return LOG_FLOOR;
    }
    let species_to_bits = topo.species_to_bits::<M>();
    let theta_of = |e: usize| edge_thetas.map(|t| t[e]).unwrap_or(theta);
    let root_th = root_theta.unwrap_or(theta);
    let mut frontier: FxHashMap<Key<M>, f64> = FxHashMap::default();
    frontier.insert(SmallVec::new(), 0.0);

    for &v in &net.topo_order {
        let down = &net.down_edges[v];
        let up = &net.up_edges[v];
        let mut new_frontier: FxHashMap<Key<M>, f64> = FxHashMap::default();
        for (key, &lp) in frontier.iter() {
            let (base, mask_at_v) = if down.is_empty() {
                let m = net.leaf_label[v]
                    .as_ref()
                    .and_then(|sp| species_to_bits.get(sp).copied())
                    .unwrap_or_else(M::zero);
                (key.clone(), m)
            } else {
                match pop_down_edges(key, down) {
                    Some(x) => x,
                    None => continue,
                }
            };

            if up.is_empty() {
                let (top, tlp) =
                    apply_branch_timed_single(mask_at_v, 0.0, sp_heights[v], None, root_th, events);
                let nk = key_insert(&base, -1, top);
                match new_frontier.get_mut(&nk) {
                    Some(x) => *x = logaddexp(*x, lp + tlp),
                    None => {
                        new_frontier.insert(nk, lp + tlp);
                    }
                }
                continue;
            }

            if up.len() >= 2 && net.is_retic[v] {
                let (e1, e2) = (up[0], up[1]);
                let (log_g1, log_g2) = gamma_logs(params.gamma[e1], params.gamma[e2]);
                let tau_low = sp_heights[v];
                let (p1, p2) = (net.edge_src[e1], net.edge_src[e2]);
                let n_total = mask_at_v.popcount();
                let mut s = mask_at_v;
                loop {
                    let k_s = s.popcount();
                    let factor = k_s as f64 * log_g1 + (n_total - k_s) as f64 * log_g2;
                    let a_s = mask_at_v.xor(s);
                    let (t1, l1) = apply_branch_timed_single(
                        s, 0.0, tau_low, Some(sp_heights[p1]), theta_of(e1), events,
                    );
                    let (t2, l2) = apply_branch_timed_single(
                        a_s, 0.0, tau_low, Some(sp_heights[p2]), theta_of(e2), events,
                    );
                    let nk = key_insert(&key_insert(&base, e1 as i32, t1), e2 as i32, t2);
                    let val = lp + factor + l1 + l2;
                    match new_frontier.get_mut(&nk) {
                        Some(x) => *x = logaddexp(*x, val),
                        None => {
                            new_frontier.insert(nk, val);
                        }
                    }
                    if s.is_zero() {
                        break;
                    }
                    s = s.pred_and(mask_at_v);
                }
                continue;
            }

            let e_up = up[0];
            let parent = net.edge_src[e_up];
            let (top, tlp) = apply_branch_timed_single(
                mask_at_v, 0.0, sp_heights[v], Some(sp_heights[parent]), theta_of(e_up), events,
            );
            let nk = key_insert(&base, e_up as i32, top);
            match new_frontier.get_mut(&nk) {
                Some(x) => *x = logaddexp(*x, lp + tlp),
                None => {
                    new_frontier.insert(nk, lp + tlp);
                }
            }
        }
        frontier = new_frontier;
    }
    finish_frontier(&frontier, topo.root_bit as usize)
}

/// Species-tree MSC DP on a displayed tree.
pub fn msc_log_prob_tree<M: Mask>(
    dt: &DisplayedTreeIndex,
    topo: &GtTopo,
    caches: &mut Caches<M>,
    kernel: &mut MSCBranchKernel,
) -> f64 {
    if topo.root_bit < 0 || topo.leaves.is_empty() {
        return 0.0;
    }
    let species_to_bits = topo.species_to_bits::<M>();
    let n = dt.n;
    let mut configs: Vec<Option<Config<M>>> = (0..n).map(|_| None).collect();
    for &leaf in &dt.leaves {
        let mask = dt
            .leaf_species
            .get(&leaf)
            .and_then(|sp| species_to_bits.get(sp).copied())
            .unwrap_or_else(M::zero);
        let mut c: Config<M> = FxHashMap::default();
        c.insert(mask, 0.0);
        configs[leaf] = Some(c);
    }
    for i in 0..n {
        if configs[i].is_some() {
            continue;
        }
        let kids = &dt.children[i];
        if kids.is_empty() {
            let mut c: Config<M> = FxHashMap::default();
            c.insert(M::zero(), 0.0);
            configs[i] = Some(c);
            continue;
        }
        let mut merged: Option<Config<M>> = None;
        for (j, &kid) in kids.iter().enumerate() {
            let child_cfg = configs[kid].take().unwrap_or_default();
            let mut top: Config<M> = FxHashMap::default();
            apply_branch(
                child_cfg.iter().map(|(&m, &l)| (m, l)),
                dt.edge_lengths[i][j],
                topo,
                caches,
                kernel,
                &mut top,
            );
            configs[kid] = Some(child_cfg);
            merged = Some(match merged {
                None => top,
                Some(prev) => combine(&prev, &top),
            });
        }
        configs[i] = merged;
    }
    let root_cfg: Config<M> = if dt.root >= 0 {
        configs[dt.root as usize].take().unwrap_or_default()
    } else {
        FxHashMap::default()
    };
    let mut top: Config<M> = FxHashMap::default();
    apply_branch(root_cfg.iter().map(|(&m, &l)| (m, l)), None, topo, caches, kernel, &mut top);
    let root_bit = topo.root_bit as usize;
    let target = M::bit(root_bit);
    if let Some(v) = top.get(&target) {
        return *v;
    }
    let terms: Vec<f64> = top
        .iter()
        .filter(|(m, _)| **m == target || (m.popcount() == 1 && m.test(root_bit)))
        .map(|(_, l)| *l)
        .collect();
    if terms.is_empty() {
        LOG_FLOOR
    } else {
        logsumexp(&terms)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Python entry points
// ═══════════════════════════════════════════════════════════════════════

fn config_from_py<M: Mask>(d: &Bound<'_, PyDict>) -> PyResult<Config<M>> {
    let mut out: Config<M> = FxHashMap::default();
    for (k, v) in d.iter() {
        out.insert(M::from_py(&k)?, v.extract::<f64>()?);
    }
    Ok(out)
}

fn config_to_py<'py, M: Mask>(py: Python<'py>, c: &Config<M>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (m, lp) in c.iter() {
        d.set_item(m.to_py(py)?, *lp)?;
    }
    Ok(d)
}

/// Mask width needed for the keys of one or more config dicts.
fn max_bits(dicts: &[&Bound<'_, PyDict>]) -> PyResult<usize> {
    let mut best = 0usize;
    for d in dicts {
        for (k, _) in d.iter() {
            let n: usize = k.call_method0("bit_length")?.extract()?;
            best = best.max(n);
        }
    }
    Ok(best)
}

#[pyfunction]
#[pyo3(name = "_msnc_log_prob_network_int")]
pub fn msnc_log_prob_network_py(
    net_idx: &Bound<'_, NetworkIndex>,
    gti: &Bound<'_, GeneTreeIndex>,
    engine: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let py = net_idx.py();
    let net = net_idx.borrow();
    let params = net.params(py);
    let mut g = gti.borrow_mut();
    let engine = engine.cast::<MSCBranchKernel>()?;
    let mut k = engine.borrow_mut();
    let g = &mut *g;
    let topo = &g.topo;
    Ok(with_mask!(&mut g.caches, |c, M| {
        msnc_log_prob_network::<M>(&net, &params, topo, c, &mut k)
    }))
}

#[pyfunction]
#[pyo3(name = "_msc_log_prob_tree_int")]
pub fn msc_log_prob_tree_py(
    dt: &Bound<'_, PyAny>,
    gti: &Bound<'_, GeneTreeIndex>,
    engine: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let py = dt.py();
    let idx = displayed_index_from(dt)?;
    let idx = idx.borrow(py);
    let mut g = gti.borrow_mut();
    let engine = engine.cast::<MSCBranchKernel>()?;
    let mut k = engine.borrow_mut();
    let g = &mut *g;
    let topo = &g.topo;
    Ok(with_mask!(&mut g.caches, |c, M| {
        msc_log_prob_tree::<M>(&idx, topo, c, &mut k)
    }))
}

#[pyfunction]
#[pyo3(name = "_apply_branch_coalescent_int")]
pub fn apply_branch_coalescent_py<'py>(
    py: Python<'py>,
    config_in: &Bound<'py, PyDict>,
    length: Option<f64>,
    gti: &Bound<'py, GeneTreeIndex>,
    engine: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut g = gti.borrow_mut();
    let engine = engine.cast::<MSCBranchKernel>()?;
    let mut k = engine.borrow_mut();
    let g = &mut *g;
    let topo = &g.topo;
    with_mask!(&mut g.caches, |c, M| {
        let cfg = config_from_py::<M>(config_in)?;
        let mut out: Config<M> = FxHashMap::default();
        apply_branch(cfg.into_iter(), length, topo, c, &mut k, &mut out);
        config_to_py(py, &out)
    })
}

#[pyfunction]
#[pyo3(name = "_combine_configs_int")]
pub fn combine_configs_py<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyDict>,
    right: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let bits = max_bits(&[left, right])?;
    macro_rules! run {
        ($m:ty) => {{
            let l = config_from_py::<$m>(left)?;
            let r = config_from_py::<$m>(right)?;
            config_to_py(py, &combine(&l, &r))
        }};
    }
    if bits <= 64 {
        run!(u64)
    } else if bits <= 128 {
        run!(u128)
    } else if bits <= 256 {
        run!(Words<4>)
    } else if bits <= 512 {
        run!(Words<8>)
    } else if bits <= 1024 {
        run!(Words<16>)
    } else {
        run!(Words<32>)
    }
}

#[pyfunction]
#[pyo3(name = "_msnc_log_density_timed")]
#[pyo3(signature = (net_idx, gti, events, sp_heights, theta, edge_thetas=None, root_theta=None))]
#[allow(clippy::too_many_arguments)]
pub fn msnc_log_density_timed_py(
    net_idx: &Bound<'_, NetworkIndex>,
    gti: &Bound<'_, GeneTreeIndex>,
    events: Vec<Event>,
    sp_heights: Vec<f64>,
    theta: f64,
    edge_thetas: Option<Vec<f64>>,
    root_theta: Option<f64>,
) -> PyResult<f64> {
    let py = net_idx.py();
    let net = net_idx.borrow();
    let params = net.params(py);
    let mut g = gti.borrow_mut();
    let g = &mut *g;
    let topo = &g.topo;
    Ok(with_mask!(&mut g.caches, |_c, M| {
        msnc_log_density_timed::<M>(
            &net,
            &params,
            topo,
            &events,
            &sp_heights,
            theta,
            edge_thetas.as_deref(),
            root_theta,
        )
    }))
}

#[pyfunction]
#[pyo3(name = "_apply_branch_timed")]
pub fn apply_branch_timed_py<'py>(
    py: Python<'py>,
    config_in: &Bound<'py, PyDict>,
    tau_low: f64,
    tau_high: Option<f64>,
    theta: f64,
    events: Vec<Event>,
) -> PyResult<Bound<'py, PyDict>> {
    let bits = max_bits(&[config_in])?;
    macro_rules! run {
        ($m:ty) => {{
            let cfg = config_from_py::<$m>(config_in)?;
            let mut out: Config<$m> = FxHashMap::default();
            for (c, lp) in cfg {
                let (nc, nlp) = apply_branch_timed_single(c, lp, tau_low, tau_high, theta, &events);
                acc(&mut out, nc, nlp);
            }
            config_to_py(py, &out)
        }};
    }
    if bits <= 64 {
        run!(u64)
    } else if bits <= 128 {
        run!(u128)
    } else if bits <= 256 {
        run!(Words<4>)
    } else if bits <= 512 {
        run!(Words<8>)
    } else if bits <= 1024 {
        run!(Words<16>)
    } else {
        run!(Words<32>)
    }
}

#[pyfunction]
#[pyo3(name = "_popcount")]
pub fn popcount_py(mask: &Bound<'_, PyAny>) -> PyResult<usize> {
    let m = mask.cast::<PyInt>()?;
    let m = m.call_method0("__abs__")?;
    Ok(m.call_method0("bit_count")
        .and_then(|v| v.extract::<usize>())
        .unwrap_or_else(|_| py_int_bits(&m).map(|b| b.len()).unwrap_or(0)))
}

#[pyfunction]
#[pyo3(name = "_bits")]
pub fn bits_py(mask: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    py_int_bits(mask)
}

#[pyfunction]
#[pyo3(name = "_logsumexp")]
pub fn logsumexp_py(values: Vec<f64>) -> f64 {
    logsumexp(&values)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MSCBranchKernel>()?;
    m.add_class::<GeneTreeIndex>()?;
    m.add_class::<NetworkIndex>()?;
    m.add_class::<DisplayedTreeIndex>()?;
    m.add_function(wrap_pyfunction!(msnc_log_prob_network_py, m)?)?;
    m.add_function(wrap_pyfunction!(msc_log_prob_tree_py, m)?)?;
    m.add_function(wrap_pyfunction!(apply_branch_coalescent_py, m)?)?;
    m.add_function(wrap_pyfunction!(combine_configs_py, m)?)?;
    m.add_function(wrap_pyfunction!(msnc_log_density_timed_py, m)?)?;
    m.add_function(wrap_pyfunction!(apply_branch_timed_py, m)?)?;
    m.add_function(wrap_pyfunction!(popcount_py, m)?)?;
    m.add_function(wrap_pyfunction!(bits_py, m)?)?;
    m.add_function(wrap_pyfunction!(logsumexp_py, m)?)?;
    let _ = PyList::empty(m.py());
    Ok(())
}
