//! Allopolyploid maximum parsimony (Hejase et al.; PhyloNet's MPAllopp):
//! allele maps and the extra-lineage score of gene trees reconciled against
//! a multi-labelled (MUL) species tree.
//!
//! ``phynetpy._infer_mp_allop`` keeps the public classes and the search
//! glue; this module holds everything that walks a gene tree or the MUL
//! tree: the MUL-tree construction (reticulation expansion and tip
//! renaming, through the same graph primitives the Python reference used,
//! in the same order), the allele-map enumeration, and the per-gene-tree
//! reconciliation (cluster -> MRCA lookups and lineage counting up the MUL
//! tree) on integer indices.
//!
//! The reference walked the MUL tree keyed by ``(parent label, label)``
//! edge names; on anything but a single-rooted tree that walk failed with
//! an ``IndexError`` (leaf root) or a ``KeyError`` (an edge it never keyed).
//! The index below reproduces those failures before scoring.

use pyo3::exceptions::{PyAttributeError, PyIndexError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple, PyType};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::{Edge, Network, Node};
use crate::util::{deepcopy, network_error};

// ═══════════════════════════════════════════════════════════════════════
// AlleleMap
// ═══════════════════════════════════════════════════════════════════════

/// Data structure that holds a mapping from gene tree leaf names to MUL tree
/// leaf names. Internally handles the mechanism for making sure gene copies
/// are not mapped to the same subgenome.
#[pyclass(name = "AlleleMap", module = "phynetpy._infer_mp_allop", subclass)]
pub struct AlleleMap {
    /// Gene-tree leaf label -> MUL-tree leaf label.
    #[pyo3(get, set)]
    pub map: Py<PyDict>,
    /// MUL-tree leaf labels already taken by a gene copy.
    #[pyo3(get, set)]
    pub disallowed: Py<PySet>,
}

impl AlleleMap {
    fn empty(py: Python<'_>) -> PyResult<AlleleMap> {
        Ok(AlleleMap {
            map: PyDict::new(py).unbind(),
            disallowed: PySet::empty(py)?.unbind(),
        })
    }

    /// The mapping dict of an ``AlleleMap``, or of any object exposing a
    /// ``map`` attribute (the reference only ever read ``f.map``).
    fn dict_of<'py>(f: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
        if let Ok(am) = f.cast::<AlleleMap>() {
            return Ok(am.borrow().map.bind(f.py()).clone());
        }
        Ok(f.getattr("map")?.cast_into::<PyDict>()?)
    }
}

#[pymethods]
impl AlleleMap {
    /// Create an empty ``AlleleMap`` with no leaves mapped yet.
    #[new]
    fn py_new(py: Python<'_>) -> PyResult<Self> {
        AlleleMap::empty(py)
    }

    /// Map a gene tree leaf to a MUL tree leaf.
    ///
    /// Args:
    ///     g_leaf (Node): a gene tree leaf
    ///     mul_leaf (str): a MUL tree leaf label
    ///
    /// Returns:
    ///     int: 0 if the mapping was unsuccessful, 1 if it was.
    fn put(&self, py: Python<'_>, g_leaf: &Bound<'_, PyAny>, mul_leaf: &Bound<'_, PyAny>) -> PyResult<i32> {
        let disallowed = self.disallowed.bind(py);
        if disallowed.contains(mul_leaf)? {
            return Ok(0);
        }
        let label = g_leaf.getattr("label")?;
        self.map.bind(py).set_item(label, mul_leaf)?;
        disallowed.add(mul_leaf)?;
        Ok(1)
    }

    fn __copy__(&self, py: Python<'_>) -> AlleleMap {
        AlleleMap {
            map: self.map.clone_ref(py),
            disallowed: self.disallowed.clone_ref(py),
        }
    }

    fn __deepcopy__(&self, py: Python<'_>, memo: Bound<'_, PyDict>) -> PyResult<AlleleMap> {
        Ok(AlleleMap {
            map: deepcopy(py, self.map.bind(py).as_any(), &memo)?
                .cast_into::<PyDict>()?
                .unbind(),
            disallowed: deepcopy(py, self.disallowed.bind(py).as_any(), &memo)?
                .cast_into::<PySet>()?
                .unbind(),
        })
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("map", self.map.bind(py))?;
        d.set_item("disallowed", self.disallowed.bind(py))?;
        Ok(d)
    }

    fn __setstate__(&mut self, state: Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(m) = state.get_item("map")? {
            self.map = m.cast_into::<PyDict>()?.unbind();
        }
        if let Some(s) = state.get_item("disallowed")? {
            self.disallowed = s.cast_into::<PySet>()?.unbind();
        }
        Ok(())
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let py = slf.py();
        let state = slf.borrow().__getstate__(py)?;
        Ok((slf.get_type(), PyTuple::empty(py), state))
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Allele-map enumeration
// ═══════════════════════════════════════════════════════════════════════

/// Search steps a bounded enumeration may spend before giving up (the
/// caller then falls back to the integer program, as the reference did).
const SEARCH_BUDGET: u64 = 2_000_000;

struct MapSearch<'a> {
    copies: &'a [Vec<usize>],
    used: Vec<bool>,
    chosen: Vec<usize>,
    out: Vec<Vec<usize>>,
    limit: Option<usize>,
    steps: u64,
}

impl MapSearch<'_> {
    /// Assign leaves ``pos-1`` down to ``0``.  Choosing the *last* leaf in
    /// the outermost loop yields complete maps in exactly the order the
    /// reference's level-wise enumeration produced them (its last level
    /// grouped the survivors by the last leaf's choice, then by the
    /// previous level's order, and so on).  Returns ``false`` once the
    /// limit or the step budget is exceeded.
    fn rec(&mut self, pos: usize) -> bool {
        if pos == 0 {
            self.out.push(self.chosen.clone());
            return self.limit.map_or(true, |l| self.out.len() <= l);
        }
        let i = pos - 1;
        for k in 0..self.copies[i].len() {
            let lab = self.copies[i][k];
            if self.used[lab] {
                continue;
            }
            if self.limit.is_some() {
                self.steps += 1;
                if self.steps > SEARCH_BUDGET {
                    return false;
                }
            }
            self.used[lab] = true;
            self.chosen[i] = lab;
            let go_on = self.rec(pos - 1);
            self.used[lab] = false;
            if !go_on {
                return false;
            }
        }
        true
    }
}

/// Every allele map compatible with ``choices`` -- one ``(leaf label,
/// allowed MUL labels)`` pair per gene-tree leaf, in ``get_leaves()`` order
/// -- as ``AlleleMap`` objects in the reference's enumeration order.  A map
/// assigns each leaf one of its allowed labels, no label twice.
///
/// With ``limit``, returns ``None`` as soon as more than ``limit`` maps exist
/// (or the search exceeds its step budget) so the caller can defer to the
/// integer program instead.
#[pyfunction]
#[pyo3(name = "_enumerate_allele_maps", signature = (choices, limit=None))]
fn enumerate_allele_maps_py(
    py: Python<'_>,
    choices: Vec<(String, Vec<String>)>,
    limit: Option<usize>,
) -> PyResult<Option<Vec<Py<AlleleMap>>>> {
    let mut names: Vec<&str> = Vec::new();
    let mut ids: FxHashMap<&str, usize> = FxHashMap::default();
    let copies: Vec<Vec<usize>> = choices
        .iter()
        .map(|(_, cs)| {
            cs.iter()
                .map(|c| {
                    *ids.entry(c.as_str()).or_insert_with(|| {
                        names.push(c.as_str());
                        names.len() - 1
                    })
                })
                .collect()
        })
        .collect();
    let n = choices.len();
    let mut search = MapSearch {
        copies: &copies,
        used: vec![false; names.len()],
        chosen: vec![usize::MAX; n],
        out: Vec::new(),
        limit,
        steps: 0,
    };
    if !search.rec(n) {
        return Ok(None);
    }
    let mut result = Vec::with_capacity(search.out.len());
    for chosen in &search.out {
        let d = PyDict::new(py);
        let s = PySet::empty(py)?;
        for (i, (leaf, _)) in choices.iter().enumerate() {
            d.set_item(leaf, names[chosen[i]])?;
            s.add(names[chosen[i]])?;
        }
        result.push(Py::new(
            py,
            AlleleMap {
                map: d.unbind(),
                disallowed: s.unbind(),
            },
        )?);
    }
    Ok(Some(result))
}

/// Most maps enumerated before ``_select_allele_maps`` picks its subset.
const SELECT_CAP: usize = 50_000;

/// Up to ``limit`` allele maps compatible with ``choices``.  When more
/// exist, the maps returned are spread evenly across the enumeration order
/// (every ``total / limit``-th map), so they differ in the assignments of
/// early leaves as well as late ones.
///
/// This replaces the reference's PuLP / CBC integer program, which solved
/// the assignment problem with a constant objective and excluded each
/// solution from the next solve: an arbitrary, solver-dependent subset of
/// the valid maps.  The subset here is deterministic and needs no external
/// solver.  Enumeration stops after ``SELECT_CAP`` maps or the step budget,
/// and the subset is then drawn from the maps enumerated so far.
#[pyfunction]
#[pyo3(name = "_select_allele_maps")]
fn select_allele_maps_py(
    py: Python<'_>,
    choices: Vec<(String, Vec<String>)>,
    limit: usize,
) -> PyResult<Vec<Py<AlleleMap>>> {
    let mut names: Vec<&str> = Vec::new();
    let mut ids: FxHashMap<&str, usize> = FxHashMap::default();
    let copies: Vec<Vec<usize>> = choices
        .iter()
        .map(|(_, cs)| {
            cs.iter()
                .map(|c| {
                    *ids.entry(c.as_str()).or_insert_with(|| {
                        names.push(c.as_str());
                        names.len() - 1
                    })
                })
                .collect()
        })
        .collect();
    let n = choices.len();
    let mut search = MapSearch {
        copies: &copies,
        used: vec![false; names.len()],
        chosen: vec![usize::MAX; n],
        out: Vec::new(),
        limit: Some(SELECT_CAP),
        steps: 0,
    };
    search.rec(n);
    search.out.truncate(SELECT_CAP);
    let total = search.out.len();
    let picks: Vec<usize> = if total <= limit {
        (0..total).collect()
    } else {
        (0..limit).map(|i| i * total / limit).collect()
    };
    let mut result = Vec::with_capacity(picks.len());
    for idx in picks {
        let chosen = &search.out[idx];
        let d = PyDict::new(py);
        let s = PySet::empty(py)?;
        for (i, (leaf, _)) in choices.iter().enumerate() {
            d.set_item(leaf, names[chosen[i]])?;
            s.add(names[chosen[i]])?;
        }
        result.push(Py::new(
            py,
            AlleleMap {
                map: d.unbind(),
                disallowed: s.unbind(),
            },
        )?);
    }
    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════
// MUL-tree construction (``MUL.to_mul``)
// ═══════════════════════════════════════════════════════════════════════

/// ``label`` without one trailing ``_copy<digits>`` suffix.
fn strip_copy_suffix(label: &str) -> Option<&str> {
    let pos = label.rfind("_copy")?;
    let digits = &label[pos + 5..];
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(&label[..pos])
}

/// The species key of a MUL tip: the label itself, or the label with one or
/// more ``_copyN`` suffixes (from ``Network.subnet``) stripped.
fn mul_tip_species(label: &str, species: &FxHashMap<String, usize>) -> Option<usize> {
    if let Some(&i) = species.get(label) {
        return Some(i);
    }
    let mut cur = label;
    loop {
        cur = strip_copy_suffix(cur)?;
        if let Some(&i) = species.get(cur) {
            return Some(i);
        }
    }
}

/// Build the multi-labelled species tree of ``net`` under ``gene_map``
/// (species -> allele labels): expand every reticulation, lowest first and
/// tie-broken by label, by grafting a ``subnet`` copy of its subtree under
/// each additional parent; drop the spacer nodes; then hand the allele
/// labels to the tips in sorted tip order.  The reference's validation and
/// error messages are kept verbatim.
#[pyfunction]
#[pyo3(name = "_allop_to_mul")]
fn allop_to_mul_py<'py>(
    py: Python<'py>,
    net: &Bound<'py, Network>,
    gene_map: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, Network>> {
    let mut species: Vec<String> = Vec::new();
    let mut remaining: Vec<Vec<String>> = Vec::new();
    for item in gene_map.call_method0("items")?.try_iter()? {
        let (k, v): (String, Vec<String>) = item?.extract()?;
        species.push(k);
        remaining.push(v);
    }
    let provided: Vec<usize> = remaining.iter().map(Vec::len).collect();
    let species_index: FxHashMap<String, usize> =
        species.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();

    let n_leaves = net.borrow().leaf_uids().len();
    if n_leaves != species.len() {
        return Err(network_error(
            py,
            format!(
                "Input network has incorrect amount of {pad}leaves. Given : {n_leaves} \
                 {pad}Expected : {}",
                species.len(),
                pad = " ".repeat(16),
            ),
        ));
    }

    let (mul, _) = Network::copy_impl(net)?;

    // Expand reticulations, always a lowest one (no reticulation below it)
    // so the copied subnetwork is already a tree.
    loop {
        let target = {
            let m = mul.borrow();
            let retics: Vec<u64> = m.nodes.keys().copied().filter(|&u| m.in_deg(u) >= 2).collect();
            if retics.is_empty() {
                break;
            }
            let retic_set: FxHashSet<u64> = retics.iter().copied().collect();
            let mut lowest: Vec<(String, u64)> = retics
                .iter()
                .copied()
                .filter(|&r| !m.subtree_uids(r).into_iter().any(|d| d != r && retic_set.contains(&d)))
                .map(|r| (m.label(py, r), r))
                .collect();
            if lowest.is_empty() {
                return Err(PyIndexError::new_err("list index out of range"));
            }
            lowest.sort();
            lowest[0].1
        };
        let parents: Vec<u64> = {
            let m = mul.borrow();
            let mut ps: Vec<(String, u64)> =
                m.parents(target).into_iter().map(|p| (m.label(py, p), p)).collect();
            ps.sort();
            ps.into_iter().map(|(_, p)| p).collect()
        };
        for &parent in parents.iter().skip(1) {
            let subtree = Network::subnet_impl(&mul, target)?;
            let euid = {
                let m = mul.borrow();
                m.edge_between(py, parent, target, None, None)?.borrow(py).uid
            };
            mul.borrow_mut().remove_edge_uid(py, euid);
            let (sub_nodes, sub_edges, sub_root) = {
                let s = subtree.borrow();
                let root = s.root_uid(py)?;
                (
                    s.nodes.values().map(|e| e.obj.clone_ref(py)).collect::<Vec<_>>(),
                    s.edges.values().map(|e| e.obj.clone_ref(py)).collect::<Vec<_>>(),
                    s.node_obj(root).clone_ref(py),
                )
            };
            {
                let mut m = mul.borrow_mut();
                for n in &sub_nodes {
                    m.add_node(mul.as_any(), n.bind(py))?;
                }
                for e in &sub_edges {
                    m.add_edge(mul.as_any(), e.bind(py))?;
                }
            }
            let parent_obj = mul.borrow().node_obj(parent).clone_ref(py);
            let link = Py::new(
                py,
                Edge::make(
                    py,
                    parent_obj,
                    sub_root,
                    Some(1.0),
                    Some(0.0),
                    Some(0.0),
                    Some("no tag assigned yet".to_string()),
                ),
            )?;
            mul.borrow_mut().add_edge(mul.as_any(), link.bind(py))?;
        }
    }

    Network::clean_impl(&mul, [true, true, true])?;

    // Hand out the allele labels, tips in sorted label order, each species'
    // labels popped from the end of its list.
    let mut leaves: Vec<(String, u64)> = {
        let m = mul.borrow();
        m.leaf_uids().into_iter().map(|u| (m.label(py, u), u)).collect()
    };
    leaves.sort();
    for (label, uid) in leaves {
        let Some(sp) = mul_tip_species(&label, &species_index) else {
            let mut keys: Vec<&str> = species.iter().map(String::as_str).collect();
            keys.sort();
            let keys_repr = PyList::new(py, keys)?.repr()?.to_string();
            return Err(network_error(
                py,
                format!(
                    "Species tip '{label}' (after reticulation expansion) does not match \
                     any key in the allele/subgenome map. Map keys: {keys_repr}."
                ),
            ));
        };
        let Some(new_name) = remaining[sp].pop() else {
            let s = &species[sp];
            return Err(network_error(
                py,
                format!(
                    "Allele/subgenome map has too few copies for species '{s}' (provided \
                     {}). Reticulation expansion created more MUL tips for this taxon \
                     than the map supplies. Provide one distinct allele label per \
                     parental lineage of each hybrid (e.g. '{s}': ['{s}_a', '{s}_b'] for \
                     a single reticulation involving '{s}').",
                    provided[sp]
                ),
            ));
        };
        mul.borrow_mut().rename_uid(py, uid, new_name);
    }

    let mut leftovers: Vec<(&str, &Vec<String>)> = species
        .iter()
        .zip(remaining.iter())
        .filter(|(_, r)| !r.is_empty())
        .map(|(s, r)| (s.as_str(), r))
        .collect();
    leftovers.sort_by(|a, b| a.0.cmp(b.0));
    if !leftovers.is_empty() {
        let detail = leftovers
            .iter()
            .map(|(sp, copies)| format!("'{sp}' has {} unused ({})", copies.len(), copies.join("/")))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(network_error(
            py,
            format!(
                "Allele/subgenome map has more copies than MUL tips after reticulation \
                 expansion: {detail}. Reduce the map to one allele label per MUL tip \
                 for each species."
            ),
        ));
    }
    Ok(mul)
}

// ═══════════════════════════════════════════════════════════════════════
// MUL-tree index
// ═══════════════════════════════════════════════════════════════════════

fn index_error() -> PyErr {
    PyIndexError::new_err("list index out of range")
}

/// One MUL tree on integer indices (``V()`` order), with the root walk the
/// reconciliation needs: first parent, children, leaf flags, DFS entry
/// times and depths for MRCA queries, and a post-order for the lineage
/// counts.
struct MulIndex {
    labels: Vec<String>,
    by_name: FxHashMap<String, usize>,
    parent: Vec<Option<usize>>,
    children: Vec<Vec<usize>>,
    is_leaf: Vec<bool>,
    root: usize,
    reach: Vec<bool>,
    depth: Vec<usize>,
    tin: Vec<usize>,
    post: Vec<usize>,
    /// ``(src, dest)`` of every edge, ``E()`` order.
    edges: Vec<(usize, usize)>,
}

impl MulIndex {
    fn build(py: Python<'_>, net: &Network) -> PyResult<MulIndex> {
        let uids: Vec<u64> = net.nodes.keys().copied().collect();
        let n = uids.len();
        let pos: FxHashMap<u64, usize> = uids.iter().enumerate().map(|(i, &u)| (u, i)).collect();
        let labels: Vec<String> = uids.iter().map(|&u| net.label(py, u)).collect();
        let by_name: FxHashMap<String, usize> = net
            .names
            .iter()
            .filter_map(|(name, u)| pos.get(u).map(|&i| (name.clone(), i)))
            .collect();
        let parent: Vec<Option<usize>> = uids
            .iter()
            .map(|&u| net.parents(u).first().map(|p| pos[p]))
            .collect();
        let children: Vec<Vec<usize>> = uids
            .iter()
            .map(|&u| net.children(u).into_iter().map(|c| pos[&c]).collect())
            .collect();
        let is_leaf: Vec<bool> = uids.iter().map(|&u| net.out_deg(u) == 0).collect();
        let root = pos[&net.root_uid(py)?];

        let mut reach = vec![false; n];
        let mut depth = vec![0usize; n];
        let mut tin = vec![0usize; n];
        let mut post = Vec::with_capacity(n);
        let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
        let mut t = 0usize;
        reach[root] = true;
        while let Some(&(v, k)) = stack.last() {
            if k < children[v].len() {
                stack.last_mut().unwrap().1 = k + 1;
                let c = children[v][k];
                if !reach[c] {
                    reach[c] = true;
                    depth[c] = depth[v] + 1;
                    t += 1;
                    tin[c] = t;
                    stack.push((c, 0));
                }
            } else {
                post.push(v);
                stack.pop();
            }
        }
        let edges: Vec<(usize, usize)> = net.edges.values().map(|e| (pos[&e.su], pos[&e.du])).collect();
        Ok(MulIndex {
            labels,
            by_name,
            parent,
            children,
            is_leaf,
            root,
            reach,
            depth,
            tin,
            post,
            edges,
        })
    }

    fn n(&self) -> usize {
        self.labels.len()
    }

    /// The reference's walk fails on anything but a single-rooted tree: a
    /// leaf root has no parent to look up, and an edge the root walk never
    /// keyed fails the per-edge lookup.  Raise the same way, up front.
    fn check_tree(&self) -> PyResult<()> {
        if self.is_leaf[self.root] {
            return Err(index_error());
        }
        for &(su, du) in &self.edges {
            if !self.reach[du] || self.parent[du] != Some(su) {
                return Err(PyKeyError::new_err((
                    (self.labels[su].clone(), self.labels[du].clone()),
                )));
            }
        }
        Ok(())
    }

    /// Index of the node a cluster member names (a label, or a ``Node``).
    fn node_named(&self, py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<usize> {
        let name = if let Ok(s) = v.extract::<String>() {
            s
        } else if let Ok(n) = v.cast::<Node>() {
            n.borrow().name.clone()
        } else {
            return Err(network_error(
                py,
                "Wrong type for parameter set_of_nodes. Expected set[Node] or set[str].",
            ));
        };
        self.by_name
            .get(&name)
            .copied()
            .ok_or_else(|| network_error(py, "A node in 'set_of_nodes' is not in the graph"))
    }

    fn lca_pair(&self, mut a: usize, mut b: usize) -> usize {
        while self.depth[a] > self.depth[b] {
            a = self.parent[a].expect("non-root tree node has a parent");
        }
        while self.depth[b] > self.depth[a] {
            b = self.parent[b].expect("non-root tree node has a parent");
        }
        while a != b {
            a = self.parent[a].expect("non-root tree node has a parent");
            b = self.parent[b].expect("non-root tree node has a parent");
        }
        a
    }

    /// Most recent common ancestor of a set of nodes: on a tree, the LCA of
    /// the members with the smallest and largest DFS entry times.
    fn mrca(&self, nodes: impl Iterator<Item = usize>) -> PyResult<usize> {
        let mut lo: Option<usize> = None;
        let mut hi: Option<usize> = None;
        for v in nodes {
            if !self.reach[v] {
                // No common ancestor at all: the reference minimised over an
                // empty candidate set.
                return Err(PyValueError::new_err("min() arg is an empty sequence"));
            }
            lo = Some(match lo {
                Some(l) if self.tin[l] <= self.tin[v] => l,
                _ => v,
            });
            hi = Some(match hi {
                Some(h) if self.tin[h] >= self.tin[v] => h,
                _ => v,
            });
        }
        match (lo, hi) {
            (Some(l), Some(h)) => Ok(self.lca_pair(l, h)),
            _ => Err(PyValueError::new_err("min() arg is an empty sequence")),
        }
    }

    /// ``(parent label, label)`` edge key of node ``v`` (``None`` above the root).
    fn edge_key(&self, v: usize) -> PyResult<(Option<String>, String)> {
        let par = if v == self.root {
            None
        } else {
            Some(self.labels[self.parent[v].ok_or_else(index_error)?].clone())
        };
        Ok((par, self.labels[v].clone()))
    }

    /// The reference's ``xl_helper`` walk from ``v``: lineages entering
    /// (bottom) and leaving (top) the edge above every node, children
    /// first.  ``count`` supplies the coalescences placed on a node's edge,
    /// ``visit`` sees every node's ``(bottom, top)``.
    fn walk(
        &self,
        v: usize,
        mapped: &[bool],
        count: &mut dyn FnMut(usize) -> PyResult<i64>,
        visit: &mut dyn FnMut(usize, i64, i64) -> PyResult<()>,
    ) -> PyResult<(i64, i64)> {
        if self.is_leaf[v] {
            if self.parent[v].is_none() {
                return Err(index_error());
            }
            let x = mapped[v] as i64;
            visit(v, x, x)?;
            return Ok((x, x));
        }
        if v != self.root && self.parent[v].is_none() {
            return Err(index_error());
        }
        let mut sum_tops = 0i64;
        for &c in &self.children[v] {
            sum_tops += self.walk(c, mapped, count, visit)?.1;
        }
        let bottom = sum_tops;
        let top = sum_tops - count(v)?;
        visit(v, bottom, top)?;
        Ok((bottom, top))
    }
}

/// Which MUL nodes some gene copy maps to under ``f`` (the reference's
/// ``label in f.map.values()`` test for leaves).
fn mapped_nodes(idx: &MulIndex, f: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
    let dict = AlleleMap::dict_of(f)?;
    let mut mapped = vec![false; idx.n()];
    for v in dict.values().iter() {
        if let Ok(s) = v.extract::<String>() {
            if let Some(&m) = idx.by_name.get(&s) {
                mapped[m] = true;
            }
        }
    }
    Ok(mapped)
}

/// ``len(mapping[(parent label, label)])``, read through ``__getitem__`` so
/// a ``defaultdict`` fills in and a plain dict raises, as in the reference.
fn count_in(mapping: &Bound<'_, PyAny>, key: (Option<String>, String)) -> PyResult<i64> {
    Ok(mapping.get_item(key)?.len()? as i64)
}

// ═══════════════════════════════════════════════════════════════════════
// Gene-tree digest and reconciliation
// ═══════════════════════════════════════════════════════════════════════

/// One gene tree: its leaves, and the leaf set below every internal node.
struct GtDigest {
    leaf_labels: Vec<String>,
    /// ``(label, leaf indices)`` of every non-leaf node, ``V()`` order.
    internal: Vec<(String, Vec<usize>)>,
}

impl GtDigest {
    fn build(py: Python<'_>, g: &Network) -> PyResult<GtDigest> {
        let leaf_uids = g.leaf_uids();
        let leaf_pos: FxHashMap<u64, usize> =
            leaf_uids.iter().enumerate().map(|(i, &u)| (u, i)).collect();
        let leaf_labels: Vec<String> = leaf_uids.iter().map(|&u| g.label(py, u)).collect();
        let mut internal = Vec::new();
        for &u in g.nodes.keys() {
            if leaf_pos.contains_key(&u) {
                continue;
            }
            let label = g.label(py, u);
            let mut idxs = Vec::new();
            for l in g.leaf_descendants_uid(u) {
                // A node with no leaf below it (a floater) had no entry in
                // the reference's leaf-descendant table.
                let Some(&i) = leaf_pos.get(&l) else {
                    return Err(PyKeyError::new_err(label));
                };
                idxs.push(i);
            }
            internal.push((label, idxs));
        }
        Ok(GtDigest {
            leaf_labels,
            internal,
        })
    }
}

/// One allele map applied to one gene tree: the MUL node each (used) gene
/// leaf maps to, and which MUL nodes receive a gene copy at all.
struct Resolved {
    f_idx: Vec<usize>,
    mapped: Vec<bool>,
}

fn resolve(py: Python<'_>, idx: &MulIndex, gt: &GtDigest, f: &Bound<'_, PyAny>) -> PyResult<Resolved> {
    let dict = AlleleMap::dict_of(f)?;
    let mut needed = vec![false; gt.leaf_labels.len()];
    for (_, leaves) in &gt.internal {
        for &l in leaves {
            needed[l] = true;
        }
    }
    let mut f_idx = vec![usize::MAX; gt.leaf_labels.len()];
    for (i, label) in gt.leaf_labels.iter().enumerate() {
        if !needed[i] {
            continue;
        }
        let Some(v) = dict.get_item(label.as_str())? else {
            return Err(PyKeyError::new_err(label.clone()));
        };
        f_idx[i] = idx.node_named(py, &v)?;
    }
    let mut mapped = vec![false; idx.n()];
    for v in dict.values().iter() {
        if let Ok(s) = v.extract::<String>() {
            if let Some(&m) = idx.by_name.get(&s) {
                mapped[m] = true;
            }
        }
    }
    Ok(Resolved { f_idx, mapped })
}

struct Scratch {
    counts: Vec<i64>,
    bottom: Vec<i64>,
    top: Vec<i64>,
}

impl Scratch {
    fn new(n: usize) -> Scratch {
        Scratch {
            counts: vec![0; n],
            bottom: vec![0; n],
            top: vec![0; n],
        }
    }
}

/// Extra lineages of one gene tree under one allele map (the reference's
/// ``XL_Allele``, before the clamp at zero): every internal gene-tree node
/// coalesces on the MUL edge above the MRCA of its (mapped) leaves; the
/// lineage count is then summed up the tree.
fn xl_allele(idx: &MulIndex, gt: &GtDigest, r: &Resolved, s: &mut Scratch) -> PyResult<i64> {
    s.counts.iter_mut().for_each(|c| *c = 0);
    for (_, leaves) in &gt.internal {
        let v = idx.mrca(leaves.iter().map(|&l| r.f_idx[l]))?;
        s.counts[v] += 1;
    }
    for &v in &idx.post {
        if idx.is_leaf[v] {
            let x = r.mapped[v] as i64;
            s.bottom[v] = x;
            s.top[v] = x;
        } else {
            let b: i64 = idx.children[v].iter().map(|&c| s.top[c]).sum();
            s.bottom[v] = b;
            s.top[v] = b - s.counts[v];
        }
    }
    let mut total = 0i64;
    for &(_, du) in &idx.edges {
        total += s.bottom[du] - 1;
        if !idx.is_leaf[du] {
            total -= s.counts[du];
        }
    }
    total += s.bottom[idx.root] - s.counts[idx.root] - 1;
    Ok(total)
}

struct MulCtx {
    idx: MulIndex,
    scratch: Scratch,
}

/// The MUL tree the reconciliation runs against; ``None`` is the
/// reference's unset ``self.mul``.
fn require_mul<'a, 'py>(
    mul: Option<&'a Bound<'py, Network>>,
    attr: &str,
) -> PyResult<&'a Bound<'py, Network>> {
    mul.ok_or_else(|| PyAttributeError::new_err(format!("'NoneType' object has no attribute '{attr}'")))
}

fn checked_index(py: Python<'_>, mul: &Bound<'_, Network>) -> PyResult<MulIndex> {
    let idx = MulIndex::build(py, &mul.borrow())?;
    idx.check_tree()?;
    Ok(idx)
}

/// The ``"allele maps"`` item of a gene tree.
fn allele_maps_item<'py>(py: Python<'py>, g: &Bound<'py, Network>) -> PyResult<Bound<'py, PyAny>> {
    let items = g.borrow().items.clone_ref(py);
    match items.bind(py).get_item("allele maps")? {
        Some(v) => Ok(v),
        None => Err(PyKeyError::new_err("allele maps")),
    }
}

/// ``XL``: the minimum over the gene tree's allele maps of its (non-negative)
/// extra-lineage count; 0 without allele maps.  The MUL index is built on
/// first use and shared across the gene trees of one ``score`` call.
fn xl_one(
    py: Python<'_>,
    mul: Option<&Bound<'_, Network>>,
    g: &Bound<'_, Network>,
    ctx: &mut Option<MulCtx>,
) -> PyResult<i64> {
    let maps = allele_maps_item(py, g)?;
    if !maps.is_truthy()? {
        return Ok(0);
    }
    let mul = require_mul(mul, "get_leaves")?;
    if ctx.is_none() {
        let idx = checked_index(py, mul)?;
        let n = idx.n();
        *ctx = Some(MulCtx {
            idx,
            scratch: Scratch::new(n),
        });
    }
    let c = ctx.as_mut().unwrap();
    let gt = GtDigest::build(py, &g.borrow())?;
    let mut best: Option<i64> = None;
    for f in maps.try_iter()? {
        let f = f?;
        let r = resolve(py, &c.idx, &gt, &f)?;
        let v = xl_allele(&c.idx, &gt, &r, &mut c.scratch)?.max(0);
        best = Some(best.map_or(v, |b| b.min(v)));
    }
    Ok(best.unwrap_or(0))
}

/// Total parsimony score of ``gt_list`` against the MUL tree ``mul``: the
/// sum over gene trees of ``XL``.
#[pyfunction]
#[pyo3(name = "_allop_score")]
fn allop_score_py(py: Python<'_>, mul: Option<&Bound<'_, Network>>, gt_list: &Bound<'_, PyAny>) -> PyResult<i64> {
    let mut ctx: Option<MulCtx> = None;
    let mut total = 0i64;
    for item in gt_list.try_iter()? {
        let g = item?.cast_into::<Network>()?;
        total += xl_one(py, mul, &g, &mut ctx)?;
    }
    Ok(total)
}

/// ``XL`` for one gene tree.
#[pyfunction]
#[pyo3(name = "_allop_xl")]
fn allop_xl_py(py: Python<'_>, mul: Option<&Bound<'_, Network>>, g: &Bound<'_, Network>) -> PyResult<i64> {
    let mut ctx: Option<MulCtx> = None;
    xl_one(py, mul, g, &mut ctx)
}

/// ``XL_Allele``: the extra lineages of ``g`` under the single allele map ``f``.
#[pyfunction]
#[pyo3(name = "_allop_xl_allele")]
fn allop_xl_allele_py(
    py: Python<'_>,
    mul: Option<&Bound<'_, Network>>,
    g: &Bound<'_, Network>,
    f: &Bound<'_, PyAny>,
) -> PyResult<i64> {
    let mul = require_mul(mul, "mrca")?;
    let idx = checked_index(py, mul)?;
    let gt = GtDigest::build(py, &g.borrow())?;
    let r = resolve(py, &idx, &gt, f)?;
    let mut scratch = Scratch::new(idx.n());
    xl_allele(&idx, &gt, &r, &mut scratch)
}

/// ``gene_tree_map``: the MUL edge each internal node of ``g`` lands on
/// under ``f``, as ``defaultdict(list)`` from ``(parent label, label)`` --
/// ``(None, root label)`` above the root -- to the gene-tree node labels.
#[pyfunction]
#[pyo3(name = "_allop_gene_tree_map")]
fn allop_gene_tree_map_py<'py>(
    py: Python<'py>,
    mul: Option<&Bound<'py, Network>>,
    g: &Bound<'py, Network>,
    f: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mul = require_mul(mul, "mrca")?;
    let idx = checked_index(py, mul)?;
    let gt = GtDigest::build(py, &g.borrow())?;
    let r = resolve(py, &idx, &gt, f)?;
    let out = py
        .import("collections")?
        .getattr("defaultdict")?
        .call1((py.get_type::<PyList>(),))?;
    for (label, leaves) in &gt.internal {
        let v = idx.mrca(leaves.iter().map(|&l| r.f_idx[l]))?;
        let bucket = out.get_item(idx.edge_key(v)?)?;
        bucket.cast::<PyList>()?.append(label)?;
    }
    Ok(out)
}

/// ``extra_lineages``: the extra-lineage count of a reconciliation given as
/// ``coal_event_map`` (edge key -> gene-tree nodes on that MUL edge) and the
/// allele map ``f``.
#[pyfunction]
#[pyo3(name = "_allop_extra_lineages")]
fn allop_extra_lineages_py(
    py: Python<'_>,
    mul: Option<&Bound<'_, Network>>,
    coal_event_map: &Bound<'_, PyAny>,
    f: &Bound<'_, PyAny>,
) -> PyResult<i64> {
    let mul = require_mul(mul, "root")?;
    let idx = checked_index(py, mul)?;
    let mapped = mapped_nodes(&idx, f)?;
    let mut bottoms = vec![0i64; idx.n()];
    let mut count = |v: usize| -> PyResult<i64> { count_in(coal_event_map, idx.edge_key(v)?) };
    let mut visit = |v: usize, bottom: i64, _top: i64| -> PyResult<()> {
        bottoms[v] = bottom;
        Ok(())
    };
    let (root_bottom, _) = idx.walk(idx.root, &mapped, &mut count, &mut visit)?;
    let root_xl = root_bottom - count(idx.root)? - 1;
    let mut total = 0i64;
    for &(_, du) in &idx.edges {
        total += bottoms[du] - 1;
        if !idx.is_leaf[du] {
            total -= count(du)?;
        }
    }
    Ok(total + root_xl)
}

/// ``xl_helper``: fill ``edge_xl_map`` with ``[bottom, top]`` lineage counts
/// for every edge in the subtree under ``start_node``.
#[pyfunction]
#[pyo3(name = "_allop_xl_helper")]
fn allop_xl_helper_py(
    py: Python<'_>,
    mul: Option<&Bound<'_, Network>>,
    start_node: &Bound<'_, PyAny>,
    edge_xl_map: &Bound<'_, PyAny>,
    coal_map: &Bound<'_, PyAny>,
    f: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let mul = require_mul(mul, "get_parents")?;
    let idx = checked_index(py, mul)?;
    let start = {
        let m = mul.borrow();
        let Some(uid) = m.uid_of_any(start_node) else {
            return Err(index_error());
        };
        m.nodes.get_index_of(&uid).ok_or_else(index_error)?
    };
    let mapped = mapped_nodes(&idx, f)?;
    let mut count = |v: usize| -> PyResult<i64> { count_in(coal_map, idx.edge_key(v)?) };
    let mut visit = |v: usize, bottom: i64, top: i64| -> PyResult<()> {
        edge_xl_map.set_item(idx.edge_key(v)?, PyList::new(py, [bottom, top])?)
    };
    idx.walk(start, &mapped, &mut count, &mut visit)?;
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AlleleMap>()?;
    m.add_function(wrap_pyfunction!(enumerate_allele_maps_py, m)?)?;
    m.add_function(wrap_pyfunction!(select_allele_maps_py, m)?)?;
    m.add_function(wrap_pyfunction!(allop_to_mul_py, m)?)?;
    m.add_function(wrap_pyfunction!(allop_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(allop_xl_py, m)?)?;
    m.add_function(wrap_pyfunction!(allop_xl_allele_py, m)?)?;
    m.add_function(wrap_pyfunction!(allop_gene_tree_map_py, m)?)?;
    m.add_function(wrap_pyfunction!(allop_extra_lineages_py, m)?)?;
    m.add_function(wrap_pyfunction!(allop_xl_helper_py, m)?)?;
    Ok(())
}
