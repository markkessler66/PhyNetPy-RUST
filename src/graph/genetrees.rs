//! GeneTrees analysis: the computational side of ``phynetpy.GeneTrees``.
//!
//! ``python/phynetpy/GeneTrees.py`` keeps the ``GeneTrees`` container (its
//! ``trees`` set, ``taxa_names`` set, mapping and naming rule are plain
//! Python attributes that other modules read directly) and delegates every
//! analysis to the ``_gt_*`` functions registered here: mapping resolution
//! and validation, cluster / split support, consensus clusters and trees,
//! Robinson-Foulds distances, the modal gene tree, gene concordance factors,
//! duplication-loss reconciliation, and the plain Newick -> ``Network``
//! conversion the ASTRAL wrapper used BioPython for.
//!
//! Every function follows its Python reference step for step.  Where the
//! reference iterated a Python ``set`` or ``dict`` the same Python object is
//! iterated here, so orders that depend on Python's hashing (and therefore
//! tie-breaks such as which modal topology wins) are unchanged.  Clusters
//! are sets of interned label ids; they are turned back into frozensets of
//! strings only when they are handed to Python.

use std::collections::{BTreeSet, VecDeque};

use indexmap::{IndexMap, IndexSet};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyType};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use super::edge::Edge;
use super::network::Network;
use super::node::Node;
use crate::io::{phylo_read, ParsedTree};
use crate::util::warn;

const DEFAULT_TAG: &str = "no tag assigned yet";

// ───────────────────────────── GeneTreeError ───────────────────────────

static GT_EXC: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// ``phynetpy.GeneTrees`` registers its ``GeneTreeError`` class here when it
/// is imported.
#[pyfunction]
fn _gt_register_gene_tree_error(py: Python<'_>, cls: Bound<'_, PyType>) {
    let _ = GT_EXC.set(py, cls.unbind());
}

/// An instance of the Python-defined ``GeneTreeError``.
fn gene_tree_error(py: Python<'_>, msg: impl Into<String>) -> PyErr {
    if GT_EXC.get(py).is_none() {
        // Importing the module registers the class as a side effect.
        let _ = py.import("phynetpy.GeneTrees");
    }
    let msg = msg.into();
    match GT_EXC.get(py) {
        Some(t) => PyErr::from_type(t.bind(py).clone(), msg),
        None => PyRuntimeError::new_err(msg),
    }
}

// ───────────────────────────── Labels ──────────────────────────────────

/// A cluster: the interned ids of its leaf labels.
type Cluster = BTreeSet<u32>;
/// ``frozenset[frozenset[str]]``: a tree's topology signature.
type Signature = BTreeSet<Cluster>;

/// Leaf labels interned for one call, so clusters hash and compare as sets
/// of small integers.
#[derive(Default)]
struct Labels {
    names: Vec<String>,
    index: FxHashMap<String, u32>,
}

impl Labels {
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(&i) = self.index.get(name) {
            return i;
        }
        let i = self.names.len() as u32;
        self.names.push(name.to_string());
        self.index.insert(name.to_string(), i);
        i
    }

    /// Intern every element of a Python iterable of labels (``str(x)`` of
    /// anything that is not already a string, which can never match a
    /// leaf label anyway).
    fn intern_iterable(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<Cluster> {
        let mut out = Cluster::new();
        for item in obj.try_iter()? {
            let item = item?;
            let text = match item.extract::<String>() {
                Ok(s) => s,
                Err(_) => item.str()?.to_string(),
            };
            out.insert(self.intern(&text));
        }
        Ok(out)
    }

    fn name(&self, i: u32) -> &str {
        &self.names[i as usize]
    }

    fn frozenset<'py>(&self, py: Python<'py>, c: &Cluster) -> PyResult<Bound<'py, PyFrozenSet>> {
        PyFrozenSet::new(py, c.iter().map(|&i| self.name(i)))
    }

    fn set<'py>(&self, py: Python<'py>, c: &Cluster) -> PyResult<Bound<'py, PySet>> {
        PySet::new(py, c.iter().map(|&i| self.name(i)))
    }
}

/// The leaf labels of one network, each read once.
struct TreeLabels<'a> {
    net: &'a Network,
    cache: FxHashMap<u64, u32>,
}

impl<'a> TreeLabels<'a> {
    fn new(net: &'a Network) -> Self {
        TreeLabels { net, cache: FxHashMap::default() }
    }

    fn get(&mut self, py: Python<'_>, labels: &mut Labels, uid: u64) -> u32 {
        if let Some(&i) = self.cache.get(&uid) {
            return i;
        }
        let i = labels.intern(&self.net.label(py, uid));
        self.cache.insert(uid, i);
        i
    }

    /// ``{leaf.label for leaf in net.get_leaves()}``.
    fn leaves(&mut self, py: Python<'_>, labels: &mut Labels) -> Cluster {
        let uids = self.net.leaf_uids();
        uids.into_iter().map(|u| self.get(py, labels, u)).collect()
    }

    /// ``{n.label for n in net.leaf_descendants(node)}``.
    fn below(&mut self, py: Python<'_>, labels: &mut Labels, uid: u64) -> Cluster {
        let uids = self.net.leaf_descendants_uid(uid);
        uids.into_iter().map(|u| self.get(py, labels, u)).collect()
    }

    /// ``{frozenset(n.label for n in c) for c in get_all_clusters(net, include_trivial)}``:
    /// the leaf-label sets below every non-root node with more than one
    /// leaf, plus the singletons when asked, without duplicates.
    fn clusters(&mut self, py: Python<'_>, labels: &mut Labels, include_trivial: bool) -> PyResult<IndexSet<Cluster, FxBuildHasher>> {
        let root = self.net.root_uid(py)?;
        let map = self.net.leaf_descendants_all_map(py)?;
        let mut out: IndexSet<Cluster, FxBuildHasher> = IndexSet::default();
        for (uid, below) in map.iter() {
            if *uid != root && below.len() > 1 {
                let c: Cluster = below.iter().map(|u| self.get(py, labels, *u)).collect();
                out.insert(c);
            }
        }
        if include_trivial {
            for leaf in self.net.leaf_uids() {
                out.insert(Cluster::from([self.get(py, labels, leaf)]));
            }
        }
        Ok(out)
    }
}

// ───────────────────────────── Helpers ─────────────────────────────────

/// The gene trees, in the order Python iterates the collection.
fn collect_trees<'py>(trees: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, Network>>> {
    let mut out = Vec::new();
    for item in trees.try_iter()? {
        out.push(item?.cast_into::<Network>()?);
    }
    Ok(out)
}

/// ``cluster_support`` before normalisation: cluster -> number of gene
/// trees containing it, in first-seen order.
fn cluster_counts(
    py: Python<'_>,
    trees: &[Bound<'_, Network>],
    labels: &mut Labels,
    include_trivial: bool,
) -> PyResult<IndexMap<Cluster, usize, FxBuildHasher>> {
    let mut counts: IndexMap<Cluster, usize, FxBuildHasher> = IndexMap::default();
    for tree in trees {
        let net = tree.borrow();
        let mut tl = TreeLabels::new(&net);
        for c in tl.clusters(py, labels, include_trivial)? {
            *counts.entry(c).or_insert(0) += 1;
        }
    }
    Ok(counts)
}

fn support_value(count: usize, num_trees: usize, normalize: bool) -> f64 {
    if normalize {
        count as f64 / num_trees as f64
    } else {
        count as f64
    }
}

/// ``{frozenset(c): value}``.
fn support_dict<'py>(
    py: Python<'py>,
    labels: &Labels,
    counts: &IndexMap<Cluster, usize, FxBuildHasher>,
    num_trees: usize,
    normalize: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (c, &n) in counts.iter() {
        out.set_item(labels.frozenset(py, c)?, support_value(n, num_trees, normalize))?;
    }
    Ok(out)
}

/// ``support_on_reference``: the support of every rooted cluster of
/// ``ref_tree`` (0.0 when no gene tree has it), in the reference's cluster
/// order.
fn reference_support(
    py: Python<'_>,
    trees: &[Bound<'_, Network>],
    ref_tree: &Network,
    labels: &mut Labels,
    include_trivial: bool,
    normalize: bool,
) -> PyResult<IndexMap<Cluster, f64, FxBuildHasher>> {
    let counts = cluster_counts(py, trees, labels, include_trivial)?;
    let mut tl = TreeLabels::new(ref_tree);
    let mut out: IndexMap<Cluster, f64, FxBuildHasher> = IndexMap::default();
    for c in tl.clusters(py, labels, include_trivial)? {
        let v = match counts.get(&c) {
            Some(&n) => support_value(n, trees.len(), normalize),
            None => 0.0,
        };
        out.insert(c, v);
    }
    Ok(out)
}

fn new_node<'py>(py: Python<'py>, name: String) -> PyResult<Bound<'py, Node>> {
    Bound::new(py, Node::make(py, name, false, None, None, None))
}

/// ``Edge(parent, child)`` with the constructor's defaults, or with an
/// explicit length.
fn new_edge<'py>(py: Python<'py>, src: &Bound<'py, Node>, dest: &Bound<'py, Node>, length: f64) -> PyResult<Bound<'py, Edge>> {
    Bound::new(
        py,
        Edge::make(
            py,
            src.clone().unbind(),
            dest.clone().unbind(),
            Some(length),
            Some(0.0),
            Some(0.0),
            Some(DEFAULT_TAG.to_string()),
        ),
    )
}

// ───────────────────────────── Mapping ─────────────────────────────────

/// ``GeneTrees._resolve_mapping``: species -> gene labels from the explicit
/// mapping (filtered to present taxa), else the naming rule, else identity.
#[pyfunction]
#[pyo3(name = "_gt_resolve_mapping")]
fn resolve_mapping<'py>(
    py: Python<'py>,
    taxa_names: &Bound<'py, PyAny>,
    species_gene_mapping: &Bound<'py, PyAny>,
    naming_rule: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    if !species_gene_mapping.is_none() {
        for item in species_gene_mapping.call_method0("items")?.try_iter()? {
            let item = item?;
            let sp = item.get_item(0)?;
            let genes = item.get_item(1)?;
            let mut present: Vec<Bound<'py, PyAny>> = Vec::new();
            for g in genes.try_iter()? {
                let g = g?;
                if taxa_names.contains(&g)? {
                    present.push(g);
                }
            }
            if !present.is_empty() {
                out.set_item(sp, PyList::new(py, present)?)?;
            }
        }
        return Ok(out);
    }
    if !naming_rule.is_none() {
        for name in taxa_names.try_iter()? {
            let name = name?;
            let key = naming_rule.call1((&name,))?;
            let bucket = out.call_method1("setdefault", (key, PyList::empty(py)))?;
            bucket.call_method1("append", (name,))?;
        }
        return Ok(out);
    }
    for name in taxa_names.try_iter()? {
        let name = name?;
        out.set_item(&name, PyList::new(py, [&name])?)?;
    }
    Ok(out)
}

/// ``GeneTrees.validate_mapping``: the list of issues with the explicit
/// mapping; warns about taxa the mapping does not cover.
#[pyfunction]
#[pyo3(name = "_gt_validate_mapping")]
fn validate_mapping<'py>(
    py: Python<'py>,
    taxa_names: &Bound<'py, PyAny>,
    species_gene_mapping: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let issues = PyList::empty(py);
    if species_gene_mapping.is_none() {
        return Ok(issues);
    }
    let mapped_genes = PySet::empty(py)?;
    for item in species_gene_mapping.call_method0("items")?.try_iter()? {
        let item = item?;
        let sp = item.get_item(0)?;
        let genes = item.get_item(1)?;
        for g in genes.try_iter()? {
            let g = g?;
            if mapped_genes.contains(&g)? {
                issues.append(format!("Gene '{}' appears under multiple species entries", g.str()?))?;
            }
            mapped_genes.add(&g)?;
            if !taxa_names.contains(&g)? {
                issues.append(format!(
                    "Mapped gene '{}' (species '{}') not found in any gene tree",
                    g.str()?,
                    sp.str()?
                ))?;
            }
        }
    }
    let unmapped = taxa_names.call_method1("difference", (&mapped_genes,))?;
    let n_unmapped = unmapped.len()?;
    if n_unmapped > 0 {
        let sorted = py.import("builtins")?.getattr("sorted")?.call1((&unmapped,))?;
        let listing = sorted.repr()?.to_string();
        issues.append(format!("Taxa not covered by mapping: {listing}"))?;
        warn(
            py,
            &format!("GeneTrees: {n_unmapped} taxa not covered by the species-gene mapping: {listing}"),
        )?;
    }
    Ok(issues)
}

// ───────────────────────────── Support ─────────────────────────────────

/// ``GeneTrees.cluster_support``.
#[pyfunction]
#[pyo3(name = "_gt_cluster_support", signature = (trees, include_trivial=false, normalize=true))]
fn cluster_support<'py>(
    py: Python<'py>,
    trees: &Bound<'py, PyAny>,
    include_trivial: bool,
    normalize: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let trees = collect_trees(trees)?;
    if trees.is_empty() {
        return Ok(PyDict::new(py));
    }
    let mut labels = Labels::default();
    let counts = cluster_counts(py, &trees, &mut labels, include_trivial)?;
    support_dict(py, &labels, &counts, trees.len(), normalize)
}

/// ``GeneTrees.split_support``: non-trivial bipartitions keyed by their
/// smaller side.
#[pyfunction]
#[pyo3(name = "_gt_split_support", signature = (trees, normalize=true))]
fn split_support<'py>(py: Python<'py>, trees: &Bound<'py, PyAny>, normalize: bool) -> PyResult<Bound<'py, PyDict>> {
    let trees = collect_trees(trees)?;
    if trees.is_empty() {
        return Ok(PyDict::new(py));
    }
    let mut labels = Labels::default();
    let mut counts: IndexMap<Cluster, usize, FxBuildHasher> = IndexMap::default();
    for tree in &trees {
        let net = tree.borrow();
        let mut tl = TreeLabels::new(&net);
        let taxa = tl.leaves(py, &mut labels);
        for a in tl.clusters(py, &mut labels, false)? {
            let b: Cluster = taxa.difference(&a).copied().collect();
            if a.len() >= 2 && b.len() >= 2 {
                let key = if a.len() <= b.len() { a } else { b };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    support_dict(py, &labels, &counts, trees.len(), normalize)
}

/// ``GeneTrees.support_on_reference``.
#[pyfunction]
#[pyo3(name = "_gt_support_on_reference", signature = (trees, ref_tree, include_trivial=false, normalize=true))]
fn support_on_reference<'py>(
    py: Python<'py>,
    trees: &Bound<'py, PyAny>,
    ref_tree: &Bound<'py, Network>,
    include_trivial: bool,
    normalize: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let trees = collect_trees(trees)?;
    let mut labels = Labels::default();
    let support = reference_support(py, &trees, &ref_tree.borrow(), &mut labels, include_trivial, normalize)?;
    let out = PyDict::new(py);
    for (c, v) in support.iter() {
        out.set_item(labels.frozenset(py, c)?, *v)?;
    }
    Ok(out)
}

/// ``GeneTrees.annotate_reference_support``: store each reference edge's
/// cluster support in the edge weight.
#[pyfunction]
#[pyo3(name = "_gt_annotate_reference_support", signature = (trees, ref_tree, include_trivial=false, normalize=true))]
fn annotate_reference_support<'py>(
    py: Python<'py>,
    trees: &Bound<'py, PyAny>,
    ref_tree: &Bound<'py, Network>,
    include_trivial: bool,
    normalize: bool,
) -> PyResult<()> {
    let trees = collect_trees(trees)?;
    let mut labels = Labels::default();
    let mut weights: Vec<(Py<Edge>, f64)> = Vec::new();
    {
        let net = ref_tree.borrow();
        let support = reference_support(py, &trees, &net, &mut labels, include_trivial, normalize)?;
        let mut tl = TreeLabels::new(&net);
        let total_taxa = tl.leaves(py, &mut labels).len();
        for (euid, entry) in net.edges.iter() {
            let below = tl.below(py, &mut labels, entry.du);
            if !include_trivial && (below.len() <= 1 || below.len() == total_taxa) {
                continue;
            }
            let val = support.get(&below).copied().unwrap_or(0.0);
            weights.push((net.edges[euid].obj.clone_ref(py), val));
        }
    }
    for (edge, val) in weights {
        edge.bind(py).call_method1("set_weight", (val,))?;
    }
    Ok(())
}

/// ``GeneTrees.consensus_clusters``: rooted clusters with frequency at
/// least ``threshold``, as sets of labels.
#[pyfunction]
#[pyo3(name = "_gt_consensus_clusters", signature = (trees, threshold=0.5, include_trivial=false))]
fn consensus_clusters<'py>(
    py: Python<'py>,
    trees: &Bound<'py, PyAny>,
    threshold: f64,
    include_trivial: bool,
) -> PyResult<Bound<'py, PyList>> {
    let trees = collect_trees(trees)?;
    let out = PyList::empty(py);
    if trees.is_empty() {
        return Ok(out);
    }
    let mut labels = Labels::default();
    let counts = cluster_counts(py, &trees, &mut labels, include_trivial)?;
    for (c, &n) in counts.iter() {
        if support_value(n, trees.len(), true) >= threshold {
            out.append(labels.set(py, c)?)?;
        }
    }
    Ok(out)
}

// ───────────────────────────── Distances ───────────────────────────────

/// ``GeneTrees.rf_distance``: the average rooted Robinson-Foulds distance
/// between each gene tree and the reference over their shared taxa.
#[pyfunction]
#[pyo3(name = "_gt_rf_distance", signature = (trees, ref_tree, normalize=false))]
fn rf_distance(py: Python<'_>, trees: &Bound<'_, PyAny>, ref_tree: &Bound<'_, Network>, normalize: bool) -> PyResult<f64> {
    let trees = collect_trees(trees)?;
    if trees.is_empty() {
        return Ok(0.0);
    }
    let mut labels = Labels::default();
    let reference = ref_tree.borrow();
    let mut rl = TreeLabels::new(&reference);
    let ref_taxa = rl.leaves(py, &mut labels);
    let ref_clusters: Vec<Cluster> = rl.clusters(py, &mut labels, false)?.into_iter().collect();

    let mut total = 0.0;
    for tree in &trees {
        let net = tree.borrow();
        let mut tl = TreeLabels::new(&net);
        let taxa = tl.leaves(py, &mut labels);
        let common: Cluster = ref_taxa.intersection(&taxa).copied().collect();
        if common.len() < 3 {
            // Too small to define internal clusters.
            continue;
        }
        let informative = |c: &Cluster| c.len() >= 2 && c.len() < common.len();
        // The reference clusters are filtered by size only; the gene tree's
        // are projected onto the shared taxa first.
        let a: FxHashSet<&Cluster> = ref_clusters.iter().filter(|c| informative(c)).collect();
        let projected: FxHashSet<Cluster> = tl
            .clusters(py, &mut labels, false)?
            .into_iter()
            .map(|c| c.intersection(&common).copied().collect())
            .collect();
        let b: FxHashSet<&Cluster> = projected.iter().filter(|c| informative(c)).collect();

        let rf = a.difference(&b).count() + b.difference(&a).count();
        if normalize {
            let max_rf = a.len() + b.len();
            total += if max_rf > 0 { rf as f64 / max_rf as f64 } else { 0.0 };
        } else {
            total += rf as f64;
        }
    }
    Ok(total / trees.len() as f64)
}

/// ``GeneTrees.most_frequent_gene_tree``: a deep copy of a representative
/// of the modal topology class.
#[pyfunction]
#[pyo3(name = "_gt_most_frequent_gene_tree", signature = (trees, include_trivial=false, restrict_to_taxa=None, tiebreak="first", seed=None))]
fn most_frequent_gene_tree<'py>(
    py: Python<'py>,
    trees: &Bound<'py, PyAny>,
    include_trivial: bool,
    restrict_to_taxa: Option<Bound<'py, PyAny>>,
    tiebreak: &str,
    seed: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let trees = collect_trees(trees)?;
    if trees.is_empty() {
        return Err(gene_tree_error(py, "No gene trees available to pick a seed from."));
    }
    let mut labels = Labels::default();
    let eligible: Vec<&Bound<'py, Network>> = match &restrict_to_taxa {
        Some(target) => {
            let target = labels.intern_iterable(target)?;
            let mut keep = Vec::new();
            for tree in &trees {
                let net = tree.borrow();
                if TreeLabels::new(&net).leaves(py, &mut labels) == target {
                    keep.push(tree);
                }
            }
            if keep.is_empty() {
                return Err(gene_tree_error(
                    py,
                    format!(
                        "No gene trees cover exactly restrict_to_taxa ({} taxa); cannot build a seed.",
                        target.len()
                    ),
                ));
            }
            keep
        }
        None => trees.iter().collect(),
    };

    // Topology signature: frozenset of rooted cluster-labels.
    let mut groups: IndexMap<Signature, Vec<usize>, FxBuildHasher> = IndexMap::default();
    for (i, tree) in eligible.iter().enumerate() {
        let net = tree.borrow();
        let sig: Signature = TreeLabels::new(&net).clusters(py, &mut labels, include_trivial)?.into_iter().collect();
        groups.entry(sig).or_default().push(i);
    }
    let max_size = groups.values().map(Vec::len).max().unwrap_or(0);
    let winners: Vec<&Vec<usize>> = groups.values().filter(|g| g.len() == max_size).collect();

    let representative = if winners.len() == 1 || tiebreak == "first" {
        winners[0][0]
    } else {
        let rng = py.import("random")?.getattr("Random")?.call1((seed,))?;
        let choices = PyList::new(py, 0..winners.len())?;
        let k: usize = rng.call_method1("choice", (choices,))?.extract()?;
        winners[k][0]
    };
    py.import("copy")?.getattr("deepcopy")?.call1((eligible[representative],))
}

// ───────────────────────────── Consensus ───────────────────────────────

/// Two rooted clusters can coexist in one tree when one contains the other
/// or they are disjoint.
fn compatible(a: &Cluster, b: &Cluster) -> bool {
    a.is_subset(b) || b.is_subset(a) || a.is_disjoint(b)
}

/// ``GeneTrees._clusters_to_tree``: realise a compatible cluster set as a
/// (possibly multifurcating) tree over ``taxa``.
fn clusters_to_tree<'py>(
    py: Python<'py>,
    labels: &Labels,
    taxa: &Cluster,
    clusters: &[Cluster],
) -> PyResult<Bound<'py, Network>> {
    let net = Network::new_object(py, None)?;
    let mut name_to_node: FxHashMap<u32, Py<Node>> = FxHashMap::default();

    fn build<'py>(
        py: Python<'py>,
        net: &Bound<'py, Network>,
        labels: &Labels,
        clusters: &[Cluster],
        subtaxa: &Cluster,
        name_to_node: &mut FxHashMap<u32, Py<Node>>,
    ) -> PyResult<Bound<'py, Node>> {
        if subtaxa.len() == 1 {
            let label = *subtaxa.iter().next().expect("one element");
            if let Some(n) = name_to_node.get(&label) {
                return Ok(n.bind(py).clone());
            }
            let node = new_node(py, labels.name(label).to_string())?;
            net.borrow_mut().add_node(net.as_any(), &node)?;
            name_to_node.insert(label, node.clone().unbind());
            return Ok(node);
        }

        // Maximal proper clusters under this subtaxa.
        let mut proper: Vec<&Cluster> = clusters
            .iter()
            .filter(|c| c.len() < subtaxa.len() && c.is_subset(subtaxa))
            .collect();
        proper.sort_by(|x, y| y.len().cmp(&x.len()));
        let mut maximal: Vec<Cluster> = Vec::new();
        for c in proper {
            if !maximal.iter().any(|m| c.is_subset(m)) {
                maximal.push(c.clone());
            }
        }
        // Fill remainder as singletons.
        let covered: Cluster = maximal.iter().flatten().copied().collect();
        for &x in subtaxa.difference(&covered) {
            maximal.push(Cluster::from([x]));
        }

        let parent = new_node(py, format!("Internal_{}", net.borrow().nodes.len()))?;
        net.borrow_mut().add_node(net.as_any(), &parent)?;
        for block in &maximal {
            let child = build(py, net, labels, clusters, block, name_to_node)?;
            let edge = new_edge(py, &parent, &child, 1.0)?;
            net.borrow_mut().add_edge(net.as_any(), &edge)?;
        }
        Ok(parent)
    }

    build(py, &net, labels, clusters, taxa, &mut name_to_node)?;
    // Clean spurious degree-1 chains.
    Network::clean_impl(&net, [false, false, true])?;
    Ok(net)
}

/// ``GeneTrees._clusters_to_tree`` for Python callers: ``taxa`` is a set of
/// labels, ``clusters`` an iterable of label sets.
#[pyfunction]
#[pyo3(name = "_gt_clusters_to_tree")]
fn clusters_to_tree_py<'py>(
    py: Python<'py>,
    taxa: &Bound<'py, PyAny>,
    clusters: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, Network>> {
    let mut labels = Labels::default();
    let taxa = labels.intern_iterable(taxa)?;
    let mut sets: IndexSet<Cluster, FxBuildHasher> = IndexSet::default();
    for c in clusters.try_iter()? {
        sets.insert(labels.intern_iterable(&c?)?);
    }
    let sets: Vec<Cluster> = sets.into_iter().collect();
    clusters_to_tree(py, &labels, &taxa, &sets)
}

/// ``GeneTrees.build_majority_rule_consensus_tree``: greedy compatible
/// clusters by support, realised as a tree over ``taxa_names``.
#[pyfunction]
#[pyo3(name = "_gt_build_majority_rule_consensus_tree", signature = (taxa_names, trees, threshold=0.5))]
fn build_majority_rule_consensus_tree<'py>(
    py: Python<'py>,
    taxa_names: &Bound<'py, PyAny>,
    trees: &Bound<'py, PyAny>,
    threshold: f64,
) -> PyResult<Bound<'py, Network>> {
    let mut labels = Labels::default();
    let taxa = labels.intern_iterable(taxa_names)?;
    if taxa.is_empty() {
        return Network::new_object(py, None);
    }
    let trees = collect_trees(trees)?;
    let counts = cluster_counts(py, &trees, &mut labels, false)?;

    // Filter by threshold and sort by support then size (large first);
    // ties keep their first-seen order.
    let mut cand: Vec<(Cluster, f64)> = counts
        .into_iter()
        .map(|(c, n)| (c, support_value(n, trees.len(), true)))
        .filter(|(_, v)| *v >= threshold)
        .collect();
    cand.sort_by(|x, y| {
        y.1.partial_cmp(&x.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| y.0.len().cmp(&x.0.len()))
    });

    let mut selected: Vec<Cluster> = Vec::new();
    for (c, _) in cand {
        if selected.iter().all(|s| compatible(&c, s)) {
            selected.push(c);
        }
    }
    clusters_to_tree(py, &labels, &taxa, &selected)
}

// ───────────────────────────── Concordance ─────────────────────────────

/// ``GeneTrees.gene_concordance_factors``: per reference edge, the fraction
/// of informative gene trees carrying the induced split, keyed by the
/// (min, max) labels of the edge's child cluster.
#[pyfunction]
#[pyo3(name = "_gt_gene_concordance_factors")]
fn gene_concordance_factors<'py>(
    py: Python<'py>,
    trees: &Bound<'py, PyAny>,
    ref_tree: &Bound<'py, Network>,
) -> PyResult<Bound<'py, PyDict>> {
    let trees = collect_trees(trees)?;
    let mut labels = Labels::default();

    // Precompute clusters (and taxa) for all gene trees.
    let mut tree_clusters: Vec<FxHashSet<Cluster>> = Vec::with_capacity(trees.len());
    let mut tree_taxa: Vec<Cluster> = Vec::with_capacity(trees.len());
    for tree in &trees {
        let net = tree.borrow();
        let mut tl = TreeLabels::new(&net);
        tree_clusters.push(tl.clusters(py, &mut labels, false)?.into_iter().collect());
        tree_taxa.push(tl.leaves(py, &mut labels));
    }

    let reference = ref_tree.borrow();
    let mut rl = TreeLabels::new(&reference);
    let all_taxa = rl.leaves(py, &mut labels);

    let mut results: IndexMap<(String, String), f64, FxBuildHasher> = IndexMap::default();
    for entry in reference.edges.values() {
        let a = rl.below(py, &mut labels, entry.du);
        let b: Cluster = all_taxa.difference(&a).copied().collect();
        if a.len() < 2 || b.len() < 2 {
            continue;
        }
        let mut informative = 0usize;
        let mut support = 0usize;
        for (i, taxa_t) in tree_taxa.iter().enumerate() {
            let a_t: Cluster = a.intersection(taxa_t).copied().collect();
            let b_t: Cluster = b.intersection(taxa_t).copied().collect();
            if a_t.is_empty() || b_t.is_empty() {
                continue;
            }
            // Both sides need at least 2 taxa for the edge to be informative.
            if a_t.len() < 2 && b_t.len() < 2 {
                continue;
            }
            informative += 1;
            if tree_clusters[i].contains(&a_t) || tree_clusters[i].contains(&b_t) {
                support += 1;
            }
        }
        if informative > 0 {
            // A short, order-independent key: the smallest and largest label.
            let mut names: Vec<&str> = a.iter().map(|&i| labels.name(i)).collect();
            names.sort_unstable();
            let key = (names[0].to_string(), names[names.len() - 1].to_string());
            results.insert(key, support as f64 / informative as f64);
        }
    }

    let out = PyDict::new(py);
    for (key, v) in results {
        out.set_item(key, v)?;
    }
    Ok(out)
}

// ───────────────────────────── Reconciliation ──────────────────────────

/// Number of edges from ``anc`` down to ``desc`` (0 when ``desc`` is not
/// below ``anc``).
fn distance_down(net: &Network, anc: u64, desc: u64) -> usize {
    if anc == desc {
        return 0;
    }
    let mut q: VecDeque<(u64, usize)> = VecDeque::new();
    q.push_back((anc, 0));
    let mut seen: FxHashSet<u64> = FxHashSet::default();
    seen.insert(anc);
    while let Some((cur, d)) = q.pop_front() {
        for ch in net.children(cur) {
            if seen.contains(&ch) {
                continue;
            }
            if ch == desc {
                return d + 1;
            }
            seen.insert(ch);
            q.push_back((ch, d + 1));
        }
    }
    0
}

/// ``GeneTrees.duplication_loss_summary``: LCA reconciliation of each gene
/// tree against ``species_tree`` under the resolved species -> genes map.
#[pyfunction]
#[pyo3(name = "_gt_duplication_loss_summary")]
fn duplication_loss_summary<'py>(
    py: Python<'py>,
    trees: &Bound<'py, PyAny>,
    species_tree: &Bound<'py, Network>,
    species_map: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let trees = collect_trees(trees)?;

    // gene label -> species name (a later species entry wins).
    let mut gene_to_species: FxHashMap<String, Option<String>> = FxHashMap::default();
    for item in species_map.call_method0("items")?.try_iter()? {
        let item = item?;
        let sp = item.get_item(0)?.extract::<String>().ok();
        for g in item.get_item(1)?.try_iter()? {
            if let Ok(g) = g?.extract::<String>() {
                gene_to_species.insert(g, sp.clone());
            }
        }
    }

    let sp = species_tree.borrow();
    let mut sp_name_to_uid: FxHashMap<String, u64> = FxHashMap::default();
    for leaf in sp.leaf_uids() {
        sp_name_to_uid.insert(sp.label(py, leaf), leaf);
    }

    let mut total_dups = 0usize;
    let mut total_losses = 0usize;
    let details = PyList::empty(py);

    for tree in &trees {
        let net = tree.borrow();
        let mut leaf_to_species: FxHashMap<u64, u64> = FxHashMap::default();
        for leaf in net.leaf_uids() {
            let label = net.label(py, leaf);
            let Some(Some(species)) = gene_to_species.get(&label) else { continue };
            if let Some(&su) = sp_name_to_uid.get(species) {
                leaf_to_species.insert(leaf, su);
            }
        }

        let mut mapped: FxHashMap<u64, u64> = FxHashMap::default();
        let mut map_node = |n: u64| -> PyResult<u64> {
            if let Some(&m) = mapped.get(&n) {
                return Ok(m);
            }
            if let Some(&su) = leaf_to_species.get(&n) {
                mapped.insert(n, su);
                return Ok(su);
            }
            // The species below ``n`` (unmapped leaves contribute nothing).
            let mut desc_species: IndexSet<u64, FxBuildHasher> = IndexSet::default();
            let mut stack = vec![n];
            while let Some(x) = stack.pop() {
                if let Some(&su) = leaf_to_species.get(&x) {
                    desc_species.insert(su);
                } else {
                    stack.extend(net.children(x));
                }
            }
            let sp_node = if desc_species.is_empty() {
                sp.root_uid(py)?
            } else {
                let targets: Vec<u64> = desc_species.into_iter().collect();
                sp.mrca_uids(py, &targets)?
            };
            mapped.insert(n, sp_node);
            Ok(sp_node)
        };

        let mut dups = 0usize;
        let mut losses = 0usize;
        for n in net.nodes.keys() {
            let ch = net.children(*n);
            if ch.len() < 2 {
                continue;
            }
            let m_n = map_node(*n)?;
            let m_c1 = map_node(ch[0])?;
            let m_c2 = map_node(ch[1])?;
            if m_n == m_c1 || m_n == m_c2 {
                dups += 1;
            }
            for m_c in [m_c1, m_c2] {
                if sp.subtree_uids(m_n).contains(&m_c) {
                    let d = distance_down(&sp, m_n, m_c);
                    losses += d.saturating_sub(1);
                }
            }
        }

        total_dups += dups;
        total_losses += losses;
        let per = PyDict::new(py);
        per.set_item("duplications", dups)?;
        per.set_item("losses", losses)?;
        details.append(per)?;
    }

    let totals = PyDict::new(py);
    totals.set_item("duplications", total_dups)?;
    totals.set_item("losses", total_losses)?;
    let out = PyDict::new(py);
    out.set_item("totals", totals)?;
    out.set_item("per_tree", details)?;
    Ok(out)
}

// ───────────────────────────── Newick -> Network ───────────────────────

/// ``GeneTrees._biopy_tree_to_network(Phylo.read(StringIO(s), "newick"))``:
/// a plain tree's Newick string into a ``Network`` with ``Internal_<n>``
/// names for unnamed clades, node times accumulated from branch lengths
/// (1.0 when missing) and those lengths on the edges.
#[pyfunction]
#[pyo3(name = "_gt_newick_to_network")]
fn newick_to_network<'py>(py: Python<'py>, newick_str: &str) -> PyResult<Bound<'py, Network>> {
    let tree: ParsedTree =
        phylo_read(newick_str.trim()).map_err(|m| gene_tree_error(py, format!("Failed to parse newick string: {m}")))?;
    let net = Network::new_object(py, None)?;
    let mut created: FxHashMap<usize, Bound<'py, Node>> = FxHashMap::default();

    // Create a clade's node (once), timed from its parent's node.
    let mut ensure_node = |idx: usize, parent: Option<&Bound<'py, Node>>| -> PyResult<Bound<'py, Node>> {
        if let Some(n) = created.get(&idx) {
            return Ok(n.clone());
        }
        let clade = &tree.clades[idx];
        let name = clade.name.clone().unwrap_or_else(|| format!("Internal_{}", created.len()));
        let node = new_node(py, name)?;
        match parent {
            None => Node::set_time(&node, 0.0)?,
            Some(p) => {
                let bl = clade.branch_length.unwrap_or(1.0);
                let pt = p.borrow().t.unwrap_or(0.0);
                Node::set_time(&node, pt + bl)?;
            }
        }
        net.borrow_mut().add_node(net.as_any(), &node)?;
        created.insert(idx, node.clone());
        Ok(node)
    };

    // Create nodes top-down, breadth first.
    let root = ensure_node(tree.root, None)?;
    let mut q: VecDeque<(usize, Bound<'py, Node>)> = VecDeque::new();
    q.push_back((tree.root, root));
    while let Some((cur, src_node)) = q.pop_front() {
        for &child in &tree.clades[cur].clades {
            let child_node = ensure_node(child, Some(&src_node))?;
            let bl = tree.clades[child].branch_length.unwrap_or(1.0);
            let edge = new_edge(py, &src_node, &child_node, bl)?;
            net.borrow_mut().add_edge(net.as_any(), &edge)?;
            q.push_back((child, child_node));
        }
    }
    Ok(net)
}

// ───────────────────────────── Registration ────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_gt_register_gene_tree_error, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(validate_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_support, m)?)?;
    m.add_function(wrap_pyfunction!(split_support, m)?)?;
    m.add_function(wrap_pyfunction!(support_on_reference, m)?)?;
    m.add_function(wrap_pyfunction!(annotate_reference_support, m)?)?;
    m.add_function(wrap_pyfunction!(consensus_clusters, m)?)?;
    m.add_function(wrap_pyfunction!(rf_distance, m)?)?;
    m.add_function(wrap_pyfunction!(most_frequent_gene_tree, m)?)?;
    m.add_function(wrap_pyfunction!(clusters_to_tree_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_majority_rule_consensus_tree, m)?)?;
    m.add_function(wrap_pyfunction!(gene_concordance_factors, m)?)?;
    m.add_function(wrap_pyfunction!(duplication_loss_summary, m)?)?;
    m.add_function(wrap_pyfunction!(newick_to_network, m)?)?;
    Ok(())
}
