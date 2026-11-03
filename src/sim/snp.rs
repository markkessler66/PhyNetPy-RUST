//! Random network generation and exact biallelic-marker simulation under
//! the MSNC: the compiled core behind ``phynetpy.SNPSimulator`` and
//! ``phynetpy._sim_markers``.
//!
//! Random draws go back to the caller's ``numpy.random.Generator`` in the
//! reference order (``uniform`` per reticulation gamma, ``choice(k, 2,
//! replace=False)`` per placement attempt, then the gene-tree draws of
//! ``sim::seq`` and one ``random()`` per genealogy node), so a seeded run
//! reproduces the Python stream.  The marker loop reuses one
//! [`GenePlan`] for every site and walks the genealogy arena directly, never
//! building a ``Network`` per site.

use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::{Edge, Network, Node};
use crate::sim::seq::{subst_unit, GTree, GenePlan, PyRng};
use crate::util::{network_error, node_error, py_float_repr, warn};

const NO_ROOT: &str = "There are no roots in this network. There is either a cycle, or \
                       nothing has been added and this is an empty network.";

fn node_label(py: Python<'_>, net: &Network, uid: u64) -> String {
    net.nodes
        .get(&uid)
        .map(|e| e.obj.borrow(py).name.clone())
        .unwrap_or_default()
}

// ═══════════════════════════════════════════════════════════════════════
// Network generation
// ═══════════════════════════════════════════════════════════════════════

/// ``_clean_node_names``: leaves become ``T1..Tn`` (by original label), the
/// root ``Root``, and the other internal nodes ``I1..Ik`` ordered by depth
/// then by the lowest-ranked leaf beneath them.
fn clean_node_names(py: Python<'_>, net: &Bound<'_, Network>) -> PyResult<()> {
    let renames: Vec<(u64, String)> = {
        let this = net.borrow();
        let root = this.root_uid(py)?;
        let mut leaves = this.leaf_uids();
        leaves.sort_by_cached_key(|u| node_label(py, &this, *u));
        let leaf_rank: FxHashMap<u64, usize> = leaves.iter().enumerate().map(|(i, u)| (*u, i)).collect();

        let mut renames: Vec<(u64, String)> = leaves
            .iter()
            .enumerate()
            .map(|(i, u)| (*u, format!("T{}", i + 1)))
            .collect();
        renames.push((root, "Root".to_string()));

        let (_, depth) = this.bfs_order(root, false);
        let below = this.leaf_descendants_all_map(py)?;
        let mut named: FxHashSet<u64> = leaves.iter().copied().collect();
        named.insert(root);
        let mut internal: Vec<(i64, usize, u64)> = Vec::new();
        for (uid, d) in depth.iter() {
            if named.contains(uid) {
                continue;
            }
            let min_rank = below
                .get(uid)
                .and_then(|ls| ls.iter().filter_map(|l| leaf_rank.get(l)).min().copied())
                .ok_or_else(|| PyValueError::new_err("min() arg is an empty sequence"))?;
            internal.push((*d, min_rank, *uid));
        }
        internal.sort_by_key(|(d, r, _)| (*d, *r));
        renames.extend(
            internal
                .iter()
                .enumerate()
                .map(|(i, (_, _, u))| (*u, format!("I{}", i + 1))),
        );
        renames
    };
    let mut this = net.borrow_mut();
    for (uid, name) in renames {
        if this.nodes.contains_key(&uid) {
            this.rename_uid(py, uid, name);
        }
    }
    Ok(())
}

/// ``Edge(src, dest, length=..., gamma=...)`` with the constructor's gamma
/// check.
fn make_edge(py: Python<'_>, src: &Py<Node>, dest: &Py<Node>, length: f64, gamma: Option<f64>) -> PyResult<Py<Edge>> {
    let gamma = gamma.unwrap_or(0.0);
    if !(0.0..=1.0).contains(&gamma) {
        return Err(PyValueError::new_err(
            "Please provide a probabilistic value for gamma (between 0 and 1, inclusive)!",
        ));
    }
    Py::new(
        py,
        Edge::make(
            py,
            src.clone_ref(py),
            dest.clone_ref(py),
            Some(length),
            Some(gamma),
            Some(0.0),
            Some("no tag assigned yet".to_string()),
        ),
    )
}

struct Cand {
    euid: u64,
    su: u64,
    du: u64,
}

/// Bisect edge ``(su -> du)`` with ``node``: returns the new node's time.
fn bisect(
    py: Python<'_>,
    net: &Bound<'_, Network>,
    cand: &Cand,
    node: &Py<Node>,
    top_gamma: Option<f64>,
) -> PyResult<f64> {
    let (src, dest, length) = {
        let this = net.borrow();
        let edge = this
            .edges
            .get(&cand.euid)
            .map(|e| e.obj.borrow(py).length)
            .ok_or_else(|| PyValueError::new_err("No matching edges found"))?;
        (
            this.node_obj(cand.su).clone_ref(py),
            this.node_obj(cand.du).clone_ref(py),
            edge,
        )
    };
    let length = length.ok_or_else(|| {
        PyTypeError::new_err("unsupported operand type(s) for /: 'NoneType' and 'float'")
    })?;
    let half = length / 2.0;
    let t_src = src
        .borrow(py)
        .t
        .ok_or_else(|| node_error(py, "No time has been set for this node!"))?;
    let t_new = t_src + half;
    Node::set_time(node.bind(py), t_new)?;
    {
        let mut this = net.borrow_mut();
        this.add_node(net.as_any(), node.bind(py))?;
        this.remove_edge_uid(py, cand.euid);
    }
    let top = make_edge(py, &src, node, half, top_gamma)?;
    let bottom = make_edge(py, node, &dest, half, None)?;
    {
        let mut this = net.borrow_mut();
        this.add_edge(net.as_any(), top.bind(py))?;
        this.add_edge(net.as_any(), bottom.bind(py))?;
    }
    Ok(t_new)
}

/// ``_add_reticulation``: graft one hybrid edge between two bisected,
/// non-nested internal edges.  ``False`` when no legal placement exists.
fn add_reticulation(
    py: Python<'_>,
    net: &Bound<'_, Network>,
    rng: &PyRng<'_>,
    gamma: f64,
    retic_id: i64,
) -> PyResult<bool> {
    // Candidates join two non-leaf, non-reticulation nodes, sorted by
    // (source label, destination label) so the draw below is reproducible.
    let candidates: Vec<Cand> = {
        let this = net.borrow();
        let leaves: FxHashSet<u64> = this.leaf_uids().into_iter().collect();
        let is_retic = |u: u64| this.nodes.get(&u).map(|e| e.obj.borrow(py).is_retic).unwrap_or(false);
        let mut cands: Vec<(String, String, Cand)> = this
            .edges
            .iter()
            .filter(|(_, e)| {
                !leaves.contains(&e.su) && !leaves.contains(&e.du) && !is_retic(e.su) && !is_retic(e.du)
            })
            .map(|(euid, e)| {
                (
                    node_label(py, &this, e.su),
                    node_label(py, &this, e.du),
                    Cand {
                        euid: *euid,
                        su: e.su,
                        du: e.du,
                    },
                )
            })
            .collect();
        cands.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        cands.into_iter().map(|(_, _, c)| c).collect()
    };
    if candidates.len() < 2 {
        return Ok(false);
    }

    // Resample until the two edges are node-disjoint and neither lies in the
    // other's subtree; the hybrid edge would otherwise close a directed cycle.
    let mut chosen: Option<(usize, usize)> = None;
    for _ in 0..50 {
        let (i, j) = rng.choice_pair(candidates.len())?;
        let (e1, e2) = (&candidates[i], &candidates[j]);
        let ends_1 = [e1.su, e1.du];
        let ends_2 = [e2.su, e2.du];
        if ends_1.iter().any(|x| ends_2.contains(x)) {
            continue;
        }
        let this = net.borrow();
        let sub_1 = this.subtree_uids(e1.du);
        if ends_2.iter().any(|x| sub_1.contains(x)) {
            continue;
        }
        let sub_2 = this.subtree_uids(e2.du);
        if ends_1.iter().any(|x| sub_2.contains(x)) {
            continue;
        }
        chosen = Some((i, j));
        break;
    }
    let Some((i, j)) = chosen else {
        return Ok(false);
    };
    let (edge1, edge2) = (&candidates[i], &candidates[j]);

    // Bisect edge 1 with a tree node, edge 2 with the hybrid node.
    let new_tree_node = Py::new(py, Node::make(py, format!("I_r{retic_id}"), false, None, None, None))?;
    let t_tree = bisect(py, net, edge1, &new_tree_node, None)?;
    let hybrid = Py::new(py, Node::make(py, format!("#H{retic_id}"), true, None, None, None))?;
    let t_hybrid = bisect(py, net, edge2, &hybrid, Some(gamma))?;

    // The reticulation edge: new_tree_node -> hybrid.
    let mut retic_len = (t_hybrid - t_tree).abs();
    if retic_len < 1e-10 {
        retic_len = 0.01; // minimum branch length to avoid numerical issues
    }
    let retic_edge = make_edge(py, &new_tree_node, &hybrid, retic_len, Some(1.0 - gamma))?;
    net.borrow_mut().add_edge(net.as_any(), retic_edge.bind(py))?;
    Ok(true)
}

/// ``random_network``: a Yule tree relabelled to PhyNetPy conventions, in
/// substitution units, with up to ``level`` reticulations grafted on.
#[pyfunction]
#[pyo3(signature = (rng, n, level, birth_rate, gamma_lo, gamma_hi))]
fn snp_random_network(
    py: Python<'_>,
    rng: Bound<'_, PyAny>,
    n: i64,
    level: i64,
    birth_rate: f64,
    gamma_lo: f64,
    gamma_hi: f64,
) -> PyResult<Py<PyAny>> {
    // Step 1: a base tree via the Yule process (draws from ``rng``).
    let yule_cls = py.import("phynetpy.BirthDeath")?.getattr("Yule")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("gamma", birth_rate)?;
    kwargs.set_item("n", n)?;
    kwargs.set_item("rng", &rng)?;
    let yule = yule_cls.call((), Some(&kwargs))?;
    let net_obj = yule.call_method0("generate_network")?;
    let net = net_obj.cast::<Network>()?;

    // Yule emits "spec*"/"internal*"/"root" labels and reports branch lengths
    // in expected substitutions per site.
    clean_node_names(py, net)?;
    net_obj.call_method1("set_branch_length_unit", (subst_unit(py)?,))?;

    // Step 2: add reticulations.
    let rng = PyRng::new(rng);
    for retic_id in 0..level.max(0) {
        let gamma = rng.uniform(gamma_lo, gamma_hi)?;
        if !add_reticulation(py, net, &rng, gamma, retic_id)? {
            warn(
                py,
                &format!(
                    "Could not place reticulation #{retic_id}. Network has {retic_id} reticulations instead of {level}."
                ),
            )?;
            break;
        }
    }
    Ok(net_obj.unbind())
}

/// ``_clean_node_names(net)``.
#[pyfunction]
fn snp_clean_node_names(py: Python<'_>, net: &Bound<'_, Network>) -> PyResult<()> {
    clean_node_names(py, net)
}

/// ``_add_reticulation(net, rng, gamma, retic_id)``.
#[pyfunction]
fn snp_add_reticulation(
    py: Python<'_>,
    net: &Bound<'_, Network>,
    rng: Bound<'_, PyAny>,
    gamma: f64,
    retic_id: i64,
) -> PyResult<bool> {
    let rng = PyRng::new(rng);
    add_reticulation(py, net, &rng, gamma, retic_id)
}

// ═══════════════════════════════════════════════════════════════════════
// Two-state mutation process
// ═══════════════════════════════════════════════════════════════════════

/// Probability of ending red after ``length`` from ``state`` under the exact
/// two-state CTMC (red -> green at rate ``u``, green -> red at rate ``v``).
fn red_probability(py: Python<'_>, state: i64, length: f64, u: f64, v: f64) -> PyResult<f64> {
    if !length.is_finite() || length < 0.0 {
        return Err(PyValueError::new_err(format!(
            "gene-tree branch lengths must be finite and non-negative; got {}.",
            py_float_repr(py, length)?
        )));
    }
    let total = u + v;
    let decay = (-total * length).exp();
    Ok(if state == 1 {
        v / total + (u / total) * decay
    } else {
        (v / total) * (1.0 - decay)
    })
}

fn mutate(py: Python<'_>, state: i64, length: f64, u: f64, v: f64, rng: &PyRng<'_>) -> PyResult<i64> {
    let p_red = red_probability(py, state, length, u, v)?;
    Ok(i64::from(rng.random()? < p_red))
}

/// ``_mutate_biallelic_state(state, length, u, v, rng)``.
#[pyfunction]
fn sim_mutate_biallelic_state(
    py: Python<'_>,
    state: i64,
    length: f64,
    u: f64,
    v: f64,
    rng: Bound<'_, PyAny>,
) -> PyResult<i64> {
    let rng = PyRng::new(rng);
    mutate(py, state, length, u, v, &rng)
}

/// One biallelic site down a genealogy arena: ``(leaf index, state)`` for
/// every leaf.  Mirrors ``simulate_biallelic_on_gene_tree`` on the
/// ``Network`` the reference parsed from the same Newick string.
fn biallelic_on_gtree(
    py: Python<'_>,
    tree: &GTree,
    root: usize,
    u: f64,
    v: f64,
    rng: &PyRng<'_>,
) -> PyResult<Vec<(usize, i64)>> {
    if tree.children(root).is_none() {
        // A single sampled lineage has no edges, so ``Network.root()`` fails.
        return Err(network_error(py, NO_ROOT));
    }
    let mut states = vec![0i64; tree.len()];
    states[root] = i64::from(rng.random()? < v / (u + v));
    let mut leaves = Vec::new();
    let mut stack = vec![root];
    while let Some(parent) = stack.pop() {
        let Some((a, b)) = tree.children(parent) else {
            leaves.push((parent, states[parent]));
            continue;
        };
        // Children in label order (a stable sort of the two).
        let (first, second) = if tree.label(a) <= tree.label(b) { (a, b) } else { (b, a) };
        for child in [first, second] {
            states[child] = mutate(py, states[parent], tree.branch_len(parent, child), u, v, rng)?;
            stack.push(child);
        }
    }
    Ok(leaves)
}

/// ``simulate_biallelic_on_gene_tree(gene_tree, u, v, rng)`` on a ``Network``.
#[pyfunction]
fn sim_biallelic_on_gene_tree<'py>(
    py: Python<'py>,
    gene_tree: &Bound<'py, Network>,
    u: f64,
    v: f64,
    rng: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // Pre-order plan: (parent slot, child slot, branch length) in the order
    // the reference visits them, children sorted by label.
    let (edges, leaves, n_slots) = {
        let net = gene_tree.borrow();
        if !net.is_acyclic_impl() {
            return Err(network_error(
                py,
                "biallelic mutation simulation requires an acyclic gene tree.",
            ));
        }
        let root = net.root_uid(py)?;
        let mut slot: FxHashMap<u64, usize> = FxHashMap::default();
        slot.insert(root, 0);
        let mut edges: Vec<(usize, usize, Option<f64>)> = Vec::new();
        let mut stack = vec![root];
        while let Some(parent) = stack.pop() {
            let ps = slot.get(&parent).copied().unwrap_or(0);
            let mut kids = net.children(parent);
            kids.sort_by_cached_key(|c| node_label(py, &net, *c));
            for child in kids {
                let edge = net.edge_between(py, parent, child, None, None)?;
                let length = edge.borrow(py).length;
                let next = slot.len();
                let cs = *slot.entry(child).or_insert(next);
                edges.push((ps, cs, length));
                stack.push(child);
            }
        }
        let mut leaves: Vec<(String, Option<usize>)> = net
            .leaf_uids()
            .into_iter()
            .map(|l| (node_label(py, &net, l), slot.get(&l).copied()))
            .collect();
        leaves.sort_by(|a, b| a.0.cmp(&b.0));
        (edges, leaves, slot.len())
    };

    let rng = PyRng::new(rng);
    let mut states = vec![0i64; n_slots];
    states[0] = i64::from(rng.random()? < v / (u + v));
    for (ps, cs, length) in edges {
        let length = length.ok_or_else(|| {
            PyTypeError::new_err("float() argument must be a string or a real number, not 'NoneType'")
        })?;
        let parent_state = states.get(ps).copied().unwrap_or(0);
        let child_state = mutate(py, parent_state, length, u, v, &rng)?;
        if let Some(s) = states.get_mut(cs) {
            *s = child_state;
        }
    }

    let out = PyDict::new(py);
    for (label, slot) in leaves {
        let state = slot
            .and_then(|s| states.get(s).copied())
            .ok_or_else(|| PyKeyError::new_err(label.clone()))?;
        out.set_item(label, state)?;
    }
    Ok(out)
}

/// The site loop of ``simulate_biallelic_markers``: one MSNC genealogy per
/// site, the two-state process down it, and per-taxon red counts.
#[pyfunction]
#[pyo3(signature = (network, n_sites, mapping, theta, u, v, branch_thetas, rng))]
#[allow(clippy::too_many_arguments)]
fn sim_biallelic_markers<'py>(
    py: Python<'py>,
    network: &Bound<'py, Network>,
    n_sites: i64,
    mapping: &Bound<'py, PyAny>,
    theta: f64,
    u: f64,
    v: f64,
    branch_thetas: Option<Bound<'py, PyAny>>,
    rng: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let plan = GenePlan::new(py, network, mapping, theta, branch_thetas.as_ref())?;

    let mut taxa: Vec<(String, Vec<String>)> = Vec::new();
    for item in mapping.call_method0("items")?.try_iter()? {
        let (k, v): (Bound<'py, PyAny>, Bound<'py, PyAny>) = item?.extract()?;
        let mut labels = Vec::new();
        for lab in v.try_iter()? {
            labels.push(lab?.str()?.to_string());
        }
        taxa.push((k.str()?.to_string(), labels));
    }

    let rng = PyRng::new(rng);
    let sites = n_sites.max(0) as usize;
    let mut data: Vec<Vec<i64>> = taxa.iter().map(|_| Vec::with_capacity(sites)).collect();
    for _ in 0..sites {
        let (tree, root) = plan.run(&rng)?;
        let leaf_states = biallelic_on_gtree(py, &tree, root, u, v, &rng)?;
        let by_label: FxHashMap<&str, i64> = leaf_states.iter().map(|(i, s)| (tree.label(*i), *s)).collect();
        for (counts, (_, labels)) in data.iter_mut().zip(&taxa) {
            let mut total = 0i64;
            for label in labels {
                total += by_label
                    .get(label.as_str())
                    .copied()
                    .ok_or_else(|| PyKeyError::new_err(label.clone()))?;
            }
            counts.push(total);
        }
    }

    let out = PyDict::new(py);
    for ((taxon, _), counts) in taxa.iter().zip(data) {
        out.set_item(taxon, counts)?;
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(snp_random_network, m)?)?;
    m.add_function(wrap_pyfunction!(snp_clean_node_names, m)?)?;
    m.add_function(wrap_pyfunction!(snp_add_reticulation, m)?)?;
    m.add_function(wrap_pyfunction!(sim_mutate_biallelic_state, m)?)?;
    m.add_function(wrap_pyfunction!(sim_biallelic_on_gene_tree, m)?)?;
    m.add_function(wrap_pyfunction!(sim_biallelic_markers, m)?)?;
    Ok(())
}
