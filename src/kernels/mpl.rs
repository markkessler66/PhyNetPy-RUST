//! Maximum pseudo-likelihood (Yu & Nakhleh 2015): the species-network
//! triplet DP and the gene-tree triplet frequencies (rho).
//!
//! The DP mirrors PhyloNet's pseudo-likelihood code path.  Each DP state is
//! an integer 0..10 encoding which of the three triplet lineages are
//! present at a node and how far they have coalesced; configurations carry
//! a vector of reticulation-parent choices so lineages that cross the same
//! reticulation agree on the parent they took.  Unlike the earlier compiled
//! kernel there are no fixed limits on node, reticulation or configuration
//! counts.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::common::{gij, LOG_FLOOR};
use crate::graph::{Network, Node};

const NUM_STATES: usize = 11;

/// Within-branch coalescent transitions: ``(target, lineages_in, lineages_out)``.
static COAL: [&[(u8, u8, u8)]; NUM_STATES] = [
    &[(0, 0, 0)],
    &[(1, 1, 1)],
    &[(2, 1, 1)],
    &[(3, 1, 1)],
    &[(4, 2, 2), (8, 2, 1)],
    &[(5, 2, 2)],
    &[(6, 2, 2)],
    &[(7, 3, 3), (9, 3, 2), (10, 3, 1)],
    &[(8, 1, 1)],
    &[(9, 2, 2), (10, 2, 1)],
    &[(10, 1, 1)],
];

/// Reticulation splits: ``(state on parent 1, state on parent 2)``.
static SPLIT: [&[(u8, u8)]; NUM_STATES] = [
    &[(0, 0)],
    &[(0, 1)],
    &[(0, 2)],
    &[(0, 3)],
    &[(0, 4), (1, 2)],
    &[(0, 5), (1, 3)],
    &[(0, 6), (2, 3)],
    &[(0, 7), (1, 6), (2, 5), (3, 4)],
    &[(0, 8)],
    &[(0, 9), (3, 8)],
    &[(0, 10)],
];

#[inline]
fn merge_lookup(s1: u8, s2: u8) -> Option<u8> {
    let (a, b) = if s1 < s2 { (s1, s2) } else { (s2, s1) };
    match (a, b) {
        (1, 2) => Some(4),
        (1, 3) => Some(5),
        (1, 6) => Some(7),
        (2, 3) => Some(6),
        (2, 5) => Some(7),
        (3, 4) => Some(7),
        (3, 8) => Some(9),
        _ => None,
    }
}

type Choices = SmallVec<[u16; 8]>;

#[derive(Clone, Debug)]
struct Config {
    p: f64,
    idx: Choices,
}

#[derive(Default, Clone)]
struct StateMap {
    slots: Vec<Vec<Config>>,
}

impl StateMap {
    fn new() -> Self {
        StateMap {
            slots: (0..NUM_STATES).map(|_| Vec::new()).collect(),
        }
    }
    #[inline]
    fn clear(&mut self) {
        for s in self.slots.iter_mut() {
            s.clear();
        }
    }
}

#[inline]
fn compatible(a: &Choices, b: &Choices) -> bool {
    a.iter().zip(b.iter()).all(|(&x, &y)| x == y || x == 0 || y == 0)
}

#[inline]
fn slot_add_dedup(slot: &mut Vec<Config>, cfg: Config) {
    for existing in slot.iter_mut() {
        if existing.idx == cfg.idx {
            existing.p += cfg.p;
            return;
        }
    }
    slot.push(cfg);
}

/// Pre-extracted topology of one species network, in leaf-to-root order.
pub struct Topo {
    pub n: usize,
    pub nr: usize,
    pub is_leaf: Vec<bool>,
    pub is_retic: Vec<bool>,
    pub in_art: Vec<bool>,
    pub in_low_art: Vec<bool>,
    pub children: Vec<Vec<usize>>,
    pub parents: Vec<Vec<usize>>,
    /// Branch length per (node, parent slot); ``None`` is an infinite branch.
    pub pa_bl: Vec<Vec<Option<f64>>>,
    pub pa_gamma: Vec<Vec<f64>>,
    /// For child ``j`` of node ``i``: index of ``i`` among the child's parents.
    pub ch_pa_slot: Vec<Vec<usize>>,
    pub labels: Vec<String>,
    pub label_to_idx: FxHashMap<String, usize>,
    /// Offset of node ``i``'s first parent slot in the DP workspace.
    pub slot_offset: Vec<usize>,
    pub n_slots: usize,
}

/// Reusable DP scratch space for one topology.
struct Workspace {
    store: Vec<StateMap>,
    active: Vec<bool>,
    tmp: StateMap,
    ap1: StateMap,
    ap2: StateMap,
    gij_cache: FxHashMap<(u64, u8, u8), f64>,
}

impl Workspace {
    fn new(topo: &Topo) -> Self {
        Workspace {
            store: (0..topo.n_slots).map(|_| StateMap::new()).collect(),
            active: vec![false; topo.n_slots],
            tmp: StateMap::new(),
            ap1: StateMap::new(),
            ap2: StateMap::new(),
            gij_cache: FxHashMap::default(),
        }
    }

}

/// ``AC+ -> AC-``: propagate configurations through one branch.
fn compute_ac_minus(
    result: &mut StateMap,
    cacs: &StateMap,
    bl: Option<f64>,
    gamma: f64,
    ws_gij: &mut dyn FnMut(Option<f64>, u8, u8) -> f64,
) {
    result.clear();
    for (sid, configs) in cacs.slots.iter().enumerate() {
        if configs.is_empty() {
            continue;
        }
        for &(tgt, lin_in, lin_out) in COAL[sid] {
            let mut prob = 1.0;
            if gamma != 1.0 {
                prob = gamma.powi(lin_in as i32);
            }
            if lin_in > 1 {
                prob *= ws_gij(bl, lin_in, lin_out);
                if lin_in == 3 && lin_out != 3 {
                    prob /= 3.0;
                }
            }
            if prob == 0.0 {
                continue;
            }
            let slot = &mut result.slots[tgt as usize];
            for cfg in configs {
                let np = (cfg.p * prob).max(0.0);
                slot_add_dedup(
                    slot,
                    Config {
                        p: np,
                        idx: cfg.idx.clone(),
                    },
                );
            }
        }
    }
}

/// Split configurations at a reticulation across its two parent edges.
fn split_at_retic(ap1: &mut StateMap, ap2: &mut StateMap, cacs: &StateMap, net_node_id: usize) {
    ap1.clear();
    ap2.clear();
    let mut net_index: u16 = 1;
    for (sid, configs) in cacs.slots.iter().enumerate() {
        if configs.is_empty() {
            continue;
        }
        for &(s1, s2) in SPLIT[sid] {
            for cfg in configs {
                let prob = cfg.p.sqrt();
                let mut c1 = cfg.clone();
                c1.idx[net_node_id] = net_index;
                c1.p = prob;
                let c2 = c1.clone();
                ap1.slots[s1 as usize].push(c1);
                ap2.slots[s2 as usize].push(c2);
                net_index += 1;
                if sid != 0 {
                    let mut c1 = cfg.clone();
                    c1.idx[net_node_id] = net_index;
                    c1.p = prob;
                    let c2 = c1.clone();
                    ap1.slots[s2 as usize].push(c1);
                    ap2.slots[s1 as usize].push(c2);
                    net_index += 1;
                }
            }
        }
    }
}

/// Copy a single child's configurations, collapsing them at a lowest
/// articulation node.
fn copy_with_art(dst: &mut StateMap, src: &StateMap, is_low_art: bool, nr: usize) {
    dst.clear();
    for (s, configs) in src.slots.iter().enumerate() {
        if configs.is_empty() {
            continue;
        }
        if is_low_art {
            let total: f64 = configs.iter().map(|c| c.p).sum();
            dst.slots[s].push(Config {
                p: total,
                idx: SmallVec::from_elem(0, nr),
            });
        } else {
            dst.slots[s].extend(configs.iter().cloned());
        }
    }
}

/// Merge the configurations of two children at a tree node.
fn merge_children(dst: &mut StateMap, ac1: &StateMap, ac2: &StateMap, is_low_art: bool, nr: usize) {
    dst.clear();
    for (s1, cfgs1) in ac1.slots.iter().enumerate() {
        if cfgs1.is_empty() {
            continue;
        }
        for (s2, cfgs2) in ac2.slots.iter().enumerate() {
            if cfgs2.is_empty() {
                continue;
            }
            let tgt = if s1 == 0 || s2 == 0 {
                if s1 != 0 {
                    s1
                } else {
                    s2
                }
            } else {
                match merge_lookup(s1 as u8, s2 as u8) {
                    Some(t) => t as usize,
                    None => continue,
                }
            };
            let slot = &mut dst.slots[tgt];
            for c1 in cfgs1 {
                for c2 in cfgs2 {
                    if !compatible(&c1.idx, &c2.idx) {
                        continue;
                    }
                    let p = (c1.p * c2.p).max(0.0);
                    if is_low_art {
                        if slot.is_empty() {
                            slot.push(Config {
                                p,
                                idx: SmallVec::from_elem(0, nr),
                            });
                        } else {
                            slot[0].p += p;
                        }
                    } else {
                        let idx: Choices = c1
                            .idx
                            .iter()
                            .zip(c2.idx.iter())
                            .map(|(&a, &b)| a.max(b))
                            .collect();
                        slot.push(Config { p, idx });
                    }
                }
            }
        }
    }
}

/// ``P(xy|z)`` for one ordered triple of leaf indices.
fn calc_triple(topo: &Topo, ws: &mut Workspace, triple: [usize; 3]) -> f64 {
    let nr = topo.nr;
    for a in ws.active.iter_mut() {
        *a = false;
    }
    let mut net_node_id = 0usize;
    let Workspace {
        store,
        active,
        tmp,
        ap1,
        ap2,
        gij_cache,
    } = ws;
    let mut gij_fn = |l: Option<f64>, i: u8, j: u8| -> f64 {
        let key = (l.map(f64::to_bits).unwrap_or(u64::MAX), i, j);
        if let Some(v) = gij_cache.get(&key) {
            return *v;
        }
        let v = gij(l, i as u32, j as u32);
        gij_cache.insert(key, v);
        v
    };

    for i in 0..topo.n {
        let mut has = false;
        if topo.is_leaf[i] {
            for j in 0..3 {
                if triple[j] == i {
                    tmp.clear();
                    tmp.slots[j + 1].push(Config {
                        p: 1.0,
                        idx: SmallVec::from_elem(0, nr),
                    });
                    has = true;
                    break;
                }
            }
        } else if topo.is_retic[i] {
            if let Some(&c0) = topo.children[i].first() {
                let k = topo.slot_offset[c0] + topo.ch_pa_slot[i][0];
                if active[k] {
                    tmp.clear();
                    for (s, cfgs) in store[k].slots.iter().enumerate() {
                        tmp.slots[s].extend(cfgs.iter().cloned());
                    }
                    has = true;
                }
            }
        } else if topo.children[i].len() >= 2 {
            let (c0, c1) = (topo.children[i][0], topo.children[i][1]);
            let k0 = topo.slot_offset[c0] + topo.ch_pa_slot[i][0];
            let k1 = topo.slot_offset[c1] + topo.ch_pa_slot[i][1];
            let (h0, h1) = (active[k0], active[k1]);
            if h0 && !h1 {
                copy_with_art(tmp, &store[k0], topo.in_low_art[i], nr);
                has = true;
            } else if h1 && !h0 {
                copy_with_art(tmp, &store[k1], topo.in_low_art[i], nr);
                has = true;
            } else if h0 && h1 {
                merge_children(tmp, &store[k0], &store[k1], topo.in_low_art[i], nr);
                has = true;
            }
        }
        if !has {
            continue;
        }

        if !tmp.slots[7].is_empty() && topo.in_art[i] {
            let mut total = tmp.slots[7][0].p / 3.0;
            if let Some(c) = tmp.slots[9].first() {
                total += c.p;
            }
            if let Some(c) = tmp.slots[10].first() {
                total += c.p;
            }
            return total;
        }

        if topo.is_retic[i] {
            split_at_retic(ap1, ap2, tmp, net_node_id);
            net_node_id += 1;
            for j in 0..topo.parents[i].len() {
                let k = topo.slot_offset[i] + j;
                let src = if j == 0 { &*ap1 } else { &*ap2 };
                compute_ac_minus(&mut store[k], src, topo.pa_bl[i][j], topo.pa_gamma[i][j], &mut gij_fn);
                active[k] = true;
            }
        } else if !topo.parents[i].is_empty() {
            let k = topo.slot_offset[i];
            compute_ac_minus(&mut store[k], tmp, topo.pa_bl[i][0], 1.0, &mut gij_fn);
            active[k] = true;
        }
    }
    0.0
}

/// Sum of ``rho_i * log(p_i)`` over the indexed triplets.
fn score_indexed(topo: &Topo, ws: &mut Workspace, trips: &[[usize; 3]], rho: &[[f64; 3]]) -> f64 {
    let mut total = 0.0;
    for (t, r) in trips.iter().zip(rho.iter()) {
        let p_xy = calc_triple(topo, ws, [t[0], t[1], t[2]]);
        let p_xz = calc_triple(topo, ws, [t[0], t[2], t[1]]);
        let p_yz = (1.0 - p_xy - p_xz).max(0.0);
        for (ri, pi) in r.iter().zip([p_xy, p_xz, p_yz]) {
            if *ri > 0.0 {
                total += ri * if pi > 0.0 { pi.ln() } else { LOG_FLOOR };
            }
        }
    }
    total
}

// ═══════════════════════════════════════════════════════════════════════
// Topology extraction
// ═══════════════════════════════════════════════════════════════════════

/// ``_is_valid_network`` from the reference engine, evaluated on an
/// adjacency snapshot with edge ``(parent -> node)`` removed.
fn disconnects_after_removal(
    children: &[Vec<usize>],
    parents: &[Vec<usize>],
    parent: usize,
    node: usize,
) -> bool {
    let n = children.len();
    let kids = |v: usize| -> Vec<usize> {
        children[v]
            .iter()
            .copied()
            .filter(|&c| !(v == parent && c == node))
            .collect()
    };
    let pars = |v: usize| -> Vec<usize> {
        parents[v]
            .iter()
            .copied()
            .filter(|&p| !(v == node && p == parent))
            .collect()
    };
    // Roots of the modified graph (in-degree 0, out-degree > 0), original
    // order first.
    let root = (0..n).find(|&v| pars(v).is_empty() && !kids(v).is_empty());
    let Some(root) = root else { return false };
    let mut visited: FxHashSet<usize> = FxHashSet::default();
    let mut stack = vec![root];
    while let Some(cur) = stack.pop() {
        if !visited.insert(cur) {
            continue;
        }
        stack.extend(kids(cur));
    }
    let mut referenced: FxHashSet<usize> = FxHashSet::default();
    for &v in visited.iter() {
        let ps = pars(v);
        let cs = kids(v);
        if ps.len() == 1 && cs.len() == 1 && v != parent {
            return false;
        }
        referenced.extend(ps);
        referenced.extend(cs);
    }
    visited.len() == referenced.len()
}

impl Topo {
    pub fn from_network(py: Python<'_>, net: &Network) -> PyResult<(Topo, Vec<Py<Node>>)> {
        let order = net.topological_order_uids(py)?;
        let uids: Vec<u64> = order.into_iter().rev().collect();
        let n = uids.len();
        let index: FxHashMap<u64, usize> = uids.iter().enumerate().map(|(i, u)| (*u, i)).collect();
        let node_objs: Vec<Py<Node>> = uids.iter().map(|u| net.node_obj(*u).clone_ref(py)).collect();
        let labels: Vec<String> = node_objs.iter().map(|o| o.borrow(py).name.clone()).collect();
        let children: Vec<Vec<usize>> = uids.iter().map(|&u| net.children(u).into_iter().map(|c| index[&c]).collect()).collect();
        let parents: Vec<Vec<usize>> = uids.iter().map(|&u| net.parents(u).into_iter().map(|p| index[&p]).collect()).collect();
        let is_leaf: Vec<bool> = uids.iter().map(|&u| net.out_deg(u) == 0).collect();
        let is_retic: Vec<bool> = uids.iter().map(|&u| net.in_deg(u) > 1).collect();
        let nr = is_retic.iter().filter(|&&r| r).count();

        // Articulation nodes, in the reference's leaf-to-root sweep.
        let root_uid = net.root_uid(py).ok();
        let mut in_art = vec![false; n];
        let mut in_low_art = vec![false; n];
        for i in 0..n {
            let is_root = root_uid == Some(uids[i]);
            let is_tree_node = parents[i].len() <= 1 && !is_leaf[i];
            if is_leaf[i] {
                in_art[i] = true;
                continue;
            }
            let all_children_art = children[i].iter().all(|&c| in_art[c]);
            if is_root {
                if !all_children_art {
                    in_low_art[i] = true;
                }
                in_art[i] = true;
                continue;
            }
            if is_tree_node {
                if all_children_art {
                    in_art[i] = true;
                } else {
                    let parent = parents[i][0];
                    if disconnects_after_removal(&children, &parents, parent, i) {
                        in_low_art[i] = true;
                        in_art[i] = true;
                    }
                }
            }
        }

        let mut slot_offset = vec![0usize; n];
        let mut n_slots = 0usize;
        for i in 0..n {
            slot_offset[i] = n_slots;
            n_slots += parents[i].len().max(1);
        }
        let ch_pa_slot: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                children[i]
                    .iter()
                    .map(|&c| parents[c].iter().position(|&p| p == i).unwrap_or(0))
                    .collect()
            })
            .collect();
        let label_to_idx: FxHashMap<String, usize> = (0..n)
            .filter(|&i| is_leaf[i])
            .map(|i| (labels[i].clone(), i))
            .collect();
        let mut topo = Topo {
            n,
            nr,
            is_leaf,
            is_retic,
            in_art,
            in_low_art,
            children,
            parents,
            pa_bl: vec![Vec::new(); n],
            pa_gamma: vec![Vec::new(); n],
            ch_pa_slot,
            labels,
            label_to_idx,
            slot_offset,
            n_slots,
        };
        topo.read_params(py, net, &uids)?;
        Ok((topo, node_objs))
    }

    /// Re-read branch lengths and inheritance probabilities (the
    /// parameters-only refresh path).
    pub fn read_params(&mut self, py: Python<'_>, net: &Network, uids: &[u64]) -> PyResult<()> {
        for i in 0..self.n {
            let n_par = self.parents[i].len();
            let mut bls = Vec::with_capacity(n_par);
            let mut gammas = Vec::with_capacity(n_par);
            for &p in &self.parents[i] {
                let e = net.first_edge_between(py, uids[p], uids[i])?;
                let eb = net.edge_obj(e).borrow(py);
                bls.push(eb.length);
                let g = if self.is_retic[i] {
                    match eb.gamma {
                        Some(g) if g > 0.0 => g,
                        _ => 1.0 / n_par as f64,
                    }
                } else {
                    1.0
                };
                gammas.push(g);
            }
            self.pa_bl[i] = bls;
            self.pa_gamma[i] = gammas;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Python classes
// ═══════════════════════════════════════════════════════════════════════

/// Indexed triplets and their rho values for one topology.
#[pyclass(name = "_TripletIndex", module = "phynetpy._mpl")]
pub struct TripletIndex {
    trips: Vec<[usize; 3]>,
    rho: Vec<[f64; 3]>,
}

#[pymethods]
impl TripletIndex {
    fn __len__(&self) -> usize {
        self.trips.len()
    }
}

/// The DP engine bound to one species network.
#[pyclass(name = "_MplTopology", module = "phynetpy._mpl")]
pub struct MplTopology {
    topo: Topo,
    uids: Vec<u64>,
    node_objs: Vec<Py<Node>>,
    ws: Workspace,
}

#[pymethods]
impl MplTopology {
    #[new]
    fn py_new(py: Python<'_>, network: &Bound<'_, Network>) -> PyResult<Self> {
        let net = network.borrow();
        let (topo, node_objs) = Topo::from_network(py, &net)?;
        let uids: Vec<u64> = node_objs.iter().map(|o| o.borrow(py).uid).collect();
        let ws = Workspace::new(&topo);
        Ok(MplTopology {
            topo,
            uids,
            node_objs,
            ws,
        })
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.topo.n
    }

    #[getter]
    fn net_node_num(&self) -> usize {
        self.topo.nr
    }

    #[getter]
    fn label_to_idx(&self) -> FxHashMap<String, usize> {
        self.topo.label_to_idx.clone()
    }

    /// Nodes in the engine's leaf-to-root order.
    fn nodes(&self, py: Python<'_>) -> Vec<Py<Node>> {
        self.node_objs.iter().map(|o| o.clone_ref(py)).collect()
    }

    fn articulation_indices(&self) -> Vec<usize> {
        (0..self.topo.n).filter(|&i| self.topo.in_art[i]).collect()
    }

    fn lowest_articulation_indices(&self) -> Vec<usize> {
        (0..self.topo.n).filter(|&i| self.topo.in_low_art[i]).collect()
    }

    /// ``(length, gamma)`` of the edge from parent ``p`` to node ``i``.
    fn edge_info(&self, i: usize, p: usize) -> Option<(Option<f64>, f64)> {
        let j = self.topo.parents[i].iter().position(|&x| x == p)?;
        Some((self.topo.pa_bl[i][j], self.topo.pa_gamma[i][j]))
    }

    /// Re-read branch lengths and gammas from ``network`` (same topology).
    fn refresh_params(&mut self, py: Python<'_>, network: &Bound<'_, Network>) -> PyResult<()> {
        let net = network.borrow();
        let uids = self.uids.clone();
        self.topo.read_params(py, &net, &uids)?;
        self.ws.gij_cache.clear();
        Ok(())
    }

    /// Translate species triplets into leaf indices, dropping triplets
    /// whose taxa are not all leaves of this network.
    fn build_triplet_index(
        &self,
        triplets: Vec<(String, String, String)>,
        rho: &Bound<'_, PyDict>,
    ) -> PyResult<TripletIndex> {
        let mut trips = Vec::with_capacity(triplets.len());
        let mut rhos = Vec::with_capacity(triplets.len());
        for t in triplets {
            let (Some(&x), Some(&y), Some(&z)) = (
                self.topo.label_to_idx.get(&t.0),
                self.topo.label_to_idx.get(&t.1),
                self.topo.label_to_idx.get(&t.2),
            ) else {
                continue;
            };
            let r: (f64, f64, f64) = match rho.get_item(&t)? {
                Some(v) => v.extract()?,
                None => continue,
            };
            trips.push([x, y, z]);
            rhos.push([r.0, r.1, r.2]);
        }
        Ok(TripletIndex { trips, rho: rhos })
    }

    /// Log pseudo-likelihood over an index built by ``build_triplet_index``.
    fn score(&mut self, index: &TripletIndex) -> f64 {
        score_indexed(&self.topo, &mut self.ws, &index.trips, &index.rho)
    }

    /// ``P(xy|z)`` for labels ``(x, y, z)``; 0 when a label is not a leaf.
    fn triple_probability(&mut self, x: &str, y: &str, z: &str) -> f64 {
        let (Some(&a), Some(&b), Some(&c)) = (
            self.topo.label_to_idx.get(x),
            self.topo.label_to_idx.get(y),
            self.topo.label_to_idx.get(z),
        ) else {
            return 0.0;
        };
        calc_triple(&self.topo, &mut self.ws, [a, b, c])
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Gene-tree triplet frequencies (rho)
// ═══════════════════════════════════════════════════════════════════════

/// Pairwise LCA depths of the relevant leaves of one gene tree.
struct LcaIndex {
    /// label id -> position in `present`, or usize::MAX when absent
    pos: Vec<usize>,
    depth: Vec<i32>, // dense L x L (L = present.len())
    l: usize,
}

impl LcaIndex {
    fn build(py: Python<'_>, net: &Network, label_ids: &FxHashMap<String, usize>, n_labels: usize) -> PyResult<LcaIndex> {
        let root = net.root_uid(py)?;
        let (order, dist) = net.bfs_order(root, false);
        let mut depth_of: FxHashMap<u64, i32> = FxHashMap::default();
        let mut parent_of: FxHashMap<u64, u64> = FxHashMap::default();
        // Reconstruct BFS parents: the reference records the first parent
        // along BFS discovery.
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((root, None::<u64>));
        let mut seen: FxHashSet<u64> = FxHashSet::default();
        while let Some((node, par)) = queue.pop_front() {
            // The reference overwrites depth/parent on every visit (no
            // visited guard), so the last BFS visit wins; trees have one.
            depth_of.insert(node, dist.get(&node).copied().unwrap_or(0) as i32);
            if let Some(p) = par {
                parent_of.insert(node, p);
            }
            if !seen.insert(node) {
                continue;
            }
            for c in net.children(node) {
                queue.push_back((c, Some(node)));
            }
        }
        let _ = order;
        // Relevant leaves present in this tree.
        let mut present: Vec<(usize, u64)> = Vec::new();
        for &leaf in &net.leaf_uids() {
            let name = net.label(py, leaf);
            if let Some(&id) = label_ids.get(&name) {
                present.push((id, leaf));
            }
        }
        present.sort_by_key(|(id, _)| *id);
        let l = present.len();
        let mut pos = vec![usize::MAX; n_labels];
        for (k, (id, _)) in present.iter().enumerate() {
            pos[*id] = k;
        }
        let mut depth = vec![0i32; l * l];
        for i in 0..l {
            let (_, na0) = present[i];
            let da0 = depth_of[&na0];
            for j in (i + 1)..l {
                let (_, nb0) = present[j];
                let (mut na, mut nb) = (na0, nb0);
                let (mut da, mut db) = (da0, depth_of[&nb0]);
                while da > db {
                    na = parent_of[&na];
                    da -= 1;
                }
                while db > da {
                    nb = parent_of[&nb];
                    db -= 1;
                }
                while na != nb {
                    na = parent_of[&na];
                    nb = parent_of[&nb];
                    da -= 1;
                }
                depth[i * l + j] = da;
                depth[j * l + i] = da;
            }
        }
        Ok(LcaIndex { pos, depth, l })
    }

    #[inline]
    fn pair(&self, a: usize, b: usize) -> i32 {
        self.depth[a * self.l + b]
    }
}

fn accumulate(
    idx: &LcaIndex,
    mapping: &[Vec<usize>],
    triplets: &[[usize; 3]],
    identity: bool,
    out: &mut [[f64; 3]],
) {
    let third = 1.0 / 3.0;
    for (t, acc) in triplets.iter().zip(out.iter_mut()) {
        if identity {
            let (x, y, z) = (mapping[t[0]][0], mapping[t[1]][0], mapping[t[2]][0]);
            let (px, py_, pz) = (idx.pos[x], idx.pos[y], idx.pos[z]);
            if px == usize::MAX || py_ == usize::MAX || pz == usize::MAX {
                continue;
            }
            let d_xy = idx.pair(px, py_);
            let d_xz = idx.pair(px, pz);
            let d_yz = idx.pair(py_, pz);
            if d_xy == d_xz && d_xz == d_yz {
                acc[0] += third;
                acc[1] += third;
                acc[2] += third;
            } else if d_xy >= d_xz && d_xy >= d_yz {
                acc[0] += 1.0;
            } else if d_xz >= d_yz {
                acc[1] += 1.0;
            } else {
                acc[2] += 1.0;
            }
        } else {
            let ax: Vec<usize> = mapping[t[0]].iter().map(|&a| idx.pos[a]).filter(|&p| p != usize::MAX).collect();
            let ay: Vec<usize> = mapping[t[1]].iter().map(|&a| idx.pos[a]).filter(|&p| p != usize::MAX).collect();
            let az: Vec<usize> = mapping[t[2]].iter().map(|&a| idx.pos[a]).filter(|&p| p != usize::MAX).collect();
            let denom = ax.len() * ay.len() * az.len();
            if denom == 0 {
                continue;
            }
            let (mut cxy, mut cxz, mut cyz) = (0.0, 0.0, 0.0);
            for &xi in &ax {
                for &yj in &ay {
                    for &zk in &az {
                        let d_xy = idx.pair(xi, yj);
                        let d_xz = idx.pair(xi, zk);
                        let d_yz = idx.pair(yj, zk);
                        if d_xy == d_xz && d_xz == d_yz {
                            cxy += third;
                            cxz += third;
                            cyz += third;
                        } else {
                            let max_d = d_xy.max(d_xz).max(d_yz);
                            if d_xy == max_d {
                                cxy += 1.0;
                            } else if d_xz == max_d {
                                cxz += 1.0;
                            } else {
                                cyz += 1.0;
                            }
                        }
                    }
                }
            }
            let inv = 1.0 / denom as f64;
            acc[0] += cxy * inv;
            acc[1] += cxz * inv;
            acc[2] += cyz * inv;
        }
    }
}

/// Rho table: for each species triplet ``(rho_XY|Z, rho_XZ|Y, rho_YZ|X)``.
#[pyfunction]
#[pyo3(name = "_compute_rhos")]
#[pyo3(signature = (gene_trees, mapping, triplets, n_workers=1))]
pub fn compute_rhos_py(
    py: Python<'_>,
    gene_trees: Vec<Bound<'_, Network>>,
    mapping: Vec<(String, Vec<String>)>,
    triplets: Vec<(String, String, String)>,
    n_workers: usize,
) -> PyResult<Vec<(f64, f64, f64)>> {
    // Intern allele labels and species.
    let mut label_ids: FxHashMap<String, usize> = FxHashMap::default();
    let mut species_ids: FxHashMap<String, usize> = FxHashMap::default();
    let mut map_vec: Vec<Vec<usize>> = Vec::with_capacity(mapping.len());
    for (sp, alleles) in &mapping {
        species_ids.insert(sp.clone(), map_vec.len());
        let ids = alleles
            .iter()
            .map(|a| {
                let next = label_ids.len();
                *label_ids.entry(a.clone()).or_insert(next)
            })
            .collect();
        map_vec.push(ids);
    }
    let identity = map_vec.iter().all(|v| v.len() == 1);
    let n_labels = label_ids.len();
    let trips: Vec<[usize; 3]> = triplets
        .iter()
        .map(|(x, y, z)| -> PyResult<[usize; 3]> {
            let get = |s: &String| {
                species_ids.get(s).copied().ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(s.clone())
                })
            };
            Ok([get(x)?, get(y)?, get(z)?])
        })
        .collect::<PyResult<_>>()?;

    // Index every tree (needs the GIL), then accumulate (does not).
    let mut indices = Vec::with_capacity(gene_trees.len());
    for t in &gene_trees {
        indices.push(LcaIndex::build(py, &t.borrow(), &label_ids, n_labels)?);
    }
    let workers = n_workers.max(1).min(indices.len().max(1));
    let mut totals = vec![[0.0f64; 3]; trips.len()];
    if workers <= 1 {
        py.detach(|| {
            for idx in &indices {
                accumulate(idx, &map_vec, &trips, identity, &mut totals);
            }
        });
    } else {
        let chunk = indices.len().div_ceil(workers);
        let partials: Vec<Vec<[f64; 3]>> = py.detach(|| {
            std::thread::scope(|s| {
                let handles: Vec<_> = indices
                    .chunks(chunk)
                    .map(|c| {
                        let map_vec = &map_vec;
                        let trips = &trips;
                        s.spawn(move || {
                            let mut part = vec![[0.0f64; 3]; trips.len()];
                            for idx in c {
                                accumulate(idx, map_vec, trips, identity, &mut part);
                            }
                            part
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            })
        });
        for part in partials {
            for (t, p) in totals.iter_mut().zip(part.iter()) {
                t[0] += p[0];
                t[1] += p[1];
                t[2] += p[2];
            }
        }
    }
    Ok(totals.into_iter().map(|t| (t[0], t[1], t[2])).collect())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MplTopology>()?;
    m.add_class::<TripletIndex>()?;
    m.add_function(wrap_pyfunction!(compute_rhos_py, m)?)?;
    Ok(())
}
