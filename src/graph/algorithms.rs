//! Graph algorithms over a ``Network``, on node and edge identities.
//! The Python-facing wrappers live in ``network.rs``.

use std::collections::VecDeque;

use indexmap::IndexMap;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use super::network::Network;
use crate::util::{network_error, node_error, py_float_repr};

impl Network {
    /// ``subgenome_count``: the number of ancestral lineages reaching ``uid``
    /// (memoized DP up the DAG).
    pub fn subgenome_count_uid(&self, py: Python<'_>, uid: u64) -> PyResult<usize> {
        let root = self.root_uid(py)?;
        let mut memo: FxHashMap<u64, usize> = FxHashMap::default();
        fn count(net: &Network, uid: u64, root: u64, memo: &mut FxHashMap<u64, usize>) -> usize {
            if let Some(c) = memo.get(&uid) {
                return *c;
            }
            let r = if uid == root {
                1
            } else {
                net.parents(uid).into_iter().map(|p| count(net, p, root, memo)).sum()
            };
            memo.insert(uid, r);
            r
        }
        Ok(count(self, uid, root, &mut memo))
    }

    /// Edge ids reachable forward from ``start`` (DFS order).
    pub fn edges_downstream(&self, start: u64) -> Vec<u64> {
        let mut stack = vec![start];
        let mut visited: FxHashSet<u64> = FxHashSet::default();
        let mut out = Vec::new();
        while let Some(cur) = stack.pop() {
            if !visited.insert(cur) {
                continue;
            }
            for e in self.out_edge_uids(cur) {
                out.push(e);
                stack.push(self.edges[&e].du);
            }
        }
        out
    }

    /// Edge ids on paths from the roots down to ``start`` (DFS order).
    pub fn edges_upstream(&self, start: u64) -> Vec<u64> {
        let mut stack = vec![start];
        let mut visited: FxHashSet<u64> = FxHashSet::default();
        let mut out = Vec::new();
        while let Some(cur) = stack.pop() {
            if !visited.insert(cur) {
                continue;
            }
            for e in self.in_edge_uids(cur) {
                out.push(e);
                stack.push(self.edges[&e].su);
            }
        }
        out
    }

    /// Leaves (out-degree 0) reachable from ``start`` (BFS order).
    pub fn leaf_descendants_uid(&self, start: u64) -> Vec<u64> {
        let mut q = VecDeque::new();
        q.push_back(start);
        let mut visited: FxHashSet<u64> = FxHashSet::default();
        let mut leaves = Vec::new();
        while let Some(cur) = q.pop_front() {
            if !visited.insert(cur) {
                continue;
            }
            if self.out_deg(cur) == 0 {
                leaves.push(cur);
            }
            for c in self.children(cur) {
                q.push_back(c);
            }
        }
        leaves
    }

    /// ``leaf_descendants_all``: every node reachable from the root mapped to
    /// its leaf descendants (memoized DFS, post-order insertion).
    pub fn leaf_descendants_all_map(
        &self,
        py: Python<'_>,
    ) -> PyResult<IndexMap<u64, Vec<u64>, FxBuildHasher>> {
        let root = self.root_uid(py)?;
        let leaves: FxHashSet<u64> = self.leaf_uids().into_iter().collect();
        let mut memo: IndexMap<u64, Vec<u64>, FxBuildHasher> = IndexMap::default();
        fn rec(
            net: &Network,
            node: u64,
            leaves: &FxHashSet<u64>,
            memo: &mut IndexMap<u64, Vec<u64>, FxBuildHasher>,
        ) {
            if memo.contains_key(&node) {
                return;
            }
            if leaves.contains(&node) {
                memo.insert(node, vec![node]);
                return;
            }
            let mut acc: Vec<u64> = Vec::new();
            let mut seen: FxHashSet<u64> = FxHashSet::default();
            for c in net.children(node) {
                rec(net, c, leaves, memo);
                for &l in &memo[&c] {
                    if seen.insert(l) {
                        acc.push(l);
                    }
                }
            }
            memo.insert(node, acc);
        }
        rec(self, root, &leaves, &mut memo);
        Ok(memo)
    }

    /// Extended-Newick string of the network.
    pub fn newick_string(&self, py: Python<'_>, annotate: bool) -> PyResult<String> {
        let root = self.root_uid(py)?;
        let leaf_set: FxHashSet<u64> = self.leaf_uids().into_iter().collect();
        let mut processed: FxHashSet<u64> = FxHashSet::default();
        let mut out = String::new();
        self.newick_help(py, root, &mut processed, &leaf_set, annotate, &mut out)?;
        out.push(';');
        Ok(out)
    }

    fn newick_help(
        &self,
        py: Python<'_>,
        node: u64,
        processed: &mut FxHashSet<u64>,
        leaf_set: &FxHashSet<u64>,
        annotate: bool,
        out: &mut String,
    ) -> PyResult<()> {
        let label = self.label(py, node);
        if leaf_set.contains(&node) {
            out.push_str(&label);
            return Ok(());
        }
        let hashed = |l: &str| -> String {
            if l.starts_with('#') {
                l.to_string()
            } else {
                format!("#{l}")
            }
        };
        let in_deg = self.in_deg(node);
        if in_deg >= 2 && processed.contains(&node) {
            out.push_str(&hashed(&label));
            return Ok(());
        }
        let node_name = if in_deg >= 2 {
            processed.insert(node);
            hashed(&label)
        } else {
            label
        };
        out.push('(');
        let children = self.children(node);
        let mut first = true;
        for child in children {
            if !first {
                out.push(',');
            }
            first = false;
            let first_visit = !processed.contains(&child);
            self.newick_help(py, child, processed, leaf_set, annotate, out)?;
            if annotate {
                let euid = self.first_edge_between(py, node, child)?;
                let (gamma, length) = {
                    let e = self.edges[&euid].obj.borrow(py);
                    (e.gamma, e.length)
                };
                if first_visit && self.in_deg(child) >= 2 {
                    if let Some(g) = gamma {
                        out.push_str(&format!("[&gamma={}]", py_float_repr(py, g)?));
                    }
                }
                if let Some(l) = length {
                    out.push_str(&format!(":{}", py_float_repr(py, l)?));
                }
            }
        }
        out.push(')');
        out.push_str(&node_name);
        Ok(())
    }

    /// Least common ancestor: the common ancestor minimising the summed
    /// upward hop distance to every target.
    pub fn mrca_uids(&self, _py: Python<'_>, targets: &[u64]) -> PyResult<u64> {
        // Every target: ancestor -> hop distance.
        let mut per_target: Vec<FxHashMap<u64, usize>> = Vec::with_capacity(targets.len());
        for &leaf in targets {
            let mut lvl: FxHashMap<u64, usize> = FxHashMap::default();
            let mut visited: FxHashSet<u64> = FxHashSet::default();
            let mut q = VecDeque::new();
            q.push_back(leaf);
            lvl.insert(leaf, 0);
            while let Some(cur) = q.pop_front() {
                let d = lvl[&cur];
                for p in self.parents(cur) {
                    if !visited.contains(&p) {
                        lvl.insert(p, d + 1);
                        q.push_back(p);
                        visited.insert(p);
                    }
                }
            }
            per_target.push(lvl);
        }
        let mut best: Option<(usize, u64)> = None;
        for uid in self.nodes.keys() {
            if !per_target.iter().all(|m| m.contains_key(uid)) {
                continue;
            }
            let total: usize = per_target.iter().map(|m| m[uid]).sum();
            if best.map(|(b, _)| total < b).unwrap_or(true) {
                best = Some((total, *uid));
            }
        }
        match best {
            Some((_, uid)) => Ok(uid),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "min() arg is an empty sequence",
            )),
        }
    }

    /// True when every component reachable from a root is acyclic.
    pub fn is_acyclic_impl(&self) -> bool {
        #[derive(Clone, Copy, PartialEq)]
        enum St {
            White,
            Gray,
            Black,
        }
        let mut state: FxHashMap<u64, St> =
            self.nodes.keys().map(|&u| (u, St::White)).collect();
        for root in self.root_uids() {
            if state[&root] != St::White {
                continue;
            }
            // Iterative DFS with an explicit (node, child index) stack.
            let mut stack: Vec<(u64, Vec<u64>, usize)> = vec![(root, self.children(root), 0)];
            state.insert(root, St::Gray);
            while let Some((node, children, idx)) = stack.last_mut() {
                if *idx < children.len() {
                    let c = children[*idx];
                    *idx += 1;
                    match state[&c] {
                        St::White => {
                            state.insert(c, St::Gray);
                            let cc = self.children(c);
                            stack.push((c, cc, 0));
                        }
                        St::Gray => return false,
                        St::Black => {}
                    }
                } else {
                    state.insert(*node, St::Black);
                    stack.pop();
                }
            }
        }
        true
    }

    /// General BFS/DFS from ``start``: the visit order and the hop distances
    /// in discovery order.
    pub fn bfs_order(&self, start: u64, dfs: bool) -> (Vec<u64>, IndexMap<u64, i64, FxBuildHasher>) {
        let mut q: VecDeque<u64> = VecDeque::new();
        q.push_back(start);
        let mut dist: IndexMap<u64, i64, FxBuildHasher> = IndexMap::default();
        dist.insert(start, 0);
        let mut visited: FxHashSet<u64> = FxHashSet::default();
        visited.insert(start);
        let mut order = Vec::new();
        while let Some(cur) = if dfs { q.pop_front() } else { q.pop_back() } {
            order.push(cur);
            let d = dist[&cur];
            for nb in self.children(cur) {
                if visited.contains(&nb) {
                    continue;
                }
                dist.insert(nb, d + 1);
                q.push_front(nb);
                visited.insert(nb);
            }
        }
        (order, dist)
    }

    /// All nodes reachable from ``start`` (inclusive), BFS order.
    pub fn subtree_uids(&self, start: u64) -> Vec<u64> {
        let mut seen: FxHashSet<u64> = FxHashSet::default();
        let mut out = Vec::new();
        let mut q = VecDeque::new();
        q.push_back(start);
        while let Some(cur) = q.pop_front() {
            if !seen.insert(cur) {
                continue;
            }
            out.push(cur);
            for c in self.children(cur) {
                q.push_back(c);
            }
        }
        out
    }

    /// Kahn's algorithm; NetworkError on a cycle.
    pub fn topological_order_uids(&self, py: Python<'_>) -> PyResult<Vec<u64>> {
        let mut indeg: FxHashMap<u64, usize> =
            self.nodes.iter().map(|(u, e)| (*u, e.in_edges.len())).collect();
        let mut q: VecDeque<u64> = self
            .nodes
            .keys()
            .filter(|u| indeg[u] == 0)
            .copied()
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(u) = q.pop_front() {
            order.push(u);
            for e in self.out_edge_uids(u) {
                let v = self.edges[&e].du;
                let d = indeg.get_mut(&v).unwrap();
                *d -= 1;
                if *d == 0 {
                    q.push_back(v);
                }
            }
        }
        if order.len() != self.nodes.len() {
            return Err(network_error(
                py,
                "Graph has cycles; no topological order exists",
            ));
        }
        Ok(order)
    }

    fn edge_length_or_type_error(&self, py: Python<'_>, euid: u64) -> PyResult<f64> {
        self.edges[&euid].obj.borrow(py).length.ok_or_else(|| {
            PyTypeError::new_err("unsupported operand type(s) for +: 'float' and 'NoneType'")
        })
    }

    /// ``distance_from_root``.
    pub fn distance_from_root_impl(&self, py: Python<'_>, node: u64, use_time: bool) -> PyResult<f64> {
        if use_time {
            // Mirrors the reference: node.get_time() raises when unset.
            let t = self.nodes[&node]
                .obj
                .borrow(py)
                .t
                .ok_or_else(|| node_error(py, "No time has been set for this node!"))?;
            if let Ok(root) = self.root_uid(py) {
                if let Some(rt) = self.nodes[&root].obj.borrow(py).t {
                    return Ok(t - rt);
                }
            }
        }
        let root = self.root_uid(py)?;
        if root == node {
            return Ok(0.0);
        }
        let mut visited: FxHashSet<u64> = FxHashSet::default();
        visited.insert(root);
        let mut q: VecDeque<(u64, f64)> = VecDeque::new();
        q.push_back((root, 0.0));
        while let Some((cur, dist)) = q.pop_front() {
            for child in self.children(cur) {
                if child == node {
                    return if use_time {
                        let e = self.first_edge_between(py, cur, child)?;
                        Ok(dist + self.edge_length_or_type_error(py, e)?)
                    } else {
                        Ok(dist + 1.0)
                    };
                }
                if !visited.contains(&child) {
                    visited.insert(child);
                    let nd = if use_time {
                        let e = self.first_edge_between(py, cur, child)?;
                        dist + self.edge_length_or_type_error(py, e)?
                    } else {
                        dist + 1.0
                    };
                    q.push_back((child, nd));
                }
            }
        }
        Err(network_error(
            py,
            format!("No path found from root to node {}", self.label(py, node)),
        ))
    }

    /// Node times from cumulative branch lengths (``by_edges``: hop counts).
    pub fn set_times_impl(&self, py: Python<'_>, by_edges: bool) -> PyResult<()> {
        let Ok(root) = self.root_uid(py) else {
            return Ok(());
        };
        super::node::Node::set_time(self.nodes[&root].obj.bind(py), 0.0)?;
        let mut visited: FxHashSet<u64> = FxHashSet::default();
        visited.insert(root);
        let mut q: VecDeque<(u64, f64)> = VecDeque::new();
        q.push_back((root, 0.0));
        while let Some((cur, t)) = q.pop_front() {
            for child in self.children(cur) {
                if visited.insert(child) {
                    let ct = if by_edges {
                        t + 1.0
                    } else {
                        let e = self.first_edge_between(py, cur, child)?;
                        t + self.edge_length_or_type_error(py, e)?
                    };
                    super::node::Node::set_time(self.nodes[&child].obj.bind(py), ct)?;
                    q.push_back((child, ct));
                }
            }
        }
        Ok(())
    }

    /// Canonical rooted-triplet encoding of every leaf triple.
    pub fn rooted_triplets(&self, py: Python<'_>) -> PyResult<FxHashSet<(String, String, String)>> {
        let leaves = self.leaf_uids();
        let labels: Vec<String> = leaves.iter().map(|&u| self.label(py, u)).collect();
        let mut out: FxHashSet<(String, String, String)> = FxHashSet::default();
        let (_, dist) = match self.root_uid(py) {
            Ok(r) => self.bfs_order(r, false),
            Err(_) => return Ok(out),
        };
        let n = leaves.len();
        let enc = |a: &str, b: &str, c: &str| -> (String, String, String) {
            let (x, y) = if a <= b { (a, b) } else { (b, a) };
            (x.to_string(), y.to_string(), c.to_string())
        };
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let (x, y, z) = (leaves[i], leaves[j], leaves[k]);
                    let try_dist = |a: u64, b: u64| -> Option<i64> {
                        let m = self.mrca_uids(py, &[a, b]).ok()?;
                        dist.get(&m).copied()
                    };
                    let (Some(dxy), Some(dxz), Some(dyz)) =
                        (try_dist(x, y), try_dist(x, z), try_dist(y, z))
                    else {
                        continue;
                    };
                    let (lx, ly, lz) = (&labels[i], &labels[j], &labels[k]);
                    let t = if dxy >= dxz && dxy >= dyz {
                        enc(lx, ly, lz)
                    } else if dxz >= dxy && dxz >= dyz {
                        enc(lx, lz, ly)
                    } else {
                        enc(ly, lz, lx)
                    };
                    out.insert(t);
                }
            }
        }
        Ok(out)
    }

    pub fn triplet_distance_impl(&self, py: Python<'_>, other: &Network) -> PyResult<f64> {
        let a = self.rooted_triplets(py)?;
        let b = other.rooted_triplets(py)?;
        if a.is_empty() && b.is_empty() {
            return Ok(0.0);
        }
        if a.is_empty() || b.is_empty() {
            return Ok(1.0);
        }
        let common = a.intersection(&b).count();
        let total = a.union(&b).count();
        if total == 0 {
            return Ok(0.0);
        }
        Ok(1.0 - (common as f64) / (total as f64))
    }
}
