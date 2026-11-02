//! ``NodeSet`` and ``EdgeSet``: the standalone adjacency containers kept for
//! API compatibility.  A ``Network`` keeps its own (faster, identity-keyed)
//! bookkeeping, but accepts these as constructor inputs.

use indexmap::IndexMap;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PySet, PyTuple};
use rustc_hash::FxHashMap;

use super::edge::Edge;
use super::node::{flatten_nodes, Node};
use crate::util::{type_name, warn};

type EdgeMap = IndexMap<u64, Py<Edge>>;

#[pyclass(name = "NodeSet", module = "phynetpy.Network")]
#[derive(Default)]
pub struct NodeSet {
    nodes: IndexMap<String, Py<Node>>,
    in_map: FxHashMap<String, EdgeMap>,
    out_map: FxHashMap<String, EdgeMap>,
}

impl NodeSet {
    pub fn node_objects(&self, py: Python<'_>) -> Vec<Py<Node>> {
        self.nodes.values().map(|n| n.clone_ref(py)).collect()
    }

    fn name_of(obj: &Bound<'_, PyAny>) -> Option<String> {
        obj.cast::<Node>().ok().map(|n| n.borrow().name.clone())
    }

    fn edge_endpoints(py: Python<'_>, edge: &Bound<'_, PyAny>) -> Option<(String, String)> {
        let e = edge.cast::<Edge>().ok()?.borrow();
        let s = e.src.borrow(py).name.clone();
        let d = e.dest.borrow(py).name.clone();
        Some((s, d))
    }
}

#[pymethods]
impl NodeSet {
    #[new]
    fn py_new() -> Self {
        NodeSet::default()
    }

    fn __contains__(&self, n: &Bound<'_, PyAny>) -> bool {
        match Self::name_of(n) {
            Some(name) => self.nodes.contains_key(&name),
            None => false,
        }
    }

    #[pyo3(signature = (*nodes))]
    fn add(&mut self, nodes: &Bound<'_, PyTuple>) -> PyResult<()> {
        for node in flatten_nodes(nodes)? {
            let name = node.borrow().name.clone();
            if !self.nodes.contains_key(&name) {
                self.nodes.insert(name, node.unbind());
            }
        }
        Ok(())
    }

    fn ready(&self, py: Python<'_>, edge: &Bound<'_, PyAny>) -> bool {
        if !edge.is_exact_instance_of::<Edge>() {
            return false;
        }
        match Self::edge_endpoints(py, edge) {
            Some((s, d)) => self.nodes.contains_key(&s) && self.nodes.contains_key(&d),
            None => false,
        }
    }

    fn in_deg(&self, node: &Bound<'_, PyAny>) -> usize {
        Self::name_of(node)
            .and_then(|n| self.in_map.get(&n))
            .map(|m| m.len())
            .unwrap_or(0)
    }

    fn out_deg(&self, node: &Bound<'_, PyAny>) -> usize {
        Self::name_of(node)
            .and_then(|n| self.out_map.get(&n))
            .map(|m| m.len())
            .unwrap_or(0)
    }

    fn in_edges<'py>(&self, py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PySet>> {
        let items = Self::name_of(node)
            .and_then(|n| self.in_map.get(&n))
            .map(|m| m.values().map(|e| e.clone_ref(py)).collect::<Vec<_>>())
            .unwrap_or_default();
        PySet::new(py, items)
    }

    fn out_edges<'py>(&self, py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PySet>> {
        let items = Self::name_of(node)
            .and_then(|n| self.out_map.get(&n))
            .map(|m| m.values().map(|e| e.clone_ref(py)).collect::<Vec<_>>())
            .unwrap_or_default();
        PySet::new(py, items)
    }

    #[pyo3(signature = (edge, removal=false))]
    fn process(&mut self, py: Python<'_>, edge: &Bound<'_, PyAny>, removal: bool) -> PyResult<()> {
        if !edge.is_exact_instance_of::<Edge>() {
            return Err(PyTypeError::new_err("Tried to process wrong type of edge!"));
        }
        if !self.ready(py, edge) {
            return Err(PyValueError::new_err(
                "Edge contains node not in network. Add node first!",
            ));
        }
        let (s, d) = Self::edge_endpoints(py, edge).unwrap();
        let e = edge.cast::<Edge>()?;
        let uid = e.borrow().uid;
        if !removal {
            self.out_map.entry(s).or_default().insert(uid, e.clone().unbind());
            self.in_map.entry(d).or_default().insert(uid, e.clone().unbind());
        } else {
            if let Some(m) = self.out_map.get_mut(&s) {
                m.swap_remove(&uid);
            }
            if let Some(m) = self.in_map.get_mut(&d) {
                m.swap_remove(&uid);
            }
        }
        Ok(())
    }

    fn get_set<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        PySet::new(py, self.nodes.values().map(|n| n.bind(py)))
    }

    fn remove(&mut self, node: &Bound<'_, PyAny>) {
        if let Some(name) = Self::name_of(node) {
            self.nodes.swap_remove(&name);
            self.in_map.remove(&name);
            self.out_map.remove(&name);
        }
    }

    fn update(&mut self, node: &Bound<'_, Node>, new_name: String) -> PyResult<()> {
        let old = node.borrow().name.clone();
        if !self.nodes.contains_key(&old) {
            return Err(PyValueError::new_err(format!(
                "Node {old} not found in NodeSet"
            )));
        }
        let obj = self.nodes.swap_remove(&old).unwrap();
        let ins = self.in_map.remove(&old).unwrap_or_default();
        let outs = self.out_map.remove(&old).unwrap_or_default();
        node.borrow_mut().set_name_inner(new_name.clone());
        self.nodes.insert(new_name.clone(), obj);
        self.in_map.insert(new_name.clone(), ins);
        self.out_map.insert(new_name, outs);
        Ok(())
    }

    fn get(&self, py: Python<'_>, name: &str) -> Option<Py<Node>> {
        self.nodes.get(name).map(|n| n.clone_ref(py))
    }
}

#[pyclass(name = "EdgeSet", module = "phynetpy.Network")]
#[derive(Default)]
pub struct EdgeSet {
    edges: IndexMap<u64, Py<Edge>>,
    pair: FxHashMap<(String, String), Vec<u64>>,
}

impl EdgeSet {
    pub fn edge_objects(&self, py: Python<'_>) -> Vec<Py<Edge>> {
        self.edges.values().map(|e| e.clone_ref(py)).collect()
    }

    fn key(py: Python<'_>, e: &Edge) -> (String, String) {
        (e.src.borrow(py).name.clone(), e.dest.borrow(py).name.clone())
    }
}

/// Pick one edge out of a (possibly bubbled) ``(src, dest)`` bucket, the
/// way ``EdgeSet.get`` does: by gamma, then by tag.
pub fn pick_edge<'py>(
    py: Python<'py>,
    candidates: &[&Py<Edge>],
    gamma: Option<f64>,
    tag: Option<&str>,
) -> PyResult<Py<Edge>> {
    match candidates.len() {
        0 => Err(PyValueError::new_err("No matching edges found")),
        1 => Ok(candidates[0].clone_ref(py)),
        2 => {
            let Some(g) = gamma else {
                warn(py, "Bubble lookup without gamma - returning first edge")?;
                return Ok(candidates[0].clone_ref(py));
            };
            let g0 = candidates[0].borrow(py).gamma;
            let g1 = candidates[1].borrow(py).gamma;
            if g0 == Some(g) && g1 == Some(g) {
                let t0 = candidates[0].borrow(py).tag.clone();
                let t1 = candidates[1].borrow(py).tag.clone();
                let tag = tag.map(|s| s.to_string());
                if t0 == tag {
                    Ok(candidates[0].clone_ref(py))
                } else if t1 == tag {
                    Ok(candidates[1].clone_ref(py))
                } else {
                    Err(PyValueError::new_err(format!(
                        "Tags don't match: {}",
                        tag.unwrap_or_else(|| "None".into())
                    )))
                }
            } else if g0 == Some(g) {
                Ok(candidates[0].clone_ref(py))
            } else if g1 == Some(g) {
                Ok(candidates[1].clone_ref(py))
            } else {
                Err(PyValueError::new_err("Gamma doesn't match any edge"))
            }
        }
        _ => Err(PyValueError::new_err(
            "More than 2 edges found - invalid topology",
        )),
    }
}

#[pymethods]
impl EdgeSet {
    #[new]
    fn py_new() -> Self {
        EdgeSet::default()
    }

    fn __contains__(&self, e: &Bound<'_, PyAny>) -> bool {
        match e.cast::<Edge>() {
            Ok(e) => self.edges.contains_key(&e.borrow().uid),
            Err(_) => false,
        }
    }

    #[pyo3(signature = (*edges))]
    fn add(&mut self, py: Python<'_>, edges: &Bound<'_, PyTuple>) -> PyResult<()> {
        for item in edges.iter() {
            let Ok(e) = item.cast::<Edge>() else {
                return Err(PyTypeError::new_err(format!(
                    "Networks are directed and hold only Edge objects. Got a {}.",
                    type_name(&item)
                )));
            };
            let (uid, key) = {
                let b = e.borrow();
                (b.uid, Self::key(py, &b))
            };
            if self.edges.contains_key(&uid) {
                continue;
            }
            self.pair.entry(key).or_default().push(uid);
            self.edges.insert(uid, e.clone().unbind());
        }
        Ok(())
    }

    fn remove(&mut self, py: Python<'_>, edge: &Bound<'_, PyAny>) {
        let Ok(e) = edge.cast::<Edge>() else { return };
        let (uid, key) = {
            let b = e.borrow();
            (b.uid, Self::key(py, &b))
        };
        if self.edges.swap_remove(&uid).is_some() {
            if let Some(v) = self.pair.get_mut(&key) {
                v.retain(|&x| x != uid);
                if v.is_empty() {
                    self.pair.remove(&key);
                }
            }
        }
    }

    #[pyo3(signature = (node, affected_edges=None))]
    fn rehash_node(
        &mut self,
        py: Python<'_>,
        node: &Bound<'_, PyAny>,
        affected_edges: Option<&Bound<'_, PyAny>>,
    ) {
        let _ = (node, affected_edges);
        let mut rebuilt: FxHashMap<(String, String), Vec<u64>> = FxHashMap::default();
        for (uid, e) in self.edges.iter() {
            rebuilt
                .entry(Self::key(py, &e.borrow(py)))
                .or_default()
                .push(*uid);
        }
        self.pair = rebuilt;
    }

    #[pyo3(signature = (n1, n2, gamma=None, tag=None))]
    fn get(
        &self,
        py: Python<'_>,
        n1: &Bound<'_, Node>,
        n2: &Bound<'_, Node>,
        gamma: Option<f64>,
        tag: Option<String>,
    ) -> PyResult<Py<Edge>> {
        let key = (n1.borrow().name.clone(), n2.borrow().name.clone());
        let Some(uids) = self.pair.get(&key) else {
            return Err(PyValueError::new_err("No matching edges found"));
        };
        let cands: Vec<&Py<Edge>> = uids.iter().filter_map(|u| self.edges.get(u)).collect();
        pick_edge(py, &cands, gamma, tag.as_deref())
    }

    fn get_set<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        PySet::new(py, self.edges.values().map(|e| e.bind(py)))
    }
}
