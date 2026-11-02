//! Yule (pure birth) and constant-rate birth-death network generators.
//!
//! The Yule generator draws from the caller's ``numpy.random.Generator`` in
//! the same order as the reference implementation, so a seeded run builds
//! the same tree.  The birth-death generator keeps the reference's use of
//! Python's global ``random`` module.

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use crate::graph::{Edge, Network, Node};

fn make_node(py: Python<'_>, name: &str, attrs: &[(&str, Bound<'_, PyAny>)], t: Option<f64>) -> PyResult<Py<Node>> {
    let d = PyDict::new(py);
    for (k, v) in attrs {
        d.set_item(*k, v)?;
    }
    Py::new(py, Node::make(py, name.to_string(), false, Some(d), None, t))
}

fn is_live(py: Python<'_>, node: &Py<Node>) -> bool {
    let n = node.borrow(py);
    match n.attrs.bind(py).get_item("live") {
        Ok(Some(v)) => v.cast::<PyBool>().map(|b| b.is_true()).unwrap_or(false),
        _ => false,
    }
}

fn default_edge(py: Python<'_>, src: &Py<Node>, dest: &Py<Node>) -> PyResult<Py<Edge>> {
    Py::new(
        py,
        Edge::make(
            py,
            src.clone_ref(py),
            dest.clone_ref(py),
            Some(1.0),
            Some(0.0),
            Some(0.0),
            Some("no tag assigned yet".to_string()),
        ),
    )
}

struct YuleRun<'py> {
    py: Python<'py>,
    rng: Bound<'py, PyAny>,
    gamma: f64,
    lin: usize,
    internal_count: usize,
    elapsed: f64,
}

impl<'py> YuleRun<'py> {
    fn live_nodes(&self, net: &Bound<'py, Network>) -> Vec<Py<Node>> {
        let n = net.borrow();
        n.nodes.values().map(|e| e.obj.clone_ref(self.py)).filter(|o| is_live(self.py, o)).collect()
    }

    fn draw_waiting_time(&self) -> PyResult<f64> {
        let scale = 1.0 / (self.lin as f64 * self.gamma);
        let u: f64 = self.rng.call_method0("random")?.extract()?;
        // scipy.stats.expon.ppf(u, scale=scale)
        Ok(-scale * (-u).ln_1p())
    }

    /// One speciation event; ``false`` when a time-conditioned run is over.
    fn event(&mut self, net: &Bound<'py, Network>, time_limit: Option<f64>) -> PyResult<bool> {
        let py = self.py;
        let mut live = self.live_nodes(net);
        live.sort_by(|a, b| a.borrow(py).name.cmp(&b.borrow(py).name));
        let idx: i64 = self.rng.call_method1("integers", (0usize, live.len()))?.extract()?;
        let spec_node = live[idx as usize].clone_ref(py);
        let (spec_uid, old_parent) = {
            let n = net.borrow();
            let su = n.uid_of(py, &spec_node).unwrap();
            let p = n.parents(su)[0];
            (su, (p, n.node_obj(p).clone_ref(py)))
        };
        let next_time = self.draw_waiting_time()?;
        let parent_time = old_parent.1.borrow(py).t.ok_or_else(|| {
            crate::util::node_error(py, "No time has been set for this node!")
        })?;
        let branch_len;
        match time_limit {
            None => {
                branch_len = self.elapsed + next_time - parent_time;
                self.elapsed += next_time;
            }
            Some(limit) => {
                if self.elapsed + next_time <= limit {
                    branch_len = self.elapsed + next_time - parent_time;
                    self.elapsed += next_time;
                } else {
                    return Ok(false);
                }
            }
        }
        let new_internal = make_node(
            py,
            &format!("internal{}", self.internal_count),
            &[("t", self.elapsed.into_pyobject(py)?.into_any()), ("live", PyBool::new(py, false).to_owned().into_any())],
            None,
        )?;
        Node::set_time(new_internal.bind(py), self.elapsed)?;
        self.internal_count += 1;
        self.lin += 1;
        let new_spec = make_node(
            py,
            &format!("spec{}", self.lin),
            &[("live", PyBool::new(py, true).to_owned().into_any())],
            None,
        )?;
        {
            let mut n = net.borrow_mut();
            n.add_node(net.as_any(), new_spec.bind(py))?;
            n.add_node(net.as_any(), new_internal.bind(py))?;
        }
        let e1 = default_edge(py, &new_internal, &new_spec)?;
        let e2 = default_edge(py, &new_internal, &spec_node)?;
        let e3 = default_edge(py, &old_parent.1, &new_internal)?;
        Edge::set_length(e3.bind(py), Some(branch_len), false, false)?;
        {
            let mut n = net.borrow_mut();
            let old_edge = n.first_edge_between(py, old_parent.0, spec_uid)?;
            n.remove_edge_uid(py, old_edge);
            n.add_edge(net.as_any(), e1.bind(py))?;
            n.add_edge(net.as_any(), e2.bind(py))?;
            n.add_edge(net.as_any(), e3.bind(py))?;
        }
        Ok(true)
    }

    fn finish_live(&self, net: &Bound<'py, Network>, final_time: f64, require_parent: bool) -> PyResult<()> {
        let py = self.py;
        for node in self.live_nodes(net) {
            node.borrow(py).attrs.bind(py).set_item("t", final_time)?;
            Node::set_time(node.bind(py), final_time)?;
            let (uid, parents) = {
                let n = net.borrow();
                let u = n.uid_of(py, &node).unwrap();
                (u, n.parents(u))
            };
            if parents.is_empty() {
                if require_parent {
                    return Err(PyIndexError::new_err("list index out of range"));
                }
                continue;
            }
            let p = parents[0];
            let (parent_time, euid) = {
                let n = net.borrow();
                let pt = n.node_obj(p).borrow(py).t.ok_or_else(|| {
                    crate::util::node_error(py, "No time has been set for this node!")
                })?;
                (pt, n.first_edge_between(py, p, uid)?)
            };
            let edge = net.borrow().edge_obj(euid).clone_ref(py);
            Edge::set_length(edge.bind(py), Some(final_time - parent_time), false, false)?;
        }
        Ok(())
    }
}

/// Build one Yule tree.  Returns ``(network, internal_count)``.
#[pyfunction]
#[pyo3(signature = (rng, gamma, condition, n, time, internal_count))]
fn yule_generate<'py>(
    py: Python<'py>,
    rng: Bound<'py, PyAny>,
    gamma: f64,
    condition: &str,
    n: usize,
    time: f64,
    internal_count: usize,
) -> PyResult<(Bound<'py, Network>, usize)> {
    let net = Network::new_object(py, None)?;
    let root = make_node(
        py,
        "root",
        &[
            ("t", 0i64.into_pyobject(py)?.into_any()),
            ("label", "root".into_pyobject(py)?.into_any()),
            ("live", PyBool::new(py, false).to_owned().into_any()),
        ],
        Some(0.0),
    )?;
    let s1 = make_node(py, "spec1", &[("live", PyBool::new(py, true).to_owned().into_any())], None)?;
    let s2 = make_node(py, "spec2", &[("live", PyBool::new(py, true).to_owned().into_any())], None)?;
    {
        let mut nb = net.borrow_mut();
        nb.add_node(net.as_any(), root.bind(py))?;
        nb.add_node(net.as_any(), s1.bind(py))?;
        nb.add_node(net.as_any(), s2.bind(py))?;
        let e1 = default_edge(py, &root, &s1)?;
        let e2 = default_edge(py, &root, &s2)?;
        nb.add_edge(net.as_any(), e1.bind(py))?;
        nb.add_edge(net.as_any(), e2.bind(py))?;
    }
    let mut run = YuleRun {
        py,
        rng,
        gamma,
        lin: 2,
        internal_count,
        elapsed: 0.0,
    };
    if condition == "N" {
        if n == 2 {
            Node::set_time(s1.bind(py), 1.0)?;
            Node::set_time(s2.bind(py), 1.0)?;
            return Ok((net, run.internal_count));
        }
        while run.live_nodes(&net).len() < n {
            run.event(&net, None)?;
        }
        let next_time = run.draw_waiting_time()?;
        let final_time = run.elapsed + next_time;
        run.finish_live(&net, final_time, false)?;
    } else {
        while run.elapsed < time {
            if !run.event(&net, Some(time))? {
                break;
            }
        }
        run.finish_live(&net, time, true)?;
    }
    Ok((net, run.internal_count))
}

/// Build one constant-rate birth-death tree with ``n`` live lineages.
#[pyfunction]
fn cbdp_generate<'py>(py: Python<'py>, gamma: f64, mu: f64, n: usize) -> PyResult<Bound<'py, Network>> {
    let random = py.import("random")?.getattr("random")?;
    let mut r = Vec::with_capacity(n);
    for _ in 0..n {
        r.push(random.call0()?.extract::<f64>()?);
    }
    let nf = n as f64;
    let qinv = |x: f64| -> f64 {
        let term1 = 1.0 / (gamma - mu);
        let term2 = 1.0 - (mu / gamma) * x.powf(1.0 / nf);
        let term3 = 1.0 - x.powf(1.0 / nf);
        term1 * (term2 / term3).ln()
    };
    let finv = |x: f64, t: f64| -> f64 {
        let rate_dif = gamma - mu;
        let term1 = 1.0 / rate_dif;
        let term2 = gamma - mu * (-t * rate_dif).exp();
        let term3 = x * (1.0 - (-t * rate_dif).exp());
        term1 * ((term2 - mu * term3) / (term2 - gamma * term3)).ln()
    };
    let t = qinv(r[0]);
    // ``s = {finv(r[i], t): i + .5}``: a dict, so equal keys collapse.
    let mut s_keys: Vec<f64> = Vec::new();
    for &ri in r.iter().skip(1) {
        let v = finv(ri, t);
        if !s_keys.contains(&v) {
            s_keys.push(v);
        }
    }
    let net = Network::new_object(py, None)?;
    let mut nodes: Vec<Py<Node>> = Vec::new();
    for j in 0..(2 * n - 1) {
        if j % 2 == 0 {
            let leaf = make_node(
                py,
                &format!("T{}", j / 2 + 1),
                &[("t", t.into_pyobject(py)?.into_any()), ("live", PyBool::new(py, true).to_owned().into_any())],
                None,
            )?;
            net.borrow_mut().add_node(net.as_any(), leaf.bind(py))?;
            Node::set_time(leaf.bind(py), t)?;
            nodes.push(leaf);
        } else {
            let k = (j - 1) / 2;
            let sk = *s_keys.get(k).ok_or_else(|| PyIndexError::new_err("list index out of range"))?;
            let internal = make_node(
                py,
                &format!("internal{k}"),
                &[("t", sk.into_pyobject(py)?.into_any()), ("live", PyBool::new(py, false).to_owned().into_any())],
                None,
            )?;
            net.borrow_mut().add_node(net.as_any(), internal.bind(py))?;
            Node::set_time(internal.bind(py), sk)?;
            nodes.push(internal);
        }
    }
    let times: Vec<f64> = nodes.iter().map(|o| o.borrow(py).t.unwrap_or(0.0)).collect();
    for index in 0..nodes.len() {
        let node_t = times[index];
        let mut right = None;
        let mut ri = index + 1;
        while ri < nodes.len() {
            if times[ri] < node_t {
                right = Some(ri);
                break;
            }
            ri += 1;
        }
        let mut left = None;
        let mut li = index as i64 - 1;
        while li >= 0 {
            if times[li as usize] < node_t {
                left = Some(li as usize);
                break;
            }
            li -= 1;
        }
        let selection = match (left, right) {
            (None, None) => continue,
            (None, Some(rr)) => rr,
            (Some(ll), None) => ll,
            (Some(ll), Some(rr)) => {
                if times[rr] - times[ll] <= 0.0 {
                    ll
                } else {
                    rr
                }
            }
        };
        let edge = default_edge(py, &nodes[selection], &nodes[index])?;
        Edge::set_length(edge.bind(py), Some(node_t - times[selection]), false, false)?;
        net.borrow_mut().add_edge(net.as_any(), edge.bind(py))?;
    }
    Ok(net)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(yule_generate, m)?)?;
    m.add_function(wrap_pyfunction!(cbdp_generate, m)?)?;
    Ok(())
}
