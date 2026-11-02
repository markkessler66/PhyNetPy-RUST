//! ``Edge``: a directed branch between two nodes with a length, an
//! inheritance probability, a weight and a tag.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::branch::Branch;
use super::journal::{Element, Entry, Old, Setter};
use super::node::{Node, Owner};
use crate::util::{deepcopy, edge_error, next_uid, stable_hash, warn};

#[pyclass(name = "Edge", module = "phynetpy.Network", weakref)]
pub struct Edge {
    pub src: Py<Node>,
    pub dest: Py<Node>,
    pub length: Option<f64>,
    pub gamma: Option<f64>,
    pub weight: Option<f64>,
    pub tag: Option<String>,
    pub hash: u64,
    pub uid: u64,
    pub owner: Option<Owner>,
}

impl Edge {
    pub fn make(
        py: Python<'_>,
        src: Py<Node>,
        dest: Py<Node>,
        length: Option<f64>,
        gamma: Option<f64>,
        weight: Option<f64>,
        tag: Option<String>,
    ) -> Edge {
        let hash = {
            let s = src.borrow(py);
            let d = dest.borrow(py);
            stable_hash(&format!("{}\0{}", s.name, d.name))
        };
        Edge {
            src,
            dest,
            length,
            gamma,
            weight,
            tag,
            hash,
            uid: next_uid(),
            owner: None,
        }
    }

    fn record(slf: &Bound<'_, Self>, this: &Edge, setter: Setter, old: Old) {
        if let Some(o) = &this.owner {
            if o.journal.is_recording() {
                o.journal.record(Entry::Attr {
                    element: Element::Edge(slf.clone().unbind()),
                    setter,
                    old,
                });
            }
        }
    }

    pub fn endpoint_uids(&self, py: Python<'_>) -> (u64, u64) {
        (self.src.borrow(py).uid, self.dest.borrow(py).uid)
    }
}

fn old_f(v: Option<f64>) -> Old {
    match v {
        Some(x) => Old::F(x),
        None => Old::Unset,
    }
}

#[pymethods]
impl Edge {
    #[new]
    #[pyo3(signature = (source, destination, length=None, gamma=None, weight=None, tag=None))]
    fn py_new(
        py: Python<'_>,
        source: Py<Node>,
        destination: Py<Node>,
        length: Option<f64>,
        gamma: Option<f64>,
        weight: Option<f64>,
        tag: Option<String>,
    ) -> PyResult<Edge> {
        let gamma = gamma.unwrap_or(0.0);
        if !(0.0..=1.0).contains(&gamma) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Please provide a probabilistic value for gamma (between 0 and 1, inclusive)!",
            ));
        }
        Ok(Edge::make(
            py,
            source,
            destination,
            Some(length.unwrap_or(1.0)),
            Some(gamma),
            Some(weight.unwrap_or(0.0)),
            Some(tag.unwrap_or_else(|| "no tag assigned yet".to_string())),
        ))
    }

    #[getter]
    fn src(&self, py: Python<'_>) -> Py<Node> {
        self.src.clone_ref(py)
    }

    #[getter]
    fn dest(&self, py: Python<'_>) -> Py<Node> {
        self.dest.clone_ref(py)
    }

    fn set_tag(slf: &Bound<'_, Self>, new_tag: Option<String>) {
        let mut this = slf.borrow_mut();
        let old = match &this.tag {
            Some(t) => Old::S(t.clone()),
            None => Old::Unset,
        };
        Edge::record(slf, &this, Setter::Tag, old);
        this.tag = new_tag;
    }

    fn get_tag(&self) -> Option<String> {
        self.tag.clone()
    }

    fn __hash__(&self) -> u64 {
        self.hash
    }

    pub fn set_gamma(slf: &Bound<'_, Self>, gamma: f64) -> PyResult<()> {
        if !(0.0..=1.0).contains(&gamma) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Please provide a probabilistic value for gamma (between 0 and 1, inclusive)!",
            ));
        }
        let mut this = slf.borrow_mut();
        let old = old_f(this.gamma);
        Edge::record(slf, &this, Setter::Gamma, old);
        this.gamma = Some(gamma);
        Ok(())
    }

    fn get_gamma(&self) -> Option<f64> {
        self.gamma
    }

    #[pyo3(signature = (new_src=None, new_dest=None))]
    pub fn copy(
        &self,
        py: Python<'_>,
        new_src: Option<Py<Node>>,
        new_dest: Option<Py<Node>>,
    ) -> PyResult<Py<Edge>> {
        let (s, d) = match (new_src, new_dest) {
            (Some(s), Some(d)) => (s, d),
            _ => (
                self.src.borrow(py).copy(py)?,
                self.dest.borrow(py).copy(py)?,
            ),
        };
        Py::new(
            py,
            Edge::make(
                py,
                s,
                d,
                self.length,
                self.gamma,
                self.weight,
                self.tag.clone(),
            ),
        )
    }

    #[pyo3(signature = (branch_length, warn_times=false, enforce_times=false))]
    pub fn set_length(
        slf: &Bound<'_, Self>,
        branch_length: Option<f64>,
        warn_times: bool,
        enforce_times: bool,
    ) -> PyResult<()> {
        let py = slf.py();
        let mut this = slf.borrow_mut();
        let check_len = {
            let d = this.dest.borrow(py).t;
            let s = this.src.borrow(py).t;
            match (d, s) {
                (Some(d), Some(s)) => d - s,
                _ => 0.0,
            }
        };
        if enforce_times {
            let bl = branch_length.ok_or_else(|| {
                pyo3::exceptions::PyTypeError::new_err(
                    "branch length must be a number when enforce_times is set",
                )
            })?;
            if (bl - check_len).abs() >= 1e-5 {
                return Err(edge_error(
                    py,
                    "Provided length is not equivalent to provided n1 and n2 times!",
                ));
            }
        } else if warn_times {
            if branch_length != Some(check_len) {
                warn(
                    py,
                    "Setting branch length of an edge to a value that is not equivalent \
                     to the difference in their set speciation times",
                )?;
            }
        }
        let old = old_f(this.length);
        Edge::record(slf, &this, Setter::Length, old);
        this.length = branch_length;
        Ok(())
    }

    fn get_length(&self) -> Option<f64> {
        self.length
    }

    fn set_weight(slf: &Bound<'_, Self>, new_weight: Option<f64>) {
        let mut this = slf.borrow_mut();
        let old = old_f(this.weight);
        Edge::record(slf, &this, Setter::Weight, old);
        this.weight = new_weight;
    }

    fn get_weight(&self) -> Option<f64> {
        self.weight
    }

    fn to_names(&self, py: Python<'_>) -> (String, String) {
        (
            self.src.borrow(py).name.clone(),
            self.dest.borrow(py).name.clone(),
        )
    }

    fn to_branch(&self, py: Python<'_>) -> Branch {
        Branch::make(
            self.length,
            self.gamma,
            Some(self.src.borrow(py).name.clone()),
        )
    }

    #[getter]
    fn _owner<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        self.owner.as_ref().and_then(|o| o.upgrade(py))
    }

    // ── copy / pickle protocol ────────────────────────────────────────

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, [self.src.bind(py), self.dest.bind(py)])
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("length", self.length)?;
        d.set_item("gamma", self.gamma)?;
        d.set_item("weight", self.weight)?;
        d.set_item("tag", self.tag.clone())?;
        Ok(d)
    }

    fn __setstate__(&mut self, state: Bound<'_, PyDict>) -> PyResult<()> {
        fn opt_f(v: Option<Bound<'_, PyAny>>) -> PyResult<Option<f64>> {
            match v {
                Some(x) if !x.is_none() => Ok(Some(x.extract()?)),
                _ => Ok(None),
            }
        }
        self.length = opt_f(state.get_item("length")?)?;
        self.gamma = opt_f(state.get_item("gamma")?)?;
        self.weight = opt_f(state.get_item("weight")?)?;
        self.tag = match state.get_item("tag")? {
            Some(x) if !x.is_none() => Some(x.extract()?),
            _ => None,
        };
        Ok(())
    }

    fn __deepcopy__(&self, py: Python<'_>, memo: Bound<'_, PyDict>) -> PyResult<Py<Edge>> {
        let s = deepcopy(py, self.src.bind(py).as_any(), &memo)?
            .cast_into::<Node>()?
            .unbind();
        let d = deepcopy(py, self.dest.bind(py).as_any(), &memo)?
            .cast_into::<Node>()?
            .unbind();
        Py::new(
            py,
            Edge::make(py, s, d, self.length, self.gamma, self.weight, self.tag.clone()),
        )
    }

    fn __copy__(&self, py: Python<'_>) -> PyResult<Py<Edge>> {
        Py::new(
            py,
            Edge::make(
                py,
                self.src.clone_ref(py),
                self.dest.clone_ref(py),
                self.length,
                self.gamma,
                self.weight,
                self.tag.clone(),
            ),
        )
    }
}
