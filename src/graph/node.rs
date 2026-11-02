//! ``Node``: a labelled vertex of a phylogenetic network.
//!
//! Equality and hashing are by label (process-independent FNV-1a), so a
//! network cannot hold two nodes with the same label, and a freshly built
//! ``Node("A")`` finds the network's own ``A`` in membership tests.

use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyDict, PyTuple, PyWeakrefReference};
use pyo3::types::PyWeakrefMethods;

use super::journal::{Element, Entry, Journal, Old, Setter};
use crate::util::{deepcopy, next_uid, node_error, stable_hash};

/// The network a node or edge belongs to: a weak reference to the Python
/// object (for ``_owner``) and a handle on its transaction journal.
pub struct Owner {
    pub net: Py<PyWeakrefReference>,
    pub journal: Journal,
}

impl Owner {
    pub fn upgrade<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        self.net.bind(py).upgrade()
    }
}

#[pyclass(name = "Node", module = "phynetpy.Network", weakref)]
pub struct Node {
    pub name: String,
    pub is_retic: bool,
    pub attrs: Py<PyDict>,
    pub seq: Option<Py<PyAny>>,
    pub t: Option<f64>,
    pub is_dirty: bool,
    pub hash: u64,
    pub uid: u64,
    pub owner: Option<Owner>,
}

impl Node {
    pub fn make(
        py: Python<'_>,
        name: String,
        is_retic: bool,
        attrs: Option<Bound<'_, PyDict>>,
        seq: Option<Py<PyAny>>,
        t: Option<f64>,
    ) -> Node {
        let hash = stable_hash(&name);
        Node {
            name,
            is_retic,
            attrs: attrs.map(|d| d.unbind()).unwrap_or_else(|| PyDict::new(py).unbind()),
            seq,
            t,
            is_dirty: false,
            hash,
            uid: next_uid(),
            owner: None,
        }
    }

    /// Rename without any bookkeeping (the network keeps its own index).
    pub fn set_name_inner(&mut self, new_name: String) {
        self.hash = stable_hash(&new_name);
        self.name = new_name;
        self.is_dirty = true;
    }

    fn record(slf: &Bound<'_, Self>, this: &Node, setter: Setter, old: Old) {
        if let Some(o) = &this.owner {
            if o.journal.is_recording() {
                o.journal.record(Entry::Attr {
                    element: Element::Node(slf.clone().unbind()),
                    setter,
                    old,
                });
            }
        }
    }

    fn comparison_key(&self) -> (f64, &str) {
        (self.t.unwrap_or(f64::INFINITY), self.name.as_str())
    }
}

#[pymethods]
impl Node {
    #[new]
    #[pyo3(signature = (name, is_reticulation=false, attr=None, seq=None, t=None))]
    fn py_new(
        py: Python<'_>,
        name: String,
        is_reticulation: bool,
        attr: Option<Bound<'_, PyDict>>,
        seq: Option<Py<PyAny>>,
        t: Option<f64>,
    ) -> Node {
        Node::make(py, name, is_reticulation, attr, seq, t)
    }

    fn get_attributes(&self, py: Python<'_>) -> Py<PyDict> {
        self.attrs.clone_ref(py)
    }

    fn set_attributes(&mut self, new_attr: Bound<'_, PyDict>) {
        self.attrs = new_attr.unbind();
    }

    fn get_time(&self, py: Python<'_>) -> PyResult<f64> {
        self.t
            .ok_or_else(|| node_error(py, "No time has been set for this node!"))
    }

    pub fn set_time(slf: &Bound<'_, Self>, new_t: f64) -> PyResult<()> {
        let py = slf.py();
        let mut this = slf.borrow_mut();
        if new_t >= 0.0 {
            let old = match this.t {
                Some(v) => Old::F(v),
                None => Old::Unset,
            };
            Node::record(slf, &this, Setter::Time, old);
            this.t = Some(new_t);
            Ok(())
        } else {
            Err(node_error(
                py,
                "Please set speciation time, t, to a non-negative number!",
            ))
        }
    }

    fn __str__(&self) -> String {
        self.name.clone()
    }

    fn _get_comparison_key(&self) -> (f64, String) {
        (self.t.unwrap_or(f64::INFINITY), self.name.clone())
    }

    fn __richcmp__(
        &self,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let Ok(other) = other.cast::<Node>() else {
            return Ok(py.NotImplemented());
        };
        let other = other.borrow();
        let result = match op {
            CompareOp::Eq => self.name == other.name,
            CompareOp::Ne => self.name != other.name,
            _ => {
                let a = self.comparison_key();
                let b = other.comparison_key();
                let ord = a
                    .0
                    .partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(b.1));
                match op {
                    CompareOp::Lt => ord.is_lt(),
                    CompareOp::Le => ord.is_le(),
                    CompareOp::Gt => ord.is_gt(),
                    CompareOp::Ge => ord.is_ge(),
                    _ => unreachable!(),
                }
            }
        };
        Ok(result.into_pyobject(py)?.to_owned().into_any().unbind())
    }

    fn __hash__(&self) -> u64 {
        self.hash
    }

    fn to_string(&self, py: Python<'_>) -> PyResult<String> {
        let mut s = format!("Node {}: ", self.name);
        if let Some(t) = self.t {
            let rounded = py
                .import("builtins")?
                .getattr("round")?
                .call1((t, 4))?
                .str()?
                .to_string();
            s.push_str(&format!("t = {rounded} "));
        }
        s.push_str(&format!(
            " is a reticulation node? {}",
            if self.is_retic { "True" } else { "False" }
        ));
        s.push_str(&format!(
            " has attributes: {}",
            self.attrs.bind(py).str()?
        ));
        Ok(s)
    }

    #[getter]
    fn label(&self) -> &str {
        &self.name
    }

    fn set_name(&mut self, new_name: String) {
        self.set_name_inner(new_name);
    }

    pub fn set_is_reticulation(slf: &Bound<'_, Self>, new_is_retic: bool) {
        let mut this = slf.borrow_mut();
        let old = Old::B(this.is_retic);
        Node::record(slf, &this, Setter::IsRetic, old);
        this.is_retic = new_is_retic;
    }

    fn is_reticulation(&self) -> bool {
        self.is_retic
    }

    fn add_attribute(
        &self,
        py: Python<'_>,
        key: Bound<'_, PyAny>,
        value: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.attrs.bind(py).set_item(key, value)
    }

    fn attribute_value<'py>(
        &self,
        py: Python<'py>,
        key: Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.attrs.bind(py).get_item(key)
    }

    fn set_seq(&mut self, new_sequence: Py<PyAny>) {
        self.seq = Some(new_sequence);
    }

    fn get_seq(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.seq {
            Some(s) => Ok(s.clone_ref(py)),
            None => Err(node_error(
                py,
                "No sequence record has been associated with this node!",
            )),
        }
    }

    pub fn copy(&self, py: Python<'_>) -> PyResult<Py<Node>> {
        let attrs = self.attrs.bind(py).copy()?;
        let mut n = Node::make(
            py,
            self.name.clone(),
            self.is_retic,
            Some(attrs),
            self.seq.as_ref().map(|s| s.clone_ref(py)),
            self.t,
        );
        n.is_dirty = self.is_dirty;
        Py::new(py, n)
    }

    #[getter]
    fn _owner<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        self.owner.as_ref().and_then(|o| o.upgrade(py))
    }

    // ── copy / pickle protocol ────────────────────────────────────────

    fn __getnewargs__(&self) -> (String, bool) {
        (self.name.clone(), self.is_retic)
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("attrs", self.attrs.bind(py))?;
        d.set_item("seq", self.seq.as_ref().map(|s| s.bind(py)))?;
        d.set_item("t", self.t)?;
        d.set_item("is_dirty", self.is_dirty)?;
        Ok(d)
    }

    fn __setstate__(&mut self, state: Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(a) = state.get_item("attrs")? {
            self.attrs = a.cast_into::<PyDict>()?.unbind();
        }
        if let Some(s) = state.get_item("seq")? {
            self.seq = if s.is_none() { None } else { Some(s.unbind()) };
        }
        if let Some(t) = state.get_item("t")? {
            self.t = if t.is_none() { None } else { Some(t.extract()?) };
        }
        if let Some(d) = state.get_item("is_dirty")? {
            self.is_dirty = d.extract()?;
        }
        Ok(())
    }

    fn __deepcopy__(&self, py: Python<'_>, memo: Bound<'_, PyDict>) -> PyResult<Py<Node>> {
        let attrs = deepcopy(py, self.attrs.bind(py).as_any(), &memo)?
            .cast_into::<PyDict>()?;
        let seq = match &self.seq {
            Some(s) => Some(deepcopy(py, s.bind(py), &memo)?.unbind()),
            None => None,
        };
        let mut n = Node::make(py, self.name.clone(), self.is_retic, Some(attrs), seq, self.t);
        n.is_dirty = self.is_dirty;
        Py::new(py, n)
    }

    fn __copy__(&self, py: Python<'_>) -> PyResult<Py<Node>> {
        let mut n = Node::make(
            py,
            self.name.clone(),
            self.is_retic,
            Some(self.attrs.bind(py).clone()),
            self.seq.as_ref().map(|s| s.clone_ref(py)),
            self.t,
        );
        n.is_dirty = self.is_dirty;
        Py::new(py, n)
    }
}

/// Flatten ``*nodes`` style arguments: Node objects, or (nested) lists,
/// tuples and sets of them.
pub fn flatten_nodes<'py>(
    args: &Bound<'py, PyTuple>,
) -> PyResult<Vec<Bound<'py, Node>>> {
    fn rec<'py>(item: &Bound<'py, PyAny>, out: &mut Vec<Bound<'py, Node>>) -> PyResult<()> {
        if let Ok(n) = item.cast::<Node>() {
            out.push(n.clone());
            return Ok(());
        }
        if item.is_instance_of::<pyo3::types::PyList>()
            || item.is_instance_of::<pyo3::types::PyTuple>()
            || item.is_instance_of::<pyo3::types::PySet>()
            || item.is_instance_of::<pyo3::types::PyFrozenSet>()
        {
            for sub in item.try_iter()? {
                rec(&sub?, out)?;
            }
            return Ok(());
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Expected Node objects (or lists of them); got {}.",
            crate::util::type_name(item)
        )))
    }
    let mut out = Vec::new();
    for item in args.iter() {
        rec(&item, &mut out)?;
    }
    Ok(out)
}
