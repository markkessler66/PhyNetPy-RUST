//! Shared helpers: process-independent hashing, object identity counters,
//! Python warning emission, and the registry of Python-defined exception
//! classes that the Rust core raises.

use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::exceptions::{PyRuntimeError, PyUserWarning};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyType};

static UID: AtomicU64 = AtomicU64::new(1);

/// A fresh identity for a Node or Edge object. Unique for the life of the
/// process; never reused.
#[inline]
pub fn next_uid() -> u64 {
    UID.fetch_add(1, Ordering::Relaxed)
}

/// 64-bit FNV-1a reduced modulo ``2**61 - 1`` so that it equals what
/// Python's ``hash()`` returns for the matching ``__hash__`` implementation.
pub fn stable_hash(text: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h % ((1u64 << 61) - 1)
}

#[pyfunction]
#[pyo3(name = "stable_hash")]
pub fn stable_hash_py(text: &str) -> u64 {
    stable_hash(text)
}

/// Exception classes defined on the Python side (``phynetpy.Network`` and
/// ``phynetpy.Newick``) that the core raises.  Registered at import time so
/// ``except NetworkError`` in user code catches errors raised from Rust.
pub struct ExcTypes {
    pub network: Py<PyType>,
    pub node: Py<PyType>,
    pub edge: Py<PyType>,
}

static EXC: PyOnceLock<ExcTypes> = PyOnceLock::new();
static NEWICK_EXC: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static NETWORK_CLS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static COERCE_UNIT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

#[pyfunction]
pub fn _register_exceptions(
    py: Python<'_>,
    network_error: Bound<'_, PyType>,
    node_error: Bound<'_, PyType>,
    edge_error: Bound<'_, PyType>,
) {
    let _ = EXC.set(
        py,
        ExcTypes {
            network: network_error.unbind(),
            node: node_error.unbind(),
            edge: edge_error.unbind(),
        },
    );
}

#[pyfunction]
pub fn _register_newick_error(py: Python<'_>, newick_error: Bound<'_, PyType>) {
    let _ = NEWICK_EXC.set(py, newick_error.unbind());
}

#[pyfunction]
pub fn _register_network_class(py: Python<'_>, cls: Bound<'_, PyType>) {
    let _ = NETWORK_CLS.set(py, cls.unbind());
}

/// The Python-level ``Network`` class (a subclass of the Rust base), if the
/// package registered one.  Rust code that builds new networks instantiates
/// this so ``type(net) is Network`` holds for every network a user sees.
pub fn network_class(py: Python<'_>) -> Option<&Bound<'_, PyType>> {
    NETWORK_CLS.get(py).map(|c| c.bind(py))
}

fn raise_registered(py: Python<'_>, ty: Option<&Py<PyType>>, msg: String) -> PyErr {
    match ty {
        Some(t) => PyErr::from_type(t.bind(py).clone(), msg),
        None => PyRuntimeError::new_err(msg),
    }
}

pub fn network_error(py: Python<'_>, msg: impl Into<String>) -> PyErr {
    raise_registered(py, EXC.get(py).map(|t| &t.network), msg.into())
}

pub fn node_error(py: Python<'_>, msg: impl Into<String>) -> PyErr {
    raise_registered(py, EXC.get(py).map(|t| &t.node), msg.into())
}

pub fn edge_error(py: Python<'_>, msg: impl Into<String>) -> PyErr {
    raise_registered(py, EXC.get(py).map(|t| &t.edge), msg.into())
}

pub fn newick_error(py: Python<'_>, msg: impl Into<String>) -> PyErr {
    if NEWICK_EXC.get(py).is_none() {
        // Importing the module registers the class as a side effect.
        let _ = py.import("phynetpy.Newick");
    }
    raise_registered(py, NEWICK_EXC.get(py), msg.into())
}

/// ``True`` when ``err`` is an instance of the registered NewickParserError.
pub fn is_newick_error(py: Python<'_>, err: &PyErr) -> bool {
    match NEWICK_EXC.get(py) {
        Some(t) => err.is_instance(py, t.bind(py)),
        None => false,
    }
}

/// Emit a ``UserWarning`` through Python's warnings machinery.
pub fn warn(py: Python<'_>, msg: &str) -> PyResult<()> {
    let cat = py.get_type::<PyUserWarning>();
    let c = CString::new(msg).unwrap_or_else(|_| CString::new("warning").unwrap());
    PyErr::warn(py, cat.as_any(), &c, 1)
}

/// ``phynetpy._units.coerce_branch_length_unit(value)``.
pub fn coerce_unit<'py>(
    py: Python<'py>,
    value: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let f = COERCE_UNIT.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py
            .import("phynetpy._units")?
            .getattr("coerce_branch_length_unit")?
            .unbind())
    })?;
    match value {
        Some(v) => f.bind(py).call1((v,)),
        None => f.bind(py).call1((py.None(),)),
    }
}

/// ``repr(float(x))`` exactly as Python prints it.
pub fn py_float_repr(py: Python<'_>, x: f64) -> PyResult<String> {
    Ok(x.into_pyobject(py)?.repr()?.to_string())
}

/// ``copy.deepcopy(obj, memo)``.
pub fn deepcopy<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    memo: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("copy")?.getattr("deepcopy")?.call1((obj, memo))
}

/// ``type(obj).__name__``.
pub fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "object".to_string())
}
