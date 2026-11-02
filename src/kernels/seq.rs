//! Sequence-likelihood kernels for ``phynetpy._seq_likelihood`` and the
//! numerical core of the model-graph substitution models in ``phynetpy.GTR``.
//!
//! * [`SubstitutionModel`]: a time-reversible 4-state nucleotide model.  It
//!   normalises ``Q`` to unit mean rate and eigendecomposes it through the
//!   ``pi^(1/2)`` similarity transform (a symmetric problem), then serves
//!   ``P(t) = exp(Q t)`` from the cached factors.  The Python classes
//!   ``SubstitutionModel`` / ``JC69`` / ``HKY85`` / ``GTR`` subclass it and
//!   fill it in through ``_set_params`` / ``_build_Q`` / ``_decompose_from``.
//! * [`FelsensteinCalculator`]: site-pattern compression of one locus and the
//!   vectorised post-order pruning over a gene tree, which is read through the
//!   uid API of [`Network`] (children, edge lengths) rather than Python calls.
//! * [`GtrCore`] plus the closed-form transition matrices (JC, F81, the
//!   Tamura-Nei family, K81) behind the ``phynetpy.GTR`` facades.

use std::cmp::Ordering;

use nalgebra::{DMatrix, DVector, Matrix4, SymmetricEigen};
use numpy::ndarray::Array2;
use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rustc_hash::FxHashMap;

use crate::graph::Network;

/// Iteration cap for the symmetric QR sweep; convergence takes a few dozen
/// iterations for a 4x4, so hitting this means the input was degenerate.
const MAX_EIGEN_ITER: usize = 100_000;

// ═══════════════════════════════════════════════════════════════════════
// numpy helpers
// ═══════════════════════════════════════════════════════════════════════

fn np1<'py>(py: Python<'py>, data: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, data)
}

fn np2<'py>(py: Python<'py>, rows: usize, cols: usize, data: Vec<f64>) -> Bound<'py, PyArray2<f64>> {
    let arr = Array2::from_shape_vec((rows, cols), data).expect("row-major data of the stated shape");
    PyArray2::from_owned_array(py, arr)
}

fn np4x4<'py>(py: Python<'py>, m: &[[f64; 4]; 4]) -> Bound<'py, PyArray2<f64>> {
    np2(py, 4, 4, flat4(m))
}

fn flat4(m: &[[f64; 4]; 4]) -> Vec<f64> {
    m.iter().flat_map(|r| r.iter().copied()).collect()
}

fn unflat4(v: &[f64], what: &str) -> PyResult<[[f64; 4]; 4]> {
    if v.len() != 16 {
        return Err(PyValueError::new_err(format!("{what} must hold 16 values.")));
    }
    let mut out = [[0.0; 4]; 4];
    for (k, x) in v.iter().enumerate() {
        out[k / 4][k % 4] = *x;
    }
    Ok(out)
}

fn arr4(v: &[f64], what: &str) -> PyResult<[f64; 4]> {
    <[f64; 4]>::try_from(v).map_err(|_| PyValueError::new_err(format!("{what} must have length 4.")))
}

/// ``np.asarray(obj, dtype=float64)`` for a one-dimensional input, as a Vec.
fn float_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    match obj.extract::<PyArrayLike1<f64, AllowTypeChange>>() {
        Ok(a) => Ok(a.as_array().iter().copied().collect()),
        Err(e) => obj.extract::<Vec<f64>>().map_err(|_| e),
    }
}

/// ``obj.ndim`` when ``obj`` is an array that has one.
fn ndim_of(obj: &Bound<'_, PyAny>) -> Option<usize> {
    obj.getattr("ndim").ok().and_then(|n| n.extract::<usize>().ok())
}

/// A square ``n x n`` matrix from anything ``np.asarray`` accepts.
fn dmatrix_of(obj: &Bound<'_, PyAny>, n: usize, what: &str) -> PyResult<DMatrix<f64>> {
    let bad = || PyValueError::new_err(format!("{what} must be a {n}x{n} matrix."));
    if let Ok(a) = obj.extract::<PyArrayLike2<f64, AllowTypeChange>>() {
        let v = a.as_array();
        if v.shape() != [n, n] {
            return Err(bad());
        }
        return Ok(DMatrix::from_fn(n, n, |i, j| v[[i, j]]));
    }
    let rows: Vec<Vec<f64>> = obj.extract().map_err(|_| bad())?;
    if rows.len() != n || rows.iter().any(|r| r.len() != n) {
        return Err(bad());
    }
    Ok(DMatrix::from_fn(n, n, |i, j| rows[i][j]))
}

fn matrix4_of(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<[[f64; 4]; 4]> {
    let m = dmatrix_of(obj, 4, what)?;
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = m[(i, j)];
        }
    }
    Ok(out)
}

/// Python's ``math.isclose(a, b, rel_tol=1e-9, abs_tol=abs_tol)``.
fn isclose(a: f64, b: f64, abs_tol: f64) -> bool {
    const REL_TOL: f64 = 1e-9;
    if a == b {
        return true;
    }
    if a.is_infinite() || b.is_infinite() {
        return false;
    }
    let diff = (a - b).abs();
    diff <= (REL_TOL * b).abs() || diff <= (REL_TOL * a).abs() || diff <= abs_tol
}

/// Ascending permutation of ``values`` (the order ``numpy.linalg.eigh`` uses).
fn ascending(values: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(Ordering::Equal));
    idx
}

// ═══════════════════════════════════════════════════════════════════════
// Nucleotide encoding (A=0, C=1, G=2, T=3) with IUPAC ambiguity
// ═══════════════════════════════════════════════════════════════════════

/// Nucleotide order shared with PhyloNet/BEAGLE: ``A,C,G,T`` -> ``0,1,2,3``.
pub const DNA_STATES: &str = "ACGT";

const A: u8 = 1;
const C: u8 = 2;
const G: u8 = 4;
const T: u8 = 8;
const ANY: u8 = 15;

/// Bitmask of the concrete nucleotides an upper-case IUPAC character covers.
/// Gaps, ``N``, ``?``, ``X``, ``.``, ``O`` and unknown characters are fully
/// ambiguous, matching PhyloNet's permissive tip handling.
#[inline]
fn iupac_mask(ch: char) -> u8 {
    match ch {
        'A' => A,
        'C' => C,
        'G' => G,
        'T' | 'U' => T,
        'R' => A | G,
        'Y' => C | T,
        'S' => C | G,
        'W' => A | T,
        'K' => G | T,
        'M' => A | C,
        'B' => C | G | T,
        'D' => A | G | T,
        'H' => A | C | T,
        'V' => A | C | G,
        _ => ANY,
    }
}

/// ``ch.upper()`` when that is a single character, else ``ch`` itself.
#[inline]
fn upper_char(ch: char) -> char {
    let mut it = ch.to_uppercase();
    match (it.next(), it.next()) {
        (Some(u), None) => u,
        _ => ch,
    }
}

#[inline]
fn mask_partials(mask: u8) -> [f64; 4] {
    [
        f64::from(mask & A != 0),
        f64::from(mask & C != 0),
        f64::from(mask & G != 0),
        f64::from(mask & T != 0),
    ]
}

/// Length-4 tip partial-likelihood vector for a DNA character.
///
/// Args:
///     ch: A single (case-insensitive) IUPAC nucleotide / ambiguity / gap
///         character.
///
/// Returns:
///     ``float64`` array of length 4 (order ``A,C,G,T``) with ``1.0`` in
///     every state the character is compatible with and ``0.0`` elsewhere.
///     Unknown characters are treated as fully ambiguous (all ones), matching
///     PhyloNet's permissive tip handling.
#[pyfunction]
fn tip_partials_for_char<'py>(
    py: Python<'py>,
    ch: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let text = ch.str()?.to_string().to_uppercase();
    let mut chars = text.chars();
    let mask = match (chars.next(), chars.next()) {
        (Some(c), None) => iupac_mask(c),
        _ => ANY,
    };
    Ok(np1(py, mask_partials(mask).to_vec()))
}

// ═══════════════════════════════════════════════════════════════════════
// SubstitutionModel
// ═══════════════════════════════════════════════════════════════════════

/// Time-reversible 4-state nucleotide substitution model (compiled core).
///
/// The Python class ``phynetpy._seq_likelihood.SubstitutionModel`` subclasses
/// this and drives construction: ``_set_params(pi, rates)`` validates and
/// stores the parameters, ``_build_Q()`` assembles the un-normalised rate
/// matrix, and ``_decompose_from(Q)`` normalises it to unit mean rate and
/// eigendecomposes it.  ``p_matrix(t)`` then costs one rescaling and one
/// ``4x4`` product.
#[pyclass(name = "SubstitutionModel", module = "phynetpy._seq_likelihood", subclass)]
pub struct SubstitutionModel {
    pi: [f64; 4],
    rates: [f64; 6],
    q: [[f64; 4]; 4],
    evals: [f64; 4],
    u: [[f64; 4]; 4],
    uinv: [[f64; 4]; 4],
    /// Serve the closed-form Jukes-Cantor matrix instead of the eigen form.
    jc: bool,
    has_params: bool,
    ready: bool,
}

impl SubstitutionModel {
    fn blank() -> Self {
        SubstitutionModel {
            pi: [0.0; 4],
            rates: [0.0; 6],
            q: [[0.0; 4]; 4],
            evals: [0.0; 4],
            u: [[0.0; 4]; 4],
            uinv: [[0.0; 4]; 4],
            jc: false,
            has_params: false,
            ready: false,
        }
    }

    fn require_params(&self) -> PyResult<()> {
        if self.has_params {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(
                "SubstitutionModel has no parameters yet; its __init__ did not run.",
            ))
        }
    }

    fn require_ready(&self) -> PyResult<()> {
        if self.ready {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(
                "SubstitutionModel has not been built yet; its __init__ did not run.",
            ))
        }
    }

    /// Validate and store ``pi`` (length 4, summing to 1) and the six rates.
    pub fn set_params(&mut self, pi: [f64; 4], rates: [f64; 6]) -> PyResult<()> {
        let mut total = 0.0;
        for p in pi {
            total += p;
        }
        if !isclose(total, 1.0, 1e-9) {
            return Err(PyValueError::new_err("Base frequencies must sum to 1."));
        }
        if pi.iter().any(|p| !(*p > 0.0) || !p.is_finite()) {
            return Err(PyValueError::new_err(
                "Base frequencies must be positive and finite (A,C,G,T).",
            ));
        }
        self.pi = pi;
        self.rates = rates;
        self.has_params = true;
        self.ready = false;
        Ok(())
    }

    /// ``Q[i,j] = rate(i,j) * pi[j]`` off the diagonal, zero on it.
    pub fn build_q(&self) -> [[f64; 4]; 4] {
        let [ac, ag, at, cg, ct, gt] = self.rates;
        let r = [
            [0.0, ac, ag, at],
            [ac, 0.0, cg, ct],
            [ag, cg, 0.0, gt],
            [at, ct, gt, 0.0],
        ];
        let mut q = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                q[i][j] = r[i][j] * self.pi[j];
            }
        }
        q
    }

    /// Fill the diagonal, normalise to unit mean rate and eigendecompose
    /// through the ``pi^(1/2)`` similarity transform.
    pub fn decompose_from(&mut self, q_in: [[f64; 4]; 4]) -> PyResult<()> {
        self.require_params()?;
        let mut q = q_in;
        for i in 0..4 {
            q[i][i] = 0.0;
        }
        for i in 0..4 {
            let mut row = 0.0;
            for j in 0..4 {
                row += q[i][j];
            }
            q[i][i] = -row;
        }
        let mut dot = 0.0;
        for i in 0..4 {
            dot += self.pi[i] * q[i][i];
        }
        let mean_rate = -dot;
        if mean_rate <= 0.0 {
            return Err(PyValueError::new_err(
                "Degenerate substitution model (non-positive rate).",
            ));
        }
        for row in q.iter_mut() {
            for x in row.iter_mut() {
                *x /= mean_rate;
            }
        }

        // Symmetric eigendecomposition via the pi^{1/2} similarity transform:
        // far more stable than a general eig() and exactly how BEAST/PhyloNet
        // decompose a reversible Q.
        let sqrt_pi = self.pi.map(f64::sqrt);
        let inv_sqrt_pi = sqrt_pi.map(|s| 1.0 / s);
        let mut b = Matrix4::<f64>::zeros();
        for i in 0..4 {
            for j in 0..4 {
                b[(i, j)] = q[i][j] * sqrt_pi[i] * inv_sqrt_pi[j];
            }
        }
        // Symmetrise away round-off.
        let bs = Matrix4::from_fn(|i, j| 0.5 * (b[(i, j)] + b[(j, i)]));
        if bs.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err(
                "Substitution model rate matrix has non-finite entries.",
            ));
        }
        let eig = SymmetricEigen::try_new(bs, f64::EPSILON, MAX_EIGEN_ITER).ok_or_else(|| {
            PyValueError::new_err("Eigendecomposition of the rate matrix did not converge.")
        })?;
        let order = ascending(eig.eigenvalues.as_slice());
        let mut evals = [0.0; 4];
        let mut u = [[0.0; 4]; 4];
        let mut uinv = [[0.0; 4]; 4];
        for (k, &src) in order.iter().enumerate() {
            evals[k] = eig.eigenvalues[src];
            for i in 0..4 {
                let v = eig.eigenvectors[(i, src)];
                // Right/left eigenvectors of Q recovered from the symmetric ones.
                u[i][k] = inv_sqrt_pi[i] * v;
                uinv[k][i] = v * sqrt_pi[i];
            }
        }
        self.q = q;
        self.evals = evals;
        self.u = u;
        self.uinv = uinv;
        self.ready = true;
        Ok(())
    }

    /// ``P(t) = exp(Q t)`` into ``out`` (negative ``t`` clamps to 0).
    #[inline]
    pub fn p_matrix_into(&self, t: f64, out: &mut [[f64; 4]; 4]) {
        let t = if t < 0.0 { 0.0 } else { t };
        if self.jc {
            let e = (-4.0 * t / 3.0).exp();
            let same = 0.25 + 0.75 * e;
            let diff = 0.25 - 0.25 * e;
            for i in 0..4 {
                for j in 0..4 {
                    out[i][j] = if i == j { same } else { diff };
                }
            }
            return;
        }
        let expd = [
            (self.evals[0] * t).exp(),
            (self.evals[1] * t).exp(),
            (self.evals[2] * t).exp(),
            (self.evals[3] * t).exp(),
        ];
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += self.u[i][k] * expd[k] * self.uinv[k][j];
                }
                out[i][j] = s;
            }
        }
    }
}

#[pymethods]
impl SubstitutionModel {
    /// Lenient constructor: Python subclasses pass their own arguments here
    /// and then configure the model from ``__init__``.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn py_new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        let _ = (args, kwargs);
        Self::blank()
    }

    /// Validate and store ``pi`` and the six exchangeabilities.
    fn _set_params(&mut self, pi: &Bound<'_, PyAny>, rates: &Bound<'_, PyAny>) -> PyResult<()> {
        let length_err = || PyValueError::new_err("Base frequencies must have length 4 (A,C,G,T).");
        let pi_v = match float_vec(pi) {
            Ok(v) => v,
            Err(e) => {
                if ndim_of(pi).is_some_and(|n| n != 1) {
                    return Err(length_err());
                }
                return Err(e);
            }
        };
        if pi_v.len() != 4 {
            return Err(length_err());
        }
        let rates_v = float_vec(rates)?;
        let rates_a: [f64; 6] = rates_v.as_slice().try_into().map_err(|_| {
            PyValueError::new_err("Exchangeability rates must have length 6 (AC,AG,AT,CG,CT,GT).")
        })?;
        let pi_a: [f64; 4] = pi_v.as_slice().try_into().map_err(|_| length_err())?;
        self.set_params(pi_a, rates_a)
    }

    /// Assemble the un-normalised, un-row-summed rate matrix.
    ///
    /// Returns:
    ///     A ``4x4`` matrix whose off-diagonal entries are
    ///     ``Q[i,j] = rate(i,j) * pi[j]`` (the time-reversible GTR
    ///     parameterisation).  Diagonal is filled in by the caller.
    #[pyo3(name = "_build_Q")]
    fn build_q_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_params()?;
        Ok(np4x4(py, &self.build_q()))
    }

    /// Normalise ``Q`` to unit mean rate and eigendecompose it.
    fn _decompose_from(&mut self, q: &Bound<'_, PyAny>) -> PyResult<()> {
        let q = matrix4_of(q, "Rate matrix")?;
        self.decompose_from(q)
    }

    /// Serve the closed-form Jukes-Cantor ``P(t)`` (``JC69``) when enabled.
    fn _use_jc69_closed_form(&mut self, enabled: bool) {
        self.jc = enabled;
    }

    /// Stationary base frequency vector (length 4, ``A,C,G,T``).
    #[getter]
    fn pi<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_params()?;
        Ok(np1(py, self.pi.to_vec()))
    }

    /// The six exchangeabilities ``AC, AG, AT, CG, CT, GT``.
    #[getter]
    fn _rates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_params()?;
        Ok(np1(py, self.rates.to_vec()))
    }

    /// The normalised instantaneous rate matrix (unit mean rate).
    #[getter(Q)]
    fn q<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_ready()?;
        Ok(np4x4(py, &self.q))
    }

    /// Eigenvalues of ``Q`` in ascending order.
    #[getter]
    fn _evals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_ready()?;
        Ok(np1(py, self.evals.to_vec()))
    }

    /// Right eigenvectors of ``Q`` (columns).
    #[getter(_U)]
    fn u_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_ready()?;
        Ok(np4x4(py, &self.u))
    }

    /// Inverse of ``_U`` (the left eigenvectors as rows).
    #[getter(_Uinv)]
    fn uinv_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_ready()?;
        Ok(np4x4(py, &self.uinv))
    }

    /// Transition-probability matrix ``P(t) = exp(Q t)``.
    ///
    /// Args:
    ///     t: Branch length in expected substitutions per site (``>= 0``).
    ///
    /// Returns:
    ///     A ``4x4`` row-stochastic matrix; ``P[i,j]`` is the probability of
    ///     ending in state ``j`` after time ``t`` starting from state ``i``.
    fn p_matrix<'py>(&self, py: Python<'py>, t: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_ready()?;
        let mut out = [[0.0; 4]; 4];
        self.p_matrix_into(t, &mut out);
        Ok(np4x4(py, &out))
    }

    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let d = PyDict::new(py);
        {
            let m = slf.borrow();
            d.set_item("pi", m.pi.to_vec())?;
            d.set_item("rates", m.rates.to_vec())?;
            d.set_item("Q", flat4(&m.q))?;
            d.set_item("evals", m.evals.to_vec())?;
            d.set_item("U", flat4(&m.u))?;
            d.set_item("Uinv", flat4(&m.uinv))?;
            d.set_item("jc", m.jc)?;
            d.set_item("has_params", m.has_params)?;
            d.set_item("ready", m.ready)?;
        }
        if let Ok(dict) = slf.getattr("__dict__") {
            d.set_item("__dict__", dict)?;
        }
        Ok(d)
    }

    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyDict>) -> PyResult<()> {
        {
            let mut m = slf.borrow_mut();
            if let Some(v) = state.get_item("pi")? {
                m.pi = arr4(&v.extract::<Vec<f64>>()?, "pi")?;
            }
            if let Some(v) = state.get_item("rates")? {
                let r: Vec<f64> = v.extract()?;
                m.rates = r
                    .as_slice()
                    .try_into()
                    .map_err(|_| PyValueError::new_err("rates must have length 6."))?;
            }
            if let Some(v) = state.get_item("Q")? {
                m.q = unflat4(&v.extract::<Vec<f64>>()?, "Q")?;
            }
            if let Some(v) = state.get_item("evals")? {
                m.evals = arr4(&v.extract::<Vec<f64>>()?, "evals")?;
            }
            if let Some(v) = state.get_item("U")? {
                m.u = unflat4(&v.extract::<Vec<f64>>()?, "U")?;
            }
            if let Some(v) = state.get_item("Uinv")? {
                m.uinv = unflat4(&v.extract::<Vec<f64>>()?, "Uinv")?;
            }
            if let Some(v) = state.get_item("jc")? {
                m.jc = v.extract()?;
            }
            if let Some(v) = state.get_item("has_params")? {
                m.has_params = v.extract()?;
            }
            if let Some(v) = state.get_item("ready")? {
                m.ready = v.extract()?;
            }
        }
        if let Some(dict) = state.get_item("__dict__")? {
            if let Ok(target) = slf.getattr("__dict__") {
                target.call_method1("update", (dict,))?;
            }
        }
        Ok(())
    }

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::empty(py)
    }
}

/// ``True`` when ``model.p_matrix`` is the compiled implementation (or a
/// Python wrapper marked ``_native_p_matrix``), so the pruning loop may
/// evaluate ``P(t)`` in Rust instead of calling back into Python.
fn native_p_matrix(py: Python<'_>, model: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !model.is_instance_of::<SubstitutionModel>() {
        return Ok(false);
    }
    let pm = model.get_type().getattr("p_matrix")?;
    let base = py.get_type::<SubstitutionModel>().getattr("p_matrix")?;
    if pm.is(&base) {
        return Ok(true);
    }
    match pm.getattr("_native_p_matrix") {
        Ok(flag) => flag.is_truthy(),
        Err(_) => Ok(false),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Felsenstein pruning likelihood (vectorised over site patterns)
// ═══════════════════════════════════════════════════════════════════════

/// Per-locus phylogenetic likelihood ``P(S | g)`` via vectorised pruning
/// (compiled core).
///
/// The Python class ``phynetpy._seq_likelihood.FelsensteinCalculator``
/// subclasses this; its ``__init__`` calls ``_compress(alignment)``, which
/// reduces the alignment to its distinct site patterns with integer weights
/// and pre-builds the ``(n_patterns, 4)`` tip partials of every taxon.
#[pyclass(name = "FelsensteinCalculator", module = "phynetpy._seq_likelihood", subclass)]
pub struct FelsensteinCalculator {
    taxa: Vec<String>,
    row_of: FxHashMap<String, usize>,
    n_patterns: usize,
    weights: Vec<f64>,
    /// ``patterns[r][p]``: taxon ``r``'s (upper-cased) character in pattern ``p``.
    patterns: Vec<Vec<char>>,
    /// ``tips[r]``: row-major ``(n_patterns, 4)`` tip partials of taxon ``r``.
    tips: Vec<Vec<f64>>,
}

impl FelsensteinCalculator {
    fn blank() -> Self {
        FelsensteinCalculator {
            taxa: Vec::new(),
            row_of: FxHashMap::default(),
            n_patterns: 0,
            weights: Vec::new(),
            patterns: Vec::new(),
            tips: Vec::new(),
        }
    }

    /// Distinct columns (first-occurrence order) and their multiplicities.
    fn compress_columns(seqs: &[Vec<char>]) -> (Vec<Vec<char>>, Vec<f64>) {
        let length = seqs.first().map_or(0, Vec::len);
        let mut index: FxHashMap<Vec<char>, usize> = FxHashMap::default();
        let mut columns: Vec<Vec<char>> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        for col in 0..length {
            let key: Vec<char> = seqs.iter().map(|s| upper_char(s[col])).collect();
            match index.get(&key) {
                Some(&i) => weights[i] += 1.0,
                None => {
                    index.insert(key.clone(), columns.len());
                    columns.push(key);
                    weights.push(1.0);
                }
            }
        }
        (columns, weights)
    }

    /// Install compressed columns (``columns[p][r]``) and their weights.
    fn install(&mut self, taxa: Vec<String>, columns: &[Vec<char>], weights: Vec<f64>) {
        let n_taxa = taxa.len();
        let n = columns.len();
        let mut patterns: Vec<Vec<char>> = vec![Vec::with_capacity(n); n_taxa];
        let mut tips: Vec<Vec<f64>> = vec![Vec::with_capacity(n * 4); n_taxa];
        for col in columns {
            for (r, &ch) in col.iter().enumerate() {
                patterns[r].push(ch);
                tips[r].extend_from_slice(&mask_partials(iupac_mask(ch)));
            }
        }
        self.row_of = taxa.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();
        self.taxa = taxa;
        self.n_patterns = n;
        self.weights = weights;
        self.patterns = patterns;
        self.tips = tips;
    }

    /// Post-order pruning over ``net`` from ``root``; ``pmat(t)`` supplies
    /// ``P(t)`` for a branch of length ``t`` (already scaled by the site rate).
    fn prune(
        &self,
        py: Python<'_>,
        net: &Network,
        root: u64,
        site_rate: f64,
        pi: &[f64; 4],
        pmat: &mut dyn FnMut(f64) -> PyResult<[[f64; 4]; 4]>,
    ) -> PyResult<f64> {
        let n = self.n_patterns;
        let width = n * 4;

        // Reversed DFS pre-order visits every child before its parent.
        let mut order: Vec<u64> = Vec::new();
        let mut stack = vec![root];
        while let Some(u) = stack.pop() {
            order.push(u);
            stack.extend(net.children(u));
        }

        let ones = vec![1.0; width];
        let leaf_partials = |uid: u64| -> &[f64] {
            match self.row_of.get(&net.label(py, uid)) {
                Some(&r) => &self.tips[r],
                // Leaf not in this locus's alignment -> fully ambiguous.
                None => &ones,
            }
        };

        let mut partials: FxHashMap<u64, Vec<f64>> = FxHashMap::default();
        for &u in order.iter().rev() {
            let kids = net.children(u);
            if kids.is_empty() {
                continue;
            }
            let mut result = vec![1.0; width];
            for c in kids {
                let euid = net.first_edge_between(py, u, c)?;
                let length = net.edge_obj(euid).borrow(py).length;
                let t = match length {
                    None => 0.0,
                    Some(l) => l * site_rate,
                };
                let p = pmat(t)?;
                let owned: Vec<f64>;
                let child: &[f64] = if net.children(c).is_empty() {
                    leaf_partials(c)
                } else if net.in_deg(c) <= 1 {
                    owned = partials.remove(&c).ok_or_else(|| {
                        PyRuntimeError::new_err("FelsensteinCalculator: missing child partials.")
                    })?;
                    &owned
                } else {
                    partials.get(&c).ok_or_else(|| {
                        PyRuntimeError::new_err("FelsensteinCalculator: missing child partials.")
                    })?
                };
                // (n_patterns,4) @ P.T: probability of the child's data given
                // each parent state; multiply across children (independent
                // lineages).
                for pat in 0..n {
                    let base = pat * 4;
                    let cp = &child[base..base + 4];
                    for i in 0..4 {
                        let s = cp[0] * p[i][0] + cp[1] * p[i][1] + cp[2] * p[i][2] + cp[3] * p[i][3];
                        result[base + i] *= s;
                    }
                }
            }
            partials.insert(u, result);
        }

        let root_partials: &[f64] = if net.children(root).is_empty() {
            leaf_partials(root)
        } else {
            &partials[&root]
        };
        let mut total = 0.0;
        for pat in 0..n {
            let base = pat * 4;
            let mut site = 0.0;
            for i in 0..4 {
                site += root_partials[base + i] * pi[i];
            }
            // Guard against underflow on pathological branch lengths.
            if site < 1e-300 {
                site = 1e-300;
            }
            total += site.ln() * self.weights[pat];
        }
        Ok(total)
    }
}

#[pymethods]
impl FelsensteinCalculator {
    /// Lenient constructor; ``__init__`` of the Python class calls
    /// ``_compress`` with the alignment.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn py_new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        let _ = (args, kwargs);
        Self::blank()
    }

    /// Compress ``alignment`` (leaf label -> aligned string) to its distinct
    /// site patterns and pre-build every taxon's tip partials.
    fn _compress(&mut self, alignment: &Bound<'_, PyAny>) -> PyResult<()> {
        if !alignment.is_truthy()? {
            return Err(PyValueError::new_err("Empty alignment passed to FelsensteinCalculator."));
        }
        let mut taxa: Vec<String> = Vec::new();
        let mut seqs: Vec<Vec<char>> = Vec::new();
        for key in alignment.call_method0("keys")?.try_iter()? {
            let key = key?;
            let label: String = match key.extract::<String>() {
                Ok(s) => s,
                Err(_) => key.str()?.to_string(),
            };
            let value = alignment.get_item(&key)?;
            let seq: String = value.extract().map_err(|_| {
                PyTypeError::new_err(format!("Sequence for taxon {label:?} must be a string."))
            })?;
            taxa.push(label);
            seqs.push(seq.chars().collect());
        }
        if seqs.is_empty() {
            return Err(PyValueError::new_err("Empty alignment passed to FelsensteinCalculator."));
        }
        let length = seqs[0].len();
        if seqs.iter().any(|s| s.len() != length) {
            return Err(PyValueError::new_err("All sequences in a locus must be equally long."));
        }
        let (columns, weights) = Self::compress_columns(&seqs);
        self.install(taxa, &columns, weights);
        Ok(())
    }

    /// Ordered list of leaf labels present in the alignment.
    #[getter]
    fn taxa(&self) -> Vec<String> {
        self.taxa.clone()
    }

    /// Number of distinct site patterns after compression.
    #[getter]
    fn n_patterns(&self) -> usize {
        self.n_patterns
    }

    /// Multiplicity of every site pattern (``float64``).
    #[getter]
    fn _weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np1(py, self.weights.clone())
    }

    /// Taxon -> ``(n_patterns, 4)`` tip partial-likelihood array.
    #[getter]
    fn _tip_partials<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (r, taxon) in self.taxa.iter().enumerate() {
            d.set_item(taxon, np2(py, self.n_patterns, 4, self.tips[r].clone()))?;
        }
        Ok(d)
    }

    /// Felsenstein log-likelihood of the alignment on ``gene_tree``.
    ///
    /// Args:
    ///     gene_tree: A rooted *tree* (one connected component, no
    ///         reticulations) whose leaves are labelled with this locus's
    ///         taxa and whose edges carry branch lengths in substitution
    ///         units.
    ///     model: The nucleotide substitution model.
    ///     site_rate: Optional per-locus relative mutation rate (PhyloNet's
    ///         ``-murate``); branch lengths are effectively scaled by this.
    ///         Defaults to 1.0 (rate variation off).
    ///
    /// Returns:
    ///     The summed log-likelihood over all (weighted) site patterns.
    #[pyo3(signature = (gene_tree, model, *, site_rate = 1.0))]
    fn log_likelihood(
        &self,
        py: Python<'_>,
        gene_tree: &Bound<'_, Network>,
        model: &Bound<'_, PyAny>,
        site_rate: f64,
    ) -> PyResult<f64> {
        let net = gene_tree.borrow();
        let root = net.root_uid(py)?;
        if native_p_matrix(py, model)? {
            let m = model.cast::<SubstitutionModel>()?.borrow();
            m.require_ready()?;
            let pi = m.pi;
            let mut p = [[0.0; 4]; 4];
            self.prune(py, &net, root, site_rate, &pi, &mut |t| {
                m.p_matrix_into(t, &mut p);
                Ok(p)
            })
        } else {
            let pi_v = float_vec(&model.getattr("pi")?)?;
            let pi = arr4(&pi_v, "model.pi")?;
            self.prune(py, &net, root, site_rate, &pi, &mut |t| {
                let p = model.call_method1("p_matrix", (t,))?;
                matrix4_of(&p, "p_matrix(t)")
            })
        }
    }

    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let d = PyDict::new(py);
        {
            let c = slf.borrow();
            d.set_item("taxa", c.taxa.clone())?;
            d.set_item("weights", c.weights.clone())?;
            let patterns: Vec<String> = c.patterns.iter().map(|p| p.iter().collect()).collect();
            d.set_item("patterns", patterns)?;
        }
        if let Ok(dict) = slf.getattr("__dict__") {
            d.set_item("__dict__", dict)?;
        }
        Ok(d)
    }

    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyDict>) -> PyResult<()> {
        {
            let mut c = slf.borrow_mut();
            let taxa: Vec<String> = match state.get_item("taxa")? {
                Some(v) => v.extract()?,
                None => Vec::new(),
            };
            let weights: Vec<f64> = match state.get_item("weights")? {
                Some(v) => v.extract()?,
                None => Vec::new(),
            };
            let patterns: Vec<String> = match state.get_item("patterns")? {
                Some(v) => v.extract()?,
                None => Vec::new(),
            };
            if patterns.len() != taxa.len() {
                return Err(PyValueError::new_err("FelsensteinCalculator state is inconsistent."));
            }
            let rows: Vec<Vec<char>> = patterns.iter().map(|s| s.chars().collect()).collect();
            let n = rows.first().map_or(0, Vec::len);
            if rows.iter().any(|r| r.len() != n) || weights.len() != n {
                return Err(PyValueError::new_err("FelsensteinCalculator state is inconsistent."));
            }
            let columns: Vec<Vec<char>> =
                (0..n).map(|p| rows.iter().map(|r| r[p]).collect()).collect();
            c.install(taxa, &columns, weights);
        }
        if let Some(dict) = state.get_item("__dict__")? {
            if let Ok(target) = slf.getattr("__dict__") {
                target.call_method1("update", (dict,))?;
            }
        }
        Ok(())
    }

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::empty(py)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// phynetpy.GTR numerical core
// ═══════════════════════════════════════════════════════════════════════

/// Map an off-diagonal cell ``(i, j)`` to its exchangeability index in
/// upper-triangular row-major order (symmetric in ``i``/``j``).
#[inline]
fn rate_index(i: usize, j: usize, states: usize) -> usize {
    let (i, j) = if i > j { (j, i) } else { (i, j) };
    i * (2 * states - i - 1) / 2 + (j - i - 1)
}

/// Eigen-factors of ``Q``: eigenvalues, right eigenvectors and their inverse.
struct Decomp {
    evals: DVector<f64>,
    u: DMatrix<f64>,
    uinv: DMatrix<f64>,
}

/// Normalised GTR rate matrix with a cached eigendecomposition: the
/// numerical core behind ``phynetpy.GTR.GTR`` (any number of states).
#[pyclass(name = "_GTRCore", module = "phynetpy.GTR")]
pub struct GtrCore {
    states: usize,
    freqs: Vec<f64>,
    rates: Vec<f64>,
    q: DMatrix<f64>,
    decomp: Option<Decomp>,
}

impl GtrCore {
    fn build(freqs: Vec<f64>, rates: Vec<f64>, states: usize) -> PyResult<Self> {
        let n = states;
        if freqs.len() != n {
            return Err(PyValueError::new_err(format!(
                "Base frequency list must have {n} entries; got {}.",
                freqs.len()
            )));
        }
        let proper = (n.saturating_sub(1) * n) / 2;
        if rates.len() != proper {
            return Err(PyValueError::new_err(format!(
                "Incorrect number of transition rates. Got {}. Expected {proper}!",
                rates.len()
            )));
        }
        // Off-diagonal Q[i][j] = r_ij * pi_j (Tavaré's parameterisation).
        let mut q = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    q[(i, j)] = rates[rate_index(i, j, n)] * freqs[j];
                }
            }
        }
        for i in 0..n {
            let mut row = 0.0;
            for j in 0..n {
                row += q[(i, j)];
            }
            q[(i, i)] = -row;
        }
        // Normalise such that -1 * SUM Q_ii * pi_i = 1.
        let mut dot = 0.0;
        for i in 0..n {
            dot += freqs[i] * q[(i, i)];
        }
        let mean_rate = -dot;
        if mean_rate <= 0.0 {
            return Err(PyValueError::new_err(
                "Degenerate substitution model: the overall substitution rate is not positive.",
            ));
        }
        q.iter_mut().for_each(|x| *x /= mean_rate);
        Ok(GtrCore {
            states: n,
            freqs,
            rates,
            q,
            decomp: None,
        })
    }

    /// Symmetric decomposition through ``B = P^(1/2) Q P^(-1/2)``; a zero
    /// frequency breaks the transform, so that case uses ``numpy.linalg.eig``
    /// exactly as the Python reference did.
    fn compute_decomp(&self, py: Python<'_>) -> PyResult<Decomp> {
        let n = self.states;
        if self.freqs.iter().all(|&f| f > 0.0) {
            let sqrt_pi: Vec<f64> = self.freqs.iter().map(|f| f.sqrt()).collect();
            let b = DMatrix::from_fn(n, n, |i, j| self.q[(i, j)] * sqrt_pi[i] / sqrt_pi[j]);
            let bs = DMatrix::from_fn(n, n, |i, j| 0.5 * (b[(i, j)] + b[(j, i)]));
            if bs.iter().any(|x| !x.is_finite()) {
                return Err(PyValueError::new_err(
                    "Substitution model rate matrix has non-finite entries.",
                ));
            }
            let eig = SymmetricEigen::try_new(bs, f64::EPSILON, MAX_EIGEN_ITER).ok_or_else(|| {
                PyValueError::new_err("Eigendecomposition of the rate matrix did not converge.")
            })?;
            let order = ascending(eig.eigenvalues.as_slice());
            let evals = DVector::from_fn(n, |k, _| eig.eigenvalues[order[k]]);
            let u = DMatrix::from_fn(n, n, |i, k| eig.eigenvectors[(i, order[k])] / sqrt_pi[i]);
            let uinv = DMatrix::from_fn(n, n, |k, j| eig.eigenvectors[(j, order[k])] * sqrt_pi[j]);
            return Ok(Decomp { evals, u, uinv });
        }
        let np = py.import("numpy")?;
        let la = np.getattr("linalg")?;
        let q_np = np2(py, n, n, Self::row_major(&self.q));
        let res = la.call_method1("eig", (q_np,))?;
        let w = np.call_method1("real", (res.get_item(0)?,))?;
        let v = np.call_method1("real", (res.get_item(1)?,))?;
        let vinv = la.call_method1("inv", (&v,))?;
        let w = float_vec(&w)?;
        if w.len() != n {
            return Err(PyValueError::new_err(
                "numpy.linalg.eig returned the wrong number of eigenvalues.",
            ));
        }
        Ok(Decomp {
            evals: DVector::from_vec(w),
            u: dmatrix_of(&v, n, "Eigenvector matrix")?,
            uinv: dmatrix_of(&vinv, n, "Inverse eigenvector matrix")?,
        })
    }

    fn ensure_decomp(&mut self, py: Python<'_>) -> PyResult<&Decomp> {
        if self.decomp.is_none() {
            self.decomp = Some(self.compute_decomp(py)?);
        }
        Ok(self.decomp.as_ref().expect("just computed"))
    }

    fn row_major(m: &DMatrix<f64>) -> Vec<f64> {
        let (r, c) = m.shape();
        let mut out = Vec::with_capacity(r * c);
        for i in 0..r {
            for j in 0..c {
                out.push(m[(i, j)]);
            }
        }
        out
    }
}

#[pymethods]
impl GtrCore {
    #[new]
    fn py_new(freqs: Vec<f64>, rates: Vec<f64>, states: usize) -> PyResult<Self> {
        Self::build(freqs, rates, states)
    }

    /// The normalised rate matrix as a fresh ``states x states`` array.
    fn q_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.states;
        np2(py, n, n, Self::row_major(&self.q))
    }

    /// Eigenvalues, right eigenvectors and their inverse (cached).
    fn decompose<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let n = self.states;
        let d = self.ensure_decomp(py)?;
        Ok((
            np1(py, d.evals.iter().copied().collect()),
            np2(py, n, n, Self::row_major(&d.u)),
            np2(py, n, n, Self::row_major(&d.uinv)),
        ))
    }

    /// ``e^(Q t)`` from the cached eigendecomposition (``t < 0`` clamps to 0).
    fn expt<'py>(&mut self, py: Python<'py>, t: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let t = if t < 0.0 { 0.0 } else { t };
        let n = self.states;
        let d = self.ensure_decomp(py)?;
        let e: Vec<f64> = d.evals.iter().map(|&l| (l * t).exp()).collect();
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += d.u[(i, k)] * e[k] * d.uinv[(k, j)];
                }
                out[i * n + j] = s;
            }
        }
        Ok(np2(py, n, n, out))
    }

    fn __getnewargs__(&self) -> (Vec<f64>, Vec<f64>, usize) {
        (self.freqs.clone(), self.rates.clone(), self.states)
    }
}

fn freqs4(freqs: &[f64]) -> PyResult<[f64; 4]> {
    arr4(freqs, "Base frequencies")
}

/// Closed-form Jukes-Cantor transition matrix.
#[pyfunction]
#[pyo3(name = "_gtr_jc_p_matrix")]
fn jc_p_matrix(py: Python<'_>, t: f64) -> Bound<'_, PyArray2<f64>> {
    let decay = (-4.0 * t / 3.0).exp();
    let same = 0.25 + 0.75 * decay;
    let diff = 0.25 - 0.25 * decay;
    let mut out = [[diff; 4]; 4];
    for i in 0..4 {
        out[i][i] = same;
    }
    np4x4(py, &out)
}

/// Closed-form Felsenstein (1981) transition matrix.
#[pyfunction]
#[pyo3(name = "_gtr_f81_p_matrix")]
fn f81_p_matrix(py: Python<'_>, freqs: Vec<f64>, t: f64) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let [pi_a, pi_c, pi_g, pi_t] = freqs4(&freqs)?;
    let u = 1.0 / (1.0 - (pi_a * pi_a + pi_c * pi_c + pi_g * pi_g + pi_t * pi_t));
    let decay = (-u * t).exp();
    let spread = 1.0 - decay;
    let (a, c, g, tt) = (pi_a * spread, pi_c * spread, pi_g * spread, pi_t * spread);
    let out = [
        [a + decay, c, g, tt],
        [a, c + decay, g, tt],
        [a, c, g + decay, tt],
        [a, c, g, tt + decay],
    ];
    Ok(np4x4(py, &out))
}

/// Closed-form transition matrix for the Tamura-Nei (1993) family (TN93,
/// HKY and K80).
#[pyfunction]
#[pyo3(name = "_gtr_tn93_p_matrix")]
fn tn93_p_matrix(
    py: Python<'_>,
    freqs: Vec<f64>,
    kappa_r: f64,
    kappa_y: f64,
    t: f64,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let [pi_a, pi_c, pi_g, pi_t] = freqs4(&freqs)?;
    let pi_r = pi_a + pi_g;
    let pi_y = pi_c + pi_t;

    // Scale so that -sum_i pi_i Q_ii == 1.
    let mean_rate = 2.0 * (pi_a * pi_g * kappa_r + pi_c * pi_t * kappa_y + pi_r * pi_y);
    let alpha_r = kappa_r / mean_rate;
    let alpha_y = kappa_y / mean_rate;
    let beta = 1.0 / mean_rate;

    // Two decay modes: the transversion eigenvalue (-beta), shared by both
    // groups, and the within-group eigenvalue, which differs between purines
    // and pyrimidines.
    let decay_tv = (-beta * t).exp();
    let decay_r = (-(pi_r * alpha_r + (1.0 - pi_r) * beta) * t).exp();
    let decay_y = (-(pi_y * alpha_y + (1.0 - pi_y) * beta) * t).exp();

    // Every same-group entry is pi_j times a per-group coefficient, and the
    // diagonal is that same product plus the group's decay term.  Every
    // cross-group entry is pi_j * (1 - decay_tv).
    let coef_r = 1.0 + (1.0 / pi_r - 1.0) * decay_tv - decay_r / pi_r;
    let coef_y = 1.0 + (1.0 / pi_y - 1.0) * decay_tv - decay_y / pi_y;
    let cross = 1.0 - decay_tv;

    let (xa, xc, xg, xt) = (pi_a * cross, pi_c * cross, pi_g * cross, pi_t * cross);
    let (ra, rg) = (pi_a * coef_r, pi_g * coef_r);
    let (yc, yt) = (pi_c * coef_y, pi_t * coef_y);

    //            A              C              G              T
    let out = [
        [ra + decay_r, xc, rg, xt],
        [xa, yc + decay_y, xg, yt],
        [ra, xc, rg + decay_r, xt],
        [xa, yc, xg, yt + decay_y],
    ];
    Ok(np4x4(py, &out))
}

/// Closed-form transition matrix for Kimura's 3-substitution-type model
/// (K81), diagonalised by the characters of the Klein four-group.
#[pyfunction]
#[pyo3(name = "_gtr_k3st_p_matrix")]
fn k3st_p_matrix(py: Python<'_>, alpha: f64, beta: f64, gamma: f64, t: f64) -> Bound<'_, PyArray2<f64>> {
    // With uniform frequencies the normalisation is just the rate sum.
    let total = alpha + beta + gamma;
    let (a, b, g) = (alpha / total, beta / total, gamma / total);

    let e_ti = (-2.0 * (a + g) * t).exp();
    let e_ac = (-2.0 * (b + g) * t).exp();
    let e_at = (-2.0 * (a + b) * t).exp();

    let same = 0.25 * (1.0 + e_ti + e_ac + e_at);
    let p_ti = 0.25 * (1.0 - e_ti + e_ac - e_at);
    let p_ac = 0.25 * (1.0 + e_ti - e_ac - e_at);
    let p_at = 0.25 * (1.0 - e_ti - e_ac + e_at);

    //         A      C      G      T
    let out = [
        [same, p_ac, p_ti, p_at],
        [p_ac, same, p_at, p_ti],
        [p_ti, p_at, same, p_ac],
        [p_at, p_ti, p_ac, same],
    ];
    np4x4(py, &out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SubstitutionModel>()?;
    m.add_class::<FelsensteinCalculator>()?;
    m.add_class::<GtrCore>()?;
    m.add_function(wrap_pyfunction!(tip_partials_for_char, m)?)?;
    m.add_function(wrap_pyfunction!(jc_p_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(f81_p_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(tn93_p_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(k3st_p_matrix, m)?)?;
    m.add("DNA_STATES", DNA_STATES)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_index_is_upper_triangular_row_major() {
        let cells = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        for (k, (i, j)) in cells.iter().enumerate() {
            assert_eq!(rate_index(*i, *j, 4), k);
            assert_eq!(rate_index(*j, *i, 4), k);
        }
    }

    #[test]
    fn jc_closed_form_matches_eigen_form() {
        let mut m = SubstitutionModel::blank();
        m.set_params([0.25; 4], [1.0; 6]).unwrap();
        let q = m.build_q();
        m.decompose_from(q).unwrap();
        let mut eigen = [[0.0; 4]; 4];
        m.p_matrix_into(0.37, &mut eigen);
        m.jc = true;
        let mut closed = [[0.0; 4]; 4];
        m.p_matrix_into(0.37, &mut closed);
        for i in 0..4 {
            for j in 0..4 {
                assert!((eigen[i][j] - closed[i][j]).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn p_matrix_rows_are_stochastic_and_reversible() {
        let pi = [0.37, 0.23, 0.27, 0.13];
        let mut m = SubstitutionModel::blank();
        m.set_params(pi, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let q = m.build_q();
        m.decompose_from(q).unwrap();
        let mut p = [[0.0; 4]; 4];
        m.p_matrix_into(0.4, &mut p);
        for i in 0..4 {
            let row: f64 = p[i].iter().sum();
            assert!((row - 1.0).abs() < 1e-12);
            for j in 0..4 {
                assert!((pi[i] * p[i][j] - pi[j] * p[j][i]).abs() < 1e-12);
            }
        }
        let mean: f64 = (0..4).map(|i| pi[i] * m.q[i][i]).sum();
        assert!((mean + 1.0).abs() < 1e-12);
    }
}
