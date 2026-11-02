//! Biallelic-marker (SNP) likelihood kernels (Bryant et al. 2012; Zhu et
//! al. 2018).
//!
//! Everything numerical behind ``phynetpy.BiMarkers`` lives here:
//!
//! * the ``(n, r)`` lineage / red-allele-count rate matrix ``Q`` (Bryant
//!   eq. 15) and ``e^{Qt}`` (scaling-and-squaring Padé, the algorithm behind
//!   ``scipy.linalg.expm``), cached per ``(n, u, v, theta)`` and ``t``;
//! * the root stationary prior: the null vector of ``Q`` restricted to the
//!   ``n >= 1`` states (Bryant eq. 20), computed by exact block
//!   back-substitution with an SVD fallback;
//! * the sparse hypergeometric merge table (rules 2 / 4) and the binomial
//!   split table (rule 3) of the network recursion;
//! * the rescaled partial-likelihood tensor operations (rules 1-4,
//!   per-vector and per-site rescaling, scale equalisation, axis moves);
//! * the bottom-up DP over the species network that threads those partials
//!   from the leaves to the root and sums ``log P(site | network)``.
//!
//! The Python classes ``SNPStrategy`` / ``SNPModelVisitor`` keep their
//! surface and call the ``_snp_*`` kernels below on NumPy arrays; the
//! likelihood entry point ``_snp_log_likelihood`` calls
//! ``_snp_log_likelihood_core``, which runs the whole DP natively.
//!
//! Tensor layout: a partial-likelihood tensor has shape ``(S, d1, .., dk)``
//! (sites first, one axis per open interface, row-major) and carries a
//! ``log_scale`` tensor of shape ``(S,)`` or ``(S, d1, .., d_{k-1})``.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use indexmap::IndexMap;
use nalgebra::{DMatrix, DMatrixView, DVector};
use numpy::ndarray::{Array1, Array2, ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyAssertionError, PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

use crate::graph::Network;

// ═══════════════════════════════════════════════════════════════════════
// State indexing
// ═══════════════════════════════════════════════════════════════════════

/// First linear index of the ``n``-lineage block; ``(0, 0)`` is index 0.
#[inline]
pub fn n_to_index(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Linear index of the ``(n, r)`` state.
#[inline]
pub fn nr_to_index(n: usize, r: usize) -> usize {
    n_to_index(n) + r
}

/// Number of ``(n, r)`` states with ``0 <= n <= m``: ``(m + 1)(m + 2) / 2``.
#[inline]
pub fn state_dim(m: usize) -> usize {
    nr_to_index(m, m) + 1
}

// ═══════════════════════════════════════════════════════════════════════
// Rate matrix and its exponential
// ═══════════════════════════════════════════════════════════════════════

/// Bryant et al. (2012) eq. 15 over the ``(n, r)`` states, ``0 <= n <= nmax``.
///
/// Index 0 is the absorbing empty state ``(0, 0)``: its row and column stay
/// zero so ``e^{Qt}`` maps an empty branch to itself.  The matrix is block
/// lower triangular in ``n`` (coalescence only lowers the lineage count),
/// which is what makes the leading ``state_dim(m)`` block of ``e^{Qt}`` the
/// exponential of the leading block of ``Q``.
pub fn build_rate_matrix(nmax: usize, u: f64, v: f64, theta: f64) -> DMatrix<f64> {
    let d = state_dim(nmax);
    let mut q = DMatrix::<f64>::zeros(d, d);
    for np in 1..=nmax {
        for rp in 0..=np {
            let n_r = nr_to_index(np, rp);
            q[(n_r, n_r)] =
                -((np * (np - 1)) as f64 / theta) - (v * (np - rp) as f64) - (rp as f64 * u);
            if rp > 0 {
                if np > 1 {
                    q[(n_r, nr_to_index(np - 1, rp - 1))] = ((rp - 1) * np) as f64 / theta;
                }
                q[(n_r, nr_to_index(np, rp - 1))] = (np - rp + 1) as f64 * v;
            }
            if rp < np {
                if np > 1 {
                    q[(n_r, nr_to_index(np - 1, rp))] = ((np - 1 - rp) * np) as f64 / theta;
                }
                q[(n_r, nr_to_index(np, rp + 1))] = (rp + 1) as f64 * u;
            }
        }
    }
    q
}

/// ``e^{Q t}`` by scaling-and-squaring Padé (nalgebra's port of
/// ``scipy.linalg.expm``).
pub fn expm(q: &DMatrix<f64>, t: f64) -> DMatrix<f64> {
    (q * t).exp()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct RateKey {
    n: usize,
    u: u64,
    v: u64,
    theta: u64,
}

impl RateKey {
    fn new(n: usize, u: f64, v: f64, theta: f64) -> Self {
        RateKey {
            n,
            u: u.to_bits(),
            v: v.to_bits(),
            theta: theta.to_bits(),
        }
    }
}

/// Process-wide caches: rate matrices and their stationary vectors per
/// ``(n, u, v, theta)`` (the Python side's ``lru_cache`` on
/// ``_get_transition``), ``e^{Qt}`` per branch time (most branch lengths
/// are unchanged between MCMC proposals), and the sparse coefficient
/// tables.
#[derive(Default)]
struct SnpCaches {
    rates: IndexMap<RateKey, Arc<DMatrix<f64>>>,
    expt: IndexMap<(RateKey, u64), Arc<DMatrix<f64>>>,
    expt_bytes: usize,
    stationary: IndexMap<RateKey, Arc<Vec<f64>>>,
    merges: FxHashMap<(usize, usize), Arc<SparseMerge>>,
    splits: IndexMap<(usize, u64), Arc<SparseSplit>>,
}

const RATE_CACHE_LIMIT: usize = 128;
const EXPT_CACHE_BYTES: usize = 256 * 1024 * 1024;
const SPLIT_CACHE_LIMIT: usize = 512;

static CACHES: LazyLock<Mutex<SnpCaches>> = LazyLock::new(|| Mutex::new(SnpCaches::default()));

fn caches() -> MutexGuard<'static, SnpCaches> {
    CACHES.lock().unwrap_or_else(|e| e.into_inner())
}

/// Cached ``Q`` for ``nmax`` lineages.
pub fn rate_matrix(nmax: usize, u: f64, v: f64, theta: f64) -> Arc<DMatrix<f64>> {
    let key = RateKey::new(nmax, u, v, theta);
    if let Some(q) = caches().rates.get(&key) {
        return q.clone();
    }
    let q = Arc::new(build_rate_matrix(nmax, u, v, theta));
    let mut c = caches();
    while c.rates.len() >= RATE_CACHE_LIMIT {
        c.rates.shift_remove_index(0);
    }
    c.rates.entry(key).or_insert_with(|| q.clone()).clone()
}

/// Cached ``e^{Q t}`` for ``nmax`` lineages (a ``state_dim(nmax)`` square
/// matrix).
pub fn transition(nmax: usize, u: f64, v: f64, theta: f64, t: f64) -> Arc<DMatrix<f64>> {
    let key = (RateKey::new(nmax, u, v, theta), t.to_bits());
    if let Some(p) = caches().expt.get(&key) {
        return p.clone();
    }
    let q = rate_matrix(nmax, u, v, theta);
    let p = Arc::new(expm(&q, t));
    let bytes = p.len() * std::mem::size_of::<f64>();
    let mut c = caches();
    if bytes <= EXPT_CACHE_BYTES {
        while c.expt_bytes + bytes > EXPT_CACHE_BYTES && !c.expt.is_empty() {
            if let Some((_, old)) = c.expt.shift_remove_index(0) {
                c.expt_bytes -= old.len() * std::mem::size_of::<f64>();
            }
        }
        if c.expt.insert(key, p.clone()).is_none() {
            c.expt_bytes += bytes;
        }
    }
    p
}

// ═══════════════════════════════════════════════════════════════════════
// Root stationary prior
// ═══════════════════════════════════════════════════════════════════════

/// Null vector of ``Q[1:d, 1:d]`` by block back-substitution.
///
/// ``Q`` is block lower triangular in ``n``; the ``n = 1`` block
/// ``[[-v, u], [v, -u]]`` is singular with null vector ``(u, v)``, and every
/// later block ``M_n`` (eigenvalues ``-n(n-1)/theta - k(u+v)``) is
/// invertible, so ``x_n = -M_n^{-1} C_n x_{n-1}``.
fn null_vector_blockwise(q: &DMatrix<f64>, m: usize) -> Option<Vec<f64>> {
    let d = state_dim(m);
    let mut x = vec![0.0; d];
    let (a, b) = (q[(1, 1)], q[(1, 2)]);
    let (c, e) = (q[(2, 1)], q[(2, 2)]);
    if a != 0.0 || b != 0.0 {
        x[1] = b;
        x[2] = -a;
    } else if c != 0.0 || e != 0.0 {
        x[1] = e;
        x[2] = -c;
    } else {
        x[1] = 1.0;
    }
    for n in 2..=m {
        let sz = n + 1;
        let base = n_to_index(n);
        let pbase = n_to_index(n - 1);
        let mut block = DMatrix::<f64>::zeros(sz, sz);
        let mut rhs = DVector::<f64>::zeros(sz);
        for r in 0..sz {
            for r2 in 0..sz {
                block[(r, r2)] = q[(base + r, base + r2)];
            }
            let mut acc = 0.0;
            for r2 in 0..n {
                acc += q[(base + r, pbase + r2)] * x[pbase + r2];
            }
            rhs[r] = -acc;
        }
        let sol = block.lu().solve(&rhs)?;
        for r in 0..sz {
            x[base + r] = sol[r];
        }
    }
    Some(x)
}

/// ``max |Q[1:d,1:d] x| <= 1e-8 * max|Q| * max|x|``.
fn residual_ok(q: &DMatrix<f64>, x: &[f64], d: usize) -> bool {
    let mut qn: f64 = 0.0;
    let mut xn: f64 = 0.0;
    let mut res: f64 = 0.0;
    for i in 1..d {
        let mut acc = 0.0;
        for j in 1..d {
            let v = q[(i, j)];
            qn = qn.max(v.abs());
            acc += v * x[j];
        }
        res = res.max(acc.abs());
        xn = xn.max(x[i].abs());
    }
    x.iter().all(|v| v.is_finite()) && res <= 1e-8 * qn * xn + f64::MIN_POSITIVE
}

/// Null vector of ``Q[1:d, 1:d]`` from its SVD (what ``scipy.linalg.null_space``
/// does): the right singular vector of the smallest singular value.
fn null_vector_svd(q: &DMatrix<f64>, d: usize) -> PyResult<Vec<f64>> {
    let sub = q.view((1, 1), (d - 1, d - 1)).into_owned();
    let svd = sub
        .try_svd(false, true, f64::EPSILON, 100_000)
        .ok_or_else(|| PyValueError::new_err("Biallelic rate matrix has no stationary distribution"))?;
    let vt = svd
        .v_t
        .ok_or_else(|| PyValueError::new_err("Biallelic rate matrix has no stationary distribution"))?;
    let sv = &svd.singular_values;
    let mut best = 0;
    for i in 1..sv.len() {
        if sv[i] < sv[best] {
            best = i;
        }
    }
    let mut x = vec![0.0; d];
    for j in 0..d - 1 {
        x[1 + j] = vt[(best, j)];
    }
    Ok(x)
}

/// Root allele-frequency prior over the ``(n, r)`` states, ``0 <= n <= m``:
/// the null vector of ``Q`` restricted to ``n >= 1``, normalised so the
/// single-lineage block ``{(1, 0), (1, 1)}`` sums to one, with
/// ``pi[(0, 0)] = 0``.
pub fn root_stationary(q: &DMatrix<f64>, m: usize) -> PyResult<Vec<f64>> {
    let d = state_dim(m);
    if m == 0 || q.nrows() < d || q.ncols() < d {
        return Err(PyValueError::new_err(
            "Biallelic rate matrix has no stationary distribution",
        ));
    }
    let x = match null_vector_blockwise(q, m) {
        Some(x) if residual_ok(q, &x, d) => x,
        _ => null_vector_svd(q, d)?,
    };
    let s = x[1] + x[2];
    if s == 0.0 || !s.is_finite() {
        return Err(PyValueError::new_err(
            "Biallelic rate matrix has no stationary distribution",
        ));
    }
    let mut pi = vec![0.0; d];
    for i in 1..d {
        pi[i] = x[i] / s;
    }
    Ok(pi)
}

/// Cached [`root_stationary`] for the ``m``-lineage rate matrix.
pub fn stationary(m: usize, u: f64, v: f64, theta: f64) -> PyResult<Arc<Vec<f64>>> {
    let key = RateKey::new(m, u, v, theta);
    if let Some(p) = caches().stationary.get(&key) {
        return Ok(p.clone());
    }
    let q = rate_matrix(m, u, v, theta);
    let pi = Arc::new(root_stationary(&q, m)?);
    let mut c = caches();
    while c.stationary.len() >= RATE_CACHE_LIMIT {
        c.stationary.shift_remove_index(0);
    }
    c.stationary.insert(key, pi.clone());
    Ok(pi)
}

// ═══════════════════════════════════════════════════════════════════════
// Sparse merge / split coefficient tables
// ═══════════════════════════════════════════════════════════════════════

/// ``C(n, k)`` exactly in 128-bit arithmetic, ``None`` on overflow.
fn comb_u128(n: usize, k: usize) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut r: u128 = 1;
    for i in 0..k {
        r = r.checked_mul((n - i) as u128)?;
        r /= (i + 1) as u128;
    }
    Some(r)
}

/// ``C(n, k)`` as a float (exact while it fits in 128 bits).
fn comb_f64(n: usize, k: usize) -> f64 {
    match comb_u128(n, k) {
        Some(v) => v as f64,
        None => {
            let k = k.min(n - k);
            let mut r = 1.0;
            for i in 0..k {
                r = r * (n - i) as f64 / (i + 1) as f64;
            }
            r
        }
    }
}

/// Hypergeometric merge coefficient ``C(rz, rx) C(nz - rz, nx - rx) / C(nz, nx)``.
fn merge_coeff(nz: usize, rz: usize, nx: usize, rx: usize) -> f64 {
    if let (Some(a), Some(b), Some(c)) =
        (comb_u128(rz, rx), comb_u128(nz - rz, nx - rx), comb_u128(nz, nx))
    {
        if let Some(num) = a.checked_mul(b) {
            return num as f64 / c as f64;
        }
    }
    comb_f64(rz, rx) * comb_f64(nz - rz, nx - rx) / comb_f64(nz, nx)
}

/// COO merge table ``M[i, j, k]`` (rules 2 and 4): one entry per
/// ``(i, j)`` state pair, ``k = (nx + ny, rx + ry)``.
pub struct SparseMerge {
    pub i: Vec<u32>,
    pub j: Vec<u32>,
    pub k: Vec<u32>,
    pub c: Vec<f64>,
    pub dim_x: usize,
    pub dim_y: usize,
    pub dim_z: usize,
    pub mx: usize,
    pub my: usize,
}

pub fn build_sparse_merge(mx: usize, my: usize) -> SparseMerge {
    let nnz = state_dim(mx) * state_dim(my);
    let mut sm = SparseMerge {
        i: Vec::with_capacity(nnz),
        j: Vec::with_capacity(nnz),
        k: Vec::with_capacity(nnz),
        c: Vec::with_capacity(nnz),
        dim_x: state_dim(mx),
        dim_y: state_dim(my),
        dim_z: state_dim(mx + my),
        mx,
        my,
    };
    for nx in 0..=mx {
        for rx in 0..=nx {
            let i = nr_to_index(nx, rx);
            for ny in 0..=my {
                for ry in 0..=ny {
                    let j = nr_to_index(ny, ry);
                    let nz = nx + ny;
                    let rz = rx + ry;
                    sm.i.push(i as u32);
                    sm.j.push(j as u32);
                    sm.k.push(nr_to_index(nz, rz) as u32);
                    sm.c.push(merge_coeff(nz, rz, nx, rx));
                }
            }
        }
    }
    sm
}

/// COO split table ``S[i, j, k]`` (rule 3): ``i = (n, r)`` splits into
/// ``j = (nb, rb)`` and ``k = (nd, rd)`` with coefficient
/// ``C(n, nb) gamma^nb (1 - gamma)^nd``; the boundary cases ``nb = 0`` and
/// ``nb = n`` are included.
pub struct SparseSplit {
    pub i: Vec<u32>,
    pub j: Vec<u32>,
    pub k: Vec<u32>,
    pub c: Vec<f64>,
    pub dim: usize,
    pub m: usize,
}

pub fn build_sparse_split(m: usize, gamma: f64) -> SparseSplit {
    let mut ss = SparseSplit {
        i: Vec::new(),
        j: Vec::new(),
        k: Vec::new(),
        c: Vec::new(),
        dim: state_dim(m),
        m,
    };
    let one_minus = 1.0 - gamma;
    for n in 0..=m {
        for r in 0..=n {
            let i = nr_to_index(n, r);
            for nb in 0..=n {
                let nd = n - nb;
                let lo = r.saturating_sub(nd);
                let hi = r.min(nb);
                if lo > hi {
                    continue;
                }
                let cnb = comb_f64(n, nb);
                for rb in lo..=hi {
                    let rd = r - rb;
                    ss.i.push(i as u32);
                    ss.j.push(nr_to_index(nb, rb) as u32);
                    ss.k.push(nr_to_index(nd, rd) as u32);
                    ss.c.push(cnb * gamma.powf(nb as f64) * one_minus.powf(nd as f64));
                }
            }
        }
    }
    ss
}

/// Cached merge table.
pub fn sparse_merge(mx: usize, my: usize) -> Arc<SparseMerge> {
    if let Some(sm) = caches().merges.get(&(mx, my)) {
        return sm.clone();
    }
    let sm = Arc::new(build_sparse_merge(mx, my));
    caches().merges.entry((mx, my)).or_insert_with(|| sm.clone()).clone()
}

/// Cached split table (bounded: ``gamma`` varies continuously under MCMC).
pub fn sparse_split(m: usize, gamma: f64) -> Arc<SparseSplit> {
    let key = (m, gamma.to_bits());
    if let Some(ss) = caches().splits.get(&key) {
        return ss.clone();
    }
    let ss = Arc::new(build_sparse_split(m, gamma));
    let mut c = caches();
    while c.splits.len() >= SPLIT_CACHE_LIMIT {
        c.splits.shift_remove_index(0);
    }
    c.splits.entry(key).or_insert_with(|| ss.clone()).clone()
}

// ═══════════════════════════════════════════════════════════════════════
// Tensors
// ═══════════════════════════════════════════════════════════════════════

/// A row-major dense tensor; axis 0 is the site axis.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn zeros(shape: Vec<usize>) -> Tensor {
        let n = shape.iter().product();
        Tensor {
            shape,
            data: vec![0.0; n],
        }
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Length of the last axis (1 for a scalar).
    #[inline]
    pub fn last(&self) -> usize {
        self.shape.last().copied().unwrap_or(1)
    }

    /// Product of every axis but the last.
    #[inline]
    pub fn prefix_len(&self) -> usize {
        self.shape[..self.ndim().saturating_sub(1)].iter().product()
    }

    fn shape_str(&self) -> String {
        match self.shape.len() {
            0 => "()".to_string(),
            1 => format!("({},)", self.shape[0]),
            _ => format!(
                "({})",
                self.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
            ),
        }
    }
}

#[inline]
fn product(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// Rule 1: ``F @ P`` along the last axis (``P`` is the leading ``d x d``
/// block of a transition matrix, ``d = F.shape[-1]``).
pub fn rule1(f: &Tensor, p: &DMatrix<f64>) -> PyResult<Tensor> {
    let d = f.last();
    if f.ndim() == 0 || p.nrows() < d || p.ncols() < d {
        return Err(PyValueError::new_err(format!(
            "transition matrix of size {}x{} cannot act on a tensor with last axis {d}",
            p.nrows(),
            p.ncols()
        )));
    }
    let rows = f.prefix_len();
    if rows == 0 || d == 0 {
        return Ok(Tensor::zeros(f.shape.clone()));
    }
    // Row-major (rows x d) data is column-major (d x rows) = F^T, so the
    // row-major result F P is the column-major product P^T F^T.
    let pt = p.view((0, 0), (d, d)).transpose();
    let ft = DMatrixView::from_slice(&f.data, d, rows);
    let out = &pt * ft;
    Ok(Tensor {
        shape: f.shape.clone(),
        data: out.data.into(),
    })
}

/// Rule 2: merge two partials from disjoint sub-networks.
/// ``(S, *xe, dim_x) x (S, *ye, dim_y) -> (S, *xe, *ye, dim_z)``.
pub fn rule2(fx: &Tensor, fy: &Tensor, sm: &SparseMerge) -> PyResult<Tensor> {
    if fx.ndim() < 2 || fy.ndim() < 2 {
        return Err(PyValueError::new_err("rule 2 needs tensors of shape (S, ..., d)"));
    }
    if fx.shape[0] != fy.shape[0] {
        return Err(PyValueError::new_err(format!(
            "rule 2: site counts differ ({} vs {})",
            fx.shape[0], fy.shape[0]
        )));
    }
    if fx.last() != sm.dim_x || fy.last() != sm.dim_y {
        return Err(PyValueError::new_err(format!(
            "rule 2: tensor state axes ({}, {}) do not match the merge table ({}, {})",
            fx.last(),
            fy.last(),
            sm.dim_x,
            sm.dim_y
        )));
    }
    let s = fx.shape[0];
    let xe = &fx.shape[1..fx.ndim() - 1];
    let ye = &fy.shape[1..fy.ndim() - 1];
    let (na, nb) = (product(xe), product(ye));
    let (dx, dy, dz) = (sm.dim_x, sm.dim_y, sm.dim_z);
    let mut shape = Vec::with_capacity(1 + xe.len() + ye.len() + 1);
    shape.push(s);
    shape.extend_from_slice(xe);
    shape.extend_from_slice(ye);
    shape.push(dz);
    let mut out = Tensor::zeros(shape);
    let (ti, tj, tk, tc) = (&sm.i, &sm.j, &sm.k, &sm.c);
    for site in 0..s {
        for a in 0..na {
            let fxr = &fx.data[(site * na + a) * dx..][..dx];
            for b in 0..nb {
                let fyr = &fy.data[(site * nb + b) * dy..][..dy];
                let o = &mut out.data[((site * na + a) * nb + b) * dz..][..dz];
                for (((&i, &j), &k), &c) in ti.iter().zip(tj).zip(tk).zip(tc) {
                    // Table indices are built below their dimensions.
                    unsafe {
                        *o.get_unchecked_mut(k as usize) +=
                            (*fxr.get_unchecked(i as usize) * *fyr.get_unchecked(j as usize)) * c;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Rule 3: split the last interface at a reticulation.
/// ``(..., dim) -> (..., dim, dim)``.
pub fn rule3(f: &Tensor, ss: &SparseSplit) -> PyResult<Tensor> {
    if f.ndim() < 1 || f.last() != ss.dim {
        return Err(PyValueError::new_err(format!(
            "rule 3: tensor state axis {} does not match the split table ({})",
            f.last(),
            ss.dim
        )));
    }
    let dim = ss.dim;
    let np = f.prefix_len();
    let mut shape = f.shape[..f.ndim() - 1].to_vec();
    shape.push(dim);
    shape.push(dim);
    let mut out = Tensor::zeros(shape);
    let (ti, tj, tk, tc) = (&ss.i, &ss.j, &ss.k, &ss.c);
    for p in 0..np {
        let fr = &f.data[p * dim..][..dim];
        let o = &mut out.data[p * dim * dim..][..dim * dim];
        for (((&i, &j), &k), &c) in ti.iter().zip(tj).zip(tk).zip(tc) {
            unsafe {
                *o.get_unchecked_mut(j as usize * dim + k as usize) += *fr.get_unchecked(i as usize) * c;
            }
        }
    }
    Ok(out)
}

/// Rule 4: merge the last two interfaces of one partial.
/// ``(..., dim_x, dim_y) -> (..., dim_z)``.
pub fn rule4(f: &Tensor, sm: &SparseMerge) -> PyResult<Tensor> {
    if f.ndim() < 2 || f.shape[f.ndim() - 2] != sm.dim_x || f.last() != sm.dim_y {
        return Err(PyValueError::new_err(format!(
            "rule 4: tensor state axes do not match the merge table ({}, {})",
            sm.dim_x, sm.dim_y
        )));
    }
    let (dx, dy, dz) = (sm.dim_x, sm.dim_y, sm.dim_z);
    let np = product(&f.shape[..f.ndim() - 2]);
    let mut shape = f.shape[..f.ndim() - 2].to_vec();
    shape.push(dz);
    let mut out = Tensor::zeros(shape);
    let (ti, tj, tk, tc) = (&sm.i, &sm.j, &sm.k, &sm.c);
    for p in 0..np {
        let fr = &f.data[p * dx * dy..][..dx * dy];
        let o = &mut out.data[p * dz..][..dz];
        for (((&i, &j), &k), &c) in ti.iter().zip(tj).zip(tk).zip(tc) {
            unsafe {
                *o.get_unchecked_mut(k as usize) += *fr.get_unchecked(i as usize * dy + j as usize) * c;
            }
        }
    }
    Ok(out)
}

/// Per-vector rescaling: every last-axis vector is divided by its max
/// absolute value (left alone when all-zero) and the log factors are
/// accumulated into a ``log_scale`` of shape ``F.shape[:-1]``.  The incoming
/// ``log_scale`` must be a leading sub-shape of that (``(S,)`` or the full
/// shape), as NumPy broadcasting with trailing singleton axes would accept.
pub fn rescale(f: &mut Tensor, log_scale: &Tensor) -> PyResult<Tensor> {
    if f.ndim() == 0 {
        return Err(PyValueError::new_err("cannot rescale a 0-d tensor"));
    }
    let new_shape = f.shape[..f.ndim() - 1].to_vec();
    let ls_nd = log_scale.ndim();
    if ls_nd > new_shape.len() || log_scale.shape[..] != new_shape[..ls_nd] {
        return Err(PyValueError::new_err(format!(
            "log_scale of shape {} cannot broadcast against a tensor of shape {}",
            log_scale.shape_str(),
            f.shape_str()
        )));
    }
    let stride = product(&new_shape[ls_nd..]);
    let d = f.last();
    let np = f.prefix_len();
    let mut out = Tensor::zeros(new_shape);
    for p in 0..np {
        let row = &mut f.data[p * d..][..d];
        let mut mx: f64 = 0.0;
        for v in row.iter() {
            mx = mx.max(v.abs());
        }
        if !(mx > 0.0) {
            mx = 1.0;
        }
        if mx != 1.0 {
            for v in row.iter_mut() {
                *v /= mx;
            }
        }
        out.data[p] = log_scale.data[p / stride] + mx.ln();
    }
    Ok(out)
}

/// Global per-site rescaling (axis-order agnostic): divide each site's
/// block by its max absolute value; ``log_scale`` must be per-site.
pub fn rescale_global(f: &mut Tensor, log_scale: &Tensor) -> PyResult<Tensor> {
    if log_scale.ndim() > 1 {
        return Err(PyValueError::new_err(format!(
            "_rescale_global requires per-site log_scale (1-D), got shape {}",
            log_scale.shape_str()
        )));
    }
    if f.ndim() == 0 {
        return Err(PyValueError::new_err("cannot rescale a 0-d tensor"));
    }
    let s = f.shape[0];
    if log_scale.data.len() != s {
        return Err(PyValueError::new_err(format!(
            "log_scale of shape {} cannot broadcast against a tensor of shape {}",
            log_scale.shape_str(),
            f.shape_str()
        )));
    }
    let block = product(&f.shape[1..]);
    let mut out = Tensor::zeros(vec![s]);
    for site in 0..s {
        let blk = &mut f.data[site * block..][..block];
        let mut mx: f64 = 0.0;
        for v in blk.iter() {
            mx = mx.max(v.abs());
        }
        if !(mx > 0.0) {
            mx = 1.0;
        }
        if mx != 1.0 {
            for v in blk.iter_mut() {
                *v /= mx;
            }
        }
        out.data[site] = log_scale.data[site] + mx.ln();
    }
    Ok(out)
}

/// Bring a multi-axis ``log_scale`` (shape ``F.shape[:-1]``) to a common
/// per-site value: each vector is multiplied by
/// ``exp(max(log_scale - common, -500))`` and the per-site maximum is
/// returned.  A per-site ``log_scale`` is returned unchanged.
pub fn equalize(f: &mut Tensor, log_scale: &Tensor) -> PyResult<Tensor> {
    if log_scale.ndim() <= 1 {
        return Ok(log_scale.clone());
    }
    if f.ndim() < 2 || log_scale.shape[..] != f.shape[..f.ndim() - 1] {
        return Err(PyValueError::new_err(format!(
            "log_scale of shape {} does not match a tensor of shape {}",
            log_scale.shape_str(),
            f.shape_str()
        )));
    }
    let s = f.shape[0];
    let per_site = product(&log_scale.shape[1..]);
    let d = f.last();
    let mut out = Tensor::zeros(vec![s]);
    for site in 0..s {
        let ls = &log_scale.data[site * per_site..][..per_site];
        let common = ls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        out.data[site] = common;
        for (e, &l) in ls.iter().enumerate() {
            let diff = (l - common).max(-500.0);
            let adj = diff.exp();
            let row = &mut f.data[(site * per_site + e) * d..][..d];
            for v in row.iter_mut() {
                *v *= adj;
            }
        }
    }
    Ok(out)
}

/// ``np.moveaxis(F, axis, -1)``.
pub fn moveaxis_to_end(f: &Tensor, axis: usize) -> Tensor {
    let nd = f.ndim();
    if nd == 0 || axis + 1 >= nd {
        return f.clone();
    }
    let outer = product(&f.shape[..axis]);
    let mid = f.shape[axis];
    let inner = product(&f.shape[axis + 1..]);
    let mut shape = Vec::with_capacity(nd);
    shape.extend_from_slice(&f.shape[..axis]);
    shape.extend_from_slice(&f.shape[axis + 1..]);
    shape.push(mid);
    let mut out = Tensor::zeros(shape);
    for o in 0..outer {
        for m in 0..mid {
            let src = &f.data[(o * mid + m) * inner..][..inner];
            for (i, &v) in src.iter().enumerate() {
                out.data[(o * inner + i) * mid + m] = v;
            }
        }
    }
    out
}

/// Swap the last two axes (``np.moveaxis(F, -2, -1)``).
pub fn swap_last_two(f: &Tensor) -> Tensor {
    let nd = f.ndim();
    if nd < 2 {
        return f.clone();
    }
    let a = f.shape[nd - 2];
    let b = f.shape[nd - 1];
    let outer = product(&f.shape[..nd - 2]);
    let mut shape = f.shape.clone();
    shape[nd - 2] = b;
    shape[nd - 1] = a;
    let mut out = Tensor::zeros(shape);
    for o in 0..outer {
        let src = &f.data[o * a * b..][..a * b];
        let dst = &mut out.data[o * a * b..][..a * b];
        for x in 0..a {
            for y in 0..b {
                dst[y * a + x] = src[x * b + y];
            }
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════
// The SNP model view of a network
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Leaf,
    Internal,
    Retic,
    Root,
}

/// One parent branch of a model node.
struct PBranch {
    length: Option<f64>,
    gamma: Option<f64>,
    parent: usize,
    parent_label: String,
    theta: f64,
}

struct MNode {
    name: String,
    kind: Kind,
    children: Vec<usize>,
    branches: Vec<PBranch>,
}

/// The typed node graph ``phynetpy._snp_model.build_snp_model`` builds,
/// read straight off the Rust network: nodes in ``V()`` order classified by
/// degree, children in ``E()`` order, parent branches in in-edge order.
struct SnpModel {
    nodes: Vec<MNode>,
    root: usize,
}

fn build_model(
    py: Python<'_>,
    net: &Network,
    theta: f64,
    edge_thetas: Option<&HashMap<(String, String), f64>>,
) -> PyResult<SnpModel> {
    let mut index: FxHashMap<u64, usize> = FxHashMap::default();
    for (pos, (&uid, _)) in net.nodes.iter().enumerate() {
        index.insert(uid, pos);
    }
    let mut nodes: Vec<MNode> = Vec::with_capacity(net.nodes.len());
    for (_, entry) in net.nodes.iter() {
        let in_deg = entry.in_edges.len();
        let out_deg = entry.out_edges.len();
        let name = entry.obj.borrow(py).name.clone();
        let mut branches = Vec::with_capacity(in_deg);
        for &e in entry.in_edges.iter() {
            let ee = &net.edges[&e];
            let eb = ee.obj.borrow(py);
            let parent_label = net.label(py, ee.su);
            let theta_b = edge_thetas
                .and_then(|m| m.get(&(name.clone(), parent_label.clone())).copied())
                .unwrap_or(theta);
            branches.push(PBranch {
                length: eb.length,
                gamma: eb.gamma,
                parent: index[&ee.su],
                parent_label,
                theta: theta_b,
            });
        }
        let kind = if out_deg == 0 {
            Kind::Leaf
        } else if in_deg == 1 {
            Kind::Internal
        } else if in_deg == 2 && out_deg == 1 {
            Kind::Retic
        } else if in_deg == 0 {
            Kind::Root
        } else {
            return Err(PyValueError::new_err(format!(
                "node '{name}' has in-degree {in_deg} and out-degree {out_deg}; the \
                 biallelic likelihood needs leaves, tree nodes, binary reticulations \
                 and a root"
            )));
        };
        if kind == Kind::Leaf && branches.is_empty() {
            return Err(PyIndexError::new_err(format!(
                "leaf '{name}' has no parent branch"
            )));
        }
        nodes.push(MNode {
            name,
            kind,
            children: Vec::new(),
            branches,
        });
    }
    for (_, ee) in net.edges.iter() {
        let s = index[&ee.su];
        let d = index[&ee.du];
        nodes[s].children.push(d);
    }
    let root = nodes
        .iter()
        .position(|n| n.kind == Kind::Root)
        .ok_or_else(|| PyIndexError::new_err("network has no root node"))?;
    Ok(SnpModel { nodes, root })
}

/// Post-order over the model graph from ``root`` (children in edge order,
/// each node once), as ``phynetpy._snp_model.postorder`` yields it.
fn postorder(model: &SnpModel, root: usize) -> Vec<usize> {
    let mut visited = vec![false; model.nodes.len()];
    let mut order = Vec::with_capacity(model.nodes.len());
    let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
    visited[root] = true;
    while let Some(&mut (node, ref mut next)) = stack.last_mut() {
        let children = &model.nodes[node].children;
        if *next < children.len() {
            let child = children[*next];
            *next += 1;
            if !visited[child] {
                visited[child] = true;
                stack.push((child, 0));
            }
        } else {
            order.push(node);
            stack.pop();
        }
    }
    order
}

// ═══════════════════════════════════════════════════════════════════════
// The DP
// ═══════════════════════════════════════════════════════════════════════

/// An open interface of a partial: a node's top (``parent = None``) or, for
/// a reticulation, its top towards one specific parent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Iface {
    node: usize,
    parent: Option<usize>,
}

struct Vpi {
    tensor: Tensor,
    ifaces: Vec<Iface>,
    max_lin: Vec<usize>,
    log_scale: Tensor,
}

struct Engine<'a> {
    model: &'a SnpModel,
    u: f64,
    v: f64,
    root_theta: f64,
    leaf_reds: &'a HashMap<String, Vec<i64>>,
    samples: &'a HashMap<String, usize>,
    lo: usize,
    sites: usize,
    vpis: Vec<Vpi>,
}

impl<'a> Engine<'a> {
    fn iface_name(&self, iface: &Iface) -> String {
        let n = &self.model.nodes[iface.node];
        match iface.parent {
            Some(p) => format!("{}_{}_top", n.name, self.model.nodes[p].name),
            None if n.kind == Kind::Root => format!("{}_bottom", n.name),
            None => format!("{}_top", n.name),
        }
    }

    fn rule1_branch(&self, f: &Tensor, m: usize, branch: &PBranch, child: &str) -> PyResult<Tensor> {
        let t = branch.length.ok_or_else(|| {
            PyTypeError::new_err(format!(
                "branch {} -> {} has no length; the biallelic likelihood needs a length on \
                 every branch",
                branch.parent_label, child
            ))
        })?;
        let p = transition(m, self.u, self.v, branch.theta, t);
        rule1(f, &p)
    }

    /// Find the partial holding ``child``'s interface towards ``parent``,
    /// move that interface to its last axis and return its position.
    fn get_vpi_for(&mut self, parent: usize, child: usize) -> PyResult<usize> {
        let retic = self.model.nodes[child].kind == Kind::Retic;
        let mut found: Option<(usize, usize)> = None;
        'outer: for (pos, vpi) in self.vpis.iter().enumerate() {
            for (ix, iface) in vpi.ifaces.iter().enumerate() {
                let hit = if retic {
                    iface.node == child && iface.parent == Some(parent)
                } else {
                    iface.node == child
                };
                if hit {
                    found = Some((pos, ix));
                    break 'outer;
                }
            }
        }
        let Some((pos, ix)) = found else {
            let available: Vec<String> = self
                .vpis
                .iter()
                .flat_map(|v| v.ifaces.iter().map(|i| format!("'{}'", self.iface_name(i))))
                .collect();
            return Err(PyValueError::new_err(format!(
                "Could not find VPI for node '{}' (parent: '{}', retic={}). Available \
                 interfaces: [{}]",
                self.model.nodes[child].name,
                self.model.nodes[parent].name,
                if retic { "True" } else { "False" },
                available.join(", ")
            )));
        };
        let vpi = &mut self.vpis[pos];
        if ix + 1 != vpi.ifaces.len() {
            if vpi.log_scale.ndim() > 1 {
                vpi.log_scale = equalize(&mut vpi.tensor, &vpi.log_scale)?;
            }
            let iface = vpi.ifaces.remove(ix);
            vpi.ifaces.push(iface);
            let ml = vpi.max_lin.remove(ix);
            vpi.max_lin.push(ml);
            vpi.tensor = moveaxis_to_end(&vpi.tensor, ix + 1);
        }
        Ok(pos)
    }

    fn take_two(&mut self, p0: usize, p1: usize) -> (Vpi, Vpi) {
        // Remove both, keeping the order of the remaining partials.
        if p0 < p1 {
            let y = self.vpis.remove(p1);
            let x = self.vpis.remove(p0);
            (x, y)
        } else {
            let x = self.vpis.remove(p0);
            let y = self.vpis.remove(p1);
            (x, y)
        }
    }

    fn visit_leaf(&mut self, idx: usize) -> PyResult<()> {
        let node = &self.model.nodes[idx];
        let samples = *self
            .samples
            .get(&node.name)
            .ok_or_else(|| PyKeyError::new_err(node.name.clone()))?;
        let reds = self.leaf_reds.get(&node.name).ok_or_else(|| {
            PyValueError::new_err(format!("Sequence with name {} not found in MSA", node.name))
        })?;
        let hi = self.lo + self.sites;
        if reds.len() < hi {
            return Err(PyValueError::new_err(format!(
                "leaf '{}' has {} sites, fewer than the {} expected",
                node.name,
                reds.len(),
                hi
            )));
        }
        let d = state_dim(samples);
        let base = n_to_index(samples) as i64;
        let mut f = Tensor::zeros(vec![self.sites, d]);
        for s in 0..self.sites {
            let idx_state = base + reds[self.lo + s];
            if idx_state < 0 || idx_state as usize >= d {
                return Err(PyIndexError::new_err(format!(
                    "index {idx_state} is out of bounds for axis 1 with size {d}"
                )));
            }
            f.data[s * d + idx_state as usize] = 1.0;
        }
        let log_scale = Tensor::zeros(vec![self.sites]);
        let mut f = self.rule1_branch(&f, samples, &node.branches[0], &node.name)?;
        let log_scale = rescale(&mut f, &log_scale)?;
        self.vpis.push(Vpi {
            tensor: f,
            ifaces: vec![Iface {
                node: idx,
                parent: None,
            }],
            max_lin: vec![samples],
            log_scale,
        });
        Ok(())
    }

    /// Rules 2 / 4 at a tree node or the root: merge the children's
    /// interfaces.  Two interfaces of the same partial merge by rule 4, two
    /// different partials by rule 2.
    fn merge_children(&mut self, idx: usize, who: &str) -> PyResult<Vpi> {
        let node = &self.model.nodes[idx];
        if node.children.len() != 2 {
            return Err(PyAssertionError::new_err(format!(
                "{who} node must have exactly two children"
            )));
        }
        let (c0, c1) = (node.children[0], node.children[1]);
        let p0 = self.get_vpi_for(idx, c0)?;
        let p1 = self.get_vpi_for(idx, c1)?;
        let top = Iface {
            node: idx,
            parent: None,
        };
        if p0 == p1 {
            let x = self.vpis.remove(p0);
            if x.max_lin.len() < 2 {
                return Err(PyValueError::new_err(format!(
                    "node '{}' needs two open interfaces to merge, found one",
                    node.name
                )));
            }
            let mx = x.max_lin[x.max_lin.len() - 2];
            let my = x.max_lin[x.max_lin.len() - 1];
            let mut xf = x.tensor;
            let x_ls = equalize(&mut xf, &x.log_scale)?;
            let f = rule4(&xf, &sparse_merge(mx, my))?;
            let mut ifaces = x.ifaces[..x.ifaces.len() - 2].to_vec();
            ifaces.push(top);
            let mut max_lin = x.max_lin[..x.max_lin.len() - 2].to_vec();
            max_lin.push(mx + my);
            Ok(Vpi {
                tensor: f,
                ifaces,
                max_lin,
                log_scale: x_ls,
            })
        } else {
            let (x, y) = self.take_two(p0, p1);
            let mx = *x.max_lin.last().unwrap();
            let my = *y.max_lin.last().unwrap();
            let mut xf = x.tensor;
            let x_ls = equalize(&mut xf, &x.log_scale)?;
            let mut yf = y.tensor;
            let y_ls = equalize(&mut yf, &y.log_scale)?;
            let f = rule2(&xf, &yf, &sparse_merge(mx, my))?;
            let mut ifaces = x.ifaces[..x.ifaces.len() - 1].to_vec();
            ifaces.extend_from_slice(&y.ifaces[..y.ifaces.len() - 1]);
            ifaces.push(top);
            let mut max_lin = x.max_lin[..x.max_lin.len() - 1].to_vec();
            max_lin.extend_from_slice(&y.max_lin[..y.max_lin.len() - 1]);
            max_lin.push(mx + my);
            let mut log_scale = x_ls;
            for (a, b) in log_scale.data.iter_mut().zip(y_ls.data.iter()) {
                *a += *b;
            }
            Ok(Vpi {
                tensor: f,
                ifaces,
                max_lin,
                log_scale,
            })
        }
    }

    fn visit_internal(&mut self, idx: usize) -> PyResult<()> {
        let mut vpi = self.merge_children(idx, "Internal")?;
        let node = &self.model.nodes[idx];
        vpi.log_scale = rescale(&mut vpi.tensor, &vpi.log_scale)?;
        let m = *vpi.max_lin.last().unwrap();
        vpi.tensor = self.rule1_branch(&vpi.tensor, m, &node.branches[0], &node.name)?;
        vpi.log_scale = rescale(&mut vpi.tensor, &vpi.log_scale)?;
        self.vpis.push(vpi);
        Ok(())
    }

    fn visit_reticulation(&mut self, idx: usize) -> PyResult<()> {
        let node = &self.model.nodes[idx];
        let child = *node.children.first().ok_or_else(|| {
            PyIndexError::new_err(format!("reticulation '{}' has no child", node.name))
        })?;
        let pos = self.get_vpi_for(idx, child)?;
        let x = self.vpis.remove(pos);
        let b0 = &node.branches[0];
        let b1 = &node.branches[1];
        let gamma = b0.gamma.ok_or_else(|| {
            PyTypeError::new_err(format!(
                "reticulation branch {} -> {} has no inheritance probability",
                b0.parent_label, node.name
            ))
        })?;
        let m = *x.max_lin.last().unwrap();
        let mut f = x.tensor;
        let mut log_scale = equalize(&mut f, &x.log_scale)?;
        let mut f = rule3(&f, &sparse_split(m, gamma))?;
        log_scale = rescale_global(&mut f, &log_scale)?;
        let f = swap_last_two(&f);
        let mut f = self.rule1_branch(&f, m, b0, &node.name)?;
        log_scale = rescale_global(&mut f, &log_scale)?;
        let f = swap_last_two(&f);
        let mut f = self.rule1_branch(&f, m, b1, &node.name)?;
        log_scale = rescale_global(&mut f, &log_scale)?;
        log_scale = rescale(&mut f, &log_scale)?;
        let mut ifaces = x.ifaces[..x.ifaces.len() - 1].to_vec();
        ifaces.push(Iface {
            node: idx,
            parent: Some(b0.parent),
        });
        ifaces.push(Iface {
            node: idx,
            parent: Some(b1.parent),
        });
        let mut max_lin = x.max_lin[..x.max_lin.len() - 1].to_vec();
        max_lin.push(m);
        max_lin.push(m);
        self.vpis.push(Vpi {
            tensor: f,
            ifaces,
            max_lin,
            log_scale,
        });
        Ok(())
    }

    fn visit_root(&mut self, idx: usize) -> PyResult<()> {
        let mut vpi = self.merge_children(idx, "Root")?;
        vpi.log_scale = rescale(&mut vpi.tensor, &vpi.log_scale)?;
        let final_lin = *vpi.max_lin.last().unwrap();
        vpi.ifaces = vec![Iface {
            node: idx,
            parent: None,
        }];
        vpi.max_lin = vec![final_lin];
        self.vpis.push(vpi);
        Ok(())
    }

    /// Fold the root partial against the stationary prior and sum the
    /// per-site log likelihoods.
    fn aggregate(&mut self, root: usize) -> PyResult<f64> {
        let pos = self.get_vpi_for(root, root)?;
        let x = self.vpis.remove(pos);
        let m = *x.max_lin.last().unwrap();
        let pi = stationary(m, self.u, self.v, self.root_theta)?;
        let mut f = x.tensor;
        let ls = equalize(&mut f, &x.log_scale)?;
        if f.ndim() != 2 {
            return Err(PyValueError::new_err(format!(
                "root partial has shape {}; every interface must be merged before the root",
                f.shape_str()
            )));
        }
        let d = f.last();
        if d != pi.len() {
            return Err(PyValueError::new_err(format!(
                "root partial has {d} states but the stationary prior has {}",
                pi.len()
            )));
        }
        let s = f.shape[0];
        let mut site_logs = Vec::with_capacity(s);
        for site in 0..s {
            let row = &f.data[site * d..][..d];
            let mut acc = 0.0;
            for (a, b) in row.iter().zip(pi.iter()) {
                acc += a * b;
            }
            let acc = acc.max(f64::MIN_POSITIVE);
            site_logs.push(acc.ln() + ls.data[site]);
        }
        Ok(pairwise_sum(&site_logs))
    }

    fn run(&mut self) -> PyResult<f64> {
        let root = self.model.root;
        for idx in postorder(self.model, root) {
            match self.model.nodes[idx].kind {
                Kind::Leaf => self.visit_leaf(idx)?,
                Kind::Internal => self.visit_internal(idx)?,
                Kind::Retic => self.visit_reticulation(idx)?,
                Kind::Root => self.visit_root(idx)?,
            }
        }
        self.aggregate(root)
    }
}

/// Pairwise summation (NumPy's ``sum`` order to within its block size).
fn pairwise_sum(v: &[f64]) -> f64 {
    if v.len() <= 8 {
        return v.iter().sum();
    }
    let mid = v.len() / 2;
    pairwise_sum(&v[..mid]) + pairwise_sum(&v[mid..])
}

// ═══════════════════════════════════════════════════════════════════════
// Python surface
// ═══════════════════════════════════════════════════════════════════════

fn dmatrix_to_py<'py>(py: Python<'py>, m: &DMatrix<f64>) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)]).into_pyarray(py)
}

fn tensor_from_py(arr: &PyReadonlyArrayDyn<'_, f64>) -> Tensor {
    let view = arr.as_array();
    let shape = view.shape().to_vec();
    let data = match view.as_slice() {
        Some(s) => s.to_vec(),
        None => view.iter().copied().collect(),
    };
    Tensor { shape, data }
}

fn tensor_to_py<'py>(py: Python<'py>, t: Tensor) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let arr = ArrayD::from_shape_vec(IxDyn(&t.shape), t.data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

fn vec_to_py<'py, T: numpy::Element>(py: Python<'py>, v: Vec<T>) -> Bound<'py, PyArray1<T>> {
    Array1::from_vec(v).into_pyarray(py)
}

/// Transition probabilities between ``(n, r)`` lineage / red-allele states
/// under the biallelic model (Bryant et al. 2012, eq. 15).
///
/// ``Q`` is sized for ``n`` sampled lineages and includes the empty state
/// ``(0, 0)`` at index 0.  ``expt(t)`` evaluates ``e^{Qt}`` by scaling-and-
/// squaring Padé (the ``scipy.linalg.expm`` algorithm); results are cached
/// per ``(n, u, v, theta, t)`` so repeated branch lengths cost nothing.
///
/// Args:
///     n (int): sample count (maximum lineages).
///     u (float): mutation rate red -> green.
///     v (float): mutation rate green -> red.
///     theta (float): population mutation rate ``4*N*mu``.
#[pyclass(name = "BiMarkersTransition", module = "phynetpy.BiMarkers")]
pub struct BiMarkersTransition {
    n: usize,
    u: f64,
    v: f64,
    theta: f64,
    q: Arc<DMatrix<f64>>,
}

#[pymethods]
impl BiMarkersTransition {
    #[new]
    fn py_new(n: usize, u: f64, v: f64, theta: f64) -> Self {
        BiMarkersTransition {
            n,
            u,
            v,
            theta,
            q: rate_matrix(n, u, v, theta),
        }
    }

    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    #[getter]
    fn u(&self) -> f64 {
        self.u
    }

    #[getter]
    fn v(&self) -> f64 {
        self.v
    }

    #[getter]
    fn theta(&self) -> f64 {
        self.theta
    }

    /// The rate matrix as a ``state_dim(n)`` square NumPy array.
    #[getter(Q)]
    fn q_py<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_py(py, &self.q)
    }

    /// Compute ``e^(Q*t)``.
    ///
    /// Args:
    ///     t (float): time, generally in coalescent units. Optional, defaults
    ///         to 1, in which case ``e^Q`` is computed.
    ///
    /// Returns:
    ///     np.ndarray: ``e^(Q*t)``.
    #[pyo3(signature = (t=1.0))]
    fn expt<'py>(&self, py: Python<'py>, t: f64) -> Bound<'py, PyArray2<f64>> {
        let p = transition(self.n, self.u, self.v, self.theta, t);
        dmatrix_to_py(py, &p)
    }

    /// The number of columns of the Q matrix.
    fn cols(&self) -> usize {
        self.q.ncols()
    }

    /// Retrieve the Q matrix.
    #[pyo3(name = "getQ")]
    fn get_q<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dmatrix_to_py(py, &self.q)
    }

    fn __repr__(&self) -> String {
        format!(
            "BiMarkersTransition(n={}, u={}, v={}, theta={})",
            self.n, self.u, self.v, self.theta
        )
    }
}

/// Root prior ``pi`` for ``m`` lineages from a full rate matrix.
#[pyfunction]
#[pyo3(name = "_snp_root_stationary")]
fn root_stationary_py<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, f64>,
    m: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = q.as_array();
    let dm = DMatrix::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
    Ok(vec_to_py(py, root_stationary(&dm, m)?))
}

/// ``(i, j, k, coeff)`` arrays of the sparse merge table for ``(mx, my)``.
#[pyfunction]
#[pyo3(name = "_snp_sparse_merge")]
fn sparse_merge_py<'py>(
    py: Python<'py>,
    mx: usize,
    my: usize,
) -> (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f64>>,
) {
    let sm = sparse_merge(mx, my);
    (
        vec_to_py(py, sm.i.iter().map(|&x| x as i32).collect()),
        vec_to_py(py, sm.j.iter().map(|&x| x as i32).collect()),
        vec_to_py(py, sm.k.iter().map(|&x| x as i32).collect()),
        vec_to_py(py, sm.c.clone()),
    )
}

/// ``(i, j, k, coeff)`` arrays of the sparse split table for ``(m, gamma)``.
#[pyfunction]
#[pyo3(name = "_snp_sparse_split")]
fn sparse_split_py<'py>(
    py: Python<'py>,
    m: usize,
    gamma: f64,
) -> (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f64>>,
) {
    let ss = sparse_split(m, gamma);
    (
        vec_to_py(py, ss.i.iter().map(|&x| x as i32).collect()),
        vec_to_py(py, ss.j.iter().map(|&x| x as i32).collect()),
        vec_to_py(py, ss.k.iter().map(|&x| x as i32).collect()),
        vec_to_py(py, ss.c.clone()),
    )
}

/// Rule 2 on NumPy arrays: ``(S, *xe, dx) x (S, *ye, dy) -> (S, *xe, *ye, dz)``.
#[pyfunction]
#[pyo3(name = "_snp_rule2")]
fn rule2_py<'py>(
    py: Python<'py>,
    f_x: PyReadonlyArrayDyn<'py, f64>,
    f_y: PyReadonlyArrayDyn<'py, f64>,
    mx: usize,
    my: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let fx = tensor_from_py(&f_x);
    let fy = tensor_from_py(&f_y);
    let out = rule2(&fx, &fy, &sparse_merge(mx, my))?;
    tensor_to_py(py, out)
}

/// Rule 3 on a NumPy array: ``(..., d) -> (..., d, d)``.
#[pyfunction]
#[pyo3(name = "_snp_rule3")]
fn rule3_py<'py>(
    py: Python<'py>,
    f: PyReadonlyArrayDyn<'py, f64>,
    m: usize,
    gamma: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let t = tensor_from_py(&f);
    let out = rule3(&t, &sparse_split(m, gamma))?;
    tensor_to_py(py, out)
}

/// Rule 4 on a NumPy array: ``(..., dx, dy) -> (..., dz)``.
#[pyfunction]
#[pyo3(name = "_snp_rule4")]
fn rule4_py<'py>(
    py: Python<'py>,
    f: PyReadonlyArrayDyn<'py, f64>,
    mx: usize,
    my: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let t = tensor_from_py(&f);
    let out = rule4(&t, &sparse_merge(mx, my))?;
    tensor_to_py(py, out)
}

/// Per-vector rescaling of a NumPy array; returns ``(F, log_scale)``.
#[pyfunction]
#[pyo3(name = "_snp_rescale")]
fn rescale_py<'py>(
    py: Python<'py>,
    f: PyReadonlyArrayDyn<'py, f64>,
    log_scale: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<f64>>)> {
    let mut t = tensor_from_py(&f);
    let ls = tensor_from_py(&log_scale);
    let new_ls = rescale(&mut t, &ls)?;
    Ok((tensor_to_py(py, t)?, tensor_to_py(py, new_ls)?))
}

/// Global per-site rescaling of a NumPy array; returns ``(F, log_scale)``.
#[pyfunction]
#[pyo3(name = "_snp_rescale_global")]
fn rescale_global_py<'py>(
    py: Python<'py>,
    f: PyReadonlyArrayDyn<'py, f64>,
    log_scale: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<f64>>)> {
    let mut t = tensor_from_py(&f);
    let ls = tensor_from_py(&log_scale);
    let new_ls = rescale_global(&mut t, &ls)?;
    Ok((tensor_to_py(py, t)?, tensor_to_py(py, new_ls)?))
}

/// Scale equalisation of a NumPy array; returns ``(F, per-site log_scale)``.
#[pyfunction]
#[pyo3(name = "_snp_equalize_scales")]
fn equalize_py<'py>(
    py: Python<'py>,
    f: PyReadonlyArrayDyn<'py, f64>,
    log_scale: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<f64>>)> {
    let mut t = tensor_from_py(&f);
    let ls = tensor_from_py(&log_scale);
    let new_ls = equalize(&mut t, &ls)?;
    Ok((tensor_to_py(py, t)?, tensor_to_py(py, new_ls)?))
}

/// The whole biallelic-marker DP for one site batch.
///
/// Args:
///     net: the species network (branch lengths in substitutions per site).
///     leaf_reds: taxon label -> per-site red-allele counts.
///     samples: taxon label -> number of sampled gene copies.
///     n_sites: total number of sites in the alignment.
///     u, v: mutation rates red -> green and green -> red.
///     theta: default population mutation rate for every branch.
///     root_theta: theta of the root population (sizes the stationary prior).
///     site_slice: optional ``(start, stop)`` restricting the batch.
///     edge_thetas: optional ``{(child_label, parent_label): theta}`` overrides.
///
/// Returns:
///     ``sum_sites log P(site | net)`` for the batch.
#[pyfunction]
#[pyo3(name = "_snp_log_likelihood_core")]
#[pyo3(signature = (net, leaf_reds, samples, n_sites, u, v, theta, root_theta, site_slice=None, edge_thetas=None))]
#[allow(clippy::too_many_arguments)]
fn snp_log_likelihood_core_py(
    py: Python<'_>,
    net: &Bound<'_, Network>,
    leaf_reds: HashMap<String, Vec<i64>>,
    samples: HashMap<String, usize>,
    n_sites: usize,
    u: f64,
    v: f64,
    theta: f64,
    root_theta: f64,
    site_slice: Option<(usize, usize)>,
    edge_thetas: Option<HashMap<(String, String), f64>>,
) -> PyResult<f64> {
    let model = {
        let n = net.borrow();
        build_model(py, &n, theta, edge_thetas.as_ref())?
    };
    let (lo, hi) = match site_slice {
        Some((a, b)) => (a.min(n_sites), b.min(n_sites)),
        None => (0, n_sites),
    };
    let sites = hi.saturating_sub(lo);
    py.detach(move || {
        let mut engine = Engine {
            model: &model,
            u,
            v,
            root_theta,
            leaf_reds: &leaf_reds,
            samples: &samples,
            lo,
            sites,
            vpis: Vec::new(),
        };
        engine.run()
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BiMarkersTransition>()?;
    m.add_function(wrap_pyfunction!(root_stationary_py, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_merge_py, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_split_py, m)?)?;
    m.add_function(wrap_pyfunction!(rule2_py, m)?)?;
    m.add_function(wrap_pyfunction!(rule3_py, m)?)?;
    m.add_function(wrap_pyfunction!(rule4_py, m)?)?;
    m.add_function(wrap_pyfunction!(rescale_py, m)?)?;
    m.add_function(wrap_pyfunction!(rescale_global_py, m)?)?;
    m.add_function(wrap_pyfunction!(equalize_py, m)?)?;
    m.add_function(wrap_pyfunction!(snp_log_likelihood_core_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexing() {
        assert_eq!(nr_to_index(0, 0), 0);
        assert_eq!(nr_to_index(1, 0), 1);
        assert_eq!(nr_to_index(1, 1), 2);
        assert_eq!(nr_to_index(2, 0), 3);
        assert_eq!(nr_to_index(3, 0), 6);
        assert_eq!(state_dim(1), 3);
        assert_eq!(state_dim(3), 10);
    }

    #[test]
    fn two_state_transition_is_closed_form() {
        let (u, v, t) = (0.8, 1.2, 0.37);
        let q = build_rate_matrix(1, u, v, 0.02);
        let p = expm(&q, t);
        let total = u + v;
        let decay = (-total * t).exp();
        // Q acts on likelihood vectors, so e^{Qt} is the transpose of the
        // forward two-state transition matrix: P[(1,0),(1,1)] is the
        // red -> green entry (u / (u + v)) (1 - e^{-(u+v) t}).
        assert!((p[(1, 2)] - (u / total) * (1.0 - decay)).abs() < 1e-12);
        // P[(1,1),(1,1)]: red stays red.
        assert!((p[(2, 2)] - (v / total + (u / total) * decay)).abs() < 1e-12);
        assert!((p[(0, 0)] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn stationary_single_lineage_block() {
        let (u, v, theta) = (0.7, 1.3, 0.02);
        let q = build_rate_matrix(4, u, v, theta);
        let pi = root_stationary(&q, 4).unwrap();
        assert_eq!(pi[0], 0.0);
        assert!((pi[1] - u / (u + v)).abs() < 1e-12);
        assert!((pi[2] - v / (u + v)).abs() < 1e-12);
        // Null vector: Q[1:,1:] pi[1:] = 0.
        let d = state_dim(4);
        for i in 1..d {
            let mut acc = 0.0;
            for j in 1..d {
                acc += q[(i, j)] * pi[j];
            }
            assert!(acc.abs() < 1e-9, "row {i} residual {acc}");
        }
        let svd = null_vector_svd(&q, d).unwrap();
        let s = svd[1] + svd[2];
        for i in 1..d {
            assert!((svd[i] / s - pi[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn split_conserves_mass_and_merge_is_hypergeometric() {
        let ss = build_sparse_split(3, 0.3);
        // Rows with r = 0 or r = n admit one red split per nb, so they sum
        // to the binomial total of one; the other rows are deliberately
        // unnormalised (the hypergeometric merge supplies the red
        // combinatorics).
        let mut sums = vec![0.0; ss.dim];
        for (&i, &c) in ss.i.iter().zip(ss.c.iter()) {
            sums[i as usize] += c;
        }
        for n in 0..=3 {
            assert!((sums[nr_to_index(n, 0)] - 1.0).abs() < 1e-12);
            assert!((sums[nr_to_index(n, n)] - 1.0).abs() < 1e-12);
        }
        let sm = build_sparse_merge(2, 1);
        assert_eq!(sm.i.len(), state_dim(2) * state_dim(1));
        for &c in &sm.c {
            assert!(c > 0.0 && c <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn rule1_matches_naive_product() {
        let q = build_rate_matrix(2, 1.0, 1.0, 0.1);
        let p = expm(&q, 0.05);
        let f = Tensor {
            shape: vec![2, 6],
            data: (0..12).map(|x| x as f64 * 0.1).collect(),
        };
        let out = rule1(&f, &p).unwrap();
        for s in 0..2 {
            for j in 0..6 {
                let mut acc = 0.0;
                for i in 0..6 {
                    acc += f.data[s * 6 + i] * p[(i, j)];
                }
                assert!((out.data[s * 6 + j] - acc).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn axis_moves() {
        let f = Tensor {
            shape: vec![1, 2, 3],
            data: (0..6).map(|x| x as f64).collect(),
        };
        let g = swap_last_two(&f);
        assert_eq!(g.shape, vec![1, 3, 2]);
        assert_eq!(g.data, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        let h = moveaxis_to_end(&f, 1);
        assert_eq!(h.shape, vec![1, 3, 2]);
        assert_eq!(h.data, g.data);
    }
}
