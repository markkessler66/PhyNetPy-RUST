//! Bounded scalar minimisation (Brent's method with bounds, the algorithm
//! behind SciPy's ``minimize_scalar(method="bounded")``), used by the
//! branch-length / inheritance-probability coordinate ascent.

use pyo3::prelude::*;

/// Minimise ``f`` on ``[lo, hi]``.  Returns ``(x, f(x), n_evaluations)``.
pub fn brent_bounded<F: FnMut(f64) -> PyResult<f64>>(
    mut f: F,
    lo: f64,
    hi: f64,
    maxiter: usize,
    xatol: f64,
) -> PyResult<(f64, f64, usize)> {
    let sqrt_eps = 2.2e-16f64.sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0f64.sqrt());
    let (mut a, mut b) = (lo, hi);
    let mut fulc = a + golden_mean * (b - a);
    let mut nfc = fulc;
    let mut xf = fulc;
    let mut rat = 0.0f64;
    let mut e = 0.0f64;
    let mut x = xf;
    let mut fx = f(x)?;
    let mut num = 1usize;
    let mut ffulc = fx;
    let mut fnfc = fx;
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;
    let sign = |v: f64| -> f64 {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    };

    while (xf - xm).abs() > (tol2 - 0.5 * (b - a)) {
        let mut golden = true;
        if e.abs() > tol1 {
            golden = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = p / q;
                x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    let si = sign(xm - xf) + if xm - xf == 0.0 { 1.0 } else { 0.0 };
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }
        let si = sign(rat) + if rat == 0.0 { 1.0 } else { 0.0 };
        x = xf + si * rat.abs().max(tol1);
        let fu = f(x)?;
        num += 1;
        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if num >= maxiter {
            break;
        }
    }
    Ok((xf, fx, num))
}

/// ``brent_bounded_minimize(func, lo, hi, maxiter, xatol) -> (x, fun)``.
#[pyfunction]
#[pyo3(signature = (func, lo, hi, maxiter=500, xatol=1e-5))]
fn brent_bounded_minimize(
    func: &Bound<'_, PyAny>,
    lo: f64,
    hi: f64,
    maxiter: usize,
    xatol: f64,
) -> PyResult<(f64, f64)> {
    let (x, fx, _) = brent_bounded(|v| func.call1((v,))?.extract::<f64>(), lo, hi, maxiter, xatol)?;
    Ok((x, fx))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(brent_bounded_minimize, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_parabola_minimum() {
        let (x, fx, _) = brent_bounded(|v| Ok::<f64, pyo3::PyErr>((v - 1.3) * (v - 1.3) + 2.0), 0.0, 5.0, 500, 1e-6).unwrap();
        assert!((x - 1.3).abs() < 1e-4);
        assert!((fx - 2.0).abs() < 1e-8);
    }
}
