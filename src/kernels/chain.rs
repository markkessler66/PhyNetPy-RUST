//! MCMC chain diagnostics, faithful to BEAST's ``TraceCorrelation``.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Largest lag considered when estimating the autocorrelation time
/// (BEAST's ``TraceCorrelation.MAX_LAG``).
pub const MAX_LAG: usize = 2000;

fn mean(a: &[f64]) -> f64 {
    a.iter().sum::<f64>() / a.len() as f64
}

/// ``(gamma0, varStat)``: lag-0 autocovariance and Geyer's initial
/// positive-sequence sum, with BEAST's ``(n - lag)`` normalisation.
pub fn gamma_and_varstat(arr: &[f64]) -> (f64, f64) {
    let n = arr.len();
    let m = mean(arr);
    let dev: Vec<f64> = arr.iter().map(|v| v - m).collect();
    let max_lag = (n - 1).min(MAX_LAG);
    let mut gamma = Vec::with_capacity(max_lag + 1);
    for lag in 0..=max_lag {
        let dot: f64 = dev[..n - lag].iter().zip(dev[lag..].iter()).map(|(a, b)| a * b).sum();
        gamma.push(dot / (n - lag) as f64);
    }
    let mut var_stat = gamma[0];
    for lag in 1..=max_lag {
        if lag % 2 == 0 {
            let pair = gamma[lag - 1] + gamma[lag];
            if pair > 0.0 {
                var_stat += 2.0 * pair;
            } else {
                break;
            }
        }
    }
    (gamma[0], var_stat)
}

#[pyfunction]
fn _gamma_and_varstat(arr: Vec<f64>) -> (f64, f64) {
    gamma_and_varstat(&arr)
}

#[pyfunction]
fn effective_sample_size(values: Vec<f64>) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let (g0, vs) = gamma_and_varstat(&values);
    if g0 == 0.0 || vs == 0.0 {
        return f64::NAN;
    }
    values.len() as f64 * g0 / vs
}

#[pyfunction]
#[pyo3(signature = (values, step_size=1))]
fn autocorrelation_time(values: Vec<f64>, step_size: i64) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let (g0, vs) = gamma_and_varstat(&values);
    if g0 == 0.0 {
        return f64::NAN;
    }
    step_size as f64 * vs / g0
}

#[pyfunction]
fn standard_error_of_mean(values: Vec<f64>) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let (_, vs) = gamma_and_varstat(&values);
    if vs > 0.0 {
        (vs / values.len() as f64).sqrt()
    } else {
        0.0
    }
}

#[pyfunction]
#[pyo3(signature = (values, prob=0.95))]
fn hpd_interval(values: Vec<f64>, prob: f64) -> PyResult<(f64, f64)> {
    if !(0.0 < prob && prob <= 1.0) {
        return Err(PyValueError::new_err("prob must be in (0, 1]."));
    }
    let mut arr = values;
    arr.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = arr.len();
    if n == 0 {
        return Err(PyValueError::new_err("Cannot compute an HPD interval of no samples."));
    }
    if n == 1 {
        return Ok((arr[0], arr[0]));
    }
    let n_in = ((prob * n as f64).ceil() as usize).max(1);
    if n_in >= n {
        return Ok((arr[0], arr[n - 1]));
    }
    let mut best_k = 0;
    let mut best_w = f64::INFINITY;
    for k in 0..=(n - n_in) {
        let w = arr[k + n_in - 1] - arr[k];
        if w < best_w {
            best_w = w;
            best_k = k;
        }
    }
    Ok((arr[best_k], arr[best_k + n_in - 1]))
}

#[pyfunction]
#[pyo3(signature = (values, first=0.1, last=0.5))]
fn geweke(values: Vec<f64>, first: f64, last: f64) -> f64 {
    let n = values.len();
    if n < 4 || first <= 0.0 || last <= 0.0 || first + last > 1.0 {
        return f64::NAN;
    }
    let na = ((first * n as f64) as usize).max(2).min(n);
    let nb = ((last * n as f64) as usize).max(2).min(n);
    let a = &values[..na];
    let b = &values[n - nb..];
    let (_, va) = gamma_and_varstat(a);
    let (_, vb) = gamma_and_varstat(b);
    let se = (va / a.len() as f64 + vb / b.len() as f64).sqrt();
    if se == 0.0 {
        return f64::NAN;
    }
    (mean(a) - mean(b)) / se
}

#[pyfunction]
fn gelman_rubin(chains: Vec<Vec<f64>>) -> f64 {
    let m = chains.len();
    if m < 2 {
        return f64::NAN;
    }
    let n = chains.iter().map(|c| c.len()).min().unwrap_or(0);
    if n < 2 {
        return f64::NAN;
    }
    let means: Vec<f64> = chains.iter().map(|c| mean(&c[..n])).collect();
    let variances: Vec<f64> = chains
        .iter()
        .zip(means.iter())
        .map(|(c, &mu)| c[..n].iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / (n - 1) as f64)
        .collect();
    let grand = mean(&means);
    let between = n as f64 / (m - 1) as f64 * means.iter().map(|x| (x - grand) * (x - grand)).sum::<f64>();
    let within = mean(&variances);
    if within == 0.0 {
        return f64::NAN;
    }
    let var_plus = (n - 1) as f64 / n as f64 * within + between / n as f64;
    (var_plus / within).sqrt()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_gamma_and_varstat, m)?)?;
    m.add_function(wrap_pyfunction!(effective_sample_size, m)?)?;
    m.add_function(wrap_pyfunction!(autocorrelation_time, m)?)?;
    m.add_function(wrap_pyfunction!(standard_error_of_mean, m)?)?;
    m.add_function(wrap_pyfunction!(hpd_interval, m)?)?;
    m.add_function(wrap_pyfunction!(geweke, m)?)?;
    m.add_function(wrap_pyfunction!(gelman_rubin, m)?)?;
    Ok(())
}
