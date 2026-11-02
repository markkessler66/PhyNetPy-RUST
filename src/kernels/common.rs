//! Shared numerics for the likelihood kernels.

/// ``log(1e-200)``: the floor every kernel clamps a vanishing probability to.
pub const LOG_FLOOR: f64 = -460.51701859880916;

/// Numerically stable ``log(exp(a) + exp(b))``.
#[inline]
pub fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

/// Stable ``log(sum(exp(v)))``, floored at [`LOG_FLOOR`] for empty or
/// all-``-inf`` input.
pub fn logsumexp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return LOG_FLOOR;
    }
    let m = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return LOG_FLOOR;
    }
    let acc: f64 = values.iter().map(|v| (v - m).exp()).sum();
    if acc <= 0.0 {
        return LOG_FLOOR;
    }
    m + acc.ln()
}

/// ``start * (start+1) * ... * end`` as a float (1.0 for an empty range).
#[inline]
pub fn fact_range(start: i64, end: i64) -> f64 {
    let mut r = 1.0;
    let mut i = start;
    while i <= end {
        r *= i as f64;
        i += 1;
    }
    r
}

/// Kingman coalescent transition probability ``g_{ij}(t)``: the chance
/// that ``i`` lineages entering a branch of length ``t`` (coalescent
/// units) leave it as ``j`` lineages.  ``None`` (or a negative length)
/// is the infinite root branch.
pub fn gij(length: Option<f64>, i: u32, j: u32) -> f64 {
    let length = match length {
        None => return if j == 1 { 1.0 } else { 0.0 },
        Some(l) if l == -1.0 => return if j == 1 { 1.0 } else { 0.0 },
        Some(l) => l,
    };
    if length == 0.0 {
        return if i == j { 1.0 } else { 0.0 };
    }
    if i == 0 {
        return 1.0;
    }
    let (i, j) = (i as i64, j as i64);
    let mut result = 0.0;
    for k in j..=i {
        let sign = if (k - j) % 2 == 0 { 1.0 } else { -1.0 };
        let kf = k as f64;
        let temp = (0.5 * kf * (1.0 - kf) * length).exp()
            * (2.0 * kf - 1.0)
            * sign
            * fact_range(j, j + k - 2)
            * fact_range(i - k + 1, i);
        let denom = fact_range(1, j) * fact_range(1, k - j) * fact_range(i, i + k - 1);
        result += temp / denom;
    }
    result
}

/// [`gij`] clamped to ``[0, 1]`` (the MSC kernel's convention).
pub fn gij_clamped(length: Option<f64>, i: u32, j: u32) -> f64 {
    gij(length, i, j).clamp(0.0, 1.0)
}

/// ``log(k!)`` by direct summation (exact enough for the DP factors).
#[inline]
pub fn ln_factorial(k: u32) -> f64 {
    (2..=k).map(|i| (i as f64).ln()).sum()
}
