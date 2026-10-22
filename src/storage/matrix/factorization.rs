//! Bounded in-house dense real factorizations with explicit numerical domains.
use crate::{GraphError, GraphResult};
fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
pub(super) fn zeros(rows: usize, cols: usize) -> GraphResult<Vec<f64>> {
    let count = rows
        .checked_mul(cols)
        .filter(|n| {
            n.checked_mul(8)
                .is_some_and(|bytes| bytes <= isize::MAX as usize)
        })
        .ok_or_else(|| invalid("Factorization size overflow"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| invalid("Factorization allocation failed"))?;
    values.resize(count, 0.);
    Ok(values)
}
fn normalize(input: &mut [f64]) -> GraphResult<f64> {
    if input.iter().any(|v| !v.is_finite()) {
        return Err(invalid("Factorization requires finite values"));
    }
    let scale = input.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
    let scale = if scale == 0. { 1. } else { scale };
    for value in input {
        let scaled = *value / scale;
        if *value != 0. && scaled == 0. {
            return Err(invalid("Factorization scaling loses input range"));
        }
        *value = scaled;
    }
    Ok(scale)
}
/// Economy Householder QR: Q is rows x min(rows,cols), R is min(rows,cols) x cols.
pub(super) fn qr(mut a: Vec<f64>, rows: usize, cols: usize) -> GraphResult<(Vec<f64>, Vec<f64>)> {
    let scale = normalize(&mut a)?;
    let k = rows.min(cols);
    let mut reflectors = Vec::with_capacity(k);
    for col in 0..k {
        let norm = (col..rows).fold(0.0_f64, |norm, row| norm.hypot(a[row * cols + col]));
        if norm == 0. {
            reflectors.push(None);
            continue;
        }
        let alpha = -norm.copysign(a[col * cols + col]);
        let mut v: Vec<_> = (col..rows).map(|row| a[row * cols + col]).collect();
        v[0] -= alpha;
        let norm = v.iter().fold(0.0_f64, |norm, v| norm.hypot(*v));
        for value in &mut v {
            *value /= norm;
        }
        for j in col..cols {
            let mut dot = super::dot_product::Dot::new();
            for (offset, &v) in v.iter().enumerate() {
                dot.add(v, a[(col + offset) * cols + j]);
            }
            let projection = dot.total();
            for (offset, &v) in v.iter().enumerate() {
                let index = (col + offset) * cols + j;
                a[index] = (-2. * v).mul_add(projection, a[index]);
            }
        }
        a[col * cols + col] = alpha;
        for row in col + 1..rows {
            a[row * cols + col] = 0.;
        }
        reflectors.push(Some(v));
    }
    let mut q = zeros(rows, k)?;
    for i in 0..k {
        q[i * k + i] = 1.;
    }
    for col in (0..k).rev() {
        if let Some(v) = &reflectors[col] {
            for j in 0..k {
                let mut dot = super::dot_product::Dot::new();
                for (offset, &v) in v.iter().enumerate() {
                    dot.add(v, q[(col + offset) * k + j]);
                }
                let projection = dot.total();
                for (offset, &v) in v.iter().enumerate() {
                    let index = (col + offset) * k + j;
                    q[index] = (-2. * v).mul_add(projection, q[index]);
                }
            }
        }
    }
    let mut r = zeros(k, cols)?;
    for row in 0..k {
        for col in row..cols {
            r[row * cols + col] = a[row * cols + col] * scale;
        }
    }
    if q.iter().chain(&r).any(|v| !v.is_finite()) {
        return Err(invalid("QR output exceeds finite range"));
    }
    Ok((q, r))
}

/// Real Cholesky A=L*L^T with norm-relative symmetry checks and checked residuals.
pub(super) fn cholesky(mut a: Vec<f64>, n: usize) -> GraphResult<Vec<f64>> {
    let scale = normalize(&mut a)?;
    let tolerance = 8. * f64::EPSILON * n.max(1) as f64;
    for row in 0..n {
        for col in 0..row {
            if (a[row * n + col] - a[col * n + row]).abs() > tolerance {
                return Err(invalid("Cholesky requires symmetric input"));
            }
            let average = a[row * n + col] + (a[col * n + row] - a[row * n + col]) * 0.5;
            a[row * n + col] = average;
            a[col * n + row] = average;
        }
    }
    let mut lower = zeros(n, n)?;
    for row in 0..n {
        for col in 0..=row {
            let mut residual = super::dot_product::Dot::new();
            residual.add(a[row * n + col], 1.);
            for k in 0..col {
                residual.add(-lower[row * n + k], lower[col * n + k]);
            }
            let residual = residual.total();
            let value = if row == col {
                if residual <= 0. {
                    return Err(invalid(
                        "Cholesky input is not positive definite at the working precision",
                    ));
                }
                residual.sqrt()
            } else {
                residual / lower[col * n + col]
            };
            if !value.is_finite() {
                return Err(invalid("Cholesky arithmetic exceeds finite range"));
            }
            lower[row * n + col] = value;
        }
    }
    let scale = scale.sqrt();
    for value in &mut lower {
        *value *= scale;
    }
    if lower.iter().any(|v| !v.is_finite()) {
        return Err(invalid("Cholesky output exceeds finite range"));
    }
    Ok(lower)
}

pub(super) struct Lu {
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub permutation: Vec<usize>,
    pub swaps: usize,
}
/// Partial-pivot PA=LU. Zero pivot columns are retained for singular factorizations.
pub(super) fn lu(mut upper: Vec<f64>, n: usize) -> GraphResult<Lu> {
    if upper.iter().any(|v| !v.is_finite()) {
        return Err(invalid("LU requires finite input"));
    }
    let mut lower = zeros(n, n)?;
    let mut permutation: Vec<_> = (0..n).collect();
    let mut swaps = 0;
    for i in 0..n {
        lower[i * n + i] = 1.;
    }
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if upper[row * n + col].abs() > upper[pivot * n + col].abs() {
                pivot = row;
            }
        }
        if upper[pivot * n + col] == 0. {
            continue;
        }
        if pivot != col {
            for j in 0..n {
                upper.swap(col * n + j, pivot * n + j);
            }
            for j in 0..col {
                lower.swap(col * n + j, pivot * n + j);
            }
            permutation.swap(col, pivot);
            swaps += 1;
        }
        for row in col + 1..n {
            let factor = upper[row * n + col] / upper[col * n + col];
            lower[row * n + col] = factor;
            upper[row * n + col] = 0.;
            for j in col + 1..n {
                let value = (-factor).mul_add(upper[col * n + j], upper[row * n + j]);
                if !value.is_finite() {
                    return Err(invalid("LU factors exceed finite range"));
                }
                upper[row * n + j] = value;
            }
        }
    }
    Ok(Lu {
        upper,
        lower,
        permutation,
        swaps,
    })
}
pub(super) fn slogdet(factor: &Lu, n: usize) -> (f64, f64) {
    let mut sign = if factor.swaps.is_multiple_of(2) {
        1.
    } else {
        -1.
    };
    let mut logs = Vec::with_capacity(n);
    for i in 0..n {
        let pivot = factor.upper[i * n + i];
        if pivot == 0. {
            return (0., f64::NEG_INFINITY);
        }
        sign *= pivot.signum();
        logs.push(pivot.abs().ln());
    }
    (sign, crate::storage::array::numeric_reductions::sum(&logs))
}
