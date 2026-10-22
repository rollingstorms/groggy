//! Direct dense row-equilibrated Gaussian elimination with partial pivoting.
use crate::{GraphError, GraphResult};
fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
fn finite(value: f64) -> GraphResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(
            "Linear solve intermediate/output exceeds finite range",
        ))
    }
}
pub(super) fn solve(
    mut a: Vec<f64>,
    mut b: Vec<f64>,
    n: usize,
    rhs: usize,
) -> GraphResult<Vec<f64>> {
    if a.iter().chain(&b).any(|v| !v.is_finite()) {
        return Err(invalid("Linear solve requires finite input"));
    }
    for row in 0..n {
        let scale = a[row * n..(row + 1) * n]
            .iter()
            .fold(0.0_f64, |s, v| s.max(v.abs()));
        if scale == 0.0 {
            return Err(invalid("Matrix is singular: zero coefficient row"));
        }
        for v in &mut a[row * n..(row + 1) * n] {
            let scaled = *v / scale;
            if *v != 0.0 && scaled == 0.0 {
                return Err(invalid("Coefficient range lost during row scaling"));
            }
            *v = scaled;
        }
        for v in &mut b[row * rhs..(row + 1) * rhs] {
            let scaled = finite(*v / scale)?;
            if *v != 0.0 && scaled == 0.0 {
                return Err(invalid("Right-hand side range lost during row scaling"));
            }
            *v = scaled;
        }
    }
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if a[row * n + col].abs() > a[pivot * n + col].abs() {
                pivot = row;
            }
        }
        if a[pivot * n + col] == 0.0 {
            return Err(invalid("Matrix is singular: zero elimination pivot"));
        }
        if pivot != col {
            for j in 0..n {
                a.swap(col * n + j, pivot * n + j);
            }
            for j in 0..rhs {
                b.swap(col * rhs + j, pivot * rhs + j);
            }
        }
        for row in col + 1..n {
            let factor = a[row * n + col] / a[col * n + col];
            a[row * n + col] = 0.0;
            for j in col + 1..n {
                a[row * n + j] = finite((-factor).mul_add(a[col * n + j], a[row * n + j]))?;
            }
            for j in 0..rhs {
                b[row * rhs + j] = finite((-factor).mul_add(b[col * rhs + j], b[row * rhs + j]))?;
            }
        }
    }
    for row in (0..n).rev() {
        for j in 0..rhs {
            let mut residual = super::dot_product::Dot::new();
            residual.add(b[row * rhs + j], 1.0);
            for col in row + 1..n {
                residual.add(-a[row * n + col], b[col * rhs + j]);
            }
            b[row * rhs + j] = finite(finite(residual.total())? / a[row * n + row])?;
        }
    }
    Ok(b)
}

/// Rounded residual B-A*X, with exact finite dot accumulation before rounding.
pub(super) fn residual(
    a: &[f64],
    b: &[f64],
    x: &[f64],
    n: usize,
    rhs: usize,
) -> GraphResult<Vec<f64>> {
    let mut result = super::factorization::zeros(n, rhs)?;
    for row in 0..n {
        for col in 0..rhs {
            let mut dot = super::dot_product::Dot::new();
            dot.add(b[row * rhs + col], 1.);
            for j in 0..n {
                dot.add(-a[row * n + j], x[j * rhs + col]);
            }
            result[row * rhs + col] = finite(dot.total())?;
        }
    }
    Ok(result)
}
// Log infinity norm avoids overflow in row sums and in the error denominator.
fn log_inf_norm(values: &[f64], cols: usize) -> f64 {
    let scale = values.iter().fold(0.0_f64, |scale, v| scale.max(v.abs()));
    if scale == 0. || cols == 0 {
        return f64::NEG_INFINITY;
    }
    let mut maximum = 0.0_f64;
    for row in values.chunks_exact(cols) {
        let mut dot = super::dot_product::Dot::new();
        for v in row {
            dot.add(v.abs() / scale, 1.);
        }
        maximum = maximum.max(dot.total());
    }
    scale.ln() + maximum.ln()
}
/// Normwise backward-error estimate using rounded residuals, not a certified bound.
pub(super) fn backward_error(
    a: &[f64],
    b: &[f64],
    x: &[f64],
    residual: &[f64],
    n: usize,
    rhs: usize,
) -> f64 {
    let numerator = log_inf_norm(residual, rhs);
    if numerator == f64::NEG_INFINITY {
        return 0.;
    }
    let product = log_inf_norm(a, n) + log_inf_norm(x, rhs);
    let right = log_inf_norm(b, rhs);
    let largest = product.max(right);
    let denominator = if largest == f64::NEG_INFINITY {
        largest
    } else {
        largest + ((product - largest).exp() + (right - largest).exp()).ln()
    };
    (numerator - denominator).exp()
}
