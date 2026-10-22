//! Dense truncated-SVD minimum-norm least squares, with explicit working range.
use crate::{GraphError, GraphResult};
fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
pub(super) fn solve(
    mut a: Vec<f64>,
    mut b: Vec<f64>,
    rows: usize,
    cols: usize,
    rhs: usize,
    relative: f64,
) -> GraphResult<Vec<f64>> {
    if a.iter().chain(&b).any(|v| !v.is_finite()) {
        return Err(invalid("Least squares requires finite input"));
    }
    let mut solution = super::factorization::zeros(cols, rhs)?;
    if rows == 0 || cols == 0 || rhs == 0 || a.iter().all(|v| *v == 0.) {
        return Ok(solution);
    }
    let scale = super::factorization::normalize(&mut a)?;
    for value in &mut b {
        let scaled = *value / scale;
        if !scaled.is_finite() || (*value != 0. && scaled == 0.) {
            return Err(invalid(
                "Least-squares RHS exceeds normalized working range",
            ));
        }
        *value = scaled;
    }
    let factor = super::decomposition::svd_economy(a, rows, cols)?;
    let largest = factor.singular.first().copied().unwrap_or(0.);
    let k = rows.min(cols);
    let mut coefficients = super::factorization::zeros(k, rhs)?;
    for component in 0..k {
        let sigma = factor.singular[component];
        if sigma == 0. || sigma / largest <= relative {
            continue;
        }
        for col in 0..rhs {
            let mut dot = super::dot_product::Dot::new();
            for row in 0..rows {
                dot.add(factor.u[row * k + component], b[row * rhs + col]);
            }
            let coefficient = dot.total() / sigma;
            if !coefficient.is_finite() {
                return Err(invalid("Least-squares projection exceeds finite range"));
            }
            coefficients[component * rhs + col] = coefficient;
        }
    }
    for row in 0..cols {
        for col in 0..rhs {
            let mut dot = super::dot_product::Dot::new();
            for component in 0..k {
                dot.add(
                    factor.vt[component * cols + row],
                    coefficients[component * rhs + col],
                );
            }
            let value = dot.total();
            if !value.is_finite() {
                return Err(invalid("Least-squares solution exceeds finite range"));
            }
            solution[row * rhs + col] = value;
        }
    }
    Ok(solution)
}
