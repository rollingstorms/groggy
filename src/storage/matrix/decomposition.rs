//! In-house dense real Jacobi decompositions. Inputs and results are row-major.
//! Bounded sweeps and explicit domain/convergence errors replace guessed outputs.
use crate::{GraphError, GraphResult};

const SWEEPS: usize = 100;

fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
fn identity(n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    out
}
fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; a.len()];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
    out
}
fn scale(a: &[f64]) -> GraphResult<f64> {
    if a.iter().any(|x| !x.is_finite()) {
        return Err(invalid("Decomposition requires finite input values"));
    }
    let maximum = a.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
    Ok(if maximum == 0.0 { 1.0 } else { maximum })
}
fn rotation(tau: f64) -> (f64, f64, f64) {
    let t = tau.signum() / (tau.abs() + tau.hypot(1.0));
    let c = 1.0 / t.hypot(1.0);
    (c, c * t, t)
}
fn rotate_columns(a: &mut [f64], rows: usize, cols: usize, p: usize, q: usize, c: f64, s: f64) {
    for row in 0..rows {
        let x = a[row * cols + p];
        let y = a[row * cols + q];
        a[row * cols + p] = c * x - s * y;
        a[row * cols + q] = s * x + c * y;
    }
}

/// Descending eigenvalues and an orthonormal column eigenvector matrix.
pub(super) fn symmetric_eigen(input: Vec<f64>, n: usize) -> GraphResult<(Vec<f64>, Vec<f64>)> {
    symmetric_eigen_with_limit(input, n, SWEEPS)
}

fn symmetric_eigen_with_limit(
    input: Vec<f64>,
    n: usize,
    sweeps: usize,
) -> GraphResult<(Vec<f64>, Vec<f64>)> {
    let scaling = scale(&input)?;
    let mut a: Vec<_> = input.iter().map(|x| x / scaling).collect();
    let mut vectors = identity(n);
    let tolerance = 8.0 * f64::EPSILON * n.max(1) as f64;
    for p in 0..n {
        for q in p + 1..n {
            if (a[p * n + q] - a[q * n + p]).abs() > tolerance {
                return Err(GraphError::NotImplemented {
                    feature:
                        "General nonsymmetric eigendecomposition; real symmetric input is required"
                            .into(),
                    tracking_issue: None,
                });
            }
            let value = a[p * n + q] / 2.0 + a[q * n + p] / 2.0;
            a[p * n + q] = value;
            a[q * n + p] = value;
        }
    }
    let mut converged = false;
    for _ in 0..sweeps {
        let mut maximum = 0.0_f64;
        for p in 0..n {
            for q in p + 1..n {
                let off = a[p * n + q];
                maximum = maximum.max(off.abs());
                if off.abs() <= tolerance {
                    continue;
                }
                let tau = (a[q * n + q] - a[p * n + p]) / (2.0 * off);
                let (c, s, t) = rotation(tau);
                a[p * n + p] -= t * off;
                a[q * n + q] += t * off;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for k in 0..n {
                    if k == p || k == q {
                        continue;
                    }
                    let x = a[k * n + p];
                    let y = a[k * n + q];
                    a[k * n + p] = c * x - s * y;
                    a[p * n + k] = a[k * n + p];
                    a[k * n + q] = s * x + c * y;
                    a[q * n + k] = a[k * n + q];
                }
                rotate_columns(&mut vectors, n, n, p, q, c, s);
            }
        }
        if maximum <= tolerance {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(invalid(&format!(
            "Symmetric eigendecomposition did not converge in {sweeps} sweeps"
        )));
    }
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&i, &j| a[j * n + j].total_cmp(&a[i * n + i]).then(i.cmp(&j)));
    let mut values = Vec::with_capacity(n);
    let mut sorted = vec![0.0; n * n];
    for (col, &old) in order.iter().enumerate() {
        let value = a[old * n + old] * scaling;
        if !value.is_finite() {
            return Err(invalid("Eigenvalue exceeds finite f64 range"));
        }
        values.push(value);
        let pivot = (0..n).max_by(|&i, &j| {
            vectors[i * n + old]
                .abs()
                .total_cmp(&vectors[j * n + old].abs())
        });
        let sign = pivot.map_or(1.0, |i| {
            if vectors[i * n + old] < 0.0 {
                -1.0
            } else {
                1.0
            }
        });
        for row in 0..n {
            sorted[row * n + col] = sign * vectors[row * n + old];
        }
    }
    Ok((values, sorted))
}

pub(super) struct Svd {
    pub u: Vec<f64>,
    pub singular: Vec<f64>,
    pub vt: Vec<f64>,
}

/// Full orthogonal U and Vt; min(rows,cols) descending singular values.
pub(super) fn svd(input: Vec<f64>, rows: usize, cols: usize) -> GraphResult<Svd> {
    // Check full output dimensions even when an empty rectangular input is cheap.
    for dimension in [rows, cols] {
        dimension
            .checked_mul(dimension)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or_else(|| invalid("Full SVD output dimensions overflow"))?;
    }
    scale(&input)?;
    if rows < cols {
        let result = tall_svd(transpose(&input, rows, cols), cols, rows, SWEEPS)?;
        return Ok(Svd {
            u: transpose(&result.vt, rows, rows),
            singular: result.singular,
            vt: transpose(&result.u, cols, cols),
        });
    }
    tall_svd(input, rows, cols, SWEEPS)
}

fn column_norm(a: &[f64], rows: usize, cols: usize, col: usize) -> f64 {
    (0..rows).fold(0.0_f64, |norm, row| norm.hypot(a[row * cols + col]))
}

fn tall_svd(input: Vec<f64>, rows: usize, cols: usize, sweeps: usize) -> GraphResult<Svd> {
    let scaling = scale(&input)?;
    let mut a: Vec<_> = input.iter().map(|x| x / scaling).collect();
    let mut v = identity(cols);
    let norm = a.iter().fold(0.0_f64, |norm, x| norm.hypot(*x));
    let dimension = rows.max(cols).max(1) as f64;
    let floor = f64::EPSILON * dimension * norm;
    let tolerance = 8.0 * f64::EPSILON * dimension;
    let mut converged = false;
    for _ in 0..sweeps {
        let mut changed = false;
        for p in 0..cols {
            for q in p + 1..cols {
                let np = column_norm(&a, rows, cols, p);
                let nq = column_norm(&a, rows, cols, q);
                if np <= floor || nq <= floor {
                    continue;
                }
                let correlation: f64 = (0..rows)
                    .map(|r| (a[r * cols + p] / np) * (a[r * cols + q] / nq))
                    .sum();
                if correlation.abs() <= tolerance {
                    continue;
                }
                let pair_scale = np.max(nq);
                let alpha = np / pair_scale;
                let beta = nq / pair_scale;
                let gamma = correlation * alpha * beta;
                if gamma == 0.0 {
                    continue;
                }
                let tau = (beta * beta - alpha * alpha) / (2.0 * gamma);
                let (c, s, _) = rotation(tau);
                rotate_columns(&mut a, rows, cols, p, q, c, s);
                rotate_columns(&mut v, cols, cols, p, q, c, s);
                changed = true;
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(invalid(&format!("SVD did not converge in {sweeps} sweeps")));
    }
    let norms: Vec<_> = (0..cols).map(|j| column_norm(&a, rows, cols, j)).collect();
    let mut order: Vec<_> = (0..cols).collect();
    order.sort_by(|&i, &j| norms[j].total_cmp(&norms[i]).then(i.cmp(&j)));
    let mut singular = Vec::with_capacity(cols);
    let mut u = vec![0.0; rows * rows];
    let mut vt = vec![0.0; cols * cols];
    let mut rank = 0;
    for (col, &old) in order.iter().enumerate() {
        let sigma = if norms[old] > floor {
            norms[old] * scaling
        } else {
            0.0
        };
        if !sigma.is_finite() {
            return Err(invalid("Singular value exceeds finite f64 range"));
        }
        singular.push(sigma);
        for row in 0..cols {
            vt[col * cols + row] = v[row * cols + old];
        }
        if sigma > 0.0 {
            let mut vector: Vec<_> = (0..rows).map(|r| a[r * cols + old] / norms[old]).collect();
            orthogonalize(&mut vector, &u, rows, col)?;
            for row in 0..rows {
                u[row * rows + col] = vector[row];
            }
            rank += 1;
        }
    }
    // Complete the left null space, including rank-deficient and zero matrices.
    let mut candidate = 0;
    for col in rank..rows {
        let mut accepted = false;
        while candidate < rows {
            let mut vector = vec![0.0; rows];
            vector[candidate] = 1.0;
            candidate += 1;
            if orthogonalize(&mut vector, &u, rows, col).is_ok() {
                for row in 0..rows {
                    u[row * rows + col] = vector[row];
                }
                accepted = true;
                break;
            }
        }
        if !accepted {
            return Err(invalid("Could not complete SVD orthonormal basis"));
        }
    }
    Ok(Svd { u, singular, vt })
}

fn orthogonalize(
    vector: &mut [f64],
    basis: &[f64],
    rows: usize,
    columns: usize,
) -> GraphResult<()> {
    for _ in 0..2 {
        for col in 0..columns {
            let dot: f64 = (0..rows).map(|r| vector[r] * basis[r * rows + col]).sum();
            for row in 0..rows {
                vector[row] -= dot * basis[row * rows + col];
            }
        }
    }
    let norm = vector.iter().fold(0.0_f64, |norm, x| norm.hypot(*x));
    if !norm.is_finite() || norm <= 64.0 * f64::EPSILON {
        return Err(invalid("Degenerate orthogonal basis vector"));
    }
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exhausted_sweep_budget_returns_errors_without_partial_answers() {
        assert!(symmetric_eigen_with_limit(vec![2., 1., 1., 2.], 2, 0).is_err());
        assert!(tall_svd(vec![1., 2., 3., 4.], 2, 2, 0).is_err());
        assert!(svd(vec![], usize::MAX, 0).is_err());
    }
}
