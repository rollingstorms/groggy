//! Checked dense scratch operations shared by decomposition derivatives.
use crate::{GraphError, GraphResult};
pub(super) fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
pub(super) fn finite(x: f64) -> GraphResult<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(invalid("Derivative arithmetic exceeds finite range"))
    }
}
pub(super) fn zeros(m: usize, n: usize) -> GraphResult<Vec<f64>> {
    super::factorization::zeros(m, n)
}
pub(super) fn transpose(a: &[f64], m: usize, n: usize) -> GraphResult<Vec<f64>> {
    let mut out = zeros(n, m)?;
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    Ok(out)
}
pub(super) fn mul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> GraphResult<Vec<f64>> {
    let mut out = zeros(m, n)?;
    for i in 0..m {
        for j in 0..n {
            let mut dot = super::dot_product::Dot::new();
            for l in 0..k {
                dot.add(a[i * k + l], b[l * n + j]);
            }
            out[i * n + j] = finite(dot.total())?;
        }
    }
    Ok(out)
}
pub(super) fn combine(a: &[f64], b: &[f64], sign: f64) -> GraphResult<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let mut dot = super::dot_product::Dot::new();
            dot.add(x, 1.);
            dot.add(y, sign);
            finite(dot.total())
        })
        .collect()
}
pub(super) fn scale(a: &mut [f64], numerator: f64, denominator: f64) -> GraphResult<()> {
    for v in a {
        *v = super::decomposition_derivatives::rescale(*v, numerator, denominator)?;
    }
    Ok(())
}
pub(super) fn symmetric(a: &mut [f64], n: usize) -> GraphResult<()> {
    let tolerance = 8. * f64::EPSILON * n.max(1) as f64;
    for i in 0..n {
        for j in 0..i {
            if (a[i * n + j] - a[j * n + i]).abs() > tolerance {
                return Err(invalid(
                    "Spectral derivative requires a symmetric input tangent",
                ));
            }
            let mean = a[i * n + j] / 2. + a[j * n + i] / 2.;
            a[i * n + j] = mean;
            a[j * n + i] = mean;
        }
    }
    Ok(())
}
pub(super) fn symmetrize(a: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..i {
            let mean = a[i * n + j] / 2. + a[j * n + i] / 2.;
            a[i * n + j] = mean;
            a[j * n + i] = mean;
        }
    }
}
pub(super) fn normalize(a: &mut [f64]) -> GraphResult<f64> {
    super::factorization::normalize(a)
}
pub(super) fn thin_columns(a: &[f64], rows: usize, cols: usize, k: usize) -> GraphResult<Vec<f64>> {
    let mut out = zeros(rows, k)?;
    for i in 0..rows {
        out[i * k..(i + 1) * k].copy_from_slice(&a[i * cols..i * cols + k]);
    }
    Ok(out)
}
/// Solve op(A) X=B for triangular A, retaining the RHS column count.
pub(super) fn solve_triangular(
    a: &[f64],
    mut b: Vec<f64>,
    n: usize,
    rhs: usize,
    lower: bool,
    trans: bool,
) -> GraphResult<Vec<f64>> {
    for step in 0..n {
        let row = if lower != trans { step } else { n - 1 - step };
        for col in 0..rhs {
            let mut dot = super::dot_product::Dot::new();
            dot.add(b[row * rhs + col], 1.);
            if lower != trans {
                for k in 0..row {
                    dot.add(
                        -if trans {
                            a[k * n + row]
                        } else {
                            a[row * n + k]
                        },
                        b[k * rhs + col],
                    );
                }
            } else {
                for k in row + 1..n {
                    dot.add(
                        -if trans {
                            a[k * n + row]
                        } else {
                            a[row * n + k]
                        },
                        b[k * rhs + col],
                    );
                }
            }
            if a[row * n + row] == 0. {
                return Err(invalid("Derivative triangular factor is singular"));
            }
            b[row * rhs + col] = finite(finite(dot.total())? / a[row * n + row])?;
        }
    }
    Ok(b)
}
pub(super) fn right_solve(
    b: &[f64],
    rows: usize,
    a: &[f64],
    n: usize,
    lower: bool,
    trans: bool,
) -> GraphResult<Vec<f64>> {
    transpose(
        &solve_triangular(a, transpose(b, rows, n)?, n, rows, lower, !trans)?,
        n,
        rows,
    )
}
