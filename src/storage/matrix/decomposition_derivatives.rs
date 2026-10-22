//! Explicit real decomposition derivatives, independent of tape ownership.
use crate::{GraphError, GraphResult};
fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
fn finite(value: f64) -> GraphResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(
            "Decomposition derivative exceeds finite working range",
        ))
    }
}
fn symmetrize(a: &mut [f64], n: usize, require_symmetric: bool) -> GraphResult<()> {
    let tolerance = 8. * f64::EPSILON * n.max(1) as f64;
    for row in 0..n {
        for col in 0..row {
            if require_symmetric && (a[row * n + col] - a[col * n + row]).abs() > tolerance {
                return Err(invalid("Cholesky input tangent must be symmetric"));
            }
            let mean = a[row * n + col] / 2. + a[col * n + row] / 2.;
            a[row * n + col] = mean;
            a[col * n + row] = mean;
        }
    }
    Ok(())
}
// Solve L X=B or L^T X=B, without forming an inverse.
fn triangular(l: &[f64], mut b: Vec<f64>, n: usize, transpose: bool) -> GraphResult<Vec<f64>> {
    for step in 0..n {
        let row = if transpose { n - 1 - step } else { step };
        for col in 0..n {
            let mut dot = super::dot_product::Dot::new();
            dot.add(b[row * n + col], 1.);
            if transpose {
                for k in row + 1..n {
                    dot.add(-l[k * n + row], b[k * n + col]);
                }
            } else {
                for k in 0..row {
                    dot.add(-l[row * n + k], b[k * n + col]);
                }
            }
            b[row * n + col] = finite(finite(dot.total())? / l[row * n + row])?;
        }
    }
    Ok(b)
}
fn transpose(a: &[f64], n: usize) -> GraphResult<Vec<f64>> {
    let mut result = super::factorization::zeros(n, n)?;
    for row in 0..n {
        for col in 0..n {
            result[row * n + col] = a[col * n + row];
        }
    }
    Ok(result)
}
// x = signed mantissa in [1,2) times 2^exponent, including subnormals.
fn parts(mut x: f64) -> (f64, i32) {
    let mut adjustment = 0;
    if x.abs() < f64::MIN_POSITIVE {
        x *= (1u64 << 52) as f64;
        adjustment = -52;
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 + adjustment;
    (
        f64::from_bits((bits & ((1u64 << 63) | ((1u64 << 52) - 1))) | (1023u64 << 52)),
        exponent,
    )
}
// Avoid forming a potentially overflowing/underflowing scale ratio first.
pub(super) fn rescale(value: f64, numerator: f64, denominator: f64) -> GraphResult<f64> {
    if value == 0. {
        return Ok(value);
    }
    let (v, ve) = parts(finite(value)?);
    let (a, ae) = parts(numerator);
    let (b, be) = parts(denominator);
    let mut result = (v * a) / b;
    let mut exponent = ve + ae - be;
    while exponent > 512 {
        result *= 2f64.powi(512);
        exponent -= 512;
    }
    while exponent < -512 {
        result *= 2f64.powi(-512);
        exponent += 512;
    }
    finite(result * 2f64.powi(exponent))
}
/// Cholesky JVP or symmetric Frobenius VJP at a finite positive-definite input.
pub(super) fn cholesky(
    mut a: Vec<f64>,
    mut direction: Vec<f64>,
    n: usize,
    reverse: bool,
) -> GraphResult<Vec<f64>> {
    if direction.iter().any(|v| !v.is_finite()) {
        return Err(invalid("Cholesky derivative requires finite directions"));
    }
    let scale = super::factorization::normalize(&mut a)?;
    let l = super::factorization::cholesky(a, n)?;
    if reverse {
        // Upper factor entries are fixed zero and have zero derivative.
        for row in 0..n {
            for col in row + 1..n {
                direction[row * n + col] = 0.;
            }
        }
    }
    let weight = super::factorization::normalize(&mut direction)?;
    let mut result;
    if reverse {
        // P=Phi(L^T G); symmetrization gives a Frobenius gradient on symmetric A.
        let mut p = super::factorization::zeros(n, n)?;
        for row in 0..n {
            for col in 0..=row {
                let mut dot = super::dot_product::Dot::new();
                for k in row.max(col)..n {
                    dot.add(l[k * n + row], direction[k * n + col]);
                }
                p[row * n + col] = finite(dot.total())? * if row == col { 0.5 } else { 1. };
            }
        }
        symmetrize(&mut p, n, false)?;
        let left = triangular(&l, p, n, true)?;
        result = transpose(&triangular(&l, transpose(&left, n)?, n, true)?, n)?;
        symmetrize(&mut result, n, false)?;
    } else {
        symmetrize(&mut direction, n, true)?;
        // dL=L Phi(L^-1 dA L^-T).
        let left = triangular(&l, direction, n, false)?;
        let middle = transpose(&triangular(&l, transpose(&left, n)?, n, false)?, n)?;
        result = super::factorization::zeros(n, n)?;
        for row in 0..n {
            for col in 0..=row {
                let mut dot = super::dot_product::Dot::new();
                for k in col..=row {
                    dot.add(
                        l[row * n + k],
                        middle[k * n + col] * if k == col { 0.5 } else { 1. },
                    );
                }
                result[row * n + col] = finite(dot.total())?;
            }
        }
    }
    for value in &mut result {
        *value = rescale(*value, weight, scale.sqrt())?;
    }
    Ok(result)
}
