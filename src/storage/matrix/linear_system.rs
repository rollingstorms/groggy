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
