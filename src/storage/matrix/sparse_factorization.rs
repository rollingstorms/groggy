//! In-house sparse numeric LU/Cholesky with explicit fill-in and no dense A.
use crate::storage::matrix::{unified_matrix::SparseMatrix, DType, NumericType};
use crate::{GraphError, GraphResult};
use std::collections::{BTreeMap, BTreeSet};
fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
fn finite(v: f64) -> GraphResult<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(invalid("Sparse factorization exceeds finite working range"))
    }
}
type Rows = Vec<BTreeMap<usize, f64>>;
fn rows<T: NumericType>(a: &SparseMatrix<T>, normalized: bool) -> GraphResult<(Rows, f64)> {
    a.validate().map_err(|e| invalid(&e.to_string()))?;
    if a.shape.rows != a.shape.cols || !matches!(T::DTYPE, DType::Float32 | DType::Float64) {
        return Err(invalid("Sparse factors require square floating input"));
    }
    let scale = a.values.iter().try_fold(0.0_f64, |scale, v| {
        let value = finite(v.to_f64())?;
        Ok::<_, GraphError>(scale.max(value.abs()))
    })?;
    let scale = if !normalized || scale == 0. {
        1.
    } else {
        scale
    };
    let mut rows = vec![BTreeMap::new(); a.shape.rows];
    for (row, out) in rows.iter_mut().enumerate() {
        for entry in a.row_pointers[row]..a.row_pointers[row + 1] {
            let value = a.values[entry].to_f64() / scale;
            if value == 0. {
                return Err(invalid("Sparse normalization loses input range"));
            }
            out.insert(a.col_indices[entry], value);
        }
    }
    Ok((rows, scale))
}
fn csr(rows: Rows, scale: f64) -> GraphResult<SparseMatrix<f64>> {
    let n = rows.len();
    let mut values = Vec::new();
    let mut indices = Vec::new();
    let mut pointers = vec![0];
    for row in rows {
        for (col, value) in row {
            let value = super::decomposition_derivatives::rescale(value, scale, 1.)?;
            if value != 0. {
                values.push(value);
                indices.push(col);
            }
        }
        pointers.push(values.len());
    }
    SparseMatrix::new(n, n, values, indices, pointers).map_err(|e| invalid(&e.to_string()))
}
pub(super) struct Lu {
    pub lower: SparseMatrix<f64>,
    pub upper: SparseMatrix<f64>,
    pub permutation: Vec<usize>,
    pub swaps: usize,
}
pub(super) fn lu<T: NumericType>(a: &SparseMatrix<T>) -> GraphResult<Lu> {
    let (mut u, scale) = rows(a, false)?;
    let n = u.len();
    let mut lower = vec![BTreeMap::new(); n];
    let mut columns = vec![BTreeSet::new(); n];
    let mut permutation: Vec<_> = (0..n).collect();
    let mut swaps = 0;
    for (row, values) in u.iter().enumerate() {
        for &col in values.keys() {
            columns[col].insert(row);
        }
    }
    for col in 0..n {
        let mut pivot = None;
        let mut magnitude = 0.;
        for &row in columns[col].range(col..) {
            let value = u[row][&col].abs();
            if value > magnitude {
                magnitude = value;
                pivot = Some(row);
            }
        }
        let Some(pivot) = pivot else {
            continue;
        };
        if pivot != col {
            swaps += 1;
            for &key in u[col].keys() {
                columns[key].remove(&col);
            }
            for &key in u[pivot].keys() {
                columns[key].remove(&pivot);
            }
            u.swap(col, pivot);
            lower.swap(col, pivot);
            permutation.swap(col, pivot);
            for &key in u[col].keys() {
                columns[key].insert(col);
            }
            for &key in u[pivot].keys() {
                columns[key].insert(pivot);
            }
        }
        let diagonal = u[col][&col];
        let pivot_row: Vec<_> = u[col].range(col + 1..).map(|(&k, &v)| (k, v)).collect();
        let affected: Vec<_> = columns[col].range(col + 1..).copied().collect();
        for row in affected {
            let multiplier = finite(u[row][&col] / diagonal)?;
            if multiplier != 0. {
                lower[row].insert(col, multiplier);
            }
            u[row].remove(&col);
            columns[col].remove(&row);
            for &(key, value) in &pivot_row {
                let old = u[row].get(&key).copied().unwrap_or(0.);
                let updated = finite((-multiplier).mul_add(value, old))?;
                if updated == 0. {
                    u[row].remove(&key);
                    columns[key].remove(&row);
                } else {
                    u[row].insert(key, updated);
                    columns[key].insert(row);
                }
            }
        }
    }
    for (i, row) in lower.iter_mut().enumerate() {
        row.insert(i, 1.);
    }
    Ok(Lu {
        lower: csr(lower, 1.)?,
        upper: csr(u, scale)?,
        permutation,
        swaps,
    })
}
pub(super) fn cholesky<T: NumericType>(a: &SparseMatrix<T>) -> GraphResult<SparseMatrix<f64>> {
    let (original, scale) = rows(a, true)?;
    let n = original.len();
    let tolerance = 8. * f64::EPSILON * n.max(1) as f64;
    let mut work = vec![BTreeMap::new(); n];
    let mut columns = vec![BTreeSet::new(); n];
    for (row, values) in original.iter().enumerate() {
        for (&col, &value) in values {
            let mirror = original[col].get(&row).copied().unwrap_or(0.);
            if (value - mirror).abs() > tolerance {
                return Err(invalid("Sparse Cholesky requires symmetric input"));
            }
            let i = row.max(col);
            let j = row.min(col);
            let mean = if row == col {
                value
            } else {
                value / 2. + mirror / 2.
            };
            if mean != 0. {
                work[i].insert(j, mean);
                if i > j {
                    columns[j].insert(i);
                }
            }
        }
    }
    let mut lower = vec![BTreeMap::new(); n];
    for col in 0..n {
        let diagonal = work[col].get(&col).copied().unwrap_or(0.);
        if diagonal <= 0. {
            return Err(invalid(
                "Sparse Cholesky input is not positive definite at working precision",
            ));
        }
        let pivot = finite(diagonal.sqrt())?;
        lower[col].insert(col, pivot);
        let affected: Vec<_> = columns[col].iter().copied().collect();
        let mut values = Vec::with_capacity(affected.len());
        for &row in &affected {
            let value = finite(work[row][&col] / pivot)?;
            values.push(value);
            if value != 0. {
                lower[row].insert(col, value);
            }
        }
        for (i, &row) in affected.iter().enumerate() {
            for (j, &other) in affected[..=i].iter().enumerate() {
                let mut dot = super::dot_product::Dot::new();
                dot.add(work[row].get(&other).copied().unwrap_or(0.), 1.);
                dot.add(-values[i], values[j]);
                let value = finite(dot.total())?;
                if value == 0. {
                    work[row].remove(&other);
                    if row > other {
                        columns[other].remove(&row);
                    }
                } else {
                    work[row].insert(other, value);
                    if row > other {
                        columns[other].insert(row);
                    }
                }
            }
        }
        for row in affected {
            work[row].remove(&col);
        }
        work[col].clear();
        columns[col].clear();
    }
    csr(lower, scale.sqrt())
}
pub(super) fn solve<T: NumericType>(
    a: &SparseMatrix<T>,
    b: Vec<f64>,
    rhs: usize,
) -> GraphResult<Vec<f64>> {
    if b.iter().any(|v| !v.is_finite()) {
        return Err(invalid("Sparse solve requires finite RHS"));
    }
    a.validate().map_err(|e| invalid(&e.to_string()))?;
    if a.shape.rows != a.shape.cols || !matches!(T::DTYPE, DType::Float32 | DType::Float64) {
        return Err(invalid("Sparse solve requires square floating input"));
    }
    let n = a.shape.rows;
    let mut b = b;
    let mut coefficients = Vec::with_capacity(a.values.len());
    for row in 0..n {
        let start = a.row_pointers[row];
        let end = a.row_pointers[row + 1];
        let scale = a.values[start..end].iter().try_fold(0.0_f64, |s, v| {
            Ok::<_, GraphError>(s.max(finite(v.to_f64())?.abs()))
        })?;
        if scale == 0. {
            return Err(invalid("Sparse solve is singular: zero coefficient row"));
        }
        for value in &a.values[start..end] {
            let scaled = value.to_f64() / scale;
            if scaled == 0. {
                return Err(invalid("Sparse row equilibration loses coefficient range"));
            }
            coefficients.push(scaled);
        }
        for value in &mut b[row * rhs..(row + 1) * rhs] {
            let scaled = finite(*value / scale)?;
            if *value != 0. && scaled == 0. {
                return Err(invalid("Sparse row equilibration loses RHS range"));
            }
            *value = scaled;
        }
    }
    let coefficients = SparseMatrix::new(
        n,
        n,
        coefficients,
        a.col_indices.clone(),
        a.row_pointers.clone(),
    )
    .map_err(|e| invalid(&e.to_string()))?;
    let factor = lu(&coefficients)?;
    let n = a.shape.rows;
    let mut out = super::factorization::zeros(n, rhs)?;
    for row in 0..n {
        for col in 0..rhs {
            let mut dot = super::dot_product::Dot::new();
            dot.add(b[factor.permutation[row] * rhs + col], 1.);
            for entry in factor.lower.row_pointers[row]..factor.lower.row_pointers[row + 1] {
                let k = factor.lower.col_indices[entry];
                if k < row {
                    dot.add(-factor.lower.values[entry], out[k * rhs + col]);
                }
            }
            out[row * rhs + col] = finite(dot.total())?;
        }
    }
    for row in (0..n).rev() {
        let diagonal = factor
            .upper
            .get(row, row)
            .map_err(|e| invalid(&e.to_string()))?;
        if diagonal == 0. {
            return Err(invalid("Sparse solve is singular: zero pivot"));
        }
        for col in 0..rhs {
            let mut dot = super::dot_product::Dot::new();
            dot.add(out[row * rhs + col], 1.);
            for entry in factor.upper.row_pointers[row]..factor.upper.row_pointers[row + 1] {
                let k = factor.upper.col_indices[entry];
                if k > row {
                    dot.add(-factor.upper.values[entry], out[k * rhs + col]);
                }
            }
            out[row * rhs + col] = finite(finite(dot.total())? / diagonal)?;
        }
    }
    Ok(out)
}

pub(super) fn slogdet<T: NumericType>(a: &SparseMatrix<T>) -> GraphResult<(f64, f64)> {
    let factor = lu(a)?;
    let mut sign = if factor.swaps.is_multiple_of(2) {
        1.
    } else {
        -1.
    };
    let mut logs = Vec::with_capacity(a.shape.rows);
    for row in 0..a.shape.rows {
        let pivot = factor
            .upper
            .get(row, row)
            .map_err(|e| invalid(&e.to_string()))?;
        if pivot == 0. {
            return Ok((0., f64::NEG_INFINITY));
        }
        sign *= pivot.signum();
        logs.push(pivot.abs().ln());
    }
    Ok((sign, crate::storage::numeric::reductions::sum(&logs)))
}
