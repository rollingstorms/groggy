//! Canonical CSR operations with sparse storage and checked typed arithmetic.
use super::{
    checked_kernels::Product,
    unified_matrix::{MatrixError, MatrixResult, SparseMatrix},
    NumericType,
};
use std::collections::{BTreeMap, BTreeSet};
fn invalid(message: &str) -> MatrixError {
    MatrixError::InvalidInput(message.into())
}
impl<T: NumericType> SparseMatrix<T> {
    fn at(&self, row: usize, col: usize) -> T {
        let start = self.row_pointers[row];
        let end = self.row_pointers[row + 1];
        match self.col_indices[start..end].binary_search(&col) {
            Ok(i) => self.values[start + i],
            Err(_) => T::zero(),
        }
    }
    pub(crate) fn binary(&self, b: &Self, op: fn(T, T) -> Option<T>) -> MatrixResult<Self> {
        self.validate()?;
        b.validate()?;
        if self.shape != b.shape {
            return Err(invalid("CSR element shapes differ"));
        }
        let mut values = Vec::new();
        let mut indices = Vec::new();
        let mut pointers = vec![0];
        for row in 0..self.shape.rows {
            let mut i = self.row_pointers[row];
            let mut j = b.row_pointers[row];
            let ie = self.row_pointers[row + 1];
            let je = b.row_pointers[row + 1];
            while i < ie || j < je {
                let col = if i < ie && (j == je || self.col_indices[i] < b.col_indices[j]) {
                    self.col_indices[i]
                } else {
                    b.col_indices[j]
                };
                let a = if i < ie && self.col_indices[i] == col {
                    let v = self.values[i];
                    i += 1;
                    v
                } else {
                    T::zero()
                };
                let rhs = if j < je && b.col_indices[j] == col {
                    let v = b.values[j];
                    j += 1;
                    v
                } else {
                    T::zero()
                };
                let value = op(a, rhs).ok_or_else(|| invalid("CSR element arithmetic overflow"))?;
                if value != T::zero() {
                    indices.push(col);
                    values.push(value);
                }
            }
            pointers.push(values.len());
        }
        Self::new(self.shape.rows, self.shape.cols, values, indices, pointers)
    }
    /// Sparse union addition, with checked integer values and zero removal.
    pub fn add(&self, b: &Self) -> MatrixResult<Self> {
        self.binary(b, T::checked_add)
    }
    /// Sparse union subtraction.
    pub fn subtract(&self, b: &Self) -> MatrixResult<Self> {
        self.binary(b, T::checked_sub)
    }
    /// Hadamard product includes implicit-zero/nonfinite IEEE interactions.
    pub fn hadamard(&self, b: &Self) -> MatrixResult<Self> {
        self.binary(b, T::checked_mul)
    }
    /// Sparse product with accurate finite dots and checked integer accumulation.
    pub fn matmul(&self, b: &Self) -> MatrixResult<Self> {
        self.validate()?;
        b.validate()?;
        if self.shape.cols != b.shape.rows {
            return Err(invalid("CSR product dimensions differ"));
        }
        let nonfinite = self.values.iter().chain(&b.values).any(|v| !v.is_finite());
        let mut values = Vec::new();
        let mut indices = Vec::new();
        let mut pointers = vec![0];
        for row in 0..self.shape.rows {
            let start = self.row_pointers[row];
            let end = self.row_pointers[row + 1];
            let mut candidates = BTreeSet::new();
            if nonfinite {
                candidates.extend(0..b.shape.cols);
            } else {
                for entry in start..end {
                    let k = self.col_indices[entry];
                    candidates.extend(
                        b.col_indices[b.row_pointers[k]..b.row_pointers[k + 1]]
                            .iter()
                            .copied(),
                    );
                }
            }
            for col in candidates {
                let mut dot = Product::<T>::new();
                if nonfinite {
                    for k in 0..self.shape.cols {
                        dot.add(self.at(row, k), b.at(k, col))?;
                    }
                } else {
                    for entry in start..end {
                        let value = b.at(self.col_indices[entry], col);
                        if value != T::zero() {
                            dot.add(self.values[entry], value)?;
                        }
                    }
                }
                let value = dot.total()?;
                if value != T::zero() {
                    indices.push(col);
                    values.push(value);
                }
            }
            pointers.push(values.len());
        }
        Self::new(self.shape.rows, b.shape.cols, values, indices, pointers)
    }
    /// Canonical CSR transpose, O(rows+cols+nnz) storage/work.
    pub fn transpose(&self) -> MatrixResult<Self> {
        self.validate()?;
        let rows = self.shape.cols;
        let count = rows
            .checked_add(1)
            .ok_or_else(|| invalid("CSR transpose dimensions overflow"))?;
        let mut pointers = Vec::new();
        pointers
            .try_reserve_exact(count)
            .map_err(|_| invalid("CSR transpose allocation failed"))?;
        pointers.resize(count, 0usize);
        for &col in &self.col_indices {
            pointers[col + 1] += 1;
        }
        for row in 0..rows {
            pointers[row + 1] += pointers[row];
        }
        let mut next = pointers[..rows].to_vec();
        let mut indices = vec![0; self.values.len()];
        let mut values = vec![T::zero(); self.values.len()];
        for row in 0..self.shape.rows {
            for entry in self.row_pointers[row]..self.row_pointers[row + 1] {
                let col = self.col_indices[entry];
                let dest = next[col];
                next[col] += 1;
                indices[dest] = row;
                values[dest] = self.values[entry];
            }
        }
        Self::new(rows, self.shape.rows, values, indices, pointers)
    }
    /// Owned positional selection; repeats create distinct output positions.
    pub fn select(&self, rows: &[usize], cols: &[usize]) -> MatrixResult<Self> {
        self.validate()?;
        if rows.iter().any(|r| *r >= self.shape.rows) || cols.iter().any(|c| *c >= self.shape.cols)
        {
            return Err(invalid("CSR selection index out of bounds"));
        }
        let mut map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (new, &old) in cols.iter().enumerate() {
            map.entry(old).or_default().push(new);
        }
        let mut values = Vec::new();
        let mut indices = Vec::new();
        let mut pointers = vec![0];
        for &row in rows {
            let mut selected = Vec::new();
            for entry in self.row_pointers[row]..self.row_pointers[row + 1] {
                if let Some(columns) = map.get(&self.col_indices[entry]) {
                    for &col in columns {
                        selected.push((col, self.values[entry]));
                    }
                }
            }
            selected.sort_unstable_by_key(|(col, _)| *col);
            for (col, value) in selected {
                indices.push(col);
                values.push(value);
            }
            pointers.push(values.len());
        }
        Self::new(rows.len(), cols.len(), values, indices, pointers)
    }
    /// Owned sparse logical reshape, preserving flattened row order.
    pub fn reshape(&self, rows: usize, cols: usize) -> MatrixResult<Self> {
        self.validate()?;
        if rows.checked_mul(cols) != self.shape.rows.checked_mul(self.shape.cols) {
            return Err(invalid("CSR reshape element counts differ"));
        }
        let count = rows
            .checked_add(1)
            .ok_or_else(|| invalid("CSR reshape dimensions overflow"))?;
        let mut pointers = Vec::new();
        pointers
            .try_reserve_exact(count)
            .map_err(|_| invalid("CSR reshape allocation failed"))?;
        pointers.resize(count, 0usize);
        let mut indices = Vec::with_capacity(self.values.len());
        for row in 0..self.shape.rows {
            for entry in self.row_pointers[row]..self.row_pointers[row + 1] {
                let index = row * self.shape.cols + self.col_indices[entry];
                pointers[index / cols + 1] += 1;
                indices.push(index % cols);
            }
        }
        for row in 0..rows {
            pointers[row + 1] += pointers[row];
        }
        Self::new(rows, cols, self.values.clone(), indices, pointers)
    }
    pub(crate) fn try_map_into<U: NumericType>(
        &self,
        operation: impl Fn(T) -> MatrixResult<U>,
    ) -> MatrixResult<SparseMatrix<U>> {
        self.validate()?;
        let count = self
            .shape
            .rows
            .checked_mul(self.shape.cols)
            .ok_or_else(|| invalid("CSR map shape overflow"))?;
        let implicit = if self.values.len() < count {
            operation(T::zero())?
        } else {
            U::zero()
        };
        let mut values = Vec::new();
        let mut indices = Vec::new();
        let mut pointers = vec![0];
        for row in 0..self.shape.rows {
            if implicit == U::zero() {
                for entry in self.row_pointers[row]..self.row_pointers[row + 1] {
                    let value = operation(self.values[entry])?;
                    if value != U::zero() {
                        indices.push(self.col_indices[entry]);
                        values.push(value);
                    }
                }
            } else {
                for col in 0..self.shape.cols {
                    let value = match self.col_indices
                        [self.row_pointers[row]..self.row_pointers[row + 1]]
                        .binary_search(&col)
                    {
                        Ok(entry) => operation(self.values[self.row_pointers[row] + entry])?,
                        Err(_) => implicit,
                    };
                    if value != U::zero() {
                        indices.push(col);
                        values.push(value);
                    }
                }
            }
            pointers.push(values.len());
        }
        SparseMatrix::<U>::new(self.shape.rows, self.shape.cols, values, indices, pointers)
    }
    /// Preserve sparse storage; nonfinite scalars can make implicit zeros nonzero.
    pub fn scale(&self, scalar: T) -> MatrixResult<Self> {
        self.validate()?;
        let mut values = Vec::new();
        let mut indices = Vec::new();
        let mut pointers = vec![0];
        for row in 0..self.shape.rows {
            if scalar.is_finite() {
                for entry in self.row_pointers[row]..self.row_pointers[row + 1] {
                    let value = self.values[entry]
                        .checked_mul(scalar)
                        .ok_or_else(|| invalid("CSR scalar overflow"))?;
                    if value != T::zero() {
                        indices.push(self.col_indices[entry]);
                        values.push(value);
                    }
                }
            } else {
                for col in 0..self.shape.cols {
                    let value = self
                        .at(row, col)
                        .checked_mul(scalar)
                        .ok_or_else(|| invalid("CSR scalar overflow"))?;
                    if value != T::zero() {
                        indices.push(col);
                        values.push(value);
                    }
                }
            }
            pointers.push(values.len());
        }
        Self::new(self.shape.rows, self.shape.cols, values, indices, pointers)
    }
}
