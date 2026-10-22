//! Validated canonical CSR construction and checked access/update helpers.
use super::{
    numeric_type::NumericType,
    unified_matrix::{MatrixError, MatrixResult, Shape, SparseMatrix},
};
fn invalid(message: &str) -> MatrixError {
    MatrixError::InvalidInput(message.into())
}
impl<T: NumericType> SparseMatrix<T> {
    /// Construct canonical CSR: sorted unique columns and no stored zeros.
    pub fn new(
        rows: usize,
        cols: usize,
        values: Vec<T>,
        col_indices: Vec<usize>,
        row_pointers: Vec<usize>,
    ) -> MatrixResult<Self> {
        let result = Self {
            values,
            col_indices,
            row_pointers,
            shape: Shape::new(rows, cols),
        };
        result.validate()?;
        Ok(result)
    }
    /// Check shape/pointer/value consistency and canonical row order.
    pub fn validate(&self) -> MatrixResult<()> {
        if self.shape.rows.checked_mul(self.shape.cols).is_none()
            || self.shape.rows.checked_add(1) != Some(self.row_pointers.len())
            || self.values.len() != self.col_indices.len()
            || self.row_pointers.first() != Some(&0)
            || self.row_pointers.last() != Some(&self.values.len())
        {
            return Err(invalid("Invalid CSR shape, pointer or value lengths"));
        }
        for row in 0..self.shape.rows {
            self.row_range(row)?;
        }
        Ok(())
    }
    fn row_range(&self, row: usize) -> MatrixResult<std::ops::Range<usize>> {
        let start = *self
            .row_pointers
            .get(row)
            .ok_or_else(|| invalid("CSR row out of bounds"))?;
        let end = *self
            .row_pointers
            .get(row + 1)
            .ok_or_else(|| invalid("Missing CSR row pointer"))?;
        let columns = self
            .col_indices
            .get(start..end)
            .ok_or_else(|| invalid("Invalid CSR row pointer range"))?;
        let values = self
            .values
            .get(start..end)
            .ok_or_else(|| invalid("Invalid CSR value range"))?;
        if columns.iter().any(|&col| col >= self.shape.cols)
            || columns.windows(2).any(|w| w[0] >= w[1])
            || values.iter().any(|&v| v == T::zero())
        {
            return Err(invalid(
                "CSR rows require sorted unique in-range columns and nonzero values",
            ));
        }
        Ok(start..end)
    }
    /// Read an entry; implicit values are zero. Invalid coordinates/layout error.
    pub fn get(&self, row: usize, col: usize) -> MatrixResult<T> {
        if row >= self.shape.rows || col >= self.shape.cols {
            return Err(invalid("CSR coordinate out of bounds"));
        }
        let range = self.row_range(row)?;
        Ok(match self.col_indices[range.clone()].binary_search(&col) {
            Ok(index) => self.values[range.start + index],
            Err(_) => T::zero(),
        })
    }
    /// Set/remove/insert an entry, retaining canonical CSR. O(nnz + rows) movement.
    pub fn set(&mut self, row: usize, col: usize, value: T) -> MatrixResult<()> {
        self.validate()?;
        if row >= self.shape.rows || col >= self.shape.cols {
            return Err(invalid("CSR coordinate out of bounds"));
        }
        let range = self.row_range(row)?;
        match self.col_indices[range.clone()].binary_search(&col) {
            Ok(offset) => {
                let index = range.start + offset;
                if value == T::zero() {
                    self.values.remove(index);
                    self.col_indices.remove(index);
                    for pointer in &mut self.row_pointers[row + 1..] {
                        *pointer -= 1;
                    }
                } else {
                    self.values[index] = value;
                }
            }
            Err(offset) if value != T::zero() => {
                let index = range.start + offset;
                self.values.insert(index, value);
                self.col_indices.insert(index, col);
                for pointer in &mut self.row_pointers[row + 1..] {
                    *pointer += 1;
                }
            }
            _ => {}
        }
        Ok(())
    }
    /// Materialize a checked owned row-major dense buffer.
    pub fn to_dense_values(&self) -> MatrixResult<Vec<T>> {
        self.validate()?;
        let count = self
            .shape
            .rows
            .checked_mul(self.shape.cols)
            .ok_or_else(|| invalid("CSR dense size overflow"))?;
        if count
            .checked_mul(std::mem::size_of::<T>())
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(invalid("CSR dense byte size overflow"));
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| invalid("CSR dense allocation failed"))?;
        values.resize(count, T::zero());
        for row in 0..self.shape.rows {
            for i in self.row_pointers[row]..self.row_pointers[row + 1] {
                values[row * self.shape.cols + self.col_indices[i]] = self.values[i];
            }
        }
        Ok(values)
    }
}
