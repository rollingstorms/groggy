//! Core Matrix Operations
//!
//! This module provides additional operations and utilities for the unified matrix system.

use crate::storage::matrix::{
    numeric_type::NumericType,
    unified_matrix::{MatrixError, MatrixResult, UnifiedMatrix},
};

use super::checked_kernels::NumericValue;
use crate::storage::array::{numeric_arithmetic, numeric_reductions as reductions};
use std::borrow::Cow;

impl<T: NumericType> UnifiedMatrix<T> {
    /// Owned positional concatenation. Two CSR inputs produce canonical CSR.
    pub fn concatenate(&self, other: &Self, axis: usize) -> MatrixResult<Self> {
        let (rows, cols) = match axis {
            0 if self.cols() == other.cols() => {
                (self.rows().checked_add(other.rows()), Some(self.cols()))
            }
            1 if self.rows() == other.rows() => {
                (Some(self.rows()), self.cols().checked_add(other.cols()))
            }
            _ => {
                return Err(MatrixError::InvalidShape(
                    "Concatenation requires axis 0/1 and matching other dimension".into(),
                ))
            }
        };
        let (rows, cols) = rows
            .zip(cols)
            .ok_or_else(|| MatrixError::InvalidShape("Concatenation dimension overflow".into()))?;
        checked_count(rows, cols)?;
        if let (Some(a), Some(b)) = (self.checked_sparse()?, other.checked_sparse()?) {
            let mut values = Vec::new();
            let mut indices = Vec::new();
            let mut pointers = vec![0];
            if axis == 0 {
                for source in [a, b] {
                    for row in 0..source.shape.rows {
                        let range = source.row_pointers[row]..source.row_pointers[row + 1];
                        values.extend_from_slice(&source.values[range.clone()]);
                        indices.extend_from_slice(&source.col_indices[range]);
                        pointers.push(values.len());
                    }
                }
            } else {
                for row in 0..rows {
                    for (source, offset) in [(a, 0), (b, self.cols())] {
                        let range = source.row_pointers[row]..source.row_pointers[row + 1];
                        values.extend_from_slice(&source.values[range.clone()]);
                        indices.extend(source.col_indices[range].iter().map(|col| col + offset));
                    }
                    pointers.push(values.len());
                }
            }
            return Self::from_csr(rows, cols, values, indices, pointers);
        }
        let (a, b) = (self.to_vec()?, other.to_vec()?);
        let mut values = reserve_data::<T>(rows, cols)?;
        if axis == 0 {
            values.extend(a);
            values.extend(b);
        } else {
            for row in 0..rows {
                values.extend_from_slice(&a[row * self.cols()..(row + 1) * self.cols()]);
                values.extend_from_slice(&b[row * other.cols()..(row + 1) * other.cols()]);
            }
        }
        Self::from_data(values, rows, cols)
    }
    /// Fallible typed map with canonical CSR and checked logical layouts.
    pub fn try_map_into<U: NumericType>(
        &self,
        operation: impl Fn(T) -> MatrixResult<U>,
    ) -> MatrixResult<UnifiedMatrix<U>> {
        if let Some(sparse) = self.checked_sparse()? {
            let mapped = sparse.try_map_into(operation)?;
            return UnifiedMatrix::<U>::from_csr(
                mapped.shape.rows,
                mapped.shape.cols,
                mapped.values,
                mapped.col_indices,
                mapped.row_pointers,
            );
        }
        let output = self
            .to_vec()?
            .into_iter()
            .map(operation)
            .collect::<MatrixResult<Vec<_>>>()?;
        UnifiedMatrix::<U>::from_data(output, self.rows(), self.cols())
    }
    /// Checked owned elementwise map; zero-preserving CSR transforms stay sparse.
    pub fn try_map_elementwise(
        &self,
        operation: impl Fn(T) -> MatrixResult<T>,
    ) -> MatrixResult<Self> {
        self.try_map_into(operation)
    }
    /// Checked logical values; CSR norms/statistics need only its stored entries.
    pub(crate) fn reduction_values(&self) -> MatrixResult<Cow<'_, [T]>> {
        if let Some(sparse) = self.checked_sparse()? {
            return Ok(Cow::Borrowed(&sparse.values));
        }
        self.to_vec().map(Cow::Owned)
    }
    /// Pass each axis's stored observations and full sample count to a reducer.
    /// Column reduction builds a CSR transpose once; dense layouts are checked once.
    pub(crate) fn reduce_axis<R>(
        &self,
        columns: bool,
        operation: impl Fn(&[T], usize) -> MatrixResult<R>,
    ) -> MatrixResult<Vec<R>> {
        let (outputs, samples) = if columns {
            (self.cols(), self.rows())
        } else {
            (self.rows(), self.cols())
        };
        if let Some(sparse) = self.checked_sparse()? {
            let transposed;
            let source = if columns {
                transposed = sparse.transpose()?;
                &transposed
            } else {
                sparse
            };
            return (0..outputs)
                .map(|row| {
                    operation(
                        &source.values[source.row_pointers[row]..source.row_pointers[row + 1]],
                        samples,
                    )
                })
                .collect();
        }
        let values = self.to_vec()?;
        (0..outputs)
            .map(|index| {
                if !columns {
                    return operation(&values[index * samples..(index + 1) * samples], samples);
                }
                let column: Vec<T> = (0..samples)
                    .map(|row| values[row * self.cols() + index])
                    .collect();
                operation(&column, samples)
            })
            .collect()
    }
}

fn checked_count(rows: usize, cols: usize) -> MatrixResult<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| MatrixError::InvalidShape("Matrix element count overflow".into()))
}

/// Operations trait for matrices
pub trait MatrixOperations<T: NumericType> {
    /// Element-wise mathematical functions
    fn map_elementwise<F>(&self, f: F) -> MatrixResult<UnifiedMatrix<T>>
    where
        F: Fn(T) -> T + Send + Sync;

    /// Element-wise in-place mathematical functions
    fn map_elementwise_inplace<F>(&mut self, f: F) -> MatrixResult<()>
    where
        F: Fn(&mut T) + Send + Sync;

    /// Apply function to pairs of elements
    fn zip_map<F>(&self, other: &UnifiedMatrix<T>, f: F) -> MatrixResult<UnifiedMatrix<T>>
    where
        F: Fn(&T, &T) -> T + Send + Sync;

    /// Reshape matrix (if possible)
    fn reshape(&self, new_rows: usize, new_cols: usize) -> MatrixResult<UnifiedMatrix<T>>;

    /// Get a submatrix view
    fn slice(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> MatrixResult<UnifiedMatrix<T>>;
}

impl<T: NumericType> MatrixOperations<T> for UnifiedMatrix<T> {
    fn map_elementwise<F>(&self, f: F) -> MatrixResult<UnifiedMatrix<T>>
    where
        F: Fn(T) -> T + Send + Sync,
    {
        self.try_map_elementwise(|value| Ok(f(value)))
    }

    fn map_elementwise_inplace<F>(&mut self, f: F) -> MatrixResult<()>
    where
        F: Fn(&mut T) + Send + Sync,
    {
        let output = self.map_elementwise(|mut value| {
            f(&mut value);
            value
        })?;
        *self = output;
        Ok(())
    }

    fn zip_map<F>(&self, other: &UnifiedMatrix<T>, f: F) -> MatrixResult<UnifiedMatrix<T>>
    where
        F: Fn(&T, &T) -> T + Send + Sync,
    {
        if self.shape() != other.shape() {
            return Err(MatrixError::DimensionMismatch {
                expected: self.shape().as_tuple(),
                got: other.shape().as_tuple(),
            });
        }

        let self_vec = self.to_vec()?;
        let other_vec = other.to_vec()?;
        let result_vec: Vec<T> = self_vec
            .iter()
            .zip(other_vec.iter())
            .map(|(a, b)| f(a, b))
            .collect();

        UnifiedMatrix::from_data(result_vec, self.rows(), self.cols())
    }

    fn reshape(&self, new_rows: usize, new_cols: usize) -> MatrixResult<UnifiedMatrix<T>> {
        if checked_count(new_rows, new_cols)? != checked_count(self.rows(), self.cols())? {
            return Err(MatrixError::InvalidShape(format!(
                "Cannot reshape {}x{} matrix to {}x{}",
                self.rows(),
                self.cols(),
                new_rows,
                new_cols
            )));
        }

        if let Some(output) = self.sparse_reshape(new_rows, new_cols)? {
            return Ok(output);
        }
        let data = self.to_vec()?;
        UnifiedMatrix::from_data(data, new_rows, new_cols)
    }

    fn slice(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        if row_end > self.rows()
            || col_end > self.cols()
            || row_start > row_end
            || col_start > col_end
        {
            return Err(MatrixError::InvalidShape(format!(
                "Invalid slice indices: rows {}..{}, cols {}..{} for shape {:?}",
                row_start,
                row_end,
                col_start,
                col_end,
                self.shape().as_tuple()
            )));
        }

        let slice_rows = row_end - row_start;
        let slice_cols = col_end - col_start;
        if let Some(output) = self.sparse_select(
            &(row_start..row_end).collect::<Vec<_>>(),
            &(col_start..col_end).collect::<Vec<_>>(),
        )? {
            return Ok(output);
        }
        // Validate all logical strides before selecting a subset.
        let source = self.to_vec()?;
        let mut result_data = reserve_data::<T>(slice_rows, slice_cols)?;

        for i in row_start..row_end {
            for j in col_start..col_end {
                result_data.push(source[i * self.cols() + j]);
            }
        }

        UnifiedMatrix::from_data(result_data, slice_rows, slice_cols)
    }
}

/// Activation functions for neural networks
pub struct ActivationFunctions;

impl ActivationFunctions {
    /// ReLU activation: max(0, x)
    pub fn relu<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        matrix.map_elementwise(|x| if x < T::zero() { T::zero() } else { x })
    }

    /// ReLU activation in-place
    pub fn relu_inplace<T: NumericType>(matrix: &mut UnifiedMatrix<T>) -> MatrixResult<()> {
        matrix.map_elementwise_inplace(|x| {
            if *x < T::zero() {
                *x = T::zero();
            }
        })
    }

    /// Sigmoid activation: 1 / (1 + exp(-x))
    pub fn sigmoid<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        matrix.try_map_elementwise(|x| {
            let exp_neg_x = (-x.to_f64()).exp();
            checked_function_value(1.0 / (1.0 + exp_neg_x))
        })
    }

    /// Tanh activation
    pub fn tanh<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        matrix.try_map_elementwise(|x| {
            let x_f64 = x.to_f64();
            checked_function_value(x_f64.tanh())
        })
    }
}

/// Mathematical functions for matrices
pub struct MathFunctions;

impl MathFunctions {
    /// Elementwise exponential; floating outputs follow IEEE nonfinite behavior.
    pub fn exp<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        matrix.try_map_elementwise(|x| checked_function_value(x.to_f64().exp()))
    }
    /// Natural log; zero/negative inputs give -inf/NaN for floats, errors for integers.
    pub fn ln<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        matrix.try_map_elementwise(|x| checked_function_value(x.to_f64().ln()))
    }
    /// Square root; negative inputs give NaN for floats, errors for integers.
    pub fn sqrt<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        matrix.try_map_elementwise(|x| checked_function_value(x.to_f64().sqrt()))
    }
    /// Elementwise real power, retaining IEEE floating domain/range results.
    pub fn pow<T: NumericType>(
        matrix: &UnifiedMatrix<T>,
        exponent: f64,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        matrix.try_map_elementwise(|x| checked_function_value(x.to_f64().powf(exponent)))
    }
    /// Exact integer absolute values; MIN errors instead of panicking/wrapping.
    pub fn abs<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        matrix.try_map_elementwise(|x| {
            if T::DTYPE.is_integer() && x < T::zero() {
                return T::zero().checked_sub(x).ok_or_else(|| {
                    MatrixError::InvalidInput("Integer absolute value overflow".into())
                });
            }
            Ok(x.abs())
        })
    }
}

fn checked_function_value<T: NumericType>(value: f64) -> MatrixResult<T> {
    T::from_f64(value).ok_or_else(|| {
        MatrixError::InvalidInput("Mathematical result cannot be represented in dtype".into())
    })
}
fn reserve_data<T: NumericType>(rows: usize, cols: usize) -> MatrixResult<Vec<T>> {
    let count = checked_count(rows, cols)?;
    count
        .checked_mul(std::mem::size_of::<T>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| MatrixError::InvalidShape("Matrix byte count overflow".into()))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| MatrixError::InvalidShape("Matrix allocation failed".into()))?;
    Ok(values)
}

/// Utility functions for creating special matrices
pub struct MatrixUtils;

impl MatrixUtils {
    /// Create a random matrix with values between 0 and 1
    pub fn random<T: NumericType>(rows: usize, cols: usize) -> MatrixResult<UnifiedMatrix<T>> {
        // Simple pseudo-random implementation
        let mut data = reserve_data::<T>(rows, cols)?;
        let mut seed = 12345u64;

        for _ in 0..checked_count(rows, cols)? {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let random_val = (seed / 65536) % 32768;
            let normalized = random_val as f64 / 32767.0;
            data.push(checked_function_value(normalized)?);
        }

        UnifiedMatrix::from_data(data, rows, cols)
    }

    /// Create a matrix filled with a specific value
    pub fn constant<T: NumericType>(
        rows: usize,
        cols: usize,
        value: T,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        let mut data = reserve_data::<T>(rows, cols)?;
        data.resize(checked_count(rows, cols)?, value);
        UnifiedMatrix::from_data(data, rows, cols)
    }

    /// Create a diagonal matrix from a vector
    pub fn diag<T: NumericType>(diagonal: Vec<T>) -> MatrixResult<UnifiedMatrix<T>> {
        let size = diagonal.len();
        let mut matrix = UnifiedMatrix::zeros(size, size)?;

        for (i, &value) in diagonal.iter().enumerate() {
            matrix.set(i, i, value)?;
        }

        Ok(matrix)
    }

    /// Extract the diagonal from a matrix
    pub fn extract_diag<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<Vec<T>> {
        let min_dim = matrix.rows().min(matrix.cols());
        let mut diagonal = Vec::with_capacity(min_dim);

        for i in 0..min_dim {
            diagonal.push(matrix.get(i, i)?);
        }

        Ok(diagonal)
    }

    /// Create a matrix with ones on the diagonal and zeros elsewhere
    pub fn eye<T: NumericType>(size: usize) -> MatrixResult<UnifiedMatrix<T>> {
        UnifiedMatrix::identity(size)
    }

    /// Create a matrix from function of indices
    pub fn from_fn<T: NumericType, F>(
        rows: usize,
        cols: usize,
        f: F,
    ) -> MatrixResult<UnifiedMatrix<T>>
    where
        F: Fn(usize, usize) -> T,
    {
        let mut data = reserve_data::<T>(rows, cols)?;

        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }

        UnifiedMatrix::from_data(data, rows, cols)
    }
}

/// Matrix statistics and analysis
pub struct MatrixStats;

impl MatrixStats {
    /// Calculate the mean of all elements
    pub fn mean<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<f64> {
        let count = checked_count(matrix.rows(), matrix.cols())?;
        let values = matrix.reduction_values()?;
        Ok(
            reductions::zero_extended_mean(values.iter().copied().map(NumericValue), count)
                .unwrap_or(f64::NAN),
        )
    }

    /// Population standard deviation, preserving integer offsets until scaling.
    pub fn std<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<f64> {
        let count = checked_count(matrix.rows(), matrix.cols())?;
        let values = matrix.reduction_values()?;
        Ok(
            reductions::zero_extended_std(values.iter().copied().map(NumericValue), count, false)
                .unwrap_or(f64::NAN),
        )
    }

    /// Calculate row-wise means
    pub fn row_means<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<Vec<f64>> {
        matrix.reduce_axis(false, |values, count| {
            Ok(
                reductions::zero_extended_mean(values.iter().copied().map(NumericValue), count)
                    .unwrap_or(0.0),
            )
        })
    }

    /// Calculate column-wise means
    pub fn col_means<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<Vec<f64>> {
        matrix.reduce_axis(true, |values, count| {
            Ok(
                reductions::zero_extended_mean(values.iter().copied().map(NumericValue), count)
                    .unwrap_or(0.0),
            )
        })
    }

    /// Calculate the Frobenius norm
    pub fn frobenius_norm<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<f64> {
        Ok(numeric_arithmetic::norm(&matrix.reduction_values()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::matrix::unified_matrix::Matrix64;

    #[test]
    fn test_relu_activation() {
        let matrix = Matrix64::from_data(vec![-1.0, 0.0, 1.0, 2.0], 2, 2).unwrap();
        let result = ActivationFunctions::relu(&matrix).unwrap();

        assert_eq!(result.get(0, 0).unwrap(), 0.0);
        assert_eq!(result.get(0, 1).unwrap(), 0.0);
        assert_eq!(result.get(1, 0).unwrap(), 1.0);
        assert_eq!(result.get(1, 1).unwrap(), 2.0);
    }

    #[test]
    fn test_matrix_reshape() {
        let matrix = Matrix64::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let reshaped = matrix.reshape(3, 2).unwrap();

        assert_eq!(reshaped.shape().as_tuple(), (3, 2));
        assert_eq!(reshaped.get(0, 0).unwrap(), 1.0);
        assert_eq!(reshaped.get(2, 1).unwrap(), 6.0);
    }

    #[test]
    fn test_matrix_slice() {
        let matrix =
            Matrix64::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3).unwrap();

        let slice = matrix.slice(1, 3, 1, 3).unwrap();

        assert_eq!(slice.shape().as_tuple(), (2, 2));
        assert_eq!(slice.get(0, 0).unwrap(), 5.0); // matrix[1,1]
        assert_eq!(slice.get(0, 1).unwrap(), 6.0); // matrix[1,2]
        assert_eq!(slice.get(1, 0).unwrap(), 8.0); // matrix[2,1]
        assert_eq!(slice.get(1, 1).unwrap(), 9.0); // matrix[2,2]
    }

    #[test]
    fn test_matrix_stats() {
        let matrix = Matrix64::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();

        let mean = MatrixStats::mean(&matrix).unwrap();
        assert!((mean - 2.5).abs() < 1e-10);

        let frobenius = MatrixStats::frobenius_norm(&matrix).unwrap();
        assert!((frobenius - (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_utils() {
        let diagonal = vec![1.0, 2.0, 3.0];
        let diag_matrix = MatrixUtils::diag(diagonal).unwrap();

        assert_eq!(diag_matrix.get(0, 0).unwrap(), 1.0);
        assert_eq!(diag_matrix.get(1, 1).unwrap(), 2.0);
        assert_eq!(diag_matrix.get(2, 2).unwrap(), 3.0);
        assert_eq!(diag_matrix.get(0, 1).unwrap(), 0.0);

        let extracted = MatrixUtils::extract_diag(&diag_matrix).unwrap();
        assert_eq!(extracted, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_element_wise_functions() {
        let matrix = Matrix64::from_data(vec![1.0, 4.0, 9.0, 16.0], 2, 2).unwrap();
        let sqrt_matrix = MathFunctions::sqrt(&matrix).unwrap();

        assert!((sqrt_matrix.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((sqrt_matrix.get(0, 1).unwrap() - 2.0).abs() < 1e-10);
        assert!((sqrt_matrix.get(1, 0).unwrap() - 3.0).abs() < 1e-10);
        assert!((sqrt_matrix.get(1, 1).unwrap() - 4.0).abs() < 1e-10);
    }
}
