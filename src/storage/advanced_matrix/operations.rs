//! Core Matrix Operations
//!
//! This module provides additional operations and utilities for the unified matrix system.

use crate::storage::advanced_matrix::{
    numeric_type::NumericType,
    unified_matrix::{MatrixError, MatrixResult, UnifiedMatrix},
};

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
        let input_vec = self.to_vec()?;
        let output_vec: Vec<T> = input_vec.iter().map(|&x| f(x)).collect();

        UnifiedMatrix::from_data(output_vec, self.rows(), self.cols())
    }

    fn map_elementwise_inplace<F>(&mut self, f: F) -> MatrixResult<()>
    where
        F: Fn(&mut T) + Send + Sync,
    {
        match &mut self.storage {
            crate::storage::advanced_matrix::unified_matrix::MatrixStorage::Dense(buffer) => {
                let mut data = buffer.data_mut()?;
                let slice = data.as_slice_mut();
                for element in slice.iter_mut() {
                    f(element);
                }
                Ok(())
            }
            _ => Err(MatrixError::UnsupportedOperation(
                "In-place operations only supported for dense matrices".to_string(),
            )),
        }
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
        if new_rows * new_cols != self.len() {
            return Err(MatrixError::InvalidShape(format!(
                "Cannot reshape {}x{} matrix to {}x{}",
                self.rows(),
                self.cols(),
                new_rows,
                new_cols
            )));
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
            || row_start >= row_end
            || col_start >= col_end
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
        let mut result_data = Vec::with_capacity(slice_rows * slice_cols);

        for i in row_start..row_end {
            for j in col_start..col_end {
                result_data.push(self.get(i, j)?);
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
        matrix.map_elementwise(|x| if x > T::zero() { x } else { T::zero() })
    }

    /// ReLU activation in-place
    pub fn relu_inplace<T: NumericType>(matrix: &mut UnifiedMatrix<T>) -> MatrixResult<()> {
        matrix.map_elementwise_inplace(|x| {
            if *x <= T::zero() {
                *x = T::zero();
            }
        })
    }

    /// Sigmoid activation: 1 / (1 + exp(-x))
    pub fn sigmoid<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        matrix.map_elementwise(|x| {
            let exp_neg_x = (-x.to_f64()).exp();
            T::from_f64(1.0 / (1.0 + exp_neg_x)).unwrap_or(T::zero())
        })
    }

    /// Tanh activation
    pub fn tanh<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        matrix.map_elementwise(|x| {
            let x_f64 = x.to_f64();
            T::from_f64(x_f64.tanh()).unwrap_or(T::zero())
        })
    }
}

/// Mathematical functions for matrices
pub struct MathFunctions;

impl MathFunctions {
    /// Element-wise exponential
    pub fn exp<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        matrix.map_elementwise(|x| T::from_f64(x.to_f64().exp()).unwrap_or(T::zero()))
    }

    /// Element-wise natural logarithm
    pub fn ln<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        matrix.map_elementwise(|x| {
            let x_f64 = x.to_f64();
            if x_f64 > 0.0 {
                T::from_f64(x_f64.ln()).unwrap_or(T::zero())
            } else {
                T::zero() // Handle non-positive values
            }
        })
    }

    /// Element-wise square root
    pub fn sqrt<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        matrix.map_elementwise(|x| {
            let x_f64 = x.to_f64();
            if x_f64 >= 0.0 {
                T::from_f64(x_f64.sqrt()).unwrap_or(T::zero())
            } else {
                T::zero() // Handle negative values
            }
        })
    }

    /// Element-wise power
    pub fn pow<T: NumericType>(
        matrix: &UnifiedMatrix<T>,
        exponent: f64,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        matrix.map_elementwise(|x| T::from_f64(x.to_f64().powf(exponent)).unwrap_or(T::zero()))
    }

    /// Element-wise absolute value
    pub fn abs<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        matrix.map_elementwise(|x| x.abs())
    }
}

/// Utility functions for creating special matrices
pub struct MatrixUtils;

impl MatrixUtils {
    /// Create a random matrix with values between 0 and 1
    pub fn random<T: NumericType>(rows: usize, cols: usize) -> MatrixResult<UnifiedMatrix<T>> {
        // Simple pseudo-random implementation
        let mut data = Vec::with_capacity(rows * cols);
        let mut seed = 12345u64;

        for _ in 0..(rows * cols) {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let random_val = (seed / 65536) % 32768;
            let normalized = random_val as f64 / 32767.0;
            data.push(T::from_f64(normalized).unwrap_or(T::zero()));
        }

        UnifiedMatrix::from_data(data, rows, cols)
    }

    /// Create a matrix filled with a specific value
    pub fn constant<T: NumericType>(
        rows: usize,
        cols: usize,
        value: T,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        let data = vec![value; rows * cols];
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
        let mut data = Vec::with_capacity(rows * cols);

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
        let sum = matrix.sum()?;
        let count = matrix.len() as f64;
        Ok(sum.to_f64() / count)
    }

    /// Calculate the standard deviation
    pub fn std<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<f64> {
        let mean_val = Self::mean(matrix)?;
        let variance_matrix = matrix.map_elementwise(|x| {
            let diff = x.to_f64() - mean_val;
            T::from_f64(diff * diff).unwrap_or(T::zero())
        })?;

        let variance = Self::mean(&variance_matrix)?;
        Ok(variance.sqrt())
    }

    /// Calculate row-wise means
    pub fn row_means<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<Vec<f64>> {
        let mut means = Vec::with_capacity(matrix.rows());

        for i in 0..matrix.rows() {
            let row_slice = matrix.slice(i, i + 1, 0, matrix.cols())?;
            means.push(Self::mean(&row_slice)?);
        }

        Ok(means)
    }

    /// Calculate column-wise means
    pub fn col_means<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<Vec<f64>> {
        let mut means = Vec::with_capacity(matrix.cols());

        for j in 0..matrix.cols() {
            let col_slice = matrix.slice(0, matrix.rows(), j, j + 1)?;
            means.push(Self::mean(&col_slice)?);
        }

        Ok(means)
    }

    /// Calculate the Frobenius norm
    pub fn frobenius_norm<T: NumericType>(matrix: &UnifiedMatrix<T>) -> MatrixResult<f64> {
        let squared = matrix.map_elementwise(|x| {
            let x_f64 = x.to_f64();
            T::from_f64(x_f64 * x_f64).unwrap_or(T::zero())
        })?;

        let sum_squares = squared.sum()?;
        Ok(sum_squares.to_f64().sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::advanced_matrix::unified_matrix::Matrix64;

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
