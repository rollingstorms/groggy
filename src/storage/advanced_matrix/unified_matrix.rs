//! UnifiedMatrix - Core matrix type with smart storage and multi-backend support
//!
//! This module implements the unified matrix type that serves as the foundation
//! for all matrix operations in the advanced matrix system.

use crate::storage::advanced_matrix::{
    backend::{BackendHint, BackendSelector, ComputeBackendExt, OperationType},
    memory::{MatrixLayout, MemoryError, SharedBuffer},
    numeric_type::NumericType,
};
use std::marker::PhantomData;
use std::sync::Arc;

/// Errors specific to matrix operations
#[derive(Debug, Clone)]
pub enum MatrixError {
    DimensionMismatch {
        expected: (usize, usize),
        got: (usize, usize),
    },
    InvalidShape(String),
    InvalidInput(String),
    MemoryError(MemoryError),
    ComputationError(String),
    UnsupportedOperation(String),
    InvalidIndex {
        row: usize,
        col: usize,
        shape: (usize, usize),
    },
}

impl std::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatrixError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Dimension mismatch: expected {:?}, got {:?}",
                    expected, got
                )
            }
            MatrixError::InvalidShape(msg) => write!(f, "Invalid shape: {}", msg),
            MatrixError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            MatrixError::MemoryError(e) => write!(f, "Memory error: {}", e),
            MatrixError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            MatrixError::UnsupportedOperation(op) => write!(f, "Unsupported operation: {}", op),
            MatrixError::InvalidIndex { row, col, shape } => {
                write!(f, "Invalid index ({}, {}) for shape {:?}", row, col, shape)
            }
        }
    }
}

impl std::error::Error for MatrixError {}

impl From<MemoryError> for MatrixError {
    fn from(e: MemoryError) -> Self {
        MatrixError::MemoryError(e)
    }
}

pub type MatrixResult<T> = Result<T, MatrixError>;

/// Strides for matrix indexing - defines how to navigate through memory
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strides {
    pub row_stride: usize,
    pub col_stride: usize,
}

impl Strides {
    /// Create strides for row-major layout
    pub fn row_major(shape: (usize, usize)) -> Self {
        let (_, cols) = shape;
        Self {
            row_stride: cols,
            col_stride: 1,
        }
    }

    /// Create strides for column-major layout  
    pub fn column_major(shape: (usize, usize)) -> Self {
        let (rows, _) = shape;
        Self {
            row_stride: 1,
            col_stride: rows,
        }
    }

    /// Calculate the linear index for given row and column
    pub fn index(&self, row: usize, col: usize) -> usize {
        row * self.row_stride + col * self.col_stride
    }
}

/// Shape information for matrices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}

impl Shape {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    pub fn as_tuple(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn total_elements(&self) -> usize {
        self.rows * self.cols
    }

    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    pub fn is_vector(&self) -> bool {
        self.rows == 1 || self.cols == 1
    }

    pub fn transpose(&self) -> Self {
        Self {
            rows: self.cols,
            cols: self.rows,
        }
    }
}

/// Different storage strategies for matrices
#[derive(Debug)]
pub enum MatrixStorage<T: NumericType> {
    /// Dense storage - all elements stored contiguously
    Dense(SharedBuffer<T>),
    /// Sparse storage - only non-zero elements stored
    Sparse(SparseMatrix<T>),
    /// View of another matrix - no owned data
    View(MatrixView<T>),
    /// Lazy computation - computed on demand
    Lazy(LazyMatrix<T>),
}

/// Sparse matrix representation (simplified CSR format)
#[derive(Debug)]
pub struct SparseMatrix<T: NumericType> {
    pub values: Vec<T>,
    pub col_indices: Vec<usize>,
    pub row_pointers: Vec<usize>,
    pub shape: Shape,
}

/// View into another matrix
pub struct MatrixView<T: NumericType> {
    pub data: *const T,
    pub shape: Shape,
    pub strides: Strides,
    pub _phantom: PhantomData<T>,
}

impl<T: NumericType> std::fmt::Debug for MatrixView<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatrixView")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("data", &"<raw pointer>")
            .finish()
    }
}

/// Lazy computation matrix (placeholder)
pub struct LazyMatrix<T: NumericType> {
    pub computation: Box<dyn Fn() -> MatrixResult<UnifiedMatrix<T>> + Send + Sync>,
    pub shape: Shape,
}

impl<T: NumericType> std::fmt::Debug for LazyMatrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyMatrix")
            .field("shape", &self.shape)
            .field("computation", &"<function pointer>")
            .finish()
    }
}

/// The unified matrix type - core of the advanced matrix system
#[derive(Debug)]
pub struct UnifiedMatrix<T: NumericType> {
    pub storage: MatrixStorage<T>,
    pub shape: Shape,
    pub strides: Strides,
    pub layout: MatrixLayout,
    pub backend_hint: BackendHint,
    pub backend_selector: Arc<BackendSelector>,
}

impl<T: NumericType> UnifiedMatrix<T> {
    /// Create a new dense matrix with the given shape
    pub fn new(rows: usize, cols: usize) -> MatrixResult<Self> {
        Self::new_with_layout(rows, cols, MatrixLayout::RowMajor)
    }

    /// Create a new dense matrix with specified layout
    pub fn new_with_layout(rows: usize, cols: usize, layout: MatrixLayout) -> MatrixResult<Self> {
        let shape = Shape::new(rows, cols);
        let buffer = SharedBuffer::new((rows, cols), layout)?;
        let strides = match layout {
            MatrixLayout::RowMajor => Strides::row_major((rows, cols)),
            MatrixLayout::ColumnMajor => Strides::column_major((rows, cols)),
            MatrixLayout::Blocked => Strides::row_major((rows, cols)), // Simplified
        };

        Ok(Self {
            storage: MatrixStorage::Dense(buffer),
            shape,
            strides,
            layout,
            backend_hint: BackendHint::AutoSelect,
            backend_selector: Arc::new(BackendSelector::new()),
        })
    }

    /// Create a matrix from existing data
    pub fn from_data(data: Vec<T>, rows: usize, cols: usize) -> MatrixResult<Self> {
        if data.len() != rows * cols {
            return Err(MatrixError::DimensionMismatch {
                expected: (rows, cols),
                got: (data.len(), 1),
            });
        }

        let shape = Shape::new(rows, cols);
        let layout = MatrixLayout::RowMajor;
        let buffer = SharedBuffer::from_data(data, (rows, cols), layout)?;
        let strides = Strides::row_major((rows, cols));

        Ok(Self {
            storage: MatrixStorage::Dense(buffer),
            shape,
            strides,
            layout,
            backend_hint: BackendHint::AutoSelect,
            backend_selector: Arc::new(BackendSelector::new()),
        })
    }

    /// Create an identity matrix
    pub fn identity(size: usize) -> MatrixResult<Self> {
        let matrix = Self::new(size, size)?;

        // Set diagonal elements to 1
        if let MatrixStorage::Dense(ref buffer) = matrix.storage {
            let mut data = buffer.data_mut()?;
            let slice = data.as_slice_mut();

            for i in 0..size {
                let idx = matrix.strides.index(i, i);
                slice[idx] = T::one();
            }
        }

        Ok(matrix)
    }

    /// Create a zeros matrix
    pub fn zeros(rows: usize, cols: usize) -> MatrixResult<Self> {
        // Buffer is initialized to zero by default
        Self::new(rows, cols)
    }

    /// Create a ones matrix
    pub fn ones(rows: usize, cols: usize) -> MatrixResult<Self> {
        let mut matrix = Self::new(rows, cols)?;
        matrix.fill(T::one())?;
        Ok(matrix)
    }

    /// Get the shape of the matrix
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// Get the number of rows
    pub fn rows(&self) -> usize {
        self.shape.rows
    }

    /// Get the number of columns
    pub fn cols(&self) -> usize {
        self.shape.cols
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.shape.total_elements()
    }

    /// Check if the matrix is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the memory layout
    pub fn layout(&self) -> MatrixLayout {
        self.layout
    }

    /// Get the strides
    pub fn strides(&self) -> Strides {
        self.strides
    }

    /// Set backend hint for future operations
    pub fn with_backend_hint(mut self, hint: BackendHint) -> Self {
        self.backend_hint = hint;
        self
    }

    /// Get element at specific position
    pub fn get(&self, row: usize, col: usize) -> MatrixResult<T> {
        if row >= self.shape.rows || col >= self.shape.cols {
            return Err(MatrixError::InvalidIndex {
                row,
                col,
                shape: self.shape.as_tuple(),
            });
        }

        match &self.storage {
            MatrixStorage::Dense(buffer) => {
                let data = buffer.data()?;
                let idx = self.strides.index(row, col);
                Ok(data.as_slice()[idx])
            }
            MatrixStorage::Sparse(_) => Err(MatrixError::UnsupportedOperation(
                "Sparse matrix indexing not implemented".to_string(),
            )),
            MatrixStorage::View(_) => Err(MatrixError::UnsupportedOperation(
                "Matrix view indexing not implemented".to_string(),
            )),
            MatrixStorage::Lazy(_) => Err(MatrixError::UnsupportedOperation(
                "Lazy matrix indexing not implemented".to_string(),
            )),
        }
    }

    /// Set element at specific position
    pub fn set(&mut self, row: usize, col: usize, value: T) -> MatrixResult<()> {
        if row >= self.shape.rows || col >= self.shape.cols {
            return Err(MatrixError::InvalidIndex {
                row,
                col,
                shape: self.shape.as_tuple(),
            });
        }

        match &mut self.storage {
            MatrixStorage::Dense(buffer) => {
                let mut data = buffer.data_mut()?;
                let idx = self.strides.index(row, col);
                data.as_slice_mut()[idx] = value;
                Ok(())
            }
            _ => Err(MatrixError::UnsupportedOperation(
                "Setting elements in non-dense matrices not implemented".to_string(),
            )),
        }
    }

    /// Fill the entire matrix with a value
    pub fn fill(&mut self, value: T) -> MatrixResult<()> {
        match &self.storage {
            MatrixStorage::Dense(buffer) => {
                let mut data = buffer.data_mut()?;
                let slice = data.as_slice_mut();
                slice.fill(value);
                Ok(())
            }
            _ => Err(MatrixError::UnsupportedOperation(
                "Filling non-dense matrices not implemented".to_string(),
            )),
        }
    }

    /// Matrix multiplication (GEMM)
    pub fn matmul(&self, other: &Self) -> MatrixResult<Self> {
        // Check dimensions
        if self.shape.cols != other.shape.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.shape.rows, other.shape.cols),
                got: (self.shape.rows, self.shape.cols),
            });
        }

        // Create result matrix
        let result = Self::new(self.shape.rows, other.shape.cols)?;

        // Select backend for computation
        let backend = self.backend_selector.select_backend(
            OperationType::GEMM,
            self.len() + other.len(),
            T::DTYPE,
            self.backend_hint.clone(),
        );

        // Perform computation based on storage types
        match (&self.storage, &other.storage, &result.storage) {
            (
                MatrixStorage::Dense(a_buf),
                MatrixStorage::Dense(b_buf),
                MatrixStorage::Dense(c_buf),
            ) => {
                let a_data = a_buf.data()?;
                let b_data = b_buf.data()?;
                let mut c_data = c_buf.data_mut()?;

                backend
                    .gemm(
                        a_data.as_slice(),
                        self.shape.as_tuple(),
                        b_data.as_slice(),
                        other.shape.as_tuple(),
                        c_data.as_slice_mut(),
                        result.shape.as_tuple(),
                        T::one(),  // alpha
                        T::zero(), // beta
                    )
                    .map_err(|e| MatrixError::ComputationError(e.to_string()))?;
            }
            _ => {
                return Err(MatrixError::UnsupportedOperation(
                    "Matrix multiplication for non-dense matrices not implemented".to_string(),
                ));
            }
        }

        Ok(result)
    }

    /// Element-wise addition
    pub fn add(&self, other: &Self) -> MatrixResult<Self> {
        if self.shape != other.shape {
            return Err(MatrixError::DimensionMismatch {
                expected: self.shape.as_tuple(),
                got: other.shape.as_tuple(),
            });
        }

        let mut result = Self::new(self.shape.rows, self.shape.cols)?;

        let backend = self.backend_selector.select_backend(
            OperationType::ElementwiseAdd,
            self.len(),
            T::DTYPE,
            self.backend_hint.clone(),
        );

        match (&self.storage, &other.storage, &mut result.storage) {
            (
                MatrixStorage::Dense(a_buf),
                MatrixStorage::Dense(b_buf),
                MatrixStorage::Dense(c_buf),
            ) => {
                let a_data = a_buf.data()?;
                let b_data = b_buf.data()?;
                let mut c_data = c_buf.data_mut()?;

                backend
                    .elementwise_add(a_data.as_slice(), b_data.as_slice(), c_data.as_slice_mut())
                    .map_err(|e| MatrixError::ComputationError(e.to_string()))?;
            }
            _ => {
                return Err(MatrixError::UnsupportedOperation(
                    "Element-wise addition for non-dense matrices not implemented".to_string(),
                ));
            }
        }

        Ok(result)
    }

    /// Element-wise subtraction
    pub fn subtract(&self, other: &Self) -> MatrixResult<Self> {
        if self.shape != other.shape {
            return Err(MatrixError::DimensionMismatch {
                expected: self.shape.as_tuple(),
                got: other.shape.as_tuple(),
            });
        }

        let mut result = Self::new(self.shape.rows, self.shape.cols)?;

        let _backend = self.backend_selector.select_backend(
            OperationType::ElementwiseSub,
            self.len(),
            T::DTYPE,
            self.backend_hint.clone(),
        );

        match (&self.storage, &other.storage, &mut result.storage) {
            (
                MatrixStorage::Dense(a_buf),
                MatrixStorage::Dense(b_buf),
                MatrixStorage::Dense(c_buf),
            ) => {
                let a_data = a_buf.data()?;
                let b_data = b_buf.data()?;
                let mut c_data = c_buf.data_mut()?;

                // If backend doesn't have subtraction, do it manually
                let a_slice = a_data.as_slice();
                let b_slice = b_data.as_slice();
                let c_slice = c_data.as_slice_mut();

                for ((a_val, b_val), c_val) in
                    a_slice.iter().zip(b_slice.iter()).zip(c_slice.iter_mut())
                {
                    *c_val = a_val.sub(*b_val);
                }
            }
            _ => {
                return Err(MatrixError::UnsupportedOperation(
                    "Element-wise subtraction for non-dense matrices not implemented".to_string(),
                ));
            }
        }

        Ok(result)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Self) -> MatrixResult<Self> {
        if self.shape != other.shape {
            return Err(MatrixError::DimensionMismatch {
                expected: self.shape.as_tuple(),
                got: other.shape.as_tuple(),
            });
        }

        let mut result = Self::new(self.shape.rows, self.shape.cols)?;

        let backend = self.backend_selector.select_backend(
            OperationType::ElementwiseMul,
            self.len(),
            T::DTYPE,
            self.backend_hint.clone(),
        );

        match (&self.storage, &other.storage, &mut result.storage) {
            (
                MatrixStorage::Dense(a_buf),
                MatrixStorage::Dense(b_buf),
                MatrixStorage::Dense(c_buf),
            ) => {
                let a_data = a_buf.data()?;
                let b_data = b_buf.data()?;
                let mut c_data = c_buf.data_mut()?;

                backend
                    .elementwise_mul(a_data.as_slice(), b_data.as_slice(), c_data.as_slice_mut())
                    .map_err(|e| MatrixError::ComputationError(e.to_string()))?;
            }
            _ => {
                return Err(MatrixError::UnsupportedOperation(
                    "Element-wise multiplication for non-dense matrices not implemented"
                        .to_string(),
                ));
            }
        }

        Ok(result)
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: T) -> MatrixResult<Self> {
        let mut result = Self::new(self.shape.rows, self.shape.cols)?;

        match (&self.storage, &mut result.storage) {
            (MatrixStorage::Dense(a_buf), MatrixStorage::Dense(c_buf)) => {
                let a_data = a_buf.data()?;
                let mut c_data = c_buf.data_mut()?;

                let a_slice = a_data.as_slice();
                let c_slice = c_data.as_slice_mut();

                for (a_val, c_val) in a_slice.iter().zip(c_slice.iter_mut()) {
                    *c_val = a_val.mul(scalar);
                }
            }
            _ => {
                return Err(MatrixError::UnsupportedOperation(
                    "Scalar multiplication for non-dense matrices not implemented".to_string(),
                ));
            }
        }

        Ok(result)
    }

    /// Alias for scale - scalar multiplication (for autodiff compatibility)
    pub fn scalar_multiply(&self, scalar: T) -> MatrixResult<Self> {
        self.scale(scalar)
    }

    /// Sum all elements
    pub fn sum(&self) -> MatrixResult<T::Accumulator> {
        let backend = self.backend_selector.select_backend(
            OperationType::Sum,
            self.len(),
            T::DTYPE,
            self.backend_hint.clone(),
        );

        match &self.storage {
            MatrixStorage::Dense(buffer) => {
                let data = buffer.data()?;
                backend
                    .reduce_sum(data.as_slice())
                    .map_err(|e| MatrixError::ComputationError(e.to_string()))
            }
            _ => Err(MatrixError::UnsupportedOperation(
                "Sum reduction for non-dense matrices not implemented".to_string(),
            )),
        }
    }

    /// Find maximum element
    pub fn max(&self) -> MatrixResult<T> {
        let backend = self.backend_selector.select_backend(
            OperationType::Max,
            self.len(),
            T::DTYPE,
            self.backend_hint.clone(),
        );

        match &self.storage {
            MatrixStorage::Dense(buffer) => {
                let data = buffer.data()?;
                backend
                    .reduce_max(data.as_slice())
                    .map_err(|e| MatrixError::ComputationError(e.to_string()))
            }
            _ => Err(MatrixError::UnsupportedOperation(
                "Max reduction for non-dense matrices not implemented".to_string(),
            )),
        }
    }

    /// Find minimum element
    pub fn min(&self) -> MatrixResult<T> {
        let backend = self.backend_selector.select_backend(
            OperationType::Min,
            self.len(),
            T::DTYPE,
            self.backend_hint.clone(),
        );

        match &self.storage {
            MatrixStorage::Dense(buffer) => {
                let data = buffer.data()?;
                backend
                    .reduce_min(data.as_slice())
                    .map_err(|e| MatrixError::ComputationError(e.to_string()))
            }
            _ => Err(MatrixError::UnsupportedOperation(
                "Min reduction for non-dense matrices not implemented".to_string(),
            )),
        }
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> MatrixResult<Self> {
        let mut result = Self::new(self.shape.cols, self.shape.rows)?;

        match (&self.storage, &mut result.storage) {
            (MatrixStorage::Dense(a_buf), MatrixStorage::Dense(c_buf)) => {
                let a_data = a_buf.data()?;
                let mut c_data = c_buf.data_mut()?;

                let a_slice = a_data.as_slice();
                let c_slice = c_data.as_slice_mut();

                for i in 0..self.shape.rows {
                    for j in 0..self.shape.cols {
                        let src_idx = self.strides.index(i, j);
                        let dst_idx = result.strides.index(j, i);
                        c_slice[dst_idx] = a_slice[src_idx];
                    }
                }
            }
            _ => {
                return Err(MatrixError::UnsupportedOperation(
                    "Transpose for non-dense matrices not implemented".to_string(),
                ));
            }
        }

        Ok(result)
    }

    /// Convert to a vector (flatten)
    pub fn to_vec(&self) -> MatrixResult<Vec<T>> {
        match &self.storage {
            MatrixStorage::Dense(buffer) => buffer.to_vec().map_err(MatrixError::from),
            _ => Err(MatrixError::UnsupportedOperation(
                "to_vec for non-dense matrices not implemented".to_string(),
            )),
        }
    }

    /// Element-wise multiplication with another matrix
    pub fn elementwise_multiply(&self, other: &Self) -> MatrixResult<Self> {
        if self.shape != other.shape {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.shape.rows, self.shape.cols),
                got: (other.shape.rows, other.shape.cols),
            });
        }

        let mut result = Self::new(self.shape.rows, self.shape.cols)?;

        // Element-wise multiplication: C[i,j] = A[i,j] * B[i,j]
        for i in 0..self.shape.rows {
            for j in 0..self.shape.cols {
                let a_val = self.get(i, j)?;
                let b_val = other.get(i, j)?;
                result.set(i, j, a_val.mul(b_val))?;
            }
        }

        Ok(result)
    }

    // === PLACEHOLDER FOR ENHANCED OPERATIONS ===
    // These will be implemented in incremental phases
}

// Implement Clone for UnifiedMatrix
impl<T: NumericType> Clone for UnifiedMatrix<T> {
    fn clone(&self) -> Self {
        match &self.storage {
            MatrixStorage::Dense(buffer) => {
                let cloned_buffer = buffer.clone();
                Self {
                    storage: MatrixStorage::Dense(cloned_buffer),
                    shape: self.shape,
                    strides: self.strides,
                    layout: self.layout,
                    backend_hint: self.backend_hint.clone(),
                    backend_selector: Arc::clone(&self.backend_selector),
                }
            }
            _ => panic!("Cloning non-dense matrices not implemented"),
        }
    }
}

// Type aliases for common matrix types
pub type Matrix64 = UnifiedMatrix<f64>;
pub type Matrix32 = UnifiedMatrix<f32>;
pub type MatrixI64 = UnifiedMatrix<i64>;
pub type MatrixI32 = UnifiedMatrix<i32>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let matrix = Matrix64::new(3, 4).unwrap();
        assert_eq!(matrix.shape().as_tuple(), (3, 4));
        assert_eq!(matrix.len(), 12);
        assert!(!matrix.is_empty());
    }

    #[test]
    fn test_matrix_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = Matrix64::from_data(data, 2, 2).unwrap();

        assert_eq!(matrix.get(0, 0).unwrap(), 1.0);
        assert_eq!(matrix.get(0, 1).unwrap(), 2.0);
        assert_eq!(matrix.get(1, 0).unwrap(), 3.0);
        assert_eq!(matrix.get(1, 1).unwrap(), 4.0);
    }

    #[test]
    fn test_identity_matrix() {
        let identity = Matrix64::identity(3).unwrap();

        // Check diagonal elements
        for i in 0..3 {
            assert_eq!(identity.get(i, i).unwrap(), 1.0);
        }

        // Check off-diagonal elements
        assert_eq!(identity.get(0, 1).unwrap(), 0.0);
        assert_eq!(identity.get(1, 0).unwrap(), 0.0);
    }

    #[test]
    fn test_matrix_addition() {
        let a = Matrix64::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = Matrix64::from_data(vec![5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();

        let c = a.add(&b).unwrap();

        assert_eq!(c.get(0, 0).unwrap(), 6.0);
        assert_eq!(c.get(0, 1).unwrap(), 8.0);
        assert_eq!(c.get(1, 0).unwrap(), 10.0);
        assert_eq!(c.get(1, 1).unwrap(), 12.0);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Matrix64::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = Matrix64::from_data(vec![5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();

        let c = a.matmul(&b).unwrap();

        // Expected: [[19, 22], [43, 50]]
        assert_eq!(c.get(0, 0).unwrap(), 19.0);
        assert_eq!(c.get(0, 1).unwrap(), 22.0);
        assert_eq!(c.get(1, 0).unwrap(), 43.0);
        assert_eq!(c.get(1, 1).unwrap(), 50.0);
    }

    #[test]
    fn test_scalar_multiplication() {
        let a = Matrix64::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = a.scale(2.0).unwrap();

        assert_eq!(b.get(0, 0).unwrap(), 2.0);
        assert_eq!(b.get(0, 1).unwrap(), 4.0);
        assert_eq!(b.get(1, 0).unwrap(), 6.0);
        assert_eq!(b.get(1, 1).unwrap(), 8.0);
    }

    #[test]
    fn test_matrix_reductions() {
        let matrix = Matrix64::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();

        assert_eq!(matrix.sum().unwrap(), 10.0);
        assert_eq!(matrix.max().unwrap(), 4.0);
        assert_eq!(matrix.min().unwrap(), 1.0);
    }

    #[test]
    fn test_matrix_transpose() {
        let a = Matrix64::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let b = a.transpose().unwrap();

        assert_eq!(b.shape().as_tuple(), (3, 2));
        assert_eq!(b.get(0, 0).unwrap(), 1.0);
        assert_eq!(b.get(0, 1).unwrap(), 4.0);
        assert_eq!(b.get(1, 0).unwrap(), 2.0);
        assert_eq!(b.get(1, 1).unwrap(), 5.0);
        assert_eq!(b.get(2, 0).unwrap(), 3.0);
        assert_eq!(b.get(2, 1).unwrap(), 6.0);
    }
}
