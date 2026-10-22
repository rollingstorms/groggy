//! UnifiedMatrix - Core matrix type with smart storage and multi-backend support
//!
//! This module implements the unified matrix type that serves as the foundation
//! for all matrix operations in the advanced matrix system.

use crate::storage::advanced_matrix::{
    backend::{BackendHint, BackendSelector},
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
#[derive(Debug, Clone)]
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
    /// Construct owned validated canonical CSR storage.
    pub fn from_csr(
        rows: usize,
        cols: usize,
        values: Vec<T>,
        col_indices: Vec<usize>,
        row_pointers: Vec<usize>,
    ) -> MatrixResult<Self> {
        let sparse = SparseMatrix::new(rows, cols, values, col_indices, row_pointers)?;
        Ok(Self {
            storage: MatrixStorage::Sparse(sparse),
            shape: Shape::new(rows, cols),
            strides: Strides::row_major((rows, cols)),
            layout: MatrixLayout::RowMajor,
            backend_hint: BackendHint::AutoSelect,
            backend_selector: Arc::new(BackendSelector::new()),
        })
    }

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

    fn owned_csr(sparse: SparseMatrix<T>) -> MatrixResult<Self> {
        Self::from_csr(
            sparse.shape.rows,
            sparse.shape.cols,
            sparse.values,
            sparse.col_indices,
            sparse.row_pointers,
        )
    }
    pub(crate) fn checked_sparse(&self) -> MatrixResult<Option<&SparseMatrix<T>>> {
        if let MatrixStorage::Sparse(s) = &self.storage {
            if s.shape != self.shape {
                return Err(MatrixError::InvalidInput(
                    "CSR shape metadata mismatch".into(),
                ));
            }
            s.validate()?;
            Ok(Some(s))
        } else {
            Ok(None)
        }
    }
    /// Explicit owned conversion to canonical CSR.
    pub fn to_csr(&self) -> MatrixResult<Self> {
        if let Some(s) = self.checked_sparse()? {
            return Self::owned_csr(s.clone());
        }
        let data = self.to_vec()?;
        let mut values = Vec::new();
        let mut indices = Vec::new();
        let mut pointers = vec![0];
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                let value = data[row * self.cols() + col];
                if value != T::zero() {
                    values.push(value);
                    indices.push(col);
                }
            }
            pointers.push(values.len());
        }
        Self::from_csr(self.rows(), self.cols(), values, indices, pointers)
    }
    /// Explicit dense materialization in logical row order.
    pub fn to_dense(&self) -> MatrixResult<Self> {
        Self::from_data(self.to_vec()?, self.rows(), self.cols())
    }
    /// Owned CSR positional selection; dense selection remains in GraphMatrix.
    pub(crate) fn sparse_select(
        &self,
        rows: &[usize],
        cols: &[usize],
    ) -> MatrixResult<Option<Self>> {
        self.checked_sparse()?
            .map(|s| s.select(rows, cols).and_then(Self::owned_csr))
            .transpose()
    }

    pub(crate) fn sparse_map(&self, operation: impl Fn(T) -> T) -> MatrixResult<Option<Self>> {
        self.checked_sparse()?
            .map(|s| s.map(operation).and_then(Self::owned_csr))
            .transpose()
    }
    pub(crate) fn sparse_try_map(
        &self,
        operation: impl Fn(T) -> MatrixResult<T>,
    ) -> MatrixResult<Option<Self>> {
        self.checked_sparse()?
            .map(|s| s.try_map(operation).and_then(Self::owned_csr))
            .transpose()
    }
    pub(crate) fn sparse_reshape(&self, rows: usize, cols: usize) -> MatrixResult<Option<Self>> {
        self.checked_sparse()?
            .map(|s| s.reshape(rows, cols).and_then(Self::owned_csr))
            .transpose()
    }

    /// Create a matrix from existing data
    pub fn from_data(data: Vec<T>, rows: usize, cols: usize) -> MatrixResult<Self> {
        let size = rows.checked_mul(cols).ok_or(MemoryError::InvalidSize)?;
        if data.len() != size {
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
                let idx = row
                    .checked_mul(self.strides.row_stride)
                    .and_then(|x| {
                        col.checked_mul(self.strides.col_stride)
                            .and_then(|y| x.checked_add(y))
                    })
                    .ok_or_else(|| MatrixError::InvalidInput("Matrix stride overflow".into()))?;
                data.as_slice().get(idx).copied().ok_or_else(|| {
                    MatrixError::InvalidInput("Matrix strides exceed storage".into())
                })
            }
            MatrixStorage::Sparse(sparse) => {
                if sparse.shape != self.shape {
                    return Err(MatrixError::InvalidInput(
                        "CSR shape metadata mismatch".into(),
                    ));
                }
                sparse.get(row, col)
            }
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
                let idx = row
                    .checked_mul(self.strides.row_stride)
                    .and_then(|x| {
                        col.checked_mul(self.strides.col_stride)
                            .and_then(|y| x.checked_add(y))
                    })
                    .ok_or_else(|| MatrixError::InvalidInput("Matrix stride overflow".into()))?;
                *data.as_slice_mut().get_mut(idx).ok_or_else(|| {
                    MatrixError::InvalidInput("Matrix strides exceed storage".into())
                })? = value;
                Ok(())
            }
            MatrixStorage::Sparse(sparse) => {
                if sparse.shape != self.shape {
                    return Err(MatrixError::InvalidInput(
                        "CSR shape metadata mismatch".into(),
                    ));
                }
                sparse.set(row, col, value)
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
        if self.cols() != other.rows() {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.rows(), self.cols()),
                got: (other.rows(), other.cols()),
            });
        }
        if let (Some(a), Some(b)) = (self.checked_sparse()?, other.checked_sparse()?) {
            return Self::owned_csr(a.matmul(b)?);
        }
        if let Some(a) = self.checked_sparse()? {
            let b = other.to_vec()?;
            if a.values.iter().all(|v| v.is_finite()) && b.iter().all(|v| v.is_finite()) {
                let mut out = Self::new(self.rows(), other.cols())?;
                for i in 0..self.rows() {
                    for j in 0..other.cols() {
                        let mut dot = super::checked_kernels::Product::<T>::new();
                        for entry in a.row_pointers[i]..a.row_pointers[i + 1] {
                            dot.add(a.values[entry], b[a.col_indices[entry] * other.cols() + j])?;
                        }
                        out.set(i, j, dot.total()?)?;
                    }
                }
                return Ok(out);
            }
        }
        if let Some(b) = other.checked_sparse()? {
            let a = self.to_vec()?;
            if b.values.iter().all(|v| v.is_finite()) && a.iter().all(|v| v.is_finite()) {
                let bt = b.transpose()?;
                let mut out = Self::new(self.rows(), other.cols())?;
                for i in 0..self.rows() {
                    for j in 0..other.cols() {
                        let mut dot = super::checked_kernels::Product::<T>::new();
                        for entry in bt.row_pointers[j]..bt.row_pointers[j + 1] {
                            dot.add(a[i * self.cols() + bt.col_indices[entry]], bt.values[entry])?;
                        }
                        out.set(i, j, dot.total()?)?;
                    }
                }
                return Ok(out);
            }
        }
        let a = self.to_vec()?;
        let b = other.to_vec()?;
        let mut out = Self::new(self.rows(), other.cols())?;
        if self.rows() == 0 || other.cols() == 0 || self.cols() == 0 {
            return Ok(out);
        }
        for i in 0..self.rows() {
            for j in 0..other.cols() {
                let mut product = super::checked_kernels::Product::<T>::new();
                for k in 0..self.cols() {
                    product.add(a[i * self.cols() + k], b[k * other.cols() + j])?;
                }
                out.set(i, j, product.total()?)?;
            }
        }
        Ok(out)
    }
    pub(crate) fn checked_zip(
        &self,
        other: &Self,
        op: fn(T, T) -> Option<T>,
    ) -> MatrixResult<Self> {
        if self.shape != other.shape {
            return Err(MatrixError::DimensionMismatch {
                expected: self.shape.as_tuple(),
                got: other.shape.as_tuple(),
            });
        }
        if let (Some(a), Some(b)) = (self.checked_sparse()?, other.checked_sparse()?) {
            return Self::owned_csr(a.binary(b, op)?);
        }
        let values = self
            .to_vec()?
            .into_iter()
            .zip(other.to_vec()?)
            .map(|(a, b)| {
                op(a, b)
                    .ok_or_else(|| MatrixError::InvalidInput("Element arithmetic overflow".into()))
            })
            .collect::<MatrixResult<Vec<_>>>()?;
        Self::from_data(values, self.rows(), self.cols())
    }
    /// Checked logical element-wise addition, independent of physical layout.
    pub fn add(&self, other: &Self) -> MatrixResult<Self> {
        if let (Some(a), Some(b)) = (self.checked_sparse()?, other.checked_sparse()?) {
            return Self::owned_csr(a.add(b)?);
        }
        self.checked_zip(other, T::checked_add)
    }
    /// Checked logical element-wise subtraction.
    pub fn subtract(&self, other: &Self) -> MatrixResult<Self> {
        if let (Some(a), Some(b)) = (self.checked_sparse()?, other.checked_sparse()?) {
            return Self::owned_csr(a.subtract(b)?);
        }
        self.checked_zip(other, T::checked_sub)
    }
    /// Checked logical Hadamard product.
    pub fn mul(&self, other: &Self) -> MatrixResult<Self> {
        if let (Some(a), Some(b)) = (self.checked_sparse()?, other.checked_sparse()?) {
            return Self::owned_csr(a.hadamard(b)?);
        }
        self.checked_zip(other, T::checked_mul)
    }
    /// Checked logical scalar multiplication.
    pub fn scale(&self, scalar: T) -> MatrixResult<Self> {
        if let Some(s) = self.checked_sparse()? {
            return Self::owned_csr(s.scale(scalar)?);
        }
        let values = self
            .to_vec()?
            .into_iter()
            .map(|v| {
                v.checked_mul(scalar)
                    .ok_or_else(|| MatrixError::InvalidInput("Scalar arithmetic overflow".into()))
            })
            .collect::<MatrixResult<Vec<_>>>()?;
        Self::from_data(values, self.rows(), self.cols())
    }

    /// Alias for scale - scalar multiplication (for autodiff compatibility)
    pub fn scalar_multiply(&self, scalar: T) -> MatrixResult<Self> {
        self.scale(scalar)
    }

    /// Sum all elements
    pub fn sum(&self) -> MatrixResult<T::Accumulator> {
        if let Some(s) = self.checked_sparse()? {
            return super::checked_kernels::sum(&s.values);
        }
        super::checked_kernels::sum(&self.to_vec()?)
    }
    /// Maximum logical value with first-tie and NaN propagation.
    pub fn max(&self) -> MatrixResult<T> {
        if let Some(s) = self.checked_sparse()? {
            let total = self
                .rows()
                .checked_mul(self.cols())
                .ok_or_else(|| MatrixError::InvalidInput("CSR dimensions overflow".into()))?;
            if s.values.is_empty() && total > 0 {
                return Ok(T::zero());
            }
            let best = super::checked_kernels::extrema(&s.values, false)?;
            if total > s.values.len() && !best.to_f64().is_nan() {
                return super::checked_kernels::extrema(&[best, T::zero()], false);
            }
            return Ok(best);
        }
        super::checked_kernels::extrema(&self.to_vec()?, false)
    }
    /// Minimum logical value with first-tie and NaN propagation.
    pub fn min(&self) -> MatrixResult<T> {
        if let Some(s) = self.checked_sparse()? {
            let total = self
                .rows()
                .checked_mul(self.cols())
                .ok_or_else(|| MatrixError::InvalidInput("CSR dimensions overflow".into()))?;
            if s.values.is_empty() && total > 0 {
                return Ok(T::zero());
            }
            let best = super::checked_kernels::extrema(&s.values, true)?;
            if total > s.values.len() && !best.to_f64().is_nan() {
                return super::checked_kernels::extrema(&[best, T::zero()], true);
            }
            return Ok(best);
        }
        super::checked_kernels::extrema(&self.to_vec()?, true)
    }
    /// Owned logical transpose.
    pub fn transpose(&self) -> MatrixResult<Self> {
        if let Some(s) = self.checked_sparse()? {
            return Self::owned_csr(s.transpose()?);
        }

        let mut out = Self::new(self.cols(), self.rows())?;
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                out.set(j, i, self.get(i, j)?)?;
            }
        }
        Ok(out)
    }

    /// Convert to a vector (flatten)
    pub fn to_vec(&self) -> MatrixResult<Vec<T>> {
        match &self.storage {
            MatrixStorage::Dense(buffer) => {
                let data = buffer.data()?;
                let values = data.as_slice();
                let mut out = Vec::new();
                let count = self
                    .rows()
                    .checked_mul(self.cols())
                    .ok_or_else(|| MatrixError::InvalidInput("Matrix shape overflow".into()))?;
                if count == 0 {
                    return Ok(out);
                }
                out.try_reserve_exact(count)
                    .map_err(|_| MatrixError::InvalidInput("Matrix allocation failed".into()))?;
                for i in 0..self.rows() {
                    for j in 0..self.cols() {
                        let index = i
                            .checked_mul(self.strides.row_stride)
                            .and_then(|x| {
                                j.checked_mul(self.strides.col_stride)
                                    .and_then(|y| x.checked_add(y))
                            })
                            .ok_or_else(|| {
                                MatrixError::InvalidInput("Matrix stride overflow".into())
                            })?;
                        out.push(*values.get(index).ok_or_else(|| {
                            MatrixError::InvalidInput("Matrix strides exceed storage".into())
                        })?);
                    }
                }
                Ok(out)
            }
            MatrixStorage::Sparse(sparse) => {
                if sparse.shape != self.shape {
                    return Err(MatrixError::InvalidInput(
                        "CSR shape metadata mismatch".into(),
                    ));
                }
                sparse.to_dense_values()
            }
            _ => Err(MatrixError::UnsupportedOperation(
                "to_vec for non-dense matrices not implemented".to_string(),
            )),
        }
    }

    /// Element-wise multiplication with another matrix
    pub fn elementwise_multiply(&self, other: &Self) -> MatrixResult<Self> {
        self.mul(other)
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
            MatrixStorage::Sparse(sparse) => Self {
                storage: MatrixStorage::Sparse(sparse.clone()),
                shape: self.shape,
                strides: self.strides,
                layout: self.layout,
                backend_hint: self.backend_hint.clone(),
                backend_selector: Arc::clone(&self.backend_selector),
            },
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
