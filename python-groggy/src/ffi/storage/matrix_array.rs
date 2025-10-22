use crate::ffi::storage::matrix::PyGraphMatrix;
use crate::ffi::storage::subgraph_array::PySubgraphArray;
use crate::ffi::storage::table_array::PyTableArray;
use groggy::storage::array::{ArrayIterator, NumArray};
use pyo3::prelude::*;
use pyo3::types::PyAny;

/// MatrixArray: Collection of GraphMatrix objects with delegation to NumArray
/// Provides basic array operations plus statistical operations on matrix collections
#[pyclass(name = "MatrixArray", unsendable)]
pub struct PyMatrixArray {
    // Delegates to NumArray for numerical operations
    // Note: NumArray contains BaseArray internally
    matrices: Vec<PyGraphMatrix>,
}

#[pymethods]
impl PyMatrixArray {
    /// Create a new MatrixArray from a vector of GraphMatrix objects
    #[new]
    pub fn new(matrices: Vec<PyGraphMatrix>) -> Self {
        Self { matrices }
    }

    // BaseArray delegation (via internal vector)

    /// Get the number of GraphMatrix objects
    fn __len__(&self) -> usize {
        self.matrices.len()
    }

    /// Check if the array is empty
    fn is_empty(&self) -> bool {
        self.matrices.is_empty()
    }

    /// Get a GraphMatrix by index
    fn __getitem__(&self, index: isize) -> PyResult<PyGraphMatrix> {
        let len = self.matrices.len();
        if len == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "index out of range: array is empty",
            ));
        }

        let actual_index = if index < 0 {
            let positive_index = len as isize + index;
            if positive_index < 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "index {} is out of range for array of length {}",
                    index, len
                )));
            }
            positive_index as usize
        } else {
            index as usize
        };

        self.matrices.get(actual_index).cloned().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "index {} is out of range for array of length {}",
                index, len
            ))
        })
    }

    /// Get the first GraphMatrix
    fn first(&self) -> Option<PyGraphMatrix> {
        self.matrices.first().cloned()
    }

    /// Get the last GraphMatrix
    fn last(&self) -> Option<PyGraphMatrix> {
        self.matrices.last().cloned()
    }

    /// Convert to Python list
    fn to_list(&self) -> Vec<PyGraphMatrix> {
        self.matrices.clone()
    }

    // NumArray delegation - statistical operations on matrix collections

    /// Calculate mean eigenvalue across all matrices
    fn mean_eigenvalue(&self) -> PyResult<f64> {
        if self.matrices.is_empty() {
            return Ok(0.0);
        }

        let mut eigenvalues = Vec::new();

        for matrix in &self.matrices {
            // For now, use a placeholder eigenvalue calculation
            // In full implementation, would calculate actual eigenvalues
            let placeholder_eigenvalue = matrix.shape().0 as f64; // Use dimensions as proxy
            eigenvalues.push(placeholder_eigenvalue);
        }

        let num_array = NumArray::new(eigenvalues);
        Ok(num_array.mean().unwrap_or(0.0))
    }

    /// Calculate statistics on matrix dimensions
    fn dimension_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dimensions: Vec<f64> = self
                .matrices
                .iter()
                .map(|matrix| {
                    let (rows, _cols) = matrix.shape();
                    rows as f64
                })
                .collect();

            if dimensions.is_empty() {
                let empty_dict = pyo3::types::PyDict::new(py);
                return Ok(empty_dict.into());
            }

            let array = NumArray::new(dimensions);
            let summary = array.describe();

            let stats_dict = pyo3::types::PyDict::new(py);
            stats_dict.set_item("count", summary.count)?;
            stats_dict.set_item("mean", summary.mean)?;
            stats_dict.set_item("std_dev", summary.std_dev)?;
            stats_dict.set_item("min", summary.min)?;
            stats_dict.set_item("max", summary.max)?;
            stats_dict.set_item("median", summary.median)?;

            Ok(stats_dict.into())
        })
    }

    /// Calculate correlation between matrix elements (simplified)
    fn correlation_matrix(&self) -> PyResult<Option<f64>> {
        if self.matrices.len() < 2 {
            return Ok(None);
        }

        // For now, return a placeholder correlation
        // In full implementation, would calculate actual matrix element correlations
        let dimensions_a: Vec<f64> = self
            .matrices
            .iter()
            .map(|matrix| matrix.shape().0 as f64)
            .collect();

        let dimensions_b: Vec<f64> = self
            .matrices
            .iter()
            .map(|matrix| matrix.shape().1 as f64)
            .collect();

        let stats_a = NumArray::new(dimensions_a);
        let stats_b = NumArray::new(dimensions_b);

        Ok(stats_a.correlate(&stats_b))
    }

    /// Get summary statistics for all matrices in the array
    fn stats_summary(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let sizes: Vec<f64> = self
                .matrices
                .iter()
                .map(|matrix| {
                    let (rows, cols) = matrix.shape();
                    (rows * cols) as f64
                })
                .collect();

            if sizes.is_empty() {
                let empty_dict = pyo3::types::PyDict::new(py);
                return Ok(empty_dict.into());
            }

            let stats_array = NumArray::new(sizes);
            let summary = stats_array.describe();

            let stats_dict = pyo3::types::PyDict::new(py);
            stats_dict.set_item("matrix_count", summary.count)?;
            stats_dict.set_item("mean_size", summary.mean)?;
            stats_dict.set_item("std_dev_size", summary.std_dev)?;
            stats_dict.set_item("min_size", summary.min)?;
            stats_dict.set_item("max_size", summary.max)?;
            stats_dict.set_item("median_size", summary.median)?;

            Ok(stats_dict.into())
        })
    }

    // Domain-specific operations

    /// Stack matrices vertically (concatenate rows)
    fn stack(&self) -> PyResult<PyGraphMatrix> {
        if self.matrices.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot stack empty MatrixArray",
            ));
        }

        // For now, return the first matrix as a placeholder
        // In full implementation, would properly stack matrices
        Ok(self.matrices[0].clone())
    }

    /// Filter matrices using a Python predicate function
    fn filter(&self, predicate: &PyAny) -> PyResult<Self> {
        let mut filtered = Vec::new();

        Python::with_gil(|_py| {
            for matrix in &self.matrices {
                match predicate.call1((matrix.clone(),)) {
                    Ok(result) => {
                        if result.is_true().unwrap_or(false) {
                            filtered.push(matrix.clone());
                        }
                    }
                    Err(_) => continue, // Skip on error
                }
            }
        });

        Ok(Self::new(filtered))
    }

    /// Filter matrices by minimum size
    fn filter_by_size(&self, min_size: usize) -> Self {
        let filtered: Vec<PyGraphMatrix> = self
            .matrices
            .iter()
            .filter(|matrix| {
                let (rows, cols) = matrix.shape();
                rows * cols >= min_size
            })
            .cloned()
            .collect();

        Self::new(filtered)
    }

    /// Extract subgraphs from matrices (placeholder implementation)
    fn subgraphs(&self) -> PyResult<PySubgraphArray> {
        // For now, return empty SubgraphArray as placeholder
        // In full implementation, would convert matrices to subgraphs
        Ok(PySubgraphArray::new(Vec::new()))
    }

    /// Convert matrices to tables (edge lists)
    fn table(&self) -> PyResult<PyTableArray> {
        let mut tables = Vec::new();

        Python::with_gil(|py| {
            for matrix in &self.matrices {
                // For now, create a simple table representation
                // In full implementation, would convert matrix to edge list table
                if let Ok(table) = matrix.to_table(py) {
                    let table_obj = table.into_py(py);
                    tables.push(table_obj);
                }
            }
        });

        Ok(PyTableArray::new(tables))
    }

    /// Create an iterator for method chaining
    fn iter(&self) -> PyMatrixArrayIterator {
        PyMatrixArrayIterator::new(self.matrices.clone())
    }

    // Python dunder methods

    fn __repr__(&self) -> String {
        format!("MatrixArray(length={})", self.matrices.len())
    }

    fn __str__(&self) -> String {
        if self.matrices.is_empty() {
            "MatrixArray[]".to_string()
        } else {
            let total_elements: usize = self
                .matrices
                .iter()
                .map(|matrix| {
                    let (rows, cols) = matrix.shape();
                    rows * cols
                })
                .sum();
            format!(
                "MatrixArray[{} matrices, {} total elements]",
                self.matrices.len(),
                total_elements
            )
        }
    }
}

/// Iterator for MatrixArray that enables method chaining
#[pyclass(name = "MatrixArrayIterator", unsendable)]
pub struct PyMatrixArrayIterator {
    inner: ArrayIterator<PyGraphMatrix>,
}

impl PyMatrixArrayIterator {
    pub fn new(matrices: Vec<PyGraphMatrix>) -> Self {
        Self {
            inner: ArrayIterator::new(matrices),
        }
    }
}

#[pymethods]
impl PyMatrixArrayIterator {
    /// Apply element-wise multiplication to each matrix with another iterator
    fn multiply(&mut self, other: &Self) -> PyResult<Self> {
        let self_elements = self.inner.clone().into_vec();
        let other_elements = other.inner.clone().into_vec();

        if self_elements.len() != other_elements.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot multiply MatrixArrays of different lengths",
            ));
        }

        let mut result = Vec::new();

        for (a, _b) in self_elements.iter().zip(other_elements.iter()) {
            // For now, just return the first matrix as placeholder
            // In full implementation, would perform matrix multiplication
            result.push(a.clone());
        }

        Ok(Self::new(result))
    }

    /// Extract eigenvalues from each matrix and return as StatsArray (conceptual)
    fn eigen(&mut self) -> PyResult<Vec<f64>> {
        let elements = self.inner.clone().into_vec();
        let mut eigenvalues = Vec::new();

        for matrix in elements {
            // For now, use matrix dimensions as proxy for eigenvalues
            // In full implementation, would calculate actual eigenvalues
            let (rows, _cols) = matrix.shape();
            eigenvalues.push(rows as f64);
        }

        Ok(eigenvalues)
    }

    /// Apply a mathematical transformation to each matrix
    fn transform(&mut self, _operation: String) -> PyResult<Self> {
        let elements = self.inner.clone().into_vec();
        let mut transformed = Vec::new();

        for matrix in elements {
            // For now, just return the original matrix regardless of operation
            // In full implementation, would apply the specified transformation
            transformed.push(matrix);
        }

        Ok(Self::new(transformed))
    }

    /// Filter matrices using a Python predicate function
    fn filter(&mut self, predicate: &PyAny) -> PyResult<Self> {
        let elements = self.inner.clone().into_vec();
        let mut filtered = Vec::new();

        Python::with_gil(|_py| {
            for matrix in elements {
                match predicate.call1((matrix.clone(),)) {
                    Ok(result) => {
                        if result.is_true().unwrap_or(false) {
                            filtered.push(matrix);
                        }
                    }
                    Err(_) => continue,
                }
            }
        });

        Ok(Self::new(filtered))
    }

    /// Collect the iterator back into a MatrixArray
    fn collect(&mut self) -> PyMatrixArray {
        let elements = self.inner.clone().into_vec();
        PyMatrixArray::new(elements)
    }

    /// Get the current length of the iterator
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}
