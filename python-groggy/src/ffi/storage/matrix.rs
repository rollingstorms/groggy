//! Matrix FFI Bindings
//!
//! Python bindings for GraphMatrix - general-purpose matrix operations.

#![allow(unused_variables)]

use groggy::storage::array::BaseArray;
use groggy::storage::array::NumArray;
use groggy::storage::GraphMatrix;
use groggy::types::AttrValue as RustAttrValue;
use pyo3::exceptions::{
    PyImportError, PyIndexError, PyKeyError, PyNotImplementedError, PyRuntimeError, PyTypeError,
    PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashMap;

// Use utility functions from utils module
use crate::ffi::storage::array::PyBaseArray;
use crate::ffi::storage::num_array::PyNumArray;

/// Python wrapper for GraphMatrix - general-purpose matrix for collections of GraphArrays
#[derive(Clone)]
#[pyclass(name = "GraphMatrix", unsendable)]
pub struct PyGraphMatrix {
    /// Core GraphMatrix
    pub inner: GraphMatrix,
}

#[pymethods]
impl PyGraphMatrix {
    /// Create a new GraphMatrix from BaseArrays (mixed types) or NumArrays (numerical)
    #[new]
    pub fn new(py: Python, arrays: Vec<PyObject>) -> PyResult<Self> {
        if arrays.is_empty() {
            return Err(PyValueError::new_err(
                "Cannot create matrix from empty array list",
            ));
        }

        let mut core_arrays: Vec<NumArray<f64>> = Vec::new();

        for (i, array_obj) in arrays.iter().enumerate() {
            // Try NumArray first (preferred for matrices)
            if let Ok(num_array) = array_obj.extract::<PyRef<PyNumArray>>(py) {
                // Convert to f64 array for matrix use
                if let Some(f64_array) = num_array.as_float64_array() {
                    core_arrays.push(f64_array.clone());
                } else {
                    // Convert other types to f64
                    let f64_values = num_array.to_float64_vec();
                    let f64_array = NumArray::new(f64_values);
                    core_arrays.push(f64_array);
                }
            }
            // Convert BaseArray to NumArray<f64>
            else if let Ok(base_array) = array_obj.extract::<PyRef<PyBaseArray>>(py) {
                let f64_values: Vec<f64> = base_array
                    .inner
                    .iter()
                    .map(|attr_value| match attr_value {
                        RustAttrValue::Float(f) => *f as f64,
                        RustAttrValue::Int(i) => *i as f64,
                        RustAttrValue::Bool(b) => {
                            if *b {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        _ => 0.0, // Default for non-numeric types
                    })
                    .collect();
                core_arrays.push(NumArray::new(f64_values));
            }
            // Convert legacy PyGraphArray to NumArray<f64>
            else if let Ok(base_array) = array_obj.extract::<PyRef<PyBaseArray>>(py) {
                let f64_values: Vec<f64> = base_array
                    .inner
                    .data()
                    .iter()
                    .map(|attr_value| match attr_value {
                        RustAttrValue::Float(f) => *f as f64,
                        RustAttrValue::Int(i) => *i as f64,
                        RustAttrValue::Bool(b) => {
                            if *b {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        _ => 0.0, // Default for non-numeric types
                    })
                    .collect();
                core_arrays.push(NumArray::new(f64_values));
            } else {
                return Err(PyTypeError::new_err(format!(
                    "Array {} must be BaseArray, NumArray, or GraphArray",
                    i
                )));
            }
        }

        // Create core GraphMatrix
        let matrix = GraphMatrix::from_arrays(core_arrays)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create matrix: {:?}", e)))?;

        Ok(Self { inner: matrix })
    }

    /// Check whether the matrix contains any rows or columns
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Create a zero matrix with specified dimensions and type
    #[classmethod]
    fn zeros(
        _cls: &PyType,
        py: Python,
        rows: usize,
        cols: usize,
        dtype: Option<&str>,
    ) -> PyResult<Py<Self>> {
        // Parse dtype string to AttrValueType
        let attr_type = match dtype.unwrap_or("float") {
            "int" | "int64" => groggy::AttrValueType::Int,
            "float" | "float64" | "f64" => groggy::AttrValueType::Float,
            "bool" => groggy::AttrValueType::Bool,
            "str" | "string" | "text" => groggy::AttrValueType::Text,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported dtype: {}",
                    dtype.unwrap_or("unknown")
                )))
            }
        };

        let matrix = GraphMatrix::zeros(rows, cols);
        Py::new(py, Self { inner: matrix })
    }

    /// Create a ones matrix with specified dimensions
    #[classmethod]
    fn ones(
        _cls: &PyType,
        py: Python,
        rows: usize,
        cols: usize,
        dtype: Option<&str>,
    ) -> PyResult<Py<Self>> {
        // Parse dtype string to AttrValueType (for consistency with zeros)
        let _attr_type = match dtype.unwrap_or("float") {
            "int" | "int64" => groggy::AttrValueType::Int,
            "float" | "float64" | "f64" => groggy::AttrValueType::Float,
            "bool" => groggy::AttrValueType::Bool,
            "str" | "string" | "text" => groggy::AttrValueType::Text,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported dtype: {}",
                    dtype.unwrap_or("unknown")
                )))
            }
        };

        let matrix = GraphMatrix::ones(rows, cols);
        Py::new(py, Self { inner: matrix })
    }

    /// Create an identity matrix with specified size
    #[classmethod]
    fn identity(_cls: &PyType, py: Python, size: usize) -> PyResult<Py<Self>> {
        let matrix = GraphMatrix::identity(size).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create identity matrix: {:?}", e))
        })?;
        Py::new(py, Self { inner: matrix })
    }

    /// Create matrix from nested Python lists (API consistency)
    ///
    /// This is a classmethod wrapper around the `groggy.matrix()` function
    /// for API consistency with other matrix libraries.
    ///
    /// Examples:
    ///   groggy.GraphMatrix.from_data([[1, 2], [3, 4]])  # 2×2 matrix
    ///   groggy.GraphMatrix.from_data([[1, 2, 3]])       # 1×3 matrix
    #[classmethod]
    fn from_data(_cls: &PyType, py: Python, data: PyObject) -> PyResult<Py<Self>> {
        // Delegate to the existing matrix() function which handles nested lists perfectly
        let matrix = crate::matrix(py, data)?;
        Py::new(py, matrix)
    }

    /// Create matrix from graph attributes
    #[classmethod]
    fn from_graph_attributes(
        _cls: &PyType,
        _py: Python,
        _graph: PyObject,
        _attrs: Vec<String>,
        _entities: Vec<u64>,
    ) -> PyResult<Py<Self>> {
        // TODO: Implement graph integration in Phase 2
        // For now, return a placeholder error
        Err(PyNotImplementedError::new_err(
            "Graph integration not yet implemented in Phase 2",
        ))
    }

    // === PROPERTIES ===

    /// Get matrix dimensions as (rows, columns) tuple
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Get matrix data type
    #[getter]
    fn dtype(&self) -> String {
        format!("{:?}", self.inner.dtype())
    }

    /// Get column names
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner.column_names().to_vec()
    }

    /// Check if matrix is square
    #[getter]
    fn is_square(&self) -> bool {
        self.inner.is_square()
    }

    /// Check if matrix is symmetric (for square numeric matrices)
    #[getter]
    fn is_symmetric(&self) -> bool {
        self.inner.is_symmetric()
    }

    /// Check if matrix contains only numeric data
    #[getter]
    fn is_numeric(&self) -> bool {
        self.inner.is_numeric()
    }

    /// Check if gradients are enabled for this matrix
    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad_enabled()
    }

    /// Get gradient matrix (None if no gradients computed yet)
    #[getter]
    fn grad(&self, py: Python) -> Option<Py<Self>> {
        self.inner
            .grad()
            .map(|grad_matrix| Py::new(py, PyGraphMatrix { inner: grad_matrix }).unwrap())
    }

    // === AUTOMATIC DIFFERENTIATION ===

    /// Enable or disable gradient tracking for this matrix
    fn requires_grad_(&self, py: Python, requires_grad: bool) -> Py<Self> {
        let new_matrix = self.inner.clone().requires_grad(requires_grad);
        Py::new(py, PyGraphMatrix { inner: new_matrix }).unwrap()
    }

    /// Compute gradients via backpropagation
    fn backward(&mut self) -> PyResult<()> {
        self.inner
            .backward()
            .map_err(|e| PyRuntimeError::new_err(format!("Backward pass failed: {:?}", e)))
    }

    /// Zero out all gradients in the computation graph
    fn zero_grad(&mut self) -> PyResult<()> {
        self.inner
            .zero_grad()
            .map_err(|e| PyRuntimeError::new_err(format!("Gradient clearing failed: {:?}", e)))
    }

    // === ACCESS & INDEXING ===

    /// Multi-index access for matrix elements: matrix[row, col] -> value, matrix[row] -> row, matrix["col"] -> column
    /// Advanced matrix indexing supporting 2D slicing operations
    /// Supports:
    /// - Single cell: matrix[row, col] -> f64
    /// - Row access: matrix[row] -> NumArray
    /// - Column access: matrix["col_name"] -> NumArray
    /// - 2D slicing: matrix[:5, :3], matrix[::2, ::2]
    /// - Advanced indexing: matrix[[0, 2], [1, 3]]
    /// - Boolean indexing: matrix[row_mask, :]
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        use crate::ffi::utils::matrix_indexing::{parse_matrix_index, MatrixIndexResult};
        use groggy::storage::matrix::MatrixSlicing;

        match parse_matrix_index(py, key)? {
            MatrixIndexResult::Cell(row, col) => {
                // Single cell access: matrix[row, col]
                self.get_cell(py, row, col)
            }
            MatrixIndexResult::Row(row_index) => {
                // Row access: matrix[row]
                self.get_row(py, row_index)
            }
            MatrixIndexResult::ColumnByName(col_name) => {
                // Column access by name: matrix["col_name"]
                self.get_column_by_name(py, col_name)
            }
            MatrixIndexResult::Slice(matrix_slice) => {
                // Advanced 2D slicing: matrix[:5, :3], matrix[[0,2], [1,3]], etc.
                match self.inner.get_submatrix(&matrix_slice) {
                    Ok(sliced_matrix) => {
                        let py_matrix = PyGraphMatrix {
                            inner: sliced_matrix,
                        };
                        Ok(py_matrix.into_py(py))
                    }
                    Err(e) => Err(pyo3::exceptions::PyIndexError::new_err(format!("{}", e))),
                }
            }
        }
    }

    /// Get single cell value at (row, col)
    fn get_cell(&self, py: Python, row: usize, col: usize) -> PyResult<PyObject> {
        match self.inner.get(row, col) {
            Some(f64_value) => Ok(f64_value.to_object(py)),
            None => {
                let (rows, cols) = self.inner.shape();
                Err(PyIndexError::new_err(format!(
                    "Index ({}, {}) out of range for {}x{} matrix",
                    row, col, rows, cols
                )))
            }
        }
    }

    /// Get single cell value at (row, col) - public interface
    fn get(&self, row: usize, col: usize) -> PyResult<f64> {
        match self.inner.get(row, col) {
            Some(value) => Ok(value),
            None => {
                let (rows, cols) = self.inner.shape();
                Err(PyIndexError::new_err(format!(
                    "Index ({}, {}) out of range for {}x{} matrix",
                    row, col, rows, cols
                )))
            }
        }
    }

    /// Set single cell value at (row, col)
    fn set(&mut self, row: usize, col: usize, value: f64) -> PyResult<()> {
        self.inner
            .set(row, col, value)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set cell value: {:?}", e)))
    }

    /// Python __setitem__ implementation for matrix[row, col] = value assignment
    fn __setitem__(&mut self, index: PyObject, value: PyObject, py: Python) -> PyResult<()> {
        // Parse index - support (row, col) tuple or single index
        if let Ok((row, col)) = index.extract::<(usize, usize)>(py) {
            // Direct (row, col) assignment: matrix[1, 2] = 5.0
            let f64_value = value.extract::<f64>(py).map_err(|_| {
                PyTypeError::new_err("Matrix assignment value must be numeric (f64)")
            })?;
            self.set(row, col, f64_value)
        } else {
            // TODO: Support advanced indexing like slices, arrays, etc.
            Err(PyNotImplementedError::new_err(
                "Advanced indexing assignment not yet implemented. Use matrix.set(row, col, value) or matrix[row, col] = value"
            ))
        }
    }

    /// Get entire row as BaseArray (mixed types) or NumArray (if all numerical)
    fn get_row(&self, py: Python, row: usize) -> PyResult<PyObject> {
        match self.inner.get_row(row) {
            Some(row_array) => {
                // GraphMatrix is always f64, so create f64 PyNumArray
                let values: Vec<f64> = row_array.to_vec();
                let py_num_array = PyNumArray::new_float64(values);
                Ok(Py::new(py, py_num_array)?.to_object(py))
            }
            None => {
                let (rows, _) = self.inner.shape();
                Err(PyIndexError::new_err(format!(
                    "Row index {} out of range for {} rows",
                    row, rows
                )))
            }
        }
    }

    /// Get column by name as BaseArray (mixed types) or NumArray (if all numerical)
    fn get_column_by_name(&self, py: Python, name: String) -> PyResult<PyObject> {
        match self.inner.get_column_by_name(&name) {
            Some(column) => {
                // GraphMatrix columns are always f64, so create f64 PyNumArray
                let values: Vec<f64> = column.to_vec();
                let py_num_array = PyNumArray::new_float64(values);
                Ok(Py::new(py, py_num_array)?.to_object(py))
            }
            None => Err(PyKeyError::new_err(format!("Column '{}' not found", name))),
        }
    }

    /// Get column by index as BaseArray (mixed types) or NumArray (if all numerical)
    fn get_column(&self, py: Python, col: usize) -> PyResult<PyObject> {
        match self.inner.get_column(col) {
            Some(column) => {
                // GraphMatrix columns are always f64, so create f64 PyNumArray
                let values: Vec<f64> = column.to_vec();
                let py_num_array = PyNumArray::new_float64(values);
                Ok(Py::new(py, py_num_array)?.to_object(py))
            }
            None => {
                let (_, cols) = self.inner.shape();
                Err(PyIndexError::new_err(format!(
                    "Column index {} out of range for {} columns",
                    col, cols
                )))
            }
        }
    }

    // === ITERATION ===

    /// Iterate over rows - returns iterator of BaseArrays or NumArrays
    fn iter_rows(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let (rows, _) = self.inner.shape();
        let mut row_arrays = Vec::with_capacity(rows);

        for i in 0..rows {
            match self.inner.get_row(i) {
                Some(row_array) => {
                    // GraphMatrix rows are always f64, so create f64 PyNumArray
                    let values: Vec<f64> = row_array.to_vec();
                    let py_num_array = PyNumArray::new_float64(values);
                    row_arrays.push(Py::new(py, py_num_array)?.to_object(py));
                }
                None => return Err(PyIndexError::new_err(format!("Row {} not found", i))),
            }
        }

        Ok(row_arrays)
    }

    /// Iterate over columns - returns iterator of BaseArrays or NumArrays
    fn iter_columns(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let (_, cols) = self.inner.shape();
        let mut col_arrays = Vec::with_capacity(cols);

        for i in 0..cols {
            match self.inner.get_column(i) {
                Some(col_array) => {
                    // GraphMatrix columns are always f64, so create f64 PyNumArray
                    let values: Vec<f64> = col_array.to_vec();
                    let py_num_array = PyNumArray::new_float64(values);
                    col_arrays.push(Py::new(py, py_num_array)?.to_object(py));
                }
                None => return Err(PyIndexError::new_err(format!("Column {} not found", i))),
            }
        }

        Ok(col_arrays)
    }

    // === LINEAR ALGEBRA OPERATIONS ===

    /// Transpose the matrix
    fn transpose(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        let transposed = self
            .inner
            .transpose()
            .map_err(|e| PyRuntimeError::new_err(format!("Transpose failed: {:?}", e)))?;
        Py::new(py, PyGraphMatrix { inner: transposed })
    }

    /// Matrix multiplication supporting either another matrix or a scalar factor
    /// - When passed a GraphMatrix, performs matrix multiplication (self * other)
    /// - When passed a numeric scalar, performs scalar multiplication
    fn multiply(&self, py: Python, operand: &PyAny) -> PyResult<Py<PyGraphMatrix>> {
        // Attempt matrix multiplication first (matrix operand)
        if let Ok(matrix_obj) = operand.extract::<Py<PyGraphMatrix>>() {
            let matrix_ref = matrix_obj.borrow(py);
            let result_matrix = self.inner.multiply(&matrix_ref.inner).map_err(|e| {
                PyRuntimeError::new_err(format!("Matrix multiplication failed: {:?}", e))
            })?;

            let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
            return Py::new(py, py_result);
        }

        // Fallback: try scalar multiplication
        if let Ok(scalar) = operand.extract::<f64>() {
            let result_matrix = self.inner.scalar_multiply(scalar).map_err(|e| {
                PyRuntimeError::new_err(format!("Scalar multiplication failed: {:?}", e))
            })?;

            let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
            return Py::new(py, py_result);
        }

        Err(PyTypeError::new_err(
            "GraphMatrix.multiply expects another GraphMatrix or a numeric scalar",
        ))
    }

    /// Matrix inverse (Phase 5 - placeholder for now)
    fn inverse(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        let result = self.inner.inverse().map_err(|e| {
            PyRuntimeError::new_err(format!("Matrix inverse calculation failed: {:?}", e))
        })?;
        let matrix = PyGraphMatrix { inner: result };
        Py::new(py, matrix)
    }

    /// Matrix power - raise matrix to integer power
    /// Returns: new GraphMatrix that is self^n
    fn power(&self, py: Python, n: u32) -> PyResult<Py<PyGraphMatrix>> {
        let result_matrix = self
            .inner
            .power(n)
            .map_err(|e| PyRuntimeError::new_err(format!("Matrix power failed: {:?}", e)))?;

        let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
        Py::new(py, py_result)
    }

    /// Reshape matrix to new dimensions while preserving total element count
    /// Returns: new GraphMatrix with specified shape
    fn reshape(&self, py: Python, new_rows: usize, new_cols: usize) -> PyResult<Py<PyGraphMatrix>> {
        let result_matrix = self
            .inner
            .reshape(new_rows, new_cols)
            .map_err(|e| PyRuntimeError::new_err(format!("Reshape failed: {:?}", e)))?;

        let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
        Py::new(py, py_result)
    }

    /// Elementwise multiplication (Hadamard product)
    /// Returns: new GraphMatrix with elementwise product
    fn elementwise_multiply(
        &self,
        py: Python,
        other: &PyGraphMatrix,
    ) -> PyResult<Py<PyGraphMatrix>> {
        let result_matrix = self.inner.elementwise_multiply(&other.inner).map_err(|e| {
            PyRuntimeError::new_err(format!("Elementwise multiplication failed: {:?}", e))
        })?;

        let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
        Py::new(py, py_result)
    }

    /// Determinant calculation (for square matrices)
    fn determinant(&self) -> PyResult<f64> {
        self.inner.determinant().map_err(|e| {
            PyRuntimeError::new_err(format!("Determinant calculation failed: {:?}", e))
        })
    }

    /// Calculate the trace (sum of diagonal elements) - only for square matrices
    fn trace(&self) -> PyResult<f64> {
        self.inner
            .trace()
            .map_err(|e| PyRuntimeError::new_err(format!("Trace calculation failed: {:?}", e)))
    }

    /// Calculate the Frobenius norm (Euclidean norm) of the matrix
    fn norm(&self) -> PyResult<f64> {
        Ok(self.inner.norm())
    }

    /// Calculate the L1 norm (sum of absolute values) of the matrix
    fn norm_l1(&self) -> PyResult<f64> {
        Ok(self.inner.norm_l1())
    }

    /// Calculate the L∞ norm (maximum absolute value) of the matrix
    fn norm_inf(&self) -> PyResult<f64> {
        Ok(self.inner.norm_inf())
    }

    // === STATISTICAL OPERATIONS ===

    /// Sum along specified axis (0=rows, 1=columns)
    fn sum_axis(&self, py: Python, axis: usize) -> PyResult<PyObject> {
        let axis_enum = match axis {
            0 => groggy::storage::Axis::Rows,
            1 => groggy::storage::Axis::Columns,
            _ => {
                return Err(PyValueError::new_err(
                    "Axis must be 0 (rows) or 1 (columns)",
                ))
            }
        };

        let result_array = self
            .inner
            .sum_axis(axis_enum)
            .map_err(|e| PyRuntimeError::new_err(format!("Sum axis operation failed: {:?}", e)))?;

        // Statistical results are numerical NumArray<f64>, convert directly to PyNumArray
        let f64_values: Vec<f64> = result_array.to_vec();
        let num_array = PyNumArray::new(f64_values);
        Ok(Py::new(py, num_array)?.to_object(py))
    }

    /// Mean along specified axis (0=rows, 1=columns)
    fn mean_axis(&self, py: Python, axis: usize) -> PyResult<PyObject> {
        let axis_enum = match axis {
            0 => groggy::storage::Axis::Rows,
            1 => groggy::storage::Axis::Columns,
            _ => {
                return Err(PyValueError::new_err(
                    "Axis must be 0 (rows) or 1 (columns)",
                ))
            }
        };

        let result_array = self
            .inner
            .mean_axis(axis_enum)
            .map_err(|e| PyRuntimeError::new_err(format!("Mean axis operation failed: {:?}", e)))?;

        // Statistical results are numerical NumArray<f64>, convert directly to PyNumArray
        let f64_values: Vec<f64> = result_array.to_vec();
        let num_array = PyNumArray::new(f64_values);
        Ok(Py::new(py, num_array)?.to_object(py))
    }

    /// Standard deviation along specified axis (0=rows, 1=columns)
    fn std_axis(&self, py: Python, axis: usize) -> PyResult<PyObject> {
        let axis_enum = match axis {
            0 => groggy::storage::Axis::Rows,
            1 => groggy::storage::Axis::Columns,
            _ => {
                return Err(PyValueError::new_err(
                    "Axis must be 0 (rows) or 1 (columns)",
                ))
            }
        };

        let result_array = self
            .inner
            .std_axis(axis_enum)
            .map_err(|e| PyRuntimeError::new_err(format!("Std axis operation failed: {:?}", e)))?;

        // Statistical results are numerical NumArray<f64>, convert directly to PyNumArray
        let f64_values: Vec<f64> = result_array.to_vec();
        let num_array = PyNumArray::new(f64_values);
        Ok(Py::new(py, num_array)?.to_object(py))
    }

    /// Variance along specified axis (0=rows, 1=columns)
    fn var_axis(&self, py: Python, axis: usize) -> PyResult<PyObject> {
        let axis_enum = match axis {
            0 => groggy::storage::Axis::Rows,
            1 => groggy::storage::Axis::Columns,
            _ => {
                return Err(PyValueError::new_err(
                    "Axis must be 0 (rows) or 1 (columns)",
                ))
            }
        };

        let result_array = self.inner.var_axis(axis_enum).map_err(|e| {
            PyRuntimeError::new_err(format!("Variance axis operation failed: {:?}", e))
        })?;

        // Statistical results are numerical NumArray<f64>, convert directly to PyNumArray
        let f64_values: Vec<f64> = result_array.to_vec();
        let num_array = PyNumArray::new(f64_values);
        Ok(Py::new(py, num_array)?.to_object(py))
    }

    /// Minimum along specified axis (0=rows, 1=columns)
    fn min_axis(&self, py: Python, axis: usize) -> PyResult<PyObject> {
        let axis_enum = match axis {
            0 => groggy::storage::Axis::Rows,
            1 => groggy::storage::Axis::Columns,
            _ => {
                return Err(PyValueError::new_err(
                    "Axis must be 0 (rows) or 1 (columns)",
                ))
            }
        };

        let result_array = self
            .inner
            .min_axis(axis_enum)
            .map_err(|e| PyRuntimeError::new_err(format!("Min axis operation failed: {:?}", e)))?;

        // Statistical results are numerical NumArray<f64>, convert directly to PyNumArray
        let f64_values: Vec<f64> = result_array.to_vec();
        let num_array = PyNumArray::new(f64_values);
        Ok(Py::new(py, num_array)?.to_object(py))
    }

    /// Maximum along specified axis (0=rows, 1=columns)
    fn max_axis(&self, py: Python, axis: usize) -> PyResult<PyObject> {
        let axis_enum = match axis {
            0 => groggy::storage::Axis::Rows,
            1 => groggy::storage::Axis::Columns,
            _ => {
                return Err(PyValueError::new_err(
                    "Axis must be 0 (rows) or 1 (columns)",
                ))
            }
        };

        let result_array = self
            .inner
            .max_axis(axis_enum)
            .map_err(|e| PyRuntimeError::new_err(format!("Max axis operation failed: {:?}", e)))?;

        // Statistical results are numerical NumArray<f64>, convert directly to PyNumArray
        let f64_values: Vec<f64> = result_array.to_vec();
        let num_array = PyNumArray::new(f64_values);
        Ok(Py::new(py, num_array)?.to_object(py))
    }

    /// Global sum of all elements in the matrix
    fn sum(&self) -> PyResult<f64> {
        Ok(self.inner.sum())
    }

    /// Global mean of all elements in the matrix  
    fn mean(&self) -> PyResult<f64> {
        Ok(self.inner.mean())
    }

    /// Global minimum value in the matrix
    fn min(&self) -> PyResult<Option<f64>> {
        Ok(self.inner.min())
    }

    /// Global maximum value in the matrix
    fn max(&self) -> PyResult<Option<f64>> {
        Ok(self.inner.max())
    }

    // === SCIENTIFIC COMPUTING CONVERSIONS ===

    /// Convert to Pandas DataFrame (when pandas available)
    fn to_pandas(&self, py: Python) -> PyResult<PyObject> {
        // Try to import pandas
        let pandas = py.import("pandas").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "pandas is required for to_pandas(). Install with: pip install pandas",
            )
        })?;

        // Convert to dictionary of column name -> values
        let dict = pyo3::types::PyDict::new(py);
        let column_names = self.inner.column_names();
        let (_, cols) = self.inner.shape();

        for (col_idx, col_name) in column_names.iter().enumerate() {
            if col_idx < cols {
                match self.inner.get_column(col_idx) {
                    Some(column) => {
                        // Convert column values directly to Python list
                        let py_list = pyo3::types::PyList::empty(py);
                        for val in column.iter() {
                            py_list.append(val.to_object(py))?;
                        }
                        dict.set_item(col_name, py_list)?;
                    }
                    None => {
                        return Err(PyIndexError::new_err(format!(
                            "Column {} not found",
                            col_idx
                        )))
                    }
                }
            }
        }

        // Create DataFrame
        let dataframe = pandas.call_method1("DataFrame", (dict,))?;
        Ok(dataframe.to_object(py))
    }

    // === DISPLAY & REPRESENTATION ===

    /// String representation with rich display hints
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let (rows, cols) = self.inner.shape();

        if rows == 0 || cols == 0 {
            return Ok("GraphMatrix(shape=(0, 0), dtype=f64)\n[]".to_string());
        }

        let mut output = format!("GraphMatrix(shape=({}, {}), dtype=f64)\n", rows, cols);

        let max_display_rows = 8;
        let max_display_cols = 6;
        let show_all_rows = rows <= max_display_rows;
        let show_all_cols = cols <= max_display_cols;

        // Calculate column widths for alignment
        let col_widths = vec![
            8;
            if show_all_cols {
                cols
            } else {
                max_display_cols
            }
        ];

        // Display matrix in mathematical bracket format
        let display_rows = if show_all_rows { rows } else { 5 };
        let display_cols = if show_all_cols { cols } else { 4 };

        for i in 0..display_rows {
            let row_idx = if !show_all_rows && i >= 3 {
                // For last 2 rows when truncating
                rows - (display_rows - i)
            } else {
                i
            };

            let row_prefix = if i == 0 { "[[" } else { " [" };
            output.push_str(row_prefix);

            for j in 0..display_cols {
                let col_idx = if !show_all_cols && j >= 2 {
                    // For last 2 columns when truncating
                    cols - (display_cols - j)
                } else {
                    j
                };

                if !show_all_cols && j == 2 {
                    output.push_str("...  ");
                    continue;
                }

                match self.inner.get(row_idx, col_idx) {
                    Some(val) => {
                        output.push_str(&format!("{:>8.3}", val));
                    }
                    None => output.push_str("    0.000"),
                }

                if j < display_cols - 1 {
                    output.push_str("  ");
                }
            }

            let row_suffix = if i == display_rows - 1 { "]]" } else { "]" };
            output.push_str(row_suffix);

            if i < display_rows - 1 {
                output.push('\n');
            }

            // Add ellipsis row if needed
            if !show_all_rows && i == 2 && rows > max_display_rows {
                output.push_str("\n ...");
            }
        }

        Ok(output)
    }

    /// String representation (same as __repr__ for consistency)
    fn __str__(&self, py: Python) -> PyResult<String> {
        self.__repr__(py)
    }

    /// Get rich display representation using Rust formatter
    pub fn rich_display(
        &self,
        config: Option<&crate::ffi::display::PyDisplayConfig>,
    ) -> PyResult<String> {
        let (total_rows, total_cols) = self.inner.shape();
        let dtype = format!("{:?}", self.inner.dtype());
        let sample_rows = std::cmp::min(5, total_rows);
        let sample_cols = std::cmp::min(5, total_cols);

        // Create custom matrix display without index column
        let mut lines = Vec::new();

        // Calculate column widths based on content
        let mut col_widths = vec![5; sample_cols]; // minimum width of 5

        // Check header widths
        for (j, width) in col_widths.iter_mut().enumerate().take(sample_cols) {
            let header = format!("col_{}", j);
            *width = std::cmp::max(*width, header.len());
        }

        // Check data widths
        for i in 0..sample_rows {
            for (j, width) in col_widths.iter_mut().enumerate().take(sample_cols) {
                if let Some(value) = self.inner.get(i, j) {
                    let value_str = format!("{}", value);
                    *width = std::cmp::max(*width, value_str.len());
                }
            }
        }

        // Build top border
        let mut top_border = "╭".to_string();
        for (j, &width) in col_widths.iter().enumerate() {
            top_border.push_str(&"─".repeat(width + 2));
            if j < col_widths.len() - 1 {
                top_border.push('┬');
            }
        }
        top_border.push('╮');
        lines.push(top_border);

        // Build header line
        let mut header_line = "│".to_string();
        for (j, &width) in col_widths.iter().enumerate() {
            let header = format!("col_{}", j);
            header_line.push_str(&format!(" {:<width$} │", header, width = width));
        }
        lines.push(header_line);

        // Build type header line
        let mut type_line = "│".to_string();
        for &width in &col_widths {
            type_line.push_str(&format!(" {:<width$} │", "obj", width = width));
        }
        lines.push(type_line);

        // Build separator
        let mut sep_border = "├".to_string();
        for (j, &width) in col_widths.iter().enumerate() {
            sep_border.push_str(&"─".repeat(width + 2));
            if j < col_widths.len() - 1 {
                sep_border.push('┼');
            }
        }
        sep_border.push('┤');
        lines.push(sep_border);

        // Build data rows
        for i in 0..sample_rows {
            let mut row_line = "│".to_string();
            for (j, &width) in col_widths.iter().enumerate() {
                let value_str = if let Some(value) = self.inner.get(i, j) {
                    format!("{}", value)
                } else {
                    "null".to_string()
                };
                row_line.push_str(&format!(" {:<width$} │", value_str, width = width));
            }
            lines.push(row_line);
        }

        // Build bottom border
        let mut bottom_border = "╰".to_string();
        for (j, &width) in col_widths.iter().enumerate() {
            bottom_border.push_str(&"─".repeat(width + 2));
            if j < col_widths.len() - 1 {
                bottom_border.push('┴');
            }
        }
        bottom_border.push('╯');
        lines.push(bottom_border);

        // Footer with matrix info
        let footer = format!(
            "rows: {} • cols: {} • type: GraphMatrix • dtype: {}",
            total_rows, total_cols, dtype
        );
        lines.push(footer);

        Ok(lines.join("\n"))
    }

    /// Rich HTML representation for Jupyter notebooks
    fn _repr_html_(&self, _py: Python) -> PyResult<String> {
        // Always use dense representation with truncation for large matrices
        self.inner
            .to_dense_html()
            .map_err(|e| PyRuntimeError::new_err(format!("HTML generation failed: {:?}", e)))
    }

    /// Generate dense HTML representation for small matrices
    fn dense_html_repr(&self) -> PyResult<String> {
        let (rows, cols) = self.inner.shape();

        let mut html = String::new();
        html.push_str(r#"<div class="groggy-matrix-container">"#);
        html.push_str(r#"<table class="groggy-matrix-table">"#);

        // // Header row with column indices
        // html.push_str("<thead><tr><th></th>");
        // for col in 0..cols {
        //     html.push_str(&format!(r#"<th class="col-header">{}</th>"#, col));
        // }
        // html.push_str("</tr></thead>");

        // Data rows
        html.push_str("<tbody>");
        for row in 0..rows {
            html.push_str("<tr>");
            // Row index header
            // html.push_str(&format!(r#"<th class="row-header">{}</th>"#, row));
            // Data cells
            for col in 0..cols {
                let value = self.inner.get(row, col).unwrap_or(0.0);
                let formatted_value = if value.abs() < 1e-10 {
                    "0.0".to_string()
                } else {
                    format!("{:.1}", value)
                };
                html.push_str(&format!(
                    r#"<td class="matrix-cell">{}</td>"#,
                    formatted_value
                ));
            }
            html.push_str("</tr>");
        }
        html.push_str("</tbody>");
        html.push_str("</table>");

        // Matrix info
        html.push_str(&format!(
            r#"<div class="matrix-info">Matrix {}×{} (dense)</div>"#,
            rows, cols
        ));
        html.push_str("</div>");

        // Add CSS
        html.push_str(
            r#"
<style>
.groggy-matrix-container {
    font-family: 'SF Mono', Monaco, Inconsolata, 'Roboto Mono', monospace;
    background: #fff;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    overflow: hidden;
    margin: 8px 0;
    max-width: fit-content;
}

.groggy-matrix-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
}

.groggy-matrix-table th, .groggy-matrix-table td {
    padding: 4px 8px;
    text-align: center;
    border: 1px solid #e1e4e8;
}

.groggy-matrix-table .col-header, .groggy-matrix-table .row-header {
    background: #f6f8fa;
    font-weight: 600;
    color: #586069;
}

.groggy-matrix-table .matrix-cell {
    font-variant-numeric: tabular-nums;
    background: #fff;
}

.groggy-matrix-table .matrix-cell:hover {
    background: #f1f8ff;
}

.matrix-info {
    padding: 6px 12px;
    background: #f6f8fa;
    font-size: 11px;
    color: #586069;
    border-top: 1px solid #e1e4e8;
}
</style>
"#,
        );

        Ok(html)
    }

    /// Extract display data for Python display formatters
    fn _get_display_data(&self, py: Python) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        let (rows, cols) = self.inner.shape();

        // Convert matrix to nested list for display
        let mut matrix_data = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut row_data = Vec::with_capacity(cols);
            for j in 0..cols {
                match self.inner.get(i, j) {
                    Some(attr_value) => {
                        row_data.push(attr_value.to_object(py));
                    }
                    None => row_data.push(py.None()),
                }
            }
            matrix_data.push(row_data);
        }

        dict.set_item("data", matrix_data)?;
        dict.set_item("shape", (rows, cols))?;
        dict.set_item("dtype", self.dtype())?;
        dict.set_item("columns", self.columns())?;
        dict.set_item("is_square", self.is_square())?;
        dict.set_item("is_symmetric", self.is_symmetric())?;

        Ok(dict.to_object(py))
    }

    /// Iterator support - iterates over rows as lists
    fn __iter__(slf: PyRef<Self>) -> PyResult<PyMatrixRowIterator> {
        let (rows, _) = slf.inner.shape();
        Ok(PyMatrixRowIterator {
            matrix: slf.into(),
            current_row: 0,
            total_rows: rows,
        })
    }

    // === PYTHON OPERATOR OVERLOADING ===

    /// Matrix multiplication operator (@)
    /// Implements: matrix1 @ matrix2
    fn __matmul__(&self, py: Python, other: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
        let result_matrix = self.inner.multiply(&other.inner).map_err(|e| {
            PyRuntimeError::new_err(format!("Matrix multiplication failed: {:?}", e))
        })?;

        let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
        Py::new(py, py_result)
    }

    /// Matrix multiplication operator (*)
    /// Implements: matrix1 * matrix2 (matrix multiplication, not element-wise)
    fn __mul__(&self, py: Python, other: &PyAny) -> PyResult<Py<PyGraphMatrix>> {
        self.multiply(py, other)
    }

    /// Matrix addition operator (+)
    /// Implements: matrix1 + matrix2 and matrix + scalar (broadcasting)
    fn __add__(&self, py: Python, other: &PyAny) -> PyResult<Py<PyGraphMatrix>> {
        // Try scalar addition first (broadcasting)
        if let Ok(scalar) = other.extract::<f64>() {
            let (rows, cols) = self.inner.shape();

            // Create arrays with scalar values
            let mut arrays = Vec::new();
            for _ in 0..cols {
                arrays.push(groggy::storage::array::NumArray::new(vec![scalar; rows]));
            }

            let scalar_matrix = groggy::storage::GraphMatrix::from_arrays(arrays)
                .map_err(|e| PyRuntimeError::new_err(format!("Scalar addition failed: {:?}", e)))?;

            let result = &self.inner + &scalar_matrix;
            return Py::new(py, PyGraphMatrix { inner: result });
        }

        // Try matrix addition
        if let Ok(other_matrix) = other.extract::<PyRef<PyGraphMatrix>>() {
            let result = &self.inner + &other_matrix.inner;
            return Py::new(py, PyGraphMatrix { inner: result });
        }

        Err(PyTypeError::new_err(
            "Unsupported operand type for matrix addition",
        ))
    }

    /// Matrix subtraction operator (-)
    /// Implements: matrix1 - matrix2 and matrix - scalar (broadcasting)
    fn __sub__(&self, py: Python, other: &PyAny) -> PyResult<Py<PyGraphMatrix>> {
        // Try scalar subtraction first (broadcasting)
        if let Ok(scalar) = other.extract::<f64>() {
            let (rows, cols) = self.inner.shape();

            // Create arrays with scalar values
            let mut arrays = Vec::new();
            for _ in 0..cols {
                arrays.push(groggy::storage::array::NumArray::new(vec![scalar; rows]));
            }

            let scalar_matrix = groggy::storage::GraphMatrix::from_arrays(arrays).map_err(|e| {
                PyRuntimeError::new_err(format!("Scalar subtraction failed: {:?}", e))
            })?;

            let result = &self.inner - &scalar_matrix;
            return Py::new(py, PyGraphMatrix { inner: result });
        }

        // Try matrix subtraction
        if let Ok(other_matrix) = other.extract::<PyRef<PyGraphMatrix>>() {
            let result = &self.inner - &other_matrix.inner;
            return Py::new(py, PyGraphMatrix { inner: result });
        }

        Err(PyTypeError::new_err(
            "Unsupported operand type for matrix subtraction",
        ))
    }

    /// Matrix power operator (**)
    /// Implements: matrix ** n
    fn __pow__(
        &self,
        py: Python,
        exponent: &PyAny,
        _modulo: Option<&PyAny>,
    ) -> PyResult<Py<PyGraphMatrix>> {
        if let Ok(n) = exponent.extract::<u32>() {
            return self.power(py, n);
        }

        Err(PyTypeError::new_err(
            "Matrix power (**) requires integer exponent",
        ))
    }

    /// Matrix division operator (/)
    /// Implements: matrix / scalar (element-wise division)
    fn __truediv__(&self, py: Python, other: &PyAny) -> PyResult<Py<PyGraphMatrix>> {
        if let Ok(scalar) = other.extract::<f64>() {
            if scalar.abs() < 1e-10 {
                return Err(PyRuntimeError::new_err("Division by zero"));
            }

            let (rows, cols) = self.inner.shape();

            // Create arrays with scalar values for division
            let mut arrays = Vec::new();
            for _ in 0..cols {
                arrays.push(groggy::storage::array::NumArray::new(vec![scalar; rows]));
            }

            let scalar_matrix = groggy::storage::GraphMatrix::from_arrays(arrays)
                .map_err(|e| PyRuntimeError::new_err(format!("Scalar division failed: {:?}", e)))?;

            // Element-wise division using multiplication with reciprocal
            let reciprocal = 1.0 / scalar;
            let mut reciprocal_arrays = Vec::new();
            for _ in 0..cols {
                reciprocal_arrays.push(groggy::storage::array::NumArray::new(vec![
                    reciprocal;
                    rows
                ]));
            }

            let reciprocal_matrix = groggy::storage::GraphMatrix::from_arrays(reciprocal_arrays)
                .map_err(|e| PyRuntimeError::new_err(format!("Scalar division failed: {:?}", e)))?;

            let result = self
                .inner
                .elementwise_multiply(&reciprocal_matrix)
                .map_err(|e| PyRuntimeError::new_err(format!("Division failed: {:?}", e)))?;

            return Py::new(py, PyGraphMatrix { inner: result });
        }

        Err(PyTypeError::new_err(
            "Matrix division only supports scalar divisor",
        ))
    }

    /// Unary negation operator (-)
    /// Implements: -matrix (element-wise negation)
    fn __neg__(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        // Multiply by -1 to negate all elements
        let (rows, cols) = self.inner.shape();

        let mut arrays = Vec::new();
        for _ in 0..cols {
            arrays.push(groggy::storage::array::NumArray::new(vec![-1.0; rows]));
        }

        let neg_one_matrix = groggy::storage::GraphMatrix::from_arrays(arrays)
            .map_err(|e| PyRuntimeError::new_err(format!("Negation failed: {:?}", e)))?;

        let result = self
            .inner
            .elementwise_multiply(&neg_one_matrix)
            .map_err(|e| PyRuntimeError::new_err(format!("Negation failed: {:?}", e)))?;

        Py::new(py, PyGraphMatrix { inner: result })
    }

    /// Absolute value operator (abs)
    /// Implements: abs(matrix) (element-wise absolute value)
    fn __abs__(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        // Apply element-wise absolute value
        let (rows, cols) = self.inner.shape();
        let mut result_arrays = Vec::new();

        for col in 0..cols {
            if let Some(column) = self.inner.get_column(col) {
                let mut abs_values = Vec::new();
                for value in column.iter() {
                    abs_values.push(value.abs());
                }
                result_arrays.push(groggy::storage::array::NumArray::new(abs_values));
            } else {
                return Err(PyRuntimeError::new_err("Failed to get matrix column"));
            }
        }

        let result_matrix =
            groggy::storage::GraphMatrix::from_arrays(result_arrays).map_err(|e| {
                PyRuntimeError::new_err(format!("Absolute value operation failed: {:?}", e))
            })?;

        Py::new(
            py,
            PyGraphMatrix {
                inner: result_matrix,
            },
        )
    }

    /// Comparison operator (>)
    /// Implements: matrix > scalar (element-wise comparison)
    fn __gt__(&self, py: Python, other: &PyAny) -> PyResult<Py<PyGraphMatrix>> {
        if let Ok(scalar) = other.extract::<f64>() {
            let (rows, cols) = self.inner.shape();
            let mut result_arrays = Vec::new();

            for col in 0..cols {
                if let Some(column) = self.inner.get_column(col) {
                    let mut bool_values = Vec::new();
                    for value in column.iter() {
                        bool_values.push(if *value > scalar { 1.0 } else { 0.0 });
                    }
                    result_arrays.push(groggy::storage::array::NumArray::new(bool_values));
                } else {
                    return Err(PyRuntimeError::new_err("Failed to get matrix column"));
                }
            }

            let result_matrix =
                groggy::storage::GraphMatrix::from_arrays(result_arrays).map_err(|e| {
                    PyRuntimeError::new_err(format!("Comparison operation failed: {:?}", e))
                })?;

            return Py::new(
                py,
                PyGraphMatrix {
                    inner: result_matrix,
                },
            );
        }

        Err(PyTypeError::new_err(
            "Comparison operations only support scalar values",
        ))
    }

    /// Comparison operator (<)
    /// Implements: matrix < scalar (element-wise comparison)
    fn __lt__(&self, py: Python, other: &PyAny) -> PyResult<Py<PyGraphMatrix>> {
        if let Ok(scalar) = other.extract::<f64>() {
            let (rows, cols) = self.inner.shape();
            let mut result_arrays = Vec::new();

            for col in 0..cols {
                if let Some(column) = self.inner.get_column(col) {
                    let mut bool_values = Vec::new();
                    for value in column.iter() {
                        bool_values.push(if *value < scalar { 1.0 } else { 0.0 });
                    }
                    result_arrays.push(groggy::storage::array::NumArray::new(bool_values));
                } else {
                    return Err(PyRuntimeError::new_err("Failed to get matrix column"));
                }
            }

            let result_matrix =
                groggy::storage::GraphMatrix::from_arrays(result_arrays).map_err(|e| {
                    PyRuntimeError::new_err(format!("Comparison operation failed: {:?}", e))
                })?;

            return Py::new(
                py,
                PyGraphMatrix {
                    inner: result_matrix,
                },
            );
        }

        Err(PyTypeError::new_err(
            "Comparison operations only support scalar values",
        ))
    }

    /// Comparison operator (>=)
    /// Implements: matrix >= scalar (element-wise comparison)
    fn __ge__(&self, py: Python, other: &PyAny) -> PyResult<Py<PyGraphMatrix>> {
        if let Ok(scalar) = other.extract::<f64>() {
            let (rows, cols) = self.inner.shape();
            let mut result_arrays = Vec::new();

            for col in 0..cols {
                if let Some(column) = self.inner.get_column(col) {
                    let mut bool_values = Vec::new();
                    for value in column.iter() {
                        bool_values.push(if *value >= scalar { 1.0 } else { 0.0 });
                    }
                    result_arrays.push(groggy::storage::array::NumArray::new(bool_values));
                } else {
                    return Err(PyRuntimeError::new_err("Failed to get matrix column"));
                }
            }

            let result_matrix =
                groggy::storage::GraphMatrix::from_arrays(result_arrays).map_err(|e| {
                    PyRuntimeError::new_err(format!("Comparison operation failed: {:?}", e))
                })?;

            return Py::new(
                py,
                PyGraphMatrix {
                    inner: result_matrix,
                },
            );
        }

        Err(PyTypeError::new_err(
            "Comparison operations only support scalar values",
        ))
    }

    /// Comparison operator (<=)
    /// Implements: matrix <= scalar (element-wise comparison)
    fn __le__(&self, py: Python, other: &PyAny) -> PyResult<Py<PyGraphMatrix>> {
        if let Ok(scalar) = other.extract::<f64>() {
            let (rows, cols) = self.inner.shape();
            let mut result_arrays = Vec::new();

            for col in 0..cols {
                if let Some(column) = self.inner.get_column(col) {
                    let mut bool_values = Vec::new();
                    for value in column.iter() {
                        bool_values.push(if *value <= scalar { 1.0 } else { 0.0 });
                    }
                    result_arrays.push(groggy::storage::array::NumArray::new(bool_values));
                } else {
                    return Err(PyRuntimeError::new_err("Failed to get matrix column"));
                }
            }

            let result_matrix =
                groggy::storage::GraphMatrix::from_arrays(result_arrays).map_err(|e| {
                    PyRuntimeError::new_err(format!("Comparison operation failed: {:?}", e))
                })?;

            return Py::new(
                py,
                PyGraphMatrix {
                    inner: result_matrix,
                },
            );
        }

        Err(PyTypeError::new_err(
            "Comparison operations only support scalar values",
        ))
    }

    // === NEURAL NETWORK OPERATIONS ===
    // Method-based neural network operations for chainable API

    /// ReLU activation applied to matrix
    /// Implements: matrix.relu()
    fn relu(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        crate::ffi::neural::activations::relu(py, self)
    }

    /// Sigmoid activation applied to matrix
    /// Implements: matrix.sigmoid()
    fn sigmoid(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        crate::ffi::neural::activations::sigmoid(py, self)
    }

    /// Tanh activation applied to matrix
    /// Implements: matrix.tanh()
    fn tanh(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        crate::ffi::neural::activations::tanh(py, self)
    }

    /// Softmax activation applied to matrix
    /// Implements: matrix.softmax(dim=1)
    #[pyo3(signature = (dim=1))]
    fn softmax(&self, py: Python, dim: i32) -> PyResult<Py<PyGraphMatrix>> {
        crate::ffi::neural::activations::softmax(py, self, dim)
    }

    /// GELU activation applied to matrix
    /// Implements: matrix.gelu()
    fn gelu(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        crate::ffi::neural::activations::gelu(py, self)
    }

    // === PLACEHOLDER OPERATIONS ===
    // These will be implemented once core matrix operations are stable

    /// Placeholder for scalar operations
    fn scalar_multiply(&self, py: Python, scalar: f64) -> PyResult<Py<PyGraphMatrix>> {
        Err(PyNotImplementedError::new_err(
            "Scalar operations will be implemented in next iteration",
        ))
    }

    // ========================================================================
    // LAZY EVALUATION & MATERIALIZATION
    // ========================================================================

    /// Get matrix data (materializes data to Python objects)
    /// This is the primary materialization method - use sparingly for large matrices
    #[getter]
    fn data(&self, py: Python) -> PyResult<PyObject> {
        let materialized = self
            .inner
            .materialize()
            .map_err(|e| PyRuntimeError::new_err(format!("Materialize failed: {:?}", e)))?;

        // Convert matrix to nested vectors for Python
        let (rows, cols) = materialized.shape();
        let mut py_matrix = Vec::new();
        for i in 0..rows {
            let mut row = Vec::new();
            for j in 0..cols {
                let val = materialized.get(i, j).unwrap_or_else(Default::default);
                row.push(val.to_object(py));
            }
            py_matrix.push(row);
        }

        Ok(py_matrix.to_object(py))
    }

    /// Get preview of matrix for display (first N rows/cols by default)
    fn preview(
        &self,
        py: Python,
        row_limit: Option<usize>,
        col_limit: Option<usize>,
    ) -> PyResult<PyObject> {
        let row_limit = row_limit.unwrap_or(10);
        let col_limit = col_limit.unwrap_or(10);
        let (preview_data, _col_names) = self.inner.preview(row_limit, col_limit);

        Ok(preview_data.to_object(py))
    }

    /// Check if matrix is sparse (has many default values)
    #[getter]
    fn is_sparse(&self) -> bool {
        self.inner.is_sparse()
    }

    /// Get summary information without materializing data
    fn summary(&self) -> String {
        self.inner.summary_info()
    }

    /// Create a dense materialized version of the matrix
    fn dense(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        let dense_matrix = self
            .inner
            .dense()
            .map_err(|e| PyRuntimeError::new_err(format!("Dense conversion failed: {:?}", e)))?;
        let py_result = PyGraphMatrix::from_graph_matrix(dense_matrix);
        Py::new(py, py_result)
    }

    /// Convert to NumPy array (when numpy available)
    /// Uses .data property to materialize data
    fn to_numpy(&self, py: Python) -> PyResult<PyObject> {
        // Try to import numpy
        let numpy = py.import("numpy").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "numpy is required for to_numpy(). Install with: pip install numpy",
            )
        })?;

        // Get materialized data using .data property
        let data = self.data(py)?;

        // Convert to numpy array
        let array = numpy.call_method1("array", (data,))?;
        Ok(array.to_object(py))
    }

    /// Convert matrix to table format for streaming visualization
    ///
    /// Creates a BaseTable with columns representing matrix data:
    /// - 'row': row index
    /// - 'column': column index  
    /// - 'value': the matrix value at that position
    fn to_table_for_streaming(&self) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        use groggy::storage::table::BaseTable;
        use groggy::types::AttrValue;
        use std::collections::HashMap;

        let (rows, cols) = self.shape();
        let mut columns = HashMap::new();

        // Prepare data vectors
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // Flatten the matrix into table format (row, column, value)
        for row in 0..rows {
            for col in 0..cols {
                row_indices.push(AttrValue::SmallInt(row as i32));
                col_indices.push(AttrValue::SmallInt(col as i32));

                // Get the value from the matrix
                match self.inner.get(row, col) {
                    Some(f64_value) => values.push(AttrValue::Float(f64_value as f32)),
                    None => values.push(AttrValue::Null),
                }
            }
        }

        // Create columns for the table
        columns.insert(
            "row".to_string(),
            groggy::storage::array::BaseArray::new(row_indices),
        );
        columns.insert(
            "column".to_string(),
            groggy::storage::array::BaseArray::new(col_indices),
        );
        columns.insert(
            "value".to_string(),
            groggy::storage::array::BaseArray::new(values),
        );

        // Create the BaseTable
        let base_table = BaseTable::from_columns(columns).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create table: {}", e))
        })?;

        Ok(crate::ffi::storage::table::PyBaseTable::from_table(
            base_table,
        ))
    }

    // === MATRIX INTEGRATION METHODS (Week 7-8: Matrix Integration) ===

    /// Flatten matrix to NumArray by concatenating all columns row-wise
    /// Returns a single NumArray containing all matrix values in row-major order
    /// OPTIMIZED: Pre-allocated vector with vectorized access
    pub fn flatten(&self, py: Python) -> PyResult<PyObject> {
        let (rows, cols) = self.inner.shape();

        if rows == 0 || cols == 0 {
            // Return empty NumArray for empty matrix
            let empty_array = PyNumArray::new(Vec::<f64>::new());
            return Ok(empty_array.into_py(py));
        }

        // OPTIMIZATION: Pre-allocate exact capacity to avoid reallocations
        let total_elements = rows * cols;
        let mut flattened_values = Vec::with_capacity(total_elements);

        // OPTIMIZATION: Flatten row-major order with direct indexing
        for row_idx in 0..rows {
            for col_idx in 0..cols {
                // Direct access - use unwrap_or for missing values
                let value = self.inner.get(row_idx, col_idx).unwrap_or(0.0);
                flattened_values.push(value);
            }
        }

        let num_array = PyNumArray::new(flattened_values);
        Ok(num_array.into_py(py))
    }

    /// Create matrix from NumArray by reshaping to specified dimensions
    #[staticmethod]
    pub fn from_flattened(
        py: Python,
        num_array: &PyNumArray,
        rows: usize,
        cols: usize,
    ) -> PyResult<Self> {
        let total_elements = rows * cols;
        let array_len = num_array.len();

        if total_elements == 0 {
            return Err(PyValueError::new_err(
                "Cannot create matrix with zero rows or columns",
            ));
        }

        if array_len != total_elements {
            return Err(PyValueError::new_err(format!(
                "Array length {} does not match matrix dimensions {}×{} = {}",
                array_len, rows, cols, total_elements
            )));
        }

        // Reshape the flattened array into column vectors
        let mut columns = Vec::new();

        for col_idx in 0..cols {
            let mut column_values = Vec::new();

            // Extract values for this column (row-major to column-major conversion)
            for row_idx in 0..rows {
                let flat_index = row_idx * cols + col_idx;
                let value = num_array.get_f64(flat_index).ok_or_else(|| {
                    PyIndexError::new_err(format!("Index {} out of bounds", flat_index))
                })?;
                column_values.push(value);
            }

            columns.push(NumArray::new(column_values));
        }

        let matrix = GraphMatrix::from_arrays(columns)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create matrix: {}", e)))?;

        Ok(PyGraphMatrix { inner: matrix })
    }

    /// Convert matrix to BaseArray containing all values as AttrValues
    pub fn to_base_array(&self, py: Python) -> PyResult<PyObject> {
        let (rows, cols) = self.inner.shape();
        let mut attr_values = Vec::new();

        // Collect all matrix values as AttrValue::Float
        for row_idx in 0..rows {
            for col_idx in 0..cols {
                match self.inner.get(row_idx, col_idx) {
                    Some(value) => attr_values.push(RustAttrValue::Float(value as f32)),
                    None => attr_values.push(RustAttrValue::Null),
                }
            }
        }

        let base_array = BaseArray::new(attr_values);
        let py_base = PyBaseArray { inner: base_array };
        Ok(py_base.into_py(py))
    }

    /// Create matrix from BaseArray containing numeric values
    #[staticmethod]
    pub fn from_base_array(
        py: Python,
        base_array: &PyBaseArray,
        rows: usize,
        cols: usize,
    ) -> PyResult<Self> {
        let total_elements = rows * cols;
        let array_len = base_array.inner.len();

        if total_elements == 0 {
            return Err(PyValueError::new_err(
                "Cannot create matrix with zero rows or columns",
            ));
        }

        if array_len != total_elements {
            return Err(PyValueError::new_err(format!(
                "Array length {} does not match matrix dimensions {}×{} = {}",
                array_len, rows, cols, total_elements
            )));
        }

        // First, try to convert BaseArray to numeric values
        let numeric_values: Result<Vec<f64>, String> = base_array
            .inner
            .iter()
            .map(|attr_val| match attr_val {
                RustAttrValue::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
                RustAttrValue::SmallInt(i) => Ok(*i as f64),
                RustAttrValue::Int(i) => Ok(*i as f64),
                RustAttrValue::Float(f) => Ok(*f as f64),
                RustAttrValue::Null => Ok(0.0), // Default null to 0.0
                other => Err(format!("Cannot convert {:?} to numeric value", other)),
            })
            .collect();

        let values = numeric_values.map_err(PyTypeError::new_err)?;

        // Reshape into column vectors (row-major to column-major)
        let mut columns = Vec::new();

        for col_idx in 0..cols {
            let mut column_values = Vec::new();

            for row_idx in 0..rows {
                let flat_index = row_idx * cols + col_idx;
                column_values.push(values[flat_index]);
            }

            columns.push(NumArray::new(column_values));
        }

        let matrix = GraphMatrix::from_arrays(columns)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create matrix: {}", e)))?;

        Ok(PyGraphMatrix { inner: matrix })
    }

    /// Get degree matrix from adjacency matrix
    fn to_degree_matrix(&self, py: Python) -> PyResult<Py<Self>> {
        let degree_matrix = self.inner.to_degree_matrix().map_err(|e| {
            PyRuntimeError::new_err(format!("Degree matrix calculation failed: {:?}", e))
        })?;
        let matrix = Self {
            inner: degree_matrix,
        };
        Py::new(py, matrix)
    }

    /// Get normalized Laplacian matrix with enhanced parameterization
    ///
    /// Args:
    ///     eps: Exponent for degree matrix (default 0.5 for standard normalization)
    ///     k: Power to raise the result to (default 1)
    ///
    /// Formula: (D^eps @ A @ D^eps)^k
    fn to_normalized_laplacian(
        &self,
        py: Python,
        eps: Option<f64>,
        k: Option<u32>,
    ) -> PyResult<Py<Self>> {
        let eps_val = eps.unwrap_or(0.5); // Standard normalization uses 0.5
        let k_val = k.unwrap_or(1); // Standard result uses power 1

        let normalized_laplacian =
            self.inner
                .to_normalized_laplacian(eps_val, k_val)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Normalized Laplacian calculation failed: {:?}",
                        e
                    ))
                })?;
        let matrix = Self {
            inner: normalized_laplacian,
        };
        Py::new(py, matrix)
    }

    /// Get standard Laplacian matrix (D - A)
    fn to_laplacian(&self, py: Python) -> PyResult<Py<Self>> {
        let laplacian = self.inner.to_laplacian().map_err(|e| {
            PyRuntimeError::new_err(format!("Laplacian calculation failed: {:?}", e))
        })?;
        let matrix = Self { inner: laplacian };
        Py::new(py, matrix)
    }

    // === ADVANCED RESHAPING OPERATIONS ===

    /// Concatenate matrices along specified axis
    fn concatenate(&self, py: Python, other: &Self, axis: usize) -> PyResult<Py<Self>> {
        let result = self.inner.concatenate(&other.inner, axis).map_err(|e| {
            PyRuntimeError::new_err(format!("Matrix concatenation failed: {:?}", e))
        })?;
        let matrix = Self { inner: result };
        Py::new(py, matrix)
    }

    /// Stack matrices along specified axis
    fn stack(&self, py: Python, other: &Self, axis: usize) -> PyResult<Py<Self>> {
        let result = self
            .inner
            .stack(&other.inner, axis)
            .map_err(|e| PyRuntimeError::new_err(format!("Matrix stacking failed: {:?}", e)))?;
        let matrix = Self { inner: result };
        Py::new(py, matrix)
    }

    /// Split matrix along specified axis
    fn split(&self, py: Python, split_points: Vec<usize>, axis: usize) -> PyResult<Vec<Py<Self>>> {
        let matrices = self
            .inner
            .split(&split_points, axis)
            .map_err(|e| PyRuntimeError::new_err(format!("Matrix split failed: {:?}", e)))?;

        let mut result = Vec::new();
        for matrix in matrices {
            let py_matrix = Self { inner: matrix };
            result.push(Py::new(py, py_matrix)?);
        }

        Ok(result)
    }

    // === ENHANCED NEURAL OPERATIONS ===

    /// Leaky ReLU activation function
    fn leaky_relu(&self, py: Python, alpha: Option<f64>) -> PyResult<Py<Self>> {
        let result = self.inner.leaky_relu(alpha).map_err(|e| {
            PyRuntimeError::new_err(format!("Leaky ReLU activation failed: {:?}", e))
        })?;
        let matrix = Self { inner: result };
        Py::new(py, matrix)
    }

    /// ELU (Exponential Linear Unit) activation function
    fn elu(&self, py: Python, alpha: Option<f64>) -> PyResult<Py<Self>> {
        let result = self
            .inner
            .elu(alpha)
            .map_err(|e| PyRuntimeError::new_err(format!("ELU activation failed: {:?}", e)))?;
        let matrix = Self { inner: result };
        Py::new(py, matrix)
    }

    /// Dropout operation for regularization
    fn dropout(&self, py: Python, p: f64, training: Option<bool>) -> PyResult<Py<Self>> {
        let training_val = training.unwrap_or(true);
        let result = self
            .inner
            .dropout(p, training_val)
            .map_err(|e| PyRuntimeError::new_err(format!("Dropout operation failed: {:?}", e)))?;
        let matrix = Self { inner: result };
        Py::new(py, matrix)
    }

    /// Solve linear system Ax = b
    fn solve(&self, py: Python, b: &Self) -> PyResult<Py<Self>> {
        let result = self
            .inner
            .solve(&b.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("Linear system solve failed: {:?}", e)))?;
        let matrix = Self { inner: result };
        Py::new(py, matrix)
    }

    /// SVD decomposition: A = U * Σ * V^T
    /// Returns tuple of (U, singular_values, V_transpose)
    fn svd(&self, py: Python) -> PyResult<(Py<Self>, PyObject, Py<Self>)> {
        let (u_matrix, singular_values, vt_matrix) = self
            .inner
            .svd()
            .map_err(|e| PyRuntimeError::new_err(format!("SVD decomposition failed: {:?}", e)))?;

        let u_py = PyGraphMatrix { inner: u_matrix };
        let vt_py = PyGraphMatrix { inner: vt_matrix };

        // Convert singular values to Python list
        let singular_values_py = pyo3::types::PyList::new(py, singular_values);

        Ok((
            Py::new(py, u_py)?,
            singular_values_py.into(),
            Py::new(py, vt_py)?,
        ))
    }

    /// QR decomposition: A = Q * R
    /// Returns tuple of (Q, R) where Q is orthogonal and R is upper triangular
    fn qr_decomposition(&self, py: Python) -> PyResult<(Py<Self>, Py<Self>)> {
        let (q_matrix, r_matrix) = self
            .inner
            .qr_decomposition()
            .map_err(|e| PyRuntimeError::new_err(format!("QR decomposition failed: {:?}", e)))?;

        let q_py = PyGraphMatrix { inner: q_matrix };
        let r_py = PyGraphMatrix { inner: r_matrix };

        Ok((Py::new(py, q_py)?, Py::new(py, r_py)?))
    }

    /// LU decomposition: PA = LU
    /// Returns tuple of (P, L, U) where P is permutation, L is lower triangular, U is upper triangular
    fn lu_decomposition(&self, py: Python) -> PyResult<(Py<Self>, Py<Self>, Py<Self>)> {
        let (p_matrix, l_matrix, u_matrix) = self
            .inner
            .lu_decomposition()
            .map_err(|e| PyRuntimeError::new_err(format!("LU decomposition failed: {:?}", e)))?;

        let p_py = PyGraphMatrix { inner: p_matrix };
        let l_py = PyGraphMatrix { inner: l_matrix };
        let u_py = PyGraphMatrix { inner: u_matrix };

        Ok((Py::new(py, p_py)?, Py::new(py, l_py)?, Py::new(py, u_py)?))
    }

    /// Cholesky decomposition: A = L * L^T
    /// Returns L (lower triangular) for positive definite matrices
    fn cholesky_decomposition(&self, py: Python) -> PyResult<Py<Self>> {
        let l_matrix = self.inner.cholesky_decomposition().map_err(|e| {
            PyRuntimeError::new_err(format!("Cholesky decomposition failed: {:?}", e))
        })?;

        Py::new(py, PyGraphMatrix { inner: l_matrix })
    }

    /// Eigenvalue decomposition: A * V = V * Λ
    /// Returns tuple of (eigenvalues, eigenvectors)
    fn eigenvalue_decomposition(&self, py: Python) -> PyResult<(PyObject, Py<Self>)> {
        let (eigenvalues, eigenvector_matrix) =
            self.inner.eigenvalue_decomposition().map_err(|e| {
                PyRuntimeError::new_err(format!("Eigenvalue decomposition failed: {:?}", e))
            })?;

        // Convert eigenvalues to Python list
        let eigenval_list: Vec<f64> = eigenvalues;
        let py_eigenvals = eigenval_list.to_object(py);

        let py_eigenvecs = PyGraphMatrix {
            inner: eigenvector_matrix,
        };

        Ok((py_eigenvals, Py::new(py, py_eigenvecs)?))
    }

    /// Matrix rank - number of linearly independent rows/columns
    /// Uses SVD with numerical tolerance for near-zero singular values
    fn rank(&self) -> PyResult<usize> {
        self.inner.rank().map_err(|e| {
            PyRuntimeError::new_err(format!("Matrix rank computation failed: {:?}", e))
        })
    }

    /// Tile (repeat) the matrix a specified number of times along each axis
    /// Args: reps - tuple of (rows_repeat, cols_repeat)
    fn tile(&self, py: Python, reps: (usize, usize)) -> PyResult<Py<Self>> {
        let result = self
            .inner
            .tile(reps)
            .map_err(|e| PyRuntimeError::new_err(format!("Matrix tiling failed: {:?}", e)))?;

        Py::new(py, PyGraphMatrix { inner: result })
    }

    /// Repeat elements of the matrix along a specified axis
    /// Args: repeats - number of times to repeat, axis - 0 for rows, 1 for columns
    fn repeat(&self, py: Python, repeats: usize, axis: usize) -> PyResult<Py<Self>> {
        let result = self
            .inner
            .repeat(repeats, axis)
            .map_err(|e| PyRuntimeError::new_err(format!("Matrix repeat failed: {:?}", e)))?;

        Py::new(py, PyGraphMatrix { inner: result })
    }

    /// Element-wise absolute value
    fn abs(&self, py: Python) -> PyResult<Py<Self>> {
        let result = self
            .inner
            .abs()
            .map_err(|e| PyRuntimeError::new_err(format!("Element-wise abs failed: {:?}", e)))?;

        Py::new(py, PyGraphMatrix { inner: result })
    }

    /// Element-wise exponential (e^x)
    fn exp(&self, py: Python) -> PyResult<Py<Self>> {
        let result = self
            .inner
            .exp()
            .map_err(|e| PyRuntimeError::new_err(format!("Element-wise exp failed: {:?}", e)))?;

        Py::new(py, PyGraphMatrix { inner: result })
    }

    /// Element-wise natural logarithm
    fn log(&self, py: Python) -> PyResult<Py<Self>> {
        let result = self
            .inner
            .log()
            .map_err(|e| PyRuntimeError::new_err(format!("Element-wise log failed: {:?}", e)))?;

        Py::new(py, PyGraphMatrix { inner: result })
    }

    /// Element-wise square root
    fn sqrt(&self, py: Python) -> PyResult<Py<Self>> {
        let result = self
            .inner
            .sqrt()
            .map_err(|e| PyRuntimeError::new_err(format!("Element-wise sqrt failed: {:?}", e)))?;

        Py::new(py, PyGraphMatrix { inner: result })
    }

    // === CONVERSION OPERATIONS ===

    /// Convert matrix to nested Python list
    /// Returns: [[row1], [row2], ...] format
    fn to_list(&self, py: Python) -> PyResult<PyObject> {
        let (rows, cols) = self.inner.shape();
        let mut result_rows = Vec::new();

        for row in 0..rows {
            let mut row_values = Vec::new();
            for col in 0..cols {
                let value = self.inner.get(row, col).unwrap_or(0.0);
                row_values.push(value);
            }
            result_rows.push(row_values);
        }

        Ok(result_rows.to_object(py))
    }

    /// Convert matrix to Python dictionary
    /// Returns: {"data": [[...]], "shape": [rows, cols], "dtype": "float64"}
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let (rows, cols) = self.inner.shape();
        let dict = pyo3::types::PyDict::new(py);

        // Add matrix data as nested list
        let data_list = self.to_list(py)?;
        dict.set_item("data", data_list)?;

        // Add shape information
        dict.set_item("shape", (rows, cols))?;

        // Add dtype information
        dict.set_item("dtype", "float64")?;

        Ok(dict.into())
    }

    // === FUNCTIONAL OPERATIONS ===

    /// Apply a Python function to each element of the matrix
    /// Returns: New matrix with function applied element-wise
    fn apply(&self, py: Python, func: PyObject) -> PyResult<Py<PyGraphMatrix>> {
        let (rows, cols) = self.inner.shape();
        let mut result_arrays = Vec::new();

        for col in 0..cols {
            if let Some(column) = self.inner.get_column(col) {
                let mut new_values = Vec::new();
                for value in column.iter() {
                    // Call Python function on each value
                    let py_value = value.to_object(py);
                    let result = func.call1(py, (py_value,))?;
                    let new_value: f64 = result.extract(py)?;
                    new_values.push(new_value);
                }
                result_arrays.push(groggy::storage::array::NumArray::new(new_values));
            } else {
                return Err(PyRuntimeError::new_err("Failed to get matrix column"));
            }
        }

        let result_matrix = groggy::storage::GraphMatrix::from_arrays(result_arrays)
            .map_err(|e| PyRuntimeError::new_err(format!("Apply operation failed: {:?}", e)))?;

        Py::new(
            py,
            PyGraphMatrix {
                inner: result_matrix,
            },
        )
    }

    /// Map a Python function over matrix elements (alias for apply)
    /// Returns: New matrix with function mapped over elements
    fn map(&self, py: Python, func: PyObject) -> PyResult<Py<PyGraphMatrix>> {
        self.apply(py, func)
    }

    /// Filter matrix elements based on a condition function
    /// Returns: Matrix with only elements where condition(element) is True, others set to 0
    fn filter(&self, py: Python, condition: PyObject) -> PyResult<Py<PyGraphMatrix>> {
        let (rows, cols) = self.inner.shape();
        let mut result_arrays = Vec::new();

        for col in 0..cols {
            if let Some(column) = self.inner.get_column(col) {
                let mut filtered_values = Vec::new();
                for value in column.iter() {
                    // Test condition on each value
                    let py_value = value.to_object(py);
                    let condition_result = condition.call1(py, (py_value,))?;
                    let passes: bool = condition_result.extract(py)?;

                    // Keep value if condition is true, otherwise set to 0
                    filtered_values.push(if passes { *value } else { 0.0 });
                }
                result_arrays.push(groggy::storage::array::NumArray::new(filtered_values));
            } else {
                return Err(PyRuntimeError::new_err("Failed to get matrix column"));
            }
        }

        let result_matrix = groggy::storage::GraphMatrix::from_arrays(result_arrays)
            .map_err(|e| PyRuntimeError::new_err(format!("Filter operation failed: {:?}", e)))?;

        Py::new(
            py,
            PyGraphMatrix {
                inner: result_matrix,
            },
        )
    }

    // === ITERATION SUPPORT ===
    // Iterator is now implemented above in the main impl block
}

/// Python iterator for matrix rows
#[pyclass]
struct PyMatrixRowIterator {
    matrix: Py<PyGraphMatrix>,
    current_row: usize,
    total_rows: usize,
}

#[pymethods]
impl PyMatrixRowIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        if self.current_row >= self.total_rows {
            return Ok(None);
        }

        let matrix = self.matrix.borrow(py);
        let row_data = matrix.get_row(py, self.current_row)?;
        self.current_row += 1;

        Ok(Some(row_data))
    }
}

// Implement display data conversion for PyGraphMatrix
impl PyGraphMatrix {
    /// Convert matrix to display data format expected by Rust formatter
    /// Future feature
    #[allow(dead_code)]
    fn to_display_data(&self) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();

        // Get matrix dimensions
        let (total_rows, total_cols) = self.inner.shape();
        let sample_rows = std::cmp::min(5, total_rows);
        let sample_cols = std::cmp::min(5, total_cols);

        // Table shape (show total dimensions, not just sample)
        data.insert(
            "shape".to_string(),
            serde_json::Value::Array(vec![
                serde_json::Value::Number(serde_json::Number::from(total_rows)),
                serde_json::Value::Number(serde_json::Number::from(total_cols)),
            ]),
        );

        // Generate column names (col_0, col_1, etc.)
        let mut column_names = Vec::new();
        for j in 0..sample_cols {
            column_names.push(serde_json::Value::String(format!("col_{}", j)));
        }
        data.insert(
            "columns".to_string(),
            serde_json::Value::Array(column_names),
        );

        // Create rows data
        let mut table_rows = Vec::new();
        for i in 0..sample_rows {
            let mut row_data = Vec::new();
            for j in 0..sample_cols {
                if let Some(value) = self.inner.get(i, j) {
                    let json_value = serde_json::Value::Number(
                        serde_json::Number::from_f64(value).unwrap_or(serde_json::Number::from(0)),
                    );
                    row_data.push(json_value);
                } else {
                    row_data.push(serde_json::Value::Null);
                }
            }
            table_rows.push(serde_json::Value::Array(row_data));
        }
        data.insert("data".to_string(), serde_json::Value::Array(table_rows));

        data
    }
}

impl PyGraphMatrix {
    /// Create PyGraphMatrix from core GraphMatrix
    pub fn from_graph_matrix(matrix: GraphMatrix) -> Self {
        Self { inner: matrix }
    }

    /// Convert matrix to table representation (placeholder implementation)
    pub fn to_table(&self, _py: Python) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        // TODO: Implement proper matrix-to-table conversion
        // This would typically convert a matrix to a tabular format
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Matrix to table conversion not yet implemented.",
        ))
    }
}
