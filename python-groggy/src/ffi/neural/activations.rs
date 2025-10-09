//! Neural Network Activation Functions
//!
//! FFI bindings for common activation functions used in neural networks.

use crate::ffi::storage::matrix::PyGraphMatrix;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;

/// ReLU activation function
/// Implements: nn.relu(matrix) -> matrix
#[pyfunction]
pub fn relu(py: Python, matrix: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
    // For now, implement ReLU element-wise using existing matrix operations
    let (_rows, cols) = matrix.inner.shape();

    // Create arrays for the result
    let mut result_arrays = Vec::new();

    for col in 0..cols {
        if let Some(column) = matrix.inner.get_column(col) {
            let mut new_values = Vec::new();
            for value in column.iter() {
                // ReLU: max(0, x)
                new_values.push(value.max(0.0));
            }
            result_arrays.push(groggy::storage::array::NumArray::new(new_values));
        } else {
            return Err(PyRuntimeError::new_err("Failed to get matrix column"));
        }
    }

    let result_matrix = groggy::storage::GraphMatrix::from_arrays(result_arrays)
        .map_err(|e| PyRuntimeError::new_err(format!("ReLU operation failed: {:?}", e)))?;

    let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
    Py::new(py, py_result)
}

/// Sigmoid activation function  
/// Implements: nn.sigmoid(matrix) -> matrix
#[pyfunction]
pub fn sigmoid(py: Python, matrix: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
    let (_rows, cols) = matrix.inner.shape();

    let mut result_arrays = Vec::new();

    for col in 0..cols {
        if let Some(column) = matrix.inner.get_column(col) {
            let mut new_values = Vec::new();
            for value in column.iter() {
                // Sigmoid: 1 / (1 + e^(-x))
                let sigmoid_val = 1.0 / (1.0 + (-value).exp());
                new_values.push(sigmoid_val);
            }
            result_arrays.push(groggy::storage::array::NumArray::new(new_values));
        } else {
            return Err(PyRuntimeError::new_err("Failed to get matrix column"));
        }
    }

    let result_matrix = groggy::storage::GraphMatrix::from_arrays(result_arrays)
        .map_err(|e| PyRuntimeError::new_err(format!("Sigmoid operation failed: {:?}", e)))?;

    let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
    Py::new(py, py_result)
}

/// Hyperbolic tangent activation function
/// Implements: nn.tanh(matrix) -> matrix  
#[pyfunction]
pub fn tanh(py: Python, matrix: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
    let (_rows, cols) = matrix.inner.shape();

    let mut result_arrays = Vec::new();

    for col in 0..cols {
        if let Some(column) = matrix.inner.get_column(col) {
            let mut new_values = Vec::new();
            for value in column.iter() {
                // Tanh: (e^x - e^(-x)) / (e^x + e^(-x))
                new_values.push(value.tanh());
            }
            result_arrays.push(groggy::storage::array::NumArray::new(new_values));
        } else {
            return Err(PyRuntimeError::new_err("Failed to get matrix column"));
        }
    }

    let result_matrix = groggy::storage::GraphMatrix::from_arrays(result_arrays)
        .map_err(|e| PyRuntimeError::new_err(format!("Tanh operation failed: {:?}", e)))?;

    let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
    Py::new(py, py_result)
}

/// Softmax activation function (applied row-wise)
/// Implements: nn.softmax(matrix, dim=1) -> matrix
#[pyfunction]
#[pyo3(signature = (matrix, dim=1))]
pub fn softmax(py: Python, matrix: &PyGraphMatrix, dim: i32) -> PyResult<Py<PyGraphMatrix>> {
    let (rows, cols) = matrix.inner.shape();

    if dim != 1 {
        return Err(PyTypeError::new_err(
            "Currently only dim=1 (row-wise) softmax is supported",
        ));
    }

    // Apply softmax row-wise
    let mut result_arrays = Vec::with_capacity(cols);
    for _ in 0..cols {
        result_arrays.push(Vec::with_capacity(rows));
    }

    for row in 0..rows {
        // Get row values
        let mut row_values = Vec::new();
        for col in 0..cols {
            if let Some(val) = matrix.inner.get(row, col) {
                row_values.push(val);
            } else {
                row_values.push(0.0);
            }
        }

        // Compute softmax for this row
        let max_val = row_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f64> = row_values.iter().map(|x| (x - max_val).exp()).collect();
        let sum_exp: f64 = exp_values.iter().sum();

        // Store softmax values back to columns
        for col in 0..cols {
            let softmax_val = exp_values[col] / sum_exp;
            result_arrays[col].push(softmax_val);
        }
    }

    // Create NumArrays from the result
    let num_arrays: Vec<groggy::storage::array::NumArray<f64>> = result_arrays
        .into_iter()
        .map(groggy::storage::array::NumArray::new)
        .collect();

    let result_matrix = groggy::storage::GraphMatrix::from_arrays(num_arrays)
        .map_err(|e| PyRuntimeError::new_err(format!("Softmax operation failed: {:?}", e)))?;

    let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
    Py::new(py, py_result)
}

/// GELU (Gaussian Error Linear Unit) activation function
/// Implements: nn.gelu(matrix) -> matrix
#[pyfunction]
pub fn gelu(py: Python, matrix: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
    let (_rows, cols) = matrix.inner.shape();

    let mut result_arrays = Vec::new();

    for col in 0..cols {
        if let Some(column) = matrix.inner.get_column(col) {
            let mut new_values = Vec::new();
            for value in column.iter() {
                // GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                let x = *value;
                let x_cubed = x * x * x;
                let inner = std::f64::consts::FRAC_2_PI.sqrt() * (x + 0.044715 * x_cubed);
                let gelu_val = 0.5 * x * (1.0 + inner.tanh());
                new_values.push(gelu_val);
            }
            result_arrays.push(groggy::storage::array::NumArray::new(new_values));
        } else {
            return Err(PyRuntimeError::new_err("Failed to get matrix column"));
        }
    }

    let result_matrix = groggy::storage::GraphMatrix::from_arrays(result_arrays)
        .map_err(|e| PyRuntimeError::new_err(format!("GELU operation failed: {:?}", e)))?;

    let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
    Py::new(py, py_result)
}
