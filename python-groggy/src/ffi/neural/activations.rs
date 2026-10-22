//! Thin Python activation adapters over the checked native matrix API.
use crate::ffi::storage::matrix::{computation, PyGraphMatrix};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

/// Native relu activation with supported tape ownership.
#[pyfunction]
pub fn relu(py: Python, matrix: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
    let inner = computation::run(py, [&matrix.inner], |[input]| input.relu())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Py::new(py, PyGraphMatrix::from_graph_matrix(inner))
}

/// Native sigmoid activation with supported tape ownership.
#[pyfunction]
pub fn sigmoid(py: Python, matrix: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
    let inner = computation::run(py, [&matrix.inner], |[input]| input.sigmoid())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Py::new(py, PyGraphMatrix::from_graph_matrix(inner))
}

/// Native tanh activation with supported tape ownership.
#[pyfunction]
pub fn tanh(py: Python, matrix: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
    let inner = computation::run(py, [&matrix.inner], |[input]| input.tanh())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Py::new(py, PyGraphMatrix::from_graph_matrix(inner))
}

/// Native gelu activation with supported tape ownership.
#[pyfunction]
pub fn gelu(py: Python, matrix: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
    let inner = computation::run(py, [&matrix.inner], |[input]| input.gelu())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Py::new(py, PyGraphMatrix::from_graph_matrix(inner))
}

/// Native row softmax; preserve the established dim=1 Python interface.
#[pyfunction]
#[pyo3(signature = (matrix, dim=1))]
pub fn softmax(py: Python, matrix: &PyGraphMatrix, dim: i32) -> PyResult<Py<PyGraphMatrix>> {
    let inner = computation::run(py, [&matrix.inner], move |[input]| input.softmax(dim))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Py::new(py, PyGraphMatrix::from_graph_matrix(inner))
}
