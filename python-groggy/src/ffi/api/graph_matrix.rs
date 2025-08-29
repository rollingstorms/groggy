//! Graph Matrix Operations - Pure FFI Delegation Layer
//!
//! All matrix operations that delegate to core implementations.

use crate::ffi::core::matrix::PyGraphMatrix;
use crate::ffi::utils::graph_error_to_py_err;
use groggy::AttrName;
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use super::graph::PyGraph;

#[pymethods]
impl PyGraph {
    /// Get adjacency matrix - PURE DELEGATION to core
    fn adjacency_matrix(&mut self, py: Python) -> PyResult<PyObject> {
        // DELEGATION: Use core adjacency_matrix implementation (graph.rs:1783)
        let matrix = py.allow_threads(|| {
            self.inner
                .borrow_mut()
                .adjacency_matrix()
                .map_err(graph_error_to_py_err)
        })?;
        
        // Convert AdjacencyMatrix to Python object
        self.adjacency_matrix_to_py_object(py, matrix)
    }

    /// Generate adjacency matrix for the entire graph (cleaner API)
    /// Returns: GraphMatrix with multi-index access (matrix[0, 1])
    /// This is a cleaner alias for adjacency_matrix() but always returns dense
    fn adjacency(&mut self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        use crate::ffi::core::matrix::PyGraphMatrix;

        // Generate dense adjacency matrix using the public API method
        let adjacency_matrix = self.inner.borrow_mut().dense_adjacency_matrix().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to generate adjacency matrix: {:?}", e))
        })?;

        // Convert AdjacencyMatrix to GraphMatrix
        let graph_matrix = self.adjacency_matrix_to_graph_matrix(adjacency_matrix)?;

        // Wrap in PyGraphMatrix with cleaner metadata
        let py_graph_matrix = PyGraphMatrix::from_graph_matrix(graph_matrix);
        Ok(Py::new(py, py_graph_matrix)?)
    }

    /// Get weighted adjacency matrix - PURE DELEGATION to core
    fn weighted_adjacency_matrix(
        &mut self,
        py: Python,
        weight_attr: &str,
    ) -> PyResult<Py<PyGraphMatrix>> {
        use crate::ffi::core::matrix::PyGraphMatrix;

        // DELEGATION: Use core weighted_adjacency_matrix implementation (graph.rs:1788)
        let matrix = py.allow_threads(|| {
            self.inner
                .borrow_mut()
                .weighted_adjacency_matrix(weight_attr)
                .map_err(graph_error_to_py_err)
        })?;

        // Convert AdjacencyMatrix to GraphMatrix
        let graph_matrix = self.adjacency_matrix_to_graph_matrix(matrix)?;
        Ok(Py::new(py, PyGraphMatrix { inner: graph_matrix })?)
    }

    /// Get dense adjacency matrix - PURE DELEGATION to core
    fn dense_adjacency_matrix(&mut self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        use crate::ffi::core::matrix::PyGraphMatrix;
        
        // DELEGATION: Use core dense_adjacency_matrix implementation (graph.rs:1797)
        let matrix = py.allow_threads(|| {
            self.inner
                .borrow_mut()
                .dense_adjacency_matrix()
                .map_err(graph_error_to_py_err)
        })?;
        
        // Convert AdjacencyMatrix to GraphMatrix
        let graph_matrix = self.adjacency_matrix_to_graph_matrix(matrix)?;
        Ok(Py::new(py, PyGraphMatrix { inner: graph_matrix })?)
    }

    /// Get sparse adjacency matrix - PURE DELEGATION to core
    fn sparse_adjacency_matrix(&mut self, py: Python) -> PyResult<PyObject> {
        // DELEGATION: Use core sparse_adjacency_matrix implementation (graph.rs:1804)
        let matrix = py.allow_threads(|| {
            self.inner
                .borrow_mut()
                .sparse_adjacency_matrix()
                .map_err(graph_error_to_py_err)
        })?;
        
        // Convert AdjacencyMatrix to Python object (sparse format)
        self.adjacency_matrix_to_py_object(py, matrix)
    }

    /// Get Laplacian matrix - PURE DELEGATION to core
    fn laplacian_matrix(
        &mut self,
        py: Python,
        normalized: Option<bool>,
    ) -> PyResult<Py<PyGraphMatrix>> {
        use crate::ffi::core::matrix::PyGraphMatrix;

        let is_normalized = normalized.unwrap_or(false);

        // DELEGATION: Use core laplacian_matrix implementation (graph.rs:1811)
        let matrix = py.allow_threads(|| {
            self.inner
                .borrow_mut()
                .laplacian_matrix(is_normalized)
                .map_err(graph_error_to_py_err)
        })?;

        // Convert AdjacencyMatrix to GraphMatrix
        let graph_matrix = self.adjacency_matrix_to_graph_matrix(matrix)?;
        Ok(Py::new(py, PyGraphMatrix { inner: graph_matrix })?)
    }

    /// Generate transition matrix - MISSING FROM CORE (needs implementation)
    fn transition_matrix(&mut self, _py: Python) -> PyResult<Py<PyGraphMatrix>> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "transition_matrix needs to be implemented in core first",
        ))
    }
}