//! Graph Matrix Operations - Internal Helper Class
//!
//! PyGraphMatrix helper class that handles all graph matrix operations.

use crate::ffi::core::matrix::PyGraphMatrix;
use crate::ffi::utils::graph_error_to_py_err;
use pyo3::prelude::*;

use super::graph::PyGraph;

/// Internal helper for graph matrix operations (not exposed to Python)
pub struct PyGraphMatrixHelper {
    pub graph: Py<PyGraph>,
}

impl PyGraphMatrixHelper {
    /// Create new PyGraphMatrixHelper instance
    pub fn new(graph: Py<PyGraph>) -> PyResult<PyGraphMatrixHelper> {
        Ok(PyGraphMatrixHelper { graph })
    }
    /// Get adjacency matrix - PURE DELEGATION to core
    pub fn adjacency_matrix(&mut self, py: Python) -> PyResult<PyObject> {
        // DELEGATION: Use core adjacency_matrix implementation (graph.rs:1783)
        let matrix = {
            let graph_ref = self.graph.borrow_mut(py);
            let result = graph_ref
                .inner
                .borrow_mut()
                .adjacency_matrix()
                .map_err(graph_error_to_py_err);
            drop(graph_ref);
            result
        }?;

        // Convert AdjacencyMatrix to Python object
        let graph_ref = self.graph.borrow(py);
        graph_ref.adjacency_matrix_to_py_object(py, matrix)
    }

    /// Simple adjacency matrix (alias) - PURE DELEGATION to core
    pub fn adjacency(&mut self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        use crate::ffi::core::matrix::PyGraphMatrix;

        // DELEGATION: Use core adjacency implementation (just added - alias for adjacency_matrix)
        let matrix = {
            let graph_ref = self.graph.borrow_mut(py);
            let result = graph_ref
                .inner
                .borrow_mut()
                .adjacency()
                .map_err(graph_error_to_py_err);
            drop(graph_ref);
            result
        }?;

        // Convert AdjacencyMatrix to GraphMatrix
        let graph_ref = self.graph.borrow(py);
        let graph_matrix = graph_ref.adjacency_matrix_to_graph_matrix(matrix)?;
        Py::new(
            py,
            PyGraphMatrix {
                inner: graph_matrix,
            },
        )
    }

    /// Get weighted adjacency matrix - PURE DELEGATION to core
    pub fn weighted_adjacency_matrix(
        &mut self,
        py: Python,
        weight_attr: &str,
    ) -> PyResult<Py<PyGraphMatrix>> {
        use crate::ffi::core::matrix::PyGraphMatrix;

        // DELEGATION: Use core weighted_adjacency_matrix implementation (graph.rs:1788)
        let matrix = {
            let graph_ref = self.graph.borrow_mut(py);
            let result = graph_ref
                .inner
                .borrow_mut()
                .weighted_adjacency_matrix(weight_attr)
                .map_err(graph_error_to_py_err);
            drop(graph_ref);
            result
        }?;

        // Convert AdjacencyMatrix to GraphMatrix
        let graph_ref = self.graph.borrow(py);
        let graph_matrix = graph_ref.adjacency_matrix_to_graph_matrix(matrix)?;
        Py::new(
            py,
            PyGraphMatrix {
                inner: graph_matrix,
            },
        )
    }

    /// Get dense adjacency matrix - PURE DELEGATION to core
    pub fn dense_adjacency_matrix(&mut self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        use crate::ffi::core::matrix::PyGraphMatrix;

        // DELEGATION: Use core dense_adjacency_matrix implementation (graph.rs:1797)
        let matrix = {
            let graph_ref = self.graph.borrow_mut(py);
            let result = graph_ref
                .inner
                .borrow_mut()
                .dense_adjacency_matrix()
                .map_err(graph_error_to_py_err);
            drop(graph_ref);
            result
        }?;

        // Convert AdjacencyMatrix to GraphMatrix
        let graph_ref = self.graph.borrow(py);
        let graph_matrix = graph_ref.adjacency_matrix_to_graph_matrix(matrix)?;
        Py::new(
            py,
            PyGraphMatrix {
                inner: graph_matrix,
            },
        )
    }

    /// Get sparse adjacency matrix - PURE DELEGATION to core
    pub fn sparse_adjacency_matrix(&mut self, py: Python) -> PyResult<PyObject> {
        // DELEGATION: Use core sparse_adjacency_matrix implementation (graph.rs:1804)
        let matrix = {
            let graph_ref = self.graph.borrow_mut(py);
            let result = graph_ref
                .inner
                .borrow_mut()
                .sparse_adjacency_matrix()
                .map_err(graph_error_to_py_err);
            drop(graph_ref);
            result
        }?;

        // Convert AdjacencyMatrix to Python object (sparse format)
        let graph_ref = self.graph.borrow(py);
        graph_ref.adjacency_matrix_to_py_object(py, matrix)
    }

    /// Get Laplacian matrix - PURE DELEGATION to core
    pub fn laplacian_matrix(
        &mut self,
        py: Python,
        normalized: Option<bool>,
    ) -> PyResult<Py<PyGraphMatrix>> {
        use crate::ffi::core::matrix::PyGraphMatrix;

        let is_normalized = normalized.unwrap_or(false);

        // DELEGATION: Use core laplacian_matrix implementation (graph.rs:1811)
        let matrix = {
            let graph_ref = self.graph.borrow_mut(py);
            let result = graph_ref
                .inner
                .borrow_mut()
                .laplacian_matrix(is_normalized)
                .map_err(graph_error_to_py_err);
            drop(graph_ref);
            result
        }?;

        // Convert AdjacencyMatrix to GraphMatrix
        let graph_ref = self.graph.borrow(py);
        let graph_matrix = graph_ref.adjacency_matrix_to_graph_matrix(matrix)?;
        Py::new(
            py,
            PyGraphMatrix {
                inner: graph_matrix,
            },
        )
    }

    /// Generate transition matrix - MISSING FROM CORE (needs implementation)
    pub fn transition_matrix(&mut self, _py: Python) -> PyResult<Py<PyGraphMatrix>> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "transition_matrix needs to be implemented in core first",
        ))
    }
}
