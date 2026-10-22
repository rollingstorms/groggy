//! Graph Matrix Operations - Internal Helper Class
//!
//! PyGraphMatrix helper class that handles all graph matrix operations.

use crate::ffi::storage::matrix::PyGraphMatrix;
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
    /// Get Laplacian matrix - PURE DELEGATION to core
    pub fn laplacian_matrix(
        &mut self,
        py: Python,
        normalized: Option<bool>,
    ) -> PyResult<Py<PyGraphMatrix>> {
        use crate::ffi::storage::matrix::PyGraphMatrix;

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

        Py::new(py, PyGraphMatrix { inner: matrix })
    }

    /// Generate transition matrix through the native graph transform.
    pub fn transition_matrix(&mut self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        let inner = self
            .graph
            .borrow(py)
            .inner
            .borrow_mut()
            .transition_matrix()
            .map_err(graph_error_to_py_err)?;
        Py::new(py, PyGraphMatrix { inner })
    }

    /// Convert graph to attribute matrix - PURE DELEGATION to core
    pub fn to_matrix(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        use crate::ffi::storage::matrix::PyGraphMatrix;

        // DELEGATION: Use core to_matrix implementation (graph.rs:2383)
        let graph_matrix = {
            let graph_ref = self.graph.borrow(py);
            let result = graph_ref
                .inner
                .borrow()
                .to_matrix_f64()
                .map_err(graph_error_to_py_err);
            drop(graph_ref);
            result
        }?;

        Py::new(
            py,
            PyGraphMatrix {
                inner: graph_matrix,
            },
        )
    }
}
