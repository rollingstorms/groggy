//! Python wrapper for regular Edge entities
//!
//! This module provides the PyEdge class that wraps our Rust Edge entity,
//! exposing all GraphEntity and EdgeOperations methods to Python.

use groggy::entities::Edge;
use groggy::types::{EdgeId, NodeId};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Python wrapper for a regular graph edge
///
/// Regular edges are the basic connections between nodes in graphs. They provide access
/// to edge attributes, topology information (source/target), and graph operations.
#[pyclass(name = "Edge", unsendable)]
pub struct PyEdge {
    /// The underlying Rust Edge entity
    pub inner: Edge,
}

impl PyEdge {
    /// Create a new PyEdge wrapper
    ///
    /// # Arguments
    /// * `edge` - The Rust Edge entity to wrap
    ///
    /// # Returns
    /// A new PyEdge instance
    pub fn from_edge(edge: Edge) -> Self {
        Self { inner: edge }
    }
}

#[pymethods]
impl PyEdge {
    /// Get the edge ID
    ///
    /// # Returns
    /// The unique EdgeId for this edge
    #[getter]
    fn id(&self) -> EdgeId {
        self.inner.id()
    }

    /// Get edge attribute value
    ///
    /// # Arguments
    /// * `key` - Attribute name to retrieve
    ///
    /// # Returns
    /// The attribute value if it exists
    ///
    /// # Raises
    /// * `KeyError` - If the attribute doesn't exist
    /// * `RuntimeError` - If there's an error accessing the attribute
    fn __getitem__(&self, py: Python, key: &str) -> PyResult<Py<PyAny>> {
        use groggy::traits::GraphEntity;

        match self.inner.get_attribute(&key.into()) {
            Ok(Some(attr_value)) => crate::ffi::utils::attr_value_to_python_value(py, &attr_value),
            Ok(None) => Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Edge {} has no attribute '{}'",
                self.inner.id(),
                key
            ))),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get edge attribute: {}",
                e
            ))),
        }
    }

    /// Set edge attribute value
    ///
    /// # Arguments
    /// * `key` - Attribute name to set
    /// * `value` - Attribute value to store
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error setting the attribute
    fn __setitem__(&self, key: &str, value: &PyAny) -> PyResult<()> {
        use groggy::traits::GraphEntity;

        // Convert PyAny to PyAttrValue
        let py_attr_value = crate::ffi::types::PyAttrValue::from_py_value(value)?;
        let attr_value = py_attr_value.to_attr_value();
        self.inner
            .set_attribute(key.into(), attr_value)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set edge attribute: {}", e)))?;
        Ok(())
    }

    /// Check if attribute exists on this edge
    ///
    /// # Arguments
    /// * `key` - Attribute name to check
    ///
    /// # Returns
    /// True if the attribute exists, False otherwise
    fn __contains__(&self, key: &str) -> PyResult<bool> {
        use groggy::traits::GraphEntity;

        match self.inner.get_attribute(&key.into()) {
            Ok(Some(_)) => Ok(true),
            Ok(None) => Ok(false),
            Err(_) => Ok(false),
        }
    }

    /// Get edge attribute via attribute access (edge.relationship)
    ///
    /// This enables dot notation access to edge attributes:
    /// edge.relationship instead of edge['relationship']
    ///
    /// # Arguments
    /// * `name` - Attribute name to retrieve
    ///
    /// # Returns
    /// The attribute value if it exists
    ///
    /// # Raises
    /// * `AttributeError` - If the attribute doesn't exist
    fn __getattr__(&self, py: Python, name: &str) -> PyResult<Py<PyAny>> {
        use groggy::traits::GraphEntity;

        match self.inner.get_attribute(&name.into()) {
            Ok(Some(attr_value)) => crate::ffi::utils::attr_value_to_python_value(py, &attr_value),
            Ok(None) => Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "'Edge' object has no attribute '{}'",
                name
            ))),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get edge attribute: {}",
                e
            ))),
        }
    }

    /// Get the edge's source node ID
    ///
    /// # Returns
    /// The NodeId of the source node
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error getting the source
    #[getter]
    fn source(&self) -> PyResult<NodeId> {
        use groggy::traits::EdgeOperations;

        self.inner
            .source()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get edge source: {}", e)))
    }

    /// Get the edge's target node ID
    ///
    /// # Returns
    /// The NodeId of the target node
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error getting the target
    #[getter]
    fn target(&self) -> PyResult<NodeId> {
        use groggy::traits::EdgeOperations;

        self.inner
            .target()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get edge target: {}", e)))
    }

    /// Get all attribute keys for this edge
    ///
    /// # Returns
    /// List of attribute names
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error accessing attributes
    fn keys(&self) -> PyResult<Vec<String>> {
        use groggy::traits::EdgeOperations;

        match self.inner.edge_attributes() {
            Ok(attrs) => Ok(attrs.keys().cloned().collect()),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get edge attributes: {}",
                e
            ))),
        }
    }

    /// Get all attribute values for this edge
    ///
    /// # Arguments
    /// * `py` - Python interpreter instance
    ///
    /// # Returns
    /// List of attribute values
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error accessing attributes
    fn values(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        use groggy::traits::EdgeOperations;

        let edge_attrs = self.inner.edge_attributes().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get edge attributes: {}", e))
        })?;

        let values = edge_attrs
            .values()
            .map(|value| crate::ffi::utils::attr_value_to_python_value(py, value))
            .collect::<PyResult<Vec<_>>>()?;

        Ok(values)
    }

    /// Get the entity type for this edge
    ///
    /// # Returns
    /// The string "edge"
    #[getter]
    fn entity_type(&self) -> &'static str {
        use groggy::traits::GraphEntity;
        self.inner.entity_type()
    }

    /// Check if this edge is currently active
    ///
    /// # Returns
    /// True if the edge is active in the graph
    #[getter]
    fn is_active(&self) -> bool {
        use groggy::traits::GraphEntity;
        self.inner.is_active()
    }

    /// Get a summary of this edge
    ///
    /// # Returns
    /// A human-readable summary string
    fn summary(&self) -> String {
        use groggy::traits::GraphEntity;
        self.inner.summary()
    }

    /// String representation of the edge
    ///
    /// # Returns
    /// A string representation showing the edge ID and key attributes
    fn __str__(&self) -> String {
        self.summary()
    }

    /// Detailed string representation of the edge
    ///
    /// # Returns
    /// A detailed string representation
    fn __repr__(&self) -> String {
        format!("PyEdge({})", self.summary())
    }
}
