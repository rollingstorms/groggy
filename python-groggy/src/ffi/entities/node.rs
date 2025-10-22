//! Python wrapper for regular Node entities
//!
//! This module provides the PyNode class that wraps our Rust Node entity,
//! exposing all GraphEntity and NodeOperations methods to Python.

use groggy::entities::Node;
use groggy::types::NodeId;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Python wrapper for a regular graph node
///
/// Regular nodes are the basic building blocks of graphs. They provide access
/// to node attributes, topology information (neighbors, degree), and graph operations.
#[pyclass(name = "Node", unsendable)]
pub struct PyNode {
    /// The underlying Rust Node entity
    pub inner: Node,
}

impl PyNode {
    /// Create a new PyNode wrapper
    ///
    /// # Arguments
    /// * `node` - The Rust Node entity to wrap
    ///
    /// # Returns
    /// A new PyNode instance
    pub fn from_node(node: Node) -> Self {
        Self { inner: node }
    }
}

#[pymethods]
impl PyNode {
    /// Get the node ID
    ///
    /// # Returns
    /// The unique NodeId for this node
    #[getter]
    fn id(&self) -> NodeId {
        self.inner.id()
    }

    /// Get node attribute value
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
                "Node {} has no attribute '{}'",
                self.inner.id(),
                key
            ))),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get node attribute: {}",
                e
            ))),
        }
    }

    /// Set node attribute value
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
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set node attribute: {}", e)))?;
        Ok(())
    }

    /// Check if attribute exists on this node
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

    /// Get node attribute via attribute access (node.name)
    ///
    /// This enables dot notation access to node attributes:
    /// node.name instead of node['name']
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
                "'Node' object has no attribute '{}'",
                name
            ))),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get node attribute: {}",
                e
            ))),
        }
    }

    /// Get the node's degree (number of connected edges)
    ///
    /// # Returns
    /// The number of edges connected to this node
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error calculating degree
    #[getter]
    fn degree(&self) -> PyResult<usize> {
        use groggy::traits::NodeOperations;

        self.inner
            .degree()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get node degree: {}", e)))
    }

    /// Get the node's neighbors
    ///
    /// # Returns
    /// List of NodeIds representing neighboring nodes
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error getting neighbors
    #[getter]
    fn neighbors(&self) -> PyResult<Vec<NodeId>> {
        use groggy::traits::NodeOperations;

        self.inner
            .neighbors()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get node neighbors: {}", e)))
    }

    /// Get all attribute keys for this node
    ///
    /// # Returns
    /// List of attribute names
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error accessing attributes
    fn keys(&self) -> PyResult<Vec<String>> {
        use groggy::traits::NodeOperations;

        match self.inner.node_attributes() {
            Ok(attrs) => Ok(attrs.keys().cloned().collect()),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get node attributes: {}",
                e
            ))),
        }
    }

    /// Get all attribute values for this node
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
        use groggy::traits::NodeOperations;

        let node_attrs = self.inner.node_attributes().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get node attributes: {}", e))
        })?;

        let values = node_attrs
            .values()
            .map(|value| crate::ffi::utils::attr_value_to_python_value(py, value))
            .collect::<PyResult<Vec<_>>>()?;

        Ok(values)
    }

    /// Get the entity type for this node
    ///
    /// # Returns
    /// The string "node"
    #[getter]
    fn entity_type(&self) -> &'static str {
        use groggy::traits::GraphEntity;
        self.inner.entity_type()
    }

    /// Check if this node is currently active
    ///
    /// # Returns
    /// True if the node is active in the graph
    #[getter]
    fn is_active(&self) -> bool {
        use groggy::traits::GraphEntity;
        self.inner.is_active()
    }

    /// Get a summary of this node
    ///
    /// # Returns
    /// A human-readable summary string
    fn summary(&self) -> String {
        use groggy::traits::GraphEntity;
        self.inner.summary()
    }

    /// String representation of the node
    ///
    /// # Returns
    /// A string representation showing the node ID and key attributes
    fn __str__(&self) -> String {
        self.summary()
    }

    /// Detailed string representation of the node
    ///
    /// # Returns
    /// A detailed string representation
    fn __repr__(&self) -> String {
        format!("PyNode({})", self.summary())
    }
}
