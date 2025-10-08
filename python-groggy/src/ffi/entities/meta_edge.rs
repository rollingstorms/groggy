//! Python wrapper for MetaEdge entities
//!
//! This module provides the PyMetaEdge class that wraps our Rust MetaEdge entity,
//! exposing all GraphEntity, EdgeOperations, and MetaEdgeOperations methods to Python.

use groggy::entities::MetaEdge;
use groggy::types::{EdgeId, NodeId};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Python wrapper for a meta-edge (aggregated edge)
///
/// Meta-edges are special edges that represent aggregated or summarized relationships
/// between nodes. They have all the capabilities of regular edges plus meta-specific
/// operations like getting aggregation counts.
#[pyclass(name = "MetaEdge", unsendable)]
pub struct PyMetaEdge {
    /// The underlying Rust MetaEdge entity
    pub inner: MetaEdge,
}

impl PyMetaEdge {
    /// Create a new PyMetaEdge wrapper
    ///
    /// # Arguments
    /// * `meta_edge` - The Rust MetaEdge entity to wrap
    ///
    /// # Returns
    /// A new PyMetaEdge instance
    pub fn from_meta_edge(meta_edge: MetaEdge) -> Self {
        Self { inner: meta_edge }
    }
}

#[pymethods]
impl PyMetaEdge {
    /// Get the meta-edge ID
    ///
    /// # Returns
    /// The unique EdgeId for this meta-edge
    #[getter]
    fn id(&self) -> EdgeId {
        self.inner.id()
    }

    /// Get meta-edge attribute value
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
                "MetaEdge {} has no attribute '{}'",
                self.inner.id(),
                key
            ))),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get meta-edge attribute: {}",
                e
            ))),
        }
    }

    /// Set meta-edge attribute value
    ///
    /// # Arguments
    /// * `key` - Attribute name to set
    /// * `value` - Attribute value to store
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error setting the attribute
    fn __setitem__(&self, key: &str, value: &PyAny) -> PyResult<()> {
        use groggy::traits::GraphEntity;

        let py_attr_value = crate::ffi::types::PyAttrValue::from_py_value(value)?;
        let attr_value = py_attr_value.to_attr_value();
        self.inner
            .set_attribute(key.into(), attr_value)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to set meta-edge attribute: {}", e))
            })?;
        Ok(())
    }

    /// Check if attribute exists on this meta-edge
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

    // === EdgeOperations (inherited from regular edges) ===

    /// Get the meta-edge's source node ID
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
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get meta-edge source: {}", e)))
    }

    /// Get the meta-edge's target node ID
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
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get meta-edge target: {}", e)))
    }

    // === MetaEdgeOperations (meta-specific methods) ===

    /// Check if this is a meta-edge
    ///
    /// Meta-edges are identified by having entity_type="meta"
    ///
    /// # Returns
    /// True if this is a meta-edge, False otherwise
    #[getter]
    fn is_meta_edge(&self) -> bool {
        use groggy::traits::MetaEdgeOperations;
        self.inner.is_meta_edge()
    }

    /// Get the count of original edges this meta-edge aggregates
    ///
    /// During subgraph collapse, multiple original edges may be aggregated
    /// into a single meta-edge. This returns how many were combined.
    ///
    /// # Returns
    /// Optional count of original edges, or None if not available
    #[getter]
    fn edge_count(&self) -> Option<i64> {
        use groggy::traits::MetaEdgeOperations;
        self.inner.edge_count()
    }

    /// Get the IDs of original edges that were aggregated into this meta-edge
    ///
    /// This is a future enhancement - currently original edge IDs are not stored
    /// during the collapse process.
    ///
    /// # Returns
    /// Optional list of original EdgeIds, or None if not available
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error accessing aggregated edges
    #[getter]
    fn aggregated_from(&self) -> PyResult<Option<Vec<EdgeId>>> {
        use groggy::traits::MetaEdgeOperations;

        self.inner
            .aggregated_from()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get aggregated edges: {}", e)))
    }

    /// Expand meta-edge back to original edges
    ///
    /// This is a future enhancement that would recreate the original edges
    /// that were aggregated into this meta-edge.
    ///
    /// # Returns
    /// Optional list of recreated EdgeIds, or None if not possible
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error expanding the meta-edge
    fn expand(&self) -> PyResult<Option<Vec<EdgeId>>> {
        use groggy::traits::MetaEdgeOperations;

        self.inner
            .expand()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to expand meta-edge: {}", e)))
    }

    /// Get meta-edge specific properties as a dictionary
    ///
    /// Returns a dictionary containing meta-edge specific attributes like
    /// edge_count, aggregation information, etc.
    ///
    /// # Arguments
    /// * `py` - Python interpreter instance
    ///
    /// # Returns
    /// Dictionary of property names to values for this meta-edge
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error accessing meta properties
    fn meta_properties(&self, py: Python) -> PyResult<HashMap<String, Py<PyAny>>> {
        use groggy::traits::MetaEdgeOperations;

        let props = self.inner.meta_properties().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get meta properties: {}", e))
        })?;

        let mut py_props = HashMap::new();
        for (key, value) in props {
            let py_value = crate::ffi::utils::attr_value_to_python_value(py, &value)?;
            py_props.insert(key, py_value);
        }

        Ok(py_props)
    }

    // === Utility methods ===

    /// Get all attribute keys for this meta-edge
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
                "Failed to get meta-edge attributes: {}",
                e
            ))),
        }
    }

    /// Get all attribute values for this meta-edge
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
            PyRuntimeError::new_err(format!("Failed to get meta-edge attributes: {}", e))
        })?;

        let values = edge_attrs
            .values()
            .map(|value| crate::ffi::utils::attr_value_to_python_value(py, value))
            .collect::<PyResult<Vec<_>>>()?;

        Ok(values)
    }

    /// Get the entity type for this meta-edge
    ///
    /// # Returns
    /// The string "meta_edge"
    #[getter]
    fn entity_type(&self) -> &'static str {
        use groggy::traits::GraphEntity;
        self.inner.entity_type()
    }

    /// Check if this meta-edge is currently active
    ///
    /// # Returns
    /// True if the meta-edge is active in the graph
    #[getter]
    fn is_active(&self) -> bool {
        use groggy::traits::GraphEntity;
        self.inner.is_active()
    }

    /// Get a summary of this meta-edge
    ///
    /// # Returns
    /// A human-readable summary string including meta-edge specific information
    fn summary(&self) -> String {
        use groggy::traits::GraphEntity;
        self.inner.summary()
    }

    /// String representation of the meta-edge
    ///
    /// # Returns
    /// A string representation showing the meta-edge ID and key information
    fn __str__(&self) -> String {
        self.summary()
    }

    /// Detailed string representation of the meta-edge
    ///
    /// # Returns
    /// A detailed string representation
    fn __repr__(&self) -> String {
        format!("PyMetaEdge({})", self.summary())
    }
}
