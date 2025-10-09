//! Python wrapper for MetaNode entities
//!
//! This module provides the PyMetaNode class that wraps our Rust MetaNode entity,
//! exposing all GraphEntity, NodeOperations, and MetaNodeOperations methods to Python.

use groggy::entities::MetaNode;
use groggy::types::{EdgeId, NodeId};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Python wrapper for a meta-node (collapsed subgraph)
///
/// Meta-nodes are special nodes that represent collapsed subgraphs. They have all
/// the capabilities of regular nodes plus meta-specific operations like expanding
/// back to the original subgraph.
#[pyclass(name = "MetaNode", unsendable)]
pub struct PyMetaNode {
    /// The underlying Rust MetaNode entity
    pub inner: MetaNode,
}

impl PyMetaNode {
    /// Create a new PyMetaNode wrapper
    ///
    /// # Arguments
    /// * `meta_node` - The Rust MetaNode entity to wrap
    ///
    /// # Returns
    /// A new PyMetaNode instance
    pub fn from_meta_node(meta_node: MetaNode) -> Self {
        Self { inner: meta_node }
    }
}

#[pymethods]
impl PyMetaNode {
    /// Get the meta-node ID
    ///
    /// # Returns
    /// The unique NodeId for this meta-node
    #[getter]
    fn id(&self) -> NodeId {
        self.inner.id()
    }

    /// Get meta-node attribute value
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
                "MetaNode {} has no attribute '{}'",
                self.inner.id(),
                key
            ))),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get meta-node attribute: {}",
                e
            ))),
        }
    }

    /// Set meta-node attribute value
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
                PyRuntimeError::new_err(format!("Failed to set meta-node attribute: {}", e))
            })?;
        Ok(())
    }

    /// Check if attribute exists on this meta-node
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

    // === NodeOperations (inherited from regular nodes) ===

    /// Get the meta-node's degree (number of connected edges)
    ///
    /// # Returns
    /// The number of edges connected to this meta-node
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error calculating degree
    #[getter]
    fn degree(&self) -> PyResult<usize> {
        use groggy::traits::NodeOperations;

        self.inner
            .degree()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get meta-node degree: {}", e)))
    }

    /// Get the meta-node's neighbors
    ///
    /// # Returns
    /// List of NodeIds representing neighboring nodes (could be regular nodes or other meta-nodes)
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error getting neighbors
    #[getter]
    fn neighbors(&self) -> PyResult<Vec<NodeId>> {
        use groggy::traits::NodeOperations;

        self.inner.neighbors().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get meta-node neighbors: {}", e))
        })
    }

    // === MetaNodeOperations (meta-specific methods) ===

    /// Check if this meta-node contains a subgraph
    ///
    /// # Returns
    /// True if this meta-node has an associated subgraph
    #[getter]
    fn has_subgraph(&self) -> bool {
        use groggy::traits::MetaNodeOperations;
        self.inner.has_subgraph()
    }

    /// Get the ID of the contained subgraph
    ///
    /// # Returns
    /// Optional subgraph ID if this meta-node contains a subgraph
    #[getter]
    fn subgraph_id(&self) -> Option<usize> {
        use groggy::traits::MetaNodeOperations;
        self.inner.subgraph_id()
    }

    /// Get the contained subgraph
    ///
    /// This returns the original subgraph that was collapsed to create this meta-node.
    ///
    /// # Returns
    /// Optional PySubgraph representing the contained subgraph
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error accessing the subgraph
    #[getter]
    fn subgraph(&self, py: Python) -> PyResult<Option<PyObject>> {
        use groggy::traits::MetaNodeOperations;

        match self.inner.subgraph() {
            Ok(Some(subgraph_trait_obj)) => {
                // Extract the data from the trait object and create a concrete subgraph
                use crate::ffi::subgraphs::subgraph::PySubgraph;
                use groggy::subgraphs::Subgraph;

                // Create a new concrete Subgraph from the trait object data
                let concrete_subgraph = Subgraph::new(
                    subgraph_trait_obj.graph_ref().clone(),
                    subgraph_trait_obj.node_set().clone(),
                    subgraph_trait_obj.edge_set().clone(),
                    format!("expanded_subgraph_{}", self.inner.id()),
                );

                let py_subgraph = PySubgraph::from_core_subgraph(concrete_subgraph)?;
                Ok(Some(Py::new(py, py_subgraph)?.to_object(py)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get meta-node subgraph: {}",
                e
            ))),
        }
    }

    /// Expand the meta-node back to its original subgraph
    ///
    /// This is an alias for the subgraph property with a more intuitive name.
    ///
    /// # Returns
    /// Optional PySubgraph representing the expanded subgraph
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error expanding the meta-node
    fn expand(&self, py: Python) -> PyResult<Option<PyObject>> {
        self.subgraph(py)
    }

    /// Get all meta-edges connected to this meta-node
    ///
    /// Meta-edges are edges with entity_type="meta" that were created during
    /// the subgraph collapse process.
    ///
    /// # Returns
    /// List of EdgeIds representing meta-edges connected to this meta-node
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error getting meta-edges
    #[getter]
    fn meta_edges(&self) -> PyResult<Vec<EdgeId>> {
        use groggy::traits::MetaNodeOperations;

        self.inner
            .meta_edges()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get meta-edges: {}", e)))
    }

    /// Re-aggregate meta-node attributes with new aggregation functions
    ///
    /// This allows updating the meta-node's aggregated attributes by re-running
    /// the aggregation process with different functions.
    ///
    /// # Arguments
    /// * `agg_functions` - Dictionary mapping attribute names to aggregation functions
    ///
    /// # Raises
    /// * `RuntimeError` - If there's an error during re-aggregation
    fn re_aggregate(
        &self,
        agg_functions: std::collections::HashMap<String, String>,
    ) -> PyResult<()> {
        use groggy::traits::MetaNodeOperations;
        use groggy::types::AttrName;

        // Convert String keys to AttrName
        let rust_agg_functions: std::collections::HashMap<AttrName, String> =
            agg_functions.into_iter().collect();

        self.inner.re_aggregate(rust_agg_functions).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to re-aggregate meta-node: {}", e))
        })
    }

    // === Utility methods ===

    /// Get all attribute keys for this meta-node
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
                "Failed to get meta-node attributes: {}",
                e
            ))),
        }
    }

    /// Get all attribute values for this meta-node
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
            PyRuntimeError::new_err(format!("Failed to get meta-node attributes: {}", e))
        })?;

        let values = node_attrs
            .values()
            .map(|value| crate::ffi::utils::attr_value_to_python_value(py, value))
            .collect::<PyResult<Vec<_>>>()?;

        Ok(values)
    }

    /// Get the entity type for this meta-node
    ///
    /// # Returns
    /// The string "meta_node"
    #[getter]
    fn entity_type(&self) -> &'static str {
        use groggy::traits::GraphEntity;
        self.inner.entity_type()
    }

    /// Check if this meta-node is currently active
    ///
    /// # Returns
    /// True if the meta-node is active in the graph
    #[getter]
    fn is_active(&self) -> bool {
        use groggy::traits::GraphEntity;
        self.inner.is_active()
    }

    /// Get a summary of this meta-node
    ///
    /// # Returns
    /// A human-readable summary string including meta-node specific information
    fn summary(&self) -> String {
        use groggy::traits::GraphEntity;
        self.inner.summary()
    }

    /// String representation of the meta-node
    ///
    /// # Returns
    /// A string representation showing the meta-node ID and key information
    fn __str__(&self) -> String {
        self.summary()
    }

    /// Detailed string representation of the meta-node
    ///
    /// # Returns
    /// A detailed string representation
    fn __repr__(&self) -> String {
        format!("PyMetaNode({})", self.summary())
    }
}
