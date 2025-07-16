// src_new/graph/edges/proxy.rs
//! EdgeProxy: Per-edge interface for attribute access, endpoints, and graph operations in Groggy.
//! Designed for agent/LLM workflows and backend extensibility.

use pyo3::prelude::*;
use crate::graph::types::{EdgeId, NodeId};
use crate::graph::managers::attributes::AttributeManager;
use crate::graph::proxy::base::EdgeProxyAttributeManager;
// use serde_json::Value; // Currently unused

#[pyclass]
pub struct EdgeProxy {
    #[pyo3(get)]
    pub edge_id: EdgeId,
    #[pyo3(get)]
    pub source: NodeId,
    #[pyo3(get)]
    pub target: NodeId,
    #[pyo3(get)]
    pub attribute_manager: AttributeManager,
    pub graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>,
}

#[pymethods]
impl EdgeProxy {
    /// Create a new EdgeProxy from Python (simplified constructor)
    #[new]
    pub fn py_new(edge_id: EdgeId, source: NodeId, target: NodeId, attribute_manager: AttributeManager) -> Self {
        let graph_store = std::sync::Arc::new(crate::storage::graph_store::GraphStore::new());
        Self { edge_id, source, target, attribute_manager, graph_store }
    }

    /// Returns a ProxyAttributeManager for this edge (per-attribute API).
    pub fn attr_manager(&self) -> EdgeProxyAttributeManager {
        EdgeProxyAttributeManager::new(self.edge_id.clone(), self.attribute_manager.clone())
    }

    /// Get the value of a single attribute for this edge (JSON string).
    pub fn get_attr(&self, attr_name: String) -> Option<String> {
        if let Some(index) = self.graph_store.edge_index(&self.edge_id) {
            self.attribute_manager.get_edge_value(attr_name, index)
                .map(|v| serde_json::to_string(&v).unwrap_or_default())
        } else {
            None
        }
    }

    /// Set the value of a single attribute for this edge (JSON string).
    pub fn set_attr(&mut self, attr_name: String, value: String) -> PyResult<()> {
        let json_value: serde_json::Value = serde_json::from_str(&value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid JSON: {}", e)))?;
        if let Some(index) = self.graph_store.edge_index(&self.edge_id) {
            self.attribute_manager.set_edge_value(attr_name, index, json_value);
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>("Edge not found in graph"))
        }
    }

    /// Get all attributes for this edge as a map (JSON string).
    pub fn attrs(&self) -> Option<String> {
        if let Some(index) = self.graph_store.edge_index(&self.edge_id) {
            // Collect all present attributes for this edge
            let mut map = serde_json::Map::new();
            for attr in self.attribute_manager.columnar.edge_attr_names() {
                if let Some(val) = self.attribute_manager.get_edge_value(attr.clone(), index) {
                    map.insert(attr, val);
                }
            }
            if map.is_empty() {
                None
            } else {
                serde_json::to_string(&map).ok()
            }
        } else {
            None
        }
    }

    /// Returns a string representation of this edge (for debugging or display).
    pub fn __str__(&self) -> String {
        format!("EdgeProxy({}, {} -> {})", self.edge_id, self.source, self.target)
    }
}

impl EdgeProxy {
    /// Regular Rust constructor - not exposed to Python
    pub fn new(edge_id: EdgeId, source: NodeId, target: NodeId, attribute_manager: AttributeManager, graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>) -> Self {
        Self { edge_id, source, target, attribute_manager, graph_store }
    }
}

// Helper: EdgeProxy factory methods: (create new edges with attribute/type-guided construction)
