// src_new/graph/edges/proxy.rs
//! EdgeProxy: Per-edge interface for attribute access, endpoints, and graph operations in Groggy.
//! Designed for agent/LLM workflows and backend extensibility.

use pyo3::prelude::*;
use crate::graph::types::{EdgeId, NodeId};
use crate::graph::managers::attributes::AttributeManager;
use crate::graph::proxy::base::EdgeProxyAttributeManager;
use serde_json::Value;

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
    #[pyo3(get)]
    pub graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>,
}

#[pymethods]
impl EdgeProxy {
    #[new]
    pub fn new(edge_id: EdgeId, source: NodeId, target: NodeId, attribute_manager: AttributeManager, graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>) -> Self {
        Self { edge_id, source, target, attribute_manager, graph_store }
    }

    /// Returns a ProxyAttributeManager for this edge (per-attribute API).
    pub fn attr_manager(&self) -> EdgeProxyAttributeManager {
        EdgeProxyAttributeManager::new(self.edge_id.clone(), self.attribute_manager.clone())
    }

    /// Get the value of a single attribute for this edge (JSON).
    pub fn get_attr(&self, attr_name: String) -> Option<serde_json::Value> {
        if let Some(index) = self.graph_store.edge_index(&self.edge_id) {
            self.attribute_manager.get_edge_value(attr_name, index)
        } else {
            None
        }
    }

    /// Set the value of a single attribute for this edge (JSON).
    pub fn set_attr(&mut self, attr_name: String, value: serde_json::Value) -> Result<(), String> {
        if let Some(index) = self.graph_store.edge_index(&self.edge_id) {
            self.attribute_manager.set_edge_value(attr_name, index, value);
            Ok(())
        } else {
            Err("Edge not found in graph".to_string())
        }
    }

    /// Get all attributes for this edge as a map (JSON).
    pub fn attrs(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
        if let Some(index) = self.graph_store.edge_index(&self.edge_id) {
            // Collect all present attributes for this edge
            let mut map = serde_json::Map::new();
            for attr in self.attribute_manager.columnar.edge_attr_names() {
                if let Some(val) = self.attribute_manager.get_edge_value(attr.clone(), index) {
                    map.insert(attr, val);
                }
            }
            Some(map)
        } else {
            None
        }
    }

    /// Returns a string representation of this edge (for debugging or display).
    pub fn __str__(&self) -> String {
        format!("EdgeProxy({}, {} -> {})", self.edge_id, self.source, self.target)
    }
}

// Helper: EdgeProxy factory methods: (create new edges with attribute/type-guided construction)
