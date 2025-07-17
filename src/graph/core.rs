// src_new/graph/core.rs

use pyo3::pyclass;
use pyo3::pymethods;
use pyo3::PyResult;
use std::sync::Arc;

/// Main graph structure with delegated collections
#[pyclass]
pub struct FastGraph {
    pub attribute_manager: crate::graph::managers::attributes::AttributeManager,
    pub graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>,
    pub node_collection: crate::graph::nodes::collection::NodeCollection,
    pub edge_collection: crate::graph::edges::collection::EdgeCollection,
    pub directed: bool,
    pub info: crate::graph::types::GraphInfo,
    // NEW: High-performance core for 10x optimization
    pub fast_core: Arc<crate::storage::fast_core::FastGraphCore>,
}

#[pymethods]
impl FastGraph {
    /// Constructor for new graph instance
    #[new]
    pub fn new() -> Self {
        let graph_store = std::sync::Arc::new(crate::storage::graph_store::GraphStore::new());
        let attribute_manager = crate::graph::managers::attributes::AttributeManager::new_with_graph_store(graph_store.clone());
        let node_collection = crate::graph::nodes::collection::NodeCollection::new(attribute_manager.clone(), graph_store.clone(), None);
        let edge_collection = crate::graph::edges::collection::EdgeCollection::new(attribute_manager.clone(), graph_store.clone(), None);
        let info = crate::graph::types::GraphInfo::default();
        let fast_core = Arc::new(crate::storage::fast_core::FastGraphCore::new());
        Self {
            attribute_manager,
            graph_store,
            node_collection,
            edge_collection,
            directed: true,
            info,
            fast_core,
        }
    }

    /// Get comprehensive graph information
    pub fn info(&self) -> crate::graph::types::GraphInfo {
        let mut info = self.info.clone();
        
        // Update counts with current values from collections and fast core
        info.node_count = self.node_collection.size();
        info.edge_count = self.edge_collection.size();
        
        // Gather memory usage in bytes
        let graph_store_bytes = self.graph_store.memory_usage_bytes();
        let content_pool_bytes = self.graph_store.content_pool_memory_usage_bytes();
        let columnar_bytes = self.attribute_manager.memory_usage_bytes();
        let fast_core_bytes = self.fast_core.memory_usage_bytes();
        
        // Convert to MB (floating point, 2 decimals)
        let graph_store_mb = (graph_store_bytes as f64) / (1024.0 * 1024.0);
        let content_pool_mb = (content_pool_bytes as f64) / (1024.0 * 1024.0);
        let columnar_mb = (columnar_bytes as f64) / (1024.0 * 1024.0);
        let fast_core_mb = (fast_core_bytes as f64) / (1024.0 * 1024.0);
        
        // Add to attributes hashmap as strings
        info.attributes.insert("memory_graph_store_mb".to_string(), format!("{:.2}", graph_store_mb));
        info.attributes.insert("memory_content_pool_mb".to_string(), format!("{:.2}", content_pool_mb));
        info.attributes.insert("memory_columnar_store_mb".to_string(), format!("{:.2}", columnar_mb));
        info.attributes.insert("memory_fast_core_mb".to_string(), format!("{:.2}", fast_core_mb));
        
        // Add fast core stats
        info.attributes.insert("fast_core_nodes".to_string(), self.fast_core.node_count().to_string());
        info.attributes.insert("fast_core_edges".to_string(), self.fast_core.edge_count().to_string());
        
        info
    }

    /// Returns a breakdown of memory usage per attribute (name, type, node/edge, bytes used)
    pub fn memory_usage_breakdown(&self) -> std::collections::HashMap<String, usize> {
        self.attribute_manager.memory_usage_breakdown()
    }

    /// Expose attribute_manager as a Python property
    #[getter]
    pub fn attribute_manager(&self) -> crate::graph::managers::attributes::AttributeManager {
        self.attribute_manager.clone()
    }

    /// Get total size (nodes + edges)
    pub fn size(&self) -> usize {
        self.node_collection.size() + self.edge_collection.size()
    }

    /// Check if graph is directed
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// Returns NodeCollection instance
    pub fn nodes(&self) -> crate::graph::nodes::collection::NodeCollection {
        let mut collection = self.node_collection.clone();
        // Ensure the node_ids are up to date with the graph store
        collection.node_ids = self.graph_store.all_node_ids();
        collection
    }

    /// Returns EdgeCollection instance
    pub fn edges(&self) -> crate::graph::edges::collection::EdgeCollection {
        self.edge_collection.clone()
    }

    /// Create subgraph with node/edge filters
    pub fn subgraph(&self, node_ids: Option<Vec<crate::graph::types::NodeId>>, edge_ids: Option<Vec<crate::graph::types::EdgeId>>) -> Self {
        let mut node_collection = self.node_collection.clone();
        let mut edge_collection = self.edge_collection.clone();
        if let Some(ids) = node_ids {
            node_collection.node_ids = ids;
        }
        if let Some(ids) = edge_ids {
            edge_collection.edge_ids = ids;
        }
        Self {
            attribute_manager: self.attribute_manager.clone(),
            graph_store: self.graph_store.clone(),
            node_collection,
            edge_collection,
            directed: self.directed,
            info: self.info.clone(),
            fast_core: self.fast_core.clone(),
        }
    }

    /// Get all subgraphs according to a given attr groups
    pub fn subgraphs(&self) {
        // TODO
    }

    // === Fast Core Methods for 10x Performance ===

    /// Add nodes using fast core (optimized batch operation)
    pub fn fast_add_nodes(&self, node_ids: Vec<String>) -> PyResult<()> {
        let str_refs: Vec<&str> = node_ids.iter().map(|s| s.as_str()).collect();
        self.fast_core.add_nodes(&str_refs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    }

    /// Add edges using fast core (optimized batch operation)
    pub fn fast_add_edges(&self, edge_pairs: Vec<(String, String)>) -> PyResult<()> {
        let str_pairs: Vec<(&str, &str)> = edge_pairs.iter()
            .map(|(s, t)| (s.as_str(), t.as_str()))
            .collect();
        self.fast_core.add_edges(&str_pairs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    }

    /// Set node attribute using fast core (optimized)
    pub fn fast_set_node_attr(&self, attr_name: String, node_id: String, value: String) -> PyResult<()> {
        let json_value: serde_json::Value = serde_json::from_str(&value)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;
        
        self.fast_core.set_node_attr(&attr_name, &node_id, &json_value)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    }

    /// Batch set node attributes using fast core (optimized)
    pub fn fast_set_node_attrs_batch(&self, attr_name: String, data: std::collections::HashMap<String, String>) -> PyResult<()> {
        use rustc_hash::FxHashMap;
        let mut json_data = FxHashMap::default();
        
        for (node_id, value_str) in data {
            let json_value: serde_json::Value = serde_json::from_str(&value_str)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON for {}: {}", node_id, e)))?;
            json_data.insert(node_id, json_value);
        }
        
        self.fast_core.set_node_attrs_batch(&attr_name, &json_data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    }

    /// Get node attribute using fast core (optimized)
    pub fn fast_get_node_attr(&self, attr_name: String, node_id: String) -> Option<String> {
        self.fast_core.get_node_attr(&attr_name, &node_id)
            .map(|v| serde_json::to_string(&v).unwrap_or_default())
    }

    /// Get fast core node IDs (optimized)
    pub fn fast_node_ids(&self) -> Vec<String> {
        self.fast_core.node_ids()
    }

    /// Get fast core edge IDs (optimized)
    pub fn fast_edge_ids(&self) -> Vec<(String, String)> {
        self.fast_core.edge_ids()
    }

    /// Get fast core memory usage in bytes
    pub fn fast_core_memory_usage(&self) -> usize {
        self.fast_core.memory_usage_bytes()
    }

    // === Ultra-Fast Bulk Operations (10x Performance Target) ===

    /// Ultra-fast bulk node addition with attributes (minimal locking)
    pub fn ultra_fast_add_nodes_with_attrs(&self, nodes_data: Vec<(String, std::collections::HashMap<String, String>)>) -> PyResult<()> {
        // Convert to internal format
        let mut nodes_with_attrs = Vec::with_capacity(nodes_data.len());
        
        for (node_id, attrs) in nodes_data {
            let mut attr_vec = Vec::with_capacity(attrs.len());
            for (key, value_str) in attrs {
                let json_value: serde_json::Value = serde_json::from_str(&value_str)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON for {}: {}", key, e)))?;
                attr_vec.push((key, json_value));
            }
            nodes_with_attrs.push((node_id, attr_vec));
        }

        self.fast_core.bulk_add_nodes_with_attrs(&nodes_with_attrs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    }

    /// Ultra-fast vectorized attribute setting (SIMD-style)
    pub fn ultra_fast_set_attrs_vectorized(&self, attr_name: String, values: Vec<(String, String)>) -> PyResult<()> {
        // Convert to internal format
        let mut value_pairs = Vec::with_capacity(values.len());
        
        for (node_id, value_str) in values {
            let json_value: serde_json::Value = serde_json::from_str(&value_str)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON for {}: {}", node_id, e)))?;
            value_pairs.push((node_id, json_value));
        }

        self.fast_core.bulk_set_attrs_vectorized(&attr_name, &value_pairs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    }

    // === Zero-Copy Ultra-Performance Methods ===

    /// Zero-copy bulk node generation (minimal allocations)
    pub fn zero_copy_bulk_add(&self, node_count: usize, base_name: String) -> PyResult<()> {
        self.fast_core.zero_copy_bulk_add(node_count, &base_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    }

    /// Native integer attribute setting (no JSON)
    pub fn set_i64_attrs_native(&self, attr_name: String, values: Vec<(u32, i64)>) -> PyResult<()> {
        self.fast_core.set_i64_attrs_native(&attr_name, &values)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    }

    /// Native string attribute setting (no JSON)
    pub fn set_string_attrs_native(&self, attr_name: String, values: Vec<(u32, u32)>) -> PyResult<()> {
        self.fast_core.set_string_attrs_native(&attr_name, &values)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    }

    /// Ultra-fast pattern attribute generation
    pub fn generate_pattern_attrs(&self, attr_name: String, pattern: String, count: usize) -> PyResult<()> {
        self.fast_core.generate_pattern_attrs(&attr_name, &pattern, count)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    }

    /// Get atomic stats (lock-free)
    pub fn get_atomic_stats(&self) -> (usize, usize) {
        self.fast_core.get_atomic_stats()
    }
}
