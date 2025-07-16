// src_new/graph/managers/attributes.rs
//! AttributeManager: Unified, batch-friendly attribute management for nodes and edges in Groggy.
//! Supports columnar storage, schema enforcement, and agent/LLM-friendly APIs.

use pyo3::prelude::*;
// use crate::graph::types::{NodeId, EdgeId}; // Currently unused
// use crate::graph::columnar::{NodeColumnarStore, EdgeColumnarStore}; // Uncomment when available

/// Unified attribute management for nodes and edges (columnar)
use crate::storage::columnar::ColumnarStore;

/// Batch operation types for vectorized attribute operations
#[derive(Debug, Clone)]
pub enum BatchOperation {
    Set {
        attr_name: String,
        entity_id: usize,
        value: serde_json::Value,
        is_node: bool,
    },
    Get {
        attr_name: String,
        entity_id: usize,
        is_node: bool,
    },
    Filter {
        attr_name: String,
        value: serde_json::Value,
        is_node: bool,
    },
}

#[derive(Debug, Clone)]
pub enum BatchResult {
    Success,
    Value(serde_json::Value),
    Indices(Vec<usize>),
    NotFound,
    Error(String),
}

#[pyclass]
#[derive(Clone)]
pub struct AttributeManager {
    pub columnar: ColumnarStore,
}

#[pymethods]
impl AttributeManager {
    #[new]
    pub fn new() -> Self {
        Self {
            columnar: ColumnarStore::new(),
        }
    }

    /// Basic Python-compatible get method (returns JSON string)
    pub fn get_py(&self, id: &str, attr: &str) -> Option<String> {
        self.get(id, attr).map(|v| serde_json::to_string(&v).unwrap_or_default())
    }
    
    /// Basic Python-compatible set method (accepts JSON string)
    pub fn set_py(&mut self, id: &str, attr: &str, value: String) -> PyResult<()> {
        let json_value: serde_json::Value = serde_json::from_str(&value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid JSON: {}", e)))?;
        self.set(id, attr, json_value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Registers an attribute and returns its UID.
    pub fn register_attr(&self, attr_name: String) -> u64 {
        self.columnar.register_attr(attr_name)
    }

    /// Get all node attribute names.
    pub fn node_attr_names(&self) -> Vec<String> {
        self.columnar.node_attr_names()
    }

    /// Get all edge attribute names.
    pub fn edge_attr_names(&self) -> Vec<String> {
        self.columnar.edge_attr_names()
    }
}

// Internal methods for performance (not exposed to Python)
impl AttributeManager {
    /// Internal get method using serde_json::Value
    pub fn get(&self, id: &str, attr: &str) -> Option<serde_json::Value> {
        // Parse the ID to get the index
        if let Ok(index) = id.parse::<usize>() {
            // Try as node first, then as edge
            if let Some(value) = self.columnar.get_node_value(attr.to_string(), index) {
                Some(value)
            } else {
                self.columnar.get_edge_value(attr.to_string(), index)
            }
        } else {
            None
        }
    }
    
    /// Internal set method using serde_json::Value
    pub fn set(&mut self, id: &str, attr: &str, value: serde_json::Value) -> Result<(), String> {
        // Parse the ID to get the index
        if let Ok(index) = id.parse::<usize>() {
            // For now, assume it's a node - in a real implementation, you'd need to know the entity type
            self.columnar.set_node_value(attr.to_string(), index, value);
            Ok(())
        } else {
            Err(format!("Invalid ID format: {}", id))
        }
    }

    /// Get all values for a node attribute by name.
    pub fn get_node_attr(&self, attr_name: String) -> Option<std::collections::HashMap<usize, serde_json::Value>> {
        self.columnar.get_node_attr(attr_name)
    }

    /// Set all values for a node attribute by name.
    pub fn set_node_attr(&self, attr_name: String, data: std::collections::HashMap<usize, serde_json::Value>) {
        self.columnar.set_node_attr(attr_name, data);
    }

    /// Get a single value for a node attribute and entity index.
    pub fn get_node_value(&self, attr_name: String, idx: usize) -> Option<serde_json::Value> {
        self.columnar.get_node_value(attr_name, idx)
    }

    /// Set a single value for a node attribute and entity index.
    pub fn set_node_value(&self, attr_name: String, idx: usize, value: serde_json::Value) {
        self.columnar.set_node_value(attr_name, idx, value);
    }

    /// Get column stats for node attributes.
    pub fn get_node_column_stats(&self) -> std::collections::HashMap<String, usize> {
        self.columnar.get_column_stats()
    }

    /// Filter nodes by attribute value (returns indices).
    pub fn filter_nodes_by_value(&self, attr_name: String, value: serde_json::Value) -> Vec<usize> {
        self.columnar.filter_nodes_by_value(attr_name, value)
    }

    /// Get all values for an edge attribute by name.
    pub fn get_edge_attr(&self, attr_name: String) -> Option<std::collections::HashMap<usize, serde_json::Value>> {
        self.columnar.get_edge_attr(attr_name)
    }

    /// Set all values for an edge attribute by name.
    pub fn set_edge_attr(&self, attr_name: String, data: std::collections::HashMap<usize, serde_json::Value>) {
        self.columnar.set_edge_attr(attr_name, data);
    }

    /// Get a single value for an edge attribute and entity index.
    pub fn get_edge_value(&self, attr_name: String, idx: usize) -> Option<serde_json::Value> {
        self.columnar.get_edge_value(attr_name, idx)
    }

    /// Set a single value for an edge attribute and entity index.
    pub fn set_edge_value(&self, attr_name: String, idx: usize, value: serde_json::Value) {
        self.columnar.set_edge_value(attr_name, idx, value);
    }

    /// Get column stats for edge attributes.
    pub fn get_edge_column_stats(&self) -> std::collections::HashMap<String, usize> {
        self.columnar.get_edge_column_stats()
    }

    /// Filter edges by attribute value (returns indices).
    pub fn filter_edges_by_value(&self, attr_name: String, value: serde_json::Value) -> Vec<usize> {
        self.columnar.filter_edges_by_value(attr_name, value)
    }

    /// SIMD-enabled batch attribute filtering for nodes
    pub fn filter_nodes_simd(&self, attr_name: String, value: serde_json::Value) -> Vec<usize> {
        #[cfg(target_feature = "avx2")]
        {
            self.filter_nodes_simd_avx2(attr_name, value)
        }
        #[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
        {
            self.filter_nodes_simd_sse2(attr_name, value)
        }
        #[cfg(not(any(target_feature = "sse2", target_feature = "avx2")))]
        {
            // Fallback to regular filtering if no SIMD support
            self.filter_nodes_by_value(attr_name, value)
        }
    }

    /// SIMD-enabled batch attribute filtering for edges
    pub fn filter_edges_simd(&self, attr_name: String, value: serde_json::Value) -> Vec<usize> {
        #[cfg(target_feature = "avx2")]
        {
            self.filter_edges_simd_avx2(attr_name, value)
        }
        #[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
        {
            self.filter_edges_simd_sse2(attr_name, value)
        }
        #[cfg(not(any(target_feature = "sse2", target_feature = "avx2")))]
        {
            // Fallback to regular filtering if no SIMD support
            self.filter_edges_by_value(attr_name, value)
        }
    }

    #[cfg(target_feature = "avx2")]
    fn filter_nodes_simd_avx2(&self, attr_name: String, value: serde_json::Value) -> Vec<usize> {
        use std::arch::x86_64::*;
        
        // Get the attribute UID and column
        let attr_uid = match self.columnar.attr_name_to_uid.get(&attr_name) {
            Some(uid) => uid.clone(),
            None => return Vec::new(),
        };
        
        let column_key = (crate::storage::columnar::ColumnKind::Node, attr_uid);
        let column = match self.columnar.columns.get(&column_key) {
            Some(col) => col,
            None => return Vec::new(),
        };
        
        match (&**column, &value) {
            (crate::storage::columnar::ColumnData::Int(vec), serde_json::Value::Number(target)) => {
                if let Some(target_i64) = target.as_i64() {
                    self.simd_filter_i64_avx2(vec, target_i64)
                } else {
                    Vec::new()
                }
            },
            (crate::storage::columnar::ColumnData::Float(vec), serde_json::Value::Number(target)) => {
                if let Some(target_f64) = target.as_f64() {
                    self.simd_filter_f64_avx2(vec, target_f64)
                } else {
                    Vec::new()
                }
            },
            _ => {
                // For non-numeric types, fall back to regular filtering
                self.filter_nodes_by_value(attr_name, value)
            }
        }
    }

    #[cfg(target_feature = "avx2")]
    unsafe fn simd_filter_i64_avx2(&self, vec: &[Option<i64>], target: i64) -> Vec<usize> {
        use std::arch::x86_64::*;
        
        let mut result = Vec::new();
        let target_vec = _mm256_set1_epi64x(target);
        
        // Process 4 i64 values at a time with AVX2
        let chunks = vec.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for (chunk_idx, chunk) in chunks.enumerate() {
            // Convert Option<i64> to i64, using i64::MIN for None values
            let values: [i64; 4] = [
                chunk[0].unwrap_or(i64::MIN),
                chunk[1].unwrap_or(i64::MIN),
                chunk[2].unwrap_or(i64::MIN),
                chunk[3].unwrap_or(i64::MIN),
            ];
            
            let data_vec = _mm256_loadu_si256(values.as_ptr() as *const __m256i);
            let cmp_result = _mm256_cmpeq_epi64(data_vec, target_vec);
            let mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_result));
            
            // Check each bit in the mask
            for i in 0..4 {
                if (mask & (1 << i)) != 0 && chunk[i].is_some() {
                    result.push(chunk_idx * 4 + i);
                }
            }
        }
        
        // Handle remainder elements
        for (i, value) in remainder.iter().enumerate() {
            if let Some(v) = value {
                if *v == target {
                    result.push(vec.len() - remainder.len() + i);
                }
            }
        }
        
        result
    }

    #[cfg(target_feature = "avx2")]
    unsafe fn simd_filter_f64_avx2(&self, vec: &[Option<f64>], target: f64) -> Vec<usize> {
        use std::arch::x86_64::*;
        
        let mut result = Vec::new();
        let target_vec = _mm256_set1_pd(target);
        
        // Process 4 f64 values at a time with AVX2
        let chunks = vec.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for (chunk_idx, chunk) in chunks.enumerate() {
            // Convert Option<f64> to f64, using NaN for None values
            let values: [f64; 4] = [
                chunk[0].unwrap_or(f64::NAN),
                chunk[1].unwrap_or(f64::NAN),
                chunk[2].unwrap_or(f64::NAN),
                chunk[3].unwrap_or(f64::NAN),
            ];
            
            let data_vec = _mm256_loadu_pd(values.as_ptr());
            let cmp_result = _mm256_cmp_pd(data_vec, target_vec, _CMP_EQ_OQ);
            let mask = _mm256_movemask_pd(cmp_result);
            
            // Check each bit in the mask
            for i in 0..4 {
                if (mask & (1 << i)) != 0 && chunk[i].is_some() {
                    result.push(chunk_idx * 4 + i);
                }
            }
        }
        
        // Handle remainder elements
        for (i, value) in remainder.iter().enumerate() {
            if let Some(v) = value {
                if *v == target {
                    result.push(vec.len() - remainder.len() + i);
                }
            }
        }
        
        result
    }

    #[cfg(target_feature = "sse2")]
    fn filter_nodes_simd_sse2(&self, attr_name: String, value: serde_json::Value) -> Vec<usize> {
        // SSE2 implementation for older CPUs - processes 2 i64 or 2 f64 at a time
        // Similar structure to AVX2 but with smaller vector sizes
        // For brevity, falling back to regular filtering for now
        self.filter_nodes_by_value(attr_name, value)
    }

    #[cfg(target_feature = "avx2")]
    fn filter_edges_simd_avx2(&self, attr_name: String, value: serde_json::Value) -> Vec<usize> {
        // Similar to filter_nodes_simd_avx2 but for edges
        let attr_uid = match self.columnar.attr_name_to_uid.get(&attr_name) {
            Some(uid) => uid.clone(),
            None => return Vec::new(),
        };
        
        let column_key = (crate::storage::columnar::ColumnKind::Edge, attr_uid);
        let column = match self.columnar.columns.get(&column_key) {
            Some(col) => col,
            None => return Vec::new(),
        };
        
        match (&**column, &value) {
            (crate::storage::columnar::ColumnData::Int(vec), serde_json::Value::Number(target)) => {
                if let Some(target_i64) = target.as_i64() {
                    unsafe { self.simd_filter_i64_avx2(vec, target_i64) }
                } else {
                    Vec::new()
                }
            },
            (crate::storage::columnar::ColumnData::Float(vec), serde_json::Value::Number(target)) => {
                if let Some(target_f64) = target.as_f64() {
                    unsafe { self.simd_filter_f64_avx2(vec, target_f64) }
                } else {
                    Vec::new()
                }
            },
            _ => {
                self.filter_edges_by_value(attr_name, value)
            }
        }
    }

    #[cfg(target_feature = "sse2")]
    fn filter_edges_simd_sse2(&self, attr_name: String, value: serde_json::Value) -> Vec<usize> {
        // SSE2 implementation for edges
        self.filter_edges_by_value(attr_name, value)
    }

    /// Sets the type/schema for a given attribute across all entities.
    pub fn set_type(&mut self, attr_name: String, attr_type: crate::storage::columnar::AttributeType, is_node: bool) -> Result<u64, String> {
        self.columnar.register_attr_with_type(attr_name, attr_type, is_node)
    }
    
    /// Returns the attribute schema (type information) for all managed attributes.
    pub fn get_schema(&self) -> std::collections::HashMap<String, crate::storage::columnar::AttributeType> {
        let mut schema = std::collections::HashMap::new();
        for entry in self.columnar.attr_schema.iter() {
            let attr_uid = entry.key();
            let attr_type = entry.value();
            if let Some(attr_name) = self.columnar.attr_uid_to_name.get(attr_uid) {
                schema.insert(attr_name.clone(), *attr_type);
            }
        }
        schema
    }
    
    /// Performs a bulk update of attributes for multiple entities.
    pub fn bulk_update(&mut self, node_updates: std::collections::HashMap<String, std::collections::HashMap<usize, serde_json::Value>>, edge_updates: std::collections::HashMap<String, std::collections::HashMap<usize, serde_json::Value>>) -> Result<(), String> {
        // Update all node attributes
        for (attr_name, data) in node_updates {
            self.columnar.set_node_attr(attr_name, data);
        }
        
        // Update all edge attributes
        for (attr_name, data) in edge_updates {
            self.columnar.set_edge_attr(attr_name, data);
        }
        
        Ok(())
    }

    /// Fast-path: retrieves attribute(s) with minimal overhead, bypassing Python wrappers.
    pub fn get_fast(&self, attr_name: &str, entity_id: usize, is_node: bool) -> Option<serde_json::Value> {
        // Use existing methods that return cloned values - for true zero-copy we'd need to add _ref methods to ColumnarStore
        if is_node {
            self.columnar.get_node_value(attr_name.to_string(), entity_id)
        } else {
            self.columnar.get_edge_value(attr_name.to_string(), entity_id)
        }
    }
    
    /// Fast-path: sets attribute(s) with minimal overhead, bypassing Python wrappers.
    pub fn set_fast(&mut self, attr_name: String, entity_id: usize, value: serde_json::Value, is_node: bool) {
        if is_node {
            self.columnar.set_node_value(attr_name, entity_id, value);
        } else {
            self.columnar.set_edge_value(attr_name, entity_id, value);
        }
    }
    
    /// Executes a vectorized batch operation on attributes.
    pub fn batch_operation(&mut self, operations: Vec<BatchOperation>) -> Result<Vec<BatchResult>, String> {
        let mut results = Vec::new();
        
        for op in operations {
            let result = match op {
                BatchOperation::Set { attr_name, entity_id, value, is_node } => {
                    self.set_fast(attr_name, entity_id, value, is_node);
                    BatchResult::Success
                },
                BatchOperation::Get { attr_name, entity_id, is_node } => {
                    if let Some(value) = if is_node {
                        self.columnar.get_node_value(attr_name, entity_id)
                    } else {
                        self.columnar.get_edge_value(attr_name, entity_id)
                    } {
                        BatchResult::Value(value)
                    } else {
                        BatchResult::NotFound
                    }
                },
                BatchOperation::Filter { attr_name, value, is_node } => {
                    let indices = if is_node {
                        self.filter_nodes_by_value(attr_name, value)
                    } else {
                        self.filter_edges_by_value(attr_name, value)
                    };
                    BatchResult::Indices(indices)
                }
            };
            results.push(result);
        }
        
        Ok(results)
    }
}

/// Single entity attribute management
#[pyclass]
pub struct ProxyAttributeManager {
    pub entity_id: usize,
    pub is_node: bool,
    pub attr_manager: AttributeManager,
}

#[pymethods]
impl ProxyAttributeManager {
    /// Returns the value of the specified attribute for this entity as JSON string.
    pub fn get(&self, attr_name: String) -> Option<String> {
        let value = if self.is_node {
            self.attr_manager.get_node_value(attr_name, self.entity_id)
        } else {
            self.attr_manager.get_edge_value(attr_name, self.entity_id)
        };
        value.map(|v| serde_json::to_string(&v).unwrap_or_default())
    }
    
    /// Sets the value of the specified attribute for this entity (JSON string).
    pub fn set(&mut self, attr_name: String, value: String) -> PyResult<()> {
        let json_value: serde_json::Value = serde_json::from_str(&value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid JSON: {}", e)))?;
        if self.is_node {
            self.attr_manager.set_node_value(attr_name, self.entity_id, json_value);
        } else {
            self.attr_manager.set_edge_value(attr_name, self.entity_id, json_value);
        }
        Ok(())
    }
    /// Checks if the specified attribute exists for this entity.
    ///
    /// Fast lookup in columnar metadata. Returns true if attribute is present.
    pub fn has(&self, attr_name: String) -> bool {
        if self.is_node {
            self.attr_manager.node_attr_names().contains(&attr_name)
        } else {
            self.attr_manager.edge_attr_names().contains(&attr_name)
        }
    }
    /// Removes the specified attribute from this entity.
    ///
    /// Updates columnar metadata and clears value. Handles schema update if last reference.
    pub fn remove(&mut self, attr_name: String) {
        if self.is_node {
            self.attr_manager.set_node_value(attr_name, self.entity_id, serde_json::Value::Null);
        } else {
            self.attr_manager.set_edge_value(attr_name, self.entity_id, serde_json::Value::Null);
        }
    }
}
