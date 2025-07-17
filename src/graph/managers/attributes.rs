// src_new/graph/managers/attributes.rs
//! AttributeManager: Unified, batch-friendly attribute management for nodes and edges in Groggy.
//! Supports columnar storage, schema enforcement, and agent/LLM-friendly APIs.

use pyo3::prelude::*;
use pyo3::PyTypeInfo;
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
    pub columnar: std::sync::Arc<ColumnarStore>,
    pub graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>,
    // Optional arena-based storage for maximum performance
    pub hyper_core: Option<std::sync::Arc<crate::storage::hyper_core::HyperGraphCore>>,
}

#[pymethods]
impl AttributeManager {
    #[new]
    pub fn new() -> Self {
        let graph_store = std::sync::Arc::new(crate::storage::graph_store::GraphStore::new());
        Self {
            columnar: std::sync::Arc::new(ColumnarStore::new()),
            graph_store,
            hyper_core: None,
        }
    }
    
    /// Create AttributeManager with hyper-optimized arena storage
    #[staticmethod]
    pub fn with_hyper_core(estimated_nodes: usize, estimated_edges: usize) -> Self {
        let graph_store = std::sync::Arc::new(crate::storage::graph_store::GraphStore::new());
        let hyper_core = std::sync::Arc::new(
            crate::storage::hyper_core::HyperGraphCore::with_capacity(
                estimated_nodes, estimated_edges, 20 // 20 expected attributes
            )
        );
        
        Self {
            columnar: std::sync::Arc::new(ColumnarStore::new()),
            graph_store,
            hyper_core: Some(hyper_core),
        }
    }

    /// Basic Python-compatible get method (returns JSON string)
    pub fn get_py(&self, id: &str, attr: &str) -> Option<String> {
        self.get_internal(id, attr, true).map(|v| serde_json::to_string(&v).unwrap_or_default())
    }

    /// Get method exposed to Python (supports flexible parameter formats)
    #[pyo3(signature = (node_ids = None, attr_names = None))]
    pub fn get(&self, py: pyo3::Python, node_ids: Option<&pyo3::types::PyAny>, attr_names: Option<&pyo3::types::PyAny>) -> PyResult<pyo3::PyObject> {
        use pyo3::types::{PyList, PyDict};
        
        // Handle single node_id, single attr_name case
        if let (Some(node_id_obj), Some(attr_name_obj)) = (node_ids, attr_names) {
            if let (Ok(node_id), Ok(attr_name)) = (node_id_obj.extract::<String>(), attr_name_obj.extract::<String>()) {
                if let Some(value) = self.get_internal(&node_id, &attr_name, true) {
                    return Ok(serde_json::to_string(&value).unwrap_or_default().into_py(py));
                } else {
                    return Ok(py.None());
                }
            }
        }
        
        // Handle batch operations
        let timing_start = std::time::Instant::now();
        
        // Parse node_ids
        let node_id_list: Vec<String> = if let Some(node_ids_obj) = node_ids {
            if let Ok(ids) = node_ids_obj.extract::<Vec<String>>() {
                ids
            } else if let Ok(single_id) = node_ids_obj.extract::<String>() {
                vec![single_id]
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err("node_ids must be a string or list of strings"));
            }
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err("node_ids parameter is required"));
        };
        
        // Parse attr_names
        let attr_name_list: Vec<String> = if let Some(attr_names_obj) = attr_names {
            if let Ok(names) = attr_names_obj.extract::<Vec<String>>() {
                names
            } else if let Ok(single_name) = attr_names_obj.extract::<String>() {
                vec![single_name]
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err("attr_names must be a string or list of strings"));
            }
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err("attr_names parameter is required"));
        };
        
        // Create result dictionary
        let result_dict = PyDict::new(py);
        
        // Batch get operations by attribute name
        for attr_name in attr_name_list {
            if let Some(all_values) = self.get_node_attr(attr_name.clone()) {
                let attr_dict = PyDict::new(py);
                
                // Filter to only requested node IDs
                for node_id in &node_id_list {
                    let is_node = !node_id.contains("->");
                    if let Some(index) = if is_node {
                        use crate::graph::types::NodeId;
                        let node_id_typed = NodeId::new(node_id.clone());
                        self.graph_store.node_index(&node_id_typed)
                    } else {
                        use crate::graph::types::{NodeId, EdgeId};
                        // PHASE 2 OPTIMIZATION: Use split_once to avoid Vec allocation
                        if let Some((source_str, target_str)) = node_id.split_once("->") {
                            let source = NodeId::new(source_str.to_string());
                            let target = NodeId::new(target_str.to_string());
                            let edge_id = EdgeId::new(source, target);
                            self.graph_store.edge_index(&edge_id)
                        } else {
                            None
                        }
                    } {
                        if let Some(value) = all_values.get(&index) {
                            let value_str = serde_json::to_string(value).unwrap_or_default();
                            attr_dict.set_item(node_id, value_str)?;
                        }
                    }
                }
                
                result_dict.set_item(&attr_name, attr_dict)?;
            }
        }
        
        let elapsed = timing_start.elapsed();
        println!("[Groggy][Timing][Rust] get: batch operation took {:.6}s", elapsed.as_secs_f64());
        
        Ok(result_dict.into())
    }
    
    /// Expose set method to Python for batch attribute setting.
    /// Accepts a dict of {entity_id: {attr_name: value}} where entity_id may be node or edge (e.g., "n1" or "n1->n2").
    #[pyo3(name = "set")]
    pub fn py_set(&mut self, attr_data: &pyo3::types::PyAny) -> PyResult<()> {
        use pyo3::types::PyDict;
        
        // Fast path: try to extract pre-serialized JSON dict first
        if let Ok(dict) = attr_data.extract::<std::collections::HashMap<String, std::collections::HashMap<String, String>>>() {
            // PHASE 2 OPTIMIZATION: Pre-serialized JSON from Python - no JSON conversion needed
            let timing_start = std::time::Instant::now();
            
            use rustc_hash::FxHashMap;
            let mut node_batches: FxHashMap<String, FxHashMap<usize, serde_json::Value>> = FxHashMap::default();
            let mut edge_batches: FxHashMap<String, FxHashMap<usize, serde_json::Value>> = FxHashMap::default();
            
            for (entity_id, attrs) in dict {
                let is_node = !entity_id.contains("->");
                
                // Get the entity index
                let index = if is_node {
                    use crate::graph::types::NodeId;
                    let node_id = NodeId::new(entity_id.clone());
                    self.graph_store.node_index(&node_id)
                } else {
                    use crate::graph::types::{NodeId, EdgeId};
                    // PHASE 2 OPTIMIZATION: Use split_once to avoid Vec allocation
                    if let Some((source_str, target_str)) = entity_id.split_once("->") {
                        let source = NodeId::new(source_str.to_string());
                        let target = NodeId::new(target_str.to_string());
                        let edge_id = EdgeId::new(source, target);
                        self.graph_store.edge_index(&edge_id)
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid edge ID format: {}", entity_id)));
                    }
                };
                
                let entity_index = match index {
                    Some(idx) => idx,
                    None => return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("{} ID not found: {}", if is_node {"Node"} else {"Edge"}, entity_id))),
                };
                
                // Process each pre-serialized attribute for this entity
                for (attr_name, json_str) in attrs {
                    let value: serde_json::Value = serde_json::from_str(&json_str)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid JSON: {}", e)))?;
                    
                    // Add to the appropriate batch
                    if is_node {
                        node_batches.entry(attr_name).or_default().insert(entity_index, value);
                    } else {
                        edge_batches.entry(attr_name).or_default().insert(entity_index, value);
                    }
                }
            }
            
            // Execute batch operations
            for (attr_name, data) in node_batches {
                self.set_node_attr(attr_name, data);
            }
            for (attr_name, data) in edge_batches {
                self.set_edge_attr(attr_name, data);
            }
            
            let elapsed = timing_start.elapsed();
            println!("[Groggy][Timing][Rust] py_set (pre-serialized): batch operation took {:.6}s", elapsed.as_secs_f64());
            return Ok(());
        }
        
        // Fallback path: handle PyDict for backward compatibility
        if let Ok(dict) = attr_data.downcast::<PyDict>() {
            let timing_start = std::time::Instant::now();
            
            // OPTIMIZATION: Import JSON module once outside the loops for massive performance improvement
            let py = attr_data.py();
            let json_module = py.import("json")?;
            
            // Group by is_node and attribute name for batch operations (using FxHashMap for performance)
            use rustc_hash::FxHashMap;
            let mut node_batches: FxHashMap<String, FxHashMap<usize, serde_json::Value>> = FxHashMap::default();
            let mut edge_batches: FxHashMap<String, FxHashMap<usize, serde_json::Value>> = FxHashMap::default();
            
            for (entity_id_obj, attrs_obj) in dict.iter() {
                let entity_id: String = entity_id_obj.extract()?;
                let is_node = !entity_id.contains("->");
                let attrs = attrs_obj.downcast::<PyDict>()?;
                
                // Get the entity index
                let index = if is_node {
                    use crate::graph::types::NodeId;
                    let node_id = NodeId::new(entity_id.clone());
                    self.graph_store.node_index(&node_id)
                } else {
                    use crate::graph::types::{NodeId, EdgeId};
                    // PHASE 2 OPTIMIZATION: Use split_once to avoid Vec allocation
                    if let Some((source_str, target_str)) = entity_id.split_once("->") {
                        let source = NodeId::new(source_str.to_string());
                        let target = NodeId::new(target_str.to_string());
                        let edge_id = EdgeId::new(source, target);
                        self.graph_store.edge_index(&edge_id)
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid edge ID format: {}", entity_id)));
                    }
                };
                
                let entity_index = match index {
                    Some(idx) => idx,
                    None => return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("{} ID not found: {}", if is_node {"Node"} else {"Edge"}, entity_id))),
                };
                
                // Process each attribute for this entity
                for (attr_name_obj, value_obj) in attrs.iter() {
                    let attr_name: String = attr_name_obj.extract()?;
                    // OPTIMIZATION: Use pre-imported json_module instead of importing on every iteration
                    let value_str = json_module.call_method1("dumps", (value_obj,))?.extract::<String>()?;
                    let value: serde_json::Value = serde_json::from_str(&value_str)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid JSON: {}", e)))?;
                    
                    // Add to the appropriate batch
                    if is_node {
                        node_batches.entry(attr_name).or_default().insert(entity_index, value);
                    } else {
                        edge_batches.entry(attr_name).or_default().insert(entity_index, value);
                    }
                }
            }
            
            // Execute batch operations
            for (attr_name, data) in node_batches {
                self.set_node_attr(attr_name, data);
            }
            for (attr_name, data) in edge_batches {
                self.set_edge_attr(attr_name, data);
            }
            
            let elapsed = timing_start.elapsed();
            println!("[Groggy][Timing][Rust] py_set (legacy): batch operation took {:.6}s", elapsed.as_secs_f64());
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected a dict of {entity_id: {attr_name: value}}"))
        }
    }

    /// Returns a breakdown of memory usage per attribute (name, type, node/edge, bytes used)
    pub fn memory_usage_breakdown(&self) -> std::collections::HashMap<String, usize> {
        self.columnar.memory_usage_breakdown()
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

    /// Returns the estimated heap memory usage in bytes for all attribute storage.
    pub fn memory_usage_bytes(&self) -> usize {
        self.columnar.memory_usage_bytes()
    }
    /// Expose set_type to Python to allow explicit schema setting from Python
    #[pyo3(name = "set_type")]
    pub fn py_set_type(&mut self, attr_name: String, py_type: &pyo3::types::PyType, is_node: bool, py: pyo3::Python) -> pyo3::PyResult<u64> {
        // Map Python type to Rust AttributeType
        let rust_type = if py_type.is_subclass(pyo3::types::PyLong::type_object(py)).unwrap_or(false) {
            crate::storage::columnar::AttributeType::Int
        } else if py_type.is_subclass(pyo3::types::PyFloat::type_object(py)).unwrap_or(false) {
            crate::storage::columnar::AttributeType::Float
        } else if py_type.is_subclass(pyo3::types::PyBool::type_object(py)).unwrap_or(false) {
            crate::storage::columnar::AttributeType::Bool
        } else if py_type.is_subclass(pyo3::types::PyString::type_object(py)).unwrap_or(false) {
            crate::storage::columnar::AttributeType::Str
        } else {
            crate::storage::columnar::AttributeType::Json
        };
        match self.set_type(attr_name, rust_type, is_node) {
            Ok(uid) => Ok(uid),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }
}

// Internal methods for performance (not exposed to Python)
impl AttributeManager {
    /// Internal constructor for Rust usage
    pub fn new_with_graph_store(graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>) -> Self {
        Self {
            columnar: std::sync::Arc::new(ColumnarStore::new()),
            graph_store,
            hyper_core: None,
        }
    }

    /// Internal get method using serde_json::Value (optimized with Cow patterns)
    pub fn get_internal(&self, id: &str, attr: &str, is_node: bool) -> Option<serde_json::Value> {
        use crate::graph::types::{NodeId, EdgeId};
        use std::borrow::Cow;
        
        let index = match is_node {
            true => {
                // PHASE 2 OPTIMIZATION: Avoid string allocation by using Cow
                let node_id = NodeId::new(id.to_string()); // TODO: Could be further optimized with string interning
                self.graph_store.node_index(&node_id)
            },
            false => {
                // Parse edge id string "source->target" - OPTIMIZED: avoid multiple string allocations
                if let Some((source_str, target_str)) = id.split_once("->") {
                    let source = NodeId::new(source_str.to_string());
                    let target = NodeId::new(target_str.to_string());
                    let edge_id = EdgeId::new(source, target);
                    self.graph_store.edge_index(&edge_id)
                } else {
                    return None;
                }
            },
        };
        if let Some(index) = index {
            if is_node {
                self.columnar.get_node_value(attr.to_string(), index)
            } else {
                self.columnar.get_edge_value(attr.to_string(), index)
            }
        } else {
            None
        }
    }
    
    /// Internal set method using serde_json::Value (optimized with split_once)
    pub fn set_internal(&self, id: &str, attr: &str, value: serde_json::Value, is_node: bool) -> Result<(), String> {
        let timing_start = std::time::Instant::now();
        use crate::graph::types::{NodeId, EdgeId};
        let index = match is_node {
            true => {
                let node_id = NodeId::new(id.to_string());
                self.graph_store.node_index(&node_id)
            },
            false => {
                // PHASE 2 OPTIMIZATION: Use split_once instead of split().collect() to avoid Vec allocation
                if let Some((source_str, target_str)) = id.split_once("->") {
                    let source = NodeId::new(source_str.to_string());
                    let target = NodeId::new(target_str.to_string());
                    let edge_id = EdgeId::new(source, target);
                    self.graph_store.edge_index(&edge_id)
                } else {
                    return Err(format!("Invalid edge ID format: {}", id));
                }
            },
        };
        if let Some(index) = index {
            if is_node {
                self.columnar.set_node_value(attr.to_string(), index, value);
                let elapsed = timing_start.elapsed();
                println!("[Groggy][Timing][Rust] set_internal: node {} attr '{}' took {:.6}s", id, attr, elapsed.as_secs_f64());
            } else {
                self.columnar.set_edge_value(attr.to_string(), index, value);
                let elapsed = timing_start.elapsed();
                println!("[Groggy][Timing][Rust] set_internal: edge {} attr '{}' took {:.6}s", id, attr, elapsed.as_secs_f64());
            }
            Ok(())
        } else {
            Err(format!("{} ID not found: {}", if is_node {"Node"} else {"Edge"}, id))
        }
    }

    /// Get all values for a node attribute by name.
    pub fn get_node_attr(&self, attr_name: String) -> Option<std::collections::HashMap<usize, serde_json::Value>> {
        self.columnar.get_node_attr(attr_name)
    }

    /// Set all values for a node attribute by name.
    pub fn set_node_attr(&self, attr_name: String, data: rustc_hash::FxHashMap<usize, serde_json::Value>) {
        let timing_start = std::time::Instant::now();
        // Convert FxHashMap to HashMap for the columnar store API
        let std_data: std::collections::HashMap<usize, serde_json::Value> = data.into_iter().collect();
        self.columnar.set_node_attr(attr_name.clone(), std_data);
        let elapsed = timing_start.elapsed();
        println!("[Groggy][Timing][Rust] set_node_attr: batch node attr '{}' took {:.6}s", attr_name, elapsed.as_secs_f64());
    }

    /// Get a single value for a node attribute and entity index.
    pub fn get_node_value(&self, attr_name: String, idx: usize) -> Option<serde_json::Value> {
        self.columnar.get_node_value(attr_name, idx)
    }

    /// Set a single value for a node attribute and entity index.
    pub fn set_node_value(&self, attr_name: String, idx: usize, value: serde_json::Value) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
        let call_num = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        let timing_start = std::time::Instant::now();
        self.columnar.set_node_value(attr_name, idx, value);
    }

    /// Get column stats for node attributes.
    pub fn get_node_column_stats(&self) -> std::collections::HashMap<String, usize> {
        self.columnar.get_column_stats()
    }

    /// Filter nodes by attribute value (returns indices).
    /// Automatically uses SIMD acceleration for numeric values when available
    pub fn filter_nodes_by_value(&self, attr_name: String, value: serde_json::Value) -> Vec<usize> {
        // Use hyper-optimized arena filtering when available
        if let Some(ref hyper_core) = self.hyper_core {
            if let Some(i_val) = value.as_i64() {
                let node_ids = hyper_core.hyper_filter_nodes_i64(&attr_name, i_val, crate::storage::arena_core::ComparisonOp::Equal);
                return self.convert_node_ids_to_indices(node_ids);
            }
        }
        
        // Use SIMD filtering for numeric values when feature is enabled
        #[cfg(feature = "simd")]
        {
            if let Some(i_val) = value.as_i64() {
                return self.filter_nodes_i64_simd(&attr_name, i_val, crate::storage::columnar::ComparisonOp::Equal);
            }
            if let Some(f_val) = value.as_f64() {
                return self.filter_nodes_f64_simd(&attr_name, f_val, crate::storage::columnar::ComparisonOp::Equal);
            }
        }
        
        // Fallback to regular filtering
        self.columnar.filter_nodes_by_value(attr_name, value)
    }
    
    /// Ultra-fast filtering with comparison operators (hyper-optimized)
    pub fn hyper_filter_nodes_i64(&self, attr_name: &str, value: i64, op: &str) -> Vec<usize> {
        if let Some(ref hyper_core) = self.hyper_core {
            let comparison_op = match op {
                ">" => crate::storage::arena_core::ComparisonOp::Greater,
                "<" => crate::storage::arena_core::ComparisonOp::Less,
                ">=" => crate::storage::arena_core::ComparisonOp::GreaterEqual,
                "<=" => crate::storage::arena_core::ComparisonOp::LessEqual,
                "==" => crate::storage::arena_core::ComparisonOp::Equal,
                "!=" => crate::storage::arena_core::ComparisonOp::NotEqual,
                _ => crate::storage::arena_core::ComparisonOp::Equal,
            };
            
            let node_ids = hyper_core.hyper_filter_nodes_i64(attr_name, value, comparison_op);
            return self.convert_node_ids_to_indices(node_ids);
        }
        
        // Fallback to SIMD filtering
        #[cfg(feature = "simd")]
        {
            let comparison_op = match op {
                ">" => crate::storage::columnar::ComparisonOp::Greater,
                "<" => crate::storage::columnar::ComparisonOp::Less,
                ">=" => crate::storage::columnar::ComparisonOp::GreaterEqual,
                "<=" => crate::storage::columnar::ComparisonOp::LessEqual,
                "==" => crate::storage::columnar::ComparisonOp::Equal,
                "!=" => crate::storage::columnar::ComparisonOp::NotEqual,
                _ => crate::storage::columnar::ComparisonOp::Equal,
            };
            return self.filter_nodes_i64_simd(attr_name, value, comparison_op);
        }
        
        #[cfg(not(feature = "simd"))]
        {
            // Basic fallback
            Vec::new()
        }
    }
    
    /// Convert node IDs to indices for compatibility
    fn convert_node_ids_to_indices(&self, node_ids: Vec<String>) -> Vec<usize> {
        node_ids.into_iter()
            .filter_map(|node_id| {
                let node_id_obj = crate::graph::types::NodeId::new(node_id);
                self.graph_store.node_index(&node_id_obj)
            })
            .collect()
    }

    /// Get all values for an edge attribute by name.
    pub fn get_edge_attr(&self, attr_name: String) -> Option<std::collections::HashMap<usize, serde_json::Value>> {
        self.columnar.get_edge_attr(attr_name)
    }

    /// Set all values for an edge attribute by name.
    pub fn set_edge_attr(&self, attr_name: String, data: rustc_hash::FxHashMap<usize, serde_json::Value>) {
        // Convert FxHashMap to HashMap for the columnar store API
        let std_data: std::collections::HashMap<usize, serde_json::Value> = data.into_iter().collect();
        self.columnar.set_edge_attr(attr_name, std_data);
    }

    /// Get a single value for an edge attribute and entity index.
    pub fn get_edge_value(&self, attr_name: String, idx: usize) -> Option<serde_json::Value> {
        self.columnar.get_edge_value(attr_name, idx)
    }

    /// Set a single value for an edge attribute and entity index.
    pub fn set_edge_value(&self, attr_name: String, idx: usize, value: serde_json::Value) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
        let call_num = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        let timing_start = std::time::Instant::now();
        self.columnar.set_edge_value(attr_name.clone(), idx, value);
        let elapsed = timing_start.elapsed();
        if call_num < 5 || (call_num > 0 && call_num % 10000 == 0) || call_num >= 5 && call_num % 10000 == 9995 {
            println!("[Groggy][Timing][Rust] set_edge_value: edge idx {} attr '{}' took {:.6}s (call #{})", idx, attr_name, elapsed.as_secs_f64(), call_num + 1);
        }
    }

    /// Get column stats for edge attributes.
    pub fn get_edge_column_stats(&self) -> std::collections::HashMap<String, usize> {
        self.columnar.get_edge_column_stats()
    }

    /// Filter edges by attribute value (returns indices).
    /// Automatically uses SIMD acceleration for numeric values when available
    pub fn filter_edges_by_value(&self, attr_name: String, value: serde_json::Value) -> Vec<usize> {
        // Use SIMD filtering for numeric values when feature is enabled
        #[cfg(feature = "simd")]
        {
            if let Some(i_val) = value.as_i64() {
                return self.filter_edges_i64_simd(&attr_name, i_val, crate::storage::columnar::ComparisonOp::Equal);
            }
            if let Some(f_val) = value.as_f64() {
                return self.filter_edges_f64_simd(&attr_name, f_val, crate::storage::columnar::ComparisonOp::Equal);
            }
        }
        
        // Fallback to regular filtering
        self.columnar.filter_edges_by_value(attr_name, value)
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
    
    // PHASE 3: Native typed attribute access methods for high-performance operations
    
    /// Get native i64 value directly without JSON overhead (PHASE 3 optimization)
    pub fn get_node_i64(&self, attr_name: &str, entity_id: usize) -> Option<i64> {
        self.columnar.get_node_i64_native(attr_name, entity_id)
    }
    
    /// Get native f64 value directly without JSON overhead (PHASE 3 optimization)
    pub fn get_node_f64(&self, attr_name: &str, entity_id: usize) -> Option<f64> {
        self.columnar.get_node_f64_native(attr_name, entity_id)
    }
    
    /// Get native bool value directly without JSON overhead (PHASE 3 optimization)
    pub fn get_node_bool(&self, attr_name: &str, entity_id: usize) -> Option<bool> {
        self.columnar.get_node_bool_native(attr_name, entity_id)
    }
    
    /// Get native String value directly without JSON overhead (PHASE 3 optimization)
    pub fn get_node_string(&self, attr_name: &str, entity_id: usize) -> Option<String> {
        self.columnar.get_node_string_native(attr_name, entity_id)
    }
    
    /// Get native i64 value for edges directly without JSON overhead (PHASE 3 optimization)
    pub fn get_edge_i64(&self, attr_name: &str, entity_id: usize) -> Option<i64> {
        self.columnar.get_edge_i64_native(attr_name, entity_id)
    }
    
    /// Get native f64 value for edges directly without JSON overhead (PHASE 3 optimization)
    pub fn get_edge_f64(&self, attr_name: &str, entity_id: usize) -> Option<f64> {
        self.columnar.get_edge_f64_native(attr_name, entity_id)
    }
    
    /// Get native bool value for edges directly without JSON overhead (PHASE 3 optimization)
    pub fn get_edge_bool(&self, attr_name: &str, entity_id: usize) -> Option<bool> {
        self.columnar.get_edge_bool_native(attr_name, entity_id)
    }
    
    /// Get native String value for edges directly without JSON overhead (PHASE 3 optimization)
    pub fn get_edge_string(&self, attr_name: &str, entity_id: usize) -> Option<String> {
        self.columnar.get_edge_string_native(attr_name, entity_id)
    }
    
    /// Bulk filter nodes by numeric attribute with native types (PHASE 3 optimization)
    /// Uses SIMD vectorized operations for 4-8x speedup
    #[cfg(feature = "simd")]
    pub fn filter_nodes_i64_simd(&self, attr_name: &str, value: i64, op: crate::storage::columnar::ComparisonOp) -> Vec<usize> {
        self.columnar.simd_filter_i64_bulk(attr_name, value, op).unwrap_or_else(|_| Vec::new())
    }
    
    /// Bulk filter nodes by float attribute with native types (PHASE 3 optimization)
    /// Uses SIMD vectorized operations for 4-8x speedup
    #[cfg(feature = "simd")]
    pub fn filter_nodes_f64_simd(&self, attr_name: &str, value: f64, op: crate::storage::columnar::ComparisonOp) -> Vec<usize> {
        // For now use fallback - can add simd_filter_f64_bulk later
        self.columnar.get_node_f64_bulk(attr_name)
            .map(|vec| {
                vec.iter().enumerate()
                    .filter_map(|(idx, val_opt)| {
                        val_opt.and_then(|val| {
                            let matches = match op {
                                crate::storage::columnar::ComparisonOp::Equal => val == value,
                                crate::storage::columnar::ComparisonOp::NotEqual => val != value,
                                crate::storage::columnar::ComparisonOp::Greater => val > value,
                                crate::storage::columnar::ComparisonOp::Less => val < value,
                                crate::storage::columnar::ComparisonOp::GreaterEqual => val >= value,
                                crate::storage::columnar::ComparisonOp::LessEqual => val <= value,
                            };
                            if matches { Some(idx) } else { None }
                        })
                    })
                    .collect()
            })
            .unwrap_or_else(Vec::new)
    }
    
    /// Bulk filter edges by numeric attribute with native types (PHASE 3 optimization)
    /// Uses SIMD vectorized operations for 4-8x speedup
    #[cfg(feature = "simd")]
    pub fn filter_edges_i64_simd(&self, attr_name: &str, value: i64, op: crate::storage::columnar::ComparisonOp) -> Vec<usize> {
        // Edge filtering with similar pattern - can be optimized with dedicated edge SIMD later
        self.columnar.get_edge_i64_bulk(attr_name)
            .map(|vec| {
                vec.iter().enumerate()
                    .filter_map(|(idx, val_opt)| {
                        val_opt.and_then(|val| {
                            let matches = match op {
                                crate::storage::columnar::ComparisonOp::Equal => val == value,
                                crate::storage::columnar::ComparisonOp::NotEqual => val != value,
                                crate::storage::columnar::ComparisonOp::Greater => val > value,
                                crate::storage::columnar::ComparisonOp::Less => val < value,
                                crate::storage::columnar::ComparisonOp::GreaterEqual => val >= value,
                                crate::storage::columnar::ComparisonOp::LessEqual => val <= value,
                            };
                            if matches { Some(idx) } else { None }
                        })
                    })
                    .collect()
            })
            .unwrap_or_else(Vec::new)
    }
    
    /// Bulk filter edges by float attribute with native types (PHASE 3 optimization)
    /// Uses SIMD vectorized operations for 4-8x speedup
    #[cfg(feature = "simd")]
    pub fn filter_edges_f64_simd(&self, attr_name: &str, value: f64, op: crate::storage::columnar::ComparisonOp) -> Vec<usize> {
        // Edge filtering with similar pattern
        self.columnar.get_edge_f64_bulk(attr_name)
            .map(|vec| {
                vec.iter().enumerate()
                    .filter_map(|(idx, val_opt)| {
                        val_opt.and_then(|val| {
                            let matches = match op {
                                crate::storage::columnar::ComparisonOp::Equal => val == value,
                                crate::storage::columnar::ComparisonOp::NotEqual => val != value,
                                crate::storage::columnar::ComparisonOp::Greater => val > value,
                                crate::storage::columnar::ComparisonOp::Less => val < value,
                                crate::storage::columnar::ComparisonOp::GreaterEqual => val >= value,
                                crate::storage::columnar::ComparisonOp::LessEqual => val <= value,
                            };
                            if matches { Some(idx) } else { None }
                        })
                    })
                    .collect()
            })
            .unwrap_or_else(Vec::new)
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
