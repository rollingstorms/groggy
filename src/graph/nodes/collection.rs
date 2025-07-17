// src_new/graph/nodes/collection.rs
//! NodeCollection: concrete implementation of BaseCollection for node storage in Groggy graphs.
//! Provides batch operations, columnar backend, and agent/LLM-friendly APIs.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use pyo3::Python;
use crate::graph::types::NodeId;
use crate::graph::managers::attributes::AttributeManager;
// use crate::graph::columnar::NodeColumnarStore; // Uncomment when available

#[pyclass]
#[derive(Clone)]
pub struct NodeCollection {
    pub attribute_manager: std::sync::Arc<AttributeManager>,
    #[pyo3(get)]
    pub node_ids: Vec<NodeId>,
    pub graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>,
}


#[pymethods]
impl NodeCollection {
    pub fn add(&mut self, py_node_ids: &pyo3::types::PyAny) -> pyo3::PyResult<()> {
        // Fast path: accept pre-processed string list from Python layer
        if let Ok(list) = py_node_ids.extract::<Vec<String>>() {
            let node_ids: Vec<NodeId> = list.into_iter().map(NodeId::new).collect();
            self.add_batch(node_ids)?;
            return Ok(());
        }
        
        // Fallback: handle other types (for direct Rust usage)
        let node_ids: Vec<NodeId> = if let Ok(list) = py_node_ids.extract::<Vec<i64>>() {
            list.into_iter().map(|i| NodeId::new(i.to_string())).collect()
        } else if let Ok(single) = py_node_ids.extract::<String>() {
            vec![NodeId::new(single)]
        } else if let Ok(single) = py_node_ids.extract::<i64>() {
            vec![NodeId::new(single.to_string())]
        } else {
            let actual_type = py_node_ids.get_type().name().unwrap_or("unknown");
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Invalid node ID format: must be str, int, list[str], or list[int]. Got type: {}",
                actual_type
            )));
        };
        self.add_batch(node_ids)?;
        Ok(())
    }
    /// Flexible filter method: supports dict, string, or chaining.
    #[pyo3(signature = (*args, **kwargs))]
    pub fn filter_py(&self, py: Python, args: &pyo3::types::PyTuple, kwargs: Option<&PyDict>) -> Self {
        let mut filtered_ids = self.node_ids.clone();
        // 1. If first arg is dict
        if let Some(first) = args.get_item(0).ok() {
            if let Ok(d) = first.downcast::<PyDict>() {
                filtered_ids = filter_nodes_by_dict(&self.attribute_manager, &self.graph_store, &filtered_ids, d, py);
            } else if let Ok(s) = first.downcast::<PyString>() {
                let query = s.to_str().unwrap_or("");
                filtered_ids = filter_nodes_by_query(&self.attribute_manager, &self.graph_store, &filtered_ids, query);
            }
        } else if let Some(d) = kwargs {
            filtered_ids = filter_nodes_by_dict(&self.attribute_manager, &self.graph_store, &filtered_ids, d, py);
        }
        Self {
            attribute_manager: self.attribute_manager.clone(),
            node_ids: filtered_ids,
            graph_store: self.graph_store.clone(),
        }
    }

    /// Create a new NodeCollection from Python (simplified constructor)
    #[new]
    pub fn py_new() -> Self {
        let graph_store = std::sync::Arc::new(crate::storage::graph_store::GraphStore::new());
        let attribute_manager = AttributeManager::new_with_graph_store(graph_store.clone());
        Self {
            attribute_manager: std::sync::Arc::new(attribute_manager),
            node_ids: Vec::new(),
            graph_store,
        }
    }

    /// Returns an AttributeManager for node attributes.
    pub fn attr(&self) -> AttributeManager {
        (*self.attribute_manager).clone()
    }

    /// Returns the number of nodes in this collection.
    #[getter]
    pub fn size(&self) -> usize {
        self.node_ids.len()
    }

    /// Flexible filter method: supports dict, string, or chaining.
    #[pyo3(name = "filter")]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn filter(&self, py: Python, args: &pyo3::types::PyTuple, kwargs: Option<&PyDict>) -> Self {
        self.filter_py(py, args, kwargs)
    }

    /// Returns all node IDs in this collection.
    pub fn ids(&self) -> Vec<String> {
        self.node_ids.iter().map(|id| id.0.clone()).collect()
    }

    /// Check if a node exists in the collection.
    pub fn has(&self, node_id: String) -> bool {
        let node_id = crate::graph::types::NodeId::new(node_id);
        self.graph_store.has_node(&node_id)
    }

    /// Get a NodeProxy for this node if it exists.
    pub fn get(&self, node_id: String) -> Option<crate::graph::nodes::proxy::NodeProxy> {
        let node_id = crate::graph::types::NodeId::new(node_id);
        if self.graph_store.has_node(&node_id) {
            Some(crate::graph::nodes::proxy::NodeProxy::new(node_id, self.graph_store.clone()))
        } else {
            None
        }
    }

}

// --- Rust-only methods ---
impl NodeCollection {
    /// Add one or more nodes to the collection (batch-oriented, Rust only).
    pub fn add_batch(&mut self, nodes: Vec<NodeId>) -> PyResult<()> {
        self.graph_store.add_nodes(&nodes);
        // MEMORY LEAK FIX: Extend existing Vec instead of replacing entire Vec 
        self.node_ids.extend(nodes);
        Ok(())
    }


    /// Remove one or more nodes from the collection (batch-oriented).
    pub fn remove(&mut self, node_ids: Vec<NodeId>) -> PyResult<()> {
        self.graph_store.remove_nodes(&node_ids);
        // MEMORY LEAK FIX: Remove specific nodes instead of replacing entire Vec
        self.node_ids.retain(|node_id| !node_ids.contains(node_id));
        Ok(())
    }

    /// Returns a FilterManager for this collection, pre-configured for nodes.
    ///
    /// Usage:
    /// let mut fm = collection.filter_manager();
    /// fm.add_filter(...);
    /// let result_ids = fm.apply(collection.ids());
    pub fn filter_manager(&self) -> crate::graph::managers::filter::FilterManager {
        // MEMORY LEAK FIX: Remove unused HashMap that was never dropped
        crate::graph::managers::filter::FilterManager::new((*self.attribute_manager).clone(), true)
    }

    /// Returns an iterator over node IDs in this collection.
    pub fn iter(&self) -> &Vec<NodeId> {
        // MEMORY LEAK FIX: Return reference instead of cloning entire Vec
        &self.node_ids
    }

    /// Get a NodeProxy for this node if it exists (Rust-only method).
    pub fn get_rust(&self, node_id: NodeId) -> Option<crate::graph::nodes::proxy::NodeProxy> {
        if self.graph_store.has_node(&node_id) {
            Some(crate::graph::nodes::proxy::NodeProxy::new(node_id, self.graph_store.clone()))
        } else {
            None
        }
    }
}

impl NodeCollection {
    /// Regular Rust constructor - not exposed to Python
    pub fn new(attribute_manager: AttributeManager, graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>, node_ids: Option<Vec<NodeId>>) -> Self {
        let ids = node_ids.unwrap_or_else(|| graph_store.all_node_ids());
        Self { attribute_manager: std::sync::Arc::new(attribute_manager), node_ids: ids, graph_store }
    }
}

// Helper: filter by Python dict - SIMD optimized
fn filter_nodes_by_dict(
    attr_manager: &AttributeManager,
    graph_store: &crate::storage::graph_store::GraphStore,
    node_ids: &Vec<NodeId>,
    d: &pyo3::types::PyDict,
    py: pyo3::Python,
) -> Vec<NodeId> {
    // Convert node_ids to indices for bulk filtering
    let mut valid_indices: Vec<usize> = node_ids.iter()
        .filter_map(|node_id| graph_store.node_index(node_id))
        .collect();
    
    // Apply each filter in the dict
    for (k, v) in d.iter() {
        let attr = k.extract::<String>().unwrap_or_default();
        
        // Try SIMD filtering for numeric comparisons
        if let Ok(tup) = v.extract::<(String, pyo3::PyObject)>() {
            // e.g. (">", 100000)
            let op_str = tup.0.as_str();
            if let Ok(cmp_val) = tup.1.extract::<i64>(py) {
                let comparison_op = match op_str {
                    ">" => crate::storage::columnar::ComparisonOp::Greater,
                    "<" => crate::storage::columnar::ComparisonOp::Less,
                    ">=" => crate::storage::columnar::ComparisonOp::GreaterEqual,
                    "<=" => crate::storage::columnar::ComparisonOp::LessEqual,
                    "==" => crate::storage::columnar::ComparisonOp::Equal,
                    "!=" => crate::storage::columnar::ComparisonOp::NotEqual,
                    _ => continue,
                };
                
                // Use SIMD filtering when available
                #[cfg(feature = "simd")]
                {
                    let matching_indices = attr_manager.filter_nodes_i64_simd(&attr, cmp_val, comparison_op);
                    valid_indices.retain(|&idx| matching_indices.contains(&idx));
                    continue;
                }
                
                #[cfg(not(feature = "simd"))]
                {
                    // Fallback to individual checks
                    valid_indices.retain(|&idx| {
                        if let Some(val) = attr_manager.get_node_value(attr.clone(), idx) {
                            if let Some(actual) = val.as_i64() {
                                match comparison_op {
                                    crate::storage::columnar::ComparisonOp::Greater => actual > cmp_val,
                                    crate::storage::columnar::ComparisonOp::Less => actual < cmp_val,
                                    crate::storage::columnar::ComparisonOp::GreaterEqual => actual >= cmp_val,
                                    crate::storage::columnar::ComparisonOp::LessEqual => actual <= cmp_val,
                                    crate::storage::columnar::ComparisonOp::Equal => actual == cmp_val,
                                    crate::storage::columnar::ComparisonOp::NotEqual => actual != cmp_val,
                                }
                            } else { false }
                        } else { false }
                    });
                    continue;
                }
            }
        } else if let Ok(val_expected) = v.extract::<i64>() {
            // Numeric equality - use SIMD
            #[cfg(feature = "simd")]
            {
                let matching_indices = attr_manager.filter_nodes_i64_simd(&attr, val_expected, crate::storage::columnar::ComparisonOp::Equal);
                valid_indices.retain(|&idx| matching_indices.contains(&idx));
                continue;
            }
            
            #[cfg(not(feature = "simd"))]
            {
                // Fallback to individual checks
                valid_indices.retain(|&idx| {
                    if let Some(val) = attr_manager.get_node_value(attr.clone(), idx) {
                        val.as_i64() == Some(val_expected)
                    } else { false }
                });
                continue;
            }
        } else if let Ok(val_expected) = v.extract::<f64>() {
            // Float equality - use SIMD
            #[cfg(feature = "simd")]
            {
                let matching_indices = attr_manager.filter_nodes_f64_simd(&attr, val_expected, crate::storage::columnar::ComparisonOp::Equal);
                valid_indices.retain(|&idx| matching_indices.contains(&idx));
                continue;
            }
            
            #[cfg(not(feature = "simd"))]
            {
                // Fallback to individual checks
                valid_indices.retain(|&idx| {
                    if let Some(val) = attr_manager.get_node_value(attr.clone(), idx) {
                        val.as_f64() == Some(val_expected)
                    } else { false }
                });
                continue;
            }
        } else if let Ok(val_expected) = v.extract::<bool>() {
            // Boolean filtering - use bulk filtering from columnar store
            let matching_indices = attr_manager.columnar.filter_nodes_by_bool(attr, val_expected);
            valid_indices.retain(|&idx| matching_indices.contains(&idx));
            continue;
        } else if let Ok(val_expected) = v.extract::<String>() {
            // String filtering - use bulk filtering from columnar store
            let matching_indices = attr_manager.columnar.filter_nodes_by_str(attr, val_expected);
            valid_indices.retain(|&idx| matching_indices.contains(&idx));
            continue;
        }
        
        // Fallback for other types - skip for now (unsupported Python type)
        // Could be enhanced to support more Python types if needed
    }
    
    // Convert indices back to NodeIds
    valid_indices.into_iter()
        .filter_map(|idx| {
            // Find the NodeId that corresponds to this index
            node_ids.iter().find(|node_id| {
                graph_store.node_index(node_id) == Some(idx)
            }).cloned()
        })
        .collect()
}

// Helper: filter by simple query string (e.g. "salary > 100000")
fn filter_nodes_by_query(
    attr_manager: &AttributeManager,
    graph_store: &crate::storage::graph_store::GraphStore,
    node_ids: &Vec<NodeId>,
    query: &str,
) -> Vec<NodeId> {
    // Very basic: support "attr > value" or "attr == value"
    let parts: Vec<&str> = query.split_whitespace().collect();
    if parts.len() == 3 {
        let attr = parts[0];
        let op = parts[1];
        let val = parts[2];
        let int_val = val.parse::<i64>().ok();
        let str_val = val.trim_matches('"').trim_matches('\'').to_string();
        node_ids.iter().filter(|node_id| {
            let node_index = match graph_store.node_index(node_id) {
                Some(idx) => idx,
                None => return false,
            };
            let v = attr_manager.get_node_value(attr.to_string(), node_index);
            match op {
                ">" => int_val.map(|i| v.as_ref().and_then(|j| j.as_i64()).map_or(false, |x| x > i)).unwrap_or(false),
                "<" => int_val.map(|i| v.as_ref().and_then(|j| j.as_i64()).map_or(false, |x| x < i)).unwrap_or(false),
                "==" => {
                    if let Some(i) = int_val {
                        v.as_ref().and_then(|j| j.as_i64()) == Some(i)
                    } else {
                        v.as_ref().and_then(|j| j.as_str()) == Some(str_val.as_str())
                    }
                },
                _ => false
            }
        }).cloned().collect()
    } else {
        node_ids.clone() // fallback: no filtering
    }
}

