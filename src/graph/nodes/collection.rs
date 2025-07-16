// src_new/graph/nodes/collection.rs
//! NodeCollection: concrete implementation of BaseCollection for node storage in Groggy graphs.
//! Provides batch operations, columnar backend, and agent/LLM-friendly APIs.

use pyo3::prelude::*;
use pyo3::{types::{PyAny, PyDict, PyString}, Python};
use crate::graph::types::NodeId;
use crate::graph::managers::attributes::AttributeManager;
// use crate::graph::columnar::NodeColumnarStore; // Uncomment when available

#[pyclass]
#[derive(Clone)]
pub struct NodeCollection {
    #[pyo3(get)]
    pub attribute_manager: AttributeManager,
    #[pyo3(get)]
    pub node_ids: Vec<NodeId>,
    pub graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>,
}


#[pymethods]
impl NodeCollection {
    /// Flexible filter method: supports dict, string, or chaining.
    #[pyo3(signature = (*args, **kwargs))]
    pub fn filter_py(&self, py: Python, args: &pyo3::types::PyTuple, kwargs: Option<&PyDict>) -> Self {
        let mut filtered_ids = self.node_ids.clone();
        // 1. If first arg is dict
        if let Some(first) = args.get_item(0).ok() {
            if let Ok(d) = first.downcast::<PyDict>() {
                filtered_ids = filter_nodes_by_dict(&self.attribute_manager, &filtered_ids, d, py);
            } else if let Ok(s) = first.downcast::<PyString>() {
                let query = s.to_str().unwrap_or("");
                filtered_ids = filter_nodes_by_query(&self.attribute_manager, &filtered_ids, query);
            }
        } else if let Some(d) = kwargs {
            filtered_ids = filter_nodes_by_dict(&self.attribute_manager, &filtered_ids, d, py);
        }
        Self {
            attribute_manager: self.attribute_manager.clone(),
            node_ids: filtered_ids,
        }
    }

    pub fn new(attribute_manager: AttributeManager, graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>, node_ids: Option<Vec<NodeId>>) -> Self {
        let ids = node_ids.unwrap_or_else(|| graph_store.all_node_ids());
        Self { attribute_manager, node_ids: ids, graph_store }
    }

    /// Create a new NodeCollection from Python (simplified constructor)
    #[new]
    pub fn py_new(attribute_manager: AttributeManager) -> Self {
        // For Python usage, create with empty node_ids and a placeholder graph_store
        // This will be properly initialized when used with a real graph
        let graph_store = std::sync::Arc::new(crate::storage::graph_store::GraphStore::new());
        Self { 
            attribute_manager, 
            node_ids: Vec::new(),
            graph_store,
        }
    }

    /// Add one or more nodes to the collection (batch-oriented).
    pub fn add(&mut self, nodes: Vec<NodeId>) -> PyResult<()> {
        self.graph_store.add_nodes(&nodes);
        self.node_ids = self.graph_store.all_node_ids();
        Ok(())
    }

    /// Remove one or more nodes from the collection (batch-oriented).
    pub fn remove(&mut self, node_ids: Vec<NodeId>) -> PyResult<()> {
        self.graph_store.remove_nodes(&node_ids);
        self.node_ids = self.graph_store.all_node_ids();
        Ok(())
    }

    /// Returns the number of nodes in this collection.
    pub fn size(&self) -> usize {
        self.graph_store.node_count()
    }

    /// Returns all node IDs in this collection.
    pub fn ids(&self) -> Vec<NodeId> {
        self.graph_store.all_node_ids()
    }

    /// Check if a node exists in the collection.
    pub fn has(&self, node_id: NodeId) -> bool {
        self.graph_store.has_node(&node_id)
    }

    /// Returns an AttributeManager for node attributes.
    pub fn attr(&self) -> AttributeManager {
        self.attribute_manager.clone()
    }

    /// Returns a FilterManager for this collection, pre-configured for nodes.
    ///
    /// Usage:
    /// let mut fm = collection.filter();
    /// fm.add_filter(...);
    /// let result_ids = fm.apply(collection.ids());
    pub fn filter(&self) -> crate::graph::managers::filter::FilterManager {
        let mut parent: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        crate::graph::managers::filter::FilterManager::new(self.attribute_manager.clone(), true)
    }

    /// Returns an iterator over node IDs in this collection.
    pub fn iter(&self) -> Vec<NodeId> {
        self.node_ids.clone()
    }

    /// Get a NodeProxy for this node if it exists.
    pub fn get(&self, node_id: NodeId) -> Option<crate::graph::nodes::proxy::NodeProxy> {
        if self.has(node_id.clone()) {
            Some(crate::graph::nodes::proxy::NodeProxy::new(node_id, self.attribute_manager.clone()))
        } else {
            None
        }
    }
}

// Helper: filter by Python dict
fn filter_nodes_by_dict(
    attr_manager: &AttributeManager,
    node_ids: &Vec<NodeId>,
    d: &pyo3::types::PyDict,
    py: pyo3::Python,
) -> Vec<NodeId> {
    let mut filtered = Vec::new();
    'outer: for node_id in node_ids {
        let mut keep = true;
        for (k, v) in d.iter() {
            let attr = k.extract::<String>().unwrap_or_default();
            let val = attr_manager.get_node_value(attr.clone(), node_id.0 as usize);
            if let Ok(tup) = v.extract::<(String, pyo3::PyObject)>() {
                // e.g. (">", 100000)
                let op = tup.0.as_str();
                let cmp_val = tup.1.extract::<i64>(py).unwrap_or(0);
                let actual = val.as_ref().and_then(|j| j.as_i64()).unwrap_or(0);
                match op {
                    ">" => if !(actual > cmp_val) { keep = false; break; },
                    "<" => if !(actual < cmp_val) { keep = false; break; },
                    ">=" => if !(actual >= cmp_val) { keep = false; break; },
                    "<=" => if !(actual <= cmp_val) { keep = false; break; },
                    "==" => if !(actual == cmp_val) { keep = false; break; },
                    _ => { keep = false; break; }
                }
            } else if let Ok(val_expected) = v.extract::<i64>() {
                // Numeric equality
                if val.as_ref().and_then(|j| j.as_i64()) != Some(val_expected) {
                    keep = false; break;
                }
            } else if let Ok(val_expected) = v.extract::<bool>() {
                if val.as_ref().and_then(|j| j.as_bool()) != Some(val_expected) {
                    keep = false; break;
                }
            } else if let Ok(val_expected) = v.extract::<String>() {
                if val.as_ref().and_then(|j| j.as_str()) != Some(val_expected.as_str()) {
                    keep = false; break;
                }
            } else {
                keep = false; break;
            }
        }
        if keep { filtered.push(node_id.clone()); }
    }
    filtered
}

// Helper: filter by simple query string (e.g. "salary > 100000")
fn filter_nodes_by_query(
    attr_manager: &AttributeManager,
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
            let v = attr_manager.get_node_value(attr.to_string(), node_id.0 as usize);
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

