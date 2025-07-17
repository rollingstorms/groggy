// src_new/graph/edges/collection.rs
//! EdgeCollection: concrete implementation of BaseCollection for edge storage in Groggy graphs.
//! Provides batch operations, columnar backend, and agent/LLM-friendly APIs.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use crate::graph::types::{EdgeId, NodeId};
use crate::graph::managers::attributes::AttributeManager;
// use crate::graph::columnar::EdgeColumnarStore; // Uncomment when available

#[pyclass]
#[derive(Clone)]
pub struct EdgeCollection {
    #[pyo3(get)]
    pub attribute_manager: AttributeManager,
    #[pyo3(get)]
    pub edge_ids: Vec<EdgeId>,
    pub graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>,
}

#[pymethods]
impl EdgeCollection {
    /// Add one or more edges to the collection (batch-oriented).
    pub fn py_add(&mut self, py_edge_ids: &pyo3::types::PyAny) -> pyo3::PyResult<()> {
        // Fast path: accept pre-processed tuple list from Python layer
        if let Ok(list) = py_edge_ids.extract::<Vec<(String, String)>>() {
            let edge_ids: Vec<EdgeId> = list.into_iter().map(|(src, tgt)| EdgeId::new(NodeId::new(src), NodeId::new(tgt))).collect();
            self.add_batch(edge_ids)?;
            return Ok(());
        }
        
        // Fallback: handle single tuple (for direct Rust usage)
        if let Ok(single) = py_edge_ids.extract::<(String, String)>() {
            let edge_ids = vec![EdgeId::new(NodeId::new(single.0), NodeId::new(single.1))];
            self.add_batch(edge_ids)?;
            return Ok(());
        }
        
        return Err(pyo3::exceptions::PyTypeError::new_err("Invalid edge ID format; expected (src, tgt) tuple or list of tuples"));
    }

    /// Flexible filter method: supports dict, string, or chaining.
    #[pyo3(signature = (*args, **kwargs))]
    pub fn filter_py(&self, py: Python, args: &pyo3::types::PyTuple, kwargs: Option<&PyDict>) -> Self {
        let mut filtered_ids = self.edge_ids.clone();
        // 1. If first arg is dict
        if let Some(first) = args.get_item(0).ok() {
            if let Ok(d) = first.downcast::<PyDict>() {
                filtered_ids = filter_edges_by_dict(&self.attribute_manager, &filtered_ids, d, py, &self.graph_store);
            } else if let Ok(s) = first.downcast::<PyString>() {
                let query = s.to_str().unwrap_or("");
                filtered_ids = filter_edges_by_query(&self.attribute_manager, &filtered_ids, query, &self.graph_store);
            }
        } else if let Some(d) = kwargs {
            filtered_ids = filter_edges_by_dict(&self.attribute_manager, &filtered_ids, d, py, &self.graph_store);
        }
        Self {
            attribute_manager: self.attribute_manager.clone(),
            edge_ids: filtered_ids,
            graph_store: self.graph_store.clone(),
        }
    }

    /// Create a new EdgeCollection from Python (simplified constructor)
    #[new]
    pub fn py_new() -> Self {
        let graph_store = std::sync::Arc::new(crate::storage::graph_store::GraphStore::new());
        let attribute_manager = AttributeManager::new_with_graph_store(graph_store.clone());
        Self {
            attribute_manager,
            edge_ids: Vec::new(),
            graph_store,
        }
    }

    /// Returns an AttributeManager for edge attributes.
    pub fn attr(&self) -> AttributeManager {
        self.attribute_manager.clone()
    }

    /// Returns the number of edges in this collection.
    #[getter]
    pub fn size(&self) -> usize {
        self.graph_store.edge_count()
    }

    /// Flexible filter method: supports dict, string, or chaining.
    #[pyo3(name = "filter")]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn filter(&self, py: Python, args: &pyo3::types::PyTuple, kwargs: Option<&PyDict>) -> Self {
        self.filter_py(py, args, kwargs)
    }

    /// Returns all edge IDs in this collection.
    pub fn ids(&self) -> Vec<String> {
        self.graph_store.all_edge_ids().iter().map(|id| format!("{}", id)).collect()
    }

    /// Check if an edge exists in the collection.
    pub fn has(&self, edge_id: String) -> bool {
        // PHASE 2 OPTIMIZATION: Use split_once instead of parsing - already optimized
        if let Some((src, tgt)) = edge_id.split_once("->") {
            let edge_id = EdgeId::new(NodeId::new(src.to_string()), NodeId::new(tgt.to_string()));
            self.graph_store.has_edge(&edge_id)
        } else {
            false
        }
    }

    /// Get an EdgeProxy for this edge if it exists.
    pub fn get(&self, edge_id: String) -> Option<crate::graph::edges::proxy::EdgeProxy> {
        if let Some((src, tgt)) = edge_id.split_once("->") {
            let edge_id = EdgeId::new(NodeId::new(src.to_string()), NodeId::new(tgt.to_string()));
            if self.graph_store.has_edge(&edge_id) {
                Some(crate::graph::edges::proxy::EdgeProxy::new(
                    edge_id,
                    NodeId::new(src.to_string()),
                    NodeId::new(tgt.to_string()),
                    self.attribute_manager.clone(),
                    self.graph_store.clone(),
                ))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Add one or more edges to the collection.
    #[pyo3(name = "add")]
    pub fn add(&mut self, py_edge_ids: &pyo3::types::PyAny) -> pyo3::PyResult<()> {
        self.py_add(py_edge_ids)
    }
}

// --- Rust-only methods ---
impl EdgeCollection {
    /// Regular Rust constructor - not exposed to Python
    pub fn new(attribute_manager: AttributeManager, graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>, edge_ids: Option<Vec<EdgeId>>) -> Self {
        let ids = edge_ids.unwrap_or_else(|| graph_store.all_edge_ids());
        Self { attribute_manager, edge_ids: ids, graph_store }
    }

    /// Add one or more edges to the collection (batch-oriented, internal use).
    pub fn add_batch(&mut self, edges: Vec<EdgeId>) -> PyResult<()> {
        self.graph_store.add_edges(&edges);
        // MEMORY LEAK FIX: Extend existing Vec instead of replacing entire Vec
        self.edge_ids.extend(edges);
        Ok(())
    }

    /// Remove one or more edges from the collection (batch-oriented, internal use).
    pub fn remove_batch(&mut self, edge_ids: Vec<EdgeId>) -> PyResult<()> {
        self.graph_store.remove_edges(&edge_ids);
        // MEMORY LEAK FIX: Remove specific edges instead of replacing entire Vec
        self.edge_ids.retain(|edge_id| !edge_ids.contains(edge_id));
        Ok(())
    }

    /// Returns an iterator over edge IDs in this collection (internal use).
    pub fn iter(&self) -> &Vec<EdgeId> {
        // MEMORY LEAK FIX: Return reference instead of cloning entire Vec
        &self.edge_ids
    }

    /// Get an EdgeProxy for this edge if it exists (internal use).
    pub fn get_rust(&self, edge_id: EdgeId) -> Option<crate::graph::edges::proxy::EdgeProxy> {
        if self.graph_store.has_edge(&edge_id) {
            Some(crate::graph::edges::proxy::EdgeProxy::new(
                edge_id.clone(),
                edge_id.source().clone(),
                edge_id.target().clone(),
                self.attribute_manager.clone(),
                self.graph_store.clone(),
            ))
        } else {
            None
        }
    }
}

// Helper: filter by Python dict
fn filter_edges_by_dict(
    attr_manager: &AttributeManager,
    edge_ids: &Vec<EdgeId>,
    d: &pyo3::types::PyDict,
    py: pyo3::Python,
    graph_store: &std::sync::Arc<crate::storage::graph_store::GraphStore>,
) -> Vec<EdgeId> {
    let mut filtered = Vec::new();
    for edge_id in edge_ids {
        let mut keep = true;
        for (k, v) in d.iter() {
            let attr = k.extract::<String>().unwrap_or_default();
            if let Some(index) = graph_store.edge_index(edge_id) {
                let val = attr_manager.get_edge_value(attr.clone(), index);
                if let Ok(tup) = v.extract::<(String, pyo3::PyObject)>() {
                    // e.g. (">", 0.7)
                    let op = tup.0.as_str();
                    let cmp_val = tup.1.extract::<f64>(py).unwrap_or(0.0);
                    let actual = val.as_ref().and_then(|j| j.as_f64()).unwrap_or(0.0);
                    match op {
                        ">" => if !(actual > cmp_val) { keep = false; break; },
                        "<" => if !(actual < cmp_val) { keep = false; break; },
                        ">=" => if !(actual >= cmp_val) { keep = false; break; },
                        "<=" => if !(actual <= cmp_val) { keep = false; break; },
                        "==" => if !(actual == cmp_val) { keep = false; break; },
                        _ => { keep = false; break; }
                    }
                } else if let Ok(val_expected) = v.extract::<f64>() {
                    // Numeric equality
                    if val.as_ref().and_then(|j| j.as_f64()) != Some(val_expected) {
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
            } else {
                // Edge not found in graph, skip it
                keep = false; break;
            }
        }
        if keep { filtered.push(edge_id.clone()); }
    }
    filtered
}

// Helper: filter by simple query string (e.g. "strength > 0.7")
fn filter_edges_by_query(
    attr_manager: &AttributeManager,
    edge_ids: &Vec<EdgeId>,
    query: &str,
    graph_store: &std::sync::Arc<crate::storage::graph_store::GraphStore>,
) -> Vec<EdgeId> {
    // Very basic: support "attr > value" or "attr == value"
    let parts: Vec<&str> = query.split_whitespace().collect();
    if parts.len() == 3 {
        let attr = parts[0];
        let op = parts[1];
        let val = parts[2];
        let float_val = val.parse::<f64>().ok();
        let str_val = val.trim_matches('"').trim_matches('\'').to_string();
        edge_ids.iter().filter(|edge_id| {
            if let Some(index) = graph_store.edge_index(edge_id) {
                let v = attr_manager.get_edge_value(attr.to_string(), index);
                match op {
                    ">" => float_val.map(|f| v.as_ref().and_then(|j| j.as_f64()).map_or(false, |x| x > f)).unwrap_or(false),
                    "<" => float_val.map(|f| v.as_ref().and_then(|j| j.as_f64()).map_or(false, |x| x < f)).unwrap_or(false),
                    "==" => {
                        if let Some(f) = float_val {
                            v.as_ref().and_then(|j| j.as_f64()) == Some(f)
                        } else {
                            v.as_ref().and_then(|j| j.as_str()) == Some(str_val.as_str())
                        }
                    },
                    _ => false
                }
            } else {
                false
            }
        }).cloned().collect()
    } else {
        edge_ids.clone() // fallback: no filtering
    }
}
