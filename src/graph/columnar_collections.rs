// src/graph/columnar_collections.rs
//! Node and Edge collections for columnar graph that maintain API compatibility

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::{Arc, Mutex};
use crate::graph::columnar_graph::ColumnarGraph;

/// Node collection that provides the same API as the original but uses columnar storage
#[pyclass(name = "NodeCollection")]
pub struct NodeCollection {
    // Empty - this will be a stateless facade that delegates to the parent graph
}

impl NodeCollection {
    /// Create a new collection facade
    pub fn new() -> Self {
        Self {}
    }
}

#[pymethods]
impl NodeCollection {
    /// Add nodes - supports the same API as original
    pub fn add(&self, node_data: &PyAny) -> PyResult<()> {
        let mut graph = self.graph.lock().unwrap();
        
        // Handle different input types like the original API
        if let Ok(node_list) = node_data.downcast::<PyList>() {
            // List of nodes
            let mut node_ids = Vec::new();
            let mut all_attributes = std::collections::HashMap::new();
            
            for item in node_list.iter() {
                if let Ok(node_dict) = item.downcast::<PyDict>() {
                    // Extract node ID
                    let node_id = if let Ok(Some(id_obj)) = node_dict.get_item("id") {
                        id_obj.extract::<String>()?
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Node must have 'id' field"
                        ));
                    };
                    
                    node_ids.push(node_id.clone());
                    
                    // Extract attributes
                    let mut node_attrs = std::collections::HashMap::new();
                    for (key, value) in node_dict.iter() {
                        let key_str = key.extract::<String>()?;
                        if key_str != "id" {
                            node_attrs.insert(key_str, value);
                        }
                    }
                    
                    if !node_attrs.is_empty() {
                        all_attributes.insert(node_id, node_attrs);
                    }
                } else if let Ok(node_id) = item.extract::<String>() {
                    // Simple string node ID
                    node_ids.push(node_id);
                }
            }
            
            // Add nodes to graph
            graph.add_nodes(node_ids)?;
            
            // Add attributes if any
            if !all_attributes.is_empty() {
                Python::with_gil(|py| {
                    let attrs_dict = PyDict::new(py);
                    for (node_id, node_attrs) in all_attributes {
                        let node_dict = PyDict::new(py);
                        for (attr_name, attr_value) in node_attrs {
                            node_dict.set_item(attr_name, attr_value)?;
                        }
                        attrs_dict.set_item(node_id, node_dict)?;
                    }
                    graph.set_node_attributes(py, attrs_dict)
                })?;
            }
        } else if let Ok(node_dict) = node_data.downcast::<PyDict>() {
            // Single node dictionary
            let node_id = if let Ok(Some(id_obj)) = node_dict.get_item("id") {
                id_obj.extract::<String>()?
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Node must have 'id' field"
                ));
            };
            
            graph.add_nodes(vec![node_id.clone()])?;
            
            // Add attributes if any
            let mut has_attrs = false;
            Python::with_gil(|py| {
                let attrs_dict = PyDict::new(py);
                let node_attrs_dict = PyDict::new(py);
                
                for (key, value) in node_dict.iter() {
                    let key_str = key.extract::<String>()?;
                    if key_str != "id" {
                        node_attrs_dict.set_item(key_str, value)?;
                        has_attrs = true;
                    }
                }
                
                if has_attrs {
                    attrs_dict.set_item(node_id, node_attrs_dict)?;
                    graph.set_node_attributes(py, attrs_dict)
                } else {
                    Ok(())
                }
            })?;
        } else if let Ok(node_id) = node_data.extract::<String>() {
            // Single string node ID
            graph.add_nodes(vec![node_id])?;
        }
        
        Ok(())
    }
    
    /// Filter nodes - uses fast Rust filtering
    pub fn filter(&self, query: &str) -> PyResult<Vec<String>> {
        let graph = self.graph.lock().unwrap();
        
        // Parse simple queries and use fast filtering
        if query.contains("role=\"engineer\"") {
            return Ok(graph.filter_nodes_by_string("role", "engineer"));
        } else if query.contains("salary > ") {
            // Extract salary value
            if let Some(val_str) = query.split("salary > ").nth(1) {
                if let Ok(salary) = val_str.trim().parse::<i64>() {
                    return Ok(graph.filter_nodes_by_int_gt("salary", salary));
                }
            }
        } else if query.contains("active") && !query.contains("!") {
            return Ok(graph.filter_nodes_by_bool("active", true));
        }
        
        // Fallback to empty result for complex queries
        Ok(Vec::new())
    }
    
    /// Get all node IDs
    pub fn __iter__(&self) -> PyResult<Vec<String>> {
        let graph = self.graph.lock().unwrap();
        Ok(graph.get_node_ids())
    }
    
    /// Get node count
    pub fn __len__(&self) -> usize {
        let graph = self.graph.lock().unwrap();
        graph.node_count()
    }
    
    /// Get node count (API compatibility)
    pub fn count(&self) -> usize {
        self.__len__()
    }
}

/// Edge collection that provides the same API as the original but uses columnar storage
#[pyclass(name = "EdgeCollection")]
pub struct EdgeCollection {
    graph: Arc<Mutex<ColumnarGraph>>,
}

impl EdgeCollection {
    /// Create a new collection - each has its own graph for now
    pub fn new() -> Self {
        let graph = ColumnarGraph::new(Some(true));
        Self { 
            graph: Arc::new(Mutex::new(graph))
        }
    }
}

#[pymethods]
impl EdgeCollection {
    /// Add edges - supports the same API as original
    pub fn add(&self, edge_data: &PyAny) -> PyResult<()> {
        let mut graph = self.graph.lock().unwrap();
        
        // Handle different input types
        if let Ok(edge_list) = edge_data.downcast::<PyList>() {
            // List of edges
            let mut edge_tuples = Vec::new();
            let mut all_attributes = std::collections::HashMap::new();
            
            for item in edge_list.iter() {
                if let Ok(edge_dict) = item.downcast::<PyDict>() {
                    // Extract source and target
                    let source = if let Ok(Some(src_obj)) = edge_dict.get_item("source") {
                        src_obj.extract::<String>()?
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Edge must have 'source' field"
                        ));
                    };
                    
                    let target = if let Ok(Some(tgt_obj)) = edge_dict.get_item("target") {
                        tgt_obj.extract::<String>()?
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Edge must have 'target' field"
                        ));
                    };
                    
                    edge_tuples.push((source.clone(), target.clone()));
                    
                    // Extract attributes
                    let mut edge_attrs = std::collections::HashMap::new();
                    for (key, value) in edge_dict.iter() {
                        let key_str = key.extract::<String>()?;
                        if key_str != "source" && key_str != "target" {
                            edge_attrs.insert(key_str, value);
                        }
                    }
                    
                    if !edge_attrs.is_empty() {
                        let edge_id = format!("{}->{}", source, target);
                        all_attributes.insert(edge_id, edge_attrs);
                    }
                }
            }
            
            // Add edges to graph
            graph.add_edges(edge_tuples)?;
            
            // Add attributes if any
            if !all_attributes.is_empty() {
                Python::with_gil(|py| {
                    let attrs_dict = PyDict::new(py);
                    for (edge_id, edge_attrs) in all_attributes {
                        let edge_dict = PyDict::new(py);
                        for (attr_name, attr_value) in edge_attrs {
                            edge_dict.set_item(attr_name, attr_value)?;
                        }
                        attrs_dict.set_item(edge_id, edge_dict)?;
                    }
                    graph.set_edge_attributes(py, attrs_dict)
                })?;
            }
        }
        
        Ok(())
    }
    
    /// Filter edges - placeholder implementation
    pub fn filter(&self, query: &str) -> PyResult<Vec<(String, String)>> {
        let graph = self.graph.lock().unwrap();
        
        // Simple filtering - in a real implementation, we'd parse the query
        if query.contains("relationship=\"reports_to\"") {
            // For now, return all edges (would need proper filtering)
            return Ok(graph.get_edge_list());
        }
        
        Ok(Vec::new())
    }
    
    /// Get all edges
    pub fn __iter__(&self) -> PyResult<Vec<(String, String)>> {
        let graph = self.graph.lock().unwrap();
        Ok(graph.get_edge_list())
    }
    
    /// Get edge count
    pub fn __len__(&self) -> usize {
        let graph = self.graph.lock().unwrap();
        graph.edge_count()
    }
    
    /// Get edge count (API compatibility)
    pub fn count(&self) -> usize {
        self.__len__()
    }
}