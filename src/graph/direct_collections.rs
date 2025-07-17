// src/graph/direct_collections.rs
//! Direct Rust collections accessible from Python without wrapper layers

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::{Arc, Mutex};
use crate::graph::columnar_graph::ColumnarGraph;

/// Direct node collection that operates on columnar storage
#[pyclass]
pub struct DirectNodeCollection {
    graph: Arc<Mutex<ColumnarGraph>>,
}

impl DirectNodeCollection {
    /// Create a new collection with a shared graph
    pub fn new_with_graph(graph: Arc<Mutex<ColumnarGraph>>) -> Self {
        Self { graph }
    }
}

#[pymethods]
impl DirectNodeCollection {
    #[new]
    pub fn new() -> Self {
        // Create a new default graph
        let graph = ColumnarGraph::new(Some(true));
        Self { 
            graph: Arc::new(Mutex::new(graph))
        }
    }
    
    /// Add nodes directly (no Python wrapper overhead)
    pub fn add(&self, node_data: &PyAny) -> PyResult<()> {
        let mut graph = self.graph.lock().unwrap();
        
        // Handle different input types
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
    
    /// Get node IDs directly
    pub fn ids(&self) -> Vec<String> {
        let graph = self.graph.lock().unwrap();
        graph.get_node_ids()
    }
    
    /// Get node count directly
    pub fn count(&self) -> usize {
        let graph = self.graph.lock().unwrap();
        graph.node_count()
    }
    
    /// Check if node exists
    pub fn has(&self, node_id: &str) -> bool {
        let graph = self.graph.lock().unwrap();
        graph.get_node_ids().contains(&node_id.to_string())
    }
    
    /// FAST: Filter nodes by string attribute (pure Rust)
    pub fn filter_by_string(&self, attr_name: &str, value: &str) -> Vec<String> {
        let graph = self.graph.lock().unwrap();
        graph.filter_nodes_by_string(attr_name, value)
    }
    
    /// FAST: Filter nodes by integer comparison (pure Rust)
    pub fn filter_by_int_gt(&self, attr_name: &str, value: i64) -> Vec<String> {
        let graph = self.graph.lock().unwrap();
        graph.filter_nodes_by_int_gt(attr_name, value)
    }
    
    /// FAST: Filter nodes by boolean (pure Rust)
    pub fn filter_by_bool(&self, attr_name: &str, value: bool) -> Vec<String> {
        let graph = self.graph.lock().unwrap();
        graph.filter_nodes_by_bool(attr_name, value)
    }
    
    /// FAST: Complex filter (pure Rust)
    pub fn filter_complex(&self, 
        role: Option<&str>, 
        min_salary: Option<i64>, 
        active: Option<bool>
    ) -> Vec<String> {
        let graph = self.graph.lock().unwrap();
        graph.filter_nodes_complex(role, min_salary, active)
    }
    
    /// Get attribute value directly
    pub fn get_attr(&self, py: Python, node_id: &str, attr_name: &str) -> PyResult<PyObject> {
        let graph = self.graph.lock().unwrap();
        graph.get_node_attribute(py, node_id, attr_name)
    }
    
    /// Set attribute value directly
    pub fn set_attr(&self, py: Python, node_id: &str, attr_name: &str, value: &PyAny) -> PyResult<()> {
        let mut graph = self.graph.lock().unwrap();
        
        // Create the batch structure
        let attrs_dict = PyDict::new(py);
        let node_dict = PyDict::new(py);
        node_dict.set_item(attr_name, value)?;
        attrs_dict.set_item(node_id, node_dict)?;
        
        graph.set_node_attributes(py, attrs_dict)
    }
    
    /// Python length
    pub fn __len__(&self) -> usize {
        self.count()
    }
    
    /// Python iteration
    pub fn __iter__(&self) -> PyResult<Vec<String>> {
        Ok(self.ids())
    }
}

/// Direct edge collection that operates on columnar storage
#[pyclass]
pub struct DirectEdgeCollection {
    graph: Arc<Mutex<ColumnarGraph>>,
}

impl DirectEdgeCollection {
    /// Create a new collection with a shared graph
    pub fn new_with_graph(graph: Arc<Mutex<ColumnarGraph>>) -> Self {
        Self { graph }
    }
}

#[pymethods]
impl DirectEdgeCollection {
    #[new]
    pub fn new() -> Self {
        // Create a new default graph
        let graph = ColumnarGraph::new(Some(true));
        Self { 
            graph: Arc::new(Mutex::new(graph))
        }
    }
    
    /// Add edges directly (no Python wrapper overhead)
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
        } else if let Ok(edge_dict) = edge_data.downcast::<PyDict>() {
            // Single edge dictionary
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
            
            graph.add_edges(vec![(source.clone(), target.clone())])?;
            
            // Add attributes if any
            let mut has_attrs = false;
            Python::with_gil(|py| {
                let attrs_dict = PyDict::new(py);
                let edge_attrs_dict = PyDict::new(py);
                
                for (key, value) in edge_dict.iter() {
                    let key_str = key.extract::<String>()?;
                    if key_str != "source" && key_str != "target" {
                        edge_attrs_dict.set_item(key_str, value)?;
                        has_attrs = true;
                    }
                }
                
                if has_attrs {
                    let edge_id = format!("{}->{}", source, target);
                    attrs_dict.set_item(edge_id, edge_attrs_dict)?;
                    graph.set_edge_attributes(py, attrs_dict)
                } else {
                    Ok(())
                }
            })?;
        }
        
        Ok(())
    }
    
    /// Get all edges as (source, target) tuples
    pub fn get_edges(&self) -> Vec<(String, String)> {
        let graph = self.graph.lock().unwrap();
        graph.get_edges()
    }
    
    /// Get edge count directly
    pub fn count(&self) -> usize {
        let graph = self.graph.lock().unwrap();
        graph.edge_count()
    }
    
    /// Get attribute value directly
    pub fn get_attr(&self, py: Python, source: &str, target: &str, attr_name: &str) -> PyResult<PyObject> {
        let graph = self.graph.lock().unwrap();
        let edge_id = format!("{}->{}", source, target);
        graph.get_edge_attribute(py, &edge_id, attr_name)
    }
    
    /// Set attribute value directly
    pub fn set_attr(&self, py: Python, source: &str, target: &str, attr_name: &str, value: &PyAny) -> PyResult<()> {
        let mut graph = self.graph.lock().unwrap();
        
        // Create the batch structure
        let attrs_dict = PyDict::new(py);
        let edge_dict = PyDict::new(py);
        edge_dict.set_item(attr_name, value)?;
        let edge_id = format!("{}->{}", source, target);
        attrs_dict.set_item(edge_id, edge_dict)?;
        
        graph.set_edge_attributes(py, attrs_dict)
    }
    
    /// Python length
    pub fn __len__(&self) -> usize {
        self.count()
    }
    
    /// Python iteration
    pub fn __iter__(&self) -> PyResult<Vec<(String, String)>> {
        Ok(self.get_edges())
    }
}