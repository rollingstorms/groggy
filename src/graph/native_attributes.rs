// src/graph/native_attributes.rs
//! Native attribute manager that accepts Python objects directly without JSON serialization

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use crate::graph::attribute_value::AttributeValue;

/// Native attribute manager that works with Python objects directly
#[pyclass]
#[derive(Clone)]
pub struct NativeAttributeManager {
    /// Storage for node attributes: node_id -> {attr_name -> value}
    node_attributes: HashMap<String, HashMap<String, AttributeValue>>,
    /// Storage for edge attributes: edge_id -> {attr_name -> value}
    edge_attributes: HashMap<String, HashMap<String, AttributeValue>>,
    /// Track memory usage
    memory_usage: usize,
}

#[pymethods]
impl NativeAttributeManager {
    #[new]
    pub fn new() -> Self {
        Self {
            node_attributes: HashMap::new(),
            edge_attributes: HashMap::new(),
            memory_usage: 0,
        }
    }
    
    /// Set attributes for nodes (batch operation)
    /// Expected format: {"node_id": {"attr_name": value, ...}, ...}
    pub fn set_node_attributes(&mut self, py: Python, attrs: &PyDict) -> PyResult<()> {
        for (node_id_obj, node_attrs_obj) in attrs.iter() {
            let node_id = node_id_obj.extract::<String>()?;
            let node_attrs = node_attrs_obj.downcast::<PyDict>()?;
            
            // Get or create attribute map for this node
            let attr_map = self.node_attributes.entry(node_id.clone()).or_insert_with(HashMap::new);
            
            // Set each attribute
            for (attr_name_obj, attr_value_obj) in node_attrs.iter() {
                let attr_name = attr_name_obj.extract::<String>()?;
                let attr_value = AttributeValue::extract(attr_value_obj)?;
                
                // Update memory usage
                if let Some(old_value) = attr_map.get(&attr_name) {
                    self.memory_usage -= old_value.memory_size();
                }
                self.memory_usage += attr_value.memory_size();
                
                attr_map.insert(attr_name, attr_value);
            }
        }
        Ok(())
    }
    
    /// Set attributes for edges (batch operation)
    /// Expected format: {"edge_id": {"attr_name": value, ...}, ...}
    pub fn set_edge_attributes(&mut self, py: Python, attrs: &PyDict) -> PyResult<()> {
        for (edge_id_obj, edge_attrs_obj) in attrs.iter() {
            let edge_id = edge_id_obj.extract::<String>()?;
            let edge_attrs = edge_attrs_obj.downcast::<PyDict>()?;
            
            // Get or create attribute map for this edge
            let attr_map = self.edge_attributes.entry(edge_id.clone()).or_insert_with(HashMap::new);
            
            // Set each attribute
            for (attr_name_obj, attr_value_obj) in edge_attrs.iter() {
                let attr_name = attr_name_obj.extract::<String>()?;
                let attr_value = AttributeValue::extract(attr_value_obj)?;
                
                // Update memory usage
                if let Some(old_value) = attr_map.get(&attr_name) {
                    self.memory_usage -= old_value.memory_size();
                }
                self.memory_usage += attr_value.memory_size();
                
                attr_map.insert(attr_name, attr_value);
            }
        }
        Ok(())
    }
    
    /// Get node attribute
    pub fn get_node_attribute(&self, py: Python, node_id: &str, attr_name: &str) -> PyResult<PyObject> {
        if let Some(attrs) = self.node_attributes.get(node_id) {
            if let Some(value) = attrs.get(attr_name) {
                return value.to_python(py);
            }
        }
        Ok(py.None())
    }
    
    /// Get edge attribute
    pub fn get_edge_attribute(&self, py: Python, edge_id: &str, attr_name: &str) -> PyResult<PyObject> {
        if let Some(attrs) = self.edge_attributes.get(edge_id) {
            if let Some(value) = attrs.get(attr_name) {
                return value.to_python(py);
            }
        }
        Ok(py.None())
    }
    
    /// Get all attributes for a node
    pub fn get_node_attributes(&self, py: Python, node_id: &str) -> PyResult<PyObject> {
        if let Some(attrs) = self.node_attributes.get(node_id) {
            let py_dict = PyDict::new(py);
            for (attr_name, value) in attrs {
                py_dict.set_item(attr_name, value.to_python(py)?)?;
            }
            Ok(py_dict.to_object(py))
        } else {
            Ok(PyDict::new(py).to_object(py))
        }
    }
    
    /// Get all attributes for an edge
    pub fn get_edge_attributes(&self, py: Python, edge_id: &str) -> PyResult<PyObject> {
        if let Some(attrs) = self.edge_attributes.get(edge_id) {
            let py_dict = PyDict::new(py);
            for (attr_name, value) in attrs {
                py_dict.set_item(attr_name, value.to_python(py)?)?;
            }
            Ok(py_dict.to_object(py))
        } else {
            Ok(PyDict::new(py).to_object(py))
        }
    }
    
    /// Get memory usage in bytes
    pub fn get_memory_usage(&self) -> usize {
        self.memory_usage
    }
    
    /// Clear all attributes
    pub fn clear(&mut self) {
        self.node_attributes.clear();
        self.edge_attributes.clear();
        self.memory_usage = 0;
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("node_count".to_string(), self.node_attributes.len());
        stats.insert("edge_count".to_string(), self.edge_attributes.len());
        stats.insert("memory_bytes".to_string(), self.memory_usage);
        stats.insert("total_node_attrs".to_string(), 
                    self.node_attributes.values().map(|attrs| attrs.len()).sum());
        stats.insert("total_edge_attrs".to_string(), 
                    self.edge_attributes.values().map(|attrs| attrs.len()).sum());
        stats
    }
}

impl Default for NativeAttributeManager {
    fn default() -> Self {
        Self::new()
    }
}