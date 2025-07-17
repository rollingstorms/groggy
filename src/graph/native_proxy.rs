// src/graph/native_proxy.rs
//! Native proxy objects that work directly with Python objects without JSON serialization

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use crate::graph::native_attributes::NativeAttributeManager;

/// Native node proxy that works with Python objects directly
#[pyclass]
#[derive(Clone)]
pub struct NativeNodeProxy {
    pub node_id: String,
    pub attr_manager: Arc<Mutex<NativeAttributeManager>>,
}

#[pymethods]
impl NativeNodeProxy {
    #[new]
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            attr_manager: Arc::new(Mutex::new(NativeAttributeManager::new())),
        }
    }
    
    /// Get an attribute value
    pub fn get_attr(&self, py: Python, attr_name: &str) -> PyResult<PyObject> {
        let manager = self.attr_manager.lock().unwrap();
        manager.get_node_attribute(py, &self.node_id, attr_name)
    }
    
    /// Set an attribute value (direct Python object, no JSON)
    pub fn set_attr(&self, py: Python, attr_name: &str, value: &PyAny) -> PyResult<()> {
        use pyo3::types::PyDict;
        use crate::graph::attribute_value::AttributeValue;
        
        let attr_value = AttributeValue::extract(value)?;
        
        // Create the nested dict structure: {node_id: {attr_name: value}}
        let py_dict = PyDict::new(py);
        let node_dict = PyDict::new(py);
        node_dict.set_item(attr_name, value)?;
        py_dict.set_item(&self.node_id, node_dict)?;
        
        let mut manager = self.attr_manager.lock().unwrap();
        manager.set_node_attributes(py, py_dict)
    }
    
    /// Get all attributes for this node
    pub fn get_attrs(&self, py: Python) -> PyResult<PyObject> {
        let manager = self.attr_manager.lock().unwrap();
        manager.get_node_attributes(py, &self.node_id)
    }
    
    /// Get node ID
    #[getter]
    pub fn id(&self) -> &str {
        &self.node_id
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!("NativeNodeProxy({})", self.node_id)
    }
}

/// Native edge proxy that works with Python objects directly
#[pyclass]
#[derive(Clone)]
pub struct NativeEdgeProxy {
    pub edge_id: String,
    pub attr_manager: Arc<Mutex<NativeAttributeManager>>,
}

#[pymethods]
impl NativeEdgeProxy {
    #[new]
    pub fn new(edge_id: String) -> Self {
        Self {
            edge_id,
            attr_manager: Arc::new(Mutex::new(NativeAttributeManager::new())),
        }
    }
    
    /// Get an attribute value
    pub fn get_attr(&self, py: Python, attr_name: &str) -> PyResult<PyObject> {
        let manager = self.attr_manager.lock().unwrap();
        manager.get_edge_attribute(py, &self.edge_id, attr_name)
    }
    
    /// Set an attribute value (direct Python object, no JSON)
    pub fn set_attr(&self, py: Python, attr_name: &str, value: &PyAny) -> PyResult<()> {
        use pyo3::types::PyDict;
        use crate::graph::attribute_value::AttributeValue;
        
        let attr_value = AttributeValue::extract(value)?;
        
        // Create the nested dict structure: {edge_id: {attr_name: value}}
        let py_dict = PyDict::new(py);
        let edge_dict = PyDict::new(py);
        edge_dict.set_item(attr_name, value)?;
        py_dict.set_item(&self.edge_id, edge_dict)?;
        
        let mut manager = self.attr_manager.lock().unwrap();
        manager.set_edge_attributes(py, py_dict)
    }
    
    /// Get all attributes for this edge
    pub fn get_attrs(&self, py: Python) -> PyResult<PyObject> {
        let manager = self.attr_manager.lock().unwrap();
        manager.get_edge_attributes(py, &self.edge_id)
    }
    
    /// Get edge ID
    #[getter]
    pub fn id(&self) -> &str {
        &self.edge_id
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!("NativeEdgeProxy({})", self.edge_id)
    }
}