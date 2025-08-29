//! Graph Attribute Operations - Pure FFI Delegation Layer
//!
//! This module contains attribute-related operations that delegate to core implementations.

use crate::ffi::types::{PyAttrValue, PyResultHandle};
use crate::ffi::utils::{python_value_to_attr_value, graph_error_to_py_err};
use groggy::{AttrName, AttrValue as RustAttrValue, EdgeId, NodeId};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use std::collections::HashMap;

use super::graph::PyGraph;

#[pymethods]
impl PyGraph {
    /// Set single node attribute - delegates to core
    pub fn set_node_attr(&mut self, node: NodeId, attr: AttrName, value: &PyAny) -> PyResult<()> {
        let attr_value = python_value_to_attr_value(value)?;
        self.inner
            .borrow_mut()
            .set_node_attr(node, attr, attr_value)
            .map_err(graph_error_to_py_err)
    }

    /// Set single edge attribute - delegates to core
    pub fn set_edge_attr(&mut self, edge: EdgeId, attr: AttrName, value: &PyAny) -> PyResult<()> {
        let attr_value = python_value_to_attr_value(value)?;
        self.inner
            .borrow_mut()
            .set_edge_attr(edge, attr, attr_value)
            .map_err(graph_error_to_py_err)
    }

    /// Get single node attribute - delegates to core
    pub fn get_node_attr(
        &self,
        node: NodeId,
        attr: AttrName,
        default: Option<&PyAny>,
        py: Python,
    ) -> PyResult<PyObject> {
        match self.inner.borrow().get_node_attr(node, &attr) {
            Ok(Some(attr_value)) => {
                let py_attr_value = PyAttrValue::new(attr_value);
                Ok(py_attr_value.to_object(py))
            }
            Ok(None) => match default {
                Some(default_value) => Ok(default_value.to_object(py)),
                None => Err(PyKeyError::new_err(format!(
                    "Attribute '{}' not found for node {}",
                    attr, node
                ))),
            },
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }

    /// Get single edge attribute - delegates to core
    pub fn get_edge_attr(
        &self,
        edge: EdgeId,
        attr: AttrName,
        default: Option<&PyAny>,
        py: Python,
    ) -> PyResult<PyObject> {
        match self.inner.borrow().get_edge_attr(edge, &attr) {
            Ok(Some(attr_value)) => {
                let py_attr_value = PyAttrValue::new(attr_value);
                Ok(py_attr_value.to_object(py))
            }
            Ok(None) => match default {
                Some(default_value) => Ok(default_value.to_object(py)),
                None => Err(PyKeyError::new_err(format!(
                    "Attribute '{}' not found for edge {}",
                    attr, edge
                ))),
            },
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }

    /// Get all attributes for a single edge - delegates to core
    fn get_edge_attrs(&self, edge: EdgeId, py: Python) -> PyResult<PyObject> {
        let result = self.inner
            .borrow()
            .get_edge_attrs(edge)
            .map_err(graph_error_to_py_err)?;

        let py_dict = PyDict::new(py);
        for (attr_name, attr_value) in result {
            let py_attr_value = PyAttrValue::new(attr_value);
            py_dict.set_item(attr_name, py_attr_value)?;
        }

        Ok(py_dict.to_object(py))
    }

    /// Set bulk node attributes - HYPER-OPTIMIZED core delegation
    fn set_node_attrs(&mut self, _py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        use groggy::AttrValue as RustAttrValue;
        use pyo3::exceptions::{PyKeyError, PyValueError};

        // Convert Python dict to Rust format: HashMap<AttrName, Vec<(NodeId, AttrValue)>>
        let mut rust_attrs: HashMap<String, Vec<(NodeId, RustAttrValue)>> = HashMap::new();

        for (attr_name_py, node_values_py) in attrs_dict.iter() {
            let attr_name: String = attr_name_py.extract()?;
            
            // Handle different input formats
            if let Ok(node_dict) = node_values_py.extract::<&PyDict>() {
                // Format: {"attr": {node1: value1, node2: value2}}
                let mut attr_values = Vec::new();
                for (node_py, value_py) in node_dict.iter() {
                    let node_id: NodeId = node_py.extract()?;
                    let attr_value = python_value_to_attr_value(value_py)?;
                    attr_values.push((node_id, attr_value));
                }
                rust_attrs.insert(attr_name, attr_values);
                
            } else if let Ok(tuples_list) = node_values_py.extract::<Vec<(NodeId, &PyAny)>>() {
                // Format: {"attr": [(node1, value1), (node2, value2)]}
                let mut attr_values = Vec::new();
                for (node_id, value_py) in tuples_list {
                    let attr_value = python_value_to_attr_value(value_py)?;
                    attr_values.push((node_id, attr_value));
                }
                rust_attrs.insert(attr_name, attr_values);
                
            } else {
                return Err(PyValueError::new_err(format!(
                    "Invalid format for attribute '{}'. Expected dict or list of tuples.", 
                    attr_name
                )));
            }
        }

        // DELEGATION: Use core bulk operation
        self.inner
            .borrow_mut()
            .set_node_attrs(rust_attrs)
            .map_err(graph_error_to_py_err)
    }

    /// Get bulk node attributes - HYPER-OPTIMIZED core delegation  
    fn get_node_attrs(&self, py: Python, nodes: Vec<NodeId>, attrs: Vec<AttrName>) -> PyResult<PyObject> {
        use groggy::AttrValue as RustAttrValue;
        
        // HYPER-OPTIMIZED bulk attribute retrieval - leverages columnar storage
        let result = self.inner
            .borrow()
            .get_node_attrs_bulk(nodes, attrs)
            .map_err(graph_error_to_py_err)?;
        
        // Convert Rust result to Python dictionary
        // Format: {node_id: {attr_name: AttrValue}}
        let py_dict = PyDict::new(py);
        for (node_id, node_attrs) in result {
            let node_dict = PyDict::new(py);
            for (attr_name, attr_value) in node_attrs {
                let py_attr_value = match attr_value {
                    RustAttrValue::Text(s) => Py::new(py, PyAttrValue { inner: RustAttrValue::Text(s) })?,
                    RustAttrValue::Int(i) => Py::new(py, PyAttrValue { inner: RustAttrValue::Int(i) })?,
                    RustAttrValue::Float(f) => Py::new(py, PyAttrValue { inner: RustAttrValue::Float(f) })?,
                    RustAttrValue::Bool(b) => Py::new(py, PyAttrValue { inner: RustAttrValue::Bool(b) })?,
                    RustAttrValue::FloatVec(arr) => Py::new(py, PyAttrValue { inner: RustAttrValue::FloatVec(arr) })?,
                    RustAttrValue::CompactText(ct) => Py::new(py, PyAttrValue { inner: RustAttrValue::CompactText(ct) })?,
                    RustAttrValue::SmallInt(si) => Py::new(py, PyAttrValue { inner: RustAttrValue::SmallInt(si) })?,
                    RustAttrValue::CompressedText(ct) => Py::new(py, PyAttrValue { inner: RustAttrValue::CompressedText(ct) })?,
                    RustAttrValue::CompressedFloatVec(cfv) => Py::new(py, PyAttrValue { inner: RustAttrValue::CompressedFloatVec(cfv) })?,
                    RustAttrValue::Null => Py::new(py, PyAttrValue { inner: RustAttrValue::Null })?,
                    RustAttrValue::SubgraphRef(sr) => Py::new(py, PyAttrValue { inner: RustAttrValue::SubgraphRef(sr) })?,
                    RustAttrValue::NodeArray(na) => Py::new(py, PyAttrValue { inner: RustAttrValue::NodeArray(na) })?,
                    RustAttrValue::EdgeArray(ea) => Py::new(py, PyAttrValue { inner: RustAttrValue::EdgeArray(ea) })?,
                    RustAttrValue::Bytes(bytes) => Py::new(py, PyAttrValue { inner: RustAttrValue::Bytes(bytes) })?,
                };
                node_dict.set_item(attr_name, py_attr_value)?;
            }
            py_dict.set_item(node_id, node_dict)?;
        }
        
        Ok(py_dict.to_object(py))
    }

    /// Set bulk edge attributes - HYPER-OPTIMIZED core delegation
    fn set_edge_attrs(&mut self, _py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        use groggy::AttrValue as RustAttrValue;
        use pyo3::exceptions::{PyKeyError, PyValueError};

        // Convert Python dict to Rust format: HashMap<AttrName, Vec<(EdgeId, AttrValue)>>
        let mut rust_attrs: HashMap<String, Vec<(EdgeId, RustAttrValue)>> = HashMap::new();

        for (attr_name_py, edge_values_py) in attrs_dict.iter() {
            let attr_name: String = attr_name_py.extract()?;
            
            // Handle different input formats  
            if let Ok(edge_dict) = edge_values_py.extract::<&PyDict>() {
                // Format: {"attr": {edge1: value1, edge2: value2}}
                let mut attr_values = Vec::new();
                for (edge_py, value_py) in edge_dict.iter() {
                    let edge_id: EdgeId = edge_py.extract()?;
                    let attr_value = python_value_to_attr_value(value_py)?;
                    attr_values.push((edge_id, attr_value));
                }
                rust_attrs.insert(attr_name, attr_values);
                
            } else if let Ok(tuples_list) = edge_values_py.extract::<Vec<(EdgeId, &PyAny)>>() {
                // Format: {"attr": [(edge1, value1), (edge2, value2)]}
                let mut attr_values = Vec::new();
                for (edge_id, value_py) in tuples_list {
                    let attr_value = python_value_to_attr_value(value_py)?;
                    attr_values.push((edge_id, attr_value));
                }
                rust_attrs.insert(attr_name, attr_values);
                
            } else {
                return Err(PyValueError::new_err(format!(
                    "Invalid format for attribute '{}'. Expected dict or list of tuples.", 
                    attr_name
                )));
            }
        }

        // DELEGATION: Use core bulk operation
        self.inner
            .borrow_mut()
            .set_edge_attrs(rust_attrs)
            .map_err(graph_error_to_py_err)
    }

    /// Check if node has specific attribute - delegates to core
    pub fn has_node_attribute(&self, node_id: NodeId, attr_name: &str) -> bool {
        self.inner.borrow().get_node_attr(node_id, attr_name)
            .map(|opt| opt.is_some())
            .unwrap_or(false)
    }

    /// Check if edge has specific attribute - delegates to core  
    pub fn has_edge_attribute(&self, edge_id: EdgeId, attr_name: &str) -> bool {
        self.inner.borrow().get_edge_attr(edge_id, attr_name)
            .map(|opt| opt.is_some())
            .unwrap_or(false)
    }

    /// Get all attribute keys for a node - delegates to core
    pub fn node_attribute_keys(&self, node_id: NodeId) -> Vec<String> {
        self.inner.borrow()
            .get_node_attrs(node_id)
            .map(|attrs| attrs.keys().cloned().collect())
            .unwrap_or_else(|_| vec![])
    }

    /// Get all attribute keys for an edge - delegates to core
    pub fn edge_attribute_keys(&self, edge_id: EdgeId) -> Vec<String> {
        self.inner.borrow()
            .get_edge_attrs(edge_id)
            .map(|attrs| attrs.keys().cloned().collect())
            .unwrap_or_else(|_| vec![])
    }
}