//! Graph Attribute Operations - Clean 12-Method Interface
//!
//! Simple, structured attribute operations with pure delegation to core.

use crate::ffi::types::PyAttrValue;
use crate::ffi::utils::{python_value_to_attr_value, graph_error_to_py_err};
use groggy::{AttrName, AttrValue, EdgeId, NodeId};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use super::graph::PyGraph;

/// Clean attribute operations - 12 essential methods only

/// Immutable attribute access (getters/utilities)
pub struct PyGraphAttr {
    pub graph: std::rc::Rc<std::cell::RefCell<groggy::Graph>>,
}

impl PyGraphAttr {
    pub fn new(graph: std::rc::Rc<std::cell::RefCell<groggy::Graph>>) -> Self {
        Self { graph }
    }

    // === FOUR CORE GETTERS ===

    pub fn get_node_attr(&self, py: Python, node: NodeId, attr: String, default: Option<&PyAny>) -> PyResult<PyObject> {
        match self.graph.borrow().get_node_attr(node, &attr) {
            Ok(Some(attr_value)) => {
                let py_attr_value = PyAttrValue::new(attr_value);
                Ok(py_attr_value.to_object(py))
            }
            Ok(None) => {
                if let Some(default_val) = default {
                    Ok(default_val.to_object(py))
                } else {
                    Ok(py.None())
                }
            }
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }

    pub fn get_node_attrs(&self, py: Python, nodes: Vec<NodeId>, attrs: Vec<AttrName>) -> PyResult<PyObject> {
        let result = self.graph.borrow()
            .get_node_attrs_bulk(nodes, attrs)
            .map_err(graph_error_to_py_err)?;
        let py_dict = PyDict::new(py);
        for (node_id, node_attrs) in result {
            let node_dict = PyDict::new(py);
            for (attr_name, attr_value) in node_attrs {
                let py_attr_value = PyAttrValue::new(attr_value);
                node_dict.set_item(attr_name, py_attr_value)?;
            }
            py_dict.set_item(node_id, node_dict)?;
        }
        Ok(py_dict.to_object(py))
    }

    pub fn get_edge_attr(&self, py: Python, edge: EdgeId, attr: String, default: Option<&PyAny>) -> PyResult<PyObject> {
        match self.graph.borrow().get_edge_attr(edge, &attr) {
            Ok(Some(attr_value)) => {
                let py_attr_value = PyAttrValue::new(attr_value);
                Ok(py_attr_value.to_object(py))
            }
            Ok(None) => {
                if let Some(default_val) = default {
                    Ok(default_val.to_object(py))
                } else {
                    Ok(py.None())
                }
            }
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }

    pub fn get_edge_attrs(&self, py: Python, edges: Vec<EdgeId>, attrs: Vec<String>) -> PyResult<PyObject> {
        let result = self.graph.borrow()
            .get_edge_attrs_bulk(edges, attrs)
            .map_err(graph_error_to_py_err)?;
        let py_dict = PyDict::new(py);
        for (edge_id, edge_attrs) in result {
            let edge_dict = PyDict::new(py);
            for (attr_name, attr_value) in edge_attrs {
                let py_attr_value = PyAttrValue::new(attr_value);
                edge_dict.set_item(attr_name, py_attr_value)?;
            }
            py_dict.set_item(edge_id, edge_dict)?;
        }
        Ok(py_dict.to_object(py))
    }

    // === FOUR UTILITY METHODS ===

    pub fn has_node_attribute(&self, _py: Python, node_id: NodeId, attr_name: &str) -> bool {
        self.graph.borrow()
            .get_node_attr(node_id, &attr_name.to_string())
            .map(|opt| opt.is_some())
            .unwrap_or(false)
    }

    pub fn has_edge_attribute(&self, _py: Python, edge_id: EdgeId, attr_name: &str) -> bool {
        self.graph.borrow()
            .get_edge_attr(edge_id, &attr_name.to_string())
            .map(|opt| opt.is_some())
            .unwrap_or(false)
    }

    pub fn node_attribute_keys(&self, _py: Python, node_id: NodeId) -> Vec<String> {
        self.graph.borrow()
            .get_node_attrs(node_id)
            .map(|attrs| attrs.keys().cloned().collect())
            .unwrap_or_else(|_| vec![])
    }

    pub fn edge_attribute_keys(&self, _py: Python, edge_id: EdgeId) -> Vec<String> {
        self.graph.borrow()
            .get_edge_attrs(edge_id)
            .map(|attrs| attrs.keys().cloned().collect())
            .unwrap_or_else(|_| vec![])
    }
}

/// Mutable attribute access (setters)
pub struct PyGraphAttrMut {
    pub graph: std::rc::Rc<std::cell::RefCell<groggy::Graph>>,
}

impl PyGraphAttrMut {
    pub fn new(graph: std::rc::Rc<std::cell::RefCell<groggy::Graph>>) -> Self {
        Self { graph }
    }

    pub fn set_node_attr(&mut self, py: Python, node: NodeId, attr: String, value: &PyAny) -> PyResult<()> {
        let attr_value = python_value_to_attr_value(value)?;
        self.graph.borrow_mut()
            .set_node_attr(node, attr, attr_value)
            .map_err(graph_error_to_py_err)
    }

    pub fn set_node_attrs(&mut self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        let mut rust_attrs: HashMap<String, Vec<(NodeId, AttrValue)>> = HashMap::new();
        for (attr_name_py, node_values_py) in attrs_dict.iter() {
            let attr_name: String = attr_name_py.extract()?;
            let node_dict: &PyDict = node_values_py.extract()?;
            let mut attr_values = Vec::new();
            for (node_py, value_py) in node_dict.iter() {
                let node_id: NodeId = node_py.extract()?;
                let attr_value = python_value_to_attr_value(value_py)?;
                attr_values.push((node_id, attr_value));
            }
            rust_attrs.insert(attr_name, attr_values);
        }
        self.graph.borrow_mut()
            .set_node_attrs(rust_attrs)
            .map_err(graph_error_to_py_err)
    }

    pub fn set_edge_attr(&mut self, py: Python, edge: EdgeId, attr: String, value: &PyAny) -> PyResult<()> {
        let attr_value = python_value_to_attr_value(value)?;
        self.graph.borrow_mut()
            .set_edge_attr(edge, attr, attr_value)
            .map_err(graph_error_to_py_err)
    }

    pub fn set_edge_attrs(&mut self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        let mut rust_attrs: HashMap<String, Vec<(EdgeId, AttrValue)>> = HashMap::new();
        for (attr_name_py, edge_values_py) in attrs_dict.iter() {
            let attr_name: String = attr_name_py.extract()?;
            let edge_dict: &PyDict = edge_values_py.extract()?;
            let mut attr_values = Vec::new();
            for (edge_py, value_py) in edge_dict.iter() {
                let edge_id: EdgeId = edge_py.extract()?;
                let attr_value = python_value_to_attr_value(value_py)?;
                attr_values.push((edge_id, attr_value));
            }
            rust_attrs.insert(attr_name, attr_values);
        }
        self.graph.borrow_mut()
            .set_edge_attrs(rust_attrs)
            .map_err(graph_error_to_py_err)
    }
}