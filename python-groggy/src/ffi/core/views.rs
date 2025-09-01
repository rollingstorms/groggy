//! Views FFI Bindings
//!
//! Python bindings for NodeView and EdgeView components.

use groggy::{EdgeId, NodeId};
use pyo3::exceptions::{PyKeyError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// Import types from our FFI modules
use crate::ffi::api::graph_attributes::PyGraphAttr;
use crate::ffi::types::PyAttrValue;

/// A view of a specific node with access to its attributes
#[pyclass(name = "NodeView", unsendable)]
pub struct PyNodeView {
    pub graph: std::rc::Rc<std::cell::RefCell<groggy::Graph>>,
    pub node_id: NodeId,
}

#[pymethods]
impl PyNodeView {
    /// Get node attribute value
    fn __getitem__(&self, _py: Python, key: &str) -> PyResult<PyAttrValue> {
        let graph = self.graph.borrow();
        match graph.get_node_attr(self.node_id, &key.to_string()) {
            Ok(Some(attr_value)) => Ok(PyAttrValue::from_attr_value(attr_value)),
            Ok(None) => Err(PyKeyError::new_err(format!(
                "Node {} has no attribute '{}'",
                self.node_id, key
            ))),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get node attribute: {}",
                e
            ))),
        }
    }

    /// Set node attribute value (chainable)
    fn __setitem__(&mut self, _py: Python, key: &str, value: PyAttrValue) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut();
        let attr_value = value.to_attr_value();
        graph
            .set_node_attr(self.node_id, key.to_string(), attr_value)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set node attribute: {}", e)))?;
        Ok(())
    }

    /// Get node ID
    #[getter]
    fn id(&self) -> PyResult<NodeId> {
        Ok(self.node_id)
    }

    /// Check if attribute exists
    fn __contains__(&self, _py: Python, key: &str) -> PyResult<bool> {
        let graph = self.graph.borrow();
        match graph.get_node_attr(self.node_id, &key.to_string()) {
            Ok(Some(_)) => Ok(true),
            Ok(None) => Ok(false),
            Err(_) => Ok(false),
        }
    }

    /// Get all attribute keys
    fn keys(&self, _py: Python) -> PyResult<Vec<String>> {
        let graph = self.graph.borrow();
        match graph.get_node_attrs(self.node_id) {
            Ok(attrs) => Ok(attrs.keys().cloned().collect()),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get node attributes: {}",
                e
            ))),
        }
    }

    /// Get all attribute values
    fn values(&self, _py: Python) -> PyResult<Vec<PyAttrValue>> {
        let graph = self.graph.borrow();
        let node_attrs = graph.get_node_attrs(self.node_id).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get node attributes: {}", e))
        })?;
        let values = node_attrs
            .values()
            .map(|value| PyAttrValue::from_attr_value(value.clone()))
            .collect();
        Ok(values)
    }

    /// Get neighbors of this node
    fn neighbors(&self, _py: Python) -> PyResult<Vec<NodeId>> {
        let graph = self.graph.borrow();
        graph.neighbors(self.node_id).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get neighbors: {}", e))
        })
    }

    /// Get all attribute items as (key, value) pairs
    fn items(&self, _py: Python) -> PyResult<Vec<(String, PyAttrValue)>> {
        let graph = self.graph.borrow();
        let node_attrs = graph.get_node_attrs(self.node_id).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get node attributes: {}", e))
        })?;
        let items = node_attrs
            .into_iter()
            .map(|(key, value)| (key, PyAttrValue::from_attr_value(value)))
            .collect();
        Ok(items)
    }

    /// Update multiple attributes at once (chainable)
    fn update(&mut self, py: Python, attributes: &PyDict) -> PyResult<PyObject> {
        let mut graph = self.graph.borrow_mut();

        for (key, value) in attributes.iter() {
            let key_str = key.extract::<String>()?;
            let attr_value = PyAttrValue::extract(value)?.to_attr_value();
            graph
                .set_node_attr(self.node_id, key_str, attr_value)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to set node attribute: {}", e))
                })?;
        }

        // Return self for chaining
        Ok(self.clone().into_py(py))
    }

    /// String representation
    fn __str__(&self, _py: Python) -> PyResult<String> {
        let graph = self.graph.borrow();
        let keys = match graph.get_node_attrs(self.node_id) {
            Ok(attrs) => attrs.keys().cloned().collect::<Vec<_>>(),
            Err(_) => Vec::new(),
        };

        if keys.is_empty() {
            Ok(format!("NodeView({})", self.node_id))
        } else {
            let mut attr_parts = Vec::new();
            for key in keys.iter().take(3) {
                // Show first 3 attributes
                if let Ok(Some(attr_value)) = graph.get_node_attr(self.node_id, key) {
                    let py_attr = PyAttrValue::from_attr_value(attr_value);
                    attr_parts.push(format!("{}={}", key, py_attr.__str__()?));
                }
            }

            let attr_str = if keys.len() > 3 {
                format!("{}, ...", attr_parts.join(", "))
            } else {
                attr_parts.join(", ")
            };

            Ok(format!("NodeView({}, {})", self.node_id, attr_str))
        }
    }

    /// Get as dictionary
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let graph = self.graph.borrow();

        if let Ok(attrs) = graph.get_node_attrs(self.node_id) {
            for (key, value) in attrs {
                let py_attr = PyAttrValue::from_attr_value(value);
                dict.set_item(key, py_attr.to_object(py))?;
            }
        }

        Ok(dict.to_object(py))
    }

    /// Alias for to_dict() - get as dictionary
    fn item(&self, py: Python) -> PyResult<PyObject> {
        self.to_dict(py)
    }

    /// Iterator support - iterates over (key, value) pairs
    fn __iter__(&self, _py: Python) -> PyResult<NodeViewIterator> {
        let graph = self.graph.borrow();

        let mut items = Vec::new();
        if let Ok(attrs) = graph.get_node_attrs(self.node_id) {
            for (key, value) in attrs {
                let py_attr = PyAttrValue::from_attr_value(value);
                items.push((key, py_attr));
            }
        }

        Ok(NodeViewIterator { items, index: 0 })
    }
}

/// Iterator for NodeView that yields (key, value) pairs
#[pyclass]
pub struct NodeViewIterator {
    items: Vec<(String, crate::ffi::types::PyAttrValue)>,
    index: usize,
}

#[pymethods]
impl NodeViewIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        if self.index < self.items.len() {
            let (key, value) = &self.items[self.index];
            self.index += 1;

            // Return (key, value) tuple
            let tuple = pyo3::types::PyTuple::new(py, [key.to_object(py), value.to_object(py)]);
            Ok(Some(tuple.to_object(py)))
        } else {
            Ok(None)
        }
    }
}

impl Clone for PyNodeView {
    fn clone(&self) -> Self {
        PyNodeView {
            graph: self.graph.clone(),
            node_id: self.node_id,
        }
    }
}

/// A view of a specific edge with access to its attributes and endpoints
#[pyclass(name = "EdgeView", unsendable)]
pub struct PyEdgeView {
    pub graph: std::rc::Rc<std::cell::RefCell<groggy::Graph>>,
    pub edge_id: EdgeId,
}

#[pymethods]
impl PyEdgeView {
    /// Get edge attribute value
    fn __getitem__(&self, _py: Python, key: &str) -> PyResult<PyAttrValue> {
        let graph = self.graph.borrow();
        match graph.get_edge_attr(self.edge_id, &key.to_string()) {
            Ok(Some(attr_value)) => Ok(PyAttrValue::from_attr_value(attr_value)),
            Ok(None) => Err(PyKeyError::new_err(format!(
                "Edge {} has no attribute '{}'",
                self.edge_id, key
            ))),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get edge attribute: {}",
                e
            ))),
        }
    }

    /// Set edge attribute value (chainable)  
    fn __setitem__(&mut self, _py: Python, key: &str, value: PyAttrValue) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut();
        let attr_value = value.to_attr_value();
        graph
            .set_edge_attr(self.edge_id, key.to_string(), attr_value)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set edge attribute: {}", e)))?;
        Ok(())
    }

    /// Get edge ID
    #[getter]
    fn id(&self) -> PyResult<EdgeId> {
        Ok(self.edge_id)
    }

    /// Get edge ID (alias for id)
    #[getter]
    fn edge_id(&self) -> PyResult<EdgeId> {
        Ok(self.edge_id)
    }

    /// Get source node ID
    #[getter]
    fn source(&self, _py: Python) -> PyResult<NodeId> {
        let graph = self.graph.borrow();
        let (source, _) = graph
            .edge_endpoints(self.edge_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get edge endpoints: {}", e)))?;
        Ok(source)
    }

    /// Get target node ID  
    #[getter]
    fn target(&self, _py: Python) -> PyResult<NodeId> {
        let graph = self.graph.borrow();
        let (_, target) = graph
            .edge_endpoints(self.edge_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get edge endpoints: {}", e)))?;
        Ok(target)
    }

    /// Get both endpoints as (source, target) tuple
    fn endpoints(&self, _py: Python) -> PyResult<(NodeId, NodeId)> {
        let graph = self.graph.borrow();
        let endpoints = graph
            .edge_endpoints(self.edge_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get edge endpoints: {}", e)))?;
        Ok(endpoints)
    }

    /// Check if attribute exists
    fn __contains__(&self, _py: Python, key: &str) -> PyResult<bool> {
        let graph = self.graph.borrow();
        match graph.get_edge_attr(self.edge_id, &key.to_string()) {
            Ok(Some(_)) => Ok(true),
            Ok(None) => Ok(false),
            Err(_) => Ok(false),
        }
    }

    /// Get all attribute keys
    fn keys(&self, _py: Python) -> PyResult<Vec<String>> {
        let attr_handler = PyGraphAttr::new(self.graph.clone());
        Ok(attr_handler.edge_attribute_keys(_py, self.edge_id))
    }

    /// Get all attribute values
    fn values(&self, _py: Python) -> PyResult<Vec<PyAttrValue>> {
        let graph = self.graph.borrow();
        let edge_attrs = graph.get_edge_attrs(self.edge_id).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get edge attributes: {}", e))
        })?;
        let values = edge_attrs
            .into_values()
            .map(PyAttrValue::from_attr_value)
            .collect();
        Ok(values)
    }

    /// Get all attribute items as (key, value) pairs
    fn items(&self, _py: Python) -> PyResult<Vec<(String, PyAttrValue)>> {
        let graph = self.graph.borrow();
        let edge_attrs = graph.get_edge_attrs(self.edge_id).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get edge attributes: {}", e))
        })?;
        let items = edge_attrs
            .into_iter()
            .map(|(key, value)| (key, PyAttrValue::from_attr_value(value)))
            .collect();
        Ok(items)
    }

    /// Update multiple attributes at once (chainable)
    fn update(&mut self, py: Python, attributes: &PyDict) -> PyResult<PyObject> {
        let mut graph = self.graph.borrow_mut();

        for (key, value) in attributes.iter() {
            let key_str = key.extract::<String>()?;
            let attr_value = PyAttrValue::extract(value)?.to_attr_value();
            graph
                .set_edge_attr(self.edge_id, key_str, attr_value)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to set edge attribute: {}",
                        e
                    ))
                })?;
        }

        // Return self for chaining
        Ok(self.clone().into_py(py))
    }

    /// String representation
    fn __str__(&self, py: Python) -> PyResult<String> {
        let attr_handler = PyGraphAttr::new(self.graph.clone());
        let keys = attr_handler.edge_attribute_keys(py, self.edge_id);

        // Get endpoints for display
        let graph = self.graph.borrow();
        let (source, target) = match graph.edge_endpoints(self.edge_id) {
            Ok(endpoints) => endpoints,
            Err(_) => return Ok(format!("EdgeView({}) [invalid]", self.edge_id)),
        };

        if keys.is_empty() {
            Ok(format!(
                "EdgeView({}: {} -> {})",
                self.edge_id, source, target
            ))
        } else {
            let mut attr_parts = Vec::new();
            for key in keys.iter().take(3) {
                // Show first 3 attributes
                if let Ok(py_obj) = attr_handler.get_edge_attr(py, self.edge_id, key.clone(), None)
                {
                    if let Ok(value) = py_obj.extract::<PyAttrValue>(py) {
                        attr_parts.push(format!("{}={}", key, value.__str__()?));
                    }
                }
            }

            let attr_str = if keys.len() > 3 {
                format!("{}, ...", attr_parts.join(", "))
            } else {
                attr_parts.join(", ")
            };

            Ok(format!(
                "EdgeView({}: {} -> {}, {})",
                self.edge_id, source, target, attr_str
            ))
        }
    }

    /// Get as dictionary
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let attr_handler = PyGraphAttr::new(self.graph.clone());
        let keys = attr_handler.edge_attribute_keys(py, self.edge_id);

        for key in keys {
            if let Ok(py_obj) = attr_handler.get_edge_attr(py, self.edge_id, key.clone(), None) {
                dict.set_item(key, py_obj)?;
            }
        }

        Ok(dict.to_object(py))
    }

    /// Alias for to_dict() - get as dictionary
    fn item(&self, py: Python) -> PyResult<PyObject> {
        self.to_dict(py)
    }

    /// Iterator support - iterates over (key, value) pairs
    fn __iter__(&self, py: Python) -> PyResult<EdgeViewIterator> {
        let attr_handler = PyGraphAttr::new(self.graph.clone());
        let keys = attr_handler.edge_attribute_keys(py, self.edge_id);

        let mut items = Vec::new();
        for key in keys {
            if let Ok(py_obj) = attr_handler.get_edge_attr(py, self.edge_id, key.clone(), None) {
                if let Ok(value) = py_obj.extract::<PyAttrValue>(py) {
                    items.push((key, value));
                }
            }
        }

        Ok(EdgeViewIterator { items, index: 0 })
    }
}

/// Iterator for EdgeView that yields (key, value) pairs
#[pyclass]
pub struct EdgeViewIterator {
    items: Vec<(String, crate::ffi::types::PyAttrValue)>,
    index: usize,
}

#[pymethods]
impl EdgeViewIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        if self.index < self.items.len() {
            let (key, value) = &self.items[self.index];
            self.index += 1;

            // Return (key, value) tuple
            let tuple = pyo3::types::PyTuple::new(py, [key.to_object(py), value.to_object(py)]);
            Ok(Some(tuple.to_object(py)))
        } else {
            Ok(None)
        }
    }
}

impl Clone for PyEdgeView {
    fn clone(&self) -> Self {
        PyEdgeView {
            graph: self.graph.clone(),
            edge_id: self.edge_id,
        }
    }
}
