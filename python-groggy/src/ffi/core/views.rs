//! Views FFI Bindings
//!
//! Python bindings for NodeView and EdgeView components.

use groggy::{EdgeId, NodeId};
use pyo3::exceptions::{PyKeyError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// Import types from our FFI modules
use crate::ffi::api::graph::PyGraph;
use crate::ffi::types::PyAttrValue;

/// A view of a specific node with access to its attributes
#[pyclass(name = "NodeView")]
pub struct PyNodeView {
    pub graph: Py<PyGraph>,
    pub node_id: NodeId,
}

#[pymethods]
impl PyNodeView {
    /// Get node attribute value
    fn __getitem__(&self, py: Python, key: &str) -> PyResult<PyAttrValue> {
        let graph = self.graph.borrow(py);
        match graph.get_node_attribute(self.node_id, key.to_string())? {
            Some(value) => Ok(value),
            None => Err(PyKeyError::new_err(format!(
                "Attribute '{}' not found on node {}",
                key, self.node_id
            ))),
        }
    }

    /// Set node attribute value (chainable)
    fn __setitem__(&mut self, py: Python, key: &str, value: PyAttrValue) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);
        graph.set_node_attribute(self.node_id, key.to_string(), &value)?;
        Ok(())
    }

    /// Get node ID
    #[getter]
    fn id(&self) -> PyResult<NodeId> {
        Ok(self.node_id)
    }

    /// Check if attribute exists
    fn __contains__(&self, py: Python, key: &str) -> PyResult<bool> {
        let graph = self.graph.borrow(py);
        Ok(graph.has_node_attribute(self.node_id, key))
    }

    /// Get all attribute keys
    fn keys(&self, py: Python) -> PyResult<Vec<String>> {
        let graph = self.graph.borrow(py);
        Ok(graph.node_attribute_keys(self.node_id))
    }

    /// Get all attribute values
    fn values(&self, py: Python) -> PyResult<Vec<PyAttrValue>> {
        let graph = self.graph.borrow(py);
        // 🚀 PERFORMANCE FIX: Use batch attribute access instead of manual loops
        let node_attrs = graph.inner.get_node_attrs(self.node_id).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get node attributes: {}", e))
        })?;

        let values = node_attrs
            .into_iter()
            .map(|(_, value)| PyAttrValue::from_attr_value(value))
            .collect();
        Ok(values)
    }

    /// Get neighbors of this node
    fn neighbors(&self, py: Python) -> PyResult<Vec<NodeId>> {
        let graph = self.graph.borrow(py);
        graph.inner.neighbors(self.node_id).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get neighbors: {}", e))
        })
    }

    /// Get all attribute items as (key, value) pairs
    fn items(&self, py: Python) -> PyResult<Vec<(String, PyAttrValue)>> {
        let graph = self.graph.borrow(py);
        // 🚀 PERFORMANCE FIX: Use batch attribute access instead of manual loops
        let node_attrs = graph.inner.get_node_attrs(self.node_id).map_err(|e| {
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
        let mut graph = self.graph.borrow_mut(py);

        for (key, value) in attributes.iter() {
            let key_str = key.extract::<String>()?;
            let attr_value = PyAttrValue::extract(value)?.to_attr_value();
            graph.set_node_attribute(
                self.node_id,
                key_str,
                &PyAttrValue::from_attr_value(attr_value),
            )?;
        }

        // Return self for chaining
        Ok(self.clone().into_py(py))
    }

    /// String representation
    fn __str__(&self, py: Python) -> PyResult<String> {
        let graph = self.graph.borrow(py);
        let keys = graph.node_attribute_keys(self.node_id);

        if keys.is_empty() {
            Ok(format!("NodeView({})", self.node_id))
        } else {
            let mut attr_parts = Vec::new();
            for key in keys.iter().take(3) {
                // Show first 3 attributes
                if let Ok(Some(value)) = graph.get_node_attribute(self.node_id, key.clone()) {
                    attr_parts.push(format!("{}={}", key, value.__str__()?));
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
        let graph = self.graph.borrow(py);
        let keys = graph.node_attribute_keys(self.node_id);

        for key in keys {
            if let Some(value) = graph
                .get_node_attribute(self.node_id, key.clone())
                .ok()
                .flatten()
            {
                dict.set_item(key, value)?;
            }
        }

        Ok(dict.to_object(py))
    }

    /// Alias for to_dict() - get as dictionary
    fn item(&self, py: Python) -> PyResult<PyObject> {
        self.to_dict(py)
    }

    /// Iterator support - iterates over (key, value) pairs
    fn __iter__(&self, py: Python) -> PyResult<NodeViewIterator> {
        let graph = self.graph.borrow(py);
        let keys = graph.node_attribute_keys(self.node_id);

        let mut items = Vec::new();
        for key in keys {
            if let Some(value) = graph
                .get_node_attribute(self.node_id, key.clone())
                .ok()
                .flatten()
            {
                items.push((key, value));
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
#[pyclass(name = "EdgeView")]
pub struct PyEdgeView {
    pub graph: Py<PyGraph>,
    pub edge_id: EdgeId,
}

#[pymethods]
impl PyEdgeView {
    /// Get edge attribute value
    fn __getitem__(&self, py: Python, key: &str) -> PyResult<PyAttrValue> {
        let graph = self.graph.borrow(py);
        match graph.get_edge_attribute(self.edge_id, key.to_string())? {
            Some(value) => Ok(value),
            None => Err(PyKeyError::new_err(format!(
                "Attribute '{}' not found on edge {}",
                key, self.edge_id
            ))),
        }
    }

    /// Set edge attribute value (chainable)
    fn __setitem__(&mut self, py: Python, key: &str, value: PyAttrValue) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);
        graph.set_edge_attribute(self.edge_id, key.to_string(), &value)?;
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
    fn source(&self, py: Python) -> PyResult<NodeId> {
        let graph = self.graph.borrow(py);
        let (source, _) = graph
            .inner
            .edge_endpoints(self.edge_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get edge endpoints: {}", e)))?;
        Ok(source)
    }

    /// Get target node ID  
    #[getter]
    fn target(&self, py: Python) -> PyResult<NodeId> {
        let graph = self.graph.borrow(py);
        let (_, target) = graph
            .inner
            .edge_endpoints(self.edge_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get edge endpoints: {}", e)))?;
        Ok(target)
    }

    /// Get both endpoints as (source, target) tuple
    fn endpoints(&self, py: Python) -> PyResult<(NodeId, NodeId)> {
        let graph = self.graph.borrow(py);
        let endpoints = graph
            .inner
            .edge_endpoints(self.edge_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get edge endpoints: {}", e)))?;
        Ok(endpoints)
    }

    /// Check if attribute exists
    fn __contains__(&self, py: Python, key: &str) -> PyResult<bool> {
        let graph = self.graph.borrow(py);
        Ok(graph.has_edge_attribute(self.edge_id, key))
    }

    /// Get all attribute keys
    fn keys(&self, py: Python) -> PyResult<Vec<String>> {
        let graph = self.graph.borrow(py);
        Ok(graph.edge_attribute_keys(self.edge_id))
    }

    /// Get all attribute values
    fn values(&self, py: Python) -> PyResult<Vec<PyAttrValue>> {
        let graph = self.graph.borrow(py);
        // 🚀 PERFORMANCE FIX: Use batch attribute access instead of manual loops
        let edge_attrs = graph.inner.get_edge_attrs(self.edge_id).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get edge attributes: {}", e))
        })?;

        let values = edge_attrs
            .into_iter()
            .map(|(_, value)| PyAttrValue::from_attr_value(value))
            .collect();
        Ok(values)
    }

    /// Get all attribute items as (key, value) pairs
    fn items(&self, py: Python) -> PyResult<Vec<(String, PyAttrValue)>> {
        let graph = self.graph.borrow(py);
        // 🚀 PERFORMANCE FIX: Use batch attribute access instead of manual loops
        let edge_attrs = graph.inner.get_edge_attrs(self.edge_id).map_err(|e| {
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
        let mut graph = self.graph.borrow_mut(py);

        for (key, value) in attributes.iter() {
            let key_str = key.extract::<String>()?;
            let attr_value = PyAttrValue::extract(value)?.to_attr_value();
            graph.set_edge_attribute(
                self.edge_id,
                key_str,
                &PyAttrValue::from_attr_value(attr_value),
            )?;
        }

        // Return self for chaining
        Ok(self.clone().into_py(py))
    }

    /// String representation
    fn __str__(&self, py: Python) -> PyResult<String> {
        let graph = self.graph.borrow(py);
        let keys = graph.edge_attribute_keys(self.edge_id);

        // Get endpoints for display
        let (source, target) = match graph.inner.edge_endpoints(self.edge_id) {
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
                if let Ok(Some(value)) = graph.get_edge_attribute(self.edge_id, key.clone()) {
                    attr_parts.push(format!("{}={}", key, value.__str__()?));
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
        let graph = self.graph.borrow(py);
        let keys = graph.edge_attribute_keys(self.edge_id);

        for key in keys {
            if let Some(value) = graph
                .get_edge_attribute(self.edge_id, key.clone())
                .ok()
                .flatten()
            {
                dict.set_item(key, value)?;
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
        let graph = self.graph.borrow(py);
        let keys = graph.edge_attribute_keys(self.edge_id);

        let mut items = Vec::new();
        for key in keys {
            if let Some(value) = graph
                .get_edge_attribute(self.edge_id, key.clone())
                .ok()
                .flatten()
            {
                items.push((key, value));
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
