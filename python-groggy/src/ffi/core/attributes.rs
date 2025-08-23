//! Attributes FFI Bindings
//!
//! Python bindings for attribute access operations.

use crate::ffi::core::array::PyGraphArray;
use crate::ffi::types::PyAttrValue;
use crate::ffi::utils::{
    attr_value_to_python_value, graph_error_to_py_err, python_value_to_attr_value,
};
use groggy::{AttrName, AttrValue as RustAttrValue, EdgeId, NodeId};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Attribute management utilities for graphs
pub struct AttributeManager;

impl AttributeManager {
    /// Set node attribute
    pub fn set_node_attribute(
        graph: &mut groggy::Graph,
        node: NodeId,
        attr: AttrName,
        value: &PyAttrValue,
    ) -> PyResult<()> {
        graph
            .set_node_attr(node, attr, value.inner.clone())
            .map_err(graph_error_to_py_err)
    }

    /// Set edge attribute
    pub fn set_edge_attribute(
        graph: &mut groggy::Graph,
        edge: EdgeId,
        attr: AttrName,
        value: &PyAttrValue,
    ) -> PyResult<()> {
        graph
            .set_edge_attr(edge, attr, value.inner.clone())
            .map_err(graph_error_to_py_err)
    }

    /// Get node attribute
    pub fn get_node_attribute(
        graph: &groggy::Graph,
        node: NodeId,
        attr: AttrName,
    ) -> PyResult<Option<PyAttrValue>> {
        match graph.get_node_attr(node, &attr) {
            Ok(Some(value)) => Ok(Some(PyAttrValue { inner: value })),
            Ok(None) => Ok(None),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }

    /// Get edge attribute
    pub fn get_edge_attribute(
        graph: &groggy::Graph,
        edge: EdgeId,
        attr: AttrName,
    ) -> PyResult<Option<PyAttrValue>> {
        match graph.get_edge_attr(edge, &attr) {
            Ok(Some(value)) => Ok(Some(PyAttrValue { inner: value })),
            Ok(None) => Ok(None),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }

    /// Get complete attribute column for ALL nodes (optimized for table() method)
    pub fn get_node_attribute_column(
        graph: &groggy::Graph,
        py: Python,
        attr_name: &str,
    ) -> PyResult<Py<PyGraphArray>> {
        match graph._get_node_attribute_column(&attr_name.to_string()) {
            Ok(values) => {
                // Convert Option<AttrValue> vector to AttrValue vector (convert None to appropriate AttrValue)
                let attr_values: Vec<RustAttrValue> = values
                    .into_iter()
                    .map(|opt_val| opt_val.unwrap_or(RustAttrValue::Int(0))) // Use default for None values
                    .collect();

                // Create GraphArray from the attribute values
                let graph_array = groggy::core::array::GraphArray::from_vec(attr_values);

                // Wrap in Python GraphArray
                let py_graph_array = PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?)
            }
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }

    /// Get complete attribute column for ALL edges (optimized for edge table() method)
    pub fn get_edge_attribute_column(
        graph: &groggy::Graph,
        py: Python,
        attr_name: &str,
    ) -> PyResult<Vec<PyObject>> {
        match graph._get_edge_attribute_column(&attr_name.to_string()) {
            Ok(values) => {
                let mut py_values = Vec::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(attr_value) => {
                            py_values.push(attr_value_to_python_value(py, &attr_value)?)
                        }
                        None => py_values.push(py.None()),
                    }
                }
                Ok(py_values)
            }
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }

    /// Get attribute column for specific nodes (optimized for subgraph tables)
    pub fn get_node_attributes_for_nodes(
        graph: &groggy::Graph,
        py: Python,
        node_ids: &[NodeId],
        attr_name: &str,
    ) -> PyResult<Vec<PyObject>> {
        match graph._get_node_attributes_for_nodes(node_ids, &attr_name.to_string()) {
            Ok(values) => {
                let mut py_values = Vec::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(attr_value) => {
                            py_values.push(attr_value_to_python_value(py, &attr_value)?)
                        }
                        None => py_values.push(py.None()),
                    }
                }
                Ok(py_values)
            }
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }

    /// Set node attributes from dictionary
    pub fn set_node_attributes_from_dict(
        graph: &mut groggy::Graph,
        node_id: NodeId,
        attrs: &PyDict,
    ) -> PyResult<()> {
        for (key, value) in attrs.iter() {
            let attr_name: String = key.extract()?;
            let attr_value = python_value_to_attr_value(value)?;
            graph
                .set_node_attr(node_id, attr_name, attr_value)
                .map_err(graph_error_to_py_err)?;
        }
        Ok(())
    }

    /// Set edge attributes from dictionary
    pub fn set_edge_attributes_from_dict(
        graph: &mut groggy::Graph,
        edge_id: EdgeId,
        attrs: &PyDict,
    ) -> PyResult<()> {
        for (key, value) in attrs.iter() {
            let attr_name: String = key.extract()?;
            let attr_value = python_value_to_attr_value(value)?;
            graph
                .set_edge_attr(edge_id, attr_name, attr_value)
                .map_err(graph_error_to_py_err)?;
        }
        Ok(())
    }

    /// Get all node attributes as dictionary
    pub fn get_node_attributes_dict(
        graph: &groggy::Graph,
        py: Python,
        node_id: NodeId,
    ) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        // Get all attributes for this node from the core graph
        match graph.get_node_attrs(node_id) {
            Ok(attrs) => {
                for (attr_name, attr_value) in attrs {
                    let py_value = attr_value_to_python_value(py, &attr_value)?;
                    dict.set_item(attr_name, py_value)?;
                }
            }
            Err(e) => return Err(graph_error_to_py_err(e)),
        }

        Ok(dict.to_object(py))
    }

    /// Get all edge attributes as dictionary
    pub fn get_edge_attributes_dict(
        graph: &groggy::Graph,
        py: Python,
        edge_id: EdgeId,
    ) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        // Get all attributes for this edge from the core graph
        match graph.get_edge_attrs(edge_id) {
            Ok(attrs) => {
                for (attr_name, attr_value) in attrs {
                    let py_value = attr_value_to_python_value(py, &attr_value)?;
                    dict.set_item(attr_name, py_value)?;
                }
            }
            Err(e) => return Err(graph_error_to_py_err(e)),
        }

        Ok(dict.to_object(py))
    }

    /// Set bulk node attributes (for performance optimization)
    pub fn set_node_attribute_bulk(
        graph: &mut groggy::Graph,
        attr_name: &str,
        attr_value: RustAttrValue,
        node_ids: &[NodeId],
    ) -> PyResult<()> {
        for &node_id in node_ids {
            graph
                .set_node_attr(node_id, attr_name.to_string(), attr_value.clone())
                .map_err(graph_error_to_py_err)?;
        }
        Ok(())
    }

    /// Set bulk edge attributes (for performance optimization)
    pub fn set_edge_attribute_bulk(
        graph: &mut groggy::Graph,
        attr_name: &str,
        attr_value: RustAttrValue,
        edge_ids: &[EdgeId],
    ) -> PyResult<()> {
        for &edge_id in edge_ids {
            graph
                .set_edge_attr(edge_id, attr_name.to_string(), attr_value.clone())
                .map_err(graph_error_to_py_err)?;
        }
        Ok(())
    }
}
