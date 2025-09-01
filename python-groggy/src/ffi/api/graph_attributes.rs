//! Graph Attribute Operations - Clean 12-Method Interface
//!
//! Simple, structured attribute operations with pure delegation to core.

use crate::ffi::core::array::PyGraphArray;
use crate::ffi::core::table::PyGraphTable;
use crate::ffi::types::PyAttrValue;
use crate::ffi::utils::{graph_error_to_py_err, python_value_to_attr_value};
use groggy::{AttrName, AttrValue, EdgeId, NodeId};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

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

    pub fn get_node_attr(
        &self,
        py: Python,
        node: NodeId,
        attr: String,
        default: Option<&PyAny>,
    ) -> PyResult<PyObject> {
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

    pub fn get_node_attrs(
        &self,
        py: Python,
        nodes: Vec<NodeId>,
        attrs: Vec<AttrName>,
    ) -> PyResult<PyObject> {
        let result = self
            .graph
            .borrow()
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

    pub fn get_edge_attr(
        &self,
        py: Python,
        edge: EdgeId,
        attr: String,
        default: Option<&PyAny>,
    ) -> PyResult<PyObject> {
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

    pub fn get_edge_attrs(
        &self,
        py: Python,
        edges: Vec<EdgeId>,
        attrs: Vec<String>,
    ) -> PyResult<PyObject> {
        let result = self
            .graph
            .borrow()
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
        self.graph
            .borrow()
            .get_node_attr(node_id, &attr_name.to_string())
            .map(|opt| opt.is_some())
            .unwrap_or(false)
    }

    pub fn has_edge_attribute(&self, _py: Python, edge_id: EdgeId, attr_name: &str) -> bool {
        self.graph
            .borrow()
            .get_edge_attr(edge_id, &attr_name.to_string())
            .map(|opt| opt.is_some())
            .unwrap_or(false)
    }

    pub fn node_attribute_keys(&self, _py: Python, node_id: NodeId) -> Vec<String> {
        self.graph
            .borrow()
            .get_node_attrs(node_id)
            .map(|attrs| attrs.keys().cloned().collect())
            .unwrap_or_else(|_| vec![])
    }

    pub fn edge_attribute_keys(&self, _py: Python, edge_id: EdgeId) -> Vec<String> {
        self.graph
            .borrow()
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

    pub fn set_node_attr(
        &mut self,
        _py: Python,
        node: NodeId,
        attr: String,
        value: &PyAny,
    ) -> PyResult<()> {
        let attr_value = python_value_to_attr_value(value)?;
        self.graph
            .borrow_mut()
            .set_node_attr(node, attr, attr_value)
            .map_err(graph_error_to_py_err)
    }

    pub fn set_node_attrs(&mut self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        // Convert any format to standardized internal format
        let standardized_attrs = self.normalize_attrs_format(py, attrs_dict, true)?;
        self.graph
            .borrow_mut()
            .set_node_attrs(standardized_attrs)
            .map_err(graph_error_to_py_err)
    }

    pub fn set_edge_attr(
        &mut self,
        _py: Python,
        edge: EdgeId,
        attr: String,
        value: &PyAny,
    ) -> PyResult<()> {
        let attr_value = python_value_to_attr_value(value)?;
        self.graph
            .borrow_mut()
            .set_edge_attr(edge, attr, attr_value)
            .map_err(graph_error_to_py_err)
    }

    pub fn set_edge_attrs(&mut self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        // Convert any format to standardized internal format
        let standardized_attrs = self.normalize_attrs_format(py, attrs_dict, false)?;
        self.graph
            .borrow_mut()
            .set_edge_attrs(standardized_attrs)
            .map_err(graph_error_to_py_err)
    }

    // === FORMAT DETECTION AND NORMALIZATION ===

    /// Intelligently detect and normalize various bulk attribute formats
    ///
    /// Supports multiple input formats:
    /// 1. Node-centric: {"attr": {node_id: value, node_id: value}}
    /// 2. Column-centric: {"attr": {"nodes": [node_ids], "values": [values]}}
    /// 3. GraphArray: {"attr": GraphArray([values])} (with matching node order)
    /// 4. GraphTable: GraphTable with node_id column + attribute columns
    fn normalize_attrs_format<T>(
        &self,
        py: Python,
        attrs_dict: &PyDict,
        is_nodes: bool,
    ) -> PyResult<HashMap<String, Vec<(T, AttrValue)>>>
    where
        T: for<'py> FromPyObject<'py> + Copy + std::fmt::Display,
    {
        let mut normalized_attrs: HashMap<String, Vec<(T, AttrValue)>> = HashMap::new();

        for (attr_name_py, attr_data_py) in attrs_dict.iter() {
            let attr_name: String = attr_name_py.extract()?;
            let attr_values = self.parse_attribute_format::<T>(py, attr_data_py, is_nodes)?;
            normalized_attrs.insert(attr_name, attr_values);
        }

        Ok(normalized_attrs)
    }

    /// Parse a single attribute's data from any supported format
    fn parse_attribute_format<T>(
        &self,
        py: Python,
        attr_data: &PyAny,
        is_nodes: bool,
    ) -> PyResult<Vec<(T, AttrValue)>>
    where
        T: for<'py> FromPyObject<'py> + Copy + std::fmt::Display,
    {
        // Format 1: Node-centric dict {node_id: value, node_id: value}
        if let Ok(node_dict) = attr_data.extract::<&PyDict>() {
            // Check if it looks like column-centric format {"nodes": [...], "values": [...]}
            if node_dict.contains("nodes")? && node_dict.contains("values")? {
                return self.parse_column_centric_format::<T>(py, node_dict);
            }
            // Otherwise treat as node-centric format
            return self.parse_node_centric_format::<T>(py, node_dict);
        }

        // Format 3: GraphArray
        if let Ok(graph_array) = attr_data.extract::<PyRef<PyGraphArray>>() {
            return self.parse_graph_array_format::<T>(py, &graph_array, is_nodes);
        }

        // Format 4: GraphTable
        if let Ok(graph_table) = attr_data.extract::<PyRef<PyGraphTable>>() {
            return self.parse_graph_table_format::<T>(py, &graph_table, is_nodes);
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            format!(
                "Unsupported attribute format. Expected: dict {{id: value}}, dict {{\"nodes\": [...], \"values\": [...]}}, GraphArray, or GraphTable. Got: {}",
                attr_data.get_type().name()?
            )
        ))
    }

    /// Parse node-centric format: {node_id: value, node_id: value}
    fn parse_node_centric_format<T>(
        &self,
        _py: Python,
        node_dict: &PyDict,
    ) -> PyResult<Vec<(T, AttrValue)>>
    where
        T: for<'py> FromPyObject<'py> + Copy + std::fmt::Display,
    {
        let mut attr_values = Vec::new();
        for (id_py, value_py) in node_dict.iter() {
            let id: T = id_py.extract()?;
            let attr_value = python_value_to_attr_value(value_py)?;
            attr_values.push((id, attr_value));
        }
        Ok(attr_values)
    }

    /// Parse column-centric format: {"nodes": [node_ids], "values": [values]}
    fn parse_column_centric_format<T>(
        &self,
        _py: Python,
        data_dict: &PyDict,
    ) -> PyResult<Vec<(T, AttrValue)>>
    where
        T: for<'py> FromPyObject<'py> + Copy + std::fmt::Display,
    {
        let ids_list = data_dict.get_item("nodes")?.ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err("Missing 'nodes' key in column-centric format")
        })?;
        let values_list = data_dict.get_item("values")?.ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err("Missing 'values' key in column-centric format")
        })?;

        let ids: Vec<T> = ids_list.extract()?;
        let values: &PyList = values_list.extract()?;

        if ids.len() != values.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Length mismatch: {} nodes vs {} values",
                ids.len(),
                values.len()
            )));
        }

        let mut attr_values = Vec::new();
        for (id, value_py) in ids.into_iter().zip(values.iter()) {
            let attr_value = python_value_to_attr_value(value_py)?;
            attr_values.push((id, attr_value));
        }

        Ok(attr_values)
    }

    /// Parse GraphArray format: values correspond to graph order
    /// Note: Complex type conversion deferred to next phase
    fn parse_graph_array_format<T>(
        &self,
        _py: Python,
        _graph_array: &PyRef<PyGraphArray>,
        is_nodes: bool,
    ) -> PyResult<Vec<(T, AttrValue)>>
    where
        T: for<'py> FromPyObject<'py> + Copy + std::fmt::Display,
    {
        // GraphArray support requires complex type conversion between NodeId/EdgeId and T
        // This will be implemented in the next phase once we have proper type mappings
        Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "GraphArray format for {} not yet implemented - complex type conversion needed",
            if is_nodes { "nodes" } else { "edges" }
        )))
    }

    /// Parse GraphTable format: table with id column + attribute columns
    fn parse_graph_table_format<T>(
        &self,
        _py: Python,
        _graph_table: &PyRef<PyGraphTable>,
        _is_nodes: bool,
    ) -> PyResult<Vec<(T, AttrValue)>>
    where
        T: for<'py> FromPyObject<'py> + Copy + std::fmt::Display,
    {
        // GraphTable support will be implemented in the next phase
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "GraphTable format support coming in next phase",
        ))
    }
}
