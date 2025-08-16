//! Graph Core FFI Bindings
//! 
//! Core graph operations: add/remove nodes/edges, basic properties.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::{PyValueError, PyTypeError, PyRuntimeError, PyKeyError};
use groggy::{Graph as RustGraph, NodeId, EdgeId, AttrValue as RustAttrValue};
use std::collections::HashMap;

/// Utility function to convert Python values to AttrValue
pub fn python_value_to_attr_value(value: &PyAny) -> PyResult<RustAttrValue> {
    if let Ok(int_val) = value.extract::<i64>() {
        Ok(RustAttrValue::Int(int_val))
    } else if let Ok(float_val) = value.extract::<f64>() {
        Ok(RustAttrValue::Float(float_val as f32))
    } else if let Ok(str_val) = value.extract::<String>() {
        Ok(RustAttrValue::Text(str_val))
    } else if let Ok(bool_val) = value.extract::<bool>() {
        Ok(RustAttrValue::Bool(bool_val))
    } else {
        Err(PyTypeError::new_err("Unsupported attribute value type"))
    }
}

/// Error conversion helper
pub fn graph_error_to_py_err(err: groggy::GraphError) -> PyErr {
    PyRuntimeError::new_err(format!("Graph error: {}", err))
}

/// Core graph operations for the main PyGraph implementation
pub struct PyGraphCore;

impl PyGraphCore {
    /// Create a new graph
    pub fn new(_config: Option<&PyDict>) -> PyResult<RustGraph> {
        Ok(RustGraph::new())
    }
    
    /// Add a single node with optional attributes
    pub fn add_node(graph: &mut RustGraph, kwargs: Option<&PyDict>) -> PyResult<NodeId> {
        let node_id = graph.add_node();
        
        if let Some(attrs) = kwargs {
            if !attrs.is_empty() {
                for (key, value) in attrs.iter() {
                    let attr_name: String = key.extract()?;
                    let attr_value = python_value_to_attr_value(value)?;
                    
                    graph.set_node_attr(node_id, attr_name, attr_value)
                        .map_err(graph_error_to_py_err)?;
                }
            }
        }
        
        Ok(node_id)
    }
    
    /// Add an edge with optional attributes
    pub fn add_edge(graph: &mut RustGraph, source: NodeId, target: NodeId, kwargs: Option<&PyDict>) -> PyResult<EdgeId> {
        let edge_id = graph.add_edge(source, target)
            .map_err(graph_error_to_py_err)?;
        
        if let Some(attrs) = kwargs {
            if !attrs.is_empty() {
                for (key, value) in attrs.iter() {
                    let attr_name: String = key.extract()?;
                    let attr_value = python_value_to_attr_value(value)?;
                    
                    graph.set_edge_attr(edge_id, attr_name, attr_value)
                        .map_err(graph_error_to_py_err)?;
                }
            }
        }
        
        Ok(edge_id)
    }
    
    /// Basic graph properties
    pub fn node_count(graph: &RustGraph) -> usize {
        graph.node_ids().len()
    }
    
    pub fn edge_count(graph: &RustGraph) -> usize {
        graph.edge_ids().len()
    }
    
    pub fn has_node(graph: &RustGraph, node_id: NodeId) -> bool {
        graph.node_ids().contains(&node_id)
    }
    
    pub fn has_edge(graph: &RustGraph, edge_id: EdgeId) -> bool {
        graph.edge_ids().contains(&edge_id)
    }
    
    /// Attribute operations
    pub fn get_node_attribute(graph: &RustGraph, node_id: NodeId, attr_name: &str) -> Option<RustAttrValue> {
        graph.get_node_attr(node_id, &attr_name.to_string()).ok().flatten()
    }
    
    pub fn set_node_attribute(graph: &mut RustGraph, node_id: NodeId, attr_name: &str, attr_value: RustAttrValue) {
        let _ = graph.set_node_attr(node_id, attr_name.to_string(), attr_value);
    }
    
    pub fn get_edge_attribute(graph: &RustGraph, edge_id: EdgeId, attr_name: &str) -> Option<RustAttrValue> {
        graph.get_edge_attr(edge_id, &attr_name.to_string()).ok().flatten()
    }
    
    pub fn set_edge_attribute(graph: &mut RustGraph, edge_id: EdgeId, attr_name: &str, attr_value: RustAttrValue) {
        let _ = graph.set_edge_attr(edge_id, attr_name.to_string(), attr_value);
    }
    
    /// Get node degree
    pub fn node_degree(graph: &RustGraph, node_id: NodeId) -> Result<usize, groggy::GraphError> {
        graph.degree(node_id)
    }
    
    /// Get edge endpoints
    pub fn edge_endpoints(graph: &RustGraph, edge_id: EdgeId) -> Result<(NodeId, NodeId), groggy::GraphError> {
        graph.get_edge_endpoints(edge_id)
    }
    
    /// Get all node IDs
    pub fn node_ids(graph: &RustGraph) -> Vec<NodeId> {
        graph.node_ids()
    }
    
    /// Get all edge IDs  
    pub fn edge_ids(graph: &RustGraph) -> Vec<EdgeId> {
        graph.edge_ids()
    }
    
    /// Get neighbors of a node
    pub fn neighbors(graph: &RustGraph, node_id: NodeId) -> Result<Vec<NodeId>, groggy::GraphError> {
        graph.neighbors(node_id)
    }
    
    /// Get edges connected to a node
    pub fn node_edges(graph: &RustGraph, node_id: NodeId) -> Result<Vec<EdgeId>, groggy::GraphError> {
        graph.node_edges(node_id)
    }
}
