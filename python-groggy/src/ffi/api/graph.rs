//! Graph FFI Coordinator
//! 
//! Main Python bindings for the Graph API.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::{PyKeyError, PyTypeError};
use groggy::{Graph as RustGraph, NodeId, EdgeId, AttrValue as RustAttrValue, AttrName, StateId};

// Import all graph modules
use crate::ffi::types::PyAttrValue;
use crate::ffi::core::accessors::{PyNodesAccessor, PyEdgesAccessor};
use crate::ffi::core::views::{PyNodeView, PyEdgeView};
use crate::ffi::core::array::PyGraphArray;
use crate::ffi::core::subgraph::PySubgraph;
use crate::ffi::utils::{graph_error_to_py_err, python_value_to_attr_value, attr_value_to_python_value};

// Placeholder imports for missing types - these need to be implemented
struct PyAttributes {
    graph: *const RustGraph,
}

struct PyNodeFilter {
    inner: groggy::core::query::NodeFilter,
}

struct PyEdgeFilter {
    inner: groggy::core::query::EdgeFilter,
}

struct PyAggregationResult {
    value: f64, // Simplified for now
}

struct PyGroupedAggregationResult {
    value: PyObject,
}

struct PyBranchInfo {
    inner: groggy::core::history::BranchInfo,
}

struct PyCommit {
    inner: groggy::core::history::Commit,
}

struct PyHistoricalView {
    state_id: StateId,
}

struct PyGraphMatrix {
    columns: Vec<Py<PyGraphArray>>,
    column_names: Vec<String>,
    num_rows: usize,
}

// Helper function for matrix conversion
fn adjacency_matrix_to_py_graph_matrix(py: Python, matrix: groggy::core::array::GraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
    // Simplified implementation - needs proper matrix conversion
    let py_matrix = PyGraphMatrix {
        columns: Vec::new(),
        column_names: Vec::new(),
        num_rows: 0,
    };
    Ok(Py::new(py, py_matrix)?)
}

/// Python wrapper for the main Graph
#[pyclass(name = "Graph", unsendable)]
pub struct PyGraph {
    inner: RustGraph,
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new(_config: Option<&PyDict>) -> PyResult<Self> {
        // For now, ignore config and just create a default graph
        // TODO: Convert Python config to GraphConfig when needed
        let rust_graph = RustGraph::new();
        Ok(Self { inner: rust_graph })
    }
    
    // === CORE GRAPH OPERATIONS ===
    
    #[pyo3(signature = (**kwargs))]
    fn add_node(&mut self, kwargs: Option<&PyDict>) -> PyResult<NodeId> {
        let node_id = self.inner.add_node();
        
        // Fast path: if no kwargs, just return the node_id
        if let Some(attrs) = kwargs {
            if !attrs.is_empty() {
                // Only do attribute setting if we actually have attributes
                for (key, value) in attrs.iter() {
                    let attr_name: String = key.extract()?;
                    let attr_value = python_value_to_attr_value(value)?;
                    
                    self.inner.set_node_attr(node_id, attr_name, attr_value)
                        .map_err(graph_error_to_py_err)?;
                }
            }
        }
        
        Ok(node_id)
    }
    
    #[pyo3(signature = (data, uid_key = None))]
    fn add_nodes(&mut self, data: &PyAny, uid_key: Option<String>) -> PyResult<PyObject> {
        // Fast path optimization: Check for integer first (most common case)
        if let Ok(count) = data.extract::<usize>() {
            // Old API: add_nodes(5) -> [0, 1, 2, 3, 4] - fastest path
            let node_ids = self.inner.add_nodes(count);
            return Python::with_gil(|py| Ok(node_ids.to_object(py)));
        }
        
        // Only use Python::with_gil for complex operations
        Python::with_gil(|py| {
            if let Ok(node_data_list) = data.extract::<Vec<&PyDict>>() {
                // New API: add_nodes([{"id": "alice", "age": 30}, ...], id_key="id")
                let mut id_mapping = std::collections::HashMap::new();
                
                // Create all nodes first
                let node_ids = self.inner.add_nodes(node_data_list.len());
                
                // Process each node's data
                for (i, node_dict) in node_data_list.iter().enumerate() {
                    let node_id = node_ids[i];
                    
                    // Extract the ID if id_key is provided
                    if let Some(ref key) = uid_key {
                        match node_dict.get_item(key) {
                            Ok(Some(id_value)) => {
                                if let Ok(user_id) = id_value.extract::<String>() {
                                    id_mapping.insert(user_id, node_id);
                                }
                            }
                            Ok(None) => {
                                return Err(PyErr::new::<PyKeyError, _>(format!("Missing key: {}", key)));
                            }
                            Err(e) => return Err(e),
                        }
                    }
                    
                    // Set all attributes from the dict
                    for (attr_key, attr_value) in node_dict.iter() {
                        let attr_name: String = attr_key.extract()?;
                        
                        // Store all attributes including the id_key for later uid_key lookups
                        // (Previous version skipped id_key, but we need it for uid_key resolution)
                        
                        let attr_val = python_value_to_attr_value(attr_value)?;
                        self.inner.set_node_attr(node_id, attr_name, attr_val)
                            .map_err(graph_error_to_py_err)?;
                    }
                }
                
                // Return the mapping if id_key was provided, otherwise return node IDs
                if uid_key.is_some() {
                    Ok(id_mapping.to_object(py))
                } else {
                    Ok(node_ids.to_object(py))
                }
            } else {
                Err(PyErr::new::<PyTypeError, _>(
                    "add_nodes expects either an integer count or a list of dictionaries"
                ))
            }
        })
    }
    
    /// Helper method to resolve string ID to NodeId using uid_key attribute
    fn resolve_string_id_to_node(&self, string_id: &str, uid_key: &str) -> PyResult<NodeId> {
        let node_ids = self.inner.node_ids();
        
        for node_id in node_ids {
            if let Ok(Some(attr_value)) = self.inner.get_node_attr(node_id, &uid_key.to_string()) {
                match attr_value {
                    RustAttrValue::Text(s) => {
                        if s == string_id {
                            return Ok(node_id);
                        }
                    },
                    RustAttrValue::CompactText(s) => {
                        if s.as_str() == string_id {
                            return Ok(node_id);
                        }
                    },
                    _ => continue, // Skip non-text attributes
                }
            }
        }
        
        Err(PyErr::new::<PyKeyError, _>(format!("No node found with {}='{}'", uid_key, string_id)))
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("Graph(nodes={}, edges={})", self.node_count(), self.edge_count())
    }
    
    /// Python len() support - returns number of nodes
    fn __len__(&self) -> usize {
        self.inner.node_ids().len()
    }
    
    /// Check if a node exists in the graph
    fn has_node(&self, node_id: NodeId) -> bool {
        self.inner.contains_node(node_id)
    }
    
    /// Check if an edge exists in the graph
    fn has_edge(&self, edge_id: EdgeId) -> bool {
        self.inner.contains_edge(edge_id)
    }
    
    /// Get the number of nodes in the graph
    fn node_count(&self) -> usize {
        self.inner.node_ids().len()
    }
    
    /// Get the number of edges in the graph
    fn edge_count(&self) -> usize {
        self.inner.edge_ids().len()
    }
    
    // === ATTRIBUTE OPERATIONS ===
    
    fn set_node_attribute(&mut self, node: NodeId, attr: AttrName, value: &PyAttrValue) -> PyResult<()> {
        self.inner.set_node_attr(node, attr, value.inner.clone())
            .map_err(graph_error_to_py_err)
    }
    
    fn set_edge_attribute(&mut self, edge: EdgeId, attr: AttrName, value: &PyAttrValue) -> PyResult<()> {
        self.inner.set_edge_attr(edge, attr, value.inner.clone())
            .map_err(graph_error_to_py_err)
    }
    
    fn get_node_attribute(&self, node: NodeId, attr: AttrName) -> PyResult<Option<PyAttrValue>> {
        match self.inner.get_node_attr(node, &attr) {
            Ok(Some(value)) => Ok(Some(PyAttrValue { inner: value })),
            Ok(None) => Ok(None),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }
    
    fn get_edge_attribute(&self, edge: EdgeId, attr: AttrName) -> PyResult<Option<PyAttrValue>> {
        match self.inner.get_edge_attr(edge, &attr) {
            Ok(Some(value)) => Ok(Some(PyAttrValue { inner: value })),
            Ok(None) => Ok(None),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }

    fn get_edge_attributes(&self, edge: EdgeId, py: Python) -> PyResult<PyObject> {
        let attrs = self.inner.get_edge_attrs(edge)
            .map_err(graph_error_to_py_err)?;
        
        // Convert HashMap to Python dict
        let dict = PyDict::new(py);
        for (attr_name, attr_value) in attrs {
            let py_value = Py::new(py, PyAttrValue { inner: attr_value })?;
            dict.set_item(attr_name, py_value)?;
        }
        Ok(dict.to_object(py))
    }
    
    // === TOPOLOGY OPERATIONS ===
    
    fn contains_node(&self, node: NodeId) -> bool {
        self.inner.contains_node(node) 
    }
    
    fn contains_edge(&self, edge: EdgeId) -> bool {
        self.inner.contains_edge(edge)
    }
    
    fn edge_endpoints(&self, edge: EdgeId) -> PyResult<(NodeId, NodeId)> {
        self.inner.edge_endpoints(edge)
            .map_err(graph_error_to_py_err)
    }

    /// Get all active node IDs as GraphArray (lazy Rust view) - use .values for Python list
    #[getter]
    fn node_ids(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let node_ids = self.inner.node_ids();
        let attr_values: Vec<groggy::AttrValue> = node_ids.into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
    }
    
    /// Get all active edge IDs as GraphArray (lazy Rust view) - use .values for Python list
    #[getter]
    fn edge_ids(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let edge_ids = self.inner.edge_ids();
        let attr_values: Vec<groggy::AttrValue> = edge_ids.into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
    }
    
    /// Get nodes accessor for fluent API (g.nodes property) 
    #[getter]
    fn nodes(self_: PyRef<Self>, py: Python) -> PyResult<Py<PyNodesAccessor>> {
        let graph_ref = self_.into();
        PyGraph::create_nodes_accessor_internal(graph_ref, py)
    }
    
    /// Get edges accessor for fluent API (g.edges property)
    #[getter]
    fn edges(self_: PyRef<Self>, py: Python) -> PyResult<Py<PyEdgesAccessor>> {
        let graph_ref = self_.into();
        PyGraph::create_edges_accessor_internal(graph_ref, py)
    }
}

// Internal methods for FFI integration (not exposed to Python)
impl PyGraph {
    /// Internal helper methods for accessors/views
    pub fn create_node_view_internal(graph: Py<PyGraph>, py: Python, node_id: NodeId) -> PyResult<Py<PyNodeView>> {
        Py::new(py, PyNodeView {
            graph,
            node_id,
        })
    }
    
    pub fn create_edge_view_internal(graph: Py<PyGraph>, py: Python, edge_id: EdgeId) -> PyResult<Py<PyEdgeView>> {
        Py::new(py, PyEdgeView {
            graph,
            edge_id,
        })
    }

    /// Create a NodesAccessor (internal helper)
    fn create_nodes_accessor_internal(graph_ref: Py<PyGraph>, py: Python) -> PyResult<Py<PyNodesAccessor>> {
        Py::new(py, PyNodesAccessor {
            graph: graph_ref,
            constrained_nodes: None,
        })
    }
    
    /// Create an EdgesAccessor (internal helper)
    fn create_edges_accessor_internal(graph_ref: Py<PyGraph>, py: Python) -> PyResult<Py<PyEdgesAccessor>> {
        Py::new(py, PyEdgesAccessor {
            graph: graph_ref,
            constrained_edges: None,
        })
    }
    
    pub fn node_attribute_keys(&self, node_id: NodeId) -> Vec<String> {
        match self.inner.get_node_attrs(node_id) {
            Ok(attrs) => attrs.keys().cloned().collect(),
            Err(_) => Vec::new(),
        }
    }
    
    pub fn edge_attribute_keys(&self, edge_id: EdgeId) -> Vec<String> {
        match self.inner.get_edge_attrs(edge_id) {
            Ok(attrs) => attrs.keys().cloned().collect(),
            Err(_) => Vec::new(),
        }
    }
}