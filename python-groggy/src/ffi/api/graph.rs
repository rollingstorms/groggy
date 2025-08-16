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
use crate::ffi::core::array::{PyGraphArray, PyGraphMatrix};
use crate::ffi::core::subgraph::PySubgraph;
use crate::ffi::core::query::{PyNodeFilter, PyEdgeFilter};
use crate::ffi::utils::{graph_error_to_py_err, python_value_to_attr_value, attr_value_to_python_value};

// Import version control types
use crate::ffi::api::graph_version::{PyCommit, PyBranchInfo};
use crate::ffi::api::graph_query::PyGraphQuery;

// Placeholder imports for missing types - these need to be implemented
struct PyAttributes {
    graph: *const RustGraph,
}


struct PyAggregationResult {
    value: f64, // Simplified for now
}

struct PyGroupedAggregationResult {
    value: PyObject,
}


struct PyHistoricalView {
    state_id: StateId,
}


// Helper function for matrix conversion
fn adjacency_matrix_to_py_graph_matrix(py: Python, matrix: groggy::AdjacencyMatrix) -> PyResult<Py<PyGraphMatrix>> {
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
    pub inner: RustGraph,
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
    
    pub fn set_node_attribute(&mut self, node: NodeId, attr: AttrName, value: &PyAttrValue) -> PyResult<()> {
        self.inner.set_node_attr(node, attr, value.inner.clone())
            .map_err(graph_error_to_py_err)
    }
    
    pub fn set_edge_attribute(&mut self, edge: EdgeId, attr: AttrName, value: &PyAttrValue) -> PyResult<()> {
        self.inner.set_edge_attr(edge, attr, value.inner.clone())
            .map_err(graph_error_to_py_err)
    }
    
    pub fn get_node_attribute(&self, node: NodeId, attr: AttrName) -> PyResult<Option<PyAttrValue>> {
        match self.inner.get_node_attr(node, &attr) {
            Ok(Some(value)) => Ok(Some(PyAttrValue { inner: value })),
            Ok(None) => Ok(None),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }
    
    pub fn get_edge_attribute(&self, edge: EdgeId, attr: AttrName) -> PyResult<Option<PyAttrValue>> {
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
    
    // === BULK OPERATIONS FOR BENCHMARK COMPATIBILITY ===
    
    /// Set bulk node attributes using the format expected by benchmark
    fn set_node_attributes(&mut self, _py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        use groggy::AttrValue as RustAttrValue;
        use pyo3::exceptions::{PyKeyError, PyValueError};
        
        // HYPER-OPTIMIZED bulk API - minimize PyO3 overhead and allocations
        let mut attrs_values = std::collections::HashMap::with_capacity(attrs_dict.len());
        
        for (attr_name, attr_data) in attrs_dict {
            let attr: AttrName = attr_name.extract()?;
            let data_dict: &PyDict = attr_data.downcast()?;
            
            // OPTIMIZATION: Extract all fields at once to reduce PyO3 calls
            let (nodes, values_obj, value_type): (Vec<NodeId>, &pyo3::PyAny, String) = {
                let nodes_item = data_dict.get_item("nodes")?.ok_or_else(|| 
                    PyErr::new::<PyKeyError, _>("Missing 'nodes' key"))?;
                let values_item = data_dict.get_item("values")?.ok_or_else(|| 
                    PyErr::new::<PyKeyError, _>("Missing 'values' key"))?;
                let type_item = data_dict.get_item("value_type")?.ok_or_else(|| 
                    PyErr::new::<PyKeyError, _>("Missing 'value_type' key"))?;
                    
                (nodes_item.extract()?, values_item, type_item.extract()?)
            };
            
            let len = nodes.len();
            
            // OPTIMIZATION: Pre-allocate result vector and use direct indexing
            let mut pairs = Vec::with_capacity(len);
            
            // OPTIMIZATION: Match on str slice to avoid repeated string comparisons
            match value_type.as_str() {
                "text" => {
                    let values: Vec<String> = values_obj.extract()?;
                    if values.len() != len {
                        return Err(PyErr::new::<PyValueError, _>("Length mismatch"));
                    }
                    
                    // OPTIMIZATION: Direct loop instead of iterator chain
                    for i in 0..len {
                        pairs.push((nodes[i], RustAttrValue::Text(values[i].clone())));
                    }
                },
                "int" => {
                    let values: Vec<i64> = values_obj.extract()?;
                    if values.len() != len {
                        return Err(PyErr::new::<PyValueError, _>("Length mismatch"));
                    }
                    
                    for i in 0..len {
                        pairs.push((nodes[i], RustAttrValue::Int(values[i])));
                    }
                },
                "float" => {
                    let values: Vec<f64> = values_obj.extract()?;
                    if values.len() != len {
                        return Err(PyErr::new::<PyValueError, _>("Length mismatch"));
                    }
                    
                    for i in 0..len {
                        pairs.push((nodes[i], RustAttrValue::Float(values[i] as f32)));
                    }
                },
                "bool" => {
                    let values: Vec<bool> = values_obj.extract()?;
                    if values.len() != len {
                        return Err(PyErr::new::<PyValueError, _>("Length mismatch"));
                    }
                    
                    for i in 0..len {
                        pairs.push((nodes[i], RustAttrValue::Bool(values[i])));
                    }
                },
                _ => return Err(PyErr::new::<PyValueError, _>("Unsupported type"))
            };
            
            attrs_values.insert(attr, pairs);
        }
        
        self.inner.set_node_attrs(attrs_values)
            .map_err(graph_error_to_py_err)
    }
    
    /// Set bulk edge attributes using the format expected by benchmark
    fn set_edge_attributes(&mut self, _py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        use groggy::AttrValue as RustAttrValue;
        use pyo3::exceptions::{PyKeyError, PyValueError};
        
        // New efficient columnar API for edges - zero PyAttrValue objects created!
        let mut attrs_values = std::collections::HashMap::new();
        
        for (attr_name, attr_data) in attrs_dict {
            let attr: AttrName = attr_name.extract()?;
            let data_dict: &PyDict = attr_data.downcast()?;
            
            // Extract components in bulk using the same pattern as node attributes
            let edges: Vec<EdgeId> = if let Ok(Some(item)) = data_dict.get_item("edges") {
                item.extract()?
            } else {
                return Err(PyErr::new::<PyKeyError, _>("Missing 'edges' key in attribute data"));
            };
            let value_type: String = if let Ok(Some(item)) = data_dict.get_item("value_type") {
                item.extract()?
            } else {
                return Err(PyErr::new::<PyKeyError, _>("Missing 'value_type' key in attribute data"));
            };
            
            // Batch convert based on known type - no individual type detection!
            let pairs = match value_type.as_str() {
                "text" => {
                    let values: Vec<String> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>("Missing 'values' key in attribute data"));
                    };
                    
                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(
                            format!("Mismatched lengths: {} edges vs {} values", edges.len(), values.len())
                        ));
                    }
                    
                    edges.into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Text(val)))
                        .collect()
                },
                "int" => {
                    let values: Vec<i64> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>("Missing 'values' key in attribute data"));
                    };
                    
                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(
                            format!("Mismatched lengths: {} edges vs {} values", edges.len(), values.len())
                        ));
                    }
                    
                    edges.into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Int(val)))
                        .collect()
                },
                "float" => {
                    let values: Vec<f64> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>("Missing 'values' key in attribute data"));
                    };
                    
                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(
                            format!("Mismatched lengths: {} edges vs {} values", edges.len(), values.len())
                        ));
                    }
                    
                    edges.into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Float(val as f32)))
                        .collect()
                },
                "bool" => {
                    let values: Vec<bool> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>("Missing 'values' key in attribute data"));
                    };
                    
                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(
                            format!("Mismatched lengths: {} edges vs {} values", edges.len(), values.len())
                        ));
                    }
                    
                    edges.into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Bool(val)))
                        .collect()
                },
                _ => {
                    return Err(PyErr::new::<PyValueError, _>(
                        format!("Unsupported value_type: {}", value_type)
                    ));
                }
            };
            
            attrs_values.insert(attr, pairs);
        }
        
        self.inner.set_edge_attrs(attrs_values)
            .map_err(graph_error_to_py_err)
    }
    
    /// Add multiple edges at once
    fn add_edges(&mut self, edges: &PyAny, node_mapping: Option<std::collections::HashMap<String, NodeId>>, _uid_key: Option<String>) -> PyResult<Vec<EdgeId>> {
        // Format 1: List of (source, target) tuples - most common case for benchmarks
        if let Ok(edge_pairs) = edges.extract::<Vec<(NodeId, NodeId)>>() {
            return Ok(self.inner.add_edges(&edge_pairs));
        }
        
        // Format 2: List of (source, target, attrs_dict) tuples  
        else if let Ok(edge_tuples) = edges.extract::<Vec<(&PyAny, &PyAny, Option<&PyDict>)>>() {
            let mut edge_ids = Vec::new();
            
            for (src_any, tgt_any, attrs_opt) in edge_tuples {
                let source: NodeId = src_any.extract()?;
                let target: NodeId = tgt_any.extract()?;
                
                let edge_id = self.inner.add_edge(source, target)
                    .map_err(graph_error_to_py_err)?;
                edge_ids.push(edge_id);
                
                // Set attributes if provided
                if let Some(attrs) = attrs_opt {
                    for (key, value) in attrs.iter() {
                        let attr_name: String = key.extract()?;
                        let attr_value = python_value_to_attr_value(value)?;
                        
                        self.inner.set_edge_attr(edge_id, attr_name, attr_value)
                            .map_err(graph_error_to_py_err)?;
                    }
                }
            }
            
            return Ok(edge_ids);
        }
        
        // Format 3: List of dictionaries with node mapping
        else if let Ok(edge_dicts) = edges.extract::<Vec<&PyDict>>() {
            let mut edge_ids = Vec::new();
            
            for edge_dict in edge_dicts {
                // Extract source and target
                let source = if let Some(mapping) = &node_mapping {
                    let source_str: String = edge_dict.get_item("source")?.unwrap().extract()?;
                    *mapping.get(&source_str).ok_or_else(|| 
                        pyo3::exceptions::PyKeyError::new_err(format!("Node {} not found in mapping", source_str)))?
                } else {
                    edge_dict.get_item("source")?.unwrap().extract()?
                };
                
                let target = if let Some(mapping) = &node_mapping {
                    let target_str: String = edge_dict.get_item("target")?.unwrap().extract()?;
                    *mapping.get(&target_str).ok_or_else(|| 
                        pyo3::exceptions::PyKeyError::new_err(format!("Node {} not found in mapping", target_str)))?
                } else {
                    edge_dict.get_item("target")?.unwrap().extract()?
                };
                
                // Add the edge
                let edge_id = self.inner.add_edge(source, target)
                    .map_err(graph_error_to_py_err)?;
                edge_ids.push(edge_id);
                
                // Set edge attributes if any
                for (key, value) in edge_dict.iter() {
                    let key_str: String = key.extract()?;
                    if key_str != "source" && key_str != "target" {
                        let attr_value = python_value_to_attr_value(value)?;
                        self.inner.set_edge_attr(edge_id, key_str, attr_value)
                            .map_err(graph_error_to_py_err)?;
                    }
                }
            }
            
            return Ok(edge_ids);
        }
        
        // If none of the formats matched, return error
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "add_edges expects a list of (source, target) tuples, (source, target, attrs) tuples, or dictionaries with node_mapping"
        ))
    }
    
    /// Filter nodes using NodeFilter object or string query
    fn filter_nodes(&mut self, py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Fast path optimization: Check for NodeFilter object first (most common case)
        let node_filter = if let Ok(filter_obj) = filter.extract::<PyNodeFilter>() {
            // Direct NodeFilter object - fastest path
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using our query parser
            let query_parser = py.import("groggy.query_parser")?;
            let parse_func = query_parser.getattr("parse_node_query")?;
            let parsed_filter: PyNodeFilter = parse_func.call1((query_str,))?.extract()?;
            parsed_filter.inner.clone()
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be a NodeFilter object or a string query (e.g., 'salary > 120000')"
            ));
        };
        
        let filtered_nodes = self.inner.find_nodes(node_filter)
            .map_err(graph_error_to_py_err)?;
        
        // O(k) Calculate induced edges using optimized core subgraph method
        use std::collections::HashSet;
        let node_set: HashSet<NodeId> = filtered_nodes.iter().copied().collect();
        
        // Get columnar topology vectors (edge_ids, sources, targets) - O(1) if cached
        let (edge_ids, sources, targets) = self.inner.get_columnar_topology();
        let mut induced_edges = Vec::new();
        
        // Iterate through parallel vectors - O(k) where k = active edges
        for i in 0..edge_ids.len() {
            let edge_id = edge_ids[i];
            let source = sources[i];
            let target = targets[i];
            
            // O(1) HashSet lookups instead of O(n) Vec::contains
            if node_set.contains(&source) && node_set.contains(&target) {
                induced_edges.push(edge_id);
            }
        }
        
        Ok(PySubgraph::new(
            filtered_nodes,
            induced_edges,
            "filtered_nodes".to_string(),
            None, // TODO: Fix graph reference later
        ))
    }
    
    /// Filter edges using EdgeFilter object or string query
    fn filter_edges(&mut self, py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Fast path optimization: Check for EdgeFilter object first (most common case)
        let edge_filter = if let Ok(filter_obj) = filter.extract::<PyEdgeFilter>() {
            // Direct EdgeFilter object - fastest path
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using our query parser
            let query_parser = py.import("groggy.query_parser")?;
            let parse_func = query_parser.getattr("parse_edge_query")?;
            let parsed_filter: PyEdgeFilter = parse_func.call1((query_str,))?.extract()?;
            parsed_filter.inner.clone()
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be an EdgeFilter object or a string query"
            ));
        };
        
        let filtered_edges = self.inner.find_edges(edge_filter)
            .map_err(graph_error_to_py_err)?;
        
        // Calculate nodes that are connected by the filtered edges
        use std::collections::HashSet;
        let mut nodes = HashSet::new();
        for &edge_id in &filtered_edges {
            if let Ok((source, target)) = self.inner.edge_endpoints(edge_id) {
                nodes.insert(source);
                nodes.insert(target);
            }
        }
        
        let node_vec: Vec<NodeId> = nodes.into_iter().collect();
        
        Ok(PySubgraph::new(
            node_vec,
            filtered_edges,
            "filtered_edges".to_string(),
            None, // TODO: Fix graph reference later
        ))
    }
    
    /// Breadth-first search traversal
    #[pyo3(signature = (start_node, max_depth = None, node_filter = None, edge_filter = None, inplace = None, attr_name = None))]
    fn bfs(&mut self, py: Python, start_node: NodeId, max_depth: Option<usize>, 
           node_filter: Option<&PyAny>, edge_filter: Option<&PyAny>,
           inplace: Option<bool>, attr_name: Option<String>) -> PyResult<PySubgraph> {
        
        // Check if start node exists
        if !self.inner.contains_node(start_node) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Start node {} does not exist in the graph", start_node)
            ));
        }
        
        // BFS implementation
        use std::collections::{VecDeque, HashSet};
        
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result_nodes = Vec::new();
        let mut result_edges = Vec::new();
        
        queue.push_back((start_node, 0)); // (node_id, depth)
        visited.insert(start_node);
        
        while let Some((current_node, depth)) = queue.pop_front() {
            result_nodes.push(current_node);
            
            // Check max depth
            if let Some(max_d) = max_depth {
                if depth >= max_d {
                    continue;
                }
            }
            
            // Get neighbors by examining all edges
            let all_edges = self.inner.edge_ids();
            for edge_id in all_edges {
                if let Ok((source, target)) = self.inner.edge_endpoints(edge_id) {
                    let neighbor = if source == current_node {
                        target
                    } else if target == current_node {
                        source
                    } else {
                        continue;
                    };
                    
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back((neighbor, depth + 1));
                        result_edges.push(edge_id);
                    }
                }
            }
        }
        
        // Handle inplace attribute setting
        if let Some(true) = inplace {
            let attr_name = attr_name.unwrap_or_else(|| "bfs_distance".to_string());
            
            for (order, &node_id) in result_nodes.iter().enumerate() {
                let order_value = groggy::AttrValue::Int(order as i64);
                let _ = self.inner.set_node_attr(node_id, attr_name.clone(), order_value);
            }
        }
        
        // Ignore filter parameters for now - they would need proper implementation
        let _ = node_filter;
        let _ = edge_filter;
        
        Ok(PySubgraph::new(
            result_nodes,
            result_edges,
            "bfs_traversal".to_string(),
            Some(Py::new(py, PyGraph { inner: groggy::Graph::new() })?)
        ))
    }
    
    /// Depth-first search traversal
    #[pyo3(signature = (start_node, max_depth = None, node_filter = None, edge_filter = None, inplace = None, node_attr = None, edge_attr = None))]
    fn dfs(&mut self, py: Python, start_node: NodeId, max_depth: Option<usize>,
           node_filter: Option<&PyAny>, edge_filter: Option<&PyAny>,
           inplace: Option<bool>, node_attr: Option<String>, edge_attr: Option<String>) -> PyResult<PySubgraph> {
        
        // Check if start node exists
        if !self.inner.contains_node(start_node) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Start node {} does not exist in the graph", start_node)
            ));
        }
        
        // DFS implementation
        use std::collections::HashSet;
        
        let mut visited = HashSet::new();
        let mut result_nodes = Vec::new();
        let mut result_edges = Vec::new();
        
        // Recursive DFS helper function
        fn dfs_recursive(
            graph: &groggy::Graph,
            current_node: NodeId,
            visited: &mut HashSet<NodeId>,
            result_nodes: &mut Vec<NodeId>,
            result_edges: &mut Vec<EdgeId>,
            depth: usize,
            max_depth: Option<usize>
        ) {
            visited.insert(current_node);
            result_nodes.push(current_node);
            
            // Check max depth
            if let Some(max_d) = max_depth {
                if depth >= max_d {
                    return;
                }
            }
            
            // Get neighbors
            let all_edges = graph.edge_ids();
            for edge_id in all_edges {
                if let Ok((source, target)) = graph.edge_endpoints(edge_id) {
                    let neighbor = if source == current_node {
                        target
                    } else if target == current_node {
                        source
                    } else {
                        continue;
                    };
                    
                    if !visited.contains(&neighbor) {
                        result_edges.push(edge_id);
                        dfs_recursive(graph, neighbor, visited, result_nodes, result_edges, depth + 1, max_depth);
                    }
                }
            }
        }
        
        dfs_recursive(&self.inner, start_node, &mut visited, &mut result_nodes, &mut result_edges, 0, max_depth);
        
        // Handle inplace attribute setting
        if let Some(true) = inplace {
            // Set node attributes
            if let Some(node_attr_name) = node_attr {
                for (order, &node_id) in result_nodes.iter().enumerate() {
                    let order_value = groggy::AttrValue::Int(order as i64);
                    let _ = self.inner.set_node_attr(node_id, node_attr_name.clone(), order_value);
                }
            }
            
            // Set edge attributes
            if let Some(edge_attr_name) = edge_attr {
                for &edge_id in &result_edges {
                    let tree_edge_value = groggy::AttrValue::Bool(true);
                    let _ = self.inner.set_edge_attr(edge_id, edge_attr_name.clone(), tree_edge_value);
                }
            }
        }
        
        // Ignore filter parameters for now - they would need proper implementation
        let _ = node_filter;
        let _ = edge_filter;
        
        Ok(PySubgraph::new(
            result_nodes,
            result_edges,
            "dfs_traversal".to_string(),
            Some(Py::new(py, PyGraph { inner: groggy::Graph::new() })?)
        ))
    }
    
    /// Find connected components
    #[pyo3(signature = (inplace = None, attr_name = None))]
    fn connected_components(&self, py: Python, inplace: Option<bool>, attr_name: Option<String>) -> PyResult<Vec<PySubgraph>> {
        // Connected components using DFS
        use std::collections::HashSet;
        
        let all_nodes = self.inner.node_ids();
        let mut visited = HashSet::new();
        let mut components = Vec::new();
        
        for start_node in all_nodes {
            if visited.contains(&start_node) {
                continue;
            }
            
            // Find all nodes in this component using DFS
            let mut component_nodes = Vec::new();
            let mut stack = vec![start_node];
            
            while let Some(current_node) = stack.pop() {
                if visited.contains(&current_node) {
                    continue;
                }
                
                visited.insert(current_node);
                component_nodes.push(current_node);
                
                // Find neighbors
                let all_edges = self.inner.edge_ids();
                for edge_id in all_edges {
                    if let Ok((source, target)) = self.inner.edge_endpoints(edge_id) {
                        let neighbor = if source == current_node {
                            target
                        } else if target == current_node {
                            source
                        } else {
                            continue;
                        };
                        
                        if !visited.contains(&neighbor) {
                            stack.push(neighbor);
                        }
                    }
                }
            }
            
            // Calculate induced edges for this component
            let component_node_set: HashSet<NodeId> = component_nodes.iter().copied().collect();
            let mut component_edges = Vec::new();
            
            let all_edges = self.inner.edge_ids();
            for edge_id in all_edges {
                if let Ok((source, target)) = self.inner.edge_endpoints(edge_id) {
                    if component_node_set.contains(&source) && component_node_set.contains(&target) {
                        component_edges.push(edge_id);
                    }
                }
            }
            
            if !component_nodes.is_empty() {
                let component = PySubgraph::new(
                    component_nodes,
                    component_edges,
                    format!("connected_component_{}", components.len()),
                    Some(Py::new(py, PyGraph { inner: groggy::Graph::new() })?)
                );
                components.push(component);
            }
        }
        
        // Handle inplace attribute setting
        if let Some(true) = inplace {
            let attr_name = attr_name.unwrap_or_else(|| "component_id".to_string());
            
            for (component_id, component) in components.iter().enumerate() {
                // Set component_id attribute on all nodes in this component
                // Note: This is a simplified implementation - in practice we'd access the nodes directly
                let component_value = groggy::AttrValue::Int(component_id as i64);
                // Would need to iterate through component nodes and set attributes
                // For now, just acknowledge the parameter
                let _ = component_value;
            }
        }
        
        Ok(components)
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
    
    // === HELPER METHODS FOR OTHER MODULES ===
    
    pub fn has_node_attribute(&self, node_id: NodeId, attr_name: &str) -> bool {
        match self.inner.get_node_attr(node_id, &attr_name.to_string()) {
            Ok(Some(_)) => true,
            _ => false,
        }
    }
    
    pub fn has_edge_attribute(&self, edge_id: EdgeId, attr_name: &str) -> bool {
        match self.inner.get_edge_attr(edge_id, &attr_name.to_string()) {
            Ok(Some(_)) => true,
            _ => false,
        }
    }
    
    
    pub fn get_edge_endpoints(&self, edge_id: EdgeId) -> Result<(NodeId, NodeId), String> {
        self.inner.edge_endpoints(edge_id).map_err(|e| e.to_string())
    }
    
    
    pub fn get_node_ids(&self) -> PyResult<Vec<NodeId>> {
        Ok(self.inner.node_ids())
    }
    
    pub fn get_edge_ids(&self) -> PyResult<Vec<EdgeId>> {
        Ok(self.inner.edge_ids())
    }
    
    // Additional public methods for internal module access
    pub fn has_node_internal(&self, node_id: NodeId) -> bool {
        self.inner.contains_node(node_id)
    }
    
    pub fn has_edge_internal(&self, edge_id: EdgeId) -> bool {
        self.inner.contains_edge(edge_id)
    }
    
    pub fn get_node_count(&self) -> usize {
        self.inner.node_ids().len()
    }
    
    pub fn get_edge_count(&self) -> usize {
        self.inner.edge_ids().len()
    }
    
    pub fn node_ids_vec(&self) -> Vec<NodeId> {
        self.inner.node_ids()
    }
    
    pub fn edge_ids_vec(&self) -> Vec<EdgeId> {
        self.inner.edge_ids()
    }
    
    pub fn get_node_ids_array(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let node_ids = self.inner.node_ids();
        let attr_values: Vec<groggy::AttrValue> = node_ids.into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::core::array::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
    }
    
    pub fn get_edge_ids_array(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let edge_ids = self.inner.edge_ids();
        let attr_values: Vec<groggy::AttrValue> = edge_ids.into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::core::array::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
    }
    
}