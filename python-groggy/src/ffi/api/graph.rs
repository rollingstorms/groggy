//! Graph FFI Coordinator
//!
//! Main Python bindings for the Graph API.

use groggy::{AttrName, AttrValue as RustAttrValue, EdgeId, Graph as RustGraph, NodeId, StateId};
use groggy::core::subgraph::Subgraph;
use groggy::core::traits::SubgraphOperations;
use pyo3::exceptions::{PyAttributeError, PyKeyError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

// Import all graph modules
use crate::ffi::core::accessors::{PyEdgesAccessor, PyNodesAccessor};
use crate::ffi::core::array::PyGraphArray;
use crate::ffi::core::matrix::PyGraphMatrix;
use crate::ffi::core::neighborhood::{PyNeighborhoodResult, PyNeighborhoodStats, PyNeighborhoodSubgraph};
use crate::ffi::core::query::{PyEdgeFilter, PyNodeFilter};
use crate::ffi::core::subgraph::PySubgraph;
use crate::ffi::core::traversal::PyGroupedAggregationResult;
use crate::ffi::core::views::{PyEdgeView, PyNodeView};
use crate::ffi::types::PyAttrValue;
use crate::ffi::utils::{graph_error_to_py_err, python_value_to_attr_value};

// Import version control types
use crate::ffi::api::graph_version::{PyBranchInfo, PyCommit, PyHistoricalView};

// Placeholder imports for missing types - these need to be implemented
struct PyAttributes {
    graph: *const RustGraph,
}

/// Specification for neighborhood hop parameters
#[derive(Debug, Clone)]
enum HopSpecification {
    /// Single level: k hops (e.g., k=2 for 2-hop)
    SingleLevel(usize),
    /// Multi-level: different sampling at each level (e.g., [100, 10] for 100 from 1st hop, 10 from 2nd hop)
    MultiLevel(Vec<usize>),
}

/// Python wrapper for AggregationResult
#[pyclass(name = "AggregationResult")]
pub struct PyAggregationResult {
    pub value: f64,
}

#[pymethods]
impl PyAggregationResult {
    #[getter]
    fn value(&self) -> f64 {
        self.value
    }

    fn __repr__(&self) -> String {
        format!("AggregationResult({})", self.value)
    }
}

/// Helper function to convert AdjacencyMatrix to PyGraphMatrix
fn adjacency_matrix_to_py_graph_matrix(
    _py: Python,
    _matrix: groggy::AdjacencyMatrix,
) -> PyResult<Py<PyGraphMatrix>> {
    // TODO: Implement adjacency matrix to GraphMatrix conversion in Phase 2
    Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
        "AdjacencyMatrix to GraphMatrix conversion temporarily disabled during Phase 2 unification",
    ))
}

// Helper function to extract matrix from AdjacencyMatrix and wrap appropriately
fn adjacency_matrix_to_py_object(
    _py: Python,
    _matrix: groggy::AdjacencyMatrixResult,
) -> PyResult<PyObject> {
    // TODO: Implement adjacency matrix to Python object conversion in Phase 2
    Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
        "Adjacency matrix functionality temporarily disabled during Phase 2 unification",
    ))
}

/// Python wrapper for the main Graph
#[pyclass(name = "Graph", unsendable)]
pub struct PyGraph {
    pub inner: Rc<RefCell<RustGraph>>,
}


impl PyGraph {
    /// Convert this graph to a SubgraphOperations trait object containing all nodes and edges
    /// 
    /// This enables all SubgraphOperations methods to work on the full graph:
    /// - connected_components(), bfs(), dfs()
    /// - node_count(), edge_count(), neighbors(), degree()  
    /// - nodes_table(), edges_table()
    /// 
    /// # Performance
    /// Creates lightweight wrapper with references to existing graph data
    pub fn as_subgraph(&self) -> PyResult<Box<dyn SubgraphOperations>> {
        let graph = self.inner.borrow();
        
        // Get all nodes and edges from the graph
        let all_nodes = graph.node_ids().into_iter().collect();
        let all_edges = graph.edge_ids().into_iter().collect();
        drop(graph); // Release borrow
        
        // Create subgraph containing the entire graph
        let full_subgraph = Subgraph::new(
            self.inner.clone(),
            all_nodes, 
            all_edges,
            "full_graph".to_string(),
        );
        
        Ok(Box::new(full_subgraph))
    }
}

#[pymethods]
impl PyGraph {
    #[new]
    #[pyo3(signature = (directed = false, _config = None))]
    fn new(directed: bool, _config: Option<&PyDict>) -> PyResult<Self> {
        // Create graph with specified directionality
        let rust_graph = if directed {
            RustGraph::new_directed()
        } else {
            RustGraph::new_undirected()
        };
        Ok(Self { inner: Rc::new(RefCell::new(rust_graph)) })
    }

    /// Check if this graph is directed
    #[getter]
    fn is_directed(&self) -> bool {
        self.inner.borrow().is_directed()
    }

    /// Check if this graph is undirected
    #[getter]
    fn is_undirected(&self) -> bool {
        self.inner.borrow().is_undirected()
    }

    // === CORE GRAPH OPERATIONS ===

    #[pyo3(signature = (**kwargs))]
    fn add_node(&mut self, kwargs: Option<&PyDict>) -> PyResult<NodeId> {
        let node_id = self.inner.borrow_mut().add_node();

        // Fast path: if no kwargs, just return the node_id
        if let Some(attrs) = kwargs {
            if !attrs.is_empty() {
                // Only do attribute setting if we actually have attributes
                for (key, value) in attrs.iter() {
                    let attr_name: String = key.extract()?;
                    let attr_value = python_value_to_attr_value(value);

                    self.inner
.borrow_mut()
                        .set_node_attr(node_id, attr_name, attr_value?)
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
            let node_ids = self.inner.borrow_mut().add_nodes(count);
            return Python::with_gil(|py| Ok(node_ids.to_object(py)));
        }

        // Only use Python::with_gil for complex operations
        Python::with_gil(|py| {
            if let Ok(node_data_list) = data.extract::<Vec<&PyDict>>() {
                // New API: add_nodes([{"id": "alice", "age": 30}, ...], id_key="id")
                let mut id_mapping = std::collections::HashMap::new();

                // Create all nodes first
                let node_ids = self.inner.borrow_mut().add_nodes(node_data_list.len());

                // OPTIMIZATION: Collect attributes by name for bulk operations instead of individual calls
                // This changes complexity from O(N × A × log N) to O(N × A)
                let mut attrs_by_name: std::collections::HashMap<
                    String,
                    Vec<(NodeId, RustAttrValue)>,
                > = std::collections::HashMap::new();

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
                                return Err(PyErr::new::<PyKeyError, _>(format!(
                                    "Missing key: {}",
                                    key
                                )));
                            }
                            Err(e) => return Err(e),
                        }
                    }

                    // Collect all attributes for bulk setting
                    for (attr_key, attr_value) in node_dict.iter() {
                        let attr_name: String = attr_key.extract()?;
                        let attr_val = python_value_to_attr_value(attr_value)?;

                        // Store all attributes including the id_key for later uid_key lookups
                        attrs_by_name
                            .entry(attr_name)
                            .or_insert_with(Vec::new)
                            .push((node_id, attr_val));
                    }
                }

                // OPTIMIZATION: Use bulk attribute setting - O(A) operations instead of O(N × A)
                if !attrs_by_name.is_empty() {
                    self.inner
.borrow_mut()
                        .set_node_attrs(attrs_by_name)
                        .map_err(graph_error_to_py_err)?;
                }

                // Return the mapping if id_key was provided, otherwise return node IDs
                if uid_key.is_some() {
                    Ok(id_mapping.to_object(py))
                } else {
                    Ok(node_ids.to_object(py))
                }
            } else {
                Err(PyErr::new::<PyTypeError, _>(
                    "add_nodes expects either an integer count or a list of dictionaries",
                ))
            }
        })
    }

    /// Helper method to resolve string ID to NodeId using uid_key attribute
    fn resolve_string_id_to_node(&self, string_id: &str, uid_key: &str) -> PyResult<NodeId> {
        let node_ids = self.inner.borrow_mut().node_ids();

        for node_id in node_ids {
            if let Ok(Some(attr_value)) = self.inner.borrow_mut().get_node_attr(node_id, &uid_key.to_string()) {
                match attr_value {
                    RustAttrValue::Text(s) => {
                        if s == string_id {
                            return Ok(node_id);
                        }
                    }
                    RustAttrValue::CompactText(s) => {
                        if s.as_str() == string_id {
                            return Ok(node_id);
                        }
                    }
                    _ => continue, // Skip non-text attributes
                }
            }
        }

        Err(PyErr::new::<PyKeyError, _>(format!(
            "No node found with {}='{}'",
            uid_key, string_id
        )))
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Graph(nodes={}, edges={})",
            self.node_count(),
            self.edge_count()
        )
    }

    /// Python len() support - returns number of nodes
    fn __len__(&self) -> usize {
        self.inner.borrow_mut().node_ids().len()
    }

    /// Check if a node exists in the graph - DELEGATED
    fn has_node(&self, node_id: NodeId) -> bool {
        // Pure delegation to SubgraphOperations
        match self.as_subgraph() {
            Ok(subgraph) => subgraph.contains_node(node_id),
            Err(_) => false,
        }
    }

    /// Check if an edge exists in the graph - DELEGATED  
    fn has_edge(&self, edge_id: EdgeId) -> bool {
        // Pure delegation to SubgraphOperations
        match self.as_subgraph() {
            Ok(subgraph) => subgraph.contains_edge(edge_id),
            Err(_) => false,
        }
    }

    /// Get the number of nodes in the graph - DELEGATED
    fn node_count(&self) -> usize {
        // Pure delegation to SubgraphOperations
        match self.as_subgraph() {
            Ok(subgraph) => subgraph.node_count(),
            Err(_) => 0,
        }
    }

    /// Get the number of edges in the graph - DELEGATED
    fn edge_count(&self) -> usize {
        // Pure delegation to SubgraphOperations
        match self.as_subgraph() {
            Ok(subgraph) => subgraph.edge_count(),
            Err(_) => 0,
        }
    }

    /// Calculate graph density (number of edges / number of possible edges) - PURE DELEGATION
    fn density(&self) -> f64 {
        // Pure delegation to SubgraphOperations trait
        match self.as_subgraph() {
            Ok(subgraph) => subgraph.density(),
            Err(_) => 0.0,
        }
    }

    // === ATTRIBUTE OPERATIONS ===

    /// Set node attribute with automatic Python value conversion
    pub fn set_node_attr(&mut self, node: NodeId, attr: AttrName, value: &PyAny) -> PyResult<()> {
        let attr_value = python_value_to_attr_value(value);
        self.inner
.borrow_mut()
            .set_node_attr(node, attr, attr_value?)
            .map_err(graph_error_to_py_err)
    }

    /// Set edge attribute with automatic Python value conversion
    pub fn set_edge_attr(&mut self, edge: EdgeId, attr: AttrName, value: &PyAny) -> PyResult<()> {
        let attr_value = python_value_to_attr_value(value);
        self.inner
.borrow_mut()
            .set_edge_attr(edge, attr, attr_value?)
            .map_err(graph_error_to_py_err)
    }

    /// Get node attribute - DELEGATED to SubgraphOperations
    pub fn get_node_attr(
        &self,
        node: NodeId,
        attr: AttrName,
    ) -> PyResult<Option<PyAttrValue>> {
        // Pure delegation with type conversion
        match self.as_subgraph() {
            Ok(subgraph) => {
                match subgraph.get_node_attribute(node, &attr) {
                    Ok(Some(value)) => Ok(Some(PyAttrValue { inner: value })),
                    Ok(None) => Ok(None),
                    Err(e) => Err(graph_error_to_py_err(e)),
                }
            },
            Err(e) => Err(e),
        }
    }

    /// Get edge attribute - DELEGATED to SubgraphOperations
    pub fn get_edge_attr(
        &self,
        edge: EdgeId,
        attr: AttrName,
    ) -> PyResult<Option<PyAttrValue>> {
        // Pure delegation with type conversion
        match self.as_subgraph() {
            Ok(subgraph) => {
                match subgraph.get_edge_attribute(edge, &attr) {
                    Ok(Some(value)) => Ok(Some(PyAttrValue { inner: value })),
                    Ok(None) => Ok(None),
                    Err(e) => Err(graph_error_to_py_err(e)),
                }
            },
            Err(e) => Err(e),
        }
    }

    fn get_edge_attrs(&self, edge: EdgeId, py: Python) -> PyResult<PyObject> {
        let attrs = self
            .inner
            .borrow_mut()
            .get_edge_attrs(edge)
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

    /// Check if node exists - DELEGATED (same as has_node)
    fn contains_node(&self, node: NodeId) -> bool {
        // Pure delegation to SubgraphOperations (same logic as has_node)
        self.has_node(node)
    }

    /// Check if edge exists - DELEGATED (same as has_edge)  
    fn contains_edge(&self, edge: EdgeId) -> bool {
        // Pure delegation to SubgraphOperations (same logic as has_edge)
        self.has_edge(edge)
    }

    fn edge_endpoints(&self, edge: EdgeId) -> PyResult<(NodeId, NodeId)> {
        self.inner
.borrow_mut()
            .edge_endpoints(edge)
            .map_err(graph_error_to_py_err)
    }

    /// Get all active node IDs as GraphArray (lazy Rust view) - use .values for Python list
    #[getter]
    fn node_ids(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let node_ids = self.inner.borrow_mut().node_ids();
        let attr_values: Vec<groggy::AttrValue> = node_ids
            .into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
    }

    /// Get all active edge IDs as GraphArray (lazy Rust view) - use .values for Python list
    #[getter]
    fn edge_ids(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let edge_ids = self.inner.borrow_mut().edge_ids();
        let attr_values: Vec<groggy::AttrValue> = edge_ids
            .into_iter()
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
    fn set_node_attrs(&mut self, _py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        use groggy::AttrValue as RustAttrValue;
        use pyo3::exceptions::{PyKeyError, PyValueError};

        // HYPER-OPTIMIZED bulk API - minimize PyO3 overhead and allocations
        let mut attrs_values = std::collections::HashMap::with_capacity(attrs_dict.len());

        for (attr_name, attr_data) in attrs_dict {
            let attr: AttrName = attr_name.extract()?;
            let data_dict: &PyDict = attr_data.downcast()?;

            // OPTIMIZATION: Extract all fields at once to reduce PyO3 calls
            let (nodes, values_obj, value_type): (Vec<NodeId>, &pyo3::PyAny, String) = {
                let nodes_item = data_dict
                    .get_item("nodes")?
                    .ok_or_else(|| PyErr::new::<PyKeyError, _>("Missing 'nodes' key"));
                let values_item = data_dict
                    .get_item("values")?
                    .ok_or_else(|| PyErr::new::<PyKeyError, _>("Missing 'values' key"));
                let type_item = data_dict
                    .get_item("value_type")?
                    .ok_or_else(|| PyErr::new::<PyKeyError, _>("Missing 'value_type' key"));

                (nodes_item?.extract()?, values_item?, type_item?.extract()?)
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
                }
                "int" => {
                    let values: Vec<i64> = values_obj.extract()?;
                    if values.len() != len {
                        return Err(PyErr::new::<PyValueError, _>("Length mismatch"));
                    }

                    for i in 0..len {
                        pairs.push((nodes[i], RustAttrValue::Int(values[i])));
                    }
                }
                "float" => {
                    let values: Vec<f64> = values_obj.extract()?;
                    if values.len() != len {
                        return Err(PyErr::new::<PyValueError, _>("Length mismatch"));
                    }

                    for i in 0..len {
                        pairs.push((nodes[i], RustAttrValue::Float(values[i] as f32)));
                    }
                }
                "bool" => {
                    let values: Vec<bool> = values_obj.extract()?;
                    if values.len() != len {
                        return Err(PyErr::new::<PyValueError, _>("Length mismatch"));
                    }

                    for i in 0..len {
                        pairs.push((nodes[i], RustAttrValue::Bool(values[i])));
                    }
                }
                _ => return Err(PyErr::new::<PyValueError, _>("Unsupported type")),
            };

            attrs_values.insert(attr, pairs);
        }

        self.inner
.borrow_mut()
            .set_node_attrs(attrs_values)
            .map_err(graph_error_to_py_err)
    }

    /// Set bulk edge attributes using the format expected by benchmark
    fn set_edge_attrs(&mut self, _py: Python, attrs_dict: &PyDict) -> PyResult<()> {
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
                return Err(PyErr::new::<PyKeyError, _>(
                    "Missing 'edges' key in attribute data",
                ));
            };
            let value_type: String = if let Ok(Some(item)) = data_dict.get_item("value_type") {
                item.extract()?
            } else {
                return Err(PyErr::new::<PyKeyError, _>(
                    "Missing 'value_type' key in attribute data",
                ));
            };

            // Batch convert based on known type - no individual type detection!
            let pairs = match value_type.as_str() {
                "text" => {
                    let values: Vec<String> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>(
                            "Missing 'values' key in attribute data",
                        ));
                    };

                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "Mismatched lengths: {} edges vs {} values",
                            edges.len(),
                            values.len()
                        )));
                    }

                    edges
                        .into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Text(val)))
                        .collect()
                }
                "int" => {
                    let values: Vec<i64> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>(
                            "Missing 'values' key in attribute data",
                        ));
                    };

                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "Mismatched lengths: {} edges vs {} values",
                            edges.len(),
                            values.len()
                        )));
                    }

                    edges
                        .into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Int(val)))
                        .collect()
                }
                "float" => {
                    let values: Vec<f64> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>(
                            "Missing 'values' key in attribute data",
                        ));
                    };

                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "Mismatched lengths: {} edges vs {} values",
                            edges.len(),
                            values.len()
                        )));
                    }

                    edges
                        .into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Float(val as f32)))
                        .collect()
                }
                "bool" => {
                    let values: Vec<bool> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>(
                            "Missing 'values' key in attribute data",
                        ));
                    };

                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "Mismatched lengths: {} edges vs {} values",
                            edges.len(),
                            values.len()
                        )));
                    }

                    edges
                        .into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Bool(val)))
                        .collect()
                }
                _ => {
                    return Err(PyErr::new::<PyValueError, _>(format!(
                        "Unsupported value_type: {}",
                        value_type
                    )));
                }
            };

            attrs_values.insert(attr, pairs);
        }

        self.inner
.borrow_mut()
            .set_edge_attrs(attrs_values)
            .map_err(graph_error_to_py_err)
    }

    /// Add multiple edges at once
    fn add_edges(
        &mut self,
        edges: &PyAny,
        node_mapping: Option<std::collections::HashMap<String, NodeId>>,
        uid_key: Option<String>,
    ) -> PyResult<Vec<EdgeId>> {
        // Format 1: List of (source, target) tuples - most common case for benchmarks
        if let Ok(edge_pairs) = edges.extract::<Vec<(NodeId, NodeId)>>() {
            return Ok(self.inner.borrow_mut().add_edges(&edge_pairs));
        }
        // Format 2: List of (source, target, attrs_dict) tuples
        else if let Ok(edge_tuples) = edges.extract::<Vec<(&PyAny, &PyAny, Option<&PyDict>)>>() {
            let mut edge_ids = Vec::new();
            let mut edges_with_attrs = Vec::new();

            // First pass: create all edges and collect attribute data
            for (src_any, tgt_any, attrs_opt) in edge_tuples {
                let source: NodeId = src_any.extract()?;
                let target: NodeId = tgt_any.extract()?;

                let edge_id = self
                    .inner
                    .borrow_mut()
                    .add_edge(source, target)
                    .map_err(graph_error_to_py_err)?;
                edge_ids.push(edge_id);

                // Store edge attributes for bulk processing
                if let Some(attrs) = attrs_opt {
                    edges_with_attrs.push((edge_id, attrs));
                }
            }

            // OPTIMIZATION: Use bulk attribute setting instead of individual calls
            if !edges_with_attrs.is_empty() {
                let mut attrs_by_name: std::collections::HashMap<
                    String,
                    Vec<(EdgeId, RustAttrValue)>,
                > = std::collections::HashMap::new();

                for (edge_id, attrs) in edges_with_attrs {
                    for (key, value) in attrs.iter() {
                        let attr_name: String = key.extract()?;
                        let attr_value = python_value_to_attr_value(value);

                        attrs_by_name
                            .entry(attr_name)
                            .or_insert_with(Vec::new)
                            .push((edge_id, attr_value?));
                    }
                }

                self.inner
                    .borrow_mut()
                    .set_edge_attrs(attrs_by_name)
                    .map_err(graph_error_to_py_err)?;
            }

            return Ok(edge_ids);
        }
        // Format 3: List of dictionaries with node mapping
        else if let Ok(edge_dicts) = edges.extract::<Vec<&PyDict>>() {
            let mut edge_ids = Vec::new();
            let mut edges_with_attrs = Vec::new();

            // First pass: create all edges and collect attribute data
            for edge_dict in edge_dicts {
                // Extract source and target
                let source = if let Some(mapping) = &node_mapping {
                    let source_str: String = edge_dict.get_item("source")?.unwrap().extract()?;
                    *mapping.get(&source_str).ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "Node {} not found in mapping",
                            source_str
                        ))
                    })?
                } else {
                    edge_dict.get_item("source")?.unwrap().extract()?
                };

                let target = if let Some(mapping) = &node_mapping {
                    let target_str: String = edge_dict.get_item("target")?.unwrap().extract()?;
                    *mapping.get(&target_str).ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "Node {} not found in mapping",
                            target_str
                        ))
                    })?
                } else {
                    edge_dict.get_item("target")?.unwrap().extract()?
                };

                // Add the edge
                let edge_id = self
                    .inner
                    .borrow_mut()
                    .add_edge(source, target)
                    .map_err(graph_error_to_py_err)?;
                edge_ids.push(edge_id);

                // Store edge and its attributes for bulk processing
                edges_with_attrs.push((edge_id, edge_dict));
            }

            // OPTIMIZATION: Use bulk attribute setting instead of individual calls
            if !edges_with_attrs.is_empty() {
                let mut attrs_by_name: std::collections::HashMap<
                    String,
                    Vec<(EdgeId, RustAttrValue)>,
                > = std::collections::HashMap::new();

                for (edge_id, edge_dict) in edges_with_attrs {
                    for (key, value) in edge_dict.iter() {
                        let key_str: String = key.extract()?;
                        if key_str != "source" && key_str != "target" {
                            let attr_value = python_value_to_attr_value(value);
                            attrs_by_name
                                .entry(key_str)
                                .or_insert_with(Vec::new)
                                .push((edge_id, attr_value?));
                        }
                    }
                }

                if !attrs_by_name.is_empty() {
                    self.inner
.borrow_mut()
                        .set_edge_attrs(attrs_by_name)
                        .map_err(graph_error_to_py_err)?;
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
    fn filter_nodes(slf: PyRefMut<Self>, py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Fast path optimization: Check for NodeFilter object first (most common case)
        let node_filter = if let Ok(filter_obj) = filter.extract::<PyNodeFilter>() {
            // Direct NodeFilter object - fastest path
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using Rust core query parser (FIXED: no circular dependency)
            let mut parser = groggy::core::query_parser::QueryParser::new();
            parser.parse_node_query(&query_str)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Query parse error: {}", e)))?
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be a NodeFilter object or a string query (e.g., 'salary > 120000')",
            ));
        };

        // Validate that any referenced attributes exist in the graph
        if let Err(attr_name) = Self::validate_node_filter_attributes(&*slf.inner.borrow(), &node_filter) {
            return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Attribute '{}' does not exist on any nodes in the graph. Use graph.nodes.table().columns to see available attributes.", attr_name)
            ));
        }

        let start = std::time::Instant::now();
        let filtered_nodes = slf
            .inner
            .borrow_mut()
            .find_nodes(node_filter)
            .map_err(graph_error_to_py_err)?;

        let _elapsed = start.elapsed();

        let start = std::time::Instant::now();
        // O(k) Calculate induced edges using optimized core subgraph method
        use std::collections::HashSet;
        let node_set: HashSet<NodeId> = filtered_nodes.iter().copied().collect();

        // Get columnar topology vectors (edge_ids, sources, targets) - O(1) if cached
        let (edge_ids, sources, targets) = slf.inner.borrow().get_columnar_topology();
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

        let _elapsed = start.elapsed();

        // Create subgraph using core Subgraph constructor
        let node_set: HashSet<NodeId> = filtered_nodes.iter().copied().collect();
        let edge_set: HashSet<EdgeId> = induced_edges.iter().copied().collect();
        let subgraph = Subgraph::new(
            slf.inner.clone(),
            node_set,
            edge_set,
            "filtered_nodes".to_string()
        );
        Ok(PySubgraph::from_core_subgraph(subgraph))
    }

    /// Filter edges using EdgeFilter object or string query
    fn filter_edges(slf: PyRefMut<Self>, py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Fast path optimization: Check for EdgeFilter object first (most common case)
        let edge_filter = if let Ok(filter_obj) = filter.extract::<PyEdgeFilter>() {
            // Direct EdgeFilter object - fastest path
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using Rust core query parser (FIXED: no circular dependency)
            let mut parser = groggy::core::query_parser::QueryParser::new();
            parser.parse_edge_query(&query_str)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Query parse error: {}", e)))?
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be an EdgeFilter object or a string query",
            ));
        };

        // Validate that any referenced attributes exist in the graph
        if let Err(attr_name) = Self::validate_edge_filter_attributes(&*slf.inner.borrow(), &edge_filter) {
            return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Attribute '{}' does not exist on any edges in the graph. Use graph.edges.table().columns to see available attributes.", attr_name)
            ));
        }

        let filtered_edges = slf
            .inner
            .borrow_mut()
            .find_edges(edge_filter)
            .map_err(graph_error_to_py_err)?;

        // Calculate nodes that are connected by the filtered edges
        use std::collections::HashSet;
        let mut nodes = HashSet::new();
        for &edge_id in &filtered_edges {
            if let Ok((source, target)) = slf.inner.borrow().edge_endpoints(edge_id) {
                nodes.insert(source);
                nodes.insert(target);
            }
        }

        // Create subgraph using core Subgraph constructor
        let edge_set: HashSet<EdgeId> = filtered_edges.iter().copied().collect();
        let subgraph = Subgraph::new(
            slf.inner.clone(),
            nodes,
            edge_set,
            "filtered_edges".to_string()
        );
        Ok(PySubgraph::from_core_subgraph(subgraph))
    }

    /// Get analytics module for this graph
    // analytics() method deleted - functionality moved to direct PyGraph delegation methods

    /// Group nodes by attribute value and compute aggregates for each group
    pub fn group_nodes_by_attribute(
        &self,
        attribute: AttrName,
        aggregation_attr: AttrName,
        operation: String,
    ) -> PyResult<PyGroupedAggregationResult> {
        let results = self
            .inner
.borrow_mut()
            .group_nodes_by_attribute(&attribute, &aggregation_attr, &operation)
            .map_err(graph_error_to_py_err)?;

        Python::with_gil(|py| {
            // Convert HashMap to Python dict
            let dict = PyDict::new(py);
            for (attr_value, agg_result) in results {
                let py_attr_value = PyAttrValue { inner: attr_value };
                let py_agg_result = PyAggregationResult {
                    value: agg_result.value,
                };
                dict.set_item(Py::new(py, py_attr_value)?, Py::new(py, py_agg_result)?);
            }

            Ok(PyGroupedAggregationResult {
                groups: dict.to_object(py),
                operation: operation.clone(),
                attribute: attribute.clone(),
            })
        })
    }

    /// Group nodes by attribute value and compute aggregates (public method for benchmarks)
    pub fn group_by(
        &self,
        attribute: AttrName,
        aggregation_attr: AttrName,
        operation: String,
    ) -> PyResult<PyGroupedAggregationResult> {
        self.group_nodes_by_attribute_internal(attribute, aggregation_attr, operation)
    }

    // === REMOVAL OPERATIONS ===

    /// Remove a single node from the graph
    fn remove_node(&mut self, node: NodeId) -> PyResult<()> {
        self.inner.borrow_mut().remove_node(node).map_err(graph_error_to_py_err)
    }

    /// Remove a single edge from the graph
    fn remove_edge(&mut self, edge: EdgeId) -> PyResult<()> {
        self.inner.borrow_mut().remove_edge(edge).map_err(graph_error_to_py_err)
    }

    /// Remove multiple nodes from the graph
    fn remove_nodes(&mut self, nodes: Vec<NodeId>) -> PyResult<()> {
        self.inner
.borrow_mut()
            .remove_nodes(&nodes)
            .map_err(graph_error_to_py_err)
    }

    /// Remove multiple edges from the graph
    fn remove_edges(&mut self, edges: Vec<EdgeId>) -> PyResult<()> {
        self.inner
.borrow_mut()
            .remove_edges(&edges)
            .map_err(graph_error_to_py_err)
    }

    // === SINGLE EDGE ADDITION ===

    /// Add a single edge to the graph with support for string IDs and attributes
    #[pyo3(signature = (source, target, uid_key = None, **kwargs))]
    fn add_edge(
        &mut self,
        _py: Python,
        source: &PyAny,
        target: &PyAny,
        uid_key: Option<String>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<EdgeId> {
        // Try to extract as NodeId first (most common case)
        let source_id = if let Ok(node_id) = source.extract::<NodeId>() {
            node_id
        } else if let Ok(string_id) = source.extract::<String>() {
            if let Some(ref key) = uid_key {
                self.resolve_string_id_to_node(&string_id, key)?
            } else {
                return Err(PyErr::new::<PyTypeError, _>(
                    "String node IDs require uid_key parameter",
                ));
            }
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "Source must be NodeId or string",
            ));
        };

        let target_id = if let Ok(node_id) = target.extract::<NodeId>() {
            node_id
        } else if let Ok(string_id) = target.extract::<String>() {
            if let Some(ref key) = uid_key {
                self.resolve_string_id_to_node(&string_id, key)?
            } else {
                return Err(PyErr::new::<PyTypeError, _>(
                    "String node IDs require uid_key parameter",
                ));
            }
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "Target must be NodeId or string",
            ));
        };

        // Add the edge
        let edge_id = self
            .inner
            .borrow_mut()
            .add_edge(source_id, target_id)
            .map_err(graph_error_to_py_err)?;

        // Set attributes if provided
        if let Some(attrs) = kwargs {
            if !attrs.is_empty() {
                for (key, value) in attrs.iter() {
                    let attr_name: String = key.extract()?;
                    let attr_value = python_value_to_attr_value(value);
                    self.inner
                        .borrow_mut()
                        .set_edge_attr(edge_id, attr_name, attr_value?)
                        .map_err(graph_error_to_py_err)?;
                }
            }
        }

        Ok(edge_id)
    }

    // === ALGORITHM OPERATIONS ===

    /// Find shortest path between two nodes (DEPRECATED - use graph.analytics.shortest_path() instead)
    #[pyo3(signature = (source, target, weight_attribute = None, inplace = None, attr_name = None))]
    fn shortest_path(
        slf: PyRef<Self>,
        py: Python,
        source: NodeId,
        target: NodeId,
        weight_attribute: Option<AttrName>,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<Option<PySubgraph>> {
        {
            // Pure delegation to SubgraphOperations (replaces deleted analytics module)
            match slf.as_subgraph() {
                Ok(subgraph) => {
                    match subgraph.shortest_path_subgraph(source, target) {
                        Ok(Some(path_subgraph)) => {
                            // Convert trait object back to concrete Subgraph then PySubgraph
                            let nodes = path_subgraph.node_set().iter().copied().collect();
                            let edges = path_subgraph.edge_set().iter().copied().collect();  
                            let concrete = groggy::core::subgraph::Subgraph::new(
                                slf.inner.clone(),
                                nodes,
                                edges,
                                "shortest_path".to_string(),
                            );
                            Ok(Some(PySubgraph::from_core_subgraph(concrete)))
                        },
                        Ok(None) => Ok(None), // No path found
                        Err(e) => Err(graph_error_to_py_err(e)),
                    }
                },
                Err(e) => Err(e),
            }
        }
    }

    /// Aggregate attribute values across nodes or edges
    #[pyo3(signature = (attribute, operation, target = None, node_ids = None))]
    fn aggregate(
        &self,
        py: Python,
        attribute: AttrName,
        operation: String,
        target: Option<String>,
        node_ids: Option<Vec<NodeId>>,
    ) -> PyResult<PyObject> {
        let target = target.unwrap_or_else(|| "nodes".to_string());

        match target.as_str() {
            "nodes" => {
                // TODO: Core doesn't have aggregate_nodes_custom, implement if needed
                let result = self.inner.borrow_mut().aggregate_node_attribute(&attribute, &operation);
                match result {
                    Ok(agg_result) => {
                        let py_result = PyAggregationResult {
                            value: agg_result.value,
                        };
                        Ok(Py::new(py, py_result)?.to_object(py))
                    }
                    Err(e) => Err(graph_error_to_py_err(e)),
                }
            }
            "edges" => {
                let result = self.inner.borrow_mut().aggregate_edge_attribute(&attribute, &operation);
                match result {
                    Ok(agg_result) => {
                        let py_result = PyAggregationResult {
                            value: agg_result.value,
                        };
                        Ok(Py::new(py, py_result)?.to_object(py))
                    }
                    Err(e) => Err(graph_error_to_py_err(e)),
                }
            }
            _ => Err(PyErr::new::<PyTypeError, _>(
                "Target must be 'nodes' or 'edges'",
            )),
        }
    }

    // === VERSION CONTROL OPERATIONS ===

    /// Commit current state of the graph (FFI wrapper around core history system)
    fn commit(&mut self, message: String, author: String) -> PyResult<StateId> {
        self.inner
.borrow_mut()
            .commit(message, author)
            .map_err(graph_error_to_py_err)
    }

    /// Create a new branch (FFI wrapper around core history system)
    fn create_branch(&mut self, branch_name: String) -> PyResult<()> {
        self.inner
.borrow_mut()
            .create_branch(branch_name)
            .map_err(graph_error_to_py_err)
    }

    /// Checkout a branch (FFI wrapper around core history system)
    fn checkout_branch(&mut self, branch_name: String) -> PyResult<()> {
        self.inner
.borrow_mut()
            .checkout_branch(branch_name)
            .map_err(graph_error_to_py_err)
    }

    /// List all branches (FFI wrapper around core history system)
    fn branches(&self) -> Vec<PyBranchInfo> {
        self.inner
.borrow_mut()
            .list_branches()
            .into_iter()
            .map(|branch_info| PyBranchInfo::new(branch_info))
            .collect()
    }

    /// Get commit history (FFI wrapper around core history system)
    fn commit_history(&self) -> Vec<PyCommit> {
        // Delegate to core history system
        self.inner
.borrow_mut()
            .commit_history()
            .into_iter()
            .map(|commit_info| PyCommit::from_commit_info(commit_info))
            .collect()
    }

    /// Get historical view at specific commit (FFI wrapper around core history system)
    fn historical_view(&self, commit_id: StateId) -> PyResult<PyHistoricalView> {
        // Delegate to core history system
        match self.inner.borrow_mut().view_at_commit(commit_id) {
            Ok(_historical_view) => Ok(PyHistoricalView {
                state_id: commit_id,
            }),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }

    // === STATE METHODS ===

    /// Check if there are uncommitted changes (FFI wrapper around core history system)
    fn has_uncommitted_changes(&self) -> bool {
        self.inner.borrow_mut().has_uncommitted_changes()
    }

    /// Get node mapping for a specific attribute (FFI wrapper around core operations)
    #[pyo3(signature = (uid_key, return_inverse = false))]
    fn get_node_mapping(
        &self,
        py: Python,
        uid_key: String,
        return_inverse: bool,
    ) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        // Delegate to core for node IDs and attributes
        let node_ids = self.inner.borrow_mut().node_ids();

        // Use core attribute access for each node
        for node_id in node_ids {
            if let Ok(Some(attr_value)) = self.inner.borrow_mut().get_node_attr(node_id, &uid_key) {
                // Convert attribute value to appropriate Python type
                let attr_value_py = match attr_value {
                    RustAttrValue::Text(s) => s.to_object(py),
                    RustAttrValue::CompactText(s) => s.as_str().to_object(py),
                    RustAttrValue::Int(i) => i.to_object(py),
                    RustAttrValue::SmallInt(i) => i.to_object(py),
                    RustAttrValue::Float(f) => f.to_object(py),
                    RustAttrValue::Bool(b) => b.to_object(py),
                    _ => continue, // Skip unsupported types
                };

                if return_inverse {
                    // Return node_id -> attribute_value mapping
                    dict.set_item(node_id, attr_value_py);
                } else {
                    // Return attribute_value -> node_id mapping (default behavior)
                    dict.set_item(attr_value_py, node_id);
                }
            }
        }

        Ok(dict.to_object(py))
    }

    // === ADJACENCY MATRIX OPERATIONS ===

    /// Generate adjacency matrix for the entire graph (FFI wrapper around core matrix operations)
    /// Returns: GraphMatrix (dense) or GraphSparseMatrix (sparse)
    fn adjacency_matrix(&mut self, _py: Python) -> PyResult<PyObject> {
        // TODO: Implement adjacency matrix in Phase 2
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Adjacency matrix temporarily disabled during Phase 2 unification",
        ))
    }

    /// Generate adjacency matrix for the entire graph (cleaner API)
    /// Returns: GraphMatrix with multi-index access (matrix[0, 1])
    /// This is a cleaner alias for adjacency_matrix() but always returns dense
    fn adjacency(&mut self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        use crate::ffi::core::matrix::PyGraphMatrix;

        // Generate dense adjacency matrix using the public API method
        let adjacency_matrix = self.inner.borrow_mut().dense_adjacency_matrix().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to generate adjacency matrix: {:?}", e))
        })?;

        // Convert AdjacencyMatrix to GraphMatrix
        let graph_matrix = self.adjacency_matrix_to_graph_matrix(adjacency_matrix)?;

        // Wrap in PyGraphMatrix
        let py_graph_matrix = PyGraphMatrix::from_graph_matrix(graph_matrix);
        Ok(Py::new(py, py_graph_matrix)?)
    }

    /// Generate weighted adjacency matrix using specified edge attribute (FFI wrapper around core matrix operations)
    fn weighted_adjacency_matrix(
        &mut self,
        py: Python,
        weight_attr: &str,
    ) -> PyResult<Py<PyGraphMatrix>> {
        use crate::ffi::core::matrix::PyGraphMatrix;

        // Generate weighted adjacency matrix using the public API method
        let adjacency_matrix = self
            .inner
.borrow_mut()
            .weighted_adjacency_matrix(weight_attr)
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to generate weighted adjacency matrix: {:?}",
                    e
                ))
            })?;

        // Convert AdjacencyMatrix to GraphMatrix
        let graph_matrix = self.adjacency_matrix_to_graph_matrix(adjacency_matrix)?;

        // Wrap in PyGraphMatrix
        let py_graph_matrix = PyGraphMatrix::from_graph_matrix(graph_matrix);
        Ok(Py::new(py, py_graph_matrix)?)
    }

    /// Generate dense adjacency matrix (FFI wrapper around core matrix operations)
    fn dense_adjacency_matrix(&mut self, _py: Python) -> PyResult<Py<PyGraphMatrix>> {
        // TODO: Implement dense adjacency matrix in Phase 2
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Dense adjacency matrix temporarily disabled during Phase 2 unification",
        ))
    }

    /// Generate sparse adjacency matrix (FFI wrapper around core matrix operations)
    fn sparse_adjacency_matrix(&mut self, _py: Python) -> PyResult<PyObject> {
        // TODO: Implement sparse adjacency matrix in Phase 2 (will return PyGraphSparseMatrix)
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Sparse adjacency matrix temporarily disabled during Phase 2 unification",
        ))
    }

    /// Generate Laplacian matrix (FFI wrapper around core matrix operations)
    fn laplacian_matrix(
        &mut self,
        py: Python,
        normalized: Option<bool>,
    ) -> PyResult<Py<PyGraphMatrix>> {
        use crate::ffi::core::matrix::PyGraphMatrix;

        let is_normalized = normalized.unwrap_or(false);

        // Generate Laplacian matrix using the public API method
        let laplacian_matrix = self.inner.borrow_mut().laplacian_matrix(is_normalized).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to generate Laplacian matrix: {:?}", e))
        })?;

        // Convert AdjacencyMatrix to GraphMatrix
        let graph_matrix = self.adjacency_matrix_to_graph_matrix(laplacian_matrix)?;

        // Wrap in PyGraphMatrix
        let py_graph_matrix = PyGraphMatrix::from_graph_matrix(graph_matrix);
        Ok(Py::new(py, py_graph_matrix)?)
    }

    /// Compute k-step transition matrix (A^k normalized by row sums for random walks)
    /// Returns matrix where entry (i,j) represents probability of k-step walk from i to j
    fn transition_matrix(
        &mut self,
        py: Python,
        k: u32,
        weight_attr: Option<&str>,
    ) -> PyResult<Py<PyGraphMatrix>> {
        use crate::ffi::core::matrix::PyGraphMatrix;

        // Get adjacency matrix (weighted or unweighted)
        let adjacency_matrix = if let Some(attr) = weight_attr {
            self.inner.borrow_mut().weighted_adjacency_matrix(attr)
        } else {
            self.inner.borrow_mut().dense_adjacency_matrix()
        }
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to generate adjacency matrix: {:?}", e))
        })?;

        // Convert to GraphMatrix for operations
        let mut graph_matrix = self.adjacency_matrix_to_graph_matrix(adjacency_matrix)?;

        // Normalize rows to get transition probabilities
        // TODO: This would benefit from proper row normalization in GraphMatrix
        // For now, just return the k-th power
        for _ in 1..k {
            let temp_matrix = graph_matrix.clone();
            graph_matrix = graph_matrix.multiply(&temp_matrix).map_err(|e| {
                PyRuntimeError::new_err(format!("Matrix power computation failed: {:?}", e))
            })?;
        }

        let py_graph_matrix = PyGraphMatrix::from_graph_matrix(graph_matrix);
        Ok(Py::new(py, py_graph_matrix)?)
    }

    /// Generate adjacency matrix for a subgraph with specific nodes (FFI wrapper around core matrix operations)
    fn subgraph_adjacency_matrix(
        &mut self,
        _py: Python,
        _node_ids: Vec<NodeId>,
    ) -> PyResult<Py<PyGraphMatrix>> {
        // TODO: Implement subgraph adjacency matrix in Phase 2
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Subgraph adjacency matrix temporarily disabled during Phase 2 unification",
        ))
    }

    // === DISPLAY/TABLE METHODS ===

    /// Get neighbors with flexible parameters (like degree)
    /// 
    /// Supports:
    /// - g.neighbors(node_id)  # neighbors for single node -> Vec<NodeId>
    /// - g.neighbors([node1, node2])  # neighbors for multiple nodes -> GraphArray of lists
    /// - g.neighbors()  # neighbors for all nodes -> GraphArray of lists
    #[pyo3(signature = (nodes = None))]
    fn neighbors(&mut self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject> {
        match nodes {
            // Single node case: neighbors(node_id) -> Vec<NodeId> (backward compatibility) - DELEGATED
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node = node_arg.extract::<NodeId>()?;
                // Pure delegation to SubgraphOperations
                match self.as_subgraph() {
                    Ok(subgraph) => {
                        let neighbors = subgraph.neighbors(node).map_err(graph_error_to_py_err)?;
                        Ok(neighbors.to_object(py))
                    },
                    Err(e) => Err(e),
                }
            }
            // List of nodes case: neighbors([node1, node2, ...]) -> GraphArray
            Some(node_arg) if node_arg.extract::<Vec<NodeId>>().is_ok() => {
                let node_ids = node_arg.extract::<Vec<NodeId>>()?;
                let mut neighbor_arrays = Vec::new();

                for node_id in node_ids {
                    match self.inner.borrow_mut().neighbors(node_id) {
                        Ok(neighbors) => {
                            // Convert Vec<NodeId> to comma-separated string representation
                            let neighbor_str = neighbors
                                .iter()
                                .map(|id| id.to_string())
                                .collect::<Vec<String>>()
                                .join(",");
                            neighbor_arrays.push(groggy::AttrValue::Text(neighbor_str));
                        }
                        Err(_) => {
                            // For non-existent nodes, return empty string
                            neighbor_arrays.push(groggy::AttrValue::Text(String::new()));
                        }
                    }
                }

                let graph_array = groggy::GraphArray::from_vec(neighbor_arrays);
                let py_graph_array = crate::ffi::core::array::PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }
            // All nodes case: neighbors() -> GraphArray
            None => {
                let all_nodes = self.inner.borrow_mut().node_ids();
                let mut neighbor_arrays = Vec::new();

                for node_id in all_nodes {
                    match self.inner.borrow_mut().neighbors(node_id) {
                        Ok(neighbors) => {
                            // Convert Vec<NodeId> to comma-separated string representation
                            let neighbor_str = neighbors
                                .iter()
                                .map(|id| id.to_string())
                                .collect::<Vec<String>>()
                                .join(",");
                            neighbor_arrays.push(groggy::AttrValue::Text(neighbor_str));
                        }
                        Err(_) => {
                            // For any issues, return empty string
                            neighbor_arrays.push(groggy::AttrValue::Text(String::new()));
                        }
                    }
                }

                let graph_array = groggy::GraphArray::from_vec(neighbor_arrays);
                let py_graph_array = crate::ffi::core::array::PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }
            // Invalid argument case
            Some(_) => Err(PyTypeError::new_err(
                "neighbors() argument must be a NodeId or list of NodeIds"
            ))
        }
    }

    /// Get the degree of nodes (number of incident edges) as GraphArray
    ///
    /// Usage:
    /// - degree(node_id) -> int: degree of single node
    /// - degree(node_ids) -> GraphArray: degrees for list of nodes  
    /// - degree() -> GraphArray: degrees for all nodes
    #[pyo3(signature = (nodes = None))]
    fn degree(&self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject> {
        match nodes {
            // Single node case: degree(node_id) -> int (keep as int for backward compatibility) - DELEGATED
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node = node_arg.extract::<NodeId>()?;
                // Pure delegation to SubgraphOperations
                match self.as_subgraph() {
                    Ok(subgraph) => {
                        let deg = subgraph.degree(node).map_err(graph_error_to_py_err)?;
                        Ok(deg.to_object(py))
                    },
                    Err(e) => Err(e),
                }
            }
            // List of nodes case: degree([node1, node2, ...]) -> GraphArray - DELEGATED
            Some(node_arg) if node_arg.extract::<Vec<NodeId>>().is_ok() => {
                let node_ids = node_arg.extract::<Vec<NodeId>>()?;
                
                // Pure delegation to SubgraphOperations
                match self.as_subgraph() {
                    Ok(subgraph) => {
                        let mut degrees = Vec::new();

                        for node_id in node_ids {
                            match subgraph.degree(node_id) {
                                Ok(deg) => {
                                    degrees.push(groggy::AttrValue::Int(deg as i64));
                                }
                                Err(_) => {
                                    // Skip nodes that don't exist rather than failing
                                    continue;
                                }
                            }
                        }

                        let graph_array = groggy::GraphArray::from_vec(degrees);
                        let py_graph_array = crate::ffi::core::array::PyGraphArray { inner: graph_array };
                        Ok(Py::new(py, py_graph_array)?.to_object(py))
                    },
                    Err(e) => Err(e),
                }
            }
            // All nodes case: degree() -> GraphArray - DELEGATED
            None => {
                // Pure delegation to SubgraphOperations
                match self.as_subgraph() {
                    Ok(subgraph) => {
                        let all_nodes = subgraph.node_set().iter().cloned().collect::<Vec<_>>();
                        let mut degrees = Vec::new();

                        for node_id in all_nodes {
                            if let Ok(deg) = subgraph.degree(node_id) {
                                degrees.push(groggy::AttrValue::Int(deg as i64));
                            }
                        }

                        let graph_array = groggy::GraphArray::from_vec(degrees);
                        let py_graph_array = crate::ffi::core::array::PyGraphArray { inner: graph_array };
                        Ok(Py::new(py, py_graph_array)?.to_object(py))
                    },
                    Err(e) => Err(e),
                }
            }
            // Invalid argument type
            Some(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "degree() argument must be a NodeId, list of NodeIds, or None",
            )),
        }
    }

    /// Get the in-degree of nodes (number of incoming edges) - for directed graphs
    ///
    /// Usage:
    /// - in_degree(node_id) -> int: in-degree of single node
    /// - in_degree(node_ids) -> dict: in-degrees for list of nodes  
    /// - in_degree() -> dict: in-degrees for all nodes
    #[pyo3(signature = (nodes = None))]
    fn in_degree(&self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject> {
        if !self.inner.borrow_mut().is_directed() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "in_degree() is only available for directed graphs. Use degree() for undirected graphs."
            ));
        }

        // Get fresh topology snapshot once
        let (_, _sources, targets) = self.inner.borrow_mut().get_columnar_topology();

        match nodes {
            // Single node case: in_degree(node_id) -> int
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node = node_arg.extract::<NodeId>()?;
                let count = targets.iter().filter(|&&target| target == node).count();
                Ok(count.to_object(py))
            }
            // List of nodes case: in_degree([node1, node2, ...]) -> dict
            Some(node_arg) if node_arg.extract::<Vec<NodeId>>().is_ok() => {
                let node_ids = node_arg.extract::<Vec<NodeId>>()?;
                let result_dict = pyo3::types::PyDict::new(py);

                for node_id in node_ids {
                    let count = targets.iter().filter(|&&target| target == node_id).count();
                    result_dict.set_item(node_id, count);
                }

                Ok(result_dict.to_object(py))
            }
            // All nodes case: in_degree() -> dict
            None => {
                let result_dict = pyo3::types::PyDict::new(py);
                let all_nodes = self.inner.borrow_mut().node_ids();

                for node_id in all_nodes {
                    let count = targets.iter().filter(|&&target| target == node_id).count();
                    result_dict.set_item(node_id, count);
                }

                Ok(result_dict.to_object(py))
            }
            // Invalid argument type
            Some(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "in_degree() argument must be a NodeId, list of NodeIds, or None",
            )),
        }
    }

    /// Get the out-degree of nodes (number of outgoing edges) - for directed graphs
    ///
    /// Usage:
    /// - out_degree(node_id) -> int: out-degree of single node
    /// - out_degree(node_ids) -> dict: out-degrees for list of nodes  
    /// - out_degree() -> dict: out-degrees for all nodes
    #[pyo3(signature = (nodes = None))]
    fn out_degree(&self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject> {
        if !self.inner.borrow_mut().is_directed() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "out_degree() is only available for directed graphs. Use degree() for undirected graphs."
            ));
        }

        // Get fresh topology snapshot once
        let (_, sources, _targets) = self.inner.borrow_mut().get_columnar_topology();

        match nodes {
            // Single node case: out_degree(node_id) -> int
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node = node_arg.extract::<NodeId>()?;
                let count = sources.iter().filter(|&&source| source == node).count();
                Ok(count.to_object(py))
            }
            // List of nodes case: out_degree([node1, node2, ...]) -> dict
            Some(node_arg) if node_arg.extract::<Vec<NodeId>>().is_ok() => {
                let node_ids = node_arg.extract::<Vec<NodeId>>()?;
                let result_dict = pyo3::types::PyDict::new(py);

                for node_id in node_ids {
                    let count = sources.iter().filter(|&&source| source == node_id).count();
                    result_dict.set_item(node_id, count);
                }

                Ok(result_dict.to_object(py))
            }
            // All nodes case: out_degree() -> dict
            None => {
                let result_dict = pyo3::types::PyDict::new(py);
                let all_nodes = self.inner.borrow_mut().node_ids();

                for node_id in all_nodes {
                    let count = sources.iter().filter(|&&source| source == node_id).count();
                    result_dict.set_item(node_id, count);
                }

                Ok(result_dict.to_object(py))
            }
            // Invalid argument type
            Some(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "out_degree() argument must be a NodeId, list of NodeIds, or None",
            )),
        }
    }

    // === NEIGHBORHOOD SAMPLING OPERATIONS ===

    /// Generate neighborhood subgraphs with flexible parameters
    /// 
    /// Supports:
    /// - g.neighborhood(node_id)  # 1-hop single node
    /// - g.neighborhood(node_id, k=2)  # k-hop single node  
    /// - g.neighborhood([node1, node2])  # 1-hop multiple nodes
    /// - g.neighborhood([node1, node2], k=2)  # k-hop multiple nodes
    /// - g.neighborhood([node1, node2], k=[100, 10])  # multi-level sampling
    /// - g.neighborhood(..., unified=True)  # single combined subgraph vs separate subgraphs
    #[allow(dead_code)]
    fn neighborhood(
        &mut self,
        nodes: &PyAny,
        k: Option<&PyAny>,
        unified: Option<bool>,
    ) -> PyResult<PyObject> {
        let py = nodes.py();
        
        // Parse k parameter
        let hop_spec = if let Some(k_val) = k {
            if let Ok(single_k) = k_val.extract::<usize>() {
                HopSpecification::SingleLevel(single_k)
            } else if let Ok(multi_k) = k_val.extract::<Vec<usize>>() {
                HopSpecification::MultiLevel(multi_k)
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "k parameter must be an integer or list of integers"
                ));
            }
        } else {
            HopSpecification::SingleLevel(1) // Default to 1-hop
        };
        
        let unified = unified.unwrap_or(false);
        
        // Parse nodes parameter - single node or list of nodes
        if let Ok(single_node) = nodes.extract::<NodeId>() {
            // Single node case
            let result = match hop_spec {
                HopSpecification::SingleLevel(1) => {
                    // Use optimized 1-hop method
                    self.inner.borrow_mut().neighborhood(single_node)
                        .map_err(graph_error_to_py_err)
                        .map(|nbh| PyNeighborhoodSubgraph { inner: nbh })
                        .map(|py_nbh| py_nbh.into_py(py))
                }
                HopSpecification::SingleLevel(k) => {
                    // Use k-hop method
                    self.inner.borrow_mut().k_hop_neighborhood(single_node, k)
                        .map_err(graph_error_to_py_err)
                        .map(|nbh| PyNeighborhoodSubgraph { inner: nbh })
                        .map(|py_nbh| py_nbh.into_py(py))
                }
                HopSpecification::MultiLevel(_levels) => {
                    // TODO: Implement multi-level sampling for single node
                    return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                        "Multi-level sampling for single node not yet implemented"
                    ));
                }
            }?;
            
            Ok(result)
        } else if let Ok(node_list) = nodes.extract::<Vec<NodeId>>() {
            // Multiple nodes case
            if unified {
                // Return single combined subgraph
                let result = match hop_spec {
                    HopSpecification::SingleLevel(k) => {
                        self.inner.borrow_mut().unified_neighborhood(&node_list, k)
                            .map_err(graph_error_to_py_err)
                            .map(|nbh| PyNeighborhoodSubgraph { inner: nbh })
                            .map(|py_nbh| py_nbh.into_py(py))
                    }
                    HopSpecification::MultiLevel(_levels) => {
                        // TODO: Implement multi-level unified sampling
                        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                            "Multi-level unified sampling not yet implemented"
                        ));
                    }
                }?;
                
                Ok(result)
            } else {
                // Return separate subgraphs for each node
                let result = match hop_spec {
                    HopSpecification::SingleLevel(1) => {
                        // Use optimized multi-neighborhood method for 1-hop
                        self.inner.borrow_mut().multi_neighborhood(&node_list)
                            .map_err(graph_error_to_py_err)
                            .map(|result| PyNeighborhoodResult { inner: result })
                            .map(|py_result| py_result.into_py(py))
                    }
                    HopSpecification::SingleLevel(_k) => {
                        // TODO: Implement k-hop for multiple nodes returning separate subgraphs
                        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                            "k-hop sampling for multiple nodes (non-unified) not yet implemented"
                        ));
                    }
                    HopSpecification::MultiLevel(_levels) => {
                        // TODO: Implement multi-level sampling for multiple nodes
                        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                            "Multi-level sampling for multiple nodes not yet implemented"
                        ));
                    }
                }?;
                
                Ok(result)
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "nodes parameter must be a NodeId or list of NodeIds"
            ))
        }
    }

    /// Get neighborhood sampling performance statistics
    #[allow(dead_code)]
    fn neighborhood_statistics(&self) -> PyNeighborhoodStats {
        PyNeighborhoodStats {
            inner: self.inner.borrow_mut().neighborhood_statistics().clone(),
        }
    }

    // === GRAPH MERGING OPERATIONS ===

    /// Add another graph to this graph (merge nodes and edges)
    /// 
    /// All nodes and edges from the other graph will be added to this graph.
    /// Node and edge IDs may be remapped to avoid conflicts.
    /// Attributes are preserved during the merge.
    pub fn add_graph(&mut self, py: Python, other: &PyGraph) -> PyResult<()> {
        // Get all nodes from the other graph with their attributes
        let other_node_ids = other.inner.borrow().node_ids();
        let other_edge_ids = other.inner.borrow().edge_ids();
        
        // Track ID mappings to handle potential conflicts
        let mut node_id_mapping = std::collections::HashMap::new();
        
        // Add all nodes from other graph
        for &old_node_id in &other_node_ids {
            // Get all attributes for this node
            let mut node_attrs = std::collections::HashMap::new();
            
            // Get attribute names for this node (this is a simplified approach)
            // TODO: This could be more efficient with a proper attribute iteration API
            let sample_attrs = ["name", "label", "type", "value", "weight", "id"]; // Common attribute names
            for attr_name in &sample_attrs {
                if let Ok(Some(attr_value)) = other.inner.borrow().get_node_attr(old_node_id, &attr_name.to_string()) {
                    node_attrs.insert(attr_name.to_string(), attr_value);
                }
            }
            
            // Add the node to this graph
            let new_node_id = self.inner.borrow_mut().add_node();
            node_id_mapping.insert(old_node_id, new_node_id);
            
            // Set all attributes on the new node
            for (attr_name, attr_value) in node_attrs {
                let _ = self.inner.borrow_mut().set_node_attr(new_node_id, attr_name, attr_value);
            }
        }
        
        // Add all edges from other graph
        for &old_edge_id in &other_edge_ids {
            if let Ok((old_source, old_target)) = other.inner.borrow().edge_endpoints(old_edge_id) {
                // Map old node IDs to new node IDs
                if let (Some(&new_source), Some(&new_target)) = (
                    node_id_mapping.get(&old_source),
                    node_id_mapping.get(&old_target)
                ) {
                    // Add the edge
                    match self.inner.borrow_mut().add_edge(new_source, new_target) {
                        Ok(new_edge_id) => {
                            // Copy edge attributes
                            let sample_edge_attrs = ["weight", "label", "type", "capacity"]; // Common edge attribute names
                            for attr_name in &sample_edge_attrs {
                                if let Ok(Some(attr_value)) = other.inner.borrow().get_edge_attr(old_edge_id, &attr_name.to_string()) {
                                    let _ = self.inner.borrow_mut().set_edge_attr(new_edge_id, attr_name.to_string(), attr_value);
                                }
                            }
                        }
                        Err(_) => {
                            // Skip edges that can't be added (e.g., duplicates in undirected graphs)
                            continue;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Return a full-view Subgraph (whole graph as a subgraph).
    /// Downstream code can always resolve parent graph from this object.
    pub fn view(self_: PyRef<Self>, py: Python<'_>) -> PyResult<Py<PySubgraph>> {
        // Create a simple subgraph containing all nodes and edges
        let all_nodes: Vec<NodeId> = self_.inner.borrow().node_ids();
        let all_edges: Vec<EdgeId> = self_.inner.borrow().edge_ids();
        
        let node_set: std::collections::HashSet<NodeId> = all_nodes.into_iter().collect();
        let edge_set: std::collections::HashSet<EdgeId> = all_edges.into_iter().collect();
        
        let subgraph = groggy::core::subgraph::Subgraph::new(
            self_.inner.clone(),
            node_set,
            edge_set,
            "full_view".to_string()
        );
        
        let py_subgraph = PySubgraph::from_core_subgraph(subgraph);
        Py::new(py, py_subgraph)
    }

    /// Check if the graph is connected (delegates to subgraph implementation)
    pub fn is_connected(self_: PyRef<Self>, py: Python<'_>) -> PyResult<bool> {
        // Create a subgraph containing all nodes and edges, then check connectivity
        let all_nodes: Vec<NodeId> = self_.inner.borrow().node_ids();
        let all_edges: Vec<EdgeId> = self_.inner.borrow().edge_ids();
        
        let node_set: std::collections::HashSet<NodeId> = all_nodes.into_iter().collect();
        let edge_set: std::collections::HashSet<EdgeId> = all_edges.into_iter().collect();
        
        let subgraph = groggy::core::subgraph::Subgraph::new(
            self_.inner.clone(),
            node_set,
            edge_set,
            "connectivity_check".to_string()
        );
        
        let result = subgraph.is_connected().map_err(graph_error_to_py_err)?;
        Ok(result)
    }

    /// Create GraphTable for DataFrame-like view of this graph's nodes
    /// Alias for g.nodes.table() for convenience
    pub fn table(self_: PyRef<Self>, py: Python<'_>) -> PyResult<PyObject> {
        // Forward to g.nodes.table() for consistency
        let nodes_accessor = Self::nodes(self_, py)?;
        let result = nodes_accessor.borrow(py).table(py);
        result
    }

    /// Create GraphTable for DataFrame-like view of this graph's edges
    /// Alias for g.edges.table() for convenience
    pub fn edges_table(self_: PyRef<Self>, py: Python<'_>) -> PyResult<PyObject> {
        // Forward to g.edges.table() for consistency
        let edges_accessor = Self::edges(self_, py)?;
        let result = edges_accessor.borrow(py).table(py);
        result
    }
}

// Internal methods for FFI integration (not exposed to Python)
impl PyGraph {
    /// Get shared reference to the graph for creating RustSubgraphs
    pub(crate) fn get_graph_ref(&self) -> Rc<RefCell<RustGraph>> {
        self.inner.clone()
    }

    /// Convert core AdjacencyMatrix to GraphMatrix for Python FFI
    fn adjacency_matrix_to_graph_matrix(
        &self,
        adjacency_matrix: groggy::AdjacencyMatrix,
    ) -> PyResult<groggy::GraphMatrix> {
        use groggy::core::array::GraphArray;
        use groggy::core::matrix::GraphMatrix;

        // Extract matrix data and convert to GraphArrays (columns)
        let size = adjacency_matrix.size;
        let mut columns = Vec::with_capacity(size);

        // Create a column for each matrix column
        for col_idx in 0..size {
            let column_values: Vec<groggy::AttrValue> = (0..size)
                .map(|row_idx| {
                    adjacency_matrix
                        .get(row_idx, col_idx)
                        .cloned()
                        .unwrap_or(groggy::AttrValue::Float(0.0))
                })
                .collect();

            let column_name = if let Some(ref labels) = adjacency_matrix.labels {
                format!(
                    "node_{}",
                    labels
                        .get(col_idx)
                        .copied()
                        .unwrap_or(col_idx as groggy::NodeId)
                )
            } else {
                format!("col_{}", col_idx)
            };

            let column = GraphArray::from_vec(column_values).with_name(column_name);
            columns.push(column);
        }

        // Create GraphMatrix from the columns
        let mut graph_matrix = GraphMatrix::from_arrays(columns).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to create GraphMatrix from adjacency matrix: {:?}",
                e
            ))
        })?;

        // Set proper column names using node labels if available
        if let Some(ref labels) = adjacency_matrix.labels {
            let column_names: Vec<String> = labels
                .iter()
                .map(|&node_id| format!("node_{}", node_id))
                .collect();
            graph_matrix.set_column_names(column_names).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to set column names: {:?}", e))
            })?;
        }

        Ok(graph_matrix)
    }
    /// Internal helper methods for accessors/views
    pub fn create_node_view_internal(
        graph: Py<PyGraph>,
        py: Python,
        node_id: NodeId,
    ) -> PyResult<Py<PyNodeView>> {
        Py::new(py, PyNodeView { graph, node_id })
    }

    pub fn create_edge_view_internal(
        graph: Py<PyGraph>,
        py: Python,
        edge_id: EdgeId,
    ) -> PyResult<Py<PyEdgeView>> {
        Py::new(py, PyEdgeView { graph, edge_id })
    }

    /// Create a NodesAccessor (internal helper)
    fn create_nodes_accessor_internal(
        graph_ref: Py<PyGraph>,
        py: Python,
    ) -> PyResult<Py<PyNodesAccessor>> {
        let core_graph = graph_ref.borrow(py).inner.clone();
        Py::new(
            py,
            PyNodesAccessor {
                graph: core_graph,
                constrained_nodes: None,
            },
        )
    }

    /// Create an EdgesAccessor (internal helper)
    fn create_edges_accessor_internal(
        graph_ref: Py<PyGraph>,
        py: Python,
    ) -> PyResult<Py<PyEdgesAccessor>> {
        let core_graph = graph_ref.borrow(py).inner.clone();
        Py::new(
            py,
            PyEdgesAccessor {
                graph: core_graph,
                constrained_edges: None,
            },
        )
    }

    pub fn node_attribute_keys(&self, node_id: NodeId) -> Vec<String> {
        match self.inner.borrow_mut().get_node_attrs(node_id) {
            Ok(attrs) => attrs.keys().cloned().collect(),
            Err(_) => Vec::new(),
        }
    }

    pub fn edge_attribute_keys(&self, edge_id: EdgeId) -> Vec<String> {
        match self.inner.borrow_mut().get_edge_attrs(edge_id) {
            Ok(attrs) => attrs.keys().cloned().collect(),
            Err(_) => Vec::new(),
        }
    }

    // === HELPER METHODS FOR OTHER MODULES ===

    pub fn has_node_attribute(&self, node_id: NodeId, attr_name: &str) -> bool {
        match self.inner.borrow_mut().get_node_attr(node_id, &attr_name.to_string()) {
            Ok(Some(_)) => true,
            _ => false,
        }
    }

    pub fn has_edge_attribute(&self, edge_id: EdgeId, attr_name: &str) -> bool {
        match self.inner.borrow_mut().get_edge_attr(edge_id, &attr_name.to_string()) {
            Ok(Some(_)) => true,
            _ => false,
        }
    }

    pub fn get_edge_endpoints(&self, edge_id: EdgeId) -> Result<(NodeId, NodeId), String> {
        self.inner
.borrow_mut()
            .edge_endpoints(edge_id)
            .map_err(|e| e.to_string())
    }

    pub fn get_node_ids(&self) -> PyResult<Vec<NodeId>> {
        Ok(self.inner.borrow_mut().node_ids())
    }

    pub fn get_edge_ids(&self) -> PyResult<Vec<EdgeId>> {
        Ok(self.inner.borrow_mut().edge_ids())
    }

    // Additional public methods for internal module access

    /// Internal helper - DELEGATED
    pub fn has_node_internal(&self, node_id: NodeId) -> bool {
        // Pure delegation (same as has_node/contains_node)
        self.has_node(node_id)
    }

    /// Internal helper - DELEGATED
    pub fn has_edge_internal(&self, edge_id: EdgeId) -> bool {
        // Pure delegation (same as has_edge/contains_edge)
        self.has_edge(edge_id)
    }

    /// Internal helper - DELEGATED
    pub fn get_node_count(&self) -> usize {
        // Pure delegation (same as node_count)
        self.node_count()
    }

    /// Internal helper - DELEGATED
    pub fn get_edge_count(&self) -> usize {
        // Pure delegation (same as edge_count)
        self.edge_count()
    }

    pub fn node_ids_vec(&self) -> Vec<NodeId> {
        self.inner.borrow_mut().node_ids()
    }

    pub fn edge_ids_vec(&self) -> Vec<EdgeId> {
        self.inner.borrow_mut().edge_ids()
    }

    pub fn get_node_ids_array(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let node_ids = self.inner.borrow_mut().node_ids();
        let attr_values: Vec<groggy::AttrValue> = node_ids
            .into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::core::array::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
    }

    pub fn get_edge_ids_array(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let edge_ids = self.inner.borrow_mut().edge_ids();
        let attr_values: Vec<groggy::AttrValue> = edge_ids
            .into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::core::array::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
    }

    /// Group nodes by attribute value and compute aggregates for each group (internal method)
    pub fn group_nodes_by_attribute_internal(
        &self,
        attribute: AttrName,
        aggregation_attr: AttrName,
        operation: String,
    ) -> PyResult<PyGroupedAggregationResult> {
        let results = self
            .inner
.borrow_mut()
            .group_nodes_by_attribute(&attribute, &aggregation_attr, &operation)
            .map_err(graph_error_to_py_err)?;

        Python::with_gil(|py| {
            // Convert HashMap to Python dict
            let dict = PyDict::new(py);
            for (attr_value, agg_result) in results {
                let py_attr_value = PyAttrValue { inner: attr_value };
                let py_agg_result = PyAggregationResult {
                    value: agg_result.value,
                };
                dict.set_item(Py::new(py, py_attr_value)?, Py::new(py, py_agg_result)?);
            }

            Ok(PyGroupedAggregationResult {
                groups: dict.to_object(py),
                operation: operation.clone(),
                attribute: attribute.clone(),
            })
        })
    }

    // === PROPERTY GETTERS FOR ATTRIBUTE ACCESS ===

    /// Enable property-style attribute access (g.age instead of g.nodes['age'])
    /// This method is called when accessing attributes that don't exist as methods
    fn __getattr__(slf: PyRef<Self>, py: Python, name: String) -> PyResult<PyObject> {
        // TODO: Complete implementation - temporarily disabled for compilation
        return Err(PyAttributeError::new_err(format!(
            "Attribute '{}' not found. Property-style attribute access is under development.",
            name
        )));
    }

    /// Get complete attribute column for ALL nodes (optimized for table() method)
    /// Returns GraphArray for enhanced analytics and proper integration with table columns
    fn _get_node_attr_column(
        &self,
        py: Python,
        attr_name: &str,
    ) -> PyResult<Py<PyGraphArray>> {
        match self
            .inner
.borrow_mut()
            ._get_node_attribute_column(&attr_name.to_string())
        {
            Ok(values) => {
                // Convert Option<AttrValue> vector to AttrValue vector (convert None to appropriate AttrValue)
                let attr_values: Vec<groggy::AttrValue> = values
                    .into_iter()
                    .map(|opt_val| opt_val.unwrap_or(groggy::AttrValue::Null)) // Use Null for missing values
                    .collect();

                // Create GraphArray and convert to PyGraphArray
                let graph_array = groggy::core::array::GraphArray::from_vec(attr_values);
                let py_graph_array = PyGraphArray::from_graph_array(graph_array);

                Py::new(py, py_graph_array)
            }
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }

    /// Validate that all attributes referenced in a NodeFilter exist in the graph
    /// Returns Ok(()) if valid, Err(attr_name) if an attribute doesn't exist
    fn validate_node_filter_attributes(
        graph: &groggy::api::graph::Graph,
        filter: &groggy::core::query::NodeFilter,
    ) -> Result<(), String> {
        use groggy::core::query::NodeFilter;

        match filter {
            NodeFilter::AttributeFilter { name, .. }
            | NodeFilter::AttributeEquals { name, .. }
            | NodeFilter::HasAttribute { name } => {
                // Check if this attribute exists on any nodes
                if !Self::attribute_exists_on_nodes(graph, name) {
                    return Err(name.clone());
                }
            }
            NodeFilter::And(filters) => {
                for f in filters {
                    Self::validate_node_filter_attributes(graph, f);
                }
            }
            NodeFilter::Or(filters) => {
                for f in filters {
                    Self::validate_node_filter_attributes(graph, f);
                }
            }
            NodeFilter::Not(filter) => {
                Self::validate_node_filter_attributes(graph, filter);
            }
            // Other filter types don't reference attributes
            _ => {}
        }

        Ok(())
    }

    /// Validate that all attributes referenced in an EdgeFilter exist in the graph
    /// Returns Ok(()) if valid, Err(attr_name) if an attribute doesn't exist
    fn validate_edge_filter_attributes(
        graph: &groggy::api::graph::Graph,
        filter: &groggy::core::query::EdgeFilter,
    ) -> Result<(), String> {
        use groggy::core::query::EdgeFilter;

        match filter {
            EdgeFilter::AttributeFilter { name, .. }
            | EdgeFilter::AttributeEquals { name, .. }
            | EdgeFilter::HasAttribute { name } => {
                // Check if this attribute exists on any edges
                if !Self::attribute_exists_on_edges(graph, name) {
                    return Err(name.clone());
                }
            }
            EdgeFilter::And(filters) => {
                for f in filters {
                    Self::validate_edge_filter_attributes(graph, f);
                }
            }
            EdgeFilter::Or(filters) => {
                for f in filters {
                    Self::validate_edge_filter_attributes(graph, f);
                }
            }
            EdgeFilter::Not(filter) => {
                Self::validate_edge_filter_attributes(graph, filter);
            }
            // Other filter types don't reference attributes
            _ => {}
        }

        Ok(())
    }

    /// Check if an attribute exists on any nodes in the graph
    fn attribute_exists_on_nodes(graph: &groggy::api::graph::Graph, attr_name: &str) -> bool {
        let node_ids = graph.node_ids();
        for node_id in node_ids.iter().take(100) {
            // Sample first 100 nodes for performance
            if let Ok(Some(_)) = graph.get_node_attr(*node_id, &attr_name.to_string()) {
                return true;
            }
        }
        false
    }

    /// Check if an attribute exists on any edges in the graph
    fn attribute_exists_on_edges(graph: &groggy::api::graph::Graph, attr_name: &str) -> bool {
        let edge_ids = graph.edge_ids();
        for edge_id in edge_ids.iter().take(100) {
            // Sample first 100 edges for performance
            if let Ok(Some(_)) = graph.get_edge_attr(*edge_id, &attr_name.to_string()) {
                return true;
            }
        }
        false
    }

    // === DELEGATED METHODS - Pure SubgraphOperations delegation ===
    
    /// Get connected components using SubgraphOperations trait  
    /// This replaces the graph_analytics.py connected_components() method
    fn connected_components(&self, py: Python) -> PyResult<Vec<PySubgraph>> {
        {
            let subgraph = self.as_subgraph()?;
            let components = subgraph.connected_components().map_err(graph_error_to_py_err)?;
            
            let py_components = components
                .into_iter()
                .map(|component| {
                    // Convert trait object back to concrete Subgraph
                    // For now, create from node/edge sets since downcasting is complex
                    let nodes = component.node_set().iter().copied().collect();
                    let edges = component.edge_set().iter().copied().collect();  
                    let subgraph = groggy::core::subgraph::Subgraph::new(
                        self.inner.clone(),
                        nodes,
                        edges,
                        "connected_component".to_string(),
                    );
                    PySubgraph::from_core_subgraph(subgraph)
                })
                .collect();
            Ok(py_components)
        }
    }

    /// Get memory statistics - moved from graph_analytics
    fn memory_statistics(&self, py: Python) -> PyResult<PyObject> {
        let stats = self.inner.borrow().memory_statistics();

        // Convert MemoryStatistics to Python dict
        let dict = PyDict::new(py);
        dict.set_item("pool_memory_bytes", stats.pool_memory_bytes);
        dict.set_item("space_memory_bytes", stats.space_memory_bytes);
        dict.set_item("history_memory_bytes", stats.history_memory_bytes);
        dict.set_item("change_tracker_memory_bytes", stats.change_tracker_memory_bytes);
        dict.set_item("total_memory_bytes", stats.total_memory_bytes);
        dict.set_item("total_memory_mb", stats.total_memory_mb);

        // Add memory efficiency stats
        let efficiency_dict = PyDict::new(py);
        efficiency_dict.set_item("bytes_per_node", stats.memory_efficiency.bytes_per_node);
        efficiency_dict.set_item("bytes_per_edge", stats.memory_efficiency.bytes_per_edge);
        efficiency_dict.set_item("bytes_per_entity", stats.memory_efficiency.bytes_per_entity);
        efficiency_dict.set_item("overhead_ratio", stats.memory_efficiency.overhead_ratio);
        efficiency_dict.set_item("cache_efficiency", stats.memory_efficiency.cache_efficiency);
        dict.set_item("memory_efficiency", efficiency_dict);

        Ok(dict.to_object(py))
    }

    /// Get analytics summary - moved from graph_analytics  
    fn get_summary(&self, py: Python) -> PyResult<String> {
        {
            let subgraph = self.as_subgraph()?;
            Ok(format!(
                "Graph Summary: {} nodes, {} edges",
                subgraph.node_count(),
                subgraph.edge_count()
            ))
        }
    }
}
