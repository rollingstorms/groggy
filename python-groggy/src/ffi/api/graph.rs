use crate::ffi::api::graph_attributes::{PyGraphAttr, PyGraphAttrMut};
// Graph FFI Coordinator
//
// Main Python bindings for the Graph API.

use groggy::core::subgraph::Subgraph;
use groggy::core::traits::SubgraphOperations;
use groggy::{AttrName, AttrValue as RustAttrValue, EdgeId, Graph as RustGraph, NodeId, StateId};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cell::RefCell;
use std::rc::Rc;

// Import all graph modules
use crate::ffi::core::accessors::{PyEdgesAccessor, PyNodesAccessor};
use crate::ffi::core::array::PyGraphArray;
use crate::ffi::core::neighborhood::PyNeighborhoodStats;
use crate::ffi::core::query::{PyEdgeFilter, PyNodeFilter};
use crate::ffi::core::subgraph::PySubgraph;
use crate::ffi::core::traversal::PyGroupedAggregationResult;
use crate::ffi::core::views::{PyEdgeView, PyNodeView};
use crate::ffi::types::PyAttrValue;
use crate::ffi::utils::{graph_error_to_py_err, python_value_to_attr_value};

// Import version control types
use crate::ffi::api::graph_version::{PyBranchInfo, PyCommit, PyHistoricalView};
// Import internal helper classes
use crate::ffi::api::graph_analysis::PyGraphAnalysis;
use crate::ffi::api::graph_matrix::PyGraphMatrixHelper;

/// Specification for neighborhood hop parameters
#[derive(Debug, Clone)]

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

/// Convert AdjacencyMatrix to Python object - DELEGATION helper
impl PyGraph {
    pub(crate) fn adjacency_matrix_to_py_object(
        &self,
        py: Python,
        matrix: groggy::AdjacencyMatrix,
    ) -> PyResult<PyObject> {
        use crate::ffi::core::matrix::PyGraphMatrix;
        use pyo3::types::PyDict;

        // Create metadata dict
        let result_dict = PyDict::new(py);
        result_dict.set_item("size", matrix.size)?;
        result_dict.set_item("is_sparse", matrix.data.is_sparse())?;
        result_dict.set_item("type", "adjacency_matrix")?;

        // Convert to GraphMatrix for structured access
        match self.adjacency_matrix_to_graph_matrix(matrix) {
            Ok(graph_matrix) => {
                let py_matrix = PyGraphMatrix {
                    inner: graph_matrix,
                };
                result_dict.set_item("matrix", Py::new(py, py_matrix)?)?;
            }
            Err(_) => {
                // Fallback to basic matrix representation
                result_dict.set_item("matrix", py.None())?;
                result_dict.set_item("error", "Matrix conversion failed")?;
            }
        }

        Ok(result_dict.to_object(py))
    }
}

/// Python wrapper for the main Graph
#[pyclass(name = "Graph", unsendable)]
pub struct PyGraph {
    pub inner: Rc<RefCell<RustGraph>>,
}

impl Clone for PyGraph {
    fn clone(&self) -> Self {
        PyGraph {
            inner: self.inner.clone(),
        }
    }
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
    #[pyo3(signature = (directed = None, _config = None))]
    fn new(directed: Option<bool>, _config: Option<&PyDict>) -> PyResult<Self> {
        // Create graph with specified directionality (defaults to false/undirected)
        let is_directed = directed.unwrap_or(false);
        let rust_graph = if is_directed {
            RustGraph::new_directed()
        } else {
            RustGraph::new_undirected()
        };
        Ok(Self {
            inner: Rc::new(RefCell::new(rust_graph)),
        })
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
                            .or_default()
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
            if let Ok(Some(attr_value)) = self
                .inner
                .borrow_mut()
                .get_node_attr(node_id, &uid_key.to_string())
            {
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
        Py::new(py, py_graph_array)
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
        Py::new(py, py_graph_array)
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

    /// Add multiple edges at once
    fn add_edges(
        &mut self,
        edges: &PyAny,
        node_mapping: Option<std::collections::HashMap<String, NodeId>>,
        _uid_key: Option<String>,
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
                            .or_default()
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
                                .or_default()
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
    fn filter_nodes(slf: PyRefMut<Self>, _py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Fast path optimization: Check for NodeFilter object first (most common case)
        let node_filter = if let Ok(filter_obj) = filter.extract::<PyNodeFilter>() {
            // Direct NodeFilter object - fastest path
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using Rust core query parser (FIXED: no circular dependency)
            let mut parser = groggy::core::query_parser::QueryParser::new();
            parser.parse_node_query(&query_str).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Query parse error: {}", e))
            })?
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be a NodeFilter object or a string query (e.g., 'salary > 120000')",
            ));
        };

        // Validate that any referenced attributes exist in the graph
        if let Err(attr_name) =
            Self::validate_node_filter_attributes(&slf.inner.borrow(), &node_filter)
        {
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
            "filtered_nodes".to_string(),
        );
        PySubgraph::from_core_subgraph(subgraph)
    }

    /// Filter edges using EdgeFilter object or string query
    fn filter_edges(slf: PyRefMut<Self>, _py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Fast path optimization: Check for EdgeFilter object first (most common case)
        let edge_filter = if let Ok(filter_obj) = filter.extract::<PyEdgeFilter>() {
            // Direct EdgeFilter object - fastest path
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using Rust core query parser (FIXED: no circular dependency)
            let mut parser = groggy::core::query_parser::QueryParser::new();
            parser.parse_edge_query(&query_str).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Query parse error: {}", e))
            })?
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be an EdgeFilter object or a string query",
            ));
        };

        // Validate that any referenced attributes exist in the graph
        if let Err(attr_name) =
            Self::validate_edge_filter_attributes(&slf.inner.borrow(), &edge_filter)
        {
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
            "filtered_edges".to_string(),
        );
        PySubgraph::from_core_subgraph(subgraph)
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
                dict.set_item(Py::new(py, py_attr_value)?, Py::new(py, py_agg_result)?)?;
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
        self.inner
            .borrow_mut()
            .remove_node(node)
            .map_err(graph_error_to_py_err)
    }

    /// Remove a single edge from the graph
    fn remove_edge(&mut self, edge: EdgeId) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .remove_edge(edge)
            .map_err(graph_error_to_py_err)
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

    // === ATTRIBUTE OPERATIONS (direct core delegation) ===

    /// Set single node attribute - delegates to PyGraphAttrMut
    #[pyo3(signature = (node, attr, value))]
    fn set_node_attr(
        &mut self,
        node: NodeId,
        attr: String,
        value: &PyAny,
        py: Python,
    ) -> PyResult<()> {
        let mut attr_handler = PyGraphAttrMut::new(self.inner.clone());
        attr_handler.set_node_attr(py, node, attr, value)
    }

    /// Get single node attribute - delegates to PyGraphAttr
    #[pyo3(signature = (node, attr, default = None))]
    fn get_node_attr(
        &self,
        node: NodeId,
        attr: String,
        default: Option<&PyAny>,
        py: Python,
    ) -> PyResult<PyObject> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        attr_handler.get_node_attr(py, node, attr, default)
    }

    /// Set single edge attribute - delegates to PyGraphAttrMut
    #[pyo3(signature = (edge, attr, value))]
    fn set_edge_attr(
        &mut self,
        edge: EdgeId,
        attr: String,
        value: &PyAny,
        py: Python,
    ) -> PyResult<()> {
        let mut attr_handler = PyGraphAttrMut::new(self.inner.clone());
        attr_handler.set_edge_attr(py, edge, attr, value)
    }

    /// Get single edge attribute - delegates to PyGraphAttr
    #[pyo3(signature = (edge, attr, default = None))]
    fn get_edge_attr(
        &self,
        edge: EdgeId,
        attr: String,
        default: Option<&PyAny>,
        py: Python,
    ) -> PyResult<PyObject> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        attr_handler.get_edge_attr(py, edge, attr, default)
    }

    /// Set bulk node attributes - delegates to PyGraphAttrMut
    #[pyo3(signature = (attrs_dict))]
    fn set_node_attrs(&mut self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        let mut attr_handler = PyGraphAttrMut::new(self.inner.clone());
        attr_handler.set_node_attrs(py, attrs_dict)
    }

    /// Get bulk node attributes - delegates to PyGraphAttr
    #[pyo3(signature = (nodes, attrs))]
    fn get_node_attrs(
        &self,
        nodes: Vec<NodeId>,
        attrs: Vec<AttrName>,
        py: Python,
    ) -> PyResult<PyObject> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        attr_handler.get_node_attrs(py, nodes, attrs)
    }

    /// Set bulk edge attributes - delegates to PyGraphAttrMut
    #[pyo3(signature = (attrs_dict))]
    fn set_edge_attrs(&mut self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        let mut attr_handler = PyGraphAttrMut::new(self.inner.clone());
        attr_handler.set_edge_attrs(py, attrs_dict)
    }

    /// Get bulk edge attributes - delegates to PyGraphAttr
    #[pyo3(signature = (edges, attrs))]
    fn get_edge_attrs(
        &self,
        edges: Vec<EdgeId>,
        attrs: Vec<String>,
        py: Python,
    ) -> PyResult<PyObject> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        attr_handler.get_edge_attrs(py, edges, attrs)
    }

    /// Check if node has specific attribute - delegates to PyGraphAttr
    #[pyo3(signature = (node_id, attr_name))]
    fn has_node_attribute(&self, node_id: NodeId, attr_name: &str, py: Python) -> PyResult<bool> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        Ok(attr_handler.has_node_attribute(py, node_id, attr_name))
    }

    /// Check if edge has specific attribute - delegates to PyGraphAttr
    #[pyo3(signature = (edge_id, attr_name))]
    fn has_edge_attribute(&self, edge_id: EdgeId, attr_name: &str, py: Python) -> PyResult<bool> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        Ok(attr_handler.has_edge_attribute(py, edge_id, attr_name))
    }

    /// Get all attribute keys for a node - delegates to PyGraphAttr
    #[pyo3(signature = (node_id))]
    fn node_attribute_keys(&self, node_id: NodeId, py: Python) -> PyResult<Vec<String>> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        Ok(attr_handler.node_attribute_keys(py, node_id))
    }

    /// Get all attribute keys for an edge - delegates to PyGraphAttr
    #[pyo3(signature = (edge_id))]
    fn edge_attribute_keys(&self, edge_id: EdgeId, py: Python) -> PyResult<Vec<String>> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        Ok(attr_handler.edge_attribute_keys(py, edge_id))
    }

    /// Get all unique node attribute names across the entire graph
    fn all_node_attribute_names(&self) -> Vec<String> {
        use std::collections::HashSet;
        let graph_ref = self.inner.borrow();
        let mut all_attrs = HashSet::new();

        for node_id in graph_ref.node_ids() {
            if let Ok(attrs) = graph_ref.get_node_attrs(node_id) {
                for attr_name in attrs.keys() {
                    all_attrs.insert(attr_name.clone());
                }
            }
        }

        all_attrs.into_iter().collect()
    }

    /// Get all unique edge attribute names across the entire graph
    fn all_edge_attribute_names(&self) -> Vec<String> {
        use std::collections::HashSet;
        let graph_ref = self.inner.borrow();
        let mut all_attrs = HashSet::new();

        for edge_id in graph_ref.edge_ids() {
            if let Ok(attrs) = graph_ref.get_edge_attrs(edge_id) {
                for attr_name in attrs.keys() {
                    all_attrs.insert(attr_name.clone());
                }
            }
        }

        all_attrs.into_iter().collect()
    }

    // === ALGORITHM OPERATIONS (delegate to PyGraphAnalysis helper) ===

    /// Get neighbors of nodes - delegates to PyGraphAnalysis helper
    fn neighbors(&mut self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject> {
        let mut analysis_handler = PyGraphAnalysis::new(Py::new(py, self.clone())?)?;
        analysis_handler.neighbors(py, nodes)
    }

    /// Get neighborhood sampling - delegates to PyGraphAnalysis helper
    fn neighborhood(
        &mut self,
        py: Python,
        center_nodes: Vec<NodeId>,
        radius: Option<usize>,
        max_nodes: Option<usize>,
    ) -> PyResult<crate::ffi::core::neighborhood::PyNeighborhoodResult> {
        let mut analysis_handler = PyGraphAnalysis::new(Py::new(py, self.clone())?)?;
        analysis_handler.neighborhood(py, center_nodes, radius, max_nodes)
    }

    /// Get shortest path - delegates to PyGraphAnalysis helper
    #[pyo3(signature = (source, target, weight_attribute = None, inplace = None, attr_name = None))]
    fn shortest_path(
        &self,
        py: Python,
        source: NodeId,
        target: NodeId,
        weight_attribute: Option<AttrName>,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<PyObject> {
        let analysis_handler = PyGraphAnalysis::new(Py::new(py, self.clone())?)?;
        analysis_handler.shortest_path(py, source, target, weight_attribute, inplace, attr_name)
    }

    // === MATRIX OPERATIONS (delegate to PyGraphMatrixHelper) ===

    /// Get adjacency matrix - delegates to PyGraphMatrixHelper
    fn adjacency_matrix(&mut self, py: Python) -> PyResult<PyObject> {
        let mut matrix_handler = PyGraphMatrixHelper::new(Py::new(py, self.clone())?)?;
        matrix_handler.adjacency_matrix(py)
    }

    /// Simple adjacency matrix (alias) - delegates to PyGraphMatrixHelper
    fn adjacency(&mut self, py: Python) -> PyResult<Py<crate::ffi::core::matrix::PyGraphMatrix>> {
        let mut matrix_handler = PyGraphMatrixHelper::new(Py::new(py, self.clone())?)?;
        matrix_handler.adjacency(py)
    }

    /// Get weighted adjacency matrix - delegates to PyGraphMatrixHelper
    fn weighted_adjacency_matrix(
        &mut self,
        py: Python,
        weight_attr: &str,
    ) -> PyResult<Py<crate::ffi::core::matrix::PyGraphMatrix>> {
        let mut matrix_handler = PyGraphMatrixHelper::new(Py::new(py, self.clone())?)?;
        matrix_handler.weighted_adjacency_matrix(py, weight_attr)
    }

    /// Get dense adjacency matrix - delegates to PyGraphMatrixHelper
    fn dense_adjacency_matrix(
        &mut self,
        py: Python,
    ) -> PyResult<Py<crate::ffi::core::matrix::PyGraphMatrix>> {
        let mut matrix_handler = PyGraphMatrixHelper::new(Py::new(py, self.clone())?)?;
        matrix_handler.dense_adjacency_matrix(py)
    }

    /// Get sparse adjacency matrix - delegates to PyGraphMatrixHelper
    fn sparse_adjacency_matrix(&mut self, py: Python) -> PyResult<PyObject> {
        let mut matrix_handler = PyGraphMatrixHelper::new(Py::new(py, self.clone())?)?;
        matrix_handler.sparse_adjacency_matrix(py)
    }

    /// Get Laplacian matrix - delegates to PyGraphMatrixHelper
    fn laplacian_matrix(
        &mut self,
        py: Python,
        normalized: Option<bool>,
    ) -> PyResult<Py<crate::ffi::core::matrix::PyGraphMatrix>> {
        let mut matrix_handler = PyGraphMatrixHelper::new(Py::new(py, self.clone())?)?;
        matrix_handler.laplacian_matrix(py, normalized)
    }

    /// Generate transition matrix - delegates to PyGraphMatrixHelper
    fn transition_matrix(
        &mut self,
        py: Python,
    ) -> PyResult<Py<crate::ffi::core::matrix::PyGraphMatrix>> {
        let mut matrix_handler = PyGraphMatrixHelper::new(Py::new(py, self.clone())?)?;
        matrix_handler.transition_matrix(py)
    }

    /// Aggregate attribute values across nodes or edges
    #[pyo3(signature = (attribute, operation, target = None, _node_ids = None))]
    fn aggregate(
        &self,
        py: Python,
        attribute: AttrName,
        operation: String,
        target: Option<String>,
        _node_ids: Option<Vec<NodeId>>,
    ) -> PyResult<PyObject> {
        let target = target.unwrap_or_else(|| "nodes".to_string());

        match target.as_str() {
            "nodes" => {
                // TODO: Core doesn't have aggregate_nodes_custom, implement if needed
                let result = self
                    .inner
                    .borrow_mut()
                    .aggregate_node_attribute(&attribute, &operation);
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
                let result = self
                    .inner
                    .borrow_mut()
                    .aggregate_edge_attribute(&attribute, &operation);
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
            .map(PyBranchInfo::new)
            .collect()
    }

    /// Get commit history (FFI wrapper around core history system)
    fn commit_history(&self) -> Vec<PyCommit> {
        // Delegate to core history system
        self.inner
            .borrow_mut()
            .commit_history()
            .into_iter()
            .map(PyCommit::from_commit_info)
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
                    dict.set_item(node_id, attr_value_py)?;
                } else {
                    // Return attribute_value -> node_id mapping (default behavior)
                    dict.set_item(attr_value_py, node_id)?;
                }
            }
        }

        Ok(dict.to_object(py))
    }

    // === ADJACENCY MATRIX OPERATIONS ===

    // === DISPLAY/TABLE METHODS ===

    // === NEIGHBORHOOD SAMPLING OPERATIONS ===
    // Moved to graph_analysis.rs module

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
    pub fn add_graph(&mut self, _py: Python, other: &PyGraph) -> PyResult<()> {
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
                if let Ok(Some(attr_value)) = other
                    .inner
                    .borrow()
                    .get_node_attr(old_node_id, &attr_name.to_string())
                {
                    node_attrs.insert(attr_name.to_string(), attr_value);
                }
            }

            // Add the node to this graph
            let new_node_id = self.inner.borrow_mut().add_node();
            node_id_mapping.insert(old_node_id, new_node_id);

            // Set all attributes on the new node
            for (attr_name, attr_value) in node_attrs {
                let _ = self
                    .inner
                    .borrow_mut()
                    .set_node_attr(new_node_id, attr_name, attr_value);
            }
        }

        // Add all edges from other graph
        for &old_edge_id in &other_edge_ids {
            if let Ok((old_source, old_target)) = other.inner.borrow().edge_endpoints(old_edge_id) {
                // Map old node IDs to new node IDs
                if let (Some(&new_source), Some(&new_target)) = (
                    node_id_mapping.get(&old_source),
                    node_id_mapping.get(&old_target),
                ) {
                    // Add the edge
                    match self.inner.borrow_mut().add_edge(new_source, new_target) {
                        Ok(new_edge_id) => {
                            // Copy edge attributes
                            let sample_edge_attrs = ["weight", "label", "type", "capacity"]; // Common edge attribute names
                            for attr_name in &sample_edge_attrs {
                                if let Ok(Some(attr_value)) = other
                                    .inner
                                    .borrow()
                                    .get_edge_attr(old_edge_id, &attr_name.to_string())
                                {
                                    let _ = self.inner.borrow_mut().set_edge_attr(
                                        new_edge_id,
                                        attr_name.to_string(),
                                        attr_value,
                                    );
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
            "full_view".to_string(),
        );

        let py_subgraph = PySubgraph::from_core_subgraph(subgraph)?;
        Py::new(py, py_subgraph)
    }

    /// Check if the graph is connected (delegates to subgraph implementation)
    pub fn is_connected(self_: PyRef<Self>, _py: Python<'_>) -> PyResult<bool> {
        // Create a subgraph containing all nodes and edges, then check connectivity
        let all_nodes: Vec<NodeId> = self_.inner.borrow().node_ids();
        let all_edges: Vec<EdgeId> = self_.inner.borrow().edge_ids();

        let node_set: std::collections::HashSet<NodeId> = all_nodes.into_iter().collect();
        let edge_set: std::collections::HashSet<EdgeId> = all_edges.into_iter().collect();

        let subgraph = groggy::core::subgraph::Subgraph::new(
            self_.inner.clone(),
            node_set,
            edge_set,
            "connectivity_check".to_string(),
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

    /// Enable property-style attribute access (g.age instead of g.nodes['age'])
    /// This method is called when accessing attributes that don't exist as methods
    fn __getattr__(&self, py: Python, name: String) -> PyResult<PyObject> {
        use pyo3::exceptions::PyAttributeError;
        use pyo3::types::PyDict;

        // Check if this is a node attribute name that exists in the graph
        let all_node_attrs = self.all_node_attribute_names();
        if all_node_attrs.contains(&name) {
            // Return a dictionary mapping node IDs to their attribute values
            let result_dict = PyDict::new(py);
            let graph_ref = self.inner.borrow();

            for node_id in graph_ref.node_ids() {
                match graph_ref.get_node_attr(node_id, &name) {
                    Ok(Some(attr_value)) => {
                        use crate::ffi::types::PyAttrValue;
                        let py_attr_value = PyAttrValue::new(attr_value);
                        result_dict.set_item(node_id, py_attr_value)?;
                    }
                    Ok(None) => {
                        // Node doesn't have this attribute, skip or set to None
                        result_dict.set_item(node_id, py.None())?;
                    }
                    Err(_) => {
                        // Error accessing attribute, skip this node
                        continue;
                    }
                }
            }

            return Ok(result_dict.to_object(py));
        }

        // Check if this is an edge attribute name that exists in the graph
        let all_edge_attrs = self.all_edge_attribute_names();
        if all_edge_attrs.contains(&name) {
            // Return a dictionary mapping edge IDs to their attribute values
            let result_dict = PyDict::new(py);
            let graph_ref = self.inner.borrow();

            for edge_id in graph_ref.edge_ids() {
                match graph_ref.get_edge_attr(edge_id, &name) {
                    Ok(Some(attr_value)) => {
                        use crate::ffi::types::PyAttrValue;
                        let py_attr_value = PyAttrValue::new(attr_value);
                        result_dict.set_item(edge_id, py_attr_value)?;
                    }
                    Ok(None) => {
                        // Edge doesn't have this attribute, skip or set to None
                        result_dict.set_item(edge_id, py.None())?;
                    }
                    Err(_) => {
                        // Error accessing attribute, skip this edge
                        continue;
                    }
                }
            }

            return Ok(result_dict.to_object(py));
        }

        // Try delegating to subgraph methods (like degree, in_degree, out_degree)
        // Create a full subgraph containing all nodes and edges
        let graph_ref = self.inner.borrow();
        let all_nodes = graph_ref.node_ids().into_iter().collect();
        let all_edges = graph_ref.edge_ids().into_iter().collect();
        drop(graph_ref); // Release borrow

        let concrete_subgraph = groggy::core::subgraph::Subgraph::new(
            self.inner.clone(),
            all_nodes,
            all_edges,
            "full_graph_delegation".to_string(),
        );

        match PySubgraph::from_core_subgraph(concrete_subgraph) {
            Ok(py_subgraph) => {
                let subgraph_obj = Py::new(py, py_subgraph)?;
                match subgraph_obj.getattr(py, name.as_str()) {
                    Ok(result) => return Ok(result),
                    Err(_) => {
                        // Subgraph doesn't have this method either, fall through to error
                    }
                }
            }
            Err(_) => {
                // Failed to create subgraph, fall through to error
            }
        }

        // Attribute not found
        Err(PyAttributeError::new_err(format!(
            "Attribute '{}' not found. Available node attributes: {:?}, Available edge attributes: {:?}",
            name, all_node_attrs, all_edge_attrs
        )))
    }
}

// Internal methods for FFI integration (not exposed to Python)
impl PyGraph {
    /// Get shared reference to the graph for creating RustSubgraphs
    /// Convert core AdjacencyMatrix to GraphMatrix for Python FFI
    pub(crate) fn adjacency_matrix_to_graph_matrix(
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
        let core_graph = graph.borrow(py).inner.clone();
        Py::new(
            py,
            PyNodeView {
                graph: core_graph,
                node_id,
            },
        )
    }

    pub fn create_edge_view_internal(
        graph: Py<PyGraph>,
        py: Python,
        edge_id: EdgeId,
    ) -> PyResult<Py<PyEdgeView>> {
        let core_graph = graph.borrow(py).inner.clone();
        Py::new(
            py,
            PyEdgeView {
                graph: core_graph,
                edge_id,
            },
        )
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

    // === HELPER METHODS FOR OTHER MODULES ===

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
        Py::new(py, py_graph_array)
    }

    pub fn get_edge_ids_array(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let edge_ids = self.inner.borrow_mut().edge_ids();
        let attr_values: Vec<groggy::AttrValue> = edge_ids
            .into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::core::array::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Py::new(py, py_graph_array)
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
                dict.set_item(Py::new(py, py_attr_value)?, Py::new(py, py_agg_result)?)?;
            }

            Ok(PyGroupedAggregationResult {
                groups: dict.to_object(py),
                operation: operation.clone(),
                attribute: attribute.clone(),
            })
        })
    }

    // === PROPERTY GETTERS FOR ATTRIBUTE ACCESS ===

    /// Get complete attribute column for ALL nodes (optimized for table() method)
    /// Returns GraphArray for enhanced analytics and proper integration with table columns
    fn _get_node_attr_column(&self, py: Python, attr_name: &str) -> PyResult<Py<PyGraphArray>> {
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
                    let _ = Self::validate_node_filter_attributes(graph, f);
                }
            }
            NodeFilter::Or(filters) => {
                for f in filters {
                    let _ = Self::validate_node_filter_attributes(graph, f);
                }
            }
            NodeFilter::Not(filter) => {
                let _ = Self::validate_node_filter_attributes(graph, filter);
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
                    let _ = Self::validate_edge_filter_attributes(graph, f);
                }
            }
            EdgeFilter::Or(filters) => {
                for f in filters {
                    let _ = Self::validate_edge_filter_attributes(graph, f);
                }
            }
            EdgeFilter::Not(filter) => {
                let _ = Self::validate_edge_filter_attributes(graph, filter);
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
}
