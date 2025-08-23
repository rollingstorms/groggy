//! Subgraph FFI Bindings
//!
//! Python bindings for PySubgraph - dual-mode architecture supporting both
//! core RustSubgraph integration and legacy compatibility.

use groggy::{AttrValue, EdgeId, NodeId};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashSet;

// Import types from our FFI modules
use crate::ffi::api::graph::PyGraph;
use crate::ffi::core::accessors::{PyEdgesAccessor, PyNodesAccessor};
use crate::ffi::core::array::PyGraphArray;
use crate::ffi::core::query::PyNodeFilter;
use crate::ffi::core::table::PyGraphTable;
use crate::ffi::types::PyAttrValue;
use crate::ffi::utils::graph_error_to_py_err;

// Import the core Subgraph type
use groggy::core::subgraph::Subgraph as RustSubgraph;

/// Utility function to convert Python values to AttrValue
fn python_value_to_attr_value(value: &PyAny) -> PyResult<AttrValue> {
    if let Ok(int_val) = value.extract::<i64>() {
        Ok(AttrValue::Int(int_val))
    } else if let Ok(float_val) = value.extract::<f64>() {
        Ok(AttrValue::Float(float_val as f32))
    } else if let Ok(str_val) = value.extract::<String>() {
        Ok(AttrValue::Text(str_val))
    } else if let Ok(bool_val) = value.extract::<bool>() {
        Ok(AttrValue::Bool(bool_val))
    } else {
        Err(PyTypeError::new_err("Unsupported attribute value type"))
    }
}

/// Utility function to convert AttrValue to Python object
fn attr_value_to_python_value(py: Python, attr_value: &AttrValue) -> PyResult<PyObject> {
    match attr_value {
        AttrValue::Int(val) => Ok(val.to_object(py)),
        AttrValue::SmallInt(val) => Ok((*val as i64).to_object(py)),
        AttrValue::Float(val) => Ok(val.to_object(py)),
        AttrValue::Bool(val) => Ok(val.to_object(py)),
        AttrValue::Text(val) => Ok(val.to_object(py)),
        AttrValue::CompactText(val) => Ok(val.as_str().to_object(py)),
        _ => Ok(py.None()),
    }
}

/// Subgraph - represents a filtered view of the main graph with same interface
#[pyclass(name = "Subgraph", unsendable)]
pub struct PySubgraph {
    // Use the core Rust Subgraph with proper Rc<RefCell<Graph>> architecture
    inner: Option<RustSubgraph>,
    // Fallback data for legacy compatibility (when inner is None)
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
    subgraph_type: String,
    graph: Option<Py<PyGraph>>,
}

impl PySubgraph {
    /// Create a PySubgraph from a core RustSubgraph (with proper graph reference)
    pub fn from_core_subgraph(subgraph: RustSubgraph) -> Self {
        let nodes = subgraph.node_ids();
        let edges = subgraph.edge_ids();
        let subgraph_type = subgraph.subgraph_type().to_string();

        PySubgraph {
            inner: Some(subgraph),
            nodes,
            edges,
            subgraph_type,
            graph: None, // Not needed when we have inner
        }
    }

    /// Standard PySubgraph constructor
    pub fn new(
        nodes: Vec<NodeId>,
        edges: Vec<EdgeId>,
        subgraph_type: String,
        graph: Option<Py<PyGraph>>,
    ) -> Self {
        PySubgraph {
            inner: None,
            nodes,
            edges,
            subgraph_type,
            graph,
        }
    }

    /// Get nodes vector (for internal module access)
    pub fn get_nodes(&self) -> &Vec<NodeId> {
        &self.nodes
    }

    /// Get edges vector (for internal module access)  
    pub fn get_edges(&self) -> &Vec<EdgeId> {
        &self.edges
    }

    /// Set graph reference (for internal module access)
    pub fn set_graph_reference(&mut self, graph: Py<PyGraph>) {
        self.graph = Some(graph);
    }
}

#[pymethods]
impl PySubgraph {
    /// Get nodes as a property that supports indexing and attribute access
    #[getter]
    fn nodes(self_: PyRef<Self>, py: Python) -> PyResult<Py<PyNodesAccessor>> {
        if let Some(graph_ref) = &self_.graph {
            Py::new(
                py,
                PyNodesAccessor {
                    graph: graph_ref.clone(),
                    constrained_nodes: Some(self_.nodes.clone()),
                },
            )
        } else {
            Err(PyRuntimeError::new_err("No graph reference available"))
        }
    }

    /// Get edges as a property that supports indexing and attribute access
    #[getter]
    fn edges(self_: PyRef<Self>, py: Python) -> PyResult<Py<PyEdgesAccessor>> {
        if let Some(graph_ref) = &self_.graph {
            Py::new(
                py,
                PyEdgesAccessor {
                    graph: graph_ref.clone(),
                    constrained_edges: Some(self_.edges.clone()),
                },
            )
        } else {
            Err(PyRuntimeError::new_err("No graph reference available"))
        }
    }

    /// Python len() support - returns number of nodes
    fn __len__(&self) -> usize {
        self.nodes.len()
    }

    /// Node count property
    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Edge count property
    fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get node IDs in this subgraph as GraphArray (lazy Rust view) - use .values for Python list
    #[getter]
    fn node_ids(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let attr_values: Vec<groggy::AttrValue> = self
            .nodes
            .iter()
            .map(|&id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
    }

    /// Get edge IDs in this subgraph as GraphArray (lazy Rust view) - use .values for Python list
    #[getter]
    fn edge_ids(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let attr_values: Vec<groggy::AttrValue> = self
            .edges
            .iter()
            .map(|&id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
    }

    /// Check if a node exists in this subgraph
    fn has_node(&self, node_id: NodeId) -> bool {
        self.nodes.contains(&node_id)
    }

    /// Check if an edge exists in this subgraph
    fn has_edge(&self, edge_id: EdgeId) -> bool {
        self.edges.contains(&edge_id)
    }

    /// String representation  
    fn __repr__(&self) -> String {
        format!(
            "Subgraph(nodes={}, edges={}, type={})",
            self.nodes.len(),
            self.edges.len(),
            self.subgraph_type
        )
    }

    /// Detailed string representation with helpful information
    fn __str__(&self) -> String {
        let mut info = format!(
            "Subgraph with {} nodes and {} edges",
            self.nodes.len(),
            self.edges.len()
        );

        if !self.subgraph_type.is_empty() {
            info.push_str(&format!("\nType: {}", self.subgraph_type));
        }

        if !self.nodes.is_empty() {
            let node_sample = if self.nodes.len() <= 5 {
                format!("{:?}", self.nodes)
            } else {
                format!(
                    "[{}, {}, {}, ... {} more]",
                    self.nodes[0],
                    self.nodes[1],
                    self.nodes[2],
                    self.nodes.len() - 3
                )
            };
            info.push_str(&format!("\nNodes: {}", node_sample));
        }

        info.push_str(
            "\nAvailable methods: .set(**attrs), .filter_nodes(filter), .table(), .nodes, .edges",
        );
        info
    }

    /// Calculate subgraph density (number of edges / number of possible edges)
    fn density(&self) -> f64 {
        let num_nodes = self.nodes.len();
        let num_edges = self.edges.len();

        if num_nodes <= 1 {
            return 0.0;
        }

        // For an undirected graph, max edges = n(n-1)/2
        // For a directed graph, max edges = n(n-1)
        // Since we don't have easy access to graph type here, we'll assume undirected
        // This is the most common case and matches standard network analysis conventions
        let max_possible_edges = (num_nodes * (num_nodes - 1)) / 2;

        if max_possible_edges > 0 {
            num_edges as f64 / max_possible_edges as f64
        } else {
            0.0
        }
    }

    /// Get degree of nodes in subgraph as GraphArray
    ///
    /// Usage:
    /// - degree(node_id, full_graph=False) -> int: degree of single node (local or full graph)
    /// - degree(node_ids, full_graph=False) -> GraphArray: degrees for list of nodes
    /// - degree(full_graph=False) -> GraphArray: degrees for all nodes in subgraph
    ///
    /// Parameters:
    /// - nodes: Optional node ID, list of node IDs, or None for all nodes
    /// - full_graph: If False (default), compute degrees within subgraph only.
    ///               If True, compute degrees from the original full graph.
    #[pyo3(signature = (nodes = None, *, full_graph = false))]
    fn degree(&self, py: Python, nodes: Option<&PyAny>, full_graph: bool) -> PyResult<PyObject> {
        // Get graph reference if we need full graph degrees
        let graph_ref = if full_graph {
            self.graph.as_ref().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Cannot compute full graph degrees: subgraph has no parent graph reference",
                )
            })?
        } else {
            // We'll handle local degrees without needing the graph reference
            &self.graph.as_ref().unwrap() // Safe because subgraphs always have graph refs
        };

        match nodes {
            // Single node case
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node_id = node_arg.extract::<NodeId>()?;

                // Verify node is in subgraph
                if !self.nodes.contains(&node_id) {
                    return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                        "Node {} is not in this subgraph",
                        node_id
                    )));
                }

                let deg = if full_graph {
                    // Get degree from full graph
                    let graph = graph_ref.borrow(py);
                    graph.inner.degree(node_id).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
                    })?
                } else {
                    // Calculate local degree within subgraph
                    self.edges
                        .iter()
                        .filter(|&&edge_id| {
                            if let Some(graph_ref) = &self.graph {
                                let graph = graph_ref.borrow(py);
                                if let Ok((source, target)) = graph.inner.edge_endpoints(edge_id) {
                                    source == node_id || target == node_id
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        })
                        .count()
                };

                Ok(deg.to_object(py))
            }

            // List of nodes case
            Some(node_arg) if node_arg.extract::<Vec<NodeId>>().is_ok() => {
                let node_ids = node_arg.extract::<Vec<NodeId>>()?;
                let mut degrees = Vec::new();

                for node_id in node_ids {
                    // Verify node is in subgraph
                    if !self.nodes.contains(&node_id) {
                        continue; // Skip nodes not in subgraph
                    }

                    let deg = if full_graph {
                        // Get degree from main graph
                        let graph = graph_ref.borrow(py);
                        match graph.inner.degree(node_id) {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local degree within subgraph
                        self.edges
                            .iter()
                            .filter(|&&edge_id| {
                                if let Some(graph_ref) = &self.graph {
                                    let graph = graph_ref.borrow(py);
                                    if let Ok((source, target)) =
                                        graph.inner.edge_endpoints(edge_id)
                                    {
                                        source == node_id || target == node_id
                                    } else {
                                        false
                                    }
                                } else {
                                    false
                                }
                            })
                            .count()
                    };

                    degrees.push(groggy::AttrValue::Int(deg as i64));
                }

                let graph_array = groggy::GraphArray::from_vec(degrees);
                let py_graph_array = crate::ffi::core::array::PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }

            // All nodes case (or None)
            None => {
                let mut degrees = Vec::new();

                for &node_id in &self.nodes {
                    let deg = if full_graph {
                        // Get degree from main graph
                        let graph = graph_ref.borrow(py);
                        match graph.inner.degree(node_id) {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local degree within subgraph
                        self.edges
                            .iter()
                            .filter(|&&edge_id| {
                                if let Some(graph_ref) = &self.graph {
                                    let graph = graph_ref.borrow(py);
                                    if let Ok((source, target)) =
                                        graph.inner.edge_endpoints(edge_id)
                                    {
                                        source == node_id || target == node_id
                                    } else {
                                        false
                                    }
                                } else {
                                    false
                                }
                            })
                            .count()
                    };

                    degrees.push(groggy::AttrValue::Int(deg as i64));
                }

                let graph_array = groggy::GraphArray::from_vec(degrees);
                let py_graph_array = crate::ffi::core::array::PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }

            // Invalid argument type
            Some(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "degree() nodes argument must be a NodeId, list of NodeIds, or None",
            )),
        }
    }

    /// Filter edges within this subgraph (chainable)
    fn filter_edges(&self, _py: Python, _filter: &PyAny) -> PyResult<PySubgraph> {
        // Placeholder implementation
        Ok(PySubgraph::new(
            self.nodes.clone(),
            self.edges.clone(),
            format!("{}_edge_filtered", self.subgraph_type),
            self.graph.clone(),
        ))
    }

    /// Connected components within this subgraph
    fn connected_components(&self) -> PyResult<Vec<PySubgraph>> {
        // Use inner subgraph if available
        if let Some(ref inner_subgraph) = self.inner {
            let components = inner_subgraph.connected_components().map_err(|e| {
                PyErr::new::<PyRuntimeError, _>(format!(
                    "Failed to get connected components: {}",
                    e
                ))
            })?;

            let mut result = Vec::new();
            for (i, component) in components.iter().enumerate() {
                // STANDARDIZED: Use PySubgraph::new() like all other subgraph creation methods
                // This ensures consistent graph reference handling for .nodes/.edges accessors
                result.push(PySubgraph::new(
                    component.node_ids(),
                    component.edge_ids(),
                    format!("connected_component_{}", i),
                    self.graph.clone(), // Pass the graph reference consistently
                ));
            }
            Ok(result)
        } else {
            // Fallback - return single component for now
            Ok(vec![PySubgraph::new(
                self.nodes.clone(),
                self.edges.clone(),
                "component".to_string(),
                self.graph.clone(),
            )])
        }
    }

    /// Check if the subgraph is connected (has exactly one connected component)
    pub fn is_connected(&self) -> PyResult<bool> {
        // Use inner subgraph if available (preferred path)
        if let Some(ref inner_subgraph) = self.inner {
            inner_subgraph.is_connected().map_err(|e| {
                PyErr::new::<PyRuntimeError, _>(format!("Failed to check connectivity: {}", e))
            })
        } else {
            // Fallback - for now assume connected if we have nodes
            Ok(!self.nodes.is_empty())
        }
    }

    /// Set attributes on all nodes in this subgraph (batch operation)
    #[pyo3(signature = (**kwargs))]
    fn set(&mut self, py: Python, kwargs: Option<&PyDict>) -> PyResult<Py<PySubgraph>> {
        // Use inner Subgraph if available (preferred path)
        if let Some(ref inner_subgraph) = self.inner {
            if let Some(kwargs) = kwargs {
                for (key, value) in kwargs.iter() {
                    let attr_name: String = key.extract()?;
                    let attr_value = python_value_to_attr_value(value)?;

                    // Use the core Subgraph's bulk set method
                    inner_subgraph
                        .set_node_attribute_bulk(&attr_name, attr_value)
                        .map_err(|e| {
                            PyErr::new::<PyRuntimeError, _>(format!(
                                "Failed to set attribute: {}",
                                e
                            ))
                        })?;
                }
            }

            // Return self for chaining
            let new_subgraph = if let Some(ref inner) = self.inner {
                PySubgraph::from_core_subgraph(inner.clone())
            } else {
                PySubgraph::new(
                    self.nodes.clone(),
                    self.edges.clone(),
                    self.subgraph_type.clone(),
                    self.graph.clone(),
                )
            };
            Ok(Py::new(py, new_subgraph)?)
        }
        // Fallback to legacy implementation
        else if let Some(graph_ref) = &self.graph {
            if let Some(kwargs) = kwargs {
                let mut graph = graph_ref.borrow_mut(py);

                // Update all nodes in this subgraph
                for &node_id in &self.nodes {
                    for (key, value) in kwargs.iter() {
                        let attr_name: String = key.extract()?;
                        let attr_value = python_value_to_attr_value(value)?;
                        let py_attr_value = PyAttrValue::from_attr_value(attr_value);

                        graph.set_node_attribute(node_id, attr_name, &py_attr_value)?;
                    }
                }
            }

            // Return self for chaining
            Ok(Py::new(
                py,
                PySubgraph::new(
                    self.nodes.clone(),
                    self.edges.clone(),
                    self.subgraph_type.clone(),
                    self.graph.clone(),
                ),
            )?)
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(
                "Cannot set attributes on subgraph without graph reference. Use graph.filter_nodes() or similar methods."
            ))
        }
    }

    /// Update attributes on all nodes in this subgraph using dict syntax
    fn update(&mut self, py: Python, data: &PyDict) -> PyResult<Py<PySubgraph>> {
        if let Some(graph_ref) = &self.graph {
            let mut graph = graph_ref.borrow_mut(py);

            // Update all nodes in this subgraph
            for &node_id in &self.nodes {
                for (key, value) in data.iter() {
                    let attr_name: String = key.extract()?;
                    let attr_value = python_value_to_attr_value(value)?;
                    let py_attr_value = PyAttrValue::from_attr_value(attr_value);

                    graph.set_node_attribute(node_id, attr_name, &py_attr_value)?;
                }
            }

            // Return self for chaining
            Ok(Py::new(
                py,
                PySubgraph::new(
                    self.nodes.clone(),
                    self.edges.clone(),
                    self.subgraph_type.clone(),
                    self.graph.clone(),
                ),
            )?)
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(
                "Cannot update attributes on subgraph without graph reference. Use graph.filter_nodes() or similar methods."
            ))
        }
    }

    /// Column access: get all values for a node attribute within this subgraph
    /// This enables: subgraph['component_id'] -> GraphArray with statistical methods
    fn get_node_attribute_column(&self, py: Python, attr_name: &str) -> PyResult<Py<PyGraphArray>> {
        // Use inner Subgraph if available (preferred path)
        if let Some(ref inner_subgraph) = self.inner {
            let attr_values = inner_subgraph
                .get_node_attribute_column(&attr_name.to_string())
                .map_err(|e| {
                    PyErr::new::<PyRuntimeError, _>(format!(
                        "Failed to get attribute column: {}",
                        e
                    ))
                })?;

            // Create GraphArray from the attribute values
            let graph_array = groggy::GraphArray::from_vec(attr_values);

            // Wrap in Python GraphArray
            let py_graph_array = PyGraphArray { inner: graph_array };
            Ok(Py::new(py, py_graph_array)?)
        }
        // Fallback to legacy implementation
        else if let Some(graph_ref) = &self.graph {
            let graph = graph_ref.borrow(py);
            let mut attr_values = Vec::new();

            for &node_id in &self.nodes {
                if let Ok(Some(attr_value)) =
                    graph.inner.get_node_attr(node_id, &attr_name.to_string())
                {
                    attr_values.push(attr_value);
                } else {
                    // Handle missing attributes with default value
                    attr_values.push(groggy::AttrValue::Int(0));
                }
            }

            // Create GraphArray from the attribute values
            let graph_array = groggy::GraphArray::from_vec(attr_values);

            // Wrap in Python GraphArray
            let py_graph_array = PyGraphArray { inner: graph_array };
            Ok(Py::new(py, py_graph_array)?)
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(
                "Cannot access attributes on subgraph without graph reference.",
            ))
        }
    }

    /// Column access: get all values for an edge attribute within this subgraph
    fn get_edge_attribute_column(&self, py: Python, attr_name: &str) -> PyResult<Vec<PyObject>> {
        if let Some(graph_ref) = &self.graph {
            let graph = graph_ref.borrow(py);
            let mut values = Vec::new();

            for &edge_id in &self.edges {
                if let Ok(Some(attr_value)) =
                    graph.inner.get_edge_attr(edge_id, &attr_name.to_string())
                {
                    // Convert AttrValue to Python object
                    let py_value = attr_value_to_python_value(py, &attr_value)?;
                    values.push(py_value);
                } else {
                    // Handle missing attributes - use None
                    values.push(py.None());
                }
            }

            Ok(values)
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(
                "Cannot access edge attributes on subgraph without graph reference.",
            ))
        }
    }

    /// Python dict-like access with multi-column support
    /// - subgraph['attr_name'] -> single column (Vec<PyObject>)  
    /// - subgraph[['age', 'height']] -> multi-column 2D GraphArray of shape (2, n)
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        // Try single string first (existing behavior)
        if let Ok(attr_name) = key.extract::<String>() {
            // CRITICAL FIX: Route to edge attributes for edge subgraphs
            if self.subgraph_type == "edge_slice_selection" {
                // This is an edge subgraph - route to edge attributes
                if attr_name == "id" {
                    // Special case: edge IDs (the edges themselves)
                    let edge_ids = self
                        .edges
                        .iter()
                        .map(|&edge_id| groggy::AttrValue::Int(edge_id as i64))
                        .collect();
                    let graph_array = groggy::GraphArray::from_vec(edge_ids);
                    let py_graph_array = PyGraphArray { inner: graph_array };
                    return Ok(Py::new(py, py_graph_array)?.to_object(py));
                } else {
                    // Regular edge attributes (strength, weight, etc.)
                    let edge_values = self.get_edge_attribute_column(py, &attr_name)?;
                    // Convert Vec<PyObject> to GraphArray for consistency
                    let mut attr_values = Vec::new();
                    for py_value in edge_values {
                        // Convert Python values back to AttrValue
                        if py_value.is_none(py) {
                            attr_values.push(groggy::AttrValue::Int(0)); // Default for missing
                        } else if let Ok(int_val) = py_value.extract::<i64>(py) {
                            attr_values.push(groggy::AttrValue::Int(int_val));
                        } else if let Ok(float_val) = py_value.extract::<f64>(py) {
                            attr_values.push(groggy::AttrValue::Float(float_val as f32));
                        } else if let Ok(str_val) = py_value.extract::<String>(py) {
                            attr_values.push(groggy::AttrValue::Text(str_val));
                        } else if let Ok(bool_val) = py_value.extract::<bool>(py) {
                            attr_values.push(groggy::AttrValue::Bool(bool_val));
                        } else {
                            attr_values.push(groggy::AttrValue::Int(0)); // Fallback
                        }
                    }
                    let graph_array = groggy::GraphArray::from_vec(attr_values);
                    let py_graph_array = PyGraphArray { inner: graph_array };
                    return Ok(Py::new(py, py_graph_array)?.to_object(py));
                }
            } else {
                // This is a node subgraph - route to node attributes (original behavior)
                let column = self.get_node_attribute_column(py, &attr_name)?;
                return Ok(column.to_object(py));
            }
        }

        // Try list of strings (multi-column access)
        if let Ok(attr_names) = key.extract::<Vec<String>>() {
            if attr_names.is_empty() {
                return Err(PyValueError::new_err("Empty attribute list"));
            }

            // Collect all columns as GraphArrays
            let mut columns = Vec::new();
            let mut num_rows = 0;

            // Type checking for mixed types
            let mut column_types = Vec::new();

            for attr_name in &attr_names {
                let column = self.get_node_attribute_column(py, attr_name)?;

                // Detect column type by borrowing temporarily
                let column_type = {
                    let graph_array = column.borrow(py);

                    // Get the length and detect column type
                    num_rows = graph_array.inner.len();

                    // Sample a few values to determine the predominant type
                    if num_rows > 0 {
                        let sample_size = std::cmp::min(num_rows, 3);
                        let mut type_counts = std::collections::HashMap::new();

                        for i in 0..sample_size {
                            let type_name = match &graph_array.inner[i] {
                                groggy::AttrValue::Int(_) | groggy::AttrValue::SmallInt(_) => "int",
                                groggy::AttrValue::Float(_) => "float",
                                groggy::AttrValue::Bool(_) => "bool",
                                groggy::AttrValue::Text(_) | groggy::AttrValue::CompactText(_) => {
                                    "str"
                                }
                                _ => "mixed",
                            };
                            *type_counts.entry(type_name).or_insert(0) += 1;
                        }

                        // Get the most common type
                        type_counts
                            .into_iter()
                            .max_by_key(|(_, count)| *count)
                            .map(|(type_name, _)| type_name)
                            .unwrap_or("mixed")
                    } else {
                        "empty"
                    }
                }; // Borrow ends here

                column_types.push(column_type);
                columns.push(column);
            }

            // Check for mixed types (GraphMatrix constraint)
            if attr_names.len() > 1 {
                let first_type = column_types[0];
                let has_mixed_types = column_types
                    .iter()
                    .any(|&t| t != first_type && t != "empty");

                if has_mixed_types {
                    let detected_types: Vec<&str> = column_types.into_iter().collect();
                    return Err(PyTypeError::new_err(format!(
                        "Mixed types detected: [{}]. GraphMatrix requires homogeneous types.\n\
                        Use subgraph.nodes.table()[{:?}] for mixed-type data.",
                        detected_types.join(", "),
                        attr_names
                    )));
                }
            }

            // For single column in list form: [['age']] -> return GraphArray (same as 'age')
            if attr_names.len() == 1 {
                return Ok(columns[0].clone_ref(py).to_object(py));
            } else {
                // Multi-column access: return a list of GraphArrays
                // This allows users to work with multiple columns programmatically
                let column_objects: Vec<PyObject> =
                    columns.into_iter().map(|col| col.to_object(py)).collect();
                return Ok(column_objects.to_object(py));
            }
        }

        Err(PyTypeError::new_err(
            "Key must be a string or list of strings",
        ))
    }

    /// Create GraphTable for DataFrame-like view of this subgraph nodes
    fn table(&self, py: Python) -> PyResult<PyObject> {
        // Get the graph reference
        let graph_py = self
            .graph
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Subgraph is not attached to a graph"))?;
        let graph = graph_py.borrow(py);

        // Get all available node attributes
        let mut all_attrs = std::collections::HashSet::new();
        for &node_id in &self.nodes {
            if let Ok(attrs) = graph.inner.get_node_attrs(node_id) {
                for attr_name in attrs.keys() {
                    all_attrs.insert(attr_name.clone());
                }
            }
        }

        // Always include node_id as first column
        let mut column_names = vec!["node_id".to_string()];
        column_names.extend(all_attrs.into_iter());

        let mut columns = Vec::new();

        // Create each column
        for column_name in &column_names {
            let mut attr_values = Vec::new();

            if column_name == "node_id" {
                // Node ID column
                for &node_id in &self.nodes {
                    attr_values.push(groggy::AttrValue::Int(node_id as i64));
                }
            } else {
                // Attribute column
                for &node_id in &self.nodes {
                    if let Ok(Some(attr_value)) = graph.inner.get_node_attr(node_id, column_name) {
                        attr_values.push(attr_value);
                    } else {
                        // Default to null/empty for missing attributes
                        attr_values.push(groggy::AttrValue::Int(0));
                    }
                }
            }

            let graph_array = groggy::GraphArray::from_vec(attr_values);
            let py_array = PyGraphArray::from_graph_array(graph_array);
            columns.push(Py::new(py, py_array)?);
        }

        let py_table = PyGraphTable::new(py, columns, Some(column_names))?;
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Create GraphTable for DataFrame-like view of this subgraph edges
    fn edges_table(&self, py: Python) -> PyResult<PyObject> {
        // Get the graph reference
        let graph_py = self
            .graph
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Subgraph is not attached to a graph"))?;
        let graph = graph_py.borrow(py);

        // Get all available edge attributes
        let mut all_attrs = std::collections::HashSet::new();
        for &edge_id in &self.edges {
            if let Ok(attrs) = graph.inner.get_edge_attrs(edge_id) {
                for attr_name in attrs.keys() {
                    all_attrs.insert(attr_name.clone());
                }
            }
        }

        // Always include edge_id, source, target as first columns
        let mut column_names = vec![
            "edge_id".to_string(),
            "source".to_string(),
            "target".to_string(),
        ];
        column_names.extend(all_attrs.into_iter());

        let mut columns = Vec::new();

        // Create each column
        for column_name in &column_names {
            let mut attr_values = Vec::new();

            if column_name == "edge_id" {
                // Edge ID column
                for &edge_id in &self.edges {
                    attr_values.push(groggy::AttrValue::Int(edge_id as i64));
                }
            } else if column_name == "source" || column_name == "target" {
                // Source/Target columns
                for &edge_id in &self.edges {
                    if let Ok((source, target)) = graph.inner.edge_endpoints(edge_id) {
                        let endpoint_id = if column_name == "source" {
                            source
                        } else {
                            target
                        };
                        attr_values.push(groggy::AttrValue::Int(endpoint_id as i64));
                    } else {
                        attr_values.push(groggy::AttrValue::Int(0));
                    }
                }
            } else {
                // Attribute column
                for &edge_id in &self.edges {
                    if let Ok(Some(attr_value)) = graph.inner.get_edge_attr(edge_id, column_name) {
                        attr_values.push(attr_value);
                    } else {
                        // Default to null/empty for missing attributes
                        attr_values.push(groggy::AttrValue::Int(0));
                    }
                }
            }

            let graph_array = groggy::GraphArray::from_vec(attr_values);
            let py_array = PyGraphArray::from_graph_array(graph_array);
            columns.push(Py::new(py, py_array)?);
        }

        let py_table = PyGraphTable::new(py, columns, Some(column_names))?;
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Python-level access to parent graph (if attached).
    #[getter]
    pub fn graph(&self) -> PyResult<Option<Py<PyGraph>>> {
        Ok(self.graph.clone())
    }

    /// Fast column accessor for node attributes on PySubgraph
    pub fn _get_node_attribute_column(
        &self,
        py: Python<'_>,
        name: &str,
    ) -> PyResult<Py<PyGraphArray>> {
        if let Some(ref inner) = self.inner {
            let attr_name = groggy::AttrName::from(name.to_string());
            let arr = inner.get_node_attribute_column(&attr_name).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to get node attribute column: {}",
                    e
                ))
            })?;
            let py_graph_array = PyGraphArray {
                inner: groggy::GraphArray::from_vec(arr),
            };
            return Py::new(py, py_graph_array);
        }
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Subgraph has no inner core; attach a graph-backed subgraph",
        ))
    }

    /// Fast column accessor for edge attributes on PySubgraph  
    pub fn _get_edge_attribute_column(
        &self,
        py: Python<'_>,
        name: &str,
    ) -> PyResult<Py<PyGraphArray>> {
        if let Some(ref inner) = self.inner {
            let attr_name = groggy::AttrName::from(name.to_string());
            let arr = inner.get_edge_attribute_column(&attr_name).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to get edge attribute column: {}",
                    e
                ))
            })?;
            let py_graph_array = PyGraphArray {
                inner: groggy::GraphArray::from_vec(arr),
            };
            return Py::new(py, py_graph_array);
        }
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Subgraph has no inner core; attach a graph-backed subgraph",
        ))
    }

    /// Filter nodes *within this subgraph* using a full NodeFilter or string expression.
    /// Returns a new PySubgraph with filtered nodes and induced edges (within this subgraph).
    pub fn filter_nodes(&self, py: Python<'_>, filter: &PyAny) -> PyResult<PySubgraph> {
        // 0) Must have a parent graph to evaluate attributes efficiently
        let Some(graph_ref) = &self.graph else {
            return Err(PyRuntimeError::new_err(
                "Subgraph has no parent graph reference",
            ));
        };

        // 1) Resolve a NodeFilter:
        //    - If caller passed a NodeFilter object -> use it
        //    - Else if they passed a string -> parse via groggy.query_parser.parse_node_query
        //    - Else optional dict[str, AttributeFilter] -> translate to NodeFilter::And(AttributeFilter...)
        let node_filter = if let Ok(py_nf) = filter.extract::<PyNodeFilter>() {
            py_nf.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            let qp = py.import("groggy.query_parser")?;
            let parse = qp.getattr("parse_node_query")?;
            let parsed: PyNodeFilter = parse.call1((query_str,))?.extract()?;
            parsed.inner.clone()
        } else if let Ok(dict) = filter.downcast::<pyo3::types::PyDict>() {
            // Optional: allow dict form {"age": AttributeFilter.greater_than(21), ...}
            use groggy::core::query::NodeFilter as NF;
            use groggy::AttrName;
            let mut clauses = Vec::new();
            for (k, v) in dict.iter() {
                let key: String = k.extract()?;
                // Expect v is a PyAttributeFilter (FFI), so call .inner on it from Python first if needed.
                // If your AttributeFilter is exposed as a Python class, try to extract it directly:
                let py_attr = v.extract::<crate::ffi::core::query::PyAttributeFilter>()?;
                clauses.push(NF::AttributeFilter {
                    name: AttrName::from(key),
                    filter: py_attr.inner.clone(),
                });
            }
            groggy::core::query::NodeFilter::And(clauses)
        } else {
            return Err(PyTypeError::new_err(
                "filter must be NodeFilter | str | dict[str, AttributeFilter]",
            ));
        };

        // 2) Evaluate filter on the *current subgraph nodes* using the core API
        let mut g = graph_ref.borrow_mut(py);

        // Get all nodes that match the filter from the entire graph
        let all_filtered_nodes: Vec<groggy::NodeId> = g
            .inner
            .find_nodes(node_filter)
            .map_err(graph_error_to_py_err)?;

        // Intersect with current subgraph's nodes to get nodes that are both:
        // 1) In this subgraph, and 2) Match the filter
        let subgraph_node_set: HashSet<groggy::NodeId> = self.nodes.iter().copied().collect();
        let filtered_nodes: Vec<groggy::NodeId> = all_filtered_nodes
            .into_iter()
            .filter(|node_id| subgraph_node_set.contains(node_id))
            .collect();

        // 3) Induce edges *within this subgraph* only
        let node_set: HashSet<groggy::NodeId> = filtered_nodes.iter().copied().collect();
        let mut induced_edges = Vec::with_capacity(self.edges.len() / 2);
        for &eid in &self.edges {
            if let Ok((s, t)) = g.inner.edge_endpoints(eid) {
                if node_set.contains(&s) && node_set.contains(&t) {
                    induced_edges.push(eid);
                }
            }
        }

        // 4) Return a new subgraph; preserve graph reference for downstream ops/tables
        let mut out = PySubgraph::new(
            filtered_nodes,
            induced_edges,
            format!("{}_filtered", self.subgraph_type),
            None, // we'll set the graph next
        );
        out.set_graph_reference(graph_ref.clone());
        Ok(out)
    }

    /// Export subgraph to a new independent Graph object
    pub fn to_graph(&self, py: Python) -> PyResult<PyObject> {
        // Import the PyGraph class
        let graph_module = py.import("groggy")?;
        let graph_class = graph_module.getattr("Graph")?;

        // Create a new empty graph with the same directed property as parent
        let is_directed = if let Some(graph_ref) = &self.graph {
            let parent_graph = graph_ref.borrow(py);
            parent_graph.inner.is_directed()
        } else {
            false // Default to undirected if no parent reference
        };

        let new_graph = graph_class.call1((is_directed,))?;

        if let Some(graph_ref) = &self.graph {
            let parent_graph = graph_ref.borrow(py);

            // Add nodes with attributes
            let mut node_id_mapping = std::collections::HashMap::new();

            for &old_node_id in &self.nodes {
                // Get node attribute keys
                let attr_keys = parent_graph.node_attribute_keys(old_node_id);

                // Create Python dict for node attributes
                let py_attrs = pyo3::types::PyDict::new(py);

                // Get each attribute using the FFI method
                for attr_key in attr_keys {
                    if let Ok(Some(py_attr_value)) =
                        parent_graph.get_node_attribute(old_node_id, attr_key.clone())
                    {
                        // Convert PyAttrValue to Python object
                        let py_value = crate::ffi::utils::attr_value_to_python_value(
                            py,
                            &py_attr_value.inner,
                        )?;
                        py_attrs.set_item(attr_key, py_value)?;
                    }
                }

                // Add node to new graph - call with **kwargs pattern
                let new_node_id = if py_attrs.len() > 0 {
                    new_graph.call_method("add_node", (), Some(py_attrs))?
                } else {
                    new_graph.call_method0("add_node")?
                };
                let new_id: u64 = new_node_id.extract()?;
                node_id_mapping.insert(old_node_id, new_id);
            }

            // Add edges with attributes
            for &old_edge_id in &self.edges {
                if let Ok((source, target)) = parent_graph.inner.edge_endpoints(old_edge_id) {
                    if let (Some(&new_source), Some(&new_target)) =
                        (node_id_mapping.get(&source), node_id_mapping.get(&target))
                    {
                        // Get edge attribute keys
                        let attr_keys = parent_graph.edge_attribute_keys(old_edge_id);

                        // Create Python dict for edge attributes
                        let py_attrs = pyo3::types::PyDict::new(py);

                        // Get each attribute using the FFI method
                        for attr_key in attr_keys {
                            if let Ok(Some(py_attr_value)) =
                                parent_graph.get_edge_attribute(old_edge_id, attr_key.clone())
                            {
                                // Convert PyAttrValue to Python object
                                let py_value = crate::ffi::utils::attr_value_to_python_value(
                                    py,
                                    &py_attr_value.inner,
                                )?;
                                py_attrs.set_item(attr_key, py_value)?;
                            }
                        }

                        // Add edge to new graph
                        if py_attrs.len() > 0 {
                            new_graph.call_method(
                                "add_edge",
                                (new_source, new_target),
                                Some(py_attrs),
                            )?;
                        } else {
                            new_graph.call_method1("add_edge", (new_source, new_target))?;
                        };
                    }
                }
            }
        }

        Ok(new_graph.into())
    }

    /// Export subgraph to NetworkX graph object
    pub fn to_networkx(&self, py: Python) -> PyResult<PyObject> {
        // Import networkx
        let nx = py.import("networkx")?;

        // Determine graph type
        let is_directed = if let Some(graph_ref) = &self.graph {
            let parent_graph = graph_ref.borrow(py);
            parent_graph.inner.is_directed()
        } else {
            false // Default to undirected if no parent reference
        };

        // Create appropriate NetworkX graph
        let nx_graph = if is_directed {
            nx.call_method0("DiGraph")?
        } else {
            nx.call_method0("Graph")?
        };

        if let Some(graph_ref) = &self.graph {
            let parent_graph = graph_ref.borrow(py);

            // Add nodes with attributes
            for &node_id in &self.nodes {
                // Get node attribute keys
                let attr_keys = parent_graph.node_attribute_keys(node_id);

                // Create Python dict for node attributes
                let py_attrs = pyo3::types::PyDict::new(py);

                // Get each attribute using the FFI method
                for attr_key in attr_keys {
                    if let Ok(Some(py_attr_value)) =
                        parent_graph.get_node_attribute(node_id, attr_key.clone())
                    {
                        // Convert PyAttrValue to Python object
                        let py_value = crate::ffi::utils::attr_value_to_python_value(
                            py,
                            &py_attr_value.inner,
                        )?;
                        py_attrs.set_item(attr_key, py_value)?;
                    }
                }

                // Add node to NetworkX graph - NetworkX expects (node_id, **attrs)
                nx_graph.call_method("add_node", (node_id,), Some(py_attrs))?;
            }

            // Add edges with attributes
            for &edge_id in &self.edges {
                if let Ok((source, target)) = parent_graph.inner.edge_endpoints(edge_id) {
                    // Get edge attribute keys
                    let attr_keys = parent_graph.edge_attribute_keys(edge_id);

                    // Create Python dict for edge attributes
                    let py_attrs = pyo3::types::PyDict::new(py);

                    // Get each attribute using the FFI method
                    for attr_key in attr_keys {
                        if let Ok(Some(py_attr_value)) =
                            parent_graph.get_edge_attribute(edge_id, attr_key.clone())
                        {
                            // Convert PyAttrValue to Python object
                            let py_value = crate::ffi::utils::attr_value_to_python_value(
                                py,
                                &py_attr_value.inner,
                            )?;
                            py_attrs.set_item(attr_key, py_value)?;
                        }
                    }

                    // Add edge to NetworkX graph - NetworkX expects (source, target, **attrs)
                    nx_graph.call_method("add_edge", (source, target), Some(py_attrs))?;
                }
            }
        }

        Ok(nx_graph.to_object(py))
    }
}
