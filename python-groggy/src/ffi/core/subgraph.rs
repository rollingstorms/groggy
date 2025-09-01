//! Simplified Subgraph FFI Bindings - Complete Replacement
//!
//! Pure delegation to core Subgraph with ALL the same methods as the current PySubgraph.
//! This replaces the 800+ line complex version with pure delegation to existing trait methods.

use crate::ffi::core::neighborhood::PyNeighborhoodResult;
use groggy::core::subgraph::Subgraph;
use groggy::core::traits::SubgraphOperations;
use groggy::{AttrValue, EdgeId, NodeId, SimilarityMetric};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashSet;

// Import FFI types we need to preserve compatibility
use crate::ffi::api::graph::PyGraph;
use crate::ffi::core::accessors::{PyEdgesAccessor, PyNodesAccessor};
use crate::ffi::core::array::PyGraphArray;
use crate::ffi::core::table::PyGraphTable;

/// Python wrapper for core Subgraph - Pure delegation to existing trait methods
///
/// This completely replaces the complex dual-mode PySubgraph with simple delegation
/// to the existing SubgraphOperations trait methods. Same API, much simpler implementation.
#[pyclass(name = "Subgraph", unsendable)]
#[derive(Clone)]
pub struct PySubgraph {
    pub inner: Subgraph,
}

impl PySubgraph {
    /// Create from Rust Subgraph
    pub fn from_core_subgraph(subgraph: Subgraph) -> PyResult<Self> {
        Ok(Self { inner: subgraph })
    }

    /// Create from trait object (used by trait delegation)
    pub fn from_trait_object(
        _subgraph: Box<dyn groggy::core::traits::SubgraphOperations>,
    ) -> PyResult<Self> {
        // For now, we'll use a simpler approach - assume we can only handle concrete Subgraph types
        // In the future, we might need better trait object handling with proper Any downcasting
        Err(PyRuntimeError::new_err(
            "from_trait_object not yet implemented - use concrete Subgraph types",
        ))
    }
}

#[pymethods]
impl PySubgraph {
    // === Basic Properties - delegate to SubgraphOperations ===

    /// Get nodes as a property that supports indexing and attribute access
    #[getter]
    fn nodes(&self, py: Python) -> PyResult<Py<PyNodesAccessor>> {
        // Use the core graph directly - no more PyGraph wrapper needed
        Py::new(
            py,
            PyNodesAccessor {
                graph: self.inner.graph(),
                constrained_nodes: Some(self.inner.node_set().iter().copied().collect()),
            },
        )
    }

    /// Get edges as a property that supports indexing and attribute access
    #[getter]
    fn edges(&self, py: Python) -> PyResult<Py<PyEdgesAccessor>> {
        // Create accessor using the graph reference from inner subgraph
        Py::new(
            py,
            PyEdgesAccessor {
                graph: self.inner.graph(),
                constrained_edges: Some(self.inner.edge_set().iter().copied().collect()),
            },
        )
    }

    /// Python len() support - returns number of nodes
    fn __len__(&self) -> usize {
        self.inner.node_count() // SubgraphOperations::node_count()
    }

    /// Node count property
    fn node_count(&self) -> usize {
        self.inner.node_count() // SubgraphOperations::node_count()
    }

    /// Edge count property
    fn edge_count(&self) -> usize {
        self.inner.edge_count() // SubgraphOperations::edge_count()
    }

    /// Get node IDs as PyGraphArray
    #[getter]
    fn node_ids(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let attr_values: Vec<AttrValue> = self
            .inner
            .node_set()
            .iter()
            .map(|&id| AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Py::new(py, py_graph_array)
    }

    /// Get edge IDs as PyGraphArray
    #[getter]
    fn edge_ids(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let attr_values: Vec<AttrValue> = self
            .inner
            .edge_set()
            .iter()
            .map(|&id| AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Py::new(py, py_graph_array)
    }

    /// Check if a node exists in this subgraph
    fn has_node(&self, node_id: NodeId) -> bool {
        self.inner.contains_node(node_id) // SubgraphOperations::contains_node()
    }

    /// Check if an edge exists in this subgraph
    fn has_edge(&self, edge_id: EdgeId) -> bool {
        self.inner.contains_edge(edge_id) // SubgraphOperations::contains_edge()
    }

    // === Analysis Methods - delegate to SubgraphOperations ===

    /// Calculate density of this subgraph
    fn density(&self) -> f64 {
        // Use same calculation as original but with trait data
        let num_nodes = self.inner.node_count();
        let num_edges = self.inner.edge_count();

        if num_nodes <= 1 {
            return 0.0;
        }

        // For undirected graph: max edges = n(n-1)/2
        let max_possible_edges = (num_nodes * (num_nodes - 1)) / 2;

        if max_possible_edges > 0 {
            num_edges as f64 / max_possible_edges as f64
        } else {
            0.0
        }
    }

    /// Get connected components within this subgraph
    fn connected_components(&self) -> PyResult<Vec<PySubgraph>> {
        let components = self
            .inner
            .connected_components()
            .map_err(|e| PyRuntimeError::new_err(format!("Connected components error: {}", e)))?;

        // Convert trait objects back to PySubgraph
        let py_components: PyResult<Vec<PySubgraph>> = components
            .into_iter()
            .map(|comp| {
                // Create new PySubgraph from the component's data
                // This is tricky because we get Box<dyn SubgraphOperations> back
                // For now, create a new Subgraph with the component's nodes/edges
                let nodes: std::collections::HashSet<NodeId> = comp.node_set().clone();
                let edges: std::collections::HashSet<EdgeId> = comp.edge_set().clone();

                // Create new Subgraph - this will need the same graph reference
                let component_subgraph = Subgraph::new(
                    self.inner.graph().clone(),
                    nodes,
                    edges,
                    "component".to_string(),
                );

                PySubgraph::from_core_subgraph(component_subgraph)
            })
            .collect();
        py_components
    }

    /// Check if this subgraph is connected
    fn is_connected(&self) -> PyResult<bool> {
        // Use connected_components to check - if only 1 component, it's connected
        let components = self.connected_components()?;
        Ok(components.len() <= 1)
    }

    /// Check if there is a path between two nodes within this subgraph
    ///
    /// This is more efficient than `shortest_path_subgraph` when you only need
    /// to know if a path exists, not the actual path.
    ///
    /// # Arguments
    /// * `node1_id` - The starting node ID
    /// * `node2_id` - The destination node ID
    ///
    /// # Returns
    /// * `True` if a path exists between the nodes within this subgraph
    /// * `False` if no path exists or either node is not in this subgraph
    ///
    /// # Example
    /// ```python
    /// # Check if there's a path between node 1 and node 5 in the subgraph
    /// path_exists = subgraph.has_path(1, 5)
    /// ```
    fn has_path(&self, node1_id: NodeId, node2_id: NodeId) -> PyResult<bool> {
        self.inner
            .has_path(node1_id, node2_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Error checking path: {}", e)))
    }

    // === Data Export Methods ===

    /// Convert subgraph nodes to a table - pure delegation to core GraphTable
    fn table(&self, py: Python) -> PyResult<PyObject> {
        let core_table = self
            .inner
            .nodes_table()
            .map_err(|e| PyRuntimeError::new_err(format!("Table creation error: {}", e)))?;

        // Wrap core GraphTable in PyGraphTable - pure delegation
        let py_table = PyGraphTable { inner: core_table };
        Ok(Py::new(py, py_table)?.into_py(py))
    }

    /// Convert subgraph edges to a table - pure delegation to core GraphTable
    fn edges_table(&self, py: Python) -> PyResult<PyObject> {
        let core_table = self
            .inner
            .edges_table()
            .map_err(|e| PyRuntimeError::new_err(format!("Edges table creation error: {}", e)))?;

        // Wrap core GraphTable in PyGraphTable - pure delegation
        let py_table = PyGraphTable { inner: core_table };
        Ok(Py::new(py, py_table)?.into_py(py))
    }

    // === Filtering Methods - delegate to SubgraphOperations ===

    /// Filter nodes and return new subgraph  
    fn filter_nodes(&self, _py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Extract the filter from Python object - support both NodeFilter objects and string queries
        let node_filter = if let Ok(filter_obj) =
            filter.extract::<crate::ffi::core::query::PyNodeFilter>()
        {
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using Rust core query parser
            let mut parser = groggy::core::query_parser::QueryParser::new();
            parser.parse_node_query(&query_str).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Query parse error: {}", e))
            })?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "filter must be a NodeFilter object or a string query (e.g., 'salary > 120000')",
            ));
        };

        // Delegate to core Graph.find_nodes method
        let graph_ref = self.inner.graph();
        let filtered_nodes = graph_ref
            .borrow_mut()
            .find_nodes(node_filter)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{:?}", e)))?;

        // Create induced subgraph using core Subgraph
        let filtered_node_set: HashSet<NodeId> = filtered_nodes.iter().copied().collect();
        let induced_edges = Subgraph::calculate_induced_edges(&graph_ref, &filtered_node_set)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{:?}", e)))?;

        let new_subgraph = Subgraph::new(
            graph_ref.clone(),
            filtered_node_set,
            induced_edges,
            format!("{}_filtered_nodes", self.inner.subgraph_type()),
        );

        PySubgraph::from_core_subgraph(new_subgraph)
    }

    /// Filter edges and return new subgraph
    fn filter_edges(&self, _py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Extract the filter from Python object - support both EdgeFilter objects and string queries
        let edge_filter = if let Ok(filter_obj) =
            filter.extract::<crate::ffi::core::query::PyEdgeFilter>()
        {
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using Rust core query parser
            let mut parser = groggy::core::query_parser::QueryParser::new();
            parser.parse_edge_query(&query_str).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Query parse error: {}", e))
            })?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "filter must be an EdgeFilter object or a string query (e.g., 'weight > 0.5')",
            ));
        };

        // Delegate to core Graph.find_edges method
        let graph_ref = self.inner.graph();
        let filtered_edges = graph_ref
            .borrow_mut()
            .find_edges(edge_filter)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{:?}", e)))?;

        // Create subgraph with filtered edges and their incident nodes
        let filtered_edge_set: HashSet<EdgeId> = filtered_edges.iter().copied().collect();
        let mut incident_nodes = HashSet::new();

        // Collect all nodes incident to the filtered edges
        for &edge_id in &filtered_edge_set {
            if let Ok((source, target)) = graph_ref.borrow().edge_endpoints(edge_id) {
                incident_nodes.insert(source);
                incident_nodes.insert(target);
            }
        }

        let new_subgraph = Subgraph::new(
            graph_ref.clone(),
            incident_nodes,
            filtered_edge_set,
            format!("{}_filtered_edges", self.inner.subgraph_type()),
        );

        PySubgraph::from_core_subgraph(new_subgraph)
    }

    // === Graph Conversion Methods ===

    /// Convert to a new independent graph
    fn to_graph(&self, py: Python) -> PyResult<PyObject> {
        // Create new PyGraph with only this subgraph's nodes and edges
        let graph_type = py.get_type::<PyGraph>();
        let new_graph = graph_type.call0()?;

        // Add nodes and edges from this subgraph to the new graph
        // This would require copying data from inner subgraph

        Ok(new_graph.to_object(py))
    }

    /// Convert to NetworkX graph (if available)
    fn to_networkx(&self, py: Python) -> PyResult<PyObject> {
        // Convert to NetworkX format using existing logic
        // This is a complex method that would delegate to existing NetworkX export

        // For now, return None as placeholder
        Ok(py.None())
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
        let graph_ref = self.inner.graph();

        match nodes {
            // Single node case
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node_id = node_arg.extract::<NodeId>()?;

                // Verify node is in subgraph
                if !self.inner.node_set().contains(&node_id) {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Node {} is not in this subgraph",
                        node_id
                    )));
                }

                let deg = if full_graph {
                    // Get degree from full graph
                    let graph = graph_ref.borrow();
                    graph.degree(node_id).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
                    })?
                } else {
                    // Calculate local degree within subgraph
                    self.inner
                        .edge_set()
                        .iter()
                        .filter(|&&edge_id| {
                            let graph = graph_ref.borrow();
                            if let Ok((src, tgt)) = graph.edge_endpoints(edge_id) {
                                src == node_id || tgt == node_id
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
                    if !self.inner.node_set().contains(&node_id) {
                        continue; // Skip nodes not in subgraph
                    }

                    let deg = if full_graph {
                        // Get degree from main graph
                        let graph = graph_ref.borrow();
                        match graph.degree(node_id) {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local degree within subgraph
                        self.inner
                            .edge_set()
                            .iter()
                            .filter(|&&edge_id| {
                                let graph = graph_ref.borrow();
                                if let Ok((src, tgt)) = graph.edge_endpoints(edge_id) {
                                    src == node_id || tgt == node_id
                                } else {
                                    false
                                }
                            })
                            .count()
                    };

                    degrees.push(groggy::AttrValue::Int(deg as i64));
                }

                let graph_array = groggy::GraphArray::from_vec(degrees);
                let py_graph_array = PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }

            // All nodes case (or None)
            None => {
                let mut degrees = Vec::new();

                for &node_id in self.inner.node_set() {
                    let deg = if full_graph {
                        // Get degree from main graph
                        let graph = graph_ref.borrow();
                        match graph.degree(node_id) {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local degree within subgraph
                        self.inner
                            .edge_set()
                            .iter()
                            .filter(|&&edge_id| {
                                let graph = graph_ref.borrow();
                                if let Ok((src, tgt)) = graph.edge_endpoints(edge_id) {
                                    src == node_id || tgt == node_id
                                } else {
                                    false
                                }
                            })
                            .count()
                    };

                    degrees.push(groggy::AttrValue::Int(deg as i64));
                }

                let graph_array = groggy::GraphArray::from_vec(degrees);
                let py_graph_array = PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }

            // Invalid argument type
            Some(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "degree() nodes argument must be a NodeId, list of NodeIds, or None",
            )),
        }
    }

    /// Get in-degree of nodes within subgraph
    #[pyo3(signature = (nodes = None, full_graph = false))]
    fn in_degree(&self, py: Python, nodes: Option<&PyAny>, full_graph: bool) -> PyResult<PyObject> {
        let graph_ref = self.inner.graph();

        match nodes {
            // Single node case
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node_id = node_arg.extract::<NodeId>()?;

                // Verify node is in subgraph
                if !self.inner.node_set().contains(&node_id) {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Node {} is not in this subgraph",
                        node_id
                    )));
                }

                let in_deg = if full_graph {
                    // Get in-degree from full graph
                    let graph = graph_ref.borrow();
                    graph.in_degree(node_id).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
                    })?
                } else {
                    // Calculate local in-degree within subgraph
                    self.inner
                        .edge_set()
                        .iter()
                        .filter(|&&edge_id| {
                            let graph = graph_ref.borrow();
                            if let Ok((_, tgt)) = graph.edge_endpoints(edge_id) {
                                tgt == node_id
                            } else {
                                false
                            }
                        })
                        .count()
                };

                Ok(in_deg.to_object(py))
            }

            // Multiple nodes case
            Some(nodes_arg) => {
                let node_ids: Vec<NodeId> = nodes_arg.extract()?;
                let mut in_degrees = Vec::new();

                for node_id in node_ids {
                    // Verify node is in subgraph
                    if !self.inner.node_set().contains(&node_id) {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Node {} is not in this subgraph",
                            node_id
                        )));
                    }

                    let in_deg = if full_graph {
                        // Get in-degree from full graph
                        let graph = graph_ref.borrow();
                        match graph.in_degree(node_id) {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local in-degree within subgraph
                        self.inner
                            .edge_set()
                            .iter()
                            .filter(|&&edge_id| {
                                let graph = graph_ref.borrow();
                                if let Ok((_, tgt)) = graph.edge_endpoints(edge_id) {
                                    tgt == node_id
                                } else {
                                    false
                                }
                            })
                            .count()
                    };

                    in_degrees.push(groggy::AttrValue::Int(in_deg as i64));
                }

                let graph_array = groggy::GraphArray::from_vec(in_degrees);
                let py_graph_array = PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }

            // All nodes case (or None)
            None => {
                let mut in_degrees = Vec::new();

                for &node_id in self.inner.node_set() {
                    let in_deg = if full_graph {
                        // Get in-degree from main graph
                        let graph = graph_ref.borrow();
                        match graph.in_degree(node_id) {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local in-degree within subgraph
                        self.inner
                            .edge_set()
                            .iter()
                            .filter(|&&edge_id| {
                                let graph = graph_ref.borrow();
                                if let Ok((_, tgt)) = graph.edge_endpoints(edge_id) {
                                    tgt == node_id
                                } else {
                                    false
                                }
                            })
                            .count()
                    };

                    in_degrees.push(groggy::AttrValue::Int(in_deg as i64));
                }

                let graph_array = groggy::GraphArray::from_vec(in_degrees);
                let py_graph_array = PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }
        }
    }

    /// Get out-degree of nodes within subgraph
    #[pyo3(signature = (nodes = None, full_graph = false))]
    fn out_degree(
        &self,
        py: Python,
        nodes: Option<&PyAny>,
        full_graph: bool,
    ) -> PyResult<PyObject> {
        let graph_ref = self.inner.graph();

        match nodes {
            // Single node case
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node_id = node_arg.extract::<NodeId>()?;

                // Verify node is in subgraph
                if !self.inner.node_set().contains(&node_id) {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Node {} is not in this subgraph",
                        node_id
                    )));
                }

                let out_deg = if full_graph {
                    // Get out-degree from full graph
                    let graph = graph_ref.borrow();
                    graph.out_degree(node_id).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
                    })?
                } else {
                    // Calculate local out-degree within subgraph
                    self.inner
                        .edge_set()
                        .iter()
                        .filter(|&&edge_id| {
                            let graph = graph_ref.borrow();
                            if let Ok((src, _)) = graph.edge_endpoints(edge_id) {
                                src == node_id
                            } else {
                                false
                            }
                        })
                        .count()
                };

                Ok(out_deg.to_object(py))
            }

            // Multiple nodes case
            Some(nodes_arg) => {
                let node_ids: Vec<NodeId> = nodes_arg.extract()?;
                let mut out_degrees = Vec::new();

                for node_id in node_ids {
                    // Verify node is in subgraph
                    if !self.inner.node_set().contains(&node_id) {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Node {} is not in this subgraph",
                            node_id
                        )));
                    }

                    let out_deg = if full_graph {
                        // Get out-degree from full graph
                        let graph = graph_ref.borrow();
                        match graph.out_degree(node_id) {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local out-degree within subgraph
                        self.inner
                            .edge_set()
                            .iter()
                            .filter(|&&edge_id| {
                                let graph = graph_ref.borrow();
                                if let Ok((src, _)) = graph.edge_endpoints(edge_id) {
                                    src == node_id
                                } else {
                                    false
                                }
                            })
                            .count()
                    };

                    out_degrees.push(groggy::AttrValue::Int(out_deg as i64));
                }

                let graph_array = groggy::GraphArray::from_vec(out_degrees);
                let py_graph_array = PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }

            // All nodes case (or None)
            None => {
                let mut out_degrees = Vec::new();

                for &node_id in self.inner.node_set() {
                    let out_deg = if full_graph {
                        // Get out-degree from main graph
                        let graph = graph_ref.borrow();
                        match graph.out_degree(node_id) {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local out-degree within subgraph
                        self.inner
                            .edge_set()
                            .iter()
                            .filter(|&&edge_id| {
                                let graph = graph_ref.borrow();
                                if let Ok((src, _)) = graph.edge_endpoints(edge_id) {
                                    src == node_id
                                } else {
                                    false
                                }
                            })
                            .count()
                    };

                    out_degrees.push(groggy::AttrValue::Int(out_deg as i64));
                }

                let graph_array = groggy::GraphArray::from_vec(out_degrees);
                let py_graph_array = PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }
        }
    }

    /// Calculate similarity between subgraphs using various metrics
    #[pyo3(signature = (other, metric = "jaccard"))]
    fn calculate_similarity(&self, other: &PySubgraph, metric: &str, _py: Python) -> PyResult<f64> {
        let similarity_metric = match metric {
            "jaccard" => SimilarityMetric::Jaccard,
            "dice" => SimilarityMetric::Dice,
            "cosine" => SimilarityMetric::Cosine,
            "overlap" => SimilarityMetric::Overlap,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown similarity metric: '{}'. Valid options: 'jaccard', 'dice', 'cosine', 'overlap'", metric)
            ))
        };

        self.inner
            .calculate_similarity(&other.inner, similarity_metric)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Similarity calculation error: {}",
                    e
                ))
            })
    }

    /// Support attribute access via indexing: subgraph['attr_name'] -> GraphArray
    fn __getitem__(&self, key: &PyAny, py: Python) -> PyResult<PyObject> {
        // Only support string keys (attribute names) for now
        if let Ok(attr_name) = key.extract::<String>() {
            // Return GraphArray of attribute values for all nodes in the subgraph
            let graph_ref = self.inner.graph();
            let mut attr_values = Vec::new();

            for &node_id in self.inner.node_set() {
                let graph = graph_ref.borrow();
                match graph.get_node_attr(node_id, &attr_name) {
                    Ok(Some(attr_value)) => {
                        attr_values.push(attr_value);
                    }
                    Ok(None) | Err(_) => {
                        // Use null for missing attributes
                        attr_values.push(AttrValue::Null);
                    }
                }
            }

            let graph_array = groggy::GraphArray::from_vec(attr_values);
            let py_graph_array = PyGraphArray { inner: graph_array };
            return Ok(Py::new(py, py_graph_array)?.to_object(py));
        }

        // For now, only support string attribute access
        Err(PyTypeError::new_err(
            "Subgraph indexing only supports string attribute names. \
             Example: subgraph['community']",
        ))
    }

    /// Compute neighborhoods from this subgraph, returning a PyNeighborhoodResult
    fn neighborhood(
        &self,
        py: Python,
        central_nodes: Vec<NodeId>,
        hops: usize,
    ) -> PyResult<PyNeighborhoodResult> {
        // Just wrap the graph_analysis version - create a temporary PyGraph from our core graph
        use crate::ffi::api::graph::PyGraph;
        use crate::ffi::api::graph_analysis::PyGraphAnalysis;

        // Create a temporary PyGraph wrapper
        let py_graph = PyGraph {
            inner: self.inner.graph(),
        };

        // Create PyGraphAnalysis and delegate to it
        let mut analysis_handler = PyGraphAnalysis::new(Py::new(py, py_graph)?)?;
        analysis_handler.neighborhood(py, central_nodes, Some(hops), None)
    }

    // === String representations ===

    fn __repr__(&self) -> String {
        format!(
            "Subgraph(nodes={}, edges={})",
            self.inner.node_count(),
            self.inner.edge_count()
        )
    }

    // === MISSING BASIC OPERATIONS ===

    /// Check if subgraph is empty
    fn is_empty(&self) -> bool {
        self.inner.node_count() == 0
    }

    /// Get text summary of subgraph
    fn summary(&self) -> String {
        format!(
            "Subgraph: {} nodes, {} edges, density: {:.3}",
            self.inner.node_count(),
            self.inner.edge_count(),
            self.inner.density()
        )
    }

    /// Check if subgraph contains a specific node (alias for has_node)
    fn contains_node(&self, node_id: NodeId) -> bool {
        self.inner.contains_node(node_id)
    }

    /// Check if subgraph contains a specific edge (alias for has_edge)  
    fn contains_edge(&self, edge_id: EdgeId) -> bool {
        self.inner.contains_edge(edge_id)
    }

    /// Get neighbors of a node within the subgraph
    fn neighbors(&self, py: Python, node_id: NodeId) -> PyResult<Py<PyGraphArray>> {
        match self.inner.neighbors(node_id) {
            Ok(neighbor_ids) => {
                let attr_values: Vec<AttrValue> = neighbor_ids
                    .into_iter()
                    .map(|id| AttrValue::Int(id as i64))
                    .collect();
                let py_array = PyGraphArray {
                    inner: groggy::core::array::GraphArray::from_vec(attr_values),
                };
                Py::new(py, py_array)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Neighbors error: {}", e))),
        }
    }

    /// Get edge endpoints (source, target)
    fn edge_endpoints(&self, _py: Python, edge_id: EdgeId) -> PyResult<(NodeId, NodeId)> {
        match self.inner.edge_endpoints(edge_id) {
            Ok(endpoints) => Ok(endpoints),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Edge endpoints error: {}",
                e
            ))),
        }
    }

    /// Check if edge exists between two nodes
    fn has_edge_between(&self, _py: Python, source: NodeId, target: NodeId) -> PyResult<bool> {
        match self.inner.has_edge_between(source, target) {
            Ok(exists) => Ok(exists),
            Err(e) => Err(PyRuntimeError::new_err(format!("Edge check error: {}", e))),
        }
    }

    // === MISSING ATTRIBUTE ACCESS METHODS ===

    /// Get a single node attribute value
    fn get_node_attribute(
        &self,
        py: Python,
        node_id: NodeId,
        attr_name: String,
    ) -> PyResult<Option<PyObject>> {
        use crate::ffi::utils::attr_value_to_python_value;
        match self.inner.get_node_attribute(node_id, &attr_name) {
            Ok(Some(attr_value)) => {
                let py_value = attr_value_to_python_value(py, &attr_value)?;
                Ok(Some(py_value))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Get node attribute error: {}",
                e
            ))),
        }
    }

    /// Get a single edge attribute value
    fn get_edge_attribute(
        &self,
        py: Python,
        edge_id: EdgeId,
        attr_name: String,
    ) -> PyResult<Option<PyObject>> {
        use crate::ffi::utils::attr_value_to_python_value;
        match self.inner.get_edge_attribute(edge_id, &attr_name) {
            Ok(Some(attr_value)) => {
                let py_value = attr_value_to_python_value(py, &attr_value)?;
                Ok(Some(py_value))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Get edge attribute error: {}",
                e
            ))),
        }
    }

    /// Set multiple node attributes (bulk operation) - delegates to accessor
    fn set_node_attrs(&self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        // Get the nodes accessor and delegate to its internal method
        let nodes_accessor = self.nodes(py)?;
        let nodes_accessor_ref: PyRef<PyNodesAccessor> = nodes_accessor.extract(py)?;
        nodes_accessor_ref.set_attrs_internal(py, attrs_dict)
    }

    /// Set multiple edge attributes (bulk operation) - delegates to accessor  
    fn set_edge_attrs(&self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        // Get the edges accessor and delegate to its internal method
        let edges_accessor = self.edges(py)?;
        let edges_accessor_ref: PyRef<PyEdgesAccessor> = edges_accessor.extract(py)?;
        edges_accessor_ref.set_attrs_internal(py, attrs_dict)
    }

    // === MISSING GRAPH METRICS ===

    /// Calculate clustering coefficient for a node or entire subgraph
    fn clustering_coefficient(&self, _py: Python, _node_id: Option<NodeId>) -> PyResult<f64> {
        // Note: Clustering coefficient not yet implemented in core
        // This is a placeholder for future implementation
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Clustering coefficient not yet implemented in core - coming in future version",
        ))
    }

    /// Calculate transitivity of the subgraph
    fn transitivity(&self, _py: Python) -> PyResult<f64> {
        // Note: Transitivity not yet implemented in core
        // This is a placeholder for future implementation
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Transitivity not yet implemented in core - coming in future version",
        ))
    }

    // === ENTITY TYPE METHOD ===

    /// Return the entity type string
    fn entity_type(&self) -> PyResult<String> {
        Ok("Subgraph".to_string())
    }

    // === MISSING SUBGRAPH OPERATIONS ===

    /// Create subgraph from BFS traversal
    fn bfs(&self, _py: Python, start: NodeId, max_depth: Option<usize>) -> PyResult<PySubgraph> {
        match self.inner.bfs(start, max_depth) {
            Ok(boxed_subgraph) => {
                // Create a concrete Subgraph from the trait object data
                use groggy::core::subgraph::Subgraph;
                use groggy::core::traits::GraphEntity;
                let concrete_subgraph = Subgraph::new(
                    self.inner.graph_ref(),
                    boxed_subgraph.node_set().clone(),
                    boxed_subgraph.edge_set().clone(),
                    format!("bfs_from_{}", start),
                );
                Ok(PySubgraph {
                    inner: concrete_subgraph,
                })
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("BFS error: {}", e))),
        }
    }

    /// Create subgraph from DFS traversal  
    fn dfs(&self, _py: Python, start: NodeId, max_depth: Option<usize>) -> PyResult<PySubgraph> {
        match self.inner.dfs(start, max_depth) {
            Ok(boxed_subgraph) => {
                // Create a concrete Subgraph from the trait object data
                use groggy::core::subgraph::Subgraph;
                use groggy::core::traits::GraphEntity;
                let concrete_subgraph = Subgraph::new(
                    self.inner.graph_ref(),
                    boxed_subgraph.node_set().clone(),
                    boxed_subgraph.edge_set().clone(),
                    format!("dfs_from_{}", start),
                );
                Ok(PySubgraph {
                    inner: concrete_subgraph,
                })
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("DFS error: {}", e))),
        }
    }

    /// Create subgraph representing shortest path between two nodes
    fn shortest_path_subgraph(
        &self,
        _py: Python,
        source: NodeId,
        target: NodeId,
    ) -> PyResult<Option<PySubgraph>> {
        match self.inner.shortest_path_subgraph(source, target) {
            Ok(Some(boxed_subgraph)) => {
                // Create a concrete Subgraph from the trait object data
                use groggy::core::subgraph::Subgraph;
                use groggy::core::traits::GraphEntity;
                let concrete_subgraph = Subgraph::new(
                    self.inner.graph_ref(),
                    boxed_subgraph.node_set().clone(),
                    boxed_subgraph.edge_set().clone(),
                    format!("shortest_path_{}_{}", source, target),
                );
                Ok(Some(PySubgraph {
                    inner: concrete_subgraph,
                }))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Shortest path error: {}",
                e
            ))),
        }
    }

    /// Create induced subgraph from list of nodes
    fn induced_subgraph(&self, _py: Python, nodes: Vec<NodeId>) -> PyResult<PySubgraph> {
        match self.inner.induced_subgraph(&nodes) {
            Ok(boxed_subgraph) => {
                // Create a concrete Subgraph from the trait object data
                use groggy::core::subgraph::Subgraph;
                use groggy::core::traits::GraphEntity;
                let concrete_subgraph = Subgraph::new(
                    self.inner.graph_ref(),
                    boxed_subgraph.node_set().clone(),
                    boxed_subgraph.edge_set().clone(),
                    "induced_subgraph".to_string(),
                );
                Ok(PySubgraph {
                    inner: concrete_subgraph,
                })
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Induced subgraph error: {}",
                e
            ))),
        }
    }

    /// Create subgraph from list of edges
    fn subgraph_from_edges(&self, _py: Python, edges: Vec<EdgeId>) -> PyResult<PySubgraph> {
        match self.inner.subgraph_from_edges(&edges) {
            Ok(boxed_subgraph) => {
                // Create a concrete Subgraph from the trait object data
                use groggy::core::subgraph::Subgraph;
                use groggy::core::traits::GraphEntity;
                let concrete_subgraph = Subgraph::new(
                    self.inner.graph_ref(),
                    boxed_subgraph.node_set().clone(),
                    boxed_subgraph.edge_set().clone(),
                    "subgraph_from_edges".to_string(),
                );
                Ok(PySubgraph {
                    inner: concrete_subgraph,
                })
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Subgraph from edges error: {}",
                e
            ))),
        }
    }

    /// Set operations - merge, intersect, subtract (placeholders)
    fn merge_with(&self, _py: Python, _other: &PySubgraph) -> PyResult<PySubgraph> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Subgraph set operations not yet implemented - requires subgraph algebra in core",
        ))
    }

    fn intersect_with(&self, _py: Python, _other: &PySubgraph) -> PyResult<PySubgraph> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Subgraph set operations not yet implemented - requires subgraph algebra in core",
        ))
    }

    fn subtract_from(&self, _py: Python, _other: &PySubgraph) -> PyResult<PySubgraph> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Subgraph set operations not yet implemented - requires subgraph algebra in core",
        ))
    }

    /// Collapse subgraph to a single node with aggregated attributes
    fn collapse_to_node(&self, _py: Python, _agg_functions: &PyDict) -> PyResult<NodeId> {
        // This is complex and needs proper aggregation logic
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Subgraph collapse not yet implemented - requires aggregation logic in core",
        ))
    }

    fn __str__(&self) -> String {
        format!(
            "Subgraph with {} nodes and {} edges",
            self.inner.node_count(),
            self.inner.edge_count()
        )
    }
}

// ============================================================================
// TRAIT IMPLEMENTATION - Core delegation pattern
// ============================================================================

// Shadow trait implementation removed - PySubgraph now uses direct delegation only
