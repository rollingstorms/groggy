//! Graph Analysis Operations - Internal Helper Class
//!
//! PyGraphAnalysis helper class that handles all graph analysis operations.

use crate::ffi::core::neighborhood::PyNeighborhoodResult;
use crate::ffi::utils::graph_error_to_py_err;
use groggy::core::traits::SubgraphOperations;
use groggy::{AttrName, GraphError, NodeId};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::graph::PyGraph;

/// Internal helper for graph analysis operations (not exposed to Python)
pub struct PyGraphAnalysis {
    pub graph: Py<PyGraph>,
}

impl PyGraphAnalysis {
    /// Create new PyGraphAnalysis instance
    pub fn new(graph: Py<PyGraph>) -> PyResult<PyGraphAnalysis> {
        Ok(PyGraphAnalysis { graph })
    }
    /// Get neighbors of nodes - PURE DELEGATION to core
    pub fn neighbors(&mut self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject> {
        if let Some(nodes_input) = nodes {
            // Handle multiple nodes
            if let Ok(node_list) = nodes_input.extract::<Vec<NodeId>>() {
                let result_dict = PyDict::new(py);

                for node in node_list {
                    // DELEGATION: Use core neighbors implementation (graph.rs:866)
                    let neighbors = {
                        let graph_ref = self.graph.borrow(py);
                        let result = graph_ref
                            .inner
                            .borrow()
                            .neighbors(node)
                            .map_err(graph_error_to_py_err);
                        drop(graph_ref);
                        result
                    }?;

                    let py_neighbors = PyList::new(py, neighbors);
                    result_dict.set_item(node, py_neighbors)?;
                }

                Ok(result_dict.to_object(py))
            } else if let Ok(single_node) = nodes_input.extract::<NodeId>() {
                // Handle single node
                let neighbors = {
                    let graph_ref = self.graph.borrow(py);
                    let result = graph_ref
                        .inner
                        .borrow()
                        .neighbors(single_node)
                        .map_err(graph_error_to_py_err);
                    drop(graph_ref);
                    result
                }?;

                Ok(PyList::new(py, neighbors).to_object(py))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "nodes must be a NodeId or list of NodeIds",
                ))
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "nodes parameter is required",
            ))
        }
    }

    /// Get neighborhood sampling - PURE DELEGATION to core
    pub fn neighborhood(
        &mut self,
        py: Python,
        center_nodes: Vec<NodeId>,
        radius: Option<usize>,
        max_nodes: Option<usize>,
    ) -> PyResult<PyNeighborhoodResult> {
        let radius = radius.unwrap_or(1);
        let _max_nodes = max_nodes.unwrap_or(100);

        // DELEGATION: Choose appropriate core method based on input parameters
        let result =
            {
                match center_nodes.len() {
                    0 => Err(GraphError::EmptyGraph {
                        operation: "neighborhood: No center nodes provided".to_string(),
                    }),
                    1 => {
                        let node_id = center_nodes[0];
                        if radius == 1 {
                            // Single node, 1-hop: use single_neighborhood
                            let graph_ref = self.graph.borrow_mut(py);
                            let result = graph_ref.inner.borrow_mut().neighborhood(node_id).map(
                                |subgraph| {
                                    let size = subgraph.node_set().len();
                                    groggy::core::neighborhood::NeighborhoodResult {
                                        neighborhoods: vec![subgraph],
                                        total_neighborhoods: 1,
                                        largest_neighborhood_size: size,
                                        execution_time: std::time::Duration::from_millis(0),
                                    }
                                },
                            );
                            drop(graph_ref);
                            result
                        } else {
                            // Single node, k-hop: use k_hop_neighborhood
                            let graph_ref = self.graph.borrow_mut(py);
                            let result = graph_ref
                                .inner
                                .borrow_mut()
                                .k_hop_neighborhood(node_id, radius)
                                .map(|subgraph| {
                                    let size = subgraph.node_set().len();
                                    groggy::core::neighborhood::NeighborhoodResult {
                                        neighborhoods: vec![subgraph],
                                        total_neighborhoods: 1,
                                        largest_neighborhood_size: size,
                                        execution_time: std::time::Duration::from_millis(0),
                                    }
                                });
                            drop(graph_ref);
                            result
                        }
                    }
                    _ => {
                        if radius == 1 {
                            // Multiple nodes, 1-hop: use multi_neighborhood
                            let graph_ref = self.graph.borrow_mut(py);
                            let result = graph_ref
                                .inner
                                .borrow_mut()
                                .multi_neighborhood(&center_nodes);
                            drop(graph_ref);
                            result
                        } else {
                            // Multiple nodes, k-hop: use unified_neighborhood
                            let graph_ref = self.graph.borrow_mut(py);
                            let result = graph_ref
                                .inner
                                .borrow_mut()
                                .unified_neighborhood(&center_nodes, radius)
                                .map(|subgraph| {
                                    let size = subgraph.node_set().len();
                                    groggy::core::neighborhood::NeighborhoodResult {
                                        neighborhoods: vec![subgraph],
                                        total_neighborhoods: 1,
                                        largest_neighborhood_size: size,
                                        execution_time: std::time::Duration::from_millis(0),
                                    }
                                });
                            drop(graph_ref);
                            result
                        }
                    }
                }
                .map_err(graph_error_to_py_err)
            }?;

        Ok(PyNeighborhoodResult { inner: result })
    }

    /// Get shortest path - PURE DELEGATION to core
    pub fn shortest_path(
        &self,
        py: Python,
        source: NodeId,
        target: NodeId,
        weight_attribute: Option<AttrName>,
        _inplace: Option<bool>,
        _attr_name: Option<String>,
    ) -> PyResult<PyObject> {
        // DELEGATION: Use core shortest_path implementation with proper options
        let path = {
            let options = groggy::core::traversal::PathFindingOptions {
                weight_attribute,
                max_path_length: None,
                heuristic: None,
            };

            let graph_ref = self.graph.borrow_mut(py);
            let result = graph_ref
                .inner
                .borrow_mut()
                .shortest_path(source, target, options)
                .map_err(graph_error_to_py_err);
            drop(graph_ref);
            result
        }?;

        match path {
            Some(path) => {
                // Create a subgraph from the path nodes and edges
                use crate::ffi::core::subgraph::PySubgraph;
                use groggy::core::subgraph::Subgraph;

                let graph_ref = self.graph.borrow(py);
                let core_graph = graph_ref.inner.clone();
                drop(graph_ref);

                // Create subgraph with path nodes and edges
                let mut node_set = std::collections::HashSet::new();
                for &node_id in &path.nodes {
                    node_set.insert(node_id);
                }

                let mut edge_set = std::collections::HashSet::new();
                for &edge_id in &path.edges {
                    edge_set.insert(edge_id);
                }

                let subgraph =
                    Subgraph::new(core_graph, node_set, edge_set, "shortest_path".to_string());

                let py_subgraph = PySubgraph { inner: subgraph };
                Ok(Py::new(py, py_subgraph)?.to_object(py))
            }
            None => Ok(py.None()),
        }
    }
}
