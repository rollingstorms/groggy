//! Graph Analysis Operations - Internal Helper Class
//!
//! PyGraphAnalysis helper class that handles all graph analysis operations.

use crate::ffi::storage::array::PyBaseArray;
use crate::ffi::storage::table::PyBaseTable;
use crate::ffi::subgraphs::neighborhood::PyNeighborhoodArray;
use crate::ffi::utils::graph_error_to_py_err;
use groggy::subgraphs::{NeighborhoodResult, NeighborhoodSubgraph};
use groggy::traits::{NeighborhoodOperations, SubgraphOperations};
use groggy::{AttrName, GraphError, NodeId};
use pyo3::prelude::*;
use pyo3::types::PyList;

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
    /// Get neighbors of nodes - Enhanced to accept int, list, BaseArray, NumArray
    ///
    /// Returns BaseTable with columns (node_id, neighbor_id) for bulk operations
    /// or List[NodeId] for single node queries (backward compatibility)
    pub fn neighbors(&mut self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject> {
        if let Some(nodes_input) = nodes {
            // Try to extract as single int first (most common case)
            if let Ok(single_node) = nodes_input.extract::<NodeId>() {
                // Single node: return list for backward compatibility
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

                return Ok(PyList::new(py, neighbors).to_object(py));
            }

            // Try to extract as list of ints
            if let Ok(node_list) = nodes_input.extract::<Vec<NodeId>>() {
                // Multiple nodes: use bulk operation, return BaseTable
                let table = {
                    let graph_ref = self.graph.borrow(py);
                    let result = graph_ref
                        .inner
                        .borrow()
                        .neighbors_bulk(&node_list)
                        .map_err(graph_error_to_py_err);
                    drop(graph_ref);
                    result
                }?;

                let py_table = Py::new(py, PyBaseTable::from_table(table))?;
                return Ok(py_table.to_object(py));
            }

            // Try to extract from BaseArray
            if let Ok(base_array) = nodes_input.extract::<PyRef<PyBaseArray>>() {
                // Extract node IDs from BaseArray
                let node_ids: Vec<NodeId> = base_array
                    .inner
                    .as_slice()
                    .iter()
                    .filter_map(|v| {
                        if let groggy::types::AttrValue::Int(id) = v {
                            Some(*id as NodeId)
                        } else {
                            None
                        }
                    })
                    .collect();

                // Use bulk operation
                let table = {
                    let graph_ref = self.graph.borrow(py);
                    let result = graph_ref
                        .inner
                        .borrow()
                        .neighbors_bulk(&node_ids)
                        .map_err(graph_error_to_py_err);
                    drop(graph_ref);
                    result
                }?;

                let py_table = Py::new(py, PyBaseTable::from_table(table))?;
                return Ok(py_table.to_object(py));
            }

            // Try to extract from NumArray (via its base property)
            if let Ok(num_array) = nodes_input.getattr("base") {
                if let Ok(base_array) = num_array.extract::<PyRef<PyBaseArray>>() {
                    let node_ids: Vec<NodeId> = base_array
                        .inner
                        .as_slice()
                        .iter()
                        .filter_map(|v| {
                            if let groggy::types::AttrValue::Int(id) = v {
                                Some(*id as NodeId)
                            } else {
                                None
                            }
                        })
                        .collect();

                    let table = {
                        let graph_ref = self.graph.borrow(py);
                        let result = graph_ref
                            .inner
                            .borrow()
                            .neighbors_bulk(&node_ids)
                            .map_err(graph_error_to_py_err);
                        drop(graph_ref);
                        result
                    }?;

                    let py_table = Py::new(py, PyBaseTable::from_table(table))?;
                    return Ok(py_table.to_object(py));
                }
            }

            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "nodes must be a NodeId, list of NodeIds, BaseArray, or NumArray",
            ))
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
    ) -> PyResult<PyNeighborhoodArray> {
        let radius = radius.unwrap_or(1);
        let _max_nodes = max_nodes.unwrap_or(100);

        let graph_handle = {
            let graph = self.graph.borrow(py);
            graph.inner.clone()
        };

        // DELEGATION: Choose appropriate core method based on input parameters
        let mut result = match center_nodes.len() {
            0 => Err(GraphError::EmptyGraph {
                operation: "neighborhood: No center nodes provided".to_string(),
            }),
            1 => {
                let node_id = center_nodes[0];
                if radius == 1 {
                    let mut graph_mut = graph_handle.borrow_mut();
                    let subgraph = graph_mut.neighborhood(node_id);
                    drop(graph_mut);
                    subgraph.map(|subgraph| {
                        let size = subgraph.node_set().len();
                        NeighborhoodResult {
                            neighborhoods: vec![subgraph],
                            total_neighborhoods: 1,
                            largest_neighborhood_size: size,
                            execution_time: std::time::Duration::from_millis(0),
                        }
                    })
                } else {
                    let mut graph_mut = graph_handle.borrow_mut();
                    let subgraph = graph_mut.k_hop_neighborhood(node_id, radius);
                    drop(graph_mut);
                    subgraph.map(|subgraph| {
                        let size = subgraph.node_set().len();
                        NeighborhoodResult {
                            neighborhoods: vec![subgraph],
                            total_neighborhoods: 1,
                            largest_neighborhood_size: size,
                            execution_time: std::time::Duration::from_millis(0),
                        }
                    })
                }
            }
            _ => {
                if radius == 1 {
                    let mut graph_mut = graph_handle.borrow_mut();
                    let result = graph_mut.multi_neighborhood(&center_nodes);
                    drop(graph_mut);
                    result
                } else {
                    let mut graph_mut = graph_handle.borrow_mut();
                    let subgraph = graph_mut.unified_neighborhood(&center_nodes, radius);
                    drop(graph_mut);
                    subgraph.map(|subgraph| {
                        let size = subgraph.node_set().len();
                        NeighborhoodResult {
                            neighborhoods: vec![subgraph],
                            total_neighborhoods: 1,
                            largest_neighborhood_size: size,
                            execution_time: std::time::Duration::from_millis(0),
                        }
                    })
                }
            }
        }
        .map_err(graph_error_to_py_err)?;

        // Rehydrate neighborhoods so their graph references point at the live graph handle
        let neighborhoods = result
            .neighborhoods
            .into_iter()
            .map(|neighborhood| {
                NeighborhoodSubgraph::from_stored(
                    graph_handle.clone(),
                    neighborhood.node_set().clone(),
                    neighborhood.edge_set().clone(),
                    neighborhood.central_nodes().to_vec(),
                    neighborhood.hops(),
                )
            })
            .collect();
        result.neighborhoods = neighborhoods;

        PyNeighborhoodArray::from_result(result)
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
            let options = groggy::query::PathFindingOptions {
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
                use crate::ffi::subgraphs::subgraph::PySubgraph;
                use groggy::subgraphs::Subgraph;

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

    /// Perform breadth-first search traversal
    pub fn bfs(
        &mut self,
        py: Python,
        start: NodeId,
        max_depth: Option<usize>,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<PyObject> {
        use crate::ffi::subgraphs::subgraph::PySubgraph;
        use groggy::query::TraversalOptions;
        use groggy::subgraphs::Subgraph;

        let inplace = inplace.unwrap_or(false);
        let graph_ref = self.graph.borrow_mut(py);

        // Create traversal options
        let options = TraversalOptions {
            node_filter: None,
            edge_filter: None,
            max_depth,
            max_nodes: None,
            target_node: None,
        };

        // Perform BFS traversal using core API
        let result = graph_ref
            .inner
            .borrow_mut()
            .bfs(start, options)
            .map_err(graph_error_to_py_err)?;

        // If inplace=True, set distance/order attributes on nodes
        if inplace {
            let attr_name = attr_name.unwrap_or_else(|| "bfs_distance".to_string());

            // Use bulk attribute setting for performance
            let mut attrs_values = std::collections::HashMap::new();
            let node_value_pairs: Vec<(NodeId, groggy::AttrValue)> = result
                .nodes
                .iter()
                .enumerate()
                .map(|(order, &node_id)| (node_id, groggy::AttrValue::Int(order as i64)))
                .collect();
            attrs_values.insert(attr_name, node_value_pairs);

            graph_ref
                .inner
                .borrow_mut()
                .set_node_attrs(attrs_values)
                .map_err(graph_error_to_py_err)?;
        }

        // Create subgraph from result
        let core_graph = graph_ref.inner.clone();
        drop(graph_ref);

        let mut node_set = std::collections::HashSet::new();
        for &node_id in &result.nodes {
            node_set.insert(node_id);
        }

        let mut edge_set = std::collections::HashSet::new();
        for &edge_id in &result.edges {
            edge_set.insert(edge_id);
        }

        let subgraph = Subgraph::new(core_graph, node_set, edge_set, "bfs_traversal".to_string());
        let py_subgraph = PySubgraph { inner: subgraph };
        Ok(Py::new(py, py_subgraph)?.to_object(py))
    }

    /// Perform depth-first search traversal
    pub fn dfs(
        &mut self,
        py: Python,
        start: NodeId,
        max_depth: Option<usize>,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<PyObject> {
        use crate::ffi::subgraphs::subgraph::PySubgraph;
        use groggy::query::TraversalOptions;
        use groggy::subgraphs::Subgraph;

        let inplace = inplace.unwrap_or(false);
        let graph_ref = self.graph.borrow_mut(py);

        // Create traversal options
        let options = TraversalOptions {
            node_filter: None,
            edge_filter: None,
            max_depth,
            max_nodes: None,
            target_node: None,
        };

        // Perform DFS traversal using core API
        let result = graph_ref
            .inner
            .borrow_mut()
            .dfs(start, options)
            .map_err(graph_error_to_py_err)?;

        // If inplace=True, set distance/order attributes on nodes
        if inplace {
            let attr_name = attr_name.unwrap_or_else(|| "dfs_order".to_string());

            // Use bulk attribute setting for performance
            let mut attrs_values = std::collections::HashMap::new();
            let node_value_pairs: Vec<(NodeId, groggy::AttrValue)> = result
                .nodes
                .iter()
                .enumerate()
                .map(|(order, &node_id)| (node_id, groggy::AttrValue::Int(order as i64)))
                .collect();
            attrs_values.insert(attr_name, node_value_pairs);

            graph_ref
                .inner
                .borrow_mut()
                .set_node_attrs(attrs_values)
                .map_err(graph_error_to_py_err)?;
        }

        // Create subgraph from result
        let core_graph = graph_ref.inner.clone();
        drop(graph_ref);

        let mut node_set = std::collections::HashSet::new();
        for &node_id in &result.nodes {
            node_set.insert(node_id);
        }

        let mut edge_set = std::collections::HashSet::new();
        for &edge_id in &result.edges {
            edge_set.insert(edge_id);
        }

        let subgraph = Subgraph::new(core_graph, node_set, edge_set, "dfs_traversal".to_string());
        let py_subgraph = PySubgraph { inner: subgraph };
        Ok(Py::new(py, py_subgraph)?.to_object(py))
    }
}
