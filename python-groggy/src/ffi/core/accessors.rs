//! Accessors FFI Bindings
//!
//! Python bindings for smart indexing accessors.

use groggy::{EdgeId, NodeId};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PySlice;

// Import types from our FFI modules
use crate::ffi::api::graph::PyGraph;
use crate::ffi::core::subgraph::PySubgraph;

/// Iterator for nodes that yields NodeViews
#[pyclass]
pub struct NodesIterator {
    graph: Py<PyGraph>,
    node_ids: Vec<NodeId>,
    index: usize,
}

#[pymethods]
impl NodesIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        if self.index < self.node_ids.len() {
            let node_id = self.node_ids[self.index];
            self.index += 1;

            // Create NodeView for this node
            let node_view = PyGraph::create_node_view_internal(self.graph.clone(), py, node_id)?;
            Ok(Some(node_view.to_object(py)))
        } else {
            Ok(None)
        }
    }
}

/// Wrapper for g.nodes that supports indexing syntax: g.nodes[id] -> NodeView
#[pyclass(name = "NodesAccessor")]
pub struct PyNodesAccessor {
    pub graph: Py<PyGraph>,
    /// Optional constraint: if Some, only these nodes are accessible
    pub constrained_nodes: Option<Vec<NodeId>>,
}

#[pymethods]
impl PyNodesAccessor {
    /// Support node access: g.nodes[0] -> NodeView, g.nodes[[0,1,2]] -> Subgraph, g.nodes[0:5] -> Subgraph
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        // Try to extract as single integer
        if let Ok(index_or_id) = key.extract::<NodeId>() {
            let actual_node_id = if let Some(ref constrained) = self.constrained_nodes {
                // Constrained case: treat as index into constrained list
                if (index_or_id as usize) >= constrained.len() {
                    return Err(PyIndexError::new_err(format!(
                        "Node index {} out of range (0-{})",
                        index_or_id,
                        constrained.len() - 1
                    )));
                }
                constrained[index_or_id as usize]
            } else {
                // Unconstrained case: treat as actual node ID (existing behavior)
                index_or_id
            };

            // Single node access - return NodeView
            let graph = self.graph.borrow(py);
            if !graph.has_node_internal(actual_node_id) {
                return Err(PyKeyError::new_err(format!(
                    "Node {} does not exist",
                    actual_node_id
                )));
            }

            let node_view =
                PyGraph::create_node_view_internal(self.graph.clone(), py, actual_node_id)?;
            return Ok(node_view.to_object(py));
        }

        // Try to extract as boolean array/list (boolean indexing) - CHECK FIRST before integers
        if let Ok(boolean_mask) = key.extract::<Vec<bool>>() {
            let graph = self.graph.borrow(py);
            let all_node_ids = if let Some(ref constrained) = self.constrained_nodes {
                constrained.clone()
            } else {
                graph.inner.node_ids()
            };

            // Check if boolean mask length matches node count
            if boolean_mask.len() != all_node_ids.len() {
                return Err(PyIndexError::new_err(format!(
                    "Boolean mask length ({}) must match number of nodes ({})",
                    boolean_mask.len(),
                    all_node_ids.len()
                )));
            }

            // Select nodes where boolean mask is True
            let selected_nodes: Vec<NodeId> = all_node_ids
                .iter()
                .zip(boolean_mask.iter())
                .filter_map(|(&node_id, &include)| if include { Some(node_id) } else { None })
                .collect();

            if selected_nodes.is_empty() {
                return Err(PyIndexError::new_err("Boolean mask selected no nodes"));
            }

            // Validate all selected nodes exist
            for &node_id in &selected_nodes {
                if !graph.has_node_internal(node_id) {
                    return Err(PyKeyError::new_err(format!(
                        "Node {} does not exist",
                        node_id
                    )));
                }
            }

            // Get induced edges for selected nodes
            let node_set: std::collections::HashSet<NodeId> =
                selected_nodes.iter().copied().collect();
            let (edge_ids, sources, targets) = graph.inner.get_columnar_topology();
            let mut induced_edges = Vec::new();

            for i in 0..edge_ids.len() {
                let edge_id = edge_ids[i];
                let source = sources[i];
                let target = targets[i];

                if node_set.contains(&source) && node_set.contains(&target) {
                    induced_edges.push(edge_id);
                }
            }

            // Create and return Subgraph
            let subgraph = PySubgraph::new(
                selected_nodes,
                induced_edges,
                "boolean_selection".to_string(),
                Some(self.graph.clone()),
            );

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // Try to extract as list of integers (batch access) - CHECK AFTER boolean arrays
        if let Ok(indices_or_ids) = key.extract::<Vec<NodeId>>() {
            // Batch node access - return Subgraph
            let graph = self.graph.borrow_mut(py);

            // Convert indices to actual node IDs if constrained
            let actual_node_ids: Result<Vec<NodeId>, PyErr> =
                if let Some(ref constrained) = self.constrained_nodes {
                    // Constrained case: treat as indices into constrained list
                    indices_or_ids
                        .into_iter()
                        .map(|index| {
                            if (index as usize) >= constrained.len() {
                                Err(PyIndexError::new_err(format!(
                                    "Node index {} out of range (0-{})",
                                    index,
                                    constrained.len() - 1
                                )))
                            } else {
                                Ok(constrained[index as usize])
                            }
                        })
                        .collect()
                } else {
                    // Unconstrained case: treat as actual node IDs
                    Ok(indices_or_ids)
                };

            let node_ids = actual_node_ids?;

            // Validate all nodes exist
            for &node_id in &node_ids {
                if !graph.has_node_internal(node_id) {
                    return Err(PyKeyError::new_err(format!(
                        "Node {} does not exist",
                        node_id
                    )));
                }
            }

            // Get induced edges for selected nodes
            let node_set: std::collections::HashSet<NodeId> = node_ids.iter().copied().collect();
            let (edge_ids, sources, targets) = graph.inner.get_columnar_topology();
            let mut induced_edges = Vec::new();

            for i in 0..edge_ids.len() {
                let edge_id = edge_ids[i];
                let source = sources[i];
                let target = targets[i];

                if node_set.contains(&source) && node_set.contains(&target) {
                    induced_edges.push(edge_id);
                }
            }

            // Create and return Subgraph
            let subgraph = PySubgraph::new(
                node_ids,
                induced_edges,
                "node_batch_selection".to_string(),
                Some(self.graph.clone()),
            );

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // Try to extract as slice (slice access)
        if let Ok(slice) = key.downcast::<PySlice>() {
            let graph = self.graph.borrow_mut(py);
            let all_node_ids = graph.inner.node_ids();

            // Convert slice to indices
            let slice_info = slice.indices(
                all_node_ids
                    .len()
                    .try_into()
                    .map_err(|_| PyValueError::new_err("Collection too large for slice"))?,
            )?;
            let start = slice_info.start as usize;
            let stop = slice_info.stop as usize;
            let step = slice_info.step as usize;

            // Extract nodes based on slice
            let mut selected_nodes = Vec::new();
            let mut i = start;
            while i < stop && i < all_node_ids.len() {
                selected_nodes.push(all_node_ids[i]);
                i += step;
            }

            // 🚀 PERFORMANCE FIX: Use core columnar topology instead of O(E) FFI algorithm
            let selected_node_set: std::collections::HashSet<NodeId> =
                selected_nodes.iter().copied().collect();
            let (edge_ids, sources, targets) = graph.inner.get_columnar_topology();
            let mut induced_edges = Vec::new();

            // O(k) where k = active edges, much better than O(E)
            for i in 0..edge_ids.len() {
                let edge_id = edge_ids[i];
                let source = sources[i];
                let target = targets[i];

                // O(1) HashSet lookups
                if selected_node_set.contains(&source) && selected_node_set.contains(&target) {
                    induced_edges.push(edge_id);
                }
            }

            // Create and return Subgraph
            let subgraph = PySubgraph::new(
                selected_nodes,
                induced_edges,
                "node_slice_selection".to_string(),
                Some(self.graph.clone()),
            );

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // If none of the above worked, return error
        Err(PyTypeError::new_err(
            "Node index must be int, list of ints, or slice. \
            To access node attributes, use: graph.nodes.table() for all attributes, \
            graph.nodes.table()['attribute_name'] for specific attributes, or \
            graph.nodes[node_id].attribute_name for a single node's attribute.",
        ))
    }

    /// Support iteration: for node_view in g.nodes
    fn __iter__(&self, py: Python) -> PyResult<NodesIterator> {
        let node_ids = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            let graph = self.graph.borrow(py);
            graph.inner.node_ids()
        };

        Ok(NodesIterator {
            graph: self.graph.clone(),
            node_ids,
            index: 0,
        })
    }

    /// Support len(g.nodes)
    fn __len__(&self, py: Python) -> PyResult<usize> {
        if let Some(ref constrained) = self.constrained_nodes {
            Ok(constrained.len())
        } else {
            let graph = self.graph.borrow(py);
            Ok(graph.get_node_count())
        }
    }

    /// String representation
    fn __str__(&self, py: Python) -> PyResult<String> {
        let graph = self.graph.borrow(py);
        let count = graph.get_node_count();
        Ok(format!("NodesAccessor({} nodes)", count))
    }

    /// Get all unique attribute names across all nodes
    #[getter]
    fn attributes(&self, py: Python) -> PyResult<Vec<String>> {
        let graph = self.graph.borrow(py);
        let mut all_attrs = std::collections::HashSet::new();

        // Determine which nodes to check
        let node_ids = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            // Get all node IDs from the graph
            (0..graph.get_node_count() as NodeId).collect()
        };

        // Collect attributes from all nodes
        for &node_id in &node_ids {
            if graph.has_node_internal(node_id) {
                let attrs = graph.node_attribute_keys(node_id);
                for attr in attrs {
                    all_attrs.insert(attr);
                }
            }
        }

        // Convert to sorted vector
        let mut result: Vec<String> = all_attrs.into_iter().collect();
        result.sort();
        Ok(result)
    }

    /// Get table view of nodes (GraphTable with node attributes)
    pub fn table(&self, py: Python) -> PyResult<PyObject> {
        use crate::ffi::core::table::PyGraphTable;

        // Determine node list based on constraints
        let node_data = if let Some(ref constrained) = self.constrained_nodes {
            // Subgraph case: use constrained nodes
            constrained.iter().map(|&id| id as u64).collect()
        } else {
            // Full graph case: get all node IDs from the graph
            let graph = self.graph.borrow(py);
            graph
                .inner
                .node_ids()
                .into_iter()
                .map(|id| id as u64)
                .collect()
        };

        // Use PyGraphTable::from_graph_nodes to create the table
        let py_table = PyGraphTable::from_graph_nodes(
            py.get_type::<PyGraphTable>(),
            py,
            self.graph.clone(),
            node_data,
            None, // Get all attributes
        )?;

        Ok(Py::new(py, py_table)?.to_object(py))
    }
}

/// Iterator for edges that yields EdgeViews
#[pyclass]
pub struct EdgesIterator {
    graph: Py<PyGraph>,
    edge_ids: Vec<EdgeId>,
    index: usize,
}

#[pymethods]
impl EdgesIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        if self.index < self.edge_ids.len() {
            let edge_id = self.edge_ids[self.index];
            self.index += 1;

            // Create EdgeView for this edge
            let edge_view = PyGraph::create_edge_view_internal(self.graph.clone(), py, edge_id)?;
            Ok(Some(edge_view.to_object(py)))
        } else {
            Ok(None)
        }
    }
}

/// Wrapper for g.edges that supports indexing syntax: g.edges[id] -> EdgeView  
#[pyclass(name = "EdgesAccessor")]
pub struct PyEdgesAccessor {
    pub graph: Py<PyGraph>,
    /// Optional constraint: if Some, only these edges are accessible
    pub constrained_edges: Option<Vec<EdgeId>>,
}

#[pymethods]
impl PyEdgesAccessor {
    /// Support edge access: g.edges[0] -> EdgeView, g.edges[[0,1,2]] -> Subgraph, g.edges[0:5] -> Subgraph
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        // Try to extract as single integer
        if let Ok(index_or_id) = key.extract::<EdgeId>() {
            let actual_edge_id = if let Some(ref constrained) = self.constrained_edges {
                // Constrained case: treat as index into constrained list
                if (index_or_id as usize) >= constrained.len() {
                    return Err(PyIndexError::new_err(format!(
                        "Edge index {} out of range (0-{})",
                        index_or_id,
                        constrained.len() - 1
                    )));
                }
                constrained[index_or_id as usize]
            } else {
                // Unconstrained case: treat as actual edge ID (existing behavior)
                index_or_id
            };

            // Single edge access - return EdgeView
            let graph = self.graph.borrow(py);
            if !graph.has_edge_internal(actual_edge_id) {
                return Err(PyKeyError::new_err(format!(
                    "Edge {} does not exist",
                    actual_edge_id
                )));
            }

            let edge_view =
                PyGraph::create_edge_view_internal(self.graph.clone(), py, actual_edge_id)?;
            return Ok(edge_view.to_object(py));
        }

        // Try to extract as boolean array/list (boolean indexing) - CHECK FIRST before integers
        if let Ok(boolean_mask) = key.extract::<Vec<bool>>() {
            let graph = self.graph.borrow(py);
            let all_edge_ids = if let Some(ref constrained) = self.constrained_edges {
                constrained.clone()
            } else {
                graph.inner.edge_ids()
            };

            // Check if boolean mask length matches edge count
            if boolean_mask.len() != all_edge_ids.len() {
                return Err(PyIndexError::new_err(format!(
                    "Boolean mask length ({}) must match number of edges ({})",
                    boolean_mask.len(),
                    all_edge_ids.len()
                )));
            }

            // Select edges where boolean mask is True
            let selected_edges: Vec<EdgeId> = all_edge_ids
                .iter()
                .zip(boolean_mask.iter())
                .filter_map(|(&edge_id, &include)| if include { Some(edge_id) } else { None })
                .collect();

            if selected_edges.is_empty() {
                return Err(PyIndexError::new_err("Boolean mask selected no edges"));
            }

            // Validate all selected edges exist
            for &edge_id in &selected_edges {
                if !graph.has_edge_internal(edge_id) {
                    return Err(PyKeyError::new_err(format!(
                        "Edge {} does not exist",
                        edge_id
                    )));
                }
            }

            // Get all endpoint nodes from selected edges
            let mut endpoint_nodes = std::collections::HashSet::new();
            for &edge_id in &selected_edges {
                if let Ok((source, target)) = graph.inner.edge_endpoints(edge_id) {
                    endpoint_nodes.insert(source);
                    endpoint_nodes.insert(target);
                }
            }

            // Create and return Subgraph
            let subgraph = PySubgraph::new(
                endpoint_nodes.into_iter().collect(),
                selected_edges,
                "boolean_edge_selection".to_string(),
                Some(self.graph.clone()),
            );

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // Try to extract as list of integers (batch access) - CHECK AFTER boolean arrays
        if let Ok(indices_or_ids) = key.extract::<Vec<EdgeId>>() {
            // Batch edge access - return Subgraph with these edges + their endpoints
            let graph = self.graph.borrow(py);

            // Convert indices to actual edge IDs if constrained
            let actual_edge_ids: Result<Vec<EdgeId>, PyErr> =
                if let Some(ref constrained) = self.constrained_edges {
                    // Constrained case: treat as indices into constrained list
                    indices_or_ids
                        .into_iter()
                        .map(|index| {
                            if (index as usize) >= constrained.len() {
                                Err(PyIndexError::new_err(format!(
                                    "Edge index {} out of range (0-{})",
                                    index,
                                    constrained.len() - 1
                                )))
                            } else {
                                Ok(constrained[index as usize])
                            }
                        })
                        .collect()
                } else {
                    // Unconstrained case: treat as actual edge IDs
                    Ok(indices_or_ids)
                };

            let edge_ids = actual_edge_ids?;

            // Validate all edges exist
            for &edge_id in &edge_ids {
                if !graph.has_edge_internal(edge_id) {
                    return Err(PyKeyError::new_err(format!(
                        "Edge {} does not exist",
                        edge_id
                    )));
                }
            }

            // Get all endpoint nodes from these edges
            let mut endpoint_nodes = std::collections::HashSet::new();
            for &edge_id in &edge_ids {
                if let Ok((source, target)) = graph.inner.edge_endpoints(edge_id) {
                    endpoint_nodes.insert(source);
                    endpoint_nodes.insert(target);
                }
            }

            // Create and return Subgraph
            let subgraph = PySubgraph::new(
                endpoint_nodes.into_iter().collect(),
                edge_ids,
                "edge_batch_selection".to_string(),
                Some(self.graph.clone()),
            );

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // Try to extract as slice (slice access)
        if let Ok(slice) = key.downcast::<PySlice>() {
            let graph = self.graph.borrow(py);
            let all_edge_ids = graph.inner.edge_ids();

            // Convert slice to indices
            let slice_info = slice.indices(
                all_edge_ids
                    .len()
                    .try_into()
                    .map_err(|_| PyValueError::new_err("Collection too large for slice"))?,
            )?;
            let start = slice_info.start as usize;
            let stop = slice_info.stop as usize;
            let step = slice_info.step as usize;

            // Extract edges based on slice
            let mut selected_edges = Vec::new();
            let mut i = start;
            while i < stop && i < all_edge_ids.len() {
                selected_edges.push(all_edge_ids[i]);
                i += step;
            }

            // Get all endpoint nodes from selected edges
            let mut endpoint_nodes = std::collections::HashSet::new();
            for &edge_id in &selected_edges {
                if let Ok((source, target)) = graph.inner.edge_endpoints(edge_id) {
                    endpoint_nodes.insert(source);
                    endpoint_nodes.insert(target);
                }
            }

            // Create and return Subgraph
            let subgraph = PySubgraph::new(
                endpoint_nodes.into_iter().collect(),
                selected_edges,
                "edge_slice_selection".to_string(),
                Some(self.graph.clone()),
            );

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // If none of the above worked, return error
        Err(PyTypeError::new_err(
            "Edge index must be int, list of ints, or slice",
        ))
    }

    /// Support iteration: for edge_view in g.edges
    fn __iter__(&self, py: Python) -> PyResult<EdgesIterator> {
        let edge_ids = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            let graph = self.graph.borrow(py);
            graph.inner.edge_ids()
        };

        Ok(EdgesIterator {
            graph: self.graph.clone(),
            edge_ids,
            index: 0,
        })
    }

    /// Support len(g.edges)
    fn __len__(&self, py: Python) -> PyResult<usize> {
        if let Some(ref constrained) = self.constrained_edges {
            Ok(constrained.len())
        } else {
            let graph = self.graph.borrow(py);
            Ok(graph.get_edge_count())
        }
    }

    /// String representation
    fn __str__(&self, py: Python) -> PyResult<String> {
        let graph = self.graph.borrow(py);
        let count = graph.get_edge_count();
        Ok(format!("EdgesAccessor({} edges)", count))
    }

    /// Get all unique attribute names across all edges
    #[getter]
    fn attributes(&self, py: Python) -> PyResult<Vec<String>> {
        let graph = self.graph.borrow(py);
        let mut all_attrs = std::collections::HashSet::new();

        // Determine which edges to check
        let edge_ids = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            // Get all edge IDs from the graph
            graph.inner.edge_ids()
        };

        // Collect attributes from all edges
        for &edge_id in &edge_ids {
            let attrs = graph.edge_attribute_keys(edge_id);
            for attr in attrs {
                all_attrs.insert(attr);
            }
        }

        // Convert to sorted vector
        let mut result: Vec<String> = all_attrs.into_iter().collect();
        result.sort();
        Ok(result)
    }

    /// Get table view of edges (GraphTable with edge attributes)  
    pub fn table(&self, py: Python) -> PyResult<PyObject> {
        use crate::ffi::core::table::PyGraphTable;

        // Determine edge list based on constraints
        let edge_data = if let Some(ref constrained) = self.constrained_edges {
            // Subgraph case: use constrained edges
            constrained.iter().map(|&id| id as u64).collect()
        } else {
            // Full graph case: get all edge IDs from the graph
            let graph = self.graph.borrow(py);
            graph
                .inner
                .edge_ids()
                .into_iter()
                .map(|id| id as u64)
                .collect()
        };

        // Use PyGraphTable::from_graph_edges to create the table
        let py_table = PyGraphTable::from_graph_edges(
            py.get_type::<PyGraphTable>(),
            py,
            self.graph.clone(),
            edge_data,
            None, // Get all attributes
        )?;

        Ok(Py::new(py, py_table)?.to_object(py))
    }
}
