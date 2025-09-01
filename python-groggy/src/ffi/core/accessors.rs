//! Accessors FFI Bindings
//!
//! Python bindings for smart indexing accessors.

use groggy::{AttrValue, EdgeId, NodeId};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};
use std::collections::HashSet;

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

// Import types from our FFI modules
use crate::ffi::core::subgraph::PySubgraph;
use crate::ffi::core::views::PyNodeView;

/// Helper function to create NodeView from core Graph
fn create_node_view_from_core(
    graph: std::rc::Rc<std::cell::RefCell<groggy::Graph>>,
    py: Python,
    node_id: NodeId,
) -> PyResult<PyObject> {
    let node_view = PyNodeView {
        graph: graph.clone(),
        node_id,
    };
    Ok(Py::new(py, node_view)?.to_object(py))
}

/// Helper function to create EdgeView from core Graph
fn create_edge_view_from_core(
    graph: std::rc::Rc<std::cell::RefCell<groggy::Graph>>,
    py: Python,
    edge_id: EdgeId,
) -> PyResult<PyObject> {
    // Create proper EdgeView object
    use crate::ffi::core::views::PyEdgeView;

    let graph_ref = graph.borrow();
    if graph_ref.has_edge(edge_id) {
        let edge_view = PyEdgeView {
            graph: graph.clone(),
            edge_id,
        };
        Ok(Py::new(py, edge_view)?.to_object(py))
    } else {
        Err(PyKeyError::new_err(format!("Edge {} not found", edge_id)))
    }
}

/// Iterator for nodes that yields NodeViews
#[pyclass(unsendable)]
pub struct NodesIterator {
    graph: std::rc::Rc<std::cell::RefCell<groggy::Graph>>,
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

            // Create PyNodeView that supports 'attr' in node and node['attr'] syntax
            let node_view = PyNodeView {
                graph: self.graph.clone(),
                node_id,
            };
            Ok(Some(Py::new(py, node_view)?.to_object(py)))
        } else {
            Ok(None)
        }
    }
}

/// Wrapper for g.nodes that supports indexing syntax: g.nodes[id] -> NodeView
#[pyclass(name = "NodesAccessor", unsendable)]
pub struct PyNodesAccessor {
    pub graph: std::rc::Rc<std::cell::RefCell<groggy::Graph>>,
    /// Optional constraint: if Some, only these nodes are accessible
    pub constrained_nodes: Option<Vec<NodeId>>,
}

#[pymethods]
impl PyNodesAccessor {
    /// Set attributes for multiple nodes (bulk operation)
    /// Supports the same formats as the main graph: node-centric, column-centric, etc.
    fn set_attrs(&self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        self.set_attrs_internal(py, attrs_dict)
    }
    /// Support node access: g.nodes[0] -> NodeView, g.nodes[[0,1,2]] -> Subgraph, g.nodes[0:5] -> Subgraph
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        // Try to extract as single integer
        if let Ok(index_or_id) = key.extract::<NodeId>() {
            let actual_node_id = if let Some(ref constrained) = self.constrained_nodes {
                // Constrained case: treat as index into constrained list
                if (index_or_id) >= constrained.len() {
                    return Err(PyIndexError::new_err(format!(
                        "Node index {} out of range (0-{})",
                        index_or_id,
                        constrained.len() - 1
                    )));
                }
                constrained[index_or_id]
            } else {
                // Unconstrained case: treat as actual node ID (existing behavior)
                index_or_id
            };

            // Single node access - return NodeView
            let graph = self.graph.borrow();
            if !graph.contains_node(actual_node_id) {
                return Err(PyKeyError::new_err(format!(
                    "Node {} does not exist",
                    actual_node_id
                )));
            }

            let node_view = create_node_view_from_core(self.graph.clone(), py, actual_node_id)?;
            return Ok(node_view.to_object(py));
        }

        // Try to extract as boolean array/list (boolean indexing) - CHECK FIRST before integers
        if let Ok(boolean_mask) = key.extract::<Vec<bool>>() {
            let graph = self.graph.borrow();
            let all_node_ids = if let Some(ref constrained) = self.constrained_nodes {
                constrained.clone()
            } else {
                {
                    graph.node_ids()
                }
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
                if !graph.contains_node(node_id) {
                    return Err(PyKeyError::new_err(format!(
                        "Node {} does not exist",
                        node_id
                    )));
                }
            }

            // Get induced edges for selected nodes
            let node_set: std::collections::HashSet<NodeId> =
                selected_nodes.iter().copied().collect();
            let (edge_ids, sources, targets) = { graph.get_columnar_topology() };
            let mut induced_edges = Vec::new();

            for i in 0..edge_ids.len() {
                let edge_id = edge_ids[i];
                let source = sources[i];
                let target = targets[i];

                if node_set.contains(&source) && node_set.contains(&target) {
                    induced_edges.push(edge_id);
                }
            }

            // Create core Subgraph first, then wrap in PySubgraph
            use std::collections::HashSet;
            let node_set: HashSet<NodeId> = selected_nodes.iter().copied().collect();
            let edge_set: HashSet<EdgeId> = induced_edges.iter().copied().collect();
            let core_subgraph = groggy::core::subgraph::Subgraph::new(
                self.graph.clone(),
                node_set,
                edge_set,
                "boolean_selection".to_string(),
            );
            let subgraph = PySubgraph::from_core_subgraph(core_subgraph)?;

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // Try to extract as list of integers (batch access) - CHECK AFTER boolean arrays
        if let Ok(indices_or_ids) = key.extract::<Vec<NodeId>>() {
            // Batch node access - return Subgraph

            // Convert indices to actual node IDs if constrained
            let actual_node_ids: Result<Vec<NodeId>, PyErr> =
                if let Some(ref constrained) = self.constrained_nodes {
                    // Constrained case: treat as indices into constrained list
                    indices_or_ids
                        .into_iter()
                        .map(|index| {
                            if (index) >= constrained.len() {
                                Err(PyIndexError::new_err(format!(
                                    "Node index {} out of range (0-{})",
                                    index,
                                    constrained.len() - 1
                                )))
                            } else {
                                Ok(constrained[index])
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
                let exists = {
                    let graph = self.graph.borrow();
                    graph.contains_node(node_id)
                };
                if !exists {
                    return Err(PyKeyError::new_err(format!(
                        "Node {} does not exist",
                        node_id
                    )));
                }
            }

            // Get induced edges for selected nodes
            let node_set: std::collections::HashSet<NodeId> = node_ids.iter().copied().collect();
            let (edge_ids, sources, targets) = {
                let graph = self.graph.borrow();
                graph.get_columnar_topology()
            };
            let mut induced_edges = Vec::new();

            for i in 0..edge_ids.len() {
                let edge_id = edge_ids[i];
                let source = sources[i];
                let target = targets[i];

                if node_set.contains(&source) && node_set.contains(&target) {
                    induced_edges.push(edge_id);
                }
            }

            // Create core Subgraph first, then wrap in PySubgraph
            let node_set: HashSet<NodeId> = node_ids.iter().copied().collect();
            let edge_set: HashSet<EdgeId> = induced_edges.iter().copied().collect();
            let core_subgraph = groggy::core::subgraph::Subgraph::new(
                self.graph.clone(),
                node_set,
                edge_set,
                "node_batch_selection".to_string(),
            );
            let subgraph = PySubgraph::from_core_subgraph(core_subgraph)?;

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // Try to extract as slice (slice access)
        if let Ok(slice) = key.downcast::<PySlice>() {
            let all_node_ids = {
                let graph = self.graph.borrow(); // Only need read access
                {
                    graph.node_ids()
                }
            };

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
            let (edge_ids, sources, targets) = {
                let graph = self.graph.borrow();
                graph.get_columnar_topology()
            };
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

            // Create core Subgraph first, then wrap in PySubgraph
            let node_set: HashSet<NodeId> = selected_nodes.iter().copied().collect();
            let edge_set: HashSet<EdgeId> = induced_edges.iter().copied().collect();
            let core_subgraph = groggy::core::subgraph::Subgraph::new(
                self.graph.clone(),
                node_set,
                edge_set,
                "node_slice_selection".to_string(),
            );
            let subgraph = PySubgraph::from_core_subgraph(core_subgraph)?;

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // Try to extract as string (attribute name access)
        if let Ok(attr_name) = key.extract::<String>() {
            // Return GraphArray of attribute values for all nodes
            let graph = self.graph.borrow();
            let node_ids = if let Some(ref constrained) = self.constrained_nodes {
                constrained.clone()
            } else {
                graph.node_ids()
            };

            let mut attr_values = Vec::new();
            for node_id in node_ids {
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
            let py_graph_array = crate::ffi::core::array::PyGraphArray { inner: graph_array };
            return Ok(Py::new(py, py_graph_array)?.to_object(py));
        }

        // If none of the above worked, return error
        Err(PyTypeError::new_err(
            "Node index must be int, list of ints, slice, or string attribute name. \
            Examples: g.nodes[0], g.nodes[0:10], g.nodes['age'], or g.age for attribute access.",
        ))
    }

    /// Support iteration: for node_view in g.nodes
    fn __iter__(&self, _py: Python) -> PyResult<NodesIterator> {
        let node_ids = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            let graph = self.graph.borrow();
            graph.node_ids()
        };

        Ok(NodesIterator {
            graph: self.graph.clone(),
            node_ids,
            index: 0,
        })
    }

    /// Support len(g.nodes)
    fn __len__(&self, _py: Python) -> PyResult<usize> {
        if let Some(ref constrained) = self.constrained_nodes {
            Ok(constrained.len())
        } else {
            let graph = self.graph.borrow();
            Ok(graph.node_ids().len())
        }
    }

    /// String representation
    fn __str__(&self, _py: Python) -> PyResult<String> {
        let graph = self.graph.borrow();
        let count = graph.node_ids().len();
        Ok(format!("NodesAccessor({} nodes)", count))
    }

    /// Get all unique attribute names across all nodes
    #[getter]
    fn attributes(&self, _py: Python) -> PyResult<Vec<String>> {
        let graph = self.graph.borrow();
        let mut all_attrs = std::collections::HashSet::new();

        // Determine which nodes to check
        let node_ids = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            // Get all node IDs from the graph
            (0..graph.node_ids().len() as NodeId).collect()
        };

        // Collect attributes from all nodes
        for &node_id in &node_ids {
            if graph.contains_node(node_id) {
                let attrs: Vec<String> = graph
                    .get_node_attrs(node_id)
                    .map(|map| map.keys().cloned().collect())
                    .unwrap_or_default();
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

    /// Get table view of nodes (GraphTable with node attributes) - DELEGATED to SubgraphOperations
    pub fn table(&self, py: Python) -> PyResult<PyObject> {
        use crate::ffi::core::table::PyGraphTable;

        // Pure delegation to SubgraphOperations through graph's as_subgraph() method
        let graph = self.graph.borrow();

        if let Some(ref constrained) = self.constrained_nodes {
            // Subgraph case: create subgraph with constrained nodes and delegate
            let all_edges = graph
                .edge_ids()
                .into_iter()
                .collect::<std::collections::HashSet<_>>();
            let constrained_set = constrained
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>();

            // Create subgraph and delegate to SubgraphOperations::nodes_table()
            let subgraph = groggy::core::subgraph::Subgraph::new(
                self.graph.clone(),
                constrained_set,
                all_edges, // TODO: Should be induced edges, but this maintains current behavior
                "nodes_accessor_subgraph".to_string(),
            );

            // Delegate to SubgraphOperations trait
            let subgraph_ops: &dyn groggy::core::traits::SubgraphOperations = &subgraph;
            let core_table = subgraph_ops
                .nodes_table()
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            let py_table = PyGraphTable::from_graph_table(core_table);
            Ok(Py::new(py, py_table)?.to_object(py))
        } else {
            // Full graph case: create full subgraph and delegate
            let all_nodes = graph
                .node_ids()
                .into_iter()
                .collect::<std::collections::HashSet<_>>();
            let all_edges = graph
                .edge_ids()
                .into_iter()
                .collect::<std::collections::HashSet<_>>();

            let subgraph = groggy::core::subgraph::Subgraph::new(
                self.graph.clone(),
                all_nodes,
                all_edges,
                "full_nodes_table".to_string(),
            );

            // Delegate to SubgraphOperations trait
            let subgraph_ops: &dyn groggy::core::traits::SubgraphOperations = &subgraph;
            let core_table = subgraph_ops
                .nodes_table()
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            let py_table = PyGraphTable::from_graph_table(core_table);
            Ok(Py::new(py, py_table)?.to_object(py))
        }
    }

    /// Get all nodes as a subgraph (equivalent to g.nodes[:]) - DELEGATED to SubgraphOperations  
    /// Returns a subgraph containing all nodes and all induced edges
    fn all(&self, _py: Python) -> PyResult<PySubgraph> {
        let graph = self.graph.borrow();

        // Pure delegation to SubgraphOperations::induced_subgraph()
        let all_node_ids = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            graph.node_ids()
        };

        // Create induced subgraph directly
        let node_set: HashSet<NodeId> = all_node_ids.iter().copied().collect();
        let induced_edges =
            groggy::core::subgraph::Subgraph::calculate_induced_edges(&self.graph, &node_set)
                .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{:?}", e)))?;

        let core_subgraph = groggy::core::subgraph::Subgraph::new(
            self.graph.clone(),
            node_set,
            induced_edges,
            "all_nodes".to_string(),
        );
        PySubgraph::from_core_subgraph(core_subgraph)
    }

    /// Get node attribute column with proper error checking
    fn _get_node_attribute_column(&self, py: Python, attr_name: &str) -> PyResult<PyObject> {
        let graph = self.graph.borrow();

        // Determine which nodes to check
        let node_ids = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            {
                graph.node_ids()
            }
        };

        if node_ids.is_empty() {
            return Err(PyValueError::new_err(format!(
                "Cannot access attribute '{}': No nodes available",
                attr_name
            )));
        }

        // Check if the attribute exists on ANY node
        let mut attribute_exists = false;
        for &node_id in &node_ids {
            if graph.contains_node(node_id) {
                let attrs: Vec<String> = graph
                    .get_node_attrs(node_id)
                    .map(|map| map.keys().cloned().collect())
                    .unwrap_or_default();
                if attrs.contains(&attr_name.to_string()) {
                    attribute_exists = true;
                    break;
                }
            }
        }

        if !attribute_exists {
            return Err(PyKeyError::new_err(format!(
                "Attribute '{}' does not exist on any nodes. Available attributes: {:?}",
                attr_name,
                {
                    let mut all_attrs = std::collections::HashSet::new();
                    for &node_id in &node_ids {
                        if graph.contains_node(node_id) {
                            let attrs: Vec<String> = graph
                                .get_node_attrs(node_id)
                                .map(|map| map.keys().cloned().collect())
                                .unwrap_or_default();
                            for attr in attrs {
                                all_attrs.insert(attr);
                            }
                        }
                    }
                    let mut result: Vec<String> = all_attrs.into_iter().collect();
                    result.sort();
                    result
                }
            )));
        }

        // Collect attribute values - allow None for nodes without the attribute
        let mut values: Vec<Option<PyObject>> = Vec::new();
        for &node_id in &node_ids {
            if graph.contains_node(node_id) {
                match graph.get_node_attr(node_id, &attr_name.to_string()) {
                    Ok(Some(value)) => {
                        // Convert the attribute value to Python object
                        let py_value = attr_value_to_python_value(py, &value)?;
                        values.push(Some(py_value));
                    }
                    Ok(None) => {
                        values.push(None);
                    }
                    Err(_) => {
                        values.push(None);
                    }
                }
            } else {
                values.push(None);
            }
        }

        // Convert to Python list
        let py_values: Vec<PyObject> = values
            .into_iter()
            .map(|opt_val| match opt_val {
                Some(val) => val,
                None => py.None(),
            })
            .collect();

        Ok(py_values.to_object(py))
    }
}

impl PyNodesAccessor {
    /// Set attributes for multiple nodes (bulk operation) - internal method callable from Rust
    /// Supports the same formats as the main graph: node-centric, column-centric, etc.
    pub fn set_attrs_internal(&self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        use crate::ffi::api::graph_attributes::PyGraphAttrMut;

        // Create a mutable graph attributes handler
        let mut attr_handler = PyGraphAttrMut::new(self.graph.clone());

        // If this accessor is constrained to specific nodes, we need to validate
        if let Some(ref constrained_nodes) = self.constrained_nodes {
            // Validate that all node IDs in attrs_dict are in our constrained set
            let constrained_set: std::collections::HashSet<NodeId> =
                constrained_nodes.iter().copied().collect();

            for (attr_name_py, node_values_py) in attrs_dict.iter() {
                let _attr_name: String = attr_name_py.extract()?;

                // Handle different formats - for now assume node-centric format
                if let Ok(node_dict) = node_values_py.extract::<&pyo3::types::PyDict>() {
                    for (node_py, _value_py) in node_dict.iter() {
                        let node_id: NodeId = node_py.extract()?;
                        if !constrained_set.contains(&node_id) {
                            return Err(pyo3::exceptions::PyPermissionError::new_err(format!(
                                "Cannot set attribute for node {} - not in this subgraph/view",
                                node_id
                            )));
                        }
                    }
                }
            }
        }

        // Delegate to the graph attributes handler (it has the smart format detection)
        attr_handler.set_node_attrs(py, attrs_dict)
    }
}

/// Iterator for edges that yields EdgeViews
#[pyclass(unsendable)]
pub struct EdgesIterator {
    graph: std::rc::Rc<std::cell::RefCell<groggy::Graph>>,
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

            // Create EdgeView for this edge using core Graph directly
            let edge_view = create_edge_view_from_core(self.graph.clone(), py, edge_id)?;
            Ok(Some(edge_view.to_object(py)))
        } else {
            Ok(None)
        }
    }
}

/// Wrapper for g.edges that supports indexing syntax: g.edges[id] -> EdgeView  
#[pyclass(name = "EdgesAccessor", unsendable)]
pub struct PyEdgesAccessor {
    pub graph: std::rc::Rc<std::cell::RefCell<groggy::Graph>>,
    /// Optional constraint: if Some, only these edges are accessible
    pub constrained_edges: Option<Vec<EdgeId>>,
}

#[pymethods]
impl PyEdgesAccessor {
    /// Set attributes for multiple edges (bulk operation)
    /// Supports the same formats as the main graph: edge-centric, column-centric, etc.
    fn set_attrs(&self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        self.set_attrs_internal(py, attrs_dict)
    }
    /// Support edge access: g.edges[0] -> EdgeView, g.edges[[0,1,2]] -> Subgraph, g.edges[0:5] -> Subgraph
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        // Try to extract as single integer
        if let Ok(index_or_id) = key.extract::<EdgeId>() {
            let actual_edge_id = if let Some(ref constrained) = self.constrained_edges {
                // Constrained case: treat as index into constrained list
                if (index_or_id) >= constrained.len() {
                    return Err(PyIndexError::new_err(format!(
                        "Edge index {} out of range (0-{})",
                        index_or_id,
                        constrained.len() - 1
                    )));
                }
                constrained[index_or_id]
            } else {
                // Unconstrained case: treat as actual edge ID (existing behavior)
                index_or_id
            };

            // Single edge access - return EdgeView
            let graph = self.graph.borrow();
            if !graph.has_edge(actual_edge_id) {
                return Err(PyKeyError::new_err(format!(
                    "Edge {} does not exist",
                    actual_edge_id
                )));
            }

            let edge_view = create_edge_view_from_core(self.graph.clone(), py, actual_edge_id)?;
            return Ok(edge_view.to_object(py));
        }

        // Try to extract as boolean array/list (boolean indexing) - CHECK FIRST before integers
        if let Ok(boolean_mask) = key.extract::<Vec<bool>>() {
            let graph = self.graph.borrow();
            let all_edge_ids = if let Some(ref constrained) = self.constrained_edges {
                constrained.clone()
            } else {
                {
                    graph.edge_ids()
                }
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
                if !graph.has_edge(edge_id) {
                    return Err(PyKeyError::new_err(format!(
                        "Edge {} does not exist",
                        edge_id
                    )));
                }
            }

            // Get all endpoint nodes from selected edges
            let mut endpoint_nodes = std::collections::HashSet::new();
            for &edge_id in &selected_edges {
                let endpoints = {
                    let graph = self.graph.borrow();
                    graph.edge_endpoints(edge_id)
                };
                if let Ok((source, target)) = endpoints {
                    endpoint_nodes.insert(source);
                    endpoint_nodes.insert(target);
                }
            }

            // Create core Subgraph first, then wrap in PySubgraph
            let node_set: HashSet<NodeId> = endpoint_nodes.into_iter().collect();
            let edge_set: HashSet<EdgeId> = selected_edges.iter().copied().collect();
            let core_subgraph = groggy::core::subgraph::Subgraph::new(
                self.graph.clone(),
                node_set,
                edge_set,
                "boolean_edge_selection".to_string(),
            );
            let subgraph = PySubgraph::from_core_subgraph(core_subgraph)?;

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // Try to extract as list of integers (batch access) - CHECK AFTER boolean arrays
        if let Ok(indices_or_ids) = key.extract::<Vec<EdgeId>>() {
            // Batch edge access - return Subgraph with these edges + their endpoints
            let graph = self.graph.borrow();

            // Convert indices to actual edge IDs if constrained
            let actual_edge_ids: Result<Vec<EdgeId>, PyErr> =
                if let Some(ref constrained) = self.constrained_edges {
                    // Constrained case: treat as indices into constrained list
                    indices_or_ids
                        .into_iter()
                        .map(|index| {
                            if (index) >= constrained.len() {
                                Err(PyIndexError::new_err(format!(
                                    "Edge index {} out of range (0-{})",
                                    index,
                                    constrained.len() - 1
                                )))
                            } else {
                                Ok(constrained[index])
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
                if !graph.has_edge(edge_id) {
                    return Err(PyKeyError::new_err(format!(
                        "Edge {} does not exist",
                        edge_id
                    )));
                }
            }

            // Get all endpoint nodes from these edges
            let mut endpoint_nodes = std::collections::HashSet::new();
            for &edge_id in &edge_ids {
                if let Ok((source, target)) = graph.edge_endpoints(edge_id) {
                    endpoint_nodes.insert(source);
                    endpoint_nodes.insert(target);
                }
            }

            // Create core Subgraph first, then wrap in PySubgraph
            let node_set: HashSet<NodeId> = endpoint_nodes.into_iter().collect();
            let edge_set: HashSet<EdgeId> = edge_ids.iter().copied().collect();
            let core_subgraph = groggy::core::subgraph::Subgraph::new(
                self.graph.clone(),
                node_set,
                edge_set,
                "edge_batch_selection".to_string(),
            );
            let subgraph = PySubgraph::from_core_subgraph(core_subgraph)?;

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // Try to extract as slice (slice access)
        if let Ok(slice) = key.downcast::<PySlice>() {
            let all_edge_ids = {
                let graph = self.graph.borrow();
                {
                    graph.edge_ids()
                }
            };

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
                let endpoints = {
                    let graph = self.graph.borrow();
                    graph.edge_endpoints(edge_id)
                };
                if let Ok((source, target)) = endpoints {
                    endpoint_nodes.insert(source);
                    endpoint_nodes.insert(target);
                }
            }

            // Create core Subgraph first, then wrap in PySubgraph
            let node_set: HashSet<NodeId> = endpoint_nodes.into_iter().collect();
            let edge_set: HashSet<EdgeId> = selected_edges.iter().copied().collect();
            let core_subgraph = groggy::core::subgraph::Subgraph::new(
                self.graph.clone(),
                node_set,
                edge_set,
                "edge_slice_selection".to_string(),
            );
            let subgraph = PySubgraph::from_core_subgraph(core_subgraph)?;

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // If none of the above worked, return error
        Err(PyTypeError::new_err(
            "Edge index must be int, list of ints, or slice. \
            Examples: g.edges[0], g.edges[0:10]. For attribute access use g.edges.weight syntax.",
        ))
    }

    /// Support iteration: for edge_view in g.edges
    fn __iter__(&self, _py: Python) -> PyResult<EdgesIterator> {
        let edge_ids = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            let graph = self.graph.borrow();
            graph.edge_ids()
        };

        Ok(EdgesIterator {
            graph: self.graph.clone(),
            edge_ids,
            index: 0,
        })
    }

    /// Support len(g.edges)
    fn __len__(&self, _py: Python) -> PyResult<usize> {
        if let Some(ref constrained) = self.constrained_edges {
            Ok(constrained.len())
        } else {
            let graph = self.graph.borrow();
            Ok(graph.edge_ids().len())
        }
    }

    /// String representation
    fn __str__(&self, _py: Python) -> PyResult<String> {
        let graph = self.graph.borrow();
        let count = graph.edge_ids().len();
        Ok(format!("EdgesAccessor({} edges)", count))
    }

    /// Get all unique attribute names across all edges
    #[getter]
    fn attributes(&self, _py: Python) -> PyResult<Vec<String>> {
        let graph = self.graph.borrow();
        let mut all_attrs = std::collections::HashSet::new();

        // Determine which edges to check
        let edge_ids = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            // Get all edge IDs from the graph
            {
                graph.edge_ids()
            }
        };

        // Collect attributes from all edges
        for &edge_id in &edge_ids {
            let attrs: Vec<String> = graph
                .get_edge_attrs(edge_id)
                .map(|map| map.keys().cloned().collect())
                .unwrap_or_default();
            for attr in attrs {
                all_attrs.insert(attr);
            }
        }

        // Convert to sorted vector
        let mut result: Vec<String> = all_attrs.into_iter().collect();
        result.sort();
        Ok(result)
    }

    /// Get table view of edges (GraphTable with edge attributes) - DELEGATED to SubgraphOperations
    pub fn table(&self, py: Python) -> PyResult<PyObject> {
        use crate::ffi::core::table::PyGraphTable;

        // Pure delegation to SubgraphOperations through graph's as_subgraph() method
        let graph = self.graph.borrow();

        if let Some(ref constrained) = self.constrained_edges {
            // Subgraph case: create subgraph with constrained edges and delegate
            let all_nodes = graph
                .node_ids()
                .into_iter()
                .collect::<std::collections::HashSet<_>>();
            let constrained_set = constrained
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>();

            // Create subgraph and delegate to SubgraphOperations::edges_table()
            let subgraph = groggy::core::subgraph::Subgraph::new(
                self.graph.clone(),
                all_nodes,
                constrained_set,
                "edges_accessor_subgraph".to_string(),
            );

            // Delegate to SubgraphOperations trait
            let subgraph_ops: &dyn groggy::core::traits::SubgraphOperations = &subgraph;
            let core_table = subgraph_ops
                .edges_table()
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            let py_table = PyGraphTable::from_graph_table(core_table);
            Ok(Py::new(py, py_table)?.to_object(py))
        } else {
            // Full graph case: create full subgraph and delegate
            let all_nodes = graph
                .node_ids()
                .into_iter()
                .collect::<std::collections::HashSet<_>>();
            let all_edges = graph
                .edge_ids()
                .into_iter()
                .collect::<std::collections::HashSet<_>>();

            let subgraph = groggy::core::subgraph::Subgraph::new(
                self.graph.clone(),
                all_nodes,
                all_edges,
                "full_edges_table".to_string(),
            );

            // Delegate to SubgraphOperations trait
            let subgraph_ops: &dyn groggy::core::traits::SubgraphOperations = &subgraph;
            let core_table = subgraph_ops
                .edges_table()
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            let py_table = PyGraphTable::from_graph_table(core_table);
            Ok(Py::new(py, py_table)?.to_object(py))
        }
    }

    /// Get all edges as a subgraph (equivalent to g.edges[:]) - DELEGATED to SubgraphOperations
    /// Returns a subgraph containing all nodes that are connected by the edges and all edges
    fn all(&self, _py: Python) -> PyResult<PySubgraph> {
        let graph = self.graph.borrow();

        // Get edge IDs based on constraint
        let all_edge_ids = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            graph.edge_ids()
        };

        // Get all endpoint nodes from these edges
        let mut endpoint_nodes = std::collections::HashSet::new();
        for &edge_id in &all_edge_ids {
            if let Ok((source, target)) = graph.edge_endpoints(edge_id) {
                endpoint_nodes.insert(source);
                endpoint_nodes.insert(target);
            }
        }

        // Create subgraph with nodes and edges
        let edge_set: HashSet<EdgeId> = all_edge_ids.iter().copied().collect();
        let core_subgraph = groggy::core::subgraph::Subgraph::new(
            self.graph.clone(),
            endpoint_nodes,
            edge_set,
            "all_edges".to_string(),
        );
        PySubgraph::from_core_subgraph(core_subgraph)
    }

    /// Support property-style attribute access: g.edges.weight
    fn __getattr__(&self, _py: Python, name: &str) -> PyResult<PyObject> {
        // TODO: Complete implementation - temporarily disabled
        Err(PyKeyError::new_err(format!(
            "Edge attribute '{}' access is under development.",
            name
        )))
    }

    /// Get edge attribute column with proper error checking
    fn _get_edge_attribute_column(&self, py: Python, attr_name: &str) -> PyResult<PyObject> {
        let graph = self.graph.borrow();

        // Determine which edges to check
        let edge_ids = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            {
                graph.edge_ids()
            }
        };

        if edge_ids.is_empty() {
            return Err(PyValueError::new_err(format!(
                "Cannot access attribute '{}': No edges available",
                attr_name
            )));
        }

        // Check if the attribute exists on ANY edge
        let mut attribute_exists = false;
        for &edge_id in &edge_ids {
            let attrs: Vec<String> = graph
                .get_edge_attrs(edge_id)
                .map(|map| map.keys().cloned().collect())
                .unwrap_or_default();
            if attrs.contains(&attr_name.to_string()) {
                attribute_exists = true;
                break;
            }
        }

        if !attribute_exists {
            return Err(PyKeyError::new_err(format!(
                "Attribute '{}' does not exist on any edges. Available attributes: {:?}",
                attr_name,
                {
                    let mut all_attrs = std::collections::HashSet::new();
                    for &edge_id in &edge_ids {
                        let attrs: Vec<String> = graph
                            .get_edge_attrs(edge_id)
                            .map(|map| map.keys().cloned().collect())
                            .unwrap_or_default();
                        for attr in attrs {
                            all_attrs.insert(attr);
                        }
                    }
                    let mut result: Vec<String> = all_attrs.into_iter().collect();
                    result.sort();
                    result
                }
            )));
        }

        // Collect attribute values - allow None for edges without the attribute
        let mut values: Vec<Option<PyObject>> = Vec::new();
        for &edge_id in &edge_ids {
            match graph.get_edge_attr(edge_id, &attr_name.to_string()) {
                Ok(Some(value)) => {
                    // Convert the attribute value to Python object
                    let py_value = attr_value_to_python_value(py, &value)?;
                    values.push(Some(py_value));
                }
                Ok(None) => {
                    values.push(None);
                }
                Err(_) => {
                    values.push(None);
                }
            }
        }

        // Convert to Python list
        let py_values: Vec<PyObject> = values
            .into_iter()
            .map(|opt_val| match opt_val {
                Some(val) => val,
                None => py.None(),
            })
            .collect();

        Ok(py_values.to_object(py))
    }
}

impl PyEdgesAccessor {
    /// Set attributes for multiple edges (bulk operation) - internal method callable from Rust
    pub fn set_attrs_internal(&self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        use crate::ffi::api::graph_attributes::PyGraphAttrMut;

        // Create a mutable graph attributes handler
        let mut attr_handler = PyGraphAttrMut::new(self.graph.clone());

        // If this accessor is constrained to specific edges, validate
        if let Some(ref constrained_edges) = self.constrained_edges {
            let constrained_set: std::collections::HashSet<EdgeId> =
                constrained_edges.iter().copied().collect();

            for (attr_name_py, edge_values_py) in attrs_dict.iter() {
                let _attr_name: String = attr_name_py.extract()?;

                if let Ok(edge_dict) = edge_values_py.extract::<&pyo3::types::PyDict>() {
                    for (edge_py, _value_py) in edge_dict.iter() {
                        let edge_id: EdgeId = edge_py.extract()?;
                        if !constrained_set.contains(&edge_id) {
                            return Err(pyo3::exceptions::PyPermissionError::new_err(format!(
                                "Cannot set attribute for edge {} - not in this subgraph/view",
                                edge_id
                            )));
                        }
                    }
                }
            }
        }

        attr_handler.set_edge_attrs(py, attrs_dict)
    }
}
