//! Accessors FFI Bindings
//!
//! Python bindings for smart indexing accessors.

use groggy::{AttrValue, EdgeId, NodeId};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};
use std::collections::HashSet;

// Import utils for conversion functions
use crate::ffi::utils::python_value_to_attr_value;

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
use crate::ffi::subgraphs::subgraph::PySubgraph;

/// Smart helper function to create appropriate Node entity from core Graph
/// Returns PyMetaNode for meta-nodes, PyNode for regular nodes
fn create_node_entity_from_core(
    graph: std::rc::Rc<std::cell::RefCell<groggy::Graph>>,
    py: Python,
    node_id: NodeId,
) -> PyResult<PyObject> {
    // Check if this is a meta-node by examining entity_type attribute
    let is_meta_node = {
        let graph_ref = graph.borrow();
        match graph_ref.get_node_attr(node_id, &"entity_type".into()) {
            Ok(Some(attr_value)) => {
                match attr_value {
                    groggy::AttrValue::Text(s) => s == "meta",
                    groggy::AttrValue::CompactText(s) => s.as_str() == "meta",
                    _ => false,
                }
            }
            _ => false,
        }
    };
    
    if is_meta_node {
        // Create PyMetaNode entity wrapper
        use crate::ffi::entities::PyMetaNode;
        use groggy::entities::MetaNode;
        
        match MetaNode::new(node_id, graph) {
            Ok(meta_node) => {
                let py_meta_node = PyMetaNode::from_meta_node(meta_node);
                Ok(Py::new(py, py_meta_node)?.to_object(py))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to create MetaNode entity: {}", e
            ))),
        }
    } else {
        // Create PyNode entity wrapper
        use crate::ffi::entities::PyNode;
        use groggy::entities::Node;
        
        match Node::new(node_id, graph) {
            Ok(node) => {
                let py_node = PyNode::from_node(node);
                Ok(Py::new(py, py_node)?.to_object(py))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to create Node entity: {}", e
            ))),
        }
    }
}


/// Smart helper function to create appropriate Edge entity from core Graph
/// Returns PyMetaEdge for meta-edges, PyEdge for regular edges
fn create_edge_entity_from_core(
    graph: std::rc::Rc<std::cell::RefCell<groggy::Graph>>,
    py: Python,
    edge_id: EdgeId,
) -> PyResult<PyObject> {
    // Check if edge exists first
    {
        let graph_ref = graph.borrow();
        if !graph_ref.has_edge(edge_id) {
            return Err(PyKeyError::new_err(format!("Edge {} not found", edge_id)));
        }
    }
    
    // Check if this is a meta-edge by examining entity_type attribute
    let is_meta_edge = {
        let graph_ref = graph.borrow();
        match graph_ref.get_edge_attr(edge_id, &"entity_type".into()) {
            Ok(Some(attr_value)) => {
                match attr_value {
                    groggy::AttrValue::Text(s) => s == "meta",
                    groggy::AttrValue::CompactText(s) => s.as_str() == "meta",
                    _ => false,
                }
            }
            _ => false,
        }
    };
    
    if is_meta_edge {
        // Create PyMetaEdge entity wrapper
        use crate::ffi::entities::PyMetaEdge;
        use groggy::entities::MetaEdge;
        
        match MetaEdge::new(edge_id, graph) {
            Ok(meta_edge) => {
                let py_meta_edge = PyMetaEdge::from_meta_edge(meta_edge);
                Ok(Py::new(py, py_meta_edge)?.to_object(py))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to create MetaEdge entity: {}", e
            ))),
        }
    } else {
        // Create PyEdge entity wrapper
        use crate::ffi::entities::PyEdge;
        use groggy::entities::Edge;
        
        match Edge::new(edge_id, graph) {
            Ok(edge) => {
                let py_edge = PyEdge::from_edge(edge);
                Ok(Py::new(py, py_edge)?.to_object(py))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to create Edge entity: {}", e
            ))),
        }
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

            // Use smart entity creation to return appropriate type
            let node_entity = create_node_entity_from_core(self.graph.clone(), py, node_id)?;
            Ok(Some(node_entity))
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

            let node_entity = create_node_entity_from_core(self.graph.clone(), py, actual_node_id)?;
            return Ok(node_entity);
        }

        // Try to extract as GraphArray (indexed boolean lookup) - CHECK FIRST
        if let Ok(graph_array) = key.extract::<PyRef<crate::ffi::storage::array::PyGraphArray>>() {
            let graph = self.graph.borrow();
            let all_node_ids = if let Some(ref constrained) = self.constrained_nodes {
                constrained.clone()
            } else {
                graph.node_ids()
            };

            let mut selected_nodes = Vec::new();

            // Use GraphArray as indexed lookup: for each node_id, check grapharray[node_id]
            // The table is indexed so that row index = node_id
            for &node_id in &all_node_ids {
                // Use node_id as index into GraphArray to get boolean value
                if let Some(attr_value) = graph_array.inner.get(node_id as usize) {
                    if let groggy::AttrValue::Bool(true) = attr_value {
                        selected_nodes.push(node_id);
                    }
                }
            }

            if selected_nodes.is_empty() {
                return Err(PyIndexError::new_err(
                    "GraphArray boolean mask selected no nodes",
                ));
            }

            // Create subgraph with selected nodes and their edges
            let graph = self.graph.borrow();
            let mut edge_ids = std::collections::HashSet::new();
            for &node_id in &selected_nodes {
                if let Ok(incident) = graph.incident_edges(node_id) {
                    for edge_id in incident {
                        // Only include edges where both endpoints are in selected_nodes
                        if let Ok((source, target)) = graph.edge_endpoints(edge_id) {
                            if selected_nodes.contains(&source) && selected_nodes.contains(&target)
                            {
                                edge_ids.insert(edge_id);
                            }
                        }
                    }
                }
            }
            drop(graph);

            let subgraph = groggy::subgraphs::Subgraph::new(
                self.graph.clone(),
                selected_nodes.into_iter().collect(),
                edge_ids,
                "boolean_filtered".to_string(),
            );

            let py_subgraph = crate::ffi::subgraphs::subgraph::PySubgraph::from_core_subgraph(subgraph)?;
            return Ok(Py::new(py, py_subgraph)?.to_object(py));
        }

        // Try to extract as boolean array/list (positional boolean indexing) - FALLBACK
        if let Ok(boolean_mask) = key.extract::<Vec<bool>>() {
            let graph = self.graph.borrow();
            let all_node_ids = if let Some(ref constrained) = self.constrained_nodes {
                constrained.clone()
            } else {
                graph.node_ids()
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
            let core_subgraph = groggy::subgraphs::Subgraph::new(
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
            let core_subgraph = groggy::subgraphs::Subgraph::new(
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
            let core_subgraph = groggy::subgraphs::Subgraph::new(
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
            // Use index-aligned attribute access to match table behavior
            return self._get_node_attribute_column(py, &attr_name);
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

    /// Get table view of nodes (GraphTable with node attributes) with optional auto-slicing
    /// 
    /// Args:
    ///     auto_slice: If True (default), automatically exclude columns that are all NaN/None
    ///                 for the current node set. If False, include all columns.
    /// 
    /// For filtered accessors (base/meta), auto_slice=True by default to exclude 
    /// irrelevant attributes. For full accessors, auto_slice=False by default.
    #[pyo3(signature = (auto_slice = None))]
    pub fn table(&self, py: Python, auto_slice: Option<bool>) -> PyResult<PyObject> {
        use crate::ffi::storage::table::PyGraphTable;

        // Determine default auto_slice behavior based on accessor type
        let should_auto_slice = match auto_slice {
            Some(explicit_value) => explicit_value,
            None => {
                // Auto-slice by default for filtered accessors (base/meta)
                // This is determined by checking if we're constrained to specific nodes
                self.constrained_nodes.is_some()
            }
        };

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
            let subgraph = groggy::subgraphs::Subgraph::new(
                self.graph.clone(),
                constrained_set.clone(),
                all_edges, // TODO: Should be induced edges, but this maintains current behavior
                "nodes_accessor_subgraph".to_string(),
            );

            // Delegate to SubgraphOperations trait
            let subgraph_ops: &dyn groggy::traits::SubgraphOperations = &subgraph;
            let mut core_table = subgraph_ops
                .nodes_table()
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            
            // Apply auto-slicing if enabled
            if should_auto_slice {
                core_table = self.apply_node_auto_slice(core_table, &constrained_set)?;
            }
            
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

            let all_nodes_clone = all_nodes.clone();
            let subgraph = groggy::subgraphs::Subgraph::new(
                self.graph.clone(),
                all_nodes,
                all_edges,
                "full_nodes_table".to_string(),
            );

            // Delegate to SubgraphOperations trait
            let subgraph_ops: &dyn groggy::traits::SubgraphOperations = &subgraph;
            let mut core_table = subgraph_ops
                .nodes_table()
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            
            // Apply auto-slicing if enabled (for full graph, should_auto_slice is false by default)
            if should_auto_slice {
                core_table = self.apply_node_auto_slice(core_table, &all_nodes_clone)?;
            }
            
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
            groggy::subgraphs::Subgraph::calculate_induced_edges(&self.graph, &node_set)
                .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{:?}", e)))?;

        let core_subgraph = groggy::subgraphs::Subgraph::new(
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

        // Create index-aligned attribute values where values[node_id] = attribute_value
        // This ensures g.nodes[g.nodes['attr'] == value] works correctly
        let max_node_id = if let Some(ref constrained) = self.constrained_nodes {
            constrained.iter().max().copied().unwrap_or(0)
        } else {
            node_ids.iter().max().copied().unwrap_or(0)
        };

        let mut values: Vec<Option<PyObject>> = vec![None; (max_node_id + 1) as usize];

        for &node_id in &node_ids {
            if graph.contains_node(node_id) {
                match graph.get_node_attr(node_id, &attr_name.to_string()) {
                    Ok(Some(value)) => {
                        // Convert the attribute value to Python object
                        let py_value = attr_value_to_python_value(py, &value)?;
                        values[node_id as usize] = Some(py_value);
                    }
                    Ok(None) => {
                        values[node_id as usize] = None;
                    }
                    Err(_) => {
                        values[node_id as usize] = None;
                    }
                }
            } else {
                values[node_id as usize] = None;
            }
        }

        // Convert to AttrValue vector for GraphArray
        let attr_values: Vec<groggy::AttrValue> = values
            .into_iter()
            .map(|opt_val| match opt_val {
                Some(val) => {
                    // Convert Python object back to AttrValue
                    python_value_to_attr_value(val.as_ref(py)).unwrap_or(groggy::AttrValue::Null)
                }
                None => groggy::AttrValue::Null,
            })
            .collect();

        // Create GraphArray and wrap in Python
        let graph_array = groggy::GraphArray::from_vec(attr_values);
        let py_array = crate::ffi::storage::array::PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_array)?.to_object(py))
    }

    /// Access all subgraph-nodes (meta-nodes) in the graph
    /// 
    /// Returns:
    ///     PyGraphArray: Array of nodes that contain subgraphs
    /// 
    /// Example:
    ///     subgraph_nodes = g.nodes.subgraphs
    ///     for meta_node in subgraph_nodes:
    ///         print(f"Meta-node {meta_node.id} contains subgraph")
    #[getter]
    fn subgraphs(&self, py: Python) -> PyResult<PyObject> {
        // Find all nodes that have a 'contained_subgraph' attribute
        let graph = self.graph.borrow();
        let mut subgraph_node_ids = Vec::new();

        // Check which nodes to iterate over
        let node_ids: Vec<NodeId> = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            graph.node_ids()
        };

        // Find nodes with contained_subgraph attribute
        for node_id in node_ids {
            if let Ok(Some(AttrValue::SubgraphRef(_))) = 
                graph.get_node_attr(node_id, &"contained_subgraph".into()) 
            {
                subgraph_node_ids.push(AttrValue::Int(node_id as i64));
            }
        }

        // Return as GraphArray
        let graph_array = groggy::GraphArray::from_vec(subgraph_node_ids);
        let py_array = crate::ffi::storage::array::PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_array)?.to_object(py))
    }

    /// Get a MetaNode object if the specified node is a meta-node
    /// 
    /// Args:
    ///     node_id: The node ID to check
    /// 
    /// Returns:
    ///     PyMetaNode if the node is a meta-node, None otherwise
    /// 
    /// Example:
    ///     meta_node = g.nodes.get_meta_node(3)
    ///     if meta_node:
    ///         subgraph = meta_node.subgraph
    fn get_meta_node(&self, py: Python, node_id: NodeId) -> PyResult<Option<PyObject>> {
        use crate::ffi::subgraphs::hierarchical::PyMetaNode;
        use groggy::subgraphs::MetaNode;
        
        // Check if this node is a meta-node by checking entity_type
        let graph = self.graph.borrow();
        let entity_type_name = "entity_type".to_string();
        if let Ok(Some(entity_type_attr)) = graph.get_node_attr(node_id, &entity_type_name) {
            // Check both Text and CompactText variants
            let entity_type_str = match entity_type_attr {
                AttrValue::Text(s) => s,
                AttrValue::CompactText(s) => s.as_str().to_string(),
                _ => {
                    return Ok(None); // Not a string type, not a meta-node
                }
            };
            
            if entity_type_str == "meta" {
                // This is a meta-node, create PyMetaNode
                drop(graph); // Release borrow before creating MetaNode
                
                match MetaNode::new(node_id, self.graph.clone()) {
                    Ok(meta_node) => {
                        let py_meta_node = PyMetaNode::from_meta_node(meta_node);
                        return Ok(Some(Py::new(py, py_meta_node)?.to_object(py)));
                    }
                    Err(e) => {
                        return Err(PyValueError::new_err(format!(
                            "Failed to create MetaNode: {}", e
                        )));
                    }
                }
            }
        }
        
        // Not a meta-node
        Ok(None)
    }

    /// Get filtered accessor for base (non-meta) nodes only
    /// 
    /// Returns:
    ///     PyNodesAccessor: Accessor that only shows base nodes (entity_type != 'meta')
    /// 
    /// Example:
    ///     base_nodes = g.nodes.base
    ///     base_count = len(base_nodes)
    ///     base_table = base_nodes.table()
    #[getter]
    fn base(&self, py: Python) -> PyResult<PyObject> {
        // Get all base nodes (entity_type != 'meta')
        let base_node_ids = self.get_filtered_node_ids("base")?;
        
        // Create new constrained accessor
        let base_accessor = PyNodesAccessor {
            graph: self.graph.clone(),
            constrained_nodes: Some(base_node_ids),
        };
        
        Ok(Py::new(py, base_accessor)?.to_object(py))
    }

    /// Get filtered accessor for meta-nodes only
    /// 
    /// Returns:
    ///     PyNodesAccessor: Accessor that only shows meta-nodes (entity_type == 'meta')
    /// 
    /// Example:
    ///     meta_nodes = g.nodes.meta
    ///     meta_count = len(meta_nodes)
    ///     meta_table = meta_nodes.table()
    #[getter]
    fn meta(&self, py: Python) -> PyResult<PyObject> {
        // Get all meta-nodes (entity_type == 'meta')
        let meta_node_ids = self.get_filtered_node_ids("meta")?;
        
        // Create new constrained accessor
        let meta_accessor = PyNodesAccessor {
            graph: self.graph.clone(),
            constrained_nodes: Some(meta_node_ids),
        };
        
        Ok(Py::new(py, meta_accessor)?.to_object(py))
    }
}

impl PyNodesAccessor {
    /// Get filtered node IDs based on entity type
    /// 
    /// Args:
    ///     filter_type: "base" for non-meta nodes, "meta" for meta-nodes
    ///     
    /// Returns:
    ///     Vec<NodeId>: List of node IDs matching the filter criteria
    fn get_filtered_node_ids(&self, filter_type: &str) -> PyResult<Vec<NodeId>> {
        let graph = self.graph.borrow();
        let entity_type_name = "entity_type".to_string();
        let mut filtered_nodes = Vec::new();
        
        // Determine which nodes to check based on existing constraints
        let nodes_to_check = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            graph.node_ids()
        };
        
        for node_id in nodes_to_check {
            // Check the entity_type attribute
            if let Ok(Some(entity_type_attr)) = graph.get_node_attr(node_id, &entity_type_name) {
                // Handle both Text and CompactText variants
                let entity_type_str = match entity_type_attr {
                    AttrValue::Text(s) => s,
                    AttrValue::CompactText(s) => s.as_str().to_string(),
                    _ => continue, // Skip nodes with non-string entity_type
                };
                
                // Filter based on requested type
                let matches_filter = match filter_type {
                    "base" => entity_type_str != "meta", // Base = everything except meta
                    "meta" => entity_type_str == "meta", // Meta = only meta nodes
                    _ => false, // Unknown filter type
                };
                
                if matches_filter {
                    filtered_nodes.push(node_id);
                }
            } else if filter_type == "base" {
                // Nodes without entity_type attribute are considered base nodes
                filtered_nodes.push(node_id);
            }
        }
        
        Ok(filtered_nodes)
    }

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

    /// Apply auto-slicing to remove columns that are all NaN/None for the given node set
    fn apply_node_auto_slice(
        &self, 
        table: groggy::storage::table::GraphTable, 
        node_set: &std::collections::HashSet<groggy::NodeId>
    ) -> PyResult<groggy::storage::table::GraphTable> {
        let graph = self.graph.borrow();
        
        // Get all attribute names in the table
        let column_names = table.columns(); 
        let mut columns_to_keep = Vec::new();
        
        for column_name in column_names {
            // Check if this column has any non-null values for the nodes in our set
            let mut has_non_null_value = false;
            
            for &node_id in node_set {
                // Try to get the attribute value for this node
                match graph.get_node_attr(node_id, &groggy::AttrName::from(column_name.clone())) {
                    Ok(Some(attr_value)) => {
                        // Check if the value is not null/none
                        if !matches!(attr_value, groggy::AttrValue::Null) {
                            has_non_null_value = true;
                            break;
                        }
                    }
                    Ok(None) => {
                        // No attribute value for this node - continue checking others
                        continue;
                    }
                    Err(_) => {
                        // Error getting attribute - continue checking others
                        continue;
                    }
                }
            }
            
            // If this column has at least one non-null value, keep it
            if has_non_null_value {
                columns_to_keep.push(column_name.as_str());
            }
        }
        
        // Create a new table with only the columns that have data
        if columns_to_keep.len() == column_names.len() {
            // All columns have data, return original table
            Ok(table)
        } else if columns_to_keep.is_empty() {
            // No columns have data, but we should keep the table structure
            // Return the original table to avoid breaking things
            Ok(table)
        } else {
            // Select only the columns with data
            table.select(&columns_to_keep).map_err(|e| {
                PyValueError::new_err(format!("Failed to slice table columns: {}", e))
            })
        }
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

            // Use smart entity creation to return appropriate type
            let edge_entity = create_edge_entity_from_core(self.graph.clone(), py, edge_id)?;
            Ok(Some(edge_entity))
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

            let edge_entity = create_edge_entity_from_core(self.graph.clone(), py, actual_edge_id)?;
            return Ok(edge_entity);
        }

        // Try to extract as GraphArray (indexed boolean lookup) - CHECK FIRST
        if let Ok(graph_array) = key.extract::<PyRef<crate::ffi::storage::array::PyGraphArray>>() {
            let graph = self.graph.borrow();
            let all_edge_ids = if let Some(ref constrained) = self.constrained_edges {
                constrained.clone()
            } else {
                graph.edge_ids()
            };

            let mut selected_edges = Vec::new();

            // Use GraphArray as indexed lookup: for each edge_id, check grapharray[edge_id]
            // The table/accessor is indexed so that row index = edge_id
            for &edge_id in &all_edge_ids {
                // Use edge_id as index into GraphArray to get boolean value
                if let Some(attr_value) = graph_array.inner.get(edge_id as usize) {
                    if let groggy::AttrValue::Bool(true) = attr_value {
                        selected_edges.push(edge_id);
                    }
                }
            }

            if selected_edges.is_empty() {
                return Err(PyIndexError::new_err(
                    "GraphArray boolean mask selected no edges",
                ));
            }

            // Create subgraph with selected edges and their endpoint nodes
            let graph = self.graph.borrow();
            let mut node_ids = std::collections::HashSet::new();
            for &edge_id in &selected_edges {
                if let Ok((source, target)) = graph.edge_endpoints(edge_id) {
                    node_ids.insert(source);
                    node_ids.insert(target);
                }
            }
            drop(graph);

            let subgraph = groggy::subgraphs::Subgraph::new(
                self.graph.clone(),
                node_ids.clone(),
                selected_edges.into_iter().collect(),
                "boolean_filtered".to_string(),
            );

            let py_subgraph = crate::ffi::subgraphs::subgraph::PySubgraph::from_core_subgraph(subgraph)?;
            return Ok(Py::new(py, py_subgraph)?.to_object(py));
        }

        // Try to extract as boolean array/list (positional boolean indexing) - FALLBACK
        if let Ok(boolean_mask) = key.extract::<Vec<bool>>() {
            let graph = self.graph.borrow();
            let all_edge_ids = if let Some(ref constrained) = self.constrained_edges {
                constrained.clone()
            } else {
                graph.edge_ids()
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
            let core_subgraph = groggy::subgraphs::Subgraph::new(
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
            let core_subgraph = groggy::subgraphs::Subgraph::new(
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
            let core_subgraph = groggy::subgraphs::Subgraph::new(
                self.graph.clone(),
                node_set,
                edge_set,
                "edge_slice_selection".to_string(),
            );
            let subgraph = PySubgraph::from_core_subgraph(core_subgraph)?;

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // Try to extract as string (attribute name access)
        if let Ok(attr_name) = key.extract::<String>() {
            // Use edge attribute column access to match nodes behavior
            return self._get_edge_attribute_column(py, &attr_name);
        }

        // If none of the above worked, return error
        Err(PyTypeError::new_err(
            "Edge index must be int, list of ints, slice, or string attribute name. \
            Examples: g.edges[0], g.edges[0:10], g.edges['weight'], or g.edges.weight for attribute access.",
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

    /// Get table view of edges (GraphTable with edge attributes) with optional auto-slicing
    /// 
    /// Args:
    ///     auto_slice: If True (default), automatically exclude columns that are all NaN/None
    ///                 for the current edge set. If False, include all columns.
    /// 
    /// For filtered accessors (base/meta), auto_slice=True by default to exclude 
    /// irrelevant attributes. For full accessors, auto_slice=False by default.
    #[pyo3(signature = (auto_slice = None))]
    pub fn table(&self, py: Python, auto_slice: Option<bool>) -> PyResult<PyObject> {
        use crate::ffi::storage::table::PyGraphTable;

        // Determine default auto_slice behavior based on accessor type
        let should_auto_slice = match auto_slice {
            Some(explicit_value) => explicit_value,
            None => {
                // Auto-slice by default for filtered accessors (base/meta)
                // This is determined by checking if we're constrained to specific edges
                self.constrained_edges.is_some()
            }
        };

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
            let subgraph = groggy::subgraphs::Subgraph::new(
                self.graph.clone(),
                all_nodes,
                constrained_set.clone(),
                "edges_accessor_subgraph".to_string(),
            );

            // Delegate to SubgraphOperations trait
            let subgraph_ops: &dyn groggy::traits::SubgraphOperations = &subgraph;
            let mut core_table = subgraph_ops
                .edges_table()
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            
            // Apply auto-slicing if enabled
            if should_auto_slice {
                core_table = self.apply_edge_auto_slice(core_table, &constrained_set)?;
            }
            
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

            let all_edges_clone = all_edges.clone();
            let subgraph = groggy::subgraphs::Subgraph::new(
                self.graph.clone(),
                all_nodes,
                all_edges,
                "full_edges_table".to_string(),
            );

            // Delegate to SubgraphOperations trait
            let subgraph_ops: &dyn groggy::traits::SubgraphOperations = &subgraph;
            let mut core_table = subgraph_ops
                .edges_table()
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            
            // Apply auto-slicing if enabled (for full graph, should_auto_slice is false by default)
            if should_auto_slice {
                core_table = self.apply_edge_auto_slice(core_table, &all_edges_clone)?;
            }
            
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
        let core_subgraph = groggy::subgraphs::Subgraph::new(
            self.graph.clone(),
            endpoint_nodes,
            edge_set,
            "all_edges".to_string(),
        );
        PySubgraph::from_core_subgraph(core_subgraph)
    }

    /// Support property-style attribute access: g.edges.weight
    fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        // Delegate to edge attribute column access
        self._get_edge_attribute_column(py, name)
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

        // Create index-aligned attribute values where values[edge_id] = attribute_value
        // This ensures g.edges[g.edges['attr'] == value] works correctly
        let max_edge_id = if let Some(ref constrained) = self.constrained_edges {
            constrained.iter().max().copied().unwrap_or(0)
        } else {
            edge_ids.iter().max().copied().unwrap_or(0)
        };

        let mut values: Vec<Option<PyObject>> = vec![None; (max_edge_id + 1) as usize];

        for &edge_id in &edge_ids {
            match graph.get_edge_attr(edge_id, &attr_name.to_string()) {
                Ok(Some(value)) => {
                    // Convert the attribute value to Python object
                    let py_value = attr_value_to_python_value(py, &value)?;
                    values[edge_id as usize] = Some(py_value);
                }
                Ok(None) => {
                    values[edge_id as usize] = None;
                }
                Err(_) => {
                    values[edge_id as usize] = None;
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

    /// Get filtered accessor for base edges (non-meta edges)
    ///
    /// Returns a new EdgesAccessor that only shows edges where entity_type != 'meta'
    /// 
    /// Example:
    ///     base_edges = g.edges.base
    ///     base_count = len(base_edges)  
    ///     base_table = base_edges.table()
    #[getter]
    fn base(&self, py: Python) -> PyResult<PyObject> {
        // Get all base edges (entity_type != 'meta')
        let base_edge_ids = self.get_filtered_edge_ids("base")?;
        
        // Create new constrained accessor
        let base_accessor = PyEdgesAccessor {
            graph: self.graph.clone(),
            constrained_edges: Some(base_edge_ids),
        };
        
        Ok(Py::new(py, base_accessor)?.to_object(py))
    }

    /// Get filtered accessor for meta-edges
    ///
    /// Returns a new EdgesAccessor that only shows edges where entity_type == 'meta'
    ///
    /// Example:
    ///     meta_edges = g.edges.meta
    ///     meta_count = len(meta_edges)
    ///     meta_table = meta_edges.table()
    #[getter]
    fn meta(&self, py: Python) -> PyResult<PyObject> {
        // Get all meta-edges (entity_type == 'meta')
        let meta_edge_ids = self.get_filtered_edge_ids("meta")?;
        
        // Create new constrained accessor
        let meta_accessor = PyEdgesAccessor {
            graph: self.graph.clone(),
            constrained_edges: Some(meta_edge_ids),
        };
        
        Ok(Py::new(py, meta_accessor)?.to_object(py))
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

    /// Get filtered edge IDs based on entity_type attribute
    ///
    /// Args:
    ///     filter_type: "base" for non-meta edges, "meta" for meta-edges
    ///     
    /// Returns:
    ///     Vec<EdgeId>: List of edge IDs matching the filter criteria
    fn get_filtered_edge_ids(&self, filter_type: &str) -> PyResult<Vec<EdgeId>> {
        let graph = self.graph.borrow();
        let entity_type_name = "entity_type".to_string();
        let mut filtered_edges = Vec::new();
        
        // Determine which edges to check based on existing constraints
        let edge_ids_to_check = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            graph.edge_ids()
        };
        
        for edge_id in edge_ids_to_check {
            // Try to get entity_type attribute for this edge
            match graph.get_edge_attr(edge_id, &entity_type_name) {
                Ok(Some(attr_value)) => {
                    // Check the entity_type value (handle both Text and CompactText)
                    let entity_type_str = match &attr_value {
                        groggy::AttrValue::Text(text) => text.as_str(),
                        groggy::AttrValue::CompactText(compact) => compact.as_str(),
                        _ => continue, // Skip edges with non-string entity_type
                    };
                    
                    // Apply filter logic
                    let should_include = match filter_type {
                        "base" => entity_type_str != "meta", // Include non-meta edges
                        "meta" => entity_type_str == "meta", // Include only meta-edges
                        _ => return Err(PyValueError::new_err(format!(
                            "Invalid filter_type: {}. Must be 'base' or 'meta'",
                            filter_type
                        ))),
                    };
                    
                    if should_include {
                        filtered_edges.push(edge_id);
                    }
                }
                Ok(None) => {
                    // No entity_type attribute - treat as base edge (non-meta)
                    if filter_type == "base" {
                        filtered_edges.push(edge_id);
                    }
                }
                Err(_) => {
                    // Error getting attribute - treat as base edge (non-meta)
                    if filter_type == "base" {
                        filtered_edges.push(edge_id);
                    }
                }
            }
        }
        
        Ok(filtered_edges)
    }

    /// Apply auto-slicing to remove columns that are all NaN/None for the given edge set
    fn apply_edge_auto_slice(
        &self, 
        table: groggy::storage::table::GraphTable, 
        edge_set: &std::collections::HashSet<groggy::EdgeId>
    ) -> PyResult<groggy::storage::table::GraphTable> {
        let graph = self.graph.borrow();
        
        // Get all attribute names in the table
        let column_names = table.columns();
        let mut columns_to_keep = Vec::new();
        
        for column_name in column_names {
            // Check if this column has any non-null values for the edges in our set
            let mut has_non_null_value = false;
            
            for &edge_id in edge_set {
                // Try to get the attribute value for this edge
                match graph.get_edge_attr(edge_id, &column_name) {
                    Ok(Some(attr_value)) => {
                        // Check if the value is not null/none
                        if !matches!(attr_value, groggy::AttrValue::Null) {
                            has_non_null_value = true;
                            break;
                        }
                    }
                    Ok(None) => {
                        // No attribute value for this edge - continue checking others
                        continue;
                    }
                    Err(_) => {
                        // Error getting attribute - continue checking others
                        continue;
                    }
                }
            }
            
            // If this column has at least one non-null value, keep it
            if has_non_null_value {
                columns_to_keep.push(column_name.as_str());
            }
        }
        
        // Create a new table with only the columns that have data
        if columns_to_keep.len() == column_names.len() {
            // All columns have data, return original table
            Ok(table)
        } else if columns_to_keep.is_empty() {
            // No columns have data, but we should keep the table structure
            // Return the original table to avoid breaking things
            Ok(table)
        } else {
            // Select only the columns with data
            table.select(&columns_to_keep).map_err(|e| {
                PyValueError::new_err(format!("Failed to slice table columns: {}", e))
            })
        }
    }
}
