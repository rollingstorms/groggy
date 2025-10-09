//! Accessors FFI Bindings
//!
//! Python bindings for smart indexing accessors.

use groggy::storage::array::BaseArray; // Modern array types
use groggy::storage::table::Table; // Add Table trait for select method
use groggy::{AttrValue, EdgeId, NodeId};
use pyo3::exceptions::{
    PyAttributeError, PyIndexError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};
use std::collections::HashSet;

// Import utils for conversion functions
use crate::ffi::utils::python_value_to_attr_value;

// Import table FFI types

// Import modern array FFI types
use crate::ffi::storage::array::PyBaseArray;
use crate::ffi::storage::num_array::PyNumArray;

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
            Ok(Some(attr_value)) => match attr_value {
                groggy::AttrValue::Text(s) => s == "meta",
                groggy::AttrValue::CompactText(s) => s.as_str() == "meta",
                _ => false,
            },
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
                "Failed to create MetaNode entity: {}",
                e
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
                "Failed to create Node entity: {}",
                e
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
            Ok(Some(attr_value)) => match attr_value {
                groggy::AttrValue::Text(s) => s == "meta",
                groggy::AttrValue::CompactText(s) => s.as_str() == "meta",
                _ => false,
            },
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
                "Failed to create MetaEdge entity: {}",
                e
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
                "Failed to create Edge entity: {}",
                e
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
#[derive(Clone)]
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

    /// List attribute names available within this accessor's node set
    fn attribute_names(&self) -> Vec<String> {
        let graph_ref = self.graph.borrow();
        let node_ids: Vec<NodeId> = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            graph_ref.node_ids()
        };

        let mut names = HashSet::new();
        for node_id in node_ids {
            if let Ok(attrs) = graph_ref.get_node_attrs(node_id) {
                for attr_name in attrs.keys() {
                    names.insert(attr_name.clone());
                }
            }
        }

        let mut result: Vec<String> = names.into_iter().collect();
        result.sort();
        result
    }

    /// Filter nodes using the same syntax as graph.filter_nodes and return a subgraph
    #[pyo3(signature = (filter))]
    fn filter(&self, py: Python, filter: &PyAny) -> PyResult<PyObject> {
        use std::collections::HashSet;

        let graph_rc = self.graph.clone();

        // Determine the seed node set (respect constrained views)
        let seed_nodes: Vec<NodeId> = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            let graph_ref = self.graph.borrow();
            graph_ref.node_ids()
        };

        let node_set: HashSet<NodeId> = seed_nodes.into_iter().collect();

        let edge_set = groggy::subgraphs::Subgraph::calculate_induced_edges(&graph_rc, &node_set)
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;

        let base_subgraph = groggy::subgraphs::Subgraph::new(
            graph_rc.clone(),
            node_set,
            edge_set,
            "nodes_accessor".to_string(),
        );

        let py_subgraph = PySubgraph::from_core_subgraph(base_subgraph)?;
        let py_obj = Py::new(py, py_subgraph)?;
        let filter_obj = filter.to_object(py);
        py_obj
            .as_ref(py)
            .call_method1("filter_nodes", (filter_obj,))
            .map(|obj| obj.to_object(py))
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

        // Try to extract as BaseArray (indexed boolean lookup) - CHECK FIRST
        if let Ok(base_array) = key.extract::<PyRef<PyBaseArray>>() {
            let graph = self.graph.borrow();
            let all_node_ids = if let Some(ref constrained) = self.constrained_nodes {
                constrained.clone()
            } else {
                graph.node_ids()
            };

            let mut selected_nodes = Vec::new();

            // Use BaseArray with positional lookup: position in array matches position in node_ids
            // This aligns with how attribute arrays are built in _get_node_attribute_column
            for (index, &node_id) in all_node_ids.iter().enumerate() {
                // Use index position into BaseArray to get boolean value
                if let Some(groggy::AttrValue::Bool(true)) = base_array.inner.get(index) {
                    selected_nodes.push(node_id);
                }
            }

            if selected_nodes.is_empty() {
                return Err(PyIndexError::new_err(
                    "BaseArray boolean mask selected no nodes",
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

            let py_subgraph =
                crate::ffi::subgraphs::subgraph::PySubgraph::from_core_subgraph(subgraph)?;
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

    /// Support column assignment: g.nodes['attr'] = array
    /// Implements MVP column assignment with array length validation
    fn __setitem__(&self, py: Python, key: &PyAny, value: &PyAny) -> PyResult<()> {
        // Extract column name from key
        let attr_name = if let Ok(name) = key.extract::<String>() {
            name
        } else {
            return Err(PyTypeError::new_err(
                "Column key must be a string. Use g.nodes['column_name'] = values",
            ));
        };

        // Convert Python value to BaseArray<AttrValue>
        let attr_array = self.convert_python_to_base_array(py, value)?;

        // Get all node IDs (constrained or full)
        let node_ids = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            let graph = self.graph.borrow();
            graph.node_ids()
        };

        // Validate array length matches node count
        if attr_array.len() != node_ids.len() {
            return Err(PyValueError::new_err(format!(
                "Array length ({}) must match number of nodes ({})",
                attr_array.len(),
                node_ids.len()
            )));
        }

        // Optimize: Use existing set_attrs bulk method for better performance on larger datasets
        if node_ids.len() > 10 {
            // For larger datasets, use existing set_attrs bulk infrastructure (inherently atomic)
            // Create a Python dict with the format: {attr_name: {node_id: value, ...}}
            let py_dict = PyDict::new(py);
            let node_values_dict = PyDict::new(py);

            // Build the inner dict mapping node_id -> value
            for (i, &node_id) in node_ids.iter().enumerate() {
                if let Some(value) = attr_array.get(i) {
                    let py_value = attr_value_to_python_value(py, value)?;
                    node_values_dict.set_item(node_id, py_value)?;
                }
            }

            py_dict.set_item(&attr_name, node_values_dict)?;
            // The set_attrs_internal method provides atomic transaction semantics
            self.set_attrs_internal(py, py_dict)?;
        } else {
            // For small datasets, implement atomic transaction semantics manually
            // Phase 1: Pre-validate all operations will succeed (fail-fast)
            let mut updates = Vec::new();
            for (i, &node_id) in node_ids.iter().enumerate() {
                if let Some(value) = attr_array.get(i) {
                    // Pre-validate node exists
                    {
                        let graph = self.graph.borrow();
                        if !graph.contains_node(node_id) {
                            return Err(PyRuntimeError::new_err(format!(
                                "Node {} not found during atomic validation",
                                node_id
                            )));
                        }
                    }
                    updates.push((node_id, value.clone()));
                }
            }

            // Phase 2: Apply all updates atomically (all-or-nothing)
            let mut graph = self.graph.borrow_mut();
            for (node_id, value) in updates {
                graph
                    .set_node_attr(node_id, attr_name.clone(), value)
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Atomic transaction failed at node {}: {}",
                            node_id, e
                        ))
                    })?;
            }
            // If we reach here, all updates succeeded atomically
        }

        Ok(())
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

    /// Get a NodesTable representation of accessible nodes  
    /// Implements: g.nodes.table()
    pub fn table(&self) -> PyResult<crate::ffi::storage::table::PyNodesTable> {
        let nodes_table = if let Some(ref constrained) = self.constrained_nodes {
            // Constrained case: create table with only the specified nodes
            self.create_constrained_nodes_table(constrained)?
        } else {
            // Unconstrained case: use the full graph table
            let graph = self.graph.borrow();
            graph.nodes_table().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create nodes table: {}", e))
            })?
        };

        Ok(crate::ffi::storage::table::PyNodesTable { table: nodes_table })
    }

    /// Get node IDs as an IntArray for integer operations
    /// Implements: g.nodes.ids()
    pub fn ids(&self) -> PyResult<crate::ffi::storage::num_array::PyIntArray> {
        let node_ids = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            let graph = self.graph.borrow();
            graph.node_ids()
        };

        // Return as IntArray to preserve integer type for node IDs
        Ok(crate::ffi::storage::num_array::PyIntArray::from_node_ids(
            node_ids,
        ))
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

        // Create attribute values in node_ids order (same as table order)
        // This ensures g.nodes[g.nodes['attr'] == value] works correctly
        let mut values: Vec<Option<PyObject>> = Vec::with_capacity(node_ids.len());

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

        // Convert to AttrValue vector for BaseArray
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

        // Create BaseArray for mixed attribute data
        let base_array = BaseArray::from_attr_values(attr_values);
        let py_array = PyBaseArray { inner: base_array };
        Ok(Py::new(py, py_array)?.to_object(py))
    }

    /// Access all subgraph-nodes (meta-nodes) in the graph
    ///
    /// Returns:
    ///     PyNumArray: Array of node IDs that contain subgraphs
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

        // Return as NumArray (node IDs are numerical)
        let values: Vec<f64> = subgraph_node_ids
            .into_iter()
            .map(|attr_val| match attr_val {
                groggy::AttrValue::Int(id) => id as f64,
                _ => 0.0, // Fallback (shouldn't happen since we're only adding Int values)
            })
            .collect();

        let py_array = PyNumArray::new(values);
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
        use crate::ffi::entities::PyMetaNode;
        use groggy::entities::MetaNode;

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
                            "Failed to create MetaNode: {}",
                            e
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

    /// Convert node attributes to matrix
    /// Implements: g.nodes.matrix()
    fn matrix(&self, py: Python) -> PyResult<Py<crate::ffi::storage::matrix::PyGraphMatrix>> {
        use crate::ffi::storage::matrix::PyGraphMatrix;

        let graph_ref = self.graph.borrow();
        let matrix = graph_ref.to_matrix_f64().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Matrix conversion failed: {}", e))
        })?;

        Py::new(py, PyGraphMatrix { inner: matrix })
    }

    /// Create a NodesArray containing this single accessor (for delegation chaining)
    /// Implements: g.nodes.array() -> enables array operations and chaining
    fn array(&self, py: Python) -> PyResult<Py<crate::ffi::storage::nodes_array::PyNodesArray>> {
        use crate::ffi::storage::nodes_array::PyNodesArray;

        // Create a NodesArray with just this accessor to enable delegation
        let nodes_array = PyNodesArray::new(vec![self.clone()]);
        Py::new(py, nodes_array)
    }

    /// Group nodes by attribute value, returning SubgraphArray
    ///
    /// Args:
    ///     attr_name: Name of the node attribute to group by
    ///
    /// Returns:
    ///     SubgraphArray: Array of subgraphs, one for each unique attribute value
    ///
    /// Example:
    ///     dept_groups = g.nodes.group_by('department')
    ///     # Returns subgraphs for each department value
    pub fn group_by(
        &self,
        attr_names: &PyAny,
    ) -> PyResult<crate::ffi::storage::subgraph_array::PySubgraphArray> {
        use groggy::types::{AttrName, NodeId};
        use std::collections::HashMap;
        use std::collections::HashSet;

        // Parse attr_names as either String or Vec<String>
        let attr_name_list: Vec<String> = if let Ok(single) = attr_names.extract::<String>() {
            vec![single]
        } else if let Ok(multiple) = attr_names.extract::<Vec<String>>() {
            multiple
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "attr_names must be a string or list of strings",
            ));
        };

        let attr_names_typed: Vec<AttrName> = attr_name_list
            .iter()
            .map(|s| AttrName::from(s.clone()))
            .collect();

        let graph = self.graph.borrow();

        // Determine which nodes to group
        let node_ids = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            graph.node_ids()
        };

        if node_ids.is_empty() {
            return Ok(crate::ffi::storage::subgraph_array::PySubgraphArray::new(
                Vec::new(),
            ));
        }

        // Group nodes by composite attribute value(s)
        // For multi-column grouping, create a tuple of values as the key
        let mut groups: HashMap<Vec<groggy::types::AttrValue>, HashSet<NodeId>> = HashMap::new();

        for &node_id in &node_ids {
            let mut key_values = Vec::new();
            let mut has_all_attrs = true;

            for attr_name in &attr_names_typed {
                if let Ok(Some(attr_value)) = graph.get_node_attr(node_id, attr_name) {
                    key_values.push(attr_value);
                } else {
                    has_all_attrs = false;
                    break;
                }
            }

            if has_all_attrs {
                groups.entry(key_values).or_default().insert(node_id);
            }
            // Skip nodes without all the attributes
        }

        // Create subgraphs for each group - sort by attribute value for deterministic order
        let mut result_subgraphs = Vec::new();
        let mut sorted_groups: Vec<_> = groups.into_iter().collect();
        sorted_groups.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for (key_values, node_group) in sorted_groups {
            if !node_group.is_empty() {
                // Find induced edges for this group of nodes
                let mut induced_edges = HashSet::new();
                let all_edges = graph.edge_ids();

                for edge_id in all_edges {
                    if let Ok((source, target)) = graph.edge_endpoints(edge_id) {
                        if node_group.contains(&source) && node_group.contains(&target) {
                            induced_edges.insert(edge_id);
                        }
                    }
                }

                // Create subgraph with descriptive name
                let subgraph_name = if attr_name_list.len() == 1 {
                    format!("nodes_{}_group_{:?}", attr_name_list[0], key_values[0])
                } else {
                    format!("nodes_{}_group_{:?}", attr_name_list.join("_"), key_values)
                };

                let core_subgraph = groggy::subgraphs::Subgraph::new(
                    self.graph.clone(),
                    node_group,
                    induced_edges,
                    subgraph_name,
                );

                let py_subgraph =
                    crate::ffi::subgraphs::subgraph::PySubgraph::from_core_subgraph(core_subgraph)
                        .map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "Failed to create subgraph: {}",
                                e
                            ))
                        })?;

                result_subgraphs.push(py_subgraph);
            }
        }

        Ok(crate::ffi::storage::subgraph_array::PySubgraphArray::new(
            result_subgraphs,
        ))
    }

    /// Get viz accessor for visualization operations
    #[getter]
    fn viz(&self, py: Python) -> PyResult<Py<crate::ffi::viz_accessor::VizAccessor>> {
        // Create a subgraph from the constrained nodes for visualization
        let mut viz_graph = groggy::api::graph::Graph::new();

        // Get the constrained node IDs or all nodes if no constraint
        let node_ids = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            self.graph.borrow().node_ids()
        };

        // Add nodes to the viz graph (basic implementation for now)
        for _node_id in &node_ids {
            let _new_node_id = viz_graph.add_node();
            // TODO: Copy attributes from original nodes
        }

        let graph_data_source = groggy::viz::streaming::GraphDataSource::new(&viz_graph);
        let viz_accessor = crate::ffi::viz_accessor::VizAccessor::with_data_source(
            graph_data_source,
            "NodesAccessor".to_string(),
        );

        Py::new(py, viz_accessor)
    }
}

impl PyNodesAccessor {
    /// Create a NodesTable for only the constrained nodes
    fn create_constrained_nodes_table(
        &self,
        constrained_nodes: &[groggy::NodeId],
    ) -> PyResult<groggy::storage::table::NodesTable> {
        use groggy::storage::array::BaseArray;
        use groggy::storage::table::{BaseTable, NodesTable};
        use std::collections::HashMap;

        let graph = self.graph.borrow();

        // Collect all constrained nodes with their attributes
        let mut attribute_columns: HashMap<String, Vec<groggy::AttrValue>> = HashMap::new();

        // Initialize with node_id column
        attribute_columns.insert("node_id".to_string(), Vec::new());

        // First pass: collect all attribute names that exist on ANY of the constrained nodes
        let mut all_attr_names = std::collections::HashSet::new();
        all_attr_names.insert("node_id".to_string());

        for &node_id in constrained_nodes {
            if let Ok(attrs) = graph.get_node_attrs(node_id) {
                for attr_name in attrs.keys() {
                    all_attr_names.insert(attr_name.clone());
                }
            }
        }

        // Initialize all columns
        for attr_name in &all_attr_names {
            if !attribute_columns.contains_key(attr_name) {
                attribute_columns.insert(attr_name.clone(), Vec::new());
            }
        }

        // Second pass: collect data for each constrained node
        for &node_id in constrained_nodes {
            // Add node_id
            attribute_columns
                .get_mut("node_id")
                .unwrap()
                .push(groggy::AttrValue::Int(node_id as i64));

            // Get all attributes for this node
            let node_attrs = graph.get_node_attrs(node_id).unwrap_or_default();

            // For each expected attribute, add the value or null
            for attr_name in &all_attr_names {
                if attr_name == "node_id" {
                    continue; // Already handled
                }

                let attr_value = node_attrs
                    .get(attr_name)
                    .cloned()
                    .unwrap_or(groggy::AttrValue::Null);

                attribute_columns
                    .get_mut(attr_name)
                    .unwrap()
                    .push(attr_value);
            }
        }

        // Apply auto-slicing: remove columns that are all null for this node set
        let _node_set: std::collections::HashSet<groggy::NodeId> =
            constrained_nodes.iter().copied().collect();
        let mut columns_to_keep = Vec::new();

        for (attr_name, values) in &attribute_columns {
            if attr_name == "node_id" {
                // Always keep node_id column
                columns_to_keep.push(attr_name.clone());
            } else {
                // Check if this column has any non-null values
                let has_non_null = values.iter().any(|v| !matches!(v, groggy::AttrValue::Null));
                if has_non_null {
                    columns_to_keep.push(attr_name.clone());
                }
            }
        }

        // Create filtered attribute columns
        let mut filtered_columns = HashMap::new();
        for attr_name in columns_to_keep {
            if let Some(values) = attribute_columns.remove(&attr_name) {
                filtered_columns.insert(attr_name, BaseArray::from_attr_values(values));
            }
        }

        // Convert to BaseTable and then NodesTable
        let base_table = BaseTable::from_columns(filtered_columns)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create BaseTable: {}", e)))?;

        NodesTable::from_base_table(base_table)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert to NodesTable: {}", e)))
    }

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
                    _ => false,                          // Unknown filter type
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

        if attrs_dict.is_empty() {
            return Err(PyValueError::new_err(
                "attrs_dict is missing required attribute mappings",
            ));
        }

        // Normalize input so we always work with attribute-centric format:
        // {"attr": {node_id: value}}. Users can also pass
        // {node_id: {"attr": value}} and we'll adapt it here.
        // Note: holder is needed to keep normalized dict alive, though not directly read
        let mut _normalized_holder: Option<Py<PyDict>> = None;
        let mut attrs_dict_ref = attrs_dict;

        if let Some((first_key, _)) = attrs_dict.iter().next() {
            if first_key.extract::<String>().is_err() {
                let normalized = PyDict::new(py);

                for (node_py, attr_map_py) in attrs_dict.iter() {
                    let node_id: NodeId = node_py.extract()?;
                    let attr_map = attr_map_py.downcast::<PyDict>().map_err(|_| {
                        PyValueError::new_err(
                            "Expected dict of attributes per node when using node-centric format",
                        )
                    })?;

                    for (attr_name_py, value_py) in attr_map.iter() {
                        let attr_name: String = attr_name_py.extract()?;
                        let target_dict = match normalized.get_item(attr_name.clone())? {
                            Some(existing) => existing.downcast::<PyDict>().map_err(|_| {
                                PyValueError::new_err(
                                    "Attribute entries must be dictionaries of node/value pairs",
                                )
                            })?,
                            None => {
                                let new_dict = PyDict::new(py);
                                normalized.set_item(attr_name.clone(), new_dict)?;
                                new_dict
                            }
                        };

                        target_dict.set_item(node_id, value_py)?;
                    }
                }

                _normalized_holder = Some(normalized.into());
                attrs_dict_ref = _normalized_holder.as_ref().unwrap().as_ref(py);
            }
        }

        // Create a mutable graph attributes handler
        let mut attr_handler = PyGraphAttrMut::new(self.graph.clone());

        // If this accessor is constrained to specific nodes, we need to validate
        if let Some(ref constrained_nodes) = self.constrained_nodes {
            // Validate that all node IDs in attrs_dict are in our constrained set
            let constrained_set: std::collections::HashSet<NodeId> =
                constrained_nodes.iter().copied().collect();

            for (attr_name_py, node_values_py) in attrs_dict_ref.iter() {
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
        attr_handler.set_node_attrs(py, attrs_dict_ref)
    }

    /// Apply auto-slicing to remove columns that are all NaN/None for the given node set
    ///
    /// # Future Feature
    ///
    /// Designed for automatic column pruning in table operations. Currently unused but
    /// planned for future optimization passes.
    #[allow(dead_code)]
    fn apply_node_auto_slice(
        &self,
        table: groggy::storage::table::BaseTable,
        node_set: &std::collections::HashSet<groggy::NodeId>,
    ) -> PyResult<groggy::storage::table::BaseTable> {
        let graph = self.graph.borrow();

        // Get all attribute names in the table
        let column_names = table.column_names();
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
                columns_to_keep.push(column_name.clone());
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
            table
                .select(&columns_to_keep)
                .map_err(|e| PyValueError::new_err(format!("Failed to slice table columns: {}", e)))
        }
    }

    /// Get the number of nodes accessible by this accessor
    pub fn node_count(&self) -> usize {
        if let Some(ref constrained) = self.constrained_nodes {
            constrained.len()
        } else {
            let graph = self.graph.borrow();
            graph.node_ids().len()
        }
    }

    /// Convert nodes to SubgraphArray via connected components analysis
    pub fn to_subgraphs(&self) -> PyResult<crate::ffi::storage::subgraph_array::PySubgraphArray> {
        // TODO: Implement proper conversion from NodesAccessor to SubgraphArray
        // This requires complex subgraph creation logic
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "NodesAccessor to SubgraphArray conversion not yet implemented.",
        ))
    }

    /// Convert nodes to EdgesAccessor showing all edges connected to these nodes
    pub fn to_edges(&self) -> PyResult<crate::ffi::storage::accessors::PyEdgesAccessor> {
        use groggy::types::{EdgeId, NodeId};
        use std::collections::HashSet;

        let graph = self.graph.borrow();
        let node_ids: Vec<NodeId> = if let Some(ref constraint) = self.constrained_nodes {
            constraint.clone()
        } else {
            graph.node_ids()
        };

        let node_set: HashSet<NodeId> = node_ids.into_iter().collect();

        // Find all edges connected to any of these nodes
        let connected_edges: Vec<EdgeId> = graph
            .edge_ids()
            .into_iter()
            .filter(|&edge_id| {
                if let Ok((source, target)) = graph.edge_endpoints(edge_id) {
                    node_set.contains(&source) || node_set.contains(&target)
                } else {
                    false
                }
            })
            .collect();

        drop(graph); // Release the borrow before creating PyEdgesAccessor

        let edges_accessor = PyEdgesAccessor {
            graph: self.graph.clone(),
            constrained_edges: Some(connected_edges),
        };
        Ok(edges_accessor)
    }

    /// Convert Python value to BaseArray<AttrValue> for column assignment
    fn convert_python_to_base_array(
        &self,
        py: Python,
        value: &PyAny,
    ) -> PyResult<BaseArray<AttrValue>> {
        // Handle BaseArray directly
        if let Ok(base_array) = value.extract::<PyRef<PyBaseArray>>() {
            return Ok(base_array.inner.clone());
        }

        // Handle Python list
        if let Ok(list) = value.extract::<Vec<PyObject>>() {
            let mut attr_values = Vec::new();
            for py_obj in list {
                let attr_value = python_value_to_attr_value(py_obj.as_ref(py)).map_err(|e| {
                    PyValueError::new_err(format!("Failed to convert list element: {}", e))
                })?;
                attr_values.push(attr_value);
            }
            return Ok(BaseArray::from_attr_values(attr_values));
        }

        // Handle scalar broadcasting (single value for all nodes)
        let attr_value = python_value_to_attr_value(value)
            .map_err(|e| PyValueError::new_err(format!("Failed to convert scalar value: {}", e)))?;

        // Get node count for broadcasting
        let node_count = if let Some(ref constrained) = self.constrained_nodes {
            constrained.len()
        } else {
            let graph = self.graph.borrow();
            graph.node_ids().len()
        };

        // Create array with repeated value
        let attr_values = vec![attr_value; node_count];
        Ok(BaseArray::from_attr_values(attr_values))
    }

    /// Append a new node to the graph with attributes (graph-aware table.append)
    ///
    /// Creates a new node in the underlying graph and sets its attributes.
    /// This maintains consistency between the graph structure and node table.
    ///
    /// # Arguments
    /// * `attrs_dict` - Dictionary mapping attribute names to values
    ///
    /// # Returns
    /// NodeId of the newly created node
    ///
    /// # Examples
    /// ```python
    /// # Add a new node with attributes
    /// node_id = g.nodes.append({'name': 'Alice', 'age': 30})
    ///
    /// # Node is immediately available in both graph and table
    /// print(g.get_node_attr(node_id, 'name'))  # 'Alice'
    /// print(g.nodes.table())  # Shows Alice in the table
    /// ```
    pub fn append(&self, _py: Python, attrs_dict: &PyDict) -> PyResult<NodeId> {
        let mut graph = self.graph.borrow_mut();

        // Create a new node in the graph
        let node_id = graph.add_node();

        // Convert Python dict to attribute map
        let mut attributes = std::collections::HashMap::new();
        for (key, value) in attrs_dict.iter() {
            let attr_name: String = key.extract()?;
            let attr_value = python_value_to_attr_value(value)?;
            attributes.insert(attr_name, attr_value);
        }

        // Set attributes using the graph's bulk set method
        for (attr_name, attr_value) in attributes {
            graph
                .set_node_attr(node_id, attr_name.clone(), attr_value)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to set attribute '{}': {}",
                        attr_name, e
                    ))
                })?;
        }

        Ok(node_id)
    }

    /// Extend the graph with multiple new nodes and their attributes (graph-aware table.extend)
    ///
    /// Creates multiple nodes in the underlying graph and sets their attributes.
    /// This is more efficient than calling append multiple times.
    ///
    /// # Arguments
    /// * `rows_data` - List of dictionaries, each representing a node's attributes
    ///
    /// # Returns
    /// List of NodeIds for the newly created nodes
    ///
    /// # Examples
    /// ```python
    /// # Add multiple nodes with attributes
    /// node_ids = g.nodes.extend([
    ///     {'name': 'Alice', 'age': 30},
    ///     {'name': 'Bob', 'age': 25},
    ///     {'name': 'Carol', 'age': 35}
    /// ])
    ///
    /// # All nodes are immediately available
    /// print(len(g.nodes))  # Increased by 3
    /// ```
    pub fn extend(&self, _py: Python, rows_data: &pyo3::types::PyList) -> PyResult<Vec<NodeId>> {
        let mut graph = self.graph.borrow_mut();
        let mut new_node_ids = Vec::new();

        // Process each row
        for row_item in rows_data.iter() {
            let row_dict: &PyDict = row_item.downcast()?;

            // Create a new node in the graph
            let node_id = graph.add_node();

            // Convert Python dict to attribute map and set attributes
            for (key, value) in row_dict.iter() {
                let attr_name: String = key.extract()?;
                let attr_value = python_value_to_attr_value(value)?;
                graph
                    .set_node_attr(node_id, attr_name.clone(), attr_value)
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Failed to set attribute '{}' on node {}: {}",
                            attr_name, node_id, e
                        ))
                    })?;
            }

            new_node_ids.push(node_id);
        }

        Ok(new_node_ids)
    }

    /// Remove nodes from the graph by their indices in the table (graph-aware table.drop)
    ///
    /// Removes nodes from the underlying graph structure, which automatically
    /// cleans up their attributes and any connected edges.
    ///
    /// # Arguments
    /// * `indices` - List of row indices in the current table view to remove
    ///
    /// # Returns
    /// Number of nodes actually removed
    ///
    /// # Examples
    /// ```python
    /// # Remove nodes at table rows 0, 2, and 5
    /// removed_count = g.nodes.drop([0, 2, 5])
    ///
    /// # Nodes and their edges are gone from the graph
    /// print(f"Removed {removed_count} nodes")
    /// ```
    pub fn drop_rows(&self, indices: Vec<usize>) -> PyResult<usize> {
        if indices.is_empty() {
            return Ok(0);
        }

        // Get the current node list (considering constraints)
        let current_nodes = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            let graph = self.graph.borrow();
            graph.node_ids()
        };

        // Validate indices and collect node IDs to remove
        let mut nodes_to_remove = Vec::new();
        for &index in &indices {
            if index >= current_nodes.len() {
                return Err(PyIndexError::new_err(format!(
                    "Row index {} out of bounds for {} nodes",
                    index,
                    current_nodes.len()
                )));
            }
            nodes_to_remove.push(current_nodes[index]);
        }

        // Remove nodes from the graph (this also removes their edges and attributes)
        let mut graph = self.graph.borrow_mut();
        let mut removed_count = 0;

        for node_id in nodes_to_remove {
            match graph.remove_node(node_id) {
                Ok(_) => removed_count += 1,
                Err(e) => {
                    // Log warning but continue with other nodes
                    eprintln!("Warning: Failed to remove node {}: {}", node_id, e);
                }
            }
        }

        Ok(removed_count)
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
#[derive(Clone)]
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

    /// List attribute names available within this accessor's edge set
    fn attribute_names(&self) -> Vec<String> {
        let graph_ref = self.graph.borrow();
        let edge_ids: Vec<EdgeId> = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            graph_ref.edge_ids()
        };

        let mut names = HashSet::new();
        for edge_id in edge_ids {
            if let Ok(attrs) = graph_ref.get_edge_attrs(edge_id) {
                for attr_name in attrs.keys() {
                    names.insert(attr_name.clone());
                }
            }
        }

        let mut result: Vec<String> = names.into_iter().collect();
        result.sort();
        result
    }

    /// Filter edges using the same syntax as graph.filter_edges and return a subgraph
    #[pyo3(signature = (filter))]
    fn filter(&self, py: Python, filter: &PyAny) -> PyResult<PyObject> {
        use std::collections::HashSet;

        let graph_rc = self.graph.clone();

        let edge_ids: Vec<EdgeId> = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            let graph_ref = self.graph.borrow();
            graph_ref.edge_ids()
        };

        let edge_set: HashSet<EdgeId> = edge_ids.iter().copied().collect();

        let mut node_set: HashSet<NodeId> = HashSet::new();
        {
            let graph_ref = self.graph.borrow();
            for &edge_id in &edge_ids {
                if let Ok((source, target)) = graph_ref.edge_endpoints(edge_id) {
                    node_set.insert(source);
                    node_set.insert(target);
                }
            }
        }

        let base_subgraph = groggy::subgraphs::Subgraph::new(
            graph_rc.clone(),
            node_set,
            edge_set,
            "edges_accessor".to_string(),
        );

        let py_subgraph = PySubgraph::from_core_subgraph(base_subgraph)?;
        let py_obj = Py::new(py, py_subgraph)?;
        let filter_obj = filter.to_object(py);
        py_obj
            .as_ref(py)
            .call_method1("filter_edges", (filter_obj,))
            .map(|obj| obj.to_object(py))
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

        // Try to extract as BaseArray (indexed boolean lookup) - CHECK FIRST
        if let Ok(base_array) = key.extract::<PyRef<PyBaseArray>>() {
            let graph = self.graph.borrow();
            let all_edge_ids = if let Some(ref constrained) = self.constrained_edges {
                constrained.clone()
            } else {
                graph.edge_ids()
            };

            let mut selected_edges = Vec::new();

            // Use BaseArray with positional lookup: position in array matches position in edge_ids
            // This aligns with how attribute arrays are built in _get_edge_attribute_column
            for (index, &edge_id) in all_edge_ids.iter().enumerate() {
                // Use index position into BaseArray to get boolean value
                if let Some(groggy::AttrValue::Bool(true)) = base_array.inner.get(index) {
                    selected_edges.push(edge_id);
                }
            }

            if selected_edges.is_empty() {
                return Err(PyIndexError::new_err(
                    "BaseArray boolean mask selected no edges",
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

            let py_subgraph =
                crate::ffi::subgraphs::subgraph::PySubgraph::from_core_subgraph(subgraph)?;
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

    /// Support column assignment: g.edges['attr'] = array
    /// Implements MVP column assignment with array length validation
    fn __setitem__(&self, py: Python, key: &PyAny, value: &PyAny) -> PyResult<()> {
        // Extract column name from key
        let attr_name = if let Ok(name) = key.extract::<String>() {
            name
        } else {
            return Err(PyTypeError::new_err(
                "Column key must be a string. Use g.edges['column_name'] = values",
            ));
        };

        // Convert Python value to BaseArray<AttrValue>
        let attr_array = self.convert_python_to_base_array(py, value)?;

        // Get all edge IDs (constrained or full)
        let edge_ids = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            let graph = self.graph.borrow();
            graph.edge_ids()
        };

        // Validate array length matches edge count
        if attr_array.len() != edge_ids.len() {
            return Err(PyValueError::new_err(format!(
                "Array length ({}) must match number of edges ({})",
                attr_array.len(),
                edge_ids.len()
            )));
        }

        // Optimize: Use existing set_attrs bulk method for better performance on larger datasets
        if edge_ids.len() > 10 {
            // For larger datasets, use existing set_attrs bulk infrastructure (inherently atomic)
            // Create a Python dict with the format: {attr_name: {edge_id: value, ...}}
            let py_dict = PyDict::new(py);
            let edge_values_dict = PyDict::new(py);

            // Build the inner dict mapping edge_id -> value
            for (i, &edge_id) in edge_ids.iter().enumerate() {
                if let Some(value) = attr_array.get(i) {
                    let py_value = attr_value_to_python_value(py, value)?;
                    edge_values_dict.set_item(edge_id, py_value)?;
                }
            }

            py_dict.set_item(&attr_name, edge_values_dict)?;
            // The set_attrs_internal method provides atomic transaction semantics
            self.set_attrs_internal(py, py_dict)?;
        } else {
            // For small datasets, implement atomic transaction semantics manually
            // Phase 1: Pre-validate all operations will succeed (fail-fast)
            let mut updates = Vec::new();
            for (i, &edge_id) in edge_ids.iter().enumerate() {
                if let Some(value) = attr_array.get(i) {
                    // Pre-validate edge exists
                    {
                        let graph = self.graph.borrow();
                        if !graph.contains_edge(edge_id) {
                            return Err(PyRuntimeError::new_err(format!(
                                "Edge {} not found during atomic validation",
                                edge_id
                            )));
                        }
                    }
                    updates.push((edge_id, value.clone()));
                }
            }

            // Phase 2: Apply all updates atomically (all-or-nothing)
            let mut graph = self.graph.borrow_mut();
            for (edge_id, value) in updates {
                graph
                    .set_edge_attr(edge_id, attr_name.clone(), value)
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Atomic transaction failed at edge {}: {}",
                            edge_id, e
                        ))
                    })?;
            }
            // If we reach here, all updates succeeded atomically
        }

        Ok(())
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
        // Skip Python internal attributes to allow proper introspection
        if name.starts_with("__") && name.ends_with("__") {
            return Err(PyAttributeError::new_err(format!(
                "'EdgesAccessor' object has no attribute '{}'",
                name
            )));
        }

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

        // Create attribute values in edge_ids order (same as table order)
        // This ensures g.edges[g.edges['attr'] == value] works correctly
        let mut values: Vec<Option<PyObject>> = Vec::with_capacity(edge_ids.len());

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

        // Convert to AttrValue vector for BaseArray (consistent with nodes accessor)
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

        // Create BaseArray for mixed attribute data (consistent with nodes accessor)
        let base_array = BaseArray::from_attr_values(attr_values);
        let py_array = PyBaseArray { inner: base_array };
        Ok(Py::new(py, py_array)?.to_object(py))
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

    /// Get an EdgesTable representation of accessible edges
    /// Implements: g.edges.table()  
    pub fn table(&self) -> PyResult<crate::ffi::storage::table::PyEdgesTable> {
        let edges_table = if let Some(ref constrained) = self.constrained_edges {
            // Constrained case: create table with only the specified edges
            self.create_constrained_edges_table(constrained)?
        } else {
            // Unconstrained case: use the full graph table
            let graph = self.graph.borrow();
            graph.edges_table().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create edges table: {}", e))
            })?
        };

        Ok(crate::ffi::storage::table::PyEdgesTable { table: edges_table })
    }

    /// Get edge IDs as a NumArray for numerical operations
    /// Implements: g.edges.ids()
    pub fn ids(&self) -> PyResult<PyNumArray> {
        let edge_ids = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            let graph = self.graph.borrow();
            graph.edge_ids()
        };

        let values: Vec<i64> = edge_ids.into_iter().map(|id| id as i64).collect();
        Ok(PyNumArray::new_int64(values))
    }

    /// Get source node IDs for all edges
    /// Returns a NumArray parallel to edge_ids where each element is the source of the corresponding edge
    #[getter]
    fn sources(&self, _py: Python) -> PyResult<PyNumArray> {
        let graph = self.graph.borrow();

        let sources = if let Some(ref constrained_edges) = self.constrained_edges {
            // Constrained case: only get sources for the constrained edges
            let mut constrained_sources = Vec::new();
            for &edge_id in constrained_edges {
                if let Ok((source, _target)) = graph.edge_endpoints(edge_id) {
                    constrained_sources.push(source);
                }
            }
            constrained_sources
        } else {
            // Unconstrained case: get all edge sources
            graph.edge_sources()
        };

        let values: Vec<i64> = sources.into_iter().map(|id| id as i64).collect();
        Ok(PyNumArray::new_int64(values))
    }

    /// Get target node IDs for all edges  
    /// Returns a NumArray parallel to edge_ids where each element is the target of the corresponding edge
    #[getter]
    fn targets(&self, _py: Python) -> PyResult<PyNumArray> {
        let graph = self.graph.borrow();

        let targets = if let Some(ref constrained_edges) = self.constrained_edges {
            // Constrained case: only get targets for the constrained edges
            let mut constrained_targets = Vec::new();
            for &edge_id in constrained_edges {
                if let Ok((_source, target)) = graph.edge_endpoints(edge_id) {
                    constrained_targets.push(target);
                }
            }
            constrained_targets
        } else {
            // Unconstrained case: get all edge targets
            graph.edge_targets()
        };

        let values: Vec<i64> = targets.into_iter().map(|id| id as i64).collect();
        Ok(PyNumArray::new_int64(values))
    }

    /// Convert edge attributes to matrix
    /// Implements: g.edges.matrix()
    fn matrix(&self, py: Python) -> PyResult<Py<crate::ffi::storage::matrix::PyGraphMatrix>> {
        use crate::ffi::storage::matrix::PyGraphMatrix;

        // For now, delegate to adjacency matrix - later we can implement edge attribute matrices
        let graph_ref = self.graph.borrow();
        let matrix = graph_ref.to_adjacency_matrix::<f64>().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Matrix conversion failed: {}", e))
        })?;

        Py::new(py, PyGraphMatrix { inner: matrix })
    }

    /// Edge weight matrix (source × target with weights)
    /// Default to 'weight' attribute, but allow custom attribute selection
    /// Implements: g.edges.weight_matrix() and g.edges.weight_matrix('strength')
    fn weight_matrix(
        &self,
        py: Python,
        attr_name: Option<String>,
    ) -> PyResult<Py<crate::ffi::storage::matrix::PyGraphMatrix>> {
        use crate::ffi::storage::matrix::PyGraphMatrix;

        let weight_attr = attr_name.unwrap_or_else(|| "weight".to_string());
        let graph_ref = self.graph.borrow();

        let matrix = graph_ref
            .to_weighted_adjacency_matrix::<f64>(&weight_attr)
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Weight matrix conversion failed: {}",
                    e
                ))
            })?;

        Py::new(py, PyGraphMatrix { inner: matrix })
    }

    /// Create an EdgesArray for delegation chains
    /// Implements: g.edges.array()
    fn array(&self, py: Python) -> PyResult<Py<crate::ffi::storage::edges_array::PyEdgesArray>> {
        use crate::ffi::storage::edges_array::PyEdgesArray;

        // Create an EdgesArray with just this accessor to enable delegation
        let edges_array = PyEdgesArray::new(vec![self.clone()]);
        Py::new(py, edges_array)
    }

    /// Group edges by attribute value, returning SubgraphArray
    ///
    /// Args:
    ///     attr_name: Name of the edge attribute to group by
    ///
    /// Returns:
    ///     SubgraphArray: Array of subgraphs, one for each unique attribute value
    ///
    /// Example:
    ///     type_groups = g.edges.group_by('interaction_type')
    ///     # Returns subgraphs for each interaction type
    pub fn group_by(
        &self,
        attr_names: &PyAny,
    ) -> PyResult<crate::ffi::storage::subgraph_array::PySubgraphArray> {
        use groggy::types::{AttrName, EdgeId};
        use std::collections::HashMap;
        use std::collections::HashSet;

        // Parse attr_names as either String or Vec<String>
        let attr_name_list: Vec<String> = if let Ok(single) = attr_names.extract::<String>() {
            vec![single]
        } else if let Ok(multiple) = attr_names.extract::<Vec<String>>() {
            multiple
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "attr_names must be a string or list of strings",
            ));
        };

        let attr_names_typed: Vec<AttrName> = attr_name_list
            .iter()
            .map(|s| AttrName::from(s.clone()))
            .collect();

        let graph = self.graph.borrow();

        // Determine which edges to group
        let edge_ids: Vec<groggy::EdgeId> = if let Some(ref constrained) = self.constrained_edges {
            constrained.to_vec()
        } else {
            graph.edge_ids()
        };

        // Group edges by composite attribute value(s)
        let mut groups: HashMap<Vec<groggy::AttrValue>, Vec<EdgeId>> = HashMap::new();

        for edge_id in edge_ids {
            let mut key_values = Vec::new();
            let mut has_all_attrs = true;

            for attr_name in &attr_names_typed {
                if let Ok(Some(attr_value)) = graph.get_edge_attr(edge_id, attr_name) {
                    key_values.push(attr_value);
                } else {
                    has_all_attrs = false;
                    break;
                }
            }

            if has_all_attrs {
                groups.entry(key_values).or_default().push(edge_id);
            }
        }

        // Create subgraphs for each group - sort by attribute value for deterministic order
        let mut subgraphs = Vec::new();
        let mut sorted_groups: Vec<_> = groups.into_iter().collect();
        sorted_groups.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for (key_values, grouped_edge_ids) in sorted_groups {
            // For each group of edges, create a subgraph containing:
            // 1. All nodes that are endpoints of these edges
            // 2. Only the edges in this group

            let mut induced_nodes = HashSet::new();
            for edge_id in &grouped_edge_ids {
                if let Ok((source, target)) = graph.edge_endpoints(*edge_id) {
                    induced_nodes.insert(source);
                    induced_nodes.insert(target);
                }
            }

            // Create subgraph with induced nodes and filtered edges
            let induced_node_set: HashSet<_> = induced_nodes.into_iter().collect();
            let grouped_edge_set: HashSet<_> = grouped_edge_ids.into_iter().collect();

            let subgraph_name = if attr_name_list.len() == 1 {
                format!("edges_{}_group_{:?}", attr_name_list[0], key_values[0])
            } else {
                format!("edges_{}_group_{:?}", attr_name_list.join("_"), key_values)
            };

            let subgraph = groggy::subgraphs::Subgraph::new(
                self.graph.clone(),
                induced_node_set,
                grouped_edge_set,
                subgraph_name,
            );

            subgraphs.push(crate::ffi::subgraphs::subgraph::PySubgraph { inner: subgraph });
        }

        Ok(crate::ffi::storage::subgraph_array::PySubgraphArray::new(
            subgraphs,
        ))
    }

    /// Get viz accessor for visualization operations
    #[getter]
    fn viz(&self, py: Python) -> PyResult<Py<crate::ffi::viz_accessor::VizAccessor>> {
        // Create a viz graph from the constrained edges
        let mut viz_graph = groggy::api::graph::Graph::new();

        // Get the constrained edge IDs or all edges if no constraint
        let edge_ids = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            self.graph.borrow().edge_ids()
        };

        // Add nodes and edges to viz graph based on the edges in this accessor
        {
            let graph_ref = self.graph.borrow();
            let mut node_map = std::collections::HashMap::new();

            // Add nodes based on the edges' endpoints
            for &edge_id in &edge_ids {
                if let Ok((source, target)) = graph_ref.edge_endpoints(edge_id) {
                    // Add source node if not already added
                    node_map
                        .entry(source)
                        .or_insert_with(|| viz_graph.add_node());

                    // Add target node if not already added
                    node_map
                        .entry(target)
                        .or_insert_with(|| viz_graph.add_node());

                    // Add the edge
                    if let Some(&new_source) = node_map.get(&source) {
                        if let Some(&new_target) = node_map.get(&target) {
                            let _new_edge_id = viz_graph.add_edge(new_source, new_target);
                            // TODO: Copy edge attributes
                        }
                    }
                }
            }
        }

        let graph_data_source = groggy::viz::streaming::GraphDataSource::new(&viz_graph);
        let viz_accessor = crate::ffi::viz_accessor::VizAccessor::with_data_source(
            graph_data_source,
            "EdgesAccessor".to_string(),
        );

        Py::new(py, viz_accessor)
    }
}

impl PyEdgesAccessor {
    /// Create an EdgesTable for only the constrained edges
    fn create_constrained_edges_table(
        &self,
        constrained_edges: &[groggy::EdgeId],
    ) -> PyResult<groggy::storage::table::EdgesTable> {
        use groggy::storage::array::BaseArray;
        use groggy::storage::table::{BaseTable, EdgesTable};
        use std::collections::HashMap;

        let graph = self.graph.borrow();

        // Collect all constrained edges with their attributes
        let mut attribute_columns: HashMap<String, Vec<groggy::AttrValue>> = HashMap::new();

        // Initialize with required edge columns: edge_id, source, target
        attribute_columns.insert("edge_id".to_string(), Vec::new());
        attribute_columns.insert("source".to_string(), Vec::new());
        attribute_columns.insert("target".to_string(), Vec::new());

        // First pass: collect all attribute names that exist on ANY of the constrained edges
        let mut all_attr_names = std::collections::HashSet::new();
        all_attr_names.insert("edge_id".to_string());
        all_attr_names.insert("source".to_string());
        all_attr_names.insert("target".to_string());

        for &edge_id in constrained_edges {
            if let Ok(attrs) = graph.get_edge_attrs(edge_id) {
                for attr_name in attrs.keys() {
                    all_attr_names.insert(attr_name.clone());
                }
            }
        }

        // Initialize all columns
        for attr_name in &all_attr_names {
            if !attribute_columns.contains_key(attr_name) {
                attribute_columns.insert(attr_name.clone(), Vec::new());
            }
        }

        // Second pass: collect data for each constrained edge
        for &edge_id in constrained_edges {
            // Add edge_id
            attribute_columns
                .get_mut("edge_id")
                .unwrap()
                .push(groggy::AttrValue::Int(edge_id as i64));

            // Add source and target
            if let Ok((source, target)) = graph.edge_endpoints(edge_id) {
                attribute_columns
                    .get_mut("source")
                    .unwrap()
                    .push(groggy::AttrValue::Int(source as i64));
                attribute_columns
                    .get_mut("target")
                    .unwrap()
                    .push(groggy::AttrValue::Int(target as i64));
            } else {
                // Handle missing endpoints
                attribute_columns
                    .get_mut("source")
                    .unwrap()
                    .push(groggy::AttrValue::Null);
                attribute_columns
                    .get_mut("target")
                    .unwrap()
                    .push(groggy::AttrValue::Null);
            }

            // Get all attributes for this edge
            let edge_attrs = graph.get_edge_attrs(edge_id).unwrap_or_default();

            // For each expected attribute, add the value or null
            for attr_name in &all_attr_names {
                if matches!(attr_name.as_str(), "edge_id" | "source" | "target") {
                    continue; // Already handled
                }

                let attr_value = edge_attrs
                    .get(attr_name)
                    .cloned()
                    .unwrap_or(groggy::AttrValue::Null);

                attribute_columns
                    .get_mut(attr_name)
                    .unwrap()
                    .push(attr_value);
            }
        }

        // Apply auto-slicing: remove columns that are all null for this edge set
        let mut columns_to_keep = Vec::new();

        for (attr_name, values) in &attribute_columns {
            if matches!(attr_name.as_str(), "edge_id" | "source" | "target") {
                // Always keep required edge columns
                columns_to_keep.push(attr_name.clone());
            } else {
                // Check if this column has any non-null values
                let has_non_null = values.iter().any(|v| !matches!(v, groggy::AttrValue::Null));
                if has_non_null {
                    columns_to_keep.push(attr_name.clone());
                }
            }
        }

        // Create filtered attribute columns
        let mut filtered_columns = HashMap::new();
        for attr_name in columns_to_keep {
            if let Some(values) = attribute_columns.remove(&attr_name) {
                filtered_columns.insert(attr_name, BaseArray::from_attr_values(values));
            }
        }

        // Convert to BaseTable and then EdgesTable
        let base_table = BaseTable::from_columns(filtered_columns)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create BaseTable: {}", e)))?;

        EdgesTable::from_base_table(base_table)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert to EdgesTable: {}", e)))
    }

    /// Set attributes for multiple edges (bulk operation) - internal method callable from Rust
    pub fn set_attrs_internal(&self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        use crate::ffi::api::graph_attributes::PyGraphAttrMut;

        if attrs_dict.is_empty() {
            return Err(PyValueError::new_err(
                "attrs_dict is missing required attribute mappings",
            ));
        }

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
                        _ => {
                            return Err(PyValueError::new_err(format!(
                                "Invalid filter_type: {}. Must be 'base' or 'meta'",
                                filter_type
                            )))
                        }
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
    ///
    /// # Future Feature
    ///
    /// Designed for automatic column pruning in table operations for edges.
    #[allow(dead_code)]
    fn apply_edge_auto_slice(
        &self,
        table: groggy::storage::table::BaseTable,
        edge_set: &std::collections::HashSet<groggy::EdgeId>,
    ) -> PyResult<groggy::storage::table::BaseTable> {
        let graph = self.graph.borrow();

        // Get all attribute names in the table
        let column_names = table.column_names();
        let mut columns_to_keep = Vec::new();

        for column_name in column_names {
            // Check if this column has any non-null values for the edges in our set
            let mut has_non_null_value = false;

            for &edge_id in edge_set {
                // Try to get the attribute value for this edge
                match graph.get_edge_attr(edge_id, column_name) {
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
                columns_to_keep.push(column_name.clone());
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
            table
                .select(&columns_to_keep)
                .map_err(|e| PyValueError::new_err(format!("Failed to slice table columns: {}", e)))
        }
    }

    /// Get the number of edges accessible by this accessor
    pub fn edge_count(&self) -> usize {
        if let Some(ref constrained) = self.constrained_edges {
            constrained.len()
        } else {
            let graph = self.graph.borrow();
            graph.edge_ids().len()
        }
    }

    /// Convert edges to NodesAccessor containing all nodes connected to these edges
    pub fn nodes(&self) -> PyResult<crate::ffi::storage::accessors::PyNodesAccessor> {
        use std::collections::HashSet;

        let graph = self.graph.borrow();
        let edge_ids: Vec<EdgeId> = if let Some(ref constraint) = self.constrained_edges {
            constraint.clone()
        } else {
            graph.edge_ids()
        };

        // Collect all nodes connected to these edges
        let mut connected_nodes: HashSet<NodeId> = HashSet::new();

        for &edge_id in &edge_ids {
            if let Ok((source, target)) = graph.edge_endpoints(edge_id) {
                connected_nodes.insert(source);
                connected_nodes.insert(target);
            }
        }

        let node_ids: Vec<NodeId> = connected_nodes.into_iter().collect();

        Ok(PyNodesAccessor {
            graph: self.graph.clone(),
            constrained_nodes: Some(node_ids),
        })
    }

    /// Convert edges to NodesAccessor showing all nodes connected by these edges
    pub fn to_nodes(&self) -> PyResult<crate::ffi::storage::accessors::PyNodesAccessor> {
        use groggy::types::{EdgeId, NodeId};
        use std::collections::HashSet;

        let graph = self.graph.borrow();
        let edge_ids: Vec<EdgeId> = if let Some(ref constraint) = self.constrained_edges {
            constraint.clone()
        } else {
            graph.edge_ids()
        };

        // Collect all nodes that are endpoints of these edges
        let mut connected_nodes = HashSet::new();

        for &edge_id in &edge_ids {
            if let Ok((source, target)) = graph.edge_endpoints(edge_id) {
                connected_nodes.insert(source);
                connected_nodes.insert(target);
            }
        }

        let node_ids: Vec<NodeId> = connected_nodes.into_iter().collect();

        drop(graph); // Release the borrow before creating PyNodesAccessor

        let nodes_accessor = PyNodesAccessor {
            graph: self.graph.clone(),
            constrained_nodes: Some(node_ids),
        };
        Ok(nodes_accessor)
    }

    /// Convert edges to SubgraphArray by creating subgraphs for each edge  
    pub fn to_subgraphs(&self) -> PyResult<crate::ffi::storage::subgraph_array::PySubgraphArray> {
        // TODO: Implement proper conversion from EdgesAccessor to SubgraphArray
        // This requires complex subgraph creation logic
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "EdgesAccessor to SubgraphArray conversion not yet implemented.",
        ))
    }

    /// Convert Python value to BaseArray<AttrValue> for column assignment
    fn convert_python_to_base_array(
        &self,
        py: Python,
        value: &PyAny,
    ) -> PyResult<BaseArray<AttrValue>> {
        // Handle BaseArray directly
        if let Ok(base_array) = value.extract::<PyRef<PyBaseArray>>() {
            return Ok(base_array.inner.clone());
        }

        // Handle Python list
        if let Ok(list) = value.extract::<Vec<PyObject>>() {
            let mut attr_values = Vec::new();
            for py_obj in list {
                let attr_value = python_value_to_attr_value(py_obj.as_ref(py)).map_err(|e| {
                    PyValueError::new_err(format!("Failed to convert list element: {}", e))
                })?;
                attr_values.push(attr_value);
            }
            return Ok(BaseArray::from_attr_values(attr_values));
        }

        // Handle scalar broadcasting (single value for all edges)
        let attr_value = python_value_to_attr_value(value)
            .map_err(|e| PyValueError::new_err(format!("Failed to convert scalar value: {}", e)))?;

        // Get edge count for broadcasting
        let edge_count = if let Some(ref constrained) = self.constrained_edges {
            constrained.len()
        } else {
            let graph = self.graph.borrow();
            graph.edge_ids().len()
        };

        // Create array with repeated value
        let attr_values = vec![attr_value; edge_count];
        Ok(BaseArray::from_attr_values(attr_values))
    }
}
