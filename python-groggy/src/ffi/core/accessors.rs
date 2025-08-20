//! Accessors FFI Bindings
//! 
//! Python bindings for smart indexing accessors.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};
use pyo3::exceptions::{PyValueError, PyTypeError, PyRuntimeError, PyKeyError, PyIndexError, PyImportError, PyNotImplementedError};
use groggy::{NodeId, EdgeId};

// Import types from our FFI modules
use crate::ffi::api::graph::PyGraph;
use crate::ffi::core::subgraph::PySubgraph;

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
        // Try to extract as single integer (existing behavior)
        if let Ok(node_id) = key.extract::<NodeId>() {
            // Check constraint first
            if let Some(ref constrained) = self.constrained_nodes {
                if !constrained.contains(&node_id) {
                    return Err(PyKeyError::new_err(format!("Node {} is not in this subgraph", node_id)));
                }
            }
            
            // Single node access - return NodeView
            let graph = self.graph.borrow(py);
            if !graph.has_node_internal(node_id) {
                return Err(PyKeyError::new_err(format!("Node {} does not exist", node_id)));
            }
            
            let node_view = PyGraph::create_node_view_internal(self.graph.clone(), py, node_id)?;
            return Ok(node_view.to_object(py));
        }
        
        // Try to extract as list of integers (batch access)
        if let Ok(node_ids) = key.extract::<Vec<NodeId>>() {
            // Batch node access - return Subgraph
            let mut graph = self.graph.borrow_mut(py);
            
            // Validate all nodes exist
            for &node_id in &node_ids {
                if !graph.has_node_internal(node_id) {
                    return Err(PyKeyError::new_err(format!("Node {} does not exist", node_id)));
                }
            }
            
            // 🚀 PERFORMANCE FIX: Use core columnar topology instead of O(E) FFI algorithm
            let node_set: std::collections::HashSet<NodeId> = node_ids.iter().copied().collect();
            let (edge_ids, sources, targets) = graph.inner.get_columnar_topology();
            let mut induced_edges = Vec::new();
            
            // O(k) where k = active edges, much better than O(E)
            for i in 0..edge_ids.len() {
                let edge_id = edge_ids[i];
                let source = sources[i];
                let target = targets[i];
                
                // O(1) HashSet lookups
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
            let mut graph = self.graph.borrow_mut(py);
            let all_node_ids = graph.inner.node_ids();
            
            // Convert slice to indices
            let slice_info = slice.indices(all_node_ids.len() as i64)?;
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
            let selected_node_set: std::collections::HashSet<NodeId> = selected_nodes.iter().copied().collect();
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
        Err(PyTypeError::new_err("Node index must be int, list of ints, or slice"))
    }
    
    /// Support iteration: for node_id in g.nodes
    fn __iter__(&self, py: Python) -> PyResult<PyObject> {
        let graph = self.graph.borrow(py);
        let node_ids = graph.get_node_ids_array(py)?;
        // Return the GraphArray directly - Python will handle iteration
        Ok(node_ids.to_object(py))
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
    fn table(&self, py: Python) -> PyResult<PyObject> {
        // Import the Python GraphTable class
        let groggy = py.import("groggy")?;
        let graph_table_class = groggy.getattr("GraphTable")?;
        
        // Get the underlying graph
        let graph = self.graph.borrow(py);
        
        // Determine node list based on constraints
        let node_data = if let Some(ref constrained) = self.constrained_nodes {
            // Subgraph case: use constrained nodes
            constrained.clone()
        } else {
            // Full graph case: get all node IDs from the graph
            // Use the inner graph's node_ids method directly
            graph.inner.node_ids()
        };
        
        // Create GraphTable with nodes data source
        let table = graph_table_class.call1((node_data, "nodes", self.graph.clone()))?;
        Ok(table.to_object(py))
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
        // Try to extract as single integer (existing behavior)
        if let Ok(edge_id) = key.extract::<EdgeId>() {
            // Check constraint first
            if let Some(ref constrained) = self.constrained_edges {
                if !constrained.contains(&edge_id) {
                    return Err(PyKeyError::new_err(format!("Edge {} is not in this subgraph", edge_id)));
                }
            }
            
            // Single edge access - return EdgeView
            let graph = self.graph.borrow(py);
            if !graph.has_edge_internal(edge_id) {
                return Err(PyKeyError::new_err(format!("Edge {} does not exist", edge_id)));
            }
            
            let edge_view = PyGraph::create_edge_view_internal(self.graph.clone(), py, edge_id)?;
            return Ok(edge_view.to_object(py));
        }
        
        // Try to extract as list of integers (batch access)
        if let Ok(edge_ids) = key.extract::<Vec<EdgeId>>() {
            // Batch edge access - return Subgraph with these edges + their endpoints
            let graph = self.graph.borrow(py);
            
            // Validate all edges exist
            for &edge_id in &edge_ids {
                if !graph.has_edge_internal(edge_id) {
                    return Err(PyKeyError::new_err(format!("Edge {} does not exist", edge_id)));
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
            let slice_info = slice.indices(all_edge_ids.len() as i64)?;
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
        Err(PyTypeError::new_err("Edge index must be int, list of ints, or slice"))
    }
    
    /// Support iteration: for edge_id in g.edges
    fn __iter__(&self, py: Python) -> PyResult<PyObject> {
        let graph = self.graph.borrow(py);
        let edge_ids = graph.get_edge_ids_array(py)?;
        // Return the GraphArray directly - Python will handle iteration
        Ok(edge_ids.to_object(py))
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
    fn table(&self, py: Python) -> PyResult<PyObject> {
        // Import the Python GraphTable class
        let groggy = py.import("groggy")?;
        let graph_table_class = groggy.getattr("GraphTable")?;
        
        // Get the underlying graph
        let graph = self.graph.borrow(py);
        
        // Determine edge list based on constraints
        let edge_data = if let Some(ref constrained) = self.constrained_edges {
            // Subgraph case: use constrained edges
            constrained.clone()
        } else {
            // Full graph case: get all edge IDs from the graph
            graph.inner.edge_ids()
        };
        
        // Create GraphTable with edges data source
        let table = graph_table_class.call1((edge_data, "edges", self.graph.clone()))?;
        Ok(table.to_object(py))
    }
}
