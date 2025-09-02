//! Path Result FFI Bindings - Pure wrapper for core PathResult

use groggy::core::path_result::PathResult;
use groggy::{EdgeId, NodeId};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashSet;

use crate::ffi::core::array::PyGraphArray;

/// Python wrapper for core PathResult - Pure delegation, no business logic
#[pyclass(name = "PathResult")]
pub struct PyPathResult {
    pub inner: PathResult,
}

impl PyPathResult {
    /// Create from core PathResult (internal use)
    pub fn from_core(inner: PathResult) -> Self {
        Self { inner }
    }
    
    /// Create new PathResult from basic data (internal use)
    pub fn new(nodes: Vec<NodeId>, edges: Vec<EdgeId>, path_type: String) -> Self {
        Self {
            inner: PathResult::new(nodes, edges, path_type),
        }
    }
    
    /// Create from HashSets (internal use)
    pub fn from_sets(
        node_set: HashSet<NodeId>, 
        edge_set: HashSet<EdgeId>, 
        path_type: String
    ) -> Self {
        Self {
            inner: PathResult::from_sets(node_set, edge_set, path_type),
        }
    }
}

#[pymethods]
impl PyPathResult {
    /// Get the number of nodes in the path result
    fn __len__(&self) -> usize {
        self.inner.node_count()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "PathResult({} type: {} nodes, {} edges)", 
            self.inner.result_type(),
            self.inner.node_count(), 
            self.inner.edge_count()
        )
    }

    /// Get node count
    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Get edge count  
    #[getter]
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Get path type (bfs, dfs, etc.)
    #[getter]
    fn path_type(&self) -> &str {
        self.inner.result_type()
    }

    /// Get nodes as a GraphArray (lightweight, no subgraph creation)
    #[getter]
    fn nodes(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let attr_values: Vec<groggy::AttrValue> = self.inner.nodes()
            .iter()
            .map(|&id| groggy::AttrValue::Int(id as i64))
            .collect();
        
        let py_array = PyGraphArray {
            inner: groggy::core::array::GraphArray::from_vec(attr_values),
        };
        
        Py::new(py, py_array)
    }

    /// Get edges as a GraphArray (lightweight, no subgraph creation)
    #[getter]
    fn edges(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let attr_values: Vec<groggy::AttrValue> = self.inner.edges()
            .iter()
            .map(|&id| groggy::AttrValue::Int(id as i64))
            .collect();
        
        let py_array = PyGraphArray {
            inner: groggy::core::array::GraphArray::from_vec(attr_values),
        };
        
        Py::new(py, py_array)
    }

    /// Get raw node IDs as Python list (ultra-lightweight)
    fn node_ids(&self) -> Vec<NodeId> {
        self.inner.nodes().clone()
    }

    /// Get raw edge IDs as Python list (ultra-lightweight)
    fn edge_ids(&self) -> Vec<EdgeId> {
        self.inner.edges().clone()
    }

    /// Check if a specific node is in the path result
    fn contains_node(&self, node_id: NodeId) -> bool {
        self.inner.contains_node(node_id)
    }

    /// Check if a specific edge is in the path result
    fn contains_edge(&self, edge_id: EdgeId) -> bool {
        self.inner.contains_edge(edge_id)
    }

    /// Get the first node (useful for paths)
    fn first_node(&self) -> PyResult<NodeId> {
        self.inner.first_node()
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    }

    /// Get the last node (useful for paths)
    fn last_node(&self) -> PyResult<NodeId> {
        self.inner.last_node()
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    }

    /// Check if the path result is empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Create a set of the nodes (for set operations)
    fn node_set(&self) -> HashSet<NodeId> {
        self.inner.to_node_set()
    }

    /// Create a set of the edges (for set operations)
    fn edge_set(&self) -> HashSet<EdgeId> {
        self.inner.to_edge_set()
    }
}