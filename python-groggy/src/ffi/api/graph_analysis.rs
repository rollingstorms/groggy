//! Graph Analysis Operations - Pure FFI Delegation Layer
//!
//! Graph algorithm operations that delegate to core implementations.

use crate::ffi::core::neighborhood::PyNeighborhoodResult;
use crate::ffi::utils::graph_error_to_py_err;
use groggy::{AttrName, EdgeId, NodeId};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::graph::PyGraph;

#[pymethods]
impl PyGraph {
    /// Get neighbors of nodes - PURE DELEGATION to core
    fn neighbors(&mut self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject> {
        if let Some(nodes_input) = nodes {
            // Handle multiple nodes
            if let Ok(node_list) = nodes_input.extract::<Vec<NodeId>>() {
                let result_dict = PyDict::new(py);
                
                for node in node_list {
                    // DELEGATION: Use core neighbors implementation (graph.rs:866)
                    let neighbors = py.allow_threads(|| {
                        self.inner
                            .borrow()
                            .neighbors(node)
                            .map_err(graph_error_to_py_err)
                    })?;
                    
                    let py_neighbors = PyList::new(py, neighbors);
                    result_dict.set_item(node, py_neighbors)?;
                }
                
                Ok(result_dict.to_object(py))
            } else if let Ok(single_node) = nodes_input.extract::<NodeId>() {
                // Handle single node
                let neighbors = py.allow_threads(|| {
                    self.inner
                        .borrow()
                        .neighbors(single_node)
                        .map_err(graph_error_to_py_err)
                })?;
                
                Ok(PyList::new(py, neighbors).to_object(py))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "nodes must be a NodeId or list of NodeIds"
                ))
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "nodes parameter is required"
            ))
        }
    }

    /// Get degree of nodes - PURE DELEGATION to core
    fn degree(&self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject> {
        if let Some(nodes_input) = nodes {
            // Handle multiple nodes
            if let Ok(node_list) = nodes_input.extract::<Vec<NodeId>>() {
                let result_dict = PyDict::new(py);
                
                for node in node_list {
                    // DELEGATION: Use core degree implementation (graph.rs:845)
                    let degree = py.allow_threads(|| {
                        self.inner
                            .borrow()
                            .degree(node)
                            .map_err(graph_error_to_py_err)
                    })?;
                    
                    result_dict.set_item(node, degree)?;
                }
                
                Ok(result_dict.to_object(py))
            } else if let Ok(single_node) = nodes_input.extract::<NodeId>() {
                // Handle single node
                let degree = py.allow_threads(|| {
                    self.inner
                        .borrow()
                        .degree(single_node)
                        .map_err(graph_error_to_py_err)
                })?;
                
                Ok(degree.to_object(py))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "nodes must be a NodeId or list of NodeIds"
                ))
            }
        } else {
            // Return degree for all nodes
            let all_nodes = self.inner.borrow().node_ids();
            let result_dict = PyDict::new(py);
            
            for node in all_nodes {
                let degree = py.allow_threads(|| {
                    self.inner
                        .borrow()
                        .degree(node)
                        .map_err(graph_error_to_py_err)
                })?;
                
                result_dict.set_item(node, degree)?;
            }
            
            Ok(result_dict.to_object(py))
        }
    }

    /// Get in-degree of nodes - MISSING FROM CORE (needs implementation)
    fn in_degree(&self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "in_degree needs to be implemented in core first"
        ))
    }

    /// Get out-degree of nodes - MISSING FROM CORE (needs implementation)
    fn out_degree(&self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "out_degree needs to be implemented in core first"
        ))
    }

    /// Get neighborhood sampling - PURE DELEGATION to core
    fn neighborhood(
        &mut self,
        py: Python,
        center_nodes: Vec<NodeId>,
        radius: Option<usize>,
        max_nodes: Option<usize>,
    ) -> PyResult<PyNeighborhoodResult> {
        let radius = radius.unwrap_or(1);
        let max_nodes = max_nodes.unwrap_or(100);

        // DELEGATION: Use core neighborhood implementation (graph.rs:1366)
        let result = py.allow_threads(|| {
            self.inner
                .borrow_mut()
                .neighborhood(&center_nodes, radius, max_nodes)
                .map_err(graph_error_to_py_err)
        })?;

        Ok(PyNeighborhoodResult::from_core_result(result))
    }

    /// Get shortest path - PURE DELEGATION to core
    fn shortest_path(
        &self,
        py: Python,
        source: NodeId,
        target: NodeId,
        weight_attribute: Option<AttrName>,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<PyObject> {
        // DELEGATION: Use core shortest_path implementation (graph.rs:1311 + traversal.rs:323)
        let path = py.allow_threads(|| {
            if let Some(weight_attr) = weight_attribute {
                // Weighted shortest path
                self.inner
                    .borrow()
                    .shortest_path_weighted(source, target, &weight_attr)
                    .map_err(graph_error_to_py_err)
            } else {
                // Unweighted shortest path
                self.inner
                    .borrow()
                    .shortest_path(source, target)
                    .map_err(graph_error_to_py_err)
            }
        })?;

        match path {
            Some(path_nodes) => Ok(PyList::new(py, path_nodes).to_object(py)),
            None => Ok(py.None()),
        }
    }
}