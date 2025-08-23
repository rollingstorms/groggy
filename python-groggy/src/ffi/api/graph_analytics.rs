//! Graph Analytics Module
//!
//! Python bindings for graph analytics and algorithms.

use crate::ffi::core::subgraph::PySubgraph;
use crate::ffi::utils::graph_error_to_py_err;
use groggy::core::traversal::{PathFindingOptions, TraversalOptions};
use groggy::{AttrName, NodeId};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Analytics operations for graphs
#[pyclass(name = "GraphAnalytics")]
pub struct PyGraphAnalytics {
    /// Reference to the parent graph
    pub graph: Py<crate::ffi::api::graph::PyGraph>,
}

#[pymethods]
impl PyGraphAnalytics {
    /// Calculate connected components with optional in-place attribute setting
    #[pyo3(signature = (inplace = false, attr_name = None))]
    pub fn connected_components(
        &self,
        py: Python,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<Vec<PySubgraph>> {
        let inplace = inplace.unwrap_or(false);
        let mut graph = self.graph.borrow_mut(py);

        // Delegate to core algorithm - THIN WRAPPER
        let options = TraversalOptions::default();
        let result = graph
            .inner
            .connected_components(options)
            .map_err(graph_error_to_py_err)?;

        let mut subgraphs = Vec::new();

        // Handle bulk attribute setting if requested
        if inplace {
            if let Some(ref attr_name) = attr_name {
                let mut attrs_values = std::collections::HashMap::new();
                let node_value_pairs: Vec<(NodeId, groggy::AttrValue)> = result
                    .components
                    .iter()
                    .enumerate()
                    .flat_map(|(i, component)| {
                        component
                            .nodes
                            .iter()
                            .map(move |&node_id| (node_id, groggy::AttrValue::Int(i as i64)))
                    })
                    .collect();
                attrs_values.insert(attr_name.clone(), node_value_pairs);

                graph
                    .inner
                    .set_node_attrs(attrs_values)
                    .map_err(graph_error_to_py_err)?;
            }
        }

        // Convert core results to FFI wrappers - ZERO-COPY: just use pre-computed edges!
        for (i, component) in result.components.into_iter().enumerate() {
            // 🚀 PERFORMANCE: Use edges already computed by Rust core - no recomputation needed!
            let subgraph = PySubgraph::new(
                component.nodes,
                component.edges, // Use pre-computed induced edges from Rust core
                format!("connected_component_{}", i),
                Some(self.graph.clone()),
            );
            subgraphs.push(subgraph);
        }

        Ok(subgraphs)
    }

    /// Perform breadth-first search traversal
    #[pyo3(signature = (start_node, max_depth = None, inplace = false, attr_name = None))]
    pub fn bfs(
        &self,
        py: Python,
        start_node: NodeId,
        max_depth: Option<usize>,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<PySubgraph> {
        let inplace = inplace.unwrap_or(false);
        let mut graph = self.graph.borrow_mut(py);

        // Create traversal options
        let mut options = TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }

        // Perform BFS traversal
        let result = graph
            .inner
            .bfs(start_node, options)
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

            graph
                .inner
                .set_node_attrs(attrs_values)
                .map_err(graph_error_to_py_err)?;
        }

        Ok(PySubgraph::new(
            result.nodes,
            result.edges,
            "bfs_traversal".to_string(),
            Some(self.graph.clone()),
        ))
    }

    /// Perform depth-first search traversal
    #[pyo3(signature = (start_node, max_depth = None, inplace = false, attr_name = None))]
    pub fn dfs(
        &self,
        py: Python,
        start_node: NodeId,
        max_depth: Option<usize>,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<PySubgraph> {
        let inplace = inplace.unwrap_or(false);
        let mut graph = self.graph.borrow_mut(py);

        // Create traversal options
        let mut options = TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }

        // Perform DFS traversal
        let result = graph
            .inner
            .dfs(start_node, options)
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

            graph
                .inner
                .set_node_attrs(attrs_values)
                .map_err(graph_error_to_py_err)?;
        }

        Ok(PySubgraph::new(
            result.nodes,
            result.edges,
            "dfs_traversal".to_string(),
            Some(self.graph.clone()),
        ))
    }

    /// Find shortest path between two nodes
    #[pyo3(signature = (source, target, weight_attribute = None, inplace = false, attr_name = None))]
    pub fn shortest_path(
        &self,
        py: Python,
        source: NodeId,
        target: NodeId,
        weight_attribute: Option<AttrName>,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<Option<PySubgraph>> {
        let inplace = inplace.unwrap_or(false);
        let mut graph = self.graph.borrow_mut(py);

        let options = PathFindingOptions {
            weight_attribute,
            max_path_length: None,
            heuristic: None,
        };

        let result = graph
            .inner
            .shortest_path(source, target, options)
            .map_err(graph_error_to_py_err)?;

        match result {
            Some(path) => {
                if inplace {
                    if let Some(attr_name) = attr_name {
                        // Use bulk attribute setting for performance
                        let mut attrs_values = std::collections::HashMap::new();
                        let node_value_pairs: Vec<(NodeId, groggy::AttrValue)> = path
                            .nodes
                            .iter()
                            .enumerate()
                            .map(|(distance, &node_id)| {
                                (node_id, groggy::AttrValue::Int(distance as i64))
                            })
                            .collect();
                        attrs_values.insert(attr_name, node_value_pairs);

                        graph
                            .inner
                            .set_node_attrs(attrs_values)
                            .map_err(graph_error_to_py_err)?;
                    }
                }

                Ok(Some(PySubgraph::new(
                    path.nodes,
                    path.edges,
                    "shortest_path".to_string(),
                    Some(self.graph.clone()),
                )))
            }
            None => Ok(None),
        }
    }

    /// Check if a path exists between two nodes
    pub fn has_path(&self, py: Python, source: NodeId, target: NodeId) -> PyResult<bool> {
        let mut graph = self.graph.borrow_mut(py);

        let options = PathFindingOptions {
            weight_attribute: None,
            max_path_length: None,
            heuristic: None,
        };

        let result = graph
            .inner
            .shortest_path(source, target, options)
            .map_err(graph_error_to_py_err)?;

        Ok(result.is_some())
    }

    /// Get node degree
    fn degree(&self, py: Python, node: NodeId) -> PyResult<usize> {
        let graph = self.graph.borrow(py);
        graph.inner.degree(node).map_err(graph_error_to_py_err)
    }

    /// Get memory statistics
    fn memory_statistics(&self, py: Python) -> PyResult<PyObject> {
        let graph = self.graph.borrow(py);
        let stats = graph.inner.memory_statistics();

        // Convert MemoryStatistics to Python dict
        let dict = PyDict::new(py);
        dict.set_item("pool_memory_bytes", stats.pool_memory_bytes)?;
        dict.set_item("space_memory_bytes", stats.space_memory_bytes)?;
        dict.set_item("history_memory_bytes", stats.history_memory_bytes)?;
        dict.set_item(
            "change_tracker_memory_bytes",
            stats.change_tracker_memory_bytes,
        )?;
        dict.set_item("total_memory_bytes", stats.total_memory_bytes)?;
        dict.set_item("total_memory_mb", stats.total_memory_mb)?;

        // Add memory efficiency stats
        let efficiency_dict = PyDict::new(py);
        efficiency_dict.set_item("bytes_per_node", stats.memory_efficiency.bytes_per_node)?;
        efficiency_dict.set_item("bytes_per_edge", stats.memory_efficiency.bytes_per_edge)?;
        efficiency_dict.set_item("bytes_per_entity", stats.memory_efficiency.bytes_per_entity)?;
        efficiency_dict.set_item("overhead_ratio", stats.memory_efficiency.overhead_ratio)?;
        efficiency_dict.set_item("cache_efficiency", stats.memory_efficiency.cache_efficiency)?;
        dict.set_item("memory_efficiency", efficiency_dict)?;

        Ok(dict.to_object(py))
    }

    /// Get analytics summary
    fn get_summary(&self, py: Python) -> PyResult<String> {
        let graph = self.graph.borrow(py);
        let node_count = graph.get_node_count();
        let edge_count = graph.get_edge_count();

        Ok(format!(
            "Graph Analytics: {} nodes, {} edges",
            node_count, edge_count
        ))
    }
}

// Placeholder - will extract analytics operations from lib_old.rs
