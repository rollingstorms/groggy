//! Graph Analytics Module
//! 
//! Python bindings for graph analytics and algorithms.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::{PyValueError, PyTypeError, PyRuntimeError};
use groggy::{NodeId, EdgeId, AttrName};
use groggy::core::traversal::{TraversalOptions, PathFindingOptions};
use crate::ffi::types::PyAttrValue;
use crate::ffi::utils::{graph_error_to_py_err, attr_value_to_python_value};
use crate::ffi::core::subgraph::PySubgraph;
use std::collections::HashSet;

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
    fn connected_components(&self, py: Python, inplace: Option<bool>, attr_name: Option<String>) -> PyResult<Vec<PySubgraph>> {
        let inplace = inplace.unwrap_or(false);
        let mut graph = self.graph.borrow_mut(py);
        
        let options = TraversalOptions::default();
        let result = graph.inner.connected_components(options)
            .map_err(graph_error_to_py_err)?;
        
        // Convert each component to a PySubgraph  
        let mut subgraphs = Vec::new();
        
        // Store components data to avoid borrow conflicts
        let components_data: Vec<_> = result.components.into_iter().enumerate().collect();
        drop(graph); // Release borrow
        
        // Process each component
        for (i, component) in components_data {
            // Re-borrow for each component to get topology
            let graph_ref = self.graph.borrow(py);
            let (edge_ids, sources, targets) = graph_ref.inner.get_columnar_topology();
            
            // Calculate induced edges using optimized columnar topology method
            let component_nodes: HashSet<NodeId> = component.nodes.iter().copied().collect();
            let mut induced_edges = Vec::new();
            
            // Iterate through parallel vectors - O(k) where k = active edges
            for j in 0..edge_ids.len() {
                let edge_id = edge_ids[j];
                let source = sources[j];
                let target = targets[j];
                
                // O(1) HashSet lookups instead of O(n) Vec::contains
                if component_nodes.contains(&source) && component_nodes.contains(&target) {
                    induced_edges.push(edge_id);
                }
            }
            drop(graph_ref); // Release immutable borrow
            
            // Create subgraph with proper induced edges
            let subgraph = PySubgraph::new(
                component.nodes.clone(),
                induced_edges,
                format!("connected_component_{}", i),
                Some(self.graph.clone()),
            );
            subgraphs.push(subgraph);
            
            // If inplace=True, set component_id attribute on nodes
            if inplace {
                let attr_name = attr_name.clone().unwrap_or_else(|| "component_id".to_string());
                let component_value = PyAttrValue { inner: groggy::AttrValue::Int(i as i64) };
                
                let mut graph_mut = self.graph.borrow_mut(py);
                for &node_id in &component.nodes {
                    graph_mut.set_node_attribute(node_id, attr_name.clone(), &component_value)?;
                }
                drop(graph_mut); // Release mutable borrow
            }
        }
        
        Ok(subgraphs)
    }
    
    /// Perform breadth-first search traversal
    #[pyo3(signature = (start_node, max_depth = None, inplace = false, attr_name = None))]
    fn bfs(&self, py: Python, start_node: NodeId, max_depth: Option<usize>, 
           inplace: Option<bool>, attr_name: Option<String>) -> PyResult<PySubgraph> {
        let inplace = inplace.unwrap_or(false);
        let mut graph = self.graph.borrow_mut(py);
        
        // Create traversal options
        let mut options = TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }
        
        // Perform BFS traversal
        let result = graph.inner.bfs(start_node, options)
            .map_err(graph_error_to_py_err)?;
        
        // If inplace=True, set distance/order attributes on nodes
        if inplace {
            let attr_name = attr_name.unwrap_or_else(|| "bfs_distance".to_string());
            
            // Set distance attributes (distance from start_node)
            for (order, &node_id) in result.nodes.iter().enumerate() {
                let order_value = PyAttrValue { inner: groggy::AttrValue::Int(order as i64) };
                graph.set_node_attribute(node_id, attr_name.clone(), &order_value)?;
            }
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
    fn dfs(&self, py: Python, start_node: NodeId, max_depth: Option<usize>,
           inplace: Option<bool>, attr_name: Option<String>) -> PyResult<PySubgraph> {
        let inplace = inplace.unwrap_or(false);
        let mut graph = self.graph.borrow_mut(py);
        
        // Create traversal options
        let mut options = TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }
        
        // Perform DFS traversal
        let result = graph.inner.dfs(start_node, options)
            .map_err(graph_error_to_py_err)?;
        
        // If inplace=True, set distance/order attributes on nodes
        if inplace {
            let attr_name = attr_name.unwrap_or_else(|| "dfs_order".to_string());
            
            // Set order attributes
            for (order, &node_id) in result.nodes.iter().enumerate() {
                let order_value = PyAttrValue { inner: groggy::AttrValue::Int(order as i64) };
                graph.set_node_attribute(node_id, attr_name.clone(), &order_value)?;
            }
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
    fn shortest_path(&self, py: Python, source: NodeId, target: NodeId, 
                    weight_attribute: Option<AttrName>, inplace: Option<bool>, 
                    attr_name: Option<String>) -> PyResult<Option<PySubgraph>> {
        let inplace = inplace.unwrap_or(false);
        let mut graph = self.graph.borrow_mut(py);
        
        let options = PathFindingOptions {
            weight_attribute,
            max_path_length: None,
            heuristic: None,
        };
        
        let result = graph.inner.shortest_path(source, target, options)
            .map_err(graph_error_to_py_err)?;
            
        match result {
            Some(path) => {
                if inplace {
                    if let Some(attr_name) = attr_name {
                        // Set path distance attribute on nodes
                        for (distance, &node_id) in path.nodes.iter().enumerate() {
                            let attr_value = groggy::AttrValue::Int(distance as i64);
                            graph.inner.set_node_attr(node_id, attr_name.clone(), attr_value)
                                .map_err(graph_error_to_py_err)?;
                        }
                    }
                }
                
                Ok(Some(PySubgraph::new(
                    path.nodes,
                    path.edges,
                    "shortest_path".to_string(),
                    Some(self.graph.clone()),
                )))
            },
            None => Ok(None),
        }
    }
    
    /// Get node degree
    fn degree(&self, py: Python, node: NodeId) -> PyResult<usize> {
        let graph = self.graph.borrow(py);
        graph.inner.degree(node)
            .map_err(graph_error_to_py_err)
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
        dict.set_item("change_tracker_memory_bytes", stats.change_tracker_memory_bytes)?;
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
        
        Ok(format!("Graph Analytics: {} nodes, {} edges", node_count, edge_count))
    }
}

use pyo3::prelude::*;

// Placeholder - will extract analytics operations from lib_old.rs
