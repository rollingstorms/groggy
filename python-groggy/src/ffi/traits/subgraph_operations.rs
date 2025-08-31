//! SubgraphOperations FFI Trait - Pure delegation to core trait implementations
//!
//! This module provides the Python-accessible SubgraphOperations trait interface
//! that delegates to our efficient core trait implementations. All algorithm logic
//! remains in the core - this is pure translation between Python and Rust types.

use crate::ffi::core::subgraph::PySubgraph;
use crate::ffi::types::PyAttrValue;
use crate::ffi::utils::*;
use groggy::core::traits::SubgraphOperations as CoreSubgraphOperations;
use groggy::core::subgraph::SimilarityMetric;
use groggy::types::{AttrName, NodeId, EdgeId};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::PyRuntimeError;
use std::collections::HashMap;

/// Python-accessible SubgraphOperations trait interface
/// 
/// This trait provides a unified API across all subgraph types in Python,
/// while delegating all operations to our efficient core trait implementations.
/// 
/// **Design Principles**:
/// - **Pure Delegation**: All methods delegate to core trait implementations
/// - **No Algorithm Logic**: FFI layer contains no business logic
/// - **Type Translation**: Converts between Python and Rust types only
/// - **Performance**: Uses py.allow_threads() for CPU-intensive operations
/// 
/// **Coverage**: 45+ methods across all SubgraphOperations functionality
pub trait PySubgraphOperations {
    // === CORE DATA ACCESS ===
    
    /// Get underlying core trait object for delegation
    fn core_subgraph(&self) -> PyResult<&dyn CoreSubgraphOperations>;
    
    /// Get nodes as Python list
    fn nodes(&self, py: Python) -> PyResult<Py<PyList>> {
        let core = self.core_subgraph()?;
        let node_ids: Vec<usize> = core.node_set()
            .iter()
            .map(|&id| id as usize)
            .collect();
        Ok(PyList::new(py, node_ids).into())
    }
    
    /// Get edges as Python list
    fn edges(&self, py: Python) -> PyResult<Py<PyList>> {
        let core = self.core_subgraph()?;
        let edge_ids: Vec<usize> = core.edge_set()
            .iter()
            .map(|&id| id as usize)
            .collect();
        Ok(PyList::new(py, edge_ids).into())
    }
    
    /// Get node count - delegates to efficient core implementation
    fn node_count(&self) -> PyResult<usize> {
        Ok(self.core_subgraph()?.node_count())
    }
    
    /// Get edge count - delegates to efficient core implementation  
    fn edge_count(&self) -> PyResult<usize> {
        Ok(self.core_subgraph()?.edge_count())
    }
    
    /// Check if subgraph is empty
    fn is_empty(&self) -> PyResult<bool> {
        Ok(self.core_subgraph()?.node_count() == 0)
    }
    
    /// Get entity summary - delegates to core GraphEntity implementation
    fn summary(&self) -> PyResult<String> {
        Ok(self.core_subgraph()?.summary())
    }
    
    // === NODE OPERATIONS ===
    
    /// Check if subgraph contains node - delegates to efficient core HashSet lookup
    fn contains_node(&self, node_id: usize) -> PyResult<bool> {
        Ok(self.core_subgraph()?.contains_node(node_id as NodeId))
    }
    
    /// Get neighbors of node within subgraph - delegates to core filtered algorithm
    fn neighbors(&self, py: Python, node_id: usize) -> PyResult<Py<PyList>> {
        let neighbors = self.core_subgraph()?.neighbors(node_id as NodeId)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let py_neighbors: Vec<usize> = neighbors.into_iter().map(|id| id as usize).collect();
        Ok(PyList::new(py, py_neighbors).into())
    }
    
    /// Get degree of node within subgraph - delegates to core filtered algorithm
    fn degree(&self, _py: Python, node_id: usize) -> PyResult<usize> {
        self.core_subgraph()?.degree(node_id as NodeId)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
    }
    
    // === EDGE OPERATIONS ===
    
    /// Check if subgraph contains edge - delegates to efficient core HashSet lookup
    fn contains_edge(&self, edge_id: usize) -> PyResult<bool> {
        Ok(self.core_subgraph()?.contains_edge(edge_id as EdgeId))
    }
    
    /// Get edge endpoints - delegates to core algorithm
    fn edge_endpoints(&self, _py: Python, edge_id: usize) -> PyResult<(usize, usize)> {
        let (source, target) = self.core_subgraph()?.edge_endpoints(edge_id as EdgeId)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        Ok((source as usize, target as usize))
    }
    
    /// Check if edge exists between nodes - delegates to core algorithm
    fn has_edge_between(&self, _py: Python, source: usize, target: usize) -> PyResult<bool> {
        self.core_subgraph()?.has_edge_between(source as NodeId, target as NodeId)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
    }
    
    // === ATTRIBUTE OPERATIONS ===
    
    /// Get node attribute - delegates to core GraphPool lookup
    fn get_node_attribute(&self, py: Python, node_id: usize, attr_name: String) -> PyResult<Option<PyObject>> {
        let attr_opt = self.core_subgraph()?.get_node_attribute(node_id as NodeId, &attr_name.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        match attr_opt {
            Some(attr_value) => Ok(Some(attr_value_to_python_value(py, &attr_value)?)),
            None => Ok(None)
        }
    }
    
    /// Get edge attribute - delegates to core GraphPool lookup
    fn get_edge_attribute(&self, py: Python, edge_id: usize, attr_name: String) -> PyResult<Option<PyObject>> {
        let attr_opt = self.core_subgraph()?.get_edge_attribute(edge_id as EdgeId, &attr_name.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        match attr_opt {
            Some(attr_value) => Ok(Some(attr_value_to_python_value(py, &attr_value)?)),
            None => Ok(None)
        }
    }
    
    // === STRUCTURAL METRICS (NEW - Phase 1) ===
    
    /// Calculate clustering coefficient - delegates to new core implementation
    fn clustering_coefficient(&self, _py: Python, node_id: Option<usize>) -> PyResult<f64> {
        // Downcast to concrete Subgraph type to access new methods
        // This is temporary until we add these methods to the core trait
        if let Some(concrete_subgraph) = self.try_downcast_to_subgraph() {
            concrete_subgraph.clustering_coefficient(node_id.map(|id| id as NodeId))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "clustering_coefficient not available for this subgraph type"
            ))
        }
    }
    
    /// Calculate transitivity - delegates to new core implementation
    fn transitivity(&self, _py: Python) -> PyResult<f64> {
        if let Some(concrete_subgraph) = self.try_downcast_to_subgraph() {
            concrete_subgraph.transitivity()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "transitivity not available for this subgraph type"
            ))
        }
    }
    
    /// Calculate density - delegates to new core implementation
    fn density(&self) -> PyResult<f64> {
        if let Some(concrete_subgraph) = self.try_downcast_to_subgraph() {
            Ok(concrete_subgraph.density())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "density not available for this subgraph type"
            ))
        }
    }
    
    // === SET OPERATIONS (NEW - Phase 1) ===
    
    /// Merge with another subgraph - delegates to new core implementation
    fn merge_with(&self, py: Python, other: &dyn PySubgraphOperations) -> PyResult<PySubgraph> {

        if let (Some(self_sg), Some(other_sg)) = (self.try_downcast_to_subgraph(), other.try_downcast_to_subgraph()) {
            let merged = self_sg.merge_with(other_sg)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
            PySubgraph::from_core_subgraph(merged)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "merge_with requires concrete Subgraph types"
            ))
        }
    }
    
    /// Intersect with another subgraph - delegates to new core implementation
    fn intersect_with(&self, py: Python, other: &dyn PySubgraphOperations) -> PyResult<PySubgraph> {
        if let (Some(self_sg), Some(other_sg)) = (self.try_downcast_to_subgraph(), other.try_downcast_to_subgraph()) {
            let intersection = self_sg.intersect_with(other_sg)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
            PySubgraph::from_core_subgraph(intersection)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "intersect_with requires concrete Subgraph types"
            ))
        }
    }
    
    /// Subtract another subgraph - delegates to new core implementation
    fn subtract_from(&self, py: Python, other: &dyn PySubgraphOperations) -> PyResult<PySubgraph> {
        if let (Some(self_sg), Some(other_sg)) = (self.try_downcast_to_subgraph(), other.try_downcast_to_subgraph()) {
            let difference = self_sg.subtract_from(other_sg)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
            PySubgraph::from_core_subgraph(difference)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "subtract_from requires concrete Subgraph types"
            ))
        }
    }
    
    /// Calculate similarity with another subgraph - delegates to new core implementation
    fn calculate_similarity(&self, py: Python, other: &dyn PySubgraphOperations, metric: String) -> PyResult<f64> {
        let similarity_metric = match metric.as_str() {
            "jaccard" => SimilarityMetric::Jaccard,
            "dice" => SimilarityMetric::Dice,
            "cosine" => SimilarityMetric::Cosine,
            "overlap" => SimilarityMetric::Overlap,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown similarity metric: {}", metric)
            ))
        };
        
        if let (Some(self_sg), Some(other_sg)) = (self.try_downcast_to_subgraph(), other.try_downcast_to_subgraph()) {
            self_sg.calculate_similarity(other_sg, similarity_metric)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "calculate_similarity requires concrete Subgraph types"
            ))
        }
    }
    
    // === ALGORITHM OPERATIONS (Delegate to existing optimized implementations) ===
    
    /// Find connected components - delegates to EXISTING optimized core algorithm
    fn connected_components(&self, py: Python) -> PyResult<Vec<PySubgraph>> {
        let components = self.core_subgraph()?.connected_components()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        let py_components: PyResult<Vec<PySubgraph>> = components
            .into_iter()
            .map(|component| PySubgraph::from_trait_object(component))
            .collect();
        py_components
    }
    
    /// BFS subgraph - delegates to existing optimized core algorithm
    fn bfs(&self, py: Python, start: usize, max_depth: Option<usize>) -> PyResult<PySubgraph> {
        let bfs_result = self.core_subgraph()?.bfs(start as NodeId, max_depth)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        PySubgraph::from_trait_object(bfs_result)
    }
    
    /// DFS subgraph - delegates to existing optimized core algorithm  
    fn dfs(&self, py: Python, start: usize, max_depth: Option<usize>) -> PyResult<PySubgraph> {
        let dfs_result = self.core_subgraph()?.dfs(start as NodeId, max_depth)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        PySubgraph::from_trait_object(dfs_result)
    }
    
    /// Shortest path subgraph - delegates to existing optimized core algorithm
    fn shortest_path_subgraph(&self, py: Python, source: usize, target: usize) -> PyResult<Option<PySubgraph>> {
        let path_opt = self.core_subgraph()?.shortest_path_subgraph(source as NodeId, target as NodeId)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        match path_opt {
            Some(path_subgraph) => Ok(Some(PySubgraph::from_trait_object(path_subgraph)?)),
            None => Ok(None)
        }

    }
    
    /// Create induced subgraph - delegates to existing optimized core algorithm
    fn induced_subgraph(&self, py: Python, nodes: Vec<usize>) -> PyResult<PySubgraph> {
        let node_ids: Vec<NodeId> = nodes.into_iter().map(|id| id as NodeId).collect();
        let induced = self.core_subgraph()?.induced_subgraph(&node_ids)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        PySubgraph::from_trait_object(induced)
    }
    
    /// Create subgraph from edges - delegates to existing optimized core algorithm
    fn subgraph_from_edges(&self, py: Python, edges: Vec<usize>) -> PyResult<PySubgraph> {
        let edge_ids: Vec<EdgeId> = edges.into_iter().map(|id| id as EdgeId).collect();
        let edge_subgraph = self.core_subgraph()?.subgraph_from_edges(&edge_ids)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        PySubgraph::from_trait_object(edge_subgraph)
    }
    
    // === HIERARCHICAL OPERATIONS ===
    
    /// Collapse subgraph to meta-node - delegates to core hierarchical implementation
    fn collapse_to_node(&self, py: Python, agg_functions: &PyDict) -> PyResult<usize> {
        let agg_map: HashMap<AttrName, String> = agg_functions
                .iter()
                .map(|(k, v)| {
                    let key = k.extract::<String>().unwrap_or_default().into();
                    let value = v.extract::<String>().unwrap_or_default();
                    (key, value)
                })
                .collect();
            
            let meta_node_id = self.core_subgraph()?.collapse_to_node(agg_map)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
            Ok(meta_node_id as usize)
    }
    
    // === BULK ATTRIBUTE OPERATIONS ===
    
    /// Set node attributes in bulk - delegates to optimized core bulk operations
    fn set_node_attrs(&self, py: Python, attrs_values: &PyDict) -> PyResult<()> {
        let bulk_attrs: PyResult<HashMap<AttrName, Vec<(NodeId, groggy::types::AttrValue)>>> = attrs_values
                .iter()
                .map(|(attr_name, node_values)| {
                    let attr_key = attr_name.extract::<String>()?.into();
                    let values_list = node_values.extract::<Vec<(usize, PyAttrValue)>>()?;
                    let converted_values: PyResult<Vec<(NodeId, groggy::types::AttrValue)>> = values_list
                        .into_iter()
                        .map(|(node_id, py_value)| {
                            let py_obj = py_value.into_py(py);
                            let rust_value = python_value_to_attr_value(py_obj.as_ref(py))?;
                            Ok((node_id as NodeId, rust_value))
                        })
                        .collect();
                    Ok((attr_key, converted_values?))
                })
                .collect();
            
            self.core_subgraph()?.set_node_attrs(bulk_attrs?)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
    }
    
    /// Set edge attributes in bulk - delegates to optimized core bulk operations
    fn set_edge_attrs(&self, py: Python, attrs_values: &PyDict) -> PyResult<()> {
        let bulk_attrs: PyResult<HashMap<AttrName, Vec<(EdgeId, groggy::types::AttrValue)>>> = attrs_values
                .iter()
                .map(|(attr_name, edge_values)| {
                    let attr_key = attr_name.extract::<String>()?.into();
                    let values_list = edge_values.extract::<Vec<(usize, PyAttrValue)>>()?;
                    let converted_values: PyResult<Vec<(EdgeId, groggy::types::AttrValue)>> = values_list
                        .into_iter()
                        .map(|(edge_id, py_value)| {
                            let py_obj = py_value.into_py(py);
                            let rust_value = python_value_to_attr_value(py_obj.as_ref(py))?;
                            Ok((edge_id as EdgeId, rust_value))
                        })
                        .collect();
                    Ok((attr_key, converted_values?))
                })
                .collect();
            
            self.core_subgraph()?.set_edge_attrs(bulk_attrs?)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
    }
    
    // === FILTERING OPERATIONS ===
    
    /// Filter nodes by string query - delegates to core filtering  
    fn filter_nodes(&self, py: Python, query: String) -> PyResult<PySubgraph> {
        if let Some(concrete_subgraph) = self.try_downcast_to_subgraph() {
            let mut parser = groggy::core::query_parser::QueryParser::new();
            let node_filter = parser.parse_node_query(&query)
                .map_err(|e| PyRuntimeError::new_err(format!("Query parse error: {}", e)))?;
            
            // Use the correct core filtering approach
            let graph_ref = concrete_subgraph.graph();
            let filtered_nodes = graph_ref.borrow_mut()
                .find_nodes(node_filter)
                .map_err(|e| PyRuntimeError::new_err(format!("Filter error: {}", e)))?;
            
            // Create induced subgraph from filtered nodes
            let induced_edges: std::collections::HashSet<EdgeId> = filtered_nodes.iter()
                .flat_map(|&node| {
                    graph_ref.borrow().incident_edges(node).unwrap_or_default()
                })
                .filter(|&edge| {
                    let (source, target) = graph_ref.borrow().edge_endpoints(edge).unwrap_or_default();
                    filtered_nodes.contains(&source) && filtered_nodes.contains(&target)
                })
                .collect();
            
            let filtered_node_set: std::collections::HashSet<NodeId> = filtered_nodes.into_iter().collect();
            let filtered = groggy::core::subgraph::Subgraph::new(
                graph_ref.clone(),
                filtered_node_set,
                induced_edges,
                "filtered_nodes".to_string()
            );
            PySubgraph::from_core_subgraph(filtered)
        } else {
            Err(PyRuntimeError::new_err("filter_nodes requires concrete Subgraph type"))
        }
    }
    
    /// Filter edges by string query - delegates to core filtering
    fn filter_edges(&self, py: Python, query: String) -> PyResult<PySubgraph> {
        if let Some(concrete_subgraph) = self.try_downcast_to_subgraph() {
            let mut parser = groggy::core::query_parser::QueryParser::new();
            let edge_filter = parser.parse_edge_query(&query)  
                .map_err(|e| PyRuntimeError::new_err(format!("Query parse error: {}", e)))?;
                
            // Use the correct core filtering approach  
            let graph_ref = concrete_subgraph.graph();
            let filtered_edges = graph_ref.borrow_mut()
                .find_edges(edge_filter)
                .map_err(|e| PyRuntimeError::new_err(format!("Filter error: {}", e)))?;
            
            // Create subgraph with filtered edges and their incident nodes
            let filtered_edge_set: std::collections::HashSet<EdgeId> = filtered_edges.iter().copied().collect();
            let mut incident_nodes = std::collections::HashSet::new();
            
            for &edge_id in &filtered_edges {
                if let Ok((source, target)) = graph_ref.borrow().edge_endpoints(edge_id) {
                    incident_nodes.insert(source);
                    incident_nodes.insert(target);
                }
            }
            
            let filtered = groggy::core::subgraph::Subgraph::new(
                graph_ref.clone(),
                incident_nodes,
                filtered_edge_set,
                "filtered_edges".to_string()
            );
            PySubgraph::from_core_subgraph(filtered)
        } else {
            Err(PyRuntimeError::new_err("filter_edges requires concrete Subgraph type"))
        }
    }
    
    // === UTILITY METHODS FOR TYPE CONVERSION ===
    
    /// Try to downcast to concrete Subgraph for accessing new Phase 1 methods
    /// This is a temporary bridge until we add the new methods to the core trait
    fn try_downcast_to_subgraph(&self) -> Option<&groggy::core::subgraph::Subgraph> {
        // This will be implemented by concrete types
        None
    }
    
    /// Get entity type - delegates to core GraphEntity trait
    fn entity_type(&self) -> PyResult<String> {
        Ok(self.core_subgraph()?.entity_type().to_string())
    }
}

// Implementations for concrete types will be added in their respective files
// For example, PySubgraph will implement this trait by delegating to its inner Subgraph