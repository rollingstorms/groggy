use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use serde_json::Value as JsonValue;
use dashmap::DashMap;
use crate::graph::types::{GraphType, NodeData, EdgeData};
use crate::utils::{python_dict_to_json_map, python_to_json_value, json_value_to_python};

/// High-performance graph structure implemented in Rust
#[pyclass]
pub struct FastGraph {
    pub graph: GraphType,
    pub node_id_to_index: DashMap<String, petgraph::graph::NodeIndex>,
    pub node_index_to_id: DashMap<petgraph::graph::NodeIndex, String>,
    pub graph_attributes: HashMap<String, JsonValue>,
    pub is_directed: bool,
}

#[pymethods]
impl FastGraph {
    #[new]
    #[pyo3(signature = (directed = true))]
    pub fn new(directed: bool) -> Self {
        let graph = if directed {
            GraphType::new_directed()
        } else {
            GraphType::new_undirected()
        };
        
        Self {
            graph,
            node_id_to_index: DashMap::new(),
            node_index_to_id: DashMap::new(),
            graph_attributes: HashMap::new(),
            is_directed: directed,
        }
    }

    /// Get whether this graph is directed
    pub fn is_directed(&self) -> bool {
        self.is_directed
    }

    /// Add a single node to the graph
    pub fn add_node(&mut self, node_id: String, attributes: Option<&PyDict>) -> PyResult<()> {
        if self.node_id_to_index.contains_key(&node_id) {
            return Ok(()); // Node already exists
        }

        let attrs = if let Some(py_attrs) = attributes {
            python_dict_to_json_map(py_attrs)?
        } else {
            HashMap::new()
        };

        let node_data = NodeData { 
            id: node_id.clone(), 
            attributes: attrs 
        };
        
        let node_index = self.graph.add_node(node_data);
        
        self.node_id_to_index.insert(node_id.clone(), node_index);
        self.node_index_to_id.insert(node_index, node_id);
        
        Ok(())
    }

    /// Remove a node from the graph
    pub fn remove_node(&mut self, node_id: String) -> PyResult<bool> {
        if let Some((_, node_idx)) = self.node_id_to_index.remove(&node_id) {
            self.node_index_to_id.remove(&node_idx);
            self.graph.remove_node(node_idx);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Add multiple nodes efficiently
    fn add_nodes(&mut self, node_data: &PyList) -> PyResult<()> {
        for item in node_data.iter() {
            let node_info: (String, Option<&PyDict>) = item.extract()?;
            self.add_node(node_info.0, node_info.1)?;
        }
        Ok(())
    }

    /// Remove multiple nodes efficiently  
    fn remove_nodes(&mut self, node_ids: Vec<String>) -> PyResult<usize> {
        let mut removed_count = 0;
        for node_id in node_ids {
            if self.remove_node(node_id)? {
                removed_count += 1;
            }
        }
        Ok(removed_count)
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, source: String, target: String, attributes: Option<&PyDict>) -> PyResult<()> {
        let source_idx = self.node_id_to_index.get(&source)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Source node '{}' not found", source)
            ))?;
        let target_idx = self.node_id_to_index.get(&target)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Target node '{}' not found", target)
            ))?;

        let attrs = if let Some(py_attrs) = attributes {
            python_dict_to_json_map(py_attrs)?
        } else {
            HashMap::new()
        };

        let edge_data = EdgeData {
            source: source.clone(),
            target: target.clone(),
            attributes: attrs,
        };

        self.graph.add_edge(*source_idx, *target_idx, edge_data);
        Ok(())
    }

    /// Remove an edge between two nodes
    pub fn remove_edge(&mut self, source: String, target: String) -> PyResult<bool> {
        let source_idx = self.node_id_to_index.get(&source)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Source node '{}' not found", source)
            ))?;
        let target_idx = self.node_id_to_index.get(&target)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Target node '{}' not found", target)
            ))?;

        if let Some(edge_idx) = self.graph.find_edge(*source_idx, *target_idx) {
            self.graph.remove_edge(edge_idx);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Add multiple edges efficiently
    fn add_edges(&mut self, edge_data: &PyList) -> PyResult<()> {
        for item in edge_data.iter() {
            let edge_info: (String, String, Option<&PyDict>) = item.extract()?;
            self.add_edge(edge_info.0, edge_info.1, edge_info.2)?;
        }
        Ok(())
    }

    /// Remove multiple edges efficiently
    fn remove_edges(&mut self, edge_pairs: Vec<(String, String)>) -> PyResult<usize> {
        let mut removed_count = 0;
        for (source, target) in edge_pairs {
            if self.remove_edge(source, target)? {
                removed_count += 1;
            }
        }
        Ok(removed_count)
    }

    /// Get node count
    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get edge count
    fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get neighbors of a node (both incoming and outgoing)
    fn get_neighbors(&self, node_id: String) -> PyResult<Vec<String>> {
        use petgraph::visit::EdgeRef;
        
        let node_idx = self.node_id_to_index.get(&node_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ))?;
        
        let mut neighbors: Vec<String> = Vec::new();
        
        // Get outgoing neighbors
        for neighbor_idx in self.graph.neighbors(*node_idx) {
            if let Some(id) = self.node_index_to_id.get(&neighbor_idx) {
                neighbors.push(id.clone());
            }
        }
        
        // Get incoming neighbors
        for edge_ref in self.graph.edges_directed(*node_idx, petgraph::Direction::Incoming) {
            let source_idx = edge_ref.source();
            if let Some(id) = self.node_index_to_id.get(&source_idx) {
                let id_str = id.clone();
                if !neighbors.contains(&id_str) {  // Avoid duplicates
                    neighbors.push(id_str);
                }
            }
        }
        
        Ok(neighbors)
    }

    /// Get outgoing neighbors of a node
    fn get_outgoing_neighbors(&self, node_id: String) -> PyResult<Vec<String>> {
        let node_idx = self.node_id_to_index.get(&node_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ))?;
        
        let mut neighbors: Vec<String> = Vec::new();
        
        // Get outgoing neighbors only
        for neighbor_idx in self.graph.neighbors(*node_idx) {
            if let Some(id) = self.node_index_to_id.get(&neighbor_idx) {
                neighbors.push(id.clone());
            }
        }
        
        Ok(neighbors)
    }

    /// Get incoming neighbors of a node
    fn get_incoming_neighbors(&self, node_id: String) -> PyResult<Vec<String>> {
        use petgraph::visit::EdgeRef;
        
        let node_idx = self.node_id_to_index.get(&node_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ))?;
        
        let mut neighbors: Vec<String> = Vec::new();
        
        // Get incoming neighbors only
        for edge_ref in self.graph.edges_directed(*node_idx, petgraph::Direction::Incoming) {
            let source_idx = edge_ref.source();
            if let Some(id) = self.node_index_to_id.get(&source_idx) {
                neighbors.push(id.clone());
            }
        }
        
        Ok(neighbors)
    }

    /// Get all neighbors of a node (both incoming and outgoing)
    fn get_all_neighbors(&self, node_id: String) -> PyResult<Vec<String>> {
        // This is the same as get_neighbors for now
        self.get_neighbors(node_id)
    }

    /// Get all node IDs
    pub fn get_node_ids(&self) -> Vec<String> {
        self.node_id_to_index.iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get all edge IDs (formatted as "source->target")
    pub fn get_edge_ids(&self) -> Vec<String> {
        let mut edge_ids = Vec::new();
        for edge_idx in self.graph.edge_indices() {
            if let Some(edge_data) = self.graph.edge_weight(edge_idx) {
                let edge_id = format!("{}->{}", edge_data.source, edge_data.target);
                edge_ids.push(edge_id);
            }
        }
        edge_ids
    }

    /// Get node attributes
    pub fn get_node_attributes(&self, node_id: String) -> PyResult<PyObject> {
        let node_idx = self.node_id_to_index.get(&node_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ))?;
        
        if let Some(node_data) = self.graph.node_weight(*node_idx) {
            Python::with_gil(|py| {
                let dict = PyDict::new(py);
                for (key, value) in &node_data.attributes {
                    let py_value = json_value_to_python(py, value)?;
                    dict.set_item(key, py_value)?;
                }
                Ok(dict.into_py(py))
            })
        } else {
            Python::with_gil(|py| Ok(PyDict::new(py).into_py(py)))
        }
    }

    /// Get edge attributes
    pub fn get_edge_attributes(&self, source: String, target: String) -> PyResult<PyObject> {
        let source_idx = self.node_id_to_index.get(&source)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Source node '{}' not found", source)
            ))?;
        let target_idx = self.node_id_to_index.get(&target)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Target node '{}' not found", target)
            ))?;
        
        if let Some(edge_idx) = self.graph.find_edge(*source_idx, *target_idx) {
            if let Some(edge_data) = self.graph.edge_weight(edge_idx) {
                Python::with_gil(|py| {
                    let dict = PyDict::new(py);
                    for (key, value) in &edge_data.attributes {
                        let py_value = json_value_to_python(py, value)?;
                        dict.set_item(key, py_value)?;
                    }
                    Ok(dict.into_py(py))
                })
            } else {
                Python::with_gil(|py| Ok(PyDict::new(py).into_py(py)))
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Edge from '{}' to '{}' not found", source, target)
            ))
        }
    }

    /// Set node attribute
    pub fn set_node_attribute(&mut self, node_id: String, key: String, value: &PyAny) -> PyResult<()> {
        let node_idx = self.node_id_to_index.get(&node_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ))?;
        
        if let Some(node_data) = self.graph.node_weight_mut(*node_idx) {
            let json_value = python_to_json_value(value)?;
            node_data.attributes.insert(key, json_value);
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ))
        }
    }

    /// Set edge attribute
    pub fn set_edge_attribute(&mut self, source: String, target: String, key: String, value: &PyAny) -> PyResult<()> {
        let source_idx = self.node_id_to_index.get(&source)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Source node '{}' not found", source)
            ))?;
        let target_idx = self.node_id_to_index.get(&target)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Target node '{}' not found", target)
            ))?;
        
        if let Some(edge_idx) = self.graph.find_edge(*source_idx, *target_idx) {
            if let Some(edge_data) = self.graph.edge_weight_mut(edge_idx) {
                let json_value = python_to_json_value(value)?;
                edge_data.attributes.insert(key, json_value);
                Ok(())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Edge from '{}' to '{}' not found", source, target)
                ))
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Edge from '{}' to '{}' not found", source, target)
            ))
        }
    }

    /// Create a deep copy of the graph
    fn copy(&self) -> Self {
        FastGraph {
            graph: self.graph.clone(),
            node_id_to_index: self.node_id_to_index.clone(),
            node_index_to_id: self.node_index_to_id.clone(),
            graph_attributes: self.graph_attributes.clone(),
            is_directed: self.is_directed,
        }
    }

    /// Get graph statistics
    fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("nodes".to_string(), self.graph.node_count());
        stats.insert("edges".to_string(), self.graph.edge_count());
        stats
    }

    /// Batch filter nodes by attribute values - highly optimized
    #[pyo3(name = "filter_nodes_by_attributes")]
    fn filter_nodes_by_attributes(&self, filters: &PyDict) -> PyResult<Vec<String>> {
        let mut results = Vec::new();
        let filter_map = python_dict_to_json_map(filters)?;
        
        // Use parallel processing for large graphs
        if self.graph.node_count() > 1000 {
            let node_ids: Vec<_> = self.node_id_to_index.iter()
                .filter_map(|entry| {
                    let node_id = entry.key();
                    let node_idx = *entry.value();
                    
                    if let Some(node_data) = self.graph.node_weight(node_idx) {
                        if self.matches_filters(&node_data.attributes, &filter_map) {
                            Some(node_id.clone())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
            
            results.extend(node_ids);
        } else {
            // Sequential for smaller graphs
            for entry in self.node_id_to_index.iter() {
                let node_id = entry.key();
                let node_idx = *entry.value();
                
                if let Some(node_data) = self.graph.node_weight(node_idx) {
                    if self.matches_filters(&node_data.attributes, &filter_map) {
                        results.push(node_id.clone());
                    }
                }
            }
        }
        
        Ok(results)
    }

    /// Batch filter edges by attribute values
    #[pyo3(name = "filter_edges_by_attributes")]
    fn filter_edges_by_attributes(&self, filters: &PyDict) -> PyResult<Vec<(String, String)>> {
        let mut results = Vec::new();
        let filter_map = python_dict_to_json_map(filters)?;
        
        for edge_idx in self.graph.edge_indices() {
            if let Some(edge_data) = self.graph.edge_weight(edge_idx) {
                if self.matches_filters(&edge_data.attributes, &filter_map) {
                    results.push((edge_data.source.clone(), edge_data.target.clone()));
                }
            }
        }
        
        Ok(results)
    }

    /// Get subgraph by node IDs - optimized batch operation
    fn get_subgraph_by_node_ids(&self, node_ids: Vec<String>) -> PyResult<FastGraph> {
        let mut subgraph = FastGraph::new(self.is_directed);
        let node_id_set: std::collections::HashSet<_> = node_ids.iter().collect();
        
        // Batch add nodes
        let mut nodes_to_add = Vec::new();
        for node_id in &node_ids {
            if let Some(node_idx) = self.node_id_to_index.get(node_id) {
                if let Some(node_data) = self.graph.node_weight(*node_idx) {
                    nodes_to_add.push((node_id.clone(), node_data.attributes.clone()));
                }
            }
        }
        
        // Add nodes to subgraph
        for (node_id, attributes) in nodes_to_add {
            let node_data = NodeData { id: node_id.clone(), attributes };
            let node_idx = subgraph.graph.add_node(node_data);
            subgraph.node_id_to_index.insert(node_id.clone(), node_idx);
            subgraph.node_index_to_id.insert(node_idx, node_id);
        }
        
        // Batch add edges between included nodes
        let mut edges_to_add = Vec::new();
        for edge_idx in self.graph.edge_indices() {
            if let Some(edge_data) = self.graph.edge_weight(edge_idx) {
                if node_id_set.contains(&edge_data.source) && node_id_set.contains(&edge_data.target) {
                    edges_to_add.push(edge_data.clone());
                }
            }
        }
        
        for edge_data in edges_to_add {
            let source_idx = *subgraph.node_id_to_index.get(&edge_data.source).unwrap();
            let target_idx = *subgraph.node_id_to_index.get(&edge_data.target).unwrap();
            subgraph.graph.add_edge(source_idx, target_idx, edge_data);
        }
        
        Ok(subgraph)
    }

    /// Get k-hop neighborhood efficiently
    fn get_k_hop_neighborhood(&self, start_node: String, k: usize) -> PyResult<Vec<String>> {
        let mut visited = std::collections::HashSet::new();
        let mut current_layer = vec![start_node.clone()];
        visited.insert(start_node);
        
        for _ in 0..k {
            let mut next_layer = Vec::new();
            
            for node_id in current_layer {
                if let Some(neighbors) = self.get_neighbors(node_id.clone()).ok() {
                    for neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor.clone());
                            next_layer.push(neighbor);
                        }
                    }
                }
            }
            
            current_layer = next_layer;
            if current_layer.is_empty() {
                break;
            }
        }
        
        Ok(visited.into_iter().collect())
    }

    /// Batch get node attributes - optimized for multiple queries
    #[pyo3(name = "get_nodes_attributes")]
    fn get_nodes_attributes(&self, py: Python, node_ids: Vec<String>) -> PyResult<PyObject> {
        let py_list = PyList::empty(py);
        
        for node_id in node_ids {
            if let Some(node_idx) = self.node_id_to_index.get(&node_id) {
                if let Some(node_data) = self.graph.node_weight(*node_idx) {
                    let py_dict = PyDict::new(py);
                    for (key, value) in &node_data.attributes {
                        py_dict.set_item(key, json_value_to_python(py, value)?)?;
                    }
                    py_list.append(py_dict)?;
                } else {
                    py_list.append(PyDict::new(py))?;
                }
            } else {
                py_list.append(PyDict::new(py))?;
            }
        }
        
        Ok(py_list.to_object(py))
    }

    /// Batch set node attributes - highly optimized for large operations
    pub fn set_nodes_attributes_batch(&mut self, node_attrs: &PyDict) -> PyResult<()> {
        let mut updates = Vec::new();
        
        // Collect all updates first to avoid borrowing conflicts
        for (node_id_py, attrs_py) in node_attrs.iter() {
            let node_id: String = node_id_py.extract()?;
            let attrs_dict: &PyDict = attrs_py.downcast()?;
            
            if let Some(node_idx) = self.node_id_to_index.get(&node_id) {
                let mut attr_updates = HashMap::new();
                for (key_py, value_py) in attrs_dict.iter() {
                    let key: String = key_py.extract()?;
                    let json_value = python_to_json_value(value_py)?;
                    attr_updates.insert(key, json_value);
                }
                updates.push((*node_idx, attr_updates));
            }
        }
        
        // Apply all updates sequentially (parallel mutation not supported)
        for (node_idx, attr_updates) in updates {
            if let Some(node_data) = self.graph.node_weight_mut(node_idx) {
                for (key, value) in attr_updates {
                    node_data.attributes.insert(key, value);
                }
            }
        }
        
        Ok(())
    }

    /// Batch set edge attributes - highly optimized for large operations
    pub fn set_edges_attributes_batch(&mut self, edge_attrs: &PyDict) -> PyResult<()> {
        let mut edge_updates = Vec::new();
        
        // Collect all edge updates first
        for (edge_key_py, attrs_py) in edge_attrs.iter() {
            let edge_key: (String, String) = edge_key_py.extract()?;
            let (source, target) = edge_key;
            let attrs_dict: &PyDict = attrs_py.downcast()?;
            
            if let (Some(source_idx), Some(target_idx)) = (
                self.node_id_to_index.get(&source),
                self.node_id_to_index.get(&target)
            ) {
                // Find the edge
                if let Some(edge_ref) = self.graph.find_edge(*source_idx, *target_idx) {
                    let mut attr_updates = HashMap::new();
                    for (key_py, value_py) in attrs_dict.iter() {
                        let key: String = key_py.extract()?;
                        let json_value = python_to_json_value(value_py)?;
                        attr_updates.insert(key, json_value);
                    }
                    edge_updates.push((edge_ref, attr_updates));
                }
            }
        }
        
        // Apply edge updates
        for (edge_ref, attr_updates) in edge_updates {
            if let Some(edge_data) = self.graph.edge_weight_mut(edge_ref) {
                for (key, value) in attr_updates {
                    edge_data.attributes.insert(key, value);
                }
            }
        }
        
        Ok(())
    }

    /// Get graph attributes
    pub fn get_graph_attributes(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for (key, value) in &self.graph_attributes {
                let py_value = json_value_to_python(py, value)?;
                dict.set_item(key, py_value)?;
            }
            Ok(dict.into_py(py))
        })
    }

    /// Set graph attribute
    pub fn set_graph_attribute(&mut self, key: String, value: &PyAny) -> PyResult<()> {
        let json_value = python_to_json_value(value)?;
        self.graph_attributes.insert(key, json_value);
        Ok(())
    }
}

impl FastGraph {
    /// Helper method to check if attributes match filters
    fn matches_filters(&self, attributes: &HashMap<String, JsonValue>, filters: &HashMap<String, JsonValue>) -> bool {
        for (key, expected_value) in filters {
            if let Some(actual_value) = attributes.get(key) {
                if actual_value != expected_value {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    /// Get node weight by index (public for other modules)
    pub fn get_node_weight(&self, node_idx: NodeIndex) -> Option<&NodeData> {
        self.graph.node_weight(node_idx)
    }

    /// Get edge weight by index (public for other modules)  
    pub fn get_edge_weight(&self, edge_idx: petgraph::graph::EdgeIndex) -> Option<&EdgeData> {
        self.graph.edge_weight(edge_idx)
    }

    /// Get edge indices (public for other modules)
    pub fn get_edge_indices(&self) -> Vec<petgraph::graph::EdgeIndex> {
        self.graph.edge_indices()
    }

    /// Get edge endpoints (public for other modules)
    pub fn get_edge_endpoints(&self, edge_idx: petgraph::graph::EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.graph.edge_endpoints(edge_idx)
    }

    /// Add node to graph directly (public for other modules)
    pub fn add_node_to_graph_public(&mut self, node_data: NodeData) -> NodeIndex {
        let node_idx = self.graph.add_node(node_data.clone());
        self.node_id_to_index.insert(node_data.id.clone(), node_idx);
        self.node_index_to_id.insert(node_idx, node_data.id);
        node_idx
    }

    /// Add edge to graph directly (public for other modules)
    pub fn add_edge_to_graph_public(&mut self, source_idx: NodeIndex, target_idx: NodeIndex, edge_data: EdgeData) -> petgraph::graph::EdgeIndex {
        self.graph.add_edge(source_idx, target_idx, edge_data)
    }

    /// Get neighbors publicly (for other modules)
    pub fn get_neighbors_public(&self, node_idx: NodeIndex) -> Vec<NodeIndex> {
        self.graph.neighbors(node_idx)
    }

    /// Get edges directed publicly (for other modules)
    pub fn get_edges_directed(&self, node_idx: NodeIndex, direction: petgraph::Direction) -> Vec<petgraph::graph::EdgeReference<EdgeData>> {
        self.graph.edges_directed(node_idx, direction)
    }

    /// Find edge publicly (for other modules)
    pub fn get_find_edge(&self, source_idx: NodeIndex, target_idx: NodeIndex) -> Option<petgraph::graph::EdgeIndex> {
        self.graph.find_edge(source_idx, target_idx)
    }
}
