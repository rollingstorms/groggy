use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use petgraph::graph::NodeIndex;

use crate::graph::types::{GraphType, NodeData, EdgeData};
use crate::storage::{ColumnarStore};
use crate::utils::{python_dict_to_json_map};

/// FastGraph struct with hybrid/columnar storage (MAIN IMPLEMENTATION)
#[pyclass]
pub struct FastGraph {
    /// The actual graph structure using the new lightweight types
    pub graph: GraphType,
    
    /// Columnar storage for attributes
    pub columnar_store: ColumnarStore,
    
    /// Bidirectional mappings for node IDs
    pub node_id_to_index: HashMap<String, NodeIndex>,
    pub node_index_to_id: HashMap<NodeIndex, String>,
    
    /// Edge tracking
    pub edge_index_to_endpoints: HashMap<petgraph::graph::EdgeIndex, (String, String)>,
}

#[pymethods]
impl FastGraph {
    #[new]
    pub fn new(directed: bool) -> Self {
        let graph = if directed {
            GraphType::new_directed()
        } else {
            GraphType::new_undirected()
        };

        Self {
            graph,
            columnar_store: ColumnarStore::new(),
            node_id_to_index: HashMap::new(),
            node_index_to_id: HashMap::new(),
            edge_index_to_endpoints: HashMap::new(),
        }
    }

    /// Add a single node with optional attributes
    pub fn add_node(&mut self, node_id: String, attributes: Option<&PyDict>) -> PyResult<()> {
        // Check if node already exists
        if self.node_id_to_index.contains_key(&node_id) {
            return Ok(()); // Node already exists
        }

        // Create node data (lightweight)
        let node_data = NodeData {
            id: node_id.clone(),
            attr_uids: std::collections::HashSet::new(),
        };
        
        // Add to graph topology
        let node_index = self.graph.add_node(node_data);
        
        // Update mappings
        self.node_id_to_index.insert(node_id.clone(), node_index);
        self.node_index_to_id.insert(node_index, node_id.clone());
        
        // Store attributes in columnar format
        if let Some(attrs) = attributes {
            let attr_map = python_dict_to_json_map(attrs)?;
            for (attr_name, attr_value) in attr_map {
                self.columnar_store.set_node_attribute(node_index.index(), &attr_name, attr_value);
            }
        }
        
        Ok(())
    }

    /// Add a single edge with optional attributes
    pub fn add_edge(&mut self, source: String, target: String, attributes: Option<&PyDict>) -> PyResult<()> {
        // Get node indices
        let source_idx = self.node_id_to_index.get(&source)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Source node '{}' not found", source)
            ))?;
        let target_idx = self.node_id_to_index.get(&target)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Target node '{}' not found", target)
            ))?;

        // Create edge data (lightweight)
        let edge_data = EdgeData {
            source: source.clone(),
            target: target.clone(),
            attr_uids: std::collections::HashSet::new(),
        };

        // Add to graph topology
        let edge_index = self.graph.add_edge(*source_idx, *target_idx, edge_data);
        
        // Store edge mapping
        self.edge_index_to_endpoints.insert(edge_index, (source, target));
        
        // Store attributes in columnar format
        if let Some(attrs) = attributes {
            let attr_map = python_dict_to_json_map(attrs)?;
            for (attr_name, attr_value) in attr_map {
                self.columnar_store.set_edge_attribute(edge_index.index(), &attr_name, attr_value);
            }
        }
        
        Ok(())
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get all node IDs
    pub fn get_node_ids(&self) -> Vec<String> {
        self.node_id_to_index.keys().cloned().collect()
    }

    /// Check if node exists
    pub fn has_node(&self, node_id: &str) -> bool {
        self.node_id_to_index.contains_key(node_id)
    }

    /// Check if edge exists
    pub fn has_edge(&self, source: &str, target: &str) -> bool {
        if let (Some(source_idx), Some(target_idx)) = (
            self.node_id_to_index.get(source),
            self.node_id_to_index.get(target)
        ) {
            self.graph.find_edge(*source_idx, *target_idx).is_some()
        } else {
            false
        }
    }

    /// Get neighbors of a node
    pub fn get_neighbors(&self, node_id: &str) -> PyResult<Vec<String>> {
        let node_idx = self.node_id_to_index.get(node_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ))?;

        let neighbor_indices = self.graph.neighbors(*node_idx);
        let neighbor_ids: Vec<String> = neighbor_indices
            .into_iter()
            .filter_map(|idx| self.node_index_to_id.get(&idx).cloned())
            .collect();

        Ok(neighbor_ids)
    }
}

// Internal methods (not exposed to Python)
impl FastGraph {
    /// Get node weight by index (internal use)
    pub fn get_node_weight(&self, node_idx: petgraph::graph::NodeIndex) -> Option<&NodeData> {
        self.graph.node_weight(node_idx)
    }

    /// Get edge weight by index (internal use) 
    pub fn get_edge_weight(&self, edge_idx: petgraph::graph::EdgeIndex) -> Option<&EdgeData> {
        self.graph.edge_weight(edge_idx)
    }

    /// Get all edge indices (internal use)
    pub fn get_edge_indices(&self) -> Vec<petgraph::graph::EdgeIndex> {
        self.graph.edge_indices()
    }

    /// Get edge endpoints (internal use)
    pub fn get_edge_endpoints(&self, edge_idx: petgraph::graph::EdgeIndex) -> Option<(petgraph::graph::NodeIndex, petgraph::graph::NodeIndex)> {
        self.graph.edge_endpoints(edge_idx)
    }

    /// Add node directly to graph (internal use)
    pub fn add_node_to_graph_public(&mut self, node_data: NodeData) -> petgraph::graph::NodeIndex {
        self.graph.add_node(node_data)
    }

    /// Add edge directly to graph (internal use)
    pub fn add_edge_to_graph_public(&mut self, source_idx: petgraph::graph::NodeIndex, target_idx: petgraph::graph::NodeIndex, edge_data: EdgeData) -> petgraph::graph::EdgeIndex {
        self.graph.add_edge(source_idx, target_idx, edge_data)
    }

    /// Get neighbors by index (internal use)
    pub fn get_neighbors_public(&self, node_idx: petgraph::graph::NodeIndex) -> Vec<petgraph::graph::NodeIndex> {
        self.graph.neighbors(node_idx)
    }

    /// Get directed edges (internal use)
    pub fn get_edges_directed(&self, node_idx: petgraph::graph::NodeIndex, direction: petgraph::Direction) -> Vec<petgraph::graph::EdgeReference<EdgeData>> {
        self.graph.edges_directed(node_idx, direction)
    }
}
