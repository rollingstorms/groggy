/// Bulk operations for efficient graph construction and modification
/// This module contains ONLY bulk/batch operations that don't belong in core
use super::core::FastGraph;
use super::types::{NodeData, EdgeData};
use crate::utils::python_dict_to_json_map;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{HashMap, HashSet};

impl FastGraph {
    /// Bulk add nodes with optimized columnar storage - INTERNAL USE ONLY
    pub(crate) fn bulk_add_nodes_internal(
        &mut self, 
        nodes_data: Vec<(String, HashMap<String, serde_json::Value>)>
    ) -> Vec<petgraph::graph::NodeIndex> {
        let mut node_indices = Vec::with_capacity(nodes_data.len());
        
        // Prepare bulk columnar operations
        let mut bulk_attributes: HashMap<String, Vec<(usize, serde_json::Value)>> = HashMap::new();
        
        // Add nodes to graph structure first
        for (node_id, attributes) in nodes_data {
            // Skip if node already exists
            if self.node_id_to_index.contains_key(&node_id) {
                continue;
            }
            
            // Create lightweight node data
            let node_data = NodeData {
                id: node_id.clone(),
                attr_uids: HashSet::new(),
            };
            
            // Add to graph topology
            let node_index = self.graph.add_node(node_data);
            
            // Update mappings
            self.node_id_to_index.insert(node_id.clone(), node_index);
            self.node_index_to_id.insert(node_index, node_id);
            
            node_indices.push(node_index);
            
            // Prepare attributes for bulk columnar insert
            for (attr_name, attr_value) in attributes {
                bulk_attributes
                    .entry(attr_name)
                    .or_insert_with(Vec::new)
                    .push((node_index.index(), attr_value));
            }
        }
        
        // Bulk insert attributes into columnar store
        for (attr_name, attr_data) in bulk_attributes {
            self.columnar_store.bulk_set_node_attributes(&attr_name, attr_data);
        }
        
        node_indices
    }
    
    /// Bulk add edges with optimized columnar storage - INTERNAL USE ONLY
    pub(crate) fn bulk_add_edges_internal(
        &mut self,
        edges_data: Vec<(String, String, HashMap<String, serde_json::Value>)>
    ) -> Vec<petgraph::graph::EdgeIndex> {
        let mut edge_indices = Vec::with_capacity(edges_data.len());
        
        // Prepare bulk columnar operations
        let mut bulk_attributes: HashMap<String, Vec<(usize, serde_json::Value)>> = HashMap::new();
        
        // Add edges to graph structure first
        for (source, target, attributes) in edges_data {
            // Get node indices
            let source_idx = match self.node_id_to_index.get(&source) {
                Some(idx) => *idx,
                None => continue, // Skip if source doesn't exist
            };
            let target_idx = match self.node_id_to_index.get(&target) {
                Some(idx) => *idx,
                None => continue, // Skip if target doesn't exist
            };
            
            // Create lightweight edge data
            let edge_data = EdgeData {
                source: source.clone(),
                target: target.clone(),
                attr_uids: HashSet::new(),
            };
            
            // Add to graph topology
            let edge_index = self.graph.add_edge(source_idx, target_idx, edge_data);
            
            // Store edge mapping
            self.edge_index_to_endpoints.insert(edge_index, (source, target));
            
            edge_indices.push(edge_index);
            
            // Prepare attributes for bulk columnar insert
            for (attr_name, attr_value) in attributes {
                bulk_attributes
                    .entry(attr_name)
                    .or_insert_with(Vec::new)
                    .push((edge_index.index(), attr_value));
            }
        }
        
        // Bulk insert attributes into columnar store
        for (attr_name, attr_data) in bulk_attributes {
            self.columnar_store.bulk_set_edge_attributes(&attr_name, attr_data);
        }
        
        edge_indices
    }
    
    /// Create nodes from Python list - PYTHON INTERFACE ONLY
    pub fn create_nodes_from_list(&mut self, py_nodes: &PyList) -> PyResult<Vec<String>> {
        let mut nodes_data = Vec::new();
        let mut created_node_ids = Vec::new();
        
        // Parse Python data first
        for item in py_nodes.iter() {
            if let Ok(tuple) = item.downcast::<pyo3::types::PyTuple>() {
                if tuple.len() >= 1 {
                    let node_id: String = tuple.get_item(0)?.extract()?;
                    
                    let attributes = if tuple.len() >= 2 {
                        if let Ok(attr_dict) = tuple.get_item(1)?.downcast::<PyDict>() {
                            python_dict_to_json_map(attr_dict)?
                        } else {
                            HashMap::new()
                        }
                    } else {
                        HashMap::new()
                    };
                    
                    created_node_ids.push(node_id.clone());
                    nodes_data.push((node_id, attributes));
                }
            }
        }
        
        // Use internal bulk method
        self.bulk_add_nodes_internal(nodes_data);
        
        Ok(created_node_ids)
    }
    
    /// Create edges from Python list - PYTHON INTERFACE ONLY
    pub fn create_edges_from_list(&mut self, py_edges: &PyList) -> PyResult<()> {
        let mut edges_data = Vec::new();
        
        // Parse Python data first
        for item in py_edges.iter() {
            if let Ok(tuple) = item.downcast::<pyo3::types::PyTuple>() {
                if tuple.len() >= 2 {
                    let source: String = tuple.get_item(0)?.extract()?;
                    let target: String = tuple.get_item(1)?.extract()?;
                    
                    let attributes = if tuple.len() >= 3 {
                        if let Ok(attr_dict) = tuple.get_item(2)?.downcast::<PyDict>() {
                            python_dict_to_json_map(attr_dict)?
                        } else {
                            HashMap::new()
                        }
                    } else {
                        HashMap::new()
                    };
                    
                    edges_data.push((source, target, attributes));
                }
            }
        }
        
        // Use internal bulk method
        self.bulk_add_edges_internal(edges_data);
        
        Ok(())
    }
    
    /// Bulk set node attributes using attribute UIDs for efficiency
    pub fn bulk_set_node_attributes_by_uid(
        &mut self,
        attr_name: &str,
        node_value_pairs: Vec<(String, serde_json::Value)>
    ) -> PyResult<()> {
        // Convert node IDs to indices
        let mut index_value_pairs = Vec::new();
        for (node_id, value) in node_value_pairs {
            if let Some(node_idx) = self.node_id_to_index.get(&node_id) {
                index_value_pairs.push((node_idx.index(), value));
            }
        }
        
        // Use columnar store's bulk method
        self.columnar_store.bulk_set_node_attributes(attr_name, index_value_pairs);
        
        Ok(())
    }
    
    /// Bulk set edge attributes using attribute UIDs for efficiency
    pub fn bulk_set_edge_attributes_by_uid(
        &mut self,
        attr_name: &str,
        edge_value_pairs: Vec<((String, String), serde_json::Value)>
    ) -> PyResult<()> {
        // Convert edge node pairs to edge indices
        let mut index_value_pairs = Vec::new();
        for ((source, target), value) in edge_value_pairs {
            if let (Some(source_idx), Some(target_idx)) = (
                self.node_id_to_index.get(&source),
                self.node_id_to_index.get(&target)
            ) {
                if let Some(edge_idx) = self.graph.find_edge(*source_idx, *target_idx) {
                    index_value_pairs.push((edge_idx.index(), value));
                }
            }
        }
        
        // Use columnar store's bulk method
        self.columnar_store.bulk_set_edge_attributes(attr_name, index_value_pairs);
        
        Ok(())
    }
}
