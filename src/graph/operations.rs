use super::core::FastGraph;
use super::types::{NodeData, EdgeData};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use std::collections::{HashSet, HashMap};
use petgraph::visit::EdgeRef;
use crate::utils::{python_dict_to_json_map, python_to_json_value, json_value_to_python};

impl FastGraph {
    /// Create subgraph with parallel node filtering
    pub fn parallel_subgraph_by_node_ids(&self, node_ids: &HashSet<String>) -> FastGraph {
        let mut subgraph = FastGraph::new(self.is_directed);
        
        // Add filtered nodes with attributes
        for node_id in node_ids {
            if let Some(node_idx) = self.node_id_to_index.get(node_id) {
                if let Some(node_data) = self.get_node_weight(*node_idx) {
                    // Add node directly to internal graph with all attributes
                    let new_node_idx = subgraph.add_node_to_graph_public(node_data.clone());
                    subgraph.node_id_to_index.insert(node_data.id.clone(), new_node_idx);
                    subgraph.node_index_to_id.insert(new_node_idx, node_data.id.clone());
                }
            }
        }
        
        // Add edges between filtered nodes in parallel
        let edges_to_add: Vec<_> = self.get_edge_indices()
            .par_iter()
            .filter_map(|edge_idx| {
                if let Some((source_idx, target_idx)) = self.get_edge_endpoints(*edge_idx) {
                    let source_id = self.node_index_to_id.get(&source_idx)?;
                    let target_id = self.node_index_to_id.get(&target_idx)?;
                    
                    if node_ids.contains(source_id.as_str()) && node_ids.contains(target_id.as_str()) {
                        let edge_data = self.get_edge_weight(*edge_idx)?;
                        Some((source_id.clone(), target_id.clone(), edge_data.clone()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        
        // Add edges to subgraph with attributes
        for (source, target, edge_data) in edges_to_add {
            let source_idx = *subgraph.node_id_to_index.get(&source).unwrap();
            let target_idx = *subgraph.node_id_to_index.get(&target).unwrap();
            // Add edge directly with all attributes
            let _ = subgraph.add_edge_to_graph_public(source_idx, target_idx, edge_data);
        }
        
        subgraph
    }
    
    /// Find connected component starting from a node
    pub fn connected_component(&self, start_node_id: &str) -> Option<FastGraph> {
        let start_idx = self.node_id_to_index.get(start_node_id)?;
        
        let mut visited = HashSet::new();
        let mut queue = vec![*start_idx];
        visited.insert(*start_idx);
        
        // BFS to find all connected nodes
        while let Some(current_idx) = queue.pop() {
            // Check all neighbors
            for neighbor_idx in self.get_neighbors_public(current_idx) {
                if !visited.contains(&neighbor_idx) {
                    visited.insert(neighbor_idx);
                    queue.push(neighbor_idx);
                }
            }
            
            // Also check incoming edges (for undirected behavior)
            for edge_ref in self.get_edges_directed(current_idx, petgraph::Direction::Incoming) {
                let source_idx = edge_ref.source();
                if !visited.contains(&source_idx) {
                    visited.insert(source_idx);
                    queue.push(source_idx);
                }
            }
        }
        
        // Convert node indices to IDs
        let node_ids: HashSet<String> = visited.iter()
            .filter_map(|idx| self.node_index_to_id.get(idx).map(|id| id.clone()))
            .collect();
        
        Some(self.parallel_subgraph_by_node_ids(&node_ids))
    }
    
    /// Get degree of a node
    pub fn node_degree(&self, node_id: &str) -> Option<usize> {
        let node_idx = self.node_id_to_index.get(node_id)?;
        
        let in_degree = self.get_edges_directed(*node_idx, petgraph::Direction::Incoming).len();
        let out_degree = self.get_edges_directed(*node_idx, petgraph::Direction::Outgoing).len();
        
        Some(in_degree + out_degree)
    }
    
    /// Get all nodes with degree greater than threshold
    pub fn high_degree_nodes(&self, min_degree: usize) -> Vec<(String, usize)> {
        self.node_id_to_index.iter()
            .filter_map(|entry| {
                let node_id = entry.key();
                let degree = self.node_degree(node_id)?;
                if degree >= min_degree {
                    Some((node_id.clone(), degree))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Internal method to remove a node
    pub fn remove_node_internal(&mut self, node_id: String) -> bool {
        if let Some(entry) = self.node_id_to_index.remove(&node_id) {
            let node_idx = entry.1;
            // Remove from reverse mapping
            self.node_index_to_id.remove(&node_idx);
            
            // Remove from graph (this also removes all connected edges)
            self.graph.remove_node(node_idx);
            
            true
        } else {
            false // Node didn't exist
        }
    }

    /// Internal method to remove multiple nodes efficiently
    pub fn remove_nodes_internal(&mut self, node_ids: Vec<String>) -> usize {
        let mut removed_count = 0;
        let mut indices_to_remove = Vec::new();

        // Collect node indices to remove
        for node_id in &node_ids {
            if let Some(entry) = self.node_id_to_index.get(node_id) {
                indices_to_remove.push((*entry.value(), node_id.clone()));
            }
        }

        // Remove nodes (this automatically removes connected edges)
        for (node_idx, node_id) in indices_to_remove {
            self.node_id_to_index.remove(&node_id);
            self.node_index_to_id.remove(&node_idx);
            self.graph.remove_node(node_idx);
            removed_count += 1;
        }

        removed_count
    }

    /// Internal method to remove an edge between two nodes
    pub fn remove_edge_internal(&mut self, source: String, target: String) -> bool {
        let source_idx = match self.node_id_to_index.get(&source) {
            Some(idx) => *idx,
            None => return false,
        };
        let target_idx = match self.node_id_to_index.get(&target) {
            Some(idx) => *idx,
            None => return false,
        };

        if let Some(edge_idx) = self.graph.find_edge(source_idx, target_idx) {
            self.graph.remove_edge(edge_idx);
            true
        } else {
            false // Edge didn't exist
        }
    }

    /// Internal method to remove multiple edges efficiently
    pub fn remove_edges_internal(&mut self, edge_pairs: Vec<(String, String)>) -> usize {
        let mut removed_count = 0;
        let mut edges_to_remove = Vec::new();

        // Collect edge indices to remove
        for (source, target) in &edge_pairs {
            if let (Some(source_idx), Some(target_idx)) = (
                self.node_id_to_index.get(source),
                self.node_id_to_index.get(target)
            ) {
                if let Some(edge_idx) = self.graph.find_edge(*source_idx, *target_idx) {
                    edges_to_remove.push(edge_idx);
                }
            }
        }

        // Remove edges
        for edge_idx in edges_to_remove {
            self.graph.remove_edge(edge_idx);
            removed_count += 1;
        }

        removed_count
    }

    /// Internal method to add an edge
    pub fn add_edge_internal(&mut self, source: String, target: String, attributes: Option<&PyDict>) -> PyResult<()> {
        // Ensure both nodes exist, create if they don't
        if !self.node_id_to_index.contains_key(&source) {
            self.add_node(source.clone(), None)?;
        }
        if !self.node_id_to_index.contains_key(&target) {
            self.add_node(target.clone(), None)?;
        }
        
        let source_idx = *self.node_id_to_index.get(&source).unwrap();
        let target_idx = *self.node_id_to_index.get(&target).unwrap();
        
        // Check if edge already exists
        if self.graph.find_edge(source_idx, target_idx).is_some() {
            return Ok(()); // Edge already exists, skip
        }
        
        let attrs = if let Some(py_attrs) = attributes {
            python_dict_to_json_map(py_attrs)?
        } else {
            HashMap::new()
        };
        
        let edge_data = EdgeData { 
            source, 
            target, 
            attributes: attrs 
        };
        
        self.graph.add_edge(source_idx, target_idx, edge_data);
        
        Ok(())
    }

    /// Internal method to add edges in bulk
    pub fn add_edges_internal(&mut self, edge_data: &PyList) -> PyResult<()> {
        let mut edges_to_add = Vec::new();
        let mut nodes_to_create = std::collections::HashSet::new();
        
        // First pass: collect edges and nodes to create
        for item in edge_data {
            let tuple = item.downcast::<pyo3::types::PyTuple>()?;
            let source: String = tuple.get_item(0)?.extract()?;
            let target: String = tuple.get_item(1)?.extract()?;
            
            if !self.node_id_to_index.contains_key(&source) {
                nodes_to_create.insert(source.clone());
            }
            if !self.node_id_to_index.contains_key(&target) {
                nodes_to_create.insert(target.clone());
            }
            
            let attributes = if tuple.len() > 2 {
                let py_attrs = tuple.get_item(2)?.downcast::<PyDict>()?;
                python_dict_to_json_map(py_attrs)?
            } else {
                HashMap::new()
            };
            
            edges_to_add.push((source, target, attributes));
        }
        
        // Create missing nodes
        for node_id in nodes_to_create {
            self.add_node(node_id, None)?;
        }
        
        // Add all edges
        for (source, target, attributes) in edges_to_add {
            let source_idx = *self.node_id_to_index.get(&source).unwrap();
            let target_idx = *self.node_id_to_index.get(&target).unwrap();
            
            let edge_data = EdgeData { 
                source, 
                target, 
                attributes 
            };
            
            self.graph.add_edge(source_idx, target_idx, edge_data);
        }
        
        Ok(())
    }
}
