use super::core::FastGraph;
use rayon::prelude::*;
use std::collections::HashSet;
use petgraph::visit::EdgeRef;

impl FastGraph {
    /// Create subgraph with parallel node filtering
    pub fn parallel_subgraph_by_node_ids(&self, node_ids: &HashSet<String>) -> FastGraph {
        let mut subgraph = FastGraph::new();
        
        // Add filtered nodes
        for node_id in node_ids {
            if let Some(node_idx) = self.node_id_to_index.get(node_id) {
                if let Some(node_data) = self.graph.node_weight(*node_idx) {
                    subgraph.add_node(node_data.id.clone(), None).unwrap();
                    // TODO: Set attributes
                }
            }
        }
        
        // Add edges between filtered nodes in parallel
        let edges_to_add: Vec<_> = self.graph.edge_indices()
            .par_bridge()
            .filter_map(|edge_idx| {
                if let Some((source_idx, target_idx)) = self.graph.edge_endpoints(edge_idx) {
                    let source_id = self.node_index_to_id.get(&source_idx)?;
                    let target_id = self.node_index_to_id.get(&target_idx)?;
                    
                    if node_ids.contains(source_id.as_str()) && node_ids.contains(target_id.as_str()) {
                        let edge_data = self.graph.edge_weight(edge_idx)?;
                        Some((source_id.clone(), target_id.clone(), edge_data.clone()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        
        // Add edges to subgraph
        for (source, target, _edge_data) in edges_to_add {
            subgraph.add_edge(source, target, None).unwrap();
            // TODO: Set edge attributes
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
            for neighbor_idx in self.graph.neighbors(current_idx) {
                if !visited.contains(&neighbor_idx) {
                    visited.insert(neighbor_idx);
                    queue.push(neighbor_idx);
                }
            }
            
            // Also check incoming edges (for undirected behavior)
            for edge_ref in self.graph.edges_directed(current_idx, petgraph::Direction::Incoming) {
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
        
        let in_degree = self.graph.edges_directed(*node_idx, petgraph::Direction::Incoming).count();
        let out_degree = self.graph.edges_directed(*node_idx, petgraph::Direction::Outgoing).count();
        
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
}
