use super::core::FastGraph;
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, VecDeque};

impl FastGraph {
    /// Breadth-First Search from a starting node
    pub fn bfs(&self, start_node_id: &str) -> Option<Vec<String>> {
        let start_idx = self.node_id_to_index.get(start_node_id)?;

        let mut visited = std::collections::HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        queue.push_back(*start_idx);
        visited.insert(*start_idx);

        while let Some(current_idx) = queue.pop_front() {
            if let Some(node_id) = self.node_index_to_id.get(&current_idx) {
                result.push(node_id.clone());
            }

            for neighbor_idx in self.get_neighbors_public(current_idx) {
                if !visited.contains(&neighbor_idx) {
                    visited.insert(neighbor_idx);
                    queue.push_back(neighbor_idx);
                }
            }
        }

        Some(result)
    }

    /// Depth-First Search from a starting node
    pub fn dfs(&self, start_node_id: &str) -> Option<Vec<String>> {
        let start_idx = self.node_id_to_index.get(start_node_id)?;

        let mut visited = std::collections::HashSet::new();
        let mut stack = Vec::new();
        let mut result = Vec::new();

        stack.push(*start_idx);

        while let Some(current_idx) = stack.pop() {
            if !visited.contains(&current_idx) {
                visited.insert(current_idx);

                if let Some(node_id) = self.node_index_to_id.get(&current_idx) {
                    result.push(node_id.clone());
                }

                // Add neighbors to stack (in reverse order for consistent traversal)
                let mut neighbors: Vec<_> = self.get_neighbors_public(current_idx);
                neighbors.reverse();

                for neighbor_idx in neighbors {
                    if !visited.contains(&neighbor_idx) {
                        stack.push(neighbor_idx);
                    }
                }
            }
        }

        Some(result)
    }

    /// Find shortest path between two nodes (BFS-based)
    pub fn shortest_path(&self, start_node_id: &str, end_node_id: &str) -> Option<Vec<String>> {
        let start_idx = self.node_id_to_index.get(start_node_id)?;
        let end_idx = self.node_id_to_index.get(end_node_id)?;

        if *start_idx == *end_idx {
            return Some(vec![start_node_id.to_string()]);
        }

        let mut visited = std::collections::HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent: HashMap<NodeIndex, NodeIndex> = HashMap::new();

        queue.push_back(*start_idx);
        visited.insert(*start_idx);

        while let Some(current_idx) = queue.pop_front() {
            if current_idx == *end_idx {
                // Reconstruct path
                let mut path = Vec::new();
                let mut current = current_idx;

                loop {
                    if let Some(node_id) = self.node_index_to_id.get(&current) {
                        path.push(node_id.clone());
                    }

                    if let Some(&parent_idx) = parent.get(&current) {
                        current = parent_idx;
                    } else {
                        break;
                    }
                }

                path.reverse();
                return Some(path);
            }

            for neighbor_idx in self.get_neighbors_public(current_idx) {
                if !visited.contains(&neighbor_idx) {
                    visited.insert(neighbor_idx);
                    parent.insert(neighbor_idx, current_idx);
                    queue.push_back(neighbor_idx);
                }
            }
        }

        None // No path found
    }

    /// Calculate clustering coefficient for a node
    pub fn clustering_coefficient(&self, node_id: &str) -> Option<f64> {
        let node_idx = self.node_id_to_index.get(node_id)?;

        // Get all neighbors
        let neighbors: Vec<_> = self.get_neighbors_public(*node_idx);
        let degree = neighbors.len();

        if degree < 2 {
            return Some(0.0);
        }

        // Count edges between neighbors
        let mut edge_count = 0;
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                // Check if there's an edge between neighbors[i] and neighbors[j]
                if let (Some(idx_i), Some(idx_j)) = (
                    self.node_id_to_index
                        .get(self.node_index_to_id.get(&neighbors[i]).unwrap()),
                    self.node_id_to_index
                        .get(self.node_index_to_id.get(&neighbors[j]).unwrap()),
                ) {
                    if self.graph.find_edge(*idx_i, *idx_j).is_some()
                        || self.graph.find_edge(*idx_j, *idx_i).is_some()
                    {
                        edge_count += 1;
                    }
                }
            }
        }

        let max_edges = degree * (degree - 1) / 2;
        Some(edge_count as f64 / max_edges as f64)
    }

    /// Calculate average clustering coefficient for the entire graph
    pub fn average_clustering_coefficient(&self) -> f64 {
        let coefficients: Vec<f64> = self
            .node_id_to_index
            .iter()
            .filter_map(|(node_id, _node_idx)| self.clustering_coefficient(node_id))
            .collect();

        if coefficients.is_empty() {
            0.0
        } else {
            coefficients.iter().sum::<f64>() / coefficients.len() as f64
        }
    }
}
