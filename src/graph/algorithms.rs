// src_new/graph/algorithms.rs

// Algorithm implementations directly on FastGraph for MVP
use crate::graph::core::FastGraph;
use crate::graph::types::{NodeId, EdgeId};
use std::collections::{VecDeque, HashSet, HashMap};

impl FastGraph {
    /// Breadth-First Search (BFS) traversal from a given node.
    /// Returns the order of visited node IDs.
    pub fn bfs(&self, start: NodeId) -> Vec<NodeId> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut order = Vec::new();
        queue.push_back(start.clone());
        visited.insert(start.clone());
        while let Some(node) = queue.pop_front() {
            order.push(node.clone());
            for edge in &self.edge_collection.edge_ids {
                if edge.0 == node && !visited.contains(&edge.1) {
                    visited.insert(edge.1.clone());
                    queue.push_back(edge.1.clone());
                }
            }
        }
        order
    }

    /// Depth-First Search (DFS) traversal from a given node.
    /// Returns the order of visited node IDs.
    pub fn dfs(&self, start: NodeId) -> Vec<NodeId> {
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        let mut order = Vec::new();
        stack.push(start.clone());
        while let Some(node) = stack.pop() {
            if !visited.insert(node.clone()) {
                continue;
            }
            order.push(node.clone());
            for edge in &self.edge_collection.edge_ids {
                if edge.0 == node && !visited.contains(&edge.1) {
                    stack.push(edge.1.clone());
                }
            }
        }
        order
    }

    /// Finds the shortest path (unweighted) between two nodes using BFS.
    /// Returns the path as a vector of NodeId, or empty if unreachable.
    pub fn shortest_path(&self, start: NodeId, goal: NodeId) -> Vec<NodeId> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent: HashMap<NodeId, NodeId> = HashMap::new();
        queue.push_back(start.clone());
        visited.insert(start.clone());
        while let Some(node) = queue.pop_front() {
            if node == goal {
                // Reconstruct path
                let mut path = vec![goal.clone()];
                let mut curr = &goal;
                while let Some(p) = parent.get(curr) {
                    path.push(p.clone());
                    curr = p;
                }
                path.reverse();
                return path;
            }
            for edge in &self.edge_collection.edge_ids {
                if edge.0 == node && !visited.contains(&edge.1) {
                    visited.insert(edge.1.clone());
                    parent.insert(edge.1.clone(), node.clone());
                    queue.push_back(edge.1.clone());
                }
            }
        }
        Vec::new()
    }

    /// Finds all connected components (for undirected graphs).
    /// Returns a vector of components, each a vector of NodeId.
    pub fn connected_components(&self) -> Vec<Vec<NodeId>> {
        let mut visited = HashSet::new();
        let mut components = Vec::new();
        for node in &self.node_collection.node_ids {
            if !visited.contains(node) {
                let mut stack = vec![node.clone()];
                let mut component = Vec::new();
                while let Some(n) = stack.pop() {
                    if visited.insert(n.clone()) {
                        component.push(n.clone());
                        for edge in &self.edge_collection.edge_ids {
                            if edge.0 == n && !visited.contains(&edge.1) {
                                stack.push(edge.1.clone());
                            }
                            if edge.1 == n && !visited.contains(&edge.0) {
                                stack.push(edge.0.clone());
                            }
                        }
                    }
                }
                components.push(component);
            }
        }
        components
    }

    /// Computes the clustering coefficient for each node (simple version).
    /// Returns a map from NodeId to clustering coefficient (f64).
    pub fn clustering_coefficient(&self) -> HashMap<NodeId, f64> {
        let mut coeffs = HashMap::new();
        let nodes = &self.node_collection.node_ids;
        let edges = &self.edge_collection.edge_ids;
        let mut neighbors: HashMap<&NodeId, HashSet<&NodeId>> = HashMap::new();
        for node in nodes {
            neighbors.insert(node, HashSet::new());
        }
        for edge in edges {
            neighbors.get_mut(&edge.0).map(|n| { n.insert(&edge.1); });
            neighbors.get_mut(&edge.1).map(|n| { n.insert(&edge.0); });
        }
        for node in nodes {
            let neigh: Vec<&NodeId> = neighbors.get(node).unwrap().iter().cloned().collect();
            let k = neigh.len();
            if k < 2 {
                coeffs.insert(node.clone(), 0.0);
                continue;
            }
            let mut links = 0;
            for i in 0..k {
                for j in (i+1)..k {
                    if neighbors.get(neigh[i]).unwrap().contains(neigh[j]) {
                        links += 1;
                    }
                }
            }
            let denom = (k * (k - 1)) / 2;
            coeffs.insert(node.clone(), links as f64 / denom as f64);
        }
        coeffs
    }
}
