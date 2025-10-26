//! Shared utilities for community detection algorithms.
//!
//! This module provides common building blocks used across different community
//! detection algorithms, including connected component detection, Union-Find
//! data structures, and helper functions.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::types::NodeId;

/// Find connected components within a subset of nodes using BFS.
///
/// This is useful for ensuring communities remain connected during refinement
/// phases of community detection algorithms.
///
/// # Arguments
/// * `nodes` - The subset of nodes to analyze
/// * `adjacency` - The adjacency list (only edges between nodes in the subset are considered)
///
/// # Returns
/// A vector of components, where each component is a vector of NodeIds
pub fn find_connected_components(
    nodes: &[NodeId],
    adjacency: &HashMap<NodeId, Vec<NodeId>>,
) -> Vec<Vec<NodeId>> {
    let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut components = Vec::new();

    for &start in nodes {
        if visited.contains(&start) {
            continue;
        }

        // BFS to find component
        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(node) = queue.pop_front() {
            component.push(node);

            if let Some(neighbors) = adjacency.get(&node) {
                for &neighbor in neighbors {
                    if node_set.contains(&neighbor) && !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if !component.is_empty() {
            components.push(component);
        }
    }

    components
}

/// Union-Find (Disjoint Set Union) data structure for efficient connected component tracking.
///
/// Provides near-constant time operations for merging sets and finding representatives
/// using path compression and union by rank optimizations.
pub struct UnionFind {
    parent: HashMap<NodeId, NodeId>,
    rank: HashMap<NodeId, usize>,
}

impl UnionFind {
    /// Create a new Union-Find structure with the given nodes.
    ///
    /// Initially, each node is in its own set.
    pub fn new(nodes: &[NodeId]) -> Self {
        let mut parent = HashMap::with_capacity(nodes.len());
        let mut rank = HashMap::with_capacity(nodes.len());

        for &node in nodes {
            parent.insert(node, node);
            rank.insert(node, 0);
        }

        Self { parent, rank }
    }

    /// Find the representative (root) of the set containing the given node.
    ///
    /// Uses path compression to flatten the tree structure for faster future queries.
    pub fn find(&mut self, node: NodeId) -> Option<NodeId> {
        if !self.parent.contains_key(&node) {
            return None;
        }

        // Path compression
        let mut root = node;
        while self.parent[&root] != root {
            root = self.parent[&root];
        }

        // Compress path
        let mut current = node;
        while current != root {
            let next = self.parent[&current];
            self.parent.insert(current, root);
            current = next;
        }

        Some(root)
    }

    /// Union two sets by merging the sets containing the two given nodes.
    ///
    /// Uses union by rank to keep the tree balanced.
    ///
    /// # Returns
    /// `true` if the nodes were in different sets and have been merged, `false` if they were already in the same set.
    pub fn union(&mut self, node1: NodeId, node2: NodeId) -> bool {
        let root1 = match self.find(node1) {
            Some(r) => r,
            None => return false,
        };

        let root2 = match self.find(node2) {
            Some(r) => r,
            None => return false,
        };

        if root1 == root2 {
            return false; // Already in the same set
        }

        // Union by rank
        let rank1 = self.rank[&root1];
        let rank2 = self.rank[&root2];

        if rank1 < rank2 {
            self.parent.insert(root1, root2);
        } else if rank1 > rank2 {
            self.parent.insert(root2, root1);
        } else {
            self.parent.insert(root2, root1);
            self.rank.insert(root1, rank1 + 1);
        }

        true
    }

    /// Get all connected components as a mapping from representative to component members.
    pub fn get_components(&mut self) -> HashMap<NodeId, Vec<NodeId>> {
        let mut components: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        let nodes: Vec<NodeId> = self.parent.keys().copied().collect();
        for node in nodes {
            if let Some(root) = self.find(node) {
                components.entry(root).or_default().push(node);
            }
        }

        components
    }

    /// Get the number of distinct components.
    pub fn count_components(&mut self) -> usize {
        self.get_components().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_connected_components_single() {
        let nodes = vec![1, 2, 3];
        let mut adjacency = HashMap::new();
        adjacency.insert(1, vec![2]);
        adjacency.insert(2, vec![1, 3]);
        adjacency.insert(3, vec![2]);

        let components = find_connected_components(&nodes, &adjacency);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 3);
    }

    #[test]
    fn test_find_connected_components_multiple() {
        let nodes = vec![1, 2, 3, 4];
        let mut adjacency = HashMap::new();
        adjacency.insert(1, vec![2]);
        adjacency.insert(2, vec![1]);
        adjacency.insert(3, vec![4]);
        adjacency.insert(4, vec![3]);

        let components = find_connected_components(&nodes, &adjacency);
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_union_find_basic() {
        let nodes = vec![1, 2, 3, 4];
        let mut uf = UnionFind::new(&nodes);

        // Initially 4 components
        assert_eq!(uf.count_components(), 4);

        // Union 1 and 2
        assert!(uf.union(1, 2));
        assert_eq!(uf.count_components(), 3);

        // Union 3 and 4
        assert!(uf.union(3, 4));
        assert_eq!(uf.count_components(), 2);

        // Union 2 and 3 (connects all)
        assert!(uf.union(2, 3));
        assert_eq!(uf.count_components(), 1);

        // Verify they're all in the same set
        let root1 = uf.find(1).unwrap();
        let root2 = uf.find(2).unwrap();
        let root3 = uf.find(3).unwrap();
        let root4 = uf.find(4).unwrap();
        assert_eq!(root1, root2);
        assert_eq!(root2, root3);
        assert_eq!(root3, root4);
    }

    #[test]
    fn test_union_find_no_merge_same_set() {
        let nodes = vec![1, 2];
        let mut uf = UnionFind::new(&nodes);

        assert!(uf.union(1, 2));
        assert!(!uf.union(1, 2)); // Already in same set
        assert_eq!(uf.count_components(), 1);
    }

    #[test]
    fn test_union_find_path_compression() {
        let nodes = vec![1, 2, 3, 4, 5];
        let mut uf = UnionFind::new(&nodes);

        // Create a chain: 1->2->3->4->5
        uf.union(1, 2);
        uf.union(2, 3);
        uf.union(3, 4);
        uf.union(4, 5);

        // All should have the same root after path compression
        let root = uf.find(1).unwrap();
        assert_eq!(uf.find(2).unwrap(), root);
        assert_eq!(uf.find(3).unwrap(), root);
        assert_eq!(uf.find(4).unwrap(), root);
        assert_eq!(uf.find(5).unwrap(), root);
    }
}
