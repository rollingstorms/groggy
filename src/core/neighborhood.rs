//! Neighborhood Subgraph Sampler
//!
//! This module provides efficient neighborhood subgraph sampling functionality.
//! Instead of creating tables (which would be slow), we generate Subgraph objects
//! that have full Graph API capabilities for further analysis and operations.
//!
//! DESIGN PHILOSOPHY:
//! - Generate Subgraph objects instead of tables for better performance
//! - Each neighborhood subgraph contains the central node(s) and their neighbors
//! - Supports 1-hop, k-hop, and multi-node neighborhoods
//! - Full composability: neighborhood().filter_nodes().bfs() etc.
//! - Follows the same pattern as connected_components result structure

use crate::core::pool::GraphPool;
use crate::core::space::GraphSpace;
use crate::core::subgraph::Subgraph;
use crate::errors::GraphResult;
use crate::types::NodeId;
use std::collections::{HashMap, HashSet};

/// Result of neighborhood sampling operations
#[derive(Debug, Clone)]
pub struct NeighborhoodResult {
    /// Vector of neighborhood subgraphs
    pub neighborhoods: Vec<NeighborhoodSubgraph>,
    /// Total number of neighborhoods generated
    pub total_neighborhoods: usize,
    /// Largest neighborhood size (in nodes)
    pub largest_neighborhood_size: usize,
    /// Execution time for the sampling operation
    pub execution_time: std::time::Duration,
}

/// A single neighborhood subgraph containing a central node and its neighbors
#[derive(Debug, Clone)]
pub struct NeighborhoodSubgraph {
    /// Node IDs in this neighborhood
    pub nodes: Vec<NodeId>,
    /// Edge IDs in this neighborhood (induced edges)
    pub edges: Vec<usize>, // EdgeId
    /// Central node(s) that define this neighborhood
    pub central_nodes: Vec<NodeId>,
    /// Distance/hop count for this neighborhood
    pub hops: usize,
    /// Number of nodes in this neighborhood
    pub size: usize,
    /// Number of edges in this neighborhood (induced edges)
    pub edge_count: usize,
}

/// Engine for generating neighborhood subgraphs
#[derive(Debug)]
pub struct NeighborhoodSampler {
    /// Performance statistics
    stats: NeighborhoodStats,
}

impl NeighborhoodSampler {
    pub fn new() -> Self {
        Self {
            stats: NeighborhoodStats::new(),
        }
    }

    /// Generate 1-hop neighborhood for a single node
    /// Returns a subgraph containing the central node and all its direct neighbors
    pub fn single_neighborhood(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        node_id: NodeId,
    ) -> GraphResult<NeighborhoodSubgraph> {
        let start = std::time::Instant::now();

        // Get neighbors using the same pattern as traversal engine
        let (_, _, _, neighbors_map) = space.snapshot(pool);

        // Get direct neighbors of the node
        let mut nodes = HashSet::new();
        nodes.insert(node_id);

        if let Some(node_neighbors) = neighbors_map.get(&node_id) {
            for &(neighbor_id, _edge_id) in node_neighbors {
                nodes.insert(neighbor_id);
            }
        }

        // Calculate induced edges using the same pattern as connected components
        let nodes_vec: Vec<NodeId> = nodes.into_iter().collect();
        let size = nodes_vec.len();
        let induced_edges = self.calculate_induced_edges(pool, space, &nodes_vec)?;
        let edge_count = induced_edges.len();

        let duration = start.elapsed();
        self.stats
            .record_neighborhood("single_neighborhood".to_string(), size, duration);

        Ok(NeighborhoodSubgraph {
            nodes: nodes_vec,
            edges: induced_edges,
            edge_count,
            size,
            central_nodes: vec![node_id],
            hops: 1,
        })
    }

    /// Helper to calculate induced edges count without creating full subgraph
    fn calculate_induced_edges_count(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        nodes: &HashSet<NodeId>,
    ) -> GraphResult<usize> {
        // Get topology vectors from space
        let (edge_ids, sources, targets, _) = space.snapshot(pool);

        let mut count = 0;
        for i in 0..edge_ids.len() {
            let source = sources[i];
            let target = targets[i];
            if nodes.contains(&source) && nodes.contains(&target) {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Calculate induced edges for a set of nodes
    /// Uses the same pattern as connected_components for proper edge calculation
    fn calculate_induced_edges(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        nodes: &[NodeId],
    ) -> GraphResult<Vec<usize>> {
        // Calculate induced edges using the same pattern as connected components
        let (edge_ids, sources, targets, _) = space.snapshot(pool);
        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
        
        let mut induced_edges = Vec::new();
        for i in 0..edge_ids.len() {
            let edge_id = edge_ids[i];
            let source = sources[i];
            let target = targets[i];
            
            if node_set.contains(&source) && node_set.contains(&target) {
                induced_edges.push(edge_id);
            }
        }
        
        Ok(induced_edges)
    }

    /// Generate 1-hop neighborhoods for multiple nodes
    /// Returns separate neighborhood subgraphs for each central node
    pub fn multi_neighborhood(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        node_ids: &[NodeId],
    ) -> GraphResult<NeighborhoodResult> {
        let start = std::time::Instant::now();
        let mut neighborhoods = Vec::new();
        let mut largest_size = 0;

        for &node_id in node_ids {
            let neighborhood = self.single_neighborhood(pool, space, node_id)?;
            largest_size = largest_size.max(neighborhood.size);
            neighborhoods.push(neighborhood);
        }

        let duration = start.elapsed();
        let total_nodes: usize = neighborhoods.iter().map(|n| n.size).sum();
        self.stats
            .record_neighborhood("multi_neighborhood".to_string(), total_nodes, duration);

        Ok(NeighborhoodResult {
            total_neighborhoods: neighborhoods.len(),
            largest_neighborhood_size: largest_size,
            neighborhoods,
            execution_time: duration,
        })
    }

    /// Generate k-hop neighborhood for a single node using BFS
    /// Returns a subgraph containing all nodes within k hops of the central node
    pub fn k_hop_neighborhood(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        node_id: NodeId,
        k: usize,
    ) -> GraphResult<NeighborhoodSubgraph> {
        let start = std::time::Instant::now();

        // For k=0, just return single node neighborhood
        if k == 0 {
            let duration = start.elapsed();
            self.stats
                .record_neighborhood("k_hop_neighborhood".to_string(), 1, duration);
            return Ok(NeighborhoodSubgraph {
                nodes: vec![node_id],
                edges: vec![], // No edges for single node
                edge_count: 0,
                size: 1,
                central_nodes: vec![node_id],
                hops: k,
            });
        }

        // For k=1, use the single_neighborhood method
        if k == 1 {
            return self.single_neighborhood(pool, space, node_id);
        }

        // For k>1, implement BFS using the same pattern as traversal engine
        let (_, _, _, neighbors_map) = space.snapshot(pool);
        let mut visited = HashSet::new();
        let mut current_level = HashSet::new();
        current_level.insert(node_id);
        visited.insert(node_id);
        
        // BFS for k hops
        for _hop in 0..k {
            let mut next_level = HashSet::new();
            for &current_node in &current_level {
                if let Some(node_neighbors) = neighbors_map.get(&current_node) {
                    for &(neighbor_id, _edge_id) in node_neighbors {
                        if !visited.contains(&neighbor_id) {
                            visited.insert(neighbor_id);
                            next_level.insert(neighbor_id);
                        }
                    }
                }
            }
            current_level = next_level;
            if current_level.is_empty() {
                break; // No more nodes to explore
            }
        }
        
        // Calculate induced edges for all visited nodes
        let nodes_vec: Vec<NodeId> = visited.into_iter().collect();
        let size = nodes_vec.len();
        let induced_edges = self.calculate_induced_edges(pool, space, &nodes_vec)?;
        let edge_count = induced_edges.len();

        let duration = start.elapsed();
        self.stats
            .record_neighborhood("k_hop_neighborhood".to_string(), size, duration);

        Ok(NeighborhoodSubgraph {
            nodes: nodes_vec,
            edges: induced_edges,
            edge_count,
            size,
            central_nodes: vec![node_id],
            hops: k,
        })
    }

    /// Generate unified neighborhood for multiple nodes
    /// Returns a single subgraph containing all nodes and their combined neighborhoods
    pub fn unified_neighborhood(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        node_ids: &[NodeId],
        k: usize,
    ) -> GraphResult<NeighborhoodSubgraph> {
        let start = std::time::Instant::now();

        // Use combined BFS from all starting nodes
        let (_, _, _, neighbors_map) = space.snapshot(pool);
        let mut visited = HashSet::new();
        let mut current_level = HashSet::new();
        
        // Start from all provided nodes
        for &node_id in node_ids {
            current_level.insert(node_id);
            visited.insert(node_id);
        }
        
        // BFS for k hops
        for _hop in 0..k {
            let mut next_level = HashSet::new();
            for &current_node in &current_level {
                if let Some(node_neighbors) = neighbors_map.get(&current_node) {
                    for &(neighbor_id, _edge_id) in node_neighbors {
                        if !visited.contains(&neighbor_id) {
                            visited.insert(neighbor_id);
                            next_level.insert(neighbor_id);
                        }
                    }
                }
            }
            current_level = next_level;
            if current_level.is_empty() {
                break; // No more nodes to explore
            }
        }
        
        // Calculate induced edges for all visited nodes
        let nodes_vec: Vec<NodeId> = visited.into_iter().collect();
        let size = nodes_vec.len();
        let induced_edges = self.calculate_induced_edges(pool, space, &nodes_vec)?;
        let edge_count = induced_edges.len();

        let duration = start.elapsed();
        self.stats.record_neighborhood(
            "unified_neighborhood".to_string(),
            size,
            duration,
        );

        Ok(NeighborhoodSubgraph {
            nodes: nodes_vec,
            edges: induced_edges,
            edge_count,
            size,
            central_nodes: node_ids.to_vec(),
            hops: k,
        })
    }

    /// Get performance statistics for neighborhood operations
    pub fn stats(&self) -> &NeighborhoodStats {
        &self.stats
    }

    /// Clear performance statistics
    pub fn clear_stats(&mut self) {
        self.stats.clear();
    }
}

impl Default for NeighborhoodSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance statistics for neighborhood sampling operations
#[derive(Debug, Clone)]
pub struct NeighborhoodStats {
    pub total_neighborhoods: usize,
    pub total_nodes_sampled: usize,
    pub total_time: std::time::Duration,
    pub operation_counts: HashMap<String, usize>,
    pub operation_times: HashMap<String, std::time::Duration>,
}

impl NeighborhoodStats {
    fn new() -> Self {
        Self {
            total_neighborhoods: 0,
            total_nodes_sampled: 0,
            total_time: std::time::Duration::new(0, 0),
            operation_counts: HashMap::new(),
            operation_times: HashMap::new(),
        }
    }

    fn record_neighborhood(
        &mut self,
        operation: String,
        nodes_sampled: usize,
        duration: std::time::Duration,
    ) {
        self.total_neighborhoods += 1;
        self.total_nodes_sampled += nodes_sampled;
        self.total_time += duration;

        *self.operation_counts.entry(operation.clone()).or_insert(0) += 1;
        *self
            .operation_times
            .entry(operation)
            .or_insert(std::time::Duration::new(0, 0)) += duration;
    }

    fn clear(&mut self) {
        *self = Self::new();
    }
}
