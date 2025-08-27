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

use crate::api::graph::Graph;
use crate::core::pool::GraphPool;
use crate::core::space::GraphSpace;
use crate::core::subgraph::Subgraph;
use crate::core::traits::{GraphEntity, SubgraphOperations};
use crate::core::traversal::TraversalEngine;
use crate::errors::GraphResult;
use crate::types::{EdgeId, EntityId, NodeId, SubgraphId};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

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
/// 
/// Uses our existing efficient HashSet<NodeId> + HashSet<EdgeId> storage
/// with specialized metadata for neighborhood analysis.
#[derive(Debug, Clone)]
pub struct NeighborhoodSubgraph {
    /// Reference to shared graph storage infrastructure
    graph_ref: Rc<RefCell<Graph>>,
    /// Efficient node storage (HashSet for O(1) contains operations)
    nodes: HashSet<NodeId>,
    /// Efficient edge storage (HashSet for O(1) contains operations)
    edges: HashSet<EdgeId>,
    /// Central node(s) that define this neighborhood
    central_nodes: Vec<NodeId>,
    /// Distance/hop count for this neighborhood
    hops: usize,
    /// Subgraph ID for GraphEntity trait
    subgraph_id: SubgraphId,
}

impl NeighborhoodSubgraph {
    /// Create a new NeighborhoodSubgraph from expansion results
    pub fn from_expansion(
        graph_ref: Rc<RefCell<Graph>>,
        central_nodes: Vec<NodeId>,
        hops: usize,
        result: crate::core::neighborhood::NeighborhoodResult,
    ) -> Self {
        // TODO: Extract nodes and edges from result properly
        // For now, create from first neighborhood in result
        if let Some(first_neighborhood) = result.neighborhoods.first() {
            let nodes: HashSet<NodeId> = first_neighborhood.nodes.iter().cloned().collect();
            let edges: HashSet<EdgeId> = first_neighborhood.edges.iter().cloned().collect();
            
            Self::new(graph_ref, central_nodes, hops, nodes, edges)
        } else {
            // Empty neighborhood
            Self::new(graph_ref, central_nodes, hops, HashSet::new(), HashSet::new())
        }
    }
    
    /// Create a new NeighborhoodSubgraph from stored data
    pub fn from_stored(
        graph_ref: Rc<RefCell<Graph>>,
        nodes: HashSet<NodeId>,
        edges: HashSet<EdgeId>,
        central_nodes: Vec<NodeId>,
        hops: usize,
    ) -> Self {
        Self::new(graph_ref, central_nodes, hops, nodes, edges)
    }
    
    /// Create a new NeighborhoodSubgraph
    pub fn new(
        graph_ref: Rc<RefCell<Graph>>,
        central_nodes: Vec<NodeId>,
        hops: usize,
        nodes: HashSet<NodeId>,
        edges: HashSet<EdgeId>,
    ) -> Self {
        // Generate subgraph ID from central nodes and hops
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        for &central in &central_nodes {
            central.hash(&mut hasher);
        }
        hops.hash(&mut hasher);
        nodes.len().hash(&mut hasher);
        let subgraph_id = hasher.finish() as SubgraphId;
        
        Self {
            graph_ref,
            nodes,
            edges,
            central_nodes,
            hops,
            subgraph_id,
        }
    }
    
    /// Get the central nodes for this neighborhood
    pub fn central_nodes(&self) -> &[NodeId] {
        &self.central_nodes
    }
    
    /// Get the number of hops for this neighborhood
    pub fn hops(&self) -> usize {
        self.hops
    }
    
    /// Expand this neighborhood by additional hops
    pub fn expand_by(&self, additional_hops: usize) -> GraphResult<NeighborhoodSubgraph> {
        let new_hops = self.hops + additional_hops;
        let mut graph = self.graph_ref.borrow_mut();
        
        // Use NeighborhoodSampler directly
        let mut neighborhood_sampler = NeighborhoodSampler::new();
        let result = neighborhood_sampler.unified_neighborhood(
            &graph.pool(),
            graph.space(),
            &self.central_nodes,
            new_hops
        )?;
        
        Ok(result)
    }
}

/// GraphEntity trait implementation for NeighborhoodSubgraph
impl GraphEntity for NeighborhoodSubgraph {
    fn entity_id(&self) -> EntityId {
        EntityId::Neighborhood(self.subgraph_id)
    }

    fn entity_type(&self) -> &'static str {
        "neighborhood"
    }

    fn graph_ref(&self) -> Rc<RefCell<Graph>> {
        self.graph_ref.clone()
    }

    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // Return central nodes as EntityNode wrappers
        let entities: Vec<Box<dyn GraphEntity>> = self.central_nodes
            .iter()
            .map(|&central_id| {
                Box::new(crate::core::node::EntityNode::new(central_id, self.graph_ref.clone()))
                    as Box<dyn GraphEntity>
            })
            .collect();
        Ok(entities)
    }

    fn summary(&self) -> String {
        format!(
            "NeighborhoodSubgraph(central={:?}, hops={}, nodes={}, edges={})",
            self.central_nodes,
            self.hops,
            self.nodes.len(),
            self.edges.len()
        )
    }
}

/// SubgraphOperations trait implementation for NeighborhoodSubgraph
impl SubgraphOperations for NeighborhoodSubgraph {
    fn node_set(&self) -> &HashSet<NodeId> {
        &self.nodes
    }

    fn edge_set(&self) -> &HashSet<EdgeId> {
        &self.edges
    }

    fn induced_subgraph(&self, nodes: &[NodeId]) -> GraphResult<Box<dyn SubgraphOperations>> {
        // Filter to nodes that exist in this neighborhood
        let filtered_nodes: HashSet<NodeId> = nodes.iter()
            .filter(|&&node_id| self.nodes.contains(&node_id))
            .cloned()
            .collect();

        // Calculate induced edges using existing method
        let induced_edges = crate::core::subgraph::Subgraph::calculate_induced_edges(&self.graph_ref, &filtered_nodes)?;

        let induced_neighborhood = NeighborhoodSubgraph::new(
            self.graph_ref.clone(),
            self.central_nodes.clone(), // Keep same central nodes
            self.hops,
            filtered_nodes,
            induced_edges
        );

        Ok(Box::new(induced_neighborhood))
    }

    fn subgraph_from_edges(&self, edges: &[EdgeId]) -> GraphResult<Box<dyn SubgraphOperations>> {
        // Filter to edges that exist in this neighborhood
        let filtered_edges: HashSet<EdgeId> = edges.iter()
            .filter(|&&edge_id| self.edges.contains(&edge_id))
            .cloned()
            .collect();

        // Calculate nodes from edge endpoints
        let mut endpoint_nodes = HashSet::new();
        let graph_borrow = self.graph_ref.borrow();
        for &edge_id in &filtered_edges {
            if let Ok((source, target)) = graph_borrow.edge_endpoints(edge_id) {
                if self.nodes.contains(&source) {
                    endpoint_nodes.insert(source);
                }
                if self.nodes.contains(&target) {
                    endpoint_nodes.insert(target);
                }
            }
        }

        let edge_neighborhood = NeighborhoodSubgraph::new(
            self.graph_ref.clone(),
            self.central_nodes.clone(),
            self.hops,
            endpoint_nodes,
            filtered_edges
        );

        Ok(Box::new(edge_neighborhood))
    }

    fn connected_components(&self) -> GraphResult<Vec<Box<dyn SubgraphOperations>>> {
        // Use existing efficient TraversalEngine for connected components within this neighborhood
        let mut graph = self.graph_ref.borrow_mut();
        let nodes_vec: Vec<NodeId> = self.nodes.iter().cloned().collect();
        let options = crate::core::traversal::TraversalOptions::default();
        
        // Use TraversalEngine directly
        let mut traversal_engine = TraversalEngine::new();
        let result = traversal_engine.connected_components_for_nodes(&graph.pool(), graph.space(), nodes_vec, options)?;

        let mut component_subgraphs = Vec::new();
        for (i, component) in result.components.into_iter().enumerate() {
            let component_nodes: std::collections::HashSet<NodeId> = component.nodes.into_iter().collect();
            let component_edges: std::collections::HashSet<EdgeId> = component.edges.into_iter().collect();
            
            let component_neighborhood = NeighborhoodSubgraph::new(
                self.graph_ref.clone(),
                self.central_nodes.clone(), // Keep same central nodes
                self.hops,
                component_nodes,
                component_edges
            );
            component_subgraphs.push(Box::new(component_neighborhood) as Box<dyn SubgraphOperations>);
        }

        Ok(component_subgraphs)
    }

    fn bfs_subgraph(&self, start: NodeId, max_depth: Option<usize>) -> GraphResult<Box<dyn SubgraphOperations>> {
        if !self.nodes.contains(&start) {
            return Err(crate::errors::GraphError::NodeNotFound { 
                node_id: start,
                operation: "bfs_subgraph".to_string(),
                suggestion: "Ensure start node is within this neighborhood".to_string(),
            });
        }

        // Use existing efficient TraversalEngine for BFS within this neighborhood
        let mut graph = self.graph_ref.borrow_mut();
        let mut options = crate::core::traversal::TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }
        
        // Use TraversalEngine directly
        let mut traversal_engine = TraversalEngine::new();
        let result = traversal_engine.bfs(&graph.pool(), &mut graph.space(), start, options)?;

        // Filter result to nodes that exist in this neighborhood
        let filtered_nodes: std::collections::HashSet<NodeId> = result.nodes
            .into_iter()
            .filter(|node| self.nodes.contains(node))
            .collect();
        let filtered_edges: std::collections::HashSet<EdgeId> = result.edges
            .into_iter()
            .filter(|edge| self.edges.contains(edge))
            .collect();

        let bfs_neighborhood = NeighborhoodSubgraph::new(
            self.graph_ref.clone(),
            self.central_nodes.clone(),
            self.hops,
            filtered_nodes,
            filtered_edges
        );

        Ok(Box::new(bfs_neighborhood))
    }

    fn dfs_subgraph(&self, start: NodeId, max_depth: Option<usize>) -> GraphResult<Box<dyn SubgraphOperations>> {
        if !self.nodes.contains(&start) {
            return Err(crate::errors::GraphError::NodeNotFound { 
                node_id: start,
                operation: "dfs_subgraph".to_string(),
                suggestion: "Ensure start node is within this neighborhood".to_string(),
            });
        }

        // Use existing efficient TraversalEngine for DFS within this neighborhood
        let mut graph = self.graph_ref.borrow_mut();
        let mut options = crate::core::traversal::TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }
        
        // Use TraversalEngine directly
        let mut traversal_engine = TraversalEngine::new();
        let result = traversal_engine.dfs(&graph.pool(), &mut graph.space(), start, options)?;

        // Filter result to nodes that exist in this neighborhood
        let filtered_nodes: std::collections::HashSet<NodeId> = result.nodes
            .into_iter()
            .filter(|node| self.nodes.contains(node))
            .collect();
        let filtered_edges: std::collections::HashSet<EdgeId> = result.edges
            .into_iter()
            .filter(|edge| self.edges.contains(edge))
            .collect();

        let dfs_neighborhood = NeighborhoodSubgraph::new(
            self.graph_ref.clone(),
            self.central_nodes.clone(),
            self.hops,
            filtered_nodes,
            filtered_edges
        );

        Ok(Box::new(dfs_neighborhood))
    }

    fn shortest_path_subgraph(&self, source: NodeId, target: NodeId) -> GraphResult<Option<Box<dyn SubgraphOperations>>> {
        if !self.nodes.contains(&source) || !self.nodes.contains(&target) {
            return Ok(None);
        }

        // Use existing efficient TraversalEngine for shortest path within this neighborhood
        let mut graph = self.graph_ref.borrow_mut();
        let options = crate::core::traversal::PathFindingOptions::default();
        
        // Use TraversalEngine directly
        let mut traversal_engine = TraversalEngine::new();
        let x = if let Some(path_result) = traversal_engine.shortest_path(
            &graph.pool(),
            &mut graph.space(),
            source,
            target,
            options
        )? {
            // Filter path to nodes/edges that exist in this neighborhood
            let filtered_nodes: std::collections::HashSet<NodeId> = path_result.nodes
                .into_iter()
                .filter(|node| self.nodes.contains(node))
                .collect();
            let filtered_edges: std::collections::HashSet<EdgeId> = path_result.edges
                .into_iter()
                .filter(|edge| self.edges.contains(edge))
                .collect();

            if !filtered_nodes.is_empty() {
                let path_neighborhood = NeighborhoodSubgraph::new(
                    self.graph_ref.clone(),
                    self.central_nodes.clone(),
                    self.hops,
                    filtered_nodes,
                    filtered_edges
                );
                Ok(Some(Box::new(path_neighborhood) as Box<dyn SubgraphOperations>))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        };
        x
    }
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

        // TODO: Fix graph_ref creation - temporary workaround
        // This is a simplified version to get compilation working
        let graph_ref = Rc::new(RefCell::new(Graph::new()));
        let subgraph_id = 0; // TODO: Generate proper SubgraphId
        Ok(NeighborhoodSubgraph {
            graph_ref,
            nodes: nodes_vec.into_iter().collect(),
            edges: induced_edges.into_iter().collect(),
            central_nodes: vec![node_id],
            hops: 1,
            subgraph_id,
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
            largest_size = largest_size.max(neighborhood.nodes.len());
            neighborhoods.push(neighborhood);
        }

        let duration = start.elapsed();
        let total_nodes: usize = neighborhoods.iter().map(|n| n.nodes.len()).sum();
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
            let graph_ref = Rc::new(RefCell::new(Graph::new()));
            let subgraph_id = 1;
            return Ok(NeighborhoodSubgraph {
                graph_ref,
                nodes: vec![node_id].into_iter().collect(),
                edges: HashSet::new(), // No edges for single node
                central_nodes: vec![node_id],
                hops: k,
                subgraph_id,
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

        let graph_ref = Rc::new(RefCell::new(Graph::new()));
        let subgraph_id = 2;
        Ok(NeighborhoodSubgraph {
            graph_ref,
            nodes: nodes_vec.into_iter().collect(),
            edges: induced_edges.into_iter().collect(),
            central_nodes: vec![node_id],
            hops: k,
            subgraph_id,
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

        let graph_ref = Rc::new(RefCell::new(Graph::new()));
        let subgraph_id = 3;
        Ok(NeighborhoodSubgraph {
            graph_ref,
            nodes: nodes_vec.into_iter().collect(),
            edges: induced_edges.into_iter().collect(),
            central_nodes: node_ids.to_vec(),
            hops: k,
            subgraph_id,
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
