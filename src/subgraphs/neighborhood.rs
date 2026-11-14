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
use crate::errors::GraphResult;
use crate::ownership::OwnerCell;
use crate::state::space::GraphSpace;
use crate::storage::pool::GraphPool;
use crate::traits::{GraphEntity, SubgraphOperations};
use crate::types::{EdgeId, EntityId, NodeId, SubgraphId};
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Weak;

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
    /// Canonical owner, selected membership, scope validation and topology caches.
    selection: crate::subgraphs::Subgraph,

    /// Central node(s) that define this neighborhood
    central_nodes: Vec<NodeId>,
    /// Distance/hop count for this neighborhood
    hops: usize,
    /// Subgraph ID for GraphEntity trait
    subgraph_id: SubgraphId,
}

impl NeighborhoodSubgraph {
    /// Retain selected values/topology while preserving specialized metadata.
    pub fn freeze(mut self) -> GraphResult<Self> {
        self.selection = self.selection.freeze()?;
        Ok(self)
    }

    /// Borrow the canonical connected selection without copying its membership.
    pub fn as_subgraph(&self) -> &crate::subgraphs::Subgraph {
        &self.selection
    }

    /// Freeze current selected values/topology; specialized metadata stays on this wrapper.
    pub fn pin_view(&self) -> GraphResult<crate::subgraphs::GraphView> {
        self.selection.pin_view()
    }

    fn selection_id(central_nodes: &[NodeId], hops: usize, node_count: usize) -> SubgraphId {
        use std::hash::Hash;
        use std::hash::Hasher;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for central in central_nodes {
            central.hash(&mut hasher);
        }
        hops.hash(&mut hasher);
        node_count.hash(&mut hasher);
        hasher.finish() as SubgraphId
    }

    fn wrap_selection(&self, result: crate::subgraphs::Subgraph) -> Self {
        Self {
            subgraph_id: Self::selection_id(&self.central_nodes, self.hops, result.node_count()),
            selection: result,
            central_nodes: self.central_nodes.clone(),
            hops: self.hops,
        }
    }

    /// Create a new NeighborhoodSubgraph from expansion results
    pub fn from_expansion(
        graph_ref: Arc<OwnerCell<Graph>>,
        central_nodes: Vec<NodeId>,
        hops: usize,
        result: crate::subgraphs::neighborhood::NeighborhoodResult,
    ) -> Self {
        // TODO: Extract nodes and edges from result properly
        // For now, create from first neighborhood in result
        if let Some(first_neighborhood) = result.neighborhoods.first() {
            let nodes: HashSet<NodeId> = first_neighborhood
                .selection
                .nodes()
                .iter()
                .cloned()
                .collect();
            let edges: HashSet<EdgeId> = first_neighborhood
                .selection
                .edges()
                .iter()
                .cloned()
                .collect();

            Self::new(graph_ref, central_nodes, hops, nodes, edges)
        } else {
            // Empty neighborhood
            Self::new(
                graph_ref,
                central_nodes,
                hops,
                HashSet::new(),
                HashSet::new(),
            )
        }
    }

    /// Create a new NeighborhoodSubgraph from stored data
    pub fn from_stored(
        graph_ref: Arc<OwnerCell<Graph>>,
        nodes: HashSet<NodeId>,
        edges: HashSet<EdgeId>,
        central_nodes: Vec<NodeId>,
        hops: usize,
    ) -> Self {
        Self::new(graph_ref, central_nodes, hops, nodes, edges)
    }

    /// Create a new NeighborhoodSubgraph
    pub fn new(
        graph_ref: Arc<OwnerCell<Graph>>,
        central_nodes: Vec<NodeId>,
        hops: usize,
        nodes: HashSet<NodeId>,
        edges: HashSet<EdgeId>,
    ) -> Self {
        let subgraph_id = Self::selection_id(&central_nodes, hops, nodes.len());

        Self {
            selection: crate::subgraphs::Subgraph::new(
                graph_ref,
                nodes,
                edges,
                format!("neighborhood_hops_{hops}"),
            ),
            central_nodes,
            hops,
            subgraph_id,
        }
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

    fn graph_ref(&self) -> Arc<OwnerCell<Graph>> {
        self.selection.graph()
    }

    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // Return central nodes as EntityNode wrappers
        let entities: Vec<Box<dyn GraphEntity>> = self
            .central_nodes
            .iter()
            .map(|&central_id| {
                Box::new(crate::entities::EntityNode::new(
                    central_id,
                    self.selection.graph(),
                )) as Box<dyn GraphEntity>
            })
            .collect();
        Ok(entities)
    }

    fn summary(&self) -> String {
        format!(
            "NeighborhoodSubgraph(central={:?}, hops={}, nodes={}, edges={})",
            self.central_nodes,
            self.hops,
            self.selection.nodes().len(),
            self.selection.edges().len()
        )
    }
}

/// SubgraphOperations trait implementation for NeighborhoodSubgraph
impl SubgraphOperations for NeighborhoodSubgraph {
    fn frozen_view(&self) -> Option<&crate::GraphView> {
        self.selection.frozen_view()
    }
    fn node_set(&self) -> &HashSet<NodeId> {
        self.selection.nodes()
    }
    fn edge_set(&self) -> &HashSet<EdgeId> {
        self.selection.edges()
    }
    fn neighbors(&self, node: NodeId) -> GraphResult<Vec<NodeId>> {
        self.selection.neighbors(node)
    }
    fn degree(&self, node: NodeId) -> GraphResult<usize> {
        self.selection.degree(node)
    }
    fn connected_components(&self) -> GraphResult<Vec<Box<dyn SubgraphOperations>>> {
        Ok(self
            .selection
            .component_selections()?
            .into_iter()
            .map(|result| Box::new(self.wrap_selection(result)) as Box<dyn SubgraphOperations>)
            .collect())
    }
    fn bfs(
        &self,
        start: NodeId,
        max_depth: Option<usize>,
    ) -> GraphResult<Box<dyn SubgraphOperations>> {
        Ok(Box::new(self.wrap_selection(
            self.selection.bfs_selection(start, max_depth)?,
        )))
    }
    fn dfs(
        &self,
        start: NodeId,
        max_depth: Option<usize>,
    ) -> GraphResult<Box<dyn SubgraphOperations>> {
        Ok(Box::new(self.wrap_selection(
            self.selection.dfs_selection(start, max_depth)?,
        )))
    }
    fn shortest_path_subgraph(
        &self,
        source: NodeId,
        target: NodeId,
    ) -> GraphResult<Option<Box<dyn SubgraphOperations>>> {
        Ok(self
            .selection
            .path_selection(source, target)?
            .map(|result| Box::new(self.wrap_selection(result)) as Box<dyn SubgraphOperations>))
    }
    fn induced_subgraph(&self, nodes: &[NodeId]) -> GraphResult<Box<dyn SubgraphOperations>> {
        Ok(Box::new(
            self.wrap_selection(self.selection.induced_selection(nodes)?),
        ))
    }
    fn subgraph_from_edges(&self, edges: &[EdgeId]) -> GraphResult<Box<dyn SubgraphOperations>> {
        Ok(Box::new(
            self.wrap_selection(self.selection.edge_selection(edges)?),
        ))
    }

    fn viz(&self) -> crate::viz::VizModule {
        use crate::subgraphs::visualization::SubgraphDataSource;
        use std::sync::Arc;

        // Create wrapper that extracts current neighborhood data
        let data_source = SubgraphDataSource::from_subgraph_operations(self);
        let data_source: Arc<dyn crate::viz::streaming::data_source::DataSource> =
            Arc::new(data_source);

        crate::viz::VizModule::new(data_source)
    }
}

/// NeighborhoodOperations trait implementation for NeighborhoodSubgraph
impl crate::traits::NeighborhoodOperations for NeighborhoodSubgraph {
    fn central_nodes(&self) -> &[NodeId] {
        &self.central_nodes
    }

    fn hops(&self) -> usize {
        self.hops
    }

    fn expand_by(
        &self,
        additional_hops: usize,
    ) -> GraphResult<Box<dyn crate::traits::NeighborhoodOperations>> {
        let new_hops = self.hops.checked_add(additional_hops).ok_or_else(|| {
            crate::GraphError::InvalidInput("Neighborhood hop count overflow".into())
        })?;
        if let Some(view) = self.frozen_view() {
            let root = crate::GraphView::all(view.read_state().clone());
            let child = root.within_hops(
                &self.central_nodes,
                new_hops,
                crate::query::TraversalDirection::Outgoing,
            )?;
            let mut result =
                self.wrap_selection(self.selection.bound_selection(child, "neighborhood".into()));
            result.hops = new_hops;
            return Ok(Box::new(result));
        }
        // Use existing neighborhood sampling to expand
        let mut sampler = NeighborhoodSampler::new();
        sampler.set_graph_ref(&self.selection.graph());

        // For simplicity, expand from the first central node
        // In a full implementation, we'd merge expansions from all central nodes
        if let Some(&first_central) = self.central_nodes.first() {
            let owner = self.selection.graph();
            let graph = owner.borrow();
            let expanded_neighborhood = sampler.k_hop_neighborhood(
                &graph.pool(),
                graph.space(),
                first_central,
                new_hops,
            )?;

            Ok(Box::new(expanded_neighborhood))
        } else {
            // No central nodes, return empty neighborhood
            let empty_neighborhood = NeighborhoodSubgraph::new(
                self.selection.graph(),
                Vec::new(),
                new_hops,
                std::collections::HashSet::new(),
                std::collections::HashSet::new(),
            );
            Ok(Box::new(empty_neighborhood))
        }
    }

    fn merge_with(
        &self,
        other: &dyn crate::traits::NeighborhoodOperations,
    ) -> GraphResult<Box<dyn crate::traits::NeighborhoodOperations>> {
        if self.frozen_view().is_some() || other.frozen_view().is_some() {
            let left = self.frozen_view().ok_or_else(|| {
                crate::GraphError::InvalidInput(
                    "Freeze both neighborhood sources before merging".into(),
                )
            })?;
            let right = other.frozen_view().ok_or_else(|| {
                crate::GraphError::InvalidInput(
                    "Freeze both neighborhood sources before merging".into(),
                )
            })?;
            let mut result = self.wrap_selection(
                self.selection
                    .bound_selection(left.merge_with(right)?, "neighborhood".into()),
            );
            for &center in other.central_nodes() {
                if !result.central_nodes.contains(&center) {
                    result.central_nodes.push(center);
                }
            }
            result.hops = self.hops.max(other.hops());
            return Ok(Box::new(result));
        }
        // Union of node and edge sets
        let mut merged_nodes = self.selection.nodes().clone();
        let other_nodes = other.node_set();
        merged_nodes.extend(other_nodes);

        let mut merged_edges = self.selection.edges().clone();
        let other_edges = other.edge_set();
        merged_edges.extend(other_edges);

        // Merge central nodes
        let mut merged_centrals = self.central_nodes.clone();
        for &central in other.central_nodes() {
            if !merged_centrals.contains(&central) {
                merged_centrals.push(central);
            }
        }

        // Use maximum hop distance
        let merged_hops = std::cmp::max(self.hops, other.hops());

        let merged_neighborhood = NeighborhoodSubgraph::new(
            self.selection.graph(),
            merged_centrals,
            merged_hops,
            merged_nodes,
            merged_edges,
        );

        Ok(Box::new(merged_neighborhood))
    }
}

/// Engine for generating neighborhood subgraphs
#[derive(Debug)]
pub struct NeighborhoodSampler {
    /// Performance statistics
    stats: NeighborhoodStats,
    /// Weak self-reference for constructing subgraphs tied to the live graph
    graph_ref: Option<Weak<OwnerCell<Graph>>>,
}

impl NeighborhoodSampler {
    pub fn new() -> Self {
        Self {
            stats: NeighborhoodStats::new(),
            graph_ref: None,
        }
    }

    /// Attach the owning graph so generated subgraphs can mutate the real graph
    pub fn set_graph_ref(&mut self, graph_ref: &Arc<OwnerCell<Graph>>) {
        self.graph_ref = Some(Arc::downgrade(graph_ref));
    }

    fn graph_ref(&self) -> Option<Arc<OwnerCell<Graph>>> {
        self.graph_ref.as_ref().and_then(|weak| weak.upgrade())
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
        let _edge_count = induced_edges.len();

        let duration = start.elapsed();
        self.stats
            .record_neighborhood("single_neighborhood".to_string(), size, duration);

        // Use the live graph when available so downstream operations mutate shared state
        let graph_ref = self
            .graph_ref()
            .unwrap_or_else(|| Arc::new(OwnerCell::new(Graph::new_manual())));

        // Generate proper SubgraphId using hash of content
        let subgraph_id = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::Hash;
            use std::hash::Hasher;
            let mut hasher = DefaultHasher::new();
            let mut sorted_nodes: Vec<NodeId> = nodes_vec.to_vec();
            sorted_nodes.sort();
            sorted_nodes.hash(&mut hasher);
            let mut sorted_edges: Vec<EdgeId> = induced_edges.to_vec();
            sorted_edges.sort();
            sorted_edges.hash(&mut hasher);
            "neighborhood".hash(&mut hasher);
            node_id.hash(&mut hasher); // Include central node for uniqueness
            (hasher.finish() as usize) as SubgraphId
        };
        Ok(NeighborhoodSubgraph {
            selection: crate::subgraphs::Subgraph::new(
                graph_ref,
                nodes_vec.into_iter().collect(),
                induced_edges.into_iter().collect(),
                "neighborhood_hops_1".into(),
            ),
            central_nodes: vec![node_id],
            hops: 1,
            subgraph_id,
        })
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
            largest_size = largest_size.max(neighborhood.selection.node_count());
            neighborhoods.push(neighborhood);
        }

        let duration = start.elapsed();
        let total_nodes: usize = neighborhoods.iter().map(|n| n.selection.node_count()).sum();
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
            let graph_ref = self
                .graph_ref()
                .unwrap_or_else(|| Arc::new(OwnerCell::new(Graph::new_manual())));
            let subgraph_id = 1;
            return Ok(NeighborhoodSubgraph {
                selection: crate::subgraphs::Subgraph::new(
                    graph_ref,
                    vec![node_id].into_iter().collect(),
                    HashSet::new(),
                    format!("neighborhood_hops_{k}"),
                ),
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
        let _edge_count = induced_edges.len();

        let duration = start.elapsed();
        self.stats
            .record_neighborhood("k_hop_neighborhood".to_string(), size, duration);

        let graph_ref = self
            .graph_ref()
            .unwrap_or_else(|| Arc::new(OwnerCell::new(Graph::new_manual())));
        let subgraph_id = 2;
        Ok(NeighborhoodSubgraph {
            selection: crate::subgraphs::Subgraph::new(
                graph_ref,
                nodes_vec.into_iter().collect(),
                induced_edges.into_iter().collect(),
                format!("neighborhood_hops_{k}"),
            ),
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
        let _edge_count = induced_edges.len();

        let duration = start.elapsed();
        self.stats
            .record_neighborhood("unified_neighborhood".to_string(), size, duration);

        let graph_ref = self
            .graph_ref()
            .unwrap_or_else(|| Arc::new(OwnerCell::new(Graph::new_manual())));
        let subgraph_id = 3;
        Ok(NeighborhoodSubgraph {
            selection: crate::subgraphs::Subgraph::new(
                graph_ref,
                nodes_vec.into_iter().collect(),
                induced_edges.into_iter().collect(),
                format!("neighborhood_hops_{k}"),
            ),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;
    use crate::ownership::OwnerCell;
    use crate::traits::NeighborhoodOperations;
    use std::sync::Arc;

    #[test]
    fn test_neighborhood_operations() {
        // Test NeighborhoodOperations implementation
        let mut graph = Graph::new_manual();

        // Create a small test graph
        let center = graph.add_node().unwrap();
        let node1 = graph.add_node().unwrap();
        let node2 = graph.add_node().unwrap();
        let node3 = graph.add_node().unwrap();

        let _edge1 = graph.add_edge(center, node1).unwrap();
        let _edge2 = graph.add_edge(center, node2).unwrap();
        let _edge3 = graph.add_edge(node1, node3).unwrap();

        let graph_rc = Arc::new(OwnerCell::new(graph));

        // Create a neighborhood manually for testing
        let nodes = std::collections::HashSet::from([center, node1, node2]);
        let edges = std::collections::HashSet::from([_edge1, _edge2]);

        let neighborhood = NeighborhoodSubgraph::new(
            graph_rc.clone(),
            vec![center], // central node
            1,            // 1-hop neighborhood
            nodes,
            edges,
        );

        // Test NeighborhoodOperations interface
        assert_eq!(neighborhood.central_nodes(), &[center]);
        assert_eq!(neighborhood.hops(), 1);
        assert!(neighborhood.is_central_node(center));
        assert!(!neighborhood.is_central_node(node1));

        // Test expansion statistics
        let stats = neighborhood.expansion_stats();
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.total_edges, 2);
        assert_eq!(stats.central_count, 1);
        assert_eq!(stats.max_hops, 1);

        // Test density calculation
        let density = neighborhood.calculate_density();
        assert!(density > 0.0 && density <= 1.0);

        // Test nodes at hop 0 (central nodes)
        let central_nodes_at_hop = neighborhood.nodes_at_hop(0).unwrap();
        assert_eq!(central_nodes_at_hop.len(), 1);
        assert_eq!(central_nodes_at_hop[0], center);

        // NeighborhoodOperations tests passed
    }
}
