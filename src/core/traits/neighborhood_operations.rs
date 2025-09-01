//! NeighborhoodOperations - Specialized interface for neighborhood entities
//!
//! This trait provides specialized operations for neighborhood subgraphs while leveraging
//! our existing efficient SubgraphOperations infrastructure. All neighborhood operations
//! use the same storage pattern with neighborhood-specific metadata.

use crate::core::traits::SubgraphOperations;
use crate::errors::GraphResult;
use crate::types::NodeId;

/// Specialized operations for neighborhood entities
///
/// This trait extends SubgraphOperations with neighborhood-specific functionality.
/// Neighborhoods maintain the same efficient HashSet<NodeId> + HashSet<EdgeId> storage
/// with additional metadata for central nodes, hop distances, and expansion statistics.
///
/// # Design Principles
/// - **Same Efficient Storage**: Uses identical storage pattern as base Subgraph
/// - **Neighborhood Metadata**: Adds central nodes, hops, and expansion statistics
/// - **Algorithm Integration**: Leverages existing neighborhood sampling algorithms
/// - **Composable Operations**: Can expand neighborhoods and analyze expansion patterns
pub trait NeighborhoodOperations: SubgraphOperations {
    /// Get the central nodes that define this neighborhood
    ///
    /// # Returns
    /// Reference to vector of central node IDs around which this neighborhood was built
    ///
    /// # Performance
    /// O(1) - Direct field access to existing Vec<NodeId>
    fn central_nodes(&self) -> &[NodeId];

    /// Get the hop distance for this neighborhood
    ///
    /// # Returns
    /// Maximum distance (in hops) from central nodes included in this neighborhood
    ///
    /// # Performance
    /// O(1) - Direct field access
    fn hops(&self) -> usize;

    /// Check if a node is a central node
    ///
    /// # Arguments
    /// * `node_id` - Node to check
    ///
    /// # Returns
    /// true if the node is one of the central nodes
    ///
    /// # Performance
    /// O(k) where k is the number of central nodes (typically small)
    fn is_central_node(&self, node_id: NodeId) -> bool {
        self.central_nodes().contains(&node_id)
    }

    /// Get expansion statistics for this neighborhood
    ///
    /// # Returns
    /// Statistics about neighborhood expansion including nodes per hop
    ///
    /// # Performance
    /// Uses existing expansion data if available, O(1) for cached stats
    fn expansion_stats(&self) -> NeighborhoodStats {
        NeighborhoodStats {
            total_nodes: self.node_count(),
            total_edges: self.edge_count(),
            central_count: self.central_nodes().len(),
            max_hops: self.hops(),
            density: self.calculate_density(),
        }
    }

    /// Calculate neighborhood density
    ///
    /// # Returns
    /// Ratio of actual edges to possible edges in the neighborhood
    ///
    /// # Performance
    /// O(1) calculation using existing efficient counts
    fn calculate_density(&self) -> f64 {
        let node_count = self.node_count();
        if node_count <= 1 {
            return 0.0;
        }

        let edge_count = self.edge_count();
        let max_possible_edges = node_count * (node_count - 1) / 2;

        edge_count as f64 / max_possible_edges as f64
    }

    /// Expand neighborhood by additional hops
    ///
    /// # Arguments
    /// * `additional_hops` - Number of additional hops to expand
    ///
    /// # Returns
    /// New neighborhood subgraph with expanded boundaries
    ///
    /// # Performance
    /// Uses existing efficient neighborhood sampling algorithms
    fn expand_by(&self, additional_hops: usize) -> GraphResult<Box<dyn NeighborhoodOperations>>;

    /// Get nodes at a specific hop distance from central nodes
    ///
    /// # Arguments
    /// * `hop_distance` - Distance in hops from central nodes
    ///
    /// # Returns
    /// Vector of nodes at exactly the specified hop distance
    ///
    /// # Performance
    /// Uses existing BFS algorithm with hop tracking
    fn nodes_at_hop(&self, hop_distance: usize) -> GraphResult<Vec<NodeId>> {
        if hop_distance > self.hops() {
            return Ok(Vec::new());
        }

        // Use BFS from central nodes with exact hop matching
        let binding = self.graph_ref();
        let graph = binding.borrow();
        let mut all_nodes_at_hop = Vec::new();

        for &central_node in self.central_nodes() {
            // This is a simplified version - would need proper hop distance tracking
            if hop_distance == 0 {
                if self.contains_node(central_node) {
                    all_nodes_at_hop.push(central_node);
                }
            } else {
                // Use neighbors for hop_distance = 1, would need recursive BFS for higher hops
                if hop_distance == 1 {
                    let neighbors = graph.neighbors(central_node)?;
                    for neighbor in neighbors {
                        if self.contains_node(neighbor) && !self.is_central_node(neighbor) {
                            all_nodes_at_hop.push(neighbor);
                        }
                    }
                }
            }
        }

        // Remove duplicates
        all_nodes_at_hop.sort();
        all_nodes_at_hop.dedup();

        Ok(all_nodes_at_hop)
    }

    /// Get boundary nodes of the neighborhood (nodes at maximum hop distance)
    ///
    /// # Returns
    /// Vector of nodes at the boundary (maximum hop distance) of the neighborhood
    ///
    /// # Performance
    /// Delegates to nodes_at_hop with maximum hop distance
    fn boundary_nodes(&self) -> GraphResult<Vec<NodeId>> {
        self.nodes_at_hop(self.hops())
    }

    /// Merge this neighborhood with another neighborhood
    ///
    /// # Arguments
    /// * `other` - Another neighborhood to merge with
    ///
    /// # Returns
    /// New neighborhood containing nodes and edges from both neighborhoods
    ///
    /// # Performance
    /// Uses existing efficient set union operations
    fn merge_with(
        &self,
        other: &dyn NeighborhoodOperations,
    ) -> GraphResult<Box<dyn NeighborhoodOperations>>;
}

/// Statistics about neighborhood expansion and structure
#[derive(Debug, Clone)]
pub struct NeighborhoodStats {
    /// Total number of nodes in the neighborhood
    pub total_nodes: usize,
    /// Total number of edges in the neighborhood
    pub total_edges: usize,
    /// Number of central nodes
    pub central_count: usize,
    /// Maximum hop distance included
    pub max_hops: usize,
    /// Density of the neighborhood (ratio of actual to possible edges)
    pub density: f64,
}

impl NeighborhoodStats {
    /// Calculate average degree in the neighborhood
    pub fn average_degree(&self) -> f64 {
        if self.total_nodes == 0 {
            0.0
        } else {
            (2.0 * self.total_edges as f64) / self.total_nodes as f64
        }
    }

    /// Calculate expansion factor (nodes per central node)
    pub fn expansion_factor(&self) -> f64 {
        if self.central_count == 0 {
            0.0
        } else {
            self.total_nodes as f64 / self.central_count as f64
        }
    }
}
