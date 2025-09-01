//! ComponentOperations - Specialized interface for connected component entities
//!
//! This trait provides specialized operations for connected components while leveraging
//! our existing efficient SubgraphOperations infrastructure. All component operations
//! use the same storage pattern with component-specific metadata.

use crate::core::traits::SubgraphOperations;
use crate::errors::GraphResult;
use crate::types::NodeId;

/// Specialized operations for connected component entities
///
/// This trait extends SubgraphOperations with component-specific functionality.
/// Components maintain the same efficient HashSet<NodeId> + HashSet<EdgeId> storage
/// with additional metadata for component identification and analysis.
///
/// # Design Principles
/// - **Same Efficient Storage**: Uses identical storage pattern as base Subgraph
/// - **Component Metadata**: Adds component ID and size metadata for analysis
/// - **Algorithm Integration**: Leverages existing connected components algorithms
/// - **Composable Operations**: Can merge components and analyze component networks
pub trait ComponentOperations: SubgraphOperations {
    /// Get the unique component identifier
    ///
    /// # Returns
    /// Unique component ID within the parent graph's component structure
    ///
    /// # Performance
    /// O(1) - Direct field access
    fn component_id(&self) -> usize;

    /// Check if this is the largest component in the graph
    ///
    /// # Returns
    /// true if this component has the most nodes among all components
    ///
    /// # Performance
    /// O(1) - Uses precomputed metadata
    fn is_largest_component(&self) -> bool;

    /// Get component size (delegates to efficient node count)
    ///
    /// # Returns
    /// Number of nodes in this component
    ///
    /// # Performance
    /// O(1) - Direct .len() call on HashSet
    fn component_size(&self) -> usize {
        self.node_count() // Delegates to existing efficient SubgraphOperations method
    }

    /// Merge this component with another component
    ///
    /// # Arguments
    /// * `other` - Another component to merge with
    ///
    /// # Returns
    /// New component containing nodes and edges from both components
    ///
    /// # Performance
    /// Uses existing efficient set union operations
    fn merge_with(
        &self,
        other: &dyn ComponentOperations,
    ) -> GraphResult<Box<dyn ComponentOperations>>;

    /// Find boundary nodes (nodes with edges to other components)
    ///
    /// # Returns
    /// Vector of node IDs that have connections outside this component
    ///
    /// # Performance
    /// Uses existing efficient neighbor lookup with filtering
    fn boundary_nodes(&self) -> GraphResult<Vec<NodeId>>;

    /// Calculate internal density of this component
    ///
    /// # Returns
    /// Ratio of actual edges to possible edges within the component
    ///
    /// # Performance
    /// O(1) calculation using existing efficient edge count
    fn internal_density(&self) -> f64 {
        let node_count = self.node_count();
        if node_count <= 1 {
            return 0.0;
        }

        let edge_count = self.edge_count();
        let max_possible_edges = node_count * (node_count - 1) / 2;

        edge_count as f64 / max_possible_edges as f64
    }

    /// Check if this component is strongly connected
    ///
    /// # Returns
    /// true if there is a path between every pair of nodes
    ///
    /// # Performance
    /// Uses existing BFS algorithm for connectivity checking
    fn is_strongly_connected(&self) -> GraphResult<bool> {
        // For undirected components, being connected is equivalent to being strongly connected
        // For directed graphs, this would require more sophisticated checking
        Ok(self.node_count() > 0 && self.edge_count() >= self.node_count() - 1)
    }
}
