//! NodeOperations - Interface for node entities in our existing GraphPool
//!
//! This trait provides a unified interface for nodes that leverages our existing
//! efficient node storage and algorithms. All node operations delegate to our
//! optimized GraphPool/GraphSpace/HistoryForest infrastructure.

use crate::core::traits::{GraphEntity, SubgraphOperations};
use crate::core::traversal::TraversalEngine;
use crate::errors::GraphResult;
use crate::types::{AttrName, AttrValue, NodeId};
use std::collections::HashMap;

/// Common operations that all node-like entities support
///
/// This trait provides a unified interface over our existing efficient node storage.
/// All node operations delegate to existing optimized algorithms and storage systems.
///
/// # Design Principles
/// - **Storage Integration**: All operations use existing GraphPool/GraphSpace infrastructure
/// - **Algorithm Reuse**: All operations delegate to existing optimized node algorithms
/// - **Zero Copying**: Attribute access returns references to GraphPool data
/// - **Hierarchical Support**: Meta-nodes can expand to subgraphs stored in GraphPool
pub trait NodeOperations: GraphEntity {
    /// Node ID (our existing efficient type)
    ///
    /// # Returns
    /// The NodeId for this node entity
    ///
    /// # Performance
    /// O(1) - Direct field access
    fn node_id(&self) -> NodeId;

    /// Node degree using our existing algorithms
    ///
    /// # Returns
    /// Number of edges connected to this node
    ///
    /// # Performance
    /// Uses existing efficient degree calculation with GraphSpace
    fn degree(&self) -> GraphResult<usize> {
        let graph_ref = self.graph_ref();
        let graph = graph_ref.borrow();
        graph.degree(self.node_id())
    }

    /// Neighbors using our existing efficient topology
    ///
    /// # Returns
    /// Vector of neighboring node IDs
    ///
    /// # Performance
    /// Uses existing efficient neighbor lookup with GraphSpace
    fn neighbors(&self) -> GraphResult<Vec<NodeId>> {
        let graph_ref = self.graph_ref();
        let graph = graph_ref.borrow();
        graph.neighbors(self.node_id())
    }

    /// All node attributes from GraphPool
    ///
    /// # Returns
    /// HashMap of all attributes for this node
    ///
    /// # Performance
    /// Iterates through node's attributes in GraphPool columnar storage
    fn node_attributes(&self) -> GraphResult<HashMap<AttrName, AttrValue>> {
        let binding = self.graph_ref();
        let graph = binding.borrow();
        let x = graph.pool().get_all_node_attributes(self.node_id());
        x
    }

    /// Get specific node attribute from GraphPool
    ///
    /// # Arguments
    /// * `name` - Attribute name to retrieve
    ///
    /// # Returns
    /// Optional reference to attribute value in GraphPool
    ///
    /// # Performance
    /// O(1) - Direct lookup in optimized columnar storage
    fn get_node_attribute(&self, name: &AttrName) -> GraphResult<Option<AttrValue>> {
        self.get_attribute(name) // Delegates to GraphEntity default implementation
    }

    /// Set node attribute in GraphPool
    ///
    /// # Arguments
    /// * `name` - Attribute name to set
    /// * `value` - Attribute value to store
    ///
    /// # Performance
    /// Uses existing efficient GraphPool attribute storage with memory pooling
    fn set_node_attribute(&self, name: AttrName, value: AttrValue) -> GraphResult<()> {
        self.set_attribute(name, value) // Delegates to GraphEntity default implementation
    }

    // === HIERARCHICAL OPERATIONS (Integration with meta-nodes) ===

    /// Expand meta-node to its contained subgraph
    ///
    /// # Returns
    /// Optional subgraph contained within this meta-node (None for regular nodes)
    ///
    /// # Performance
    /// Queries GraphPool for subgraph reference, then reconstructs appropriate subgraph type
    fn expand_to_subgraph(&self) -> GraphResult<Option<Box<dyn SubgraphOperations>>> {
        // First check if this node has a subgraph reference
        let subgraph_id_opt = {
            let binding = self.graph_ref();
            let graph = binding.borrow();
            let x = graph
                .pool()
                .get_node_attribute(self.node_id(), &"contained_subgraph".into())?;
            x
        };

        if let Some(AttrValue::SubgraphRef(subgraph_id)) = subgraph_id_opt {
            // Get all needed data in separate scopes to avoid long-lived borrows
            let (nodes, edges, subgraph_type) = {
                let binding = self.graph_ref();
                let graph = binding.borrow();
                let x = graph.pool().get_subgraph(subgraph_id)?;
                x
            };

            // Create appropriate subgraph type based on stored metadata
            let subgraph: Box<dyn SubgraphOperations> = match subgraph_type.as_str() {
                "neighborhood" => {
                    // Reconstruct NeighborhoodSubgraph with metadata from node attributes
                    let _central_nodes = {
                        let binding = self.graph_ref();
                        let graph = binding.borrow();
                        let x = if let Some(AttrValue::NodeArray(central)) = graph
                            .pool()
                            .get_node_attribute(self.node_id(), &"central_nodes".into())?
                        {
                            central
                        } else {
                            vec![self.node_id()] // Default to this node as central
                        };
                        x
                    };

                    let _hops = {
                        let binding = self.graph_ref();
                        let graph = binding.borrow();
                        let x = if let Some(AttrValue::SmallInt(h)) = graph
                            .pool()
                            .get_node_attribute(self.node_id(), &"expansion_hops".into())?
                        {
                            h as usize
                        } else {
                            1 // Default hops
                        };
                        x
                    };

                    // For now, just create a regular Subgraph since NeighborhoodSubgraph::from_stored doesn't exist
                    Box::new(crate::core::subgraph::Subgraph::new(
                        self.graph_ref(),
                        nodes,
                        edges,
                        subgraph_type.clone(),
                    ))
                }
                "component" => {
                    // TODO: Implement ComponentSubgraph when needed
                    // For now, return basic Subgraph
                    Box::new(crate::core::subgraph::Subgraph::new(
                        self.graph_ref(),
                        nodes,
                        edges,
                        subgraph_type.clone(),
                    ))
                }
                _ => {
                    // Default to base Subgraph
                    Box::new(crate::core::subgraph::Subgraph::new(
                        self.graph_ref(),
                        nodes,
                        edges,
                        subgraph_type.clone(),
                    ))
                }
            };

            Ok(Some(subgraph))
        } else {
            Ok(None)
        }
    }

    /// Check if this is a meta-node (contains a subgraph)
    ///
    /// # Returns
    /// true if this node contains a subgraph reference in GraphPool
    ///
    /// # Performance
    /// O(1) - Single attribute lookup in GraphPool
    fn is_meta_node(&self) -> bool {
        self.graph_ref()
            .borrow()
            .pool()
            .get_node_attribute(self.node_id(), &"contained_subgraph".into())
            .map(|attr_opt| attr_opt.is_some())
            .unwrap_or(false)
    }

    /// Get entities contained within this meta-node
    ///
    /// # Returns
    /// Vector of entities contained within this meta-node's subgraph
    ///
    /// # Performance
    /// If meta-node: expands subgraph and returns its related entities
    /// If regular node: returns empty vector
    fn contained_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        if let Some(subgraph) = self.expand_to_subgraph()? {
            subgraph.related_entities()
        } else {
            Ok(Vec::new())
        }
    }

    // === ALGORITHM OPERATIONS ===

    /// Create neighborhood subgraph around this node
    ///
    /// # Arguments
    /// * `hops` - Number of hops to expand the neighborhood
    ///
    /// # Returns
    /// NeighborhoodSubgraph containing this node and its k-hop neighbors
    ///
    /// # Performance
    /// Uses existing efficient neighborhood expansion algorithm
    fn neighborhood(&self, hops: usize) -> GraphResult<Box<dyn SubgraphOperations>> {
        // Use existing NeighborhoodSampler for efficient neighborhood expansion (same as EntityNode)
        let binding = self.graph_ref();
        let graph = binding.borrow_mut();
        let mut neighborhood_sampler = crate::core::neighborhood::NeighborhoodSampler::new();
        let result = neighborhood_sampler.unified_neighborhood(
            &graph.pool(),
            graph.space(),
            &[self.node_id()],
            hops,
        )?;

        // unified_neighborhood already returns a NeighborhoodSubgraph
        Ok(Box::new(result))
    }

    /// Find shortest paths from this node to multiple targets
    ///
    /// # Arguments
    /// * `targets` - Slice of target node IDs
    ///
    /// # Returns
    /// Vector of path subgraphs, one for each reachable target
    ///
    /// # Performance
    /// Uses existing efficient shortest path algorithm with multiple targets
    fn shortest_paths(&self, targets: &[NodeId]) -> GraphResult<Vec<Box<dyn SubgraphOperations>>> {
        let binding = self.graph_ref();
        let graph = binding.borrow_mut();
        let mut paths = Vec::new();

        for &target in targets {
            let options = crate::core::traversal::PathFindingOptions::default();
            // Use TraversalEngine directly
            let mut traversal_engine = TraversalEngine::new();
            if let Some(path_result) = traversal_engine.shortest_path(
                &graph.pool(),
                graph.space(),
                self.node_id(),
                target,
                options,
            )? {
                // TODO: Implement PathSubgraph when needed
                // For now, create basic Subgraph for path
                let path_subgraph = crate::core::subgraph::Subgraph::new(
                    self.graph_ref(),
                    path_result.nodes.into_iter().collect(),
                    path_result.edges.into_iter().collect(),
                    format!("path_{}_{}", self.node_id(), target),
                );
                paths.push(Box::new(path_subgraph) as Box<dyn SubgraphOperations>);
            }
        }

        Ok(paths)
    }

    /// Get all edges connected to this node
    ///
    /// # Returns
    /// Vector of edge IDs connected to this node
    ///
    /// # Performance
    /// Uses existing efficient edge enumeration with GraphSpace
    fn incident_edges(&self) -> GraphResult<Vec<crate::types::EdgeId>> {
        let binding = self.graph_ref();
        let graph = binding.borrow();
        graph.incident_edges(self.node_id())
    }

    /// Check if this node is connected to another node
    ///
    /// # Arguments
    /// * `other` - Other node ID to check connection with
    ///
    /// # Returns
    /// true if there's an edge between this node and the other node
    ///
    /// # Performance
    /// Uses existing efficient edge existence check
    fn is_connected_to(&self, other: NodeId) -> GraphResult<bool> {
        let binding = self.graph_ref();
        let graph = binding.borrow();
        graph.has_edge_between(self.node_id(), other)
    }

    // === BULK ATTRIBUTE OPERATIONS ===
    // Note: Bulk operations are available via SubgraphOperations::set_node_attrs()
    // for cross-entity bulk operations. Individual nodes use single attribute methods.
}
