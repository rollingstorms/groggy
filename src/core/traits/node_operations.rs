//! NodeOperations - Interface for node entities in our existing GraphPool
//!
//! This trait provides a unified interface for nodes that leverages our existing
//! efficient node storage and algorithms. All node operations delegate to our
//! optimized GraphPool/GraphSpace/HistoryForest infrastructure.

use crate::core::traits::{GraphEntity, SubgraphOperations};
use crate::types::{AttrName, AttrValue, NodeId};
use crate::errors::GraphResult;
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
        let graph = self.graph_ref().borrow();
        graph.pool().get_all_node_attributes(self.node_id())
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
    fn get_node_attribute(&self, name: &AttrName) -> GraphResult<Option<&AttrValue>> {
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
        let graph = self.graph_ref().borrow();
        
        // Check if this node has a subgraph reference in GraphPool
        if let Some(AttrValue::SubgraphRef(subgraph_id)) =
            graph.pool().get_node_attribute(self.node_id(), &"contained_subgraph".into())? {
            
            // Retrieve subgraph data from GraphPool storage
            let (nodes, edges, subgraph_type) = graph.pool().get_subgraph(*subgraph_id)?;
            
            // Create appropriate subgraph type based on stored metadata
            let subgraph: Box<dyn SubgraphOperations> = match subgraph_type.as_str() {
                "neighborhood" => {
                    // Reconstruct NeighborhoodSubgraph with metadata from node attributes
                    let central_nodes = if let Some(AttrValue::NodeArray(central)) =
                        graph.pool().get_node_attribute(self.node_id(), &"central_nodes".into())? {
                        central.clone()
                    } else {
                        vec![self.node_id()] // Default to this node as central
                    };
                    let hops = if let Some(AttrValue::SmallInt(h)) =
                        graph.pool().get_node_attribute(self.node_id(), &"expansion_hops".into())? {
                        *h as usize
                    } else {
                        1 // Default hops
                    };
                    
                    Box::new(crate::core::neighborhood::NeighborhoodSubgraph::from_stored(
                        self.graph_ref(),
                        nodes,
                        edges,
                        central_nodes,
                        hops
                    ))
                },
                "component" => {
                    // TODO: Implement ComponentSubgraph when needed
                    // For now, return basic Subgraph
                    Box::new(crate::core::subgraph::Subgraph::new(
                        self.graph_ref(),
                        nodes,
                        edges,
                        subgraph_type.clone()
                    ))
                },
                _ => {
                    // Default to base Subgraph
                    Box::new(crate::core::subgraph::Subgraph::from_stored(
                        self.graph_ref(),
                        nodes,
                        edges,
                        subgraph_type.clone()
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
        self.graph_ref().borrow().pool()
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
        let graph = self.graph_ref().borrow();
        let neighborhood_result = graph.expand_neighborhood(&[self.node_id()], hops, None)?;
        
        let neighborhood = crate::core::neighborhood::NeighborhoodSubgraph::from_expansion(
            self.graph_ref(),
            vec![self.node_id()],
            hops,
            neighborhood_result
        );
        
        Ok(Box::new(neighborhood))
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
        let graph = self.graph_ref().borrow();
        let mut paths = Vec::new();
        
        for &target in targets {
            if let Some(path_result) = graph.shortest_path(self.node_id(), target)? {
                // TODO: Implement PathSubgraph when needed  
                // For now, create basic Subgraph for path
                let path_subgraph = crate::core::subgraph::Subgraph::new(
                    self.graph_ref(),
                    path_result.nodes.into_iter().collect(),
                    path_result.edges.into_iter().collect(),
                    format!("path_{}_{}", self.node_id(), target)
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
        let graph = self.graph_ref().borrow();
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
        let graph = self.graph_ref().borrow();
        graph.has_edge_between(self.node_id(), other)
    }
}