//! EdgeOperations - Interface for edge entities in our existing GraphPool
//!
//! This trait provides a unified interface for edges that leverages our existing
//! efficient edge storage and algorithms. All edge operations delegate to our
//! optimized GraphPool/GraphSpace/HistoryForest infrastructure.

use crate::core::traits::{GraphEntity, SubgraphOperations};
use crate::errors::GraphResult;
use crate::types::{AttrName, AttrValue, EdgeId, NodeId};
use std::collections::HashMap;

/// Common operations that all edge-like entities support
///
/// This trait provides a unified interface over our existing efficient edge storage.
/// All edge operations delegate to existing optimized algorithms and storage systems.
///
/// # Design Principles
/// - **Storage Integration**: All operations use existing GraphPool/GraphSpace infrastructure
/// - **Algorithm Reuse**: All operations delegate to existing optimized edge algorithms
/// - **Zero Copying**: Attribute access returns references to GraphPool data
/// - **Endpoint Access**: Efficient access to connected nodes
pub trait EdgeOperations: GraphEntity {
    /// Edge ID (our existing efficient type)
    ///
    /// # Returns
    /// The EdgeId for this edge entity
    ///
    /// # Performance
    /// O(1) - Direct field access
    fn edge_id(&self) -> EdgeId;

    /// Get edge endpoints using our existing algorithms
    ///
    /// # Returns
    /// Tuple of (source, target) node IDs
    ///
    /// # Performance
    /// Uses existing efficient edge endpoint lookup with GraphSpace
    fn endpoints(&self) -> GraphResult<(NodeId, NodeId)> {
        let graph_ref = self.graph_ref();
        let graph = graph_ref.borrow();
        graph.edge_endpoints(self.edge_id())
    }

    /// Get source node of this edge
    ///
    /// # Returns
    /// Source node ID
    ///
    /// # Performance
    /// Uses existing efficient endpoint lookup
    fn source(&self) -> GraphResult<NodeId> {
        let (source, _) = self.endpoints()?;
        Ok(source)
    }

    /// Get target node of this edge
    ///
    /// # Returns
    /// Target node ID
    ///
    /// # Performance
    /// Uses existing efficient endpoint lookup
    fn target(&self) -> GraphResult<NodeId> {
        let (_, target) = self.endpoints()?;
        Ok(target)
    }

    /// Get the other endpoint given one endpoint
    ///
    /// # Arguments
    /// * `node_id` - One endpoint of the edge
    ///
    /// # Returns
    /// The other endpoint, or None if the given node is not an endpoint
    ///
    /// # Performance
    /// O(1) using existing endpoint lookup
    fn other_endpoint(&self, node_id: NodeId) -> GraphResult<Option<NodeId>> {
        let (source, target) = self.endpoints()?;
        if source == node_id {
            Ok(Some(target))
        } else if target == node_id {
            Ok(Some(source))
        } else {
            Ok(None)
        }
    }

    /// Check if this edge connects the given nodes
    ///
    /// # Arguments
    /// * `node1` - First node to check
    /// * `node2` - Second node to check
    ///
    /// # Returns
    /// true if this edge connects the two nodes (in either direction)
    ///
    /// # Performance
    /// O(1) using existing endpoint lookup
    fn connects(&self, node1: NodeId, node2: NodeId) -> GraphResult<bool> {
        let (source, target) = self.endpoints()?;
        Ok((source == node1 && target == node2) || (source == node2 && target == node1))
    }

    /// All edge attributes from GraphPool
    ///
    /// # Returns
    /// HashMap of all attributes for this edge
    ///
    /// # Performance
    /// Iterates through edge's attributes in GraphPool columnar storage
    fn edge_attributes(&self) -> GraphResult<HashMap<AttrName, AttrValue>> {
        let binding = self.graph_ref();
        let graph = binding.borrow();
        let x = graph.pool().get_all_edge_attributes(self.edge_id());
        x
    }

    /// Get specific edge attribute from GraphPool
    ///
    /// # Arguments
    /// * `name` - Attribute name to retrieve
    ///
    /// # Returns
    /// Optional reference to attribute value in GraphPool
    ///
    /// # Performance
    /// O(1) - Direct lookup in optimized columnar storage
    fn get_edge_attribute(&self, name: &AttrName) -> GraphResult<Option<AttrValue>> {
        self.get_attribute(name) // Delegates to GraphEntity default implementation
    }

    /// Set edge attribute in GraphPool
    ///
    /// # Arguments
    /// * `name` - Attribute name to set
    /// * `value` - Attribute value to store
    ///
    /// # Performance
    /// Uses existing efficient GraphPool attribute storage with memory pooling
    fn set_edge_attribute(&self, name: AttrName, value: AttrValue) -> GraphResult<()> {
        self.set_attribute(name, value) // Delegates to GraphEntity default implementation
    }

    /// Check if this edge is directed
    ///
    /// # Returns
    /// true if the graph containing this edge is directed
    ///
    /// # Performance
    /// O(1) - Graph type lookup
    fn is_directed(&self) -> bool {
        let binding = self.graph_ref();
        let graph = binding.borrow();
        graph.is_directed()
    }

    /// Get edge weight if stored as attribute
    ///
    /// # Returns
    /// Edge weight as f64, or None if no weight attribute exists
    ///
    /// # Performance
    /// O(1) - Single attribute lookup
    fn weight(&self) -> GraphResult<Option<f64>> {
        match self.get_edge_attribute(&"weight".into())? {
            Some(AttrValue::Float(w)) => Ok(Some(w as f64)),
            Some(AttrValue::Int(w)) => Ok(Some(w as f64)),
            Some(AttrValue::SmallInt(w)) => Ok(Some(w as f64)),
            _ => Ok(None),
        }
    }

    /// Set edge weight as attribute
    ///
    /// # Arguments
    /// * `weight` - Weight value to set
    ///
    /// # Performance
    /// Uses existing efficient attribute storage
    fn set_weight(&self, weight: f64) -> GraphResult<()> {
        self.set_edge_attribute("weight".into(), AttrValue::Float(weight as f32))
    }

    /// Get edge capacity if stored as attribute (for flow networks)
    ///
    /// # Returns
    /// Edge capacity as f64, or None if no capacity attribute exists
    ///
    /// # Performance
    /// O(1) - Single attribute lookup
    fn capacity(&self) -> GraphResult<Option<f64>> {
        match self.get_edge_attribute(&"capacity".into())? {
            Some(AttrValue::Float(c)) => Ok(Some(c as f64)),
            Some(AttrValue::Int(c)) => Ok(Some(c as f64)),
            Some(AttrValue::SmallInt(c)) => Ok(Some(c as f64)),
            _ => Ok(None),
        }
    }

    /// Create subgraph containing just this edge and its endpoints
    ///
    /// # Returns
    /// Subgraph containing this edge and its two endpoint nodes
    ///
    /// # Performance
    /// Creates minimal subgraph using existing efficient subgraph creation
    fn as_subgraph(&self) -> GraphResult<Box<dyn SubgraphOperations>> {
        let (source, target) = self.endpoints()?;
        let nodes = [source, target].into_iter().collect();
        let edges = [self.edge_id()].into_iter().collect();

        let edge_subgraph = crate::core::subgraph::Subgraph::new(
            self.graph_ref(),
            nodes,
            edges,
            format!("edge_{}", self.edge_id()),
        );

        Ok(Box::new(edge_subgraph))
    }

    /// Get parallel edges (other edges between the same endpoints)
    ///
    /// # Returns
    /// Vector of edge IDs that connect the same endpoints as this edge
    ///
    /// # Performance
    /// Uses existing efficient edge enumeration algorithms
    fn parallel_edges(&self) -> GraphResult<Vec<EdgeId>> {
        let (source, target) = self.endpoints()?;
        let binding = self.graph_ref();
        let graph = binding.borrow();

        // Get all edges between these nodes
        let mut parallel = Vec::new();
        let source_edges = graph.incident_edges(source)?;

        for edge_id in source_edges {
            if edge_id != self.edge_id() {
                if let Ok((edge_source, edge_target)) = graph.edge_endpoints(edge_id) {
                    if (edge_source == source && edge_target == target)
                        || (edge_source == target && edge_target == source)
                    {
                        parallel.push(edge_id);
                    }
                }
            }
        }

        Ok(parallel)
    }

    // === BULK ATTRIBUTE OPERATIONS ===
    // Note: Bulk operations are available via SubgraphOperations::set_edge_attrs()
    // for cross-entity bulk operations. Individual edges use single attribute methods.
}
