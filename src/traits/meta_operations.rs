//! Meta Operations - Traits for meta-node and meta-edge specific functionality
//!
//! This module defines traits that provide meta-entity specific operations
//! in addition to the regular node/edge operations they inherit.

use crate::errors::GraphResult;
use crate::traits::{EdgeOperations, NodeOperations, SubgraphOperations};
use crate::types::{AttrName, AttrValue, EdgeId};
use std::collections::HashMap;

/// Operations specific to meta-nodes
///
/// Meta-nodes are nodes that represent collapsed subgraphs. They inherit all
/// regular node operations and add meta-specific capabilities like expanding
/// back to the original subgraph.
pub trait MetaNodeOperations: NodeOperations {
    /// Check if this meta-node contains a subgraph
    ///
    /// # Returns
    /// true if the meta-node has an associated subgraph, false otherwise
    fn has_subgraph(&self) -> bool {
        self.subgraph_id().is_some()
    }

    /// Get the ID of the contained subgraph
    ///
    /// # Returns
    /// Optional subgraph ID if this meta-node contains a subgraph
    fn subgraph_id(&self) -> Option<usize>;

    /// Get the contained subgraph as a SubgraphOperations trait object
    ///
    /// # Returns
    /// Optional subgraph that can be used for further operations
    fn subgraph(&self) -> GraphResult<Option<Box<dyn SubgraphOperations>>>;

    /// Expand meta-node back to its original subgraph
    ///
    /// This is an alias for subgraph() with a more intuitive name
    ///
    /// # Returns
    /// Optional subgraph representing the expanded meta-node
    fn expand(&self) -> GraphResult<Option<Box<dyn SubgraphOperations>>> {
        self.subgraph()
    }

    /// Get all meta-edges connected to this meta-node
    ///
    /// Meta-edges are edges with entity_type="meta" that were created during
    /// the subgraph collapse process.
    ///
    /// # Returns
    /// Vector of EdgeIds representing meta-edges connected to this meta-node
    fn meta_edges(&self) -> GraphResult<Vec<EdgeId>> {
        let graph_ref = self.graph_ref();
        let graph = graph_ref.borrow();

        // Get all incident edges
        let incident_edges = graph.incident_edges(self.node_id())?;

        // Filter to only meta-edges (those with entity_type="meta")
        let mut meta_edges = Vec::new();
        for edge_id in incident_edges {
            if let Ok(Some(attr_value)) = graph.get_edge_attr(edge_id, &"entity_type".into()) {
                if let Some(entity_type) = attr_value.as_text() {
                    if entity_type == "meta" {
                        meta_edges.push(edge_id);
                    }
                }
            }
        }

        Ok(meta_edges)
    }

    /// Re-aggregate meta-node attributes with new aggregation functions
    ///
    /// This allows updating the meta-node's aggregated attributes by re-running
    /// the aggregation process with different functions.
    ///
    /// # Arguments
    /// * `agg_functions` - New aggregation functions to apply
    ///
    /// # Returns
    /// Ok(()) if re-aggregation succeeded, GraphError otherwise
    fn re_aggregate(&self, _agg_functions: HashMap<AttrName, String>) -> GraphResult<()> {
        // TODO: Implement re-aggregation logic
        // This would need to:
        // 1. Get the original subgraph
        // 2. Re-run aggregation with new functions
        // 3. Update the meta-node's attributes

        // For now, return a placeholder
        Err(crate::errors::GraphError::NotImplemented {
            feature: "Meta-node re-aggregation".to_string(),
            tracking_issue: None,
        })
    }
}

/// Operations specific to meta-edges
///
/// Meta-edges are edges that represent aggregated or summarized relationships
/// between meta-nodes and other nodes. They inherit all regular edge operations
/// and add meta-specific capabilities.
pub trait MetaEdgeOperations: EdgeOperations {
    /// Check if this is a meta-edge
    ///
    /// Meta-edges are identified by having entity_type="meta"
    ///
    /// # Returns
    /// true if this is a meta-edge, false otherwise
    fn is_meta_edge(&self) -> bool {
        let graph_ref = self.graph_ref();
        let graph = graph_ref.borrow();

        match graph.get_edge_attr(self.edge_id(), &"entity_type".into()) {
            Ok(Some(attr_value)) => attr_value.as_text() == Some("meta"),
            _ => false,
        }
    }

    /// Get the count of original edges this meta-edge aggregates
    ///
    /// During subgraph collapse, multiple original edges may be aggregated
    /// into a single meta-edge. This returns how many were combined.
    ///
    /// # Returns
    /// Optional count of original edges, or None if not available
    fn edge_count(&self) -> Option<i64> {
        let graph_ref = self.graph_ref();
        let graph = graph_ref.borrow();

        match graph.get_edge_attr(self.edge_id(), &"edge_count".into()) {
            Ok(Some(attr_value)) => attr_value.as_int(),
            _ => None,
        }
    }

    /// Get the IDs of original edges that were aggregated into this meta-edge
    ///
    /// This is a future enhancement - currently original edge IDs are not stored
    /// during the collapse process.
    ///
    /// # Returns
    /// Optional vector of original EdgeIds, or None if not available
    fn aggregated_from(&self) -> GraphResult<Option<Vec<EdgeId>>> {
        // TODO: Store and retrieve original edge IDs during collapse
        // For now, this information is not preserved
        Ok(None)
    }

    /// Expand meta-edge back to original edges
    ///
    /// This is a future enhancement that would recreate the original edges
    /// that were aggregated into this meta-edge.
    ///
    /// # Returns
    /// Optional vector of recreated EdgeIds, or None if not possible
    fn expand(&self) -> GraphResult<Option<Vec<EdgeId>>> {
        // TODO: Implement meta-edge expansion
        // This would need to:
        // 1. Get the original edge information (if stored)
        // 2. Recreate the original edges
        // 3. Remove the meta-edge
        Ok(None)
    }

    /// Get meta-edge specific properties as a summary
    ///
    /// Returns a HashMap containing meta-edge specific attributes like
    /// edge_count, aggregation information, etc.
    ///
    /// # Returns
    /// HashMap of property names to values for this meta-edge
    fn meta_properties(&self) -> GraphResult<HashMap<String, AttrValue>> {
        let mut properties = HashMap::new();

        // Add edge count if available
        if let Some(count) = self.edge_count() {
            properties.insert("edge_count".to_string(), AttrValue::Int(count));
        }

        // Add entity type
        properties.insert(
            "entity_type".to_string(),
            AttrValue::Text("meta".to_string()),
        );

        // Add is_meta_edge flag
        properties.insert(
            "is_meta_edge".to_string(),
            AttrValue::Text("true".to_string()),
        );

        Ok(properties)
    }
}
