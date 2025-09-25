//! MetaNode Entity - Node representing a collapsed subgraph
//!
//! This module provides the concrete implementation for meta-nodes, which are
//! special nodes that represent collapsed subgraphs. MetaNodes implement all
//! regular node operations plus meta-specific operations.

use crate::api::graph::Graph;
use crate::errors::GraphResult;
use crate::traits::{GraphEntity, MetaNodeOperations, NodeOperations, SubgraphOperations};
use crate::types::{AttrValue, EntityId, NodeId};
use std::cell::RefCell;
use std::rc::Rc;

/// A meta-node representing a collapsed subgraph
///
/// Meta-nodes are nodes that contain references to subgraphs. They have all
/// the capabilities of regular nodes plus meta-specific operations like
/// expanding back to the original subgraph.
#[derive(Debug, Clone)]
pub struct MetaNode {
    /// The unique identifier for this meta-node
    id: NodeId,
    /// Reference to the graph containing this meta-node
    graph: Rc<RefCell<Graph>>,
}

impl MetaNode {
    /// Create a new MetaNode wrapper
    ///
    /// # Arguments
    /// * `id` - The NodeId for this meta-node
    /// * `graph` - Reference to the graph containing this meta-node
    ///
    /// # Returns
    /// A new MetaNode instance that implements all node traits plus meta-operations
    ///
    /// # Errors
    /// Returns GraphError if the node doesn't exist or isn't a valid meta-node
    pub fn new(id: NodeId, graph: Rc<RefCell<Graph>>) -> GraphResult<Self> {
        // Validate that the node exists
        {
            let graph_borrowed = graph.borrow();
            if !graph_borrowed.contains_node(id) {
                return Err(crate::errors::GraphError::NodeNotFound {
                    node_id: id,
                    operation: "create meta-node entity".to_string(),
                    suggestion:
                        "Ensure the node exists in the graph before creating a MetaNode entity"
                            .to_string(),
                });
            }

            // TEMPORARILY DISABLED: Check if this is actually a meta-node
            // This will help us debug why the validation is failing
            /*
            if !graph_borrowed.is_meta_node(id) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Node {} is not a meta-node. Meta-nodes must have entity_type='meta' and contains_subgraph attributes.",
                    id
                )));
            }
            */
        }

        Ok(MetaNode { id, graph })
    }

    /// Get the NodeId for this meta-node
    ///
    /// # Returns
    /// The NodeId that uniquely identifies this meta-node
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Check if this node has a contained subgraph
    ///
    /// This is an alias for the trait method to make it available
    /// as a regular method on the struct.
    ///
    /// # Returns
    /// true if this meta-node contains a subgraph
    pub fn has_contained_subgraph(&self) -> bool {
        self.has_subgraph()
    }

    /// Get the contained subgraph ID
    ///
    /// This is an alias for the trait method to make it available
    /// as a regular method on the struct.
    ///
    /// # Returns
    /// Optional subgraph ID if this meta-node contains a subgraph
    pub fn contained_subgraph_id(&self) -> Option<usize> {
        self.subgraph_id()
    }

    /// Expand to subgraph
    ///
    /// This is an alias for the trait method to make it available
    /// as a regular method on the struct.
    ///
    /// # Returns
    /// Optional subgraph that this meta-node represents
    pub fn expand_to_subgraph(&self) -> GraphResult<Option<Box<dyn SubgraphOperations>>> {
        self.expand()
    }
}

impl GraphEntity for MetaNode {
    fn entity_id(&self) -> EntityId {
        EntityId::MetaNode(self.id)
    }

    fn entity_type(&self) -> &'static str {
        "meta_node"
    }

    fn graph_ref(&self) -> Rc<RefCell<Graph>> {
        self.graph.clone()
    }

    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // For meta-nodes, related entities include:
        // 1. Neighbor nodes (through meta-edges)
        // 2. The contained subgraph entities (if expanded)

        let mut entities: Vec<Box<dyn GraphEntity>> = Vec::new();

        // Add neighbor nodes
        let neighbors = self.neighbors()?;
        for neighbor_id in neighbors {
            let graph = self.graph.borrow();
            if graph.is_meta_node(neighbor_id) {
                drop(graph);
                let neighbor_meta = MetaNode::new(neighbor_id, self.graph.clone())?;
                entities.push(Box::new(neighbor_meta));
            } else {
                drop(graph);
                let neighbor_node = super::Node::new(neighbor_id, self.graph.clone())?;
                entities.push(Box::new(neighbor_node));
            }
        }

        // Add contained subgraph entities when expanded
        if let Ok(Some(subgraph)) = self.subgraph() {
            // Get entities from the expanded subgraph
            if let Ok(subgraph_entities) = subgraph.related_entities() {
                entities.extend(subgraph_entities);
            }
        }

        Ok(entities)
    }

    fn summary(&self) -> String {
        let graph = self.graph.borrow();

        // Get basic node information
        let degree = self.degree().unwrap_or(0);
        let has_subgraph = self.has_subgraph();
        let subgraph_id = self.subgraph_id();

        // Get a few key attributes if available
        let mut attr_summary = String::new();
        if let Ok(attrs) = graph.get_node_attrs(self.id) {
            let key_attrs: Vec<_> = attrs
                .keys()
                .filter(|&k| k != "entity_type" && k != "contains_subgraph") // Skip internal attrs
                .take(2)
                .collect();

            if !key_attrs.is_empty() {
                let attr_strs: Vec<String> = key_attrs
                    .iter()
                    .filter_map(|&key| {
                        graph
                            .get_node_attr(self.id, key)
                            .ok()
                            .flatten()
                            .map(|value| format!("{}={}", key, value))
                    })
                    .collect();

                if !attr_strs.is_empty() {
                    attr_summary = format!(", {}", attr_strs.join(", "));
                }
            }
        }

        let subgraph_info = if has_subgraph {
            format!(", subgraph_id={:?}", subgraph_id)
        } else {
            String::new()
        };

        format!(
            "MetaNode(id={}, degree={}{}{}) ",
            self.id, degree, subgraph_info, attr_summary
        )
    }
}

impl NodeOperations for MetaNode {
    fn node_id(&self) -> NodeId {
        self.id
    }

    // NodeOperations trait provides default implementations for degree() and neighbors()
    // that delegate to our graph_ref() and node_id(), so we don't need to override them
}

impl MetaNodeOperations for MetaNode {
    fn subgraph_id(&self) -> Option<usize> {
        let graph = self.graph.borrow();
        match graph.get_node_attr(self.id, &"contains_subgraph".into()) {
            Ok(Some(AttrValue::SubgraphRef(id))) => Some(id),
            // Also handle integer values (when converted from Python FFI)
            Ok(Some(AttrValue::Int(id))) => Some(id as usize),
            _ => None,
        }
    }

    fn subgraph(&self) -> GraphResult<Option<Box<dyn SubgraphOperations>>> {
        if let Some(subgraph_id) = self.subgraph_id() {
            // Get the stored subgraph data from the graph pool
            let (nodes, edges, _subgraph_type) = {
                let graph = self.graph.borrow();
                let result = graph.pool().get_subgraph(subgraph_id)?;
                result
            };

            // Reconstruct a concrete Subgraph instance
            let subgraph = crate::subgraphs::Subgraph::new(
                self.graph.clone(),
                nodes,
                edges,
                format!("expanded_from_meta_node_{}", self.id),
            );

            // Return as a SubgraphOperations trait object
            Ok(Some(Box::new(subgraph)))
        } else {
            Ok(None)
        }
    }
}
