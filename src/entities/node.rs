//! Regular Node Entity - Standard graph node implementation
//!
//! This module provides the concrete implementation for regular graph nodes.
//! Regular nodes implement GraphEntity and NodeOperations traits, delegating
//! all operations to our existing optimized storage and algorithms.

use crate::api::graph::Graph;
use crate::errors::GraphResult;
use crate::traits::{GraphEntity, NodeOperations};
use crate::types::{EntityId, NodeId};
use std::cell::RefCell;
use std::rc::Rc;

/// A regular graph node
///
/// Regular nodes are the basic building blocks of graphs. They have attributes
/// and can be connected by edges. All operations are delegated to our existing
/// optimized GraphPool storage and algorithms.
#[derive(Debug, Clone)]
pub struct Node {
    /// The unique identifier for this node
    id: NodeId,
    /// Reference to the graph containing this node
    graph: Rc<RefCell<Graph>>,
}

impl Node {
    /// Create a new Node wrapper
    ///
    /// # Arguments
    /// * `id` - The NodeId for this node
    /// * `graph` - Reference to the graph containing this node
    ///
    /// # Returns
    /// A new Node instance that implements GraphEntity and NodeOperations
    ///
    /// # Errors
    /// Returns GraphError if the node ID doesn't exist in the graph
    pub fn new(id: NodeId, graph: Rc<RefCell<Graph>>) -> GraphResult<Self> {
        // Validate that the node exists in the graph
        {
            let graph_borrowed = graph.borrow();
            if !graph_borrowed.contains_node(id) {
                return Err(crate::errors::GraphError::NodeNotFound {
                    node_id: id,
                    operation: "create node entity".to_string(),
                    suggestion: "Ensure the node exists in the graph before creating a Node entity"
                        .to_string(),
                });
            }
        }

        Ok(Node { id, graph })
    }

    /// Get the NodeId for this node
    ///
    /// # Returns
    /// The NodeId that uniquely identifies this node
    pub fn id(&self) -> NodeId {
        self.id
    }
}

impl GraphEntity for Node {
    fn entity_id(&self) -> EntityId {
        EntityId::Node(self.id)
    }

    fn entity_type(&self) -> &'static str {
        "node"
    }

    fn graph_ref(&self) -> Rc<RefCell<Graph>> {
        self.graph.clone()
    }

    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // For regular nodes, related entities are neighbor nodes
        let neighbors = self.neighbors()?;
        let mut entities: Vec<Box<dyn GraphEntity>> = Vec::new();

        for neighbor_id in neighbors {
            let neighbor_node = Node::new(neighbor_id, self.graph.clone())?;
            entities.push(Box::new(neighbor_node));
        }

        Ok(entities)
    }

    fn summary(&self) -> String {
        let graph = self.graph.borrow();

        // Get basic node information
        let degree = self.degree().unwrap_or(0);

        // Get a few key attributes if available
        let mut attr_summary = String::new();
        if let Ok(attrs) = graph.get_node_attrs(self.id) {
            let key_attrs: Vec<_> = attrs.keys().take(3).collect();
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

        format!("Node(id={}, degree={}{}) ", self.id, degree, attr_summary)
    }
}

impl NodeOperations for Node {
    fn node_id(&self) -> NodeId {
        self.id
    }

    // NodeOperations trait provides default implementations for degree() and neighbors()
    // that delegate to our graph_ref() and node_id(), so we don't need to override them
}
