//! Regular Edge Entity - Standard graph edge implementation
//!
//! This module provides the concrete implementation for regular graph edges.
//! Regular edges implement GraphEntity and EdgeOperations traits, delegating
//! all operations to our existing optimized storage and algorithms.

use crate::api::graph::Graph;
use crate::errors::GraphResult;
use crate::traits::{EdgeOperations, GraphEntity};
use crate::types::{EdgeId, EntityId};
use std::cell::RefCell;
use std::rc::Rc;

/// A regular graph edge
///
/// Regular edges connect two nodes and can have attributes. All operations
/// are delegated to our existing optimized GraphPool storage and algorithms.
#[derive(Debug, Clone)]
pub struct Edge {
    /// The unique identifier for this edge
    id: EdgeId,
    /// Reference to the graph containing this edge
    graph: Rc<RefCell<Graph>>,
}

impl Edge {
    /// Create a new Edge wrapper
    ///
    /// # Arguments
    /// * `id` - The EdgeId for this edge
    /// * `graph` - Reference to the graph containing this edge
    ///
    /// # Returns
    /// A new Edge instance that implements GraphEntity and EdgeOperations
    ///
    /// # Errors
    /// Returns GraphError if the edge ID doesn't exist in the graph
    pub fn new(id: EdgeId, graph: Rc<RefCell<Graph>>) -> GraphResult<Self> {
        // Validate that the edge exists in the graph
        {
            let graph_borrowed = graph.borrow();
            if !graph_borrowed.contains_edge(id) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Edge {} does not exist in graph",
                    id
                )));
            }
        }

        Ok(Edge { id, graph })
    }

    /// Get the EdgeId for this edge
    ///
    /// # Returns
    /// The EdgeId that uniquely identifies this edge
    pub fn id(&self) -> EdgeId {
        self.id
    }
}

impl GraphEntity for Edge {
    fn entity_id(&self) -> EntityId {
        EntityId::Edge(self.id)
    }

    fn entity_type(&self) -> &'static str {
        "edge"
    }

    fn graph_ref(&self) -> Rc<RefCell<Graph>> {
        self.graph.clone()
    }

    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // For edges, related entities are the endpoint nodes
        let (source, target) = self.endpoints()?;
        let mut entities: Vec<Box<dyn GraphEntity>> = Vec::new();

        let source_node = super::Node::new(source, self.graph.clone())?;
        let target_node = super::Node::new(target, self.graph.clone())?;

        entities.push(Box::new(source_node));
        entities.push(Box::new(target_node));

        Ok(entities)
    }

    fn summary(&self) -> String {
        let graph = self.graph.borrow();

        // Get endpoints
        let endpoints = match graph.edge_endpoints(self.id) {
            Ok((source, target)) => format!("{} → {}", source, target),
            Err(_) => "? → ?".to_string(),
        };

        // Get a few key attributes if available
        let mut attr_summary = String::new();
        if let Ok(attrs) = graph.get_edge_attrs(self.id) {
            let key_attrs: Vec<_> = attrs.keys().take(2).collect();
            if !key_attrs.is_empty() {
                let attr_strs: Vec<String> = key_attrs
                    .iter()
                    .filter_map(|&key| {
                        graph
                            .get_edge_attr(self.id, key)
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

        format!("Edge(id={}, {}{}) ", self.id, endpoints, attr_summary)
    }
}

impl EdgeOperations for Edge {
    fn edge_id(&self) -> EdgeId {
        self.id
    }

    // EdgeOperations trait provides default implementations for endpoints(), source(), target()
    // that delegate to our graph_ref() and edge_id(), so we don't need to override them
}
