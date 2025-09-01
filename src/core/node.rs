//! EntityNode - GraphEntity implementation for individual nodes
//!
//! This module provides EntityNode, which wraps individual nodes with
//! the GraphEntity and NodeOperations traits, enabling unified access
//! to nodes through the trait system.

use crate::api::graph::Graph;
use crate::core::neighborhood::NeighborhoodSampler;
use crate::core::traits::{GraphEntity, NodeOperations, SubgraphOperations};
use crate::core::traversal::TraversalEngine;
use crate::errors::GraphResult;
use crate::types::{AttrName, AttrValue, EdgeId, EntityId, NodeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// A wrapper around a NodeId that implements GraphEntity and NodeOperations
///
/// EntityNode provides a unified interface to individual nodes, enabling
/// them to participate in the trait system while leveraging all existing
/// efficient storage and algorithms.
#[derive(Debug, Clone)]
pub struct EntityNode {
    /// The node ID this entity represents
    node_id: NodeId,
    /// Reference to the shared graph storage infrastructure
    graph_ref: Rc<RefCell<Graph>>,
}

impl EntityNode {
    /// Create a new EntityNode wrapper for the given node
    pub fn new(node_id: NodeId, graph_ref: Rc<RefCell<Graph>>) -> Self {
        Self { node_id, graph_ref }
    }

    /// Get the wrapped node ID
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Check if this node exists in the graph
    pub fn exists(&self) -> bool {
        self.graph_ref.borrow().has_node(self.node_id)
    }
}

impl GraphEntity for EntityNode {
    fn entity_id(&self) -> EntityId {
        EntityId::Node(self.node_id)
    }

    fn entity_type(&self) -> &'static str {
        "node"
    }

    fn graph_ref(&self) -> Rc<RefCell<Graph>> {
        self.graph_ref.clone()
    }

    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // Return neighboring nodes as EntityNode wrappers
        let graph = self.graph_ref.borrow();
        let neighbor_ids = graph.neighbors(self.node_id)?;

        let entities: Vec<Box<dyn GraphEntity>> = neighbor_ids
            .into_iter()
            .map(|neighbor_id| {
                Box::new(EntityNode::new(neighbor_id, self.graph_ref.clone()))
                    as Box<dyn GraphEntity>
            })
            .collect();

        Ok(entities)
    }

    fn summary(&self) -> String {
        let graph = self.graph_ref.borrow();
        let degree = graph.degree(self.node_id).unwrap_or(0);
        format!("EntityNode(id={}, degree={})", self.node_id, degree)
    }
}

impl NodeOperations for EntityNode {
    fn node_id(&self) -> NodeId {
        self.node_id
    }

    fn degree(&self) -> GraphResult<usize> {
        let graph_ref = self.graph_ref();
        let graph = graph_ref.borrow();
        graph.degree(self.node_id)
    }

    fn neighbors(&self) -> GraphResult<Vec<NodeId>> {
        let graph_ref = self.graph_ref();
        let graph = graph_ref.borrow();
        graph.neighbors(self.node_id)
    }

    fn node_attributes(&self) -> GraphResult<HashMap<AttrName, AttrValue>> {
        let graph = self.graph_ref.borrow();
        let x = graph.pool().get_all_node_attributes(self.node_id);
        x
    }

    fn get_node_attribute(&self, name: &AttrName) -> GraphResult<Option<AttrValue>> {
        self.get_attribute(name) // Delegates to GraphEntity default implementation
    }

    fn set_node_attribute(&self, name: AttrName, value: AttrValue) -> GraphResult<()> {
        self.set_attribute(name, value) // Delegates to GraphEntity default implementation
    }

    fn expand_to_subgraph(&self) -> GraphResult<Option<Box<dyn SubgraphOperations>>> {
        // First check if this node has a subgraph reference
        let subgraph_id_opt = {
            let graph = self.graph_ref.borrow();
            let x = graph
                .pool()
                .get_node_attribute(self.node_id, &"contained_subgraph".into())?;
            x
        };

        if let Some(AttrValue::SubgraphRef(subgraph_id)) = subgraph_id_opt {
            // Get all needed data in separate scopes to avoid long-lived borrows
            let (nodes, edges, subgraph_type) = {
                let graph = self.graph_ref.borrow();
                let x = graph.pool().get_subgraph(subgraph_id)?;
                x
            };

            let central_nodes = {
                let graph = self.graph_ref.borrow();
                let x = if let Some(AttrValue::NodeArray(central)) = graph
                    .pool()
                    .get_node_attribute(self.node_id, &"central_nodes".into())?
                {
                    central
                } else {
                    vec![self.node_id] // Default to this node as central
                };
                x
            };

            let hops = {
                let graph = self.graph_ref.borrow();
                let x = if let Some(AttrValue::SmallInt(h)) = graph
                    .pool()
                    .get_node_attribute(self.node_id, &"expansion_hops".into())?
                {
                    h as usize
                } else {
                    1 // Default hops
                };
                x
            };

            // Create appropriate subgraph type based on stored metadata
            let subgraph: Box<dyn SubgraphOperations> = match subgraph_type.as_str() {
                "neighborhood" => Box::new(
                    crate::core::neighborhood::NeighborhoodSubgraph::from_stored(
                        self.graph_ref(),
                        nodes,
                        edges,
                        central_nodes,
                        hops,
                    ),
                ),
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

    fn is_meta_node(&self) -> bool {
        self.graph_ref
            .borrow()
            .pool()
            .get_node_attribute(self.node_id, &"contained_subgraph".into())
            .map(|attr_opt| attr_opt.is_some())
            .unwrap_or(false)
    }

    fn contained_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        if let Some(subgraph) = self.expand_to_subgraph()? {
            subgraph.related_entities()
        } else {
            Ok(Vec::new())
        }
    }

    fn neighborhood(&self, hops: usize) -> GraphResult<Box<dyn SubgraphOperations>> {
        // Use existing NeighborhoodSampler for efficient neighborhood expansion
        let graph = self.graph_ref.borrow_mut();
        // Use NeighborhoodSampler directly
        let mut neighborhood_sampler = NeighborhoodSampler::new();
        let result = neighborhood_sampler.unified_neighborhood(
            &graph.pool(),
            graph.space(),
            &[self.node_id],
            hops,
        )?;

        // unified_neighborhood already returns a NeighborhoodSubgraph
        let neighborhood_subgraph = result;

        Ok(Box::new(neighborhood_subgraph))
    }

    fn shortest_paths(&self, targets: &[NodeId]) -> GraphResult<Vec<Box<dyn SubgraphOperations>>> {
        let graph = self.graph_ref.borrow_mut();
        let mut paths = Vec::new();
        let options = crate::core::traversal::PathFindingOptions::default();

        for &target in targets {
            // Use TraversalEngine directly
            let mut traversal_engine = TraversalEngine::new();
            if let Some(path_result) = traversal_engine.shortest_path(
                &graph.pool(),
                graph.space(),
                self.node_id,
                target,
                options.clone(),
            )? {
                let path_subgraph = crate::core::subgraph::Subgraph::new(
                    self.graph_ref(),
                    path_result.nodes.into_iter().collect(),
                    path_result.edges.into_iter().collect(),
                    format!("path_{}_{}", self.node_id, target),
                );
                paths.push(Box::new(path_subgraph) as Box<dyn SubgraphOperations>);
            }
        }

        Ok(paths)
    }

    fn incident_edges(&self) -> GraphResult<Vec<EdgeId>> {
        let graph = self.graph_ref.borrow();
        graph.incident_edges(self.node_id)
    }

    fn is_connected_to(&self, other: NodeId) -> GraphResult<bool> {
        let graph = self.graph_ref.borrow();
        graph.has_edge_between(self.node_id, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;
    use crate::types::AttrValue;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_entity_node_operations() {
        // Test EntityNode and NodeOperations functionality
        let mut graph = Graph::new();

        // Create test nodes and edges
        let node1 = graph.add_node();
        let node2 = graph.add_node();
        let node3 = graph.add_node();

        let _edge1 = graph.add_edge(node1, node2).unwrap();
        let _edge2 = graph.add_edge(node2, node3).unwrap();

        // Set some attributes
        graph
            .set_node_attr(
                node1,
                "name".to_string(),
                AttrValue::Text("Alice".to_string()),
            )
            .unwrap();
        graph
            .set_node_attr(node1, "age".to_string(), AttrValue::Int(25))
            .unwrap();

        // Create EntityNode
        let graph_rc = Rc::new(RefCell::new(graph));
        let entity_node = EntityNode::new(node1, graph_rc.clone());

        // Test GraphEntity interface
        assert_eq!(entity_node.entity_type(), "node");
        assert!(entity_node.exists());

        // Test NodeOperations interface
        assert_eq!(entity_node.node_id(), node1);

        // Test degree calculation
        let degree = entity_node.degree().unwrap();
        assert_eq!(degree, 1); // node1 has one edge to node2

        // Test neighbors
        let neighbors = entity_node.neighbors().unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], node2);

        // Test attribute access
        let name = entity_node.get_node_attribute(&"name".to_string()).unwrap();
        assert!(matches!(
            name,
            Some(AttrValue::Text(_)) | Some(AttrValue::CompactText(_))
        ));

        // Test related entities (should return neighbors as EntityNodes)
        let related = entity_node.related_entities().unwrap();
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].entity_type(), "node");

        // Test connectivity
        let is_connected = entity_node.is_connected_to(node2).unwrap();
        assert!(is_connected);

        let is_not_connected = entity_node.is_connected_to(node3).unwrap();
        assert!(!is_not_connected);

        println!("EntityNode tests passed!");
    }
}
