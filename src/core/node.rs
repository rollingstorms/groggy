//! EntityNode - GraphEntity implementation for individual nodes
//!
//! This module provides EntityNode, which wraps individual nodes with
//! the GraphEntity and NodeOperations traits, enabling unified access
//! to nodes through the trait system.

use crate::api::graph::Graph;
use crate::core::traits::{GraphEntity, NodeOperations, SubgraphOperations};
use crate::errors::{GraphError, GraphResult};
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
        Self {
            node_id,
            graph_ref,
        }
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
        format!(
            "EntityNode(id={}, degree={})",
            self.node_id,
            degree
        )
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
        graph.pool().get_all_node_attributes(self.node_id)
    }

    fn get_node_attribute(&self, name: &AttrName) -> GraphResult<Option<&AttrValue>> {
        self.get_attribute(name) // Delegates to GraphEntity default implementation
    }

    fn set_node_attribute(&self, name: AttrName, value: AttrValue) -> GraphResult<()> {
        self.set_attribute(name, value) // Delegates to GraphEntity default implementation
    }

    fn expand_to_subgraph(&self) -> GraphResult<Option<Box<dyn SubgraphOperations>>> {
        let graph = self.graph_ref.borrow();
        
        // Check if this node has a subgraph reference in GraphPool
        if let Some(AttrValue::SubgraphRef(subgraph_id)) =
            graph.pool().get_node_attribute(self.node_id, &"contained_subgraph".into())? {
            
            // Retrieve subgraph data from GraphPool storage
            let (nodes, edges, subgraph_type) = graph.pool().get_subgraph(*subgraph_id)?;
            
            // Create appropriate subgraph type based on stored metadata
            let subgraph: Box<dyn SubgraphOperations> = match subgraph_type.as_str() {
                "neighborhood" => {
                    // Reconstruct NeighborhoodSubgraph with metadata from node attributes
                    let central_nodes = if let Some(AttrValue::NodeArray(central)) =
                        graph.pool().get_node_attribute(self.node_id, &"central_nodes".into())? {
                        central.clone()
                    } else {
                        vec![self.node_id] // Default to this node as central
                    };
                    let hops = if let Some(AttrValue::SmallInt(h)) =
                        graph.pool().get_node_attribute(self.node_id, &"expansion_hops".into())? {
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
                _ => {
                    // Default to base Subgraph
                    Box::new(crate::core::subgraph::Subgraph::new(
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

    fn is_meta_node(&self) -> bool {
        self.graph_ref.borrow().pool()
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
        let mut graph = self.graph_ref.borrow_mut();
        let result = graph.neighborhood_sampler.unified_neighborhood(
            &graph.pool.borrow(),
            &graph.space,
            &vec![self.node_id],
            hops
        )?;
        
        // Create proper NeighborhoodSubgraph with trait support
        let neighborhood_subgraph = crate::core::neighborhood::NeighborhoodSubgraph::from_expansion(
            self.graph_ref(),
            vec![self.node_id],
            hops,
            result
        );
        
        Ok(Box::new(neighborhood_subgraph))
    }

    fn shortest_paths(&self, targets: &[NodeId]) -> GraphResult<Vec<Box<dyn SubgraphOperations>>> {
        let mut graph = self.graph_ref.borrow_mut();
        let mut paths = Vec::new();
        let options = crate::core::traversal::PathFindingOptions::default();
        
        for &target in targets {
            if let Some(path_result) = graph.traversal_engine.shortest_path(
                &graph.pool.borrow(),
                &mut graph.space,
                self.node_id,
                target,
                options.clone()
            )? {
                let path_subgraph = crate::core::subgraph::Subgraph::new(
                    self.graph_ref(),
                    path_result.nodes.into_iter().collect(),
                    path_result.edges.into_iter().collect(),
                    format!("path_{}_{}", self.node_id, target)
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