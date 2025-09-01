//! EntityEdge - GraphEntity implementation for individual edges
//!
//! This module provides EntityEdge, which wraps individual edges with
//! the GraphEntity and EdgeOperations traits, enabling unified access
//! to edges through the trait system.

use crate::api::graph::Graph;
use crate::core::traits::{EdgeOperations, GraphEntity, SubgraphOperations};
use crate::errors::GraphResult;
use crate::types::{AttrName, AttrValue, EdgeId, EntityId, NodeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// A wrapper around an EdgeId that implements GraphEntity and EdgeOperations
///
/// EntityEdge provides a unified interface to individual edges, enabling
/// them to participate in the trait system while leveraging all existing
/// efficient storage and algorithms.
#[derive(Debug, Clone)]
pub struct EntityEdge {
    /// The edge ID this entity represents
    edge_id: EdgeId,
    /// Reference to the shared graph storage infrastructure
    graph_ref: Rc<RefCell<Graph>>,
}

impl EntityEdge {
    /// Create a new EntityEdge wrapper for the given edge
    pub fn new(edge_id: EdgeId, graph_ref: Rc<RefCell<Graph>>) -> Self {
        Self { edge_id, graph_ref }
    }

    /// Get the wrapped edge ID
    pub fn edge_id(&self) -> EdgeId {
        self.edge_id
    }

    /// Check if this edge exists in the graph
    pub fn exists(&self) -> bool {
        self.graph_ref.borrow().has_edge(self.edge_id)
    }
}

impl GraphEntity for EntityEdge {
    fn entity_id(&self) -> EntityId {
        EntityId::Edge(self.edge_id)
    }

    fn entity_type(&self) -> &'static str {
        "edge"
    }

    fn graph_ref(&self) -> Rc<RefCell<Graph>> {
        self.graph_ref.clone()
    }

    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // Return the endpoint nodes as EntityNode wrappers
        let graph = self.graph_ref.borrow();
        let (source, target) = graph.edge_endpoints(self.edge_id)?;

        let entities: Vec<Box<dyn GraphEntity>> = vec![
            Box::new(crate::core::node::EntityNode::new(
                source,
                self.graph_ref.clone(),
            )),
            Box::new(crate::core::node::EntityNode::new(
                target,
                self.graph_ref.clone(),
            )),
        ];

        Ok(entities)
    }

    fn summary(&self) -> String {
        let graph = self.graph_ref.borrow();
        if let Ok((source, target)) = graph.edge_endpoints(self.edge_id) {
            let weight = self
                .weight()
                .ok()
                .flatten()
                .map(|w| format!(", weight={:.2}", w))
                .unwrap_or_default();
            format!(
                "EntityEdge(id={}, {}->{}{}, directed={})",
                self.edge_id,
                source,
                target,
                weight,
                graph.is_directed()
            )
        } else {
            format!("EntityEdge(id={}, invalid)", self.edge_id)
        }
    }
}

impl EdgeOperations for EntityEdge {
    fn edge_id(&self) -> EdgeId {
        self.edge_id
    }

    fn endpoints(&self) -> GraphResult<(NodeId, NodeId)> {
        let graph_ref = self.graph_ref();
        let graph = graph_ref.borrow();
        graph.edge_endpoints(self.edge_id)
    }

    fn source(&self) -> GraphResult<NodeId> {
        let (source, _) = self.endpoints()?;
        Ok(source)
    }

    fn target(&self) -> GraphResult<NodeId> {
        let (_, target) = self.endpoints()?;
        Ok(target)
    }

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

    fn connects(&self, node1: NodeId, node2: NodeId) -> GraphResult<bool> {
        let (source, target) = self.endpoints()?;
        Ok((source == node1 && target == node2) || (source == node2 && target == node1))
    }

    fn edge_attributes(&self) -> GraphResult<HashMap<AttrName, AttrValue>> {
        let graph = self.graph_ref.borrow();
        let x = graph.pool().get_all_edge_attributes(self.edge_id);
        x
    }

    fn get_edge_attribute(&self, name: &AttrName) -> GraphResult<Option<AttrValue>> {
        self.get_attribute(name) // Delegates to GraphEntity default implementation
    }

    fn set_edge_attribute(&self, name: AttrName, value: AttrValue) -> GraphResult<()> {
        self.set_attribute(name, value) // Delegates to GraphEntity default implementation
    }

    fn is_directed(&self) -> bool {
        let graph = self.graph_ref.borrow();
        graph.is_directed()
    }

    fn weight(&self) -> GraphResult<Option<f64>> {
        match self.get_edge_attribute(&"weight".into())? {
            Some(AttrValue::Float(w)) => Ok(Some(w as f64)),
            Some(AttrValue::Int(w)) => Ok(Some(w as f64)),
            Some(AttrValue::SmallInt(w)) => Ok(Some(w as f64)),
            _ => Ok(None),
        }
    }

    fn set_weight(&self, weight: f64) -> GraphResult<()> {
        self.set_edge_attribute("weight".into(), AttrValue::Float(weight as f32))
    }

    fn capacity(&self) -> GraphResult<Option<f64>> {
        match self.get_edge_attribute(&"capacity".into())? {
            Some(AttrValue::Float(c)) => Ok(Some(c as f64)),
            Some(AttrValue::Int(c)) => Ok(Some(c as f64)),
            Some(AttrValue::SmallInt(c)) => Ok(Some(c as f64)),
            _ => Ok(None),
        }
    }

    fn as_subgraph(&self) -> GraphResult<Box<dyn SubgraphOperations>> {
        let (source, target) = self.endpoints()?;
        let nodes = [source, target].into_iter().collect();
        let edges = [self.edge_id].into_iter().collect();

        let edge_subgraph = crate::core::subgraph::Subgraph::new(
            self.graph_ref(),
            nodes,
            edges,
            format!("edge_{}", self.edge_id),
        );

        Ok(Box::new(edge_subgraph))
    }

    fn parallel_edges(&self) -> GraphResult<Vec<EdgeId>> {
        let (source, target) = self.endpoints()?;
        let graph = self.graph_ref.borrow();

        // Get all edges between these nodes
        let mut parallel = Vec::new();
        let source_edges = graph.incident_edges(source)?;

        for edge_id in source_edges {
            if edge_id != self.edge_id {
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
}
