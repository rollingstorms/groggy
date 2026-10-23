//! EntityEdge - GraphEntity implementation for individual edges
//!
//! This module provides EntityEdge, which wraps individual edges with
//! the GraphEntity and EdgeOperations traits, enabling unified access
//! to edges through the trait system.

use crate::api::graph::Graph;
use crate::errors::GraphResult;
use crate::traits::{EdgeOperations, GraphEntity, SubgraphOperations};
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
    // Canonical handle ownership; only legacy API/behavior adaptation lives here.
    inner: crate::entities::Edge,
}

impl EntityEdge {
    /// Create a new EntityEdge wrapper for the given edge
    pub fn new(edge_id: EdgeId, graph_ref: Rc<RefCell<Graph>>) -> Self {
        Self {
            inner: crate::entities::Edge::from_id_unchecked(edge_id, graph_ref),
        }
    }

    /// Get the wrapped edge ID
    pub fn edge_id(&self) -> EdgeId {
        self.inner.id()
    }

    /// Check if this edge exists in the graph
    pub fn exists(&self) -> bool {
        self.inner.graph_handle().borrow().has_edge(self.inner.id())
    }
}

impl GraphEntity for EntityEdge {
    fn entity_id(&self) -> EntityId {
        EntityId::Edge(self.inner.id())
    }

    fn entity_type(&self) -> &'static str {
        "edge"
    }

    fn graph_ref(&self) -> Rc<RefCell<Graph>> {
        self.inner.graph_handle().clone()
    }

    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // Return the endpoint nodes as EntityNode wrappers
        let graph = self.inner.graph_handle().borrow();
        let (source, target) = graph.edge_endpoints(self.inner.id())?;

        let entities: Vec<Box<dyn GraphEntity>> = vec![
            Box::new(crate::entities::EntityNode::new(
                source,
                self.inner.graph_handle().clone(),
            )),
            Box::new(crate::entities::EntityNode::new(
                target,
                self.inner.graph_handle().clone(),
            )),
        ];

        Ok(entities)
    }

    fn summary(&self) -> String {
        let graph = self.inner.graph_handle().borrow();
        if let Ok((source, target)) = graph.edge_endpoints(self.inner.id()) {
            let weight = self
                .weight()
                .ok()
                .flatten()
                .map(|w| format!(", weight={:.2}", w))
                .unwrap_or_default();
            format!(
                "EntityEdge(id={}, {}->{}{}, directed={})",
                self.inner.id(),
                source,
                target,
                weight,
                graph.is_directed()
            )
        } else {
            format!("EntityEdge(id={}, invalid)", self.inner.id())
        }
    }
}

impl EdgeOperations for EntityEdge {
    fn edge_id(&self) -> EdgeId {
        self.inner.id()
    }

    fn endpoints(&self) -> GraphResult<(NodeId, NodeId)> {
        let graph_ref = self.graph_ref();
        let graph = graph_ref.borrow();
        graph.edge_endpoints(self.inner.id())
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
        let graph = self.inner.graph_handle().borrow();
        graph.get_edge_attrs(self.inner.id())
    }

    fn get_edge_attribute(&self, name: &AttrName) -> GraphResult<Option<AttrValue>> {
        self.get_attribute(name) // Delegates to GraphEntity default implementation
    }

    fn set_edge_attribute(&self, name: AttrName, value: AttrValue) -> GraphResult<()> {
        self.set_attribute(name, value) // Delegates to GraphEntity default implementation
    }

    fn is_directed(&self) -> bool {
        let graph = self.inner.graph_handle().borrow();
        graph.is_directed()
    }

    fn as_subgraph(&self) -> GraphResult<Box<dyn SubgraphOperations>> {
        let (source, target) = self.endpoints()?;
        let nodes = [source, target].into_iter().collect();
        let edges = [self.inner.id()].into_iter().collect();

        let edge_subgraph = crate::subgraphs::subgraph::Subgraph::new(
            self.graph_ref(),
            nodes,
            edges,
            format!("edge_{}", self.inner.id()),
        );

        Ok(Box::new(edge_subgraph))
    }

    fn parallel_edges(&self) -> GraphResult<Vec<EdgeId>> {
        let (source, target) = self.endpoints()?;
        let graph = self.inner.graph_handle().borrow();

        // Get all edges between these nodes
        let mut parallel = Vec::new();
        let source_edges = graph.incident_edges(source)?;

        for edge_id in source_edges {
            if edge_id != self.inner.id() {
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
