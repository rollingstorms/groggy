//! Staging and validation for one logical graph write.

use crate::errors::{GraphError, GraphResult};
use crate::types::{AttrName, AttrValue, EdgeId, NodeId, StateId};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchNodeRef {
    Existing(NodeId),
    /// Zero-based position in `GraphBatch::add_nodes`.
    Added(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchEdgeRef {
    Existing(EdgeId),
    /// Zero-based position in `GraphBatch::add_edges`.
    Added(usize),
}

/// One mixed write. Operations execute in this order: add nodes, add edges,
/// assign properties, remove explicit edges, then remove nodes and their edges.
#[derive(Debug, Default)]
pub struct GraphBatch {
    pub add_nodes: usize,
    pub add_edges: Vec<(BatchNodeRef, BatchNodeRef)>,
    pub node_attrs: HashMap<AttrName, Vec<(BatchNodeRef, AttrValue)>>,
    pub edge_attrs: HashMap<AttrName, Vec<(BatchEdgeRef, AttrValue)>>,
    pub remove_edges: Vec<BatchEdgeRef>,
    pub remove_nodes: Vec<BatchNodeRef>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct BatchResult {
    pub state_id: StateId,
    pub added_nodes: Vec<NodeId>,
    pub added_edges: Vec<EdgeId>,
}

pub(crate) struct PreparedBatch {
    pub add_nodes: usize,
    pub add_edges: Vec<(NodeId, NodeId)>,
    pub node_attrs: HashMap<AttrName, Vec<(NodeId, AttrValue)>>,
    pub edge_attrs: HashMap<AttrName, Vec<(EdgeId, AttrValue)>>,
    pub remove_edges: Vec<EdgeId>,
    pub remove_nodes: Vec<NodeId>,
}

fn node_id(reference: BatchNodeRef, first: NodeId, count: usize) -> Result<NodeId, String> {
    match reference {
        BatchNodeRef::Existing(id) => Ok(id),
        BatchNodeRef::Added(position) if position < count => Ok(first + position),
        BatchNodeRef::Added(position) => Err(format!(
            "new node reference {position} is outside add_nodes count {count}"
        )),
    }
}

fn edge_id(reference: BatchEdgeRef, first: EdgeId, count: usize) -> Result<EdgeId, String> {
    match reference {
        BatchEdgeRef::Existing(id) => Ok(id),
        BatchEdgeRef::Added(position) if position < count => Ok(first + position),
        BatchEdgeRef::Added(position) => Err(format!(
            "new edge reference {position} is outside add_edges count {count}"
        )),
    }
}

impl GraphBatch {
    pub(crate) fn prepare<NodeActive, EdgeActive, EdgeEndpoints>(
        self,
        node_active: NodeActive,
        edge_active: EdgeActive,
        edge_endpoints: EdgeEndpoints,
        next_node_id: NodeId,
        next_edge_id: EdgeId,
    ) -> GraphResult<PreparedBatch>
    where
        NodeActive: Fn(NodeId) -> bool,
        EdgeActive: Fn(EdgeId) -> bool,
        EdgeEndpoints: Fn(EdgeId) -> Option<(NodeId, NodeId)>,
    {
        if next_node_id.checked_add(self.add_nodes).is_none()
            || next_edge_id.checked_add(self.add_edges.len()).is_none()
        {
            return Err(GraphError::InvalidInput(
                "Batch ID allocation would overflow".into(),
            ));
        }
        let mut errors = Vec::new();
        let node_end = next_node_id + self.add_nodes;
        let edge_end = next_edge_id + self.add_edges.len();
        let known_node = |id| node_active(id) || (next_node_id..node_end).contains(&id);
        let known_edge = |id| edge_active(id) || (next_edge_id..edge_end).contains(&id);
        let mut resolved_edges = Vec::with_capacity(self.add_edges.len());

        for (position, &(source_ref, target_ref)) in self.add_edges.iter().enumerate() {
            let source = node_id(source_ref, next_node_id, self.add_nodes);
            let target = node_id(target_ref, next_node_id, self.add_nodes);
            match (source, target) {
                (Ok(source), Ok(target)) => {
                    if !known_node(source) || !known_node(target) {
                        errors.push(format!(
                            "add_edges[{position}]: endpoints ({source}, {target}) are not both active"
                        ));
                    }
                    resolved_edges.push((source, target));
                }
                (source, target) => {
                    for error in source.err().into_iter().chain(target.err()) {
                        errors.push(format!("add_edges[{position}]: {error}"));
                    }
                    resolved_edges.push((usize::MAX, usize::MAX));
                }
            }
        }

        let mut removed_nodes = Vec::new();
        let mut seen_removed_nodes = HashSet::new();
        for (position, &reference) in self.remove_nodes.iter().enumerate() {
            match node_id(reference, next_node_id, self.add_nodes) {
                Ok(id) if !known_node(id) => {
                    errors.push(format!("remove_nodes[{position}]: node {id} is not active"));
                }
                Ok(id) if !seen_removed_nodes.insert(id) => {
                    errors.push(format!("remove_nodes[{position}]: duplicate node {id}"));
                }
                Ok(id) => removed_nodes.push(id),
                Err(error) => errors.push(format!("remove_nodes[{position}]: {error}")),
            }
        }

        let mut removed_edges = Vec::new();
        let mut seen_removed_edges = HashSet::new();
        for (position, &reference) in self.remove_edges.iter().enumerate() {
            match edge_id(reference, next_edge_id, self.add_edges.len()) {
                Ok(id) if !known_edge(id) => {
                    errors.push(format!("remove_edges[{position}]: edge {id} is not active"));
                }
                Ok(id) if !seen_removed_edges.insert(id) => {
                    errors.push(format!("remove_edges[{position}]: duplicate edge {id}"));
                }
                Ok(id) => removed_edges.push(id),
                Err(error) => errors.push(format!("remove_edges[{position}]: {error}")),
            }
        }

        for (position, &(source, target)) in resolved_edges.iter().enumerate() {
            if source != usize::MAX
                && (seen_removed_nodes.contains(&source) || seen_removed_nodes.contains(&target))
            {
                errors.push(format!(
                    "add_edges[{position}]: endpoint is removed in this batch"
                ));
            }
        }

        let mut node_attrs = HashMap::new();
        let mut node_keys: Vec<_> = self.node_attrs.into_iter().collect();
        node_keys.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
        for (name, values) in node_keys {
            let mut resolved = Vec::with_capacity(values.len());
            let mut seen = HashSet::new();
            if (name == "entity_type" || name == super::graph::EXTERNAL_ID) && !values.is_empty() {
                errors.push(format!("node_attrs.{name}: system-managed attribute"));
            }
            for (position, (reference, mut value)) in values.into_iter().enumerate() {
                if let Err(error) =
                    super::graph::kinds::normalize_kind_property_in_place(&name, &mut value, true)
                {
                    errors.push(format!("node_attrs.{name}[{position}]: {error}"));
                }
                match node_id(reference, next_node_id, self.add_nodes) {
                    Ok(id) => {
                        if !known_node(id) || seen_removed_nodes.contains(&id) {
                            errors.push(format!("node_attrs.{name}[{position}]: node {id} is not active in the resulting state"));
                        }
                        if !seen.insert(id) {
                            errors.push(format!(
                                "node_attrs.{name}[{position}]: duplicate node {id}"
                            ));
                        }
                        resolved.push((id, value));
                    }
                    Err(error) => errors.push(format!("node_attrs.{name}[{position}]: {error}")),
                }
            }
            node_attrs.insert(name, resolved);
        }

        let mut edge_attrs = HashMap::new();
        let mut edge_keys: Vec<_> = self.edge_attrs.into_iter().collect();
        edge_keys.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
        for (name, values) in edge_keys {
            let mut resolved = Vec::with_capacity(values.len());
            let mut seen = HashSet::new();
            if name == super::graph::EXTERNAL_ID && !values.is_empty() {
                errors.push("edge_attrs.external_id: system-managed attribute".into());
            }
            for (position, (reference, mut value)) in values.into_iter().enumerate() {
                if let Err(error) =
                    super::graph::kinds::normalize_kind_property_in_place(&name, &mut value, false)
                {
                    errors.push(format!("edge_attrs.{name}[{position}]: {error}"));
                }
                match edge_id(reference, next_edge_id, self.add_edges.len()) {
                    Ok(id) => {
                        let endpoints = if (next_edge_id..edge_end).contains(&id) {
                            resolved_edges.get(id - next_edge_id).copied()
                        } else {
                            edge_endpoints(id)
                        };
                        let retained = known_edge(id)
                            && !seen_removed_edges.contains(&id)
                            && endpoints.is_some_and(|(source, target)| {
                                !seen_removed_nodes.contains(&source)
                                    && !seen_removed_nodes.contains(&target)
                            });
                        if !retained {
                            errors.push(format!("edge_attrs.{name}[{position}]: edge {id} is not active in the resulting state"));
                        }
                        if !seen.insert(id) {
                            errors.push(format!(
                                "edge_attrs.{name}[{position}]: duplicate edge {id}"
                            ));
                        }
                        resolved.push((id, value));
                    }
                    Err(error) => errors.push(format!("edge_attrs.{name}[{position}]: {error}")),
                }
            }
            edge_attrs.insert(name, resolved);
        }

        if !errors.is_empty() {
            return Err(GraphError::InvalidInput(format!(
                "Invalid graph batch: {}",
                errors.join("; ")
            )));
        }
        if self.add_nodes == 0
            && resolved_edges.is_empty()
            && node_attrs.values().all(Vec::is_empty)
            && edge_attrs.values().all(Vec::is_empty)
            && removed_edges.is_empty()
            && removed_nodes.is_empty()
        {
            return Err(GraphError::InvalidInput(
                "Graph batch has no changes".into(),
            ));
        }
        Ok(PreparedBatch {
            add_nodes: self.add_nodes,
            add_edges: resolved_edges,
            node_attrs,
            edge_attrs,
            remove_edges: removed_edges,
            remove_nodes: removed_nodes,
        })
    }
}
