//! Staging and validation for one logical graph write.

use crate::errors::{GraphError, GraphResult};
use crate::state::publication::{ChangeKey, TimestampMicros};
use crate::types::{AttrName, AttrValue, EdgeId, NodeId, StateId};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BatchNodeRef {
    Existing(NodeId),
    /// Zero-based position in `GraphBatch::add_nodes`.
    Added(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BatchEdgeRef {
    Existing(EdgeId),
    /// Zero-based position in `GraphBatch::add_edges`.
    Added(usize),
}

/// Reference to the staged operation whose effective time is being overridden.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BatchChangeRef {
    NodeAdded(BatchNodeRef),
    NodeRemoved(BatchNodeRef),
    EdgeAdded(BatchEdgeRef),
    EdgeRemoved(BatchEdgeRef),
    NodeAttribute(BatchNodeRef, AttrName),
    EdgeAttribute(BatchEdgeRef, AttrName),
}

/// One mixed write. Operations execute in this order: add nodes, add edges,
/// assign properties, remove explicit edges, then remove nodes and their edges.
#[derive(Debug, Clone, Default)]
pub struct GraphBatch {
    pub add_nodes: usize,
    pub add_edges: Vec<(BatchNodeRef, BatchNodeRef)>,
    pub node_attrs: HashMap<AttrName, Vec<(BatchNodeRef, AttrValue)>>,
    pub edge_attrs: HashMap<AttrName, Vec<(BatchEdgeRef, AttrValue)>>,
    /// Explicit absence, distinct from assigning AttrValue::Null.
    pub unset_node_attrs: HashMap<AttrName, Vec<BatchNodeRef>>,
    pub unset_edge_attrs: HashMap<AttrName, Vec<BatchEdgeRef>>,
    pub remove_edges: Vec<BatchEdgeRef>,
    pub remove_nodes: Vec<BatchNodeRef>,
    /// Explicit initial assignment or rename; ordinary external_id property writes remain forbidden.
    pub node_external_ids: Vec<(BatchNodeRef, String)>,
    pub edge_external_ids: Vec<(BatchEdgeRef, String)>,
    /// Default effective time in signed Unix microseconds; None uses publication time.
    pub effective_at: Option<TimestampMicros>,
    /// Explicit per-operation overrides. Duplicate or unrelated targets reject the batch.
    pub change_times: Vec<(BatchChangeRef, TimestampMicros)>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct BatchResult {
    pub state_id: StateId,
    pub added_nodes: Vec<NodeId>,
    pub added_edges: Vec<EdgeId>,
}

pub(crate) struct PreparedPublication {
    pub batch: PreparedBatch,
    pub metadata: crate::PublicationMetadata,
}

pub(crate) struct PreparedBatch {
    pub add_nodes: usize,
    pub add_edges: Vec<(NodeId, NodeId)>,
    pub node_attrs: HashMap<AttrName, Vec<(NodeId, AttrValue)>>,
    pub edge_attrs: HashMap<AttrName, Vec<(EdgeId, AttrValue)>>,
    pub unset_node_attrs: HashMap<AttrName, Vec<NodeId>>,
    pub unset_edge_attrs: HashMap<AttrName, Vec<EdgeId>>,
    pub remove_edges: Vec<EdgeId>,
    pub remove_nodes: Vec<NodeId>,
    pub identity_bindings: Vec<(String, crate::ExternalEntityRef)>,
    pub effective_at: Option<TimestampMicros>,
    pub change_times: BTreeMap<ChangeKey, TimestampMicros>,
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
    /// Whether any mutation is staged (time metadata alone is not a mutation).
    pub fn has_operations(&self) -> bool {
        self.add_nodes > 0
            || !self.add_edges.is_empty()
            || !self.remove_nodes.is_empty()
            || !self.remove_edges.is_empty()
            || !self.node_external_ids.is_empty()
            || !self.edge_external_ids.is_empty()
            || self.unset_node_attrs.values().any(|v| !v.is_empty())
            || self.unset_edge_attrs.values().any(|v| !v.is_empty())
            || self.node_attrs.values().any(|v| !v.is_empty())
            || self.edge_attrs.values().any(|v| !v.is_empty())
    }

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
        if std::alloc::Layout::array::<NodeId>(self.add_nodes).is_err()
            || std::alloc::Layout::array::<AttrValue>(self.add_nodes).is_err()
        {
            return Err(GraphError::InvalidInput(
                "Batch node allocation exceeds vector capacity".into(),
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

        let mut unset_node_attrs = HashMap::new();
        let mut columns: Vec<_> = self.unset_node_attrs.into_iter().collect();
        columns.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        for (name, refs) in columns {
            if (name == "entity_type" || name == super::graph::EXTERNAL_ID) && !refs.is_empty() {
                errors.push(format!("unset_node_attrs.{name}: system-managed attribute"));
            }
            let mut seen = HashSet::new();
            let assigned: HashSet<_> = node_attrs
                .get(&name)
                .into_iter()
                .flatten()
                .map(|(id, _)| *id)
                .collect();
            let mut ids = Vec::with_capacity(refs.len());
            for (position, reference) in refs.into_iter().enumerate() {
                match node_id(reference, next_node_id, self.add_nodes) {
                    Ok(id) => {
                        if !known_node(id) || seen_removed_nodes.contains(&id) {
                            errors.push(format!("unset_node_attrs.{name}[{position}]: node {id} is not active in the resulting state"));
                        }
                        if !seen.insert(id) || assigned.contains(&id) {
                            errors.push(format!("unset_node_attrs.{name}[{position}]: duplicate/conflicting node {id}"));
                        }
                        ids.push(id);
                    }
                    Err(error) => {
                        errors.push(format!("unset_node_attrs.{name}[{position}]: {error}"))
                    }
                }
            }
            unset_node_attrs.insert(name, ids);
        }

        let mut unset_edge_attrs = HashMap::new();
        let mut columns: Vec<_> = self.unset_edge_attrs.into_iter().collect();
        columns.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        for (name, refs) in columns {
            if (name == super::graph::EXTERNAL_ID) && !refs.is_empty() {
                errors.push(format!("unset_edge_attrs.{name}: system-managed attribute"));
            }
            let mut seen = HashSet::new();
            let assigned: HashSet<_> = edge_attrs
                .get(&name)
                .into_iter()
                .flatten()
                .map(|(id, _)| *id)
                .collect();
            let mut ids = Vec::with_capacity(refs.len());
            for (position, reference) in refs.into_iter().enumerate() {
                match edge_id(reference, next_edge_id, self.add_edges.len()) {
                    Ok(id) => {
                        let endpoints = if (next_edge_id..edge_end).contains(&id) {
                            resolved_edges.get(id - next_edge_id).copied()
                        } else {
                            edge_endpoints(id)
                        };
                        if !(known_edge(id)
                            && !seen_removed_edges.contains(&id)
                            && endpoints.is_some_and(|(a, b)| {
                                !seen_removed_nodes.contains(&a) && !seen_removed_nodes.contains(&b)
                            }))
                        {
                            errors.push(format!("unset_edge_attrs.{name}[{position}]: edge {id} is not active in the resulting state"));
                        }
                        if !seen.insert(id) || assigned.contains(&id) {
                            errors.push(format!("unset_edge_attrs.{name}[{position}]: duplicate/conflicting edge {id}"));
                        }
                        ids.push(id);
                    }
                    Err(error) => {
                        errors.push(format!("unset_edge_attrs.{name}[{position}]: {error}"))
                    }
                }
            }
            unset_edge_attrs.insert(name, ids);
        }

        let mut identity_bindings = Vec::new();
        {
            use crate::ExternalEntityRef;
            let mut seen_entities = HashSet::new();
            let mut seen_names = HashSet::new();
            let mut register =
                |entity: Result<ExternalEntityRef, String>, name: String, context: String| {
                    match entity {
                        Err(error) => errors.push(format!("{context}: {error}")),
                        Ok(entity) => {
                            let retained = match entity {
                                ExternalEntityRef::Node(id) => {
                                    known_node(id) && !seen_removed_nodes.contains(&id)
                                }
                                ExternalEntityRef::Edge(id) => {
                                    let endpoints = if (next_edge_id..edge_end).contains(&id) {
                                        resolved_edges.get(id - next_edge_id).copied()
                                    } else {
                                        edge_endpoints(id)
                                    };
                                    known_edge(id)
                                        && !seen_removed_edges.contains(&id)
                                        && endpoints.is_some_and(|(a, b)| {
                                            !seen_removed_nodes.contains(&a)
                                                && !seen_removed_nodes.contains(&b)
                                        })
                                }
                            };
                            if !retained {
                                errors.push(format!(
                                    "{context}: {entity:?} is not active in resulting state"
                                ));
                            }
                            if name.is_empty() {
                                errors.push(format!("{context}: external ID must not be empty"));
                            }
                            if !seen_entities.insert(entity) {
                                errors.push(format!(
                                    "{context}: duplicate identity assignment for {entity:?}"
                                ));
                            }
                            if !seen_names.insert(name.clone()) {
                                errors.push(format!("{context}: duplicate external ID {name:?}"));
                            }
                            match entity {
                                ExternalEntityRef::Node(id) => node_attrs
                                    .entry(super::graph::EXTERNAL_ID.into())
                                    .or_default()
                                    .push((id, AttrValue::Text(name.clone()))),
                                ExternalEntityRef::Edge(id) => edge_attrs
                                    .entry(super::graph::EXTERNAL_ID.into())
                                    .or_default()
                                    .push((id, AttrValue::Text(name.clone()))),
                            }
                            identity_bindings.push((name, entity));
                        }
                    }
                };
            for (position, (reference, name)) in self.node_external_ids.into_iter().enumerate() {
                register(
                    node_id(reference, next_node_id, self.add_nodes).map(ExternalEntityRef::Node),
                    name,
                    format!("node_external_ids[{position}]"),
                );
            }
            for (position, (reference, name)) in self.edge_external_ids.into_iter().enumerate() {
                register(
                    edge_id(reference, next_edge_id, self.add_edges.len())
                        .map(ExternalEntityRef::Edge),
                    name,
                    format!("edge_external_ids[{position}]"),
                );
            }
        }

        let mut available = HashSet::new();
        available.extend((next_node_id..node_end).map(ChangeKey::NodeAdded));
        available.extend(removed_nodes.iter().copied().map(ChangeKey::NodeRemoved));
        available.extend((next_edge_id..edge_end).map(ChangeKey::EdgeAdded));
        available.extend(removed_edges.iter().copied().map(ChangeKey::EdgeRemoved));
        for (name, values) in &node_attrs {
            available.extend(
                values
                    .iter()
                    .map(|(id, _)| ChangeKey::NodeAttribute(*id, name.clone())),
            );
        }
        for (name, values) in &edge_attrs {
            available.extend(
                values
                    .iter()
                    .map(|(id, _)| ChangeKey::EdgeAttribute(*id, name.clone())),
            );
        }
        for (name, ids) in &unset_node_attrs {
            available.extend(
                ids.iter()
                    .map(|id| ChangeKey::NodeAttribute(*id, name.clone())),
            );
        }
        for (name, ids) in &unset_edge_attrs {
            available.extend(
                ids.iter()
                    .map(|id| ChangeKey::EdgeAttribute(*id, name.clone())),
            );
        }
        let mut change_times = BTreeMap::new();
        for (position, (target, at)) in self.change_times.into_iter().enumerate() {
            let key = match target {
                BatchChangeRef::NodeAdded(reference) => {
                    node_id(reference, next_node_id, self.add_nodes).map(ChangeKey::NodeAdded)
                }
                BatchChangeRef::NodeRemoved(reference) => {
                    node_id(reference, next_node_id, self.add_nodes).map(ChangeKey::NodeRemoved)
                }
                BatchChangeRef::EdgeAdded(reference) => {
                    edge_id(reference, next_edge_id, self.add_edges.len()).map(ChangeKey::EdgeAdded)
                }
                BatchChangeRef::EdgeRemoved(reference) => {
                    edge_id(reference, next_edge_id, self.add_edges.len())
                        .map(ChangeKey::EdgeRemoved)
                }
                BatchChangeRef::NodeAttribute(reference, name) => {
                    node_id(reference, next_node_id, self.add_nodes)
                        .map(|id| ChangeKey::NodeAttribute(id, name))
                }
                BatchChangeRef::EdgeAttribute(reference, name) => {
                    edge_id(reference, next_edge_id, self.add_edges.len())
                        .map(|id| ChangeKey::EdgeAttribute(id, name))
                }
            };
            match key {
                Ok(key) if !available.contains(&key) => errors.push(format!(
                    "change_times[{position}]: no staged operation for {key:?}"
                )),
                Ok(key) => {
                    if change_times.insert(key.clone(), at).is_some() {
                        errors.push(format!(
                            "change_times[{position}]: duplicate time for {key:?}"
                        ));
                    }
                }
                Err(error) => errors.push(format!("change_times[{position}]: {error}")),
            }
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
            && unset_node_attrs.values().all(Vec::is_empty)
            && unset_edge_attrs.values().all(Vec::is_empty)
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
            unset_node_attrs,
            unset_edge_attrs,
            remove_edges: removed_edges,
            remove_nodes: removed_nodes,
            identity_bindings,
            effective_at: self.effective_at,
            change_times,
        })
    }
}
