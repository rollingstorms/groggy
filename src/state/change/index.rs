//! Borrowed physical change references, explicitly distinct from logical values.
use super::ChangeSet;
use crate::state::delta::{ColumnIndexDelta, DeltaObject};
use crate::storage::pool::GraphPool;
use crate::{AttrName, EdgeEndpointChange, EdgeId, GraphError, GraphResult, NodeId};
use std::collections::HashMap;

/// Borrowed pending index references. Resolving requires the matching retained pool.
#[derive(Debug, Clone, Copy)]
pub struct IndexChangeView<'a> {
    /// Added node identities.
    pub nodes_added: &'a [NodeId],
    /// Removed node identities.
    pub nodes_removed: &'a [NodeId],
    /// Added edge identities and endpoints.
    pub edges_added: &'a [(EdgeId, NodeId, NodeId)],
    /// Removed edge identities.
    pub edges_removed: &'a [EdgeId],
    /// Coalesced ordered endpoint changes keyed by stable edge ID.
    pub edge_endpoint_changes: &'a HashMap<EdgeId, EdgeEndpointChange>,
    /// Node property before/after physical column references, not values.
    pub node_attr_changes: &'a [(NodeId, AttrName, Option<usize>, Option<usize>)],
    /// Edge property before/after physical column references, not values.
    pub edge_attr_changes: &'a [(EdgeId, AttrName, Option<usize>, Option<usize>)],
}
impl IndexChangeView<'_> {
    /// Convert a topology-only record without inventing property payloads.
    pub fn without_values(&self) -> GraphResult<ChangeSet> {
        if !self.node_attr_changes.is_empty() || !self.edge_attr_changes.is_empty() {
            return Err(GraphError::InvalidInput("Logical property changes require the retained pool; use create_changeset_resolved(pool)".into()));
        }
        Ok(self.topology())
    }
    fn topology(&self) -> ChangeSet {
        ChangeSet {
            nodes_added: self.nodes_added.to_vec(),
            nodes_removed: self.nodes_removed.to_vec(),
            edges_added: self.edges_added.to_vec(),
            edges_removed: self.edges_removed.to_vec(),
            edge_endpoint_changes: self.edge_endpoint_changes.values().copied().collect(),
            ..ChangeSet::new()
        }
    }
    /// Preserve physical references in the explicitly indexed delta representation.
    pub fn to_index_delta(&self) -> GraphResult<DeltaObject> {
        let mut nodes = HashMap::new();
        let mut edges = HashMap::new();
        for (rows, output) in [
            (self.node_attr_changes, &mut nodes),
            (self.edge_attr_changes, &mut edges),
        ] {
            for (id, name, old, new) in rows {
                output
                    .entry(name.clone())
                    .or_insert_with(ColumnIndexDelta::new)
                    .add_index_change(*id, *old, *new);
            }
        }
        Ok(DeltaObject::new_with_indices(
            nodes,
            edges,
            self.nodes_added.to_vec(),
            self.nodes_removed.to_vec(),
            self.edges_added.to_vec(),
            self.edges_removed.to_vec(),
        )?
        .with_endpoint_changes(self.edge_endpoint_changes.values().copied().collect()))
    }
    /// Estimate occupied pending record bytes, excluding spare capacity/hash buckets.
    pub fn estimated_bytes(&self) -> usize {
        std::mem::size_of_val(self.nodes_added)
            + std::mem::size_of_val(self.nodes_removed)
            + std::mem::size_of_val(self.edges_added)
            + std::mem::size_of_val(self.edges_removed)
            + self.edge_endpoint_changes.len() * std::mem::size_of::<EdgeEndpointChange>()
            + std::mem::size_of_val(self.node_attr_changes)
            + std::mem::size_of_val(self.edge_attr_changes)
            + self
                .node_attr_changes
                .iter()
                .chain(self.edge_attr_changes)
                .map(|(_, name, _, _)| name.len())
                .sum::<usize>()
    }
    /// Resolve actual typed values, preserving first-before/last-after per property.
    pub fn resolve(&self, pool: &GraphPool) -> GraphResult<ChangeSet> {
        let mut changes = ChangeSet {
            nodes_added: self.nodes_added.to_vec(),
            nodes_removed: self.nodes_removed.to_vec(),
            edges_added: self.edges_added.to_vec(),
            edges_removed: self.edges_removed.to_vec(),
            edge_endpoint_changes: self.edge_endpoint_changes.values().copied().collect(),
            node_attr_changes: Vec::new(),
            edge_attr_changes: Vec::new(),
        };

        let mut node_positions: HashMap<(NodeId, AttrName), usize> = HashMap::new();
        for (node_id, attr, old_index, new_index) in self.node_attr_changes {
            let old_value = old_index
                .map(|index| {
                    pool.get_attr_by_index(attr, index, true)
                        .cloned()
                        .ok_or_else(|| GraphError::InvalidInput(format!(
                            "Missing old node attribute value for node {node_id}, {attr}, index {index}"
                        )))
                })
                .transpose()?;
            let new_value = new_index
                .map(|index| {
                    pool.get_attr_by_index(attr, index, true)
                        .cloned()
                        .ok_or_else(|| {
                            GraphError::InvalidInput(format!(
                    "Missing new node attribute value for node {node_id}, {attr}, index {index}"
                ))
                        })
                })
                .transpose()?;
            if let Some(&position) = node_positions.get(&(*node_id, attr.clone())) {
                changes.node_attr_changes[position].3 = new_value;
            } else {
                node_positions.insert((*node_id, attr.clone()), changes.node_attr_changes.len());
                changes
                    .node_attr_changes
                    .push((*node_id, attr.clone(), old_value, new_value));
            }
        }

        let mut edge_positions: HashMap<(EdgeId, AttrName), usize> = HashMap::new();
        for (edge_id, attr, old_index, new_index) in self.edge_attr_changes {
            let old_value = old_index
                .map(|index| {
                    pool.get_attr_by_index(attr, index, false)
                        .cloned()
                        .ok_or_else(|| GraphError::InvalidInput(format!(
                            "Missing old edge attribute value for edge {edge_id}, {attr}, index {index}"
                        )))
                })
                .transpose()?;
            let new_value = new_index
                .map(|index| {
                    pool.get_attr_by_index(attr, index, false)
                        .cloned()
                        .ok_or_else(|| {
                            GraphError::InvalidInput(format!(
                    "Missing new edge attribute value for edge {edge_id}, {attr}, index {index}"
                ))
                        })
                })
                .transpose()?;
            if let Some(&position) = edge_positions.get(&(*edge_id, attr.clone())) {
                changes.edge_attr_changes[position].3 = new_value;
            } else {
                edge_positions.insert((*edge_id, attr.clone()), changes.edge_attr_changes.len());
                changes
                    .edge_attr_changes
                    .push((*edge_id, attr.clone(), old_value, new_value));
            }
        }

        Ok(changes)
    }
}
