//! External identity is separate from active membership and internal ID allocation.
//! Bindings are retained for the lifetime of the graph, including old names and
//! inactive entities. The current name is an ordinary versioned column value.

use super::Graph;
use crate::{AttrName, AttrValue, EdgeId, GraphError, GraphResult, NodeId};
use std::collections::HashMap;

/// System-managed property storing an entity's current external identifier.
pub const EXTERNAL_ID: &str = "external_id";

/// Internal identity of an allocated entity, whether currently active or not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ExternalEntityRef {
    Node(NodeId),
    Edge(EdgeId),
}

/// Insert or update a node using a globally unique external identifier.
/// Missing properties are preserved; supplied properties replace their current values.
#[derive(Debug, Clone)]
pub struct NodeUpsert {
    pub external_id: String,
    pub attrs: HashMap<AttrName, AttrValue>,
}

/// Insert or update an edge. Existing endpoints must match; rewiring is separate.
#[derive(Debug, Clone)]
pub struct EdgeUpsert {
    pub external_id: String,
    pub source: NodeId,
    pub target: NodeId,
    pub attrs: HashMap<AttrName, AttrValue>,
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExternalIdentityIndex {
    pub(super) bindings: HashMap<String, ExternalEntityRef>,
}

impl ExternalIdentityIndex {
    pub(super) fn validate_binding(
        &self,
        name: &str,
        entity: ExternalEntityRef,
    ) -> GraphResult<()> {
        if name.is_empty() {
            return Err(GraphError::InvalidInput(
                "External ID must not be empty".into(),
            ));
        }
        if let Some(bound) = self.bindings.get(name) {
            if *bound != entity {
                return Err(GraphError::InvalidInput(format!(
                    "External ID {name:?} is already reserved for {bound:?}"
                )));
            }
        }
        Ok(())
    }
    pub(super) fn reserve_bindings(&mut self, bindings: Vec<(String, ExternalEntityRef)>) {
        self.bindings.extend(bindings);
    }
}

fn check_attrs(
    attrs: &HashMap<AttrName, AttrValue>,
    is_node: bool,
    position: usize,
    errors: &mut Vec<String>,
) {
    for (name, value) in attrs {
        if let Err(error) = super::kinds::validate_kind_property(name, value, is_node) {
            errors.push(format!("{position}.{name}: {error}"));
        }
    }
    if attrs.contains_key(EXTERNAL_ID) {
        errors.push(format!(
            "{position}.external_id: use the explicit identity field"
        ));
    }
    if is_node && attrs.contains_key("entity_type") {
        errors.push(format!(
            "{position}.entity_type: system-managed node property"
        ));
    }
}

fn reject(errors: Vec<String>) -> GraphResult<()> {
    if errors.is_empty() {
        Ok(())
    } else {
        Err(GraphError::InvalidInput(format!(
            "Invalid external-ID operation: {}",
            errors.join("; ")
        )))
    }
}

impl Graph {
    /// Resolve an allocated identity, including inactive entities and old names.
    /// This does not make the entity visible in the current graph state.
    pub fn allocated_entity_for_external_id(&self, external_id: &str) -> Option<ExternalEntityRef> {
        self.identities.bindings.get(external_id).copied()
    }

    fn current_external_name_matches(&self, entity: ExternalEntityRef, name: &str) -> bool {
        let value = match entity {
            ExternalEntityRef::Node(id) => self.get_node_attr(id, &EXTERNAL_ID.into()),
            ExternalEntityRef::Edge(id) => self.get_edge_attr(id, &EXTERNAL_ID.into()),
        };
        value
            .ok()
            .flatten()
            .is_some_and(|value| value.get_text().as_deref() == Some(name))
    }

    fn group_matches(&self, entity: ExternalEntityRef, group: Option<&str>) -> bool {
        let Some(group) = group else {
            return true;
        };
        let value = match entity {
            ExternalEntityRef::Node(id) => self.get_node_attr(id, &"group_id".into()),
            ExternalEntityRef::Edge(id) => self.get_edge_attr(id, &"group_id".into()),
        };
        value
            .ok()
            .flatten()
            .is_some_and(|value| value.get_text().as_deref() == Some(group))
    }

    /// Find an active node by its current external ID, optionally checking group_id.
    /// Groups filter visibility; they do not create separate uniqueness namespaces.
    pub fn node_by_external_id(&self, external_id: &str, group_id: Option<&str>) -> Option<NodeId> {
        let entity = self.allocated_entity_for_external_id(external_id)?;
        match entity {
            ExternalEntityRef::Node(id)
                if self.current_external_name_matches(entity, external_id)
                    && self.group_matches(entity, group_id) =>
            {
                Some(id)
            }
            _ => None,
        }
    }

    /// Find an active edge by its current external ID, optionally checking group_id.
    pub fn edge_by_external_id(&self, external_id: &str, group_id: Option<&str>) -> Option<EdgeId> {
        let entity = self.allocated_entity_for_external_id(external_id)?;
        match entity {
            ExternalEntityRef::Edge(id)
                if self.current_external_name_matches(entity, external_id)
                    && self.group_matches(entity, group_id) =>
            {
                Some(id)
            }
            _ => None,
        }
    }

    /// Explicitly assign or rename a node's external ID. Old names remain reserved.
    pub fn set_node_external_id(&mut self, node: NodeId, external_id: String) -> GraphResult<()> {
        if self.automatic_publication {
            return self
                .writer("groggy")
                .set_node_external_id(node, external_id)
                .map(|_| ());
        }

        if !self.has_node(node) {
            return Err(GraphError::node_not_found(node, "set external ID"));
        }
        let entity = ExternalEntityRef::Node(node);
        self.identities.validate_binding(&external_id, entity)?;
        if self.current_external_name_matches(entity, &external_id) {
            return Ok(());
        }
        self.set_node_attr_internal(
            node,
            EXTERNAL_ID.into(),
            AttrValue::Text(external_id.clone()),
        )?;
        self.identities.bindings.insert(external_id, entity);
        Ok(())
    }

    /// Explicitly assign or rename an edge's external ID. Old names remain reserved.
    pub fn set_edge_external_id(&mut self, edge: EdgeId, external_id: String) -> GraphResult<()> {
        if self.automatic_publication {
            return self
                .writer("groggy")
                .set_edge_external_id(edge, external_id)
                .map(|_| ());
        }

        if !self.has_edge(edge) {
            return Err(GraphError::edge_not_found(edge, "set external ID"));
        }
        let entity = ExternalEntityRef::Edge(edge);
        self.identities.validate_binding(&external_id, entity)?;
        if self.current_external_name_matches(entity, &external_id) {
            return Ok(());
        }
        self.write_edge_attrs(HashMap::from([(
            EXTERNAL_ID.into(),
            vec![(edge, AttrValue::Text(external_id.clone()))],
        )]))?;
        self.identities.bindings.insert(external_id, entity);
        Ok(())
    }

    fn resolve_upsert(
        &self,
        name: &str,
        is_node: bool,
        position: usize,
        seen: &mut HashMap<String, usize>,
        errors: &mut Vec<String>,
    ) -> Option<usize> {
        if name.is_empty() {
            errors.push(format!("{position}: external ID must not be empty"));
        }
        if let Some(first) = seen.get(name) {
            errors.push(format!(
                "{position}: duplicate external ID {name:?} (first at {first})"
            ));
        } else {
            seen.insert(name.to_owned(), position);
        }
        let entity = self.allocated_entity_for_external_id(name)?;
        let id = match (entity, is_node) {
            (ExternalEntityRef::Node(id), true) | (ExternalEntityRef::Edge(id), false) => id,
            _ => {
                errors.push(format!(
                    "{position}: external ID {name:?} belongs to {entity:?}"
                ));
                return None;
            }
        };
        if !self.current_external_name_matches(entity, name) {
            errors.push(format!("{position}: external ID {name:?} is reserved for inactive or renamed {entity:?}; restore or rename explicitly"));
        }
        Some(id)
    }

    fn prepare_node_upserts(
        &self,
        rows: &[NodeUpsert],
        positions: Option<&[usize]>,
    ) -> GraphResult<Vec<Option<NodeId>>> {
        let mut errors = Vec::new();
        let mut seen = HashMap::new();
        let mut resolved = Vec::with_capacity(rows.len());
        for (offset, row) in rows.iter().enumerate() {
            let position = positions
                .and_then(|positions| positions.get(offset))
                .copied()
                .unwrap_or(offset);
            check_attrs(&row.attrs, true, position, &mut errors);
            resolved.push(self.resolve_upsert(
                &row.external_id,
                true,
                position,
                &mut seen,
                &mut errors,
            ));
        }
        reject(errors)?;
        Ok(resolved)
    }

    /// Preflight node upserts without writing; optional positions preserve adapter input offsets.
    pub fn validate_node_upserts(
        &self,
        rows: &[NodeUpsert],
        positions: Option<&[usize]>,
    ) -> GraphResult<()> {
        self.prepare_node_upserts(rows, positions).map(|_| ())
    }

    /// Atomically validate a node upsert list before allocation or property writes.
    /// A repeated identical upsert is a no-op. Like other direct writers, this
    /// records changes for the existing explicit commit API.
    pub fn upsert_nodes(&mut self, rows: Vec<NodeUpsert>) -> GraphResult<Vec<NodeId>> {
        if self.automatic_publication {
            return self.writer("groggy").upsert_nodes(rows);
        }

        let resolved = self.prepare_node_upserts(&rows, None)?;
        let count = resolved.iter().filter(|id| id.is_none()).count();
        let mut added = self.add_nodes(count)?.into_iter();
        let mut columns: HashMap<AttrName, Vec<(NodeId, AttrValue)>> = HashMap::new();
        let mut ids = Vec::with_capacity(rows.len());
        let mut bindings = Vec::new();
        for (row, existing) in rows.into_iter().zip(resolved) {
            let id = match existing {
                Some(id) => id,
                None => added.next().ok_or_else(|| {
                    GraphError::internal("Node allocation count mismatch", "upsert_nodes")
                })?,
            };
            let mut attrs = row.attrs;
            if existing.is_none() {
                attrs.insert(EXTERNAL_ID.into(), AttrValue::Text(row.external_id.clone()));
                bindings.push((row.external_id, ExternalEntityRef::Node(id)));
            }
            for (name, value) in attrs {
                let value = super::kinds::normalize_kind_property(&name, value, true)?;
                if self.get_node_attr(id, &name)?.as_ref() != Some(&value) {
                    columns.entry(name).or_default().push((id, value));
                }
            }
            ids.push(id);
        }
        self.write_node_attrs(columns)?;
        self.identities.bindings.extend(bindings);
        Ok(ids)
    }

    fn prepare_edge_upserts(
        &self,
        rows: &[EdgeUpsert],
        positions: Option<&[usize]>,
    ) -> GraphResult<Vec<Option<EdgeId>>> {
        let mut errors = Vec::new();
        let mut seen = HashMap::new();
        let mut resolved = Vec::with_capacity(rows.len());
        for (offset, row) in rows.iter().enumerate() {
            let position = positions
                .and_then(|positions| positions.get(offset))
                .copied()
                .unwrap_or(offset);
            check_attrs(&row.attrs, false, position, &mut errors);
            let existing =
                self.resolve_upsert(&row.external_id, false, position, &mut seen, &mut errors);
            if !self.has_node(row.source) || !self.has_node(row.target) {
                errors.push(format!(
                    "{position}: endpoints ({}, {}) are not both active",
                    row.source, row.target
                ));
            }
            if let Some(id) = existing {
                let endpoints = self.edge_endpoints(id).ok();
                let matches = endpoints == Some((row.source, row.target))
                    || (self.is_undirected() && endpoints == Some((row.target, row.source)));
                if !matches {
                    errors.push(format!(
                        "{position}: edge {id} endpoints conflict; rewire explicitly"
                    ));
                }
            }
            resolved.push(existing);
        }
        reject(errors)?;
        Ok(resolved)
    }

    /// Preflight edge upserts without writing; optional positions preserve adapter input offsets.
    pub fn validate_edge_upserts(
        &self,
        rows: &[EdgeUpsert],
        positions: Option<&[usize]>,
    ) -> GraphResult<()> {
        self.prepare_edge_upserts(rows, positions).map(|_| ())
    }

    /// Upsert edges without conflating parallel edge identities or changing endpoints.
    pub fn upsert_edges(&mut self, rows: Vec<EdgeUpsert>) -> GraphResult<Vec<EdgeId>> {
        if self.automatic_publication {
            return self.writer("groggy").upsert_edges(rows);
        }

        let resolved = self.prepare_edge_upserts(&rows, None)?;
        let pairs: Vec<_> = rows
            .iter()
            .zip(&resolved)
            .filter(|(_, id)| id.is_none())
            .map(|(row, _)| (row.source, row.target))
            .collect();
        let mut added = self.add_edges(&pairs)?.into_iter();
        let mut columns: HashMap<AttrName, Vec<(EdgeId, AttrValue)>> = HashMap::new();
        let mut ids = Vec::with_capacity(rows.len());
        let mut bindings = Vec::new();
        for (row, existing) in rows.into_iter().zip(resolved) {
            let id = match existing {
                Some(id) => id,
                None => added.next().ok_or_else(|| {
                    GraphError::internal("Edge allocation count mismatch", "upsert_edges")
                })?,
            };
            let mut attrs = row.attrs;
            if existing.is_none() {
                attrs.insert(EXTERNAL_ID.into(), AttrValue::Text(row.external_id.clone()));
                bindings.push((row.external_id, ExternalEntityRef::Edge(id)));
            }
            for (name, value) in attrs {
                let value = super::kinds::normalize_kind_property(&name, value, false)?;
                if self.get_edge_attr(id, &name)?.as_ref() != Some(&value) {
                    columns.entry(name).or_default().push((id, value));
                }
            }
            ids.push(id);
        }
        self.write_edge_attrs(columns)?;
        self.identities.bindings.extend(bindings);
        Ok(ids)
    }

    pub(super) fn snapshot_identity_bindings(
        &self,
        snapshot: &crate::state::state::GraphSnapshot,
    ) -> GraphResult<Vec<(String, ExternalEntityRef)>> {
        let mut staged = HashMap::new();
        let mut errors = Vec::new();
        for id in &snapshot.active_nodes {
            if let Some(attrs) = snapshot.node_attributes.get(id) {
                self.stage_snapshot_identity(
                    ExternalEntityRef::Node(*id),
                    attrs,
                    &mut staged,
                    &mut errors,
                );
            }
        }
        for (id, attrs) in &snapshot.edge_attributes {
            if snapshot.edges.contains_key(id) {
                self.stage_snapshot_identity(
                    ExternalEntityRef::Edge(*id),
                    attrs,
                    &mut staged,
                    &mut errors,
                );
            }
        }
        reject(errors)?;
        Ok(staged.into_iter().collect())
    }

    fn stage_snapshot_identity(
        &self,
        entity: ExternalEntityRef,
        attrs: &HashMap<AttrName, AttrValue>,
        staged: &mut HashMap<String, ExternalEntityRef>,
        errors: &mut Vec<String>,
    ) {
        let Some(value) = attrs.get(EXTERNAL_ID) else {
            return;
        };
        let Some(name) = value.get_text() else {
            errors.push(format!("{entity:?}: external ID must be text"));
            return;
        };
        if let Err(error) = self.identities.validate_binding(&name, entity) {
            errors.push(error.to_string());
        }
        if let Some(other) = staged.insert(name.clone(), entity) {
            if other != entity {
                errors.push(format!(
                    "External ID {name:?} appears on both {other:?} and {entity:?}"
                ));
            }
        }
    }

    pub(super) fn retain_snapshot_identity_bindings(
        &mut self,
        bindings: Vec<(String, ExternalEntityRef)>,
    ) {
        self.identities.bindings.extend(bindings);
    }

    /// Preflight new IDs for the legacy uid_key property mapping.
    /// Ordinary properties remain editable; use managed external IDs for a lasting constraint.
    pub fn validate_new_text_ids(&self, attr: &str, ids: &[String]) -> GraphResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut existing: HashMap<String, Vec<NodeId>> = HashMap::new();
        let attr = attr.to_owned();
        for node in self.node_ids() {
            if let Some(value) = self.get_node_attr(node, &attr)? {
                if let Some(text) = value.get_text() {
                    existing.entry(text).or_default().push(node);
                }
            }
        }
        let mut seen = HashMap::new();
        let mut errors = Vec::new();
        for (position, id) in ids.iter().enumerate() {
            if let Some(nodes) = existing.get(id) {
                let mut nodes = nodes.clone();
                nodes.sort_unstable();
                errors.push(format!(
                    "{position}: {attr}={id:?} already identifies nodes {nodes:?}"
                ));
            }
            if let Some(first) = seen.get(id) {
                errors.push(format!(
                    "{position}: duplicate {attr}={id:?} (first at {first})"
                ));
            } else {
                seen.insert(id, position);
            }
        }
        reject(errors)
    }

    /// Resolve a text property only if it identifies one active node unambiguously.
    /// Legacy uid_key operations use this scan; external-ID lookup uses its index.
    pub fn node_by_unique_text_attr(&self, attr: &str, value: &str) -> GraphResult<Option<NodeId>> {
        let mut matches = Vec::new();
        for node in self.node_ids() {
            if self
                .get_node_attr(node, &attr.to_owned())?
                .is_some_and(|v| v.get_text().as_deref() == Some(value))
            {
                matches.push(node);
            }
        }
        matches.sort_unstable();
        if matches.len() > 1 {
            return Err(GraphError::InvalidInput(format!(
                "Ambiguous {attr}={value:?}: nodes {matches:?}"
            )));
        }
        Ok(matches.first().copied())
    }
}

impl Graph {
    /// Prepare published upserts without allocating IDs or touching reservations.
    pub(crate) fn published_node_upserts(
        &self,
        rows: Vec<NodeUpsert>,
    ) -> GraphResult<(crate::GraphBatch, Vec<NodeId>)> {
        use crate::{BatchNodeRef, GraphBatch};
        let resolved = self.prepare_node_upserts(&rows, None)?;
        let first = self.pool.borrow().statistics().node_count;
        let mut batch = GraphBatch::default();
        let mut ids = Vec::with_capacity(rows.len());
        for (row, existing) in rows.into_iter().zip(resolved) {
            let (reference, id) = match existing {
                Some(id) => (BatchNodeRef::Existing(id), id),
                None => {
                    let position = batch.add_nodes;
                    let id = first.checked_add(position).ok_or_else(|| {
                        GraphError::InvalidInput("Node ID allocation overflow".into())
                    })?;
                    batch.add_nodes += 1;
                    let reference = BatchNodeRef::Added(position);
                    batch.node_external_ids.push((reference, row.external_id));
                    (reference, id)
                }
            };
            for (name, value) in row.attrs {
                let value = super::kinds::normalize_kind_property(&name, value, true)?;
                if existing.is_none() || self.get_node_attr(id, &name)?.as_ref() != Some(&value) {
                    batch
                        .node_attrs
                        .entry(name)
                        .or_default()
                        .push((reference, value));
                }
            }
            ids.push(id);
        }
        Ok((batch, ids))
    }

    pub(crate) fn published_edge_upserts(
        &self,
        rows: Vec<EdgeUpsert>,
    ) -> GraphResult<(crate::GraphBatch, Vec<EdgeId>)> {
        use crate::{BatchEdgeRef, BatchNodeRef, GraphBatch};
        let resolved = self.prepare_edge_upserts(&rows, None)?;
        let first = self.pool.borrow().statistics().edge_count;
        let mut batch = GraphBatch::default();
        let mut ids = Vec::with_capacity(rows.len());
        for (row, existing) in rows.into_iter().zip(resolved) {
            let (reference, id) = match existing {
                Some(id) => (BatchEdgeRef::Existing(id), id),
                None => {
                    let position = batch.add_edges.len();
                    let id = first.checked_add(position).ok_or_else(|| {
                        GraphError::InvalidInput("Edge ID allocation overflow".into())
                    })?;
                    batch.add_edges.push((
                        BatchNodeRef::Existing(row.source),
                        BatchNodeRef::Existing(row.target),
                    ));
                    let reference = BatchEdgeRef::Added(position);
                    batch.edge_external_ids.push((reference, row.external_id));
                    (reference, id)
                }
            };
            for (name, value) in row.attrs {
                let value = super::kinds::normalize_kind_property(&name, value, false)?;
                if existing.is_none() || self.get_edge_attr(id, &name)?.as_ref() != Some(&value) {
                    batch
                        .edge_attrs
                        .entry(name)
                        .or_default()
                        .push((reference, value));
                }
            }
            ids.push(id);
        }
        Ok((batch, ids))
    }
}
