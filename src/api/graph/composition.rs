//! Accept an entirely staged meta-node composition, including its retained group.
use super::*;
use crate::{BatchEdgeRef, BatchNodeRef};

/// Fully evaluated user attributes and boundary edges, without allocated IDs.
pub(crate) struct MetaComposition {
    pub nodes: HashSet<NodeId>,
    pub edges: HashSet<EdgeId>,
    pub kind: String,
    pub attrs: HashMap<AttrName, AttrValue>,
    pub boundary: Vec<(BatchNodeRef, BatchNodeRef, HashMap<AttrName, AttrValue>)>,
    pub remove_originals: bool,
}
impl Graph {
    pub(crate) fn apply_meta_composition(
        &mut self,
        composition: MetaComposition,
    ) -> GraphResult<NodeId> {
        self.apply_meta_compositions(vec![composition])
            .map(|nodes| nodes[0])
    }

    /// Prepare every effect against one source, then mutate and publish once.
    pub(crate) fn apply_meta_compositions(
        &mut self,
        compositions: Vec<MetaComposition>,
    ) -> GraphResult<Vec<NodeId>> {
        if compositions.is_empty() {
            return Ok(Vec::new());
        }
        if self.automatic_publication {
            self.validate_write_head(self.current_state_id())?;
        }
        let mut removed = HashSet::new();
        for composition in &compositions {
            self.validate_new_node_attrs(std::slice::from_ref(&composition.attrs))?;
            if composition.attrs.contains_key("contains_subgraph") {
                return Err(GraphError::InvalidInput(
                    "contains_subgraph is managed by meta composition".into(),
                ));
            }
            if composition.remove_originals {
                for &node in &composition.nodes {
                    if !removed.insert(node) {
                        return Err(GraphError::InvalidInput(
                            "Destructive composition selections must not overlap".into(),
                        ));
                    }
                }
            }
        }
        // A collection uses the original source for every plan. Cross-selection
        // destructive rewiring needs an explicit policy; never silently create
        // edges whose original endpoints this same operation removes.
        for composition in &compositions {
            for (source, target, _) in &composition.boundary {
                for endpoint in [source, target] {
                    if matches!(endpoint, BatchNodeRef::Existing(id) if removed.contains(id)) {
                        return Err(GraphError::InvalidInput(
                            "Composition boundary targets a node removed by another plan".into(),
                        ));
                    }
                }
            }
        }
        let source = self.pin_current_state()?;
        let counts = self.pool.borrow().statistics();
        let mut groups = Vec::with_capacity(compositions.len());
        let mut batch = GraphBatch {
            add_nodes: compositions.len(),
            ..Default::default()
        };
        for (position, composition) in compositions.into_iter().enumerate() {
            let view = crate::GraphView::from_membership(
                source.clone(),
                composition.nodes,
                composition.edges,
            )?;
            let stored = self
                .pool
                .borrow_mut()
                .prepare_frozen_group(view, composition.kind)?;
            for (name, value) in composition.attrs {
                batch
                    .node_attrs
                    .entry(name)
                    .or_default()
                    .push((BatchNodeRef::Added(position), value));
            }
            for (from, to, attrs) in composition.boundary {
                let resolve = |reference| match reference {
                    BatchNodeRef::Added(0) => BatchNodeRef::Added(position),
                    other => other,
                };
                let edge = batch.add_edges.len();
                batch.add_edges.push((resolve(from), resolve(to)));
                for (name, value) in attrs {
                    batch
                        .edge_attrs
                        .entry(name)
                        .or_default()
                        .push((BatchEdgeRef::Added(edge), value));
                }
            }
            groups.push(stored);
        }
        self.pool
            .borrow_mut()
            .validate_and_reserve_group_batch(&groups)?;
        let mut removed: Vec<_> = removed.into_iter().collect();
        removed.sort_unstable();
        batch.remove_nodes = removed.into_iter().map(BatchNodeRef::Existing).collect();
        let managed = |prepared: &mut crate::api::batch::PreparedBatch| {
            for (position, group) in groups.iter().enumerate() {
                let node = counts.node_count + position;
                prepared
                    .node_attrs
                    .entry("contains_subgraph".into())
                    .or_default()
                    .push((node, AttrValue::SubgraphRef(group.id)));
                prepared
                    .node_attrs
                    .entry("entity_type".into())
                    .or_default()
                    .push((node, AttrValue::Text("meta".into())));
            }
        };
        if self.automatic_publication {
            let mut prepared = self.prepare_publication(batch, self.current_state_id(), None)?;
            managed(&mut prepared.batch);
            for group in groups {
                self.pool.borrow_mut().install_stored_subgraph(group);
            }
            Ok(self
                .execute_publication(prepared, "compose_meta_nodes".into(), "groggy".into())?
                .added_nodes)
        } else {
            let mut prepared = batch.prepare(
                |id| self.has_node(id),
                |id| self.has_edge(id),
                |id| self.pool.borrow().get_edge_endpoints(id),
                counts.node_count,
                counts.edge_count,
            )?;
            managed(&mut prepared);
            for group in groups {
                self.pool.borrow_mut().install_stored_subgraph(group);
            }
            self.execute_prepared_mutation(prepared)
                .map(|(nodes, _)| nodes)
        }
    }
}

impl Graph {
    /// Retain an explicit frozen source selection. Changed source state or values
    /// produce a new group reference; no graph delta is published by retaining it.
    pub fn retain_group(
        &mut self,
        source: &crate::GraphView,
        kind: String,
    ) -> GraphResult<SubgraphId> {
        let prepared = self
            .pool
            .borrow_mut()
            .prepare_frozen_group(source.clone(), kind)?;
        let id = prepared.id;
        self.pool.borrow_mut().install_stored_subgraph(prepared);
        self.space.invalidate_current_read();
        Ok(id)
    }
    /// Install a revision prepared independently of the mutable graph owner.
    pub fn retain_group_revision(
        &mut self,
        group: crate::storage::retained_group::RetainedGroup,
    ) -> GraphResult<SubgraphId> {
        let prepared = self.pool.borrow_mut().prepare_group_revision(group)?;
        let id = prepared.id;
        self.pool.borrow_mut().install_stored_subgraph(prepared);
        self.space.invalidate_current_read();
        Ok(id)
    }
    /// Upgrade a membership-only group using a caller-chosen source state.
    pub fn pin_legacy_group(
        &mut self,
        id: SubgraphId,
        source: &crate::GraphView,
    ) -> GraphResult<SubgraphId> {
        let (nodes, edges, kind) = self.pool.borrow().get_subgraph(id)?;
        let view = crate::GraphView::from_membership(source.read_state().clone(), nodes, edges)?;
        self.retain_group(&view, kind)
    }
}
