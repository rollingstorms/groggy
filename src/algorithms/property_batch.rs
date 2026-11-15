//! Property staging and terminal removals for explicitly atomic builder execution.
use super::Context;
use crate::{
    api::graph::WriteCoordinate,
    api::graph::{kinds, Graph},
    AttrName, AttrValue, BatchChangeRef, BatchEdgeRef, BatchNodeRef, EdgeId, GraphBatch,
    GraphResult, NodeId,
};
use crate::{subgraphs::Subgraph, traits::SubgraphOperations};
use anyhow::{anyhow, Result};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug)]
pub(crate) struct PropertyBatch {
    coordinate: WriteCoordinate,
    allocation_base: Option<(NodeId, EdgeId)>,
    add_nodes: usize,
    add_edges: Vec<(BatchNodeRef, BatchNodeRef)>,
    nodes: BTreeMap<AttrName, BTreeMap<usize, AttrValue>>,
    edges: BTreeMap<AttrName, BTreeMap<usize, AttrValue>>,
    remove_nodes: Vec<BatchNodeRef>,
    remove_edges: Vec<BatchEdgeRef>,
    edge_endpoints: Vec<(EdgeId, BatchNodeRef, BatchNodeRef)>,
    change_times: Vec<(BatchChangeRef, crate::TimestampMicros)>,
}
impl PropertyBatch {
    /// Materialize exactly the selected pending working state as a detached,
    /// read-only value. It has no accepted write provenance or history claim.
    /// This copies selected values, never clones a writable Graph/history/pool.
    pub(crate) fn capture(&self, subgraph: &Subgraph) -> GraphResult<crate::GraphView> {
        subgraph
            .graph()
            .borrow()
            .validate_write_coordinate(&self.coordinate)?;
        let base = subgraph.pin_view()?;
        let mut revisions = HashMap::with_capacity(self.edge_endpoints.len());
        for &(edge, source, target) in &self.edge_endpoints {
            let (BatchNodeRef::Existing(source), BatchNodeRef::Existing(target)) = (source, target)
            else {
                return Err(crate::GraphError::InvalidInput(
                    "Unsupported staged added endpoint".into(),
                ));
            };
            if revisions.insert(edge, (source, target)).is_some() {
                return Err(crate::GraphError::InvalidInput(format!(
                    "Duplicate staged endpoint revision for edge {edge}"
                )));
            }
        }
        let mut removed_nodes = HashSet::with_capacity(self.remove_nodes.len());
        for reference in &self.remove_nodes {
            let BatchNodeRef::Existing(id) = reference else {
                return Err(crate::GraphError::InvalidInput(
                    "Unsupported staged added node removal".into(),
                ));
            };
            removed_nodes.insert(*id);
        }
        let mut removed_edges = HashSet::with_capacity(self.remove_edges.len());
        for reference in &self.remove_edges {
            let BatchEdgeRef::Existing(id) = reference else {
                return Err(crate::GraphError::InvalidInput(
                    "Unsupported staged added edge removal".into(),
                ));
            };
            removed_edges.insert(*id);
        }
        let (added_nodes, added_edges) = self.added_ids();
        let nodes: HashSet<_> = base
            .nodes()
            .iter()
            .copied()
            .chain(added_nodes.iter().copied())
            .filter(|id| !removed_nodes.contains(id))
            .collect();
        // Revisions precede node cascades in the canonical batch model. An edge
        // moved off a removed endpoint must survive in this provisional value.
        let original = base.read_state().snapshot();
        let mut snapshot = crate::state::state::GraphSnapshot::empty_with_type(
            original.state_id,
            original.graph_type,
        );
        snapshot.active_nodes = nodes.iter().copied().collect();
        snapshot.active_nodes.sort_unstable();
        for id in &snapshot.active_nodes {
            if let Some(attrs) = original.node_attributes.get(id) {
                snapshot.node_attributes.insert(*id, attrs.clone());
            }
        }
        for &id in &added_nodes {
            if nodes.contains(&id) {
                snapshot
                    .node_attributes
                    .entry(id)
                    .or_default()
                    .insert("entity_type".into(), AttrValue::Text("base".into()));
            }
        }
        for (&id, &(source, target)) in added_edges.iter().zip(&self.add_edges) {
            let resolve = |reference| match reference {
                BatchNodeRef::Existing(id) => id,
                BatchNodeRef::Added(position) => added_nodes[position],
            };
            let (source, target) = (resolve(source), resolve(target));
            if !removed_edges.contains(&id) && nodes.contains(&source) && nodes.contains(&target) {
                snapshot.edges.insert(id, (source, target));
            }
        }
        for &id in base.edges() {
            let (source, target) = revisions
                .get(&id)
                .copied()
                .map(Ok)
                .unwrap_or_else(|| base.edge_endpoints(id))?;
            if !removed_edges.contains(&id)
                && !removed_nodes.contains(&source)
                && !removed_nodes.contains(&target)
            {
                snapshot.edges.insert(id, (source, target));
                if let Some(attrs) = original.edge_attributes.get(&id) {
                    snapshot.edge_attributes.insert(id, attrs.clone());
                }
            }
        }
        for (name, rows) in &self.nodes {
            for (&id, value) in rows {
                if !nodes.contains(&id) {
                    return Err(crate::GraphError::InvalidInput(format!(
                        "Staged node property target {id} is outside the resulting selection"
                    )));
                }
                snapshot
                    .node_attributes
                    .entry(id)
                    .or_default()
                    .insert(name.clone(), value.clone());
            }
        }
        for (name, rows) in &self.edges {
            for (&id, value) in rows {
                if !snapshot.edges.contains_key(&id) {
                    return Err(crate::GraphError::InvalidInput(format!(
                        "Staged edge property target {id} is outside the resulting selection"
                    )));
                }
                snapshot
                    .edge_attributes
                    .entry(id)
                    .or_default()
                    .insert(name.clone(), value.clone());
            }
        }
        let state = crate::state::read_state::PinnedReadState::from_snapshot(snapshot)?
            .with_groups(base.read_state().groups().clone());
        Ok(crate::GraphView::all(state))
    }

    pub fn new(coordinate: WriteCoordinate) -> Self {
        Self {
            coordinate,
            allocation_base: None,
            add_nodes: 0,
            add_edges: Vec::new(),
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            remove_nodes: Vec::new(),
            remove_edges: Vec::new(),
            edge_endpoints: Vec::new(),
            change_times: Vec::new(),
        }
    }
    pub fn get(&self, node: bool, id: usize, name: &AttrName) -> Option<&AttrValue> {
        (if node { &self.nodes } else { &self.edges })
            .get(name)?
            .get(&id)
    }
    pub fn stage(
        &mut self,
        node: bool,
        attrs: HashMap<AttrName, Vec<(usize, AttrValue)>>,
    ) -> GraphResult<()> {
        // Caller validates the entire input before any staging. Normalization is
        // identical to ordinary graph writes, including empty labels vectors.
        let mut attrs = attrs;
        for (name, rows) in &mut attrs {
            for (_, value) in rows {
                kinds::normalize_kind_property_in_place(name, value, node)?;
            }
        }
        let columns = if node {
            &mut self.nodes
        } else {
            &mut self.edges
        };
        for (name, rows) in attrs {
            if !rows.is_empty() {
                columns.entry(name).or_default().extend(rows);
            }
        }
        Ok(())
    }
    pub(crate) fn stage_removals(
        &mut self,
        nodes: bool,
        ids: Vec<usize>,
        effective_at: Option<crate::TimestampMicros>,
    ) {
        for id in ids {
            let target = if nodes {
                let reference = BatchNodeRef::Existing(id);
                self.remove_nodes.push(reference);
                BatchChangeRef::NodeRemoved(reference)
            } else {
                let reference = BatchEdgeRef::Existing(id);
                self.remove_edges.push(reference);
                BatchChangeRef::EdgeRemoved(reference)
            };
            if let Some(at) = effective_at {
                self.change_times.push((target, at));
            }
        }
    }
    pub(crate) fn stage_endpoint_revisions(
        &mut self,
        rows: Vec<(EdgeId, NodeId, NodeId)>,
        effective_at: Option<crate::TimestampMicros>,
    ) {
        for (edge, source, target) in rows {
            self.edge_endpoints.push((
                edge,
                BatchNodeRef::Existing(source),
                BatchNodeRef::Existing(target),
            ));
            if let Some(at) = effective_at {
                self.change_times.push((
                    BatchChangeRef::EdgeEndpoints(BatchEdgeRef::Existing(edge)),
                    at,
                ));
            }
        }
    }
    fn allocation_base(&mut self, graph: &Graph) -> GraphResult<(NodeId, EdgeId)> {
        graph.validate_write_coordinate(&self.coordinate)?;
        let stats = graph.pool().statistics();
        let current = (stats.node_count, stats.edge_count);
        if self.allocation_base.is_some_and(|base| base != current) {
            return Err(crate::GraphError::InvalidInput(
                "Staged allocation coordinates changed".into(),
            ));
        }
        self.allocation_base = Some(current);
        Ok(current)
    }

    pub(crate) fn added_ids(&self) -> (Vec<NodeId>, Vec<EdgeId>) {
        let Some((nodes, edges)) = self.allocation_base else {
            return (Vec::new(), Vec::new());
        };
        (
            (nodes..nodes + self.add_nodes).collect(),
            (edges..edges + self.add_edges.len()).collect(),
        )
    }

    pub(crate) fn stage_nodes(
        &mut self,
        graph: &Graph,
        count: usize,
        attrs: &HashMap<AttrName, AttrValue>,
        at: Option<crate::TimestampMicros>,
    ) -> GraphResult<Vec<NodeId>> {
        let (base, _) = self.allocation_base(graph)?;
        let total = self
            .add_nodes
            .checked_add(count)
            .filter(|&n| base.checked_add(n).is_some())
            .ok_or_else(|| crate::GraphError::InvalidInput("Staged node IDs overflow".into()))?;
        let mut ids = Vec::new();
        ids.try_reserve_exact(count)
            .map_err(|e| crate::GraphError::InvalidInput(e.to_string()))?;
        ids.extend(base + self.add_nodes..base + total);
        let first = self.add_nodes;
        self.add_nodes = total;
        self.stage(
            true,
            attrs
                .iter()
                .map(|(name, value)| {
                    (
                        name.clone(),
                        ids.iter().map(|&id| (id, value.clone())).collect(),
                    )
                })
                .collect(),
        )?;
        if let Some(at) = at {
            self.change_times
                .extend((first..total).map(|position| {
                    (BatchChangeRef::NodeAdded(BatchNodeRef::Added(position)), at)
                }));
        }
        Ok(ids)
    }

    pub(crate) fn stage_edges(
        &mut self,
        graph: &Graph,
        selected: &Subgraph,
        edges: &[(BatchNodeRef, BatchNodeRef)],
        attrs: &HashMap<AttrName, AttrValue>,
        at: Option<crate::TimestampMicros>,
    ) -> GraphResult<Vec<EdgeId>> {
        let (_, base) = self.allocation_base(graph)?;
        let total = self
            .add_edges
            .len()
            .checked_add(edges.len())
            .filter(|&n| base.checked_add(n).is_some())
            .ok_or_else(|| crate::GraphError::InvalidInput("Staged edge IDs overflow".into()))?;
        for &(source, target) in edges {
            for reference in [source, target] {
                let valid = match reference {
                    BatchNodeRef::Existing(id) => {
                        selected.nodes().contains(&id) && graph.has_node(id)
                    }
                    BatchNodeRef::Added(position) => position < self.add_nodes,
                };
                if !valid {
                    return Err(crate::GraphError::InvalidInput(
                        "Added edge endpoint is outside the selected or staged nodes".into(),
                    ));
                }
            }
        }
        let first = self.add_edges.len();
        let ids: Vec<_> = (base + first..base + total).collect();
        self.add_edges.extend_from_slice(edges);
        self.stage(
            false,
            attrs
                .iter()
                .map(|(name, value)| {
                    (
                        name.clone(),
                        ids.iter().map(|&id| (id, value.clone())).collect(),
                    )
                })
                .collect(),
        )?;
        if let Some(at) = at {
            self.change_times
                .extend((first..total).map(|position| {
                    (BatchChangeRef::EdgeAdded(BatchEdgeRef::Added(position)), at)
                }));
        }
        Ok(ids)
    }

    pub fn accept(self, graph: &mut Graph) -> GraphResult<()> {
        graph.validate_write_coordinate(&self.coordinate)?;
        if self.add_nodes == 0
            && self.add_edges.is_empty()
            && self.nodes.is_empty()
            && self.edges.is_empty()
            && self.remove_nodes.is_empty()
            && self.remove_edges.is_empty()
            && self.edge_endpoints.is_empty()
        {
            return Ok(());
        }
        let batch = GraphBatch {
            add_nodes: self.add_nodes,
            add_edges: self.add_edges,
            remove_nodes: self.remove_nodes,
            remove_edges: self.remove_edges,
            change_times: self.change_times,
            edge_endpoints: self.edge_endpoints,
            node_attrs: self
                .nodes
                .into_iter()
                .map(|(name, rows)| {
                    (
                        name,
                        rows.into_iter()
                            .map(|(id, value)| (BatchNodeRef::Existing(id), value))
                            .collect(),
                    )
                })
                .collect(),
            edge_attrs: self
                .edges
                .into_iter()
                .map(|(name, rows)| {
                    (
                        name,
                        rows.into_iter()
                            .map(|(id, value)| (BatchEdgeRef::Existing(id), value))
                            .collect(),
                    )
                })
                .collect(),
            ..Default::default()
        };
        graph.apply_batch(
            batch,
            self.coordinate.state_id(),
            "atomic builder effects".into(),
            "groggy".into(),
        )?;
        Ok(())
    }
}

impl Context {
    /// Read a node property, including prior accepted attachments in this run.
    pub(crate) fn node_attribute(
        &self,
        subgraph: &Subgraph,
        id: NodeId,
        name: &AttrName,
    ) -> crate::GraphResult<Option<AttrValue>> {
        if !subgraph.contains_node(id) {
            return Err(crate::GraphError::InvalidInput(format!(
                "Node {id} is outside the subgraph selection"
            )));
        }
        if let Some(value) = self
            .property_batch
            .as_ref()
            .and_then(|batch| batch.get(true, id, name))
        {
            if !subgraph.graph().borrow().has_node(id) {
                return Err(crate::GraphError::InvalidInput(format!(
                    "Node {id} is inactive"
                )));
            }
            return Ok(Some(value.clone()));
        }
        subgraph.get_node_attribute(id, name)
    }
    /// Read an edge property through the same property-only overlay.
    pub(crate) fn edge_attribute(
        &self,
        subgraph: &Subgraph,
        id: EdgeId,
        name: &AttrName,
    ) -> crate::GraphResult<Option<AttrValue>> {
        if !subgraph.contains_edge(id) {
            return Err(crate::GraphError::InvalidInput(format!(
                "Edge {id} is outside the subgraph selection"
            )));
        }
        if let Some(value) = self
            .property_batch
            .as_ref()
            .and_then(|batch| batch.get(false, id, name))
        {
            if !subgraph.graph().borrow().has_edge(id) {
                return Err(crate::GraphError::InvalidInput(format!(
                    "Edge {id} is inactive"
                )));
            }
            return Ok(Some(value.clone()));
        }
        subgraph.get_edge_attribute(id, name)
    }

    /// Publish a property column or return an owned output according to Context.
    /// Transient columns still use canonical target/kind validation.
    pub(crate) fn attach_node_column(
        &mut self,
        subgraph: &Subgraph,
        name: AttrName,
        values: Vec<(NodeId, AttrValue)>,
    ) -> Result<()> {
        if self.is_cancelled() {
            return Err(anyhow!("node attribute attachment cancelled"));
        }
        if self.persist_results() && self.property_batch.is_none() {
            return subgraph
                .set_node_attr_column(name, values)
                .map_err(|err| anyhow!(err));
        }
        self.attach_node_attrs(subgraph, HashMap::from([(name, values)]))
    }

    /// Attach multiple node columns through one validated persistence boundary.
    pub(crate) fn attach_node_attrs(
        &mut self,
        subgraph: &Subgraph,
        attrs: HashMap<AttrName, Vec<(NodeId, AttrValue)>>,
    ) -> Result<()> {
        if self.is_cancelled() {
            return Err(anyhow!("node attribute attachment cancelled"));
        }
        if self.persist_results() && self.property_batch.is_none() {
            return subgraph.set_node_attrs(attrs).map_err(|err| anyhow!(err));
        }
        subgraph
            .graph()
            .borrow()
            .validate_node_attrs(&attrs, None)?;
        if let Some(batch) = self.property_batch.as_mut() {
            return batch.stage(true, attrs).map_err(|err| anyhow!(err));
        }
        for (name, mut values) in attrs {
            values.sort_unstable_by_key(|(id, _)| *id);
            self.add_output(
                format!("node_attributes.{name}"),
                super::AlgorithmOutput::NodeAttributes(values),
            );
        }
        Ok(())
    }

    /// Attach edge columns or expose owned results without changing graph state.
    pub(crate) fn attach_edge_attrs(
        &mut self,
        subgraph: &Subgraph,
        attrs: HashMap<AttrName, Vec<(EdgeId, AttrValue)>>,
    ) -> Result<()> {
        if self.is_cancelled() {
            return Err(anyhow!("edge attribute attachment cancelled"));
        }
        if self.persist_results() && self.property_batch.is_none() {
            return subgraph.set_edge_attrs(attrs).map_err(|err| anyhow!(err));
        }
        subgraph
            .graph()
            .borrow()
            .validate_edge_attrs(&attrs, None)?;
        if let Some(batch) = self.property_batch.as_mut() {
            return batch.stage(false, attrs).map_err(|err| anyhow!(err));
        }
        for (name, mut values) in attrs {
            values.sort_unstable_by_key(|(id, _)| *id);
            self.add_output(
                format!("edge_attributes.{name}"),
                super::AlgorithmOutput::EdgeAttributes(values),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlay_reads_cannot_escape_selection_or_serve_deactivated_entities() {
        let mut graph = Graph::new();
        graph.add_nodes(3).unwrap();
        graph.add_edges(&[(0, 1), (0, 1)]).unwrap();
        let owner = graph.into_shared();
        let subgraph = Subgraph::new(owner.clone(), [0, 1].into(), [0].into(), "selected".into());
        let mut pending = PropertyBatch::new(owner.borrow().write_coordinate());
        pending
            .stage(
                true,
                HashMap::from([(
                    "x".into(),
                    vec![(0, AttrValue::Int(7)), (2, AttrValue::Int(8))],
                )]),
            )
            .unwrap();
        pending
            .stage(
                false,
                HashMap::from([(
                    "x".into(),
                    vec![(0, AttrValue::Int(7)), (1, AttrValue::Int(8))],
                )]),
            )
            .unwrap();
        let mut ctx = Context::new();
        ctx.property_batch = Some(pending);
        assert_eq!(
            ctx.node_attribute(&subgraph, 0, &"x".into()).unwrap(),
            Some(AttrValue::Int(7))
        );
        assert!(ctx.node_attribute(&subgraph, 2, &"x".into()).is_err());
        assert!(ctx.edge_attribute(&subgraph, 1, &"x".into()).is_err());
        owner.borrow_mut().remove_node(0).unwrap();
        assert!(ctx.node_attribute(&subgraph, 0, &"x".into()).is_err());
        assert!(ctx.edge_attribute(&subgraph, 0, &"x".into()).is_err());
    }
    #[test]
    fn final_values_are_one_mixed_batch_and_prior_graph_values_stay_unchanged_until_accept() {
        let mut graph = Graph::new_publishing();
        graph.insert_nodes(2).expect("nodes");
        graph.add_edge(0, 1).expect("edge");
        let before = graph.current_state_id();
        let mut pending = PropertyBatch::new(graph.write_coordinate());
        pending
            .stage(
                true,
                HashMap::from([("x".into(), vec![(0, AttrValue::Int(1))])]),
            )
            .expect("stage");
        pending
            .stage(
                true,
                HashMap::from([(
                    "x".into(),
                    vec![(0, AttrValue::Int(2)), (1, AttrValue::Null)],
                )]),
            )
            .expect("overwrite");
        pending
            .stage(
                false,
                HashMap::from([("w".into(), vec![(0, AttrValue::Float64(1.0000000001))])]),
            )
            .expect("edge");
        assert_eq!(pending.get(true, 0, &"x".into()), Some(&AttrValue::Int(2)));
        assert_eq!(graph.get_node_attr(0, &"x".into()).expect("old x"), None);
        pending.accept(&mut graph).expect("accept");
        assert_eq!(graph.current_state_id(), before + 1);
        assert!(!graph.has_uncommitted_changes());
        assert_eq!(
            graph.get_node_attr(0, &"x".into()).expect("x"),
            Some(AttrValue::Int(2))
        );
        assert_eq!(
            graph.get_edge_attr(0, &"w".into()).expect("w"),
            Some(AttrValue::Float64(1.0000000001))
        );
    }
    #[test]
    fn normalization_is_visible_to_later_reads_and_empty_columns_accept_nothing() {
        let mut graph = Graph::new_publishing();
        graph.insert_nodes(1).expect("node");
        let before = graph.current_state_id();
        let mut pending = PropertyBatch::new(graph.write_coordinate());
        pending
            .stage(
                true,
                HashMap::from([(
                    "labels".into(),
                    vec![(0, AttrValue::Float64Vec(Vec::new()))],
                )]),
            )
            .expect("normalize");
        assert_eq!(
            pending.get(true, 0, &"labels".into()),
            Some(&AttrValue::TextVec(Vec::new()))
        );
        let mut empty = PropertyBatch::new(graph.write_coordinate());
        empty
            .stage(true, HashMap::from([("x".into(), Vec::new())]))
            .expect("empty");
        empty.accept(&mut graph).expect("no-op");
        assert_eq!(graph.current_state_id(), before);
    }
    #[test]
    fn stale_or_dirty_heads_reject_without_accepting_staged_values() {
        let mut graph = Graph::new_publishing();
        graph.insert_nodes(1).expect("node");
        let mut pending = PropertyBatch::new(graph.write_coordinate());
        pending
            .stage(
                true,
                HashMap::from([("x".into(), vec![(0, AttrValue::Int(1))])]),
            )
            .expect("stage");
        graph
            .set_node_attr(0, "other".into(), AttrValue::Int(2))
            .expect("external write");
        assert!(pending.accept(&mut graph).is_err());
        assert_eq!(graph.get_node_attr(0, &"x".into()).expect("x"), None);
        graph.set_automatic_publication(false).expect("manual");
        let mut pending = PropertyBatch::new(graph.write_coordinate());
        pending
            .stage(
                true,
                HashMap::from([("x".into(), vec![(0, AttrValue::Int(1))])]),
            )
            .expect("stage");
        graph
            .set_node_attr(0, "other".into(), AttrValue::Int(3))
            .expect("dirty write");
        assert!(pending.accept(&mut graph).is_err());
        assert_eq!(graph.get_node_attr(0, &"x".into()).expect("x"), None);
    }

    #[test]
    fn branch_or_owner_changes_reject_even_when_the_head_and_ids_match() {
        for foreign_owner in [false, true] {
            let mut graph = Graph::new_publishing();
            graph.insert_nodes(1).expect("node");
            let head = graph.current_state_id();
            let mut pending = PropertyBatch::new(graph.write_coordinate());
            pending
                .stage(
                    true,
                    HashMap::from([("x".into(), vec![(0, AttrValue::Int(1))])]),
                )
                .expect("stage");
            if foreign_owner {
                graph = Graph::new_publishing();
                graph.insert_nodes(1).expect("same foreign head");
            } else {
                graph.create_branch("other".into()).expect("branch");
                graph.checkout_branch("other".into()).expect("switch");
            }
            assert_eq!(graph.current_state_id(), head);
            let error = pending.accept(&mut graph).expect_err("coordinate rejects");
            assert!(error.to_string().contains("coordinate"));
            assert_eq!(graph.current_state_id(), head);
            assert!(!graph.has_uncommitted_changes());
            assert_eq!(graph.get_node_attr(0, &"x".into()).expect("x"), None);
            assert_eq!(graph.add_node().expect("no consumed IDs"), 1);
        }
    }
}
