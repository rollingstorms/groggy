//! Checked mutations through immutable selection pins; write ownership stays in Graph.
use super::*;
use crate::{
    api::batch::PreparedBatch,
    state::{state::GraphSnapshot, PinnedReadState},
    subgraphs::GraphView,
};

/// One accepted batch and its refreshed immutable selection. The original view stays unchanged.
#[derive(Debug)]
pub struct ViewBatchResult {
    /// IDs and accepted state from the canonical batch coordinator.
    pub batch: BatchResult,
    /// Refreshed immutable membership and values, independent of later writes.
    pub view: GraphView,
}

pub(crate) struct PreparedViewPublication {
    pub publication: crate::api::batch::PreparedPublication,
    pub view: GraphView,
    pub expected_state: StateId,
}

impl Graph {
    /// Publish a scoped batch against the view's captured source state and return a new view.
    /// Foreign/stale/dirty pins and cascades outside the selection reject before writes.
    /// This explicit checked operation publishes even when ordinary mutations are manual.
    pub fn apply_view_batch(
        &mut self,
        view: &GraphView,
        batch: GraphBatch,
        message: String,
        author: String,
    ) -> GraphResult<ViewBatchResult> {
        self.apply_view_batch_internal(view, batch, message, author, None)
    }

    /// Apply a scoped batch with the same proposed-state schema validation as apply_batch.
    pub fn apply_view_batch_with_schema(
        &mut self,
        view: &GraphView,
        batch: GraphBatch,
        message: String,
        author: String,
        schema: &GraphSchema,
    ) -> GraphResult<ViewBatchResult> {
        self.apply_view_batch_internal(view, batch, message, author, Some(schema))
    }

    fn apply_view_batch_internal(
        &mut self,
        view: &GraphView,
        batch: GraphBatch,
        message: String,
        author: String,
        schema: Option<&GraphSchema>,
    ) -> GraphResult<ViewBatchResult> {
        let prepared = self.prepare_view_publication(view, batch, schema, None)?;
        let batch = self.execute_publication(prepared.publication, message, author)?;
        Ok(self.accepted_view_result(batch, prepared.view))
    }

    /// Reuse only the complete output prepared before this accepted write. This
    /// is shared with durable publication after the journal append and execution.
    pub(crate) fn accepted_view_result(
        &self,
        batch: BatchResult,
        view: GraphView,
    ) -> ViewBatchResult {
        self.space.cache_accepted_read(view.read_state());
        ViewBatchResult { batch, view }
    }

    pub(crate) fn validate_bound_source(
        &self,
        view: &GraphView,
        require_accepted: bool,
    ) -> GraphResult<PinnedReadState> {
        let origin = view.read_state().origin().ok_or_else(|| {
            invalid("View has no current graph write provenance; capture a fresh read_view")
        })?;
        if !Arc::ptr_eq(&origin.history, self.history.identity()) {
            return Err(invalid("View belongs to a different graph history"));
        }
        if require_accepted && !origin.clean {
            return Err(invalid(
                "View captured unpublished/uncommitted state and is read-only; use a fresh accepted-state view",
            ));
        }
        if origin.branch != self.current_branch {
            return Err(invalid(
                "View belongs to a different branch; capture a fresh view",
            ));
        }
        if require_accepted {
            self.validate_write_head(origin.state)?;
        } else if origin.state != self.current_commit {
            return Err(invalid(
                "View belongs to a stale state; capture a fresh view",
            ));
        }
        // A live current payload has already passed immutable capture validation.
        // Distinct payloads still receive exact comparison after cache invalidation
        // (for example physical maintenance that leaves the logical head unchanged).
        let current = self.pin_current_state()?;
        if !std::ptr::eq(current.snapshot(), view.read_state().snapshot())
            && !same_values(current.snapshot(), view.read_state().snapshot())?
        {
            return Err(invalid(
                "Working graph no longer matches the view's captured state",
            ));
        }
        Ok(current)
    }

    pub(crate) fn prepare_view_publication(
        &self,
        view: &GraphView,
        batch: GraphBatch,
        schema: Option<&GraphSchema>,
        additional_schema: Option<&GraphSchema>,
    ) -> GraphResult<PreparedViewPublication> {
        let current = self.validate_bound_source(view, true)?;
        let origin = view
            .read_state()
            .origin()
            .ok_or_else(|| invalid("Missing view source"))?;
        let prepared = self.prepare_publication_with_additional_schema(
            batch,
            origin.state,
            schema,
            additional_schema,
        )?;
        let counts = self.pool.borrow().statistics();
        validate_scope(
            view,
            current.snapshot(),
            &prepared.batch,
            counts.node_count,
            counts.edge_count,
        )?;
        let mut snapshot = current.snapshot().clone();
        let (nodes, edges) = project(
            &mut snapshot,
            view,
            &prepared.batch,
            counts.node_count,
            counts.edge_count,
        );
        snapshot.state_id = self.history.next_state_id();
        // Construct and validate ALL returned data before publication. No fallible
        // capture or selected-membership validation runs after the graph changes.
        let state = PinnedReadState::from_snapshot(snapshot)?
            .with_groups(self.pool.borrow().retained_groups())
            .with_origin(
                Arc::clone(self.history.identity()),
                self.current_branch.clone(),
                self.history.next_state_id(),
                true,
            );
        let refreshed = GraphView::from_membership(state, nodes, edges)?;
        Ok(PreparedViewPublication {
            publication: prepared,
            view: refreshed,
            expected_state: origin.state,
        })
    }
}

fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}

fn same_values(a: &GraphSnapshot, b: &GraphSnapshot) -> GraphResult<bool> {
    Ok(a.state_id == b.state_id
        && a.graph_type == b.graph_type
        && a.active_nodes == b.active_nodes
        && a.edges == b.edges
        && same_property_rows(&a.node_attributes, &b.node_attributes)?
        && same_property_rows(&a.edge_attributes, &b.edge_attributes)?)
}

fn same_property_rows(
    a: &HashMap<usize, HashMap<AttrName, AttrValue>>,
    b: &HashMap<usize, HashMap<AttrName, AttrValue>>,
) -> GraphResult<bool> {
    // Missing rows and empty rows both mean no properties. Exact retained payload
    // equality still distinguishes floating widths, signed zero and NaN bits.
    if a.values().filter(|row| !row.is_empty()).count()
        != b.values().filter(|row| !row.is_empty()).count()
    {
        return Ok(false);
    }
    for (id, row) in a.iter().filter(|(_, row)| !row.is_empty()) {
        let Some(other) = b.get(id) else {
            return Ok(false);
        };
        if row.len() != other.len() {
            return Ok(false);
        }
        for (name, value) in row {
            let Some(other) = other.get(name) else {
                return Ok(false);
            };
            if !crate::state::change::same_value(value, other)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn validate_scope(
    view: &GraphView,
    snapshot: &GraphSnapshot,
    batch: &PreparedBatch,
    first_node: NodeId,
    first_edge: EdgeId,
) -> GraphResult<()> {
    // Keep this match exhaustive: a new batch operation must be considered here.
    // Identity bindings are normalized into the property columns by preparation;
    // clock fields describe acceptance, not additional selected write targets.
    let PreparedBatch {
        add_nodes,
        add_edges,
        edge_endpoint_changes,
        node_attrs,
        edge_attrs,
        unset_node_attrs,
        unset_edge_attrs,
        remove_edges,
        remove_nodes,
        identity_bindings: _,
        effective_at: _,
        change_times: _,
    } = batch;

    let node_ok =
        |id: NodeId| view.contains_node(id) || (first_node..first_node + *add_nodes).contains(&id);
    let edge_ok = |id: EdgeId| {
        view.contains_edge(id) || (first_edge..first_edge + add_edges.len()).contains(&id)
    };
    let mut errors = Vec::new();
    let node_targets = node_attrs
        .values()
        .flat_map(|rows| rows.iter().map(|(id, _)| *id))
        .chain(unset_node_attrs.values().flatten().copied())
        .chain(remove_nodes.iter().copied());
    for id in node_targets {
        if !node_ok(id) {
            errors.push(format!("node {id} is outside the view"));
        }
    }
    let edge_targets = edge_attrs
        .values()
        .flat_map(|rows| rows.iter().map(|(id, _)| *id))
        .chain(unset_edge_attrs.values().flatten().copied())
        .chain(edge_endpoint_changes.iter().map(|change| change.edge_id))
        .chain(remove_edges.iter().copied());
    for id in edge_targets {
        if !edge_ok(id) {
            errors.push(format!("edge {id} is outside the view"));
        }
    }
    for (source, target) in add_edges.iter().copied().chain(
        edge_endpoint_changes
            .iter()
            .map(|change| change.new_endpoints),
    ) {
        for id in [source, target] {
            if !node_ok(id) {
                errors.push(format!("edge endpoint {id} is outside the view"));
            }
        }
    }
    let removed: HashSet<_> = remove_nodes.iter().copied().collect();
    for (&edge, &(source, target)) in &snapshot.edges {
        if (removed.contains(&source) || removed.contains(&target)) && !view.contains_edge(edge) {
            errors.push(format!(
                "node removal would cascade to edge {edge} outside the view"
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        errors.sort_unstable();
        errors.dedup();
        Err(GraphError::InvalidInput(format!(
            "View batch scope: {}",
            errors.join("; ")
        )))
    }
}

// The input was fully prepared by the normal coordinator. Apply its logical
// overlay to the owned capture, preserving values without reading a changed owner.
fn project(
    snapshot: &mut GraphSnapshot,
    view: &GraphView,
    batch: &PreparedBatch,
    first_node: NodeId,
    first_edge: EdgeId,
) -> (HashSet<NodeId>, HashSet<EdgeId>) {
    // Keep this match exhaustive: a new batch operation must be considered here.
    // Identity bindings are normalized into the property columns by preparation;
    // clock fields describe acceptance, not additional selected write targets.
    let PreparedBatch {
        add_nodes,
        add_edges,
        edge_endpoint_changes,
        node_attrs,
        edge_attrs,
        unset_node_attrs,
        unset_edge_attrs,
        remove_edges,
        remove_nodes,
        identity_bindings: _,
        effective_at: _,
        change_times: _,
    } = batch;

    let added_nodes = first_node..first_node + *add_nodes;
    let added_edges = first_edge..first_edge + add_edges.len();
    let mut nodes = view.nodes().clone();
    let mut edges = view.edges().clone();
    nodes.extend(added_nodes.clone());
    edges.extend(added_edges);
    for node in added_nodes {
        snapshot.active_nodes.push(node);
        snapshot.node_attributes.insert(
            node,
            HashMap::from([("entity_type".into(), AttrValue::Text("base".into()))]),
        );
    }
    for (offset, &endpoints) in add_edges.iter().enumerate() {
        snapshot.edges.insert(first_edge + offset, endpoints);
    }
    for change in edge_endpoint_changes {
        snapshot.edges.insert(change.edge_id, change.new_endpoints);
    }
    for (name, rows) in node_attrs {
        for (id, value) in rows {
            snapshot
                .node_attributes
                .entry(*id)
                .or_default()
                .insert(name.clone(), value.clone());
        }
    }
    for (name, rows) in edge_attrs {
        for (id, value) in rows {
            snapshot
                .edge_attributes
                .entry(*id)
                .or_default()
                .insert(name.clone(), value.clone());
        }
    }
    for (columns, unsets) in [
        (&mut snapshot.node_attributes, unset_node_attrs),
        (&mut snapshot.edge_attributes, unset_edge_attrs),
    ] {
        for (name, ids) in unsets {
            for id in ids {
                if let Some(row) = columns.get_mut(id) {
                    row.remove(name);
                }
            }
        }
        columns.retain(|_, row| !row.is_empty());
    }
    let removed_nodes: HashSet<_> = remove_nodes.iter().copied().collect();
    let removed_edges: HashSet<_> = remove_edges.iter().copied().collect();
    snapshot
        .active_nodes
        .retain(|id| !removed_nodes.contains(id));
    snapshot
        .node_attributes
        .retain(|id, _| !removed_nodes.contains(id));
    snapshot.edges.retain(|id, (source, target)| {
        !removed_edges.contains(id)
            && !removed_nodes.contains(source)
            && !removed_nodes.contains(target)
    });
    snapshot
        .edge_attributes
        .retain(|id, _| snapshot.edges.contains_key(id));
    nodes.retain(|id| !removed_nodes.contains(id));
    edges.retain(|id| snapshot.edges.contains_key(id));
    (nodes, edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_comparison_uses_exact_payloads_and_normalizes_empty_rows() {
        let mut a = GraphSnapshot::empty(1);
        a.active_nodes.push(0);
        a.edges.insert(0, (0, 0));
        let mut b = a.clone();
        a.edge_attributes.insert(0, HashMap::new());
        assert!(same_values(&a, &b).unwrap());
        for (left, right) in [
            (AttrValue::Float(1.), AttrValue::Float64(1.)),
            (AttrValue::Float64(0.), AttrValue::Float64(-0.)),
            (
                AttrValue::Float64(f64::from_bits(0x7ff8000000000001)),
                AttrValue::Float64(f64::from_bits(0x7ff8000000000002)),
            ),
            (
                AttrValue::Float64Vec(vec![0.]),
                AttrValue::Float64Vec(vec![-0.]),
            ),
        ] {
            a.node_attributes
                .insert(0, HashMap::from([("value".into(), left)]));
            b.node_attributes
                .insert(0, HashMap::from([("value".into(), right)]));
            assert!(!same_values(&a, &b).unwrap());
            assert!(same_values(&a, &a).unwrap());
        }
    }
    #[test]
    fn accepted_payload_cache_is_weak_and_fresh_captures_keep_distinct_identity() {
        let mut graph = Graph::new();
        graph.add_node().unwrap();
        let source = graph.read_view().unwrap();
        let batch = GraphBatch {
            node_attrs: HashMap::from([(
                "value".into(),
                vec![(crate::BatchNodeRef::Existing(0), AttrValue::Int(7))],
            )]),
            ..Default::default()
        };
        let result = graph
            .apply_view_batch(&source, batch, "update".into(), "test".into())
            .unwrap();
        let current = graph.read_view().unwrap();
        assert!(std::ptr::eq(
            current.read_state().snapshot(),
            result.view.read_state().snapshot()
        ));
        assert!(!current
            .read_state()
            .shares_capture(result.view.read_state()));
        let weak = result.view.read_state().downgrade_data();
        drop(current);
        drop(result);
        assert!(weak.upgrade().is_none());
        assert_eq!(source.node_attr(0, &"value".into()).unwrap(), None);
        assert_eq!(
            graph
                .read_view()
                .unwrap()
                .node_attr(0, &"value".into())
                .unwrap(),
            Some(AttrValue::Int(7))
        );
    }
}
