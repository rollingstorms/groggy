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
        Ok(ViewBatchResult {
            batch,
            view: prepared.view,
        })
    }

    pub(crate) fn prepare_view_publication(
        &self,
        view: &GraphView,
        batch: GraphBatch,
        schema: Option<&GraphSchema>,
        additional_schema: Option<&GraphSchema>,
    ) -> GraphResult<PreparedViewPublication> {
        let origin = view.read_state().origin().ok_or_else(|| {
            invalid("View has no current graph write provenance; capture a fresh read_view")
        })?;
        if !Arc::ptr_eq(&origin.history, self.history.identity()) {
            return Err(invalid("View belongs to a different graph history"));
        }
        if !origin.clean {
            return Err(invalid(
                "View captured unpublished state and is read-only; use a fresh accepted-state view",
            ));
        }
        if origin.branch != self.current_branch {
            return Err(invalid(
                "View belongs to a different branch; capture a fresh view",
            ));
        }
        self.validate_write_head(origin.state)?;
        // One current capture both validates raw indices/endpoints and detects low-level
        // storage escapes that changed data without advancing the checked head.
        let mut snapshot = self
            .space
            .capture_read_snapshot(&self.pool.borrow(), self.current_commit)?;
        if !same_values(&snapshot, view.read_state().snapshot()) {
            return Err(invalid(
                "Working graph no longer matches the view's captured state",
            ));
        }
        let prepared = self.prepare_publication_with_additional_schema(
            batch,
            origin.state,
            schema,
            additional_schema,
        )?;
        let counts = self.pool.borrow().statistics();
        validate_scope(
            view,
            &snapshot,
            &prepared.batch,
            counts.node_count,
            counts.edge_count,
        )?;
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
        let state = PinnedReadState::from_snapshot(snapshot)?.with_origin(
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

fn same_values(a: &GraphSnapshot, b: &GraphSnapshot) -> bool {
    a.state_id == b.state_id
        && a.graph_type == b.graph_type
        && a.active_nodes == b.active_nodes
        && a.edges == b.edges
        && a.node_attributes == b.node_attributes
        && a.edge_attributes == b.edge_attributes
}

fn validate_scope(
    view: &GraphView,
    snapshot: &GraphSnapshot,
    batch: &PreparedBatch,
    first_node: NodeId,
    first_edge: EdgeId,
) -> GraphResult<()> {
    let node_ok = |id: NodeId| {
        view.contains_node(id) || (first_node..first_node + batch.add_nodes).contains(&id)
    };
    let edge_ok = |id: EdgeId| {
        view.contains_edge(id) || (first_edge..first_edge + batch.add_edges.len()).contains(&id)
    };
    let mut errors = Vec::new();
    let node_targets = batch
        .node_attrs
        .values()
        .flat_map(|rows| rows.iter().map(|(id, _)| *id))
        .chain(batch.remove_nodes.iter().copied());
    for id in node_targets {
        if !node_ok(id) {
            errors.push(format!("node {id} is outside the view"));
        }
    }
    let edge_targets = batch
        .edge_attrs
        .values()
        .flat_map(|rows| rows.iter().map(|(id, _)| *id))
        .chain(batch.remove_edges.iter().copied());
    for id in edge_targets {
        if !edge_ok(id) {
            errors.push(format!("edge {id} is outside the view"));
        }
    }
    for &(source, target) in &batch.add_edges {
        for id in [source, target] {
            if !node_ok(id) {
                errors.push(format!("edge endpoint {id} is outside the view"));
            }
        }
    }
    let removed: HashSet<_> = batch.remove_nodes.iter().copied().collect();
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
    let added_nodes = first_node..first_node + batch.add_nodes;
    let added_edges = first_edge..first_edge + batch.add_edges.len();
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
    for (offset, &endpoints) in batch.add_edges.iter().enumerate() {
        snapshot.edges.insert(first_edge + offset, endpoints);
    }
    for (name, rows) in &batch.node_attrs {
        for (id, value) in rows {
            snapshot
                .node_attributes
                .entry(*id)
                .or_default()
                .insert(name.clone(), value.clone());
        }
    }
    for (name, rows) in &batch.edge_attrs {
        for (id, value) in rows {
            snapshot
                .edge_attributes
                .entry(*id)
                .or_default()
                .insert(name.clone(), value.clone());
        }
    }
    let removed_nodes: HashSet<_> = batch.remove_nodes.iter().copied().collect();
    let removed_edges: HashSet<_> = batch.remove_edges.iter().copied().collect();
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
