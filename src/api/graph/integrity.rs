//! Explicit retained-state verification; never runs on ordinary mutation paths.
use super::*;
use crate::state::{Delta, GraphSnapshot};

/// Cardinalities of a successfully verified retained graph. Allocation counts
/// include inactive/reserved IDs; value counts include retained raw values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntegrityReport {
    /// Accepted state captured before detached verification.
    pub selected_state: StateId,
    /// Selected branch captured with the checked state.
    pub selected_branch: String,
    /// Captured durable schema revision; None for an ordinary Graph.
    pub schema_revision: Option<u64>,
    /// Active nodes in the selected state.
    pub active_nodes: usize,
    /// Active edges in the selected state.
    pub active_edges: usize,
    /// Reserved node ID extent, including inactive IDs.
    pub allocated_nodes: usize,
    /// Reserved edge ID extent, including inactive IDs.
    pub allocated_edges: usize,
    /// Explicitly purged IDs within the monotonic lifetime node extent.
    pub purged_nodes: usize,
    /// Explicitly purged IDs within the monotonic lifetime edge extent.
    pub purged_edges: usize,
    /// Retained raw node property columns.
    pub node_columns: usize,
    /// Retained raw edge property columns.
    pub edge_columns: usize,
    /// Values in raw node columns, excluding frozen-source payloads.
    pub node_values: usize,
    /// Values in raw edge columns, excluding frozen-source payloads.
    pub edge_values: usize,
    /// All retained recorded versions, including unreferenced versions.
    pub retained_versions: usize,
    /// Verified versions plus the implicit empty state zero.
    pub verified_states: usize,
    /// Retained versions without precise publication time metadata.
    pub legacy_versions: usize,
    /// Retained branch references.
    pub branches: usize,
    /// Frozen source references/aliases in the owner pool.
    pub retained_group_references: usize,
    /// Retained external ID reservations, including historical aliases.
    pub external_id_reservations: usize,
    /// Largest node count in any verified recorded state.
    pub maximum_active_nodes: usize,
    /// Largest edge count in any verified recorded state.
    pub maximum_active_edges: usize,
    /// Whether a captured current durable schema policy was enforced.
    pub current_schema_checked: bool,
}

/// Owned portable capture. Verification can run outside a language runtime lock;
/// it does not retain a mutable graph or grant mutation authority.
pub struct IntegrityCheckPlan {
    archive: archive::GraphArchive,
    schema: Option<GraphSchema>,
    schema_revision: Option<u64>,
}
impl Graph {
    /// Verify recoverable storage invariants and every retained recorded state.
    /// Requires a clean accepted head. Does not publish, prune or rebuild indexes.
    pub fn check_integrity(&self) -> GraphResult<IntegrityReport> {
        self.prepare_integrity_check()?.execute()
    }
    /// Freeze input for detached full retained-state verification. Capture itself
    /// runs on the owner thread and owns portable copies of retained values.
    pub fn prepare_integrity_check(&self) -> GraphResult<IntegrityCheckPlan> {
        Ok(IntegrityCheckPlan {
            archive: self.archive_document()?,
            schema: None,
            schema_revision: None,
        })
    }
}
impl crate::storage::journal::DurableGraph {
    /// Verify the accepted in-memory retained graph and current schema. Journal
    /// bytes are verified by reopen; this does not rescan an already open file.
    pub fn check_integrity(&self) -> GraphResult<IntegrityReport> {
        self.prepare_integrity_check()?.execute()
    }
    /// Prepare an owned retained-graph/schema check, suitable for detached work.
    pub fn prepare_integrity_check(&self) -> GraphResult<IntegrityCheckPlan> {
        let mut plan = self.graph()?.prepare_integrity_check()?;
        plan.schema = self.schema().cloned();
        plan.schema_revision = Some(self.schema_revision());
        Ok(plan)
    }
}
impl IntegrityCheckPlan {
    /// Revalidate archive/config/allocation/references/hashes/clocks, then verify
    /// every first-parent transition, including unreferenced retained versions.
    /// It keeps only parent snapshots still needed by first-parent descendants;
    /// a linear history reuses one buffer. Wide branch frontiers can cost memory.
    pub fn execute(self) -> GraphResult<IntegrityReport> {
        let graph = self.archive.restore()?;
        if let Some(schema) = &self.schema {
            graph.check_schema_policy(schema)?;
        }
        let pool = graph.pool();
        let stats = pool.statistics();
        let (node_names, edge_names) = pool.attribute_names();
        let mut report = IntegrityReport {
            selected_state: graph.current_state_id(),
            selected_branch: graph.current_branch.clone(),
            schema_revision: self.schema_revision,
            active_nodes: graph.node_count(),
            active_edges: graph.edge_count(),
            allocated_nodes: stats.node_count,
            allocated_edges: stats.edge_count,
            purged_nodes: pool.purged_node_count(),
            purged_edges: pool.purged_edge_count(),
            node_columns: node_names.len(),
            edge_columns: edge_names.len(),
            node_values: node_names
                .iter()
                .filter_map(|name| pool.get_node_attribute_column(name))
                .map(|column| column.len())
                .sum(),
            edge_values: edge_names
                .iter()
                .filter_map(|name| pool.get_edge_attribute_column(name))
                .map(|column| column.len())
                .sum(),
            retained_versions: 0,
            verified_states: 1,
            legacy_versions: 0,
            branches: graph.list_branches().len(),
            retained_group_references: pool.retained_groups().len(),
            external_id_reservations: graph.identities.bindings.len(),
            maximum_active_nodes: 0,
            maximum_active_edges: 0,
            current_schema_checked: self.schema.is_some(),
        };
        let mut versions = graph.history_records();
        versions.sort_unstable_by_key(|version| version.id);
        report.retained_versions = versions.len();
        report.legacy_versions = versions
            .iter()
            .filter(|version| version.publication.is_none())
            .count();
        let mut remaining: HashMap<StateId, usize> = HashMap::new();
        for version in &versions {
            *remaining
                .entry(version.parents.first().copied().unwrap_or(0))
                .or_default() += 1;
        }
        let mut frontier =
            HashMap::from([(0, GraphSnapshot::empty_with_type(0, graph.graph_type()))]);
        for version in versions {
            let parent = version.parents.first().copied().unwrap_or(0);
            let count = remaining
                .get_mut(&parent)
                .ok_or_else(|| invalid(version.id, "missing first-parent reference count"))?;
            let mut state = if *count == 1 {
                frontier
                    .remove(&parent)
                    .ok_or_else(|| invalid(version.id, "missing first-parent state"))?
            } else {
                frontier
                    .get(&parent)
                    .ok_or_else(|| invalid(version.id, "missing shared first-parent state"))?
                    .clone()
            };
            *count -= 1;
            validate_before(&state, &version.delta, version.id)?;
            for &(edge, u, v) in &version.delta.edges_added {
                if pool.get_edge_endpoints(edge) != Some((u, v)) {
                    return Err(invalid(
                        version.id,
                        format!("edge {edge} differs from its allocated endpoint identity"),
                    ));
                }
            }
            state.apply_delta_in_place(&version.delta, version.id);
            validate_snapshot(&state)?;
            report.verified_states += 1;
            report.maximum_active_nodes = report.maximum_active_nodes.max(state.active_nodes.len());
            report.maximum_active_edges = report.maximum_active_edges.max(state.edges.len());
            if remaining.contains_key(&version.id) {
                frontier.insert(version.id, state);
            }
        }
        Ok(report)
    }
}
fn invalid(state: StateId, reason: impl std::fmt::Display) -> GraphError {
    GraphError::InvalidInput(format!(
        "Integrity check failed in recorded state {state}: {reason}"
    ))
}
fn validate_before(state: &GraphSnapshot, delta: &Delta, id: StateId) -> GraphResult<()> {
    for node in &delta.nodes_added {
        if state.active_nodes.contains(node) {
            return Err(invalid(
                id,
                format!("node {node} is added while already active"),
            ));
        }
    }
    for node in &delta.nodes_removed {
        if !state.active_nodes.contains(node) && !delta.nodes_added.contains(node) {
            return Err(invalid(
                id,
                format!("node {node} is removed while inactive"),
            ));
        }
    }
    for (edge, _, _) in &delta.edges_added {
        if state.edges.contains_key(edge) {
            return Err(invalid(
                id,
                format!("edge {edge} is added while already active"),
            ));
        }
    }
    for edge in &delta.edges_removed {
        if !state.edges.contains_key(edge) && !delta.edges_added.iter().any(|row| row.0 == *edge) {
            return Err(invalid(
                id,
                format!("edge {edge} is removed while inactive"),
            ));
        }
    }
    for (kind, rows, changes) in [
        ("node", &state.node_attributes, &delta.node_attr_changes),
        ("edge", &state.edge_attributes, &delta.edge_attr_changes),
    ] {
        for (entity, name, old, _) in changes {
            let actual = rows.get(entity).and_then(|row| row.get(name));
            let same = match (actual, old.as_ref()) {
                (None, None) => true,
                (Some(a), Some(b)) => archive::same_value(a, b).map_err(|e| invalid(id, e))?,
                _ => false,
            };
            if !same {
                return Err(invalid(id,format!("{kind} {entity} property {name:?} before-value differs from its first-parent state")));
            }
        }
    }
    Ok(())
}
fn validate_snapshot(state: &GraphSnapshot) -> GraphResult<()> {
    let nodes: HashSet<_> = state.active_nodes.iter().copied().collect();
    if nodes.len() != state.active_nodes.len() {
        return Err(invalid(state.state_id, "duplicate active node"));
    }
    for (&edge, &(u, v)) in &state.edges {
        if !nodes.contains(&u) || !nodes.contains(&v) {
            return Err(invalid(
                state.state_id,
                format!("edge {edge} has an inactive endpoint"),
            ));
        }
    }
    for (&node, row) in &state.node_attributes {
        if !row.is_empty() && !nodes.contains(&node) {
            return Err(invalid(
                state.state_id,
                format!("inactive node {node} has visible properties"),
            ));
        }
    }
    for (&edge, row) in &state.edge_attributes {
        if !row.is_empty() && !state.edges.contains_key(&edge) {
            return Err(invalid(
                state.state_id,
                format!("inactive edge {edge} has visible properties"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::change_tracker::ChangeSet;
    #[test]
    fn invalid_unused_raw_values_report_column_and_offset_without_changing_the_owner() {
        let mut graph = Graph::new();
        graph.add_node().unwrap();
        let state = graph.current_state_id();
        graph
            .pool
            .borrow_mut()
            .set_node_attribute(0, "broken".into(), AttrValue::Json("{".into()))
            .unwrap();
        let error = graph.check_integrity().unwrap_err();
        let GraphError::InvalidInput(reason) = error else {
            panic!("expected typed integrity error");
        };
        assert!(
            reason.contains("node column \"broken\" value 0"),
            "{reason}"
        );
        assert_eq!(graph.current_state_id(), state);
        assert_eq!(graph.get_node_attr(0, &"broken".into()).unwrap(), None);
    }

    #[test]
    fn unreferenced_invalid_before_values_fail_with_the_exact_recorded_state() {
        let mut graph = Graph::new();
        graph.add_node().unwrap();
        graph
            .set_node_attr(0, "x".into(), AttrValue::Int(0))
            .unwrap();
        let current = graph.current_state_id();
        let mut bad = ChangeSet::new();
        bad.node_attr_changes.push((
            0,
            "x".into(),
            Some(AttrValue::Int(999)),
            Some(AttrValue::Int(1)),
        ));
        let id = graph
            .history
            .create_commit(
                bad,
                "corrupt retained version".into(),
                "test".into(),
                Some(current),
            )
            .unwrap();
        let err = graph.check_integrity().unwrap_err().to_string();
        assert!(err.contains(&format!("state {id}")), "{err}");
        assert!(err.contains("before-value"), "{err}");
        assert_eq!(graph.current_state_id(), current);
        assert_eq!(
            graph.get_node_attr(0, &"x".into()).unwrap(),
            Some(AttrValue::Int(0))
        );
    }
    #[test]
    fn inactive_properties_in_a_nonhead_retained_state_cannot_hide_behind_a_valid_head() {
        let mut graph = Graph::new();
        graph.add_nodes(2).unwrap();
        graph.remove_node(1).unwrap();
        let current = graph.current_state_id();
        let mut bad = ChangeSet::new();
        bad.node_attr_changes
            .push((1, "x".into(), None, Some(AttrValue::Int(1))));
        let id = graph
            .history
            .create_commit(bad, "bad hidden state".into(), "test".into(), Some(current))
            .unwrap();
        let err = graph.check_integrity().unwrap_err().to_string();
        assert!(err.contains(&format!("state {id}")), "{err}");
        assert!(err.contains("inactive node"), "{err}");
    }
    #[test]
    fn physical_verification_detects_semantic_corruption_inside_a_valid_baseline() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal");
        let mut graph = Graph::new();
        graph.add_node().unwrap();
        graph
            .set_node_attr(0, "x".into(), AttrValue::Int(0))
            .unwrap();
        let current = graph.current_state_id();
        let mut bad = ChangeSet::new();
        bad.node_attr_changes.push((
            0,
            "x".into(),
            Some(AttrValue::Int(999)),
            Some(AttrValue::Int(1)),
        ));
        let id = graph
            .history
            .create_commit(
                bad,
                "bad retained transition".into(),
                "test".into(),
                Some(current),
            )
            .unwrap();
        // The internal fixture has valid hashes/framing and a valid selected head,
        // but an unreferenced retained version has an impossible before-value.
        let mut store = crate::DurableGraph::create_from_graph(&path, &graph, None).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let error = store.verify_storage().unwrap_err().to_string();
        assert!(
            error.contains(&format!("state {id}")) && error.contains("before-value"),
            "{error}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert!(store.graph().is_err());
    }
}
