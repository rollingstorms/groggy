//! Temporal extensions for algorithm context.
//!
//! This module provides temporal awareness to algorithms, enabling:
//! - Time-scoped algorithm execution
//! - Delta computation between snapshots
//! - Change tracking within time windows
//! - Temporal metadata access during pipeline execution

use std::collections::{BTreeSet, HashSet};

use crate::errors::{GraphError, GraphResult};
use crate::temporal::TemporalSnapshot;
use crate::types::{AttrName, AttrValue, EdgeId, NodeId, StateId};

/// Temporal scope for algorithm execution.
///
/// This structure defines the temporal context in which an algorithm runs,
/// allowing it to:
/// - Know which commit(s) it's analyzing
/// - Access reference snapshots for comparison
/// - Query time windows for change detection
#[derive(Debug, Clone)]
pub struct TemporalScope {
    /// Current commit being analyzed.
    pub current_commit: StateId,

    /// Optional time window for windowed operations.
    /// Format: (start_commit, end_commit)
    pub window: Option<(StateId, StateId)>,

    /// Optional reference snapshot for delta/comparison operations.
    pub reference_snapshot: Option<TemporalSnapshot>,

    /// Additional temporal metadata (tags, descriptions, etc.)
    pub metadata: TemporalMetadata,
}

impl TemporalScope {
    /// Create a new temporal scope for a specific commit.
    pub fn at_commit(commit_id: StateId) -> Self {
        Self {
            current_commit: commit_id,
            window: None,
            reference_snapshot: None,
            metadata: TemporalMetadata::default(),
        }
    }

    /// Create a temporal scope with a time window.
    pub fn with_window(current: StateId, start: StateId, end: StateId) -> Self {
        Self {
            current_commit: current,
            window: Some((start, end)),
            reference_snapshot: None,
            metadata: TemporalMetadata::default(),
        }
    }

    /// Set a reference snapshot for comparison operations.
    pub fn with_reference(mut self, snapshot: TemporalSnapshot) -> Self {
        self.reference_snapshot = Some(snapshot);
        self
    }

    /// Add metadata to this scope.
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.tags.insert(key, value);
        self
    }

    /// Check if this scope has a time window defined.
    pub fn has_window(&self) -> bool {
        self.window.is_some()
    }

    /// Check if this scope has a reference snapshot.
    pub fn has_reference(&self) -> bool {
        self.reference_snapshot.is_some()
    }

    /// Get the window duration in commits (if window is defined).
    pub fn window_size(&self) -> Option<usize> {
        self.window
            .map(|(start, end)| (end.saturating_sub(start)) as usize)
    }
}

/// Additional metadata for temporal scopes.
#[derive(Debug, Clone, Default)]
pub struct TemporalMetadata {
    /// User-defined tags for categorizing temporal contexts.
    pub tags: std::collections::HashMap<String, String>,

    /// Optional description of what this temporal scope represents.
    pub description: Option<String>,
}

/// Represents the difference between two temporal snapshots.
///
/// This structure captures all changes that occurred between two points
/// in time, organized by entity type and operation type.
#[derive(Debug, Clone)]
pub struct TemporalDelta {
    /// Commit ID of the "before" snapshot.
    pub from_commit: StateId,

    /// Commit ID of the "after" snapshot.
    pub to_commit: StateId,

    /// Nodes that were added between snapshots.
    pub nodes_added: Vec<NodeId>,

    /// Nodes that were removed between snapshots.
    pub nodes_removed: Vec<NodeId>,

    /// Edges that were added between snapshots.
    pub edges_added: Vec<EdgeId>,

    /// Edges that were removed between snapshots.
    pub edges_removed: Vec<EdgeId>,

    /// Node attribute changes: (node_id, attr_name, old_value, new_value)
    pub node_attr_changes: Vec<NodeAttrChange>,

    /// Edge attribute changes: (edge_id, attr_name, old_value, new_value)
    pub edge_attr_changes: Vec<EdgeAttrChange>,
}

/// Represents a single node attribute change.
#[derive(Debug, Clone)]
pub struct NodeAttrChange {
    pub node_id: NodeId,
    pub attr_name: AttrName,
    pub old_value: Option<AttrValue>,
    pub new_value: Option<AttrValue>,
}

/// Represents a single edge attribute change.
#[derive(Debug, Clone)]
pub struct EdgeAttrChange {
    pub edge_id: EdgeId,
    pub attr_name: AttrName,
    pub old_value: Option<AttrValue>,
    pub new_value: Option<AttrValue>,
}

impl TemporalDelta {
    /// Compute the delta between two temporal snapshots.
    ///
    /// Compare retained states from the same in-process history instance.
    /// Results are sorted net membership/property differences; property changes
    /// cover entities surviving in both states. Branch siblings and reverse
    /// comparisons are allowed. Intermediate events are not replayed, and
    /// endpoint rewiring rejects until represented by this delta type.
    pub fn compute(prev: &TemporalSnapshot, cur: &TemporalSnapshot) -> GraphResult<Self> {
        Self::compute_cancellable(prev, cur, || false)
    }

    pub(crate) fn compute_cancellable(
        prev: &TemporalSnapshot,
        cur: &TemporalSnapshot,
        cancelled: impl Fn() -> bool,
    ) -> GraphResult<Self> {
        let check = || {
            if cancelled() {
                Err(GraphError::InvalidInput("temporal diff cancelled".into()))
            } else {
                Ok(())
            }
        };
        check()?;
        if !prev.same_history(cur) {
            return Err(GraphError::InvalidInput(
                "Temporal snapshots must belong to the same history instance".into(),
            ));
        }
        let before = prev.read_state().snapshot();
        let after = cur.read_state().snapshot();
        let from_commit = prev.lineage().commit_id;
        let to_commit = cur.lineage().commit_id;
        let prev_nodes = prev.existence().nodes();
        let cur_nodes = cur.existence().nodes();
        let prev_edges = prev.existence().edges();
        let cur_edges = cur.existence().edges();
        let mut nodes_added = Vec::new();
        let mut nodes_removed = Vec::new();
        let mut edges_added = Vec::new();
        let mut edges_removed = Vec::new();
        let mut node_attr_changes = Vec::new();
        let mut edge_attr_changes = Vec::new();
        for &id in cur_nodes {
            check()?;
            if !prev_nodes.contains(&id) {
                nodes_added.push(id);
            } else {
                for (name, old_value, new_value) in property_changes(
                    before.node_attributes.get(&id),
                    after.node_attributes.get(&id),
                    &check,
                )? {
                    node_attr_changes.push(NodeAttrChange {
                        node_id: id,
                        attr_name: name,
                        old_value,
                        new_value,
                    });
                }
            }
        }
        for &id in prev_nodes {
            check()?;
            if !cur_nodes.contains(&id) {
                nodes_removed.push(id);
            }
        }
        for &id in cur_edges {
            check()?;
            if !prev_edges.contains(&id) {
                edges_added.push(id);
            } else {
                if before.edges.get(&id) != after.edges.get(&id) {
                    return Err(GraphError::InvalidInput(
                        "Temporal edge endpoint-change deltas are not supported".into(),
                    ));
                }
                for (name, old_value, new_value) in property_changes(
                    before.edge_attributes.get(&id),
                    after.edge_attributes.get(&id),
                    &check,
                )? {
                    edge_attr_changes.push(EdgeAttrChange {
                        edge_id: id,
                        attr_name: name,
                        old_value,
                        new_value,
                    });
                }
            }
        }
        for &id in prev_edges {
            check()?;
            if !cur_edges.contains(&id) {
                edges_removed.push(id);
            }
        }
        nodes_added.sort_unstable();
        nodes_removed.sort_unstable();
        edges_added.sort_unstable();
        edges_removed.sort_unstable();
        node_attr_changes.sort_by(|a, b| (a.node_id, &a.attr_name).cmp(&(b.node_id, &b.attr_name)));
        edge_attr_changes.sort_by(|a, b| (a.edge_id, &a.attr_name).cmp(&(b.edge_id, &b.attr_name)));
        check()?;

        Ok(Self {
            from_commit,
            to_commit,
            nodes_added,
            nodes_removed,
            edges_added,
            edges_removed,
            node_attr_changes,
            edge_attr_changes,
        })
    }

    /// Check if this delta represents any changes.
    pub fn is_empty(&self) -> bool {
        self.nodes_added.is_empty()
            && self.nodes_removed.is_empty()
            && self.edges_added.is_empty()
            && self.edges_removed.is_empty()
            && self.node_attr_changes.is_empty()
            && self.edge_attr_changes.is_empty()
    }

    /// Get a summary of changes.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        if !self.nodes_added.is_empty() {
            parts.push(format!("+{} nodes", self.nodes_added.len()));
        }
        if !self.nodes_removed.is_empty() {
            parts.push(format!("-{} nodes", self.nodes_removed.len()));
        }
        if !self.edges_added.is_empty() {
            parts.push(format!("+{} edges", self.edges_added.len()));
        }
        if !self.edges_removed.is_empty() {
            parts.push(format!("-{} edges", self.edges_removed.len()));
        }

        let total_attr_changes = self.node_attr_changes.len() + self.edge_attr_changes.len();
        if total_attr_changes > 0 {
            parts.push(format!("{} attr changes", total_attr_changes));
        }

        if parts.is_empty() {
            "no changes".to_string()
        } else {
            parts.join(", ")
        }
    }

    /// Get all nodes that were affected by changes (added, removed, or modified).
    pub fn affected_nodes(&self) -> HashSet<NodeId> {
        let mut affected = HashSet::new();
        affected.extend(&self.nodes_added);
        affected.extend(&self.nodes_removed);
        for change in &self.node_attr_changes {
            affected.insert(change.node_id);
        }
        affected
    }

    /// Get all edges that were affected by changes.
    pub fn affected_edges(&self) -> HashSet<EdgeId> {
        let mut affected = HashSet::new();
        affected.extend(&self.edges_added);
        affected.extend(&self.edges_removed);
        for change in &self.edge_attr_changes {
            affected.insert(change.edge_id);
        }
        affected
    }
}

type PropertyDifference = (AttrName, Option<AttrValue>, Option<AttrValue>);

fn property_changes(
    before: Option<&std::collections::HashMap<AttrName, AttrValue>>,
    after: Option<&std::collections::HashMap<AttrName, AttrValue>>,
    check: &impl Fn() -> GraphResult<()>,
) -> GraphResult<Vec<PropertyDifference>> {
    let mut names = BTreeSet::new();
    for attrs in [before, after].into_iter().flatten() {
        for name in attrs.keys() {
            check()?;
            names.insert(name);
        }
    }
    let mut changes = Vec::new();
    for name in names {
        check()?;
        let old = before.and_then(|attrs| attrs.get(name));
        let new = after.and_then(|attrs| attrs.get(name));
        if old != new {
            changes.push((name.clone(), old.cloned(), new.cloned()));
        }
    }
    Ok(changes)
}

/// Represents entities (nodes/edges) that changed within a time window.
///
/// This is useful for identifying which parts of the graph were active
/// during a specific time period.
#[derive(Debug, Clone)]
pub struct ChangedEntities {
    /// Nodes that were created, deleted, or had attributes modified.
    pub modified_nodes: HashSet<NodeId>,

    /// Edges that were created, deleted, or had attributes modified.
    pub modified_edges: HashSet<EdgeId>,

    /// Type of change for each node (created, deleted, modified).
    pub node_change_types: std::collections::HashMap<NodeId, ChangeType>,

    /// Type of change for each edge.
    pub edge_change_types: std::collections::HashMap<EdgeId, ChangeType>,
}

impl ChangedEntities {
    /// Create an empty set of changed entities.
    pub fn empty() -> Self {
        Self {
            modified_nodes: HashSet::new(),
            modified_edges: HashSet::new(),
            node_change_types: std::collections::HashMap::new(),
            edge_change_types: std::collections::HashMap::new(),
        }
    }

    /// Check if any entities changed.
    pub fn is_empty(&self) -> bool {
        self.modified_nodes.is_empty() && self.modified_edges.is_empty()
    }

    /// Get the total number of changed entities.
    pub fn total_changes(&self) -> usize {
        self.modified_nodes.len() + self.modified_edges.len()
    }

    /// Add a node change.
    pub fn add_node(&mut self, node_id: NodeId, change_type: ChangeType) {
        self.modified_nodes.insert(node_id);
        self.node_change_types
            .entry(node_id)
            .and_modify(|value| *value = value.combine(change_type))
            .or_insert(change_type);
    }

    /// Add an edge change.
    pub fn add_edge(&mut self, edge_id: EdgeId, change_type: ChangeType) {
        self.modified_edges.insert(edge_id);
        self.edge_change_types
            .entry(edge_id)
            .and_modify(|value| *value = value.combine(change_type))
            .or_insert(change_type);
    }

    /// Merge another ChangedEntities into this one.
    pub fn merge(&mut self, other: ChangedEntities) {
        self.modified_nodes.extend(other.modified_nodes);
        self.modified_edges.extend(other.modified_edges);
        for (id, change) in other.node_change_types {
            self.add_node(id, change);
        }
        for (id, change) in other.edge_change_types {
            self.add_edge(id, change);
        }
    }
}

/// Type of change that occurred to an entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChangeType {
    /// Entity was created.
    Created,
    /// Entity was deleted.
    Deleted,
    /// Entity had attributes modified.
    AttributeModified,
    /// Multiple types of changes (e.g., created and modified in same window).
    Multiple,
}

impl ChangeType {
    /// Combine two change types.
    pub fn combine(self, other: ChangeType) -> ChangeType {
        if self == other {
            self
        } else {
            ChangeType::Multiple
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::Graph;

    #[test]
    fn retained_delta_reports_sorted_logical_property_changes_and_membership() {
        let mut graph = Graph::new();
        graph.add_nodes(4);
        let edge = graph.add_edge(0, 1).unwrap();
        let removed_edge = graph.add_edge(2, 3).unwrap();
        graph
            .set_node_attr(0, "score".into(), AttrValue::Int(1))
            .unwrap();
        graph
            .set_node_attr(0, "text".into(), AttrValue::Text("index_123".into()))
            .unwrap();
        graph
            .set_edge_attr(edge, "weight".into(), AttrValue::Float64(1.0))
            .unwrap();
        let first = graph.commit("before".into(), "test".into()).unwrap();
        let before = graph.snapshot_at_commit(first).unwrap();
        graph
            .set_node_attr(0, "score".into(), AttrValue::Int(2))
            .unwrap();
        graph
            .set_node_attr(1, "nullable".into(), AttrValue::Null)
            .unwrap();
        graph
            .set_edge_attr(edge, "weight".into(), AttrValue::Float64(2.0))
            .unwrap();
        graph.remove_node(3).unwrap();
        let new = graph.add_node();
        let added_edge = graph.add_edge(0, new).unwrap();
        let second = graph.commit("after".into(), "test".into()).unwrap();
        let after = graph.snapshot_at_commit(second).unwrap();
        let delta = TemporalDelta::compute(&before, &after).unwrap();
        assert_eq!(delta.nodes_added, vec![new]);
        assert_eq!(delta.nodes_removed, vec![3]);
        assert_eq!(delta.edges_added, vec![added_edge]);
        assert_eq!(delta.edges_removed, vec![removed_edge]);
        assert_eq!(delta.node_attr_changes.len(), 2);
        assert_eq!(delta.node_attr_changes[0].node_id, 0);
        assert_eq!(
            delta.node_attr_changes[0].old_value,
            Some(AttrValue::Int(1))
        );
        assert_eq!(
            delta.node_attr_changes[0].new_value,
            Some(AttrValue::Int(2))
        );
        assert_eq!(delta.node_attr_changes[1].node_id, 1);
        assert_eq!(delta.node_attr_changes[1].old_value, None);
        assert_eq!(delta.node_attr_changes[1].new_value, Some(AttrValue::Null));
        assert_eq!(delta.edge_attr_changes.len(), 1);
        assert_eq!(delta.edge_attr_changes[0].edge_id, edge);
        assert!(delta.affected_nodes().contains(&0));
        assert!(delta.summary().contains("3 attr changes"));
        let reverse = TemporalDelta::compute(&after, &before).unwrap();
        assert_eq!(reverse.nodes_added, vec![3]);
        assert_eq!(
            reverse.node_attr_changes[1].old_value,
            Some(AttrValue::Null)
        );
        assert_eq!(reverse.node_attr_changes[1].new_value, None);
        drop(graph);
        assert!(TemporalDelta::compute(&before, &before.clone())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn retained_delta_rejects_foreign_history_even_when_numeric_ids_match() {
        let mut first = Graph::new();
        let mut second = Graph::new();
        first.add_node();
        second.add_node();
        let id = first.commit("same".into(), "test".into()).unwrap();
        assert_eq!(second.commit("same".into(), "test".into()).unwrap(), id);
        let before = first.snapshot_at_commit(id).unwrap();
        let after = second.snapshot_at_commit(id).unwrap();
        assert!(TemporalDelta::compute(&before, &after).is_err());
    }

    #[test]
    fn property_delta_preserves_absence_null_types_and_stable_name_order() {
        use std::collections::HashMap;
        let before: HashMap<_, _> = [
            ("removed".into(), AttrValue::Int(1)),
            ("type".into(), AttrValue::Int(3)),
            ("unchanged".into(), AttrValue::Float64(f64::NAN)),
        ]
        .into_iter()
        .collect();
        let after: HashMap<_, _> = [
            ("null".into(), AttrValue::Null),
            ("type".into(), AttrValue::Text("3".into())),
            ("unchanged".into(), AttrValue::Float64(f64::NAN)),
        ]
        .into_iter()
        .collect();
        let changes = property_changes(Some(&before), Some(&after), &|| Ok(())).unwrap();
        assert_eq!(
            changes.iter().map(|row| row.0.as_str()).collect::<Vec<_>>(),
            vec!["null", "removed", "type"]
        );
        assert_eq!(changes[0].1, None);
        assert_eq!(changes[0].2, Some(AttrValue::Null));
        assert_eq!(changes[1].2, None);
        assert_eq!(changes[2].2, Some(AttrValue::Text("3".into())));
    }

    #[test]
    fn retained_delta_rejects_unrepresented_endpoint_rewiring() {
        use crate::state::{change_tracker::ChangeSet, history::HistoryForest};
        let mut history = HistoryForest::new();
        let mut first = ChangeSet::new();
        first.nodes_added = vec![0, 1, 2];
        first.edges_added = vec![(0, 0, 1)];
        let before_id = history
            .create_commit(first, "first".into(), "test".into(), None)
            .unwrap();
        let before = TemporalSnapshot::at_commit(&history, before_id).unwrap();
        let mut next = ChangeSet::new();
        next.edges_added = vec![(0, 0, 2)];
        let after_id = history
            .create_commit(next, "next".into(), "test".into(), Some(before_id))
            .unwrap();
        let after = TemporalSnapshot::at_commit(&history, after_id).unwrap();
        assert!(TemporalDelta::compute(&before, &after).is_err());
    }

    #[test]
    fn retained_delta_checks_cancellation_during_traversal() {
        use std::cell::Cell;
        let mut graph = Graph::new();
        graph.add_nodes(20);
        let id = graph.commit("fixture".into(), "test".into()).unwrap();
        let snapshot = graph.snapshot_at_commit(id).unwrap();
        let checks = Cell::new(0);
        let result = TemporalDelta::compute_cancellable(&snapshot, &snapshot, || {
            checks.set(checks.get() + 1);
            checks.get() > 5
        });
        assert!(result.is_err());
        assert_eq!(checks.get(), 6);
    }

    #[test]
    fn retained_delta_compares_sibling_branches_without_numeric_time_assumptions() {
        let mut graph = Graph::new();
        graph.add_node();
        graph
            .set_node_attr(0, "x".into(), AttrValue::Int(0))
            .unwrap();
        graph.commit("root".into(), "test".into()).unwrap();
        graph.create_branch("sibling".into()).unwrap();
        graph
            .set_node_attr(0, "x".into(), AttrValue::Int(1))
            .unwrap();
        let left = graph.commit("left".into(), "test".into()).unwrap();
        let before = graph.snapshot_at_commit(left).unwrap();
        graph.checkout_branch("sibling".into()).unwrap();
        graph
            .set_node_attr(0, "x".into(), AttrValue::Int(2))
            .unwrap();
        let right = graph.commit("right".into(), "test".into()).unwrap();
        let after = graph.snapshot_at_commit(right).unwrap();
        let delta = TemporalDelta::compute(&before, &after).unwrap();
        assert!(delta.nodes_added.is_empty());
        assert_eq!(delta.node_attr_changes.len(), 1);
        assert_eq!(
            delta.node_attr_changes[0].old_value,
            Some(AttrValue::Int(1))
        );
        assert_eq!(
            delta.node_attr_changes[0].new_value,
            Some(AttrValue::Int(2))
        );
    }

    #[test]
    fn test_temporal_scope_creation() {
        let scope = TemporalScope::at_commit(42);
        assert_eq!(scope.current_commit, 42);
        assert!(!scope.has_window());
        assert!(!scope.has_reference());
    }

    #[test]
    fn test_temporal_scope_with_window() {
        let scope = TemporalScope::with_window(50, 10, 50);
        assert_eq!(scope.current_commit, 50);
        assert!(scope.has_window());
        assert_eq!(scope.window_size(), Some(40));
    }

    #[test]
    fn test_temporal_delta_empty() {
        let delta = TemporalDelta {
            from_commit: 1,
            to_commit: 2,
            nodes_added: Vec::new(),
            nodes_removed: Vec::new(),
            edges_added: Vec::new(),
            edges_removed: Vec::new(),
            node_attr_changes: Vec::new(),
            edge_attr_changes: Vec::new(),
        };

        assert!(delta.is_empty());
        assert_eq!(delta.summary(), "no changes");
    }

    #[test]
    fn test_changed_entities() {
        let mut entities = ChangedEntities::empty();
        assert!(entities.is_empty());

        entities.add_node(1, ChangeType::Created);
        entities.add_node(2, ChangeType::AttributeModified);
        entities.add_edge(100, ChangeType::Deleted);

        assert!(!entities.is_empty());
        assert_eq!(entities.total_changes(), 3);
        assert!(entities.modified_nodes.contains(&1));
        assert!(entities.modified_nodes.contains(&2));
        assert!(entities.modified_edges.contains(&100));
    }

    #[test]
    fn test_change_type_combine() {
        assert_eq!(
            ChangeType::Created.combine(ChangeType::Created),
            ChangeType::Created
        );
        assert_eq!(
            ChangeType::Created.combine(ChangeType::Deleted),
            ChangeType::Multiple
        );
    }
}
