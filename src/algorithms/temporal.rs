//! Temporal extensions for algorithm context.
//!
//! This module provides temporal awareness to algorithms, enabling:
//! - Time-scoped algorithm execution
//! - Delta computation between snapshots
//! - Change tracking within time windows
//! - Temporal metadata access during pipeline execution

use std::collections::HashSet;

use crate::errors::GraphResult;
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
    /// This compares the existence sets and attributes of both snapshots
    /// to identify all changes that occurred.
    pub fn compute(prev: &TemporalSnapshot, cur: &TemporalSnapshot) -> GraphResult<Self> {
        let from_commit = prev.lineage().commit_id;
        let to_commit = cur.lineage().commit_id;

        // Compute node changes
        let prev_nodes: HashSet<NodeId> = prev.existence().nodes().clone();
        let cur_nodes: HashSet<NodeId> = cur.existence().nodes().clone();

        let nodes_added: Vec<NodeId> = cur_nodes.difference(&prev_nodes).copied().collect();
        let nodes_removed: Vec<NodeId> = prev_nodes.difference(&cur_nodes).copied().collect();

        // Compute edge changes
        let prev_edges: HashSet<EdgeId> = prev.existence().edges().clone();
        let cur_edges: HashSet<EdgeId> = cur.existence().edges().clone();

        let edges_added: Vec<EdgeId> = cur_edges.difference(&prev_edges).copied().collect();
        let edges_removed: Vec<EdgeId> = prev_edges.difference(&cur_edges).copied().collect();

        // Note: Attribute change detection would require iterating through all
        // attributes on all nodes/edges that exist in both snapshots.
        // For now, we'll leave these empty as a placeholder for future implementation.
        let node_attr_changes = Vec::new();
        let edge_attr_changes = Vec::new();

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
        self.node_change_types.insert(node_id, change_type);
    }

    /// Add an edge change.
    pub fn add_edge(&mut self, edge_id: EdgeId, change_type: ChangeType) {
        self.modified_edges.insert(edge_id);
        self.edge_change_types.insert(edge_id, change_type);
    }

    /// Merge another ChangedEntities into this one.
    pub fn merge(&mut self, other: ChangedEntities) {
        self.modified_nodes.extend(other.modified_nodes);
        self.modified_edges.extend(other.modified_edges);
        self.node_change_types.extend(other.node_change_types);
        self.edge_change_types.extend(other.edge_change_types);
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
