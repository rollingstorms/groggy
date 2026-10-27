//! Temporal Index for efficient history-aware queries.
//!
//! The TemporalIndex enables fast lookups of:
//! - When nodes/edges were created/deleted
//! - Which nodes/edges existed at any commit
//! - Attribute value timelines for nodes/edges
//! - Neighbors at specific commits or within time windows
//!
//! This structure is built once from ChangeTracker history and provides
//! O(log n) or better performance for most temporal queries.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::errors::{GraphError, GraphResult};
use crate::state::history::{Delta, HistoryForest};
use crate::types::{AttrName, AttrValue, EdgeId, NodeId, StateId};
use std::sync::Arc;

/// Temporal index for efficient history queries.
///
/// Design principles:
/// - Build once from HistoryForest, then use for many queries
/// - Store lifetimes (creation/deletion commits) for all entities
/// - Index attribute changes chronologically for timeline queries
/// - Support range queries efficiently via sorted data structures
#[derive(Debug, Clone)]
pub struct TemporalIndex {
    provenance: Option<(Arc<()>, StateId)>,
    records: BTreeMap<StateId, Arc<Delta>>,
    node_intervals: HashMap<NodeId, Vec<(StateId, Option<StateId>)>>,
    edge_intervals: HashMap<EdgeId, Vec<(StateId, Option<StateId>)>>,
    /// Map from node ID to (creation_commit, deletion_commit).
    /// None for deletion_commit means node still exists.
    node_lifetime: HashMap<NodeId, (StateId, Option<StateId>)>,

    /// Map from edge ID to (creation_commit, deletion_commit).
    /// None for deletion_commit means edge still exists.
    edge_lifetime: HashMap<EdgeId, (StateId, Option<StateId>)>,

    /// Attribute change timeline: (node_id, attr_name) -> sorted vec of (commit, value).
    /// Sorted by commit ID for binary search.
    node_attr_timeline: HashMap<(NodeId, AttrName), Vec<(StateId, Option<AttrValue>)>>,

    /// Edge attribute change timeline: (edge_id, attr_name) -> sorted vec of (commit, value).
    edge_attr_timeline: HashMap<(EdgeId, AttrName), Vec<(StateId, Option<AttrValue>)>>,

    /// Map from edge ID to (source, target) nodes.
    /// This allows neighbor queries without reconstructing full state.
    edge_endpoints: HashMap<EdgeId, (NodeId, NodeId)>,
    /// Allocated only for edges with an actual endpoint revision.
    edge_endpoint_revisions: HashMap<EdgeId, Vec<(StateId, (NodeId, NodeId))>>,

    /// Commit timestamp index for range queries.
    /// Maps commit ID to timestamp for temporal window queries.
    commit_timestamps: BTreeMap<StateId, u64>,

    /// Reverse index: timestamp -> commit ID for time-based lookups.
    timestamp_to_commit: BTreeMap<u64, Vec<StateId>>,
}

impl TemporalIndex {
    /// Create a new empty temporal index.
    pub fn new() -> Self {
        Self {
            provenance: None,
            records: BTreeMap::new(),
            node_intervals: HashMap::new(),
            edge_intervals: HashMap::new(),
            node_lifetime: HashMap::new(),
            edge_lifetime: HashMap::new(),
            node_attr_timeline: HashMap::new(),
            edge_attr_timeline: HashMap::new(),
            edge_endpoints: HashMap::new(),
            edge_endpoint_revisions: HashMap::new(),
            commit_timestamps: BTreeMap::new(),
            timestamp_to_commit: BTreeMap::new(),
        }
    }

    /// Build index from HistoryForest.
    ///
    /// This processes all commits in chronological order, tracking:
    /// - When each node/edge was created
    /// - When each node/edge was deleted (if applicable)
    /// - All attribute changes with their commit IDs
    ///
    /// Time complexity: O(total_changes) where total_changes is the sum
    /// of all changes across all commits.
    pub fn from_history(history: &HistoryForest) -> GraphResult<Self> {
        let mut index = Self::new();

        // Get commits in chronological order
        let commits = history.get_commit_history();

        // Process each commit in order
        for commit in commits {
            index.process_commit(history, commit.id)?;
        }

        index.sort_timelines();
        Ok(index)
    }

    /// Build the retained first-parent lineage ending at an explicit state.
    /// This matches snapshot reconstruction and excludes sibling branches.
    pub fn from_history_at(history: &HistoryForest, head: StateId) -> GraphResult<Self> {
        Self::from_history_at_cancellable(history, head, || false)
    }

    pub(crate) fn from_history_at_cancellable(
        history: &HistoryForest,
        head: StateId,
        cancelled: impl Fn() -> bool,
    ) -> GraphResult<Self> {
        let mut ids = Vec::new();
        let mut seen = HashSet::new();
        let mut current = head;
        while current != 0 {
            if cancelled() {
                return Err(GraphError::InvalidInput("temporal index cancelled".into()));
            }
            if !seen.insert(current) {
                return Err(GraphError::InvalidInput("cyclic temporal history".into()));
            }
            let commit = history.get_commit(current)?;
            ids.push(current);
            current = commit.parents.first().copied().unwrap_or(0);
        }
        ids.reverse();
        if ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(GraphError::InvalidInput(
                "temporal state IDs must increase along the retained lineage".into(),
            ));
        }
        let mut index = Self::new();
        index.provenance = Some((Arc::clone(history.identity()), head));
        for id in ids {
            if cancelled() {
                return Err(GraphError::InvalidInput("temporal index cancelled".into()));
            }
            index.process_commit(history, id)?;
        }
        index.sort_timelines();
        if cancelled() {
            return Err(GraphError::InvalidInput("temporal index cancelled".into()));
        }
        Ok(index)
    }

    pub(crate) fn matches(&self, history: &HistoryForest, head: StateId) -> bool {
        self.provenance
            .as_ref()
            .is_some_and(|(identity, captured)| {
                Arc::ptr_eq(identity, history.identity()) && *captured == head
            })
    }

    pub(crate) fn validate_captured_head(&self, head: StateId) -> GraphResult<()> {
        if self
            .provenance
            .as_ref()
            .is_some_and(|(_, captured)| *captured == head)
        {
            Ok(())
        } else {
            Err(GraphError::InvalidInput(
                "temporal index has a different or unscoped captured head".into(),
            ))
        }
    }

    /// Validate a retained state coordinate on this captured lineage (0 is empty root).
    pub fn validate_state(&self, state: StateId) -> GraphResult<()> {
        if state == 0 || self.records.contains_key(&state) {
            Ok(())
        } else {
            Err(GraphError::InvalidInput(format!(
                "state {state} is outside the captured temporal lineage"
            )))
        }
    }

    /// Validate inclusive state-ID bounds without expanding missing numeric IDs.
    pub fn validate_range(&self, start: StateId, end: StateId) -> GraphResult<()> {
        if start > end {
            return Err(GraphError::InvalidInput(
                "temporal range start exceeds end".into(),
            ));
        }
        self.validate_state(start)?;
        self.validate_state(end)
    }

    pub(crate) fn changes_in_range(
        &self,
        start: StateId,
        end: StateId,
    ) -> impl Iterator<Item = &Delta> {
        self.records
            .range(start..=end)
            .map(|(_, delta)| delta.as_ref())
    }

    fn sort_timelines(&mut self) {
        for timeline in self
            .node_attr_timeline
            .values_mut()
            .chain(self.edge_attr_timeline.values_mut())
        {
            timeline.sort_by_key(|(id, _)| *id);
        }
    }

    /// Whether any activation interval overlaps the inclusive retained-state window.
    pub fn node_existed_in_window(&self, node: NodeId, start: StateId, end: StateId) -> bool {
        if start > end {
            return false;
        }
        self.node_intervals.get(&node).map_or_else(
            || {
                self.node_lifetime
                    .get(&node)
                    .is_some_and(|&(created, deleted)| {
                        created <= end && deleted.is_none_or(|deleted| start < deleted)
                    })
            },
            |intervals| {
                intervals.iter().any(|&(created, deleted)| {
                    created <= end && deleted.is_none_or(|deleted| start < deleted)
                })
            },
        )
    }

    /// Process a single commit and update the index.
    fn process_commit(&mut self, history: &HistoryForest, commit_id: StateId) -> GraphResult<()> {
        let commit = history.get_commit(commit_id)?;
        let delta = &commit.delta;
        let (added_nodes, removed_nodes, added_edges, removed_edges): (
            HashSet<_>,
            HashSet<_>,
            HashSet<_>,
            HashSet<_>,
        ) = if delta.edge_endpoint_changes.is_empty() {
            (
                HashSet::new(),
                HashSet::new(),
                HashSet::new(),
                HashSet::new(),
            )
        } else {
            (
                delta.nodes_added.iter().copied().collect(),
                delta.nodes_removed.iter().copied().collect(),
                delta.edges_added.iter().map(|e| e.0).collect(),
                delta.edges_removed.iter().copied().collect(),
            )
        };
        let mut revisions = HashSet::new();
        for change in &delta.edge_endpoint_changes {
            if !revisions.insert(change.edge_id)
                || !self.edge_exists_at(change.edge_id, commit_id.saturating_sub(1))
                || self.edge_endpoints.get(&change.edge_id).copied() != Some(change.old_endpoints)
                || added_edges.contains(&change.edge_id)
                || [change.new_endpoints.0, change.new_endpoints.1]
                    .iter()
                    .any(|&node| {
                        removed_nodes.contains(&node)
                            || !(added_nodes.contains(&node)
                                || self.node_exists_at(node, commit_id.saturating_sub(1)))
                    })
                || removed_edges.contains(&change.edge_id)
            {
                return Err(GraphError::InvalidInput(
                    "Invalid temporal endpoint revision".into(),
                ));
            }
        }
        self.records.insert(commit_id, Arc::clone(delta));

        // Record commit timestamp
        self.commit_timestamps.insert(commit_id, commit.timestamp);
        self.timestamp_to_commit
            .entry(commit.timestamp)
            .or_insert_with(Vec::new)
            .push(commit_id);

        // Track node creations
        for &node_id in &delta.nodes_added {
            self.node_lifetime
                .entry(node_id)
                .or_insert((commit_id, None))
                .1 = None;
            open_interval(&mut self.node_intervals, node_id, commit_id);
        }

        // Track node deletions
        for &node_id in &delta.nodes_removed {
            if let Some(lifetime) = self.node_lifetime.get_mut(&node_id) {
                lifetime.1 = Some(commit_id);
            }
            close_interval(&mut self.node_intervals, node_id, commit_id);
        }

        // Track edge creations
        for &(edge_id, source, target) in &delta.edges_added {
            self.edge_lifetime
                .entry(edge_id)
                .or_insert((commit_id, None))
                .1 = None;
            open_interval(&mut self.edge_intervals, edge_id, commit_id);
            if self
                .edge_endpoints
                .get(&edge_id)
                .is_some_and(|existing| *existing != (source, target))
            {
                return Err(GraphError::InvalidInput(
                    "temporal index does not support endpoint rewiring".into(),
                ));
            }
            self.edge_endpoints.insert(edge_id, (source, target));
        }

        for c in &delta.edge_endpoint_changes {
            self.edge_endpoint_revisions
                .entry(c.edge_id)
                .or_insert_with(|| vec![(0, c.old_endpoints)])
                .push((commit_id, c.new_endpoints));
            self.edge_endpoints.insert(c.edge_id, c.new_endpoints);
        }

        // Track edge deletions
        for &edge_id in &delta.edges_removed {
            if let Some(lifetime) = self.edge_lifetime.get_mut(&edge_id) {
                lifetime.1 = Some(commit_id);
            }
            close_interval(&mut self.edge_intervals, edge_id, commit_id);
        }

        // Track node attribute changes
        for &(node_id, ref attr_name, ref _old, ref new_value) in &delta.node_attr_changes {
            let key = (node_id, attr_name.clone());
            self.node_attr_timeline
                .entry(key)
                .or_insert_with(Vec::new)
                .push((commit_id, new_value.clone()));
        }

        // Track edge attribute changes
        for &(edge_id, ref attr_name, ref _old, ref new_value) in &delta.edge_attr_changes {
            let key = (edge_id, attr_name.clone());
            self.edge_attr_timeline
                .entry(key)
                .or_insert_with(Vec::new)
                .push((commit_id, new_value.clone()));
        }

        Ok(())
    }

    /// Check if a node existed at a specific commit.
    pub fn node_exists_at(&self, node_id: NodeId, commit_id: StateId) -> bool {
        if let Some(intervals) = self.node_intervals.get(&node_id) {
            return intervals.iter().any(|&(created, deleted)| {
                created <= commit_id && deleted.is_none_or(|deleted| commit_id < deleted)
            });
        }
        self.node_lifetime
            .get(&node_id)
            .is_some_and(|&(created, deleted)| {
                created <= commit_id && deleted.is_none_or(|deleted| commit_id < deleted)
            })
    }

    /// Return the lifetime range (creation, optional deletion) for a node.
    pub fn node_lifetime_range(&self, node_id: NodeId) -> Option<(StateId, Option<StateId>)> {
        self.node_lifetime
            .get(&node_id)
            .map(|(created, deleted)| (*created, *deleted))
    }

    /// Get the commit where a node was created.
    pub fn node_creation_commit(&self, node_id: NodeId) -> Option<StateId> {
        self.node_lifetime
            .get(&node_id)
            .map(|(created, _)| *created)
    }

    /// Get the commit where a node was deleted, if ever.
    pub fn node_deletion_commit(&self, node_id: NodeId) -> Option<StateId> {
        self.node_lifetime
            .get(&node_id)
            .and_then(|(_, deleted)| *deleted)
    }

    /// Check if an edge existed at a specific commit.
    pub fn edge_exists_at(&self, edge_id: EdgeId, commit_id: StateId) -> bool {
        if let Some(intervals) = self.edge_intervals.get(&edge_id) {
            return intervals.iter().any(|&(created, deleted)| {
                created <= commit_id && deleted.is_none_or(|deleted| commit_id < deleted)
            });
        }
        self.edge_lifetime
            .get(&edge_id)
            .is_some_and(|&(created, deleted)| {
                created <= commit_id && deleted.is_none_or(|deleted| commit_id < deleted)
            })
    }

    /// Get all nodes that existed at a specific commit.
    pub fn nodes_at_commit(&self, commit_id: StateId) -> Vec<NodeId> {
        self.node_lifetime
            .iter()
            .filter(|&(&id, _)| self.node_exists_at(id, commit_id))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get all edges that existed at a specific commit.
    pub fn edges_at_commit(&self, commit_id: StateId) -> Vec<EdgeId> {
        self.edge_lifetime
            .iter()
            .filter(|&(&id, _)| self.edge_exists_at(id, commit_id))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get neighbors of a node at a specific commit.
    ///
    /// Returns the IDs of all nodes connected to the given node at the
    /// specified commit, considering only edges that existed at that time.
    pub fn neighbors_at_commit(&self, node_id: NodeId, commit_id: StateId) -> Vec<NodeId> {
        let mut neighbors = Vec::new();

        for (&edge_id, &pair) in &self.edge_endpoints {
            let (source, target) = if let Some(values) = self.edge_endpoint_revisions.get(&edge_id)
            {
                let position = values.partition_point(|(state, _)| *state <= commit_id);
                let Some((_, pair)) = position.checked_sub(1).and_then(|i| values.get(i)).copied()
                else {
                    continue;
                };
                pair
            } else {
                pair
            };
            // Check if edge existed at this commit
            if !self.edge_exists_at(edge_id, commit_id) {
                continue;
            }

            // Add neighbor if this edge connects to our node
            if source == node_id {
                neighbors.push(target);
            } else if target == node_id {
                neighbors.push(source);
            }
        }

        neighbors.sort_unstable();
        neighbors
    }

    /// Bulk neighbor query at a specific commit.
    ///
    /// Returns neighbors for multiple nodes efficiently.
    pub fn neighbors_bulk_at_commit(
        &self,
        nodes: &[NodeId],
        commit_id: StateId,
    ) -> HashMap<NodeId, Vec<NodeId>> {
        let mut result = HashMap::new();

        for &node in nodes {
            let neighbors = self.neighbors_at_commit(node, commit_id);
            result.insert(node, neighbors);
        }

        result
    }

    /// Get neighbors that existed at any point within a commit range.
    ///
    /// Returns all neighbors that were connected during [start_commit, end_commit],
    /// even if they weren't connected for the entire window.
    pub fn neighbors_in_window(
        &self,
        node_id: NodeId,
        start_commit: StateId,
        end_commit: StateId,
    ) -> Vec<NodeId> {
        let mut neighbors = std::collections::HashSet::new();

        for (&edge_id, &pair) in &self.edge_endpoints {
            let fixed = (0, pair);
            let values = self
                .edge_endpoint_revisions
                .get(&edge_id)
                .map(Vec::as_slice)
                .unwrap_or_else(|| std::slice::from_ref(&fixed));
            let Some(intervals) = self.edge_intervals.get(&edge_id) else {
                continue;
            };
            let (mut i, mut j) = (0, 0);
            while start_commit <= end_commit && i < values.len() && j < intervals.len() {
                let (revision, (source, target)) = values[i];
                let revision_end = values.get(i + 1).map(|v| v.0);
                let (created, deleted) = intervals[j];
                let start = start_commit.max(revision).max(created);
                if start <= end_commit
                    && revision_end.is_none_or(|end| start < end)
                    && deleted.is_none_or(|end| start < end)
                {
                    if source == node_id {
                        neighbors.insert(target);
                    } else if target == node_id {
                        neighbors.insert(source);
                    }
                }
                match (revision_end, deleted) {
                    (None, None) => break,
                    (Some(_), None) => i += 1,
                    (None, Some(_)) => j += 1,
                    (Some(a), Some(b)) if a <= b => i += 1,
                    _ => j += 1,
                }
            }
        }

        let mut neighbors: Vec<_> = neighbors.into_iter().collect();
        neighbors.sort_unstable();
        neighbors
    }

    /// Get attribute value for a node at a specific commit.
    ///
    /// Uses binary search on the attribute timeline to find the most recent
    /// value at or before the given commit.
    pub fn node_attr_at_commit(
        &self,
        node_id: NodeId,
        attr: &AttrName,
        commit_id: StateId,
    ) -> Option<AttrValue> {
        if !self.node_exists_at(node_id, commit_id) {
            return None;
        }
        let key = (node_id, attr.clone());
        let timeline = self.node_attr_timeline.get(&key)?;

        let activation = self
            .node_intervals
            .get(&node_id)
            .and_then(|rows| {
                rows.iter().rev().find(|&&(start, end)| {
                    start <= commit_id && end.is_none_or(|end| commit_id < end)
                })
            })
            .map(|(start, _)| *start)
            .unwrap_or(0);
        timeline
            .partition_point(|(state, _)| *state <= commit_id)
            .checked_sub(1)
            .and_then(|index| {
                (timeline[index].0 >= activation)
                    .then(|| timeline[index].1.clone())
                    .flatten()
            })
    }

    /// Get attribute timeline for a node within a commit range.
    ///
    /// Returns all attribute changes that occurred during [from_commit, to_commit].
    pub fn node_attr_history(
        &self,
        node_id: NodeId,
        attr: &AttrName,
        from_commit: StateId,
        to_commit: StateId,
    ) -> Vec<(StateId, Option<AttrValue>)> {
        self.node_attr_history_slice(node_id, attr, from_commit, to_commit)
            .to_vec()
    }

    pub(crate) fn node_attr_history_slice(
        &self,
        node: NodeId,
        attr: &AttrName,
        start: StateId,
        end: StateId,
    ) -> &[(StateId, Option<AttrValue>)] {
        if start > end {
            return &[];
        }
        let Some(rows) = self.node_attr_timeline.get(&(node, attr.clone())) else {
            return &[];
        };
        let first = rows.partition_point(|(state, _)| *state < start);
        let last = rows.partition_point(|(state, _)| *state <= end);
        &rows[first..last]
    }

    /// Get edge attribute value at a specific commit.
    pub fn edge_attr_at_commit(
        &self,
        edge_id: EdgeId,
        attr: &AttrName,
        commit_id: StateId,
    ) -> Option<AttrValue> {
        if !self.edge_exists_at(edge_id, commit_id) {
            return None;
        }
        let key = (edge_id, attr.clone());
        let timeline = self.edge_attr_timeline.get(&key)?;

        let activation = self
            .edge_intervals
            .get(&edge_id)
            .and_then(|rows| {
                rows.iter().rev().find(|&&(start, end)| {
                    start <= commit_id && end.is_none_or(|end| commit_id < end)
                })
            })
            .map(|(start, _)| *start)
            .unwrap_or(0);
        timeline
            .partition_point(|(state, _)| *state <= commit_id)
            .checked_sub(1)
            .and_then(|index| {
                (timeline[index].0 >= activation)
                    .then(|| timeline[index].1.clone())
                    .flatten()
            })
    }

    /// Get edge attribute timeline within a commit range.
    pub fn edge_attr_history(
        &self,
        edge_id: EdgeId,
        attr: &AttrName,
        from_commit: StateId,
        to_commit: StateId,
    ) -> Vec<(StateId, Option<AttrValue>)> {
        let key = (edge_id, attr.clone());
        let timeline = match self.edge_attr_timeline.get(&key) {
            Some(t) => t,
            None => return Vec::new(),
        };

        timeline
            .iter()
            .filter(|&&(commit, _)| commit >= from_commit && commit <= to_commit)
            .cloned()
            .collect()
    }

    /// Find commits within a timestamp range.
    pub fn commits_in_time_range(&self, start_ts: u64, end_ts: u64) -> Vec<StateId> {
        self.commit_timestamps
            .iter()
            .filter(|&(_, &ts)| ts >= start_ts && ts <= end_ts)
            .map(|(&cid, _)| cid)
            .collect()
    }

    /// Get all nodes that changed (created, deleted, or had attributes modified) in a commit.
    pub fn nodes_changed_in_commit(&self, commit_id: StateId) -> Vec<NodeId> {
        let mut changed = HashSet::new();
        if let Some(delta) = self.records.get(&commit_id) {
            changed.extend(delta.nodes_added.iter().map(|entry| *entry));
            changed.extend(delta.nodes_removed.iter().copied());
            changed.extend(delta.node_attr_changes.iter().map(|(id, _, _, _)| *id));
        }
        let mut ids: Vec<_> = changed.into_iter().collect();
        ids.sort_unstable();
        ids
    }

    /// Get all nodes that changed (created, deleted, or modified) within a commit range.
    pub fn nodes_changed_in_range(&self, start: StateId, end: StateId) -> Vec<NodeId> {
        if start > end {
            return Vec::new();
        }
        let mut changed = HashSet::new();
        for (&commit, _) in self.records.range(start..=end) {
            changed.extend(self.nodes_changed_in_commit(commit));
        }
        let mut ids: Vec<_> = changed.into_iter().collect();
        ids.sort_unstable();
        ids
    }

    /// Get all edges that changed in a commit.
    pub fn edges_changed_in_commit(&self, commit_id: StateId) -> Vec<EdgeId> {
        let mut changed = HashSet::new();
        if let Some(delta) = self.records.get(&commit_id) {
            changed.extend(delta.edges_added.iter().map(|entry| entry.0));
            changed.extend(delta.edges_removed.iter().copied());
            changed.extend(delta.edge_attr_changes.iter().map(|(id, _, _, _)| *id));
            changed.extend(delta.edge_endpoint_changes.iter().map(|c| c.edge_id));
        }
        let mut ids: Vec<_> = changed.into_iter().collect();
        ids.sort_unstable();
        ids
    }

    /// Get statistics about the index.
    pub fn statistics(&self) -> IndexStatistics {
        IndexStatistics {
            total_nodes: self.node_lifetime.len(),
            total_edges: self.edge_lifetime.len(),
            total_commits: self.commit_timestamps.len(),
            node_attr_timelines: self.node_attr_timeline.len(),
            edge_attr_timelines: self.edge_attr_timeline.len(),
        }
    }
}

fn open_interval(
    intervals: &mut HashMap<usize, Vec<(StateId, Option<StateId>)>>,
    id: usize,
    state: StateId,
) {
    let rows = intervals.entry(id).or_default();
    if rows.last().is_none_or(|(_, end)| end.is_some()) {
        rows.push((state, None));
    }
}
fn close_interval(
    intervals: &mut HashMap<usize, Vec<(StateId, Option<StateId>)>>,
    id: usize,
    state: StateId,
) {
    if let Some((_, end)) = intervals.get_mut(&id).and_then(|rows| rows.last_mut()) {
        *end = Some(state);
    }
}

impl Default for TemporalIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the temporal index.
#[derive(Debug, Clone)]
pub struct IndexStatistics {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub total_commits: usize,
    pub node_attr_timelines: usize,
    pub edge_attr_timelines: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::state::change_tracker::ChangeSet;

    #[test]
    fn scoped_index_tracks_reactivation_gaps_and_exclusive_deletion() {
        let mut history = HistoryForest::new();
        let mut changes = ChangeSet::new();
        changes.nodes_added = vec![0, 1];
        changes.edges_added = vec![(0, 0, 1)];
        changes.node_attr_changes = vec![(0, "x".into(), None, Some(AttrValue::Int(1)))];
        let first = history
            .create_commit(changes, "birth".into(), "test".into(), None)
            .unwrap();
        let mut changes = ChangeSet::new();
        changes.nodes_removed = vec![0];
        changes.edges_removed = vec![0];
        let deleted = history
            .create_commit(changes, "delete".into(), "test".into(), Some(first))
            .unwrap();
        let gap = history
            .create_commit(ChangeSet::new(), "gap".into(), "test".into(), Some(deleted))
            .unwrap();
        let mut changes = ChangeSet::new();
        changes.nodes_added = vec![0];
        changes.edges_added = vec![(0, 0, 1)];
        let restored = history
            .create_commit(changes, "restore".into(), "test".into(), Some(gap))
            .unwrap();
        let index = TemporalIndex::from_history_at(&history, restored).unwrap();
        assert!(index.node_exists_at(0, first));
        assert!(!index.node_exists_at(0, deleted));
        assert!(!index.node_exists_at(0, gap));
        assert!(index.node_exists_at(0, restored));
        assert!(!index.edge_exists_at(0, gap));
        assert!(index.edge_exists_at(0, restored));
        assert!(!index.node_existed_in_window(0, deleted, gap));
        assert!(index.node_existed_in_window(0, gap, restored));
        assert!(index.neighbors_in_window(1, deleted, gap).is_empty());
        assert_eq!(index.neighbors_in_window(1, restored, restored), vec![0]);
        assert_eq!(index.node_attr_at_commit(0, &"x".into(), restored), None);
        assert_eq!(index.node_creation_commit(0), Some(first));
        assert_eq!(index.nodes_changed_in_commit(restored), vec![0]);
        assert_eq!(index.nodes_changed_in_range(0, u64::MAX), vec![0, 1]);
    }

    #[test]
    fn scoped_index_excludes_siblings_and_validates_sparse_bounds_and_provenance() {
        let mut history = HistoryForest::new();
        let mut changes = ChangeSet::new();
        changes.nodes_added = vec![0];
        let first = history
            .create_commit(changes, "root".into(), "test".into(), None)
            .unwrap();
        let mut changes = ChangeSet::new();
        changes.nodes_added = vec![1];
        let sibling = history
            .create_commit(changes, "sibling".into(), "test".into(), Some(first))
            .unwrap();
        let mut changes = ChangeSet::new();
        changes.nodes_added = vec![2];
        let head = history
            .create_commit(changes, "head".into(), "test".into(), Some(first))
            .unwrap();
        let index = TemporalIndex::from_history_at(&history, head).unwrap();
        assert!(index.matches(&history, head));
        assert!(!index.matches(&history, sibling));
        assert!(!index.matches(&HistoryForest::new(), head));
        assert_eq!(index.nodes_changed_in_range(first, head), vec![0, 2]);
        assert!(!index.node_exists_at(1, head));
        assert!(index.validate_range(0, head).is_ok());
        assert!(index.validate_range(sibling, head).is_err());
        assert!(index.validate_range(head, first).is_err());
        assert!(index.validate_state(u64::MAX).is_err());
        assert!(!TemporalIndex::from_history(&history)
            .unwrap()
            .matches(&history, head));
    }

    #[test]
    fn scoped_index_build_checks_cancellation_without_returning_partial_index() {
        use std::cell::Cell;
        let mut history = HistoryForest::new();
        let mut parent = None;
        for _ in 0..5 {
            parent = Some(
                history
                    .create_commit(ChangeSet::new(), "state".into(), "test".into(), parent)
                    .unwrap(),
            );
        }
        let checks = Cell::new(0);
        assert!(
            TemporalIndex::from_history_at_cancellable(&history, parent.unwrap(), || {
                checks.set(checks.get() + 1);
                checks.get() > 3
            })
            .is_err()
        );
        assert_eq!(checks.get(), 4);
    }

    #[test]
    fn test_empty_index() {
        let index = TemporalIndex::new();
        assert!(!index.node_exists_at(1, 1));
        assert!(!index.edge_exists_at(1, 1));
        assert_eq!(index.nodes_at_commit(1).len(), 0);
    }

    #[test]
    fn test_node_lifetime_tracking() {
        let mut index = TemporalIndex::new();

        // Node 1 created at commit 5
        index.node_lifetime.insert(1, (5, None));

        assert!(!index.node_exists_at(1, 4)); // Before creation
        assert!(index.node_exists_at(1, 5)); // At creation
        assert!(index.node_exists_at(1, 10)); // After creation

        // Node 1 deleted at commit 15
        index.node_lifetime.insert(1, (5, Some(15)));

        assert!(index.node_exists_at(1, 10)); // Before deletion
        assert!(!index.node_exists_at(1, 15)); // At deletion
        assert!(!index.node_exists_at(1, 20)); // After deletion
    }

    #[test]
    fn test_attribute_timeline() {
        let mut index = TemporalIndex::new();
        let node_id = 1;
        let attr = "status".to_string();

        // Add attribute changes
        let key = (node_id, attr.clone());
        index.node_lifetime.insert(node_id, (0, None));
        index.node_attr_timeline.insert(
            key.clone(),
            vec![
                (5, Some(AttrValue::Text("active".into()))),
                (10, Some(AttrValue::Text("inactive".into()))),
                (15, Some(AttrValue::Text("active".into()))),
            ],
        );

        // Query at different commits
        assert_eq!(
            index.node_attr_at_commit(node_id, &attr, 3),
            None // Before first change
        );
        assert_eq!(
            index.node_attr_at_commit(node_id, &attr, 5),
            Some(AttrValue::Text("active".into()))
        );
        assert_eq!(
            index.node_attr_at_commit(node_id, &attr, 12),
            Some(AttrValue::Text("inactive".into()))
        );
        assert_eq!(
            index.node_attr_at_commit(node_id, &attr, 20),
            Some(AttrValue::Text("active".into()))
        );
    }
    #[test]
    fn endpoint_timelines_allocate_only_for_revised_edges() {
        let mut history = crate::HistoryForest::new();
        let root = history
            .create_commit(
                crate::state::change_tracker::ChangeSet {
                    nodes_added: vec![0, 1, 2],
                    edges_added: vec![(0, 0, 1), (1, 1, 2)],
                    ..crate::state::change_tracker::ChangeSet::new()
                },
                "root".into(),
                "test".into(),
                None,
            )
            .unwrap();
        let plain = TemporalIndex::from_history(&history).unwrap();
        assert!(plain.edge_endpoint_revisions.is_empty());
        history
            .create_commit(
                crate::state::change_tracker::ChangeSet {
                    edge_endpoint_changes: vec![crate::EdgeEndpointChange {
                        edge_id: 0,
                        old_endpoints: (0, 1),
                        new_endpoints: (2, 1),
                    }],
                    ..crate::state::change_tracker::ChangeSet::new()
                },
                "revision".into(),
                "test".into(),
                Some(root),
            )
            .unwrap();
        let revised = TemporalIndex::from_history(&history).unwrap();
        assert_eq!(revised.edge_endpoint_revisions.len(), 1);
        assert!(!revised.edge_endpoint_revisions.contains_key(&1));
        assert_eq!(revised.neighbors_at_commit(0, root), vec![1]);
    }
}
