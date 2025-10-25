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

use crate::errors::GraphResult;
use crate::state::history::HistoryForest;
use crate::types::{AttrName, AttrValue, EdgeId, NodeId, StateId};

/// Temporal index for efficient history queries.
///
/// Design principles:
/// - Build once from HistoryForest, then use for many queries
/// - Store lifetimes (creation/deletion commits) for all entities
/// - Index attribute changes chronologically for timeline queries
/// - Support range queries efficiently via sorted data structures
#[derive(Debug, Clone)]
pub struct TemporalIndex {
    /// Map from node ID to (creation_commit, deletion_commit).
    /// None for deletion_commit means node still exists.
    node_lifetime: HashMap<NodeId, (StateId, Option<StateId>)>,

    /// Map from edge ID to (creation_commit, deletion_commit).
    /// None for deletion_commit means edge still exists.
    edge_lifetime: HashMap<EdgeId, (StateId, Option<StateId>)>,

    /// Attribute change timeline: (node_id, attr_name) -> sorted vec of (commit, value).
    /// Sorted by commit ID for binary search.
    node_attr_timeline: HashMap<(NodeId, AttrName), Vec<(StateId, AttrValue)>>,

    /// Edge attribute change timeline: (edge_id, attr_name) -> sorted vec of (commit, value).
    edge_attr_timeline: HashMap<(EdgeId, AttrName), Vec<(StateId, AttrValue)>>,

    /// Map from edge ID to (source, target) nodes.
    /// This allows neighbor queries without reconstructing full state.
    edge_endpoints: HashMap<EdgeId, (NodeId, NodeId)>,

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
            node_lifetime: HashMap::new(),
            edge_lifetime: HashMap::new(),
            node_attr_timeline: HashMap::new(),
            edge_attr_timeline: HashMap::new(),
            edge_endpoints: HashMap::new(),
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

        Ok(index)
    }

    /// Process a single commit and update the index.
    fn process_commit(&mut self, history: &HistoryForest, commit_id: StateId) -> GraphResult<()> {
        let commit = history.get_commit(commit_id)?;
        let delta = &commit.delta;

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
                .or_insert((commit_id, None));
        }

        // Track node deletions
        for &node_id in &delta.nodes_removed {
            if let Some(lifetime) = self.node_lifetime.get_mut(&node_id) {
                lifetime.1 = Some(commit_id);
            }
        }

        // Track edge creations
        for &(edge_id, source, target) in &delta.edges_added {
            self.edge_lifetime
                .entry(edge_id)
                .or_insert((commit_id, None));
            self.edge_endpoints.insert(edge_id, (source, target));
        }

        // Track edge deletions
        for &edge_id in &delta.edges_removed {
            if let Some(lifetime) = self.edge_lifetime.get_mut(&edge_id) {
                lifetime.1 = Some(commit_id);
            }
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
        if let Some(&(created, deleted)) = self.node_lifetime.get(&node_id) {
            created <= commit_id && deleted.map_or(true, |d| commit_id < d)
        } else {
            false
        }
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
        if let Some(&(created, deleted)) = self.edge_lifetime.get(&edge_id) {
            created <= commit_id && deleted.map_or(true, |d| commit_id < d)
        } else {
            false
        }
    }

    /// Get all nodes that existed at a specific commit.
    pub fn nodes_at_commit(&self, commit_id: StateId) -> Vec<NodeId> {
        self.node_lifetime
            .iter()
            .filter(|&(_id, &(created, deleted))| {
                created <= commit_id && deleted.map_or(true, |d| commit_id < d)
            })
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get all edges that existed at a specific commit.
    pub fn edges_at_commit(&self, commit_id: StateId) -> Vec<EdgeId> {
        self.edge_lifetime
            .iter()
            .filter(|&(_id, &(created, deleted))| {
                created <= commit_id && deleted.map_or(true, |d| commit_id < d)
            })
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get neighbors of a node at a specific commit.
    ///
    /// Returns the IDs of all nodes connected to the given node at the
    /// specified commit, considering only edges that existed at that time.
    pub fn neighbors_at_commit(&self, node_id: NodeId, commit_id: StateId) -> Vec<NodeId> {
        let mut neighbors = Vec::new();

        for (&edge_id, &(source, target)) in &self.edge_endpoints {
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

        for (&edge_id, &(source, target)) in &self.edge_endpoints {
            if let Some(&(created, deleted)) = self.edge_lifetime.get(&edge_id) {
                // Check if edge existed at any point in the window
                let edge_start = created;
                let edge_end = deleted.unwrap_or(StateId::MAX);

                // Intervals overlap if: edge_start <= end_commit AND start_commit <= edge_end
                if edge_start <= end_commit && start_commit <= edge_end {
                    // This edge was active during the window
                    if source == node_id {
                        neighbors.insert(target);
                    } else if target == node_id {
                        neighbors.insert(source);
                    }
                }
            }
        }

        neighbors.into_iter().collect()
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
        let key = (node_id, attr.clone());
        let timeline = self.node_attr_timeline.get(&key)?;

        // Binary search for the most recent change at or before commit_id
        match timeline.binary_search_by_key(&commit_id, |&(cid, _)| cid) {
            Ok(idx) => Some(timeline[idx].1.clone()),
            Err(0) => None, // No changes before this commit
            Err(idx) => Some(timeline[idx - 1].1.clone()),
        }
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
    ) -> Vec<(StateId, AttrValue)> {
        let key = (node_id, attr.clone());
        let timeline = match self.node_attr_timeline.get(&key) {
            Some(t) => t,
            None => return Vec::new(),
        };

        timeline
            .iter()
            .filter(|&&(commit, _)| commit >= from_commit && commit <= to_commit)
            .cloned()
            .collect()
    }

    /// Get edge attribute value at a specific commit.
    pub fn edge_attr_at_commit(
        &self,
        edge_id: EdgeId,
        attr: &AttrName,
        commit_id: StateId,
    ) -> Option<AttrValue> {
        let key = (edge_id, attr.clone());
        let timeline = self.edge_attr_timeline.get(&key)?;

        match timeline.binary_search_by_key(&commit_id, |&(cid, _)| cid) {
            Ok(idx) => Some(timeline[idx].1.clone()),
            Err(0) => None,
            Err(idx) => Some(timeline[idx - 1].1.clone()),
        }
    }

    /// Get edge attribute timeline within a commit range.
    pub fn edge_attr_history(
        &self,
        edge_id: EdgeId,
        attr: &AttrName,
        from_commit: StateId,
        to_commit: StateId,
    ) -> Vec<(StateId, AttrValue)> {
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
        let mut changed = std::collections::HashSet::new();

        // Nodes created at this commit
        for (&node_id, &(created, _)) in &self.node_lifetime {
            if created == commit_id {
                changed.insert(node_id);
            }
        }

        // Nodes deleted at this commit
        for (&node_id, &(_, deleted)) in &self.node_lifetime {
            if deleted == Some(commit_id) {
                changed.insert(node_id);
            }
        }

        // Nodes with attribute changes at this commit
        for ((node_id, _), timeline) in &self.node_attr_timeline {
            if timeline.iter().any(|&(cid, _)| cid == commit_id) {
                changed.insert(*node_id);
            }
        }

        changed.into_iter().collect()
    }

    /// Get all nodes that changed (created, deleted, or modified) within a commit range.
    pub fn nodes_changed_in_range(&self, start: StateId, end: StateId) -> Vec<NodeId> {
        if start > end {
            return Vec::new();
        }
        let mut changed = HashSet::new();
        for commit in start..=end {
            for node in self.nodes_changed_in_commit(commit) {
                changed.insert(node);
            }
        }
        changed.into_iter().collect()
    }

    /// Get all edges that changed in a commit.
    pub fn edges_changed_in_commit(&self, commit_id: StateId) -> Vec<EdgeId> {
        let mut changed = std::collections::HashSet::new();

        // Edges created at this commit
        for (&edge_id, &(created, _)) in &self.edge_lifetime {
            if created == commit_id {
                changed.insert(edge_id);
            }
        }

        // Edges deleted at this commit
        for (&edge_id, &(_, deleted)) in &self.edge_lifetime {
            if deleted == Some(commit_id) {
                changed.insert(edge_id);
            }
        }

        // Edges with attribute changes at this commit
        for ((edge_id, _), timeline) in &self.edge_attr_timeline {
            if timeline.iter().any(|&(cid, _)| cid == commit_id) {
                changed.insert(*edge_id);
            }
        }

        changed.into_iter().collect()
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
        index.node_attr_timeline.insert(
            key.clone(),
            vec![
                (5, AttrValue::Text("active".into())),
                (10, AttrValue::Text("inactive".into())),
                (15, AttrValue::Text("active".into())),
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
}
