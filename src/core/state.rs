//! State objects, snapshots, and reconstruction for graph history tracking.
//!
//! ARCHITECTURE ROLE:
//! This module provides the complete state management system for graph history.
//! It includes both the historical state metadata (StateObject) and the complete
//! graph state representation (GraphSnapshot) with reconstruction algorithms.
//!
//! DESIGN PHILOSOPHY:
//! - StateObject: Metadata and delta storage for efficient history
//! - GraphSnapshot: Complete state representation for fast access
//! - Reconstruction: Algorithms to build snapshots from deltas
//! - State comparison: Diffing and analysis utilities

use crate::core::delta::DeltaObject;
use crate::core::history::Delta;
use crate::errors::{GraphError, GraphResult};
use crate::types::{AttrName, AttrValue, EdgeId, NodeId, StateId};
use crate::util::timestamp_now;
use std::collections::HashMap;
use std::sync::Arc;

/// Immutable state object - a point in the graph's history
#[derive(Debug, Clone)]
pub struct StateObject {
    /// Parent state (None for root)
    pub parent: Option<StateId>,
    /// Changes from parent
    pub delta: Arc<DeltaObject>,
    /// Metadata
    pub metadata: Arc<StateMetadata>,
}

/// Metadata associated with a state
#[derive(Debug, Clone)]
pub struct StateMetadata {
    /// Human-readable label
    pub label: String,
    /// When this state was created (Unix timestamp)
    pub timestamp: u64,
    /// Who created this state
    pub author: String,
    /// Content hash for verification/deduplication
    pub hash: [u8; 32],
    /// Optional commit message
    pub message: Option<String>,
    /// Tags associated with this state
    pub tags: Vec<String>,
}

impl StateObject {
    /// Create a new state object
    pub fn new(
        parent: Option<StateId>,
        delta: DeltaObject,
        label: String,
        author: String,
        message: Option<String>,
    ) -> Self {
        let metadata = StateMetadata {
            label,
            timestamp: timestamp_now(),
            author,
            hash: delta.content_hash,
            message,
            tags: Vec::new(),
        };

        Self {
            parent,
            delta: Arc::new(delta),
            metadata: Arc::new(metadata),
        }
    }

    /// Create a root state (no parent)
    pub fn new_root(delta: DeltaObject, label: String, author: String) -> Self {
        Self::new(None, delta, label, author, None)
    }

    /// Get the parent state ID
    pub fn parent(&self) -> Option<StateId> {
        self.parent
    }

    /// Get the delta object
    pub fn delta(&self) -> &DeltaObject {
        &self.delta
    }

    /// Get the metadata
    pub fn metadata(&self) -> &StateMetadata {
        &self.metadata
    }

    /// Check if this is a root state (no parent)
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// Get the content hash
    pub fn content_hash(&self) -> [u8; 32] {
        self.metadata.hash
    }

    /// Get the timestamp
    pub fn timestamp(&self) -> u64 {
        self.metadata.timestamp
    }

    /// Get the author
    pub fn author(&self) -> &str {
        &self.metadata.author
    }

    /// Get the label
    pub fn label(&self) -> &str {
        &self.metadata.label
    }

    /// Get the commit message
    pub fn message(&self) -> Option<&str> {
        self.metadata.message.as_deref()
    }

    /// Get tags
    pub fn tags(&self) -> &[String] {
        &self.metadata.tags
    }

    /// Add a tag to this state's metadata
    pub fn add_tag(&mut self, tag: String) {
        // Since metadata is Arc, we need to clone and modify
        let mut metadata = (*self.metadata).clone();
        metadata.tags.push(tag);
        self.metadata = Arc::new(metadata);
    }

    /// Remove a tag from this state's metadata
    pub fn remove_tag(&mut self, tag: &str) {
        let mut metadata = (*self.metadata).clone();
        metadata.tags.retain(|t| t != tag);
        self.metadata = Arc::new(metadata);
    }

    /// Check if this state has a specific tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.metadata.tags.contains(&tag.to_string())
    }

    /// Update the label
    pub fn set_label(&mut self, label: String) {
        let mut metadata = (*self.metadata).clone();
        metadata.label = label;
        self.metadata = Arc::new(metadata);
    }

    /// Update the message
    pub fn set_message(&mut self, message: Option<String>) {
        let mut metadata = (*self.metadata).clone();
        metadata.message = message;
        self.metadata = Arc::new(metadata);
    }

    /// Get the size of this state's delta in terms of change count
    pub fn delta_size(&self) -> usize {
        self.delta.change_count()
    }

    /// Check if this state represents an empty delta
    pub fn is_empty_delta(&self) -> bool {
        self.delta.is_empty()
    }
}

impl StateMetadata {
    /// Create new metadata
    pub fn new(label: String, author: String, hash: [u8; 32]) -> Self {
        Self {
            label,
            timestamp: timestamp_now(),
            author,
            hash,
            message: None,
            tags: Vec::new(),
        }
    }

    /// Create metadata with a message
    pub fn with_message(label: String, author: String, hash: [u8; 32], message: String) -> Self {
        Self {
            label,
            timestamp: timestamp_now(),
            author,
            hash,
            message: Some(message),
            tags: Vec::new(),
        }
    }

    /// Get a human-readable timestamp
    pub fn timestamp_string(&self) -> String {
        // Convert Unix timestamp to readable format
        // This is a simplified implementation
        format!("timestamp:{}", self.timestamp)
    }

    /// Get a short hash representation
    pub fn short_hash(&self) -> String {
        format!(
            "{:02x}{:02x}{:02x}{:02x}",
            self.hash[0], self.hash[1], self.hash[2], self.hash[3]
        )
    }
}

/*
=== GRAPH SNAPSHOTS AND STATE RECONSTRUCTION ===
Complete state representation and reconstruction algorithms
*/

/// Complete snapshot of a graph state
///
/// DESIGN: This represents the complete graph as it existed at a specific
/// point in time. It's expensive to compute but provides fast access once built.
///
/// USAGE: Used by HistoricalView for time travel and potentially
/// GraphSpace for current state representation
///
/// PERFORMANCE: Memory-intensive but provides O(1) access to all data
#[derive(Debug, Clone)]
pub struct GraphSnapshot {
    /// All nodes that were active at this state
    /// DESIGN: Vec for iteration, but could be HashSet for contains() queries
    pub active_nodes: Vec<NodeId>,

    /// All edges that were active at this state
    /// Maps edge_id -> (source_node, target_node)
    pub edges: HashMap<EdgeId, (NodeId, NodeId)>,

    /// All node attributes at this state
    /// Maps node_id -> (attribute_name -> attribute_value)
    pub node_attributes: HashMap<NodeId, HashMap<AttrName, AttrValue>>,

    /// All edge attributes at this state
    /// Maps edge_id -> (attribute_name -> attribute_value)
    pub edge_attributes: HashMap<EdgeId, HashMap<AttrName, AttrValue>>,

    /// The state ID this snapshot represents
    pub state_id: StateId,
}

impl GraphSnapshot {
    /// Create an empty snapshot for the given state
    pub fn empty(state_id: StateId) -> Self {
        Self {
            active_nodes: Vec::new(),
            edges: HashMap::new(),
            node_attributes: HashMap::new(),
            edge_attributes: HashMap::new(),
            state_id,
        }
    }

    /// Apply a delta to this snapshot to create a new snapshot
    ///
    /// ALGORITHM:
    /// 1. Start with a copy of this snapshot
    /// 2. Apply node additions/removals from delta
    /// 3. Apply edge additions/removals from delta
    /// 4. Apply attribute changes from delta
    /// 5. Update state_id to target state
    /// 6. Return the new snapshot
    ///
    /// PERFORMANCE: O(changes in delta + size of current snapshot for cloning)
    pub fn apply_delta(&self, delta: &Delta, target_state: StateId) -> Self {
        let mut new_snapshot = self.clone();
        new_snapshot.state_id = target_state;

        // Apply node additions
        for &node_id in &delta.nodes_added {
            if !new_snapshot.active_nodes.contains(&node_id) {
                new_snapshot.active_nodes.push(node_id);
            }
        }

        // Apply node removals
        for &node_id in &delta.nodes_removed {
            new_snapshot.active_nodes.retain(|&id| id != node_id);
            new_snapshot.node_attributes.remove(&node_id);
        }

        // Apply edge additions
        for &(edge_id, source, target) in &delta.edges_added {
            new_snapshot.edges.insert(edge_id, (source, target));
        }

        // Apply edge removals
        for &edge_id in &delta.edges_removed {
            new_snapshot.edges.remove(&edge_id);
            new_snapshot.edge_attributes.remove(&edge_id);
        }

        // Apply node attribute changes
        for (node_id, attr_name, _old_value, new_value) in &delta.node_attr_changes {
            let attrs = new_snapshot
                .node_attributes
                .entry(*node_id)
                .or_insert_with(HashMap::new);
            attrs.insert(attr_name.clone(), new_value.clone());
        }

        // Apply edge attribute changes
        for (edge_id, attr_name, _old_value, new_value) in &delta.edge_attr_changes {
            let attrs = new_snapshot
                .edge_attributes
                .entry(*edge_id)
                .or_insert_with(HashMap::new);
            attrs.insert(attr_name.clone(), new_value.clone());
        }

        new_snapshot
    }

    /// Create a snapshot by applying a sequence of deltas
    ///
    /// ALGORITHM:
    /// 1. Start with base snapshot (or empty)
    /// 2. Apply each delta in sequence
    /// 3. Return final snapshot
    ///
    /// USAGE: This is the main reconstruction function used by HistoryForest
    pub fn reconstruct_from_deltas(
        base: Option<&GraphSnapshot>,
        deltas: &[(Delta, StateId)],
    ) -> GraphResult<Self> {
        let mut current = base.cloned().unwrap_or_else(|| GraphSnapshot::empty(0));

        for (delta, target_state) in deltas {
            current = current.apply_delta(delta, *target_state);
        }

        Ok(current)
    }

    /// Compare this snapshot with another to produce a diff
    ///
    /// ALGORITHM:
    /// 1. Compare active nodes (added/removed)
    /// 2. Compare edges (added/removed)  
    /// 3. Compare all attributes (changed values)
    /// 4. Return structured diff
    pub fn diff_with(&self, other: &GraphSnapshot) -> StateDiff {
        use std::collections::HashSet;

        let self_nodes: HashSet<_> = self.active_nodes.iter().collect();
        let other_nodes: HashSet<_> = other.active_nodes.iter().collect();

        let nodes_added = other
            .active_nodes
            .iter()
            .filter(|&&node| !self_nodes.contains(&node))
            .cloned()
            .collect();

        let nodes_removed = self
            .active_nodes
            .iter()
            .filter(|&&node| !other_nodes.contains(&node))
            .cloned()
            .collect();

        // Compare edges
        let edges_added = other
            .edges
            .iter()
            .filter(|(edge_id, _)| !self.edges.contains_key(edge_id))
            .map(|(edge_id, &(source, target))| (*edge_id, source, target))
            .collect();

        let edges_removed = self
            .edges
            .iter()
            .filter(|(edge_id, _)| !other.edges.contains_key(edge_id))
            .map(|(edge_id, _)| *edge_id)
            .collect();

        // Compare attributes
        let mut attribute_changes = Vec::new();

        // Node attribute changes
        for &node_id in &other.active_nodes {
            let self_attrs = self
                .node_attributes
                .get(&node_id)
                .cloned()
                .unwrap_or_default();
            let other_attrs = other
                .node_attributes
                .get(&node_id)
                .cloned()
                .unwrap_or_default();

            for (attr_name, new_value) in &other_attrs {
                let old_value = self_attrs.get(attr_name).cloned();
                if old_value.as_ref() != Some(new_value) {
                    attribute_changes.push(AttributeChange {
                        entity_type: EntityType::Node,
                        entity_id: node_id as u64,
                        attr_name: attr_name.clone(),
                        old_value,
                        new_value: Some(new_value.clone()),
                    });
                }
            }

            // Check for removed attributes
            for (attr_name, old_value) in &self_attrs {
                if !other_attrs.contains_key(attr_name) {
                    attribute_changes.push(AttributeChange {
                        entity_type: EntityType::Node,
                        entity_id: node_id as u64,
                        attr_name: attr_name.clone(),
                        old_value: Some(old_value.clone()),
                        new_value: None,
                    });
                }
            }
        }

        // Edge attribute changes
        for &edge_id in other.edges.keys() {
            let self_attrs = self
                .edge_attributes
                .get(&edge_id)
                .cloned()
                .unwrap_or_default();
            let other_attrs = other
                .edge_attributes
                .get(&edge_id)
                .cloned()
                .unwrap_or_default();

            for (attr_name, new_value) in &other_attrs {
                let old_value = self_attrs.get(attr_name).cloned();
                if old_value.as_ref() != Some(new_value) {
                    attribute_changes.push(AttributeChange {
                        entity_type: EntityType::Edge,
                        entity_id: edge_id as u64,
                        attr_name: attr_name.clone(),
                        old_value,
                        new_value: Some(new_value.clone()),
                    });
                }
            }

            // Check for removed attributes
            for (attr_name, old_value) in &self_attrs {
                if !other_attrs.contains_key(attr_name) {
                    attribute_changes.push(AttributeChange {
                        entity_type: EntityType::Edge,
                        entity_id: edge_id as u64,
                        attr_name: attr_name.clone(),
                        old_value: Some(old_value.clone()),
                        new_value: None,
                    });
                }
            }
        }

        StateDiff {
            from_state: self.state_id,
            to_state: other.state_id,
            nodes_added,
            nodes_removed,
            edges_added,
            edges_removed,
            attribute_changes,
        }
    }

    /// Get basic statistics about this snapshot
    pub fn statistics(&self) -> SnapshotStatistics {
        SnapshotStatistics {
            node_count: self.active_nodes.len(),
            edge_count: self.edges.len(),
            node_attr_count: self.node_attributes.values().map(|attrs| attrs.len()).sum(),
            edge_attr_count: self.edge_attributes.values().map(|attrs| attrs.len()).sum(),
            memory_usage: self.estimate_memory_usage(),
        }
    }

    /// Estimate memory usage of this snapshot in bytes
    fn estimate_memory_usage(&self) -> usize {
        let mut total = 0;

        // Vec<NodeId>
        total += self.active_nodes.len() * std::mem::size_of::<NodeId>();

        // HashMap<EdgeId, (NodeId, NodeId)>
        total += self.edges.len()
            * (std::mem::size_of::<EdgeId>() + std::mem::size_of::<(NodeId, NodeId)>());

        // HashMap<NodeId, HashMap<AttrName, AttrValue>>
        for (_, attrs) in &self.node_attributes {
            total += std::mem::size_of::<NodeId>();
            for (_name, value) in attrs {
                total += std::mem::size_of::<AttrName>();
                total += match value {
                    AttrValue::Text(s) => s.len(),
                    AttrValue::Int(_) => std::mem::size_of::<i64>(),
                    AttrValue::Float(_) => std::mem::size_of::<f64>(),
                    AttrValue::Bool(_) => std::mem::size_of::<bool>(),
                    AttrValue::FloatVec(v) => v.len() * std::mem::size_of::<f32>(),
                    AttrValue::CompactText(cs) => cs.memory_size(),
                    AttrValue::SmallInt(_) => std::mem::size_of::<i32>(),
                    AttrValue::Bytes(b) => b.len(),
                    AttrValue::CompressedText(cd) => cd.memory_size(),
                    AttrValue::CompressedFloatVec(cd) => cd.memory_size(),
                    AttrValue::Null => 0,
                };
            }
        }

        // HashMap<EdgeId, HashMap<AttrName, AttrValue>>
        for (_, attrs) in &self.edge_attributes {
            total += std::mem::size_of::<EdgeId>();
            for (_name, value) in attrs {
                total += std::mem::size_of::<AttrName>();
                total += match value {
                    AttrValue::Text(s) => s.len(),
                    AttrValue::Int(_) => std::mem::size_of::<i64>(),
                    AttrValue::Float(_) => std::mem::size_of::<f64>(),
                    AttrValue::Bool(_) => std::mem::size_of::<bool>(),
                    AttrValue::FloatVec(v) => v.len() * std::mem::size_of::<f32>(),
                    AttrValue::CompactText(cs) => cs.memory_size(),
                    AttrValue::SmallInt(_) => std::mem::size_of::<i32>(),
                    AttrValue::Bytes(b) => b.len(),
                    AttrValue::CompressedText(cd) => cd.memory_size(),
                    AttrValue::CompressedFloatVec(cd) => cd.memory_size(),
                    AttrValue::Null => 0,
                };
            }
        }

        total
    }

    /// Check if a node exists in this snapshot
    pub fn contains_node(&self, node_id: NodeId) -> bool {
        self.active_nodes.contains(&node_id)
    }

    /// Check if an edge exists in this snapshot
    pub fn contains_edge(&self, edge_id: EdgeId) -> bool {
        self.edges.contains_key(&edge_id)
    }

    /// Get all neighbors of a node in this snapshot
    /// NOTE: For current state, use Graph::neighbors() which is optimized with columnar topology
    pub fn get_neighbors(&self, node_id: NodeId) -> GraphResult<Vec<NodeId>> {
        if !self.contains_node(node_id) {
            return Err(GraphError::NodeNotFound {
                node_id,
                operation: "get neighbors".to_string(),
                suggestion: "Check if node exists in this snapshot".to_string(),
            });
        }

        let mut neighbors = Vec::new();
        for &(source, target) in self.edges.values() {
            if source == node_id {
                neighbors.push(target);
            } else if target == node_id {
                neighbors.push(source);
            }
        }

        neighbors.sort();
        neighbors.dedup();
        Ok(neighbors)
    }
}

/*
=== STATE COMPARISON AND DIFFING ===
*/

/// Difference between two graph states
///
/// DESIGN: Structured representation of all changes between two snapshots
/// USAGE: Used for commit diffs, merge analysis, change visualization
#[derive(Debug, Clone)]
pub struct StateDiff {
    pub from_state: StateId,
    pub to_state: StateId,
    pub nodes_added: Vec<NodeId>,
    pub nodes_removed: Vec<NodeId>,
    pub edges_added: Vec<(EdgeId, NodeId, NodeId)>,
    pub edges_removed: Vec<EdgeId>,
    pub attribute_changes: Vec<AttributeChange>,
}

impl StateDiff {
    /// Create an empty diff between two states
    pub fn empty(from_state: StateId, to_state: StateId) -> Self {
        Self {
            from_state,
            to_state,
            nodes_added: Vec::new(),
            nodes_removed: Vec::new(),
            edges_added: Vec::new(),
            edges_removed: Vec::new(),
            attribute_changes: Vec::new(),
        }
    }

    /// Check if this diff represents any changes
    pub fn is_empty(&self) -> bool {
        self.nodes_added.is_empty()
            && self.nodes_removed.is_empty()
            && self.edges_added.is_empty()
            && self.edges_removed.is_empty()
            && self.attribute_changes.is_empty()
    }

    /// Get a summary of the changes in this diff
    pub fn summary(&self) -> DiffSummary {
        DiffSummary {
            from_state: self.from_state,
            to_state: self.to_state,
            nodes_changed: self.nodes_added.len() + self.nodes_removed.len(),
            edges_changed: self.edges_added.len() + self.edges_removed.len(),
            attributes_changed: self.attribute_changes.len(),
        }
    }
}

/// A single attribute change between states
#[derive(Debug, Clone)]
pub struct AttributeChange {
    pub entity_type: EntityType,
    pub entity_id: u64,
    pub attr_name: AttrName,
    pub old_value: Option<AttrValue>,
    pub new_value: Option<AttrValue>,
}

#[derive(Debug, Clone)]
pub enum EntityType {
    Node,
    Edge,
}

/// Statistics about a snapshot
#[derive(Debug, Clone)]
pub struct SnapshotStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub node_attr_count: usize,
    pub edge_attr_count: usize,
    pub memory_usage: usize,
}

/// Summary of changes in a StateDiff
#[derive(Debug, Clone)]
pub struct DiffSummary {
    pub from_state: StateId,
    pub to_state: StateId,
    pub nodes_changed: usize,
    pub edges_changed: usize,
    pub attributes_changed: usize,
}

/*
=== STATE UTILITIES ===
Helper functions for working with states
*/

/// Merge two snapshots (for branch merging)
///
/// ALGORITHM:
/// 1. Union of active nodes and edges
/// 2. Merge attributes (conflict resolution needed)
/// 3. Create new snapshot with merged state
pub fn merge_snapshots(
    _base: &GraphSnapshot,
    _branch1: &GraphSnapshot,
    _branch2: &GraphSnapshot,
    _target_state: StateId,
) -> GraphResult<GraphSnapshot> {
    // TODO: Complex merge algorithm
    // This is needed for git-like branch merging
    todo!("Implement merge_snapshots")
}

/// Validate that a snapshot is internally consistent
///
/// CHECKS:
/// 1. All edges reference active nodes
/// 2. All attribute maps reference active entities
/// 3. No duplicate IDs
pub fn validate_snapshot(_snapshot: &GraphSnapshot) -> GraphResult<()> {
    // TODO:
    // // Check edge endpoints are active nodes
    // for &(source, target) in snapshot.edges.values() {
    //     if !snapshot.contains_node(source) || !snapshot.contains_node(target) {
    //         return Err(GraphError::InvalidSnapshot);
    //     }
    // }
    //
    // // Check attribute maps only reference active entities
    // // ... more validation logic
    //
    // Ok(())
    todo!("Implement validate_snapshot")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::delta::DeltaObject;

    #[test]
    fn test_state_object_creation() {
        let delta = DeltaObject::empty();
        let state =
            StateObject::new_root(delta, "Initial state".to_string(), "test_user".to_string());

        assert!(state.is_root());
        assert_eq!(state.label(), "Initial state");
        assert_eq!(state.author(), "test_user");
        assert!(state.is_empty_delta());
    }

    #[test]
    fn test_state_tags() {
        let delta = DeltaObject::empty();
        let mut state =
            StateObject::new_root(delta, "Tagged state".to_string(), "test_user".to_string());

        state.add_tag("important".to_string());
        state.add_tag("milestone".to_string());

        assert!(state.has_tag("important"));
        assert!(state.has_tag("milestone"));
        assert!(!state.has_tag("nonexistent"));

        state.remove_tag("important");
        assert!(!state.has_tag("important"));
        assert!(state.has_tag("milestone"));
    }
}
