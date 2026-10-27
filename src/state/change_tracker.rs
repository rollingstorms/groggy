//! Change Tracking System - Strategy-based transaction management for graph modifications.
//!
//! ARCHITECTURE ROLE:
//! This component provides a pluggable interface for different temporal storage strategies.
//! It delegates to the selected strategy while maintaining a consistent API for the rest
//! of the system.
//!
//! DESIGN PHILOSOPHY:
//! - Strategy Pattern: Pluggable algorithms for different storage approaches
//! - Compatibility: Legacy import paths remain; logical extraction is fallible.
//! - Performance Transparency: Each strategy optimizes for different workloads
//! - Configuration Driven: Strategy selection based on requirements

/*
=== STRATEGY-BASED CHANGE TRACKING ===

The change tracker now uses the Strategy Pattern to support different temporal
storage approaches:

1. INDEX DELTAS: Current implementation using column indices (default)
2. FULL SNAPSHOTS: Complete state snapshots (future)
3. HYBRID: Snapshots + deltas (future)
4. COMPRESSED: Space-optimized storage (future)

KEY DESIGN DECISIONS:
- Strategy Pattern: Pluggable storage algorithms
- Delegation: ChangeTracker forwards to selected strategy
- Compatibility: Existing APIs work unchanged
- Configuration: Strategy selection at creation time
*/

/// Strategy-based change tracker
///
/// RESPONSIBILITIES:
/// - Provide consistent API for change tracking
/// - Delegate to selected temporal storage strategy
/// - Support strategy-specific operations (like index-based changes)
/// - Maintain backward compatibility with existing code
///
/// NOT RESPONSIBLE FOR:
/// - Actual storage implementation (that's the strategy's job)
/// - Strategy selection logic (that's configuration driven)
/// - Storage optimization (that's strategy-specific)
#[derive(Debug)]
pub struct ChangeTracker {
    /// The selected temporal storage strategy
    /// This handles the actual change tracking implementation
    strategy: Box<dyn TemporalStorageStrategy>,
}

use crate::errors::{GraphError, GraphResult};
use crate::state::delta::DeltaObject;
use crate::storage::pool::GraphPool;
use crate::types::{AttrName, AttrValue, EdgeId, NodeId};
use crate::utils::strategies::{
    create_strategy, StorageCharacteristics, StorageStrategyType, TemporalStorageStrategy,
};
use std::collections::{HashMap, HashSet};

impl ChangeTracker {
    /// Create a new change tracker with default strategy (IndexDeltas)
    pub fn new() -> Self {
        Self::with_strategy(StorageStrategyType::default())
    }

    /// Create a new change tracker with specific strategy
    pub fn with_strategy(strategy_type: StorageStrategyType) -> Self {
        Self {
            strategy: create_strategy(strategy_type),
        }
    }

    /// Create a change tracker with a custom strategy instance
    pub fn with_custom_strategy(strategy: Box<dyn TemporalStorageStrategy>) -> Self {
        Self { strategy }
    }

    /*
    === RECORDING CHANGES ===
    These methods are called by Graph when operations modify the graph
    */

    /*
    === CHANGE RECORDING API ===
    These methods delegate to the selected strategy
    */

    /// Record that a new node was added
    pub fn record_node_addition(&mut self, node_id: NodeId) {
        self.strategy.record_node_addition(node_id);
    }

    /// Record that multiple nodes were added (bulk operation)
    pub fn record_nodes_addition(&mut self, node_ids: &[NodeId]) {
        for &node_id in node_ids {
            self.strategy.record_node_addition(node_id);
        }
    }

    /// Record that a node was removed
    pub fn record_node_removal(&mut self, node_id: NodeId) {
        self.strategy.record_node_removal(node_id);
    }

    /// Record that a new edge was added
    pub fn record_edge_addition(&mut self, edge_id: EdgeId, source: NodeId, target: NodeId) {
        self.strategy.record_edge_addition(edge_id, source, target);
    }

    /// Record that multiple edges were added (bulk operation)
    pub fn record_edges_addition(&mut self, edges: &[(EdgeId, NodeId, NodeId)]) {
        for &(edge_id, source, target) in edges {
            self.strategy.record_edge_addition(edge_id, source, target);
        }
    }

    /// Whether the installed strategy can retain typed endpoint changes.
    pub fn supports_endpoint_changes(&self) -> bool {
        self.strategy.supports_endpoint_changes()
    }
    /// Retain the original and final endpoints of one stable edge instance.
    pub fn record_edge_endpoint_change(
        &mut self,
        change: crate::EdgeEndpointChange,
    ) -> GraphResult<()> {
        self.strategy.record_edge_endpoint_change(change)
    }

    /// Record that an edge was removed
    pub fn record_edge_removal(&mut self, edge_id: EdgeId) {
        self.strategy.record_edge_removal(edge_id);
    }

    /// Record attribute changes for any entity type (index-based, efficient bulk recording)
    /// This is the main API - all attribute changes are recorded as indices
    pub fn record_attr_changes<T>(
        &mut self,
        changes: &[(T, AttrName, Option<usize>, usize)],
        is_node: bool,
    ) where
        T: Into<usize> + Copy,
    {
        // Convert to appropriate type and delegate to strategy's bulk method
        if is_node {
            let node_changes: Vec<(NodeId, AttrName, Option<usize>, Option<usize>)> = changes
                .iter()
                .map(|&(entity_id, ref attr_name, old_index, new_index)| {
                    (
                        entity_id.into() as NodeId,
                        attr_name.clone(),
                        old_index,
                        Some(new_index),
                    )
                })
                .collect();
            self.strategy.record_node_attr_changes_bulk(&node_changes);
        } else {
            let edge_changes: Vec<(EdgeId, AttrName, Option<usize>, Option<usize>)> = changes
                .iter()
                .map(|&(entity_id, ref attr_name, old_index, new_index)| {
                    (
                        entity_id.into() as EdgeId,
                        attr_name.clone(),
                        old_index,
                        Some(new_index),
                    )
                })
                .collect();
            self.strategy.record_edge_attr_changes_bulk(&edge_changes);
        }
    }

    /// Record single attribute change (convenience wrapper)
    pub fn record_attr_change<T>(
        &mut self,
        entity_id: T,
        attr_name: AttrName,
        old_index: Option<usize>,
        new_index: usize,
        is_node: bool,
    ) where
        T: Into<usize> + Copy,
    {
        self.record_attr_changes(&[(entity_id, attr_name, old_index, new_index)], is_node);
    }

    /// Record an explicit absence without allocating a sentinel pool value.
    pub fn record_attr_unset(
        &mut self,
        entity: usize,
        name: AttrName,
        old: Option<usize>,
        is_node: bool,
    ) {
        if is_node {
            self.strategy
                .record_node_attr_changes_bulk(&[(entity, name, old, None)]);
        } else {
            self.strategy
                .record_edge_attr_changes_bulk(&[(entity, name, old, None)]);
        }
    }

    // NOTE: update_change_metadata and current_timestamp are now handled by the strategy
    // These methods have been moved to IndexDeltaStrategy in strategies.rs

    /*
    === BULK CHANGE RECORDING ===
    Efficient recording of multiple changes at once
    */

    /// Record multiple node additions efficiently
    pub fn record_node_additions(&mut self, node_ids: &[NodeId]) {
        for &node_id in node_ids {
            self.record_node_addition(node_id);
        }
    }

    // NOTE: Bulk change recording methods moved above as main API

    /*
    === CHANGE INSPECTION ===
    Query what has changed
    */

    /*
    === CHANGE INSPECTION API ===
    These methods delegate to the selected strategy
    */

    /// Check if there are any uncommitted changes
    pub fn has_changes(&self) -> bool {
        self.strategy.has_changes()
    }

    /// Get the total number of changes recorded
    pub fn change_count(&self) -> usize {
        self.strategy.change_count()
    }

    // NOTE: change_summary is now handled by the strategy-based implementation below (line ~318)

    /// Get all nodes that have been modified (added, removed, or attrs changed)
    pub fn get_modified_nodes(&self) -> HashSet<NodeId> {
        let mut modified = HashSet::new();
        let changeset = self.strategy.indexed_changes();

        // Add all nodes that were added or removed
        modified.extend(changeset.nodes_added.iter());
        modified.extend(changeset.nodes_removed.iter());

        // Add all nodes that had attribute changes
        for (node_id, _, _, _) in changeset.node_attr_changes {
            modified.insert(*node_id);
        }

        modified
    }

    /// Get all edges that have been modified
    pub fn get_modified_edges(&self) -> HashSet<EdgeId> {
        let mut modified = HashSet::new();
        let changeset = self.strategy.indexed_changes();

        // Add all edges that were added or removed
        for (edge_id, _, _) in changeset.edges_added {
            modified.insert(*edge_id);
        }
        modified.extend(changeset.edges_removed.iter());
        modified.extend(changeset.edge_endpoint_changes.keys());

        // Add all edges that had attribute changes
        for (edge_id, _, _, _) in changeset.edge_attr_changes {
            modified.insert(*edge_id);
        }

        modified
    }

    /// Check if a specific node has been modified
    pub fn is_node_modified(&self, node_id: NodeId) -> bool {
        let changeset = self.strategy.indexed_changes();

        // Check if node was added or removed
        if changeset.nodes_added.contains(&node_id) || changeset.nodes_removed.contains(&node_id) {
            return true;
        }

        // Check if node had attribute changes
        changeset
            .node_attr_changes
            .iter()
            .any(|(changed_node_id, _, _, _)| *changed_node_id == node_id)
    }

    /// Check if a specific edge has been modified
    pub fn is_edge_modified(&self, edge_id: EdgeId) -> bool {
        let changeset = self.strategy.indexed_changes();

        // Check if edge was added or removed
        if changeset
            .edges_added
            .iter()
            .any(|(id, _, _)| *id == edge_id)
            || changeset.edges_removed.contains(&edge_id)
            || changeset.edge_endpoint_changes.contains_key(&edge_id)
        {
            return true;
        }

        // Check if edge had attribute changes
        changeset
            .edge_attr_changes
            .iter()
            .any(|(changed_edge_id, _, _, _)| *changed_edge_id == edge_id)
    }

    /*
    === DELTA GENERATION ===
    Convert tracked changes into history deltas
    */

    /*
    === DELTA CREATION API ===
    */

    /// Borrow actual pending physical references without cloning or synthesizing values.
    pub fn indexed_changes(&self) -> crate::state::change::IndexChangeView<'_> {
        self.strategy.indexed_changes()
    }

    /// Create an explicitly indexed pending delta; portable history requires pool resolution.
    pub fn create_delta(&self) -> GraphResult<DeltaObject> {
        self.strategy.create_delta()
    }

    /// Extract topology-only logical changes; property references require explicit pool resolution.
    pub fn create_change_set(&self) -> GraphResult<ChangeSet> {
        self.strategy.create_change_set()
    }

    /// Alias for create_change_set (backward compatibility)
    pub fn create_changeset(&self) -> GraphResult<ChangeSet> {
        self.create_change_set()
    }

    /// Resolve stable column references into logical values for historical storage.
    pub fn create_changeset_resolved(&self, pool: &GraphPool) -> GraphResult<ChangeSet> {
        self.strategy.create_change_set_resolved(pool)
    }

    /*
    === ROLLBACK OPERATIONS ===
    Undo changes back to last commit
    */

    /*
    === CHANGE MANAGEMENT API ===
    */

    /// Clear all recorded changes (rollback to last commit state)
    pub fn clear(&mut self) {
        self.strategy.clear_changes();
    }

    /*
    === STRATEGY MANAGEMENT API ===
    */

    /// Get the name of the current strategy
    pub fn strategy_name(&self) -> &'static str {
        self.strategy.strategy_name()
    }

    /// Get the storage characteristics of the current strategy
    pub fn storage_characteristics(&self) -> StorageCharacteristics {
        self.strategy.storage_characteristics()
    }

    /// Get a summary of what has changed (compatibility method)
    pub fn change_summary(&self) -> ChangeSummary {
        let changeset = self.strategy.indexed_changes();

        ChangeSummary {
            nodes_added: changeset.nodes_added.len(),
            nodes_removed: changeset.nodes_removed.len(),
            edges_added: changeset.edges_added.len(),
            edges_removed: changeset.edges_removed.len(),
            edge_endpoint_changes: changeset.edge_endpoint_changes.len(),
            node_attr_changes: changeset.node_attr_changes.len(),
            edge_attr_changes: changeset.edge_attr_changes.len(),
            total_changes: self.change_count(),
            first_change_time: None, // TODO: Could track timestamps in strategy
        }
    }

    /// Empty trackers need no reversal. Pending changes require an explicit
    /// retained-state target; a physical tracker cannot infer inverse payloads.
    pub fn generate_reverse_operations(&self) -> GraphResult<Vec<ReverseOperation>> {
        if !self.has_changes() {
            return Ok(Vec::new());
        }
        Err(crate::GraphError::InvalidInput(
            "Reverse operations require an explicit retained state; use Graph::restore_state with Tracked mode".into()
        ))
    }

    /*
    === CHANGE MERGING ===
    Combine changes from different sources (useful for merging branches)
    */

    /// An empty source is a no-op. Nonempty trackers lack retained parent/base
    /// provenance and cannot authorize merging physical column references.
    pub fn merge(&mut self, other: &ChangeTracker) -> GraphResult<()> {
        if !other.has_changes() {
            return Ok(());
        }
        Err(crate::GraphError::InvalidInput(
            "Tracker merging requires retained parents and a merge base; use Graph::prepare_merge_read".into()
        ))
    }

    /// Conflict-free empty inputs are trivial; pending trackers require retained
    /// parent/base provenance rather than a fabricated empty conflict result.
    pub fn would_conflict_with(&self, other: &ChangeTracker) -> GraphResult<Vec<MergeConflict>> {
        if !self.has_changes() && !other.has_changes() {
            return Ok(Vec::new());
        }
        Err(crate::GraphError::InvalidInput(
            "Conflict analysis requires retained parents and a merge base; use Graph::prepare_merge_read".into()
        ))
    }

    /*
    === OPTIMIZATION OPERATIONS ===
    Clean up and optimize the change log
    */

    /// Optimize the change log by removing redundant entries
    /// For example, if a node attribute is changed multiple times,
    /// we only need to track the final change
    pub fn optimize(&mut self) {
        // Basic implementation is a no-op - optimization not yet implemented
    }

    /// Estimate the memory usage of the change tracker
    pub fn memory_usage(&self) -> usize {
        self.strategy.indexed_changes().estimated_bytes()
    }

    /*
    === STATISTICS AND DEBUGGING ===
    Information about the change tracker state
    */

    /// Get statistics about the changes
    pub fn statistics(&self) -> ChangeStatistics {
        ChangeStatistics {
            change_summary: self.change_summary(),
            memory_usage_bytes: self.memory_usage(),
            average_changes_per_node: 0.0,
            average_changes_per_edge: 0.0,
            most_changed_nodes: Vec::new(),
            most_changed_edges: Vec::new(),
            change_rate_per_second: 0.0,
        }
    }

    /// Get the time elapsed since the first change
    pub fn time_since_first_change(&self) -> Option<u64> {
        // Basic implementation returns None - time tracking not yet implemented
        None
    }
}

/*
=== SUPPORTING DATA STRUCTURES ===
*/

/// Compatibility export of the canonical logical mutation payload.
pub use super::change::ChangeSet;

/// Summary of all changes in the tracker
#[derive(Debug, Clone)]
pub struct ChangeSummary {
    pub nodes_added: usize,
    pub nodes_removed: usize,
    pub edges_added: usize,
    pub edges_removed: usize,
    pub edge_endpoint_changes: usize,
    pub node_attr_changes: usize,
    pub edge_attr_changes: usize,
    pub total_changes: usize,
    pub first_change_time: Option<u64>,
}

impl ChangeSummary {
    /// Check if any changes have been made
    pub fn is_empty(&self) -> bool {
        self.total_changes == 0
    }

    /// Get a human-readable description of the changes
    pub fn description(&self) -> String {
        format!(
            "+{} nodes, -{} nodes, +{} edges, -{} edges, {} node attrs, {} edge attrs",
            self.nodes_added,
            self.nodes_removed,
            self.edges_added,
            self.edges_removed,
            self.node_attr_changes,
            self.edge_attr_changes
        ) + &if self.edge_endpoint_changes > 0 {
            format!(", {} endpoint changes", self.edge_endpoint_changes)
        } else {
            String::new()
        }
    }
}

/// A reversible operation that can undo a change
#[derive(Debug, Clone)]
pub enum ReverseOperation {
    /// Remove a node that was added
    RemoveNode(NodeId),

    /// Add back a node that was removed (with its attributes)
    AddNode(NodeId, HashMap<AttrName, AttrValue>),

    /// Remove an edge that was added
    RemoveEdge(EdgeId),

    /// Add back an edge that was removed (with its attributes)
    AddEdge(EdgeId, NodeId, NodeId, HashMap<AttrName, AttrValue>),

    /// Restore an attribute to its previous value
    RestoreNodeAttr(NodeId, AttrName, Option<AttrValue>),

    /// Restore an edge attribute to its previous value
    RestoreEdgeAttr(EdgeId, AttrName, Option<AttrValue>),
}

impl ReverseOperation {
    /// Execute this reverse operation on a graph pool
    pub fn execute(&self, pool: &mut GraphPool) -> Result<(), GraphError> {
        let _ = pool; // Silence unused parameter warning
                      // For now return not implemented error
        match self {
            ReverseOperation::RemoveNode(_) => Err(GraphError::NotImplemented {
                feature: "reverse operations".to_string(),
                tracking_issue: None,
            }),
            ReverseOperation::AddNode(_, _) => Err(GraphError::NotImplemented {
                feature: "reverse operations".to_string(),
                tracking_issue: None,
            }),
            ReverseOperation::RemoveEdge(_) => Err(GraphError::NotImplemented {
                feature: "reverse operations".to_string(),
                tracking_issue: None,
            }),
            ReverseOperation::AddEdge(_, _, _, _) => Err(GraphError::NotImplemented {
                feature: "reverse operations".to_string(),
                tracking_issue: None,
            }),
            ReverseOperation::RestoreNodeAttr(_, _, _) => Err(GraphError::NotImplemented {
                feature: "reverse operations".to_string(),
                tracking_issue: None,
            }),
            ReverseOperation::RestoreEdgeAttr(_, _, _) => Err(GraphError::NotImplemented {
                feature: "reverse operations".to_string(),
                tracking_issue: None,
            }),
        }
    }
}

/// A conflict that occurs when merging change trackers
#[derive(Debug, Clone)]
pub struct MergeConflict {
    pub conflict_type: ConflictType,
    pub entity_id: u64, // NodeId or EdgeId
    pub attribute: Option<AttrName>,
    pub our_change: String,   // Description of our change
    pub their_change: String, // Description of their change
}

#[derive(Debug, Clone)]
pub enum ConflictType {
    /// Both sides modified the same node attribute
    NodeAttributeConflict,

    /// Both sides modified the same edge attribute
    EdgeAttributeConflict,

    /// One side added a node, other side removed it
    NodeExistenceConflict,

    /// One side added an edge, other side removed it
    EdgeExistenceConflict,
}

/// Detailed statistics about the change tracker
#[derive(Debug, Clone)]
pub struct ChangeStatistics {
    pub change_summary: ChangeSummary,
    pub memory_usage_bytes: usize,
    pub average_changes_per_node: f64,
    pub average_changes_per_edge: f64,
    pub most_changed_nodes: Vec<(NodeId, usize)>, // Top 10 most modified nodes
    pub most_changed_edges: Vec<(EdgeId, usize)>, // Top 10 most modified edges
    pub change_rate_per_second: f64,              // Changes per second since first change
}

impl Default for ChangeTracker {
    fn default() -> Self {
        Self::new()
    }
}

/*
=== STRATEGY-BASED IMPLEMENTATION NOTES ===

STRATEGY PATTERN BENEFITS:
- Pluggable storage algorithms for different workloads
- Easy to add new temporal storage strategies
- Configuration-driven strategy selection
- Backward compatibility with existing code

PERFORMANCE CHARACTERISTICS:
- Strategy-dependent: IndexDeltas optimized for frequent small changes
- Delegation overhead: Minimal - single virtual function call
- Memory usage: Strategy-specific optimizations
- Extensibility: New strategies can optimize for different patterns

STRATEGY SELECTION:
- Default: IndexDeltaStrategy (preserves current performance)
- Configuration: Can specify different strategies at creation
- Runtime: Strategy switching not currently supported (by design)
- Future: Could be adaptive based on workload characteristics

INTEGRATION POINTS:
- Space: Calls index-specific methods for IndexDeltaStrategy
- History: Resolves create_changeset_resolved(pool) into typed logical deltas.
- Graph: Same API regardless of underlying strategy
- Config: Strategy selection happens at ChangeTracker creation

BACKWARD COMPATIBILITY:
- No-pool extraction returns GraphResult and rejects unresolved properties.
- Performance characteristics preserved for default strategy
- New strategy-specific methods available for optimization
- Use indexed_changes() for borrowed diagnostics and pool resolution for logical values.
*/
