//! Change Tracking System - Strategy-based transaction management for graph modifications.
//!
//! ARCHITECTURE ROLE:
//! This component provides a pluggable interface for different temporal storage strategies.
//! It delegates to the selected strategy while maintaining a consistent API for the rest
//! of the system.
//!
//! DESIGN PHILOSOPHY:
//! - Strategy Pattern: Pluggable algorithms for different storage approaches
//! - Backward Compatibility: Existing APIs continue to work unchanged
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

use crate::core::delta::DeltaObject;
use crate::core::pool::GraphPool;
use crate::core::strategies::{
    create_strategy, StorageCharacteristics, StorageStrategyType, TemporalStorageStrategy,
};
use crate::errors::GraphError;
use crate::types::{AttrName, AttrValue, EdgeId, NodeId};
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
        // Delegate to strategy for bulk recording
        for &(entity_id, ref attr_name, old_index, new_index) in changes {
            let id = entity_id.into();
            if is_node {
                self.strategy
                    .record_node_attr_change(id, attr_name.clone(), old_index, new_index);
            } else {
                self.strategy
                    .record_edge_attr_change(id, attr_name.clone(), old_index, new_index);
            }
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
        let changeset = self.strategy.create_change_set();

        // Add all nodes that were added or removed
        modified.extend(changeset.nodes_added.iter());
        modified.extend(changeset.nodes_removed.iter());

        // Add all nodes that had attribute changes
        for (node_id, _, _, _) in &changeset.node_attr_changes {
            modified.insert(*node_id);
        }

        modified
    }

    /// Get all edges that have been modified
    pub fn get_modified_edges(&self) -> HashSet<EdgeId> {
        let mut modified = HashSet::new();
        let changeset = self.strategy.create_change_set();

        // Add all edges that were added or removed
        for (edge_id, _, _) in &changeset.edges_added {
            modified.insert(*edge_id);
        }
        modified.extend(changeset.edges_removed.iter());

        // Add all edges that had attribute changes
        for (edge_id, _, _, _) in &changeset.edge_attr_changes {
            modified.insert(*edge_id);
        }

        modified
    }

    /// Check if a specific node has been modified
    pub fn is_node_modified(&self, node_id: NodeId) -> bool {
        let changeset = self.strategy.create_change_set();

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
        let changeset = self.strategy.create_change_set();

        // Check if edge was added or removed
        if changeset
            .edges_added
            .iter()
            .any(|(id, _, _)| *id == edge_id)
            || changeset.edges_removed.contains(&edge_id)
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

    /// Create a delta object representing all current changes
    /// This is used when committing to history
    pub fn create_delta(&self) -> DeltaObject {
        self.strategy.create_delta()
    }

    /// Create a change set that can be passed to HistoryForest
    pub fn create_change_set(&self) -> ChangeSet {
        self.strategy.create_change_set()
    }

    /// Alias for create_change_set (backward compatibility)
    pub fn create_changeset(&self) -> ChangeSet {
        self.create_change_set()
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
        let changeset = self.strategy.create_change_set();

        ChangeSummary {
            nodes_added: changeset.nodes_added.len(),
            nodes_removed: changeset.nodes_removed.len(),
            edges_added: changeset.edges_added.len(),
            edges_removed: changeset.edges_removed.len(),
            node_attr_changes: changeset.node_attr_changes.len(),
            edge_attr_changes: changeset.edge_attr_changes.len(),
            total_changes: self.change_count(),
            first_change_time: None, // TODO: Could track timestamps in strategy
        }
    }

    /// Generate the reverse operations needed to undo all changes
    /// This is useful for implementing rollback functionality
    pub fn generate_reverse_operations(&self) -> Vec<ReverseOperation> {
        // Basic implementation returns empty - full implementation would analyze changes and generate reverses
        Vec::new()
    }

    /*
    === CHANGE MERGING ===
    Combine changes from different sources (useful for merging branches)
    */

    /// Merge changes from another change tracker
    /// This is complex because changes might conflict
    pub fn merge(&mut self, other: &ChangeTracker) -> Result<(), MergeConflict> {
        let _ = other; // Silence unused parameter warning
                       // Basic implementation returns error - merging not yet implemented
        Err(MergeConflict {
            conflict_type: ConflictType::NodeAttributeConflict,
            entity_id: 0,
            attribute: None,
            our_change: "not implemented".to_string(),
            their_change: "not implemented".to_string(),
        })
    }

    /// Check if merging with another change tracker would cause conflicts
    pub fn would_conflict_with(&self, other: &ChangeTracker) -> Vec<MergeConflict> {
        let _ = other; // Silence unused parameter warning
                       // Basic implementation returns empty - conflict analysis not yet implemented
        Vec::new()
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
        let changeset = self.strategy.create_change_set();
        let mut total = 0;

        // Size of vectors holding IDs
        total += changeset.nodes_added.len() * std::mem::size_of::<NodeId>();
        total += changeset.nodes_removed.len() * std::mem::size_of::<NodeId>();
        total += changeset.edges_added.len() * std::mem::size_of::<(EdgeId, NodeId, NodeId)>();
        total += changeset.edges_removed.len() * std::mem::size_of::<EdgeId>();

        // Size of attribute changes
        for (_, _attr_name, old_val, new_val) in &changeset.node_attr_changes {
            total += std::mem::size_of::<NodeId>();
            total += std::mem::size_of::<AttrName>();
            if let Some(val) = old_val {
                total += match val {
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
            total += match new_val {
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

        for (_, _attr_name, old_val, new_val) in &changeset.edge_attr_changes {
            total += std::mem::size_of::<EdgeId>();
            total += std::mem::size_of::<AttrName>();
            if let Some(val) = old_val {
                total += match val {
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
            total += match new_val {
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

        total
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

/// A set of changes that can be committed to history
#[derive(Debug, Clone)]
pub struct ChangeSet {
    pub nodes_added: Vec<NodeId>,
    pub nodes_removed: Vec<NodeId>,
    pub edges_added: Vec<(EdgeId, NodeId, NodeId)>,
    pub edges_removed: Vec<EdgeId>,
    pub node_attr_changes: Vec<(NodeId, AttrName, Option<AttrValue>, AttrValue)>,
    pub edge_attr_changes: Vec<(EdgeId, AttrName, Option<AttrValue>, AttrValue)>,
}

impl ChangeSet {
    /// Create an empty change set
    pub fn new() -> Self {
        Self {
            nodes_added: Vec::new(),
            nodes_removed: Vec::new(),
            edges_added: Vec::new(),
            edges_removed: Vec::new(),
            node_attr_changes: Vec::new(),
            edge_attr_changes: Vec::new(),
        }
    }

    /// Check if the change set is empty
    pub fn is_empty(&self) -> bool {
        self.nodes_added.is_empty()
            && self.nodes_removed.is_empty()
            && self.edges_added.is_empty()
            && self.edges_removed.is_empty()
            && self.node_attr_changes.is_empty()
            && self.edge_attr_changes.is_empty()
    }
}

/// Summary of all changes in the tracker
#[derive(Debug, Clone)]
pub struct ChangeSummary {
    pub nodes_added: usize,
    pub nodes_removed: usize,
    pub edges_added: usize,
    pub edges_removed: usize,
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
        )
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
- History: Uses create_delta() output regardless of strategy
- Graph: Same API regardless of underlying strategy
- Config: Strategy selection happens at ChangeTracker creation

BACKWARD COMPATIBILITY:
- All existing APIs work unchanged
- Performance characteristics preserved for default strategy
- New strategy-specific methods available for optimization
- Migration path: zero code changes required
*/
