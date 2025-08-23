//! Temporal Storage Strategies - Pluggable approaches for storing graph changes over time
//!
//! This module contains both the trait definition and all concrete implementations
//! of different temporal storage strategies.

use crate::core::change_tracker::ChangeSet;
use crate::core::delta::DeltaObject;
use crate::types::{AttrName, AttrValue, EdgeId, NodeId};
use std::collections::HashMap;

/*
=== TEMPORAL STRATEGY OVERVIEW ===

Different workloads benefit from different temporal storage approaches:

1. INDEX DELTAS (current implementation):
   - Best for: Frequent small changes, analytical workloads
   - Trade-off: Small storage + fast commits vs. reconstruction cost
   - Use case: Real-time graph updates, attribute-heavy workloads

2. FULL SNAPSHOTS (future):
   - Best for: Infrequent large changes, time-travel heavy workloads
   - Trade-off: Large storage vs. instant recall
   - Use case: Versioned datasets, audit trails

3. HYBRID APPROACHES (future):
   - Best for: Mixed workloads with both patterns
   - Trade-off: Balanced storage vs. balanced performance
   - Use case: Production systems with varied access patterns

The strategy pattern allows runtime selection based on workload characteristics.
*/

/// Core trait for temporal storage strategies
///
/// DESIGN: Each strategy implements a different approach to storing and reconstructing
/// temporal graph states. The trait provides a common interface that ChangeTracker
/// and History systems can work with regardless of the underlying storage strategy.
pub trait TemporalStorageStrategy: std::fmt::Debug {
    /*
    === CHANGE RECORDING ===
    How this strategy tracks modifications
    */

    /// Record that a node was added to the graph
    fn record_node_addition(&mut self, node_id: NodeId);

    /// Record that a node was removed from the graph
    fn record_node_removal(&mut self, node_id: NodeId);

    /// Record that an edge was added to the graph
    fn record_edge_addition(&mut self, edge_id: EdgeId, source: NodeId, target: NodeId);

    /// Record that an edge was removed from the graph
    fn record_edge_removal(&mut self, edge_id: EdgeId);

    /// Record that a node attribute changed (index-based)
    /// All strategies now work with indices for consistency and efficiency
    fn record_node_attr_change(
        &mut self,
        node_id: NodeId,
        attr_name: AttrName,
        old_index: Option<usize>,
        new_index: usize,
    );

    /// Record that an edge attribute changed (index-based)
    fn record_edge_attr_change(
        &mut self,
        edge_id: EdgeId,
        attr_name: AttrName,
        old_index: Option<usize>,
        new_index: usize,
    );

    /*
    === DELTA CREATION ===
    Convert tracked changes into committable deltas
    */

    /// Create a delta object representing all current changes
    /// This is used when committing to history
    fn create_delta(&self) -> DeltaObject;

    /// Create a change set that can be passed to HistoryForest
    fn create_change_set(&self) -> ChangeSet;

    /*
    === CHANGE MANAGEMENT ===
    State management for the tracking system
    */

    /// Check if there are any uncommitted changes
    fn has_changes(&self) -> bool;

    /// Get the total number of changes recorded
    fn change_count(&self) -> usize;

    /// Clear all recorded changes (after successful commit)
    fn clear_changes(&mut self);

    /*
    === STRATEGY METADATA ===
    Information about this storage strategy
    */

    /// Get the name of this strategy
    fn strategy_name(&self) -> &'static str;

    /// Get performance characteristics of this strategy
    fn storage_characteristics(&self) -> StorageCharacteristics;

    /// Support for downcasting to specific strategy types
    /// This allows access to strategy-specific methods
    fn as_any(&mut self) -> &mut dyn std::any::Any;
}

/// Performance and storage characteristics of a temporal strategy
///
/// DESIGN: This allows the system to choose strategies based on workload requirements
/// and provides transparency about trade-offs between different approaches.
#[derive(Debug, Clone)]
pub struct StorageCharacteristics {
    /// Typical storage overhead per change (relative scale 1-10)
    pub storage_overhead: u8,

    /// Typical commit performance (relative scale 1-10, higher = faster)
    pub commit_speed: u8,

    /// Typical reconstruction performance (relative scale 1-10, higher = faster)
    pub reconstruction_speed: u8,

    /// Whether this strategy supports efficient partial reconstruction
    pub supports_partial_reconstruction: bool,

    /// Whether this strategy benefits from frequent commits
    pub prefers_frequent_commits: bool,

    /// Human-readable description of the strategy's trade-offs
    pub description: &'static str,
}

/// Enumeration of available storage strategies
///
/// DESIGN: This enum allows configuration-driven strategy selection
/// and provides a type-safe way to specify which strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageStrategyType {
    /// Index-based deltas (current implementation)
    /// Stores attribute changes as column index mappings
    IndexDeltas,
    // Future strategies:
    // /// Full snapshots at each commit
    // FullSnapshots,
    //
    // /// Hybrid: snapshots + deltas
    // Hybrid,
    //
    // /// Compressed snapshots
    // CompressedSnapshots,
}

impl StorageStrategyType {
    /// Get a human-readable name for this strategy
    pub fn name(&self) -> &'static str {
        match self {
            StorageStrategyType::IndexDeltas => "Index-Based Deltas",
        }
    }

    /// Get a description of this strategy
    pub fn description(&self) -> &'static str {
        match self {
            StorageStrategyType::IndexDeltas => {
                "Efficient delta storage using column indices for temporal graph versioning"
            }
        }
    }
}

impl Default for StorageStrategyType {
    fn default() -> Self {
        StorageStrategyType::IndexDeltas
    }
}

/// Factory function to create a strategy instance
///
/// DESIGN: This factory pattern allows the system to create strategy instances
/// based on configuration without tight coupling to specific implementations.
pub fn create_strategy(strategy_type: StorageStrategyType) -> Box<dyn TemporalStorageStrategy> {
    match strategy_type {
        StorageStrategyType::IndexDeltas => Box::new(IndexDeltaStrategy::new()), // Future strategies would be added here
    }
}

/*
=== INDEX DELTA STRATEGY IMPLEMENTATION ===
*/

use crate::core::delta::ColumnIndexDelta;

/// Index-based delta storage strategy
///
/// DESIGN: This strategy tracks attribute changes using column indices instead of values.
/// When an attribute changes, it stores the old and new column indices rather than the
/// actual values. This enables efficient temporal versioning where multiple graph states
/// can share the same columnar data.
///
/// PERFORMANCE CHARACTERISTICS:
/// - Storage: Very efficient (only stores indices)
/// - Commits: Fast (no value copying)
/// - Reconstruction: Moderate (requires index resolution)
/// - Best for: Frequent changes, analytical workloads
#[derive(Debug)]
pub struct IndexDeltaStrategy {
    /// Nodes that have been added since last commit
    nodes_added: Vec<NodeId>,

    /// Nodes that have been removed since last commit
    nodes_removed: Vec<NodeId>,

    /// Edges that have been added since last commit
    /// Format: (edge_id, source_node, target_node)
    edges_added: Vec<(EdgeId, NodeId, NodeId)>,

    /// Edges that have been removed since last commit
    edges_removed: Vec<EdgeId>,

    /// Node attribute changes since last commit
    /// Format: (node_id, attr_name, old_index, new_index)
    /// old_index = None means attribute was newly set
    node_attr_index_changes: Vec<(NodeId, AttrName, Option<usize>, usize)>,

    /// Edge attribute changes since last commit
    /// Same format as node_attr_index_changes
    edge_attr_index_changes: Vec<(EdgeId, AttrName, Option<usize>, usize)>,

    /// When the first change was made (for timing statistics)
    first_change_timestamp: Option<u64>,

    /// Total number of changes recorded
    total_changes: usize,
}

impl IndexDeltaStrategy {
    /// Create a new empty index delta strategy
    pub fn new() -> Self {
        Self {
            nodes_added: Vec::new(),
            nodes_removed: Vec::new(),
            edges_added: Vec::new(),
            edges_removed: Vec::new(),
            node_attr_index_changes: Vec::new(),
            edge_attr_index_changes: Vec::new(),
            first_change_timestamp: None,
            total_changes: 0,
        }
    }

    /// Record node attribute change with indices (strategy-specific API)
    /// This is the index-specific version that stores column indices
    pub fn record_node_attr_index_change(
        &mut self,
        node_id: NodeId,
        attr_name: AttrName,
        old_index: Option<usize>,
        new_index: usize,
    ) {
        // Look for existing change for this node/attribute
        if let Some(existing) = self
            .node_attr_index_changes
            .iter_mut()
            .find(|(id, name, _, _)| *id == node_id && name == &attr_name)
        {
            // Update the new index, keep the original old index
            existing.3 = new_index;
        } else {
            // New change
            self.node_attr_index_changes
                .push((node_id, attr_name, old_index, new_index));
        }

        self.update_change_metadata();
    }

    /// Record edge attribute change with indices (strategy-specific API)
    pub fn record_edge_attr_index_change(
        &mut self,
        edge_id: EdgeId,
        attr_name: AttrName,
        old_index: Option<usize>,
        new_index: usize,
    ) {
        // Same pattern as node attributes but for edges
        if let Some(existing) = self
            .edge_attr_index_changes
            .iter_mut()
            .find(|(id, name, _, _)| *id == edge_id && name == &attr_name)
        {
            existing.3 = new_index;
        } else {
            self.edge_attr_index_changes
                .push((edge_id, attr_name, old_index, new_index));
        }

        self.update_change_metadata();
    }

    /// Helper method to update change metadata
    fn update_change_metadata(&mut self) {
        self.total_changes += 1;
        if self.first_change_timestamp.is_none() {
            self.first_change_timestamp = Some(self.current_timestamp());
        }
    }

    /// Get current timestamp
    fn current_timestamp(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

impl TemporalStorageStrategy for IndexDeltaStrategy {
    fn record_node_addition(&mut self, node_id: NodeId) {
        self.nodes_added.push(node_id);
        self.update_change_metadata();
    }

    fn record_node_removal(&mut self, node_id: NodeId) {
        self.nodes_removed.push(node_id);
        // Remove any attribute changes for this node since it's being deleted
        self.node_attr_index_changes
            .retain(|(id, _, _, _)| *id != node_id);
        self.update_change_metadata();
    }

    fn record_edge_addition(&mut self, edge_id: EdgeId, source: NodeId, target: NodeId) {
        self.edges_added.push((edge_id, source, target));
        self.update_change_metadata();
    }

    fn record_edge_removal(&mut self, edge_id: EdgeId) {
        self.edges_removed.push(edge_id);
        // Remove any attribute changes for this edge since it's being deleted
        self.edge_attr_index_changes
            .retain(|(id, _, _, _)| *id != edge_id);
        self.update_change_metadata();
    }

    fn record_node_attr_change(
        &mut self,
        node_id: NodeId,
        attr_name: AttrName,
        old_index: Option<usize>,
        new_index: usize,
    ) {
        // Now the trait method directly receives indices - much cleaner!
        self.record_node_attr_index_change(node_id, attr_name, old_index, new_index);
    }

    fn record_edge_attr_change(
        &mut self,
        edge_id: EdgeId,
        attr_name: AttrName,
        old_index: Option<usize>,
        new_index: usize,
    ) {
        // Same clean delegation to index-specific method
        self.record_edge_attr_index_change(edge_id, attr_name, old_index, new_index);
    }

    fn create_delta(&self) -> DeltaObject {
        let mut node_attr_indices = HashMap::new();

        // Group node attribute index changes by attribute name
        for (node_id, attr_name, old_index, new_index) in &self.node_attr_index_changes {
            let column_delta = node_attr_indices
                .entry(attr_name.clone())
                .or_insert_with(ColumnIndexDelta::new);

            column_delta.add_index_change(*node_id as usize, *old_index, *new_index);
        }

        // Similar for edge attributes
        let mut edge_attr_indices = HashMap::new();
        for (edge_id, attr_name, old_index, new_index) in &self.edge_attr_index_changes {
            let column_delta = edge_attr_indices
                .entry(attr_name.clone())
                .or_insert_with(ColumnIndexDelta::new);

            column_delta.add_index_change(*edge_id as usize, *old_index, *new_index);
        }

        // Create delta with index-based changes
        DeltaObject::new_with_indices(
            node_attr_indices,
            edge_attr_indices,
            self.nodes_added.clone(),
            self.nodes_removed.clone(),
            self.edges_added.clone(),
            self.edges_removed.clone(),
        )
    }

    fn create_change_set(&self) -> ChangeSet {
        // Convert index-based changes to ChangeSet format for HistoryForest
        let mut node_attr_changes = Vec::new();
        for (node_id, attr_name, _old_index, _new_index) in &self.node_attr_index_changes {
            // For now, we don't have the actual values, so we use placeholder values
            // In a full implementation, we would resolve the indices to actual values
            node_attr_changes.push((
                *node_id,
                attr_name.clone(),
                None, // old_value - would need to resolve from old_index
                AttrValue::Text(format!("index_{}", _new_index)), // placeholder new value
            ));
        }

        let mut edge_attr_changes = Vec::new();
        for (edge_id, attr_name, _old_index, _new_index) in &self.edge_attr_index_changes {
            edge_attr_changes.push((
                *edge_id,
                attr_name.clone(),
                None,                                             // old_value
                AttrValue::Text(format!("index_{}", _new_index)), // placeholder new value
            ));
        }

        ChangeSet {
            nodes_added: self.nodes_added.clone(),
            nodes_removed: self.nodes_removed.clone(),
            edges_added: self.edges_added.clone(),
            edges_removed: self.edges_removed.clone(),
            node_attr_changes,
            edge_attr_changes,
        }
    }

    fn has_changes(&self) -> bool {
        !self.nodes_added.is_empty()
            || !self.nodes_removed.is_empty()
            || !self.edges_added.is_empty()
            || !self.edges_removed.is_empty()
            || !self.node_attr_index_changes.is_empty()
            || !self.edge_attr_index_changes.is_empty()
    }

    fn change_count(&self) -> usize {
        self.total_changes
    }

    fn clear_changes(&mut self) {
        self.nodes_added.clear();
        self.nodes_removed.clear();
        self.edges_added.clear();
        self.edges_removed.clear();
        self.node_attr_index_changes.clear();
        self.edge_attr_index_changes.clear();
        self.first_change_timestamp = None;
        self.total_changes = 0;
    }

    fn strategy_name(&self) -> &'static str {
        "Index Delta Strategy"
    }

    fn storage_characteristics(&self) -> StorageCharacteristics {
        StorageCharacteristics {
            storage_overhead: 2,     // Very low (only indices)
            commit_speed: 9,         // Very fast (no value copying)
            reconstruction_speed: 6, // Moderate (requires index resolution)
            supports_partial_reconstruction: true,
            prefers_frequent_commits: true,
            description: "Efficient temporal storage using column indices. \
                         Best for frequent changes and analytical workloads.",
        }
    }

    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Default for IndexDeltaStrategy {
    fn default() -> Self {
        Self::new()
    }
}
