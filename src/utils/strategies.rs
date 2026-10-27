//! Temporal Storage Strategies - Pluggable approaches for storing graph changes over time
//!
//! This module contains both the trait definition and all concrete implementations
//! of different temporal storage strategies.

use crate::errors::{GraphError, GraphResult};
use crate::state::change_tracker::ChangeSet;
use crate::state::delta::DeltaObject;
use crate::storage::pool::GraphPool;
use crate::types::{AttrName, EdgeId, NodeId};
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

    /// Existing external strategies explicitly opt into the new typed change kind.
    fn supports_endpoint_changes(&self) -> bool {
        false
    }
    /// Reject unsupported endpoint tracking instead of losing historical topology.
    fn record_edge_endpoint_change(
        &mut self,
        _change: crate::EdgeEndpointChange,
    ) -> GraphResult<()> {
        Err(GraphError::InvalidInput(
            "Temporal strategy does not support endpoint changes".into(),
        ))
    }

    /// Record that an edge was removed from the graph
    fn record_edge_removal(&mut self, edge_id: EdgeId);

    /// Record that a node attribute changed (index-based)
    /// All strategies now work with indices for consistency and efficiency
    fn record_node_attr_change(
        &mut self,
        node_id: NodeId,
        attr_name: AttrName,
        old_index: Option<usize>,
        new_index: Option<usize>,
    );

    /// Record multiple node attribute changes efficiently (bulk operation)
    /// Default implementation delegates to individual calls, but strategies can optimize
    fn record_node_attr_changes_bulk(
        &mut self,
        changes: &[(NodeId, AttrName, Option<usize>, Option<usize>)],
    ) {
        for &(node_id, ref attr_name, old_index, new_index) in changes {
            self.record_node_attr_change(node_id, attr_name.clone(), old_index, new_index);
        }
    }

    /// Record that an edge attribute changed (index-based)
    fn record_edge_attr_change(
        &mut self,
        edge_id: EdgeId,
        attr_name: AttrName,
        old_index: Option<usize>,
        new_index: Option<usize>,
    );

    /// Record multiple edge attribute changes efficiently (bulk operation)
    /// Default implementation delegates to individual calls, but strategies can optimize
    fn record_edge_attr_changes_bulk(
        &mut self,
        changes: &[(EdgeId, AttrName, Option<usize>, Option<usize>)],
    ) {
        for &(edge_id, ref attr_name, old_index, new_index) in changes {
            self.record_edge_attr_change(edge_id, attr_name.clone(), old_index, new_index);
        }
    }

    /*
    === DELTA CREATION ===
    Convert tracked changes into committable deltas
    */

    /// Borrow canonical pending physical records without fabricating logical values.
    fn indexed_changes(&self) -> crate::state::change::IndexChangeView<'_>;

    /// Create a complete explicitly indexed delta; no raw reference becomes a value.
    fn create_delta(&self) -> GraphResult<DeltaObject> {
        self.indexed_changes().to_index_delta()
    }
    /// Topology-only logical extraction. Property records require explicit pool resolution.
    fn create_change_set(&self) -> GraphResult<ChangeSet> {
        self.indexed_changes().without_values()
    }
    /// Resolve retained column references before publishing logical history.
    fn create_change_set_resolved(&self, pool: &GraphPool) -> GraphResult<ChangeSet> {
        self.indexed_changes().resolve(pool)
    }

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
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
    edge_add_positions: Option<HashMap<EdgeId, usize>>,

    /// Edges that have been removed since last commit
    edges_removed: Vec<EdgeId>,
    edge_endpoint_changes: HashMap<EdgeId, crate::EdgeEndpointChange>,

    /// Node attribute changes since last commit
    /// Format: (node_id, attr_name, old_index, new_index)
    /// old_index = None means attribute was newly set
    node_attr_index_changes: Vec<(NodeId, AttrName, Option<usize>, Option<usize>)>,

    /// Edge attribute changes since last commit
    /// Same format as node_attr_index_changes
    edge_attr_index_changes: Vec<(EdgeId, AttrName, Option<usize>, Option<usize>)>,

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
            edge_add_positions: None,
            edges_removed: Vec::new(),
            edge_endpoint_changes: HashMap::new(),
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
        new_index: Option<usize>,
    ) {
        // Append in operation order. Resolution coalesces all scalar/bulk
        // records together, keeping the first old value and final new value.
        self.node_attr_index_changes
            .push((node_id, attr_name, old_index, new_index));

        self.update_change_metadata();
    }

    /// Record edge attribute change with indices (strategy-specific API)
    pub fn record_edge_attr_index_change(
        &mut self,
        edge_id: EdgeId,
        attr_name: AttrName,
        old_index: Option<usize>,
        new_index: Option<usize>,
    ) {
        // Append in operation order. Resolution coalesces all scalar/bulk
        // records together, keeping the first old value and final new value.
        self.edge_attr_index_changes
            .push((edge_id, attr_name, old_index, new_index));

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
    fn supports_endpoint_changes(&self) -> bool {
        true
    }
    fn record_edge_endpoint_change(
        &mut self,
        change: crate::EdgeEndpointChange,
    ) -> GraphResult<()> {
        if !self.edges_added.is_empty() {
            let positions = self.edge_add_positions.get_or_insert_with(|| {
                self.edges_added
                    .iter()
                    .enumerate()
                    .map(|(i, row)| (row.0, i))
                    .collect()
            });
            if let Some(&position) = positions.get(&change.edge_id) {
                self.edges_added[position].1 = change.new_endpoints.0;
                self.edges_added[position].2 = change.new_endpoints.1;
                self.update_change_metadata();
                return Ok(());
            }
        }
        self.edge_endpoint_changes
            .entry(change.edge_id)
            .and_modify(|old| old.new_endpoints = change.new_endpoints)
            .or_insert(change);
        self.update_change_metadata();
        Ok(())
    }

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
        if let Some(positions) = &mut self.edge_add_positions {
            positions.insert(edge_id, self.edges_added.len() - 1);
        }
        self.update_change_metadata();
    }

    fn record_edge_removal(&mut self, edge_id: EdgeId) {
        self.edges_removed.push(edge_id);
        self.edge_endpoint_changes.remove(&edge_id);
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
        new_index: Option<usize>,
    ) {
        // Now the trait method directly receives indices - much cleaner!
        self.record_node_attr_index_change(node_id, attr_name, old_index, new_index);
    }

    // Optimized bulk recording for node attributes
    fn record_node_attr_changes_bulk(
        &mut self,
        changes: &[(NodeId, AttrName, Option<usize>, Option<usize>)],
    ) {
        // Reserve capacity upfront to avoid reallocations
        self.node_attr_index_changes.reserve(changes.len());

        // Bulk append without individual update_change_metadata calls
        for &(node_id, ref attr_name, old_index, new_index) in changes {
            self.node_attr_index_changes
                .push((node_id, attr_name.clone(), old_index, new_index));
        }

        // Single metadata update for the entire bulk operation
        self.total_changes += changes.len();
        if self.first_change_timestamp.is_none() {
            self.first_change_timestamp = Some(self.current_timestamp());
        }
    }

    fn record_edge_attr_change(
        &mut self,
        edge_id: EdgeId,
        attr_name: AttrName,
        old_index: Option<usize>,
        new_index: Option<usize>,
    ) {
        // Same clean delegation to index-specific method
        self.record_edge_attr_index_change(edge_id, attr_name, old_index, new_index);
    }

    // Optimized bulk recording for edge attributes
    fn record_edge_attr_changes_bulk(
        &mut self,
        changes: &[(EdgeId, AttrName, Option<usize>, Option<usize>)],
    ) {
        // Reserve capacity upfront to avoid reallocations
        self.edge_attr_index_changes.reserve(changes.len());

        // Bulk append without individual update_change_metadata calls
        for &(edge_id, ref attr_name, old_index, new_index) in changes {
            self.edge_attr_index_changes
                .push((edge_id, attr_name.clone(), old_index, new_index));
        }

        // Single metadata update for the entire bulk operation
        self.total_changes += changes.len();
        if self.first_change_timestamp.is_none() {
            self.first_change_timestamp = Some(self.current_timestamp());
        }
    }

    fn indexed_changes(&self) -> crate::state::change::IndexChangeView<'_> {
        crate::state::change::IndexChangeView {
            nodes_added: &self.nodes_added,
            nodes_removed: &self.nodes_removed,
            edges_added: &self.edges_added,
            edges_removed: &self.edges_removed,
            edge_endpoint_changes: &self.edge_endpoint_changes,
            node_attr_changes: &self.node_attr_index_changes,
            edge_attr_changes: &self.edge_attr_index_changes,
        }
    }

    fn has_changes(&self) -> bool {
        !self.nodes_added.is_empty()
            || !self.nodes_removed.is_empty()
            || !self.edges_added.is_empty()
            || !self.edges_removed.is_empty()
            || !self.edge_endpoint_changes.is_empty()
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
        self.edge_add_positions = None;
        self.edges_removed.clear();
        self.edge_endpoint_changes.clear();
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
