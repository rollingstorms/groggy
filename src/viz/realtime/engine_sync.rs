//! Engine Synchronization and Update Coalescing
//!
//! This module provides ordered apply/merge logic and update coalescing for
//! the realtime visualization engine to ensure predictable state management.

use crate::errors::{GraphError, GraphResult};
use crate::types::{AttrValue, NodeId};
use crate::viz::realtime::accessor::{EngineSnapshot, EngineUpdate};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Handles ordered application and coalescing of engine updates
#[derive(Debug)]
pub struct EngineSyncManager {
    /// Pending updates queue (maintains order)
    pending_updates: VecDeque<TimestampedUpdate>,

    /// Coalescing buffer for position deltas
    position_coalescing: HashMap<NodeId, CoalescedPositionDelta>,

    /// Coalescing buffer for attribute changes
    attribute_coalescing: HashMap<(String, String, String), CoalescedAttributeChange>, // (entity_type, entity_id, attr_name)

    /// Last snapshot timestamp (for ordering guarantee)
    last_snapshot_time: Option<Instant>,

    /// Coalescing window duration
    coalescing_window: Duration,

    /// Maximum batch size for processing
    max_batch_size: usize,
}

/// Update with timestamp for ordering
#[derive(Debug, Clone)]
struct TimestampedUpdate {
    update: EngineUpdate,
    timestamp: Instant,
    #[allow(dead_code)]
    sequence_id: u64,
}

/// Coalesced position delta for a node
#[derive(Debug, Clone)]
struct CoalescedPositionDelta {
    node_id: NodeId,
    accumulated_delta: Vec<f64>,
    #[allow(dead_code)]
    first_timestamp: Instant,
    last_timestamp: Instant,
    update_count: usize,
}

/// Coalesced attribute change
#[derive(Debug, Clone)]
struct CoalescedAttributeChange {
    entity_type: String,
    entity_id: String,
    attr_name: String,
    final_value: AttrValue,
    #[allow(dead_code)]
    first_timestamp: Instant,
    last_timestamp: Instant,
    update_count: usize,
}

impl EngineSyncManager {
    /// Create new synchronization manager
    pub fn new() -> Self {
        Self {
            pending_updates: VecDeque::new(),
            position_coalescing: HashMap::new(),
            attribute_coalescing: HashMap::new(),
            last_snapshot_time: None,
            coalescing_window: Duration::from_millis(16), // ~60 FPS
            max_batch_size: 50,
        }
    }

    /// Configure coalescing window
    pub fn set_coalescing_window(&mut self, window: Duration) {
        self.coalescing_window = window;
    }

    /// Configure maximum batch size
    pub fn set_max_batch_size(&mut self, size: usize) {
        self.max_batch_size = size;
    }

    /// Queue a snapshot (clears all pending updates to ensure ordering)
    pub async fn queue_snapshot(
        &mut self,
        snapshot: EngineSnapshot,
    ) -> GraphResult<Vec<EngineUpdate>> {
        let timestamp = Instant::now();

        // Clear all pending updates - snapshot takes precedence
        self.pending_updates.clear();
        self.position_coalescing.clear();
        self.attribute_coalescing.clear();

        self.last_snapshot_time = Some(timestamp);

        // Convert snapshot to a special update for consistency
        let snapshot_update = EngineUpdate::SnapshotLoaded {
            node_count: snapshot.node_count(),
            edge_count: snapshot.edge_count(),
        };

        Ok(vec![snapshot_update])
    }

    /// Queue an update with ordering and coalescing
    pub async fn queue_update(
        &mut self,
        update: EngineUpdate,
        sequence_id: u64,
    ) -> GraphResult<()> {
        let timestamp = Instant::now();

        // Ensure snapshot precedes any update
        if let Some(snapshot_time) = self.last_snapshot_time {
            if timestamp < snapshot_time {
                return Err(GraphError::InvalidState {
                    operation: "queue_update".to_string(),
                    expected_state: "timestamp after snapshot".to_string(),
                    actual_state: "timestamp before snapshot".to_string(),
                    suggestion: "Updates must come after snapshot loading".to_string(),
                });
            }
        }

        // Try to coalesce the update
        if self.try_coalesce_update(&update, timestamp) {
            return Ok(());
        }

        // Add to pending queue if not coalesced
        self.pending_updates.push_back(TimestampedUpdate {
            update,
            timestamp,
            sequence_id,
        });

        // Ensure queue doesn't grow too large
        if self.pending_updates.len() > self.max_batch_size * 2 {
            self.flush_oldest_updates().await?;
        }

        Ok(())
    }

    /// Try to coalesce an update with existing ones
    fn try_coalesce_update(&mut self, update: &EngineUpdate, timestamp: Instant) -> bool {
        match update {
            EngineUpdate::PositionDelta { node_id, delta } => {
                // Coalesce position deltas for the same node
                if let Some(existing) = self.position_coalescing.get_mut(node_id) {
                    // Add to accumulated delta
                    for (i, d) in delta.iter().enumerate() {
                        if i < existing.accumulated_delta.len() {
                            existing.accumulated_delta[i] += d;
                        } else {
                            existing.accumulated_delta.push(*d);
                        }
                    }
                    existing.last_timestamp = timestamp;
                    existing.update_count += 1;
                    true
                } else {
                    // Start new coalescing entry
                    self.position_coalescing.insert(
                        *node_id,
                        CoalescedPositionDelta {
                            node_id: *node_id,
                            accumulated_delta: delta.clone(),
                            first_timestamp: timestamp,
                            last_timestamp: timestamp,
                            update_count: 1,
                        },
                    );
                    true
                }
            }
            EngineUpdate::NodeChanged { id, attributes } => {
                // For node attribute changes, coalesce by node ID and attribute name
                let mut coalesced = false;
                for (attr_name, new_value) in attributes {
                    let key = ("node".to_string(), id.to_string(), attr_name.clone());

                    if let Some(existing) = self.attribute_coalescing.get_mut(&key) {
                        // Update final value (overwrites previous)
                        existing.final_value = new_value.clone();
                        existing.last_timestamp = timestamp;
                        existing.update_count += 1;
                        coalesced = true;
                    } else {
                        // Start new coalescing entry
                        self.attribute_coalescing.insert(
                            key,
                            CoalescedAttributeChange {
                                entity_type: "node".to_string(),
                                entity_id: id.to_string(),
                                attr_name: attr_name.clone(),
                                final_value: new_value.clone(),
                                first_timestamp: timestamp,
                                last_timestamp: timestamp,
                                update_count: 1,
                            },
                        );
                        coalesced = true;
                    }
                }
                coalesced
            }
            _ => {
                // Don't coalesce other update types
                false
            }
        }
    }

    /// Get ready updates (respecting coalescing window and ordering)
    pub async fn get_ready_updates(&mut self) -> GraphResult<Vec<EngineUpdate>> {
        let now = Instant::now();
        let mut ready_updates = Vec::new();

        // Check coalesced position deltas
        let mut expired_positions = Vec::new();
        for (node_id, delta) in &self.position_coalescing {
            if now.duration_since(delta.last_timestamp) >= self.coalescing_window {
                expired_positions.push(*node_id);
            }
        }

        for node_id in expired_positions {
            if let Some(delta) = self.position_coalescing.remove(&node_id) {
                ready_updates.push(EngineUpdate::PositionDelta {
                    node_id: delta.node_id,
                    delta: delta.accumulated_delta,
                });
            }
        }

        // Check coalesced attribute changes
        let mut expired_attributes = Vec::new();
        for (key, change) in &self.attribute_coalescing {
            if now.duration_since(change.last_timestamp) >= self.coalescing_window {
                expired_attributes.push(key.clone());
            }
        }

        for key in expired_attributes {
            if let Some(change) = self.attribute_coalescing.remove(&key) {
                // Convert coalesced attribute change back to appropriate EngineUpdate variant
                match change.entity_type.as_str() {
                    "node" => {
                        if let Ok(node_id) = change.entity_id.parse::<crate::types::NodeId>() {
                            let mut attributes = std::collections::HashMap::new();
                            attributes.insert(change.attr_name, change.final_value);
                            ready_updates.push(EngineUpdate::NodeChanged {
                                id: node_id,
                                attributes,
                            });
                        }
                    }
                    "edge" => {
                        if let Ok(edge_id) = change.entity_id.parse::<crate::types::EdgeId>() {
                            let mut attributes = std::collections::HashMap::new();
                            attributes.insert(change.attr_name, change.final_value);
                            ready_updates.push(EngineUpdate::EdgeChanged {
                                id: edge_id,
                                attributes,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }

        // Process pending non-coalesced updates (respect ordering and batch size)
        let mut processed_count = 0;
        while let Some(timestamped) = self.pending_updates.front() {
            if processed_count >= self.max_batch_size {
                break;
            }

            // Check if update is ready (not too recent)
            if now.duration_since(timestamped.timestamp) >= self.coalescing_window {
                let timestamped = self.pending_updates.pop_front().unwrap();
                ready_updates.push(timestamped.update);
                processed_count += 1;
            } else {
                break; // Maintain order - don't skip newer updates
            }
        }

        Ok(ready_updates)
    }

    /// Force flush oldest updates (emergency pressure relief)
    async fn flush_oldest_updates(&mut self) -> GraphResult<()> {
        let flush_count = self.max_batch_size;
        for _ in 0..flush_count {
            if self.pending_updates.pop_front().is_some() {
                // Just discard - this is emergency mode
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Force flush all coalesced updates immediately
    pub async fn flush_all_coalesced(&mut self) -> GraphResult<Vec<EngineUpdate>> {
        let mut updates = Vec::new();

        // Flush all position deltas
        for (_, delta) in self.position_coalescing.drain() {
            updates.push(EngineUpdate::PositionDelta {
                node_id: delta.node_id,
                delta: delta.accumulated_delta,
            });
        }

        // Flush all attribute changes
        for (_, change) in self.attribute_coalescing.drain() {
            // Convert coalesced attribute change back to appropriate EngineUpdate variant
            match change.entity_type.as_str() {
                "node" => {
                    if let Ok(node_id) = change.entity_id.parse::<crate::types::NodeId>() {
                        let mut attributes = std::collections::HashMap::new();
                        attributes.insert(change.attr_name, change.final_value);
                        updates.push(EngineUpdate::NodeChanged {
                            id: node_id,
                            attributes,
                        });
                    }
                }
                "edge" => {
                    if let Ok(edge_id) = change.entity_id.parse::<crate::types::EdgeId>() {
                        let mut attributes = std::collections::HashMap::new();
                        attributes.insert(change.attr_name, change.final_value);
                        updates.push(EngineUpdate::EdgeChanged {
                            id: edge_id,
                            attributes,
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(updates)
    }

    /// Get synchronization statistics
    pub fn get_stats(&self) -> SyncStats {
        SyncStats {
            pending_update_count: self.pending_updates.len(),
            coalesced_position_count: self.position_coalescing.len(),
            coalesced_attribute_count: self.attribute_coalescing.len(),
            coalescing_window_ms: self.coalescing_window.as_millis() as u64,
            max_batch_size: self.max_batch_size,
        }
    }
}

/// Synchronization statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SyncStats {
    pub pending_update_count: usize,
    pub coalesced_position_count: usize,
    pub coalesced_attribute_count: usize,
    pub coalescing_window_ms: u64,
    pub max_batch_size: usize,
}

// Add SnapshotLoaded variant to EngineUpdate enum if it doesn't exist
// This would need to be added to the accessor module
