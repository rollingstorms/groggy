//! Graph Space - Active State Tracker
//!
//! ARCHITECTURE ROLE:
//! GraphSpace is a **minimal active state tracker**. It simply knows which
//! nodes and edges are currently "active" and tracks changes. The actual
//! graph operations happen in the Graph coordinator.
//!
//! DESIGN PHILOSOPHY:
//! - GraphSpace = Active Set + Change Tracking (minimal responsibility)
//! - Graph = Operations & Coordination (delegates to pool for data)
//! - GraphPool = Pure Data Storage (the "database tables")
//! - Keep GraphSpace minimal and focused

/*
=== GRAPH SPACE RESPONSIBILITIES ===

GraphSpace is JUST the active state tracker and change recorder.
It's not a full interface - that's Graph's job.

KEY RESPONSIBILITIES:
1. ACTIVE SET MANAGEMENT: Track which nodes/edges are currently active
2. CHANGE TRACKING: Record every modification for history commits
3. WORKSPACE STATE: Basic queries about current active state

WHAT BELONGS HERE:
- active_nodes, active_edges HashSets
- change_tracker methods
- Simple queries: node_count(), contains_node(), etc.
- Workspace management: has_changes(), reset(), etc.

WHAT DOESN'T BELONG HERE:
- Graph operations like add_node(), set_attr() (that's Graph's job)
- Complex topology queries (Graph delegates to Pool)
- History operations (that's HistoryForest's job)
- Analytics (that's QueryEngine's job)
*/

use std::collections::{HashMap, HashSet};
use crate::types::{NodeId, EdgeId, AttrName, AttrValue, StateId};
use crate::core::change_tracker::ChangeTracker;
use crate::core::strategies::StorageStrategyType;
use crate::errors::{GraphError, GraphResult};
use crate::core::pool::GraphPool;
use crate::core::delta::DeltaObject;

/// Minimal active state tracker for the current graph
/// 
/// DESIGN: GraphSpace is just the "active set" tracker. It knows which
/// nodes and edges are currently active, and tracks changes for commits.
/// The Graph coordinator handles actual operations.
/// 
/// KEY INSIGHT: 
/// - GraphSpace = Active Sets + Change Tracking (minimal)
/// - Graph = Operations & Coordination
/// - GraphPool = Pure Data Storage
/// 
/// LIFECYCLE:
/// 1. Created from a historical state (or empty)  
/// 2. Graph calls methods to update active sets
/// 3. Changes tracked automatically
/// 4. Eventually committed to create new historical state
#[derive(Debug)]
pub struct GraphSpace {
    
    /*
    === ACTIVE SET TRACKING ===
    GraphSpace only tracks which entities are currently active
    */
    
    /// All currently active (not deleted) nodes
    /// DESIGN: HashSet for O(1) contains() and fast iteration
    active_nodes: HashSet<NodeId>,
    
    /// All currently active (not deleted) edges  
    /// DESIGN: HashSet for O(1) contains() and fast iteration
    active_edges: HashSet<EdgeId>,
    
    /*
    === ATTRIBUTE INDEX MAPPINGS ===
    Map entities to their current attribute indices in Pool columns
    */
    
    /// Maps node -> attribute_name -> column_index
    /// This is how we resolve "what is node X's current value for attribute Y"
    node_attribute_indices: HashMap<NodeId, HashMap<AttrName, usize>>,
    
    /// Maps edge -> attribute_name -> column_index
    /// Same pattern as nodes but for edges
    edge_attribute_indices: HashMap<EdgeId, HashMap<AttrName, usize>>,
    
    /*
    === WORKSPACE METADATA ===
    Information about this workspace state
    */
    
    /// Which historical state this workspace is based on
    /// Used for computing deltas when committing
    base_state: StateId,
    
    /// Tracks all modifications since the last commit
    /// Used to create deltas for history storage
    change_tracker: ChangeTracker,
    
    /// Whether this workspace has been modified
    /// Optimization to avoid change tracking overhead
    has_changes: bool,
}

impl GraphSpace {
    /// Create a new empty graph space with default strategy
    pub fn new(base_state: StateId) -> Self {
        Self::with_strategy(base_state, StorageStrategyType::default())
    }
    
    /// Create a new graph space with specific temporal storage strategy
    pub fn with_strategy(base_state: StateId, strategy: StorageStrategyType) -> Self {
        Self {
            active_nodes: HashSet::new(),
            active_edges: HashSet::new(),
            node_attribute_indices: HashMap::new(),
            edge_attribute_indices: HashMap::new(),
            base_state,
            change_tracker: ChangeTracker::with_strategy(strategy),
            has_changes: false,
        }
    }


    /*
    === ACTIVE SET MANAGEMENT ===
    Legacy methods for external activation/deactivation
    */

    /// Add a node to the active set (called by Graph.add_node())
    pub fn activate_node(&mut self, node_id: NodeId) {
        self.active_nodes.insert(node_id);
        self.change_tracker.record_node_addition(node_id);
        self.has_changes = true;
    }

    /// Remove a node from the active set (called by Graph.remove_node())
    pub fn deactivate_node(&mut self, node_id: NodeId) {
        self.active_nodes.remove(&node_id);
        // Also remove all attribute indices for this node
        self.node_attribute_indices.remove(&node_id);
        self.change_tracker.record_node_removal(node_id);
        self.has_changes = true;
    }

    /// Add an edge to the active set (called by Graph.add_edge())
    pub fn activate_edge(&mut self, edge_id: EdgeId, source: NodeId, target: NodeId) {
        self.active_edges.insert(edge_id);
        self.change_tracker.record_edge_addition(edge_id, source, target);
        self.has_changes = true;
    }

    /// Remove an edge from the active set (called by Graph.remove_edge())
    pub fn deactivate_edge(&mut self, edge_id: EdgeId) {
        self.active_edges.remove(&edge_id);
        // Also remove all attribute indices for this edge
        self.edge_attribute_indices.remove(&edge_id);
        self.change_tracker.record_edge_removal(edge_id);
        self.has_changes = true;
    }

    /*
    === ATTRIBUTE INDEX MANAGEMENT ===
    Methods for managing node/edge -> attribute -> index mappings
    */
    
    /// Set node attribute using index-based storage
    pub fn set_node_attr(&mut self, pool: &mut GraphPool, node_id: NodeId, attr_name: AttrName, value: AttrValue) -> GraphResult<()> {
        // ALGORITHM: Index-based attribute setting
        // 1. Get current index (if any) for change tracking
        // 2. Append new value to Pool and get new index
        // 3. Update our index mapping
        // 4. Record the index change for history
        
        let old_index = self.node_attribute_indices
            .get(&node_id)
            .and_then(|attrs| attrs.get(&attr_name))
            .copied();
        
        // Append new value to pool and get new index
        let new_index = pool.append_node_attr_value(attr_name.clone(), value);
        
        // Update our index mapping
        self.node_attribute_indices
            .entry(node_id)
            .or_insert_with(HashMap::new)
            .insert(attr_name.clone(), new_index);
        
        // Record the change with indices instead of values
        self.change_tracker.record_node_attr_index_change(node_id, attr_name, old_index, new_index);
        self.has_changes = true;
        
        Ok(())
    }
    
    /// Get node attribute by resolving index through pool
    pub fn get_node_attr(&self, pool: &GraphPool, node_id: NodeId, attr_name: &AttrName) -> GraphResult<Option<&AttrValue>> {
        // ALGORITHM: Index resolution
        // 1. Get the index for this node/attribute combination
        // 2. Use Pool to resolve index to value
        
        if let Some(attrs) = self.node_attribute_indices.get(&node_id) {
            if let Some(&index) = attrs.get(attr_name) {
                return Ok(pool.get_node_attr_by_index(attr_name, index));
            }
        }
        Ok(None)
    }

    /// Set edge attribute using index-based storage
    pub fn set_edge_attr(&mut self, pool: &mut GraphPool, edge_id: EdgeId, attr_name: AttrName, value: AttrValue) -> GraphResult<()> {
        let old_index = self.edge_attribute_indices
            .get(&edge_id)
            .and_then(|attrs| attrs.get(&attr_name))
            .copied();
        
        let new_index = pool.append_edge_attr_value(attr_name.clone(), value);
        
        self.edge_attribute_indices
            .entry(edge_id)
            .or_insert_with(HashMap::new)
            .insert(attr_name.clone(), new_index);
        
        self.change_tracker.record_edge_attr_index_change(edge_id, attr_name, old_index, new_index);
        self.has_changes = true;
        
        Ok(())
    }
    
    /// Get edge attribute by resolving index through pool
    pub fn get_edge_attr(&self, pool: &GraphPool, edge_id: EdgeId, attr_name: &AttrName) -> GraphResult<Option<&AttrValue>> {
        if let Some(attrs) = self.edge_attribute_indices.get(&edge_id) {
            if let Some(&index) = attrs.get(attr_name) {
                return Ok(pool.get_edge_attr_by_index(attr_name, index));
            }
        }
        Ok(None)
    }
    
    /// Get all attributes for a node by resolving indices through pool
    pub fn get_all_node_attrs(&self, pool: &GraphPool, node_id: NodeId) -> HashMap<AttrName, AttrValue> {
        let mut result = HashMap::new();
        if let Some(attrs) = self.node_attribute_indices.get(&node_id) {
            for (attr_name, &index) in attrs {
                if let Some(value) = pool.get_node_attr_by_index(attr_name, index) {
                    result.insert(attr_name.clone(), value.clone());
                }
            }
        }
        result
    }
    
    /// Get all attributes for an edge by resolving indices through pool
    pub fn get_all_edge_attrs(&self, pool: &GraphPool, edge_id: EdgeId) -> HashMap<AttrName, AttrValue> {
        let mut result = HashMap::new();
        if let Some(attrs) = self.edge_attribute_indices.get(&edge_id) {
            for (attr_name, &index) in attrs {
                if let Some(value) = pool.get_edge_attr_by_index(attr_name, index) {
                    result.insert(attr_name.clone(), value.clone());
                }
            }
        }
        result
    }

    /*
    === SIMPLE ACTIVE STATE QUERIES ===
    Basic information about what's currently active
    */

    /// Get the number of active nodes
    pub fn node_count(&self) -> usize {
        self.active_nodes.len()
    }

    /// Get the number of active edges
    pub fn edge_count(&self) -> usize {
        self.active_edges.len()
    }

    /// Check if a node is currently active
    pub fn contains_node(&self, node_id: NodeId) -> bool {
        self.active_nodes.contains(&node_id)
    }

    /// Check if an edge is currently active
    pub fn contains_edge(&self, edge_id: EdgeId) -> bool {
        self.active_edges.contains(&edge_id)
    }

    /// Get all active node IDs (for iteration)
    pub fn get_active_nodes(&self) -> &HashSet<NodeId> {
        &self.active_nodes
    }

    /// Get all active edge IDs (for iteration)
    pub fn get_active_edges(&self) -> &HashSet<EdgeId> {
        &self.active_edges
    }

    /*
    === BASIC QUERIES ===
    Simple active set queries - topology handled by Graph coordinator
    */

    /// Get all active node IDs as a vector
    pub fn node_ids(&self) -> Vec<NodeId> {
        // TODO: self.active_nodes.iter().copied().collect()
        todo!("Implement GraphSpace::node_ids")
    }

    /// Get all active edge IDs as a vector
    pub fn edge_ids(&self) -> Vec<EdgeId> {
        // TODO: self.active_edges.iter().copied().collect()
        todo!("Implement GraphSpace::edge_ids")
    }
    
    /// NOTE: Topology queries (neighbors, degree, connectivity) now handled by
    /// Graph coordinator which queries Pool for topology and filters by Space's active sets

    /*
    === CHANGE TRACKING ===
    Workspace state and change management
    */

    /// Check if there are uncommitted changes
    pub fn has_uncommitted_changes(&self) -> bool {
        // TODO: self.has_changes
        todo!("Implement GraphSpace::has_uncommitted_changes")
    }

    /// Get the number of uncommitted changes
    pub fn uncommitted_change_count(&self) -> usize {
        // TODO: self.change_tracker.change_count()
        todo!("Implement GraphSpace::uncommitted_change_count")
    }

    /// Get summary of uncommitted changes
    pub fn change_summary(&self) -> ChangeSummary {
        // TODO: self.change_tracker.change_summary()
        todo!("Implement GraphSpace::change_summary")
    }

    /// Get the base state this workspace is built on
    pub fn get_base_state(&self) -> StateId {
        // TODO: self.base_state
        todo!("Implement GraphSpace::get_base_state")
    }

    /// Create a delta object representing current changes
    /// USAGE: Called when committing changes to history
    pub fn create_change_delta(&self, pool: &GraphPool) -> DeltaObject {
        // TODO: ALGORITHM - Efficient delta creation using Pool's change tracking
        // 1. self.change_tracker.create_delta(pool)
        
        // PERFORMANCE: O(changed_entities) - leverages Pool's efficient iteration
        // INTEGRATION: ChangeTracker uses Pool's BitVec-based change detection
        todo!("Implement GraphSpace::create_change_delta")
    }

    /// Clear all uncommitted changes (reset to base state)
    /// WARNING: This loses all work since the last commit
    pub fn reset_hard(&mut self) -> GraphResult<()> {
        // TODO:
        // 1. Clear active sets (reload from base_state)
        // 2. Clear change tracker
        // 3. Reset has_changes flag
        todo!("Implement GraphSpace::reset_hard")
    }
}

/*
=== SUPPORTING DATA STRUCTURES ===
*/

use crate::core::change_tracker::ChangeSummary;

/*
=== IMPLEMENTATION NOTES ===

GraphSpace is now properly minimal - it just manages:
1. Active sets (which nodes/edges are currently active)
2. Change tracking (what's changed since last commit)
3. Basic queries (count, contains, etc.)

All actual graph operations happen in Graph, which coordinates between:
- GraphSpace (active state)
- GraphPool (data storage) 
- HistoryForest (version control)
- QueryEngine (analytics)

This separation makes the architecture much cleaner and easier to understand.
*/
