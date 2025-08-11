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
- NOTE: change tracking moved to Graph for cleaner separation
- Simple queries: node_count(), contains_node(), etc.
- Workspace management: reset(), state queries, etc.

WHAT DOESN'T BELONG HERE:
- Graph operations like add_node(), set_attr() (that's Graph's job)
- Complex topology queries (Graph delegates to Pool)
- History operations (that's HistoryForest's job)
- Analytics (that's QueryEngine's job)
*/

use std::collections::{HashMap, HashSet};
use crate::types::{NodeId, EdgeId, AttrName, AttrValue, StateId};
// NOTE: ChangeTracker import removed - Graph manages it directly now
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
    
    // NOTE: ChangeTracker moved to Graph for cleaner separation
    // Graph coordinates between Space (current state) and ChangeTracker (deltas)
}

impl GraphSpace {
    /// Create a new empty graph space 
    pub fn new(base_state: StateId) -> Self {
        Self {
            active_nodes: HashSet::new(),
            active_edges: HashSet::new(),
            node_attribute_indices: HashMap::new(),
            edge_attribute_indices: HashMap::new(),
            base_state,
        }
    }


    /*
    === ACTIVE SET MANAGEMENT ===
    Legacy methods for external activation/deactivation
    */

    /// Add a node to the active set (called by Graph.add_node())
    pub fn activate_node(&mut self, node_id: NodeId) {
        self.active_nodes.insert(node_id);
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
    }

    /// Add multiple nodes to the active set (called by Graph.add_nodes())
    pub fn activate_nodes(&mut self, nodes: &[NodeId]) {
        self.active_nodes.extend(nodes);
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
    }

    /// Remove multiple nodes from the active set (called by Graph.remove_nodes())
    pub fn deactivate_nodes(&mut self, nodes: &[NodeId]) {
        self.active_nodes.retain(|node| !nodes.contains(node));
        // Also remove all attribute indices for these nodes
        for node in nodes {
            self.node_attribute_indices.remove(node);
        }
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
    }

    /// Remove a node from the active set (called by Graph.remove_node())
    pub fn deactivate_node(&mut self, node_id: NodeId) {
        self.active_nodes.remove(&node_id);
        // Also remove all attribute indices for this node
        self.node_attribute_indices.remove(&node_id);
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
    }

    /// Add an edge to the active set (called by Graph.add_edge())
    pub fn activate_edge(&mut self, edge_id: EdgeId, source: NodeId, target: NodeId) {
        self.active_edges.insert(edge_id);
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
    }

    /// Remove an edge from the active set (called by Graph.remove_edge())
    pub fn deactivate_edge(&mut self, edge_id: EdgeId) {
        self.active_edges.remove(&edge_id);
        // Also remove all attribute indices for this edge
        self.edge_attribute_indices.remove(&edge_id);
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
    }

    /// Add multiple edges to the active set (called by Graph.add_edges())
    pub fn activate_edges(&mut self, edges: &[EdgeId]) {
        self.active_edges.extend(edges);
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
    }

    /// Remove multiple edges from the active set (called by Graph.remove_edges())
    pub fn deactivate_edges(&mut self, edges: &[EdgeId]) {
        self.active_edges.retain(|edge| !edges.contains(edge));
        // Also remove all attribute indices for these edges
        for edge in edges {
            self.edge_attribute_indices.remove(edge);
        }
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
    }

    /*
    === CURRENT INDEX MAPPINGS ===
    Space ONLY manages which entities have which current attribute indices
    Graph coordinates between Space (current state) and ChangeTracker (deltas)
    */
    
    /// Update the current attribute index for any entity (called by Graph after Pool storage)
    pub fn set_attr_index<T>(&mut self, entity_id: T, attr_name: AttrName, new_index: usize, is_node: bool) 
    where T: Into<u64> + Copy {
        let id = entity_id.into();
        if is_node {
            self.node_attribute_indices
                .entry(id)
                .or_insert_with(HashMap::new)
                .insert(attr_name, new_index);
        } else {
            self.edge_attribute_indices
                .entry(id)
                .or_insert_with(HashMap::new)
                .insert(attr_name, new_index);
        }
    }
    
    /// Get current attribute index for any entity (used by Graph for change tracking)
    pub fn get_attr_index<T>(&self, entity_id: T, attr_name: &AttrName, is_node: bool) -> Option<usize> 
    where T: Into<u64> + Copy {
        let id = entity_id.into();
        let attribute_map = if is_node {
            &self.node_attribute_indices
        } else {
            &self.edge_attribute_indices
        };
        
        attribute_map
            .get(&id)
            .and_then(|attrs| attrs.get(attr_name))
            .copied()
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
        // TODO: Space no longer tracks changes directly
        todo!("Implement GraphSpace::has_uncommitted_changes")
    }

    /// Get the number of uncommitted changes
    pub fn uncommitted_change_count(&self) -> usize {
        // TODO: Graph tracks changes now
        todo!("Implement GraphSpace::uncommitted_change_count")
    }

    /// Get summary of uncommitted changes
    pub fn change_summary(&self) -> ChangeSummary {
        // TODO: Graph provides change summary now
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
        // NOTE: Graph creates delta via ChangeTracker now
        
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
        // NOTE: Graph manages change state now
        todo!("Implement GraphSpace::reset_hard")
    }
}

/*
=== SUPPORTING DATA STRUCTURES ===
*/

// NOTE: ChangeSummary import removed - Graph manages change tracking directly now

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
