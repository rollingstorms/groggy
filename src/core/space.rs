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
use crate::types::{NodeId, EdgeId, AttrName, StateId};
// NOTE: ChangeTracker import removed - Graph manages it directly now
use crate::errors::GraphResult;
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
    === COLUMNAR TOPOLOGY CACHE ===
    Cached columnar representation of edges for vectorized topology queries
    */
    
    /// Edge sources in columnar format (parallel to edge_targets)
    /// DESIGN: Separate vectors for vectorized/SIMD operations
    edge_sources: Vec<NodeId>,
    
    /// Edge targets in columnar format (parallel to edge_sources) 
    edge_targets: Vec<NodeId>,
    
    /// Active edge IDs in columnar format (parallel to sources/targets)
    active_edge_ids: Vec<EdgeId>,
    
    /// Flag indicating if columnar cache needs rebuild
    topology_cache_dirty: bool,
    
    /// Topology generation counter - increments when structure changes
    topology_generation: usize,
    
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
            edge_sources: Vec::new(),
            edge_targets: Vec::new(),
            active_edge_ids: Vec::new(),
            topology_cache_dirty: false,
            topology_generation: 0,
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

    
    /// Bulk activate nodes from Vec (BULK OPTIMIZED for arbitrary IDs)
    /// 
    /// PERFORMANCE: Single HashSet::extend call instead of individual inserts
    pub fn activate_nodes(&mut self, nodes: Vec<NodeId>) {
        // Pre-grow HashSet
        self.active_nodes.reserve(nodes.len());
        
        // Single bulk extend operation
        self.active_nodes.extend(nodes);
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
    pub fn activate_edge(&mut self, edge_id: EdgeId, _source: NodeId, _target: NodeId) {
        self.active_edges.insert(edge_id);
        
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
        self.topology_cache_dirty = true;
        self.topology_generation += 1;
    }

    /// Remove an edge from the active set (called by Graph.remove_edge())
    pub fn deactivate_edge(&mut self, edge_id: EdgeId) {
        self.active_edges.remove(&edge_id);
        // Also remove all attribute indices for this edge
        self.edge_attribute_indices.remove(&edge_id);
        
        // Mark topology cache as dirty (will be rebuilt on next access)
        self.topology_cache_dirty = true;
        self.topology_generation += 1;
        
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
    }
    
    /// Bulk activate edges with pre-allocation (BULK OPTIMIZED)
    /// 
    /// PERFORMANCE: Single HashSet operation with capacity management
    pub fn activate_edges(&mut self, edges: Vec<EdgeId>) {
        // Pre-grow HashSet to prevent rehashing
        self.active_edges.reserve(edges.len());
        
        // Single bulk extend operation
        self.active_edges.extend(&edges);
        
        // Mark topology cache as dirty since we can't efficiently update columnar cache
        // for bulk operations without knowing source/target info
        self.topology_cache_dirty = true;
        self.topology_generation += 1;
    }
    

    /// Remove multiple edges from the active set (called by Graph.remove_edges())
    pub fn deactivate_edges(&mut self, edges: &[EdgeId]) {
        self.active_edges.retain(|edge| !edges.contains(edge));
        // Also remove all attribute indices for these edges
        for edge in edges {
            self.edge_attribute_indices.remove(edge);
        }
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
        self.topology_cache_dirty = true;
        self.topology_generation += 1;
    }

    /*
    === CURRENT INDEX MAPPINGS ===
    Space ONLY manages which entities have which current attribute indices
    Graph coordinates between Space (current state) and ChangeTracker (deltas)
    */
    
    /// Update the current attribute index for any entity (called by Graph after Pool storage)
    pub fn set_attr_index<T>(&mut self, entity_id: T, attr_name: AttrName, new_index: usize, is_node: bool) 
    where T: Into<usize> + Copy {
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
    where T: Into<usize> + Copy {
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
    
    /// Convenience method: Get current attribute index for a node
    pub fn get_node_attr_index(&self, node_id: NodeId, attr_name: &AttrName) -> Option<usize> {
        self.get_attr_index(node_id, attr_name, true)
    }
    
    /// Convenience method: Set current attribute index for a node
    pub fn set_node_attr_index(&mut self, node_id: NodeId, attr_name: AttrName, new_index: usize) {
        self.set_attr_index(node_id, attr_name, new_index, true)
    }
    
    /// Convenience method: Get current attribute index for an edge
    pub fn get_edge_attr_index(&self, edge_id: EdgeId, attr_name: &AttrName) -> Option<usize> {
        self.get_attr_index(edge_id, attr_name, false)
    }
    
    /// Convenience method: Set current attribute index for an edge  
    pub fn set_edge_attr_index(&mut self, edge_id: EdgeId, attr_name: AttrName, new_index: usize) {
        self.set_attr_index(edge_id, attr_name, new_index, false)
    }
    
    /// Get all attribute names and indices for a node
    pub fn get_node_attr_indices(&self, node_id: NodeId) -> HashMap<AttrName, usize> {
        self.node_attribute_indices
            .get(&(node_id as usize))
            .cloned()
            .unwrap_or_default()
    }
    
    /// Get all attribute names and indices for an edge
    pub fn get_edge_attr_indices(&self, edge_id: EdgeId) -> HashMap<AttrName, usize> {
        self.edge_attribute_indices
            .get(&(edge_id as usize))
            .cloned()
            .unwrap_or_default()
    }

    /*
    === COLUMNAR FILTERING OPTIMIZATION ===
    Bulk attribute operations for vectorized filtering
    */
    
    /// Get attribute indices for all active nodes/edges in bulk (VECTORIZED)
    /// 
    /// PERFORMANCE: Single bulk operation replaces individual per-node lookups
    /// This is the key enabler for columnar filtering operations
    pub fn get_attribute_indices(&self, 
        attr_name: &AttrName,
        is_node: bool
    ) -> Vec<(NodeId, Option<usize>)> {
        
        if is_node {
            // Get all active nodes and their attribute indices in one operation
            self.active_nodes
                .iter()
                .map(|&node_id| {
                    let index = self.get_node_attr_index(node_id, attr_name);
                    (node_id, index)
                })
                .collect()
        } else {
            // Get all active edges and their attribute indices in one operation
            self.active_edges
                .iter()
                .map(|&edge_id| {
                    let index = self.get_edge_attr_index(edge_id, attr_name);
                    (edge_id as NodeId, index)  // Cast EdgeId to NodeId for consistent interface
                })
                .collect()
        }
    }
    
    /// Get attribute values for all active nodes/edges in columnar format
    /// 
    /// PERFORMANCE: This method combines index lookup + value retrieval in single bulk operation
    /// Replaces the pattern: for node in nodes { get_index(); get_value() } 
    pub fn get_attributes<'a>(&self, 
        pool: &'a GraphPool, 
        attr_name: &AttrName,
        is_node: bool
    ) -> Vec<(NodeId, Option<&'a crate::types::AttrValue>)> {
        
        // STEP 1: Get all indices for active entities in bulk
        let entity_indices = self.get_attribute_indices(attr_name, is_node);
        
        // STEP 2: Bulk attribute retrieval from pool
        pool.get_attribute_values(attr_name, &entity_indices, is_node)
    }

    /// Get attribute values for a specific subset of nodes in bulk (NEW OPTIMIZED METHOD)
    /// 
    /// PERFORMANCE: Single bulk operation replaces N individual lookups
    /// This is what the query system should actually use
    pub fn get_attributes_for_nodes<'a>(
        &self,
        pool: &'a GraphPool,
        attr_name: &AttrName,
        node_ids: &[NodeId]
    ) -> Vec<(NodeId, Option<&'a crate::types::AttrValue>)> {
        // STEP 1: Bulk index lookup for specific nodes
        let entity_indices: Vec<(NodeId, Option<usize>)> = node_ids
            .iter()
            .map(|&node_id| {
                let index = self.node_attribute_indices
                    .get(&(node_id as usize))
                    .and_then(|attrs| attrs.get(attr_name))
                    .copied();
                (node_id, index)
            })
            .collect();
        
        // STEP 2: Single bulk retrieval from pool
        pool.get_attribute_values(attr_name, &entity_indices, true)
    }

    /// Get attribute values for a specific subset of edges in bulk (NEW OPTIMIZED METHOD)
    pub fn get_attributes_for_edges<'a>(
        &self,
        pool: &'a GraphPool,
        attr_name: &AttrName,
        edge_ids: &[EdgeId]
    ) -> Vec<(NodeId, Option<&'a crate::types::AttrValue>)> {
        // STEP 1: Bulk index lookup for specific edges
        let entity_indices: Vec<(NodeId, Option<usize>)> = edge_ids
            .iter()
            .map(|&edge_id| {
                let index = self.edge_attribute_indices
                    .get(&(edge_id as usize))
                    .and_then(|attrs| attrs.get(attr_name))
                    .copied();
                (edge_id as NodeId, index) // Cast for consistent interface
            })
            .collect();
        
        // STEP 2: Single bulk retrieval from pool
        pool.get_attribute_values(attr_name, &entity_indices, false)
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
        self.active_nodes.iter().copied().collect()
    }

    /// Get all active edge IDs as a vector
    pub fn edge_ids(&self) -> Vec<EdgeId> {
        self.active_edges.iter().copied().collect()
    }
    
    /*
    === COLUMNAR TOPOLOGY ACCESS ===
    High-performance vectorized topology queries
    */
    
    /// Get columnar topology vectors for vectorized neighbor queries
    /// 
    /// PERFORMANCE: Returns references to cached vectors for zero-copy access
    /// Returns (edge_ids, sources, targets) as parallel vectors
    pub fn get_columnar_topology(&self) -> (&[EdgeId], &[NodeId], &[NodeId]) {
        (&self.active_edge_ids, &self.edge_sources, &self.edge_targets)
    }
    
    /// Ensure topology cache is up-to-date and return columnar topology
    /// This is the method that should be called from Graph API when cache rebuilding is needed
    pub fn get_columnar_topology_with_rebuild(&mut self, pool: &crate::core::pool::GraphPool) -> (&[EdgeId], &[NodeId], &[NodeId]) {
        if self.topology_cache_dirty {
            self.rebuild_topology_cache(pool);
        }
        self.get_columnar_topology()
    }
    /// Rebuild columnar topology cache from current active edges
    /// 
    /// PERFORMANCE: Called automatically when cache is dirty
    fn rebuild_topology_cache(&mut self, pool: &crate::core::pool::GraphPool) {
        // Clear existing cache
        self.active_edge_ids.clear();
        self.edge_sources.clear();
        self.edge_targets.clear();
        
        // Reserve capacity
        let edge_count = self.active_edges.len();
        self.active_edge_ids.reserve(edge_count);
        self.edge_sources.reserve(edge_count);
        self.edge_targets.reserve(edge_count);
        
        // Rebuild from active edges
        for &edge_id in &self.active_edges {
            if let Some((source, target)) = pool.get_edge_endpoints(edge_id) {
                self.active_edge_ids.push(edge_id);
                self.edge_sources.push(source);
                self.edge_targets.push(target);
            }
        }
        
        self.topology_cache_dirty = false;
    }
    

    
    /// Get current topology generation
    pub fn get_topology_generation(&self) -> usize {
        self.topology_generation
    }
    
    /// Check if topology cache is dirty
    pub fn is_topology_cache_dirty(&self) -> bool {
        self.topology_cache_dirty
    }
    
    /// NOTE: Topology queries (neighbors, degree, connectivity) now handled by
    /// Graph coordinator which queries Pool for topology and filters by Space's active sets

    /*
    === CHANGE TRACKING ===
    Workspace state and change management
    */

    /// Check if there are uncommitted changes
    pub fn has_uncommitted_changes(&self) -> bool {
        // NOTE: Graph manages change tracking now - Space doesn't track changes
        false
    }

    /// Get the number of uncommitted changes
    pub fn uncommitted_change_count(&self) -> usize {
        // NOTE: Graph manages change tracking now - Space doesn't count changes
        0
    }

    /// Get summary of uncommitted changes
    pub fn change_summary(&self) -> String {
        // TODO: Graph provides change summary now - placeholder for now
        "No changes tracked in Space".to_string()
    }

    /// Get the base state this workspace is built on
    pub fn get_base_state(&self) -> StateId {
        self.base_state
    }

    /// Create a delta object representing current changes
    /// USAGE: Called when committing changes to history
    pub fn create_change_delta(&self, _pool: &GraphPool) -> DeltaObject {
        // NOTE: Graph creates delta via ChangeTracker now - Space doesn't track changes
        // Return empty delta as placeholder
        DeltaObject::empty()
    }

    /// Clear all uncommitted changes (reset to base state)
    /// WARNING: This loses all work since the last commit
    pub fn reset_hard(&mut self) -> GraphResult<()> {
        // NOTE: Graph manages change state now - Space just clears active sets
        self.active_nodes.clear();
        self.active_edges.clear();
        self.node_attribute_indices.clear();
        self.edge_attribute_indices.clear();
        Ok(())
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
