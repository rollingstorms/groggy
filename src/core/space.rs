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
use std::sync::Arc;
use std::rc::Rc;
use std::cell::RefCell;
use crate::types::{NodeId, EdgeId, AttrName, StateId};
// NOTE: ChangeTracker import removed - Graph manages it directly now
use crate::errors::GraphResult;
use crate::core::pool::GraphPool;
use crate::core::delta::DeltaObject;

/// Snapshot of topology data that can be used without holding locks
#[derive(Debug, Clone)]
pub struct TopologySnapshot {
    pub edge_ids: Arc<Vec<EdgeId>>,
    pub sources: Arc<Vec<NodeId>>,
    pub targets: Arc<Vec<NodeId>>,
    pub neighbors: Arc<HashMap<NodeId, Vec<(NodeId, EdgeId)>>>,
    pub version: u64,
}

/// Internal cache state with interior mutability
#[derive(Debug)]
struct CacheState {
    built_version: u64,
    snapshot: Option<TopologySnapshot>,
}

impl CacheState {
    fn new() -> Self {
        Self {
            built_version: 0,
            snapshot: None,
        }
    }
    
    fn try_get_snapshot(&self, current_version: u64) -> Option<TopologySnapshot> {
        if self.built_version == current_version {
            self.snapshot.clone()
        } else {
            None
        }
    }
    
    fn set_snapshot(&mut self, snapshot: TopologySnapshot) {
        self.built_version = snapshot.version;
        self.snapshot = Some(snapshot);
    }
}

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
    
    /// All currently active (not deleted) nodes with interior mutability
    /// DESIGN: HashSet for O(1) contains() and fast iteration
    active_nodes: RefCell<HashSet<NodeId>>,
    
    /// All currently active (not deleted) edges with interior mutability  
    /// DESIGN: HashSet for O(1) contains() and fast iteration
    active_edges: RefCell<HashSet<EdgeId>>,
    
    /*
    === UNIFIED CACHE SYSTEM ===
    Single versioned cache with interior mutability for lock-free access
    */
    
    /// Shared reference to the graph pool for topology rebuilding
    pool: Rc<RefCell<GraphPool>>,
    
    /// Version counter with interior mutability - increments on structural changes
    version: RefCell<u64>,
    
    /// Cache state with interior mutability for lock-free rebuilding
    cache: RefCell<CacheState>,
    
    /*
    === ATTRIBUTE INDEX MAPPINGS ===
    Map entities to their current attribute indices in Pool columns
    */
    
    /// Maps node -> attribute_name -> column_index with interior mutability
    /// This is how we resolve "what is node X's current value for attribute Y"
    node_attribute_indices: RefCell<HashMap<NodeId, HashMap<AttrName, usize>>>,
    
    /// Maps edge -> attribute_name -> column_index with interior mutability
    /// Same pattern as nodes but for edges
    edge_attribute_indices: RefCell<HashMap<EdgeId, HashMap<AttrName, usize>>>,
    
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
    pub fn new(pool: Rc<RefCell<GraphPool>>, base_state: StateId) -> Self {
        Self {
            active_nodes: RefCell::new(HashSet::new()),
            active_edges: RefCell::new(HashSet::new()),
            pool,
            version: RefCell::new(1), // Start at 1 so empty cache (version 0) is immediately stale
            cache: RefCell::new(CacheState::new()),
            node_attribute_indices: RefCell::new(HashMap::new()),
            edge_attribute_indices: RefCell::new(HashMap::new()),
            base_state,
        }
    }


    /*
    === ACTIVE SET MANAGEMENT ===
    Legacy methods for external activation/deactivation
    */

    /// Add a node to the active set (called by Graph.add_node())
    pub fn activate_node(&self, node_id: NodeId) {
        self.active_nodes.borrow_mut().insert(node_id);
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
    }

    
    /// Bulk activate nodes from Vec (BULK OPTIMIZED for arbitrary IDs)
    /// 
    /// PERFORMANCE: Single HashSet::extend call instead of individual inserts
    pub fn activate_nodes(&self, nodes: Vec<NodeId>) {
        let mut active_nodes = self.active_nodes.borrow_mut();
        // Pre-grow HashSet
        active_nodes.reserve(nodes.len());
        
        // Single bulk extend operation
        active_nodes.extend(nodes);
    }

    /// Remove multiple nodes from the active set (called by Graph.remove_nodes())
    pub fn deactivate_nodes(&self, nodes: &[NodeId]) {
        self.active_nodes.borrow_mut().retain(|node| !nodes.contains(node));
        // Also remove all attribute indices for these nodes
        let mut node_attrs = self.node_attribute_indices.borrow_mut();
        for node in nodes {
            node_attrs.remove(node);
        }
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
    }

    /// Remove a node from the active set (called by Graph.remove_node())
    pub fn deactivate_node(&self, node_id: NodeId) {
        self.active_nodes.borrow_mut().remove(&node_id);
        // Also remove all attribute indices for this node
        self.node_attribute_indices.borrow_mut().remove(&node_id);
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
    }

    /// Add an edge to the active set (called by Graph.add_edge())
    pub fn activate_edge(&self, edge_id: EdgeId, _source: NodeId, _target: NodeId) {
        self.active_edges.borrow_mut().insert(edge_id);
        
        // Increment version to invalidate cache lazily
        *self.version.borrow_mut() += 1;
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
    }

    /// Remove an edge from the active set (called by Graph.remove_edge())
    pub fn deactivate_edge(&self, edge_id: EdgeId) {
        self.active_edges.borrow_mut().remove(&edge_id);
        // Also remove all attribute indices for this edge
        self.edge_attribute_indices.borrow_mut().remove(&edge_id);
        
        // Increment version to invalidate cache lazily
        *self.version.borrow_mut() += 1;
        
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
    }
    
    /// Bulk activate edges with pre-allocation (BULK OPTIMIZED)
    /// 
    /// PERFORMANCE: Single HashSet operation with capacity management
    pub fn activate_edges(&self, edges: Vec<EdgeId>) {
        let mut active_edges = self.active_edges.borrow_mut();
        // Pre-grow HashSet to prevent rehashing
        active_edges.reserve(edges.len());
        
        // Single bulk extend operation
        active_edges.extend(&edges);
        
        // Increment version to invalidate cache lazily
        // Single version bump for entire bulk operation
        *self.version.borrow_mut() += 1;
    }
    

    /// Remove multiple edges from the active set (called by Graph.remove_edges())
    pub fn deactivate_edges(&self, edges: &[EdgeId]) {
        self.active_edges.borrow_mut().retain(|edge| !edges.contains(edge));
        // Also remove all attribute indices for these edges
        let mut edge_attrs = self.edge_attribute_indices.borrow_mut();
        for edge in edges {
            edge_attrs.remove(edge);
        }
        
        // Increment version to invalidate cache lazily
        // Single version bump for entire bulk operation
        *self.version.borrow_mut() += 1;
        // NOTE: Graph calls ChangeTracker directly for cleaner separation
    }

    /*
    === CURRENT INDEX MAPPINGS ===
    Space ONLY manages which entities have which current attribute indices
    Graph coordinates between Space (current state) and ChangeTracker (deltas)
    */
    
    /// Update the current attribute index for any entity (called by Graph after Pool storage)
    pub fn set_attr_index<T>(&self, entity_id: T, attr_name: AttrName, new_index: usize, is_node: bool) 
    where T: Into<usize> + Copy {
        let id = entity_id.into();
        if is_node {
            self.node_attribute_indices
                .borrow_mut()
                .entry(id)
                .or_insert_with(HashMap::new)
                .insert(attr_name, new_index);
        } else {
            self.edge_attribute_indices
                .borrow_mut()
                .entry(id)
                .or_insert_with(HashMap::new)
                .insert(attr_name, new_index);
        }
    }
    
    /// Get current attribute index for any entity (used by Graph for change tracking)
    pub fn get_attr_index<T>(&self, entity_id: T, attr_name: &AttrName, is_node: bool) -> Option<usize> 
    where T: Into<usize> + Copy {
        let id = entity_id.into();
        if is_node {
            self.node_attribute_indices
                .borrow()
                .get(&id)
                .and_then(|attrs| attrs.get(attr_name))
                .copied()
        } else {
            self.edge_attribute_indices
                .borrow()
                .get(&id)
                .and_then(|attrs| attrs.get(attr_name))
                .copied()
        }
    }
    
    /// Convenience method: Get current attribute index for a node
    pub fn get_node_attr_index(&self, node_id: NodeId, attr_name: &AttrName) -> Option<usize> {
        self.get_attr_index(node_id, attr_name, true)
    }
    
    /// Convenience method: Set current attribute index for a node
    pub fn set_node_attr_index(&self, node_id: NodeId, attr_name: AttrName, new_index: usize) {
        self.set_attr_index(node_id, attr_name, new_index, true)
    }
    
    /// Convenience method: Get current attribute index for an edge
    pub fn get_edge_attr_index(&self, edge_id: EdgeId, attr_name: &AttrName) -> Option<usize> {
        self.get_attr_index(edge_id, attr_name, false)
    }
    
    /// Convenience method: Set current attribute index for an edge  
    pub fn set_edge_attr_index(&self, edge_id: EdgeId, attr_name: AttrName, new_index: usize) {
        self.set_attr_index(edge_id, attr_name, new_index, false)
    }
    
    /// Get all attribute names and indices for a node
    pub fn get_node_attr_indices(&self, node_id: NodeId) -> HashMap<AttrName, usize> {
        self.node_attribute_indices
            .borrow()
            .get(&(node_id as usize))
            .cloned()
            .unwrap_or_default()
    }
    
    /// Get all attribute names and indices for an edge
    pub fn get_edge_attr_indices(&self, edge_id: EdgeId) -> HashMap<AttrName, usize> {
        self.edge_attribute_indices
            .borrow()
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
                .borrow().iter()
                .map(|&node_id| {
                    let index = self.get_node_attr_index(node_id, attr_name);
                    (node_id, index)
                })
                .collect()
        } else {
            // Get all active edges and their attribute indices in one operation
            self.active_edges
                .borrow().iter()
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

    /// Get nodes that have a specific attribute (REVERSE INDEX OPTIMIZATION)
    /// 
    /// PERFORMANCE: Instead of checking N nodes to see if they have an attribute,
    /// find all nodes that have the attribute first (much faster for sparse attributes)
    pub fn get_nodes_with_attribute(&self, attr_name: &AttrName) -> Vec<NodeId> {
        let node_attrs = self.node_attribute_indices.borrow();
        let mut nodes_with_attr = Vec::new();
        
        // Iterate through all nodes that have any attributes
        for (&node_id, attrs) in node_attrs.iter() {
            if attrs.contains_key(attr_name) {
                nodes_with_attr.push(node_id as NodeId);
            }
        }
        
        nodes_with_attr
    }

    /// Get attribute values for a specific subset of nodes in bulk (ULTRA-OPTIMIZED)
    /// 
    /// PERFORMANCE: Eliminates all HashMap lookups for large sparse attribute queries
    /// This is what the query system should actually use
    pub fn get_attributes_for_nodes<'a>(
        &self,
        pool: &'a GraphPool,
        attr_name: &AttrName,
        node_ids: &[NodeId]
    ) -> Vec<(NodeId, Option<&'a crate::types::AttrValue>)> {
        
        if node_ids.len() < 1000 {
            // Direct lookup is fine for small sets - use original optimized version
            let node_attrs = self.node_attribute_indices.borrow();
            let entity_indices: Vec<(NodeId, Option<usize>)> = node_ids
                .iter()
                .map(|&node_id| {
                    let index = node_attrs
                        .get(&(node_id as usize))
                        .and_then(|attrs| attrs.get(attr_name))
                        .copied();
                    (node_id, index)
                })
                .collect();
            drop(node_attrs);
            
            pool.get_attribute_values(attr_name, &entity_indices, true)
        } else {
            // RADICAL OPTIMIZATION: For large sets, completely avoid individual HashMap lookups
            // Build result by iterating through attribute index ONCE instead of N lookups
            
            let node_attrs = self.node_attribute_indices.borrow();
            let query_set: std::collections::HashSet<NodeId> = node_ids.iter().copied().collect();
            let mut entity_indices = Vec::new();
            
            // Single pass through attribute indices - O(A) where A = nodes with any attributes
            // Instead of O(N) where N = query size
            for (&node_id_usize, attrs) in node_attrs.iter() {
                let node_id = node_id_usize as NodeId;
                if query_set.contains(&node_id) {
                    if let Some(&index) = attrs.get(attr_name) {
                        entity_indices.push((node_id, Some(index)));
                    }
                }
            }
            drop(node_attrs);
            
            if entity_indices.is_empty() {
                // No nodes in our query have this attribute
                return node_ids.iter().map(|&id| (id, None)).collect();
            }
            
            // Get attribute values for nodes that have the attribute
            let attr_results = pool.get_attribute_values(attr_name, &entity_indices, true);
            let attr_map: std::collections::HashMap<NodeId, Option<&'a crate::types::AttrValue>> = 
                attr_results.into_iter().collect();
            
            // Build final result maintaining order
            node_ids.iter().map(|&node_id| {
                (node_id, attr_map.get(&node_id).copied().flatten())
            }).collect()
        }
    }

    /// Get attribute values for a specific subset of edges in bulk (NEW OPTIMIZED METHOD)
    pub fn get_attributes_for_edges<'a>(
        &self,
        pool: &'a GraphPool,
        attr_name: &AttrName,
        edge_ids: &[EdgeId]
    ) -> Vec<(NodeId, Option<&'a crate::types::AttrValue>)> {
        // STEP 1: Single borrow for bulk index lookup - fixes O(n) RefCell borrows
        let edge_attrs = self.edge_attribute_indices.borrow();
        let entity_indices: Vec<(NodeId, Option<usize>)> = edge_ids
            .iter()
            .map(|&edge_id| {
                let index = edge_attrs
                    .get(&(edge_id as usize))
                    .and_then(|attrs| attrs.get(attr_name))
                    .copied();
                (edge_id as NodeId, index) // Cast for consistent interface
            })
            .collect();
        // Drop the borrow before calling pool
        drop(edge_attrs);
        
        // STEP 2: Single bulk retrieval from pool
        pool.get_attribute_values(attr_name, &entity_indices, false)
    }

    /*
    === SIMPLE ACTIVE STATE QUERIES ===
    Basic information about what's currently active
    */

    /// Get the number of active nodes
    pub fn node_count(&self) -> usize {
        self.active_nodes.borrow().len()
    }

    /// Get the number of active edges
    pub fn edge_count(&self) -> usize {
        self.active_edges.borrow().len()
    }

    /// Check if a node is currently active
    pub fn contains_node(&self, node_id: NodeId) -> bool {
        self.active_nodes.borrow().contains(&node_id)
    }

    /// Check if an edge is currently active
    pub fn contains_edge(&self, edge_id: EdgeId) -> bool {
        self.active_edges.borrow().contains(&edge_id)
    }

    /// Get all active node IDs (for iteration)
    pub fn get_active_nodes(&self) -> HashSet<NodeId> {
        self.active_nodes.borrow().clone()
    }
    
    /// Get all active edge IDs (for iteration)
    pub fn get_active_edges(&self) -> HashSet<EdgeId> {
        self.active_edges.borrow().clone()
    }

    /*
    === BASIC QUERIES ===
    Simple active set queries - topology handled by Graph coordinator
    */

    /// Get all active node IDs as a vector
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.active_nodes.borrow().iter().copied().collect()
    }

    /// Get all active edge IDs as a vector
    pub fn edge_ids(&self) -> Vec<EdgeId> {
        self.active_edges.borrow().iter().copied().collect()
    }
    
    /*
    === UNIFIED CACHE ACCESS ===
    Single method to get consistent topology and adjacency data with interior mutability
    */
    
    /// Get consistent snapshot of topology and adjacency data
    /// 
    /// PERFORMANCE: Rebuilds cache atomically only when version has changed
    /// All returned data is guaranteed to be consistent with each other
    pub fn snapshot(&self, pool: &GraphPool) -> (Arc<Vec<EdgeId>>, Arc<Vec<NodeId>>, Arc<Vec<NodeId>>, Arc<HashMap<NodeId, Vec<(NodeId, EdgeId)>>>) {
        let current_version = *self.version.borrow();
        
        // Fast path: try to get existing snapshot
        if let Some(snapshot) = self.cache.borrow().try_get_snapshot(current_version) {
            return (snapshot.edge_ids, snapshot.sources, snapshot.targets, snapshot.neighbors);
        }
        
        // Slow path: rebuild cache
        let (edge_ids, sources, targets) = self.rebuild_topology(pool);

        // Build adjacency map from columnar topology (O(E))
        let mut neighbors = HashMap::<NodeId, Vec<(NodeId, EdgeId)>>::new();
        for (i, &edge_id) in edge_ids.iter().enumerate() {
            let u = sources[i];
            let v = targets[i];
            neighbors.entry(u).or_default().push((v, edge_id));
            neighbors.entry(v).or_default().push((u, edge_id));
        }

        // Create new snapshot
        let edge_ids_arc = Arc::new(edge_ids);
        let sources_arc = Arc::new(sources);
        let targets_arc = Arc::new(targets);
        let neighbors_arc = Arc::new(neighbors);

        let snapshot = TopologySnapshot {
            edge_ids: edge_ids_arc.clone(),
            sources: sources_arc.clone(), 
            targets: targets_arc.clone(),
            neighbors: neighbors_arc.clone(),
            version: current_version,
        };

        // Update cache and return tuple
        self.cache.borrow_mut().set_snapshot(snapshot);
        (edge_ids_arc, sources_arc, targets_arc, neighbors_arc)
    }
    
    /// Rebuild columnar topology from active edges
    /// 
    /// PERFORMANCE: O(E) where E = number of active edges
    /// Returns owned vectors that will be wrapped in Arc
    fn rebuild_topology(&self, pool: &GraphPool) -> (Vec<EdgeId>, Vec<NodeId>, Vec<NodeId>) {
        let mut edge_ids = Vec::new();
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        
        // Get active edges with interior mutability
        let active_edges = self.active_edges.borrow();
        
        // Reserve capacity
        let edge_count = active_edges.len();
        edge_ids.reserve(edge_count);
        sources.reserve(edge_count);
        targets.reserve(edge_count);
        
        // Rebuild from active edges
        for &edge_id in active_edges.iter() {
            if let Some((source, target)) = pool.get_edge_endpoints(edge_id) {
                edge_ids.push(edge_id);
                sources.push(source);
                targets.push(target);
            }
        }
        
        (edge_ids, sources, targets)
    }
    
    /// Get current version (for debugging/monitoring)
    pub fn get_version(&self) -> u64 {
        *self.version.borrow()
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
    pub fn reset_hard(&self) -> GraphResult<()> {
        // NOTE: Graph manages change state now - Space just clears active sets
        self.active_nodes.borrow_mut().clear();
        self.active_edges.borrow_mut().clear();
        self.node_attribute_indices.borrow_mut().clear();
        self.edge_attribute_indices.borrow_mut().clear();
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
