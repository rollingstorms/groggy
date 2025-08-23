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

use crate::core::delta::DeltaObject;
use crate::core::pool::GraphPool;
use crate::types::{AttrName, EdgeId, NodeId, StateId};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Arc, RwLock};

/// Snapshot of topology data that can be used without holding locks
#[derive(Debug, Clone)]
pub struct TopologySnapshot {
    pub edge_ids: Arc<Vec<EdgeId>>,
    pub sources: Arc<Vec<NodeId>>,
    pub targets: Arc<Vec<NodeId>>,
    pub neighbors: Arc<HashMap<NodeId, Vec<(NodeId, EdgeId)>>>,
    /// Cached adjacency matrix (computed from neighbors for fast access)
    pub adjacency_matrix: Option<Arc<crate::core::adjacency::AdjacencyMatrix>>,
    pub version: u64,
}

/// Internal cache state with RwLock for read-heavy access
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
    === ACTIVE SETS - Plain containers (no RefCell) ===
    */
    /// All currently active (not deleted) nodes
    active_nodes: HashSet<NodeId>,

    /// All currently active (not deleted) edges  
    active_edges: HashSet<EdgeId>,

    /*
    === POOL REFERENCE ===
    */
    /// Shared reference to the graph pool for topology rebuilding
    #[allow(dead_code)]
    pool: Rc<std::cell::RefCell<GraphPool>>,

    /*
    === VERSION COUNTER - Plain u64 ===
    */
    /// Version counter - increments on structural changes
    version: u64,

    /*
    === CACHE - RwLock for read-heavy, occasional write ===
    */
    /// Cache state with RwLock for lock-free reads
    cache: RwLock<CacheState>,

    /*
    === ATTRIBUTE INDEX MAPPINGS - Plain HashMaps ===
    */
    /// Maps attribute_name -> node -> column_index (OPTIMIZED: attribute-first for bulk filtering)
    node_attribute_indices: HashMap<AttrName, HashMap<NodeId, usize>>,

    /// Maps attribute_name -> edge -> column_index (OPTIMIZED: attribute-first for bulk filtering)
    edge_attribute_indices: HashMap<AttrName, HashMap<EdgeId, usize>>,

    /*
    === WORKSPACE METADATA ===
    */
    /// Which historical state this workspace is based on
    base_state: StateId,
}

impl GraphSpace {
    /// Create a new empty graph space
    pub fn new(pool: Rc<std::cell::RefCell<GraphPool>>, base_state: StateId) -> Self {
        Self {
            active_nodes: HashSet::new(),
            active_edges: HashSet::new(),
            pool,
            version: 1, // Start at 1 so empty cache (version 0) is immediately stale
            cache: RwLock::new(CacheState::new()),
            node_attribute_indices: HashMap::new(),
            edge_attribute_indices: HashMap::new(),
            base_state,
        }
    }

    /*
    === ACTIVE SET MANAGEMENT ===
    Legacy methods for external activation/deactivation
    */

    /// Add a node to the active set (&mut self)
    pub fn activate_node(&mut self, node_id: NodeId) {
        self.active_nodes.insert(node_id);
    }

    /// Bulk activate nodes (&mut self - PERFORMANCE: single extend call)
    pub fn activate_nodes<I: IntoIterator<Item = NodeId>>(&mut self, nodes: I) {
        self.active_nodes.extend(nodes);
    }

    /// Remove multiple nodes from the active set (&mut self)
    pub fn deactivate_nodes(&mut self, nodes: &[NodeId]) {
        let to_remove: HashSet<NodeId> = nodes.iter().copied().collect();
        self.active_nodes.retain(|n| !to_remove.contains(n));
        // Remove nodes from all attribute indices
        for (_, nodes_map) in self.node_attribute_indices.iter_mut() {
            for node_id in &to_remove {
                nodes_map.remove(node_id);
            }
        }
    }

    /// Remove a node from the active set (&mut self)
    pub fn deactivate_node(&mut self, node_id: NodeId) {
        self.active_nodes.remove(&node_id);
        // Remove node from all attribute indices
        for (_, nodes) in self.node_attribute_indices.iter_mut() {
            nodes.remove(&node_id);
        }
    }

    /// Add an edge to the active set (&mut self - increments version for topology cache invalidation)
    pub fn activate_edge(&mut self, edge_id: EdgeId, _source: NodeId, _target: NodeId) {
        self.active_edges.insert(edge_id);
        self.version += 1; // invalidate topology cache
    }

    /// Remove an edge from the active set (&mut self)
    pub fn deactivate_edge(&mut self, edge_id: EdgeId) {
        self.active_edges.remove(&edge_id);
        // Remove edge from all attribute indices
        for (_, edges) in self.edge_attribute_indices.iter_mut() {
            edges.remove(&edge_id);
        }
        self.version += 1;
    }

    /// Bulk activate edges (&mut self)
    pub fn activate_edges<I: IntoIterator<Item = EdgeId>>(&mut self, edges: I) {
        self.active_edges.extend(edges);
        self.version += 1;
    }

    /// Remove multiple edges from the active set (&mut self)
    pub fn deactivate_edges(&mut self, edges: &[EdgeId]) {
        let to_remove: HashSet<EdgeId> = edges.iter().copied().collect();
        self.active_edges.retain(|e| !to_remove.contains(e));
        // Remove edges from all attribute indices
        for (_, edges_map) in self.edge_attribute_indices.iter_mut() {
            for edge_id in &to_remove {
                edges_map.remove(edge_id);
            }
        }
        self.version += 1;
    }

    /*
    === CURRENT INDEX MAPPINGS ===
    Space ONLY manages which entities have which current attribute indices
    Graph coordinates between Space (current state) and ChangeTracker (deltas)
    */

    /// Update the current attribute index for any entity (&mut self)
    pub fn set_attr_index<T>(
        &mut self,
        entity_id: T,
        attr_name: AttrName,
        new_index: usize,
        is_node: bool,
    ) where
        T: Into<usize> + Copy,
    {
        let id = entity_id.into();
        if is_node {
            self.node_attribute_indices
                .entry(attr_name)
                .or_default()
                .insert(id as NodeId, new_index);
        } else {
            self.edge_attribute_indices
                .entry(attr_name)
                .or_default()
                .insert(id as EdgeId, new_index);
        }
    }

    /// Get current attribute index for any entity (&self - read-only)
    pub fn get_attr_index<T>(
        &self,
        entity_id: T,
        attr_name: &AttrName,
        is_node: bool,
    ) -> Option<usize>
    where
        T: Into<usize> + Copy,
    {
        let id = entity_id.into();
        if is_node {
            self.node_attribute_indices
                .get(attr_name)
                .and_then(|nodes| nodes.get(&(id as NodeId)))
                .copied()
        } else {
            self.edge_attribute_indices
                .get(attr_name)
                .and_then(|edges| edges.get(&(id as EdgeId)))
                .copied()
        }
    }

    /// Get current attribute index for a node (&self - read-only)
    pub fn get_node_attr_index(&self, node_id: NodeId, attr_name: &AttrName) -> Option<usize> {
        self.node_attribute_indices
            .get(attr_name)
            .and_then(|nodes| nodes.get(&node_id))
            .copied()
    }

    /// Get specific attribute indices for multiple nodes in one call (ULTRA-OPTIMIZED: attribute-first)
    ///
    /// PERFORMANCE: Single attribute lookup + fast node iteration instead of N*2 HashMap lookups
    /// OLD: 50k * (node_lookup + attr_lookup) = 100k HashMap operations
    /// NEW: 1 * attr_lookup + 50k * (direct HashMap get) = 1 + 50k operations
    pub fn get_node_attr_indices_for_attr(
        &self,
        node_ids: &[NodeId],
        attr_name: &AttrName,
    ) -> Vec<(NodeId, Option<usize>)> {
        // Single attribute lookup to get all nodes with this attribute
        if let Some(attr_nodes) = self.node_attribute_indices.get(attr_name) {
            // Fast iteration through requested nodes with direct HashMap access
            node_ids
                .iter()
                .map(|&node_id| (node_id, attr_nodes.get(&node_id).copied()))
                .collect()
        } else {
            // No nodes have this attribute - return all None
            node_ids.iter().map(|&node_id| (node_id, None)).collect()
        }
    }

    /// Set current attribute index for a node (&mut self)
    pub fn set_node_attr_index(&mut self, node_id: NodeId, attr_name: AttrName, new_index: usize) {
        self.node_attribute_indices
            .entry(attr_name)
            .or_default()
            .insert(node_id, new_index);
    }

    /// Get current attribute index for an edge (&self - read-only)
    pub fn get_edge_attr_index(&self, edge_id: EdgeId, attr_name: &AttrName) -> Option<usize> {
        self.edge_attribute_indices
            .get(attr_name)
            .and_then(|edges| edges.get(&edge_id))
            .copied()
    }

    /// Set current attribute index for an edge (&mut self)
    pub fn set_edge_attr_index(&mut self, edge_id: EdgeId, attr_name: AttrName, new_index: usize) {
        self.edge_attribute_indices
            .entry(attr_name)
            .or_default()
            .insert(edge_id, new_index);
    }

    /// Get specific attribute indices for multiple edges in one call (ULTRA-OPTIMIZED: attribute-first)
    pub fn get_edge_attr_indices_for_attr(
        &self,
        edge_ids: &[EdgeId],
        attr_name: &AttrName,
    ) -> Vec<(EdgeId, Option<usize>)> {
        // Single attribute lookup to get all edges with this attribute
        if let Some(attr_edges) = self.edge_attribute_indices.get(attr_name) {
            // Fast iteration through requested edges with direct HashMap access
            edge_ids
                .iter()
                .map(|&edge_id| (edge_id, attr_edges.get(&edge_id).copied()))
                .collect()
        } else {
            // No edges have this attribute - return all None
            edge_ids.iter().map(|&edge_id| (edge_id, None)).collect()
        }
    }

    /// Get all attribute names and indices for a node (&self - read-only)
    pub fn get_node_attr_indices(&self, node_id: NodeId) -> HashMap<AttrName, usize> {
        let mut result = HashMap::new();
        for (attr_name, nodes) in &self.node_attribute_indices {
            if let Some(&index) = nodes.get(&node_id) {
                result.insert(attr_name.clone(), index);
            }
        }
        result
    }

    /// Get all attribute names and indices for an edge (&self - read-only)
    pub fn get_edge_attr_indices(&self, edge_id: EdgeId) -> HashMap<AttrName, usize> {
        let mut result = HashMap::new();
        for (attr_name, edges) in &self.edge_attribute_indices {
            if let Some(&index) = edges.get(&edge_id) {
                result.insert(attr_name.clone(), index);
            }
        }
        result
    }

    /*
    === COLUMNAR FILTERING OPTIMIZATION ===
    Bulk attribute operations for vectorized filtering
    */

    /// Get attribute indices for all active nodes (&self - ULTRA-OPTIMIZED: attribute-first)
    pub fn get_attribute_indices_nodes(
        &self,
        attr_name: &AttrName,
    ) -> Vec<(NodeId, Option<usize>)> {
        if let Some(attr_nodes) = self.node_attribute_indices.get(attr_name) {
            self.active_nodes
                .iter()
                .map(|&node_id| (node_id, attr_nodes.get(&node_id).copied()))
                .collect()
        } else {
            self.active_nodes
                .iter()
                .map(|&node_id| (node_id, None))
                .collect()
        }
    }

    /// Get attribute indices for all active edges (&self - ULTRA-OPTIMIZED: attribute-first)
    pub fn get_attribute_indices_edges(
        &self,
        attr_name: &AttrName,
    ) -> Vec<(EdgeId, Option<usize>)> {
        if let Some(attr_edges) = self.edge_attribute_indices.get(attr_name) {
            self.active_edges
                .iter()
                .map(|&edge_id| (edge_id, attr_edges.get(&edge_id).copied()))
                .collect()
        } else {
            self.active_edges
                .iter()
                .map(|&edge_id| (edge_id, None))
                .collect()
        }
    }

    /// Get attribute values for all active nodes in columnar format (&self - OPTIMIZED)
    pub fn get_attributes_nodes<'a>(
        &self,
        pool: &'a GraphPool,
        attr_name: &AttrName,
    ) -> Vec<(NodeId, Option<&'a crate::types::AttrValue>)> {
        let entity_indices = self.get_attribute_indices_nodes(attr_name);
        pool.get_attribute_values(attr_name, &entity_indices, true)
    }

    /// Get attribute values for all active edges in columnar format (&self - OPTIMIZED)
    pub fn get_attributes_edges<'a>(
        &self,
        pool: &'a GraphPool,
        attr_name: &AttrName,
    ) -> Vec<(EdgeId, Option<&'a crate::types::AttrValue>)> {
        let entity_indices = self.get_attribute_indices_edges(attr_name);
        pool.get_attribute_values(attr_name, &entity_indices, false)
    }

    /// Get nodes that have a specific attribute (ULTRA-OPTIMIZED: direct attribute lookup)
    ///
    /// PERFORMANCE: Direct O(1) lookup + iteration instead of checking N nodes
    /// OLD: Iterate all nodes, check if each has attribute
    /// NEW: Direct lookup of attribute -> get all nodes with it
    pub fn get_nodes_with_attribute(&self, attr_name: &AttrName) -> Vec<NodeId> {
        if let Some(attr_nodes) = self.node_attribute_indices.get(attr_name) {
            attr_nodes.keys().copied().collect()
        } else {
            Vec::new()
        }
    }

    /// Get attribute values for a specific subset of nodes in bulk (ULTRA-OPTIMIZED)
    ///
    /// PERFORMANCE: Eliminates all HashMap lookups for large sparse attribute queries
    /// This is what the query system should actually use
    /// Get attribute values for specific nodes (ULTRA-OPTIMIZED: no RefCell churn)
    pub fn get_attributes_for_nodes<'a>(
        &self,
        pool: &'a GraphPool,
        attr_name: &AttrName,
        node_ids: &[NodeId],
    ) -> Vec<(NodeId, Option<&'a crate::types::AttrValue>)> {
        let start_time = std::time::Instant::now();

        // PERFORMANCE: Bulk index lookup in single call
        let entity_indices = self.get_node_attr_indices_for_attr(node_ids, attr_name);

        let _index_time = start_time.elapsed();
        let start_time = std::time::Instant::now();

        // Single pool call for bulk attribute retrieval
        let values = pool.get_attribute_values(attr_name, &entity_indices, true);

        let _pool_time = start_time.elapsed();

        values
    }

    /// Get attribute values for specific edges (ULTRA-OPTIMIZED: no RefCell churn)
    pub fn get_attributes_for_edges<'a>(
        &self,
        pool: &'a GraphPool,
        attr_name: &AttrName,
        edge_ids: &[EdgeId],
    ) -> Vec<(EdgeId, Option<&'a crate::types::AttrValue>)> {
        // PERFORMANCE: Bulk index lookup in single call
        let entity_indices = self.get_edge_attr_indices_for_attr(edge_ids, attr_name);

        // Single pool call for bulk attribute retrieval
        pool.get_attribute_values(attr_name, &entity_indices, false)
    }

    /*
    === SIMPLE ACTIVE STATE QUERIES ===
    Basic information about what's currently active
    */

    /// Get the number of active nodes (&self - read-only)
    pub fn node_count(&self) -> usize {
        self.active_nodes.len()
    }

    /// Get the number of active edges (&self - read-only)
    pub fn edge_count(&self) -> usize {
        self.active_edges.len()
    }

    /// Check if a node is currently active (&self - read-only)
    pub fn contains_node(&self, node_id: NodeId) -> bool {
        self.active_nodes.contains(&node_id)
    }

    /// Check if an edge is currently active (&self - read-only)
    pub fn contains_edge(&self, edge_id: EdgeId) -> bool {
        self.active_edges.contains(&edge_id)
    }

    /// Get all active node IDs (compatibility with old RefCell API)
    pub fn get_active_nodes(&self) -> HashSet<NodeId> {
        self.active_nodes.clone()
    }

    /// Get all active edge IDs (compatibility with old RefCell API)
    pub fn get_active_edges(&self) -> HashSet<EdgeId> {
        self.active_edges.clone()
    }

    /*
    === BASIC QUERIES ===
    Simple active set queries - topology handled by Graph coordinator
    */

    /// Get all active node IDs as a vector (&self - read-only)
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.active_nodes.iter().copied().collect()
    }

    /// Get all active edge IDs as a vector (&self - read-only)
    pub fn edge_ids(&self) -> Vec<EdgeId> {
        self.active_edges.iter().copied().collect()
    }

    /*
    === UNIFIED CACHE ACCESS ===
    Single method to get consistent topology and adjacency data with interior mutability
    */

    /// Get consistent snapshot of topology and adjacency data (RwLock for read-heavy access)
    pub fn snapshot(
        &self,
        pool: &GraphPool,
    ) -> (
        Arc<Vec<EdgeId>>,
        Arc<Vec<NodeId>>,
        Arc<Vec<NodeId>>,
        Arc<HashMap<NodeId, Vec<(NodeId, EdgeId)>>>,
    ) {
        let current_version = self.version;

        // Fast path: read-lock and try to get cached snapshot
        {
            let cache = self.cache.read().unwrap();
            if let Some(snapshot) = cache.try_get_snapshot(current_version) {
                return (
                    snapshot.edge_ids,
                    snapshot.sources,
                    snapshot.targets,
                    snapshot.neighbors,
                );
            }
        }

        // Slow path: write-lock and rebuild if still stale
        let mut cache = self.cache.write().unwrap();

        // Double-check after acquiring write lock (another thread might have rebuilt)
        if let Some(snapshot) = cache.try_get_snapshot(current_version) {
            return (
                snapshot.edge_ids,
                snapshot.sources,
                snapshot.targets,
                snapshot.neighbors,
            );
        }

        // Rebuild topology cache
        let (edge_ids, sources, targets) = self.rebuild_topology(pool);

        // Build adjacency map from columnar topology (O(E))
        // DIRECTED: Only add source → target edges
        // UNDIRECTED: Add both source ↔ target edges for traversal
        let mut neighbors = HashMap::<NodeId, Vec<(NodeId, EdgeId)>>::new();
        let graph_type = pool.graph_type();

        for (i, &edge_id) in edge_ids.iter().enumerate() {
            let u = sources[i];
            let v = targets[i];

            // Always add the primary direction (source → target)
            neighbors.entry(u).or_default().push((v, edge_id));

            // For undirected graphs, also add the reverse direction (target → source)
            if graph_type == crate::types::GraphType::Undirected {
                neighbors.entry(v).or_default().push((u, edge_id));
            }
        }

        // Create Arc-wrapped data for zero-copy sharing
        let edge_ids_arc = Arc::new(edge_ids);
        let sources_arc = Arc::new(sources);
        let targets_arc = Arc::new(targets);
        let neighbors_arc = Arc::new(neighbors);

        let snapshot = TopologySnapshot {
            edge_ids: edge_ids_arc.clone(),
            sources: sources_arc.clone(),
            targets: targets_arc.clone(),
            neighbors: neighbors_arc.clone(),
            adjacency_matrix: None, // Will be computed on-demand later
            version: current_version,
        };

        cache.set_snapshot(snapshot);
        (edge_ids_arc, sources_arc, targets_arc, neighbors_arc)
    }

    /// Rebuild columnar topology from active edges (PERFORMANCE: O(E))
    fn rebuild_topology(&self, pool: &GraphPool) -> (Vec<EdgeId>, Vec<NodeId>, Vec<NodeId>) {
        let edge_count = self.active_edges.len();
        let mut edge_ids = Vec::with_capacity(edge_count);
        let mut sources = Vec::with_capacity(edge_count);
        let mut targets = Vec::with_capacity(edge_count);

        for &edge_id in &self.active_edges {
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
        self.version
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

    /// Create a delta object representing current changes (placeholder)
    pub fn create_change_delta(&self, _pool: &GraphPool) -> DeltaObject {
        DeltaObject::empty()
    }

    /*
    === COMPATIBILITY METHODS FOR OLD REFCELL API ===
    */

    /// Generic attribute index getter (compatibility with old API)
    pub fn get_attribute_indices(
        &self,
        attr_name: &AttrName,
        is_node: bool,
    ) -> Vec<(NodeId, Option<usize>)> {
        if is_node {
            self.get_attribute_indices_nodes(attr_name)
        } else {
            // Convert EdgeId result to NodeId for compatibility
            self.get_attribute_indices_edges(attr_name)
                .into_iter()
                .map(|(edge_id, index)| (edge_id as NodeId, index))
                .collect()
        }
    }

    /// Generic attribute values getter (compatibility with old API)
    pub fn get_attributes<'a>(
        &self,
        pool: &'a GraphPool,
        attr_name: &AttrName,
        is_node: bool,
    ) -> Vec<(NodeId, Option<&'a crate::types::AttrValue>)> {
        if is_node {
            self.get_attributes_nodes(pool, attr_name)
        } else {
            // Convert EdgeId result to NodeId for compatibility
            self.get_attributes_edges(pool, attr_name)
                .into_iter()
                .map(|(edge_id, value)| (edge_id as NodeId, value))
                .collect()
        }
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
